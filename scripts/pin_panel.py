#!/usr/bin/env python
"""Pin the model panel: revision hashes, licences, gating, tokenizer and chat-template SHA-256 -> a hashed manifest.

DESIGN_DECISIONS 7.1 / PREREGISTRATION §7 (panel freeze, week of 19-25 October 2026): every self-hosted entry of
configs/models.yaml gets a full commit hash; the freeze commit records each entry's tokenizer files (a
"tokenizer-fixed" control pair needs byte-identical files) and licence. huggingface.co is not reachable from the
build machine, so this script is a KIT: run it where the Hub is reachable (a Kaggle or Colab cell, or the author's
laptop) with an HF token whose account has accepted the gated licences (Gemma, Llama 3.1, Vistral).

    python scripts/pin_panel.py --offline                       # draft manifest from models.yaml only (no network)
    python scripts/pin_panel.py                                 # resolve every self-hosted entry on the Hub
    python scripts/pin_panel.py --names gemma-3-1b-it qwen3.5-0.8b
    python scripts/pin_panel.py --apply                         # ... and write the hashes into configs/models.yaml
    python scripts/pin_panel.py --verify                        # models.yaml revisions == manifest, manifest hash intact

The manifest (configs/panel_manifest.json) carries `manifest_sha256` = SHA-256 of its canonical JSON without that
field, the SHA-256 of configs/models.yaml and configs/env_pins.txt it was made against, and per entry: name, hf_id,
the checkpoint actually loaded (a quantized checkpoint when the entry names one), revision, status (pinned /
unpinned / error), gated, licence (the model card's `license` field and tags), last-modified date, the SHA-256 of
each tokenizer file present (tokenizer.json, tokenizer.model, tokenizer_config.json, special_tokens_map.json,
chat_template.jinja) and of the chat template text. `--apply` edits only the `revision:` value of the entries it
pinned and leaves every comment in place. Nothing is downloaded but tokenizer and config files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "configs" / "models.yaml"
ENV_PINS = ROOT / "configs" / "env_pins.txt"
OUT = ROOT / "configs" / "panel_manifest.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str | None:
    return sha256_bytes(p.read_bytes()) if p.exists() else None


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_hash(manifest: dict) -> str:
    return sha256_bytes(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}).encode("utf-8"))


def self_hosted(cfg: dict, names: list[str] | None = None) -> list[dict]:
    """Every entry that loads weights (an hf_id and a non-API backend), panel and serving variants alike."""
    out = []
    for e in cfg.get("models") or []:
        if names and e.get("name") not in names:
            continue
        if not e.get("hf_id") or e.get("hardware") == "api" or e.get("backend") in ("gemini", "openai_compat"):
            continue
        out.append(e)
    return out


def checkpoint_of(e: dict) -> str:
    q = e.get("quantization") or {}
    return q.get("checkpoint") or e["hf_id"]


def quantized_checkpoint_missing(e: dict) -> bool:
    """A quantized entry (AWQ / GPTQ / GGUF) whose pre-quantized checkpoint has not been chosen yet: pinning the base
    repository would pin the wrong weights."""
    q = e.get("quantization") or {}
    return bool(q.get("method")) and str(q.get("method")).lower() not in ("bitsandbytes", "bnb", "nf4", "int4", "int8") \
        and not q.get("checkpoint")


def offline_entry(e: dict) -> dict:
    rev = e.get("revision")
    rec = {"name": e["name"], "hf_id": e["hf_id"], "checkpoint": checkpoint_of(e), "hf_id_status": e.get("hf_id_status"),
           "group": e.get("group"), "revision": rev, "status": "pinned" if rev and FULL_SHA.match(str(rev)) else "unpinned",
           "gated": e.get("gated"), "license": None, "tokenizer_files": {}, "chat_template_sha256": None}
    if quantized_checkpoint_missing(e):
        rec["note"] = "quantized checkpoint not chosen (quantization.checkpoint is null): choose it, then pin it"
    return rec


def resolve_entry(e: dict, api, download) -> dict:
    """One entry against the Hub. `api` is a huggingface_hub.HfApi (or a stand-in with model_info); `download(repo,
    filename, revision)` returns a local path or raises."""
    rec = offline_entry(e)
    if quantized_checkpoint_missing(e):
        rec.update(status="unpinned")
        return rec
    repo = checkpoint_of(e)
    try:
        info = api.model_info(repo)
    except Exception as exc:                      # gated without access, wrong id, network
        rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return rec
    sha = getattr(info, "sha", None)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    tags = list(getattr(info, "tags", None) or [])
    rec.update(revision=sha, status="pinned" if sha and FULL_SHA.match(sha) else "error",
               gated=getattr(info, "gated", None),
               license={"card": (card or {}).get("license"), "license_name": (card or {}).get("license_name"),
                        "tags": [t for t in tags if t.startswith("license:")]},
               last_modified=str(getattr(info, "last_modified", None) or getattr(info, "lastModified", "") or ""),
               siblings=sorted(s.rfilename for s in (getattr(info, "siblings", None) or []))[:200])
    files = {}
    file_errors = {}
    template = None
    for fn in TOKENIZER_FILES:
        if rec["siblings"] and fn not in rec["siblings"]:
            continue
        try:
            p = Path(download(repo, fn, sha))
        except Exception as exc:                  # recorded, not fatal: some repositories ship only some files
            file_errors[fn] = f"{type(exc).__name__}: {str(exc)[:120]}"
            continue
        files[fn] = sha256_file(p)
        if fn == "chat_template.jinja":
            template = p.read_text(encoding="utf-8")
        elif fn == "tokenizer_config.json" and template is None:
            try:
                template = json.loads(p.read_text(encoding="utf-8")).get("chat_template")
            except Exception:
                template = None
    if isinstance(template, list):                # several named templates
        template = canonical(template)
    rec["tokenizer_files"] = files
    if file_errors:
        rec["tokenizer_file_errors"] = file_errors
    rec["chat_template_sha256"] = sha256_bytes(template.encode("utf-8")) if template else None
    return rec


def build_manifest(entries: list[dict], mode: str) -> dict:
    m = {"created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": mode,
         "models_yaml_sha256": sha256_file(MODELS), "env_pins_sha256": sha256_file(ENV_PINS),
         "n_entries": len(entries), "n_pinned": sum(e["status"] == "pinned" for e in entries),
         "entries": entries}
    m["manifest_sha256"] = manifest_hash(m)
    return m


def apply_revisions(text: str, pins: dict[str, str]) -> tuple[str, list[str]]:
    """Write `revision: <sha>` for the named entries, block style (`- name: X` ... `revision: null`) and flow style
    (`- {name: X, ..., revision: null, ...}`); comments untouched. Returns (new text, names changed)."""
    lines = text.splitlines(keepends=True)
    changed = []
    current = None
    for i, line in enumerate(lines):
        m = re.match(r"^\s*-\s*name:\s*([^\s#]+)", line)
        if m:
            current = m.group(1)
        flow = re.match(r"^\s*-\s*\{\s*name:\s*([^,\s]+)", line)
        if flow and flow.group(1) in pins:
            new = re.sub(r"revision:\s*null", f"revision: {pins[flow.group(1)]}", line, count=1)
            if new != line:
                lines[i] = new
                changed.append(flow.group(1))
            continue
        if current in pins and re.match(r"^\s+revision:\s*null\b", line):
            lines[i] = re.sub(r"revision:\s*null", f"revision: {pins[current]}", line, count=1)
            changed.append(current)
    return "".join(lines), changed


def verify(manifest: dict, cfg: dict) -> list[str]:
    problems = []
    if manifest_hash(manifest) != manifest.get("manifest_sha256"):
        problems.append("manifest_sha256 does not match the manifest's content")
    by_name = {e["name"]: e for e in cfg.get("models") or []}
    for rec in manifest["entries"]:
        e = by_name.get(rec["name"])
        if e is None:
            problems.append(f"{rec['name']}: not in models.yaml")
        elif rec["status"] == "pinned" and e.get("revision") != rec["revision"]:
            problems.append(f"{rec['name']}: models.yaml revision {e.get('revision')!r} != manifest {rec['revision']!r}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(MODELS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--offline", action="store_true", help="no network: a draft manifest from models.yaml")
    ap.add_argument("--apply", action="store_true", help="write the pinned hashes into models.yaml")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.models).read_text(encoding="utf-8"))
    if args.verify:
        man = json.loads(Path(args.out).read_text(encoding="utf-8"))
        problems = verify(man, cfg)
        print(json.dumps({"ok": not problems, "problems": problems, "n_pinned": man.get("n_pinned")}, ensure_ascii=False))
        return 1 if problems else 0
    entries = self_hosted(cfg, args.names)
    if args.offline:
        recs = [offline_entry(e) for e in entries]
    else:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()

        def download(repo, fn, rev):
            return hf_hub_download(repo, fn, revision=rev)
        recs = [resolve_entry(e, api, download) for e in entries]
    man = build_manifest(recs, "offline" if args.offline else "hub")
    Path(args.out).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.apply and not args.offline:
        pins = {r["name"]: r["revision"] for r in recs if r["status"] == "pinned"}
        text, changed = apply_revisions(Path(args.models).read_text(encoding="utf-8"), pins)
        Path(args.models).write_text(text, encoding="utf-8")
        man = build_manifest(recs, "hub")                     # re-hash against the edited models.yaml
        Path(args.out).write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"revisions written for {len(changed)} entries")
    print(json.dumps({"out": args.out, "n_entries": man["n_entries"], "n_pinned": man["n_pinned"],
                      "errors": [r["name"] for r in recs if r["status"] == "error"],
                      "manifest_sha256": man["manifest_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
