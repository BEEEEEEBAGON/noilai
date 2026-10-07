#!/usr/bin/env python
"""Pin the self-hosted model panel: resolve every `hf_id` of configs/models.yaml to a full
commit hash, record the SHA-256 of its tokenizer files, and write the hashes back into the
`revision:` lines (DESIGN_DECISIONS 7.1: the runner refuses an unpinned self-hosted entry).

    python scripts/pin_panel.py --out experiments/panel_pins.json          # resolve (needs hub access)
    python scripts/pin_panel.py --apply --from experiments/panel_pins.json # offline: write the pins
    python scripts/pin_panel.py --apply --from ... --dry-run               # show what would change
    python scripts/pin_panel.py --apply --from ... --partial               # pin what resolved, keep the rest null
    python scripts/pin_panel.py --apply --from ... --force                 # overwrite a different existing pin
    python scripts/pin_panel.py --names gemma-3-1b-it qwen3.5-2b --out ... # a subset

Which entries: every entry whose `backend` downloads weights (hf, vllm, llama_cpp: the
LOCAL_WEIGHT_BACKENDS of noilai.eval.run) and that carries an `hf_id`, i.e. the self-hosted
members of the seven panel groups plus their `bf16_reference` and `reasoning_substudy` serving
variants (same weights, so the same hash). API entries with an open-weight `hf_id` (gpt-oss on
Groq) are resolved only with --include-api, for the tokenizer record; they have no `revision`
line and are never written. The hub is asked once per distinct `hf_id`.

What is recorded, one record per entry, in the JSON (`records`): `name`, `hf_id`, the config's
own `hf_id_status` (known/uncertain), the resolution outcome `hf_id_status` (found / not_found /
gated / error), `revision` (the 40-hex commit hash of the repository's main branch at
`resolved_utc`), `hub_gated` (the hub's gating flag), `tokenizer_sha256` (one SHA-256 per
tokenizer file present in the repository: tokenizer.model, tokenizer.json,
tokenizer_config.json; downloaded with hf_hub_download at the pinned revision and hashed) and
`error`. A gated repository whose licence the token has not accepted still resolves its hash
(the metadata is public) and reports `gated` with empty tokenizer hashes.

--apply (text-preserving): only the value of each entry's `revision:` line changes, written as
a double-quoted scalar (a hash made of digits only would otherwise parse as an integer);
comments, order and every other field stay, and the rewritten file is re-parsed and compared
field by field with the original before it is written (`hf_id_status` is never flipped: the
author verifies the uncertain ids by hand, DD 7.1). It refuses when any PANEL entry is
unresolved unless --partial, and when an entry is already pinned (a full hash or a Kaggle
Models slug) to something other than the resolved hash unless --force; an entry already pinned
to the resolved hash is left as it is.

Exit codes: 0 when every selected panel entry resolved (resolution) or the file was written
(apply); 1 when the resolution is incomplete (the JSON is still written), when apply refuses,
or when the rewrite does not verify.

Kaggle CPU usage (huggingface.co is unreachable from the build machine, so the resolution runs
where the hub is reachable and the apply step runs here):
  1. New Kaggle notebook, accelerator None (CPU), Internet ON; attach the code bundle dataset
     (README "Running from a private repository") or clone the repository; add the Hugging
     Face token as the Kaggle secret HF_TOKEN (gated families -- Gemma, Llama, Vistral -- need
     the licence accepted on that account) and export it:
         from kaggle_secrets import UserSecretsClient
         import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  2. `pip install huggingface_hub pyyaml` (no torch, no transformers needed), then
         python scripts/pin_panel.py --out /kaggle/working/panel_pins.json
     (or `python scripts/kaggle_cpu_jobs.py pin`, which also records the output's hash).
  3. Download panel_pins.json from the notebook's output, put it at experiments/panel_pins.json
     in the repository and run
         python scripts/pin_panel.py --apply --from experiments/panel_pins.json
     then commit models.yaml and the JSON together (the pin is part of the panel freeze).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = ROOT / "configs" / "models.yaml"
DEFAULT_OUT = ROOT / "experiments" / "panel_pins.json"
# DESIGN_DECISIONS 7.1: the backends that download weights and therefore need a pinned revision
# (kept equal to noilai.eval.run.LOCAL_WEIGHT_BACKENDS; tests/test_pin_panel.py checks that)
LOCAL_WEIGHT_BACKENDS = ("hf", "vllm", "llama_cpp")
# the files whose SHA-256 identifies a tokenizer (DD 7.5: `tokenizer_sha256` in every manifest);
# hashed only when the repository has them
TOKENIZER_FILES = ("tokenizer.model", "tokenizer.json", "tokenizer_config.json")
STATUS_FOUND, STATUS_NOT_FOUND, STATUS_GATED, STATUS_ERROR = "found", "not_found", "gated", "error"
# DESIGN_DECISIONS 7.1: a full HF commit hash, or a Kaggle Models slug + version (owner/model/framework/
# variation/version, the framework segment optional in older slugs); the two "already pinned" forms
COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
KAGGLE_SLUG = re.compile(r"^[\w.-]+(?:/[\w.-]+){2,3}/\d+$")
# the entry shapes of models.yaml: block entries start with `- name: x` and carry `revision:` on a
# line of their own; the serving variants are one-line flow mappings `- {name: x, ..., revision: null, ...}`
_BLOCK_NAME = re.compile(r"^\s*-\s+name:\s*(?P<name>[^\s#]+)")
_FLOW_NAME = re.compile(r"^\s*-\s*\{\s*name:\s*(?P<name>[^,}\s]+)")
_BLOCK_REV = re.compile(r"^(?P<lead>\s*revision:\s*)(?P<value>[^#\s]+)(?P<rest>.*)$")
_FLOW_REV = re.compile(r"(?P<lead>\brevision:\s*)(?P<value>[^,}\s]+)")
_ERROR_CHARS = 400          # how much of an exception message the record keeps

try:                        # the hub library is optional: --apply --from runs without it
    from huggingface_hub import errors as _hub_errors
except Exception:           # pragma: no cover - exercised only where huggingface_hub is absent
    _hub_errors = None


# ------------------------------------------------------------------ selection
def load_models(path: Path = MODELS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def panel_groups(cfg: dict) -> set[str]:
    return {g for g, meta in (cfg.get("groups") or {}).items() if (meta or {}).get("panel")}


def is_self_hosted(entry: dict) -> bool:
    """An entry whose backend downloads weights and that names a hub repository."""
    return entry.get("backend") in LOCAL_WEIGHT_BACKENDS and bool(entry.get("hf_id"))


def select_entries(cfg: dict, include_api: bool = False, names: list[str] | None = None) -> list[dict]:
    """The entries to resolve: the self-hosted ones, plus API entries with an hf_id when asked."""
    out = []
    for m in cfg.get("models") or []:
        if names is not None and m.get("name") not in names:
            continue
        if is_self_hosted(m) or (include_api and m.get("hf_id")):
            out.append(m)
    if names is not None:
        unknown = sorted(set(names) - {m.get("name") for m in out})
        if unknown:
            raise KeyError(f"no resolvable entry named {unknown} (self-hosted with an hf_id, or API with --include-api)")
    return out


# ------------------------------------------------------------------ resolution
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HubResolver:
    """The two hub calls the resolution needs, behind one object so tests can inject a fake:
    `model_info(hf_id)` returns an object with `.sha`, `.gated` and `.siblings` (each with
    `.rfilename`), `download(hf_id, filename, revision)` returns the local path of that file."""

    def __init__(self, token: str | None = None, cache_dir: str | None = None):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.token = token
        self.cache_dir = cache_dir

    def model_info(self, hf_id: str):
        return self.api.model_info(hf_id, token=self.token)

    def download(self, hf_id: str, filename: str, revision: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(hf_id, filename, revision=revision, token=self.token, cache_dir=self.cache_dir)


def classify_error(exc: BaseException) -> str:
    """found / not_found / gated / error from the hub's exception classes (by name when the hub
    library is not importable, so a fake resolver can raise look-alikes)."""
    names = {c.__name__ for c in type(exc).__mro__}
    if "GatedRepoError" in names:
        return STATUS_GATED
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return STATUS_NOT_FOUND
    return STATUS_ERROR


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]


def resolve_hf_id(hf_id: str, resolver) -> dict:
    """One hub repository -> {hf_id_status, revision, hub_gated, tokenizer_sha256, error, resolved_utc}."""
    rec = {"hf_id_status": STATUS_FOUND, "revision": None, "hub_gated": None, "tokenizer_sha256": {},
           "error": None, "resolved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    try:
        info = resolver.model_info(hf_id)
    except Exception as e:
        rec.update(hf_id_status=classify_error(e), error=_error_text(e))
        return rec
    sha = str(getattr(info, "sha", "") or "").strip().lower()
    if not COMMIT_HASH.fullmatch(sha):
        rec.update(hf_id_status=STATUS_ERROR, error=f"model_info returned no full commit hash (sha={sha!r})")
        return rec
    rec["revision"] = sha
    gated = getattr(info, "gated", None)
    rec["hub_gated"] = gated if isinstance(gated, (bool, str)) or gated is None else str(gated)
    siblings = getattr(info, "siblings", None)
    present = list(TOKENIZER_FILES)
    if siblings is not None:
        files = {getattr(s, "rfilename", None) for s in siblings}
        present = [f for f in TOKENIZER_FILES if f in files]
    for filename in present:
        try:
            path = resolver.download(hf_id, filename, sha)
        except Exception as e:
            kind = classify_error(e)
            if kind == STATUS_NOT_FOUND or any("EntryNotFoundError" in c.__name__ for c in type(e).__mro__):
                continue                                   # the file is not in the repository
            rec.update(hf_id_status=kind, error=_error_text(e))
            return rec
        rec["tokenizer_sha256"][filename] = sha256_file(Path(path))
    return rec


def resolve_entries(entries: list[dict], resolver, panel: set[str] | None = None) -> list[dict]:
    """One record per entry; the hub is asked once per distinct hf_id."""
    panel = panel or set()
    cache: dict[str, dict] = {}
    out = []
    for m in entries:
        hf_id = m["hf_id"]
        if hf_id not in cache:
            cache[hf_id] = resolve_hf_id(hf_id, resolver)
        out.append({"name": m["name"], "hf_id": hf_id, "group": m.get("group"), "backend": m.get("backend"),
                    "panel": m.get("group") in panel, "config_hf_id_status": m.get("hf_id_status"),
                    "config_revision": m.get("revision"), **cache[hf_id]})
    return out


def is_resolved(record: dict) -> bool:
    return bool(COMMIT_HASH.fullmatch(str(record.get("revision") or "")))


def write_pins(records: list[dict], out: Path, models_path: Path) -> dict:
    doc = {"models_config": _relative(models_path), "models_config_sha256": sha256_file(models_path),
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "huggingface_hub_version": _hub_version(),
           "n_records": len(records), "n_resolved": sum(is_resolved(r) for r in records),
           "n_unresolved_panel": sum(1 for r in records if r["panel"] and not is_resolved(r)),
           "records": records}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return doc


def read_pins(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    records = doc["records"] if isinstance(doc, dict) else doc
    if not isinstance(records, list) or not all(isinstance(r, dict) and "name" in r for r in records):
        raise ValueError(f"{path}: expected a `records` list of {{name, hf_id, revision, ...}}")
    return records


def _hub_version() -> str | None:
    try:
        import huggingface_hub

        return huggingface_hub.__version__
    except Exception:
        return None


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ apply
def already_pinned(value) -> bool:
    """True when `revision` is one of the DD 7.1 pin forms the runner accepts (a GGUF SHA-256 lives
    in `gguf_sha256`, not here)."""
    v = str(value or "").strip()
    return bool(COMMIT_HASH.fullmatch(v.lower()) or KAGGLE_SLUG.fullmatch(v))


def plan_apply(cfg: dict, records: list[dict], partial: bool = False, force: bool = False) -> dict:
    """Which entries get which hash, and why the apply would be refused. Returns {changes: {name:
    hash}, unchanged: [names already at the hash], unresolved: [names], unresolved_panel: [names],
    conflicts: {name: (current, resolved)}, refusals: [messages]}."""
    by_name = {r["name"]: r for r in records}
    panel = panel_groups(cfg)
    changes, unchanged, unresolved, unresolved_panel, conflicts = {}, [], [], [], {}
    for m in cfg.get("models") or []:
        if not is_self_hosted(m):
            continue
        rec = by_name.get(m["name"])
        if rec is None or not is_resolved(rec):
            unresolved.append(m["name"])
            if m.get("group") in panel:
                unresolved_panel.append(m["name"])
            continue
        if rec.get("hf_id") != m.get("hf_id"):
            conflicts[m["name"]] = (f"hf_id {m.get('hf_id')!r} != record's {rec.get('hf_id')!r}", rec["revision"])
            continue
        new = rec["revision"].lower()
        cur = m.get("revision")
        if str(cur or "").strip().lower() == new:
            unchanged.append(m["name"])
        elif already_pinned(cur) and not force:
            conflicts[m["name"]] = (cur, new)
        else:
            changes[m["name"]] = new
    refusals = []
    if unresolved_panel and not partial:
        refusals.append(f"{len(unresolved_panel)} panel entries are unresolved: {unresolved_panel}; "
                        "resolve them (or pass --partial to pin only what resolved)")
    if conflicts:
        refusals.append("already pinned to something else (pass --force to overwrite): "
                        + "; ".join(f"{n}: {cur!r} -> {new}" for n, (cur, new) in conflicts.items()))
    return {"changes": changes, "unchanged": unchanged, "unresolved": unresolved, "unresolved_panel": unresolved_panel,
            "conflicts": conflicts, "refusals": refusals}


def rewrite_revisions(text: str, changes: dict[str, str]) -> str:
    """Replace the value of the `revision:` line of every named entry, keeping everything else
    (indentation, trailing comment, the flow mapping around it) byte for byte."""
    out, seen, current = [], Counter(), None
    for line in text.splitlines(keepends=True):
        m = _FLOW_NAME.match(line)
        if m:
            current = None                               # a flow entry is complete on its line
            name = m.group("name")
            if name in changes:
                line, n = _FLOW_REV.subn(rf'\g<lead>"{changes[name]}"', line, count=1)
                seen[name] += n
            out.append(line)
            continue
        m = _BLOCK_NAME.match(line)
        if m:
            current = m.group("name")
            out.append(line)
            continue
        if current in changes:
            body = line.rstrip("\r\n")
            m = _BLOCK_REV.match(body)
            if m:
                line = f'{m.group("lead")}"{changes[current]}"{m.group("rest")}{line[len(body):]}'
                seen[current] += 1
        out.append(line)
    bad = {n: seen[n] for n in changes if seen[n] != 1}
    if bad:
        raise ValueError(f"could not rewrite exactly one `revision:` line for {bad} (occurrences per entry)")
    return "".join(out)


def verify_rewrite(old_cfg: dict, new_text: str, changes: dict[str, str]) -> None:
    """The rewritten text parses to the old configuration with only the requested revisions changed."""
    new_cfg = yaml.safe_load(new_text)
    old_rest = {k: v for k, v in old_cfg.items() if k != "models"}
    new_rest = {k: v for k, v in new_cfg.items() if k != "models"}
    if old_rest != new_rest:
        raise ValueError("rewrite changed something outside `models`")
    old_models, new_models = old_cfg.get("models") or [], new_cfg.get("models") or []
    if len(old_models) != len(new_models):
        raise ValueError("rewrite changed the number of model entries")
    for o, n in zip(old_models, new_models):
        if {k: v for k, v in o.items() if k != "revision"} != {k: v for k, v in n.items() if k != "revision"}:
            raise ValueError(f"rewrite changed a field other than `revision` of {o.get('name')!r}")
        want = changes.get(o.get("name"), o.get("revision"))
        if n.get("revision") != want:
            raise ValueError(f"entry {o.get('name')!r}: revision {n.get('revision')!r}, expected {want!r}")


def apply_pins(models_path: Path, records: list[dict], partial: bool = False, force: bool = False,
               dry_run: bool = False) -> dict:
    """Plan, refuse or rewrite models.yaml in place. Returns the plan plus `written` (bool)."""
    text = models_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    plan = plan_apply(cfg, records, partial=partial, force=force)
    plan["written"] = False
    if plan["refusals"] or dry_run:
        return plan
    new_text = rewrite_revisions(text, plan["changes"])
    verify_rewrite(cfg, new_text, plan["changes"])
    models_path.write_text(new_text, encoding="utf-8")
    plan["written"] = True
    return plan


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-config", type=Path, default=MODELS_FILE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the resolution JSON goes")
    ap.add_argument("--names", nargs="*", default=None, help="resolve only these entries")
    ap.add_argument("--include-api", action="store_true",
                    help="also resolve API entries that carry an hf_id (tokenizer record only; never applied)")
    ap.add_argument("--token-env", default="HF_TOKEN", help="environment variable holding the hub token")
    ap.add_argument("--cache-dir", default=None, help="hf_hub_download cache directory")
    ap.add_argument("--apply", action="store_true", help="write the resolved hashes into --models-config")
    ap.add_argument("--from", dest="from_file", type=Path, default=None,
                    help="with --apply: read the records from this JSON instead of resolving (no network)")
    ap.add_argument("--partial", action="store_true", help="with --apply: allow unresolved panel entries")
    ap.add_argument("--force", action="store_true", help="with --apply: overwrite an entry pinned to another value")
    ap.add_argument("--dry-run", action="store_true", help="with --apply: report the plan, write nothing")
    args = ap.parse_args(argv)

    cfg = load_models(args.models_config)
    if args.apply and args.from_file is not None:
        records = read_pins(args.from_file)
    else:
        import os

        entries = select_entries(cfg, include_api=args.include_api, names=args.names)
        resolver = HubResolver(token=os.environ.get(args.token_env) or None, cache_dir=args.cache_dir)
        records = resolve_entries(entries, resolver, panel=panel_groups(cfg))
        for r in records:
            print(json.dumps({k: r.get(k) for k in ("name", "hf_id", "hf_id_status", "revision", "error")}))
        doc = write_pins(records, args.out, args.models_config)
        print(f"wrote {args.out}: {doc['n_resolved']}/{doc['n_records']} resolved, "
              f"{doc['n_unresolved_panel']} panel entries unresolved")
        if not args.apply:
            return 0 if doc["n_unresolved_panel"] == 0 else 1

    plan = apply_pins(args.models_config, records, partial=args.partial, force=args.force, dry_run=args.dry_run)
    summary = {k: plan[k] for k in ("changes", "unchanged", "unresolved", "unresolved_panel", "conflicts", "written")}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for msg in plan["refusals"]:
        print(f"refused: {msg}", file=sys.stderr)
    if plan["refusals"]:
        return 1
    if args.dry_run:
        print(f"dry run: {len(plan['changes'])} entries would be pinned in {args.models_config}")
    else:
        print(f"pinned {len(plan['changes'])} entries in {args.models_config} "
              f"({len(plan['unchanged'])} already at their hash, {len(plan['unresolved'])} left unresolved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
