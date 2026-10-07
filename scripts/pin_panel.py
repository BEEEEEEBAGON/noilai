#!/usr/bin/env python
"""Pin the self-hosted model panel: resolve the hub repository each entry of configs/models.yaml
loads to a full commit hash, record the SHA-256 of its tokenizer files and chat template and its
licence, and write the hashes back into the `revision:` lines (DESIGN_DECISIONS 7.1: the runner
refuses an unpinned self-hosted entry).

    python scripts/pin_panel.py --out experiments/panel_pins.json          # resolve (needs hub access)
    python scripts/pin_panel.py --apply --from experiments/panel_pins.json # offline: write the pins
    python scripts/pin_panel.py --apply --from ... --dry-run               # show what would change
    python scripts/pin_panel.py --apply --from ... --partial               # pin what resolved, keep the rest null
    python scripts/pin_panel.py --apply --from ... --force                 # overwrite a different existing pin
    python scripts/pin_panel.py --names gemma-3-1b-it qwen3.5-2b --out ... # a subset
    python scripts/pin_panel.py --names <chunk models> --apply --partial --out <path>
                                        # a CPU chunk: resolve its models, then pin them in the clone

Which entries: every entry whose `backend` downloads weights (hf, vllm, llama_cpp: the
LOCAL_WEIGHT_BACKENDS of noilai.eval.run) and that carries an `hf_id`, i.e. the self-hosted
members of the seven panel groups plus their `bf16_reference` and `reasoning_substudy` serving
variants. API entries with an open-weight `hf_id` (gpt-oss on Groq) are resolved only with
--include-api, for the tokenizer record; they have no `revision` line and are never written.
A name in --names that the config knows but that is not resolvable (an API entry without
--include-api) is skipped with a message on stderr; a name the config does not know is an error.

Which repository: the one the runner loads (noilai.eval.backends: `quantization.checkpoint` or
`hf_id`, at the entry's `revision`). An entry that names a pre-quantized checkpoint (AWQ, GPTQ,
QAT int4) is therefore resolved against that checkpoint, and its bf16 serving variant against
the base `hf_id`: two hashes from two repositories although they share an `hf_id`. The hub is
asked once per distinct repository. An entry whose pre-quantized checkpoint is not chosen yet
(`quantization.method` set, `checkpoint` null, the method not bitsandbytes, which quantizes the
base weights at load) gets the status `checkpoint_unchosen` and no hash, because pinning the
base repository would pin the wrong weights; it counts as unresolved (a panel entry refuses the
apply unless --partial) whatever a pins file says about it.

What is recorded, one record per entry, in the JSON (`records`): `name`, `hf_id`, `checkpoint`
(the repository resolved), `group`, `backend`, `panel`, the config's own `hf_id_status`
(`config_hf_id_status`: known/uncertain) and `config_revision`, the resolution outcome
`hf_id_status` (found / not_found / gated / error / checkpoint_unchosen), `revision` (the
40-hex commit hash of the repository's main branch at `resolved_utc`), `hub_gated` (the hub's
gating flag), `license` ({card, license_name, tags}: the model card's `license` and
`license_name` fields and the hub's `license:*` tags), `last_modified` (the hub's date),
`tokenizer_sha256` (one SHA-256 per tokenizer file present in the repository: tokenizer.model,
tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja; downloaded
with hf_hub_download at the pinned revision and hashed), `chat_template_sha256` (SHA-256 of
chat_template.jinja when the repository ships one, else of tokenizer_config.json's
`chat_template`; a list of named templates is hashed as canonical JSON; DD 7.5's second
manifest identity) and `error`. The document also carries `models_config` and its SHA-256,
`env_pins_sha256` (configs/env_pins.txt, the run-environment pins; null when absent),
`huggingface_hub_version`, `written_utc` and the counts `n_records`, `n_resolved`,
`n_unresolved_panel`, `n_checkpoint_unchosen`. A gated repository whose licence the token has
not accepted still resolves its hash (the metadata is public) and reports `gated` with empty
tokenizer hashes.

--apply (text-preserving): only the value of each entry's `revision:` line changes, written as
a double-quoted scalar (a hash made of digits only would otherwise parse as an integer);
comments, order and every other field stay, and the rewritten file is re-parsed and compared
field by field with the original before it is written (`hf_id_status` is never flipped: the
author verifies the uncertain ids by hand, DD 7.1). It refuses when any PANEL entry is
unresolved unless --partial, and when an entry is already pinned (a full hash or a Kaggle
Models slug) to something other than the resolved hash unless --force; an entry already pinned
to the resolved hash is left as it is. A record whose `hf_id` or `checkpoint` is not the
entry's (a stale pins file after an id fix or a checkpoint choice) is a conflict, never applied.
--apply without --from resolves first, then applies (the Kaggle CPU chunks do this in the clone).

Exit codes: 0 when every selected panel entry resolved (resolution) or the file was written
(apply); 1 when the resolution is incomplete (the JSON is still written), when apply refuses,
or when the rewrite does not verify; 2 for a name the config does not know.

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
ENV_PINS_FILE = ROOT / "configs" / "env_pins.txt"
DEFAULT_OUT = ROOT / "experiments" / "panel_pins.json"
# DESIGN_DECISIONS 7.1: the backends that download weights and therefore need a pinned revision
# (kept equal to noilai.eval.run.LOCAL_WEIGHT_BACKENDS; tests/test_pin_panel.py checks that)
LOCAL_WEIGHT_BACKENDS = ("hf", "vllm", "llama_cpp")
# quantization methods applied to the base weights at load (noilai.eval.backends' bitsandbytes set; models.yaml
# also spells them bitsandbytes_<type>): the base repository is what the runner downloads, so it is what gets pinned
INFLIGHT_QUANT_METHODS = ("bitsandbytes", "bnb", "nf4", "int4", "int8")
# the files whose SHA-256 identifies a tokenizer (DD 7.5: `tokenizer_sha256` and `chat_template_sha256` in every
# manifest); hashed only when the repository has them
TOKENIZER_FILES = ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja")
STATUS_FOUND, STATUS_NOT_FOUND, STATUS_GATED, STATUS_ERROR = "found", "not_found", "gated", "error"
STATUS_UNCHOSEN = "checkpoint_unchosen"
UNCHOSEN_NOTE = "quantized checkpoint not chosen (quantization.checkpoint is null): choose it, then pin it"
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


def checkpoint_of(entry: dict) -> str:
    """The repository the runner loads: `quantization.checkpoint` when the entry names a pre-quantized
    checkpoint, else `hf_id` (noilai.eval.backends loads `q.get("checkpoint") or model_id` at `revision`)."""
    q = entry.get("quantization") or {}
    return q.get("checkpoint") or entry["hf_id"]


def quantized_checkpoint_missing(entry: dict) -> bool:
    """A quantized entry (AWQ / GPTQ / QAT int4) whose pre-quantized checkpoint has not been chosen yet: pinning
    the base repository would pin the wrong weights. Bitsandbytes quantizes the base weights at load, so an
    entry using it loads (and pins) its `hf_id`."""
    q = entry.get("quantization") or {}
    method = str(q.get("method") or "").strip().lower()
    if not method or q.get("checkpoint"):
        return False
    return method not in INFLIGHT_QUANT_METHODS and not method.startswith(("bitsandbytes", "bnb"))


def select_entries(cfg: dict, include_api: bool = False, names: list[str] | None = None) -> list[dict]:
    """The entries to resolve: the self-hosted ones, plus API entries with an hf_id when asked. With `names`
    only those; a name the config does not know is a KeyError, a known but unresolvable one is left out
    (skipped_names() says why)."""
    models = cfg.get("models") or []
    if names is not None:
        unknown = sorted(set(names) - {m.get("name") for m in models})
        if unknown:
            raise KeyError(f"no entry named {unknown} in the models config")
    return [m for m in models
            if (names is None or m.get("name") in names) and (is_self_hosted(m) or (include_api and m.get("hf_id")))]


def skipped_names(cfg: dict, names: list[str], include_api: bool = False) -> dict[str, str]:
    """The names select_entries() leaves out although the config knows them, with the reason."""
    selected = {m.get("name") for m in select_entries(cfg, include_api=include_api, names=names)}
    out = {}
    for m in cfg.get("models") or []:
        name = m.get("name")
        if name not in names or name in selected:
            continue
        if not m.get("hf_id"):
            out[name] = f"no hf_id (backend {m.get('backend')!r}): nothing to pin"
        else:
            out[name] = (f"API entry (backend {m.get('backend')!r}): no revision line to pin; "
                         "--include-api records its tokenizer")
    return out


# ------------------------------------------------------------------ resolution
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class HubResolver:
    """The two hub calls the resolution needs, behind one object so tests can inject a fake:
    `model_info(repo_id)` returns an object with `.sha`, `.gated`, `.card_data`, `.tags`, `.last_modified`
    and `.siblings` (each with `.rfilename`); `download(repo_id, filename, revision)` returns the local
    path of that file."""

    def __init__(self, token: str | None = None, cache_dir: str | None = None):
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.token = token
        self.cache_dir = cache_dir

    def model_info(self, repo_id: str):
        return self.api.model_info(repo_id, token=self.token)

    def download(self, repo_id: str, filename: str, revision: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id, filename, revision=revision, token=self.token, cache_dir=self.cache_dir)


def classify_error(exc: BaseException) -> str:
    """found / not_found / gated / error from the hub's exception classes (by name when the hub
    library is not importable, so a fake resolver can raise look-alikes)."""
    names = {c.__name__ for c in type(exc).__mro__}
    if "GatedRepoError" in names:
        return STATUS_GATED
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return STATUS_NOT_FOUND
    return STATUS_ERROR


def _is_absent_file(exc: BaseException) -> bool:
    """hf_hub_download's "the repository has no such file" (EntryNotFoundError / RemoteEntryNotFoundError);
    a LocalEntryNotFoundError is the opposite case, the hub could not be reached and the cache has no copy,
    so it stays an error rather than an empty tokenizer record."""
    names = {c.__name__ for c in type(exc).__mro__}
    return "EntryNotFoundError" in names and "LocalEntryNotFoundError" not in names


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]


def _license_of(info) -> dict:
    """{card, license_name, tags}: the model card's `license` / `license_name` and the hub's `license:*` tags."""
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None) or {}
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    if not isinstance(card, dict):
        card = {}
    tags = [str(t) for t in (getattr(info, "tags", None) or [])]
    return {"card": card.get("license"), "license_name": card.get("license_name"),
            "tags": [t for t in tags if t.startswith("license:")]}


def _last_modified_of(info) -> str | None:
    value = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def chat_template_sha256(jinja_path: Path | None, tokenizer_config_path: Path | None) -> str | None:
    """SHA-256 of the chat template text: chat_template.jinja when the repository ships one (transformers
    prefers it), else tokenizer_config.json's `chat_template`; a list of named templates is hashed as
    canonical JSON (sorted keys, no spaces). None when the repository has neither."""
    template = None
    if jinja_path is not None:
        template = Path(jinja_path).read_text(encoding="utf-8")
    elif tokenizer_config_path is not None:
        try:
            template = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8")).get("chat_template")
        except (ValueError, AttributeError):
            template = None
    if isinstance(template, (list, dict)):
        template = json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not template:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def _blank_record() -> dict:
    return {"hf_id_status": STATUS_FOUND, "revision": None, "hub_gated": None, "license": None, "last_modified": None,
            "tokenizer_sha256": {}, "chat_template_sha256": None, "error": None, "resolved_utc": _now()}


def resolve_repo(repo_id: str, resolver) -> dict:
    """One hub repository -> {hf_id_status, revision, hub_gated, license, last_modified, tokenizer_sha256,
    chat_template_sha256, error, resolved_utc}."""
    rec = _blank_record()
    try:
        info = resolver.model_info(repo_id)
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
    rec["license"] = _license_of(info)
    rec["last_modified"] = _last_modified_of(info)
    siblings = getattr(info, "siblings", None)
    present = list(TOKENIZER_FILES)
    if siblings is not None:
        files = {getattr(s, "rfilename", None) for s in siblings}
        present = [f for f in TOKENIZER_FILES if f in files]
    paths: dict[str, Path] = {}
    for filename in present:
        try:
            path = resolver.download(repo_id, filename, sha)
        except Exception as e:
            kind = classify_error(e)
            if kind == STATUS_NOT_FOUND or _is_absent_file(e):
                continue                                   # the file is not in the repository
            rec.update(hf_id_status=kind, error=_error_text(e))
            return rec
        paths[filename] = Path(path)
        rec["tokenizer_sha256"][filename] = sha256_file(paths[filename])
    rec["chat_template_sha256"] = chat_template_sha256(paths.get("chat_template.jinja"), paths.get("tokenizer_config.json"))
    return rec


def resolve_entries(entries: list[dict], resolver, panel: set[str] | None = None) -> list[dict]:
    """One record per entry; the hub is asked once per distinct repository (checkpoint_of); an entry whose
    pre-quantized checkpoint is not chosen gets STATUS_UNCHOSEN, no hash and no hub call."""
    panel = panel or set()
    cache: dict[str, dict] = {}
    out = []
    for m in entries:
        repo = checkpoint_of(m)
        if quantized_checkpoint_missing(m):
            res = _blank_record()
            res.update(hf_id_status=STATUS_UNCHOSEN, error=UNCHOSEN_NOTE)
        else:
            if repo not in cache:
                cache[repo] = resolve_repo(repo, resolver)
            res = cache[repo]
        out.append({"name": m["name"], "hf_id": m["hf_id"], "checkpoint": repo, "group": m.get("group"),
                    "backend": m.get("backend"), "panel": m.get("group") in panel,
                    "config_hf_id_status": m.get("hf_id_status"), "config_revision": m.get("revision"), **res})
    return out


def is_resolved(record: dict) -> bool:
    return bool(COMMIT_HASH.fullmatch(str(record.get("revision") or "")))


def write_pins(records: list[dict], out: Path, models_path: Path, env_pins_path: Path = ENV_PINS_FILE) -> dict:
    env_pins_path = Path(env_pins_path)
    doc = {"models_config": _relative(models_path), "models_config_sha256": sha256_file(models_path),
           "env_pins_sha256": sha256_file(env_pins_path) if env_pins_path.exists() else None,
           "written_utc": _now(), "huggingface_hub_version": _hub_version(),
           "n_records": len(records), "n_resolved": sum(is_resolved(r) for r in records),
           "n_unresolved_panel": sum(1 for r in records if r["panel"] and not is_resolved(r)),
           "n_checkpoint_unchosen": sum(1 for r in records if r.get("hf_id_status") == STATUS_UNCHOSEN),
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
    checkpoint_unchosen: [names whose pre-quantized checkpoint is not chosen; always unresolved],
    conflicts: {name: (current, resolved)}, refusals: [messages]}."""
    by_name = {r["name"]: r for r in records}
    panel = panel_groups(cfg)
    changes, unchanged, unresolved, unresolved_panel, unchosen, conflicts = {}, [], [], [], [], {}
    for m in cfg.get("models") or []:
        if not is_self_hosted(m):
            continue
        rec = by_name.get(m["name"])
        if quantized_checkpoint_missing(m):
            unchosen.append(m["name"])
        if rec is None or not is_resolved(rec) or quantized_checkpoint_missing(m):
            unresolved.append(m["name"])
            if m.get("group") in panel:
                unresolved_panel.append(m["name"])
            continue
        if rec.get("hf_id") != m.get("hf_id"):
            conflicts[m["name"]] = (f"hf_id {m.get('hf_id')!r} != record's {rec.get('hf_id')!r}", rec["revision"])
            continue
        want, got = checkpoint_of(m), rec.get("checkpoint") or rec.get("hf_id")   # a record without `checkpoint`
        if got != want:                                                           # was resolved against its hf_id
            conflicts[m["name"]] = (f"checkpoint {want!r} != record's {got!r}", rec["revision"])
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
        msg = (f"{len(unresolved_panel)} panel entries are unresolved: {unresolved_panel}; "
               "resolve them (or pass --partial to pin only what resolved)")
        waiting = [n for n in unresolved_panel if n in unchosen]
        if waiting:
            msg += f"; {len(waiting)} of them wait for a quantization.checkpoint: {waiting}"
        refusals.append(msg)
    if conflicts:
        refusals.append("already pinned to something else (pass --force to overwrite): "
                        + "; ".join(f"{n}: {cur!r} -> {new}" for n, (cur, new) in conflicts.items()))
    return {"changes": changes, "unchanged": unchanged, "unresolved": unresolved, "unresolved_panel": unresolved_panel,
            "checkpoint_unchosen": unchosen, "conflicts": conflicts, "refusals": refusals}


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
    ap.add_argument("--names", nargs="+", default=None,
                    help="resolve only these entries (a known but unresolvable name, e.g. an API model, is skipped)")
    ap.add_argument("--include-api", action="store_true",
                    help="also resolve API entries that carry an hf_id (tokenizer record only; never applied)")
    ap.add_argument("--token-env", default="HF_TOKEN", help="environment variable holding the hub token")
    ap.add_argument("--cache-dir", default=None, help="hf_hub_download cache directory")
    ap.add_argument("--apply", action="store_true",
                    help="write the resolved hashes into --models-config (after resolving, unless --from)")
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

        try:
            entries = select_entries(cfg, include_api=args.include_api, names=args.names)
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 2
        if args.names:
            for name, reason in skipped_names(cfg, args.names, include_api=args.include_api).items():
                print(f"skipped {name}: {reason}", file=sys.stderr)
        resolver = HubResolver(token=os.environ.get(args.token_env) or None, cache_dir=args.cache_dir)
        records = resolve_entries(entries, resolver, panel=panel_groups(cfg))
        for r in records:
            print(json.dumps({k: r.get(k) for k in ("name", "hf_id", "checkpoint", "hf_id_status", "revision", "error")}))
        doc = write_pins(records, args.out, args.models_config)
        print(f"wrote {args.out}: {doc['n_resolved']}/{doc['n_records']} resolved, "
              f"{doc['n_unresolved_panel']} panel entries unresolved "
              f"({doc['n_checkpoint_unchosen']} wait for a quantization.checkpoint)")
        if not args.apply:
            return 0 if doc["n_unresolved_panel"] == 0 else 1

    plan = apply_pins(args.models_config, records, partial=args.partial, force=args.force, dry_run=args.dry_run)
    summary = {k: plan[k] for k in ("changes", "unchanged", "unresolved", "unresolved_panel", "checkpoint_unchosen",
                                    "conflicts", "written")}
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
