#!/usr/bin/env python
"""Turn configs/run_plan.yaml into `scripts/run_eval.py` commands, run them, and log the time.

The notebooks' RUN cells call this so that the flag mapping lives in one tested place:

    python scripts/kaggle_run_plan.py --list
    python scripts/kaggle_run_plan.py --materialize                              # write every derived (seeded) item file
    python scripts/kaggle_run_plan.py --verify E1_main E1_attested                # item-file gate for these runs
    python scripts/kaggle_run_plan.py --run E1_main --dry-run                     # print every command
    python scripts/kaggle_run_plan.py --run E1_main --models gemma-3-1b-it qwen3.5-2b \
        --platform kaggle --session-hardware 2xt4 --max-session-hours 9         # execute, log hours
    python scripts/kaggle_run_plan.py --run E1_api_core --platform api            # quota-aware API loop

For each (run, model) the command is

    <python> scripts/run_eval.py --model-config <name> --items <path> --tasks ... --variants ...
        --paraphrases ... --shots N --arms ... [--input-format F] [--instruction I] [--language L]
        [--smoke --limit N] [--in-core-only] --resume --run-id <run_id>__<name> --out-root data/runs [extra args]

The output flags follow `run_eval_out_style` in run_plan.yaml: "run_id_root" (default; the CLI
as landed writes data/runs/<run-id>/) or "out" (the first specification's `--out <dir>`).
The run directory is data/runs/<run_id>__<name> either way.

Guards (raise before anything runs): an API-served model must run with --in-core-only, and an
item file flagged `never_to_api` never reaches an API backend (plan 2.3: APIs see only the
core; the sealed split reaches no API); a canary-bearing item file (`canary_required`, i.e. a
test-split file) never reaches a provider whose `terms.trains_on_inputs` is not exactly false
(DESIGN_DECISIONS 11.2 / 12.41, item 33: Gemini's unpaid tier), unless the provider block
records `terms.paid_key_data_use_opt_out: true`, in which case the command carries
--core-to-training-provider-opt-out and the manifest records it; `limit` is accepted on
`experiment: smoke` lines only (DESIGN_DECISIONS 12.32: every other sub-sample is a seeded
file); a run flagged `not_a_run_eval_line` (E4) is refused. `execute` reports a guarded
(run, model) as "refused: ..." and goes on with the next model, so a mixed model set (Groq
and Gemini) still runs its permitted members; the CLI exits 1 when anything was refused.

Revision pins (DESIGN_DECISIONS 7.1): a self-hosted model whose `revision` is not a full
commit hash, a Kaggle Models slug + version or a GGUF SHA-256 is skipped on every non-smoke
line ("skipped: revision not pinned"), mirroring run_eval.py's own refusal, unless
`allow_unpinned_revision`; dry runs still print the command.

Item files. `{release}` in a plan path is the plan's `release` directory. A file with a
`derive` block is a seeded sub-sample of another item file (per task x variant cell counts,
T3 as yes/no pairs, the core kept whole, vulgar rows excluded, balanced over source and output
lexicality like scripts/sample_items.py); `derive_item_file` / `--materialize` write it
deterministically, `verify_run_items` checks every run's file through kaggle_verify_items
before the first command (hash, header record, canary, counts).

Session guards. `session_hardware` (the notebook's accelerator: t4, 2xt4, tpu, l4, api) skips a
model whose configured hardware cannot run in the session unless `allow_hardware_mismatch`;
the compute-log row always records the SESSION's device type and count. `max_session_hours`
stops launching new models past the budget (DD 7.6: <= 9 h of work per 12-h Kaggle session);
`after_each` (a callable) runs after every executed model, e.g. a dataset push, so that a
timed-out session keeps what it finished.

Accounting: every executed command appends TWO compute-log rows through scripts/compute_log.py:
a provisional 0-hour row (`purpose = "<run_id> started HH:MMZ"`) before the child is launched,
so that a session killed mid-model leaves its start on record, and the final row with the wall
hours afterwards (a provisional row without a final row means the session died; its hours are
reconstructed from the platform's session log). API models keep a daily ledger
(data/runs/api_ledger.json: requests AND tokens per model per UTC day, counted from the new
rows of outputs.jsonl; `n_prompt_tokens + n_output_tokens`, or the plan's
`assumed_tokens_per_call` when the provider returned no usage). Under `scope: per_model` the
budget pools every models.yaml entry that names the same (provider, provider_model_id), so a
thinking variant shares its base entry's daily allowance. A model whose daily request or token
budget is spent is skipped until tomorrow; a running command is stopped (SIGINT) by a watchdog
the moment the day's cap is reached and reported as "parked", never as failed; `--resume`
continues it the next day.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compute_log as CL
import kaggle_verify_items as KVI

CONFIGS = ROOT / "configs"
API_BACKENDS = {"openai_compat", "gemini"}
# hardware key of models.yaml -> (compute-log device type, device count)
HARDWARE_DEVICES = {"t4": ("t4", 1), "2xt4": ("t4", 2), "tpu": ("tpu-v5e-8", 1), "api": ("api", 0),
                    "p100": ("p100", 1), "l4": ("l4", 1), "cpu": ("cpu", 0), "mixed": (None, None)}
# "cpu": a Kaggle CPU session (no GPU quota). No entry is configured for it: a CPU run passes
# --allow-hardware-mismatch and `--backend hf --device cpu --dtype float32` (notebooks/kaggle_cpu_pilot.ipynb),
# and the compute log records device "cpu" x 0 accelerators, i.e. zero GPU-hours.
# entry hardware -> the sessions that can host it (a 1xT4 entry runs on a 2xT4 session; nothing else crosses)
SESSION_COMPATIBLE = {"t4": {"t4", "2xt4"}, "2xt4": {"2xt4"}, "tpu": {"tpu"}, "l4": {"l4"}, "p100": {"p100"},
                      "api": {"t4", "2xt4", "tpu", "l4", "p100", "api", "cpu"}}
ABLATION_FLAGS = (("input_format", "--input-format"), ("instruction", "--instruction"), ("language", "--language"),
                  ("arm_scope", "--arm-scope"), ("backend", "--backend"))
TASKS = ("T1", "T2", "T3")
VARIANTS = ("V1", "V2", "V3", "V4")
PARKED = "parked"
REFUSED = "refused"
OPT_OUT_FLAG = "--core-to-training-provider-opt-out"
# DESIGN_DECISIONS 7.1: a full HF commit hash, a Kaggle Models slug + version, or a GGUF SHA-256
# (a Kaggle Models slug is owner/model/framework/variation/version; the framework segment is optional in older slugs)
REVISION_PATTERNS = (re.compile(r"^[0-9a-f]{40}$"), re.compile(r"^[\w.-]+(?:/[\w.-]+){2,3}/\d+$"), re.compile(r"^[0-9a-f]{64}$"))


# ------------------------------------------------------------------ loading
def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plan(path: Path = CONFIGS / "run_plan.yaml") -> dict:
    """run_plan.yaml with `{release}` resolved in every item-file path (kaggle_verify_items.load_plan)."""
    return KVI.load_plan(path)


def load_models(path: Path = CONFIGS / "models.yaml") -> dict:
    """models.yaml with an added `by_name` index."""
    cfg = load_yaml(path)
    cfg["by_name"] = {m["name"]: m for m in cfg["models"]}
    return cfg


def find_run(plan: dict, run_id: str) -> dict:
    for r in plan["runs"]:
        if r["id"] == run_id:
            return r
    raise KeyError(f"no run {run_id!r}; have {[r['id'] for r in plan['runs']]}")


def item_path(plan: dict, key: str, project_root: Path | None = None) -> Path:
    spec = plan["item_files"][key]
    p = Path(spec["path"])
    if project_root is not None and not p.is_absolute():
        p = Path(project_root) / p
    return p


def item_keys_for_runs(plan: dict, run_ids: Sequence[str]) -> list[str]:
    """The item-file keys the given runs read, in order, without duplicates (for the verify cell)."""
    keys: list[str] = []
    for rid in run_ids:
        k = find_run(plan, rid)["items"]
        if k not in keys:
            keys.append(k)
    return keys


# ------------------------------------------------------------------ model sets
def expand_set(name: str, plan: dict, models_cfg: dict, _seen: set | None = None) -> list[str]:
    """A model set is a list of names, {union: [sets]} or {group: <models.yaml group>}."""
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"model_sets cycle at {name!r}")
    _seen.add(name)
    sets = plan.get("model_sets", {})
    if name not in sets:
        raise KeyError(f"unknown model set {name!r}")
    spec = sets[name]
    if isinstance(spec, list):
        out = list(spec)
    elif "union" in spec:
        out = []
        for s in spec["union"]:
            out.extend(x for x in expand_set(s, plan, models_cfg, _seen) if x not in out)
    elif "group" in spec:
        out = [m["name"] for m in models_cfg["models"] if m.get("group") == spec["group"]]
    else:
        raise ValueError(f"model set {name!r}: expected a list, {{union: [...]}} or {{group: ...}}")
    unknown = [m for m in out if m not in models_cfg["by_name"]]
    if unknown:
        raise KeyError(f"model set {name!r} names unknown models {unknown}")
    return out


def expand_models(run: dict, plan: dict, models_cfg: dict) -> list[str]:
    spec = run["models"]
    if isinstance(spec, dict) and "set" in spec:
        return expand_set(spec["set"], plan, models_cfg)
    if isinstance(spec, list):
        unknown = [m for m in spec if m not in models_cfg["by_name"]]
        if unknown:
            raise KeyError(f"run {run['id']!r} names unknown models {unknown}")
        return list(spec)
    raise ValueError(f"run {run['id']!r}: models must be a list or {{set: name}}")


# ------------------------------------------------------------------ commands
def run_dir(plan: dict, run: dict, model: str) -> str:
    suffix = f"__{run['chunk_tag']}" if run.get("chunk_tag") else ""
    return f"{plan.get('runs_root', 'data/runs')}/{run['id']}__{model}{suffix}"


NARROWABLE = ("paraphrases", "arms", "tasks", "variants")


def narrow_run(run: dict, overrides: dict | None, tag: str | None = None) -> dict:
    """A compute chunk's view of a run line (docs/COMPUTE_PLAN.md): it may only NARROW the line's paraphrases, arms,
    tasks or variants (never add one), and writes to its own run directory `<run>__<model>__<tag>` so that a manifest
    always describes exactly the rows beside it. The analysis pools the directories of a line by its run id. A tag
    without overrides keeps the line whole in a directory of its own (the CPU notebook's `cpu` tag: an engine apart)."""
    if not overrides:
        return run if not tag else {**run, "chunk_tag": tag}
    bad = set(overrides) - set(NARROWABLE)
    if bad:
        raise ValueError(f"a chunk may only narrow {NARROWABLE}, not {sorted(bad)}")
    out = dict(run)
    for k, v in overrides.items():
        wider = set(v) - set(run.get(k) or [])
        if wider:
            raise ValueError(f"chunk override widens {k} of {run['id']!r} by {sorted(wider)}")
        if not v:
            raise ValueError(f"chunk override empties {k} of {run['id']!r}")
        out[k] = list(v)
    out["chunk_tag"] = tag or "-".join(f"{k}-{'+'.join(v)}" for k, v in sorted(overrides.items()))
    return out


def is_api(entry: dict) -> bool:
    return entry.get("backend") in API_BACKENDS or entry.get("hardware") == "api"


def provider_terms(entry: dict, models_cfg: dict) -> dict:
    """The `terms` block of the entry's provider ({} when there is none)."""
    prov = entry.get("provider")
    block = (models_cfg.get("providers") or {}).get(prov) or {}
    terms = block.get("terms")
    return terms if isinstance(terms, dict) else {}


def check_training_provider(run: dict, entry: dict, plan: dict, models_cfg: dict) -> bool:
    """DESIGN_DECISIONS 11.2 / 12.41 (item 33): a canary-bearing (test-split) item file never reaches a
    provider whose tier trains on inputs or whose terms are not cleared. Returns True when the
    command must carry --core-to-training-provider-opt-out (a recorded paid-key opt-out), False when
    nothing is needed; raises ValueError when the combination is refused."""
    item_spec = plan["item_files"][run["items"]]
    if not (is_api(entry) and item_spec.get("canary_required")):
        return False
    terms = provider_terms(entry, models_cfg)
    trains = terms.get("trains_on_inputs")
    if trains is False:
        return False
    if terms.get("paid_key_data_use_opt_out") is True:
        return True
    state = "trains on inputs" if trains is True else f"has no cleared no-training terms (trains_on_inputs={trains!r})"
    raise ValueError(f"item file {run['items']!r} carries the canary (test split) and provider {entry.get('provider')!r} "
                     f"of {entry['name']!r} {state}: a provider whose tier trains on inputs never receives the core "
                     f"(DESIGN_DECISIONS 11.2 / 12.41, item 33). Either record a paid key with a verified data-use "
                     f"opt-out (`providers.<name>.terms.paid_key_data_use_opt_out: true`) or run the dev-derived API set "
                     f"(noilai_api_dev, DD 4.5) on a line of its own; the author decides (DD 13.18)")


def revision_pinned(entry: dict) -> bool:
    """DESIGN_DECISIONS 7.1: True for an API entry, or a self-hosted entry whose `revision` is a full
    HF commit hash, a Kaggle Models slug + version or a GGUF SHA-256."""
    if is_api(entry):
        return True
    rev = entry.get("revision")
    return isinstance(rev, str) and any(p.fullmatch(rev.strip()) for p in REVISION_PATTERNS)


# DESIGN_DECISIONS 6.2: for a tokenizer whose census verdict for an arm is "normalizes" (identical token ids), that arm
# is not run: its effect is 0 by construction and the analysis emits a "0 by construction" row (DEVIATIONS 1 Oct 2026:
# the runner used to run these arms anyway, ~2/5 of E3 on the normalizing families).
CENSUS_SKIPPABLE_ARMS = {"nfd": "verdict_nfd", "pc": "verdict_pc", "win1258": "verdict_pc"}


def census_skipped_arms(run: dict, entry: dict) -> dict[str, str]:
    """{arm: reason} for the run's arms that the model's tokenizer census makes 0 by construction (DD 6.2). Empty when
    the model is not censused (data/audit/<family or tokenizer_audit>.json absent): nothing is skipped on a guess."""
    arms = [a for a in run.get("arms") or [] if a in CENSUS_SKIPPABLE_ARMS]
    if not arms:
        return {}
    from noilai.eval.run import normalization_census_for
    census = normalization_census_for(entry, {})
    if census.get("status") != "censused":
        return {}
    out = {}
    for a in arms:
        if census.get(CENSUS_SKIPPABLE_ARMS[a]) == "normalizes":
            out[a] = f"0 by construction: {CENSUS_SKIPPABLE_ARMS[a]} = normalizes ({census.get('source')})"
    return out


def build_command(run: dict, model: str, plan: dict, models_cfg: dict, project_root: Path = ROOT,
                  python: str = sys.executable, extra: Sequence[str] = ()) -> list[str]:
    """The run_eval.py argv for one (run, model). Raises on a guarded combination."""
    if run.get("not_a_run_eval_line"):
        raise ValueError(f"run {run['id']!r} is not a run_eval line (see its purpose)")
    entry = models_cfg["by_name"].get(model)
    if entry is None:
        raise KeyError(f"unknown model {model!r}")
    item_spec = plan["item_files"][run["items"]]
    api = is_api(entry)
    if api and item_spec.get("never_to_api"):
        raise ValueError(f"item file {run['items']!r} must never reach an API ({model})")
    if api and not run.get("in_core_only"):
        raise ValueError(f"API model {model!r} may only run with in_core_only: true (run {run['id']!r})")
    opt_out = check_training_provider(run, entry, plan, models_cfg)
    if run.get("limit") and run.get("experiment") != "smoke":
        raise ValueError(f"run {run['id']!r} uses `limit`, which is smoke-only (DESIGN_DECISIONS 12.32): "
                         f"select_items truncates a file sorted by task, so a limit measures T1 only; "
                         f"reference a seeded item file instead")
    cli = plan.get("run_eval_cli", "scripts/run_eval.py")
    cmd = [python, str(Path(project_root) / cli), "--model-config", model, "--items", item_spec["path"],
           "--tasks", *run["tasks"]]
    if run.get("variants"):
        cmd += ["--variants", *run["variants"]]
    if run.get("paraphrases"):
        cmd += ["--paraphrases", *run["paraphrases"]]
    skipped = census_skipped_arms(run, entry)
    arms = [a for a in run["arms"] if a not in skipped]
    if not arms:
        raise ValueError(f"run {run['id']!r}: every arm is 0 by construction for {model} ({skipped})")
    cmd += ["--shots", str(run.get("shots", 0)), "--arms", *arms]
    for key, flag in ABLATION_FLAGS:
        if run.get(key):
            cmd += [flag, str(run[key])]
    if run.get("limit"):
        cmd += ["--smoke", "--limit", str(run["limit"])]      # run_eval.py refuses --limit without --smoke (DD 12.32)
    if run.get("in_core_only"):
        cmd.append("--in-core-only")
    if opt_out:
        cmd.append(OPT_OUT_FLAG)      # recorded by run.py in api_safety / api_privacy_settings (DD 11.2)
    cmd.append("--resume")
    rd = run_dir(plan, run, model)
    style = plan.get("run_eval_out_style", "run_id_root")
    if style == "run_id_root":          # the CLI as landed: --run-id <dir name> --out-root <runs root>
        cmd += ["--run-id", Path(rd).name, "--out-root", plan.get("runs_root", "data/runs")]
    elif style == "out":                # the CLI as first specified: --out <dir>
        cmd += ["--out", rd]
    else:
        raise ValueError(f"run_eval_out_style must be 'run_id_root' or 'out', not {style!r}")
    cmd += list(extra)
    return cmd


def device_for(entry: dict) -> tuple[str, int]:
    """(device type, count) for the compute log, from the model's hardware assignment."""
    hw = entry.get("hardware", "mixed")
    dev, n = HARDWARE_DEVICES.get(hw, (hw, 1))
    return (dev or hw, 1 if n is None else n)


def session_device(session_hardware: str) -> tuple[str, int]:
    dev, n = HARDWARE_DEVICES.get(session_hardware, (session_hardware, 1))
    return (dev or session_hardware, 1 if n is None else n)


def hardware_compatible(entry: dict, session_hardware: str | None) -> bool:
    """Can a model configured for `entry['hardware']` run in a session on `session_hardware`?"""
    if session_hardware is None:
        return True
    hw = entry.get("hardware")
    if is_api(entry):
        return True
    return session_hardware in SESSION_COMPATIBLE.get(hw, {hw})


# ------------------------------------------------------------------ derived item files
def _read_rows(path: Path) -> tuple[dict | None, list[dict]]:
    header, rows = None, []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            if KVI.is_header(d) and header is None and not rows:
                header = d
            else:
                rows.append(d)
    return header, rows


def _write_rows(path: Path, header: dict | None, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if header:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    tmp.replace(path)


def _balanced_draw(rng: random.Random, pool: list[dict], n: int, key: Callable[[dict], object]) -> list[dict]:
    """Round-robin over strata after a seeded shuffle inside each stratum (as scripts/sample_items.py)."""
    groups: dict = defaultdict(list)
    for it in pool:
        groups[key(it)].append(it)
    for g in groups.values():
        g.sort(key=lambda it: it["item_id"])
        rng.shuffle(g)
    keys = sorted(groups, key=str)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
    return out


def _per_cell(spec: dict, task: str) -> int:
    pc = spec.get("per_cell", 0)
    if isinstance(pc, dict):
        return int(pc.get(task, 0))
    return int(pc)


def _is_vulgar(it: dict) -> bool:
    v = it.get("vulgar")
    return v is True or (isinstance(v, str) and v.strip().lower() in ("yes", "true", "1"))


def sample_noilai(rows: list[dict], derive: dict) -> tuple[list[dict], dict]:
    """Seeded per-cell sub-sample of NóiLái items (DD 4.5 rules): the core kept whole, vulgar rows
    excluded, T3 drawn as yes/no pairs, strata balanced over (source, output_lexical) for T1/T2
    and (source, twin_type of the 'no' mate) for T3. Returns (items sorted by item_id, counts)."""
    rng = random.Random(int(derive["seed"]))
    tasks = tuple(derive.get("tasks") or TASKS)
    variants = tuple(derive.get("variants") or VARIANTS)
    by_id = {it["item_id"]: it for it in rows}
    chosen: dict[str, dict] = {}
    counts: Counter = Counter()

    def eligible(it):
        return (not _is_vulgar(it)) and (it.get("in_core") or not derive.get("in_core_only"))

    for task in tasks:
        for v in variants:
            n_cell = _per_cell(derive, task)
            if task == "T3":
                pool = [it for it in rows if it["task"] == "T3" and it["variant"] == v and it.get("gold") == "yes"
                        and eligible(it) and it.get("pair_item_id") in by_id and eligible(by_id[it["pair_item_id"]])]
                n_pairs = n_cell // 2
                core = [it for it in pool if it.get("in_core")]
                rest = [it for it in pool if not it.get("in_core")]
                key = lambda it: (it.get("source"), by_id[it["pair_item_id"]].get("twin_type"))
                if len(core) >= n_pairs:     # a core larger than the cell (a core-derived subset): a seeded, balanced draw from it
                    sel = _balanced_draw(rng, core, n_pairs, key)
                else:                        # the core kept whole, the rest drawn balanced
                    sel = core + _balanced_draw(rng, rest, n_pairs - len(core), key)
                for it in sel:
                    mate = by_id[it["pair_item_id"]]
                    chosen[it["item_id"]] = it
                    chosen[mate["item_id"]] = mate
                counts[f"T3-{v}"] = 2 * len(sel)
            else:
                pool = [it for it in rows if it["task"] == task and it["variant"] == v and eligible(it)]
                core = [it for it in pool if it.get("in_core")]
                rest = [it for it in pool if not it.get("in_core")]
                key = lambda it: (it.get("source"), (it.get("strata") or {}).get("output_lexical"))
                if len(core) >= n_cell:
                    sel = _balanced_draw(rng, core, n_cell, key)
                else:
                    sel = core + _balanced_draw(rng, rest, n_cell - len(core), key)
                for it in sel:
                    chosen[it["item_id"]] = it
                counts[f"{task}-{v}"] = len(sel)
    items = sorted(chosen.values(), key=lambda it: it["item_id"])
    return items, dict(sorted(counts.items()))


def sample_xcopa(rows: list[dict], derive: dict) -> tuple[list[dict], dict]:
    rng = random.Random(int(derive["seed"]))
    pool = sorted(rows, key=lambda d: int(d["idx"]))
    n = min(int(derive["n"]), len(pool))
    sel = sorted(rng.sample(pool, n), key=lambda d: int(d["idx"]))
    return sel, {"XCOPA": len(sel)}


def derive_item_file(key: str, plan: dict, project_root: Path = ROOT, force: bool = False) -> dict:
    """Write the seeded sub-sample described by `item_files[key].derive`; idempotent (same seed, same bytes).

    Returns {"key", "path", "status" ('written' | 'exists' | 'rewritten'), "sha256", "n_items", "counts"}.
    """
    spec = plan["item_files"][key]
    derive = spec.get("derive")
    if not derive:
        raise ValueError(f"item file {key!r} has no `derive` block")
    src_key = derive["source"]
    src = item_path(plan, src_key, project_root)
    dst = item_path(plan, key, project_root)
    if not src.exists():
        raise FileNotFoundError(f"source item file {src_key!r} for {key!r} is missing: {src}")
    status = "written"
    if dst.exists() and not force:
        status = "exists"
    else:
        if dst.exists():
            status = "rewritten"
        header, rows = _read_rows(src)
        kind = spec.get("kind") or plan["item_files"][src_key].get("kind") or "noilai"
        if kind == "xcopa":
            items, counts = sample_xcopa(rows, derive)
            header = None
        else:
            items, counts = sample_noilai(rows, derive)
        _write_rows(dst, header, items)
    _h, rows = _read_rows(dst)
    if spec.get("kind") == "xcopa":
        counts = {"XCOPA": len(rows)}
    else:
        counts = dict(sorted(Counter(f"{r['task']}-{r.get('variant')}" for r in rows).items()))
    return {"key": key, "path": str(dst), "status": status, "sha256": KVI.sha256_file(dst), "n_items": len(rows), "counts": counts}


def materialize(plan: dict, project_root: Path = ROOT, keys: Sequence[str] | None = None, force: bool = False) -> list[dict]:
    """Derive every item file with a `derive` block (or the given keys); skips sources that are absent."""
    out = []
    for key, spec in plan["item_files"].items():
        if not spec.get("derive") or (keys and key not in keys):
            continue
        try:
            out.append(derive_item_file(key, plan, project_root, force=force))
        except FileNotFoundError as e:
            out.append({"key": key, "path": str(item_path(plan, key, project_root)), "status": f"skipped: {e}",
                        "sha256": None, "n_items": 0, "counts": {}})
    return out


def verify_run_items(run: dict, plan: dict, project_root: Path = ROOT, materialize_missing: bool = True) -> dict:
    """The item-file gate for one run: derive the file if the plan says how and it is absent, then verify."""
    key = run["items"]
    spec = plan["item_files"][key]
    if spec.get("derive") and materialize_missing and not item_path(plan, key, project_root).exists():
        try:
            derive_item_file(key, plan, project_root)
        except FileNotFoundError as e:
            return {"key": key, "ok": False, "problems": [str(e)], "path": str(item_path(plan, key, project_root))}
    res = KVI.verify_spec(spec, root=Path(project_root))
    res["key"] = key
    return res


# ------------------------------------------------------------------ API ledger
def ledger_path(plan: dict, project_root: Path = ROOT) -> Path:
    return Path(project_root) / plan.get("runs_root", "data/runs") / "api_ledger.json"


def ledger_load(path: Path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def ledger_save(path: Path, ledger: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_day() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _day_entry(ledger: dict, model: str, day: str) -> dict:
    """{'requests': n, 'tokens': n} for (model, day); an old integer-valued ledger is read as requests."""
    d = ledger.setdefault(model, {})
    cur = d.get(day)
    if cur is None:
        cur = d[day] = {"requests": 0, "tokens": 0}
    elif not isinstance(cur, dict):
        cur = d[day] = {"requests": int(cur), "tokens": 0}
    return cur


def ledger_add(ledger: dict, model: str, n_calls: int, n_tokens: int = 0, day: str | None = None) -> dict:
    day = day or utc_day()
    if model.startswith("_"):
        raise ValueError("model names starting with '_' are reserved for ledger metadata")
    cur = _day_entry(ledger, model, day)
    cur["requests"] += int(n_calls)
    cur["tokens"] += int(n_tokens)
    return ledger


def ledger_used(ledger: dict, model: str, day: str | None = None) -> int:
    """Requests recorded for (model, day)."""
    day = day or utc_day()
    cur = ledger.get(model, {}).get(day, 0)
    return int(cur["requests"]) if isinstance(cur, dict) else int(cur or 0)


def ledger_tokens_used(ledger: dict, model: str, day: str | None = None) -> int:
    day = day or utc_day()
    cur = ledger.get(model, {}).get(day, 0)
    return int(cur.get("tokens", 0)) if isinstance(cur, dict) else 0


def daily_budget(entry: dict, models_cfg: dict) -> dict | None:
    """The free-tier daily caps for an API entry: {'requests_per_day', 'tokens_per_day', 'scope', 'provider'}
    (None when the entry is not served by a provider). Either cap may be None (unknown)."""
    prov = entry.get("provider")
    if not prov:
        return None
    tier = models_cfg.get("providers", {}).get(prov, {}).get("free_tier", {})
    rpd, tpd = tier.get("requests_per_day"), tier.get("tokens_per_day")
    return {"requests_per_day": int(rpd) if rpd else None, "tokens_per_day": int(tpd) if tpd else None,
            "scope": tier.get("scope") or ("per_model" if tier.get("per_model", True) else "per_org"), "provider": prov}


def _budget_models(entry: dict, budget: dict, models_cfg: dict) -> list[str]:
    """The models whose usage counts against this entry's daily budget: every model of the provider
    under `per_org`; otherwise every entry served as the same (provider, provider_model_id), so a
    thinking variant (same provider model id) shares its base entry's daily cap (DD 7.2 / 7.3)."""
    if budget.get("scope") == "per_org":
        return [m["name"] for m in models_cfg["models"] if m.get("provider") == budget["provider"]]
    pmid = entry.get("provider_model_id")
    if not pmid:
        return [entry["name"]]
    same = [m["name"] for m in models_cfg["models"]
            if m.get("provider") == entry.get("provider") and m.get("provider_model_id") == pmid]
    return same if entry["name"] in same else [entry["name"], *same]


def budget_status(entry: dict, models_cfg: dict, ledger: dict, day: str | None = None,
                  threshold: float = 1.0) -> dict | None:
    """Usage against the daily caps; `spent` when requests or tokens reach threshold x cap."""
    budget = daily_budget(entry, models_cfg)
    if budget is None:
        return None
    day = day or utc_day()
    names = _budget_models(entry, budget, models_cfg)
    req = sum(ledger_used(ledger, m, day) for m in names)
    tok = sum(ledger_tokens_used(ledger, m, day) for m in names)
    rpd, tpd = budget["requests_per_day"], budget["tokens_per_day"]
    reasons = []
    if rpd and req >= threshold * rpd:
        reasons.append(f"requests {req}/{rpd}")
    if tpd and tok >= threshold * tpd:
        reasons.append(f"tokens {tok}/{tpd}")
    return {**budget, "day": day, "requests_used": req, "tokens_used": tok, "spent": bool(reasons),
            "reason": ", ".join(reasons), "counts_models": names}


def count_outputs(run_directory: Path) -> int:
    p = Path(run_directory) / "outputs.jsonl"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def outputs_usage(run_directory: Path, skip_rows: int = 0, assumed_tokens_per_call: int = 500) -> dict:
    """Rows and tokens of outputs.jsonl after the first `skip_rows` rows. Tokens are
    n_prompt_tokens + n_output_tokens per row (`usage` when the backend stored it whole); a row
    without usage counts `assumed_tokens_per_call` and is tallied in `n_missing_usage`."""
    p = Path(run_directory) / "outputs.jsonl"
    out = {"rows": 0, "tokens": 0, "n_missing_usage": 0}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if not ln.strip() or i < skip_rows:
                continue
            out["rows"] += 1
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:      # a row cut by a stop: counted by the assumption, redone by --resume
                out["tokens"] += assumed_tokens_per_call
                out["n_missing_usage"] += 1
                continue
            usage = d.get("usage") or {}
            pt = d.get("n_prompt_tokens", usage.get("prompt_tokens"))
            ot = d.get("n_output_tokens", usage.get("completion_tokens"))
            if pt is None and ot is None:
                out["tokens"] += assumed_tokens_per_call
                out["n_missing_usage"] += 1
            else:
                out["tokens"] += int(pt or 0) + int(ot or 0)
    return out


def trim_partial_last_line(path: Path) -> bool:
    """Drop a trailing line that is not complete JSON (a row cut by a stop); `--resume` redoes it.
    Returns True when something was removed."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    tail = data[cut + 1:]
    try:
        json.loads(tail.decode("utf-8"))
        return False
    except (UnicodeDecodeError, json.JSONDecodeError):
        path.write_bytes(data[: cut + 1] if cut >= 0 else b"")
        return True


class BudgetWatchdog:
    """Polls a run's outputs.jsonl while run_eval.py runs and interrupts the process (SIGINT) as
    soon as the day's request or token cap is reached; the rows already flushed are kept."""

    def __init__(self, out_dir: Path, budget: dict, requests_before: int, tokens_before: int, skip_rows: int,
                 assumed_tokens_per_call: int = 500, poll_s: float = 5.0):
        self.out_dir = Path(out_dir)
        self.budget = budget
        self.requests_before = requests_before
        self.tokens_before = tokens_before
        self.skip_rows = skip_rows
        self.assumed = assumed_tokens_per_call
        self.poll_s = poll_s
        self.stopped = False
        self.reason = ""
        self._stop = threading.Event()

    def check(self) -> str | None:
        u = outputs_usage(self.out_dir, self.skip_rows, self.assumed)
        rpd, tpd = self.budget.get("requests_per_day"), self.budget.get("tokens_per_day")
        if rpd and self.requests_before + u["rows"] >= rpd:
            return f"requests {self.requests_before + u['rows']}/{rpd}"
        if tpd and self.tokens_before + u["tokens"] >= tpd:
            return f"tokens {self.tokens_before + u['tokens']}/{tpd}"
        return None

    def watch(self, proc: subprocess.Popen) -> None:
        while not self._stop.wait(self.poll_s):
            if proc.poll() is not None:
                return
            reason = self.check()
            if reason:
                self.stopped, self.reason = True, reason
                try:
                    proc.send_signal(signal.SIGINT)
                except (ProcessLookupError, OSError):
                    pass
                return

    def run(self, cmd: Sequence[str], cwd: Path) -> int:
        proc = subprocess.Popen(list(cmd), cwd=str(cwd))
        t = threading.Thread(target=self.watch, args=(proc,), daemon=True)
        t.start()
        try:
            rc = proc.wait()
        finally:
            self._stop.set()
            t.join(timeout=max(self.poll_s * 2, 1.0))
        return rc


# ------------------------------------------------------------------ execution
def execute(run_id: str, models: Sequence[str] | None = None, platform: str = "kaggle", dry_run: bool = False,
            continue_on_error: bool = True, extra: Sequence[str] = (), plan: dict | None = None,
            models_cfg: dict | None = None, project_root: Path = ROOT, log_path: Path | None = None,
            python: str = sys.executable, session_hardware: str | None = None, allow_hardware_mismatch: bool = False,
            session_t0: float | None = None, max_session_hours: float | None = None,
            after_each: Callable[[dict], None] | None = None, verify: bool = True, poll_s: float = 5.0,
            allow_unpinned_revision: bool = False, line_overrides: dict | None = None,
            chunk_tag: str | None = None) -> list[dict]:
    """Run every (run, model) command in order; return one result dict per model.

    Statuses: 'dry-run', 'ok', 'failed (<rc>)', 'parked: ...' (an API day's cap reached; resume
    tomorrow), 'refused: ...' (a build_command guard: never_to_api, a training provider on a
    test-split file, in_core_only missing, ...), 'skipped: ...' (budget spent, hardware not
    runnable in this session, session budget reached, item file failed verification, revision
    not pinned on a non-smoke line).
    """
    plan = plan or load_plan()
    models_cfg = models_cfg or load_models()
    run = narrow_run(find_run(plan, run_id), line_overrides, chunk_tag)
    names = list(models) if models else expand_models(run, plan, models_cfg)
    log_path = Path(log_path) if log_path else Path(project_root) / plan.get("compute_log", "data/compute_log.csv")
    lpath = ledger_path(plan, project_root)
    ledger = ledger_load(lpath)
    api_cfg = plan.get("api") or {}
    assumed = int(api_cfg.get("assumed_tokens_per_call", 500))
    park_threshold = float(api_cfg.get("park_threshold", 0.95))
    results: list[dict] = []
    gate = None
    if verify and not dry_run:
        gate = verify_run_items(run, plan, project_root)
        if not gate["ok"]:
            print(f"[gate ] {run_id}: item file {run['items']!r} failed verification: {gate['problems']}")
    for name in names:
        entry = models_cfg["by_name"][name]
        out_dir = Path(project_root) / run_dir(plan, run, name)
        res = {"run": run_id, "model": name, "cmd": None, "out": str(out_dir), "status": "dry-run",
               "hardware": entry.get("hardware"), "session_hardware": session_hardware}
        res["census_skipped_arms"] = census_skipped_arms(run, entry)
        try:
            cmd = build_command(run, name, plan, models_cfg, project_root=project_root, python=python, extra=extra)
        except ValueError as e:               # a guard of build_command: report, go on with the next model
            res["status"] = f"{REFUSED}: {e}"
            print(f"\n[plan ] {run_id} / {name}\n[refuse] {e}")
            results.append(res)
            continue
        res["cmd"] = cmd
        print(f"\n[plan ] {run_id} / {name}\n[cmd  ] {shlex.join(cmd)}")
        if not dry_run and not run.get("limit") and not allow_unpinned_revision and not revision_pinned(entry):
            res["status"] = (f"skipped: revision not pinned ({entry.get('revision')!r}); DESIGN_DECISIONS 7.1 requires a full "
                             f"commit hash, a Kaggle Models slug + version or a GGUF SHA-256 before a paper run "
                             f"(run_eval.py refuses it too; --allow-unpinned-revision overrides for a rehearsal)")
            print(f"[skip ] {name}: {res['status']}")
            results.append(res)
            continue
        if gate is not None and not gate["ok"]:
            res["status"] = f"skipped: item file {run['items']} failed verification"
            res["verify"] = gate
            results.append(res)
            continue
        if not hardware_compatible(entry, session_hardware):
            msg = f"hardware {entry.get('hardware')} is not runnable in a {session_hardware} session"
            if not allow_hardware_mismatch:
                res["status"] = f"skipped: {msg}"
                print(f"[skip ] {name}: {msg}")
                results.append(res)
                continue
            res["hardware_mismatch"] = msg
            print(f"[warn ] {name}: {msg}; running anyway (--allow-hardware-mismatch), logged as the session's device")
        if max_session_hours is not None and session_t0 is not None and CL.hours_since(session_t0) >= max_session_hours:
            res["status"] = f"skipped: session budget of {max_session_hours} h reached before this model started"
            print(f"[skip ] {name}: {res['status']}")
            results.append(res)
            continue
        status = budget_status(entry, models_cfg, ledger) if is_api(entry) else None
        if status is not None:
            res["api_budget_today"] = {k: status[k] for k in ("requests_per_day", "tokens_per_day", "requests_used",
                                                              "tokens_used", "scope", "day")}
            if status["spent"]:
                res["status"] = f"skipped: daily budget spent ({status['reason']}); resume tomorrow (UTC)"
                print(f"[quota] {name}: {status['reason']} used today; skipping until tomorrow (UTC)")
                results.append(res)
                continue
            print(f"[quota] {name}: requests {status['requests_used']}/{status['requests_per_day']}, "
                  f"tokens {status['tokens_used']}/{status['tokens_per_day']} used today ({status['scope']})")
        if dry_run:
            results.append(res)
            continue
        before = count_outputs(out_dir)
        if (out_dir / "outputs.jsonl").exists() and trim_partial_last_line(out_dir / "outputs.jsonl"):
            print(f"[fix  ] {name}: dropped a partial trailing row of outputs.jsonl (redone by --resume)")
            before = count_outputs(out_dir)
        t0 = time.time()
        dev, n = session_device(session_hardware) if session_hardware else device_for(entry)
        # provisional row (checklist C1): a session killed mid-model still has this run's start on record
        CL.append_entry(log_path, CL.Entry(date=CL.today(), platform=platform, gpu_type=dev, n_gpus=n, hours=0.0,
                                           run_id=out_dir.name, purpose=f"{run_id} started {dt.datetime.now(dt.timezone.utc):%H:%M}Z"))
        stopped_by_budget = ""
        if status is not None:
            dog = BudgetWatchdog(out_dir, status, status["requests_used"], status["tokens_used"], before, assumed, poll_s)
            rc = dog.run(cmd, cwd=Path(project_root))
            stopped_by_budget = dog.reason if dog.stopped else ""
        else:
            rc = subprocess.run(cmd, cwd=str(project_root), check=False).returncode
        hours = CL.hours_since(t0)
        if (out_dir / "outputs.jsonl").exists():
            trim_partial_last_line(out_dir / "outputs.jsonl")
        usage = outputs_usage(out_dir, before, assumed)
        res.update({"returncode": rc, "hours": round(hours, 4), "new_outputs": usage["rows"], "new_tokens": usage["tokens"],
                    "n_missing_usage": usage["n_missing_usage"]})
        if status is not None:
            ledger_add(ledger, name, usage["rows"], usage["tokens"])
            ledger.setdefault("_budgets", {}).setdefault(name, {})[utc_day()] = {
                "requests_per_day": status["requests_per_day"], "tokens_per_day": status["tokens_per_day"], "scope": status["scope"]}
            ledger_save(lpath, ledger)
            after = budget_status(entry, models_cfg, ledger, threshold=park_threshold)
            res["api_budget_after"] = {k: after[k] for k in ("requests_used", "tokens_used")}
        if rc == 0:
            res["status"] = "ok"
        elif stopped_by_budget:
            res["status"] = f"{PARKED}: daily budget reached ({stopped_by_budget}); resume tomorrow (UTC)"
        elif status is not None and after["spent"]:
            res["status"] = f"{PARKED}: command failed with >= {int(park_threshold * 100)}% of a daily cap used ({after['reason']}); resume tomorrow (UTC)"
        else:
            res["status"] = f"failed ({rc})"
        res["logged_device"] = {"gpu_type": dev, "n_gpus": n}
        CL.append_entry(log_path, CL.Entry(date=CL.today(), platform=platform, gpu_type=dev, n_gpus=n,
                                           hours=hours, run_id=out_dir.name, purpose=run_id))
        print(f"[done ] {name}: {res['status']}, {hours:.3f} h, {usage['rows']} new outputs, {usage['tokens']} tokens")
        if after_each is not None:
            try:
                after_each(res)
            except Exception as e:
                res["after_each_error"] = repr(e)
                print(f"[warn ] after_each hook failed for {name}: {e!r}")
        results.append(res)
        if rc != 0 and not stopped_by_budget and not res["status"].startswith(PARKED) and not continue_on_error:
            break
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=CONFIGS / "run_plan.yaml")
    ap.add_argument("--models-config", type=Path, default=CONFIGS / "models.yaml")
    ap.add_argument("--list", action="store_true", help="list the runs and their expanded models")
    ap.add_argument("--materialize", nargs="*", default=None, metavar="KEY",
                    help="write the derived (seeded) item files: all of them, or the given keys")
    ap.add_argument("--force", action="store_true", help="with --materialize: rewrite existing files")
    ap.add_argument("--verify", nargs="*", default=None, metavar="RUN_ID", help="item-file gate for these runs (exit 1 on failure)")
    ap.add_argument("--run", help="run id from run_plan.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="subset of the run's models (default: all)")
    ap.add_argument("--platform", default="kaggle", choices=CL.PLATFORMS)
    ap.add_argument("--session-hardware", default=None, choices=sorted(k for k in HARDWARE_DEVICES if k != "mixed"),
                    help="the session's accelerator; models configured for other hardware are skipped")
    ap.add_argument("--allow-hardware-mismatch", action="store_true")
    ap.add_argument("--max-session-hours", type=float, default=None, help="launch no new model past this budget")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the item-file gate (tests only)")
    ap.add_argument("--allow-unpinned-revision", action="store_true",
                    help="run a self-hosted model whose `revision` is not pinned (rehearsals only; DD 7.1)")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--extra", default="", help="extra run_eval.py args, one shell-quoted string")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log", type=Path, default=None, help="compute log path (default from the plan)")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    models_cfg = load_models(args.models_config)
    if args.list:
        for r in plan["runs"]:
            try:
                names = expand_models(r, plan, models_cfg)
            except KeyError as e:
                names = [f"ERROR {e}"]
            est = r.get("estimate", {})
            flags = " ".join(f"{k}={r[k]}" for k, _ in ABLATION_FLAGS if r.get(k))
            print(f"{r['id']:<22} {r['experiment']:<12} {est.get('low')}-{est.get('high')} {est.get('unit')}"
                  f"   items={r['items']} {flags}  {len(names)} models: {', '.join(names)}")
        return 0
    if args.materialize is not None:
        rc = 0
        for info in materialize(plan, ROOT, keys=args.materialize or None, force=args.force):
            print(json.dumps(info, ensure_ascii=False))
            if str(info["status"]).startswith("skipped"):
                rc = 1
        return rc
    if args.verify is not None:
        rc = 0
        for rid in args.verify:
            res = verify_run_items(find_run(plan, rid), plan, ROOT)
            print(json.dumps({k: res.get(k) for k in ("key", "path", "ok", "n_items", "sha256", "problems")}, ensure_ascii=False))
            rc = rc or (0 if res["ok"] else 1)
        return rc
    if not args.run:
        ap.error("--run, --list, --materialize or --verify is required")
    results = execute(args.run, models=args.models, platform=args.platform, dry_run=args.dry_run,
                      continue_on_error=not args.stop_on_error, extra=shlex.split(args.extra), plan=plan,
                      models_cfg=models_cfg, log_path=args.log, python=args.python,
                      session_hardware=args.session_hardware, allow_hardware_mismatch=args.allow_hardware_mismatch,
                      session_t0=time.time(), max_session_hours=args.max_session_hours, verify=not args.no_verify,
                      allow_unpinned_revision=args.allow_unpinned_revision)
    failed = [r for r in results if str(r["status"]).startswith("failed")]
    parked = [r for r in results if str(r["status"]).startswith(PARKED)]
    refused = [r for r in results if str(r["status"]).startswith(REFUSED)]
    print(f"\n[summary] {len(results)} commands, {len(failed)} failed, {len(parked)} parked, {len(refused)} refused")
    return 1 if failed or refused else 0


if __name__ == "__main__":
    sys.exit(main())
