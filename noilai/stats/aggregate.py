"""From scored run directories to the cells of the pre-registered analysis (the aggregation helper that
docs/DESIGN_DECISIONS.md 8.1 names). The code implements the binding text and never changes it:

  * PREREG 8.1 (docs/PREREGISTRATION.md:83): "The base pair is the sampled unit and the cluster ... Paraphrases
    are within-item repeated measures: aggregate each item x model to its paraphrase mean (values in
    {0, 1/3, 2/3, 1}), or keep rows and always cluster by base pair."
  * DD 5.1 (:253): the primary metric is "strict T1 accuracy averaged over the three paraphrases, pooled over
    V1-V4 with equal weight, per open model"; API models report "p0 accuracy on the core, with the 300-item
    paraphrase range reported beside it"; the aggregation "never mixes 1- and 3-trial means without saying so
    (item 52)". PREREG 3 (:35): the default condition is Vietnamese instructions, the explained variant, three
    demonstrations, the raw input format and the baseline encoding.
  * PREREG 8.13 (:132): "Main tables report p0 with the mean and range over {p0, p1, p2}; a claim about a model
    holds only if its direction holds under all three paraphrases ... every headline size carries its cluster
    count".
  * PREREG 8.2 (:86): base-pair cluster bootstrap, B = 2,000, fixed seed, BCa from 50 base pairs, stratified by
    base-pair type (lexical vs pseudo); Wilson on n / DEFF for cells at 0 % / 100 % or below 20 base pairs;
    "Every cell reports point estimate, CI, number of items, number of base pairs, realized design effect and
    unparseable rate."
  * PREREG 8.5 (:108): tau_m(a) = E_i[Y_i(a) - Y_i(base)] over the sampled items, base-pair cluster-bootstrap CI
    and the paired t on base-pair-level mean differences as the primary inference, McNemar mid-p only as the
    secondary p0 item-level statistic labelled as ignoring clustering; baseline = NFC, old-style placement,
    canonical i/y; `arm_scope: whole_prompt` is the primary scope.
  * DD 5.3 (:261): T3 is reported as accuracy and balanced accuracy / d'; `t3_yesno_lp` and `t3_pair_lp` are
    named, reported in separate columns and never compared with generated accuracy; the headline T3 number
    excludes the `spelling` twins and reports them as their own column.
  * DD 8.5 (:434): ">= 300-350 items per cell for any cell-level 10-point claim" (the `claim_eligible` flag; a
    different thing from the small-cell interval rule, both are columns).
  * DD 12.17 (:568) and PREREG 8.3 (:94): the explicit onset-rime-tone input gap is a main result, "estimation
    with CIs, no Holm". PREREG 8.3 (:90): a normalizing tokenizer's `nfd` / `pc` cells are "0 by construction"
    rows, not tests; DD 6.2: a corrupting tokenizer path is refused and reported.

Every interval comes from noilai.stats.bootstrap (this module owns no bootstrap), every paired test from
noilai.stats.tests through noilai.stats.tables.intervention_table, and the small-cell rule is the one inside
bootstrap.cluster_bootstrap (a binary cell -> Wilson on n / DEFF, a non-binary statistic such as a paraphrase mean
or a difference -> the percentile interval, both flagged `small_cell`); this module only forwards `ci_method` and
`small_cell`. Nothing is written under paper/, and no item-level row is ever written out.

A run directory is `<runs>/<run>__<model>[__<tag>]` (scripts/kaggle_run_plan.run_dir). The directories of one plan
line and one model are pooled BEFORE anything is aggregated (the chunk plan's `p0p1` + `p2` paraphrase split of a
2xT4 / TPU model), the `cpu` tag (another engine) stays a unit of its own, `tpu_main` is the TPU trio's part of
`E1_main`. Score rows carry no model column: the model is the manifest's identity.model_key. XCOPA has no base
pair and clusters by item_id. scripts/ are never imported here; the three-line run-id split of
scripts/interim_analysis.py is re-implemented in `split_run_id`.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants
from ..audit.tokenizers import CENSUS_VERDICTS
from ..eval.backends import API_BACKENDS
from ..eval.run import (
    AUDIT_DIR,
    CENSUS_ARM_VERDICTS,
    STATUS_FINISHED,
    STATUS_THINKING,
    census_classification,
    load_item_file,
    read_manifest,
)
from ..eval.score import _spelling_pair_ids, balanced_stats, read_jsonl
from . import bootstrap, tables

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"                           # CLAUDE.md: this workstream never writes under paper/
RUNS_DIR = ROOT / "data" / "runs"
MODELS_YAML = ROOT / "configs" / "models.yaml"
DEFAULT_OUT_DIR = ROOT / "experiments" / "analysis"

CLUSTER = constants.BOOTSTRAP_CLUSTER                # base_pair_id: the sampled unit and the cluster (PREREG 8.1)
STRATUM = "source"                                   # lexicon / pseudo: the PREREG 8.2 stratification
NO_STRATUM = "none"                                  # the stratum label of rows without a base-pair type (XCOPA)
BASELINE_ARM = tables.runner_arm_name("base")        # nfc (PREREG 8.5: baseline = NFC)
ARM_SCOPE_PRIMARY = constants.ARM_SCOPE_PRIMARY      # whole_prompt
PARAPHRASES = ("p0", "p1", "p2")
PRIMARY_PARAPHRASE = "p0"
DEFAULT_CONDITION = "vi-*-s3-explained-raw"          # PREREG 3: vi, explained, 3 shots, raw input (any paraphrase)
COMPONENTS_CONDITION = "vi-*-s3-explained-components"   # DD 12.17: the explicit onset-rime-tone input
LINE_ALIASES = {"tpu_main": "E1_main"}               # scripts/ledger.py RUN_ALIAS: the TPU trio's E1 rows are E1_main's
POOLED_TAGS = ("p0p1", "p2")                         # a paraphrase split of one line: pooled per (line, model)
SEPARATE_TAGS = ("cpu",)                             # another engine: a unit apart, never pooled with the GPU rows
ROW_KEY = ("item_id", "arm", "arm_scope", "prompt_id")   # unique over the pooled rows of one (line, model)
MEANS_KEYS = ("model", "line", "item_id", "arm", "arm_scope", "condition", "task", "variant")
CELL_KEYS = ("model", "line", "task", "variant", "arm", "arm_scope", "condition")
CARRY_COLUMNS = (CLUSTER, STRATUM, "in_core", "c2_affected", "c2_enriched", "item_file", "item_key", "pair_item_id",
                 "twin_type", "gold", "is_api", "census_nfd", "census_pc", "input_lexical", "output_lexical",
                 "attested_overlap", "tone_pair", "spelling_triggers", "degenerate", "exact", "eligible_h6")
NO_BASE_PAIR_TASKS = ("XCOPA",)                      # cluster by item_id (scripts/check_run.py does the same)
OUTCOME_BY_TASK = {"XCOPA": "logprob_correct"}       # DD 5.7: the summed log-probability is XCOPA's primary outcome
LOGPROB_COLUMNS = ("forced_choice_correct", "paired_correct", "paired_correct_norm", "logprob_correct",
                   "logprob_correct_norm")           # never compared with a generated `correct` column (DD 5.3)
BASELINE_LINES = {"E3_api_core": "E1_api_core"}      # run_plan: the API C1/C2 arms' nfc baseline is E1_api_core's p0
APPENDIX_LINES = ("E3_scope_item", "E3_engine_hf")   # two-scope / two-engine agreement checks, not Table 3 cells
T1_VARIANTS = constants.T1_CELLS
UNPARSEABLE = "unparseable"
ZERO_BY_CONSTRUCTION = "zero_by_construction"
OPEN_METRIC = "mean of 3 paraphrases (strict T1, V1-V4 equal weight, default condition)"
API_METRIC = "p0 on the core (strict T1, V1-V4 equal weight); range on the paraphrase subset"
EXPLICIT_INPUT_LABEL = ("explicit-input gap, p0, core items, nfc, whole_prompt, paired by item (DD 12.17): "
                        "estimation with CIs, no Holm (PREREG 8.3); the estimand sentence is derived, not quoted")
NOT_CENSUSED = {"status": "not_censused", "verdict_nfd": None, "verdict_pc": None, "legacy_census": False,
                "source": None, "engine": None}
_PARAPHRASE = re.compile(r"p\d+")


class Unscorable(RuntimeError):
    """A run directory whose rows cannot enter the analysis: reported in `skipped`, never a crash."""


# ------------------------------------------------------------------ names: run ids, prompt ids, item files
def split_run_id(run_id: str) -> tuple[str, str | None, str]:
    """(line, model, tag) of a run id `<run>__<model>[__<tag>]`; model names contain `--`, never `__`. A run id
    without `__` is its own line with no model and no tag (the model then comes from the manifest)."""
    parts = run_id.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return run_id, None, ""


def pool_key(line: str, model: str, tag: str = "") -> tuple[str, str, str]:
    """The rows of every directory with the same key are one set of cells: the aliased line, the model, and
    the tag only when it marks another engine (`cpu`); the paraphrase split tags (`p0p1`, `p2`) pool."""
    return LINE_ALIASES.get(line, line), model, tag if tag in SEPARATE_TAGS else ""


def paraphrase_of(prompt_id: object) -> str | None:
    """`p0` of `vi-p0-s3-explained-raw` (noilai.eval.prompts.prompt_id) or of XCOPA's `xcopa-p0`; None otherwise."""
    parts = str(prompt_id).split("-")
    return parts[1] if len(parts) >= 2 and _PARAPHRASE.fullmatch(parts[1]) else None


def condition_of(prompt_id: object) -> str | None:
    """The prompt id with its paraphrase replaced by `*`: `vi-*-s3-explained-raw` is the default condition."""
    parts = str(prompt_id).split("-")
    if len(parts) >= 2 and _PARAPHRASE.fullmatch(parts[1]):
        parts[1] = "*"
        return "-".join(parts)
    return None


def item_file_kind(path: object) -> str:
    """`c2` for the C2-enriched item file (configs/run_plan.yaml `noilai_c2`), `main` for every other file: the
    `file` coordinate of constants.HOLM_FAMILY_TABLE3."""
    stem = Path(str(path or "")).name.split(".")[0].lower()
    return "c2" if "c2" in stem.split("_") else "main"


# ------------------------------------------------------------------ the panel and the census
def load_models_cfg(path: Path = MODELS_YAML) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def model_entry(models_cfg: Mapping | None, name: str) -> dict | None:
    for e in (models_cfg or {}).get("models") or []:
        if e.get("name") == name:
            return e
    return None


def is_api_entry(entry: Mapping | None) -> bool:
    """configs/models.yaml: an API model has an API backend or `hardware: api` (scripts/kaggle_run_plan.is_api)."""
    return bool(entry) and (entry.get("backend") in API_BACKENDS or entry.get("hardware") == "api")


def is_api_manifest(manifest: Mapping) -> bool:
    m = manifest.get("model") or {}
    return bool(m.get("provider")) or m.get("hardware_entry") == "api" or is_api_entry(m.get("config_entry"))


def _census_block(census: Mapping | None, source: str | None = None) -> dict:
    """{status, verdict_nfd, verdict_pc, legacy_census, source, engine} from a census block as noilai.eval.run
    writes it into a manifest or scripts/audit_tokenizers.py into data/audit/<family>.json."""
    if not census:
        return dict(NOT_CENSUSED)
    cls, legacy = census_classification(dict(census))
    pc = census.get("verdict_pc") if census.get("verdict_pc") in CENSUS_VERDICTS else None
    return {"status": "censused", "verdict_nfd": cls, "verdict_pc": pc, "legacy_census": legacy,
            "source": source or census.get("source"), "engine": census.get("engine")}


def census_from_manifest(manifest: Mapping) -> dict | None:
    """The run's own census (manifest data.normalization_census) when the run was censused, else None."""
    c = (manifest.get("data") or {}).get("normalization_census") or {}
    return _census_block(c, c.get("source")) if c.get("status") == "censused" else None


def census_from_audit(names: Sequence[str | None], audit_dir: Path = AUDIT_DIR) -> dict | None:
    """The fallback of noilai.eval.run.normalization_census_for: data/audit/<name>.json for the first name
    (the entry's `tokenizer_audit`, its `family`, its `name`) whose file carries a normalization_census."""
    for name in names:
        if not name:
            continue
        p = Path(audit_dir) / f"{name}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        census = d.get("normalization_census") or {}
        if census:
            rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
            return _census_block(census, str(rel))
    return None


def census_for(model: str, manifest: Mapping | None = None, models_cfg: Mapping | None = None,
               audit_dir: Path = AUDIT_DIR) -> dict:
    """The census verdicts that decide a model's nfd / pc cells: the run manifest's block first (the engine the
    rows were produced on), else data/audit/<family>.json through the models.yaml entry, else not censused."""
    c = census_from_manifest(manifest) if manifest else None
    if c:
        return c
    entry = model_entry(models_cfg, model) or {}
    c = census_from_audit([entry.get("tokenizer_audit"), entry.get("family"), model], audit_dir)
    return c or dict(NOT_CENSUSED)


def normalizing_arms(census: Mapping | None) -> set[str]:
    """Arms the census makes identical to nfc (token ids unchanged): their cells are 0 by construction."""
    return {arm for arm, key in CENSUS_ARM_VERDICTS.items() if (census or {}).get(key) == "normalizes"}


def corrupting_arms(census: Mapping | None) -> set[str]:
    """Arms whose re-encoded string does not survive the tokenizer path: refused and reported (DD 6.2)."""
    return {arm for arm, key in CENSUS_ARM_VERDICTS.items() if (census or {}).get(key) == "corrupts"}


# ------------------------------------------------------------------ clusters from the rows or the item file
def resolve_item_path(manifest: Mapping, root: Path = ROOT) -> Path | None:
    """The run's item file as recorded (item_file.path / data.item_file), relative to the project root, or by its
    tail from `data/` when the path was absolute on another machine; None when no candidate exists."""
    info = manifest.get("item_file") if isinstance(manifest.get("item_file"), dict) else {}
    recorded = str(info.get("path") or (manifest.get("data") or {}).get("item_file") or "")
    if not recorded:
        return None
    rec = Path(recorded)
    candidates = [rec, root / rec]
    if "data" in rec.parts:
        candidates.append(root / Path(*rec.parts[rec.parts.index("data"):]))
    return next((c for c in candidates if c.exists()), None)


def attach_clusters(rows: list[dict], manifest: Mapping | None = None, *, root: Path = ROOT) -> str | None:
    """Make sure every row carries CLUSTER (and STRATUM when the item file knows it). scores.jsonl copies
    base_pair_id and source from the item; a task without base pairs (XCOPA) clusters by item_id; older scores
    are completed from the item file by item_id. Returns a note, or raises Unscorable."""
    missing = [r for r in rows if not r.get(CLUSTER)]
    notes = []
    no_pair = [r for r in rows if r.get("task") in NO_BASE_PAIR_TASKS]
    for r in no_pair:
        if not r.get(CLUSTER):
            r[CLUSTER] = r["item_id"]
        if r.get(STRATUM) is None:
            r[STRATUM] = NO_STRATUM
    if any(r in missing for r in no_pair):
        notes.append(f"{sum(1 for r in no_pair if r in missing)} rows of {'/'.join(NO_BASE_PAIR_TASKS)} cluster by item_id")
    rest = [r for r in missing if r.get("task") not in NO_BASE_PAIR_TASKS]
    if rest:
        path = resolve_item_path(manifest or {}, root)
        if path is None:
            raise Unscorable(f"{len(rest)} score rows carry no {CLUSTER} and the item file is not on this machine")
        try:
            items, _ = load_item_file(path)
        except (OSError, ValueError) as e:              # unreadable or not a jsonl file (JSONDecodeError is a ValueError)
            raise Unscorable(f"{len(rest)} score rows carry no {CLUSTER} and the item file cannot be read: {e}") from e
        by_id = {it["item_id"]: it for it in items}
        unresolved = 0
        for r in rest:
            it = by_id.get(r.get("item_id"))
            if it and it.get(CLUSTER):
                r[CLUSTER] = it[CLUSTER]
                if r.get(STRATUM) is None and it.get(STRATUM) is not None:
                    r[STRATUM] = it[STRATUM]
            else:
                unresolved += 1
        if unresolved:
            raise Unscorable(f"{unresolved} score rows have no {CLUSTER} in the scores or in the item file {path.name}")
        notes.append(f"{len(rest)} rows took {CLUSTER} from the item file {path.name}")
    return "; ".join(notes) or None


# ------------------------------------------------------------------ run directories
@dataclass
class RunDir:
    """One scored run directory as the analysis sees it."""
    path: Path
    run_id: str
    line: str                      # the plan line after LINE_ALIASES (tpu_main -> E1_main)
    raw_line: str
    model: str
    tag: str
    status: str
    item_path: str | None
    item_sha256: str | None
    item_key: str | None           # the item file's stem (noilai_main, noilai_c2, xcopa_vi_test, ...)
    item_file: str                 # main | c2 (the HOLM_FAMILY_TABLE3 coordinate)
    smoke: bool
    is_api: bool
    census: dict
    results_sha256: str | None
    n_rows: int
    arms: list[str]
    prompt_ids: list[str]
    cluster_note: str | None = None
    unit_ids: list[str] = field(default_factory=list)
    hash_verified: str = "n/a (no ledger)"

    def as_dict(self) -> dict:
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items() if k != "census"}
        d["census_nfd"] = self.census.get("verdict_nfd")
        d["census_pc"] = self.census.get("verdict_pc")
        d["census_status"] = self.census.get("status")
        return d


@dataclass
class LoadedRuns:
    """The pooled score rows (one DataFrame), the run directories, what was skipped and why, and the census per
    model. Added row columns: model, run_id, line, tag, item_key, item_file, c2_enriched, is_api, census_nfd,
    census_pc, paraphrase, condition (plus base_pair_id / source when they had to be attached)."""
    rows: pd.DataFrame
    runs: list[RunDir]
    skipped: list[dict]
    census: dict[str, dict]

    @property
    def runs_table(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_dict() for r in self.runs])

    def n(self) -> dict:
        df = self.rows
        return {"items": int(df["item_id"].nunique()) if len(df) else 0,
                "base_pairs": int(df[CLUSTER].nunique()) if len(df) else 0,
                "models": len({r.model for r in self.runs}), "runs": len(self.runs),
                "lines": len({(r.line, r.model) for r in self.runs}), "rows": len(df),
                "skipped": len(self.skipped)}


def discover_run_dirs(runs: Path | str | Sequence[Path | str]) -> list[Path]:
    """Run directories from a runs root (its children holding a manifest.json), explicit run directories or
    manifest paths; duplicates removed, sorted."""
    paths = [runs] if isinstance(runs, (str, Path)) else list(runs)
    out: set[Path] = set()
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "manifest.json":
            out.add(p.parent.resolve())
        elif (p / "manifest.json").exists():
            out.add(p.resolve())
        elif p.is_dir():
            out |= {d.resolve() for d in p.iterdir() if d.is_dir() and (d / "manifest.json").exists()}
    return sorted(out, key=str)


def read_ledger(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def locate_run_dir(row: Mapping, runs_root: Path, root: Path = ROOT) -> Path | None:
    """The run directory of a done ledger row: the parent of its result_file, else `<runs>/<run>__<model>__<tag>`
    when the row carries a tag, else `<runs>/<run>__<model>`; None when no candidate holds a manifest.json."""
    candidates: list[Path] = []
    recorded = (row.get("result_file") or "").strip()
    if recorded:
        parent = Path(recorded).parent
        candidates += [parent, root / parent, runs_root / parent.name]
    if row.get("run_id") and row.get("model"):
        tag = (row.get("tag") or "").strip()
        if tag:
            candidates.append(runs_root / f"{row['run_id']}__{row['model']}__{tag}")
        candidates.append(runs_root / f"{row['run_id']}__{row['model']}")
    return next((c for c in candidates if (c / "manifest.json").exists()), None)


def ledger_done_units(ledger: Path, runs_root: Path, skipped: list[dict], root: Path = ROOT) -> dict[Path, dict]:
    """resolved run directory -> {arms (runner names), unit_ids, hash_verified} over the ledger's `done` units
    (scripts/ledger.py: the run finished and every expected row is present); a done unit whose item-file hash
    does not match the plan (hash_verified `no`, DD 4.6) and a unit without a directory are skipped with reasons."""
    found: dict[Path, dict] = {}
    for row in read_ledger(ledger):
        if row.get("status") != "done":
            continue
        unit = row.get("unit_id")
        if (row.get("hash_verified") or "").strip().lower() == "no":
            skipped.append({"unit_id": unit, "run_dir": row.get("result_file") or None,
                            "reason": "done but the item-file hash does not match the plan (DD 4.6): not analysed"})
            continue
        d = locate_run_dir(row, runs_root, root)
        if d is None:
            skipped.append({"unit_id": unit, "run_dir": None,
                            "reason": "done but no run directory with a manifest.json on this machine "
                                      f"(result_file {row.get('result_file') or '-'})"})
            continue
        ref = found.setdefault(d.resolve(), {"arms": set(), "unit_ids": [], "hash_verified": row.get("hash_verified") or "n/a"})
        if ref["hash_verified"] != (row.get("hash_verified") or "n/a"):
            ref["hash_verified"] = "mixed"
        ref["unit_ids"].append(unit or "")
        ref["arms"].add(tables.runner_arm_name(row.get("arm") or "base"))
    return found


def _read_results_sha(run_dir: Path) -> str | None:
    p = run_dir / "results_hashes.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("results_sha256")
    except (OSError, json.JSONDecodeError):
        return None


def _read_run(d: Path, unit: dict | None, models_cfg: Mapping | None, audit_dir: Path, include_smoke: bool,
              allow_thinking: bool, root: Path) -> tuple[RunDir, list[dict]]:
    manifest = read_manifest(d)
    status = manifest.get("status")
    identity = manifest.get("identity") or {}
    run_id = identity.get("run_id") or manifest.get("run_id") or d.name
    raw_line, model_from_id, tag = split_run_id(run_id)
    model = identity.get("model_key") or (manifest.get("model") or {}).get("name") or model_from_id or "?"
    if status == STATUS_THINKING and not allow_thinking:
        raise Unscorable(f"manifest status {STATUS_THINKING!r} ({manifest.get('n_thinking_chars_total')} thinking characters "
                         "in a main run): reasoning sub-study material, never a main-table row (DD 7.3)")
    if status not in (STATUS_FINISHED, STATUS_THINKING):
        raise Unscorable(f"not completed (manifest status {status!r})")
    opts = manifest.get("options") or {}
    smoke = bool((opts.get("notes") or {}).get("smoke") or (manifest.get("outcome") or {}).get("smoke"))
    if smoke and not include_smoke:
        raise Unscorable("a smoke run (options.notes.smoke): never a paper row (DD 8.8)")
    scores = d / "scores.jsonl"
    if not scores.exists():
        raise Unscorable("no scores.jsonl (run scripts/score_run.py first)")
    rows = read_jsonl(scores)
    if not rows:
        raise Unscorable("scores.jsonl is empty")
    note = attach_clusters(rows, manifest, root=root)
    arms_done = None
    if unit is not None:
        arms_done = set(unit["arms"])
        rows = [r for r in rows if (r.get("arm") or BASELINE_ARM) in arms_done]
        if not rows:
            raise Unscorable(f"no score rows for the done arms {sorted(arms_done)}")
    info = manifest.get("item_file") if isinstance(manifest.get("item_file"), dict) else {}
    data = manifest.get("data") or {}
    item_path = info.get("path") or data.get("item_file")
    item_key = Path(str(item_path)).name.split(".")[0] if item_path else None
    item_file = item_file_kind(item_path)
    census = census_for(model, manifest, models_cfg, audit_dir)
    entry = model_entry(models_cfg, model)
    api = is_api_entry(entry) if entry else is_api_manifest(manifest)
    line = LINE_ALIASES.get(raw_line, raw_line)
    for r in rows:
        if not r.get("arm"):
            r["arm"] = BASELINE_ARM
        if not r.get("arm_scope"):
            r["arm_scope"] = ARM_SCOPE_PRIMARY
        if r.get("c2_enriched") is None:
            r["c2_enriched"] = item_file == "c2"
        r.update(model=model, run_id=run_id, line=line, tag=tag, item_key=item_key, item_file=item_file, is_api=api,
                 census_nfd=census.get("verdict_nfd"), census_pc=census.get("verdict_pc"),
                 paraphrase=paraphrase_of(r.get("prompt_id")), condition=condition_of(r.get("prompt_id")))
    run = RunDir(path=d, run_id=run_id, line=line, raw_line=raw_line, model=model, tag=tag, status=str(status),
                 item_path=str(item_path) if item_path else None,
                 item_sha256=info.get("sha256") or data.get("item_file_sha256"), item_key=item_key, item_file=item_file,
                 smoke=smoke, is_api=api, census=census, results_sha256=_read_results_sha(d), n_rows=len(rows),
                 arms=sorted({str(r["arm"]) for r in rows}), prompt_ids=sorted({str(r.get("prompt_id")) for r in rows}),
                 cluster_note=note, unit_ids=list(unit["unit_ids"]) if unit else [],
                 hash_verified=unit["hash_verified"] if unit else "n/a (no ledger)")
    return run, rows


def load_runs(runs: Path | str | Sequence[Path | str] = RUNS_DIR, *, ledger: Path | str | None = None,
              models_cfg: Mapping | None = None, audit_dir: Path = AUDIT_DIR, include_smoke: bool = False,
              allow_thinking: bool = False, root: Path = ROOT) -> LoadedRuns:
    """Pooled score rows of every completed run directory. `runs` is the runs root, one or several run
    directories, or manifest paths. With a `ledger` that exists, only the `done` units' directories and arms
    enter (hash_verified `no` skipped, DD 4.6); without one, every directory whose manifest status is `finished`.
    `finished_thinking_present` main runs and smoke runs are skipped unless asked for. The rows of one
    (line, model[, cpu]) are pooled over directories and (item_id, arm, arm_scope, prompt_id) must then be unique:
    a duplicate means a paraphrase was scored twice for the line -> ValueError naming the directories (duplicates
    are never averaged away)."""
    skipped: list[dict] = []
    if isinstance(runs, (str, Path)):
        runs_root = Path(runs) if Path(runs).is_dir() and not (Path(runs) / "manifest.json").exists() else Path(runs).parent
    else:                                   # explicit run directories or manifest paths: their common parent is the root
        parents = {(Path(r).parent if Path(r).is_dir() else Path(r).parent.parent) for r in runs}
        runs_root = parents.pop() if len(parents) == 1 else RUNS_DIR
    done: dict[Path, dict] | None = None
    if ledger is not None and Path(ledger).exists():
        done = ledger_done_units(Path(ledger), runs_root, skipped, root)
        dirs = sorted(done, key=str)
    else:
        dirs = discover_run_dirs(runs)
    loaded: list[tuple[RunDir, list[dict]]] = []
    for d in dirs:
        try:
            run, rows = _read_run(d, done.get(d.resolve()) if done else None, models_cfg, audit_dir, include_smoke,
                                  allow_thinking, root)
        except Unscorable as e:
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": str(e)})
            continue
        except (OSError, json.JSONDecodeError, KeyError) as e:
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": f"unreadable run directory: {type(e).__name__}: {e}"})
            continue
        loaded.append((run, rows))
    groups: dict[tuple, list[tuple[RunDir, list[dict]]]] = defaultdict(list)
    for run, rows in loaded:
        groups[pool_key(run.raw_line, run.model, run.tag)].append((run, rows))
    frames = []
    for key in sorted(groups):
        members = groups[key]
        seen: dict[tuple, Path] = {}
        clash: set[Path] = set()
        for run, rows in members:
            for r in rows:
                k = tuple(r.get(c) for c in ROW_KEY)
                if k in seen:
                    clash |= {seen[k], run.path}
                else:
                    seen[k] = run.path
        if clash:
            raise ValueError(f"line {key[0]!r}, model {key[1]!r}: duplicate {ROW_KEY} rows across "
                             f"{sorted(p.name for p in clash)}: a paraphrase scored twice is never averaged away")
        frames.append(pd.DataFrame([r for _, rows in members for r in rows]))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*ROW_KEY, "model", "line", "tag"])
    census: dict[str, dict] = {}
    for run, _ in loaded:              # a censused block (from a run manifest) wins over a not_censused one
        if run.model not in census or (run.census.get("status") == "censused" and census[run.model].get("status") != "censused"):
            census[run.model] = run.census
    return LoadedRuns(df, [run for run, _ in loaded], skipped, census)


# ------------------------------------------------------------------ paraphrase means (PREREG 8.1)
def _as_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s.astype(object).where(s.notna(), np.nan), errors="coerce").to_numpy(dtype=float)


def paraphrase_means(df: pd.DataFrame, *, condition: str | None = None, value: str = "correct",
                     allow_mixed: bool = False) -> pd.DataFrame:
    """One row per (model, line, item_id, arm, arm_scope, condition, task, variant) with `correct_mean`, the mean
    of `value` over the DISTINCT paraphrases present (values in {0, 1/3, 2/3, 1} for three), `n_paraphrases`,
    `paraphrases` (sorted tuple), `correct_p0` (NaN when p0 is absent), `n_rows`, `n_unparseable`, and the item
    covariates of CARRY_COLUMNS. A missing paraphrase is never filled in. A cell (CELL_KEYS) mixing items with
    different numbers of paraphrases is refused unless `allow_mixed`, and is then reported through
    `n_paraphrases_cell` (item 52: 1- and 3-trial means are never mixed without saying so)."""
    d = df if condition is None else df[df["condition"] == condition]
    cols = [*MEANS_KEYS, "correct_mean", "n_paraphrases", "paraphrases", "correct_p0", "n_rows", "n_unparseable",
            "n_paraphrases_cell"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    keys = list(MEANS_KEYS)
    dup = d.duplicated(subset=[*keys, "paraphrase"], keep=False)
    if dup.any():
        ex = d.loc[dup, [*keys, "paraphrase"]].head(3).to_dict("records")
        raise ValueError(f"{int(dup.sum())} rows repeat an (item, arm, scope, condition, paraphrase): e.g. {ex}")
    d = d.assign(_v=_as_float(d[value]),
                 _unp=(d["error_class"] == UNPARSEABLE).astype(int) if "error_class" in d else 0)
    g = d.groupby(keys, sort=True, dropna=False)
    out = g.agg(correct_mean=("_v", "mean"), n_paraphrases=("paraphrase", "nunique"), n_rows=("_v", "size"),
                n_unparseable=("_unp", "sum"))
    out["paraphrases"] = g["paraphrase"].agg(lambda s: tuple(sorted(set(s.dropna()))))
    p0 = d[d["paraphrase"] == PRIMARY_PARAPHRASE].set_index(keys)["_v"]
    out["correct_p0"] = p0.reindex(out.index).to_numpy()
    carry = [c for c in CARRY_COLUMNS if c in d.columns]
    if carry:
        out = out.join(g[carry].first())
    out = out.reset_index()
    cells = out.groupby(list(CELL_KEYS), dropna=False)["n_paraphrases"].agg(lambda s: "|".join(str(x) for x in sorted(set(s))))
    mixed = cells[cells.str.contains("|", regex=False)]
    if len(mixed) and not allow_mixed:
        raise ValueError(f"{len(mixed)} cells mix items with different numbers of paraphrases "
                         f"(item 52; e.g. {mixed.head(3).to_dict()}): pass allow_mixed=True to keep them, reported per cell")
    out = out.merge(cells.rename("n_paraphrases_cell").reset_index(), on=list(CELL_KEYS), how="left")
    return out


# ------------------------------------------------------------------ cells (PREREG 8.2)
def _cell_ci(vals: np.ndarray, clusters: np.ndarray, strata: np.ndarray | None, n_boot: int, seed: int, level: float,
             stat: Callable | None = None, proportion: bool = True) -> tuple[bootstrap.CI, bool]:
    """bootstrap.cluster_bootstrap, stratified by base-pair type when the strata are well defined (a base pair in
    two strata is not a base-pair stratification: unstratified then, flagged)."""
    kw = {"n_boot": n_boot, "seed": seed, "level": level, "stat": stat, "proportion": proportion}
    if strata is not None:
        try:
            return bootstrap.cluster_bootstrap(vals, clusters, strata=strata, **kw), True
        except ValueError:
            pass
    return bootstrap.cluster_bootstrap(vals, clusters, **kw), False


def _strata_of(g: pd.DataFrame) -> np.ndarray | None:
    return g[STRATUM].to_numpy() if STRATUM in g and g[STRATUM].notna().all() else None


def _cluster_stats(vals: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.ravel()
    k = int(inv.max()) + 1 if len(inv) else 0
    return (float(bootstrap.design_effect_from_clusters(vals, inv, k)), float(bootstrap.anova_icc(vals, inv, k)), k)


def _n_pairs(g: pd.DataFrame) -> int:
    """Distinct yes/no pairs: the {item_id, pair_item_id} sets (an item without a partner counts alone)."""
    if "pair_item_id" not in g:
        return int(g["item_id"].nunique())
    pairs = {frozenset((i, p)) if isinstance(p, str) and p else frozenset((i,))
             for i, p in zip(g["item_id"], g["pair_item_id"])}
    return len(pairs)


def cell_table(data: pd.DataFrame, *, by: Sequence[str] = ("model", "task", "variant", "arm"), value: str | None = None,
               n_boot: int = constants.BOOTSTRAP_B, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Per cell of `by`: estimate, lo, hi, ci_method, small_cell, n_items, n_base_pairs, n_pairs (T3), deff, icc,
    unparseable_rate, n_paraphrases, claim_eligible (>= MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM items), n_rows.
    `data` is either the output of paraphrase_means (value `correct_mean`: non-binary from two paraphrases on, so
    the bootstrap keeps the percentile / BCa interval and flags small cells) or score rows (value `correct`:
    binary, so a small or 0 % / 100 % cell gets Wilson on n / DEFF). The interval is
    bootstrap.cluster_bootstrap(values, base_pair_id, strata=source) at every size; rows whose value is undefined
    (NaN) are dropped and counted in `n_undefined`."""
    means = "correct_mean" in data.columns
    value = value or ("correct_mean" if means else "correct")
    by = list(by)
    rows = []
    for keys, g in data.groupby(by, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        vals_all = _as_float(g[value])
        keep = np.isfinite(vals_all)
        g2, vals = g[keep], vals_all[keep]
        task = dict(zip(by, keys)).get("task", g2["task"].iloc[0] if len(g2) and "task" in g2 else None)
        rec = {**dict(zip(by, keys)), "value": value, "n_undefined": int((~keep).sum())}
        if len(vals) == 0:
            rows.append({**rec, "estimate": np.nan, "lo": np.nan, "hi": np.nan, "ci_method": None, "small_cell": None,
                         "level": level, "n_items": 0, "n_base_pairs": 0, "n_pairs": None, "deff": np.nan, "icc": np.nan,
                         "unparseable_rate": np.nan, "n_paraphrases": None, "n_paraphrases_set": "", "claim_eligible": False,
                         "n_rows": 0, "stratified": False})
            continue
        clusters = g2[CLUSTER].to_numpy()
        ci, stratified = _cell_ci(vals, clusters, _strata_of(g2), n_boot, seed, level)
        deff, icc, _ = _cluster_stats(vals, clusters)
        n_items = int(g2["item_id"].nunique())
        if means:
            n_rows = int(g2["n_rows"].sum())
            unp = float(g2["n_unparseable"].sum() / n_rows) if n_rows else np.nan
            n_par = sorted({int(x) for x in g2["n_paraphrases"]})
        else:
            n_rows = len(g2)
            unp = float((g2["error_class"] == UNPARSEABLE).mean()) if "error_class" in g2 else np.nan
            n_par = [int(g2["paraphrase"].nunique())] if "paraphrase" in g2 else []
        rows.append({**rec, "estimate": ci.estimate, "lo": ci.lo, "hi": ci.hi, "ci_method": ci.method,
                     "small_cell": ci.small_cell, "level": level, "n_items": n_items, "n_base_pairs": ci.n_clusters,
                     "n_pairs": _n_pairs(g2) if task == "T3" else None, "deff": deff, "icc": icc, "unparseable_rate": unp,
                     "n_paraphrases": n_par[0] if len(n_par) == 1 else None, "n_paraphrases_set": "|".join(map(str, n_par)),
                     "claim_eligible": n_items >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM, "n_rows": n_rows,
                     "stratified": stratified})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the primary metric (DD 5.1)
def _variant_mean_stat(variants: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    """stat(values, weights) for bootstrap.cluster_bootstrap: the weighted mean within each variant cell, then the
    unweighted mean over the cells present in the replicate (V1-V4 with equal weight, DD 5.1)."""
    masks = [np.asarray(variants) == c for c in np.unique(variants)]

    def stat(x: np.ndarray, w: np.ndarray) -> float:
        parts = [np.average(x[m], weights=w[m]) for m in masks if w[m].sum() > 0]
        return float(np.mean(parts)) if parts else float("nan")
    return stat


def _equal_weight_accuracy(rows: pd.DataFrame, value: str = "correct") -> float:
    if rows.empty:
        return float("nan")
    v = _as_float(rows[value])
    return _variant_mean_stat(rows["variant"].to_numpy())(v, np.ones_like(v))


def _default_rows(df: pd.DataFrame, condition: str, task: str = "T1") -> pd.DataFrame:
    return df[(df["task"] == task) & (df["arm"] == BASELINE_ARM) & (df["arm_scope"] == ARM_SCOPE_PRIMARY)
              & (df["condition"] == condition)]


def primary_metric(df: pd.DataFrame, *, api_models: Collection[str] | None = None, main_line: str = "E1_main",
                   api_core_line: str = "E1_api_core", api_paraphrase_line: str = "E1_api_paraphrase",
                   condition: str = DEFAULT_CONDITION, n_boot: int = constants.BOOTSTRAP_B, seed: int = 0,
                   level: float = 0.95) -> pd.DataFrame:
    """Per model: `acc_mean3` = the mean over V1-V4 of the four T1 cell means of the paraphrase mean (equal cell
    weight; base-pair cluster-bootstrap CI with the variant-mean statistic), `acc_p0`, `acc_p1`, `acc_p2` (the
    same statistic on each paraphrase alone), `range_lo` / `range_hi` over them, cluster counts, and the
    `metric` label. API models (`api_models`, else the rows' `is_api`) report p0 on the core (`api_core_line`)
    and the range on the paraphrase subset (`api_paraphrase_line`, p1 / p2) with p0 restricted to the same items
    (DD 5.1); `acc_mean3` is left undefined for them, never a 3-trial mean mixed with 1-trial ones."""
    base = _default_rows(df, condition)
    if api_models is not None:
        api_set = set(api_models)
    else:
        api_set = set(base.loc[base["is_api"].fillna(False).astype(bool), "model"]) if "is_api" in base else set()
    rows = []
    for model in sorted(base["model"].unique()):
        sub = base[base["model"] == model]
        rows.append(_api_primary(sub, model, api_core_line, api_paraphrase_line, n_boot, seed, level) if model in api_set
                    else _open_primary(sub[sub["line"] == main_line], model, main_line, n_boot, seed, level))
    return pd.DataFrame(rows)


def _open_primary(sub: pd.DataFrame, model: str, line: str, n_boot: int, seed: int, level: float) -> dict:
    rec = {"model": model, "metric": OPEN_METRIC, "is_api": False, "line": line}
    if sub.empty:
        return {**rec, "note": f"no {line} rows under the default condition"}
    means = paraphrase_means(sub)
    vals = means["correct_mean"].to_numpy(dtype=float)
    ci, stratified = _cell_ci(vals, means[CLUSTER].to_numpy(), _strata_of(means), n_boot, seed, level,
                              stat=_variant_mean_stat(means["variant"].to_numpy()))
    per_p = {p: _equal_weight_accuracy(sub[sub["paraphrase"] == p]) for p in PARAPHRASES if (sub["paraphrase"] == p).any()}
    cells = means.groupby("variant")["item_id"].nunique()
    return {**rec, "acc_mean3": ci.estimate, "lo": ci.lo, "hi": ci.hi, "ci_method": ci.method, "small_cell": ci.small_cell,
            "level": level, "acc_p0": per_p.get("p0", np.nan), "acc_p1": per_p.get("p1", np.nan),
            "acc_p2": per_p.get("p2", np.nan), "range_lo": min(per_p.values()) if per_p else np.nan,
            "range_hi": max(per_p.values()) if per_p else np.nan, "range_on": "all items",
            "paraphrases": "|".join(sorted(per_p)), "n_items": int(means["item_id"].nunique()),
            "n_base_pairs": ci.n_clusters, "n_paraphrases_set": "|".join(sorted(set(means["n_paraphrases_cell"]))),
            "n_variants": len(cells), "min_cell_items": int(cells.min()),
            "claim_eligible": bool(cells.min() >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM), "stratified": stratified,
            "note": None}


def _api_primary(sub: pd.DataFrame, model: str, core_line: str, para_line: str, n_boot: int, seed: int, level: float) -> dict:
    rec = {"model": model, "metric": API_METRIC, "is_api": True, "line": f"{core_line}+{para_line}"}
    core = sub[(sub["line"] == core_line) & (sub["paraphrase"] == PRIMARY_PARAPHRASE)]
    if core.empty:
        return {**rec, "note": f"no {core_line} p0 rows under the default condition"}
    vals = _as_float(core["correct"])
    ci, stratified = _cell_ci(vals, core[CLUSTER].to_numpy(), _strata_of(core), n_boot, seed, level,
                              stat=_variant_mean_stat(core["variant"].to_numpy()))
    para = sub[(sub["line"] == para_line) & (sub["paraphrase"] != PRIMARY_PARAPHRASE)]
    ids = set(para["item_id"])
    per_p: dict[str, float] = {}
    if ids:
        per_p["p0"] = _equal_weight_accuracy(core[core["item_id"].isin(ids)])
        for p in PARAPHRASES[1:]:
            sp = para[para["paraphrase"] == p]
            if not sp.empty:
                per_p[p] = _equal_weight_accuracy(sp)
    cells = core.groupby("variant")["item_id"].nunique()
    return {**rec, "acc_mean3": np.nan, "lo": ci.lo, "hi": ci.hi, "ci_method": ci.method, "small_cell": ci.small_cell,
            "level": level, "acc_p0": ci.estimate, "acc_p0_subset": per_p.get("p0", np.nan),
            "acc_p1": per_p.get("p1", np.nan), "acc_p2": per_p.get("p2", np.nan),
            "range_lo": min(per_p.values()) if per_p else np.nan, "range_hi": max(per_p.values()) if per_p else np.nan,
            "range_on": f"{len(ids)} paraphrase-subset items" if ids else "n/a (p0 only)",
            "paraphrases": "|".join(sorted(per_p)) if per_p else "p0", "n_items": int(core["item_id"].nunique()),
            "n_base_pairs": ci.n_clusters, "n_paraphrases_set": "1", "n_variants": len(cells),
            "min_cell_items": int(cells.min()), "claim_eligible": bool(cells.min() >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM),
            "stratified": stratified, "note": None}


# ------------------------------------------------------------------ the all-three-paraphrases rule (PREREG 8.13)
def direction_holds(diffs: Mapping[str, float], direction: int | None = None,
                    paraphrases: Sequence[str] = PARAPHRASES) -> dict:
    """PREREG 8.13: a claim about a model holds only if its direction holds under all three paraphrases.
    `diffs` maps paraphrase -> contrast (b - a); `direction` (+1 / -1) is the predicted sign when there is one.
    Returns holds (all paraphrases present, every sign equal, non-zero, and equal to `direction` when given),
    sign, n_paraphrases, missing."""
    present = {p: float(diffs[p]) for p in paraphrases if p in diffs and np.isfinite(diffs[p])}
    missing = [p for p in paraphrases if p not in present]
    signs = {int(np.sign(v)) for v in present.values()}
    one = len(signs) == 1 and 0 not in signs
    holds = not missing and one and (direction is None or signs == {int(direction)})
    if missing:
        reason = f"paraphrase(s) {', '.join(missing)} absent"
    elif not one:
        reason = "the sign differs between paraphrases" if len(signs) > 1 else "a zero contrast"
    elif direction is not None and signs != {int(direction)}:
        reason = "the common sign is not the predicted one"
    else:
        reason = None
    return {"holds": bool(holds), "sign": next(iter(signs)) if len(signs) == 1 else None, "n_paraphrases": len(present),
            "missing": missing, "reason": reason}


def _select(df: pd.DataFrame, where: Mapping[str, object]) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for col, val in where.items():
        m &= df[col].isin(list(val)) if isinstance(val, (list, tuple, set, frozenset)) else (df[col] == val)
    return df[m]


def paraphrase_contrast(df: pd.DataFrame, a: Mapping[str, object], b: Mapping[str, object], *, by: Sequence[str] = ("model",),
                        pair_on: Sequence[str] = ("item_id",), value: str = "correct", direction: int | None = None) -> pd.DataFrame:
    """mean(b) - mean(a) per paraphrase, rows paired on `pair_on` within each paraphrase, one row per `by` group,
    with `diff_p0`, `diff_p1`, `diff_p2`, `n_p*` and direction_holds(...) (`holds`, `sign`, `reason`). `a` and `b`
    are column filters, e.g. a={"line": "E1_main", "arm": "nfc"}, b={"line": "E1_abl_zero_shot"}."""
    by, pair_on = list(by), list(pair_on)
    da, db = _select(df, a), _select(df, b)
    rows = []
    for keys in sorted({tuple(x) for x in pd.concat([da[by], db[by]]).itertuples(index=False)}):
        where = dict(zip(by, keys))
        ga, gb = _select(da, where), _select(db, where)
        rec = dict(where)
        diffs: dict[str, float] = {}
        for p in PARAPHRASES:
            pa = ga[ga["paraphrase"] == p].set_index(pair_on)[value]
            pb = gb[gb["paraphrase"] == p].set_index(pair_on)[value]
            common = pa.index.intersection(pb.index)
            rec[f"n_{p}"] = len(common)
            if len(common):
                diffs[p] = float(_as_float(pb.loc[common]).mean() - _as_float(pa.loc[common]).mean())
            rec[f"diff_{p}"] = diffs.get(p, np.nan)
        rows.append({**rec, **direction_holds(diffs, direction)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ within-item ATE tau_m(a) (PREREG 8.5)
def _cell_rows(g: pd.DataFrame, task: str | None, file: str | None) -> pd.DataFrame:
    """The (task, file) cell exactly as tables.intervention_table selects it (its `_cell_rows`): the file
    coordinate applies through the `c2_enriched` column."""
    sub = g[g["task"] == task] if task is not None and "task" in g else g
    if file is not None and "c2_enriched" in sub:
        flag = sub["c2_enriched"].fillna(False).astype(bool)
        sub = sub[flag] if file == "c2" else sub[~flag]
    return sub


def _first(g: pd.DataFrame, col: str):
    if col not in g or g.empty:
        return None
    s = g[col].dropna()
    return s.iloc[0] if len(s) else None


def _e3_frame(dm: pd.DataFrame, lines: Sequence[str], baseline_arm: str, baseline_lines: Mapping[str, str]) -> pd.DataFrame | None:
    """One model's p0 whole_prompt rows of the E3 lines with their baseline: the line's own baseline rows, else
    (E3_api_core) the mapped line's baseline rows on the same items and prompts. XCOPA's outcome is
    `logprob_correct` (DD 5.7; rows without it leave), the placement arm keeps C2-affected items only (H4 "on
    C2-affected items"; on the C2-enriched file every row is one), and rows without a stratum get NO_STRATUM."""
    parts = []
    for line in lines:
        dl = dm[dm["line"] == line]
        arms = dl[dl["arm"] != baseline_arm]
        if arms.empty:
            continue
        base, base_line = dl[dl["arm"] == baseline_arm], line
        if base.empty and line in baseline_lines:
            cand = dm[(dm["line"] == baseline_lines[line]) & (dm["arm"] == baseline_arm)]
            base = cand[cand["item_id"].isin(set(arms["item_id"])) & cand["prompt_id"].isin(set(arms["prompt_id"]))]
            base_line = baseline_lines[line]
        if base.empty:
            continue
        parts.append(pd.concat([arms, base]).assign(e3_line=line, baseline_line=base_line))
    if not parts:
        return None
    frame = pd.concat(parts, ignore_index=True)
    frame["correct"] = _as_float(frame["correct"])
    for task, col in OUTCOME_BY_TASK.items():
        m = frame["task"] == task
        if m.any():
            frame.loc[m, "correct"] = _as_float(frame.loc[m, col]) if col in frame else np.nan
    frame = frame[np.isfinite(frame["correct"].to_numpy(dtype=float))].copy()
    frame["correct"] = frame["correct"].astype(bool)
    if "c2_affected" in frame:
        unaffected = (frame["arm"] == "placement_new") & ~frame["c2_affected"].fillna(True).astype(bool)
        frame = frame[~unaffected]
    if "c2_enriched" not in frame:
        frame["c2_enriched"] = frame["item_file"] == "c2"
    frame["c2_enriched"] = frame["c2_enriched"].fillna(False).astype(bool)
    if STRATUM not in frame:
        frame[STRATUM] = NO_STRATUM
    frame[STRATUM] = frame[STRATUM].fillna(NO_STRATUM)
    dup = frame.duplicated(subset=["arm", "task", "c2_enriched", "item_id", "prompt_id"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} rows of model {dm['model'].iloc[0]!r} repeat an (arm, task, file, item, prompt) "
                         f"across lines {sorted(set(frame.loc[dup, 'e3_line']))}: one baseline per cell, never a duplicate")
    return frame


def _pair_stats(cell: pd.DataFrame, baseline_arm: str, arm: str) -> dict:
    """What tables.intervention_table does not return for a paired cell: distinct items, the realized design
    effect of the item-level differences over base pairs, the paired-t SE (its square is the within-model
    variance a pooled estimate needs) and the claim flag. Pairing by (item_id, prompt_id), as the table does."""
    base = cell[cell["arm"] == baseline_arm].set_index(["item_id", "prompt_id"])
    other = cell[cell["arm"] == arm].set_index(["item_id", "prompt_id"])
    common = base.index.intersection(other.index)
    if len(common) == 0:
        return {"n_items": 0, "deff": np.nan, "se_paired_t": np.nan, "claim_eligible": False, "line": _first(cell, "e3_line")}
    a = base.loc[common, "correct"].astype(float).to_numpy()
    b = other.loc[common, "correct"].astype(float).to_numpy()
    cl = base.loc[common, CLUSTER].to_numpy()
    deff, _, _ = _cluster_stats(b - a, cl)
    n_items = len({i for i, _ in common})
    return {"n_items": n_items, "deff": deff, "se_paired_t": bootstrap.paired_t_base_pairs(a, b, cl).se,
            "claim_eligible": n_items >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM, "line": _first(cell, "e3_line")}


EFFECT_COLUMNS = ("model", "line", "task", "variant", "arm", "item_file", "cell_kind", "outcome", "acc_base", "acc_arm", "effect",
                  "ci_lo", "ci_hi", "ci_method", "small_cell", "se_paired_t", "p_paired_t", "p_bootstrap", "p_mcnemar",
                  "p_primary", "p_source", "p_holm", "significant", "family_size", "family_defined", "holm_family", "n",
                  "n_items", "n_base_pairs", "deff", "claim_eligible", "c2_affected_only", ZERO_BY_CONSTRUCTION, "refused",
                  "reason", "is_api")


def _effect_row(r: Mapping, extra: Mapping, **override) -> dict:
    rec = {**dict(r), **dict(extra), **override}
    rec.setdefault("outcome", OUTCOME_BY_TASK.get(rec.get("task"), "correct"))
    rec.setdefault("c2_affected_only", rec.get("arm") == "placement_new")
    rec.setdefault(ZERO_BY_CONSTRUCTION, False)
    rec.setdefault("refused", False)
    rec.setdefault("reason", None)
    return {c: rec.get(c) for c in EFFECT_COLUMNS}


def _undefined_row(model: str, task: str, arm: str, file: str | None, base: pd.DataFrame, kind: str, reason: str,
                   api: bool) -> dict:
    acc = float(_as_float(base["correct"]).mean())
    zero = kind == ZERO_BY_CONSTRUCTION
    return _effect_row({"model": model, "task": task, "arm": arm, "item_file": file}, {},
                       line=_first(base, "e3_line"), variant=None, cell_kind=kind, acc_base=acc, acc_arm=acc if zero else None,
                       effect=0.0 if zero else None, ci_lo=None, ci_hi=None, ci_method=None, small_cell=None,
                       se_paired_t=None, p_paired_t=None, p_bootstrap=None, p_mcnemar=None, p_primary=None, p_source=None,
                       p_holm=None, significant=None, family_size=None, family_defined=constants.HOLM_FAMILY_TABLE3_CELLS,
                       holm_family=None, n=len(base), n_items=int(base["item_id"].nunique()),
                       n_base_pairs=int(base[CLUSTER].nunique()), deff=None, claim_eligible=False,
                       **{ZERO_BY_CONSTRUCTION: zero}, refused=not zero, reason=reason, is_api=api)


def intervention_effects(df: pd.DataFrame, *, baseline_arm: str = BASELINE_ARM,
                         family: Sequence[tuple[str, str, str | None]] | None = None, census: Mapping[str, Mapping] | None = None,
                         api_models: Collection[str] | None = None, lines: Sequence[str] | None = None,
                         exclude_lines: Sequence[str] = APPENDIX_LINES, baseline_lines: Mapping[str, str] | None = None,
                         variants: Sequence[str] = T1_VARIANTS, n_boot: int = constants.BOOTSTRAP_B, seed: int = 0,
                         use: str = "primary") -> pd.DataFrame:
    """tau_m(a) per (model, task[, variant], arm) on p0 whole_prompt rows (the E3 lines are p0-only, so tau is a
    binary quantity and tests.paired_test applies). The Table 3 family (tables.table3_family(), 14 cells) is
    FILTERED PER MODEL before tables.intervention_table: the nfd cells leave when the census says the tokenizer
    normalizes NFD, the win1258 cells when it normalizes PC (both become "0 by construction" rows: effect 0, no
    interval, no p), a corrupting arm is refused and reported, API models have no XCOPA cell and their C2 arm
    runs on the core subset (C2-affected rows, item file `main`, kept in their family: a reading of PREREG 8.3
    recorded in the handoff). Holm is applied within the model row (`cell_kind == "family"`); the per-variant T1
    rows (`cell_kind == "variant"`, H3's tau(nfd, T1-V3) among them) carry no Holm column. Every row: effect,
    CI (base-pair cluster bootstrap), p_paired_t, p_bootstrap, p_mcnemar (secondary, ignores clustering),
    p_primary + p_source, n, n_items, n_base_pairs, deff, claim_eligible, se_paired_t."""
    base_lines = dict(BASELINE_LINES if baseline_lines is None else baseline_lines)
    fam_all = list(tables.table3_family() if family is None else family)
    d = df[(df["paraphrase"] == PRIMARY_PARAPHRASE) & (df["arm_scope"] == ARM_SCOPE_PRIMARY)]
    if lines is None:
        lines = sorted(set(d.loc[d["arm"] != baseline_arm, "line"]) - set(exclude_lines))
    if api_models is not None:
        api_set = set(api_models)
    else:
        api_set = set(d.loc[d["is_api"].fillna(False).astype(bool), "model"]) if "is_api" in d else set()
    census = dict(census or {})
    out: list[dict] = []
    for model in sorted(d["model"].unique()):
        dm = d[d["model"] == model]
        cm = census.get(model) or {"verdict_nfd": _first(dm, "census_nfd"), "verdict_pc": _first(dm, "census_pc")}
        zero, bad = normalizing_arms(cm), corrupting_arms(cm)
        frame = _e3_frame(dm, list(lines), baseline_arm, base_lines)
        if frame is None:
            continue
        api = model in api_set

        def cell_file(arm: str, file: str | None, api: bool = api) -> str | None:
            return None if api and arm == "placement_new" else file

        fam_m = [(arm, task, cell_file(arm, file)) for arm, task, file in fam_all
                 if arm not in zero and arm not in bad and not (api and task == "XCOPA")]
        tab = tables.intervention_table(frame, baseline_arm=baseline_arm, family=fam_m, n_boot=n_boot, use=use)
        for _, r in tab.iterrows():
            cell = _cell_rows(frame, r["task"], r["item_file"])
            out.append(_effect_row(r, _pair_stats(cell, baseline_arm, r["arm"]), variant=None, cell_kind="family",
                                   holm_family="Table 3 model row (PREREG 8.3)", is_api=api))
        for arm, task, file in fam_all:
            if (arm not in zero and arm not in bad) or (api and task == "XCOPA"):
                continue
            base = _cell_rows(frame, task, cell_file(arm, file))
            base = base[base["arm"] == baseline_arm]
            if base.empty:
                continue
            if arm in zero:
                reason = (f"census {CENSUS_ARM_VERDICTS[arm]} == normalizes: identical token ids, Delta = 0 exactly, "
                          "a sanity assertion and not a test (PREREG 8.3)")
                out.append(_undefined_row(model, task, arm, file, base, ZERO_BY_CONSTRUCTION, reason, api))
            else:
                reason = (f"census {CENSUS_ARM_VERDICTS[arm]} == corrupts: the tokenizer path does not return the "
                          "re-encoded input; the arm is refused on this engine and reported (DD 6.2, PREREG 5.5)")
                out.append(_undefined_row(model, task, arm, file, base, "refused", reason, api))
        t1 = frame[frame["task"] == "T1"]
        for arm, task, file in fam_m:
            if task != "T1":
                continue
            for v in variants:
                fv = t1[t1["variant"] == v]
                if fv.empty:
                    continue
                tv = tables.intervention_table(fv, baseline_arm=baseline_arm, family=[(arm, "T1", file)], n_boot=n_boot, use=use)
                for _, r in tv.iterrows():
                    cell = _cell_rows(fv, "T1", file)
                    out.append(_effect_row(r, _pair_stats(cell, baseline_arm, arm), variant=v, cell_kind="variant",
                                           holm_family=None, p_holm=None, significant=None, family_size=None,
                                           family_defined=None, is_api=api))
    return pd.DataFrame(out, columns=list(EFFECT_COLUMNS))


# ------------------------------------------------------------------ the explicit-input gap (DD 12.17)
def explicit_input_gap(df: pd.DataFrame, *, line: str = "E1_explicit_input", baseline_lines: Sequence[str] = ("E1_main", "E1_api_core"),
                       components_condition: str = COMPONENTS_CONDITION, raw_condition: str = DEFAULT_CONDITION,
                       n_boot: int = constants.BOOTSTRAP_B, seed: int = 0, level: float = 0.95, by_task: bool = True) -> pd.DataFrame:
    """gap_m = mean_i[Y_i(components) - Y_i(raw)] over the core items under p0, nfc, whole_prompt: the
    components rows of `line` against the raw rows of the first baseline line with rows for the model (E1_main's
    p0 rows with in_core, or E1_api_core's), paired by item_id. CI by bootstrap.paired_difference_ci (stratified
    by base-pair type), paired t on base-pair means (primary from PAIRED_T_MIN_BASE_PAIRS base pairs) and the
    bootstrap p beside it; one row per model (`scope == "all"`) and per task when `by_task`. Estimation only,
    no Holm (PREREG 8.3); the label says the estimand is derived, not quoted."""
    comp_all = df[(df["line"] == line) & (df["paraphrase"] == PRIMARY_PARAPHRASE) & (df["arm"] == BASELINE_ARM)
                  & (df["arm_scope"] == ARM_SCOPE_PRIMARY) & (df["condition"] == components_condition)]
    raw_all = df[(df["paraphrase"] == PRIMARY_PARAPHRASE) & (df["arm"] == BASELINE_ARM)
                 & (df["arm_scope"] == ARM_SCOPE_PRIMARY) & (df["condition"] == raw_condition)]
    rows = []
    for model in sorted(comp_all["model"].unique()):
        comp = comp_all[comp_all["model"] == model]
        raw, base_line = pd.DataFrame(), None
        for bl in baseline_lines:
            cand = raw_all[(raw_all["model"] == model) & (raw_all["line"] == bl)]
            if "in_core" in cand and bl != "E1_api_core":
                cand = cand[cand["in_core"].fillna(False).astype(bool)]
            if not cand.empty:
                raw, base_line = cand, bl
                break
        if raw.empty:
            rows.append({"model": model, "scope": "all", "line": line, "baseline_line": None, "label": EXPLICIT_INPUT_LABEL,
                         "note": f"no raw-condition core baseline among {list(baseline_lines)}"})
            continue
        scopes = [("all", comp, raw)]
        if by_task:
            scopes += [(t, comp[comp["task"] == t], raw[raw["task"] == t]) for t in sorted(comp["task"].unique())]
        for scope, c, r in scopes:
            rows.append(_gap_row(model, scope, c, r, line, base_line, n_boot, seed, level))
    return pd.DataFrame(rows)


def _gap_row(model: str, scope: str, comp: pd.DataFrame, raw: pd.DataFrame, line: str, base_line: str | None,
             n_boot: int, seed: int, level: float) -> dict:
    rec = {"model": model, "scope": scope, "line": line, "baseline_line": base_line, "label": EXPLICIT_INPUT_LABEL, "note": None}
    r = raw.drop_duplicates("item_id").set_index("item_id")
    c = comp.drop_duplicates("item_id").set_index("item_id")
    common = r.index.intersection(c.index)
    if len(common) == 0:
        return {**rec, "note": "no item paired between the components and raw conditions", "n_items": 0}
    a, b = _as_float(r.loc[common, "correct"]), _as_float(c.loc[common, "correct"])
    cl = r.loc[common, CLUSTER].to_numpy()
    strata = r.loc[common, STRATUM].to_numpy() if STRATUM in r and r.loc[common, STRATUM].notna().all() else None
    try:
        ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed, level=level, strata=strata)
        p_bs = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed, strata=strata)
        stratified = strata is not None
    except ValueError:
        ci = bootstrap.paired_difference_ci(a, b, cl, n_boot=n_boot, seed=seed, level=level)
        p_bs = bootstrap.paired_bootstrap_pvalue(a, b, cl, n_boot=n_boot, seed=seed)
        stratified = False
    pt = bootstrap.paired_t_base_pairs(a, b, cl)
    deff, _, _ = _cluster_stats(b - a, cl)
    n_items = len(common)
    return {**rec, "acc_raw": float(a.mean()), "acc_components": float(b.mean()), "gap": ci.estimate, "lo": ci.lo, "hi": ci.hi,
            "ci_method": ci.method, "small_cell": ci.small_cell, "level": level, "se_paired_t": pt.se, "t_paired": pt.t,
            "p_paired_t": pt.p, "p_bootstrap": p_bs, "p_primary": pt.p if pt.primary else p_bs,
            "p_source": "paired_t" if pt.primary else "bootstrap", "n_items": n_items, "n_base_pairs": ci.n_clusters,
            "deff": deff, "claim_eligible": n_items >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM, "stratified": stratified}


# ------------------------------------------------------------------ the T3 headline (DD 5.3)
def _records(g: pd.DataFrame) -> list[dict]:
    return g.astype(object).where(g.notna(), None).to_dict("records")


def _rate(g: pd.DataFrame, col: str) -> tuple[float, int]:
    if col not in g:
        return float("nan"), 0
    v = _as_float(g[col])
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else float("nan")), len(v)


def t3_headline(df: pd.DataFrame, *, by: Sequence[str] = ("model", "line", "arm"), n_boot: int = constants.BOOTSTRAP_B,
                seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Per `by` group of T3 whole_prompt rows: generated accuracy with all items, WITHOUT the `spelling` twins and
    their partners (noilai.eval.score._spelling_pair_ids; the headline, DD 5.3) and on the spelling pairs alone,
    balanced accuracy / d' / yes-rate (noilai.eval.score.balanced_stats) on the headline rows, n_items, n_pairs,
    and the log-probability modes in their own columns: `t3_yesno_lp` (forced_choice_correct), `t3_pair_lp`
    (paired_correct, the summed log-probability) and `t3_pair_lp_norm` (paired_correct_norm). No column of this
    table contrasts a generated number with a log-probability one (DD 5.3 item 63; PREREG 8.2)."""
    assert constants.T3_HEADLINE_EXCLUDES_TWIN == "spelling"
    d = df[(df["task"] == "T3") & (df["arm_scope"] == ARM_SCOPE_PRIMARY)]
    by = list(by)
    rows = []
    for keys, g in d.groupby(by, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        recs = _records(g)
        spelling = _spelling_pair_ids(recs)
        excl_mask = ~g["item_id"].isin(spelling)
        excl, only = g[excl_mask], g[~excl_mask]

        def acc(sub: pd.DataFrame) -> bootstrap.CI | None:
            if sub.empty:
                return None
            ci, _ = _cell_ci(_as_float(sub["correct"]), sub[CLUSTER].to_numpy(), _strata_of(sub), n_boot, seed, level)
            return ci

        ci_all, ci_excl, ci_only = acc(g), acc(excl), acc(only)
        bal = balanced_stats(_records(excl), "pred") if len(excl) else {}
        yes_lp, n_yes_lp = _rate(g, "forced_choice_correct")
        pair_lp, n_pair_lp = _rate(g, "paired_correct")
        pair_lp_norm, _ = _rate(g, "paired_correct_norm")
        rows.append({**dict(zip(by, keys)), "acc_all": ci_all.estimate if ci_all else np.nan,
                     "acc_excl_spelling": ci_excl.estimate if ci_excl else np.nan,
                     "lo_excl_spelling": ci_excl.lo if ci_excl else np.nan, "hi_excl_spelling": ci_excl.hi if ci_excl else np.nan,
                     "ci_method": ci_excl.method if ci_excl else None, "small_cell": ci_excl.small_cell if ci_excl else None,
                     "acc_spelling_pairs": ci_only.estimate if ci_only else np.nan,
                     "balanced_accuracy": bal.get("balanced_accuracy"), "d_prime": bal.get("d_prime"), "yes_rate": bal.get("yes_rate"),
                     "n_items": int(g["item_id"].nunique()), "n_items_excl_spelling": int(excl["item_id"].nunique()),
                     "n_pairs": _n_pairs(g), "n_pairs_excl_spelling": _n_pairs(excl) if len(excl) else 0,
                     "n_spelling_items": int(only["item_id"].nunique()), "n_base_pairs": ci_excl.n_clusters if ci_excl else 0,
                     "n_rows": len(g), "n_paraphrases": int(g["paraphrase"].nunique()) if "paraphrase" in g else None,
                     "t3_yesno_lp": yes_lp, "t3_yesno_lp_n": n_yes_lp, "t3_pair_lp": pair_lp, "t3_pair_lp_norm": pair_lp_norm,
                     "t3_pair_lp_n": n_pair_lp, "headline_excludes_twin": constants.T3_HEADLINE_EXCLUDES_TWIN,
                     "logprob_columns_note": "t3_yesno_lp / t3_pair_lp are separate scoring modes, never compared with acc_* (DD 5.3)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ writing (never under paper/, never item rows)
def refuse_paper(path: Path) -> None:
    """CLAUDE.md: nothing of this workstream is written under paper/."""
    if Path(path).resolve().is_relative_to(PAPER_DIR.resolve()):
        raise ValueError(f"refusing to write under paper/ ({path}): the experiments workstream never writes paper files")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset, tuple)):
        return sorted(o) if isinstance(o, (set, frozenset)) else list(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _frame_records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def write_aggregates(out_dir: Path, tables_: Mapping[str, pd.DataFrame | Mapping | Sequence], *, label: str = "INTERIM",
                     meta: Mapping | None = None) -> dict[str, list[Path]]:
    """Each table as `<name>.csv` (DataFrames) and `<name>.json` ({label, table, n_rows, columns, rows, meta})
    under `out_dir` (default experiments/analysis/ at the entry point). Refuses a directory under paper/ and any
    DataFrame carrying an `item_id` column (item-level rows are never written out)."""
    out_dir = Path(out_dir)
    refuse_paper(out_dir)
    for name, t in tables_.items():
        if isinstance(t, pd.DataFrame) and "item_id" in t.columns:
            raise ValueError(f"table {name!r} carries item-level rows (an item_id column): aggregates only are written")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for name, t in tables_.items():
        paths = []
        if isinstance(t, pd.DataFrame):
            csv_path = out_dir / f"{name}.csv"
            t.to_csv(csv_path, index=False)
            paths.append(csv_path)
            body = {"label": label, "table": name, "n_rows": len(t), "columns": list(map(str, t.columns)),
                    "rows": _frame_records(t), "meta": dict(meta or {})}
        else:
            body = {"label": label, "table": name, "n_rows": len(t) if hasattr(t, "__len__") else None, "columns": None,
                    "rows": t, "meta": dict(meta or {})}
        js = out_dir / f"{name}.json"
        js.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        paths.append(js)
        written[name] = paths
    return written
