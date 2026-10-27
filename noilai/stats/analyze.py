"""The pre-registered analysis entry point (PREREGISTRATION 8.13, DESIGN_DECISIONS 8.8): every table and figure
file of the paper from the completed runs, labelled, with its provenance.

    python -m noilai.stats.analyze --config configs/analysis.yaml --runs data/runs [--ledger experiments/ledger.csv]
                                   [--out experiments/analysis] [--label INTERIM|FINAL] [--date YYYY-MM-DD]
                                   [--n-boot B] [--seed S] [--include-smoke] [--check] [--quiet]

What it does, in order: (1) reads configs/analysis.yaml and asserts that every mirrored threshold agrees with
noilai/constants.py (alpha, the bootstrap settings, the Table 3 Holm family, the arm aliases); (2) selects the
COMPLETED units -- with a ledger, rows with status `done` and hash_verified `yes` / `n/a` (a `no` is skipped and
reported, DESIGN_DECISIONS 4.6; a `dev` split or a non-confirmatory kind never enters a table); without one, every
run directory under --runs whose manifest says `finished` (smoke manifests are skipped unless --include-smoke);
(3) maps every directory `<run>__<model>[__<tag>]` to (line, model, tag) with the `tpu_main -> E1_main` alias,
pools the `p0p1` + `p2` directories of one (line, model) into one set of rows and keeps a `cpu` unit apart;
(4) computes the tables through noilai.stats.bootstrap / tests / tables / pooling (and, when present,
noilai.stats.aggregate as a cross-check written into the .json siblings), with statsmodels'
multipletests(method="holm") for the Holm families of PREREGISTRATION 8.3 taken from noilai.constants
HOLM_FAMILY_TABLE3 (the yaml carries the same definition, asserted); (5) writes every table as .tex (header:
generator, commit, content sha256, generated-at; then the label line with n) AND .json, plus analysis_summary.json,
results_log_entry.md (the docs/RESULTS_LOG.md entry text with the exact call; this module never edits docs/) and
analysis_report.md, to --out. Every artefact is labelled INTERIM with n (items, base pairs, models) unless
--label FINAL is asked AND every requirement of labels.final_requires holds (the registration recorded, the
scoring rule frozen and `--check` passing, every live unit of the Table 2 / Table 3 lines done and hash-verified,
no smoke run analysed); otherwise FINAL is refused with the reasons (exit 3). Zero completed runs -> labelled
empty tables, exit 0. --out under paper/ is refused (exit 2). --check recomputes and compares every .tex body
(the stamp lines ignored), exit 1 when stale. Nothing is item-level in the outputs.

The code implements docs/PREREGISTRATION.md section 8 and docs/DESIGN_DECISIONS.md sections 1, 8 and 9; where
the two differ the difference is recorded in the handoff, never edited into either. The H1 item-audit join, the
H3b tone-isolation DiD and the H6 matched contrast live in noilai.stats.matching (PREREG 8.4, 8.7, 8.8) and run here
when their inputs (the item audits under paths.audit_dir / --audit-dir, the item files) are on the machine; an analysis
whose inputs are absent (those, the probes) is written as a `not computable` artefact with the reason, never silently
omitted. scripts/ is never imported: the run-directory conventions of
scripts/interim_analysis.py are re-implemented here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from .. import constants
from ..eval.prompts import ARM_ALIASES
from ..eval.run import CENSUS_ARM_VERDICTS, STATUS_FINISHED, STATUS_THINKING, load_item_file, read_manifest
from ..eval.score import _spelling_pair_ids, balanced_stats, read_jsonl
from . import aggregate as AGG  # the aggregation helper (DD 8.1): the Table 2 cross-check
from . import bootstrap, dose_response, tables
from . import e2 as E2  # the E2 regression and the H1 nested test (PREREG 8.4)
from . import matching as MT  # the item-audit join, the H3b DiD and the H6 matching (PREREG 8.4, 8.7, 8.8)
from . import pooling as PL  # the hierarchical pool, k of N and TOST (DD 1, PREREG 8.1)
from . import tests as T

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"                         # CLAUDE.md: this workstream never writes under paper/
DEFAULT_CONFIG = ROOT / "configs" / "analysis.yaml"
GENERATOR = "noilai.stats.analyze"
LABEL_INTERIM, LABEL_FINAL = "INTERIM", "FINAL"
NOTICE = ("interim numbers are for monitoring only; no design or analysis change without a "
          "docs/DEVIATIONS.md entry; no paper prose")
EXIT_OK, EXIT_STALE, EXIT_USAGE, EXIT_FINAL_REFUSED = 0, 1, 2, 3
STAMP_PREFIXES = ("% GENERATED", "% generator commit")      # the two stamp lines --check ignores
NON_LIVE_STATUSES = frozenset({"cut", "zero_by_construction", "unscheduled", "alternative"})   # scripts/ledger.py
CLUSTER_KEY = constants.BOOTSTRAP_CLUSTER
STRATUM_KEY = "source"
SHA_SHORT = 12
# the pre-registered analyses as scripts/progress.py:117-128 names them (re-typed here: scripts/ is never imported)
ANALYSES = ("Table 2 (E1 estimation per done model)", "Explicit-input gap, DD 12.17 main result",
            "H1 nested LR test (E2), PREREG 8.4", "H3 pooled crossover (E3), PREREG 8.5", "H3b tone-isolation DiD",
            "H4 placement (C2 set)", "H5 probes and patching (E4)", "H6 attested vs matched (needs the 100-row floor)",
            "Human-baseline comparison (mean-human band)")
# the score-row fields the analyses read (docs/DATA_FORMAT.md; noilai/eval/score.py)
ROW_FIELDS = ("item_id", "task", "variant", "arm", "arm_scope", "prompt_id", "correct", "correct_lenient", "error_class",
              "base_pair_id", "source", "in_core", "c2_affected", "pair_item_id", "twin_type", "forced_choice_correct",
              "paired_correct", "paired_correct_norm", "logprob_correct", "logprob_correct_norm", "delta_prompt_tokens",
              "delta_n_tokens", "input_lexical", "output_lexical", "attested_overlap", "spelling_triggers", "tone_pair",
              "degenerate", "exact", "eligible_h6", "n_thinking_chars", "pred", "gold", "input_freq", "output_freq")
T3_LOGPROB_KEYS = ("forced_choice_correct", "paired_correct", "paired_correct_norm")


class Unscorable(RuntimeError):
    """A completed run whose numbers cannot be computed here (reported, never a crash)."""


# ============================================================================ configuration
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict:
    """configs/analysis.yaml, validated against noilai.constants (ValueError naming every disagreement)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise TypeError(f"{path}: not a mapping")
    validate_config(cfg)
    return dict(cfg)


def config_table3_family(cfg: Mapping) -> tuple[tuple[str, str, str], ...]:
    return tuple(tuple(str(x) for x in cell) for cell in (cfg.get("holm") or {}).get("table3_family") or ())


def validate_config(cfg: Mapping) -> None:
    """The yaml mirrors noilai/constants.py; the constants module is the source (DD 8.8, item 29)."""
    bad = []
    holm = cfg.get("holm") or {}
    if float(holm.get("alpha", -1)) != constants.ALPHA:
        bad.append(f"holm.alpha {holm.get('alpha')!r} != constants.ALPHA {constants.ALPHA}")
    if config_table3_family(cfg) != constants.HOLM_FAMILY_TABLE3:
        bad.append("holm.table3_family differs from constants.HOLM_FAMILY_TABLE3")
    bs = cfg.get("bootstrap") or {}
    mirror = {"B": constants.BOOTSTRAP_B, "cluster": constants.BOOTSTRAP_CLUSTER, "bca_min_clusters": constants.BCA_MIN_CLUSTERS,
              "small_cell_max_base_pairs": constants.SMALL_CELL_MAX_BASE_PAIRS,
              "paired_t_min_base_pairs": constants.PAIRED_T_MIN_BASE_PAIRS}
    bad += [f"bootstrap.{k} {bs.get(k)!r} != constants {v!r}" for k, v in mirror.items() if bs.get(k) != v]
    aliases = (cfg.get("arms") or {}).get("aliases") or {}
    if dict(aliases) != dict(ARM_ALIASES):
        bad.append(f"arms.aliases {aliases!r} != noilai.eval.prompts.ARM_ALIASES {ARM_ALIASES!r}")
    if cfg.get("arm_scope_primary", constants.ARM_SCOPE_PRIMARY) != constants.ARM_SCOPE_PRIMARY:
        bad.append("arm_scope_primary differs from constants.ARM_SCOPE_PRIMARY")
    if cfg.get("h4_direction_from", "BASELINE_PLACEMENT") != "BASELINE_PLACEMENT":
        bad.append("h4_direction_from must name constants.BASELINE_PLACEMENT (never a hard-coded sign)")
    if bad:
        raise ValueError("configs/analysis.yaml disagrees with noilai/constants.py: " + "; ".join(bad))


def resolve_path(cfg: Mapping, key: str, root: Path = ROOT) -> Path:
    p = Path(str((cfg.get("paths") or {}).get(key, key)))
    return p if p.is_absolute() else root / p


def tokenizer_family_map(cfg: Mapping) -> dict[str, str]:
    """model -> tokenizer family (families.by_model); empty -> ValueError (pooling never guesses it)."""
    if PL is not None:
        return PL.tokenizer_family_map(cfg)
    m = (cfg.get("families") or {}).get("by_model") or {}
    if not m:
        raise ValueError("configs/analysis.yaml carries no families.by_model map (model -> tokenizer family of PREREG 8.1)")
    return {str(k): str(v) for k, v in m.items()}


def holm_adjust(pvalues: Sequence[float], alpha: float = constants.ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' Holm step-down (PREREG 8.3): (reject, p_adjusted). Numerically identical to noilai.stats.tests.holm
    (tests/test_analyze.py asserts it on random inputs); empty input -> empty arrays."""
    p = np.asarray(list(pvalues), dtype=float)
    if len(p) == 0:
        return np.zeros(0, dtype=bool), p
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method="holm")
    return np.asarray(reject, dtype=bool), np.asarray(p_adj, dtype=float)


def apply_holm(tests_: Sequence[T.PairedTest], alpha: float = constants.ALPHA) -> list[T.PairedTest]:
    """One Holm family over the given tests (their primary p of design 8.2), through multipletests."""
    if not tests_:
        return []
    reject, p_adj = holm_adjust([t.p_primary for t in tests_], alpha)
    for t, r, a in zip(tests_, reject, p_adj, strict=True):
        t.p_adj, t.significant, t.p_primary_source = float(a), bool(r), t.p_primary_name
    return list(tests_)


# ============================================================================ run discovery
@dataclass
class RunRef:
    """One completed run directory and the ledger units that point at it."""
    run_dir: Path
    unit_ids: list[str] = field(default_factory=list)
    arms: set[str] | None = None                      # ledger mode: the arms whose units are done; None = every arm
    hash_verified: str = "n/a (no ledger)"
    ledger_rows: list[dict] = field(default_factory=list)
    smoke: bool = False


def split_run_id(run_id: str) -> tuple[str, str, str]:
    """`<run>__<model>[__<tag>]` -> (line, model, tag); a run id without `__` is its own line (model unknown)."""
    parts = run_id.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return run_id, "", ""


def paraphrase_of(prompt_id: str | None) -> str | None:
    """`vi-p0-s3-explained-raw` / `xcopa-p0` -> `p0` (noilai.eval.prompts.prompt_id; run.py:459-460)."""
    parts = str(prompt_id or "").split("-")
    if len(parts) >= 2 and parts[1].startswith("p") and parts[1][1:].isdigit():
        return parts[1]
    return None


def condition_of(prompt_id: str | None) -> str:
    """The prompt id with the paraphrase replaced by `*` (the condition every paraphrase shares)."""
    parts = str(prompt_id or "").split("-")
    if len(parts) >= 2 and paraphrase_of(prompt_id):
        parts[1] = "*"
    return "-".join(parts)


def item_file_class(item_key: str | None, path: str | None) -> str:
    """`main` / `c2` / `xcopa` / `core` / `attested` / ... from the ledger's item_key or the manifest's item file."""
    stem = (item_key or Path(path or "").stem or "").lower()
    if "c2" in stem.split("_"):
        return "c2"
    if "xcopa" in stem:
        return "xcopa"
    if stem in ("noilai_main", "noilai_test"):
        return "main"
    if stem.startswith("noilai_"):
        return stem[len("noilai_"):]
    return stem or "unknown"


def read_ledger(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def locate_run_dir(row: Mapping, runs_root: Path) -> Path | None:
    """The run directory of a done ledger row (the recorded result_file's parent, the tagged path, the untagged path)."""
    candidates: list[Path] = []
    recorded = (row.get("result_file") or "").strip()
    if recorded:
        parent = Path(recorded).parent
        candidates += [parent, ROOT / parent, runs_root / parent.name]
    if row.get("run_id") and row.get("model"):
        tag = (row.get("tag") or "").strip()
        if tag:
            candidates.append(runs_root / f"{row['run_id']}__{row['model']}__{tag}")
        candidates.append(runs_root / f"{row['run_id']}__{row['model']}")
    for cand in candidates:
        if (cand / "manifest.json").exists():
            return cand
    return None


def manifest_is_smoke(manifest: Mapping) -> bool:
    opts = manifest.get("options") or {}
    notes = opts.get("notes") if isinstance(opts, Mapping) else None
    return bool((isinstance(notes, Mapping) and notes.get("smoke")) or (manifest.get("outcome") or {}).get("smoke"))


def completed_from_ledger(ledger: Path, runs_root: Path, cfg: Mapping, skipped: list[dict]) -> list[RunRef]:
    """The run directories of the ledger's completed units, one RunRef per directory with its done arms."""
    comp = cfg.get("completed") or {}
    statuses = set(comp.get("ledger_statuses") or ["done"])
    allowed_hv = {str(x).lower() for x in (comp.get("hash_verified_allowed") or ["yes", "n/a"])}
    kinds = set(comp.get("kinds") or [])
    excl_splits = set(comp.get("exclude_splits") or [])
    refs: dict[Path, RunRef] = {}
    for row in read_ledger(ledger):
        if row.get("status") not in statuses:
            continue
        uid = row.get("unit_id")
        hv = (row.get("hash_verified") or "n/a").strip().lower()
        if hv not in allowed_hv:
            skipped.append({"unit_id": uid, "run_dir": row.get("result_file") or None,
                            "reason": f"done but hash_verified {hv!r}: the item-file hash does not match the plan (DD 4.6); "
                                      "not analysed"})
            continue
        if kinds and (row.get("kind") or "") and row.get("kind") not in kinds:
            skipped.append({"unit_id": uid, "run_dir": None, "reason": f"kind {row.get('kind')!r}: never a paper-table row"})
            continue
        if (row.get("split") or "") in excl_splits:
            skipped.append({"unit_id": uid, "run_dir": None, "reason": f"split {row.get('split')!r}: dev results appear in no table"})
            continue
        run_dir = locate_run_dir(row, runs_root)
        if run_dir is None:
            skipped.append({"unit_id": uid, "run_dir": None,
                            "reason": "done but no run directory with a manifest.json "
                                      f"(result_file {row.get('result_file') or '-'}): a human or probe deliverable, or a "
                                      "run not mirrored on this machine"})
            continue
        key = run_dir.resolve()
        ref = refs.get(key)
        if ref is None:
            ref = refs[key] = RunRef(key, arms=set(), hash_verified=row.get("hash_verified") or "n/a")
        elif ref.hash_verified != (row.get("hash_verified") or "n/a"):
            ref.hash_verified = "mixed"
        ref.unit_ids.append(uid or "")
        ref.ledger_rows.append(dict(row))
        ref.arms.add(row.get("arm") or "nfc")
    return [refs[k] for k in sorted(refs)]


def expand_run_paths(paths: Sequence[Path], skipped: list[dict]) -> list[Path]:
    """--runs: a runs root, explicit run directories or manifest paths -> run directories (with a manifest.json)."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "manifest.json":
            out.append(p.parent)
        elif p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
        elif p.is_dir():
            out += sorted(d for d in p.iterdir() if d.is_dir() and (d / "manifest.json").exists())
        else:
            skipped.append({"unit_id": None, "run_dir": str(p), "reason": "no such runs root, run directory or manifest.json"})
    return out


def completed_from_runs(paths: Sequence[Path], cfg: Mapping, skipped: list[dict], include_smoke: bool = False) -> list[RunRef]:
    """Every run directory whose manifest says the run finished (no ledger: no hash gate, reported as such)."""
    comp = cfg.get("completed") or {}
    finished = set(comp.get("manifest_statuses") or [STATUS_FINISHED])
    exclude_smoke = bool(comp.get("exclude_smoke_manifests", True)) and not include_smoke
    refs: list[RunRef] = []
    for d in expand_run_paths(paths, skipped):
        try:
            manifest = read_manifest(d)
        except (OSError, json.JSONDecodeError) as e:
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": f"unreadable manifest.json: {e}"})
            continue
        status = manifest.get("status")
        if status not in finished and status != STATUS_THINKING:     # a thinking-present run is judged per line below
            skipped.append({"unit_id": None, "run_dir": str(d), "reason": f"not completed (manifest status {status!r})"})
            continue
        smoke = manifest_is_smoke(manifest)
        if smoke and exclude_smoke:
            skipped.append({"unit_id": None, "run_dir": str(d),
                            "reason": "smoke manifest (options.notes.smoke): never a paper-table row (--include-smoke to exercise)"})
            continue
        refs.append(RunRef(d.resolve(), smoke=smoke))
    return refs


# ============================================================================ loading the rows
def models_index(models_cfg: Mapping | None) -> dict[str, dict]:
    """configs/models.yaml entries by name with their group, family, hardware, provider, reference_of, panel flag."""
    if not models_cfg:
        return {}
    groups = models_cfg.get("groups") or {}
    panel_groups = {g for g, spec in groups.items() if isinstance(spec, Mapping) and spec.get("panel")}
    out = {}
    for e in models_cfg.get("models") or []:
        out[str(e.get("name"))] = {"group": e.get("group"), "family": e.get("family"), "hardware": e.get("hardware"),
                                  "provider": e.get("provider"), "reference_of": e.get("reference_of"), "hf_id": e.get("hf_id"),
                                  "panel": e.get("group") in panel_groups}
    return out


def census_of(manifest: Mapping, model: str, midx: Mapping, audit_dir: Path) -> dict:
    """The model's census verdicts: the manifest's data.normalization_census when censused, else the fallback
    data/audit/<models.yaml family>.json (PREREG 8.6; kaggle_run_plan census_skipped_arms)."""
    cen = ((manifest.get("data") or {}).get("normalization_census") or {}) if isinstance(manifest.get("data"), Mapping) else {}
    out = {"status": cen.get("status"), "source": cen.get("source"), "engine": cen.get("engine"),
           "legacy_census": cen.get("legacy_census"), "classification": cen.get("classification"),
           "verdict_nfd": cen.get("verdict_nfd"), "verdict_pc": cen.get("verdict_pc"), "verdict_source": "manifest"}
    if out["verdict_nfd"] is None and out["verdict_pc"] is None:
        fam = (midx.get(model) or {}).get("family")
        path = audit_dir / f"{fam}.json" if fam else None
        if path is not None and path.exists():
            try:
                fb = (json.loads(path.read_text(encoding="utf-8")).get("normalization_census") or {})
                out.update(verdict_nfd=fb.get("verdict_nfd"), verdict_pc=fb.get("verdict_pc"),
                           verdict_source=f"fallback data/audit/{fam}.json", fallback_file=str(path.relative_to(ROOT))
                           if path.is_relative_to(ROOT) else str(path))
            except (OSError, json.JSONDecodeError, AttributeError):
                out["verdict_source"] = f"fallback data/audit/{fam}.json unreadable"
        else:
            out["verdict_source"] = "none (not censused, no data/audit/<family>.json)"
    return out


def manifest_excerpt(manifest: Mapping, run_dir: Path) -> dict:
    ident = manifest.get("identity") or {}
    eng = manifest.get("engine") if isinstance(manifest.get("engine"), Mapping) else {}
    hw = manifest.get("hardware") if isinstance(manifest.get("hardware"), Mapping) else {}
    info = manifest.get("item_file") if isinstance(manifest.get("item_file"), Mapping) else {}
    data = manifest.get("data") if isinstance(manifest.get("data"), Mapping) else {}
    results_sha = None
    rh = run_dir / "results_hashes.json"
    if rh.exists():
        try:
            results_sha = json.loads(rh.read_text(encoding="utf-8")).get("results_sha256")
        except (OSError, json.JSONDecodeError):
            results_sha = None
    return {"model_key": ident.get("model_key"), "revision": ident.get("revision"), "revision_kind": ident.get("revision_kind"),
            "tokenizer_sha256": ident.get("tokenizer_sha256"), "engine": eng.get("engine") or manifest.get("backend"),
            "engine_version": eng.get("engine_version"), "dtype_requested": eng.get("dtype_requested"),
            "dtype_resolved": eng.get("dtype_resolved"), "quantization": eng.get("quantization"),
            "accelerator_kind": hw.get("accelerator_kind"), "n_gpus": hw.get("n_gpus"), "wall_s": manifest.get("wall_s"),
            "gpu_hours": manifest.get("gpu_hours"), "tpu_hours": manifest.get("tpu_hours"), "status": manifest.get("status"),
            "item_file": info.get("path") or data.get("item_file"),
            "item_file_sha256": info.get("sha256") or data.get("item_file_sha256"), "results_sha256": results_sha,
            "git_commit": manifest.get("git_commit"), "git_dirty": manifest.get("git_dirty"),
            "n_thinking_chars_total": manifest.get("n_thinking_chars_total")}


@dataclass
class RunRecord:
    run_id: str
    line: str                    # after the alias (tpu_main -> E1_main)
    raw_line: str
    model: str
    tag: str
    unit: str                    # "cpu" for the CPU-engine unit kept apart, else ""
    run_dir: Path
    ref: RunRef
    excerpt: dict
    census: dict
    smoke: bool
    item_class: str
    n_rows: int
    n_items: int
    n_base_pairs: int
    arms: list[str]
    prompt_ids: list[str]
    cluster_note: str | None
    inputs: list[dict]

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("ref", "run_dir")}
        d["run_dir"] = str(self.run_dir)
        d["unit_ids"] = list(self.ref.unit_ids)
        d["hash_verified"] = self.ref.hash_verified
        return d


def _sha_record(path: Path, kind: str) -> dict:
    return {"path": str(path), "kind": kind, "sha256": sha256_file(path)}


def load_run(ref: RunRef, cfg: Mapping, midx: Mapping, audit_dir: Path) -> tuple[RunRecord, list[dict]]:
    """One completed run directory: its record and its score rows (the completed arms only), each row carrying
    model, line, tag, paraphrase, condition, the census verdicts and the item-file class. No rescoring here
    (scripts/ is never imported): a run without scores.jsonl is unscorable."""
    manifest = read_manifest(ref.run_dir)
    ident = manifest.get("identity") or {}
    run_id = ident.get("run_id") or manifest.get("run_id") or ref.run_dir.name
    raw_line, model_from_id, tag = split_run_id(run_id)
    aliases = (cfg.get("lines") or {}).get("aliases") or {}
    line = str(aliases.get(raw_line, raw_line))
    model = ident.get("model_key") or (manifest.get("model") or {}).get("name") or model_from_id or "?"
    comp = cfg.get("completed") or {}
    if manifest.get("status") == STATUS_THINKING and line not in set(comp.get("thinking_status_allowed_for_lines") or []):
        raise Unscorable(f"manifest status {STATUS_THINKING!r} ({manifest.get('n_thinking_chars_total')} thinking characters "
                         "in a main run, DD 7.3): reasoning sub-study material, never a main-table row")
    scores = ref.run_dir / "scores.jsonl"
    if not scores.exists():
        raise Unscorable("no scores.jsonl (run scripts/score_run.py first; analyze never scores)")
    rows = read_jsonl(scores)
    if not rows:
        raise Unscorable("scores.jsonl is empty")
    if ref.arms is not None:
        rows = [r for r in rows if (r.get("arm") or "nfc") in ref.arms]
        if not rows:
            raise Unscorable(f"no score rows for the completed arms {sorted(ref.arms)}")
    separate = set((cfg.get("lines") or {}).get("separate_tags") or ["cpu"])
    unit = tag if tag in separate else ""
    census = census_of(manifest, model, midx, audit_dir)
    item_key = next((r.get("item_key") for r in ref.ledger_rows if r.get("item_key")), None)
    info = manifest.get("item_file") if isinstance(manifest.get("item_file"), Mapping) else {}
    item_class = item_file_class(item_key, info.get("path") or (manifest.get("data") or {}).get("item_file"))
    api_lines = set((cfg.get("lines") or {}).get("api") or [])
    is_api = ((midx.get(model) or {}).get("hardware") == "api") or line in api_lines
    missing_cluster = 0
    out_rows = []
    for r in rows:
        row = {k: r.get(k) for k in ROW_FIELDS}
        if not row.get(CLUSTER_KEY):
            row[CLUSTER_KEY] = row.get("item_id")          # XCOPA has no base pair: cluster by item (check_run.py)
            missing_cluster += 1
        row.update(model=model, run_id=run_id, line=line, raw_line=raw_line, tag=tag, unit=unit,
                   paraphrase=paraphrase_of(row.get("prompt_id")), condition=condition_of(row.get("prompt_id")),
                   census_nfd=census.get("verdict_nfd"), census_pc=census.get("verdict_pc"), item_class=item_class,
                   c2_enriched=item_class == "c2", is_api=bool(is_api), smoke=ref.smoke)
        out_rows.append(row)
    inputs = [_sha_record(ref.run_dir / "manifest.json", "manifest"), _sha_record(scores, "scores")]
    if (ref.run_dir / "results_hashes.json").exists():
        inputs.append(_sha_record(ref.run_dir / "results_hashes.json", "results_hashes"))
    rec = RunRecord(run_id=run_id, line=line, raw_line=raw_line, model=model, tag=tag, unit=unit, run_dir=ref.run_dir, ref=ref,
                    excerpt=manifest_excerpt(manifest, ref.run_dir), census=census, smoke=ref.smoke, item_class=item_class,
                    n_rows=len(out_rows), n_items=len({r["item_id"] for r in out_rows}),
                    n_base_pairs=len({r[CLUSTER_KEY] for r in out_rows}), arms=sorted({str(r.get("arm") or "nfc") for r in out_rows}),
                    prompt_ids=sorted({str(r.get("prompt_id")) for r in out_rows}),
                    cluster_note=(f"{missing_cluster} rows without {CLUSTER_KEY}: clustered by item_id" if missing_cluster else None),
                    inputs=inputs)
    return rec, out_rows


@dataclass
class Context:
    """Everything the analyses read: the pooled frame, the run records, the configuration and the settings."""
    cfg: dict
    df: pd.DataFrame
    runs: list[RunRecord]
    unscorable: list[dict]
    skipped: list[dict]
    n_boot: int
    paired_boot: int
    seed: int
    level: float
    midx: dict
    ledger_rows: list[dict] | None
    paths: dict
    label: str = LABEL_INTERIM
    inputs: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_cells: int = 0                 # the Table 2 cells, set once the artefacts are computed
    cache: dict = field(default_factory=dict)   # item audits and item files read once (keyed by path)

    @property
    def n(self) -> dict:
        d = self.df
        return {"items": int(d["item_id"].nunique()) if len(d) else 0,
                "base_pairs": int(d[CLUSTER_KEY].nunique()) if len(d) else 0,
                "models": int(d["model"].nunique()) if len(d) else 0, "runs": len(self.runs),
                "lines": len({(r.line, r.model, r.unit) for r in self.runs}), "cells": self.n_cells,
                "unscorable": len(self.unscorable), "skipped": len(self.skipped)}

    def family_map(self) -> dict[str, str] | None:
        try:
            return tokenizer_family_map(self.cfg)
        except ValueError as e:
            self.notes.append(str(e))
            return None

    def rows(self, line: str | Sequence[str] | None = None, *, arm: str | None = None, paraphrase: str | None = None,
             condition: str | None = None, task: str | None = None, scope: bool = True) -> pd.DataFrame:
        """The frame restricted to a line (or lines), arm, paraphrase, condition and task; `scope` keeps the primary
        arm scope only (constants.ARM_SCOPE_PRIMARY)."""
        d = self.df
        if len(d) == 0:
            return d
        m = np.ones(len(d), dtype=bool)
        if line is not None:
            lines = [line] if isinstance(line, str) else list(line)
            m &= d["line"].isin(lines).values
        if arm is not None:
            m &= (d["arm"] == arm).values
        if paraphrase is not None:
            m &= (d["paraphrase"] == paraphrase).values
        if condition is not None:
            m &= (d["condition"] == condition).values
        if task is not None:
            m &= (d["task"] == task).values
        if scope:
            m &= (d["arm_scope"].fillna(constants.ARM_SCOPE_PRIMARY) == constants.ARM_SCOPE_PRIMARY).values
        return d[m]


def build_frame(runs: list[RunRecord], rows_by_run: dict[str, list[dict]], unscorable: list[dict]) -> tuple[pd.DataFrame, list[RunRecord]]:
    """Pool the rows of every directory of one (line, model, unit) and refuse duplicates: a row key (item_id, arm,
    arm_scope, prompt_id) seen twice within one pool marks the whole pool unscorable (the directories named)."""
    pools: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in runs:
        pools[(r.line, r.model, r.unit)].append(r)
    kept_runs, frames = [], []
    for key in sorted(pools):
        recs = pools[key]
        seen: dict[tuple, str] = {}
        dup = None
        for rec in recs:
            for row in rows_by_run[rec.run_id]:
                k = (row.get("item_id"), row.get("arm"), row.get("arm_scope"), row.get("prompt_id"))
                if k in seen and seen[k] != rec.run_id:
                    dup = (k, seen[k], rec.run_id)
                    break
                seen.setdefault(k, rec.run_id)
            if dup:
                break
        if dup:
            k, a, b = dup
            unscorable.append({"run_dir": ", ".join(str(x.run_dir) for x in recs), "unit_ids": [u for x in recs for u in x.ref.unit_ids],
                               "reason": f"duplicate row {k} in directories {a!r} and {b!r} of one (line, model): the paraphrase "
                                         "split must be disjoint; not analysed"})
            continue
        kept_runs += recs
        frames += [pd.DataFrame(rows_by_run[rec.run_id]) for rec in recs]
    if not frames:
        return pd.DataFrame(columns=[*ROW_FIELDS, "model", "run_id", "line", "raw_line", "tag", "unit", "paraphrase", "condition",
                                     "census_nfd", "census_pc", "item_class", "c2_enriched", "is_api", "smoke"]), kept_runs
    df = pd.concat(frames, ignore_index=True)
    df[STRATUM_KEY] = df[STRATUM_KEY].where(df[STRATUM_KEY].notna(), "none")
    df["correct"] = df["correct"].fillna(False).astype(bool)
    return df, kept_runs


def collect(refs: list[RunRef], cfg: Mapping, midx: Mapping, audit_dir: Path, skipped: list[dict]) -> tuple[list[RunRecord], pd.DataFrame, list[dict]]:
    runs, unscorable = [], []
    rows_by_run: dict[str, list[dict]] = {}
    for ref in refs:
        try:
            rec, rows = load_run(ref, cfg, midx, audit_dir)
        except Unscorable as e:
            unscorable.append({"run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids), "reason": str(e)})
            continue
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            unscorable.append({"run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids),
                               "reason": f"unreadable run directory: {type(e).__name__}: {e}"})
            continue
        if rec.run_id in rows_by_run:
            unscorable.append({"run_dir": str(ref.run_dir), "unit_ids": list(ref.unit_ids),
                               "reason": f"run id {rec.run_id!r} appears in two directories"})
            continue
        runs.append(rec)
        rows_by_run[rec.run_id] = rows
    df, runs = build_frame(runs, rows_by_run, unscorable)
    return runs, df, unscorable


# ============================================================================ statistics helpers
def _finite(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _strata(g: pd.DataFrame):
    s = g[STRATUM_KEY].values if STRATUM_KEY in g else None
    return None if s is None or any(x is None for x in s) else s


def acc_cell(g: pd.DataFrame, ctx: Context, value: str = "correct") -> dict:
    """Accuracy of one cell with the base-pair cluster-bootstrap CI (DD 8.2), items, base pairs, DEFF, ICC and the
    unparseable rate (DD 8.9); stratified by `source` when every base pair lies in one stratum."""
    v = g[value].fillna(False).astype(bool).values if value in g else np.zeros(len(g), dtype=bool)
    cl = g[CLUSTER_KEY].values
    kw = {"n_boot": ctx.n_boot, "seed": ctx.seed, "level": ctx.level}
    strata = _strata(g)
    stratified = strata is not None
    try:
        ci = bootstrap.accuracy_ci(v, cl, strata=strata, **kw)
    except ValueError:
        stratified = False
        ci = bootstrap.accuracy_ci(v, cl, **kw)
    vals = np.asarray(v, dtype=float)
    inv, k = bootstrap._cluster_index(cl)
    deff = bootstrap.design_effect_from_clusters(vals, inv, k)
    icc = bootstrap.anova_icc(vals, inv, k)
    n_unp = int((g["error_class"] == "unparseable").sum()) if "error_class" in g else 0
    return {"estimate": ci.estimate, "lo": ci.lo, "hi": ci.hi, "ci_method": ci.method, "small_cell": bool(ci.small_cell),
            "n_rows": len(g), "n_items": int(g["item_id"].nunique()), "n_base_pairs": int(ci.n_clusters),
            "deff": round(float(deff), 3), "icc": round(float(icc), 3), "unparseable_rate": n_unp / len(g) if len(g) else None,
            "stratified": stratified, "level": ctx.level, "n_boot": ci.n_boot,
            "claim_eligible": int(g["item_id"].nunique()) >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM}


def align_pairs(base: pd.DataFrame, other: pd.DataFrame, keys: Sequence[str] = ("item_id", "prompt_id"),
                value_a: str = "correct", value_b: str = "correct") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Rows of `other` paired with `base` on `keys`: (a, b, clusters, strata) or None when nothing pairs."""
    if len(base) == 0 or len(other) == 0:
        return None
    keys = list(keys)
    b_ = base.drop_duplicates(keys).set_index(keys)
    o_ = other.drop_duplicates(keys).set_index(keys)
    common = b_.index.intersection(o_.index)
    if len(common) == 0:
        return None
    a = b_.loc[common, value_a].fillna(False).astype(bool).values
    b = o_.loc[common, value_b].fillna(False).astype(bool).values
    cl = b_.loc[common, CLUSTER_KEY].values
    st = b_.loc[common, STRATUM_KEY].values if STRATUM_KEY in b_ else None
    return a, b, cl, st


def paired_contrast(base: pd.DataFrame, other: pd.DataFrame, ctx: Context, name: str, keys: Sequence[str] = ("item_id", "prompt_id"),
                    value_a: str = "correct", value_b: str = "correct") -> tuple[T.PairedTest, dict] | None:
    """tests.paired_test on the paired rows plus the base-pair paired t's se (for pooling) and the counts."""
    al = align_pairs(base, other, keys, value_a, value_b)
    if al is None:
        return None
    a, b, cl, st = al
    try:
        t = T.paired_test(a, b, cl, name=name, n_boot=ctx.paired_boot, seed=ctx.seed, strata=st)
    except ValueError:
        t = T.paired_test(a, b, cl, name=name, n_boot=ctx.paired_boot, seed=ctx.seed)
    pt = bootstrap.paired_t_base_pairs(a, b, cl)
    extra = {"se": pt.se, "n": len(a), "n_base_pairs": int(pt.n_clusters), "arrays": (a, b, cl)}
    return t, extra


def test_row(t: T.PairedTest, extra: dict) -> dict:
    """One JSON-safe record of a PairedTest (every number the paper may quote, with its cluster count)."""
    return {"effect": t.diff, "acc_base": t.acc_a, "acc_arm": t.acc_b, "ci_lo": t.ci_lo, "ci_hi": t.ci_hi, "ci_method": t.ci_method,
            "small_cell": t.small_cell, "p_paired_t": t.p_paired_t, "p_bootstrap": t.p_bootstrap, "p_mcnemar": t.p_mcnemar,
            "p_primary": t.p_primary, "p_source": t.p_primary_name, "p_holm": t.p_adj, "significant": t.significant,
            "n": extra["n"], "n_base_pairs": extra["n_base_pairs"], "se_paired_t": extra["se"],
            "claim_eligible": extra["n"] >= constants.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM}


def pooled(ctx: Context, records: list[dict], *, direction: int | None, min_families: int, label: str) -> dict:
    """noilai.stats.pooling.pool across tokenizer families (DD 1 / PREREG 8.1); a dict with `computable` False and
    the reason when the helper, the family map or an informative family is missing."""
    if PL is None:
        return {"computable": False, "reason": "noilai.stats.pooling is not importable"}
    fam = ctx.family_map()
    if fam is None:
        return {"computable": False, "reason": ctx.notes[-1]}
    if not records:
        return {"computable": False, "reason": "no per-model estimate to pool"}
    try:
        res = PL.pool(records, families=fam, method=str((ctx.cfg.get("pooling") or {}).get("method", "reml_hk")), level=ctx.level,
                      n_boot=ctx.n_boot, seed=ctx.seed, direction=direction, min_reliable_families=min_families, label=label)
    except ValueError as e:
        return {"computable": False, "reason": str(e)}
    out = res.as_dict()
    out["computable"] = True
    return out


def _audit_key_of(ctx: Context, model: str, line: str) -> str | None:
    """The run plan's item-file key of the file a model's line ran on (`noilai_main` for .../noilai_main.jsonl): the
    `__<key>` suffix of scripts/kaggle_cpu_jobs.py's item audits."""
    for r in ctx.runs:
        if r.model == model and r.line == line and r.excerpt.get("item_file"):
            return Path(str(r.excerpt["item_file"])).stem
    return None


def items_audit_for(ctx: Context, model: str, key: str | None) -> tuple[pd.DataFrame | None, str]:
    """The model's item audit (scripts/audit_items.py rows for its tokenizer and item file; data/audit/items_<tokenizer>__<key>.jsonl,
    named by hf_id, then models.yaml family, then the model name) as a frame, or (None, why)."""
    entry = ctx.midx.get(model) or {}
    cands = MT.items_audit_candidates(ctx.paths["audit_dir"], hf_id=entry.get("hf_id"), family=entry.get("family"), key=key)
    cands += [p for p in MT.items_audit_candidates(ctx.paths["audit_dir"], hf_id=None, family=model, key=key) if p not in cands]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        return None, f"no item audit for {model} under {ctx.paths['audit_dir']} (tried {', '.join(p.name for p in cands)})"
    cache = ctx.cache.setdefault("items_audit", {})
    if path not in cache:
        cache[path] = MT.read_items_audit(path)
        ctx.inputs.append(_sha_record(path, "items_audit"))
    return cache[path], str(path)


def _item_file_of(ctx: Context, model: str, line: str) -> Path | None:
    """The item file a model's line ran on, when it is on this machine (aggregate.resolve_item_path's rules)."""
    for r in ctx.runs:
        if r.model == model and r.line == line and r.excerpt.get("item_file"):
            p = AGG.resolve_item_path({"item_file": {"path": r.excerpt["item_file"]}}) if AGG is not None else None
            if p is not None:
                return p
    return None


def _items_by_id(ctx: Context, path: Path) -> dict[str, dict]:
    cache = ctx.cache.setdefault("items", {})
    if path not in cache:
        items, _ = load_item_file(path)
        cache[path] = {it["item_id"]: it for it in items}
        ctx.inputs.append(_sha_record(path, "items"))
    return cache[path]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def min_families(ctx: Context, key: str, default: int) -> int:
    return int(((ctx.cfg.get("pooling") or {}).get("min_reliable_families") or {}).get(key, default))


def pct(x) -> str:
    return "--" if not _finite(x) else f"{100 * float(x):.1f}"


def pts(x) -> str:
    return "--" if not _finite(x) else f"{100 * float(x):+.1f}"


def pval(p) -> str:
    if not _finite(p):
        return "--"
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci_pct(est, lo, hi) -> str:
    return "--" if not _finite(est) else f"{pct(est)} [{pct(lo)}, {pct(hi)}]"


def ci_pts(est, lo, hi) -> str:
    return "--" if not _finite(est) else f"{pts(est)} [{pts(lo)}, {pts(hi)}]"


def jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, Path):
        return str(x)
    return x


# ============================================================================ artefacts
@dataclass
class Artefact:
    """One output table / figure-data file: headers and pre-formatted tex rows, the machine-readable rows, the
    payload of its .json sibling, and whether the analysis was computable (with the reason when not)."""
    name: str
    title: str
    headers: list[str]
    tex_rows: list[list[str]]
    rows: list[dict]
    computable: bool = True
    reason: str | None = None
    caption: str = ""
    extra: dict = field(default_factory=dict)
    tex_body: str | None = None        # a custom body (Table 3 through tables.to_latex_interventions)
    analysis: str | None = None        # the ANALYSES name this artefact decides, when it does

    def empty(self) -> bool:
        return not self.rows


def not_computable(name: str, title: str, reason: str, headers: Sequence[str] = ("cell",), analysis: str | None = None,
                   extra: dict | None = None) -> Artefact:
    return Artefact(name, title, list(headers), [], [], computable=False, reason=reason, extra=extra or {}, analysis=analysis)


def _tex(s) -> str:
    return tables._tex(s)


def _model_label(row: Mapping) -> str:
    return f"{row['model']} (cpu)" if row.get("unit") == "cpu" else str(row["model"])


def _groups(d: pd.DataFrame, by: Sequence[str]):
    if len(d) == 0:
        return []
    for keys, g in d.groupby(list(by), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yield dict(zip(by, keys, strict=True)), g


# ---------------------------------------------------------------------------- Table 2
def _rate(g: pd.DataFrame, key: str) -> float | None:
    if key not in g:
        return None
    s = g[key].dropna()
    return float(s.astype(bool).mean()) if len(s) else None


def _t3_extras(g: pd.DataFrame) -> dict:
    """T3 balanced accuracy / d' with the spelling twins AND their partners out (DD 5.3; score._spelling_pair_ids),
    plus the log-prob columns kept apart (t3_yesno_lp = forced choice; t3_pair_lp = pair) and never compared."""
    recs = g.to_dict("records")
    drop = _spelling_pair_ids(recs)
    kept = [r for r in recs if r["item_id"] not in drop]
    bs = balanced_stats(kept, pred_key="pred") if kept else {"balanced_accuracy": None, "d_prime": None}
    return {"balanced_accuracy_excl_spelling": bs.get("balanced_accuracy"), "d_prime_excl_spelling": bs.get("d_prime"),
            "n_items_excl_spelling": len({r["item_id"] for r in kept}), "n_spelling_pair_items": len(drop & {r["item_id"] for r in recs}),
            "n_pairs": len({frozenset((r["item_id"], r.get("pair_item_id"))) for r in recs if r.get("pair_item_id")}),
            "t3_yesno_lp": _rate(g, "forced_choice_correct"), "t3_pair_lp": _rate(g, "paired_correct"),
            "t3_pair_lp_norm": _rate(g, "paired_correct_norm")}


def primary_metric(ctx: Context, g: pd.DataFrame) -> dict | None:
    """constants.PRIMARY_METRIC: strict T1 accuracy averaged over the paraphrases present, pooled over V1-V4 with equal
    weight (the mean of the (variant, paraphrase) cell accuracies), with a base-pair cluster-bootstrap CI through a
    custom statistic (every cell re-weighted by the same draw)."""
    t1 = g[(g["task"] == "T1") & g["paraphrase"].notna()]
    if len(t1) == 0:
        return None
    cells = (t1["variant"].astype(str) + "|" + t1["paraphrase"].astype(str)).values
    _, idx = np.unique(cells, return_inverse=True)
    idx = idx.ravel()
    n_cells = int(idx.max()) + 1

    def stat(v: np.ndarray, w: np.ndarray) -> float:
        num = np.bincount(idx, weights=v * w, minlength=n_cells)
        den = np.bincount(idx, weights=w, minlength=n_cells)
        ok = den > 0
        return float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")

    v = t1["correct"].astype(float).values
    kw = {"n_boot": ctx.n_boot, "seed": ctx.seed, "level": ctx.level, "stat": stat}
    try:
        ci = bootstrap.cluster_bootstrap(v, t1[CLUSTER_KEY].values, strata=_strata(t1), **kw)
    except ValueError:
        ci = bootstrap.cluster_bootstrap(v, t1[CLUSTER_KEY].values, **kw)
    paras = sorted(t1["paraphrase"].unique())
    return {"acc_mean": ci.estimate, "lo": ci.lo, "hi": ci.hi, "ci_method": ci.method, "small_cell": bool(ci.small_cell),
            "n_cells": n_cells, "paraphrases": paras, "n_paraphrases": len(paras), "variants": sorted(t1["variant"].unique()),
            "n_items": int(t1["item_id"].nunique()), "n_base_pairs": int(ci.n_clusters),
            "label": constants.PRIMARY_METRIC + (" (fewer than three paraphrases present)" if len(paras) < 3 else "")}


def table2(ctx: Context) -> Artefact:
    """Table 2: strict accuracy per model x task x variant on the baseline arm, p0 with the mean and range over the
    paraphrases present (PREREG 8.13), the T3 headline without spelling twins, the log-prob columns apart, and the
    primary metric per model; API models on the core with the range over the 300-item paraphrase subset, labelled."""
    L = ctx.cfg.get("lines") or {}
    spec = L.get("table2") or {}
    lines = list(spec.get("lines") or ["E1_main"])
    arm = str(spec.get("arm") or ctx.cfg.get("baseline_arm") or "nfc")
    p_primary = str(spec.get("primary_paraphrase") or "p0")
    cond = str(ctx.cfg.get("default_condition") or "vi-*-s3-explained-raw")
    d = ctx.rows(lines, arm=arm, condition=cond)
    title = "Table 2: strict accuracy per model, task and variant (baseline arm)"
    if len(d) == 0:
        return not_computable("table2_main", title, f"no completed rows on {lines} (arm {arm}, condition {cond})",
                              ["Model", "Task", "Variant", f"{p_primary} % [95% CI]", "mean (range)", "items", "base pairs"],
                              analysis=ANALYSES[0])
    rows, tex_rows, metrics, cells_agg = [], [], {}, []
    for keys, g in _groups(d, ("model", "unit", "task", "variant")):
        api = bool(g["is_api"].fillna(False).astype(bool).any())
        per_p: dict[str, dict] = {}
        p_items: dict[str, set] = {}
        for p, gp in _groups(g, ("paraphrase",)):
            if p["paraphrase"] is None:
                continue
            per_p[str(p["paraphrase"])] = acc_cell(gp, ctx)
            p_items[str(p["paraphrase"])] = set(gp["item_id"])
        if not per_p:
            continue
        prim = per_p.get(p_primary)
        others = [p for p in per_p if p != p_primary]
        note = ""
        if api and others:            # the range on the paraphrase subset, p0 restricted to the same items (item 52)
            sub = set.union(*(p_items[p] for p in others))
            g0 = g[(g["paraphrase"] == p_primary) & g["item_id"].isin(sub)]
            vals = [acc_cell(g0, ctx)["estimate"]] if len(g0) else []
            vals += [per_p[p]["estimate"] for p in others]
            note = f"API: {p_primary} on the core; range over the {len(sub)}-item paraphrase subset ({p_primary} restricted to it)"
        else:
            vals = [c["estimate"] for c in per_p.values()]
        if not others:
            note = (note + "; " if note else "") + f"range: n/a ({p_primary} only)"
        if prim is None:
            note = (note + "; " if note else "") + f"{p_primary} absent (paraphrases present: {', '.join(sorted(per_p))})"
        row = {**keys, "model_label": _model_label(keys), "is_api": api, "line": "+".join(sorted(g["line"].unique())),
               "tags": "+".join(sorted(x for x in g["tag"].unique() if x)), "arm": arm, "condition": cond,
               "paraphrases": sorted(per_p), "n_paraphrases": len(per_p),
               "primary": {"paraphrase": p_primary, **prim} if prim else None,
               "mean_over_paraphrases": float(np.mean(vals)) if vals else None,
               "range_over_paraphrases": [float(min(vals)), float(max(vals))] if vals else None,
               "per_paraphrase": per_p, "note": note}
        if keys["task"] == "T3":
            row["t3"] = _t3_extras(g[g["paraphrase"] == p_primary] if prim else g)
        rows.append(row)
        rng = f"{pct(row['mean_over_paraphrases'])} ({pct(row['range_over_paraphrases'][0])}-{pct(row['range_over_paraphrases'][1])})" if vals else "--"
        est = ci_pct(prim["estimate"], prim["lo"], prim["hi"]) if prim else "--"
        flag = "$^{\\dagger}$" if prim and prim["small_cell"] else ""
        tex_rows.append([_tex(row["model_label"]), str(keys["task"]), str(keys["variant"] or "--"), est + flag, rng,
                         str(prim["n_items"] if prim else sum(c["n_items"] for c in per_p.values())),
                         str(prim["n_base_pairs"] if prim else "--")])
        cells_agg.append(row)
    for keys, g in _groups(d, ("model", "unit")):
        api = bool(g["is_api"].fillna(False).astype(bool).any())
        # an API model's p0 covers the whole core and its p1 / p2 the 300-item subset: the per-model metric is p0 on the
        # core, the range stays per cell (DD 5.1; 1- and 3-trial means are never mixed without saying so, item 52)
        pm = primary_metric(ctx, g[g["paraphrase"] == p_primary] if api else g)
        if pm:
            if api:
                pm["label"] = (f"API model: {p_primary} accuracy on the core (strict T1, V1-V4 equal weight); the paraphrase "
                               "range is reported per cell on the paraphrase subset, never mixed into this mean (DD 5.1, item 52)")
            metrics[_model_label(keys)] = pm
    cross = _aggregate_cross_check(ctx, d)
    agg_pm = (cross.get("primary_metric") or {}).get("acc_mean3") or {}
    agree = {m: (abs(metrics[m]["acc_mean"] - agg_pm[m]) < 1e-9) for m in metrics if agg_pm.get(m) is not None}
    extra = {"primary_metric_per_model": metrics, "primary_paraphrase": p_primary, "lines": lines, "arm": arm, "condition": cond,
             "primary_metric_agrees_with_aggregate_module": agree or None,
             "rule": "PREREG 8.13: p0 with the mean and range over {p0, p1, p2}; a claim about a model holds only if its "
                     "direction holds under all three paraphrases; every headline size carries its cluster count",
             "t3_headline_excludes_twin": constants.T3_HEADLINE_EXCLUDES_TWIN,
             "aggregate_cross_check": cross}
    headers = ["Model", "Task", "Variant", f"{p_primary} % [95% CI]", "mean (range) over paraphrases", "items", "base pairs"]
    cap = (f"Strict accuracy in percent on the baseline arm ({arm}), condition {cond}; {p_primary} with its base-pair "
           "cluster-bootstrap interval, then the mean and range over the paraphrases present; $^{\\dagger}$ small cell "
           f"(Wilson on n / DEFF). Estimation only: no pairwise model test (PREREG 8.3).")
    return Artefact("table2_main", title, headers, tex_rows, rows, caption=cap, extra=extra, analysis=ANALYSES[0])


def _aggregate_cross_check(ctx: Context, d: pd.DataFrame) -> dict:
    """noilai.stats.aggregate (task B2) as a cross-check when present: its paraphrase means and primary metric are
    written beside ours; the rendered table never depends on it."""
    if AGG is None:
        return {"module": None, "note": "noilai.stats.aggregate not importable: cross-check skipped"}
    out: dict = {"module": "noilai.stats.aggregate"}
    kw = {"n_boot": ctx.n_boot, "seed": ctx.seed, "level": ctx.level}
    for fn, kwargs in (("paraphrase_means", {}), ("primary_metric", kw)):
        f = getattr(AGG, fn, None)
        if f is None:
            out[fn] = {"note": "function absent"}
            continue
        try:
            res = f(d, **kwargs)
            rec = {"n_rows": len(res) if hasattr(res, "__len__") else None,
                   "columns": list(res.columns) if isinstance(res, pd.DataFrame) else None}
            if fn == "primary_metric" and isinstance(res, pd.DataFrame) and {"model", "acc_mean3"} <= set(res.columns):
                rec["acc_mean3"] = {str(r["model"]): (None if pd.isna(r["acc_mean3"]) else float(r["acc_mean3"])) for _, r in res.iterrows()}
            out[fn] = rec
        except Exception as e:  # a concurrent module's API may differ; the cross-check is optional (BLE001 is ignored repo-wide)
            out[fn] = {"note": f"not comparable: {type(e).__name__}: {e}"}
    return out


def table2_range(ctx: Context, t2: Artefact) -> Artefact:
    """Appendix: every per-paraphrase cell behind Table 2, the lenient T1 accuracy, the spelling-twin T3 column and
    the non-degenerate restriction (PREREG 8.13 range; appendix.tex extras)."""
    title = "Table 2 appendix: paraphrase range, lenient T1, spelling-twin T3 and non-degenerate cells"
    if not t2.computable:
        return not_computable("table2_paraphrase_range", title, t2.reason or "Table 2 not computable",
                              ["Model", "Task", "Variant", "paraphrase", "% [95% CI]", "items", "base pairs"])
    L = ctx.cfg.get("lines") or {}
    spec = L.get("table2") or {}
    d = ctx.rows(list(spec.get("lines") or ["E1_main"]), arm=str(spec.get("arm") or "nfc"),
                 condition=str(ctx.cfg.get("default_condition") or "vi-*-s3-explained-raw"))
    rows, tex_rows = [], []
    for r in t2.rows:
        for p, c in sorted(r["per_paraphrase"].items()):
            rows.append({"model": r["model"], "unit": r["unit"], "task": r["task"], "variant": r["variant"], "paraphrase": p, **c})
            tex_rows.append([_tex(r["model_label"]), r["task"], str(r["variant"] or "--"), p, ci_pct(c["estimate"], c["lo"], c["hi"]),
                             str(c["n_items"]), str(c["n_base_pairs"])])
    lenient, nondeg, spelling = [], [], []
    for keys, g in _groups(d[d["task"] == "T1"], ("model", "unit", "variant", "paraphrase")):
        if "correct_lenient" in g and g["correct_lenient"].notna().any():
            lenient.append({**keys, **acc_cell(g, ctx, value="correct_lenient")})
    if "degenerate" in d:
        nd = d[~d["degenerate"].fillna(False).astype(bool)]
        for keys, g in _groups(nd, ("model", "unit", "task", "variant", "paraphrase")):
            nondeg.append({**keys, **acc_cell(g, ctx)})
    for keys, g in _groups(d[d["task"] == "T3"], ("model", "unit", "variant", "paraphrase")):
        recs = g.to_dict("records")
        sp = _spelling_pair_ids(recs)
        sub = [r for r in recs if r["item_id"] in sp]
        if sub:
            bs = balanced_stats(sub, pred_key="pred")
            spelling.append({**keys, "n_items": len(sub), "accuracy": float(np.mean([bool(r["correct"]) for r in sub])), **bs})
    extra = {"lenient_t1": lenient, "non_degenerate": nondeg, "spelling_pairs_t3": spelling}
    cap = "Every per-paraphrase cell behind Table 2 (strict accuracy, base-pair cluster-bootstrap intervals)."
    return Artefact("table2_paraphrase_range", title, ["Model", "Task", "Variant", "paraphrase", "% [95% CI]", "items", "base pairs"],
                    tex_rows, rows, caption=cap, extra=extra)


# ---------------------------------------------------------------------------- Table 3
def _verdicts(ctx: Context, model: str) -> dict:
    g = ctx.df[ctx.df["model"] == model]
    nfd = next((x for x in g["census_nfd"].dropna().unique()), None) if len(g) else None
    pc = next((x for x in g["census_pc"].dropna().unique()), None) if len(g) else None
    return {"nfd": nfd, "win1258": pc}


def pass_through(ctx: Context, model: str) -> tuple[bool, str]:
    """Whether the model's tokenizer passes NFD through (H3's informative set): the census verdict when recorded,
    else the DD 7.2 expectation by tokenizer family (families.pass_through_expected), labelled."""
    v = _verdicts(ctx, model)["nfd"]
    if v is not None:
        return v == "passes_through", f"census verdict_nfd={v}"
    fam = ctx.family_map() or {}
    expected = set((ctx.cfg.get("families") or {}).get("pass_through_expected") or [])
    label = fam.get(model)
    return (label in expected), f"not censused: expectation for family {label!r} ({'pass-through' if label in expected else 'normalizing/unknown'})"


def table3_frame(ctx: Context) -> tuple[pd.DataFrame, dict]:
    """The rows behind Table 3: E3_noilai (main file), E3_xcopa with the summed log-prob outcome copied into `correct`
    (5.7; never compared with the generated outcome), E3_c2_enriched (c2 file), E3_api_core with its E1_api_core
    baseline rows on the same items."""
    L = (ctx.cfg.get("lines") or {}).get("table3") or {}
    main, c2 = str(L.get("main", "E3_noilai")), str(L.get("c2", "E3_c2_enriched"))
    xs = L.get("xcopa") or {}
    api = L.get("api") or {}
    notes: dict = {"dropped_xcopa_rows_without_logprob": 0}
    parts = [ctx.rows(main, paraphrase="p0"), ctx.rows(c2, paraphrase="p0")]
    x = ctx.rows(str(xs.get("line", "E3_xcopa")), paraphrase="p0")
    if len(x):
        outcome = str(xs.get("outcome", "logprob_correct"))
        x = x.copy()
        ok = x[outcome].notna() if outcome in x else pd.Series(False, index=x.index)
        notes["dropped_xcopa_rows_without_logprob"] = int((~ok).sum())
        x = x[ok]
        x["correct"] = x[outcome].astype(bool)
        x["outcome"] = outcome
        parts.append(x)
    a = ctx.rows(str(api.get("line", "E3_api_core")), paraphrase="p0")
    if len(a):
        base = ctx.rows(str(api.get("baseline_line", "E1_api_core")), paraphrase="p0", arm=str(L.get("baseline_arm", "nfc")))
        base = base[base["item_id"].isin(set(a["item_id"]))]
        parts += [a, base]
    parts = [p for p in parts if len(p)]
    if not parts:
        return ctx.df.iloc[0:0], notes
    f = pd.concat(parts, ignore_index=True)
    if "outcome" not in f:
        f["outcome"] = "correct"
    f["outcome"] = f["outcome"].fillna("correct")
    return f, notes


def table3(ctx: Context) -> tuple[Artefact, Artefact]:
    """Table 3 (DD 8.3, item 65): paired effect of every arm vs the baseline, ONE Holm family per model row from
    constants.HOLM_FAMILY_TABLE3 FILTERED PER MODEL before tables.intervention_table: nfd / win1258 cells of a
    normalizing tokenizer are `0 by construction` rows (effect 0, no test), corrupting arms are refused, API models
    keep only nfd / placement_new on T1 / T3; the primary p (paired t from 200 base pairs, else the bootstrap p) is
    Holm-corrected within the row, McNemar in the appendix; the realized family size sits in every row and the footer."""
    title = "Table 3: re-encoding interventions vs the NFC baseline, Holm within each model row"
    headers = ["Model", "Census", "Task", "Arm", "File", "effect pts [95% CI]", "p (Holm)", "sig.", "items", "base pairs", "family"]
    f, notes = table3_frame(ctx)
    if len(f) == 0:
        a = not_computable("table3_interventions", title, "no completed E3 rows", headers, extra={"notes": notes})
        b = not_computable("table3_mcnemar_appendix", "Table 3 appendix: McNemar (ignores clustering)", "no completed E3 rows", headers)
        return a, b
    base_arm = str(((ctx.cfg.get("lines") or {}).get("table3") or {}).get("baseline_arm", "nfc"))
    family_all = tables.table3_family()
    rows, mc_rows, tabs, zero_rows, refused, normalizes, cross = [], [], [], [], [], {}, True
    for (model, unit), g in f.groupby(["model", "unit"], sort=True):
        v = _verdicts(ctx, model)
        api = bool(g["is_api"].fillna(False).astype(bool).any())
        fam = []
        for arm, task, file in family_all:
            verdict = v.get(arm)
            if arm in CENSUS_ARM_VERDICTS and verdict == "normalizes":
                zero_rows.append({"model": model, "unit": unit, "task": task, "arm": arm, "item_file": file, "effect": 0.0,
                                  "census": verdict, "note": "0 by construction (census: normalizes; PREREG 8.6, not a test)"})
                continue
            if arm in CENSUS_ARM_VERDICTS and verdict == "corrupts":
                refused.append({"model": model, "unit": unit, "task": task, "arm": arm, "item_file": file, "census": verdict,
                                "note": "refused: the census says the engine corrupts this encoding (PREREG 8.6)"})
                continue
            if api and (task == "XCOPA" or arm in ("strip_tones", "strip_all")):
                continue           # API lines carry only nfd / placement_new on T1 / T3 (run_plan E3_api_core)
            fam.append((arm, task, file))
        normalizes[model] = v.get("nfd") == "normalizes"
        if not fam:
            continue
        tab = tables.intervention_table(g, baseline_arm=base_arm, family=fam, n_boot=ctx.paired_boot, use="primary")
        if len(tab) == 0:
            continue
        mc = tables.intervention_table(g, baseline_arm=base_arm, family=fam, n_boot=ctx.paired_boot, use="mcnemar")
        tab["family_defined"] = constants.HOLM_FAMILY_TABLE3_CELLS
        tab["unit"] = unit
        tab["census_nfd"], tab["census_pc"] = v.get("nfd"), v.get("win1258")
        tab["outcome"] = [("logprob_correct" if t == "XCOPA" else "correct") for t in tab["task"]]
        _, p_adj = holm_adjust(tab["p_primary"].astype(float).values)       # the statsmodels cross-check of tests.holm
        cross = cross and bool(np.allclose(p_adj, tab["p_holm"].astype(float).values, atol=1e-12))
        tabs.append(tab)
        rows.extend(tab.to_dict("records"))
        for r in mc.to_dict("records"):
            mc_rows.append({**r, "unit": unit, "label": "McNemar mid-p (secondary; ignores clustering; p0 item level)"})
    def _tex_rows(rs: list[dict]) -> list[list[str]]:
        out = []
        for r in rs:
            star = "$^{*}$" if r.get("significant") else ""
            dag = "$^{\\dagger}$" if r.get("small_cell") else ""
            cen = f"nfd:{r.get('census_nfd') or '?'}/pc:{r.get('census_pc') or '?'}"
            out.append([_tex(_model_label(r)), _tex(cen), str(r["task"]), _tex(r["arm"]), str(r.get("item_file") or "--"),
                        ci_pts(r["effect"], r["ci_lo"], r["ci_hi"]) + dag, pval(r.get("p_holm")) + star, "yes" if r.get("significant") else "no",
                        str(r["n"]), str(r["n_base_pairs"]), f"{r['family_size']} of {constants.HOLM_FAMILY_TABLE3_CELLS}"])
        return out
    tex_rows = _tex_rows(rows)
    for z in zero_rows:
        tex_rows.append([_tex(_model_label(z)), _tex(z["census"]), z["task"], _tex(z["arm"]), z["item_file"], "0 (by construction)",
                         "--", "--", "--", "--", "--"])
    body = None
    if tabs:
        full = pd.concat(tabs, ignore_index=True)
        cap = (f"Table 3 ({ctx.label}). Realized family sizes per model row out of {constants.HOLM_FAMILY_TABLE3_CELLS} cells; "
               "cells `0 by construction` and refused cells are listed in the .json sibling.")
        body = tables.to_latex_interventions(full, cap, "tab:arms", normalizes=normalizes or None)
        body += "\n" + "\n".join(f"% 0 by construction: {z['model']} {z['arm']} {z['task']} ({z['item_file']}; census {z['census']})" for z in zero_rows)
        body += "\n" + "\n".join(f"% refused: {z['model']} {z['arm']} {z['task']} (census {z['census']})" for z in refused)
    extra = {"zero_by_construction": zero_rows, "refused": refused, "normalizes": normalizes, "notes": notes,
             "holm_cross_check_multipletests_equals_tests_holm": cross, "family_defined": constants.HOLM_FAMILY_TABLE3_CELLS,
             "family": [list(c) for c in constants.HOLM_FAMILY_TABLE3], "baseline_arm": base_arm,
             "realized_family_size": {f"{r['model']}": r["family_size"] for r in rows},
             "note_c2_T3": "the (placement_new, T3, c2) cell is never realized: E3_c2_enriched is T1-only (handoff item 5)"}
    cap = ("Effects in points vs the NFC baseline on the same items (base-pair cluster-bootstrap intervals); $^{*}$ Holm-adjusted "
           f"$p<{constants.ALPHA}$ within the model row over its defined cells; $^{{\\dagger}}$ small cell; XCOPA outcome = summed "
           "log-probability of the correct alternative (never compared with the generated outcome).")
    a = Artefact("table3_interventions", title, headers, tex_rows, rows, caption=cap, extra=extra, tex_body=body)
    b = Artefact("table3_mcnemar_appendix", "Table 3 appendix: McNemar mid-p within each model row (ignores clustering)", headers,
                 _tex_rows(mc_rows), mc_rows, caption="Secondary p0 item-level statistic (PREREG 8.2); labelled, never primary.")
    return a, b


# ---------------------------------------------------------------------------- H3, H4
def table_h3(ctx: Context) -> Artefact:
    """H3 (PREREG 8.5, :25): tau(NFD, T1-V3) from the E3_noilai V3 rows and tau(NFD, XCOPA) from E3_xcopa, pass-through
    models only, a Holm family of two per model; the per-family table is primary and each contrast is pooled across
    tokenizer families (direction +1 for T1-V3, -1 for XCOPA); below five informative families the pooled CI is
    labelled unreliable and H3 is decided by the per-family CIs."""
    title = "H3: NFD on T1-V3 and on XCOPA, pass-through models, family of two per model, pooled across tokenizer families"
    headers = ["Model", "Family", "Contrast", "effect pts [95% CI]", "p", "p (Holm, 2)", "sig.", "items", "base pairs", "pass-through"]
    L = ctx.cfg.get("lines") or {}
    t3 = L.get("table3") or {}
    h3 = L.get("h3") or {}
    main = ctx.rows(str(t3.get("main", "E3_noilai")), paraphrase="p0", task="T1")
    main = main[main["variant"] == str(h3.get("t1_variant", "V3"))]
    x = ctx.rows(str((t3.get("xcopa") or {}).get("line", "E3_xcopa")), paraphrase="p0")
    if len(main) == 0 and len(x) == 0:
        return not_computable("table_h3", title, "no completed E3_noilai V3 or E3_xcopa rows", headers, analysis=ANALYSES[3])
    fam_map = ctx.family_map() or {}
    base_arm = str(t3.get("baseline_arm", "nfc"))
    rows, tex_rows, recs = [], [], {"T1-V3": [], "XCOPA": []}
    models = sorted(set(main["model"]) | set(x["model"]))
    for model in models:
        ok, why = pass_through(ctx, model)
        if not ok and bool(h3.get("pass_through_only", True)):
            rows.append({"model": model, "contrast": None, "pass_through": False, "pass_through_source": why,
                         "note": "not informative for H3 (normalizing tokenizer: NFD = NFC by construction)"})
            continue
        tests_, metas = [], []
        for name, d in (("T1-V3", main[main["model"] == model]), ("XCOPA", x[x["model"] == model])):
            if len(d) == 0:
                continue
            value = "logprob_correct" if name == "XCOPA" else "correct"
            d = d[d[value].notna()] if value in d else d.iloc[0:0]
            pc = paired_contrast(d[d["arm"] == base_arm], d[d["arm"] == "nfd"], ctx, name, value_a=value, value_b=value)
            if pc is None:
                continue
            tests_.append(pc[0])
            metas.append((name, pc[1]))
        apply_holm(tests_)
        for t, (name, extra) in zip(tests_, metas, strict=True):
            r = {"model": model, "family": fam_map.get(model), "contrast": name, "outcome": "logprob_correct" if name == "XCOPA" else "correct",
                 "pass_through": True, "pass_through_source": why, "family_size": len(tests_), **test_row(t, extra)}
            rows.append(r)
            recs[name].append({"model": model, "estimate": t.diff, "se": extra["se"]})
            tex_rows.append([_tex(model), _tex(r["family"] or "?"), name, ci_pts(t.diff, t.ci_lo, t.ci_hi), pval(t.p_primary), pval(t.p_adj),
                             "yes" if t.significant else "no", str(extra["n"]), str(extra["n_base_pairs"]), "yes"])
    mf = min_families(ctx, "H3", 5)
    pooled_t1 = pooled(ctx, recs["T1-V3"], direction=+1, min_families=mf, label="H3 tau(NFD, T1-V3)")
    pooled_x = pooled(ctx, recs["XCOPA"], direction=-1, min_families=mf, label="H3 tau(NFD, XCOPA)")
    def _side(p: dict) -> bool | None:
        return p.get("on_predicted_side") if p.get("computable") else None
    decision = {"rule": "both pooled estimates on the predicted side with CIs excluding 0 (T1-V3 > 0, XCOPA < 0); with fewer than "
                        f"{mf} informative families the pooled CI is unreliable and H3 is decided by the per-family CIs (PREREG:25)",
                "t1v3_on_side": _side(pooled_t1), "xcopa_on_side": _side(pooled_x),
                "per_family_decision": {"T1-V3": pooled_t1.get("all_informative_families_on_side"),
                                        "XCOPA": pooled_x.get("all_informative_families_on_side")},
                "unreliable": {"T1-V3": pooled_t1.get("unreliable"), "XCOPA": pooled_x.get("unreliable")}}
    extra = {"pooled_t1_v3": pooled_t1, "pooled_xcopa": pooled_x, "decision": decision, "phrase": "across the panel",
             "k_of_n": {"T1-V3": pooled_t1.get("k_of_n"), "XCOPA": pooled_x.get("k_of_n")}}
    cap = ("tau_m(NFD) = E_i[Y_i(NFD) - Y_i(NFC)] per pass-through model on T1-V3 (strict) and XCOPA (summed log-prob), Holm over "
           "the two contrasts of a model; the pooled random-effects estimates (REML + Hartung-Knapp, k of N tokenizer "
           "families) are in the .json sibling.")
    return Artefact("table_h3", title, headers, tex_rows, rows, caption=cap, extra=extra, analysis=ANALYSES[3])


def _c2_affected_rows(d: pd.DataFrame) -> pd.DataFrame:
    """The C2-affected rows (H4 'on C2-affected items') when the scores carry the flag; every row otherwise."""
    if "c2_affected" in d and d["c2_affected"].fillna(False).astype(bool).any():
        return d[d["c2_affected"].fillna(False).astype(bool)]
    return d


def table_h4(ctx: Context) -> Artefact:
    """H4 (PREREG :27): placement_new vs nfc on the C2-affected rows (E3_c2_enriched, T1, p0; API: E3_api_core vs
    E1_api_core), one paired test per model, Holm ACROSS the panel, k of N models (Holm-significant and on the
    predicted side), the pooled estimate, and TOST (delta = 2 points) for models whose CI includes 0. The expected
    sign is read from constants.BASELINE_PLACEMENT (`old` -> the new placement is expected to cost accuracy)."""
    title = "H4: placement_new vs nfc on the C2-affected items, Holm across the panel, k of N, pooled, TOST"
    headers = ["Model", "Family", "Line", "effect pts [95% CI]", "p", "p (Holm, panel)", "sig.", "items", "base pairs", "TOST"]
    L = ctx.cfg.get("lines") or {}
    h4 = L.get("h4") or {}
    arm = str(h4.get("arm", "placement_new"))
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    direction = -1 if constants.BASELINE_PLACEMENT == "old" else +1
    c2 = _c2_affected_rows(ctx.rows(str(h4.get("line", "E3_c2_enriched")), paraphrase="p0", task="T1"))
    # the API C2 arm runs on the whole 296-item core (run_plan E3_api_core): H4 is "on C2-affected items", so the
    # unaffected rows (identical text under both placements) leave before pairing, as on the C2-enriched file
    api = _c2_affected_rows(ctx.rows(str(h4.get("api_line", "E3_api_core")), paraphrase="p0", task="T1"))
    api_base = ctx.rows(str(h4.get("api_baseline_line", "E1_api_core")), paraphrase="p0", arm=base_arm, task="T1")
    if len(c2) == 0 and len(api) == 0:
        return not_computable("table_h4", title, "no completed E3_c2_enriched (or E3_api_core) rows", headers, analysis=ANALYSES[5])
    fam_map = ctx.family_map() or {}
    tests_, metas = [], []
    for model, g in c2.groupby("model", sort=True):
        pc = paired_contrast(g[g["arm"] == base_arm], g[g["arm"] == arm], ctx, model)
        if pc:
            tests_.append(pc[0])
            metas.append((model, str(h4.get("line", "E3_c2_enriched")), pc[1]))
    for model, g in api.groupby("model", sort=True):
        pc = paired_contrast(api_base[api_base["model"] == model], g[g["arm"] == arm], ctx, model)
        if pc:
            tests_.append(pc[0])
            metas.append((model, str(h4.get("api_line", "E3_api_core")), pc[1]))
    if not tests_:
        return not_computable("table_h4", title, f"no model has paired {arm} vs {base_arm} rows", headers, analysis=ANALYSES[5])
    apply_holm(tests_)
    rows, tex_rows, recs, tosts = [], [], [], {}
    for t, (model, line, extra) in zip(tests_, metas, strict=True):
        tost = None
        if PL is not None and PL.tost_applies(t):
            a, b, cl = extra["arrays"]
            tost = PL.tost_paired(a, b, cl)
            tosts[model] = tost
        r = {"model": model, "family": fam_map.get(model), "line": line, "arm": arm, **test_row(t, extra),
             "tost": tost, "expected_direction": direction}
        rows.append(r)
        recs.append({"model": model, "estimate": t.diff, "se": extra["se"]})
        ts = "--" if tost is None else ("equivalent" if tost.get("equivalent") else f"p={pval(tost.get('p_tost'))}")
        tex_rows.append([_tex(model), _tex(r["family"] or "?"), line, ci_pts(t.diff, t.ci_lo, t.ci_hi), pval(t.p_primary), pval(t.p_adj),
                         "yes" if t.significant else "no", str(extra["n"]), str(extra["n_base_pairs"]), ts])
    kofn = PL.k_of_n_models(tests_, direction) if PL is not None else {"note": "pooling module absent"}
    pool = pooled(ctx, recs, direction=direction, min_families=min_families(ctx, "H4", 3), label="H4 tau(placement_new)")
    falsified = None
    if pool.get("computable"):
        falsified = not (pool.get("on_predicted_side") and (pool["lo"] > 0 or pool["hi"] < 0))
    extra = {"k_of_n_models": kofn, "pooled": pool, "tost": tosts, "expected_direction": direction,
             "c2_affected_only": True,
             "direction_source": f"constants.BASELINE_PLACEMENT == {constants.BASELINE_PLACEMENT!r}",
             "tost_margin_points": constants.TOST_EQUIVALENCE_MARGIN_POINTS,
             "decision": {"falsified": falsified, "rule": "falsified when the pooled CI includes 0 or lies on the wrong side; "
                                                           "reported as k of N models, never as 'in every model'"},
             "c2_on_xcopa": "never tested (PREREG)"}
    cap = ("tau_m(placement_new) on the C2-affected items (T1, p0); Holm across the panel's models; TOST with delta = "
           f"{constants.TOST_EQUIVALENCE_MARGIN_POINTS} points for models whose unadjusted CI includes 0; k of N and the pooled "
           "estimate in the .json sibling.")
    return Artefact("table_h4", title, headers, tex_rows, rows, caption=cap, extra=extra, analysis=ANALYSES[5])


# ---------------------------------------------------------------------------- contrasts against the E1 baseline rows
def _baseline_core_rows(ctx: Context, model: str, paraphrase: str = "p0") -> pd.DataFrame:
    """The E1_main p0 nfc rows restricted to the core (in_core), or the E1_api_core rows for an API model."""
    L = ctx.cfg.get("lines") or {}
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    cond = str(ctx.cfg.get("default_condition") or "vi-*-s3-explained-raw")
    main = ctx.rows(str((L.get("explicit_input") or {}).get("baseline_line", "E1_main")), arm=base_arm, paraphrase=paraphrase, condition=cond)
    main = main[main["model"] == model]
    if "in_core" in main:
        main = main[main["in_core"].fillna(False).astype(bool)]
    if len(main) == 0:
        api = ctx.rows("E1_api_core", arm=base_arm, paraphrase=paraphrase, condition=cond)
        main = api[api["model"] == model]
    return main


def contrast_table(ctx: Context, name: str, title: str, lines: Sequence[str], condition: str | None, label: str,
                   analysis: str | None = None) -> Artefact:
    """Per model: a line's p0 nfc rows vs the E1 baseline rows on the same core items, paired by item_id; CIs and the
    paired t / bootstrap p, NO Holm (PREREG 8.3: explicit input and ablations are estimation); pooled across families."""
    headers = ["Model", "Line", "gap pts [95% CI]", "p (unadjusted)", "items", "base pairs"]
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    rows, tex_rows, recs = [], [], defaultdict(list)
    found = False
    for line in lines:
        d = ctx.rows(line, arm=base_arm, paraphrase="p0", condition=condition)
        if len(d) == 0:
            continue
        found = True
        for model, g in d.groupby("model", sort=True):
            base = _baseline_core_rows(ctx, model)
            pc = paired_contrast(base, g, ctx, f"{line}:{model}", keys=("item_id",))
            if pc is None:
                rows.append({"model": model, "line": line, "note": "no E1 baseline rows on the same items"})
                continue
            t, extra = pc
            r = {"model": model, "line": line, **test_row(t, extra), "label": label}
            r.pop("p_holm"), r.pop("significant")
            rows.append(r)
            recs[line].append({"model": model, "estimate": t.diff, "se": extra["se"]})
            tex_rows.append([_tex(model), _tex(line), ci_pts(t.diff, t.ci_lo, t.ci_hi), pval(t.p_primary), str(extra["n"]), str(extra["n_base_pairs"])])
    if not found:
        return not_computable(name, title, f"no completed rows on {list(lines)}", headers, analysis=analysis)
    pools = {line: pooled(ctx, rs, direction=None, min_families=3, label=f"{label} {line}") for line, rs in recs.items()}
    cap = (f"{label}: mean over the core items of Y_i(line) - Y_i(E1 baseline, p0, {base_arm}) per model, paired by item; base-pair "
           "cluster-bootstrap intervals; estimation only (no Holm, PREREG 8.3).")
    return Artefact(name, title, headers, tex_rows, rows, caption=cap, extra={"pooled": pools, "label": label}, analysis=analysis)


def table_explicit_input(ctx: Context) -> Artefact:
    spec = (ctx.cfg.get("lines") or {}).get("explicit_input") or {}
    return contrast_table(ctx, "table_explicit_input", "Explicit-input gap (DD 12.17): components vs raw input on the core items",
                          [str(spec.get("line", "E1_explicit_input"))], str(spec.get("condition", "vi-*-s3-explained-components")),
                          "derived estimand: explicit-input gap", analysis=ANALYSES[1])


def table_ablations(ctx: Context) -> Artefact:
    lines = list((ctx.cfg.get("lines") or {}).get("ablations") or [])
    return contrast_table(ctx, "table_ablations", "Ablations vs the E1 baseline on the core items (estimation only)", lines, None,
                          "ablation")


# ---------------------------------------------------------------------------- human, census, runs, errors
def _read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def table_human(ctx: Context) -> Artefact:
    """Human baseline (PREREG 8.11): the report's aggregates (mean human per cell) beside every model's E1_main p0 rows
    restricted to the human items; the mean-human band is a reference, no cell-level test."""
    title = "Human baseline: mean-human accuracy per cell and every model on the same items"
    headers = ["Model / human", "Task", "Variant", "% [95% CI]", "items", "base pairs"]
    rep_path, items_path = ctx.paths["human_report"], ctx.paths["human_items"]
    rep = _read_json(rep_path) if rep_path.exists() else None
    if not isinstance(rep, Mapping):
        return not_computable("table_human", title, f"no human baseline report at {rep_path} (make ingest-sheets)", headers,
                              analysis=ANALYSES[8])
    forms = rep.get("forms_included", rep.get("n_forms", rep.get("n_respondents")))
    ids_raw = _read_json(items_path) if items_path.exists() else None
    ids = set(ids_raw) if isinstance(ids_raw, list) else set((ids_raw or {}).get("items", (ids_raw or {}).get("item_ids", []))) \
        if isinstance(ids_raw, Mapping) else set()
    ctx.inputs.append(_sha_record(rep_path, "human_report"))
    rows, tex_rows = [], []
    acc = rep.get("accuracy") or {}
    for cell, c in sorted((acc.get("by_task_variant") or {}).items()):
        mh = ((c or {}).get("strict") or {}).get("mean_human") or {}
        task, _, variant = cell.partition("-")
        rows.append({"who": "mean human", "task": task, "variant": variant, **{k: mh.get(k) for k in ("estimate", "lo", "hi")},
                     "n_items": (c or {}).get("n_items"), "forms_included": forms})
        tex_rows.append(["mean human", task, variant, ci_pct(mh.get("estimate"), mh.get("lo"), mh.get("hi")), str((c or {}).get("n_items") or "--"), "--"])
    if ids:
        cond = str(ctx.cfg.get("default_condition") or "vi-*-s3-explained-raw")
        d = ctx.rows("E1_main", arm=str(ctx.cfg.get("baseline_arm") or "nfc"), paraphrase="p0", condition=cond)
        d = d[d["item_id"].isin(ids)]
        for keys, g in _groups(d, ("model", "unit", "task", "variant")):
            c = acc_cell(g, ctx)
            rows.append({"who": _model_label(keys), **keys, **c})
            tex_rows.append([_tex(_model_label(keys)), keys["task"], str(keys["variant"]), ci_pct(c["estimate"], c["lo"], c["hi"]), str(c["n_items"]), str(c["n_base_pairs"])])
    extra = {"forms_included": forms, "human_items": len(ids), "by_output_lexicality": acc.get("by_output_lexicality"),
             "by_region": rep.get("by_region"), "agreement": rep.get("agreement"),
             "omit_row_until": "forms_included >= 20 (HUMAN_BASELINE_PROTOCOL)", "row_eligible": bool(forms is not None and forms >= 20)}
    cap = "Mean-human accuracy per cell (reference band, no cell-level test) and every model's p0 accuracy on the same items."
    return Artefact("table_human", title, headers, tex_rows, rows, caption=cap, extra=extra, analysis=ANALYSES[8])


def table_census(ctx: Context) -> Artefact:
    """Every analysed model's census verdicts (manifest, else data/audit/<family>.json) and its 0-by-construction cells."""
    title = "Tokenizer census per analysed model (NFD / PC verdicts) and the cells that are 0 by construction"
    headers = ["Model", "verdict NFD", "verdict PC", "source", "engine", "0 by construction"]
    if not ctx.runs:
        return not_computable("table_census", title, "no completed run", headers)
    rows, tex_rows = [], []
    seen = set()
    for r in sorted(ctx.runs, key=lambda x: (x.model, x.unit)):
        if (r.model, r.unit) in seen:
            continue
        seen.add((r.model, r.unit))
        c = r.census
        zero = [f"{a}:{CENSUS_ARM_VERDICTS[a]}" for a in CENSUS_ARM_VERDICTS if c.get(CENSUS_ARM_VERDICTS[a]) == "normalizes"]
        rows.append({"model": r.model, "unit": r.unit, **c, "zero_by_construction_arms": [a.split(":")[0] for a in zero],
                     "models_yaml_family": (ctx.midx.get(r.model) or {}).get("family")})
        tex_rows.append([_tex(_model_label({"model": r.model, "unit": r.unit})), _tex(c.get("verdict_nfd") or "--"), _tex(c.get("verdict_pc") or "--"),
                         _tex(c.get("verdict_source") or "--"), _tex(c.get("engine") or "--"), _tex(", ".join(a.split(":")[0] for a in zero) or "none")])
    cap = "Census verdicts per model (PREREG 8.6); `normalizes` makes the nfd / pc cells 0 by construction, never a test."
    return Artefact("table_census", title, headers, tex_rows, rows, caption=cap)


def table_runs(ctx: Context) -> Artefact:
    """Manifest excerpts of every analysed run (identity, engine, hardware, hours, hashes) and the models that failed
    smoke per the ledger."""
    title = "Analysed runs: manifest excerpts, compute and result hashes"
    headers = ["Run", "Line", "Model", "Engine", "dtype", "Accel.", "wall s", "GPU h", "TPU h", "item sha", "results sha"]
    rows, tex_rows = [], []
    for r in ctx.runs:
        e = r.excerpt
        rows.append({**r.as_dict()})
        tex_rows.append([_tex(r.run_id), _tex(r.line), _tex(r.model), _tex(e.get("engine") or "--"), _tex(e.get("dtype_resolved") or e.get("dtype_requested") or "--"),
                         _tex(e.get("accelerator_kind") or "--"), f"{e.get('wall_s') or 0:.0f}", f"{e.get('gpu_hours') or 0:.2f}",
                         f"{e.get('tpu_hours') or 0:.2f}", (e.get("item_file_sha256") or "--")[:SHA_SHORT], (e.get("results_sha256") or "--")[:SHA_SHORT]])
    failed = []
    if ctx.ledger_rows is not None:
        failed = [{"unit_id": x.get("unit_id"), "model": x.get("model"), "status": x.get("status"), "notes": x.get("notes")}
                  for x in ctx.ledger_rows if x.get("status") in ("failed", "refused")]
    totals = {"wall_s": float(sum((r.excerpt.get("wall_s") or 0) for r in ctx.runs)),
              "gpu_hours": float(sum((r.excerpt.get("gpu_hours") or 0) for r in ctx.runs)),
              "tpu_hours": float(sum((r.excerpt.get("tpu_hours") or 0) for r in ctx.runs)), "runs": len(ctx.runs)}
    art = Artefact("table_runs", title, headers, tex_rows, rows, caption="One row per analysed run directory; totals and the ledger's failed / refused units in the .json sibling.",
                   extra={"totals": totals, "failed_or_refused": failed})
    if not rows:
        art.computable, art.reason = False, "no completed run"
    return art


def table_errors(ctx: Context) -> Artefact:
    """Error-class breakdown (Fig. 3): tables.error_breakdown by model, task and variant on the E1 p0 baseline rows."""
    title = "Error classes per model, task and variant (shares of rows; Fig. 3 data)"
    headers = ["Model", "Task", "Variant", "error class", "share %", "rows"]
    cond = str(ctx.cfg.get("default_condition") or "vi-*-s3-explained-raw")
    d = ctx.rows(["E1_main", "E1_api_core"], arm=str(ctx.cfg.get("baseline_arm") or "nfc"), paraphrase="p0", condition=cond)
    if len(d) == 0:
        return not_computable("table_errors", title, "no completed E1 p0 rows", headers)
    tab = tables.error_breakdown(d, by=("model", "task", "variant"))
    counts = d.groupby(["model", "task", "variant"]).size()
    rows, tex_rows, by_group = [], [], {}
    for idx, r in tab.iterrows():
        model, task, variant = idx
        n = int(counts.loc[idx])
        for cls, share in r.items():
            if share > 0:
                rows.append({"model": model, "task": task, "variant": variant, "error_class": cls, "share": float(share), "n_rows": n})
                tex_rows.append([_tex(model), task, str(variant), _tex(cls), pct(share), str(n)])
    groups = {m: (ctx.midx.get(m) or {}).get("group") for m in d["model"].unique()}
    d2 = d.assign(group=d["model"].map(groups).fillna("unknown"))
    for idx, r in tables.error_breakdown(d2, by=("group", "task")).iterrows():
        by_group[f"{idx[0]}|{idx[1]}"] = {k: float(v) for k, v in r.items() if v > 0}
    return Artefact("table_errors", title, headers, tex_rows, rows, caption="Shares of score rows per error class (noilai.eval.score ERROR_CLASSES).",
                    extra={"by_models_yaml_group": by_group})


# ---------------------------------------------------------------------------- dose-response, H6, H1 / E2, probes, appendix
def fig_deltatok(ctx: Context) -> Artefact:
    """Fig. 4 data: dose_response.decompose per model x arm on the E3_noilai p0 rows (delta tokens from the arm rows'
    delta_prompt_tokens); associational, no 'share mediated' (DD 8.6 items 2-3); the zero-dose contrast where the
    stratum holds >= 100 items (PREREG 8.7)."""
    title = "Delta-token dose-response per model and arm (associational; Fig. 4 data)"
    headers = ["Model", "Task", "Arm", "total pts [95% CI]", "form-only (dtok = 0) pts [95% CI]", "slope / token", "n dtok=0", "n dtok>0"]
    main = str(((ctx.cfg.get("lines") or {}).get("table3") or {}).get("main", "E3_noilai"))
    d = ctx.rows(main, paraphrase="p0")
    if len(d) == 0:
        return not_computable("fig_deltatok", title, f"no completed {main} rows", headers)
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    rows, tex_rows = [], []
    for keys, g in _groups(d, ("model", "unit", "task", "arm")):
        if keys["arm"] == base_arm:
            continue
        base = d[(d["model"] == keys["model"]) & (d["unit"] == keys["unit"]) & (d["task"] == keys["task"]) & (d["arm"] == base_arm)]
        al = align_pairs(base, g)
        if al is None:
            continue
        a, b, cl, _ = al
        o_ = g.drop_duplicates(["item_id", "prompt_id"]).set_index(["item_id", "prompt_id"])
        b_ = base.drop_duplicates(["item_id", "prompt_id"]).set_index(["item_id", "prompt_id"])
        common = b_.index.intersection(o_.index)
        dtok = pd.to_numeric(o_.loc[common, "delta_prompt_tokens"], errors="coerce").fillna(0.0).values
        if len(set(cl)) < 2:
            continue
        dec = dose_response.decompose(a, b, dtok, cl, n_boot=ctx.n_boot, seed=ctx.seed, level=ctx.level)
        zero_ok = dec.n_delta_zero >= 100
        r = {**keys, "total_effect": dec.total_effect.as_dict(), "form_only_effect": dec.form_only_effect.as_dict(),
             "slope_per_token": dec.slope_per_token.as_dict(), "token_associated_part": dec.token_associated_part.as_dict(),
             "effect_delta_zero": dec.effect_delta_zero.as_dict() if (dec.effect_delta_zero and zero_ok) else None,
             "effect_delta_pos": dec.effect_delta_pos.as_dict() if dec.effect_delta_pos else None,
             "n_delta_zero": dec.n_delta_zero, "n_delta_pos": dec.n_delta_pos, "label": dec.label,
             "zero_dose_reported": zero_ok, "mean_delta_tokens": float(np.mean(dtok)) if len(dtok) else None}
        rows.append(r)
        te, fo, sl = dec.total_effect, dec.form_only_effect, dec.slope_per_token
        tex_rows.append([_tex(_model_label(keys)), keys["task"], _tex(keys["arm"]), ci_pts(te.estimate, te.lo, te.hi), ci_pts(fo.estimate, fo.lo, fo.hi),
                         f"{100 * sl.estimate:+.3f}" if _finite(sl.estimate) else "--", str(dec.n_delta_zero), str(dec.n_delta_pos)])
    if not rows:
        return not_computable("fig_deltatok", title, "no arm pairs with two or more base pairs", headers)
    return Artefact("fig_deltatok", title, headers, tex_rows, rows,
                    caption="Associational decomposition of each arm's effect along delta prompt tokens (no share mediated is estimated).")


def _attested_covariates(items: Mapping[str, dict], freqs: Mapping[str, int]) -> tuple[dict[str, dict], dict]:
    """The PREREG 8.8 covariates of every exact, H6-eligible two-syllable attested item (from the attested release file:
    the parsed input syllables, the rule output, the Viet74K syllable frequencies), plus the floor counts."""
    from ..gen.generate import spelling_triggers, syl_from_dict
    from ..vi.syllable import spell, try_parse
    cov: dict[str, dict] = {}
    n_exact = n_elig = n_verified = 0
    collections: set[str] = set()
    for iid, it in items.items():
        if not it.get("exact"):
            continue
        n_exact += 1
        if not it.get("eligible_h6") or it.get("n_syllables") != 2 or not it.get("input_syllables"):
            continue
        n_elig += 1
        if it.get("verified_by"):
            n_verified += 1
            collections.add(str(it.get("source") or "?"))
        inp = [syl_from_dict(d) for d in it["input_syllables"]]
        out_text = it.get("rule_output") or (it["gold"][0] if isinstance(it.get("gold"), list) else it.get("gold"))
        parsed = [try_parse(w, strict=False) for w in str(out_text or "").split()]
        if len(parsed) != 2 or any(p is None for p in parsed):
            continue
        outs = [p.syllable for p in parsed]
        try:
            cov[iid] = MT.h6_covariates(it["input"], out_text, variant=str(it.get("variant")), output_lexical=True,
                                        spelling_triggers=spelling_triggers((inp[0], inp[1]), (outs[0], outs[1])),
                                        tone_pair=MT.tone_pair_of(it["input_syllables"]),
                                        input_freq=[freqs.get(spell(x), 0) for x in inp], output_freq=[freqs.get(spell(x), 0) for x in outs])
        except ValueError:
            continue
    return cov, {"n_exact": n_exact, "n_eligible_h6": n_elig, "n_verified_exact": n_verified, "n_collections": len(collections)}


def _generated_covariates(rows: pd.DataFrame, items: Mapping[str, dict]) -> pd.DataFrame:
    """The matching covariates of the generated control pool: the row's strata (frequencies, tone pair, spelling triggers,
    lexicality) and the syllable lengths from the item text."""
    recs = []
    for r in rows.drop_duplicates("item_id").itertuples(index=False):
        it = items.get(r.item_id)
        if it is None:
            continue
        gold = it["gold"][0] if isinstance(it.get("gold"), list) else it.get("gold")
        try:
            c = MT.h6_covariates(it["input"], str(gold), variant=str(r.variant), output_lexical=bool(r.output_lexical),
                                 spelling_triggers=r.spelling_triggers if isinstance(r.spelling_triggers, (list, tuple)) else None,
                                 tone_pair=r.tone_pair, input_freq=r.input_freq if isinstance(r.input_freq, (list, tuple)) else None,
                                 output_freq=r.output_freq if isinstance(r.output_freq, (list, tuple)) else None)
        except ValueError:
            continue
        recs.append({"item_id": r.item_id, **c})
    return pd.DataFrame(recs)


def table_h6(ctx: Context) -> Artefact:
    """H6 (PREREG 8.8 / DD 8.7, :29): every exact attested item matched to 3-5 generated items with lexical output from base pairs
    outside the attested set (noilai.stats.matching: exact on variant, task class, output lexicality, spelling-trigger presence and
    tone pair; nearest on the log syllable frequencies and syllable lengths, 0.2-SD caliper; SMDs after matching reported), the
    estimate = mean over attested items of (acc_attested - mean acc of its controls) with the base-pair bootstrap, Holm across the
    panel and the pooled estimate. The floor (>= 100 verified exact two-syllable rows from >= 3 collections and the
    attested__expansion unit done) decides whether the test runs; below it the matched estimate is DESCRIPTIVE (no p, no pool)."""
    title = "H6: exact attested items vs matched generated items (matched contrast; descriptive below the 100-row floor)"
    headers = ["Model", "Family", "attested - matched, pts [95% CI]", "p", "p (Holm)", "attested matched/total", "controls", "balance (max |SMD|)", "status"]
    spec = (ctx.cfg.get("lines") or {}).get("attested") or {}
    line, control_line = str(spec.get("line", "E1_attested")), str(spec.get("control_line", "E1_main"))
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    floor = int(spec.get("floor", constants.ATTESTED_EXACT_FLOOR_FOR_H6))
    k_min, k_max, caliper = int(spec.get("k_min", MT.H6_K_MIN)), int(spec.get("k_max", MT.H6_K_MAX)), spec.get("caliper_sd", MT.CALIPER_SD)
    floor_unit_done = False
    if ctx.ledger_rows is not None:
        floor_unit_done = any(r.get("unit_id") == spec.get("floor_unit", "attested__expansion") and r.get("status") == "done" for r in ctx.ledger_rows)
    d = ctx.rows(line, arm=base_arm, paraphrase="p0")
    extra: dict = {"floor": floor, "floor_unit_done": floor_unit_done, "k_min": k_min, "k_max": k_max, "caliper_sd": caliper,
                   "exact_on": list(MT.H6_EXACT), "nearest_on": list(MT.H6_NEAREST), "smd_balance": MT.SMD_BALANCE}
    if len(d) == 0:
        return not_computable("table_h6", title, f"no completed {line} rows under {base_arm} (p0); floor unit done: {floor_unit_done}", headers,
                              analysis=ANALYSES[7], extra=extra)
    try:
        from ..vi import lexicon as L
        freqs = L.syllable_frequencies()
    except (OSError, FileNotFoundError) as e:
        return not_computable("table_h6", title, f"the Viet74K word list is not fetched (syllable frequencies for the matching): {e}", headers,
                              analysis=ANALYSES[7], extra=extra)
    fam_map = ctx.family_map() or {}
    rows, tex_rows, records, tests_ = [], [], [], []
    descriptive_rows = []
    counts_any: dict = {}
    for model in sorted(d["model"].unique()):
        dm = d[d["model"] == model]
        rec: dict = {"model": model, "family": fam_map.get(model), "n_attested_rows": len(dm), "n_attested_items": int(dm["item_id"].nunique())}
        c = acc_cell(dm[dm["exact"].fillna(False).astype(bool)] if "exact" in dm and dm["exact"].notna().any() else dm, ctx)
        descriptive_rows.append({"model": model, **c})
        att_path = _item_file_of(ctx, model, line)
        if att_path is None:
            rec["note"] = f"the attested item file of {line} is not on this machine (the matching covariates need it)"
            rows.append(rec)
            continue
        att_items = _items_by_id(ctx, att_path)
        cov_att, counts = _attested_covariates(att_items, freqs)
        counts_any = counts
        pool_rows = ctx.rows(control_line, arm=base_arm, paraphrase="p0", task="T1")
        pool_rows = pool_rows[pool_rows["model"] == model]
        if "output_lexical" in pool_rows:
            pool_rows = pool_rows[pool_rows["output_lexical"].fillna(False).astype(bool)]
        if "attested_overlap" in pool_rows:
            pool_rows = pool_rows[~pool_rows["attested_overlap"].fillna(False).astype(bool)]
        main_path = _item_file_of(ctx, model, control_line)
        if len(pool_rows) == 0 or main_path is None:
            rec["note"] = (f"no {control_line} T1 rows with lexical output for {model}" if len(pool_rows) == 0
                           else f"the {control_line} item file is not on this machine (syllable lengths for the matching need it)")
            rows.append(rec)
            continue
        treated = pd.DataFrame([{"item_id": iid, **cov_att[iid]} for iid in dm["item_id"].unique() if iid in cov_att])
        rec["n_exact_eligible_with_rows"] = len(treated)
        if len(treated) == 0:
            rec["note"] = "no exact, H6-eligible attested item among the scored rows"
            rows.append(rec)
            continue
        pool = _generated_covariates(pool_rows, _items_by_id(ctx, main_path))
        if len(pool) == 0:
            rec["note"] = "no control pool item could be read from the item file"
            rows.append(rec)
            continue
        match = MT.nearest_neighbour_match(treated, pool, exact=MT.H6_EXACT, nearest=MT.H6_NEAREST, k_min=k_min, k_max=k_max, caliper_sd=caliper)
        rec["matching"] = match.as_dict()
        if match.n_matched == 0:
            rec["note"] = "no attested item found k_min controls within the caliper"
            rows.append(rec)
            continue
        y_t = dict(zip(dm.drop_duplicates("item_id")["item_id"], dm.drop_duplicates("item_id")["correct"].fillna(False).astype(float)))
        pr = pool_rows.drop_duplicates("item_id")
        y_c = dict(zip(pr["item_id"], pr["correct"].fillna(False).astype(float)))
        clusters = dict(zip(dm["item_id"], dm[CLUSTER_KEY]))
        try:
            mc = MT.matched_contrast(y_t, y_c, match, clusters, n_boot=ctx.paired_boot, seed=ctx.seed, level=ctx.level)
        except ValueError as e:
            rec["note"] = f"matched contrast not computable: {e}"
            rows.append(rec)
            continue
        rec.update(contrast=mc.as_dict(), effect=mc.estimate, ci_lo=mc.lo, ci_hi=mc.hi, p_bootstrap=mc.p_bootstrap)
        records.append({"model": model, "values": list(mc.per_treated["contribution"]), "clusters": list(mc.per_treated["cluster"])})
        tests_.append(rec)
        rows.append(rec)
    floor_met = bool(counts_any) and counts_any.get("n_verified_exact", 0) >= floor and counts_any.get("n_collections", 0) >= constants.ATTESTED_MIN_COLLECTIONS \
        and floor_unit_done
    status = "test run" if floor_met else "descriptive: floor not met (PREREG :29); no p-values, no pooled estimate"
    if tests_ and floor_met:
        reject, adj = holm_adjust([r["p_bootstrap"] for r in tests_])
        for r, rj, pa in zip(tests_, reject, adj):
            r.update(p_holm=float(pa), significant=bool(rj), holm_family_size=len(tests_))
    for r in rows:
        r["status"] = status if "effect" in r else (r.get("note") or "not matched")
        if "effect" in r:
            m = r["matching"]
            bal = f"{'yes' if m['balanced'] else 'no'} ({m['max_abs_smd_after']:.2f})" if m.get("max_abs_smd_after") is not None else "?"
            tex_rows.append([_tex(r["model"]), _tex(r["family"] or "?"), ci_pts(r["effect"], r["ci_lo"], r["ci_hi"]),
                             pval(r["p_bootstrap"]) if floor_met else "--", pval(r.get("p_holm")) if floor_met else "--",
                             f"{m['n_matched']}/{m['n_treated']}", str(m["n_controls_distinct"]), bal, "test" if floor_met else "descriptive"])
        else:
            tex_rows.append([_tex(r["model"]), _tex(r.get("family") or "?"), "--", "--", "--", str(r.get("n_attested_items", "")), "--", "--", _tex(r["status"])])
    pooled_h6 = None
    if tests_ and floor_met:
        pooled_h6 = pooled(ctx, records, direction=+1, min_families=min_families(ctx, "H6", 3), label="H6 attested - matched")
    decision = {"rule": "pooled acc(attested) - acc(matched) > 0 with the CI excluding 0, per-model tests Holm across the panel; H6 is tested only above the floor "
                        f"({floor} verified exact two-syllable rows from >= {constants.ATTESTED_MIN_COLLECTIONS} collections and the expansion unit done), else descriptive",
                "floor_met": floor_met, "floor_counts": counts_any, "on_predicted_side": (pooled_h6 or {}).get("on_predicted_side") if pooled_h6 else None}
    extra.update(floor_met=floor_met, floor_counts=counts_any, status=status, descriptive_accuracy=descriptive_rows, pooled=pooled_h6, decision=decision,
                 test="run" if floor_met else "not run")
    cap = (f"Matched contrast per model (mean over attested items of acc_attested - mean acc of 3-5 matched generated items; base-pair bootstrap); "
           f"{status}.")
    return Artefact("table_h6", title, headers, tex_rows, rows, computable=bool(tests_), reason=None if tests_ else "no model yielded a matched contrast; see the status column",
                    caption=cap, extra=extra, analysis=ANALYSES[7])


def table_h1_e2(ctx: Context) -> Artefact:
    """H1 / E2 (PREREG 8.4): the E1_main T1 rows under the baseline arm joined per model to the item audit of its tokenizer
    (data/audit/items_<tokenizer>__<key>.jsonl; noilai.stats.matching.e2_frame), the identifiability floor per tokenizer
    (e2.identifiability_floor: a model with fewer than 300 misaligned split syllables contributes descriptively only), then
    e2.nested_h1_test (LR / ΔAIC of M0 + tps_w vs + split + align_w) and the base-pair cluster bootstrap of the mean per-model
    slopes. H1 is supported when the test passes, >= 4 families clear the floor and the minimum viable panel holds."""
    title = "H1 / E2: nested LR test of alignment and fragmentation (PREREG 8.4) with the identifiability floor"
    headers = ["Model", "Family", "rows", "items", "misaligned split syll.", "clears floor", "slope tps_w", "slope align_w", "in fit"]
    if not ctx.runs:
        return not_computable("table_h1_e2", title, "no completed run (the E1_main T1 rows and the item audits are the inputs)", headers,
                              analysis=ANALYSES[2])
    e2cfg = ctx.cfg.get("e2") or {}
    floor = int(e2cfg.get("misaligned_split_floor", E2.H1_MISALIGNED_SPLIT_FLOOR))
    min_fam = int(e2cfg.get("min_families", E2.H1_MIN_FAMILIES))
    base_arm = str(ctx.cfg.get("baseline_arm") or "nfc")
    main = str((((ctx.cfg.get("lines") or {}).get("table2") or {}).get("lines") or ["E1_main"])[0])
    d = ctx.rows(main, arm=base_arm, task="T1")
    fam = ctx.family_map() or {}
    models = sorted(d["model"].unique()) if len(d) else []
    fams = {fam.get(m) for m in models} - {None}
    mvp = (ctx.cfg.get("families") or {}).get("min_viable_panel") or {}
    expected = set((ctx.cfg.get("families") or {}).get("pass_through_expected") or [])
    non_gemma = bool((fams & expected) - {"gemma_sp", "gemma3", "gemma4"})
    ok = len(models) >= int(mvp.get("models", constants.MIN_VIABLE_PANEL_MODELS)) and len(fams) >= int(mvp.get("families", constants.MIN_VIABLE_PANEL_FAMILIES)) \
        and (non_gemma or not mvp.get("needs_non_gemma_pass_through", True))
    audits = sorted(p.name for p in ctx.paths["audit_dir"].glob("items_*.jsonl")) if ctx.paths["audit_dir"].exists() else []
    extra: dict = {"models": models, "families": sorted(fams), "mvp": ok, "audit_files": audits, "floor": floor, "min_families": min_fam,
                   "baseline_arm": base_arm, "line": main}
    if len(d) == 0:
        return not_computable("table_h1_e2", title, f"no completed {main} T1 rows under {base_arm}", headers, analysis=ANALYSES[2], extra=extra)
    rows, parts, floor_counts, notes = [], [], {}, []
    for model in models:
        dm = d[d["model"] == model]
        rec = {"model": model, "family": fam.get(model), "n_rows": len(dm), "n_items": int(dm["item_id"].nunique()), "n_misaligned_split": None,
               "clears_floor": None, "slope_tps": None, "slope_align": None, "in_fit": False, "note": None}
        audit, where = items_audit_for(ctx, model, _audit_key_of(ctx, model, main))
        if audit is None:
            rec["note"] = where
            notes.append(where)
            rows.append(rec)
            continue
        a = audit[(audit["encoding"] == "nfc") & audit["item_id"].isin(set(dm["item_id"]))]
        n_mis = int(pd.to_numeric(a["n_misaligned_split"], errors="coerce").fillna(0).sum())
        floor_counts[model] = n_mis
        joined = MT.e2_frame(dm, audit, encoding="nfc")
        rec.update(n_misaligned_split=n_mis, clears_floor=n_mis >= floor, audit=where, n_audited_items=int(a["item_id"].nunique()), n_joined=len(joined))
        if len(joined) and rec["clears_floor"]:
            parts.append(joined)
            rec["in_fit"] = True
        elif len(joined) == 0:
            rec["note"] = "no score row joins an audit row (item ids differ?)"
        else:
            rec["note"] = f"below the floor ({n_mis} < {floor}): descriptive only"
        rows.append(rec)
    fams_clear = sorted({fam.get(m) for m, n in floor_counts.items() if n >= floor and fam.get(m)})
    identified = len(fams_clear) >= min_fam
    extra.update(families_clearing=fams_clear, h1_identified=identified, floor_counts=floor_counts, notes=notes)

    def _tex_rows():
        out = []
        for r in rows:
            out.append([_tex(r["model"]), _tex(r["family"] or "?"), str(r["n_rows"]), str(r["n_items"]),
                        "--" if r["n_misaligned_split"] is None else str(r["n_misaligned_split"]),
                        "--" if r["clears_floor"] is None else ("yes" if r["clears_floor"] else "no"),
                        "--" if r["slope_tps"] is None else f"{r['slope_tps']:+.2f}", "--" if r["slope_align"] is None else f"{r['slope_align']:+.2f}",
                        "yes" if r["in_fit"] else "no"])
        return out

    if not parts:
        reason = "no model with E1_main T1 rows both has an item audit and clears the identifiability floor" + (f"; {notes[0]}" if notes else "")
        return Artefact("table_h1_e2", title, headers, _tex_rows(), rows, computable=False, reason=reason, analysis=ANALYSES[2], extra=extra)
    frame = E2.prepare_covariates(pd.concat(parts, ignore_index=True))
    try:
        h1 = E2.nested_h1_test(frame)
        fe = E2.fit_fixed_effects_clustered(frame, sandwich=False)
        boot = E2.bootstrap_mean_slopes(frame, n_boot=min(int(e2cfg.get("n_boot", 200)), ctx.n_boot), seed=ctx.seed, level=ctx.level)
    except Exception as e:
        extra["fit_error"] = f"{type(e).__name__}: {e}"
        return Artefact("table_h1_e2", title, headers, _tex_rows(), rows, computable=False,
                        reason=f"E2 fit failed on {len(frame)} rows: {type(e).__name__}: {str(e)[:160]}", analysis=ANALYSES[2], extra=extra)
    for r in rows:
        if r["in_fit"]:
            r["slope_tps"], r["slope_align"] = _f(fe.slopes_tps.get(r["model"])), _f(fe.slopes_align.get(r["model"]))
    supported = bool(h1["h1_supported"]) and identified and ok
    verdict = "supported" if supported else ("descriptive (floor or minimum viable panel not met)" if not (identified and ok) else "falsified")
    decision = {"h1_supported_by_test": bool(h1["h1_supported"]), "identified": identified, "mvp": ok, "h1": verdict,
                "rule": "LR p < alpha and ΔAIC >= 2 and beta(align_w) > 0 and beta(tps_w) < 0 (PREREG 8.4), with >= 4 families clearing the "
                        f"{floor}-misaligned-split floor and the minimum viable panel (>= 8 models, >= 4 families, a non-Gemma pass-through family)"}
    extra.update(nested=h1, bootstrap=boot, n_fit_rows=len(frame), n_fit_models=int(frame["model"].nunique()), decision=decision,
                 primary_formula=E2.PRIMARY_FORMULA)
    cap = (f"Per model: rows and items joined to the item audit, misaligned split syllables under NFC against the floor of {floor}, the per-model "
           f"slopes of tps_w and align_w from the primary fit; the nested LR test (p = {pval(h1['p'])}, ΔAIC = {h1['delta_aic']:.1f}) and the "
           f"bootstrap CIs of the mean slopes are in the .json sibling. H1: {verdict}.")
    return Artefact("table_h1_e2", title, headers, _tex_rows(), rows, caption=cap, extra=extra, analysis=ANALYSES[2])


def table_h3b(ctx: Context) -> Artefact:
    """H3b (PREREG 8.7 / DD 1, :26): the tone-isolation difference-in-differences on the E3_noilai T1 rows (p0) of every pass-through
    model with an item audit: ISO items (the audit's NFD `tone_isolated`) matched to NON-ISO items nearest on syllable frequency, NFC
    and NFD token count within the lexical / pseudo strata (noilai.stats.matching.h3b_did); Δ = (y(NFD) - y(NFC))_ISO minus the
    matched NON-ISO items' effect, on V3 (expected > 0) and on V1 (expected ≈ 0); Holm over the four configured contrasts."""
    cfg = (ctx.cfg.get("lines") or {}).get("h3b") or {}
    t3 = (ctx.cfg.get("lines") or {}).get("table3") or {}
    line = str(cfg.get("line", t3.get("main", "E3_noilai")))
    conf_models = [str(m) for m in (cfg.get("models") or ["gemma-3-4b-it", "gemma-3-1b-it"])]
    variants = [str(v) for v in (cfg.get("variants") or ["V3", "V1"])]
    base_arm, arm = str(t3.get("baseline_arm", ctx.cfg.get("baseline_arm") or "nfc")), str(cfg.get("arm", "nfd"))
    min_items = int(cfg.get("min_items", MT.H3B_MIN_ITEMS))
    k_min, k_max, caliper = int(cfg.get("k_min", MT.H3B_K_MIN)), int(cfg.get("k_max", MT.H3B_K_MAX)), cfg.get("caliper_sd", MT.CALIPER_SD)
    title = "H3b: tone-isolation DiD on T1 (ISO items matched to NON-ISO items), V3 expected > 0 and V1 expected ≈ 0"
    headers = ["Model", "variant", "Δ pts [95% CI]", "p", "p (Holm)", "sig.", "ISO matched/total", "items in cell", "balance (max |SMD|)", "status"]
    d = ctx.rows(line, paraphrase="p0", task="T1")
    d = d[d["arm"].isin([base_arm, arm])] if len(d) else d
    extra: dict = {"line": line, "configured_models": conf_models, "variants": variants, "baseline_arm": base_arm, "arm": arm, "min_items": min_items,
                   "k_min": k_min, "k_max": k_max, "caliper_sd": caliper, "exact_on": list(MT.H3B_EXACT), "nearest_on": list(MT.H3B_NEAREST)}
    if len(d) == 0:
        return not_computable("table_h3b", title, f"no completed {line} T1 rows under {base_arm} / {arm} (p0)", headers, analysis=ANALYSES[4], extra=extra)
    fam_map = ctx.family_map() or {}
    rows, results = [], []
    for model in sorted(d["model"].unique()):
        ok, why = pass_through(ctx, model)
        if not ok:
            rows.append({"model": model, "family": fam_map.get(model), "variant": None, "status": f"not informative: {why} (NFD = NFC by construction)"})
            continue
        audit, where = items_audit_for(ctx, model, _audit_key_of(ctx, model, line))
        if audit is None:
            rows.append({"model": model, "family": fam_map.get(model), "variant": None, "status": where})
            continue
        dm = d[d["model"] == model]
        for v in variants:
            r = MT.h3b_did(dm, audit, variant=v, baseline_arm=base_arm, arm=arm, k_min=k_min, k_max=k_max, caliper_sd=caliper, min_items=min_items,
                           n_boot=ctx.paired_boot, seed=ctx.seed, level=ctx.level)
            r.update(model=model, family=fam_map.get(model), configured=model in conf_models, audit=where)
            results.append(r)
            rows.append(r)
    conf = [r for r in results if r.get("computable") and r["configured"]]
    if conf:
        reject, adj = holm_adjust([r["p_bootstrap"] for r in conf])
        for r, rj, pa in zip(conf, reject, adj):
            r.update(p_holm=float(pa), significant=bool(rj), holm_family_size=len(conf))
    decisions = {}
    for model in {r["model"] for r in results}:
        by_v = {r["variant"]: r for r in results if r["model"] == model and r.get("computable")}
        v3, v1 = by_v.get(variants[0]), by_v.get(variants[1]) if len(variants) > 1 else None
        pos = bool(v3 and v3["lo"] > 0)
        null_ok = v1 is None or bool(v1["lo"] <= 0 <= v1["hi"])
        decisions[model] = {"v3_positive_ci_excludes_0": pos if v3 else None, "v1_ci_includes_0": (v1["lo"] <= 0 <= v1["hi"]) if v1 else None,
                            "supported": bool(v3 and pos and v1 is not None and null_ok), "underpowered": any(r.get("underpowered") for r in by_v.values())}
    tex_rows = []
    for r in rows:
        if r.get("computable"):
            m = r["matching"]
            bal = f"{'yes' if m['balanced'] else 'no'} ({m['max_abs_smd_after']:.2f})" if m.get("max_abs_smd_after") is not None else "?"
            status = ("Holm family" if r["configured"] else "exploratory model") + (f"; underpowered (< {min_items} items)" if r["underpowered"] else "")
            tex_rows.append([_tex(r["model"]), r["variant"], ci_pts(r["estimate"], r["lo"], r["hi"]), pval(r["p_bootstrap"]), pval(r.get("p_holm")),
                             "yes" if r.get("significant") else ("no" if r.get("significant") is False else "--"), f"{m['n_matched']}/{r['n_iso']}",
                             str(r["n_items"]), bal, status])
        else:
            tex_rows.append([_tex(r["model"]), r.get("variant") or "--", "--", "--", "--", "--", "--", str(r.get("n_items", "--")), "--",
                             _tex(r.get("status") or r.get("reason") or "not computable")])
    extra.update(decisions=decisions, rule="Δ(V3) CI > 0 (excluding 0) and Δ(V1) CI including 0 per model (PREREG :26); Holm over the four configured contrasts; "
                                           "fewer than 500 items in a cell is reported as underpowered (PREREG 8.7)")
    computable = any(r.get("computable") for r in results)
    reason = None if computable else ("; ".join(str(r.get("status") or r.get("reason")) for r in rows[:3]) or "no contrast computable")
    cap = ("Tone-isolation DiD per model and variant: ISO items (NFD isolates the tone mark) against NON-ISO items matched within lexical / pseudo "
           "strata on syllable frequency and NFC / NFD token counts; base-pair bootstrap over the ISO items; Holm over the configured contrasts.")
    return Artefact("table_h3b", title, headers, tex_rows, rows, computable=computable, reason=reason, caption=cap, extra=extra, analysis=ANALYSES[4])


def fig_probes(ctx: Context) -> Artefact:
    """E4 (H5): the probe directories' files are listed; the retention table and figure data are produced by the
    probe scripts, not here (reported regardless of outcome, DD 8.8)."""
    title = "E4 probes and patching: deliverables found (reported regardless of outcome)"
    headers = ["directory", "files"]
    probe_line = str((ctx.cfg.get("lines") or {}).get("probes", "E4_probe"))
    root = ctx.paths["runs_root"]
    rows, tex_rows = [], []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith(probe_line + "__"):
                files = sorted(p.name for p in d.iterdir() if p.suffix in (".csv", ".json"))
                rows.append({"directory": d.name, "files": files})
                tex_rows.append([_tex(d.name), _tex(", ".join(files))])
    if not rows:
        return not_computable("fig_probes", title, f"no {probe_line}__<model> directory under {root}", headers, analysis=ANALYSES[6])
    return Artefact("fig_probes", title, headers, tex_rows, rows, caption="Probe deliverables present; the retention table is not computed here.",
                    analysis=ANALYSES[6], extra={"note": "not run here: probes.csv / excess.csv / patching*.json are summarized by the probe scripts"})


def table_bf16_drift(ctx: Context) -> Artefact:
    """Appendix: `<model>--bf16` runs of bf16_drift_200 against the T4 E1_main rows of their reference model on the same items."""
    title = "bf16 drift: serving variant vs the reference run on the same items"
    headers = ["Model", "reference", "drift pts [95% CI]", "p", "items", "base pairs"]
    line = str(((ctx.cfg.get("lines") or {}).get("appendix") or {}).get("bf16", "bf16_drift_200"))
    d = ctx.rows(line, paraphrase="p0")
    if len(d) == 0:
        return not_computable("table_bf16_drift", title, f"no completed {line} rows", headers)
    rows, tex_rows = [], []
    for model, g in d.groupby("model", sort=True):
        ref = (ctx.midx.get(model) or {}).get("reference_of") or model.split("--")[0]
        base = ctx.rows("E1_main", paraphrase="p0", arm=str(ctx.cfg.get("baseline_arm") or "nfc"))
        base = base[base["model"] == ref]
        pc = paired_contrast(base, g, ctx, model, keys=("item_id",))
        if pc is None:
            rows.append({"model": model, "reference": ref, "note": "no reference rows on the same items"})
            continue
        t, extra = pc
        rows.append({"model": model, "reference": ref, **test_row(t, extra)})
        tex_rows.append([_tex(model), _tex(ref), ci_pts(t.diff, t.ci_lo, t.ci_hi), pval(t.p_primary), str(extra["n"]), str(extra["n_base_pairs"])])
    return Artefact("table_bf16_drift", title, headers, tex_rows, rows, caption="Accuracy difference of the bf16 serving variant vs its reference on the same items.")


def table_engine_scope_agreement(ctx: Context) -> Artefact:
    """Appendix: E3_scope_item (item scope) and E3_engine_hf (hf engine) vs E3_noilai on the same rows: agreement rate."""
    title = "Two-scope and two-engine agreement with E3_noilai on the same rows"
    headers = ["Model", "line", "arm", "agreement %", "accuracy line", "accuracy E3", "rows"]
    app = (ctx.cfg.get("lines") or {}).get("appendix") or {}
    main = str(((ctx.cfg.get("lines") or {}).get("table3") or {}).get("main", "E3_noilai"))
    rows, tex_rows = [], []
    for key in ("scope", "engine"):
        line = str(app.get(key, ""))
        d = ctx.df[(ctx.df["line"] == line)] if line and len(ctx.df) else ctx.df.iloc[0:0]
        for (model, arm), g in d.groupby(["model", "arm"], sort=True):
            ref = ctx.rows(main, paraphrase="p0", arm=arm)
            ref = ref[ref["model"] == model]
            al = align_pairs(ref, g, keys=("item_id",))
            if al is None:
                continue
            a, b, _, _ = al
            rows.append({"model": model, "line": line, "arm": arm, "agreement": float(np.mean(a == b)), "accuracy_line": float(b.mean()),
                         "accuracy_e3": float(a.mean()), "n": len(a)})
            tex_rows.append([_tex(model), _tex(line), _tex(arm), pct(np.mean(a == b)), pct(b.mean()), pct(a.mean()), str(len(a))])
    if not rows:
        return not_computable("table_engine_scope_agreement", title, "no completed E3_scope_item / E3_engine_hf rows paired with E3_noilai", headers)
    return Artefact("table_engine_scope_agreement", title, headers, tex_rows, rows, caption="Row-level agreement of the secondary scope / engine with the primary run.")


def table_reasoning_curve(ctx: Context) -> Artefact:
    """Appendix: accuracy and thinking length of the reasoning sub-study runs (never a main-table row)."""
    title = "Reasoning sub-study: accuracy against thinking characters (appendix only)"
    headers = ["Model", "Task", "accuracy % [95% CI]", "mean thinking chars", "items", "base pairs"]
    lines = list(((ctx.cfg.get("lines") or {}).get("appendix") or {}).get("reasoning") or [])
    d = ctx.rows(lines)
    if len(d) == 0:
        return not_computable("table_reasoning_curve", title, f"no completed rows on {lines}", headers)
    rows, tex_rows = [], []
    for keys, g in _groups(d, ("model", "task")):
        c = acc_cell(g, ctx)
        think = pd.to_numeric(g["n_thinking_chars"], errors="coerce").mean() if "n_thinking_chars" in g else None
        rows.append({**keys, **c, "mean_thinking_chars": None if think is None or not np.isfinite(think) else float(think)})
        tex_rows.append([_tex(keys["model"]), keys["task"], ci_pct(c["estimate"], c["lo"], c["hi"]), f"{think:.0f}" if _finite(think) else "--", str(c["n_items"]), str(c["n_base_pairs"])])
    return Artefact("table_reasoning_curve", title, headers, tex_rows, rows, caption="Thinking ON runs (`--thinking` variants); appendix material only.")


def table_deff_icc(ctx: Context, t2: Artefact) -> Artefact:
    """Appendix: the realized design effect and ICC of every Table 2 p0 cell (DD 8.5 / 8.9)."""
    title = "Design effect and ICC per Table 2 cell (p0)"
    headers = ["Model", "Task", "Variant", "DEFF", "ICC", "items", "base pairs"]
    if not t2.computable:
        return not_computable("table_deff_icc", title, t2.reason or "Table 2 not computable", headers)
    rows, tex_rows = [], []
    for r in t2.rows:
        p = r.get("primary")
        if not p:
            continue
        rows.append({"model": r["model"], "unit": r["unit"], "task": r["task"], "variant": r["variant"], "deff": p["deff"], "icc": p["icc"],
                     "n_items": p["n_items"], "n_base_pairs": p["n_base_pairs"]})
        tex_rows.append([_tex(r["model_label"]), r["task"], str(r["variant"] or "--"), f"{p['deff']:.2f}", f"{p['icc']:.3f}", str(p["n_items"]), str(p["n_base_pairs"])])
    return Artefact("table_deff_icc", title, headers, tex_rows, rows, caption="DEFF = 1 + (m - 1) ICC with the ANOVA ICC over base pairs (DD 8.5).")


def all_artefacts(ctx: Context) -> list[Artefact]:
    t2 = table2(ctx)
    ctx.n_cells = len(t2.rows)
    t3, t3mc = table3(ctx)
    arts = [t2, table2_range(ctx, t2), t3, t3mc, table_h3(ctx), table_h3b(ctx), table_h4(ctx), table_h1_e2(ctx), table_explicit_input(ctx),
            table_human(ctx), table_census(ctx), table_runs(ctx), table_h6(ctx), table_ablations(ctx), table_errors(ctx)]
    errors = arts[-1]
    fig_err = Artefact("fig_errors", "Fig. 3 data: error-class shares", errors.headers, errors.tex_rows, errors.rows, errors.computable,
                       errors.reason, errors.caption, dict(errors.extra))
    arts += [fig_err, fig_deltatok(ctx), fig_probes(ctx), table_bf16_drift(ctx), table_engine_scope_agreement(ctx),
             table_reasoning_curve(ctx), table_deff_icc(ctx, t2)]
    return arts


# ============================================================================ labels, FINAL, provenance
def git_info(root: Path = ROOT) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def refuse_paper(path: Path) -> None:
    """CLAUDE.md: nothing of this workstream is written under paper/."""
    if Path(path).resolve().is_relative_to(PAPER_DIR.resolve()):
        raise ValueError(f"refusing to write under paper/ ({path}): the analysis never writes paper files; the paper workstream copies from --out")


def table_lines(cfg: Mapping) -> set[str]:
    """Every run-plan line behind Table 2 and Table 3, aliases included (the units FINAL requires done)."""
    L = cfg.get("lines") or {}
    t3 = L.get("table3") or {}
    lines = set((L.get("table2") or {}).get("lines") or [])
    lines |= {str(t3.get("main", "E3_noilai")), str(t3.get("c2", "E3_c2_enriched")), str((t3.get("xcopa") or {}).get("line", "E3_xcopa")),
              str((t3.get("api") or {}).get("line", "E3_api_core")), str((t3.get("api") or {}).get("baseline_line", "E1_api_core"))}
    for alias, target in (L.get("aliases") or {}).items():
        if target in lines:
            lines.add(str(alias))
    return lines


def final_blockers(ctx: Context) -> list[str]:
    """Why FINAL cannot be claimed (labels.final_requires); empty when every requirement holds."""
    out = []
    gates_path = ctx.paths["gates"]
    gates = _read_json(gates_path) if gates_path.suffix == ".json" else None
    if gates is None:
        try:
            gates = yaml.safe_load(gates_path.read_text(encoding="utf-8")) if gates_path.exists() else None
        except (OSError, yaml.YAMLError):
            gates = None
    if not isinstance(gates, Mapping):
        out.append(f"gates file {gates_path} missing or unreadable: the registration and the scoring-rule freeze are not recorded")
    else:
        pre = gates.get("preregistration") or {}
        if not (pre.get("registration_url") and pre.get("registration_date")):
            out.append("preregistration.registration_url / registration_date not recorded in experiments/gates.yaml")
        if not (gates.get("scoring_rule_freeze") or {}).get("commit"):
            out.append("scoring_rule_freeze.commit not recorded (scripts/freeze_scoring_rule.py --record is the author's command)")
        else:
            script = ctx.paths["freeze_check"]
            if script.exists():
                r = subprocess.run([sys.executable, str(script), "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
                if r.returncode != 0:
                    out.append(f"scripts/freeze_scoring_rule.py --check failed: {(r.stdout + r.stderr).strip()[:200]}")
    if ctx.ledger_rows is None:
        out.append("no ledger: FINAL needs experiments/ledger.csv to show every live Table 2 / Table 3 unit done and hash-verified")
    else:
        lines = table_lines(ctx.cfg)
        missing = [r.get("unit_id") for r in ctx.ledger_rows if r.get("run_id") in lines and r.get("status") not in NON_LIVE_STATUSES
                   and not (r.get("status") == "done" and (r.get("hash_verified") or "").lower() != "no")]
        if missing:
            out.append(f"{len(missing)} live Table 2 / Table 3 units are not done and hash-verified (first: {', '.join(str(m) for m in missing[:3])})")
    if any(r.smoke for r in ctx.runs):
        out.append("a smoke run (options.notes.smoke) is among the analysed runs")
    if ctx.unscorable:
        out.append(f"{len(ctx.unscorable)} completed run directories are unscorable")
    return out


def n_label(label: str, n: Mapping) -> str:
    return f"{label} (n = {n['items']:,} items, {n['base_pairs']:,} base pairs, {n['models']} models)"


def analyses_status(ctx: Context, arts: Sequence[Artefact]) -> dict:
    """Every pre-registered analysis (the names of scripts/progress.py) with whether it was computable here."""
    by_name = {a.analysis: a for a in arts if a.analysis}
    out = {}
    for name in ANALYSES:
        a = by_name.get(name)
        if a is None:
            out[name] = {"computable": False, "reason": "no artefact of this module decides it"}
        else:
            out[name] = {"computable": bool(a.computable and not a.empty()), "reason": a.reason if not a.computable else (None if a.rows else "no rows"),
                         "artefact": a.name}
    return out


# ============================================================================ rendering and writing
def render_tabular(art: Artefact, label_line: str) -> str:
    """The .tex body of an artefact: the label line, then a tabular (an empty table shows one `(no completed run)` row)."""
    if art.tex_body is not None:
        return label_line + "\n" + art.tex_body + "\n"
    n_cols = max(1, len(art.headers))
    lines = [label_line, "\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "c" * (n_cols - 1) + "}", "\\toprule",
             " & ".join(_tex(h) for h in art.headers) + " \\\\", "\\midrule"]
    if art.tex_rows:
        lines += [" & ".join(r) + " \\\\" for r in art.tex_rows]
    else:
        reason = art.reason or "no completed run"
        lines.append("(no completed run)" + (" & " * (n_cols - 1)) + " \\\\")
        lines.append(f"% not computable: {reason}")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{art.caption or art.title}}}", f"\\label{{tab:{art.name}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def stamp_lines(inputs_desc: str, stamp: str, git: Mapping, body: str) -> list[str]:
    dirty = "-dirty" if git.get("dirty") else ""
    return [f"% GENERATED by {GENERATOR} from {inputs_desc} on {stamp}; do not edit by hand.",
            f"% generator commit {(git.get('commit') or 'unknown')[:12]}{dirty}, content sha256 {sha256_text(body)}, generated {stamp}"]


def body_of(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.startswith(STAMP_PREFIXES)]


def results_log_entry(ctx: Context, arts: Sequence[Artefact], date: str, call: str, git: Mapping, cfg_sha: str, out_dir: Path) -> str:
    n = ctx.n
    kind = "model outputs" if ctx.runs else "no model outputs"
    run_list = ", ".join(f"`{r.run_id}` results_sha256 `{(r.excerpt.get('results_sha256') or '-')[:SHA_SHORT]}`" for r in ctx.runs) or "none"
    commit = (git.get("commit") or "unknown")[:12] + (" (dirty tree)" if git.get("dirty") else "")
    source = (f"Source: `{GENERATOR}` ({n_label(ctx.label, n)}), the exact call `{call}` at commit {commit}; configuration "
              f"`configs/analysis.yaml` sha256 `{cfg_sha[:SHA_SHORT]}…`; {len(ctx.runs)} completed run directories ({run_list}); "
              f"outputs under `{out_dir}`. NN is assigned when the entry is pasted into docs/RESULTS_LOG.md (this module never edits docs/).")
    L = [f"## {date} — Pre-registered analysis run, {ctx.label} ({kind}) [RL-{date}-NN]", "", source, ""]
    if ctx.label == LABEL_INTERIM:
        L += [f"**{NOTICE}**", ""]
    L += ["Numbers (each read from the named file, never typed):", ""]
    for a in arts:
        if a.computable and a.rows:
            L.append(f"- `{a.name}.json`: {a.title} — {len(a.rows)} rows.")
        else:
            L.append(f"- `{a.name}.json`: not computable — {a.reason or 'no rows'}.")
    return "\n".join(L) + "\n"


def render_report(ctx: Context, arts: Sequence[Artefact], date: str, call: str, blockers: Sequence[str]) -> str:
    n = ctx.n
    L = [f"# {n_label(ctx.label, n)} analysis {date}", ""]
    if ctx.label == LABEL_INTERIM:
        L += [f"**{NOTICE}**", ""]
    L += [(f"_Exact call: `{call}`. {n['runs']} completed run directories analysed, {n['unscorable']} unscorable, {n['skipped']} skipped. "
           f"Bootstrap B = {ctx.n_boot} (paired {ctx.paired_boot}), seed {ctx.seed}, clusters = `{CLUSTER_KEY}`, stratified by `{STRATUM_KEY}`._"), ""]
    if blockers:
        L += ["FINAL would be refused because:", *[f"- {b}" for b in blockers], ""]
    for a in arts:
        L += [f"## {n_label(ctx.label, n)}: {a.title}", ""]
        if not a.computable:
            L += [f"_not computable: {a.reason}_", ""]
        if a.tex_rows:
            L += ["| " + " | ".join(a.headers) + " |", "|" + "---|" * len(a.headers)]
            L += ["| " + " | ".join(c.replace("$^{*}$", "*").replace("$^{\\dagger}$", "†").replace("\\_", "_") for c in r) + " |" for r in a.tex_rows]
        else:
            L += ["| (no completed run) |" + " |" * (len(a.headers) - 1)]
        L.append("")
    L += [f"## {n_label(ctx.label, n)}: unscorable runs ({n['unscorable']})", "", "| run directory | units | reason |", "|---|---|---|"]
    L += [f"| `{Path(u['run_dir']).name}` | {', '.join(u['unit_ids']) or '-'} | {u['reason']} |" for u in ctx.unscorable] or ["| (none) | | |"]
    L += ["", f"## {n_label(ctx.label, n)}: skipped units and directories ({n['skipped']})", "", "| unit / directory | reason |", "|---|---|"]
    L += [f"| `{s.get('unit_id') or Path(s.get('run_dir') or '-').name}` | {s['reason']} |" for s in ctx.skipped] or ["| (none) | |"]
    if ctx.label == LABEL_INTERIM:
        L += ["", NOTICE]
    return "\n".join(L) + "\n"


def write_outputs(ctx: Context, arts: Sequence[Artefact], out_dir: Path, *, date: str, stamp: str, call: str, cfg_path: Path,
                  check: bool = False) -> int:
    """Write (or, with `check`, compare) every artefact's .tex and .json, analysis_summary.json, results_log_entry.md and
    analysis_report.md. Returns EXIT_OK, or EXIT_STALE when --check finds a differing or missing .tex body."""
    refuse_paper(out_dir)
    git = git_info()
    cfg_sha = sha256_file(cfg_path)
    n = ctx.n
    label_line = f"% {n_label(ctx.label, n)}" + (f"; {NOTICE}" if ctx.label == LABEL_INTERIM else "")
    inputs_desc = f"{len(ctx.runs)} run directories" + (f" and {ctx.paths['ledger']}" if ctx.ledger_rows is not None else " (no ledger)")
    stale = []
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)
    summary_arts = {}
    for a in arts:
        body = render_tabular(a, label_line)
        head = stamp_lines(inputs_desc, stamp, git, body)
        tex = "\n".join(head) + "\n" + body
        payload = {"label": ctx.label, "n": n, "date": date, "generated": stamp, "generator": GENERATOR, "git": git, "config_sha256": cfg_sha,
                   "title": a.title, "name": a.name, "computable": a.computable, "reason": a.reason, "columns": a.headers,
                   "rows": a.rows, "n_rows": len(a.rows), "content_sha256": sha256_text(body), "caption": a.caption,
                   "inputs": [i for r in ctx.runs for i in r.inputs], **a.extra}
        if ctx.label == LABEL_INTERIM:
            payload["notice"] = NOTICE
        tex_path, js_path = out_dir / f"{a.name}.tex", out_dir / f"{a.name}.json"
        summary_arts[a.name] = {"tex": tex_path.name, "json": js_path.name, "content_sha256": sha256_text(body), "computable": a.computable,
                                "n_rows": len(a.rows), "reason": a.reason}
        if check:
            if not tex_path.exists() or body_of(tex_path.read_text(encoding="utf-8")) != body_of(tex):
                stale.append(str(tex_path))
            continue
        tex_path.write_text(tex, encoding="utf-8")
        js_path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    blockers = final_blockers(ctx) if ctx.label == LABEL_INTERIM else []
    summary = {"label": ctx.label, "date": date, "generated": stamp, "generator": GENERATOR, "exact_call": call, "git": git,
               "config": {"path": str(cfg_path), "sha256": cfg_sha}, "n": n,
               "bootstrap": {"n_boot": ctx.n_boot, "paired_n_boot": ctx.paired_boot, "seed": ctx.seed, "level": ctx.level, "cluster": CLUSTER_KEY,
                             "strata": STRATUM_KEY, "bca_min_clusters": constants.BCA_MIN_CLUSTERS,
                             "small_cell_max_base_pairs": constants.SMALL_CELL_MAX_BASE_PAIRS,
                             "paired_t_min_base_pairs": constants.PAIRED_T_MIN_BASE_PAIRS, "alpha": constants.ALPHA},
               "inputs": [*(i for r in ctx.runs for i in r.inputs), *ctx.inputs], "runs": [r.as_dict() for r in ctx.runs],
               "analyses": analyses_status(ctx, arts), "artefacts": summary_arts, "unscorable": ctx.unscorable, "skipped": ctx.skipped,
               "final_blockers": blockers, "notes": ctx.notes, "modules": {"aggregate": AGG is not None, "pooling": PL is not None}}
    if ctx.label == LABEL_INTERIM:
        summary["notice"] = NOTICE
    if check:
        for name in ("analysis_summary.json", "results_log_entry.md", "analysis_report.md"):
            if not (out_dir / name).exists():
                stale.append(str(out_dir / name))
        for s in stale:
            print("stale:", s)
        return EXIT_STALE if stale else EXIT_OK
    (out_dir / "analysis_summary.json").write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "results_log_entry.md").write_text(results_log_entry(ctx, arts, date, call, git, cfg_sha, out_dir), encoding="utf-8")
    (out_dir / "analysis_report.md").write_text(render_report(ctx, arts, date, call, blockers), encoding="utf-8")
    return EXIT_OK


# ============================================================================ the driver
def build_context(cfg: dict, *, runs: Sequence[Path], ledger: Path | None, n_boot: int, paired_boot: int, seed: int,
                  include_smoke: bool = False, label: str = LABEL_INTERIM, audit_dir: Path | None = None) -> Context:
    paths = {k: resolve_path(cfg, k) for k in ("runs_root", "ledger", "models", "audit_dir", "human_report", "human_items", "gates", "freeze_check")}
    if audit_dir is not None:
        paths["audit_dir"] = Path(audit_dir)
    if runs:
        paths["runs_root"] = Path(runs[0]) if len(runs) == 1 and Path(runs[0]).is_dir() and not (Path(runs[0]) / "manifest.json").exists() else paths["runs_root"]
    if ledger is not None:
        paths["ledger"] = ledger
    models_cfg = None
    if paths["models"].exists():
        try:
            models_cfg = yaml.safe_load(paths["models"].read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            models_cfg = None
    midx = models_index(models_cfg)
    skipped: list[dict] = []
    ledger_rows = None
    if ledger is not None and ledger.exists():
        ledger_rows = read_ledger(ledger)
        refs = completed_from_ledger(ledger, paths["runs_root"], cfg, skipped)
    else:
        refs = completed_from_runs(list(runs) or [paths["runs_root"]], cfg, skipped, include_smoke=include_smoke)
    run_recs, df, unscorable = collect(refs, cfg, midx, paths["audit_dir"], skipped)
    level = float((cfg.get("bootstrap") or {}).get("level", 0.95))
    ctx = Context(cfg=cfg, df=df, runs=run_recs, unscorable=unscorable, skipped=skipped, n_boot=n_boot, paired_boot=paired_boot, seed=seed,
                  level=level, midx=midx, ledger_rows=ledger_rows, paths=paths, label=label)
    if ledger_rows is not None:
        ctx.inputs.append(_sha_record(ledger, "ledger"))
    for key in ("models",):
        if paths[key].exists():
            ctx.inputs.append(_sha_record(paths[key], key))
    return ctx


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog=f"python -m {GENERATOR}", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="configs/analysis.yaml (validated against noilai.constants)")
    ap.add_argument("--runs", type=Path, nargs="*", default=None, help="the runs root, or explicit run directories / manifest paths")
    ap.add_argument("--ledger", type=Path, default=None, help="experiments/ledger.csv (default: the config's path when --runs is not given)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: the config's out_dir; never under paper/)")
    ap.add_argument("--label", choices=(LABEL_INTERIM, LABEL_FINAL), default=LABEL_INTERIM,
                    help="FINAL is refused (exit 3) unless every requirement of labels.final_requires holds")
    ap.add_argument("--date", default=None, help="the date of the stamp and the entry, YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default: the config's B; also sets the paired B)")
    ap.add_argument("--seed", type=int, default=None, help="bootstrap seed (default: the config's)")
    ap.add_argument("--include-smoke", action="store_true", help="analyse smoke manifests too (exercise only; FINAL stays refused)")
    ap.add_argument("--audit-dir", type=Path, default=None, help="where the census and item audits live (default: the config's paths.audit_dir)")
    ap.add_argument("--check", action="store_true", help="recompute and compare every .tex body in --out (stamp lines ignored); exit 1 when stale")
    ap.add_argument("--quiet", action="store_true", help="do not print the Markdown report")
    args = ap.parse_args(argv)
    call = f"python -m {GENERATOR} " + " ".join(shlex.quote(a) for a in argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as e:
        ap.error(str(e))
    out_dir = args.out if args.out is not None else resolve_path(cfg, "out_dir")
    try:
        refuse_paper(out_dir)
    except ValueError as e:
        ap.error(str(e))
    if args.date is None:
        now = dt.datetime.now(dt.timezone.utc)
        date, stamp = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H:%MZ")
    else:
        try:
            date = dt.date.fromisoformat(args.date).isoformat()
        except ValueError:
            ap.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
        stamp = date
    bs = cfg.get("bootstrap") or {}
    n_boot = args.n_boot if args.n_boot is not None else int(bs.get("B", constants.BOOTSTRAP_B))
    paired = args.n_boot if args.n_boot is not None else int(bs.get("paired_B", 4000))
    seed = args.seed if args.seed is not None else int(bs.get("seed", 0))
    ledger = args.ledger if args.ledger is not None else (resolve_path(cfg, "ledger") if args.runs is None else None)
    if ledger is not None and not ledger.exists():
        print(f"ledger {ledger} does not exist: scanning the runs for finished runs", file=sys.stderr)
        ledger = None
    ctx = build_context(cfg, runs=args.runs or [], ledger=ledger, n_boot=n_boot, paired_boot=paired, seed=seed,
                        include_smoke=args.include_smoke, label=args.label, audit_dir=args.audit_dir)
    if args.label == LABEL_FINAL:
        blockers = final_blockers(ctx)
        if blockers:
            print("FINAL refused; the artefacts stay INTERIM until every requirement holds:", file=sys.stderr)
            for b in blockers:
                print(f"  - {b}", file=sys.stderr)
            return EXIT_FINAL_REFUSED
    arts = all_artefacts(ctx)
    code = write_outputs(ctx, arts, out_dir, date=date, stamp=stamp, call=call, cfg_path=args.config, check=args.check)
    if args.check:
        print("analysis outputs current" if code == EXIT_OK else "analysis outputs stale", file=sys.stderr)
        return code
    if not args.quiet:
        print((out_dir / "analysis_report.md").read_text(encoding="utf-8"))
    print(f"wrote {len(arts)} artefacts (.tex + .json), analysis_summary.json, results_log_entry.md and analysis_report.md to {out_dir}",
          file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
