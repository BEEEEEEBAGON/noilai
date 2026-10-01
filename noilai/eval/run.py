"""One evaluation run: model config x item file -> data/runs/<run_id>/{outputs.jsonl, manifest.json}.

    items, kind = load_item_file(path)                 # NóiLái jsonl (header record skipped) or XCOPA jsonl
    opts = RunOptions(tasks=..., variants=..., paraphrases=..., shots=..., arms=..., arm_scope=..., ...)
    run_dir = run(backend, entry, items, path, opts)

Selection (DESIGN_DECISIONS 4.5, 3.7, 12.32)
  * T1/T2/T3 items are filtered by `tasks` and `variants`; attested items by `attested_policy`
    (`exact2`, the default: exact two-syllable rows, the ones that enter T1/T3-style scoring
    and H6; approx and three-syllable rows are dropped and counted; `all` keeps every row).
    Attested V5/V6 rows render with their own variant templates.
  * `sample_n` draws a seeded stratified sample (noilai.eval.sample: per task x variant cell,
    core forced in, T3 pairs kept together, vulgar excluded); the ids, their SHA-256, the seed
    and the per-cell counts go into the manifest and `sample_ids.json`. It is SMOKE/PILOT-ONLY
    (the CLI refuses it without --smoke): every paper sub-sample is a seeded FILE written by
    scripts/sample_items.py, whose strata differ from this draw (DESIGN_DECISIONS 4.5, 12.32).
  * `limit` is a head truncation ("first N") and is SMOKE-ONLY: the CLI refuses it without
    --smoke because the first rows of a release file are one task.

Requests are the product item x paraphrase x shots x arm. A request whose rendered prompt is
byte-identical to the base-arm prompt (C2 on an unaffected item) is skipped and recorded in
`unchanged.jsonl` (DESIGN_DECISIONS 6.1); an arm that changes no prompt at all fails the run
before any generation. outputs.jsonl is appended after every batch and `--resume` skips
(item_id, arm, prompt_id) keys already present. manifest.json is written at start (status
running) and rewritten at the end with the fields of DESIGN_DECISIONS 7.5 (identity, engine,
data, sampling, hardware/time, outcome) plus the flat keys of docs/DATA_FORMAT.md.

Safety guards
  * the canary is never part of a prompt (render() asserts it) and the manifest records
    whether it appears in any output;
  * an API backend refuses an item file whose path contains validation / human / sealed, one
    flagged never_to_api (header, item or sibling manifest), non-core items unless
    allow_noncore_api (the core set and the public attested examples are API-eligible), and
    never sends an attested item whose `vulgar` flag is set (counted in the manifest);
  * a provider whose terms train on inputs, or whose terms are unknown (no explicit
    `trains_on_inputs: false` in the models file), never receives a core item
    (DESIGN_DECISIONS 11.2 / 12.41): run it on the dev-derived API set, or pass
    core_to_training_provider_opt_out for a key with a verified data-use opt-out (recorded);
  * the runner refuses to start a local (hf / vllm / llama_cpp) non-smoke run unless the entry's
    `revision` is a full commit hash, a Kaggle Models slug + version, or (llama_cpp) a GGUF
    SHA-256 (`check_revision`, DESIGN_DECISIONS 7.1); the accepted form is in the manifest;
  * a main run (not the reasoning sub-study, not a smoke run, allow_thinking unset) whose model
    emitted thinking text finishes with status `finished_thinking_present`, which the scorer
    refuses without --allow-thinking (DESIGN_DECISIONS 5.5 step 2 / 7.3: MUST be 0);
  * an item whose input / gold / reading / candidate EQUALS a demonstration phrase in either
    order is refused unless allow_demo_overlap (DESIGN_DECISIONS 7.4); items that merely share
    a SYLLABLE with the demonstrations or the instruction examples are flagged per row and
    listed in the manifest (`demo_overlap_policy` = flag, the default; drop; or refuse).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..audit.tokenizers import CENSUS_VERDICTS
from ..gen import variants as V
from ..vi import reencode as R
from . import prompts as P
from . import xcopa as X
from .backends import Backend, backend_versions, slug
from .extract import extract, t3_label
from .sample import DEFAULT_SEED, Sample, head_sample, stratified_sample

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs"
HASHES_FILE = ROOT / "data" / "HASHES.json"
ATTESTED_SEED = ROOT / "data" / "attested_seed.tsv"
AUDIT_DIR = ROOT / "data" / "audit"
FORBIDDEN_API_PATH_PARTS = ("validation", "human", "sealed")
DEMO_OVERLAP_POLICIES = ("flag", "drop", "refuse")
ATTESTED_POLICIES = ("exact2", "all")
LOCAL_WEIGHT_BACKENDS = ("hf", "vllm", "llama_cpp")      # backends that download weights: revision pinned (7.1)
REASONING_SUBSTUDY_GROUP = "reasoning_substudy"          # configs/models.yaml group whose runs may think
STATUS_FINISHED = "finished"
STATUS_THINKING = "finished_thinking_present"
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
_KAGGLE_SLUG = re.compile(r"[\w.-]+/[\w.-]+/[\w.-]+/[\w.-]+/\d+")      # owner/model/framework/variation/version
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ApiSafetyError(RuntimeError):
    pass


class RevisionError(ValueError):
    pass


def classify_revision(entry: dict, backend_kind: str | None) -> str | None:
    """The pinned form of the entry's model revision (DESIGN_DECISIONS 7.1): 'commit_hash' (40
    hex), 'kaggle_slug' (owner/model/framework/variation/version), 'gguf_sha256' (llama_cpp: a
    64-hex `gguf_sha256`), or None when nothing pins it."""
    rev = str(entry.get("revision") or "").strip()
    if _COMMIT_HASH.fullmatch(rev.lower()):
        return "commit_hash"
    if _KAGGLE_SLUG.fullmatch(rev):
        return "kaggle_slug"
    if backend_kind == "llama_cpp" and _SHA256.fullmatch(str(entry.get("gguf_sha256") or "").lower()):
        return "gguf_sha256"
    return None


def check_revision(entry: dict, backend_kind: str | None, smoke: bool = False) -> str | None:
    """Refuse a non-smoke run of a weight-downloading backend whose revision is not pinned
    (DESIGN_DECISIONS 7.1, item 67); returns the accepted form (None for other backends / smoke)."""
    kind = classify_revision(entry, backend_kind)
    if backend_kind in LOCAL_WEIGHT_BACKENDS and not smoke and kind is None:
        raise RevisionError(f"model entry {entry.get('name')!r} ({backend_kind}) has revision={entry.get('revision')!r}: "
                            "DESIGN_DECISIONS 7.1 requires a full 40-hex commit hash, a Kaggle Models slug + version "
                            "(owner/model/framework/variation/version) or, for llama_cpp, a 64-hex `gguf_sha256` before "
                            "any non-smoke run; pin it in configs/models.yaml or pass --smoke for a smoke test")
    return kind


class DemoOverlapError(RuntimeError):
    pass


class CorruptingCensusError(RuntimeError):
    """A re-encoding arm was requested on an engine path whose normalization census verdict is
    `corrupts` (decoding the ids does not return the input): the arm is refused on that engine
    (DESIGN_DECISIONS 6.2, PREREGISTRATION 5.5 / 8.6)."""


class UnchangedArmError(RuntimeError):
    pass


@dataclass
class RunOptions:
    tasks: tuple = ("T1", "T2", "T3", "attested")
    variants: tuple = V.VARIANTS              # T1/T2/T3 cells; attested rows are selected by attested_policy
    paraphrases: tuple = ("p0",)
    shots: tuple = (3,)
    arms: tuple = ("nfc",)
    arm_scope: str = P.DEFAULT_ARM_SCOPE      # whole_prompt (primary) or item (DESIGN_DECISIONS 6.1)
    limit: int = 0                            # smoke-only head truncation ("first N")
    sample_n: int = 0                         # seeded stratified sample size (0 = the whole selection)
    sample_seed: int = DEFAULT_SEED
    sample_exclude_vulgar: bool = True
    in_core_only: bool = False
    resume: bool = False
    allow_noncore_api: bool = False
    core_to_training_provider_opt_out: bool = False   # the API key has a verified data-use opt-out (11.2); recorded
    allow_thinking: bool = False              # thinking text tolerated (reasoning sub-study / smoke set it too)
    allow_demo_overlap: bool = False          # override the phrase-level refusal (dev pilots only)
    demo_overlap_policy: str = "flag"         # syllable-level overlap: flag | drop | refuse
    attested_policy: str = "exact2"           # exact2 | all
    max_new_tokens: int = 64
    batch_size: int = 8
    logprobs: bool = True                 # when the backend supports them (T3 forced choice, XCOPA)
    input_format: str = "raw"
    instruction: str = "explained"
    language: str = "vi"
    seed: int = 0
    run_id: str | None = None
    out_root: Path = RUNS_DIR
    system_prompt: str | None = None
    n_accelerators: int | None = None  # override for TPUs, which torch.cuda cannot count
    xcopa_logprob_mode: str = "choice"    # 'choice' (candidate continuations) or 'none'
    t1_forced_choice: bool = False        # EXPLORATORY T1 multi-distractor forced choice (noilai.eval.forced_choice)
    account_holder: str | None = None     # ROLE of the API / compute / hub account holder, never a name (11.2)
    require_census: bool = False          # fail when no normalization census exists for the model
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arms = tuple(P.normalize_arm(a) for a in self.arms)
        P.check_arm_scope(self.arm_scope)
        if self.demo_overlap_policy not in DEMO_OVERLAP_POLICIES:
            raise ValueError(f"demo_overlap_policy must be one of {DEMO_OVERLAP_POLICIES}")
        if self.attested_policy not in ATTESTED_POLICIES:
            raise ValueError(f"attested_policy must be one of {ATTESTED_POLICIES}")


# ------------------------------------------------------------------ items
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_header(record: dict) -> bool:
    return "_header" in record or "item_id" not in record


def load_item_file(path: Path) -> tuple[list[dict], str]:
    """(items, kind) with kind 'noilai' or 'xcopa'. The canary header record that v0.2 test
    and core files start with (docs/DATA_FORMAT.md) is skipped, as is any record without
    an item_id."""
    path = Path(path)
    if X.is_xcopa_file(path):
        return X.load_xcopa(path), "xcopa"
    items = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if not is_header(d):
                    items.append(d)
    return items, "noilai"


def read_header(path: Path) -> dict | None:
    """The header record of a release file (via noilai.gen.generate.read_header), or None."""
    with contextlib.suppress(Exception):
        from ..gen.generate import read_header as _rh
        return _rh(Path(path))
    return None


def read_canary(path: Path, items: list[dict]) -> str | None:
    header = read_header(path)
    if header and header.get("canary"):
        return header["canary"]
    return next((it["canary"] for it in items if it.get("canary")), None)


def is_vulgar(item: dict) -> bool:
    v = item.get("vulgar")
    return v is True or (isinstance(v, str) and v.strip().lower() in ("yes", "true", "1"))


def attested_eligible(item: dict, policy: str = "exact2") -> tuple[bool, str | None]:
    """(eligible, reason when not). `exact2`: exact two-syllable rows only (DESIGN_DECISIONS 3.7,
    4.7: approx rows never enter T1/T3-style scoring or H6; three-syllable rows keep their
    positions and are not rendered as T1)."""
    if policy == "all":
        return True, None
    n = item.get("n_syllables") or len(str(item.get("input", "")).split())
    if n != 2:
        return False, "three_syllable" if n == 3 else "not_two_syllables"
    exactness = str(item.get("exactness") or ("exact" if item.get("exact") else "unknown"))
    if item.get("exact") is False or exactness.startswith("approx"):
        return False, "approx"
    if item.get("variant") not in V.ALL_VARIANTS:
        return False, "unknown_variant"
    return True, None


def select_items(items: list[dict], opts: RunOptions, kind: str, report: dict | None = None) -> list[dict]:
    """The items a run scores, in a deterministic order. `report` (a dict, filled in place)
    receives the attested drops and the sample description."""
    report = report if report is not None else {}
    if kind == "xcopa":
        sel = list(items)
    else:
        sel = []
        att = {"n_total": 0, "n_selected": 0, "dropped": {}}
        for it in items:
            if is_header(it):
                continue
            if it.get("task") == "attested":
                if "attested" not in opts.tasks:
                    continue
                att["n_total"] += 1
                ok, reason = attested_eligible(it, opts.attested_policy)
                if not ok:
                    att["dropped"][reason] = att["dropped"].get(reason, 0) + 1
                    continue
                att["n_selected"] += 1
                sel.append(it)
            elif it.get("task") in opts.tasks and it.get("variant") in opts.variants:
                sel.append(it)
        if att["n_total"]:
            report["attested"] = {**att, "policy": opts.attested_policy}
    if opts.in_core_only:
        sel = [it for it in sel if it.get("in_core") or it.get("task") == "attested"]
    if opts.sample_n and kind != "xcopa":
        cells_tasks = tuple(t for t in opts.tasks if t in ("T1", "T2", "T3"))
        sample = stratified_sample(sel, opts.sample_n, seed=opts.sample_seed, tasks=cells_tasks, variants=opts.variants,
                                   exclude_vulgar=opts.sample_exclude_vulgar)
        attested = [it for it in sel if it.get("task") == "attested"]
        sel = sample.items + attested
        report["sample"] = sample.describe()
        report["_sample"] = sample
    elif opts.sample_n:
        sel = head_sample(sorted(sel, key=lambda x: x["item_id"]), opts.sample_n)
        report["sample"] = {"n": len(sel), "kind": "head", "note": "XCOPA has no cells; first N by item_id"}
    if opts.limit:
        sel = head_sample(sel, opts.limit)
        report["limit"] = {"n": opts.limit, "note": "smoke-only head truncation (first N)"}
    return sel


def _never_to_api(item_path: Path, items: list[dict]) -> str | None:
    """Why a file must never reach an API: a forbidden path part, or a never_to_api / sealed flag
    on the header, an item or the sibling release manifest."""
    parts = [p.lower() for p in Path(item_path).parts]
    for bad in FORBIDDEN_API_PATH_PARTS:
        if any(bad in p for p in parts):
            return f"path contains {bad!r}"
    header = read_header(item_path) or {}
    if header.get("never_to_api") or header.get("sealed"):
        return "header record is flagged never_to_api"
    if any(it.get("never_to_api") for it in items):
        return "an item is flagged never_to_api"
    man = Path(item_path).parent / "manifest.json"
    with contextlib.suppress(Exception):
        d = json.loads(man.read_text(encoding="utf-8"))
        if d.get("never_to_api") or d.get("sealed"):
            return "the release manifest is flagged never_to_api / sealed"
    return None


def check_api_safety(backend: Backend, items: list[dict], opts: RunOptions,
                     item_path: Path | None = None, privacy: dict | None = None) -> tuple[list[dict], dict]:
    """Enforce the API rules (DESIGN_DECISIONS 7.3, 11.2); returns (items to send, report).
    `privacy` is `provider_privacy(entry)`: a provider whose `trains_on_inputs` is not exactly
    False (True, a string, or unknown = not cleared) never receives a core item unless
    `opts.core_to_training_provider_opt_out` documents a key with a data-use opt-out."""
    privacy = privacy or {}
    report = {"is_api": backend.is_api, "n_excluded_vulgar": 0, "n_noncore": 0, "n_core": 0,
              "trains_on_inputs": privacy.get("trains_on_inputs"),
              "core_to_training_provider_opt_out": bool(opts.core_to_training_provider_opt_out)}
    if not backend.is_api:
        return items, report
    if item_path is not None:
        why = _never_to_api(item_path, items)
        if why:
            raise ApiSafetyError(f"{item_path} must never reach an API backend: {why}")
    # the core set and the (public, folk) attested examples may go to an API; nothing else
    noncore = [it for it in items if not (it.get("in_core") or it.get("task") == "attested"
                                          or it.get("source") == "attested")]
    report["n_noncore"] = len(noncore)
    if noncore and not opts.allow_noncore_api:
        raise ApiSafetyError(f"{len(noncore)} items are not in the core set; an API backend receives the core only "
                             f"(pass --allow-noncore-api to override for a dev-set pilot)")
    kept = []
    for it in items:
        if is_vulgar(it):
            report["n_excluded_vulgar"] += 1
            continue
        kept.append(it)
    core = [it for it in kept if it.get("in_core")]
    report["n_core"] = len(core)
    trains = privacy.get("trains_on_inputs")
    if core and trains is not False and not opts.core_to_training_provider_opt_out:
        prov = privacy.get("provider")
        state = "trains on inputs" if trains is True else f"has no cleared no-training terms (trains_on_inputs={trains!r})"
        raise ApiSafetyError(f"{len(core)} core items would reach provider {prov!r}, which {state}: a provider whose "
                             "tier trains on inputs never receives the core (DESIGN_DECISIONS 11.2 / 12.41, item 33); "
                             "run the dev-derived API set (noilai_api_dev, 4.5) instead, set `terms.trains_on_inputs: "
                             "false` for a provider with no-training terms, or pass "
                             "--core-to-training-provider-opt-out for a paid key with a verified data-use opt-out")
    return kept, report


def check_demo_overlap(items: list[dict], opts: RunOptions, kind: str) -> dict:
    """Syllable-level overlap {syllable: [item_id, ...]} between the demonstrations / instruction
    examples and the items (recorded, then handled by `apply_demo_policy`). Raises when an item
    EQUALS a demonstration phrase in either order (the 7.4 exclusion rule) unless
    allow_demo_overlap, or on any overlap under demo_overlap_policy='refuse'."""
    if kind == "xcopa":
        return {}
    demos = P.default_demos()
    phrases = P.demo_phrase_overlap(demos, items)
    overlap = P.demo_overlap(demos, items)
    if phrases and not opts.allow_demo_overlap:
        raise DemoOverlapError("items equal to a demonstration phrase (either order) are in the item file: "
                               + ", ".join(f"{ph} ({', '.join(ids)})" for ph, ids in phrases.items())
                               + "; the demonstrations are the H6 control and must not be scored (DESIGN_DECISIONS "
                               "7.4); regenerate with these pairs reserved or pass --allow-demo-overlap for a pilot")
    if overlap and opts.demo_overlap_policy == "refuse" and not opts.allow_demo_overlap:
        raise DemoOverlapError("demonstration/instruction syllables occur in the item file: "
                               + ", ".join(f"{s} ({len(ids)} items)" for s, ids in overlap.items())
                               + "; regenerate with these syllables reserved, pass --allow-demo-overlap, or choose "
                               "--demo-overlap-policy flag|drop")
    return overlap


def apply_demo_policy(items: list[dict], overlap: dict, opts: RunOptions) -> tuple[list[dict], dict]:
    """Drop or flag the items sharing a syllable with the demonstrations. Returns (items, report)."""
    ids = sorted({i for lst in overlap.values() for i in lst})
    report = {"policy": opts.demo_overlap_policy, "n_items": len(ids), "item_ids": ids,
              "phrases": P.demo_phrase_overlap(P.default_demos(), items) if items and items[0].get("task") != X.XCOPA_TASK
              else {}}
    if opts.demo_overlap_policy == "drop" and ids:
        idset = set(ids)
        items = [it for it in items if it["item_id"] not in idset]
        report["n_dropped"] = len(ids)
    return items, report


# ------------------------------------------------------------------ requests
@dataclass
class Request:
    item: dict
    arm: str
    paraphrase: str
    shots: int
    prompt_id: str
    unchanged: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.item["item_id"], self.arm, self.prompt_id)


def plan_requests(items: list[dict], opts: RunOptions, kind: str) -> list[Request]:
    reqs = []
    for it in items:
        for p in opts.paraphrases:
            for s in (opts.shots if kind == "noilai" else (0,)):
                for a in opts.arms:
                    pid = (P.prompt_id(p, s, opts.instruction, opts.input_format, opts.language, opts.arm_scope)
                           if kind == "noilai" else f"xcopa-{p}" + ("" if opts.arm_scope == P.DEFAULT_ARM_SCOPE
                                                                  else f"-{opts.arm_scope}"))
                    reqs.append(Request(it, a, p, s, pid))
    return reqs


def render_request(req: Request, opts: RunOptions, kind: str, arm: str | None = None) -> list[dict]:
    arm = arm or req.arm
    if kind == "xcopa":
        return X.render_xcopa(req.item, req.paraphrase, arm, system=opts.system_prompt, arm_scope=opts.arm_scope)
    return P.render(req.item, paraphrase=req.paraphrase, shots=req.shots, arm=arm,
                    input_format=opts.input_format, instruction=opts.instruction, language=opts.language,
                    system=opts.system_prompt, arm_scope=opts.arm_scope)


def mark_unchanged(reqs: list[Request], opts: RunOptions, kind: str) -> tuple[list[Request], list[Request]]:
    """Split the requests into (to run, unchanged): a non-base arm whose rendered prompt is
    byte-identical to the base arm's prompt for the same item is skipped (DESIGN_DECISIONS
    6.1). Raises UnchangedArmError when an arm changes no prompt at all (the C2 assertion of
    item 49), before any generation."""
    todo, unchanged = [], []
    changed_per_arm: dict[str, int] = {a: 0 for a in opts.arms if a != P.BASE_ARM}
    total_per_arm: dict[str, int] = {a: 0 for a in opts.arms if a != P.BASE_ARM}
    for r in reqs:
        if r.arm == P.BASE_ARM:
            todo.append(r)
            continue
        total_per_arm[r.arm] += 1
        base = render_request(r, opts, kind, arm=P.BASE_ARM)
        cur = render_request(r, opts, kind)
        if P.messages_text(base) == P.messages_text(cur):
            r.unchanged = True
            unchanged.append(r)
        else:
            changed_per_arm[r.arm] += 1
            todo.append(r)
    dead = [a for a, n in total_per_arm.items() if n and changed_per_arm[a] == 0]
    if dead:
        raise UnchangedArmError(f"arm(s) {dead} change no prompt on this item file under arm_scope="
                                f"{opts.arm_scope!r}: the column would be empty (DESIGN_DECISIONS 6.1, item 49); "
                                f"for C2 the baseline is old style, so the arm must be placement_new")
    return todo, unchanged


def existing_keys(outputs_path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    d = json.loads(ln)
                    if "item_id" in d:
                        keys.add((d["item_id"], d["arm"], d["prompt_id"]))
    return keys


# ------------------------------------------------------------------ logprobs
def _marker_for(arm: str, arm_scope: str = P.DEFAULT_ARM_SCOPE) -> str:
    return P.marker_for(arm, arm_scope)


def t3_logprobs(backend: Backend, req: Request, messages: list[dict], opts: RunOptions) -> dict:
    """Answer-token log-probabilities P(Có), P(Không) after the rendered T3 prompt, and the
    candidate's string log-probability under a T1 context for the same input and variant
    (shared by the yes/no pair -> BLiMP-style paired scoring). Under a strip arm the candidate
    log-probability is None: tone twins become identical strings once stripped, so the pair
    comparison is undefined there (only the yes/no answer tokens are scored).

    Boundary convention (DESIGN_DECISIONS 7.3, item 55): the scoring context ends with the
    answer marker ("Đáp án:") and every continuation carries its leading space (" Có",
    " Không", " " + candidate), the piece the model itself produces after the marker and the
    convention of xcopa.completion_pair. A context ending in "Đáp án: " would merge the space
    into the first continuation token on SentencePiece and byte-level BPE tokenizers alike and
    violate the prefix property on every row. A violated continuation is recorded as None; the
    continuation token counts are recorded for the per-token means (5.3 item 40)."""
    out: dict = {}
    arm, scope = req.arm, opts.arm_scope
    encode_marker = R.meaning_preserving(arm) and scope == "whole_prompt"
    prompt_text = backend.chat_to_text(messages) + _marker_for(arm, scope)
    conts = [" " + (R.reencode(w, arm) if encode_marker else w) for w in ("Có", "Không")]
    lp = backend.logprobs(prompt_text, conts)
    viol = _violations(backend, 2)
    n_tok = _n_tokens(backend, 2)
    out["Có"], out["Không"] = _lp(lp[0], viol[0]), _lp(lp[1], viol[1])
    out["n_tokens_Có"], out["n_tokens_Không"] = n_tok
    out["prompt_ids_sha256"] = getattr(backend, "last_prompt_ids_sha256", None)
    # per continuation (DESIGN_DECISIONS 7.3): the scorer nulls only the violated continuation's
    # prediction; the row-level flag is the any() of them, for the manifest count
    out["prefix_property_violations"] = {"Có": viol[0], "Không": viol[1], "candidate": None}
    out["prefix_property_violation"] = any(viol) if any(v is not None for v in viol) else None
    it = req.item
    out["candidate"] = None
    out["candidate_n_tokens"] = None
    out["candidate_context_hash"] = None
    if arm in P.STRIP_ARMS:
        out["candidate_skipped_reason"] = "strip arm: stripped twins are not distinct strings"
        return out
    pseudo = {"task": "T1", "variant": it["variant"], "input": it["input"], "item_id": it["item_id"],
              "canary": it.get("canary"), "gold": [it.get("correct_output", "")]}
    ctx_messages = P.render(pseudo, paraphrase=req.paraphrase, shots=req.shots, arm=arm, input_format=opts.input_format,
                            instruction=opts.instruction, language=opts.language, system=opts.system_prompt,
                            arm_scope=scope)
    ctx = backend.chat_to_text(ctx_messages) + _marker_for(arm, scope)
    cand = " " + R.reencode(it["candidate"], arm)
    lp2 = backend.logprobs(ctx, [cand])
    viol2 = _violations(backend, 1)
    out["candidate"] = _lp(lp2[0], viol2[0])
    out["candidate_n_tokens"] = _n_tokens(backend, 1)[0]
    out["candidate_context_ids_sha256"] = getattr(backend, "last_prompt_ids_sha256", None)
    out["prefix_property_violations"]["candidate"] = viol2[0]
    if viol2[0]:
        out["prefix_property_violation"] = True
    out["candidate_context_hash"] = P.prompt_hash(ctx_messages)
    return out


def t1_forced_choice(backend: Backend, req: Request, messages: list[dict], opts: RunOptions) -> dict:
    """EXPLORATORY (docs/FORCED_CHOICE_EXPLORATORY.md): the summed log-probability of the gold and of the rule-built
    near misses as continuations of the rendered T1 prompt + answer marker, the boundary convention of t3_logprobs."""
    from noilai.eval import forced_choice as FC
    arm, scope = req.arm, opts.arm_scope
    encode = R.meaning_preserving(arm) and scope == "whole_prompt"
    context = backend.chat_to_text(messages) + _marker_for(arm, scope)
    return FC.score_candidates(backend, context, FC.t1_candidates(req.item), arm, encode)


def _violations(backend: Backend, n: int) -> list[bool | None]:
    v = list(getattr(backend, "last_prefix_violations", []) or [])
    return [bool(x) for x in v] if len(v) == n else [None] * n


def _n_tokens(backend: Backend, n: int) -> list[int | None]:
    v = list(getattr(backend, "last_n_tokens", []) or [])
    return [int(x) for x in v] if len(v) == n else [None] * n


def _lp(value, violated) -> float | None:
    """A continuation's summed log-probability, or None when its prefix property was violated
    (asserted, not degraded: DESIGN_DECISIONS 7.3) or the backend returned no number."""
    if violated or value is None:
        return None
    value = float(value)
    if math.isnan(value) and violated is None:      # a backend without violation flags signalled it by NaN
        return None
    return value                                    # a NaN with violated=False is unexpected and is counted by run()


def xcopa_logprobs(backend: Backend, req: Request) -> dict:
    ctx, conts = X.completion_pair(req.item, req.arm)
    lp = backend.logprobs(ctx, conts)
    viol = _violations(backend, 2)
    n_tok = _n_tokens(backend, 2)
    return {"1": _lp(lp[0], viol[0]), "2": _lp(lp[1], viol[1]), "n_tokens_1": n_tok[0], "n_tokens_2": n_tok[1],
            "context": ctx, "prompt_ids_sha256": getattr(backend, "last_prompt_ids_sha256", None),
            "prefix_property_violations": {"1": viol[0], "2": viol[1]},
            "prefix_property_violation": any(viol) if any(v is not None for v in viol) else None}


# ------------------------------------------------------------------ manifest helpers
def git_info() -> dict:
    def _run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    commit = _run("rev-parse", "HEAD")
    dirty = _run("status", "--porcelain", "--", str(ROOT))
    return {"git_commit": commit or "unknown", "git_dirty": (bool(dirty) if dirty is not None else None)}


def hardware_info(n_override: int | None = None) -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "python_version": sys.version.split()[0],
            "cpu": platform.processor() or None, "gpus": [], "n_gpus": 0, "cuda": None, "cuda_version": None,
            "driver_version": None, "torch_version": None}
    with contextlib.suppress(Exception):      # no torch, or a broken CUDA install: report no GPU
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["n_gpus"] = torch.cuda.device_count()
            info["cuda"] = info["cuda_version"] = torch.version.cuda
    with contextlib.suppress(Exception):
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
        info["driver_version"] = out.strip().splitlines()[0].strip() if out.strip() else None
    if n_override is not None:
        info["n_gpus"] = int(n_override)
        info["n_accelerators_override"] = int(n_override)
    return info


def resource_hashes() -> dict:
    if HASHES_FILE.exists():
        return json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    return {}


def code_sha256() -> str:
    """One hash over the harness code the run depends on (noilai/**/*.py, prompts/*.yaml, the
    two CLIs), file names included, sorted."""
    h = hashlib.sha256()
    files = sorted(list((ROOT / "noilai").rglob("*.py")) + list((ROOT / "prompts").glob("*.yaml"))
                   + [ROOT / "scripts" / "run_eval.py", ROOT / "scripts" / "score_run.py"])
    for f in files:
        if f.exists() and "__pycache__" not in f.parts:
            h.update(str(f.relative_to(ROOT)).encode("utf-8") + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def pip_freeze() -> tuple[str, str]:
    """(freeze text, sha256) from importlib.metadata; the text is written next to the manifest."""
    try:
        from importlib import metadata
        lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]},
                       key=str.lower)
    except Exception:
        lines = []
    text = "\n".join(lines) + ("\n" if lines else "")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_content_sha256(items: list[dict]) -> str:
    """The dataset's content hash: items sorted by item_id with the canary keys removed
    (DESIGN_DECISIONS 4.6)."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.get("item_id", "")):
        d = {k: v for k, v in it.items() if k not in ("canary", "do_not_train", "evaluation_only")}
        h.update(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")
    return h.hexdigest()


def placement_baseline_for(item_path: Path, items: list[dict]) -> dict:
    """The placement convention of the stored items: from the sibling release manifest
    (`placement_style`), else measured on the inputs (old if no input changes under
    `placement_old`)."""
    man = Path(item_path).parent / "manifest.json"
    with contextlib.suppress(Exception):
        d = json.loads(man.read_text(encoding="utf-8"))
        if d.get("placement_style"):
            return {"placement_baseline": d["placement_style"], "source": str(man)}
    texts = [it.get("input", "") for it in items if isinstance(it.get("input"), str)]
    changed_old = sum(1 for t in texts if R.reencode(t, "placement_old") != R.to_nfc(t))
    changed_new = sum(1 for t in texts if R.reencode(t, "placement_new") != R.to_nfc(t))
    if changed_old == 0 and changed_new == 0:
        style = "undetermined"          # no C2-affected input: both conventions spell every input the same way
    elif changed_old == 0:
        style = "old"
    elif changed_new == 0:
        style = "new"
    else:
        style = "mixed"
    return {"placement_baseline": style if texts else None, "source": "measured", "n_inputs_changed_by_old": changed_old,
            "n_inputs_changed_by_new": changed_new}


# which census verdict each re-encoding arm depends on (DESIGN_DECISIONS 6.2: NFD = the C1 arm, PC = the
# partial Windows-1258 decomposition); the strip and placement arms produce NFC strings and need no verdict
CENSUS_ARM_VERDICTS = {"nfd": "verdict_nfd", "win1258": "verdict_pc"}


def census_classification(census: dict) -> tuple[str | None, bool]:
    """(classification, legacy) from a census block. The three-valued `verdict_nfd` of
    noilai.audit.tokenizers.normalization_census is authoritative; a census written before the
    verdicts existed carries only the two-valued `normalizes_nfd`, which cannot tell a pass-through
    from a corrupting path -- it is read as a fallback and flagged `legacy_census` in the manifest."""
    if not census:
        return None, False
    verdict = census.get("verdict_nfd")
    if verdict in CENSUS_VERDICTS:
        return verdict, False
    return ("normalizes" if census.get("normalizes_nfd") else "passes_through"), True


def normalization_census_for(entry: dict, backend_info: dict) -> dict:
    """The tokenizer normalization census (DESIGN_DECISIONS 6.2) recorded for this model: read
    from data/audit/<name>.json (entry `tokenizer_audit` or `family`), else marked absent.
    `classification` is the NFD verdict (normalizes | passes_through | corrupts), `verdict_pc` the
    PC arm's; `legacy_census` is True when only the two-valued `normalizes_nfd` was available."""
    names = [entry.get("tokenizer_audit"), entry.get("family"), entry.get("name")]
    for name in names:
        if not name:
            continue
        p = AUDIT_DIR / f"{name}.json"
        if p.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError, AttributeError):   # unreadable file: not censused
                d = json.loads(p.read_text(encoding="utf-8"))
                census = d.get("normalization_census") or {}
                cls, legacy = census_classification(census)
                verdict_pc = census.get("verdict_pc") if census.get("verdict_pc") in CENSUS_VERDICTS else None
                source = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                return {"status": "censused", "source": source, "classification": cls,
                        "engine": backend_info.get("engine") or backend_info.get("backend"),
                        "legacy_census": legacy, **census, "verdict_nfd": cls, "verdict_pc": verdict_pc}
    return {"status": "not_censused", "source": None, "classification": None, "legacy_census": False,
            "note": "no data/audit/<family>.json for this model; run scripts/audit_tokenizers.py before GPU time (6.2)"}


def check_census_arms(census: dict, arms, engine=None) -> None:
    """DESIGN_DECISIONS 6.2 / PREREGISTRATION 5.5: an arm whose census verdict on this engine path is
    `corrupts` is refused (raises CorruptingCensusError), because the string the model would receive is
    not the re-encoded input. A missing verdict (uncensused model, legacy census, strip or placement
    arm) refuses nothing; `require_census` is the switch for that."""
    if not census or census.get("status") != "censused":
        return
    bad = {a: census.get(CENSUS_ARM_VERDICTS[a]) for a in arms
           if a in CENSUS_ARM_VERDICTS and census.get(CENSUS_ARM_VERDICTS[a]) == "corrupts"}
    if bad:
        raise CorruptingCensusError(
            f"arm(s) {sorted(bad)} refused: the normalization census ({census.get('source')}) says the tokenizer path "
            f"CORRUPTS the re-encoded form (decoding the ids does not return the input) on engine {engine!r}; "
            "the arm is refused on this engine and reported (DESIGN_DECISIONS 6.2, PREREGISTRATION 5.5)")


def provider_privacy(entry: dict) -> dict:
    """What the provider does with inputs (models.yaml `providers` block) plus the request-side
    privacy settings (`extra_body`), for `api_privacy_settings`."""
    prov = entry.get("provider")
    out = {"provider": prov, "extra_body": entry.get("extra_body"), "terms": None, "data_policy": None,
           "trains_on_inputs": None}
    if "provider_terms" in entry:
        # copied from the models file the entry was read from (backends.get_model_entry), or set
        # directly on an in-memory entry (tests, pilots): the entry's own statement wins
        terms = entry.get("provider_terms") or {}
        out["terms"] = entry.get("provider_terms")
        out["data_policy"] = entry.get("provider_data_policy")
        out["trains_on_inputs"] = terms.get("trains_on_inputs") if isinstance(terms, dict) else None
        return out
    if prov:
        with contextlib.suppress(Exception):
            from .backends import MODELS_FILE, load_models_config
            cfg = load_models_config(MODELS_FILE)
            block = (cfg.get("providers") or {}).get(prov) or {}
            out["terms"] = block.get("terms")
            out["data_policy"] = block.get("data_policy")
            out["trains_on_inputs"] = (block.get("terms") or {}).get("trains_on_inputs")
    return out


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, frozenset):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, Sample):
        return x.describe()
    return x


def write_manifest(run_dir: Path, manifest: dict) -> None:
    tmp = run_dir / "manifest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)
    os.replace(tmp, run_dir / "manifest.json")


def default_run_id(entry: dict, item_path: Path) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slug(entry.get('name') or entry.get('model_id') or 'model')}__{slug(Path(item_path).stem)}__{stamp}"


def _item_text(item: dict, kind: str) -> str:
    if kind == "xcopa":
        return " ".join(str(item.get(k, "")) for k in ("premise", "choice1", "choice2"))
    return " ".join(x for x in (item.get("input"), item.get("candidate")) if x)


# ------------------------------------------------------------------ the run
def run(backend: Backend, entry: dict, items: list[dict], item_path: Path, opts: RunOptions,
        kind: str | None = None, log=print) -> Path:
    item_path = Path(item_path)
    items = [it for it in items if not is_header(it)]
    kind = kind or ("xcopa" if items and items[0].get("task") == X.XCOPA_TASK else "noilai")
    if kind == "noilai" and opts.input_format != "raw" and any(a in P.STRIP_ARMS for a in opts.arms):
        raise ValueError("strip arms are defined on the raw input format only")
    selection: dict = {}
    selected = select_items(items, opts, kind, selection)
    sample_obj: Sample | None = selection.pop("_sample", None)
    privacy = provider_privacy(entry)
    selected, api_report = check_api_safety(backend, selected, opts, item_path, privacy)
    overlap = check_demo_overlap(selected, opts, kind)
    selected, demo_report = apply_demo_policy(selected, overlap, opts)
    flagged_ids = set(demo_report.get("item_ids", [])) if opts.demo_overlap_policy == "flag" else set()
    canary = read_canary(item_path, items)
    backend_info = backend.info()
    census = normalization_census_for(entry, backend_info)
    if opts.require_census and census["status"] != "censused":
        raise RuntimeError(census["note"])
    check_census_arms(census, opts.arms, engine=census.get("engine"))

    run_id = opts.run_id or default_run_id(entry, item_path)
    run_dir = Path(opts.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = run_dir / "outputs.jsonl"
    done = existing_keys(outputs_path) if opts.resume else set()
    if outputs_path.exists() and not opts.resume:
        raise FileExistsError(f"{outputs_path} exists; pass --resume to continue it or choose another --run-id")

    reqs = plan_requests(selected, opts, kind)
    runnable, unchanged = mark_unchanged(reqs, opts, kind)
    with open(run_dir / "unchanged.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"item_id": r.item["item_id"], "arm": r.arm, "prompt_id": r.prompt_id,
                                 "identical_to": {"arm": P.BASE_ARM, "prompt_id": r.prompt_id}}, ensure_ascii=False) + "\n"
                     for r in unchanged)
    n_changed = {a: sum(1 for r in runnable if r.arm == a) for a in opts.arms if a != P.BASE_ARM}
    n_unchanged = {a: sum(1 for r in unchanged if r.arm == a) for a in opts.arms if a != P.BASE_ARM}
    todo = [r for r in runnable if r.key not in done]
    if sample_obj is not None:
        with open(run_dir / "sample_ids.json", "w", encoding="utf-8") as f:
            json.dump({"seed": sample_obj.seed, "n": len(sample_obj.items), "ids_sha256": sample_obj.ids_sha256(),
                       "item_ids": sample_obj.item_ids}, f, ensure_ascii=False, indent=0)
    freeze_text, freeze_sha = pip_freeze()
    (run_dir / "pip_freeze.txt").write_text(freeze_text, encoding="utf-8")
    t_start = time.monotonic()
    started = _now()
    demos_hash = hashlib.sha256(json.dumps(_jsonable({f"{k[0]}-{k[1]}": v for k, v in P.default_demos().items()}),
                                           ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() \
        if kind == "noilai" else None
    account_holder = (opts.account_holder or entry.get("account_holder") or os.environ.get("NOILAI_ACCOUNT_HOLDER_ROLE")
                      or opts.notes.get("account_holder"))
    is_tpu = str(entry.get("hardware") or "").lower() == "tpu"
    smoke = bool(opts.notes.get("smoke"))
    thinking_allowed = bool(opts.allow_thinking or smoke or entry.get("group") == REASONING_SUBSTUDY_GROUP)
    backend_kind = backend_info.get("backend") or entry.get("backend")
    manifest = {
        "run_id": run_id, "status": "running", "started_utc": started, "finished_utc": None,
        "start_utc": started, "end_utc": None,          # DESIGN_DECISIONS 7.5 names; started_utc/finished_utc are aliases
        "model": {"name": entry.get("name"), "model_id": entry.get("hf_id") or entry.get("model_id")
                  or entry.get("provider_model_id"), "revision": entry.get("revision"),
                  "provider": entry.get("provider"), "provider_model_id": entry.get("provider_model_id"),
                  "hardware_entry": entry.get("hardware"), "config_entry": entry},
        "backend": backend_info, "backend_versions": backend_versions(),
        "seed": opts.seed, "options": asdict(opts),
        "prompt_files_sha256": P.prompt_file_hashes(),
        "item_file": {"path": str(item_path), "sha256": sha256_file(item_path), "content_sha256": item_content_sha256(items),
                      "kind": kind, "n_items": len(items), "n_selected": len(selected),
                      "canary_present": canary is not None,
                      "canary_sha256": hashlib.sha256(canary.encode()).hexdigest() if canary else None,
                      "header": {k: v for k, v in (read_header(item_path) or {}).items() if k != "canary"} or None},
        "selection": selection, "api_safety": api_report, "demo_overlap": overlap, "demo_overlap_report": demo_report,
        "n_requests": len(reqs), "n_unchanged": len(unchanged), "n_resumed": len(runnable) - len(todo), "n_written": 0,
        "hardware": {**hardware_info(opts.n_accelerators), "accelerator_kind": "tpu" if is_tpu else "gpu",
                     "kaggle_kernel": kaggle_kernel_info()},
        "resource_sha256": resource_hashes(), **git_info(),
        "code_sha256": code_sha256(),
        "gpu_hours": None, "tpu_hours": None, "wall_s": None, "canary_check": None,
        # DESIGN_DECISIONS 7.5, grouped
        "identity": {"run_id": run_id, "model_key": entry.get("name"),
                     "model_id": entry.get("hf_id") or entry.get("model_id") or entry.get("provider_model_id"),
                     "revision": entry.get("revision"), "revision_kind": classify_revision(entry, backend_kind),
                     "tokenizer_sha256": backend_info.get("tokenizer_sha256"),
                     "tokenizer_sha256_source": backend_info.get("tokenizer_sha256_source"),
                     "chat_template_sha256": backend_info.get("chat_template_sha256"),
                     "template_kwargs": (backend_info.get("engine_flags") or {}).get("chat_template_kwargs"),
                     "thinking_knob_detected": backend_info.get("thinking_knob_detected"),
                     "thinking_knobs": backend_info.get("thinking_knobs")},
        "engine": {"engine": backend_info.get("engine") or backend_info.get("backend"),
                   "engine_version": backend_info.get("backend_version"),
                   "torch_version": backend_info.get("torch_version") or backend_versions().get("torch"),
                   "cuda_version": None, "driver_version": None, "python_version": sys.version.split()[0],
                   "pip_freeze_sha256": freeze_sha, "pip_freeze_file": "pip_freeze.txt",
                   "engine_args": backend_info.get("engine_flags"), "dtype_requested": backend_info.get("dtype_requested")
                   or entry.get("dtype"), "dtype_resolved": backend_info.get("dtype"),
                   "quantization": backend_info.get("quantization"),
                   "attention_backend": (backend_info.get("engine_flags") or {}).get("attention_backend")
                   or entry.get("attention_backend"), "tensor_parallel_size": entry.get("tensor_parallel_size")},
        "data": {"item_file": str(item_path), "item_file_sha256": None, "item_content_sha256": None,
                 "attested_list_sha256": sha256_file(ATTESTED_SEED) if ATTESTED_SEED.exists() else None,
                 "prompt_file_sha256": None, "demo_hashes": demos_hash, "arms": list(opts.arms), "arm_scope": opts.arm_scope,
                 "prompt_ids": sorted({r.prompt_id for r in reqs}), "n_items": len(selected), "canary_check": None,
                 "resource_hashes": resource_hashes(), "normalization_census": census,
                 **placement_baseline_for(item_path, items), "n_changed_prompts": n_changed,
                 "n_unchanged_prompts": n_unchanged, "prefix_property_violations": 0,
                 "prefix_property_violations_by_continuation": {},
                 "sample": selection.get("sample"), "demo_overlap_item_ids": demo_report.get("item_ids", [])},
        "sampling": {"temperature": 0.0, "max_tokens": opts.max_new_tokens, "seed": opts.seed,
                     "stop": getattr(backend, "stop", None),      # the stop list handed to the backend, if it has one
                     "greedy": True, "logprobs_mode": ("t3_yesno+t3_pair+xcopa_choice" if opts.logprobs and
                                                      backend.supports_logprobs else "none")},
        "outcome": {"n_done": 0, "n_unparseable": None, "n_truncated": None, "n_thinking_chars_total": None,
                    "thinking_allowed": thinking_allowed, "smoke": smoke,
                    "n_hedged": None, "nan_inf_events": 0, "api_model_echo": [], "rate_limit_events": None,
                    "account_holder": account_holder if account_holder else ("unknown" if backend.is_api else None),
                    "api_account_holder": account_holder if account_holder else ("unknown" if backend.is_api else None),
                    "api_account_holder_missing": bool(backend.is_api and not account_holder),
                    "api_privacy_settings": provider_privacy(entry) if backend.is_api else None,
                    "prefill_tok_s": None, "decode_tok_s": None},
    }
    manifest["data"]["item_file_sha256"] = manifest["item_file"]["sha256"]
    manifest["data"]["item_content_sha256"] = manifest["item_file"]["content_sha256"]
    manifest["data"]["prompt_file_sha256"] = manifest["prompt_files_sha256"]
    manifest["engine"]["cuda_version"] = manifest["hardware"].get("cuda_version")
    manifest["engine"]["driver_version"] = manifest["hardware"].get("driver_version")
    write_manifest(run_dir, manifest)
    log(f"[run {run_id}] {len(todo)} requests to do ({len(runnable) - len(todo)} resumed, {len(unchanged)} unchanged) "
        f"-> {outputs_path}")

    use_lp = opts.logprobs and backend.supports_logprobs
    n_written = 0
    nan_inf = 0
    canary_in_prompts = False        # computed over every rendered prompt of this run, never assumed
    model_echo: set = set()
    base_tokens: dict[tuple[str, str], int | None] = {}
    bs = max(1, opts.batch_size)
    with open(outputs_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), bs):
            batch = todo[start: start + bs]
            messages_batch = [render_request(r, opts, kind) for r in batch]
            if canary and any(canary in P.messages_text(m) for m in messages_batch):
                canary_in_prompts = True
            t0 = time.monotonic()
            gens = backend.generate(messages_batch, max_new_tokens=opts.max_new_tokens, greedy=True)
            batch_latency = time.monotonic() - t0
            for req, msgs, g in zip(batch, messages_batch, gens):
                raw = g.get("text", "")
                ex = extract(raw, req.item["task"])
                reasoning = g.get("reasoning")
                n_think = ex.n_thinking_chars + (len(reasoning) if isinstance(reasoning, str) else 0)
                n_out = g.get("n_output_tokens")
                finish = g.get("finish_reason")
                if finish is None and n_out is not None:
                    finish = "length" if n_out >= opts.max_new_tokens else "stop"
                truncated = (finish == "length") if finish is not None else (n_out is not None and n_out >= opts.max_new_tokens)
                templated = backend.chat_to_text(msgs)
                n_tok = g.get("n_prompt_tokens")
                key = (req.item["item_id"], req.prompt_id)
                if req.arm == P.BASE_ARM:
                    base_tokens.setdefault(key, n_tok if n_tok is not None else backend.count_tokens(templated))
                    n_base = base_tokens[key]
                else:
                    if key not in base_tokens:
                        base_tokens[key] = backend.count_tokens(backend.chat_to_text(render_request(req, opts, kind,
                                                                                                  arm=P.BASE_ARM)))
                    n_base = base_tokens[key]
                    if n_tok is None:
                        n_tok = backend.count_tokens(templated)
                row = {
                    "item_id": req.item["item_id"], "task": req.item["task"], "variant": req.item["variant"],
                    "arm": req.arm, "arm_scope": opts.arm_scope, "prompt_id": req.prompt_id, "paraphrase": req.paraphrase,
                    "shots": req.shots, "prompt_hash": P.prompt_hash(msgs),
                    "templated_prompt_hash": hashlib.sha256(templated.encode("utf-8")).hexdigest(),
                    "prompt_ids_sha256": g.get("prompt_ids_sha256"),      # the id sequence the engine received (7.3)
                    "raw": raw, "answer": ex.answer, "extraction_method": ex.method, "hedged": ex.hedged,
                    "n_marker_lines": ex.n_marker_lines, "n_thinking_chars": n_think,
                    "thinking_unclosed": ex.thinking_unclosed, "reasoning": reasoning,
                    "n_thinking_tokens": g.get("n_thinking_tokens"),
                    "n_prompt_tokens": n_tok, "n_output_tokens": n_out, "n_prompt_tokens_base": n_base,
                    "delta_prompt_tokens": (n_tok - n_base) if (n_tok is not None and n_base is not None) else None,
                    "latency_s": g.get("latency_s", batch_latency / len(batch)), "finish_reason": finish,
                    "truncated": bool(truncated), "prompt_truncated": g.get("prompt_truncated", False),
                    "n_placement_changes": R.count_placement_changes(_item_text(req.item, kind)),
                    "demo_syllable_overlap": req.item["item_id"] in flagged_ids,
                    "timestamp_utc": _now(),
                }
                if g.get("model_version"):
                    model_echo.add(str(g["model_version"]))
                if use_lp:
                    try:
                        if req.item["task"] == "T3":
                            row["logprobs"] = t3_logprobs(backend, req, msgs, opts)
                        elif req.item["task"] == "T1" and opts.t1_forced_choice and req.arm not in P.STRIP_ARMS:
                            row["forced_choice"] = t1_forced_choice(backend, req, msgs, opts)
                        elif req.item["task"] == X.XCOPA_TASK and opts.xcopa_logprob_mode == "choice":
                            row["logprobs"] = xcopa_logprobs(backend, req)
                    except NotImplementedError:
                        use_lp = False
                    for v in (row.get("logprobs") or {}).values():
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            nan_inf += 1
                    if (row.get("logprobs") or {}).get("prefix_property_violation"):
                        manifest["data"]["prefix_property_violations"] += 1
                    for cont, v in ((row.get("logprobs") or {}).get("prefix_property_violations") or {}).items():
                        if v:
                            by_cont = manifest["data"]["prefix_property_violations_by_continuation"]
                            by_cont[cont] = by_cont.get(cont, 0) + 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            fout.flush()
            manifest["n_written"] = n_written
            if (start // bs) % 10 == 0:
                write_manifest(run_dir, manifest)
                log(f"[run {run_id}] {start + len(batch)}/{len(todo)}")

    wall = time.monotonic() - t_start
    outcome = count_outcomes(outputs_path)
    ended = _now()
    n_acc = manifest["hardware"]["n_gpus"]
    hours = wall / 3600.0 * n_acc
    status = STATUS_FINISHED
    if outcome["n_thinking_chars_total"] > 0 and not thinking_allowed:
        # DESIGN_DECISIONS 5.5 step 2 / 7.3: thinking text MUST be 0 in a main run; the scorer refuses this status
        status = STATUS_THINKING
        log(f"[run {run_id}] WARNING: {outcome['n_thinking_chars_total']} thinking characters in a main run -> status "
            f"{status}; score with --allow-thinking only for the reasoning sub-study")
    manifest.update(status=status, finished_utc=ended, end_utc=ended, wall_s=wall, n_written=n_written,
                    gpu_hours=0.0 if is_tpu else hours, tpu_hours=hours if is_tpu else 0.0,
                    canary_check=canary_check(outputs_path, canary, canary_in_prompts), backend=backend.info())
    manifest["data"]["canary_check"] = manifest["canary_check"]
    manifest["identity"]["tokenizer_sha256"] = manifest["backend"].get("tokenizer_sha256")
    manifest["outcome"].update(n_done=outcome["n_rows"], n_unparseable=outcome["n_unparseable"],
                               n_truncated=outcome["n_truncated"], n_thinking_chars_total=outcome["n_thinking_chars_total"],
                               thinking_present_in_main_run=(status == STATUS_THINKING),
                               n_hedged=outcome["n_hedged"], n_prompt_truncated=outcome["n_prompt_truncated"],
                               nan_inf_events=nan_inf, api_model_echo=sorted(model_echo),
                               rate_limit_events=getattr(backend, "n_retries", None),
                               n_prefix_property_violations=manifest["backend"].get("n_prefix_property_violations"),
                               n_truncated_prompts_backend=manifest["backend"].get("n_truncated_prompts"))
    manifest["n_unparseable"] = outcome["n_unparseable"]
    manifest["n_truncated"] = outcome["n_truncated"]
    manifest["n_thinking_chars_total"] = outcome["n_thinking_chars_total"]
    write_manifest(run_dir, manifest)
    log(f"[run {run_id}] {status}: {n_written} rows in {wall:.1f}s")
    return run_dir


def kaggle_kernel_info() -> dict:
    """The KAGGLE_* environment of the process (kernel slug, run type, ...; DESIGN_DECISIONS 7.5
    `kaggle_kernel_slug/version`), secrets excluded; an empty dict off Kaggle."""
    return {k: os.environ[k] for k in sorted(os.environ)
            if k.startswith("KAGGLE_") and not any(s in k for s in ("SECRET", "KEY", "TOKEN", "PASSWORD"))}


def is_unparseable_row(d: dict) -> bool:
    """The scorer's notion of unparseable (5.6) for one outputs row: no answer; a T3 answer that
    maps to neither label (a kept hedge such as 'Có thể'); an XCOPA answer that is not 1 or 2."""
    ans = d.get("answer")
    if ans is None:
        return True
    task = d.get("task")
    if task == "T3":
        return t3_label(ans) is None
    if task == X.XCOPA_TASK:
        return ans not in ("1", "2")
    return False


def count_outcomes(outputs_path: Path) -> dict:
    """Outcome counts over EVERY row of outputs.jsonl (resumed rows included)."""
    out = {"n_rows": 0, "n_unparseable": 0, "n_truncated": 0, "n_thinking_chars_total": 0, "n_hedged": 0,
           "n_prompt_truncated": 0}
    if not outputs_path.exists():
        return out
    with open(outputs_path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            out["n_rows"] += 1
            if is_unparseable_row(d):
                out["n_unparseable"] += 1
            if d.get("truncated"):
                out["n_truncated"] += 1
            if d.get("hedged"):
                out["n_hedged"] += 1
            if d.get("prompt_truncated"):
                out["n_prompt_truncated"] += 1
            out["n_thinking_chars_total"] += int(d.get("n_thinking_chars") or 0)
    return out


def canary_check(outputs_path: Path, canary: str | None, canary_in_prompts: bool = False) -> dict:
    """The canary must never be echoed: it is never put in a prompt, and a model producing it
    would be evidence of contamination. `canary_in_prompts` is what run() measured over every
    rendered prompt (the outputs rows carry prompt hashes, not prompt text), never a constant."""
    if canary is None:
        return {"item_file_has_canary": False, "canary_in_outputs": False, "canary_in_prompts": bool(canary_in_prompts)}
    hit = False
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            for ln in f:
                if canary in ln:
                    hit = True
                    break
    return {"item_file_has_canary": True, "canary_in_outputs": hit, "canary_in_prompts": bool(canary_in_prompts)}


def read_outputs(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "outputs.jsonl", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_unchanged(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "unchanged.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
