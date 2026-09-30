"""One evaluation run: model config x item file -> data/runs/<run_id>/{outputs.jsonl, manifest.json}.

    items, kind = load_item_file(path)                 # NóiLái jsonl or XCOPA jsonl
    opts = RunOptions(tasks=..., variants=..., paraphrases=..., shots=..., arms=..., ...)
    run_dir = run(backend, entry, items, path, opts)

Requests are the product item x paraphrase x shots x arm (T2 ignores the variant filter's
effect on rendering but keeps it for selection). outputs.jsonl is appended after every
batch and `--resume` skips (item_id, arm, prompt_id) keys already present. manifest.json is
written at start (status running) and rewritten at the end with the fields of
docs/DATA_FORMAT.md: model id and revision, backend and version, dtype, quantization,
engine flags, seed, prompt file hashes, item file hash, canary check, hardware, start/end,
GPU-hours (wall x n_gpus), resource hashes, options and counts.

Safety guards
  * the canary is never part of a prompt (render() asserts it) and the manifest records
    whether it appears in any output;
  * an API backend refuses an item file with non-core items unless allow_noncore_api (the
    core set and the public attested examples are API-eligible), and never sends an attested
    item whose `vulgar` flag is set (counted in the manifest);
  * an item file sharing a syllable with the demonstrations is refused unless
    allow_demo_overlap (the overlap is recorded either way).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..gen import variants as V
from ..vi import reencode as R
from . import prompts as P
from . import xcopa as X
from .backends import Backend, backend_versions, slug
from .extract import extract_answer

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs"
HASHES_FILE = ROOT / "data" / "HASHES.json"


class ApiSafetyError(RuntimeError):
    pass


class DemoOverlapError(RuntimeError):
    pass


@dataclass
class RunOptions:
    tasks: tuple = ("T1", "T2", "T3", "attested")
    variants: tuple = V.VARIANTS
    paraphrases: tuple = ("p0",)
    shots: tuple = (3,)
    arms: tuple = ("nfc",)
    limit: int = 0
    in_core_only: bool = False
    resume: bool = False
    allow_noncore_api: bool = False
    allow_demo_overlap: bool = False
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
    notes: dict = field(default_factory=dict)


# ------------------------------------------------------------------ items
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_item_file(path: Path) -> tuple[list[dict], str]:
    """(items, kind) with kind 'noilai' or 'xcopa'."""
    path = Path(path)
    if X.is_xcopa_file(path):
        return X.load_xcopa(path), "xcopa"
    items = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                items.append(json.loads(ln))
    return items, "noilai"


def is_vulgar(item: dict) -> bool:
    v = item.get("vulgar")
    return v is True or (isinstance(v, str) and v.strip().lower() in ("yes", "true", "1"))


def select_items(items: list[dict], opts: RunOptions, kind: str) -> list[dict]:
    if kind == "xcopa":
        sel = list(items)
    else:
        sel = [it for it in items if it["task"] in opts.tasks and it["variant"] in opts.variants]
    if opts.in_core_only:
        sel = [it for it in sel if it.get("in_core")]
    if opts.limit:
        sel = sel[: opts.limit]
    return sel


def check_api_safety(backend: Backend, items: list[dict], opts: RunOptions) -> tuple[list[dict], dict]:
    """Enforce the API rules; returns (items to send, report)."""
    report = {"is_api": backend.is_api, "n_excluded_vulgar": 0, "n_noncore": 0}
    if not backend.is_api:
        return items, report
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
    return kept, report


def check_demo_overlap(items: list[dict], opts: RunOptions, kind: str) -> dict:
    if kind == "xcopa":
        return {}
    overlap = P.demo_overlap(P.default_demos(), items)
    if overlap and not opts.allow_demo_overlap:
        raise DemoOverlapError("demonstration/instruction syllables occur in the item file: "
                               + ", ".join(f"{s} ({len(ids)} items)" for s, ids in overlap.items())
                               + "; regenerate with these syllables reserved or pass --allow-demo-overlap")
    return overlap


# ------------------------------------------------------------------ requests
@dataclass
class Request:
    item: dict
    arm: str
    paraphrase: str
    shots: int
    prompt_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.item["item_id"], self.arm, self.prompt_id)


def plan_requests(items: list[dict], opts: RunOptions, kind: str) -> list[Request]:
    reqs = []
    for it in items:
        for p in opts.paraphrases:
            for s in (opts.shots if kind == "noilai" else (0,)):
                for a in opts.arms:
                    pid = (P.prompt_id(p, s, opts.instruction, opts.input_format, opts.language) if kind == "noilai"
                           else f"xcopa-{p}")
                    reqs.append(Request(it, a, p, s, pid))
    return reqs


def existing_keys(outputs_path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    d = json.loads(ln)
                    keys.add((d["item_id"], d["arm"], d["prompt_id"]))
    return keys


def render_request(req: Request, opts: RunOptions, kind: str) -> list[dict]:
    if kind == "xcopa":
        return X.render_xcopa(req.item, req.paraphrase, req.arm, system=opts.system_prompt)
    return P.render(req.item, paraphrase=req.paraphrase, shots=req.shots, arm=req.arm,
                    input_format=opts.input_format, instruction=opts.instruction, language=opts.language,
                    system=opts.system_prompt)


# ------------------------------------------------------------------ logprobs
def _marker_for(arm: str) -> str:
    return R.reencode(P.answer_marker(), arm) if R.meaning_preserving(arm) else P.answer_marker()


def t3_logprobs(backend: Backend, req: Request, messages: list[dict], opts: RunOptions) -> dict:
    """Answer-token log-probabilities P(Có), P(Không) after the rendered T3 prompt, and the
    candidate's string log-probability under a T1 context for the same input and variant
    (shared by the yes/no pair -> BLiMP-style paired scoring)."""
    out: dict = {}
    arm = req.arm
    prompt_text = backend.chat_to_text(messages) + _marker_for(arm) + " "
    conts = [R.reencode(w, arm) if R.meaning_preserving(arm) else w for w in ("Có", "Không")]
    lp = backend.logprobs(prompt_text, conts)
    out["Có"], out["Không"] = float(lp[0]), float(lp[1])
    it = req.item
    pseudo = {"task": "T1", "variant": it["variant"], "input": it["input"], "item_id": it["item_id"],
              "canary": it.get("canary"), "gold": [it.get("correct_output", "")]}
    ctx_messages = P.render(pseudo, paraphrase=req.paraphrase, shots=req.shots, arm=arm, input_format=opts.input_format,
                            instruction=opts.instruction, language=opts.language, system=opts.system_prompt)
    ctx = backend.chat_to_text(ctx_messages) + _marker_for(arm) + " "
    cand = R.reencode(it["candidate"], arm)
    out["candidate"] = float(backend.logprobs(ctx, [cand])[0])
    out["candidate_context_hash"] = P.prompt_hash(ctx_messages)
    return out


def xcopa_logprobs(backend: Backend, req: Request) -> dict:
    ctx, conts = X.completion_pair(req.item, req.arm)
    lp = backend.logprobs(ctx, conts)
    return {"1": float(lp[0]), "2": float(lp[1]), "context": ctx}


# ------------------------------------------------------------------ manifest
def git_info() -> dict:
    def _run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return None
    commit = _run("rev-parse", "HEAD")
    dirty = _run("status", "--porcelain", "--", str(ROOT))
    return {"git_commit": commit or "unknown", "git_dirty": (bool(dirty) if dirty is not None else None)}


def hardware_info(n_override: int | None = None) -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "cpu": platform.processor() or None,
            "gpus": [], "n_gpus": 0}
    with contextlib.suppress(Exception):      # no torch, or a broken CUDA install: report no GPU
        import torch

        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["n_gpus"] = torch.cuda.device_count()
            info["cuda"] = torch.version.cuda
    if n_override is not None:
        info["n_gpus"] = int(n_override)
        info["n_accelerators_override"] = int(n_override)
    return info


def resource_hashes() -> dict:
    if HASHES_FILE.exists():
        return json.loads(HASHES_FILE.read_text(encoding="utf-8"))
    return {}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def write_manifest(run_dir: Path, manifest: dict) -> None:
    tmp = run_dir / "manifest.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)
    os.replace(tmp, run_dir / "manifest.json")


def default_run_id(entry: dict, item_path: Path) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slug(entry.get('name') or entry.get('model_id') or 'model')}__{slug(Path(item_path).stem)}__{stamp}"


# ------------------------------------------------------------------ the run
def run(backend: Backend, entry: dict, items: list[dict], item_path: Path, opts: RunOptions,
        kind: str | None = None, log=print) -> Path:
    item_path = Path(item_path)
    kind = kind or ("xcopa" if items and items[0].get("task") == X.XCOPA_TASK else "noilai")
    if kind == "noilai" and opts.input_format != "raw" and any(a in P.STRIP_ARMS for a in opts.arms):
        raise ValueError("strip arms are defined on the raw input format only")
    selected = select_items(items, opts, kind)
    selected, api_report = check_api_safety(backend, selected, opts)
    overlap = check_demo_overlap(selected, opts, kind)
    canary = next((it["canary"] for it in items if it.get("canary")), None)

    run_id = opts.run_id or default_run_id(entry, item_path)
    run_dir = Path(opts.out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = run_dir / "outputs.jsonl"
    done = existing_keys(outputs_path) if opts.resume else set()
    if outputs_path.exists() and not opts.resume:
        raise FileExistsError(f"{outputs_path} exists; pass --resume to continue it or choose another --run-id")

    reqs = plan_requests(selected, opts, kind)
    todo = [r for r in reqs if r.key not in done]
    t_start = time.monotonic()
    started = _now()
    manifest = {
        "run_id": run_id, "status": "running", "started_utc": started, "finished_utc": None,
        "model": {"name": entry.get("name"), "model_id": entry.get("hf_id") or entry.get("model_id")
                  or entry.get("provider_model_id"), "revision": entry.get("revision"),
                  "provider": entry.get("provider"), "provider_model_id": entry.get("provider_model_id"),
                  "hardware_entry": entry.get("hardware"), "config_entry": entry},
        "backend": backend.info(), "backend_versions": backend_versions(),
        "seed": opts.seed, "options": asdict(opts),
        "prompt_files_sha256": P.prompt_file_hashes(),
        "item_file": {"path": str(item_path), "sha256": sha256_file(item_path), "kind": kind, "n_items": len(items),
                      "n_selected": len(selected), "canary_present": canary is not None,
                      "canary_sha256": hashlib.sha256(canary.encode()).hexdigest() if canary else None},
        "api_safety": api_report, "demo_overlap": overlap,
        "n_requests": len(reqs), "n_resumed": len(reqs) - len(todo), "n_written": 0,
        "hardware": hardware_info(opts.n_accelerators), "resource_sha256": resource_hashes(), **git_info(),
        "gpu_hours": None, "wall_s": None, "canary_check": None,
    }
    write_manifest(run_dir, manifest)
    log(f"[run {run_id}] {len(todo)} requests to do ({len(reqs) - len(todo)} resumed) -> {outputs_path}")

    use_lp = opts.logprobs and backend.supports_logprobs
    n_written = 0
    bs = max(1, opts.batch_size)
    with open(outputs_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), bs):
            batch = todo[start: start + bs]
            messages_batch = [render_request(r, opts, kind) for r in batch]
            t0 = time.monotonic()
            gens = backend.generate(messages_batch, max_new_tokens=opts.max_new_tokens, greedy=True)
            batch_latency = time.monotonic() - t0
            for req, msgs, g in zip(batch, messages_batch, gens):
                raw = g.get("text", "")
                answer, method = extract_answer(raw, req.item["task"])
                row = {
                    "item_id": req.item["item_id"], "task": req.item["task"], "variant": req.item["variant"],
                    "arm": req.arm, "prompt_id": req.prompt_id, "paraphrase": req.paraphrase, "shots": req.shots,
                    "prompt_hash": P.prompt_hash(msgs), "raw": raw, "answer": answer, "extraction_method": method,
                    "n_prompt_tokens": g.get("n_prompt_tokens"), "n_output_tokens": g.get("n_output_tokens"),
                    "latency_s": g.get("latency_s", batch_latency / len(batch)), "finish_reason": g.get("finish_reason"),
                    "timestamp_utc": _now(),
                }
                if use_lp:
                    try:
                        if req.item["task"] == "T3":
                            row["logprobs"] = t3_logprobs(backend, req, msgs, opts)
                        elif req.item["task"] == X.XCOPA_TASK and opts.xcopa_logprob_mode == "choice":
                            row["logprobs"] = xcopa_logprobs(backend, req)
                    except NotImplementedError:
                        use_lp = False
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            fout.flush()
            manifest["n_written"] = n_written
            if (start // bs) % 10 == 0:
                write_manifest(run_dir, manifest)
                log(f"[run {run_id}] {start + len(batch)}/{len(todo)}")

    wall = time.monotonic() - t_start
    manifest.update(status="finished", finished_utc=_now(), wall_s=wall, n_written=n_written,
                    gpu_hours=wall / 3600.0 * manifest["hardware"]["n_gpus"],
                    canary_check=canary_check(outputs_path, canary), backend=backend.info())
    write_manifest(run_dir, manifest)
    log(f"[run {run_id}] finished: {n_written} rows in {wall:.1f}s")
    return run_dir


def canary_check(outputs_path: Path, canary: str | None) -> dict:
    """The canary must never be echoed: it is never put in a prompt, and a model producing it
    would be evidence of contamination."""
    if canary is None:
        return {"item_file_has_canary": False, "canary_in_outputs": False, "canary_in_prompts": False}
    hit = False
    if outputs_path.exists():
        with open(outputs_path, encoding="utf-8") as f:
            for ln in f:
                if canary in ln:
                    hit = True
                    break
    return {"item_file_has_canary": True, "canary_in_outputs": hit, "canary_in_prompts": False}


def read_outputs(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "outputs.jsonl", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_manifest(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
