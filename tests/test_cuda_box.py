"""A rented CUDA box as a run session: the planner decides what fits by VRAM and bf16 support, logs the
real device, and a one-command driver runs the chunk plan's units (configs/compute_chunks.yaml) on the box:
chunk-scoped weight purge, prefetch of the next chunk, check_run / sync / Kaggle push after every executed
model, and `--lanes N` scheduling of the box's cards.

Everything here runs offline: the session parser, the fit rule, the chunk selection, the purge and prefetch
rules and the lane schedule are exercised against configs/models.yaml and configs/compute_chunks.yaml; the
driver's loop runs against a stub run_eval.py in a temporary project root with a temporary hub cache.
"""
import copy
import json
import shlex
import shutil
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import compute_log as CL
import cuda_box as CB
import kaggle_run_plan as KRP
import test_cloud as TC


@pytest.fixture(scope="module")
def models_cfg():
    return KRP.load_models()


# ------------------------------------------------------------------ compute log
def test_compute_log_accepts_a_rented_box_and_aws_as_platforms():
    for platform in ("rented", "aws"):
        e = CL.Entry(date="2026-10-12", platform=platform, gpu_type="rtx3090", n_gpus=1, hours=0.5, run_id="r", purpose="p")
        assert e.platform == platform and e.kind == "gpu"


# ------------------------------------------------------------------ the session spec
def test_parse_cuda_session_reads_name_vram_count_and_bf16():
    s = KRP.parse_cuda_session("rtx3090:24")
    assert (s.gpu_type, s.vram_gb, s.n_gpus, s.bf16) == ("rtx3090", 24.0, 1, True)
    s = KRP.parse_cuda_session("a100-80gb:80x1")
    assert (s.gpu_type, s.vram_gb, s.n_gpus, s.bf16) == ("a100-80gb", 80.0, 1, True)
    s = KRP.parse_cuda_session("v100-32gb:32x2:nobf16")
    assert (s.gpu_type, s.vram_gb, s.n_gpus, s.bf16) == ("v100-32gb", 32.0, 2, False)
    assert str(s) == "cuda v100-32gb:32x2:nobf16"
    assert KRP.parse_cuda_session("rtx3090:24").usable_gb == pytest.approx(24 * 0.9)      # vLLM's gpu_memory_utilization
    assert KRP.parse_cuda_session("rtx3090:24x2").usable_gb == pytest.approx(48 * 0.9)
    for bad in ("rtx3090", "rtx3090:abc", "rtx3090:24x0", ":24", "rtx3090:24:maybe"):
        with pytest.raises(ValueError):
            KRP.parse_cuda_session(bad)


def test_cuda_session_from_nvidia_smi_output():
    text = "NVIDIA GeForce RTX 3090, 24576, 8.6\nNVIDIA GeForce RTX 3090, 24576, 8.6\n"
    s = KRP.cuda_session_from_nvidia_smi(text)
    assert (s.gpu_type, s.n_gpus, s.bf16) == ("geforce-rtx-3090", 2, True) and s.vram_gb == pytest.approx(24.0)
    s = KRP.cuda_session_from_nvidia_smi("Tesla V100-SXM2-32GB, 32768, 7.0\n")
    assert (s.gpu_type, s.n_gpus, s.bf16) == ("v100-sxm2-32gb", 1, False) and s.vram_gb == pytest.approx(32.0)
    s = KRP.cuda_session_from_nvidia_smi("NVIDIA A100-SXM4-80GB, 81920, 8.0\n")
    assert s.gpu_type == "a100-sxm4-80gb" and s.bf16
    with pytest.raises(ValueError):
        KRP.cuda_session_from_nvidia_smi("")
    with pytest.raises(ValueError):                                   # two different cards is not one session
        KRP.cuda_session_from_nvidia_smi("NVIDIA A100-SXM4-80GB, 81920, 8.0\nNVIDIA GeForce RTX 3090, 24576, 8.6\n")


def test_cuda_visible_devices_narrows_the_indexed_nvidia_smi_rows(monkeypatch):
    assert KRP.NVIDIA_SMI_QUERY[1].startswith("--query-gpu=index,name,"), "the index comes first so a lane can filter by it"
    two = "0, NVIDIA A800-SXM4-40GB, 40960, 8.0\n1, NVIDIA A800-SXM4-40GB, 40960, 8.0\n"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    s = KRP.cuda_session_from_nvidia_smi(two)
    assert (s.gpu_type, s.vram_gb, s.n_gpus, s.bf16) == ("a800-sxm4-40gb", 40.0, 2, True) and str(s) == "cuda a800-sxm4-40gb:40x2"
    assert KRP.cuda_session_from_nvidia_smi(two, visible="0").n_gpus == 1
    assert KRP.cuda_session_from_nvidia_smi(two, visible="1").n_gpus == 1
    assert KRP.cuda_session_from_nvidia_smi(two, visible="1,0").n_gpus == 2
    assert KRP.cuda_session_from_nvidia_smi(two, visible="GPU-aaaa,GPU-bbbb").n_gpus == 2, "a UUID list keeps every row"
    with pytest.raises(ValueError, match="no GPU visible"):
        KRP.cuda_session_from_nvidia_smi(two, visible="")
    with pytest.raises(ValueError, match="no GPU visible"):
        KRP.cuda_session_from_nvidia_smi(two, visible="5")
    assert KRP.visible_gpu_indices(None) is None and KRP.visible_gpu_indices("") == [] and KRP.visible_gpu_indices("0, 1") == [0, 1]
    assert KRP.visible_gpu_indices("GPU-aaaa") is None
    # the environment is the default, and detect_cuda_session passes it; the old three-field rows carry no index to filter
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert KRP.cuda_session_from_nvidia_smi(two).n_gpus == 1
    assert KRP.cuda_session_from_nvidia_smi("NVIDIA A800-SXM4-40GB, 40960, 8.0\nNVIDIA A800-SXM4-40GB, 40960, 8.0\n").n_gpus == 2
    monkeypatch.setattr(KRP.subprocess, "check_output", lambda *a, **k: two)
    assert KRP.detect_cuda_session().n_gpus == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert KRP.detect_cuda_session().n_gpus == 2


# ------------------------------------------------------------------ the fit rule
def test_cuda_fit_decides_by_estimated_weight_size_and_bf16_support(models_cfg):
    by = models_cfg["by_name"]
    s24 = KRP.parse_cuda_session("rtx3090:24")
    s48 = KRP.parse_cuda_session("a40:48")
    s80 = KRP.parse_cuda_session("a100-80gb:80")
    s24x2 = KRP.parse_cuda_session("rtx3090:24x2")
    s32_nobf16 = KRP.parse_cuda_session("v100-32gb:32:nobf16")
    # the estimate: params x bytes per parameter x 1.1 (runtime overhead) + 1 GB (CUDA context); fp32 1B = 5.4 GB
    ok, need, _ = KRP.cuda_fit(by["gemma-3-1b-it"], s24, models_cfg)
    assert ok and need == pytest.approx(1.0 * 4 * 1.1 + 1.0)
    assert KRP.cuda_fit(by["gemma-3-4b-it"], s24, models_cfg)[0], "fp32 4.3B (~17 GB of weights) fits one 24 GB card"
    assert not KRP.cuda_fit(by["gemma-3-12b-it"], s24, models_cfg)[0], "bf16 12B (~24 GB of weights) does not"
    assert KRP.cuda_fit(by["gemma-3-12b-it"], s24x2, models_cfg)[0], "two 24 GB cards do, under tensor parallelism"
    assert not KRP.cuda_fit(by["qwen3.8-27b"], s48, models_cfg)[0], "bf16 27B (~54 GB of weights) needs the 80 GB card"
    assert KRP.cuda_fit(by["qwen3.8-27b"], s80, models_cfg)[0]
    assert KRP.cuda_fit(by["qwen3.5-9b"], s24, models_cfg)[0], "4-bit AWQ 9B is small"
    assert KRP.cuda_fit(by["phogpt-4b-chat"], s24, models_cfg)[0], "the HF backend counts the same way"
    # a bf16 reference entry takes its size from the entry it references, and needs a bf16-capable card
    ok, need, reason = KRP.cuda_fit(by["gemma-3-1b-it--bf16"], s24, models_cfg)
    assert ok and need == pytest.approx(1.0 * 2 * 1.1 + 1.0)
    ok, _, reason = KRP.cuda_fit(by["gemma-3-1b-it--bf16"], s32_nobf16, models_cfg)
    assert not ok and "bf16" in reason
    assert KRP.cuda_fit(by["qwen3.5-2b"], s32_nobf16, models_cfg)[0], "fp16 entries run on a pre-Ampere card"
    # API entries run from any session
    assert KRP.cuda_fit(by["gpt-oss-20b"], s24, models_cfg)[0]
    # the planner's guard and device accounting dispatch on the session type
    assert KRP.hardware_compatible(by["gemma-3-12b-it"], s80, models_cfg) and not KRP.hardware_compatible(by["gemma-3-12b-it"], s24, models_cfg)
    assert KRP.hardware_compatible(by["gemma-3-1b-it"], "2xt4"), "string sessions are unchanged"
    assert KRP.session_device(s24x2) == ("rtx3090", 2) and KRP.session_device("2xt4") == ("t4", 2)
    # tensor parallelism: one GPU unless the estimate exceeds a single card
    assert KRP.tensor_parallel_for(by["gemma-3-4b-it"], s24x2, models_cfg) == 1, "a 2xT4 entry needs no tp=2 on a 24 GB card"
    assert KRP.tensor_parallel_for(by["gemma-3-12b-it"], s24x2, models_cfg) == 2
    assert KRP.tensor_parallel_for(by["qwen3.8-27b"], s80, models_cfg) == 1


# ------------------------------------------------------------------ the planner's loop on a box


@pytest.fixture(scope="module")
def plan():
    return TC._unpinned(KRP.load_plan())


@pytest.fixture(scope="module")
def pinned_cfg(models_cfg):
    cfg = copy.deepcopy({k: v for k, v in models_cfg.items() if k != "by_name"})
    for m in cfg["models"]:
        if not KRP.is_api(m):
            m["revision"] = TC.PINNED_REVISION
    cfg["by_name"] = {m["name"]: m for m in cfg["models"]}
    return cfg


@pytest.fixture(scope="module")
def chunk_spec():
    return CB.load_chunks()


@pytest.fixture(scope="module")
def chunk_by_id(chunk_spec):
    return {c["id"]: c for c in chunk_spec["chunks"]}


def _stub_project(tmp_path: Path, plan: dict) -> Path:
    """test_cloud's stub project, with a run_eval.py that also accepts --tensor-parallel-size."""
    proj = TC._stub_project(tmp_path, plan, with_items=False)
    stub = TC.STUB_RUN_EVAL.replace('ap.add_argument("--in-core-only", action="store_true")',
                                    'ap.add_argument("--in-core-only", action="store_true")\nap.add_argument("--tensor-parallel-size")')
    (proj / "scripts" / "run_eval.py").write_text(stub)
    return proj


def test_execute_on_a_cuda_session_skips_by_fit_logs_the_card_and_sets_tensor_parallelism(tmp_path, plan, pinned_cfg):
    proj = _stub_project(tmp_path, plan)
    log = proj / "log.csv"
    s = KRP.parse_cuda_session("rtx3090:24x2")
    res = KRP.execute("E1_main", models=["gemma-3-1b-it", "gemma-3-12b-it", "qwen3.8-27b"], plan=plan, models_cfg=pinned_cfg,
                      project_root=proj, log_path=log, verify=False, session_hardware=s, platform="rented")
    status = {r["model"]: r["status"] for r in res}
    assert status["gemma-3-1b-it"] == "ok" and status["gemma-3-12b-it"] == "ok", status
    assert status["qwen3.8-27b"].startswith("skipped: hardware tpu is not runnable in a cuda rtx3090:24x2 session"), status
    assert "needs ~60.4 GB" in status["qwen3.8-27b"] and "43.2 GB usable" in status["qwen3.8-27b"]
    assert res[0]["session_hardware"] == "cuda rtx3090:24x2"
    rows = CL.read_entries(log)
    assert rows and all((r.gpu_type, r.n_gpus, r.platform) == ("rtx3090", 2, "rented") for r in rows), "rows carry the box's card and count"
    # the command tells run_eval how many cards to shard over: one unless the estimate exceeds a single card
    tp = {m: json.loads((proj / "data" / "runs" / f"E1_main__{m}" / "manifest.json").read_text())["tensor_parallel_size"]
          for m in ("gemma-3-1b-it", "gemma-3-12b-it")}
    assert tp == {"gemma-3-1b-it": "1", "gemma-3-12b-it": "2"}
    assert res[0]["tensor_parallel_size"] == 1 and res[1]["tensor_parallel_size"] == 2
    # only vLLM entries take the flag: the HF backend (PhoGPT) and API entries do not
    res = KRP.execute("E1_main", models=["phogpt-4b-chat"], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log,
                      verify=False, session_hardware=s, platform="rented", dry_run=True)
    assert "--tensor-parallel-size" not in res[0]["cmd"]
    res = KRP.execute("E1_api_core", models=["gpt-oss-20b"], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log,
                      verify=False, session_hardware=s, platform="api", dry_run=True)
    assert res[0]["status"] == "dry-run" and "--tensor-parallel-size" not in res[0]["cmd"]
    # --allow-hardware-mismatch still forces a run, logged as the box's card
    res = KRP.execute("E1_main", models=["qwen3.8-27b"], plan=plan, models_cfg=pinned_cfg, project_root=proj, log_path=log,
                      verify=False, session_hardware=s, platform="rented", allow_hardware_mismatch=True)
    assert res[0]["status"] == "ok" and res[0]["hardware_mismatch"] and res[0]["logged_device"] == {"gpu_type": "rtx3090", "n_gpus": 2}


def test_run_eval_takes_a_tensor_parallel_override_over_the_hardware_block():
    import run_eval as RE

    ap = RE.build_parser()
    args = ap.parse_args(["--items", "x.jsonl", "--model-config", "gemma-3-4b-it", "--tensor-parallel-size", "1"])
    assert RE.resolve_entry(args)["tensor_parallel_size"] == 1
    args = ap.parse_args(["--items", "x.jsonl", "--model-config", "gemma-3-4b-it"])
    assert RE.resolve_entry(args)["tensor_parallel_size"] == 2, "without the flag the 2xt4 hardware block decides"


def test_cli_takes_a_cuda_session_spec_or_detects_the_box(capsys, monkeypatch):
    rc = KRP.main(["--run", "E1_main", "--models", "gemma-3-1b-it", "qwen3.8-27b", "--dry-run", "--platform", "rented",
                   "--session-hardware", "cuda", "--cuda-session", "rtx3090:24"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "E1_main__gemma-3-1b-it" in out and "--tensor-parallel-size 1" in out
    assert "qwen3.8-27b: hardware tpu is not runnable in a cuda rtx3090:24x1 session (needs ~60.4 GB" in out
    # no spec: the box is asked (nvidia-smi); here the probe is stubbed
    monkeypatch.setattr(KRP, "detect_cuda_session", lambda: KRP.parse_cuda_session("a100-80gb:80"))
    rc = KRP.main(["--run", "E1_main", "--models", "qwen3.8-27b", "--dry-run", "--platform", "rented", "--session-hardware", "cuda"])
    out = capsys.readouterr().out
    assert rc == 0 and "[session] cuda a100-80gb:80x1" in out and "E1_main__qwen3.8-27b" in out and "not runnable" not in out
    # a spec with a Kaggle session name is a contradiction
    with pytest.raises(SystemExit):
        KRP.main(["--run", "E1_main", "--models", "gemma-3-1b-it", "--dry-run", "--session-hardware", "2xt4", "--cuda-session", "rtx3090:24"])
    # no nvidia-smi on this machine: a clear error, not a traceback
    monkeypatch.setattr(KRP, "detect_cuda_session", lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    with pytest.raises(SystemExit):
        KRP.main(["--run", "E1_main", "--models", "gemma-3-1b-it", "--dry-run", "--session-hardware", "cuda"])
    assert "nvidia-smi" in capsys.readouterr().err


# ------------------------------------------------------------------ the box driver: bootstrap and whole-line mode


def test_vllm_pin_is_read_from_pyproject():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    want = next(s.split("==")[1] for s in extras["vllm"] if s.startswith("vllm=="))
    assert CB.vllm_pin(ROOT / "pyproject.toml") == want
    with pytest.raises(ValueError):
        CB.vllm_pin(ROOT / "requirements.txt")


def test_bootstrap_commands_install_the_pinned_engine_fetch_resources_and_materialize_item_files():
    cmds = CB.bootstrap_commands(Path("/box/noilai"), python="py", vllm_version="0.30.0")
    assert cmds[0] == ["py", "-m", "pip", "install", "-q", "vllm==0.30.0"]
    assert cmds[1] == ["py", "-m", "pip", "install", "-q", "-e", "/box/noilai[eval]"]
    assert cmds[2] == ["py", "scripts/fetch_resources.py"]
    assert cmds[3] == ["py", "scripts/kaggle_run_plan.py", "--materialize"]
    assert len(cmds) == 4
    # the Kaggle CLI only when the dataset push is wanted
    cmds = CB.bootstrap_commands(Path("/box/noilai"), python="py", vllm_version="0.30.0", kaggle=True)
    assert len(cmds) == 5 and cmds[2] == ["py", "-m", "pip", "install", "-q", "kaggle"] and cmds[3] == ["py", "scripts/fetch_resources.py"]


def test_default_run_order_is_the_plans_self_hosted_lines_without_api_e4_or_duplicate_booking_lines(plan, models_cfg):
    ids = CB.default_run_ids(plan, models_cfg)
    assert ids[:2] == ["pilot_t1_200", "pilot_xcopa_200"], "the Gate 1 pilot goes first"
    assert "E4_probe" not in ids, "E4 is not a run_eval line"
    assert not any(KRP.find_run(plan, r).get("hardware") == "api" for r in ids), "API lines run from anywhere, not from the box"
    assert "tpu_main" not in ids, "tpu_main books TPU time for models E1_main already covers on a box"
    assert "smoke_20_tpu" in ids and "reasoning_500_tpu" in ids and "bf16_drift_200" in ids, "the TPU-only models' lines stay"
    order = [r["id"] for r in plan["runs"]]
    assert ids == [r for r in order if r in ids], "plan order is the cut order"
    assert CB.duplicate_of(KRP.find_run(plan, "tpu_main"), [KRP.find_run(plan, "E1_main")], plan, models_cfg) == "E1_main"
    assert CB.duplicate_of(KRP.find_run(plan, "smoke_20_tpu"), [KRP.find_run(plan, "smoke_20")], plan, models_cfg) is None


def test_run_all_runs_the_lines_in_order_and_syncs_after_every_executed_model(tmp_path, plan, pinned_cfg):
    proj = _stub_project(tmp_path, plan)
    marker = tmp_path / "synced"
    sync = f"{sys.executable} -c \"open({str(marker)!r}, 'a').write('x')\""
    s = KRP.parse_cuda_session("rtx3090:24")
    res = CB.run_all(["smoke_20", "E1_main"], session=s, platform="rented", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                     log_path=proj / "log.csv", models=["gemma-3-1b-it", "qwen3.8-27b"], sync_cmd=sync, verify=False)
    assert [(r["run"], r["model"], r["status"][:7]) for r in res] == [
        ("smoke_20", "gemma-3-1b-it", "ok"), ("smoke_20", "qwen3.8-27b", "skipped"),
        ("E1_main", "gemma-3-1b-it", "ok"), ("E1_main", "qwen3.8-27b", "skipped")]
    assert marker.read_text() == "xx", "one sync per executed model, none for a skipped one"
    assert {r.platform for r in CL.read_entries(proj / "log.csv")} == {"rented"}
    # a dry run executes nothing and syncs nothing
    marker.unlink()
    res = CB.run_all(["E1_main"], session=s, platform="rented", plan=plan, models_cfg=pinned_cfg, project_root=proj,
                     log_path=proj / "log.csv", models=["gemma-3-1b-it"], sync_cmd=sync, verify=False, dry_run=True)
    assert res[0]["status"] == "dry-run" and not marker.exists()


def test_cli_dry_run_prints_the_bootstrap_and_the_commands(capsys, monkeypatch):
    rc = CB.main(["--bootstrap", "--dry-run", "--session", "rtx3090:24", "--runs", "E1_main", "--models", "gemma-3-1b-it",
                  "--platform", "rented", "--sync-cmd", "echo sync"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"vllm=={CB.vllm_pin(ROOT / 'pyproject.toml')}" in out and "fetch_resources.py" in out
    assert "[session] cuda rtx3090:24x1" in out and "E1_main__gemma-3-1b-it" in out and "--tensor-parallel-size 1" in out
    monkeypatch.setattr(KRP, "detect_cuda_session", lambda: KRP.parse_cuda_session("a100-80gb:80"))
    rc = CB.main(["--dry-run", "--runs", "E1_main", "--models", "qwen3.8-27b"])
    out = capsys.readouterr().out
    assert rc == 0 and "[session] cuda a100-80gb:80x1" in out and "E1_main__qwen3.8-27b" in out


# ------------------------------------------------------------------ chunk mode: selection and the command plan


def test_select_chunks_by_selector_in_chunk_order(chunk_spec):
    with pytest.raises(ValueError, match="tier9: no gpu/tpu chunk has tier 9; tiers:"):
        CB.select_chunks(chunk_spec, ["tier9"])
    every = CB.select_chunks(chunk_spec, ["all"])
    ids = [c["id"] for c in every]
    assert ids[0] == "c05_gpu_t0" and ids[-1] == "c31_tpu_t5" and len(ids) == 27
    assert all(c["queue"] in ("gpu", "tpu") for c in every), "the CPU chunks are the HF-on-CPU route"
    assert [c["id"] for c in CB.select_chunks(chunk_spec, ["tier0"])] == ["c05_gpu_t0", "c06_gpu_t0", "c07_tpu_t0"]
    ids = [c["id"] for c in CB.select_chunks(chunk_spec, ["from:c20_tpu_t2"])]
    assert ids[0] == "c20_tpu_t2" and ids[-1] == "c31_tpu_t5" and len(ids) == 12
    assert [c["id"] for c in CB.select_chunks(chunk_spec, ["c07_tpu_t0", "c05_gpu_t0", "c07_tpu_t0"])] == ["c05_gpu_t0", "c07_tpu_t0"]
    with pytest.raises(ValueError, match=r"unknown chunk 'c99'.*c05_gpu_t0"):
        CB.select_chunks(chunk_spec, ["c99"])
    with pytest.raises(ValueError, match="CPU chunks run the HF backend on a CPU session"):
        CB.select_chunks(chunk_spec, ["c01_cpu_t0"])
    with pytest.raises(ValueError, match="CPU chunks run the HF backend"):
        CB.select_chunks(chunk_spec, ["from:c02_cpu_t0"])


def test_command_plan_and_the_chunk_scoped_purge_rule(chunk_by_id, models_cfg):
    by = models_cfg["by_name"]
    # c05 -> c06: the three pilot models are in both chunks, so nothing of c05 is purged
    cmds = CB.command_plan([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"], chunk_by_id["c07_tpu_t0"]], models_cfg)
    assert cmds[0] == {"chunk": "c05_gpu_t0", "run": "pilot_t1_200", "model": "gemma-3-1b-it", "tag": None, "repo": "google/gemma-3-1b-it"}
    points = CB.purge_points(cmds)
    assert [p for c, p in zip(cmds, points, strict=True) if c["chunk"] == "c05_gpu_t0"] == [None] * 6
    c06 = [(c["model"], p) for c, p in zip(cmds, points, strict=True) if c["chunk"] == "c06_gpu_t0"]
    assert len(c06) == 14 and all(p == CB.repo_of(by[m]) for m, p in c06), "c06 purges each model right after its smoke (c07 shares none)"
    assert dict(c06)["llama-3.1-8b-instruct"] == "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "the quantized checkpoint is the repository"
    assert [p for c, p in zip(cmds, points, strict=True) if c["chunk"] == "c07_tpu_t0"] == [
        "google/gemma-3-12b-it", "Qwen/Qwen3.8-27B", "aisingapore/Qwen-SEA-LION-v4.5-27B-IT"], "the last chunk purges after the last use"
    # c16 -> c17: the 27B is purged at c16's end although c20 uses it again (c17 is next; bounded disk beats cached weights)
    cmds = CB.command_plan([chunk_by_id["c16_tpu_t1"], chunk_by_id["c17_gpu_t2"], chunk_by_id["c18_gpu_t2"]], models_cfg)
    points = CB.purge_points(cmds)
    assert [p for c, p in zip(cmds, points, strict=True) if c["chunk"] == "c16_tpu_t1"] == [None] * 5 + ["Qwen/Qwen3.8-27B"]
    c17 = [(c["run"], p) for c, p in zip(cmds, points, strict=True) if c["chunk"] == "c17_gpu_t2"]
    assert len(c17) == 32
    assert all(p is None for r, p in c17 if r != "E1_abl_spaced"), "c17's eight models stay for its four lines"
    assert all(p for r, p in c17 if r == "E1_abl_spaced"), "and go one by one during the last line"
    # `keep` (a lane's shared weights) exempts a repository; --models restricts the plan; an API entry has no repository
    assert CB.purge_points(cmds, keep=["Qwen/Qwen3.8-27B"])[5] is None
    cmds = CB.command_plan([chunk_by_id["c17_gpu_t2"]], models_cfg, models=["gemma-3-1b-it", "nobody"])
    assert [c["model"] for c in cmds] == ["gemma-3-1b-it"] * 4 and CB.purge_points(cmds) == [None, None, None, "google/gemma-3-1b-it"]
    assert CB.repo_of(by["gpt-oss-20b"]) is None and CB.repo_of(by["gemma-3-1b-it--bf16"]) == "google/gemma-3-1b-it"
    assert CB.job_models({"models": ["a", "b"]}, None) == ["a", "b"] and CB.job_models({"models": ["a", "b"]}, ["b"]) == ["b"]


# ------------------------------------------------------------------ the hub cache: purge and prefetch


def test_hf_cache_dir_follows_the_hub_library_and_purge_repo_removes_a_cached_repository(tmp_path, capsys):
    assert CB.hf_cache_dir({"HF_HUB_CACHE": "/a", "HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/c"}) == Path("/a")
    assert CB.hf_cache_dir({"HUGGINGFACE_HUB_CACHE": "/b", "HF_HOME": "/c"}) == Path("/b")
    assert CB.hf_cache_dir({"HF_HOME": "/c"}) == Path("/c/hub")
    assert CB.hf_cache_dir({"XDG_CACHE_HOME": "/x"}) == Path("/x/huggingface/hub")
    assert CB.hf_cache_dir({}) == Path.home() / ".cache" / "huggingface" / "hub"
    assert CB.repo_folder("google/gemma-3-1b-it") == "models--google--gemma-3-1b-it"
    cache = tmp_path / "hub"
    snap = cache / "models--google--gemma-3-1b-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x" * 2048)
    (cache / ".locks" / "models--google--gemma-3-1b-it").mkdir(parents=True)
    assert CB.repo_cached(cache, "google/gemma-3-1b-it") and not CB.repo_cached(cache, "Qwen/Qwen3.5-2B")
    assert CB.dir_size_gb(cache / "models--google--gemma-3-1b-it") == pytest.approx(2048 / 1e9)
    assert CB.purge_repo(cache, "google/gemma-3-1b-it") is True
    assert not (cache / "models--google--gemma-3-1b-it").exists() and not (cache / ".locks" / "models--google--gemma-3-1b-it").exists()
    assert CB.purge_repo(cache, "google/gemma-3-1b-it") is False, "nothing left to remove"
    assert capsys.readouterr().out.strip() == "[purge] google/gemma-3-1b-it (0.0 GB freed)"


def test_prefetch_plan_applies_the_disk_guard_and_prefetch_commands_pin_commits(chunk_by_id, pinned_cfg, tmp_path):
    s = KRP.parse_cuda_session("a800-sxm4-40gb:40x2")
    by = pinned_cfg["by_name"]
    cmds = CB.command_plan([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], pinned_cfg)
    cache = tmp_path / "hub"
    (cache / CB.repo_folder("Qwen/Qwen3.5-9B")).mkdir(parents=True)                 # already cached: not fetched again
    want = CB.prefetch_plan(cmds, "c05_gpu_t0", "c06_gpu_t0", pinned_cfg, s, cache, free_gb=500.0)
    assert "google/gemma-3-1b-it" not in want["repos"], "used by the running chunk: it stays, no fetch"
    assert "Qwen/Qwen3.5-9B" not in want["repos"] and want["repos"][0] == "Qwen/Qwen3.5-0.8B" and len(want["repos"]) == 10
    assert want["revisions"] == dict.fromkeys(want["repos"], TC.PINNED_REVISION) and want["skip"] is None
    # the guard: this chunk's repositories plus the fetched ones must leave DISK_HEADROOM_GB of the free space
    est = {c["repo"]: KRP.cuda_fit(by[c["model"]], s, pinned_cfg)[1] for c in cmds}
    need = sum(est[r] for r in ("google/gemma-3-1b-it", "Qwen/Qwen3.5-2B", "vinai/PhoGPT-4B-Chat")) + sum(est[r] for r in want["repos"])
    assert want["need_gb"] == pytest.approx(need) and 80 < need < 150
    tight = CB.prefetch_plan(cmds, "c05_gpu_t0", "c06_gpu_t0", pinned_cfg, s, cache, free_gb=need + 19.0)
    assert tight["skip"] == f"needs ~{need:.0f} GB, {need + 19:.0f} GB free"
    assert CB.prefetch_plan(cmds, "c05_gpu_t0", "c06_gpu_t0", pinned_cfg, s, cache, free_gb=need + 20.0)["skip"] is None
    # the current chunk's repositories count once each (the pilot's two lines share them)
    assert CB.repo_estimate_gb("google/gemma-3-1b-it", cmds, s, pinned_cfg) == pytest.approx(1.0 * 4 * 1.1 + 1.0)
    # argv: one process; a repository rides with its pinned commit, an unpinned one takes the default branch
    (cmd,) = CB.prefetch_commands(["a/b", "c/d"], {"a/b": TC.PINNED_REVISION, "c/d": "main"}, python="py")
    assert cmd[:2] == ["py", "-c"] and "snapshot_download" in cmd[2] and cmd[3:] == [f"a/b@{TC.PINNED_REVISION}", "c/d"]
    assert "'*.safetensors'" in cmd[2] and "'consolidated*'" in cmd[2]
    assert CB.prefetch_commands([], {}, "py") == []


def test_kaggle_push_command_stages_the_runs_and_the_compute_log():
    cmd = CB.kaggle_push_command(Path("/box/noilai"), Path("/box/stage"), "noilai-runs", "c06_gpu_t0/smoke_20/gemma-3-1b-it ok @ 036f603a", python="py")
    assert cmd == ["py", "scripts/kaggle_dataset.py", "--src", "data/runs", "--src", "data/compute_log.csv", "--dest", "/box/stage",
                   "--slug", "noilai-runs", "--message", "c06_gpu_t0/smoke_20/gemma-3-1b-it ok @ 036f603a"]
    cmd = CB.kaggle_push_command(Path("/box/noilai"), None, "noilai-runs", "m", create=True, python="py")
    assert cmd[-1] == "--create" and cmd[cmd.index("--dest") + 1] == "/box/noilai_runs_dataset", "the default staging dir is beside the repo"
    assert not CB.kaggle_credentials({}) and not CB.kaggle_credentials({"KAGGLE_USERNAME": "u"})
    assert CB.kaggle_credentials({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    assert CB.check_command("/r/x", "py") == ["py", "scripts/check_run.py", "--run", "/r/x"]


# ------------------------------------------------------------------ the driver's loop on the stub project


def test_run_chunks_writes_the_ledgers_unit_directories_and_purges_per_chunk(tmp_path, plan, pinned_cfg, chunk_by_id, capsys):
    proj = _stub_project(tmp_path, plan)
    cache = tmp_path / "hub"
    s = KRP.parse_cuda_session("rtx3090:24")
    common = {"platform": "rented", "plan": plan, "models_cfg": pinned_cfg, "project_root": proj, "log_path": proj / "log.csv",
              "verify": False, "check": False, "prefetch": False, "cache_dir": cache}
    # c22 restricted to gemma-3-4b-it: ten lines, two of them the p0p1 split of a tagged job
    res = CB.run_chunks([chunk_by_id["c22_gpu_t3"]], s, models=["gemma-3-4b-it"], **common)
    assert len(res) == 10 and all(r["status"] == "ok" and r["chunk"] == "c22_gpu_t3" for r in res), [(r["run"], r["status"]) for r in res]
    first = Path(res[0]["out"])
    assert first.name == "E1_main__gemma-3-4b-it__p0p1", "the tag comes from the job: the directory is the ledger's unit"
    assert json.loads((first / "manifest.json").read_text())["paraphrases"] == ["p0", "p1"]
    assert Path(res[1]["out"]).name == "E1_explicit_input__gemma-3-4b-it", "an untagged job writes <run>__<model>"
    assert Path(res[5]["out"]).name == "E1_attested__gemma-3-4b-it__p0p1"
    assert {r.platform for r in CL.read_entries(proj / "log.csv")} == {"rented"}
    out = capsys.readouterr().out
    assert "[chunk] c22_gpu_t3 (order 22, tier 3, queue gpu): 10 jobs" in out
    assert "[chunk] c22_gpu_t3 done: 10 commands, 0 failed, 0 refused, 0 skipped, 0 gated" in out
    # two chunks sharing a repository: the shared one survives the first chunk and goes after its last use in the second
    for repo in ("google/gemma-3-1b-it", "Qwen/Qwen3.5-0.8B"):
        (cache / CB.repo_folder(repo) / "snapshots").mkdir(parents=True)
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, models=["gemma-3-1b-it", "qwen3.5-0.8b"], **common)
    assert [(r["chunk"], r["run"], r["model"], r["status"], r.get("purged")) for r in res] == [
        ("c05_gpu_t0", "pilot_t1_200", "gemma-3-1b-it", "ok", None), ("c05_gpu_t0", "pilot_xcopa_200", "gemma-3-1b-it", "ok", None),
        ("c06_gpu_t0", "smoke_20", "gemma-3-1b-it", "ok", "google/gemma-3-1b-it"), ("c06_gpu_t0", "smoke_20", "qwen3.5-0.8b", "ok", "Qwen/Qwen3.5-0.8B")]
    assert not list(cache.iterdir()), "both repositories are gone at the end"
    assert Path(res[0]["out"]).name == "pilot_t1_200__gemma-3-1b-it"
    out = capsys.readouterr().out
    assert "[chunk] c05_gpu_t0 (order 5, tier 0, queue gpu): 2 jobs" in out and "[purge] google/gemma-3-1b-it (0.0 GB freed)" in out
    # a job with no model left is skipped silently; --keep-weights (purge=False) leaves the cache alone
    (cache / CB.repo_folder("Qwen/Qwen3.5-0.8B")).mkdir(parents=True)
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, models=["qwen3.5-0.8b"], purge=False, **common)
    assert [(r["chunk"], r["model"]) for r in res] == [("c06_gpu_t0", "qwen3.5-0.8b")] and "purged" not in res[0]
    assert (cache / CB.repo_folder("Qwen/Qwen3.5-0.8B")).exists()
    assert "[chunk] c05_gpu_t0 (order 5, tier 0, queue gpu): 0 jobs" in capsys.readouterr().out
    # a dry run executes, purges and fetches nothing and says what it would do (the 0.8B is not cached: it would be fetched)
    shutil.rmtree(cache / CB.repo_folder("Qwen/Qwen3.5-0.8B"))
    (cache / CB.repo_folder("google/gemma-3-1b-it")).mkdir(parents=True)
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, models=["gemma-3-1b-it", "qwen3.5-0.8b"],
                        **{**common, "prefetch": True}, dry_run=True, free_gb=500.0)
    out = capsys.readouterr().out
    assert all(r["status"] == "dry-run" for r in res) and (cache / CB.repo_folder("google/gemma-3-1b-it")).exists()
    assert "[purge] (dry-run) google/gemma-3-1b-it" in out and "[purge] (dry-run) Qwen/Qwen3.5-0.8B" in out and "GB freed" not in out
    assert "[prefetch] (dry-run) would fetch for c06_gpu_t0: Qwen/Qwen3.5-0.8B" in out and "[prefetch] started" not in out
    assert not (proj / "data" / "runs" / "_prefetch.log").exists()


def test_hooks_check_sync_and_push_after_each_executed_model_and_skip_cleanly(tmp_path, plan, pinned_cfg, chunk_by_id, capsys, monkeypatch):
    proj = _stub_project(tmp_path, plan)
    s = KRP.parse_cuda_session("rtx3090:24")
    marker = tmp_path / "synced"
    sync = f"{sys.executable} -c \"open({str(marker)!r}, 'a').write('x')\""
    common = {"platform": "rented", "plan": plan, "models_cfg": pinned_cfg, "project_root": proj, "log_path": proj / "log.csv",
              "verify": False, "prefetch": False, "cache_dir": tmp_path / "hub"}
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    res = CB.run_chunks([chunk_by_id["c06_gpu_t0"]], s, models=["gemma-3-1b-it", "qwen3.5-0.8b"], sync_cmd=sync,
                        kaggle_dataset="noilai-runs", create_dataset=True, **common)
    out = capsys.readouterr().out
    assert [r["status"] for r in res] == ["ok", "ok"]
    assert all(r["checked"] is None for r in res) and out.count("check_run skipped") == 1, "no check_run.py in the stub project: said once"
    assert marker.read_text() == "xx", "one background sync per executed model, all waited for by the end"
    assert out.count("[sync ] started") == 2 and out.count("[sync ] rc=0") == 2
    assert out.count("KAGGLE_USERNAME / KAGGLE_KEY are not set") == 1 and "[kaggle] started" not in out
    # with credentials the push is a background child of scripts/kaggle_dataset.py (absent here: its failure is printed, never fatal)
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    res = CB.run_chunks([chunk_by_id["c06_gpu_t0"]], s, models=["gemma-3-1b-it"], kaggle_dataset="noilai-runs", create_dataset=True,
                        staging=tmp_path / "stage", **common)
    out = capsys.readouterr().out
    assert res[0]["status"] == "ok"
    assert "[kaggle] started: c06_gpu_t0/smoke_20/gemma-3-1b-it ok @ nogit" in out and "[kaggle] rc=2" in out
    # a check_run.py in the project runs after each ok model; its verdict is recorded and a failure never stops the loop
    (proj / "scripts" / "check_run.py").write_text("import sys\nprint('checked', sys.argv[-1])\nsys.exit(0 if 'gemma' in sys.argv[-1] else 3)\n")
    res = CB.run_chunks([chunk_by_id["c06_gpu_t0"]], s, models=["gemma-3-1b-it", "qwen3.5-0.8b"], **common)
    out = capsys.readouterr().out
    assert [(r["status"], r["checked"]) for r in res] == [("ok", True), ("ok", False)]
    assert "[check] checked" in out and "[check] FAILED" in out and "smoke_20__qwen3.5-0.8b (rc=3)" in out
    assert CB.run_check(proj / "data" / "runs" / "smoke_20__gemma-3-1b-it", proj, sys.executable) is True


def test_prefetch_runs_one_background_process_for_the_next_chunk(tmp_path, plan, pinned_cfg, chunk_by_id, capsys, monkeypatch):
    proj = _stub_project(tmp_path, plan)
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(CB, "PREFETCH_SCRIPT", "import sys\nprint('fetched', *sys.argv[1:])\n")
    monkeypatch.setattr(CB, "prefetch_available", lambda python: True)
    s = KRP.parse_cuda_session("rtx3090:24")
    common = {"platform": "rented", "plan": plan, "models_cfg": pinned_cfg, "project_root": proj, "log_path": proj / "log.csv",
              "verify": False, "check": False, "cache_dir": cache, "models": ["gemma-3-1b-it", "qwen3.5-0.8b"]}
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, free_gb=500.0, **common)
    out = capsys.readouterr().out
    assert [r["status"] for r in res] == ["ok"] * 4
    assert "[prefetch] started: c06_gpu_t0: Qwen/Qwen3.5-0.8B" in out, "c06's repository that c05 does not use"
    assert out.index("[prefetch] rc=0") < out.index("[chunk] c06_gpu_t0"), "the prefetch lands before the next chunk starts"
    log = (proj / "data" / "runs" / "_prefetch.log").read_text()
    assert f"fetched Qwen/Qwen3.5-0.8B@{TC.PINNED_REVISION}" in log, "the pinned commit rides along"
    # too little free space (5.4 GB of gemma-3-1b-it + 2.8 GB to fetch must leave 20 GB): no process; none either when nothing is missing
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, free_gb=25.0, **common)
    out = capsys.readouterr().out
    assert "[prefetch] skipped for c06_gpu_t0: needs ~8 GB, 25 GB free" in out and "[prefetch] started" not in out
    (cache / CB.repo_folder("Qwen/Qwen3.5-0.8B")).mkdir(parents=True)
    CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, free_gb=500.0, purge=False, **common)
    assert "[prefetch]" not in capsys.readouterr().out
    # a spent session budget: no model starts, so nothing is fetched for the next chunk either
    res = CB.run_chunks([chunk_by_id["c05_gpu_t0"], chunk_by_id["c06_gpu_t0"]], s, free_gb=500.0, max_hours=0.0, **common)
    out = capsys.readouterr().out
    assert all(r["status"].startswith("skipped: session budget of 0.0 h reached") for r in res) and len(res) == 4
    assert "[prefetch] skipped for c06_gpu_t0: session budget of 0.0 h reached" in out and "[prefetch] started" not in out
    # the pre-registration gate holds every command of the next chunk (gates.yaml all null, as on the box today): no download
    (proj / "experiments").mkdir()
    shutil.copy(ROOT / "experiments" / "gates.yaml", proj / "experiments" / "gates.yaml")
    gates = KRP.gate_status(proj)
    assert gates["present"] and CB.chunk_gated(chunk_by_id["c08_gpu_t1"], plan, pinned_cfg, gates, ["gemma-3-1b-it"])
    assert not CB.chunk_gated(chunk_by_id["c06_gpu_t0"], plan, pinned_cfg, gates) and not CB.chunk_gated(chunk_by_id["c07_tpu_t0"], plan, pinned_cfg, gates, ["gemma-3-1b-it"])
    assert not CB.chunk_gated(chunk_by_id["c08_gpu_t1"], plan, pinned_cfg, {"present": False})
    assert not (cache / CB.repo_folder("Qwen/Qwen3.5-0.8B")).exists(), "the purge point of a skipped command still holds"
    res = CB.run_chunks([chunk_by_id["c07_tpu_t0"], chunk_by_id["c08_gpu_t1"]], s, free_gb=500.0, **{**common, "models": ["gemma-3-1b-it"]})
    out = capsys.readouterr().out
    assert [r["chunk"] for r in res] == ["c08_gpu_t1"] * 8 and all(r["status"].startswith("skipped: gate:") for r in res)
    assert CB.status_counts(res)["gated"] == 8 and "[chunk] c08_gpu_t1 done: 8 commands, 0 failed, 0 refused, 8 skipped, 8 gated" in out
    assert "[prefetch] skipped for c08_gpu_t1: every command gated" in out and "[prefetch] started" not in out
    assert not (cache / CB.repo_folder("google/gemma-3-1b-it")).exists()
    # a dry run on the same chunks says the same and counts the gated commands although their status stays `dry-run`
    res = CB.run_chunks([chunk_by_id["c07_tpu_t0"], chunk_by_id["c08_gpu_t1"]], s, free_gb=500.0, dry_run=True, **{**common, "models": ["gemma-3-1b-it"]})
    out = capsys.readouterr().out
    assert all(r["status"] == "dry-run" and r["prereg_gate"] for r in res) and CB.status_counts(res)["gated"] == 8
    assert "[prefetch] skipped for c08_gpu_t1: every command gated" in out and "would fetch" not in out


# ------------------------------------------------------------------ lanes: the driver schedules the cards


def test_schedule_balances_one_card_chunks_over_lanes_and_runs_27b_chunks_exclusively(chunk_spec, models_cfg):
    s = KRP.parse_cuda_session("a800-sxm4-40gb:40x2")
    chunks = CB.select_chunks(chunk_spec, ["from:c08_gpu_t1"])
    by_id = {c["id"]: c for c in chunks}
    phases = CB.schedule(chunks, s, models_cfg, 2)
    assert [(p["tier"], p["kind"]) for p in phases] == [(1, "lanes"), (1, "exclusive"), (2, "lanes"), (2, "exclusive"), (3, "lanes"),
                                                        (3, "exclusive"), (4, "lanes"), (5, "lanes"), (5, "exclusive")], "tiers in order, barriers"
    exclusive = [c for p in phases if p["kind"] == "exclusive" for c in p["chunks"]]
    assert exclusive == ["c16_tpu_t1", "c20_tpu_t2", "c21_tpu_t2", "c26_tpu_t3", "c31_tpu_t5"], "a 27B bf16 model (~60 GB) needs both cards"
    laned = [c for p in phases if p["kind"] == "lanes" for lane in p["lanes"] for c in lane]
    assert sorted(laned + exclusive) == sorted(by_id) and len(laned) + len(exclusive) == len(chunks), "every chunk exactly once"
    assert "c29_tpu_t4" in laned, "gemma-3-12b-it (27.8 GB) fits one 36 GB card"
    tier1 = phases[0]
    hours = [sum(by_id[c]["hours"]["high"] for c in lane) for lane in tier1["lanes"]]
    assert len(hours) == 2 and abs(hours[0] - hours[1]) <= 6.62, hours
    assert all(lane == sorted(lane, key=lambda c: by_id[c]["order"]) for lane in tier1["lanes"]), "each lane back in chunk order"
    assert tier1["shared"] == [], "tier 1's chunks are one model each"
    assert all(not CB.chunk_is_big(by_id[c], s, models_cfg) for c in laned) and all(CB.chunk_is_big(by_id[c], s, models_cfg) for c in exclusive)
    # tier 0: c05 and c06 in lanes (sharing the three pilot models' repositories), c07 exclusive
    tier0 = CB.select_chunks(chunk_spec, ["tier0"])
    assert CB.schedule(tier0, s, models_cfg, 2) == [
        {"kind": "lanes", "tier": 0, "lanes": [["c05_gpu_t0"], ["c06_gpu_t0"]],
         "shared": ["Qwen/Qwen3.5-2B", "google/gemma-3-1b-it", "vinai/PhoGPT-4B-Chat"]},
        {"kind": "exclusive", "tier": 0, "chunks": ["c07_tpu_t0"]}]
    # one lane, or one card: one exclusive phase per tier with every chunk
    whole = [{"kind": "exclusive", "tier": 0, "chunks": ["c05_gpu_t0", "c06_gpu_t0", "c07_tpu_t0"]}]
    assert CB.schedule(tier0, s, models_cfg, 1) == whole
    assert CB.schedule(tier0, KRP.parse_cuda_session("a800-sxm4-40gb:40"), models_cfg, 2) == whole
    # the greedy balance: largest first onto the least-loaded lane
    lanes = CB.balance_lanes([{"id": "a", "order": 1, "hours": {"high": 1}}, {"id": "b", "order": 2, "hours": {"high": 3}},
                              {"id": "c", "order": 3, "hours": {"high": 2}}], 2)
    assert [[c["id"] for c in lane] for lane in lanes] == [["b"], ["a", "c"]]


def test_lane_command_forwards_the_parents_flags_to_a_one_card_child():
    s = KRP.parse_cuda_session("a800-sxm4-40gb:40x2")
    cmd = CB.lane_command(["c05_gpu_t0", "c08_gpu_t1"], s, 1, platform="rented", python="py", check=False, keep_weights=True, prefetch=False,
                          verify=False, allow_unpinned_revision=True, extra="--n 2", models=["gemma-3-1b-it"], log=Path("/l.csv"),
                          max_hours=2.5, keep_repos=["google/gemma-3-1b-it"], sync_cmd="echo s", kaggle_dataset="noilai-runs",
                          create_dataset=True, staging=Path("/stage"), headroom_gb=40.0)
    assert cmd == ["py", "scripts/cuda_box.py", "--session", "a800-sxm4-40gb:40x1", "--chunks", "c05_gpu_t0", "c08_gpu_t1",
                   "--platform", "rented", "--lanes", "1", "--no-check", "--keep-weights", "--no-prefetch", "--no-verify",
                   "--allow-unpinned-revision", "--extra", "--n 2", "--models", "gemma-3-1b-it", "--log", "/l.csv", "--python", "py",
                   "--max-hours", "2.500", "--keep-repos", "google/gemma-3-1b-it", "--disk-headroom-gb", "40", "--sync-cmd", "echo s",
                   "--kaggle-dataset", "noilai-runs", "--create-dataset", "--staging", "/stage"]
    # an exclusive child takes the full session; the defaults forward nothing but the python
    cmd = CB.lane_command(["c07_tpu_t0"], KRP.parse_cuda_session("v100-32gb:32x2:nobf16"), 2, python="py")
    assert cmd == ["py", "scripts/cuda_box.py", "--session", "v100-32gb:32x2:nobf16", "--chunks", "c07_tpu_t0", "--platform", "rented",
                   "--lanes", "1", "--python", "py"]


def test_run_lanes_starts_one_child_per_card_and_an_exclusive_child_with_the_full_session(tmp_path, chunk_spec, models_cfg, capsys, monkeypatch):
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    (proj / "scripts" / "fake_box.py").write_text(
        "import json, os, sys\nprint(json.dumps({'argv': sys.argv[1:], 'cvd': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "sys.exit(1 if 'c07_tpu_t0' in sys.argv else 0)\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    s = KRP.parse_cuda_session("a800-sxm4-40gb:40x2")
    rc = CB.run_lanes(CB.select_chunks(chunk_spec, ["tier0"]), s, models_cfg, 2, platform="rented", project_root=proj, python=sys.executable,
                      max_hours=5.0, kaggle_dataset="noilai-runs", create_dataset=True, script="scripts/fake_box.py", check=False)
    out = capsys.readouterr().out
    assert rc == 1, "the exclusive child exited non-zero"
    logs = {p.name: json.loads(p.read_text().strip().splitlines()[-1]) for p in (proj / "data" / "runs").glob("_box_*.log")}
    assert set(logs) == {"_box_t0-lanes_0.log", "_box_t0-lanes_1.log", "_box_t0-exclusive_0.log"}
    a, b, x = (logs[f"_box_t0-{k}.log"] for k in ("lanes_0", "lanes_1", "exclusive_0"))
    assert (a["cvd"], b["cvd"], x["cvd"]) == ("0", "1", None), "one card per lane, every card for the exclusive child"
    for argv, chunks, spec in ((a["argv"], "c05_gpu_t0", "a800-sxm4-40gb:40x1"), (b["argv"], "c06_gpu_t0", "a800-sxm4-40gb:40x1"),
                               (x["argv"], "c07_tpu_t0", "a800-sxm4-40gb:40x2")):
        joined = " ".join(argv)
        assert f"--session {spec} --chunks {chunks} --platform rented --lanes 1 --no-check" in joined and "--max-hours" in joined, joined
    assert "--keep-repos Qwen/Qwen3.5-2B google/gemma-3-1b-it vinai/PhoGPT-4B-Chat" in " ".join(a["argv"]) and "--keep-repos" in " ".join(b["argv"])
    assert "--disk-headroom-gb 40" in " ".join(a["argv"]) and "--disk-headroom-gb 40" in " ".join(b["argv"]), "two lanes: twice the head-room"
    assert "--disk-headroom-gb" not in x["argv"] and "--keep-repos" not in x["argv"]
    assert "--kaggle-dataset noilai-runs --create-dataset" in " ".join(a["argv"]), "lane 0 pushes, and creates the dataset once"
    assert "--kaggle-dataset" not in " ".join(b["argv"]) and "--kaggle-dataset noilai-runs" in " ".join(x["argv"]) and "--create-dataset" not in x["argv"]
    assert "[phase] tier 0 lanes: A=[c05_gpu_t0] B=[c06_gpu_t0]" in out and "[phase] tier 0 exclusive: [c07_tpu_t0]" in out
    assert "[lane 0] rc=0: {" in out and "[lane 1] rc=0: {" in out and "[lane 0] rc=1: {" in out, "each child's last log line is printed"
    # a spent budget launches no phase
    rc = CB.run_lanes(CB.select_chunks(chunk_spec, ["tier0"]), s, models_cfg, 2, project_root=proj, python=sys.executable,
                      max_hours=0.0, script="scripts/fake_box.py")
    assert rc == 0 and "session budget of 0.0 h reached" in capsys.readouterr().out
    # the operator's --keep-repos reach every child (with the shared ones in a lane); the final push's failure is printed, never fatal
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    rc = CB.run_lanes(CB.select_chunks(chunk_spec, ["tier0"]), s, models_cfg, 2, project_root=proj, python=sys.executable,
                      models=["gemma-3-1b-it", "qwen3.8-27b"], kaggle_dataset="noilai-runs", keep_repos=["x/kept"], script="scripts/fake_box.py")
    out = capsys.readouterr().out
    logs = {p.name: json.loads(p.read_text().strip().splitlines()[-1]) for p in (proj / "data" / "runs").glob("_box_*.log")}
    a, x = logs["_box_t0-lanes_0.log"], logs["_box_t0-exclusive_0.log"]
    assert "--keep-repos google/gemma-3-1b-it x/kept" in " ".join(a["argv"]) and "--keep-repos x/kept --sync-cmd" not in " ".join(x["argv"])
    assert x["argv"][x["argv"].index("--keep-repos") + 1] == "x/kept" and "--models gemma-3-1b-it qwen3.8-27b" in " ".join(x["argv"])
    assert "--chunks c07_tpu_t0" in " ".join(x["argv"]), "the 27B model keeps c07 exclusive"
    assert rc == 1 and "[kaggle] final push rc=2" in out, "no scripts/kaggle_dataset.py in the stub project: rc 2 printed; exit 1 is c07's child"
    (proj / "scripts" / "fake_box.py").write_text("import sys\nprint('ok')\n")
    rc = CB.run_lanes(CB.select_chunks(chunk_spec, ["c05_gpu_t0", "c06_gpu_t0"]), s, models_cfg, 2, project_root=proj,
                      python=sys.executable, kaggle_dataset="noilai-runs", script="scripts/fake_box.py")
    assert rc == 0 and "[kaggle] final push rc=2" in capsys.readouterr().out, "every child exited 0: the failed push does not change the exit code"


# ------------------------------------------------------------------ the CLI in chunk mode


def test_cli_dry_run_in_chunk_mode_prints_the_chunks_the_fit_and_no_purge(capsys):
    rc = CB.main(["--dry-run", "--chunks", "tier0", "--session", "a800-sxm4-40gb:40x2"])
    out = capsys.readouterr().out
    assert rc == 0
    for needle in ("[session] cuda a800-sxm4-40gb:40x2", "[chunks] c05_gpu_t0 c06_gpu_t0 c07_tpu_t0",
                   "[chunk] c05_gpu_t0 (order 5, tier 0, queue gpu): 2 jobs", "[chunk] c06_gpu_t0 (order 6, tier 0, queue gpu): 1 jobs",
                   "[chunk] c07_tpu_t0 (order 7, tier 0, queue tpu): 1 jobs", "[summary] 23 commands, 0 failed, 0 refused, 0 skipped, 0 gated"):
        assert needle in out, needle
    cmds = {ln.split("--model-config ")[1].split()[0]: ln for ln in out.splitlines() if ln.startswith("[cmd  ]")}
    assert "--tensor-parallel-size 2" in cmds["qwen3.8-27b"] and "smoke_20_tpu__qwen3.8-27b" in cmds["qwen3.8-27b"]
    assert "--tensor-parallel-size 2" in cmds["qwen-sea-lion-v4.5-27b-it"]
    assert "--tensor-parallel-size 1" in cmds["gemma-3-12b-it"], "27.8 GB fits one 36 GB card"
    assert "GB freed" not in out and "[prefetch] started" not in out and "[purge] (dry-run) google/gemma-3-1b-it" in out
    assert out.index("[purge] (dry-run) google/gemma-3-1b-it") > out.index("[chunk] c06_gpu_t0"), "c05's repositories stay for c06"
    # the default is every gpu/tpu chunk (c05..c31); a chunk without a selected model has no job
    rc = CB.main(["--dry-run", "--session", "a800-sxm4-40gb:40x2", "--models", "gemma-3-1b-it"])
    out = capsys.readouterr().out
    assert rc == 0 and "[chunks] c05_gpu_t0 c06_gpu_t0 c07_tpu_t0 c08_gpu_t1" in out and out.count("[chunk] c") == 27 * 2
    assert "[chunk] c07_tpu_t0 (order 7, tier 0, queue tpu): 0 jobs" in out and "[chunk] c08_gpu_t1 (order 8, tier 1, queue gpu): 8 jobs" in out
    # the gate (experiments/gates.yaml all null today): a dry run counts c08's commands as gated and prefetches nothing for it
    rc = CB.main(["--dry-run", "--chunks", "c07_tpu_t0", "c08_gpu_t1", "--session", "a800-sxm4-40gb:40x2"])
    out = capsys.readouterr().out
    assert rc == 0 and "[chunk] c08_gpu_t1 done: 8 commands, 0 failed, 0 refused, 0 skipped, 8 gated" in out
    assert "[summary] 11 commands, 0 failed, 0 refused, 0 skipped, 8 gated" in out and out.count("[gate ] gemma-3-1b-it:") == 8
    assert "[prefetch] skipped for c08_gpu_t1: every command gated" in out and "would fetch" not in out
    # a tier no chunk has, and an empty --models, are refused
    with pytest.raises(SystemExit):
        CB.main(["--dry-run", "--session", "rtx3090:24", "--chunks", "tier9"])
    assert "tier9: no gpu/tpu chunk has tier 9" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        CB.main(["--dry-run", "--session", "rtx3090:24", "--chunks", "c05_gpu_t0", "--models"])
    assert "--models needs at least one entry" in capsys.readouterr().err
    # --runs and --chunks exclude each other; a CPU chunk is refused with the reason
    with pytest.raises(SystemExit):
        CB.main(["--dry-run", "--session", "rtx3090:24", "--runs", "E1_main", "--chunks", "c05_gpu_t0"])
    assert "mutually exclusive" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        CB.main(["--dry-run", "--session", "rtx3090:24", "--chunks", "c01_cpu_t0"])
    assert "CPU chunks run the HF backend" in capsys.readouterr().err


def test_cli_dry_run_with_lanes_prints_the_schedule_and_the_child_commands(capsys):
    rc = CB.main(["--dry-run", "--lanes", "2", "--chunks", "tier0", "--session", "a800-sxm4-40gb:40x2", "--kaggle-dataset", "noilai-runs",
                  "--create-dataset", "--no-check"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[phase] tier 0 lanes: A=[c05_gpu_t0] B=[c06_gpu_t0] (shared: Qwen/Qwen3.5-2B, google/gemma-3-1b-it, vinai/PhoGPT-4B-Chat)" in out
    assert "[phase] tier 0 exclusive: [c07_tpu_t0]" in out
    lanes = [ln for ln in out.splitlines() if ln.startswith("[lane ") and "scripts/cuda_box.py" in ln]
    assert len(lanes) == 3
    assert "--session a800-sxm4-40gb:40x1 --chunks c05_gpu_t0 --platform rented --lanes 1 --no-check" in lanes[0]
    assert "--keep-repos Qwen/Qwen3.5-2B google/gemma-3-1b-it vinai/PhoGPT-4B-Chat --disk-headroom-gb 40 --kaggle-dataset noilai-runs --create-dataset" in lanes[0]
    assert "--chunks c06_gpu_t0" in lanes[1] and "--kaggle-dataset" not in lanes[1] and "--keep-repos" in lanes[1]
    assert "--session a800-sxm4-40gb:40x2 --chunks c07_tpu_t0" in lanes[2] and "--kaggle-dataset noilai-runs" in lanes[2]
    assert "--create-dataset" not in lanes[2], "only the first child creates the dataset"
    assert "_box_t0-lanes_0.log" in out and "_box_t0-lanes_1.log" in out and "_box_t0-exclusive_0.log" in out
    assert "[plan ]" not in out and "[cmd  ]" not in out, "the parent runs nothing on a dry run"
    # a lane child's argv parses back into this script: one card, tp=1 everywhere, the shared repository kept
    child = shlex.split(lanes[1].split("] ", 1)[1])
    rc = CB.main([*child[2:], "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "[session] cuda a800-sxm4-40gb:40x1" in out and "[chunk] c06_gpu_t0 (order 6, tier 0, queue gpu): 1 jobs" in out
    assert "--tensor-parallel-size 2" not in out and "[purge] (dry-run) Qwen/Qwen3.5-0.8B" in out
    assert "[purge] (dry-run) google/gemma-3-1b-it" not in out, "a repository shared with the other lane is never purged"
    # one card: --lanes 2 falls back to one process
    rc = CB.main(["--dry-run", "--lanes", "2", "--chunks", "c05_gpu_t0", "--session", "rtx3090:24", "--models", "gemma-3-1b-it"])
    out = capsys.readouterr().out
    assert rc == 0 and "needs >= 2 cards" in out and "[chunk] c05_gpu_t0" in out
    # more lanes than cards: one lane per card (a third lane would have no card); the operator's --keep-repos ride along
    rc = CB.main(["--dry-run", "--lanes", "3", "--chunks", "c08_gpu_t1", "c09_gpu_t1", "c10_gpu_t1", "--session", "a800-sxm4-40gb:40x2",
                  "--no-check", "--keep-repos", "x/kept", "--disk-headroom-gb", "25"])
    out = capsys.readouterr().out
    lanes = [ln for ln in out.splitlines() if ln.startswith("[lane ") and "scripts/cuda_box.py" in ln]
    assert rc == 0 and "[note ] --lanes 3 > 2 cards: using 2 lanes" in out and len(lanes) == 2, out
    assert "[phase] tier 1 lanes: A=[" in out and "C=[" not in out and "CUDA_VISIBLE_DEVICES" not in out
    assert all("--keep-repos x/kept --disk-headroom-gb 50" in ln for ln in lanes), lanes
