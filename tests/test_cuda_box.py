"""A rented CUDA box as a run session: the planner decides what fits by VRAM and bf16 support, logs the
real device, and a one-command driver runs the plan's self-hosted lines in cut order.

Everything here runs offline: the session parser, the fit rule and the driver's command construction
are exercised against configs/models.yaml and a stub run_eval.py in a temporary project root.
"""
import copy
import json
import sys
import tomllib
from pathlib import Path

import pytest

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


# ------------------------------------------------------------------ the box driver


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
