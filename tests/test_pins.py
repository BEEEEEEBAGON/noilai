"""Panel pins (DD 7.1, PREREG §7): the pin kit, the hashed panel manifest and the run-environment pins."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pin_panel as PP  # noqa: E402


def test_committed_manifest_hash_is_intact_and_matches_models_yaml():
    man = json.loads((ROOT / "configs" / "panel_manifest.json").read_text(encoding="utf-8"))
    assert PP.manifest_hash(man) == man["manifest_sha256"]
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert PP.verify(man, cfg) == []
    names = {e["name"] for e in PP.self_hosted(cfg)}
    assert {e["name"] for e in man["entries"]} == names and "gpt-oss-120b" not in names and "gemini-flash" not in names


def test_apply_revisions_edits_only_the_revision_values():
    text = ("models:\n  - name: a-model   # comment\n    hf_id: x/a\n    revision: null            # keep me\n"
            "  - name: b-model\n    revision: null\n"
            "variants:\n  - {name: c-model, hf_id: x/c, revision: null, gated: true}\n")
    sha = "a" * 40
    new, changed = PP.apply_revisions(text, {"a-model": sha, "c-model": "b" * 40})
    assert sorted(changed) == ["a-model", "c-model"]
    assert f"revision: {sha}            # keep me" in new and "  - name: b-model\n    revision: null\n" in new
    assert "{name: c-model, hf_id: x/c, revision: " + "b" * 40 + ", gated: true}" in new
    assert len(new.splitlines()) == len(text.splitlines())


def test_resolve_entry_against_a_stand_in_hub(tmp_path):
    tok = tmp_path / "tokenizer_config.json"
    tok.write_text(json.dumps({"chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"}))
    spm = tmp_path / "tokenizer.model"
    spm.write_bytes(b"\x00spm")
    info = SimpleNamespace(sha="c" * 40, gated="manual", card_data={"license": "gemma"}, tags=["license:gemma", "text-generation"],
                           last_modified="2025-03-12", siblings=[SimpleNamespace(rfilename=n) for n in ("tokenizer_config.json", "tokenizer.model", "model.safetensors")])
    api = SimpleNamespace(model_info=lambda repo: info)
    files = {"tokenizer_config.json": tok, "tokenizer.model": spm}

    def download(repo, fn, rev):
        assert rev == "c" * 40
        return files[fn]
    rec = PP.resolve_entry({"name": "g", "hf_id": "google/gemma-3-1b-it", "revision": None, "gated": True}, api, download)
    assert rec["status"] == "pinned" and rec["revision"] == "c" * 40 and rec["license"]["card"] == "gemma"
    assert set(rec["tokenizer_files"]) == {"tokenizer_config.json", "tokenizer.model"} and rec["chat_template_sha256"]

    def boom(repo):
        raise PermissionError("gated: request access")
    bad = PP.resolve_entry({"name": "g", "hf_id": "x/y", "revision": None}, SimpleNamespace(model_info=boom), download)
    assert bad["status"] == "error" and "gated" in bad["error"]
    q = PP.resolve_entry({"name": "q", "hf_id": "x/q", "quantization": {"method": "awq", "bits": 4, "checkpoint": None}}, api, download)
    assert q["status"] == "unpinned" and "quantized checkpoint" in q["note"]


def test_verify_catches_a_tampered_manifest():
    cfg = {"models": [{"name": "m", "hf_id": "x/m", "revision": None, "backend": "hf", "hardware": "t4"}]}
    man = PP.build_manifest([{**PP.offline_entry(cfg["models"][0]), "status": "pinned", "revision": "d" * 40}], "hub")
    assert any("models.yaml revision" in p for p in PP.verify(man, cfg))
    man["n_pinned"] = 99
    assert any("manifest_sha256" in p for p in PP.verify(man, cfg))


def test_env_pins_agree_with_pyproject_and_the_notebooks():
    import tomllib
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s#]*)", (ROOT / "configs" / "env_pins.txt").read_text(encoding="utf-8"), re.MULTILINE))
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["vllm"] == [f"vllm=={pins['vllm']}"] and extras["llama_cpp"] == [f"llama-cpp-python=={pins['llama-cpp-python']}"]
    nb = (ROOT / "scripts" / "kaggle_build_notebooks.py").read_text(encoding="utf-8")
    assert f'VLLM_VERSION = "{pins["vllm"]}"' in nb
    lock = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", (ROOT / "requirements-lock.txt").read_text(encoding="utf-8"), re.MULTILINE))
    lock = {k.lower().replace("_", "-"): v for k, v in lock.items()}
    for name in ("transformers", "tokenizers", "safetensors", "sentencepiece", "accelerate", "numpy", "scipy", "statsmodels", "pandas"):
        assert pins[name] == lock[name], (name, pins[name], lock[name])
