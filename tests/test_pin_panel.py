"""Panel pinning (scripts/pin_panel.py, DESIGN_DECISIONS 7.1) and the CPU-only Kaggle job driver
(scripts/kaggle_cpu_jobs.py): resolution with a fake hub, the JSON record, the text-preserving
--apply rewrite of a copy of configs/models.yaml and its refusals, and the job planning / report
of the driver against the real configs, all without network access."""
import copy
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from huggingface_hub.errors import GatedRepoError, RemoteEntryNotFoundError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

KCJ = importlib.import_module("kaggle_cpu_jobs")   # scripts/kaggle_cpu_jobs.py (imported after the path insert)
PP = importlib.import_module("pin_panel")          # scripts/pin_panel.py

MODELS_YAML = ROOT / "configs" / "models.yaml"
PY = sys.executable


def _hub_error(cls, status):
    return cls("hub says no", response=httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x")))


def fake_sha(hf_id: str) -> str:
    return hashlib.sha1(hf_id.encode()).hexdigest()           # 40 hex: shaped like a commit hash


class _Sibling:
    def __init__(self, name):
        self.rfilename = name


class _Info:
    def __init__(self, sha, gated, files):
        self.sha, self.gated, self.siblings = sha, gated, [_Sibling(f) for f in files]


class FakeResolver:
    """A hub double: `behaviour[hf_id]` is 'ok', 'not_found', 'gated_info', 'gated_download', 'boom' or
    'no_siblings'; downloads write small files whose content depends on (hf_id, filename)."""

    def __init__(self, tmp: Path, behaviour=None, files=("tokenizer.model", "tokenizer.json", "tokenizer_config.json")):
        self.tmp, self.behaviour, self.files = tmp, behaviour or {}, list(files)
        self.calls = []

    def model_info(self, hf_id):
        self.calls.append(("info", hf_id))
        b = self.behaviour.get(hf_id, "ok")
        if b == "not_found":
            raise _hub_error(RepositoryNotFoundError, 404)
        if b == "gated_info":
            raise _hub_error(GatedRepoError, 403)
        if b == "boom":
            raise ConnectionError("proxy refused")
        if b == "no_siblings":
            info = _Info(fake_sha(hf_id), None, [])
            info.siblings = None
            return info
        return _Info(fake_sha(hf_id), "auto" if "gemma" in hf_id else False, [*self.files, "config.json"])

    def download(self, hf_id, filename, revision):
        self.calls.append(("download", hf_id, filename, revision))
        if self.behaviour.get(hf_id) == "gated_download":
            raise _hub_error(GatedRepoError, 403)
        if filename not in self.files:
            raise _hub_error(RemoteEntryNotFoundError, 404)         # what hf_hub_download raises for an absent file
        p = self.tmp / f"{hf_id.replace('/', '__')}__{filename}"
        p.write_text(f"{hf_id}:{filename}:{revision}")
        return str(p)


@pytest.fixture(scope="module")
def cfg():
    return PP.load_models(MODELS_YAML)


@pytest.fixture
def models_copy(tmp_path):
    p = tmp_path / "models.yaml"
    shutil.copy(MODELS_YAML, p)
    return p


# ------------------------------------------------------------------ selection and resolution
def test_selection_is_every_self_hosted_entry_with_an_hf_id(cfg):
    from noilai.eval import run as RN

    assert PP.LOCAL_WEIGHT_BACKENDS == RN.LOCAL_WEIGHT_BACKENDS      # DD 7.1: one list of weight-downloading backends
    entries = PP.select_entries(cfg)
    names = {m["name"] for m in entries}
    assert all(m["backend"] in PP.LOCAL_WEIGHT_BACKENDS and m["hf_id"] and "revision" in m for m in entries)
    panel = PP.panel_groups(cfg)
    assert panel == {"interpretability_ladder", "general_multilingual", "sea_specialised", "vietnamese_specialised",
                     "legacy_baseline", "large_open_reference", "api"}
    groups = {m["group"] for m in entries}
    assert groups == (panel - {"api"}) | {"bf16_reference", "reasoning_substudy"}
    assert {"gemma-3-1b-it", "phogpt-4b-chat", "gemma-3-1b-it--bf16", "qwen3.5-4b--thinking"} <= names
    assert not names & {"gemini-flash", "gpt-oss-120b", "gpt-oss-20b--thinking"}
    with_api = {m["name"] for m in PP.select_entries(cfg, include_api=True)}
    assert with_api - names == {"gpt-oss-120b", "gpt-oss-20b", "gpt-oss-120b--thinking", "gpt-oss-20b--thinking"}
    assert [m["name"] for m in PP.select_entries(cfg, names=["qwen3.5-2b"])] == ["qwen3.5-2b"]
    with pytest.raises(KeyError):
        PP.select_entries(cfg, names=["gemini-flash"])


def test_resolution_records_hash_tokenizer_hashes_and_the_three_outcomes(cfg, tmp_path):
    behaviour = {"google/gemma-3-4b-it": "gated_download", "Qwen/Qwen3.5-9B": "not_found",
                 "sail/Sailor2-8B-Chat": "boom", "vinai/PhoGPT-4B-Chat": "no_siblings",
                 "Viet-Mistral/Vistral-7B-Chat": "gated_info"}
    hub = FakeResolver(tmp_path, behaviour, files=["tokenizer.model", "tokenizer_config.json"])
    entries = PP.select_entries(cfg)
    records = PP.resolve_entries(entries, hub, panel=PP.panel_groups(cfg))
    by = {r["name"]: r for r in records}
    assert len(records) == len(entries)
    # the hub is asked once per distinct id, and the variants share their reference's answer
    assert sum(1 for c in hub.calls if c[0] == "info") == len({m["hf_id"] for m in entries})
    ok = by["gemma-3-1b-it"]
    assert ok["hf_id_status"] == PP.STATUS_FOUND and ok["revision"] == fake_sha("google/gemma-3-1b-it") and ok["error"] is None
    assert ok["panel"] is True and ok["config_hf_id_status"] == "known" and ok["config_revision"] is None
    assert ok["hub_gated"] == "auto" and ok["resolved_utc"].endswith("+00:00")
    assert set(ok["tokenizer_sha256"]) == {"tokenizer.model", "tokenizer_config.json"}       # only the files present
    expect = hashlib.sha256(f"google/gemma-3-1b-it:tokenizer.model:{ok['revision']}".encode()).hexdigest()
    assert ok["tokenizer_sha256"]["tokenizer.model"] == expect
    assert by["gemma-3-1b-it--bf16"]["revision"] == ok["revision"] and by["gemma-3-1b-it--bf16"]["panel"] is False
    gated = by["gemma-3-4b-it"]
    assert gated["hf_id_status"] == PP.STATUS_GATED and gated["revision"] == fake_sha("google/gemma-3-4b-it")
    assert gated["tokenizer_sha256"] == {} and "GatedRepoError" in gated["error"]
    assert by["vistral-7b-chat"]["hf_id_status"] == PP.STATUS_GATED and by["vistral-7b-chat"]["revision"] is None
    nf = by["qwen3.5-9b"]
    assert nf["hf_id_status"] == PP.STATUS_NOT_FOUND and nf["revision"] is None and "RepositoryNotFoundError" in nf["error"]
    assert by["sailor2-8b-chat"]["hf_id_status"] == PP.STATUS_ERROR and "ConnectionError" in by["sailor2-8b-chat"]["error"]
    # no sibling listing: every candidate file is tried, the absent one (EntryNotFoundError) is simply left out
    pho = by["phogpt-4b-chat"]
    assert pho["hf_id_status"] == PP.STATUS_FOUND and set(pho["tokenizer_sha256"]) == {"tokenizer.model", "tokenizer_config.json"}
    assert sum(1 for c in hub.calls if c[0] == "download" and c[1] == "vinai/PhoGPT-4B-Chat") == 3
    assert [PP.is_resolved(by[n]) for n in ("gemma-3-1b-it", "gemma-3-4b-it", "qwen3.5-9b", "sailor2-8b-chat")] == [True, True, False, False]
    # the JSON: one record per entry, counts, and the config it was resolved against
    out = tmp_path / "pins.json"
    doc = PP.write_pins(records, out, MODELS_YAML)
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back["n_records"] == len(entries) and back["n_resolved"] == doc["n_resolved"] == sum(PP.is_resolved(r) for r in records)
    assert back["n_unresolved_panel"] == 3           # qwen3.5-9b, sailor2, vistral (gated at download still has its hash)
    assert back["models_config"] == "configs/models.yaml" and len(back["models_config_sha256"]) == 64
    assert [r["name"] for r in back["records"]] == [m["name"] for m in entries]
    assert {"name", "hf_id", "revision", "tokenizer_sha256", "resolved_utc", "error", "hf_id_status"} <= set(back["records"][0])
    assert [r["name"] for r in PP.read_pins(out)] == [m["name"] for m in entries]


def test_a_short_or_missing_sha_is_an_error_not_a_pin(tmp_path):
    class Short(FakeResolver):
        def model_info(self, hf_id):
            return _Info("abc123", False, [])

    rec = PP.resolve_hf_id("org/m", Short(tmp_path))
    assert rec["hf_id_status"] == PP.STATUS_ERROR and rec["revision"] is None and "no full commit hash" in rec["error"]


# ------------------------------------------------------------------ apply
def _pins_for(cfg, tmp_path, behaviour=None):
    hub = FakeResolver(tmp_path, behaviour)
    records = PP.resolve_entries(PP.select_entries(cfg), hub, panel=PP.panel_groups(cfg))
    out = tmp_path / "pins.json"
    PP.write_pins(records, out, MODELS_YAML)
    return out, records


def test_apply_rewrites_only_the_revision_lines_and_keeps_comments(cfg, tmp_path, models_copy):
    pins, records = _pins_for(cfg, tmp_path)
    before = models_copy.read_text(encoding="utf-8")
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins)],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    after = models_copy.read_text(encoding="utf-8")
    new_cfg = yaml.safe_load(after)
    selected = PP.select_entries(cfg)
    assert len(new_cfg["models"]) == len(cfg["models"])
    for old, new in zip(cfg["models"], new_cfg["models"]):
        assert {k: v for k, v in old.items() if k != "revision"} == {k: v for k, v in new.items() if k != "revision"}
        if old["name"] in {m["name"] for m in selected}:
            assert new["revision"] == fake_sha(old["hf_id"]), old["name"]
        else:
            assert new.get("revision") == old.get("revision"), old["name"]
    assert {k: v for k, v in new_cfg.items() if k != "models"} == {k: v for k, v in cfg.items() if k != "models"}
    assert new_cfg["defaults"]["revision"] is None                   # the defaults block is not an entry
    # text-preserving: the same lines except the revision values; the per-line comments survive
    b_lines, a_lines = before.splitlines(), after.splitlines()
    assert len(b_lines) == len(a_lines)
    changed = [(i, b, a) for i, (b, a) in enumerate(zip(b_lines, a_lines)) if b != a]
    assert len(changed) == len(selected)
    for i, b, a in changed:
        assert "revision: null" in b and "revision: null" not in a and 'revision: "' in a
        assert b.replace("revision: null", "", 1) == a.replace(f'revision: "{fake_sha(_hf_id_of(cfg, b_lines, i))}"', "", 1)
    block = next(a for _, b, a in changed if a.lstrip().startswith("revision:"))
    assert block.rstrip().endswith("# full commit hash, required before a paper run (DD 7.1)")
    flow = next(a for _, b, a in changed if a.lstrip().startswith("- {name: gemma-3-1b-it--bf16"))
    assert f'revision: "{fake_sha("google/gemma-3-1b-it")}", gated: true' in flow
    assert after.count("[UNCERTAIN: verify]") == before.count("[UNCERTAIN: verify]")   # hf_id_status is never flipped
    # the config now passes the runner's pin check for every self-hosted entry
    from noilai.eval import run as RN

    for m in new_cfg["models"]:
        if m["backend"] in RN.LOCAL_WEIGHT_BACKENDS:
            assert RN.check_revision(m, m["backend"]) == "commit_hash", m["name"]
    # a second apply is a no-op that reports every entry as already at its hash
    plan = PP.apply_pins(models_copy, records)
    assert plan["written"] is True and plan["changes"] == {} and len(plan["unchanged"]) == len(selected)
    assert models_copy.read_text(encoding="utf-8") == after


def _hf_id_of(cfg, lines: list[str], idx: int) -> str:
    """The hf_id of the entry whose line `idx` changed: a flow line names it; a block `revision:` line
    (identical across entries) belongs to the nearest `- name:` line above it."""
    if lines[idx].lstrip().startswith("- {name:"):
        name = lines[idx].split("name:")[1].split(",")[0].strip()
    else:
        while not lines[idx].lstrip().startswith("- name:"):
            idx -= 1
        name = lines[idx].split("name:")[1].strip()
    return next(m["hf_id"] for m in cfg["models"] if m["name"] == name)


def test_apply_refuses_an_unresolved_panel_entry_unless_partial(cfg, tmp_path, models_copy):
    pins, _ = _pins_for(cfg, tmp_path, {"Qwen/Qwen3.5-2B": "not_found"})
    before = models_copy.read_text(encoding="utf-8")
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins)],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 1 and "qwen3.5-2b" in r.stderr and "--partial" in r.stderr
    assert models_copy.read_text(encoding="utf-8") == before
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins),
                        "--partial"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    new_cfg = {m["name"]: m for m in yaml.safe_load(models_copy.read_text(encoding="utf-8"))["models"]}
    assert new_cfg["qwen3.5-2b"]["revision"] is None and new_cfg["qwen3.5-2b--bf16"]["revision"] is None
    assert new_cfg["qwen3.5-4b"]["revision"] == fake_sha("Qwen/Qwen3.5-4B")
    # an unresolved NON-panel entry alone does not block the apply
    cfg2 = copy.deepcopy(cfg)
    records = PP.resolve_entries(PP.select_entries(cfg2), FakeResolver(tmp_path), panel=PP.panel_groups(cfg2))
    for rec in records:
        if rec["name"] == "qwen3.5-4b--thinking":
            rec["revision"], rec["hf_id_status"] = None, PP.STATUS_ERROR
    plan = PP.plan_apply(cfg2, records)
    assert plan["refusals"] == [] and plan["unresolved"] == ["qwen3.5-4b--thinking"] and plan["unresolved_panel"] == []


def test_apply_refuses_to_overwrite_a_different_existing_pin_unless_force(cfg, tmp_path, models_copy):
    pins, records = _pins_for(cfg, tmp_path)
    other = "f" * 40
    text = models_copy.read_text(encoding="utf-8")
    text = PP.rewrite_revisions(text, {"gemma-3-1b-it": other, "phogpt-4b-chat": fake_sha("vinai/PhoGPT-4B-Chat")})
    models_copy.write_text(text, encoding="utf-8")
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins)],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 1 and "gemma-3-1b-it" in r.stderr and "--force" in r.stderr and "phogpt" not in r.stderr
    assert models_copy.read_text(encoding="utf-8") == text                     # nothing written, not even the others
    plan = PP.plan_apply(yaml.safe_load(text), records)
    assert plan["conflicts"] == {"gemma-3-1b-it": (other, fake_sha("google/gemma-3-1b-it"))}
    assert "phogpt-4b-chat" in plan["unchanged"] and "phogpt-4b-chat" not in plan["changes"]
    # a Kaggle Models slug is a pin too (DD 7.1) and is protected the same way
    slug_cfg = yaml.safe_load(text)
    for m in slug_cfg["models"]:
        if m["name"] == "gemma-3-4b-it":
            m["revision"] = "google/gemma-3/transformers/gemma-3-4b-it/2"
    assert set(PP.plan_apply(slug_cfg, records)["conflicts"]) == {"gemma-3-1b-it", "gemma-3-4b-it"}
    assert PP.plan_apply(slug_cfg, records, force=True)["conflicts"] == {}
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins),
                        "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    new_cfg = {m["name"]: m for m in yaml.safe_load(models_copy.read_text(encoding="utf-8"))["models"]}
    assert new_cfg["gemma-3-1b-it"]["revision"] == fake_sha("google/gemma-3-1b-it")


def test_apply_dry_run_and_a_record_for_another_hf_id_are_safe(cfg, tmp_path, models_copy):
    pins, records = _pins_for(cfg, tmp_path)
    before = models_copy.read_text(encoding="utf-8")
    r = subprocess.run([PY, str(SCRIPTS / "pin_panel.py"), "--models-config", str(models_copy), "--apply", "--from", str(pins),
                        "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0 and "dry run" in r.stdout and models_copy.read_text(encoding="utf-8") == before
    # a record whose hf_id is not the entry's (a stale pins file after an id fix) is a conflict, never applied
    stale = copy.deepcopy(records)
    next(rec for rec in stale if rec["name"] == "qwen3.5-2b")["hf_id"] = "Qwen/Qwen3.5-2B-Instruct"
    plan = PP.plan_apply(cfg, stale)
    assert "qwen3.5-2b" in plan["conflicts"] and plan["refusals"]


def test_rewrite_guards_and_quoting():
    text = MODELS_YAML.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        PP.rewrite_revisions(text, {"no-such-entry": "a" * 40})
    digits = "1" * 40                                             # would parse as an int if written bare
    new = PP.rewrite_revisions(text, {"gemma-3-1b-it": digits})
    assert {m["name"]: m for m in yaml.safe_load(new)["models"]}["gemma-3-1b-it"]["revision"] == digits
    PP.verify_rewrite(yaml.safe_load(text), new, {"gemma-3-1b-it": digits})
    with pytest.raises(ValueError):
        PP.verify_rewrite(yaml.safe_load(text), new.replace('display_name: "Gemma 3 1B (IT)"', 'display_name: "x"'),
                          {"gemma-3-1b-it": digits})
    assert PP.already_pinned("a" * 40) and PP.already_pinned("google/gemma-3/transformers/gemma-3-1b-it/2")
    assert not PP.already_pinned(None) and not PP.already_pinned("main") and not PP.already_pinned("abc123")


def test_pin_panel_cli_resolution_path_writes_json_and_exit_code(cfg, tmp_path, monkeypatch, capsys):
    """The CLI's resolve path with the hub constructor swapped for the fake: writes --out, exit 1 while a
    panel entry is unresolved, 0 once everything resolved."""
    hub = FakeResolver(tmp_path, {"Qwen/Qwen3.5-2B": "not_found"})
    monkeypatch.setattr(PP, "HubResolver", lambda token=None, cache_dir=None: hub)
    out = tmp_path / "x" / "pins.json"
    rc = PP.main(["--out", str(out), "--names", "qwen3.5-2b", "gemma-3-1b-it"])
    assert rc == 1 and out.exists()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert [r["name"] for r in doc["records"]] == ["gemma-3-1b-it", "qwen3.5-2b"] and doc["n_unresolved_panel"] == 1
    assert "1 panel entries unresolved" in capsys.readouterr().out
    hub.behaviour = {}
    assert PP.main(["--out", str(out), "--names", "gemma-3-1b-it"]) == 0


# ------------------------------------------------------------------ kaggle_cpu_jobs
def _project(tmp_path, spm=True, items=("noilai_main", "noilai_core", "noilai_dev")) -> Path:
    """A project root with placeholder inputs: the two SentencePiece files and the named release files."""
    root = tmp_path / "proj"
    (root / "data" / "external").mkdir(parents=True)
    (root / "data" / "release" / "v0.3").mkdir(parents=True)
    if spm:
        for fname, _ in KCJ.SPM_FILES:
            (root / "data" / "external" / fname).write_bytes(b"spm")
    for key in items:
        (root / "data" / "release" / "v0.3" / f"{key}.jsonl").write_text("{}\n")
    return root


def test_cpu_jobs_plan_covers_every_distinct_tokenizer_and_item_key(cfg, tmp_path):
    root = _project(tmp_path)
    ctx = KCJ.Context(root=root, python="py")
    cmds = KCJ.plan_jobs(["all"], ctx)
    hf_ids = []
    for m in cfg["models"]:
        if m.get("hf_id") and m["hf_id"] not in hf_ids:
            hf_ids.append(m["hf_id"])
    toks = [c for c in cmds if c.job == "audit-tokenizers"]
    assert [c.name for c in toks] == ["gemma3", "gemma2"] + [h.replace("/", "__") for h in hf_ids]
    assert all(c.skip_reason is None for c in toks)
    assert toks[0].argv == ["py", str(ROOT / "scripts" / "audit_tokenizers.py"), "--spm",
                            f"{root / 'data' / 'external' / 'gemma3_tokenizer.model'}:gemma3", "--out", str(root / "data" / "audit"),
                            "--items", str(root / "data" / "release" / "v0.3" / "noilai_main.jsonl")]
    assert toks[2].argv[2:4] == ["--hf", "google/gemma-3-1b-it"]
    assert toks[2].outputs == [root / "data" / "audit" / "google__gemma-3-1b-it.json", root / "data" / "audit" / "google__gemma-3-1b-it_rows.csv"]
    items = [c for c in cmds if c.job == "audit-items"]
    assert len(items) == len(toks) * 3 and all(c.skip_reason is None for c in items)
    first = items[0]
    assert first.argv == ["py", str(ROOT / "scripts" / "audit_items.py"), "--items",
                          str(root / "data" / "release" / "v0.3" / "noilai_main.jsonl"), "--spm",
                          f"{root / 'data' / 'external' / 'gemma3_tokenizer.model'}:gemma3", "--out",
                          str(root / "data" / "audit" / "items_gemma3__noilai_main.jsonl")]
    assert {c.name for c in items} == {f"{t.name}__{k}" for t in toks for k in KCJ.DEFAULT_ITEMS_KEYS}
    assert all(c.argv[c.argv.index("--out") + 1].endswith(f"items_{c.name}.jsonl") for c in items)
    pin = [c for c in cmds if c.job == "pin"]
    assert len(pin) == 1 and pin[0].argv[1] == str(ROOT / "scripts" / "pin_panel.py") and "--include-api" in pin[0].argv
    assert pin[0].argv[pin[0].argv.index("--out") + 1] == str(root / "experiments" / "panel_pins.json")
    assert pin[0].outputs == [root / "experiments" / "panel_pins.json"]
    # order: the three jobs in sequence; subsets and the key override
    assert [c.job for c in cmds] == ["audit-tokenizers"] * len(toks) + ["audit-items"] * len(items) + ["pin"]
    only = KCJ.plan_jobs(["pin", "audit-items"], ctx, items_keys=("noilai_dev",))
    assert [c.job for c in only][:len(toks)] == ["audit-items"] * len(toks) and only[-1].job == "pin"
    assert all(c.name.endswith("__noilai_dev") for c in only[:-1])
    with pytest.raises(ValueError):
        KCJ.plan_jobs(["nope"], ctx)
    with pytest.raises(KeyError):
        KCJ.plan_jobs(["audit-items"], ctx, items_keys=("no_such_key",))
    # --release-dir points the plan's {release} elsewhere
    other = tmp_path / "rel"
    other.mkdir()
    (other / "noilai_dev.jsonl").write_text("{}\n")
    alt = KCJ.plan_jobs(["audit-items"], KCJ.Context(root=root, release_dir=other), items_keys=("noilai_dev",))
    assert alt[0].argv[alt[0].argv.index("--items") + 1] == str(other / "noilai_dev.jsonl") and alt[0].skip_reason is None


def test_cpu_jobs_skip_missing_inputs_with_a_clear_message_and_exit_code(tmp_path):
    root = _project(tmp_path, spm=False, items=("noilai_dev",))
    ctx = KCJ.Context(root=root)
    cmds = KCJ.plan_jobs(["all"], ctx)
    spm = [c for c in cmds if c.job == "audit-tokenizers" and c.name in ("gemma3", "gemma2")]
    assert all("fetch_resources" in c.skip_reason and "gemma" in c.skip_reason for c in spm)
    hf = [c for c in cmds if c.job == "audit-tokenizers" and c.name not in ("gemma3", "gemma2")]
    assert all(c.skip_reason is None and "--items" not in c.argv for c in hf)             # no noilai_main: census from words only
    assert all("noilai_main" in c.note for c in hf)
    items = [c for c in cmds if c.job == "audit-items"]
    main_missing = [c for c in items if c.name.endswith("__noilai_main")]
    assert all("missing item file" in c.skip_reason and "noilai_main" in c.skip_reason for c in main_missing)
    dev = [c for c in items if c.name.endswith("__noilai_dev")]
    assert [c.skip_reason is None for c in dev] == [False, False] + [True] * (len(dev) - 2)
    # the CLI: a job with no runnable command fails the run; one that still has commands does not
    report = tmp_path / "r.json"
    r = subprocess.run([PY, str(SCRIPTS / "kaggle_cpu_jobs.py"), "audit-items", "--items-keys", "noilai_main", "--project-root",
                        str(root), "--report", str(report), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 1 and "could not run: ['audit-items']" in r.stderr and "skipped: missing item file" in r.stdout
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert doc["jobs"]["audit-items"]["ran"] is False and doc["jobs"]["audit-items"]["skipped"] == doc["jobs"]["audit-items"]["n_commands"]
    r = subprocess.run([PY, str(SCRIPTS / "kaggle_cpu_jobs.py"), "audit-tokenizers", "pin", "--project-root", str(root),
                        "--report", str(report), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert doc["dry_run"] is True and set(doc["jobs"]) == {"audit-tokenizers", "pin"}
    assert doc["jobs"]["audit-tokenizers"]["skipped"] == 2 and doc["jobs"]["audit-tokenizers"]["planned"] == len(hf)
    assert all(c["status"] in ("planned", "skipped") and c["produced"] == {} for c in doc["commands"])
    assert not (root / "data" / "audit" / "gemma3.json").exists() and not (root / "experiments").exists()


def test_cpu_jobs_dry_run_against_the_repository_plans_every_hub_tokenizer(tmp_path):
    """The real configs: 2 SentencePiece + every distinct hf_id of models.yaml, one pin command, the report written."""
    report = tmp_path / "report.json"
    r = subprocess.run([PY, str(SCRIPTS / "kaggle_cpu_jobs.py"), "audit-tokenizers", "pin", "--report", str(report), "--dry-run"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    doc = json.loads(report.read_text(encoding="utf-8"))
    names = [c["name"] for c in doc["commands"] if c["job"] == "audit-tokenizers"]
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"]
    assert set(names) == {"gemma3", "gemma2"} | {m["hf_id"].replace("/", "__") for m in models if m.get("hf_id")}
    assert len(names) == len(set(names))
    assert doc["models_config"] == "configs/models.yaml" and doc["plan"] == "configs/run_plan.yaml"
    pin = next(c for c in doc["commands"] if c["job"] == "pin")
    assert pin["status"] == "planned" and pin["outputs"] == ["experiments/panel_pins.json"]
    assert "scripts/pin_panel.py" in pin["command"] and pin["returncode"] is None


def test_cpu_jobs_execute_records_hashes_and_failures(tmp_path):
    root = _project(tmp_path, items=("noilai_dev",))
    ctx = KCJ.Context(root=root, python="py")
    cmds = KCJ.plan_jobs(["audit-tokenizers", "pin"], ctx)
    calls = []

    def runner(argv, cwd, text, capture_output, check):
        calls.append(argv)
        cmd = next(c for c in cmds if c.argv == argv)
        if cmd.name == "gemma2":
            return subprocess.CompletedProcess(argv, 3, stdout="", stderr="boom: no such tokenizer\n")
        for p in cmd.outputs:
            p.write_text(f"out of {cmd.name}")
        return subprocess.CompletedProcess(argv, 0, stdout=f"did {cmd.name}\n", stderr="")

    KCJ.execute(cmds, ctx, runner=runner)
    assert len(calls) == len(cmds) and all(c.wall_s is not None for c in cmds)
    by = {c.name: c for c in cmds}
    assert by["gemma3"].status == "ok" and by["gemma3"].returncode == 0 and by["gemma3"].stdout_tail == "did gemma3\n"
    produced = by["gemma3"].produced
    assert set(produced) == {"data/audit/gemma3.json", "data/audit/gemma3_rows.csv"}
    assert produced["data/audit/gemma3.json"]["sha256"] == hashlib.sha256(b"out of gemma3").hexdigest()
    assert produced["data/audit/gemma3.json"]["bytes"] == len(b"out of gemma3")
    assert by["gemma2"].status == "failed" and by["gemma2"].returncode == 3 and "boom" in by["gemma2"].stderr_tail
    assert by["gemma2"].produced == {}
    assert by["pin"].status == "ok" and set(by["pin"].produced) == {"experiments/panel_pins.json"}
    summary = KCJ.job_summary(cmds, ["audit-tokenizers", "pin"])
    assert summary["audit-tokenizers"]["failed"] == 1 and summary["audit-tokenizers"]["ok"] == len(cmds) - 2
    assert summary["pin"] == {"n_commands": 1, "planned": 0, "ok": 1, "failed": 0, "skipped": 0, "ran": True}
    report = tmp_path / "r.json"
    doc = KCJ.write_report(report, cmds, ["audit-tokenizers", "pin"], ctx, False, "2026-10-07T00:00:00+00:00")
    back = json.loads(report.read_text(encoding="utf-8"))
    assert back["jobs"] == doc["jobs"] == summary and back["started_utc"] == "2026-10-07T00:00:00+00:00"
    rec = next(c for c in back["commands"] if c["name"] == "gemma3")
    assert rec["produced"] == produced and rec["command"].startswith("py ") and rec["status"] == "ok"
    # an interpreter that cannot start is a failure with the exception recorded, not a crash
    def broken(argv, **kw):
        raise OSError("no such interpreter")

    KCJ.execute(cmds[:1], ctx, runner=broken)
    assert cmds[0].status == "failed" and cmds[0].returncode == -1 and "OSError" in cmds[0].stderr_tail


def test_scripts_run_as_programs_with_help():
    for name in ("pin_panel.py", "kaggle_cpu_jobs.py"):
        r = subprocess.run([PY, str(SCRIPTS / name), "--help"], capture_output=True, text=True, check=False)
        assert r.returncode == 0 and "Kaggle" in r.stdout, (name, r.stderr[-500:])
