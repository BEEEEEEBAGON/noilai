"""End-to-end tests of the command-line scripts on small builds."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args, **kw):
    return subprocess.run([PY, *args], cwd=ROOT, text=True, capture_output=True, check=True, **kw)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    out = tmp_path_factory.mktemp("rel")
    run("scripts/build_data.py", "--out", str(out), "--seed", "3", "--n-lexicon", "120", "--n-pseudo", "60",
        "--per-cell-t1", "40", "--per-cell-t2", "20", "--per-cell-t3", "15", "--core-per-cell", "8")
    return out


def test_build_data_writes_release_and_manifest(release):
    m = json.loads((release / "manifest.json").read_text())
    assert m["n_items"] > 0 and m["seed"] == 3 and "git_commit" in m and m["canary"].startswith("NOILAI-CANARY-")
    for name in ("noilai_dev.jsonl", "noilai_test.jsonl", "noilai_core.jsonl"):
        assert (release / name).exists()
    from noilai.gen.generate import load_items, read_header
    core = load_items(release / "noilai_core.jsonl")
    assert all(it["in_core"] and it["split"] == "test" for it in core)
    assert read_header(release / "noilai_core.jsonl")["canary"] == m["canary"]
    assert sum(m["core_counts"].values()) == len(core)


def test_build_attested(tmp_path):
    out = tmp_path / "attested.jsonl"
    r = run("scripts/build_attested.py", "--out", str(out))
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) >= 30 and "rule reproduces" in r.stdout
    by_in = {(r["input"], r["declared_variant"]): r for r in rows}
    assert by_in[("mèo cái", "V1")]["rule_matches_attested"] is True and by_in[("mèo cái", "V1")]["eligible_h6"]
    assert by_in[("bí mật", "V3")]["variant"] == "V3" and set(by_in[("bí mật", "V3")]["variant_labels"]) == {"V2", "V3"}
    assert by_in[("đầu tiên", "V2")]["rule_matches_attested"] is True
    assert by_in[("bí mật", "V3")]["rule_matches_attested"] is True
    # the ây/ay case: rule gives 'tháo giầy', attested 'tháo giày' -> approx, both forms in gold, not H6-eligible
    r_ = by_in[("thầy giáo", "V4")]
    assert r_["rule_matches_attested"] is False and "tháo giầy" in r_["gold"] and "tháo giày" in r_["gold"]
    assert r_["exactness"].startswith("approx") and not r_["eligible_h6"]
    # three-syllable rows use their declared positions; the reproducing labels record every match
    assert by_in[("chà đồ nhôm", "V4")]["rule_output"] == "chôm đồ nhà" and by_in[("chà đồ nhôm", "V4")]["positions"] == "0-2"
    k = by_in[("khoái ăn sang", "V2")]
    assert k["rule_matches_attested"] is True and "V2@0-2" in k["reproducing_labels"] and "V3@0-2 reversed" in k["reproducing_labels"]
    assert by_in[("con cá đối", "V1")]["rule_matches_attested"] is True and by_in[("con cá đối", "V1")]["positions"] == "1-2"
    # the six-way textbook illustration: every variant reproduces its row
    for v in ("V1", "V2", "V3", "V4", "V5", "V6"):
        assert by_in[("thay đổi", v)]["rule_matches_attested"] is True, v
    assert by_in[("trái gió", "V6")]["vulgar"] is True
    assert all(r["parse_ok"] for r in rows) and not any(r["eligible_h6"] for r in rows if r["n_syllables"] == 3)


def test_validation_forms_sample_and_score(release, tmp_path):
    out = tmp_path / "val"
    run("scripts/make_validation_forms.py", "sample", "--items", str(release / "noilai_test.jsonl"), "--dev",
        str(release / "noilai_dev.jsonl"), "--out", str(out), "--n", "120", "--overlap", "30", "--validators", "A", "B", "C")
    forms = sorted(out.glob("validation_form_*.csv"))
    assert len(forms) == 3
    meta = json.loads((out / "validation_manifest.json").read_text())
    assert meta["n_items"] == 120 and meta["overlap"] == 30
    # every non-overlap item on exactly two forms, overlap items on three
    from collections import Counter
    c = Counter()
    for f in forms:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            c[r["item_id"]] += 1
    assert Counter(c.values()) == {2: 90, 3: 30}
    # fill in sheets and score
    ret = out / "returned"
    ret.mkdir()
    for f in forms:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        for r in rows:
            r.update(correct="yes", spelling="yes", lexical="no", offensive="no")
        with open(ret / f.name, "w", newline="", encoding="utf-8") as g:
            w = csv.DictWriter(g, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    r = run("scripts/make_validation_forms.py", "score", "--out", str(out), "--returned", str(ret / "*.csv"))
    rep = json.loads((out / "validation_report.json").read_text())
    assert rep["correct"]["percent_agreement"] == 1.0
    assert rep["generator_precision"]["estimate"] == 1.0


def test_human_baseline_forms(release, tmp_path):
    out = tmp_path / "human"
    r = run("scripts/make_validation_forms.py", "baseline", "--items", str(release / "noilai_core.jsonl"), "--out", str(out),
            "--n-forms", "4", "--per-form", "24")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["forms"] == 4 and len(list(out.glob("baseline_form_*.csv"))) == 4
    rows = list(csv.DictReader(open(out / "baseline_form_01.csv", encoding="utf-8")))
    assert len(rows) == 24 and {r["task"] for r in rows} == {"T1", "T2", "T3"}


def test_audit_script_on_gemma3_if_present(tmp_path):
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    run("scripts/audit_tokenizers.py", "--spm", f"{spm}:g3", "--out", str(tmp_path), "--max-syllables", "200")
    d = json.loads((tmp_path / "g3.json").read_text())
    assert 190 <= d["summary"]["nfc"]["n"] <= 200 and d["normalization_census"]["normalizes_nfd"] is False
    assert (tmp_path / "g3_rows.csv").exists()


def test_fetch_resources_keeps_existing_and_verifies_hashes():
    r = run("scripts/fetch_resources.py")
    assert "HASH MISMATCH" not in r.stdout and "[FAIL]" not in r.stdout


def test_audit_items_script(release, tmp_path):
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    out = tmp_path / "items.jsonl"
    run("scripts/audit_items.py", "--items", str(release / "noilai_core.jsonl"), "--spm", f"{spm}:g3", "--out", str(out))
    from noilai.gen.generate import load_items
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    core = load_items(release / "noilai_core.jsonl")
    assert len(rows) == 2 * len(core)
    nfd = [r for r in rows if r["encoding"] == "nfd"]
    assert all(r["delta_tokens_vs_nfc"] >= 0 for r in nfd) and any(r["delta_tokens_vs_nfc"] > 0 for r in nfd)
    assert all(len(r["input"]["tokens_per_syllable"]) == 2 for r in rows)


def test_count_placement_on_xcopa(tmp_path):
    xc = ROOT / "data" / "external" / "xcopa_test_vi.jsonl"
    if not xc.exists():
        pytest.skip("XCOPA not downloaded")
    out = tmp_path / "pl.json"
    run("scripts/count_placement.py", str(xc), "--out", str(out))
    rep = json.loads(out.read_text())
    assert rep["affected_tokens"] > 30 and rep["old"] + rep["new"] == rep["affected_tokens"]
    assert rep["majority"] == "old" and rep["old_share"] > 0.9


def test_sample_items_main_and_c2(release):
    r = run("scripts/sample_items.py", "main", "--release", str(release), "--per-cell", "20", "--seed", "1")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    from noilai.gen.generate import load_items, read_header
    main_items = load_items(release / "noilai_main.jsonl")
    core = load_items(release / "noilai_core.jsonl")
    assert {it["item_id"] for it in core} <= {it["item_id"] for it in main_items}       # the core is inside the main sample
    assert all(v <= 20 or k.startswith("T3") for k, v in info["counts"].items())
    assert read_header(release / "noilai_main.jsonl")["canary"] == read_header(release / "noilai_test.jsonl")["canary"]
    r2 = run("scripts/sample_items.py", "c2", "--release", str(release), "--n", "30", "--seed", "7")
    c2 = load_items(release / "noilai_c2.jsonl")
    rel_bp = {it["base_pair_id"] for it in load_items(release / "noilai_test.jsonl") + load_items(release / "noilai_dev.jsonl")}
    assert c2 and all(it["strata"]["c2_affected"] for it in c2) and not ({it["base_pair_id"] for it in c2} & rel_bp)
    m = json.loads((release / "manifest.json").read_text())
    assert m["samples"]["main"]["sha256"] and m["samples"]["c2"]["disjoint_from_release"] is True


def test_attested_labels_are_engine_derived(tmp_path):
    out = tmp_path / "attested.jsonl"
    run("scripts/build_attested.py", "--out", str(out))
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    by = {(r["input"], r["declared_variant"]): r for r in rows}
    assert "V2" in by[("khoái ăn sang", "V2")]["variant_labels"] and by[("khoái ăn sang", "V2")]["declared_matches_engine"]
    assert set(by[("cây còn", "V1")]["variant_labels"]) >= {"V1", "V2"}
    assert not by[("thầy giáo", "V4")]["declared_matches_engine"] and by[("thầy giáo", "V4")]["exactness"].startswith("approx(substitution")
    assert not any(r["input"] == "mộng mơ" for r in rows)
