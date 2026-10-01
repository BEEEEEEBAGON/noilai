"""End-to-end tests of the command-line scripts on small builds."""
import csv
import json
import re
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
    """DD 4.6: the public manifest.json carries neither the build seed nor the canary GUID (its SHA-256 instead); the
    private manifest_private.json carries everything and is the only place the seed lives."""
    import hashlib
    m = json.loads((release / "manifest.json").read_text())
    priv = json.loads((release / "manifest_private.json").read_text())
    assert priv["n_items"] > 0 and priv["seed"] == 3 and "git_commit" in priv and priv["canary"].startswith("NOILAI-CANARY-")
    assert priv["generator_args"]["seed"] == 3
    assert "seed" not in m and "canary" not in m and "seed" not in m["generator_args"] and "canary" not in m["generator_args"]
    assert m["canary_sha256"] == priv["canary_sha256"] == hashlib.sha256(priv["canary"].encode()).hexdigest()
    assert m["private_manifest"] == "manifest_private.json" and m["n_items"] == priv["n_items"] and m["content_sha256"] == priv["content_sha256"]
    for name in ("noilai_dev.jsonl", "noilai_test.jsonl", "noilai_core.jsonl"):
        assert (release / name).exists()
    from noilai.gen.generate import load_items, read_header, release_canary
    core = load_items(release / "noilai_core.jsonl")
    assert all(it["in_core"] and it["split"] == "test" for it in core)
    assert read_header(release / "noilai_core.jsonl")["canary"] == priv["canary"] == release_canary(release)
    assert sum(m["core_counts"].values()) == len(core)


def test_build_data_requires_a_seed(tmp_path):
    """DD 4.6 (item 51): no default build seed anywhere; `--seed` is required and the Makefile has no SEED default."""
    r = subprocess.run([PY, "scripts/build_data.py", "--out", str(tmp_path / "x")], cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 2 and "--seed" in r.stderr and "required" in r.stderr
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert not re.search(r"^SEED\s*\??=", mk, re.MULTILINE), "the Makefile must not carry a build-seed default"
    assert 'test -n "$(SEED)"' in mk and "exit 1" in mk[mk.index("data: resources"):mk.index("audit:")]
    r = subprocess.run(["make", "-n", "data", "SEED="], cwd=ROOT, text=True, capture_output=True, check=False)
    if r.returncode == 0:                                     # `make -n` prints the recipe; run the guard line itself
        guard = next(ln for ln in r.stdout.splitlines() if 'test -n "$(SEED)"' in ln or "test -n \"\"" in ln)
        g = subprocess.run(["sh", "-c", guard], cwd=ROOT, text=True, capture_output=True, check=False)
        assert g.returncode == 1 and "SEED is unset" in g.stderr


def test_build_attested(tmp_path):
    out = tmp_path / "attested.jsonl"
    r = run("scripts/build_attested.py", "--out", str(out))
    with open(out, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
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
    # design 12.5: attested text is stored in the release placement style like every other file
    from noilai.vi.reencode import convert_placement
    n_texts = 0
    for r_ in rows:
        for t in [r_["input"], r_["attested_output"], r_["rule_output"], *r_["gold"]]:
            assert convert_placement(t, "old") == t, (r_["item_id"], t)
            n_texts += 1
        assert [d["spelled"] for d in r_["input_syllables"]] == r_["input"].split()
    assert n_texts > 100
    assert by_in[("thụy điển", "V3")]["input"] == "thụy điển" and by_in[("thụy điển", "V3")]["attested_output"] == "thủy điện"
    assert "0 exactness problems" in r.stdout


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
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                c[r["item_id"]] += 1
    assert Counter(c.values()) == {2: 90, 3: 30}
    # fill in sheets and score
    ret = out / "returned"
    ret.mkdir()
    for f in forms:
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
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
    with open(out / "baseline_form_01.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24 and {r["task"] for r in rows} == {"T1", "T2", "T3"}


def test_audit_script_on_gemma3_if_present(tmp_path):
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    run("scripts/audit_tokenizers.py", "--spm", f"{spm}:g3", "--out", str(tmp_path), "--max-syllables", "200")
    d = json.loads((tmp_path / "g3.json").read_text())
    assert 190 <= d["summary"]["nfc"]["n"] <= 200 and d["normalization_census"]["normalizes_nfd"] is False
    assert (tmp_path / "g3_rows.csv").exists()


def _fetch_resources_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetch_resources", ROOT / "scripts" / "fetch_resources.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_resources_keeps_existing_and_verifies_hashes():
    """Offline only: the script is run when every resource is already under data/external with the hash recorded in
    data/HASHES.json, so that it takes the `[keep]` path for each file and reaches the network for nothing. Missing
    files skip the test (the fetch is the author's step, never a test side effect); a hash mismatch on disk fails it
    before the script could rename the file."""
    FR = _fetch_resources_module()
    known = FR.load_hashes()
    assert known, "data/HASHES.json is empty"
    missing = [n for n in FR.RESOURCES if not (FR.EXTERNAL / n).exists()]
    if missing:
        pytest.skip(f"resources not downloaded ({missing}); run scripts/fetch_resources.py by hand")
    mismatched = {n: (FR.sha256(FR.EXTERNAL / n)[:16], (known.get(n) or FR.RESOURCES[n].get("sha256") or "")[:16])
                  for n in FR.RESOURCES if (known.get(n) or FR.RESOURCES[n].get("sha256"))
                  and FR.sha256(FR.EXTERNAL / n) != (known.get(n) or FR.RESOURCES[n].get("sha256"))}
    assert not mismatched, f"data/external differs from data/HASHES.json: {mismatched}"
    r = run("scripts/fetch_resources.py")
    assert "[get ]" not in r.stdout, "the script must not fetch when every file is present"
    assert r.stdout.count("[keep]") == len(FR.RESOURCES) and r.stdout.count("[ok  ]") == len(FR.RESOURCES)
    assert "HASH MISMATCH" not in r.stdout and "[FAIL]" not in r.stdout


def test_audit_items_script(release, tmp_path):
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    out = tmp_path / "items.jsonl"
    run("scripts/audit_items.py", "--items", str(release / "noilai_core.jsonl"), "--spm", f"{spm}:g3", "--out", str(out))
    from noilai.gen.generate import load_items
    with open(out, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
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


def _expected_main_counts(test_items: list[dict], per_cell: int) -> dict[str, int]:
    """DD 4.5 / scripts/sample_items.py: per T1/T2 cell `per_cell` non-vulgar test items (the core first, then the pool
    until it is exhausted); per T3 cell `per_cell // 2` yes/no PAIRS drawn from the `yes` members that have a mate."""
    from collections import Counter

    from noilai.gen import variants as V
    by_id = {it["item_id"]: it for it in test_items}
    want = Counter()
    for task in ("T1", "T2"):
        for v in V.VARIANTS:
            pool = [it for it in test_items if it["task"] == task and it["variant"] == v and not it["vulgar"]]
            core = sum(1 for it in pool if it["in_core"])
            want[f"{task}-{v}"] = core + min(max(per_cell - core, 0), len(pool) - core)
    for v in V.VARIANTS:
        pool = [it for it in test_items if it["task"] == "T3" and it["variant"] == v and it["gold"] == "yes" and not it["vulgar"]
                and it.get("pair_item_id") in by_id]
        core = sum(1 for it in pool if it["in_core"])
        want[f"T3-{v}"] = 2 * (core + min(max(per_cell // 2 - core, 0), len(pool) - core))
    return dict(want)


def test_sample_items_main_and_c2(release):
    """tests-health-2: EXACT per-cell counts (350 per T1/T2 cell and 175 yes/no pairs per T3 cell on the paper's build;
    on this fixture the exact numbers its pools allow, computed independently here), not an upper bound."""
    from collections import Counter
    r = run("scripts/sample_items.py", "main", "--release", str(release), "--per-cell", "20", "--seed", "1")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    from noilai.gen.generate import load_items, read_header
    main_items = load_items(release / "noilai_main.jsonl")
    core = load_items(release / "noilai_core.jsonl")
    test_items = load_items(release / "noilai_test.jsonl")
    assert {it["item_id"] for it in core} <= {it["item_id"] for it in main_items}       # the core is inside the main sample
    expected = _expected_main_counts(test_items, 20)
    assert info["counts"] == expected, (info["counts"], expected)
    observed = dict(Counter(f"{it['task']}-{it['variant']}" for it in main_items))
    assert observed == expected and info["n_items"] == len(main_items) == sum(expected.values())
    assert len(expected) == 12 and all(v % 2 == 0 for k, v in expected.items() if k.startswith("T3"))
    assert any(v == 20 for k, v in expected.items() if not k.startswith("T3")), "no cell reaches the requested size: fixture too small"
    assert all(v <= 20 for v in expected.values())
    t3 = [it for it in main_items if it["task"] == "T3"]
    t3_ids = {it["item_id"] for it in t3}
    assert all(it["pair_item_id"] in t3_ids for it in t3) and sum(1 for it in t3 if it["gold"] == "yes") * 2 == len(t3)
    assert not any(it["vulgar"] for it in main_items)
    assert read_header(release / "noilai_main.jsonl")["canary"] == read_header(release / "noilai_test.jsonl")["canary"]
    run("scripts/sample_items.py", "c2", "--release", str(release), "--n", "30", "--seed", "7")
    c2 = load_items(release / "noilai_c2.jsonl")
    rel_bp = {it["base_pair_id"] for it in load_items(release / "noilai_test.jsonl") + load_items(release / "noilai_dev.jsonl")}
    assert c2 and all(it["strata"]["c2_affected"] for it in c2) and not ({it["base_pair_id"] for it in c2} & rel_bp)
    assert all(it["c2_enriched"] is True and it["strata"]["c2_enriched"] is True for it in c2)     # design 4.3 / 4.5
    m = json.loads((release / "manifest.json").read_text())
    assert m["samples"]["main"]["sha256"] and m["samples"]["c2"]["disjoint_from_release"] is True
    assert m["samples"]["main"]["counts"] == expected and m["samples"]["main"]["seed"] == 1 and m["samples"]["c2"]["seed"] == 7
    priv = json.loads((release / "manifest_private.json").read_text())
    assert priv["samples"] == m["samples"] and "canary" not in m and "seed" not in m      # both manifests record the samples


def test_attested_labels_are_engine_derived(tmp_path):
    out = tmp_path / "attested.jsonl"
    run("scripts/build_attested.py", "--out", str(out))
    with open(out, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    by = {(r["input"], r["declared_variant"]): r for r in rows}
    assert "V2" in by[("khoái ăn sang", "V2")]["variant_labels"] and by[("khoái ăn sang", "V2")]["declared_matches_engine"]
    assert set(by[("cây còn", "V1")]["variant_labels"]) >= {"V1", "V2"}
    assert not by[("thầy giáo", "V4")]["declared_matches_engine"] and by[("thầy giáo", "V4")]["exactness"].startswith("approx(substitution")
    assert not any(r["input"] == "mộng mơ" for r in rows)
    # design 4.7(b) / 3.7: exactness is engine-derived and every approx row is reachable by exactly its named change
    approx = [r for r in rows if not r["exact"]]
    assert len(approx) >= 5 and all(r["exact"] == r["rule_matches_attested"] for r in rows)
    assert all(r["exactness"] == "exact" for r in rows if r["rule_matches_attested"])
    assert all(r["exactness"].startswith("approx(") and r["approx_reachable"] is True for r in approx), \
        [(r["input"], r["exactness"], r["approx_reachable"]) for r in approx]
    assert all(r["approx_reachable"] is None for r in rows if r["exact"])
    assert {r["exactness"].split("(")[1].split(":")[0] for r in approx} == {"merger", "substitution"}


def test_build_attested_refuses_a_declared_exact_row_the_engine_does_not_reproduce(tmp_path):
    """A hand-typed `exact` is never trusted: the row is labelled by the engine and --strict fails."""
    import importlib.util
    seed = tmp_path / "seed.tsv"
    header = "input\toutput\tvariant\tpositions\texactness\tregion_tag\tgloss_input\tgloss_output\tnote\tconfidence\tvulgar\tsource\tverified_by\n"
    seed.write_text(header + "đại học\tđộc hại\tV1\t0-1\texact\t\tuniversity\tharmful\t\thigh\tno\ttest\t\n"
                    + "đại học\tđộc hại\tV1\t0-1\tapprox(substitution: o→a)\t\tuniversity\tharmful\t\thigh\tno\ttest\t\n", encoding="utf-8")
    out = tmp_path / "att.jsonl"
    r = subprocess.run([PY, "scripts/build_attested.py", "--seed-file", str(seed), "--out", str(out), "--strict"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 1 and "[WARN]" in r.stdout and "2 exactness problems" in r.stdout
    with open(out, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    assert rows[0]["exact"] is False and rows[0]["exactness"] == "approx(unspecified)" and rows[0]["declared_exactness"] == "exact"
    assert rows[0]["approx_reachable"] is False and not rows[0]["eligible_h6"]
    assert rows[1]["approx_reachable"] is False                  # o→a is not the change that yields độc
    spec = importlib.util.spec_from_file_location("build_attested", ROOT / "scripts" / "build_attested.py")
    ba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ba)
    from noilai.vi.syllable import try_parse

    def S(t):
        return [try_parse(w, strict=False).syllable for w in t.split()]
    assert ba.reachable_by_named_change(S("đọc hại"), S("độc hại"), "approx(substitution: o→ô)") is True
    assert ba.reachable_by_named_change(S("đan giởn"), S("đang giỡn"), "approx(merger: n/ng + hỏi/ngã)") is True
    assert ba.reachable_by_named_change(S("đan giởn"), S("đang giỡn"), "approx(merger: hỏi/ngã)") is False   # n/ng unexplained
    assert ba.reachable_by_named_change(S("vẫn như củ"), S("vẫn như cũ"), "approx(merger: n/ng + hỏi/ngã)") is False  # n/ng unused
