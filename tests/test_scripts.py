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


def test_validation_packet_and_score(release, tmp_path):
    """docs/gate1/VALIDATION_PROTOCOL.md: Parts A-E per validator, overlap rows on every sheet and the rest on exactly
    two, the planted controls and the item ids only in the author's key, then fill the sheets and score them."""
    from collections import Counter
    out = tmp_path / "val"
    r = run("scripts/make_validation_forms.py", "packet", "--release", str(release), "--out", str(out), "--validators", "A", "B", "C",
            "--per-cell", "3", "--controls-per-cell", "1", "--overlap", "6", "--t2-overlap", "4",
            "--candidates", str(tmp_path / "none.tsv"))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["sizes"]["n_sample"] == 36 and info["sizes"]["n_controls"] == 12 and info["sizes"]["calibration"] == 16
    key = json.loads((out / "B_key.json").read_text())
    assert len(key) == 48 and sum(k["control"] for k in key.values()) == 12
    c = Counter()
    for v in "ABC":
        with open(out / f"B_items_{v}.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert "item_id" not in rows[0] and "control" not in rows[0]          # the key never reaches a validator
        text = (out / f"B_items_{v}.csv").read_text(encoding="utf-8")
        assert not any(k["item_id"] in text for k in key.values())
        c.update(row["row_id"] for row in rows)
    assert Counter(c.values()) == {3: 6, 2: 42}
    m = json.loads((out / "validation_manifest.json").read_text())
    assert all(st["n_sampled"] == 0 or abs(st["weight"] * st["n_sampled"] - st["n_population"]) < 1e-6 for st in m["strata"].values())
    with open(out / "A_calibration_A.csv", encoding="utf-8") as fh:
        head = next(csv.reader(fh))
    assert "key" not in head and "explanation_vi" not in head and "manipulation" not in head
    # fill: everyone right on everything (controls answered "no"), and score
    ret = out / "returned"
    ret.mkdir()
    for v in "ABC":
        for sheet in ("B_items", "C_attested", "A_calibration"):
            with open(out / f"{sheet}_{v}.csv", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                if sheet == "B_items":
                    row.update(correct="no" if key[row["row_id"]]["control"] else "yes", spelling="yes", offensive="no")
                elif sheet == "C_attested":
                    row.update(valid="yes", known="no", offensive="no")
            if v == "A":                                  # A flags one sampled B row and one C row offensive
                b_row = next(r_ for r_ in rows if sheet == "B_items" and not key[r_["row_id"]]["control"]) if sheet == "B_items" else None
                if b_row:
                    b_row["offensive"], flagged_b = "yes", b_row
                if sheet == "C_attested":
                    rows[0]["offensive"], flagged_c = "yes", rows[0]
            with open(ret / f"{sheet}_{v}.csv", "w", newline="", encoding="utf-8") as g:
                w = csv.DictWriter(g, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    run("scripts/make_validation_forms.py", "score", "--dir", str(out), "--n-boot", "100")
    rep = json.loads((out / "report" / "validation_report.json").read_text())
    b = rep["B"]
    assert b["agreement"]["correct"]["percent_agreement"] == 1.0 and b["agreement"]["correct"]["alpha"] == 1.0
    assert b["generator_precision"]["pooled"]["weighted_precision"] == 1.0 and b["generator_precision"]["pooled"]["n"] == 36
    assert all(x["control_catch_rate"] == 1.0 for x in b["validators"].values())
    assert rep["C"]["n_verified"] == rep["C"]["n_rows"] > 0
    # DD 11.5: the flags file names the flagged item and the flagged texts by hash only (DD 11.1: no test-split text)
    from noilai.eval.run import load_validator_flags, text_digest
    raw = (out / "report" / "validator_flags.json").read_text(encoding="utf-8")
    flags = load_validator_flags(out / "report" / "validator_flags.json")
    assert flags["item_ids"] == {key[flagged_b["row_id"]]["item_id"]}
    assert {text_digest(flagged_b["candidate"]), text_digest(flagged_c["input"]), text_digest(flagged_c["output"])} <= flags["text_sha256"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in flags["text_sha256"])
    assert flagged_b["candidate"] not in raw and flagged_c["output"] not in raw and '"A"' not in raw


def test_human_baseline_forms(release, tmp_path):
    """DD 10.2: the forms carry the models' exact p0 prompt per item (and its hash), one instruction-check row."""
    from noilai.eval import prompts as P
    from noilai.gen.generate import load_items
    out = tmp_path / "human"
    r = run("scripts/make_validation_forms.py", "baseline", "--items", str(release / "noilai_core.jsonl"), "--out", str(out),
            "--n-forms", "4", "--per-form", "24")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["forms"] == 4 and len(list(out.glob("baseline_form_*.csv"))) == 4 and info["check_items_per_form"] == 1
    with open(out / "baseline_form_01.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    main = [r_ for r_ in rows if r_["block"] == "main"]
    assert len(main) == 24 and {r_["task"] for r_ in main} == {"T1", "T2", "T3"} and sum(r_["block"] == "check" for r_ in rows) == 1
    items = {it["item_id"]: it for it in load_items(release / "noilai_core.jsonl")}
    for r_ in main[:5]:
        msgs = P.render(items[r_["item_id"]], paraphrase="p0", shots=3, arm="nfc")
        assert r_["prompt_model"] == msgs[-1]["content"] and r_["prompt_hash"] == P.prompt_hash(msgs)


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


# ------------------------------------------------------------------ returned human sheets (CLAUDE.md step 2)
def _gold_answer(it: dict) -> str:
    if it["task"] == "T1":
        return it["gold"][0]
    if it["task"] == "T2":
        return it["gold"][0]["output"]
    return "Có" if it["gold"] == "yes" else "Không"


def _swap(text: str) -> str:
    a, b = text.split()
    return f"{b} {a}"


def _no_personal_data(root: Path, *needles: str) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            for needle in needles:
                assert needle not in text, (p, needle)


def test_human_baseline_import_and_score(release, tmp_path):
    """The one baseline scorer on the forms the author sends: Google Forms response CSVs -> `import-responses` (first
    submission per form, e-mail in an answer cell redacted, coarse demographics only) -> `score-baseline` with the models'
    scorer; the PREREG 5.7 exclusions plus WAS_VALIDATOR (docs/DEVIATIONS.md 1 Oct) and a duplicate response, every
    reason listed and counted; strict / lenient / tolerant per row (DD 10.2 item 32); mean-human with the two-way
    (person, item) bootstrap and the any-human ceiling (PREREG 8.11); alpha between the raters on the double-judged
    items; the region split from demographics.csv; item-level rows in the git-ignored data tree, aggregates with the
    file's SHA-256 under --out (CLAUDE.md: no names or e-mails there)."""
    import hashlib
    import importlib
    import shutil
    sys.path.insert(0, str(ROOT / "scripts"))
    MVF = importlib.import_module("make_validation_forms")
    from noilai.gen.generate import load_items
    email = "respondent.07" + "@" + "example.invalid"                       # built here: no address literal in the tree
    items = {it["item_id"]: it for it in load_items(release / "noilai_core.jsonl")}
    out = tmp_path / "human"
    run("scripts/make_validation_forms.py", "baseline", "--items", str(release / "noilai_core.jsonl"), "--out", str(out),
        "--n-forms", "4", "--per-form", "24", "--no-model-prompt")
    forms = {}
    for k in range(1, 5):
        with open(out / f"baseline_form_{k:02d}.csv", encoding="utf-8") as fh:
            forms[k] = list(csv.DictReader(fh))
    demo_titles = [t for t, _c, _r in MVF.DEMOGRAPHICS_VI]
    region_title = next(t for t in demo_titles if "vùng" in t)

    def answers(k: int, edit) -> dict[str, str]:
        """{position: answer} for form k: gold everywhere, then `edit` changes some."""
        got = {}
        for r in forms[k]:
            if r["block"] == "main":
                got[r["position"]] = _gold_answer(items[r["item_id"]])
            elif r["item_id"] == MVF.CHECK_ITEM["item_id"]:
                got[r["position"]] = "Đã đọc."
            elif r["item_id"] in ("TOOLS", "WAS_VALIDATOR"):
                got[r["position"]] = "Không"
        edit(got)
        return got

    main_rows = {k: [r for r in forms[k] if r["block"] == "main"] for k in forms}
    t1_2 = [r for r in main_rows[2] if r["task"] == "T1"]
    t2_2 = next(r for r in main_rows[2] if r["task"] == "T2")
    t3_2 = next(r for r in main_rows[2] if r["task"] == "T3")
    special = {k: {r["item_id"]: r["position"] for r in forms[k] if r["block"] != "main"} for k in forms}

    def edit2(got):                                                 # a swapped T1 (lenient), a blank T1, a wrong T3, an e-mail
        got[t1_2[0]["position"]] = _swap(got[t1_2[0]["position"]])
        got[t1_2[1]["position"]] = ""
        got[t3_2["position"]] = "Không" if got[t3_2["position"]] == "Có" else "Có"
        got[t2_2["position"]] = email

    def edit3(got):                                                 # five answers only, and tool use reported
        for r in main_rows[3][5:]:
            got[r["position"]] = ""
        got[special[3]["TOOLS"]] = "Có"

    def edit4(got):                                                 # the check without diacritics fails; a validator
        got[special[4]["CHECK-01"]] = "da doc"
        got[special[4]["WAS_VALIDATOR"]] = "Có"

    edits = {1: lambda got: None, 2: edit2, 3: edit3, 4: edit4}
    regions = {1: "Nam", 2: "Bắc", 3: "", 4: "Trung"}
    (out / "responses").mkdir()
    for k in range(1, 5):
        got = answers(k, edits[k])
        header = ["Timestamp", "Email Address", *demo_titles, *[f"{r['position']}. {r['question_short']}" for r in forms[k]],
                  "Góp ý (không bắt buộc)"]
        demo = dict.fromkeys(demo_titles, "")
        demo[demo_titles[0]], demo[region_title] = MVF.DEMOGRAPHICS_VI[0][1][0], regions[k]       # the first age band
        with open(out / "responses" / f"responses_form_{k:02d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(["2026/11/03 10:00:00", email, *[demo[t] for t in demo_titles], *[got.get(r["position"], "") for r in forms[k]], ""])
            if k == 1:                                              # a second submission: dropped by the importer, unread
                w.writerow(["2026/11/03 11:00:00", email, *[demo[t] for t in demo_titles], *["x" for _ in forms[k]], ""])
    r = run("scripts/make_validation_forms.py", "import-responses", "--dir", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info == {"responses": 4, "extra_submissions_dropped": {"1": 1}, "emails_redacted": 1, "written_to": str(out / "returned")}
    assert sorted(p.name for p in (out / "returned").glob("*.csv")) == [f"baseline_form_{k:02d}_r1.csv" for k in range(1, 5)] + ["demographics.csv"]
    _no_personal_data(out / "returned", email, "2026/11/03")
    shutil.copy(out / "returned" / "baseline_form_01_r1.csv", out / "returned" / "baseline_form_01_r2.csv")   # a hand-imported duplicate

    agg = tmp_path / "aggregates"
    r = run("scripts/make_validation_forms.py", "score-baseline", "--items", str(release / "noilai_core.jsonl"), "--dir", str(out),
            "--returned", str(out / "returned" / "baseline_form_*.csv"), "--out", str(agg), "--n-boot", "100", "--seed", "0")
    assert "mean-human strict accuracy" in r.stdout
    rep = json.loads((agg / "human_baseline_report.json").read_text(encoding="utf-8"))
    assert sorted(p.name for p in agg.iterdir()) == ["human_baseline_report.json"]           # aggregates only under --out
    _no_personal_data(agg, email, "2026/11/03")
    # exclusions: every reason that applies is listed; forms_excluded gives the first (HUMAN_BASELINE_PROTOCOL section 5)
    assert rep["forms_returned"] == 5 and rep["forms_scored"] == 2 and rep["forms_included"] == ["01", "02"]
    assert rep["exclusions"]["by_reason"] == {"duplicate_form_response": 1, "failed_instruction_check": 1, "too_few_answers": 1,
                                              "tool_use": 1, "was_validator": 1}
    assert rep["forms_excluded"] == {"baseline_form_01_r2.csv": "duplicate_form_response", "baseline_form_03_r1.csv": "too_few_answers",
                                     "baseline_form_04_r1.csv": "failed_instruction_check"}
    reasons = {e["form"]: e["reasons"] for e in rep["exclusions"]["respondents"]}
    assert reasons["03"] == ["too_few_answers", "tool_use"] and reasons["04"] == ["failed_instruction_check", "was_validator"]
    assert rep["instruction_check"] == {"passed": 4, "failed": 1} and rep["tool_use"] == {"no": 4, "yes": 1}
    assert rep["was_validator"] == {"no": 4, "yes": 1} and rep["problems"] == {}
    assert rep["per_form"]["01"]["answered"] == 24 and rep["per_form"]["02"]["answered"] == 23
    # item-level rows: in the git-ignored data tree, never under --out; the report carries their path and hash
    item_level = out / "report" / "human_scores.jsonl"
    rows = [json.loads(ln) for ln in item_level.read_text(encoding="utf-8").splitlines()]
    assert rep["item_level"]["path"] == str(item_level) and rep["item_level"]["n_rows"] == len(rows) == 48
    assert rep["item_level"]["sha256"] == hashlib.sha256(item_level.read_bytes()).hexdigest()
    assert {r["respondent"] for r in rows} == {"01-r1", "02-r1"} and {r["form"] for r in rows} == {"01", "02"}
    assert all(r["strict"] for r in rows if r["form"] == "01") and all(r["region"] == "Nam" for r in rows if r["form"] == "01")
    by = {(r["form"], r["item_id"]): r for r in rows}
    swapped = by[("02", t1_2[0]["item_id"])]
    assert swapped["strict"] is False and swapped["lenient"] is True and swapped["tolerant"] is True and swapped["error_class"] == "lenient_only"
    blank = by[("02", t1_2[1]["item_id"])]
    assert blank["answer"] is None and blank["strict"] is False and blank["error_class"] == "unparseable"
    assert by[("02", t3_2["item_id"])]["error_class"] == "wrong"
    assert by[("02", t2_2["item_id"])]["answer"] == "[redacted]" and by[("02", t2_2["item_id"])]["strict"] is False
    assert all(r["region"] == "Bắc" for r in rows if r["form"] == "02")
    # PREREG 8.11: mean-human = the mean over items of the item's mean judgment, with the two-way bootstrap; any-human the ceiling
    per_item = {}
    for r in rows:
        per_item.setdefault(r["item_id"], []).append(r["strict"])
    expected = sum(sum(v) / len(v) for v in per_item.values()) / len(per_item)
    mh = rep["accuracy"]["overall"]["strict"]["mean_human"]
    assert mh["estimate"] == pytest.approx(expected) and mh["lo"] <= mh["estimate"] <= mh["hi"]
    assert mh["n_boot"] == 100 and mh["n_persons"] == 2 and mh["n_items"] == len(per_item) and mh["method"] == "two_way_person_item_bootstrap_mean"
    assert rep["accuracy"]["overall"]["strict"]["any_human"]["estimate"] >= mh["estimate"]
    assert rep["accuracy"]["overall"]["lenient"]["mean_human"]["estimate"] > mh["estimate"]           # the swapped answer
    assert set(rep["accuracy"]["by_task"]) == {"T1", "T2", "T3"} and set(rep["mean_human_by_task"]) == {"T1", "T2", "T3"}
    assert any(k.startswith("T1-V") for k in rep["accuracy"]["by_task_variant"]) and rep["accuracy"]["by_output_lexicality"]
    n_double = sum(1 for v in per_item.values() if len(v) == 2)
    assert rep["items"]["n_double_judged"] == n_double >= 6                                 # the anchors at least
    ag = rep["agreement"]["double_judged"]
    assert ag["correct_strict"]["n_items"] == n_double and ag["correct_strict"]["n_ratings"] == 2 * n_double
    assert ag["correct_strict"]["alpha_ci"][0] <= ag["correct_strict"]["alpha"] <= ag["correct_strict"]["alpha_ci"][1]
    assert ag["answer"]["n_items"] == n_double and -1.0 <= ag["answer"]["alpha"] <= 1.0
    assert rep["error_classes"]["T1"]["lenient_only"] == 1 and rep["error_classes"]["T1"]["unparseable"] == 1
    assert set(rep["by_region"]) == {"Bắc", "Nam"} and rep["by_region"]["Nam"]["n_respondents"] == 1
    assert rep["items"]["not_in_human_design"] == [] and rep["items"]["design_items_unanswered"] == 42 - len(per_item)
    # nothing is written under paper/
    bad = subprocess.run([PY, "scripts/make_validation_forms.py", "score-baseline", "--items", str(release / "noilai_core.jsonl"), "--dir", str(out),
                          "--returned", str(out / "returned" / "baseline_form_*.csv"), "--out", str(ROOT / "paper" / "x")],
                         cwd=ROOT, text=True, capture_output=True, check=False)
    assert bad.returncode != 0 and "refusing to write under paper/" in bad.stderr and not (ROOT / "paper" / "x").exists()


def test_sheet_scorers_write_nothing_without_returns_and_make_ingest_sheets_runs_both(tmp_path):
    """`make ingest-sheets` is the one entry point for returned human sheets: it delegates to the validation `score`
    and to `score-baseline`, and either writes nothing when no sheet has come back (CLAUDE.md step 2)."""
    import shutil
    r = run("scripts/make_validation_forms.py", "score", "--dir", str(tmp_path / "val"), "--returned", str(tmp_path / "val" / "returned" / "*"))
    assert "nothing scored, nothing written" in r.stdout and not (tmp_path / "val").exists()
    r = run("scripts/make_validation_forms.py", "score-baseline", "--items", str(tmp_path / "none.jsonl"), "--dir", str(tmp_path / "human"),
            "--returned", str(tmp_path / "human" / "returned" / "baseline_form_*.csv"), "--out", str(tmp_path / "agg"))
    assert "nothing scored, nothing written" in r.stdout and not (tmp_path / "human").exists() and not (tmp_path / "agg").exists()
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^ingest-sheets: validation-score baseline-score\s*$", mk, re.MULTILINE)
    assert re.search(r"^\.PHONY:.*\bingest-sheets\b", mk, re.MULTILINE) and "scripts/ingest_sheets.py" not in mk
    vs = mk[mk.index("validation-score:"):mk.index("baseline-score:")]
    assert "make_validation_forms.py score --dir data/validation --returned 'data/validation/returned/*'" in vs
    assert "--flags-out data/audit/validator_flags.json" in vs
    bs = mk[mk.index("baseline-score:"):mk.index("ingest-sheets:")]
    assert "make_validation_forms.py score-baseline --items $(RELEASE)/noilai_main.jsonl --dir data/human" in bs
    assert "--returned 'data/human/returned/baseline_form_*.csv' --out experiments/human" in bs
    if shutil.which("make"):
        r = subprocess.run(["make", "-n", "ingest-sheets"], cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.returncode == 0 and "make_validation_forms.py score --dir data/validation" in r.stdout
        assert "make_validation_forms.py score-baseline" in r.stdout and "ingest_sheets.py" not in r.stdout
