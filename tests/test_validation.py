"""noilai.validation: the Gate 1 validation packet (docs/gate1/VALIDATION_PROTOCOL.md; DD 10.1 / 10.2 amended 1 Oct 2026)."""
import random

import pytest

from noilai import constants as C
from noilai import validation as VA
from noilai.gen import variants as V
from noilai.stats.agreement import krippendorff_alpha, krippendorff_alpha_nominal
from noilai.vi.reencode import canonical_text


def test_sizes_are_the_amended_design():
    assert C.VALIDATION_ITEMS == 12 * (C.VALIDATION_PER_CELL + C.VALIDATION_CONTROLS_PER_CELL) == 408
    assert len(VA.CALIBRATION_SPECS) == C.VALIDATION_CALIBRATION_ITEMS
    assert 0 < C.VALIDATION_CALIBRATION_PASS < 1 and C.VALIDATION_OVERLAP < C.VALIDATION_ITEMS


@pytest.mark.parametrize("raw,lab", [("Có", "yes"), ("có", "yes"), ("YES", "yes"), ("Không", "no"), ("no", "no"),
                                     ("Không chắc", "unsure"), ("unsure", "unsure"), ("", None), (None, None), ("  ", None)])
def test_norm_label(raw, lab):
    assert VA.norm_label(raw) == lab


def test_norm_label_refuses_free_text():
    with pytest.raises(ValueError):
        VA.norm_label("maybe yes")


def test_resolve_is_the_published_adjudication_rule():
    assert VA.resolve({"A": "yes", "B": "yes"}) == ("yes", "unanimous")
    assert VA.resolve({"A": "yes", "B": "no"}) == (None, "needs_author")
    assert VA.resolve({"A": "yes", "B": "no", "C": "no"}) == ("no", "majority")
    assert VA.resolve({"A": "yes", "B": "no"}, author="no") == ("no", "author")
    assert VA.resolve({"A": "yes", "B": "unsure"}) == (None, "unsure")          # one definite label is not a decision
    assert VA.resolve({"A": "yes", "B": "unsure"}, author="yes") == ("yes", "author")
    assert VA.resolve({"A": "yes", "B": "yes"}, author="no") == ("yes", "unanimous")   # the author never overrides agreement


def test_assignment_two_and_three_validators():
    assert VA.assign(5, ["A", "B"], 0) == [["A", "B"]] * 5
    a = VA.assign(12, ["A", "B", "C"], 3)
    assert a[:3] == [["A", "B", "C"]] * 3 and all(len(x) == 2 for x in a[3:])
    from collections import Counter
    load = Counter(v for x in a for v in x)
    assert max(load.values()) - min(load.values()) <= 1
    with pytest.raises(ValueError):
        VA.assign(3, ["A"], 0)


def test_stratified_precision_weights_by_population():
    # cell T1-V1: lexical stratum (N=900) 10/10 correct, pseudo stratum (N=100) 5/10 correct -> 0.9 x 1 + 0.1 x 0.5 = 0.95
    key, final = {}, {}
    for i in range(20):
        lex = i < 10
        rid = f"B-{i}"
        key[rid] = {"control": False, "cell": "T1-V1", "stratum": "T1|V1|" + ("lexicon" if lex else "pseudo"),
                    "stratum_population": 900 if lex else 100, "weight": 90.0 if lex else 10.0}
        final[rid] = "yes" if lex or i < 15 else "no"
    key["B-c"] = {"control": True, "cell": "T1-V1", "stratum": "T1|V1|lexicon", "stratum_population": None, "weight": None}
    final["B-c"] = "no"
    p = VA.stratified_precision(final, key)
    assert p["T1-V1"]["weighted_precision"] == pytest.approx(0.95) and p["T1-V1"]["n"] == 20 and p["T1-V1"]["n_yes"] == 15
    assert p["pooled"]["weighted_precision"] == pytest.approx(0.95)


def test_masi_and_jaccard():
    a, b = frozenset({"x"}), frozenset({"x", "y"})
    assert VA.jaccard_distance(a, a) == 0 and VA.masi_distance(a, a) == 0
    assert VA.jaccard_distance(a, b) == pytest.approx(0.5) and VA.masi_distance(a, b) == pytest.approx(1 - 0.5 * 2 / 3)
    assert VA.masi_distance(frozenset({"x"}), frozenset({"y"})) == 1


def test_generic_alpha_equals_nominal_and_the_krippendorff_package():
    rng = random.Random(3)
    rs = []
    for i in range(80):
        truth = rng.random() < 0.85
        for c in "ABC":
            if rng.random() < 0.8:
                rs.append((i, c, "yes" if (truth if rng.random() < 0.9 else not truth) else "no"))
    a = krippendorff_alpha(rs, lambda x, y: float(x != y))
    assert a == pytest.approx(krippendorff_alpha_nominal(rs))
    kd = pytest.importorskip("krippendorff")
    import numpy as np
    mat = np.full((3, 80), np.nan)
    for i, c, lab in rs:
        mat["ABC".index(c), i] = 1.0 if lab == "yes" else 0.0
    assert a == pytest.approx(kd.alpha(reliability_data=mat, level_of_measurement="nominal"))


def test_calibration_keys_follow_the_engine():
    rows = VA.build_calibration(set())
    assert len(rows) == C.VALIDATION_CALIBRATION_ITEMS and len({r["row_id"] for r in rows}) == len(rows)
    by = {r["manipulation"]: r for r in rows if r["task"] == "T1"}
    for r in rows:
        assert set(r["key"]) == {"correct", "spelling", "offensive"} and r["explanation_vi"] and r["explanation_en"]
        if r["task"] == "T1" and r["manipulation"] in ("none", "placement_new", "iy_variant", "vulgar", "dialect_hoi_nga"):
            a, b = VA.phrase_syllables(r["input"])
            assert canonical_text(r["candidate"]) == canonical_text(VA.spell_pair(V.apply(r["kind"], a, b)))
            assert r["key"]["correct"] == "yes"
    assert by["wrong_variant"]["key"]["correct"] == "no" and by["misspell_k"]["key"]["spelling"] == "no"
    assert by["vulgar"]["key"]["offensive"] == "yes"
    assert by["placement_new"]["candidate"] != VA.spell_pair(VA.phrase_syllables(by["placement_new"]["candidate"]))  # new style shown


def test_calibration_avoids_release_phrases():
    first = VA.build_calibration(set())
    used = {VA.phrase_key(first[0]["input"])}
    again = VA.build_calibration(used)
    assert VA.phrase_key(again[0]["input"]) not in used


def test_controls_are_never_correct():
    from noilai.gen.generate import syl_dict
    rng = random.Random(0)
    a, b = VA.phrase_syllables("bàn ghế")
    gold = V.apply("V1", a, b)
    it = {"task": "T1", "variant": "V1", "input": "bàn ghế", "input_syllables": [syl_dict(a), syl_dict(b)],
          "gold": [VA.spell_pair(gold)], "gold_syllables": [syl_dict(gold[0]), syl_dict(gold[1])]}
    for _ in range(30):
        c = VA.make_control(it, rng)
        cc = canonical_text(c["candidate"])
        assert cc != canonical_text(it["gold"][0]) and set(cc.split()) != set(canonical_text(it["gold"][0]).split())
    t3 = {"task": "T3", "variant": "V1", "input": "bàn ghế", "input_syllables": it["input_syllables"], "gold": "yes", "candidate": "x y"}
    assert VA.make_control(t3, rng)["system_verdict"] == "no"


def test_qu_analyses_match_the_design_example():
    # DD 2.1 O5: quả hồng -> cổng hòa (glide in the rime, the parser's analysis) vs quổng hà (qu as the onset)
    a, b = VA.phrase_syllables("quả hồng")
    assert VA.spell_pair(V.apply("V1", a, b)) == "cổng hòa"
    assert VA.spell_pair(VA.qu_onset_analysis("V1", a, b)) == "quổng hà"


def test_supplementary_sheets():
    sup = VA.build_supplementary()
    assert len(sup["D1"]) == 20 and len(sup["D2"]) == 20 and len(sup["D3"]) >= 12
    for r in sup["D1"]:
        a, b = VA.phrase_syllables(r["phrase"])
        assert a.tone != b.tone and a.onset != b.onset and a.rime != b.rime      # the produced form identifies the kind
    assert any(r["phrase"] == "quốc gia" for r in sup["D2"])                     # DD 2.1 O5 (b): a quốc-type item


def test_documents_quote_the_validation_constants():
    """The DD 10.1 amendment marker and the protocol quote the sizes the code uses (repo convention: doc numbers are
    test-pinned). docs/PREREGISTRATION.md does not fix the validation size and is not edited."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    dd = (root / "docs" / "DESIGN_DECISIONS.md").read_text(encoding="utf-8")
    proto = (root / "docs" / "gate1" / "VALIDATION_PROTOCOL.md").read_text(encoding="utf-8")
    marker = dd.split("[Amendment, 1 October 2026 — what the Gate 1 packet does")[1].split("]\n")[0]
    for q in (f"VALIDATION_PER_CELL` = {C.VALIDATION_PER_CELL}", f"VALIDATION_CONTROLS_PER_CELL` = {C.VALIDATION_CONTROLS_PER_CELL}",
              f"VALIDATION_OVERLAP` = {C.VALIDATION_OVERLAP}", f"{C.VALIDATION_ITEMS} rows", f"{C.VALIDATION_CALIBRATION_ITEMS}-item calibration"):
        assert q in marker, q
    for q in (f"{C.VALIDATION_PER_CELL} + {C.VALIDATION_CONTROLS_PER_CELL} per cell", f"{C.VALIDATION_OVERLAP} rows to every validator",
              f"{round(C.VALIDATION_CONTROL_CATCH_MIN * 100)}% of the planted controls", f"{C.VALIDATION_CALIBRATION_ITEMS} keyed items",
              f"fewer than {round(C.VALIDATION_CALIBRATION_PASS * 100)}%", f"{C.VALIDATION_T2_GOLD_OVERLAP} items by everyone"):
        assert q in proto, q


def test_baseline_closing_rows_exclude(tmp_path):
    """PREREG 5.7: a respondent who used tools or who was a validator is excluded; the rows exist on every form."""
    import csv
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import make_validation_forms as MV
    assert {c["item_id"] for c in MV.CLOSING_ITEMS} == {"TOOLS", "WAS_VALIDATOR"}
    items = []
    for t in ("T1", "T2", "T3"):
        for v in ("V1", "V2", "V3", "V4"):
            for i in range(6):
                base = {"task": t, "variant": v, "input": "a b", "vulgar": False, "split": "test"}
                if t == "T3":
                    items.append({**base, "item_id": f"T3-{v}-{2 * i:06d}", "gold": "yes", "candidate": "b a", "pair_item_id": f"T3-{v}-{2 * i + 1:06d}"})
                    items.append({**base, "item_id": f"T3-{v}-{2 * i + 1:06d}", "gold": "no", "candidate": "c a", "pair_item_id": f"T3-{v}-{2 * i:06d}"})
                else:
                    items.append({**base, "item_id": f"{t}-{v}-{i:06d}", "gold": ["b a"] if t == "T1" else [{"output": "b a"}]})
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    out = tmp_path / "human"
    subprocess.run([sys.executable, "scripts/make_validation_forms.py", "baseline", "--items", str(p), "--out", str(out),
                    "--n-forms", "4", "--per-form", "24", "--no-model-prompt"], cwd=root, check=True, capture_output=True, text=True)
    with open(out / "baseline_form_01.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["item_id"] for r in rows if r["block"] == "closing"] == ["TOOLS", "WAS_VALIDATOR"]


def test_google_form_script_and_response_import(tmp_path):
    """The Apps Script parses as JavaScript (when node is installed) and a Google Forms response CSV maps back to item
    ids by the position prefix of each question title."""
    import csv
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    items = []
    for t in ("T1", "T2", "T3"):
        for v in ("V1", "V2", "V3", "V4"):
            for i in range(6):
                base = {"task": t, "variant": v, "input": "a b", "vulgar": False, "split": "test"}
                if t == "T3":
                    items.append({**base, "item_id": f"T3-{v}-{2 * i:06d}", "gold": "yes", "candidate": "b a", "pair_item_id": f"T3-{v}-{2 * i + 1:06d}"})
                    items.append({**base, "item_id": f"T3-{v}-{2 * i + 1:06d}", "gold": "no", "candidate": "c a", "pair_item_id": f"T3-{v}-{2 * i:06d}"})
                else:
                    items.append({**base, "item_id": f"{t}-{v}-{i:06d}", "gold": ["b a"] if t == "T1" else [{"output": "b a"}]})
    p = tmp_path / "items.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    out = tmp_path / "human"
    run = [sys.executable, "scripts/make_validation_forms.py"]
    subprocess.run(run + ["baseline", "--items", str(p), "--out", str(out), "--n-forms", "4", "--per-form", "24", "--no-model-prompt"],
                   cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(run + ["google-form", "--dir", str(out)], cwd=root, check=True, capture_output=True, text=True)
    js = (out / "build_forms.gs").read_text(encoding="utf-8")
    assert "setCollectEmail(false)" in js and "requireSelectExactly" in js
    if shutil.which("node"):
        r = subprocess.run(["node", "-e", "new Function(require('fs').readFileSync(process.argv[1], 'utf8'))", str(out / "build_forms.gs")],
                           capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
    with open(out / "baseline_form_02.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    (out / "responses").mkdir()
    with open(out / "responses" / "responses_form_02.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "Nhóm tuổi"] + [f"{r['position']}. {r['question_short']}" for r in rows])
        w.writerow(["t", "18–29"] + [f"ans{r['position']}" for r in rows])
    subprocess.run(run + ["import-responses", "--dir", str(out)], cwd=root, check=True, capture_output=True, text=True)
    with open(out / "returned" / "baseline_form_02_r1.csv", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert [(r["item_id"], r["answer"]) for r in back] == [(r["item_id"], f"ans{r['position']}") for r in rows]
    with open(out / "returned" / "demographics.csv", encoding="utf-8") as fh:
        demo = list(csv.DictReader(fh))
    assert demo[0]["Nhóm tuổi"] == "18–29" and "Timestamp" not in demo[0]
