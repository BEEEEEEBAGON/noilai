"""Tests for the generator: legality, determinism, splits, T2/T3 invariants (v0.2 design)."""
import json
from collections import Counter

import pytest

from noilai.gen import variants as V
from noilai.gen.generate import (
    TWIN_TYPES,
    Generator,
    VulgarLexicon,
    c2_affected,
    load_items,
    read_header,
    write_release,
)
from noilai.vi import lexicon as L
from noilai.vi.reencode import canonical_text, convert_placement
from noilai.vi.syllable import Syllable, spell, try_parse

CANARY = "NOILAI-CANARY-00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module")
def build():
    g = Generator(seed=11)
    b = g.build(n_lexicon=150, n_pseudo=80, per_cell_t1=60, per_cell_t2=30, per_cell_t3=25, core_per_cell=10, canary=CANARY)
    return g, b


def _syl(d):
    return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])


def _texts(it):
    texts = [it["input"]]
    if it["task"] == "T1":
        texts.append(it["gold"][0])
    elif it["task"] == "T2":
        texts.extend(x["output"] for x in it["gold"])
    else:
        texts.append(it["correct_output"])
        if it["gold"] == "yes":
            texts.append(it["candidate"])
    return texts


def test_every_syllable_is_legal_stored_old_style_with_attested_iy(build):
    g, b = build
    for it in b["items"]:
        for t in _texts(it):
            assert convert_placement(t, "old") == t, (it["item_id"], t)           # stored in the baseline style
            for w in t.split():
                p = try_parse(w, strict=True)
                assert p is not None, (it["item_id"], t)
                assert L.emit(p.syllable, "old", g.inv) == w                    # emitted spelling round-trips
                assert g.inv.is_legal(p.syllable, "onset_rime"), (it["item_id"], w)
                assert p.syllable.onset != "p"
        for w in it["input"].split():        # inputs never contain qu-
            s = try_parse(w).syllable
            assert not (s.onset == "c" and s.glide), it["item_id"]


def test_t1_gold_is_the_variant_output_filed_under_the_lowest_label(build):
    g, b = build
    for it in (x for x in b["items"] if x["task"] == "T1"):
        a, c = (_syl(d) for d in it["input_syllables"])
        labels = it["variant_labels"]
        assert it["variant"] == labels[0] and it["variant"] in V.VARIANTS
        for v in labels:
            x, y = V.apply(v, a, c)
            assert canonical_text(it["gold"][0]) == canonical_text(f"{spell(x)} {spell(y)}")
        assert canonical_text(it["gold"][0]) not in (canonical_text(it["input"]), canonical_text(" ".join(reversed(it["input"].split()))))
        assert set(it["strata"]) >= {"variant_labels", "c2_affected", "same_onset", "same_tone", "same_rime"}


def test_no_duplicate_output_strings_per_base_pair_and_variant_cells_are_v1_v4(build):
    g, b = build
    seen = Counter()
    for it in (x for x in b["items"] if x["task"] == "T1"):
        seen[(it["base_pair_id"], canonical_text(it["gold"][0]))] += 1
    assert max(seen.values()) == 1
    assert {it["variant"] for it in b["items"]} <= set(V.VARIANTS)


def test_t2_gold_contains_the_base_pair_and_only_lexical_readings(build):
    g, b = build
    for it in (x for x in b["items"] if x["task"] == "T2"):
        assert it["source"] == "lexicon" and it["gold_validated"] is False
        outs = [canonical_text(r["output"]) for r in it["gold"]]
        assert len(outs) == len(set(outs)) == it["strata"]["n_readings"]
        x, y = (_syl(d) for d in it["input_syllables"])
        for r in it["gold"]:
            for lab in r["variant_labels"]:
                v, rev = lab.rstrip("r"), lab.endswith("r")
                o = V.apply(v, x, y)
                text = f"{spell(o[1])} {spell(o[0])}" if rev else f"{spell(o[0])} {spell(o[1])}"
                assert canonical_text(text) == canonical_text(r["output"]) and tuple(text.split()) in g.lex_pairs
        for w in it["input"].split():
            s = try_parse(w).syllable
            assert not (s.onset == "c" and s.glide)


def test_t3_pairs_are_balanced_and_twins_differ(build):
    g, b = build
    t3 = [x for x in b["items"] if x["task"] == "T3"]
    assert Counter(x["gold"] for x in t3)["yes"] == Counter(x["gold"] for x in t3)["no"]
    by_id = {x["item_id"]: x for x in t3}
    for it in t3:
        mate = by_id[it["pair_item_id"]]
        assert mate["input"] == it["input"] and mate["gold"] != it["gold"]
        if it["gold"] == "no":
            assert it["twin_type"] in TWIN_TYPES
            assert canonical_text(it["candidate"]) != canonical_text(it["correct_output"])
            assert canonical_text(it["candidate"]) != canonical_text(it["input"])
            if it["twin_type"] == "spelling":
                bad = [w for w in it["candidate"].split() if try_parse(w, strict=True) is None]
                assert len(bad) == 1
        else:
            assert it["candidate"] == it["correct_output"]


def test_split_core_canary_and_vulgar_rules(build):
    g, b = build
    split_of = {}
    for it in b["items"]:
        if not it["vulgar"]:
            split_of.setdefault(it["base_pair_id"], it["split"])
            assert split_of[it["base_pair_id"]] == it["split"]
        else:
            assert it["split"] == "test" and not it["in_core"] and it["vulgar_reason"]
        if it["in_core"]:
            assert it["split"] == "test"
        if it["split"] == "test":
            assert it["canary"] == CANARY and it["do_not_train"] and it["evaluation_only"]
        else:
            assert "canary" not in it
    core = Counter((it["task"], it["variant"]) for it in b["items"] if it["in_core"])
    assert all(v == 10 for v in core.values()) and len(core) == 12


def test_reserved_pairs_and_syllables_are_excluded(build):
    g, b = build
    for it in b["items"]:
        a, c = (spell(_syl(d)) for d in it["input_syllables"])
        if it["task"] != "T2":
            assert frozenset((a, c)) not in g.reserved.pairs
            assert a not in g.reserved.syllables and c not in g.reserved.syllables
    assert frozenset(("mèo", "cái")) in g.reserved.pairs and frozenset(("mài", "kéo")) in g.reserved.pairs
    assert "trung" in g.reserved.syllables            # demonstration pair (prompts/demos.yaml)


def test_vulgar_lexicon_and_c2_stratum():
    vl = VulgarLexicon()
    assert vl.reason("đéo hiểu") and vl.reason("Lồn bàn") and vl.reason("dái chó") and vl.reason("chó dái")
    assert vl.reason("mèo cái") is None
    assert c2_affected([try_parse("hoà").syllable]) and not c2_affected([try_parse("quả").syllable])
    assert not c2_affected([try_parse("hoàn").syllable]) and not c2_affected([try_parse("hoa").syllable])


def test_qu_syllables_are_excluded_by_default_but_allowed_on_request():
    g2 = Generator(seed=11, exclude_qu=False)
    pairs = g2.base_pairs(200, 100)
    assert any((p.a.onset == "c" and p.a.glide) or (p.b.onset == "c" and p.b.glide) for p in pairs)


def test_build_is_deterministic_given_seed_and_canary():
    kw = dict(n_lexicon=40, n_pseudo=20, per_cell_t1=15, per_cell_t2=8, per_cell_t3=6, core_per_cell=4, canary=CANARY)
    a = Generator(seed=5).build(**kw)
    b = Generator(seed=5).build(**kw)
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
    c = Generator(seed=5).build(**{**kw, "canary": None})
    assert c["canary"].startswith("NOILAI-CANARY-") and c["canary"] != CANARY


def test_write_and_load_release_with_header(build, tmp_path):
    g, b = build
    m = write_release(b, tmp_path)
    dev, test, core = (load_items(tmp_path / f"noilai_{k}.jsonl") for k in ("dev", "test", "core"))
    assert len(dev) + len(test) == m["n_items"] == len(b["items"])
    assert all(it["in_core"] for it in core) and len(core) == sum(m["core_counts"].values())
    assert read_header(tmp_path / "noilai_test.jsonl")["canary"] == CANARY and read_header(tmp_path / "noilai_dev.jsonl") is None
    assert "NEVER APPEAR IN TRAINING CORPORA" in read_header(tmp_path / "noilai_core.jsonl")["_header"]
    assert m["placement_style"] == "old" and "drops" in m and m["resource_sha256"]["vulgar_lexicon.tsv"]


def test_iy_emission_rules_and_zero_onset_exclusion(build):
    g, b = build
    forbidden = {"sỹ", "vỹ", "ỳ", "ỵ", "ỹ"}
    for it in b["items"]:
        for t in _texts(it):
            for w in t.split():
                assert w not in forbidden, (it["item_id"], w)
                s = try_parse(w, strict=True).syllable
                assert not (s.onset == "" and s.rime == "i"), (it["item_id"], w)        # no zero-onset bare /i/ anywhere
        assert len(it["strata"]["iy_forms"]) == 4 and set(it["strata"]["iy_forms"]) <= {"i", "y", None}
    assert L.iy_table().get("mĩ") == "y" and L.iy_table().get("lí", "i") == "i"        # Viet74K majority forms


def test_pseudo_quota_matches_lexical_marginals_and_manifest_has_content_hash(build, tmp_path):
    g, b = build
    q = b["pseudo_quota"]
    assert q["n_pseudo"] == 80 and set(q["features"]) == {"zero_onset", "glide", "stop_coda", "tone_class"}
    # every achieved share is within 0.1 of its target share (small build: 80 pseudo pairs)
    for k, target in q["target_share"].items():
        assert abs(q["achieved_share"].get(k, 0.0) - target) <= 0.1, k
    m = write_release(b, tmp_path)
    assert len(m["content_sha256"]) == 64 and m["pseudo_quota"]["n_pseudo"] == 80 and m["n_marginal_rimes"] >= 13
    # the content hash ignores the canary: two builds with different canaries share it
    kw = dict(n_lexicon=40, n_pseudo=20, per_cell_t1=15, per_cell_t2=8, per_cell_t3=6, core_per_cell=4)
    a = Generator(seed=5).build(**kw, canary=CANARY)
    c = Generator(seed=5).build(**kw, canary="NOILAI-CANARY-00000000-0000-0000-0000-000000000002")
    assert a["content_sha256"] == c["content_sha256"]
