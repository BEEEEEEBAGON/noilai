"""Tests for the generator: legality, determinism, splits, T2/T3 invariants."""
import json
from collections import Counter

import pytest

from noilai.gen import variants as V
from noilai.gen.generate import Generator, TWIN_TYPES, load_items, write_release
from noilai.vi import lexicon as L
from noilai.vi.reencode import canonical_text
from noilai.vi.syllable import Syllable, spell, try_parse


@pytest.fixture(scope="module")
def build():
    g = Generator(seed=11)
    b = g.build(n_lexicon=150, n_pseudo=80, per_cell_t1=60, per_cell_t2=30, per_cell_t3=25, core_per_cell=10)
    return g, b


def _syl(d):
    return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])


def test_every_output_syllable_is_legal_and_standard(build):
    g, b = build
    for it in b["items"]:
        texts = []
        if it["task"] == "T1":
            texts.append(it["gold"][0])
        elif it["task"] == "T2":
            texts.extend(x["output"] for x in it["gold"])
        else:
            texts.append(it["correct_output"])
            if it["gold"] == "yes":
                texts.append(it["candidate"])
        texts.append(it["input"])
        for t in texts:
            for w in t.split():
                p = try_parse(w, strict=True)
                assert p is not None, (it["item_id"], t)
                assert spell(p.syllable) == w                       # standard, new-style, lowercase
                assert g.inv.is_legal(p.syllable, "onset_rime"), (it["item_id"], w)


def test_t1_gold_is_the_variant_output_and_not_the_input(build):
    g, b = build
    for it in (x for x in b["items"] if x["task"] == "T1"):
        a, c = (_syl(d) for d in it["input_syllables"])
        x, y = V.apply(it["variant"], a, c)
        assert it["gold"][0] == f"{spell(x)} {spell(y)}"
        assert it["gold"][0] != it["input"]


def test_t2_gold_contains_the_base_pair_and_only_lexical_readings(build):
    g, b = build
    for it in (x for x in b["items"] if x["task"] == "T2"):
        outs = [r["output"] for r in it["gold"]]
        assert it["source"] == "lexicon"
        assert any(tuple(o.split()) in g.lex_pairs for o in outs)
        for r in it["gold"]:
            x, y = (_syl(d) for d in it["input_syllables"])
            o = V.apply(r["variant"], x, y)
            text = f"{spell(o[1])} {spell(o[0])}" if r["reversed"] else f"{spell(o[0])} {spell(o[1])}"
            assert text == r["output"] and tuple(text.split()) in g.lex_pairs


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
            assert it["candidate"] != it["input"]
            if it["twin_type"] == "spelling":
                # a misspelling: not parseable in strict mode, but same structure as the gold
                bad = [w for w in it["candidate"].split() if try_parse(w, strict=True) is None]
                assert len(bad) == 1
        else:
            assert it["candidate"] == it["correct_output"]


def test_split_is_by_base_pair_and_core_is_inside_test(build):
    g, b = build
    split_of = {}
    for it in b["items"]:
        split_of.setdefault(it["base_pair_id"], it["split"])
        assert split_of[it["base_pair_id"]] == it["split"]
        if it["in_core"]:
            assert it["split"] == "test" and it["canary"] == b["canary"]
        if it["split"] == "test":
            assert it["canary"].startswith("NOILAI-CANARY-")
        else:
            assert "canary" not in it
    core = Counter((it["task"], it["variant"]) for it in b["items"] if it["in_core"])
    assert all(v == 10 for v in core.values()) and len(core) == 12


def test_qu_syllables_are_excluded_by_default(build):
    g, b = build
    for it in b["items"]:
        for w in it["input"].split():
            s = try_parse(w).syllable
            assert not (s.onset == "c" and s.glide), it["item_id"]
    g2 = Generator(seed=11, exclude_qu=False)
    pairs = g2.base_pairs(200, 100)
    assert any(p.a.onset == "c" and p.a.glide or p.b.onset == "c" and p.b.glide for p in pairs)


def test_build_is_deterministic():
    a = Generator(seed=5).build(n_lexicon=40, n_pseudo=20, per_cell_t1=15, per_cell_t2=8, per_cell_t3=6, core_per_cell=4)
    b = Generator(seed=5).build(n_lexicon=40, n_pseudo=20, per_cell_t1=15, per_cell_t2=8, per_cell_t3=6, core_per_cell=4)
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
    c = Generator(seed=6).build(n_lexicon=40, n_pseudo=20, per_cell_t1=15, per_cell_t2=8, per_cell_t3=6, core_per_cell=4)
    assert c["canary"] != a["canary"]


def test_write_and_load_release(build, tmp_path):
    g, b = build
    m = write_release(b, tmp_path)
    dev, test, core = (load_items(tmp_path / f"noilai_{k}.jsonl") for k in ("dev", "test", "core"))
    assert len(dev) + len(test) == m["n_items"] == len(b["items"])
    assert all(it["in_core"] for it in core) and len(core) == sum(m["core_counts"].values())
    assert set(m["resource_sha256"]) >= {"vi-DauMoi.dic", "Viet74K.txt"}
