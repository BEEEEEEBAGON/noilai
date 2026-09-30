"""Invariants of a built release (skipped when data/release/v0.1 is absent). Run before any
model run: the item files must match their manifest and every item must be well formed."""
import json
from collections import Counter
from pathlib import Path

import pytest

from noilai.gen.generate import TWIN_TYPES, load_items
from noilai.vi import lexicon as L
from noilai.vi.syllable import spell, try_parse

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data" / "release" / "v0.1"

pytestmark = pytest.mark.skipif(not (REL / "noilai_test.jsonl").exists(), reason="no built release")


@pytest.fixture(scope="module")
def release():
    m = json.loads((REL / "manifest.json").read_text())
    dev, test, core = (load_items(REL / f"noilai_{k}.jsonl") for k in ("dev", "test", "core"))
    return m, dev, test, core


def test_counts_match_manifest(release):
    m, dev, test, core = release
    assert len(dev) + len(test) == m["n_items"]
    counts = Counter(f"{it['task']}-{it['variant']}-{it['split']}" for it in dev + test)
    assert counts == Counter(m["counts"])
    assert Counter(f"{it['task']}-{it['variant']}" for it in core) == Counter(m["core_counts"])
    assert all(it["in_core"] for it in core) and {it["item_id"] for it in core} <= {it["item_id"] for it in test}


def test_every_item_is_well_formed(release):
    m, dev, test, core = release
    inv = L.load_inventory()
    ids = set()
    for it in dev + test:
        assert it["item_id"] not in ids
        ids.add(it["item_id"])
        if it["split"] == "test":
            assert it["canary"] == m["canary"]
        else:
            assert "canary" not in it
        texts = [it["input"]]
        if it["task"] == "T1":
            texts.append(it["gold"][0])
        elif it["task"] == "T2":
            texts += [g["output"] for g in it["gold"]]
        else:
            texts.append(it["correct_output"])
            if it["gold"] == "yes":
                texts.append(it["candidate"])
            else:
                assert it["twin_type"] in TWIN_TYPES
        for t in texts:
            for w in t.split():
                p = try_parse(w, strict=True)
                assert p is not None and spell(p.syllable) == w, (it["item_id"], w)
                assert inv.is_legal(p.syllable, "onset_rime"), (it["item_id"], w)
        for w in it["input"].split():        # qu- is excluded from INPUTS only (outputs spell /k/+glide as qu unambiguously)
            p = try_parse(w, strict=True)
            assert not (p.syllable.onset == "c" and p.syllable.glide), (it["item_id"], w)


def test_t3_pairs_are_complete(release):
    m, dev, test, core = release
    by_id = {it["item_id"]: it for it in dev + test if it["task"] == "T3"}
    for it in by_id.values():
        mate = by_id[it["pair_item_id"]]
        assert mate["pair_item_id"] == it["item_id"] and mate["gold"] != it["gold"] and mate["input"] == it["input"]
        assert mate["split"] == it["split"] and mate["in_core"] == it["in_core"]


def test_attested_file(release):
    rows = load_items(REL / "attested.jsonl")
    assert rows and all(r["task"] == "attested" for r in rows)
    assert sum(1 for r in rows if r["rule_matches_attested"]) >= 10
