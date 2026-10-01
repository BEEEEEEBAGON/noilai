"""Invariants of a built release (skipped when no release is present). Run before any
model run: the item files must match their manifest and every item must be well formed."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from noilai.gen.generate import (
    TWIN_TYPES,
    content_hash,
    load_items,
    load_reserved,
    read_header,
    release_canary,
)
from noilai.vi import lexicon as L
from noilai.vi.reencode import convert_placement
from noilai.vi.syllable import spell, try_parse

ROOT = Path(__file__).resolve().parents[1]
REL = next((p for p in sorted((ROOT / "data" / "release").glob("v*"), reverse=True) if (p / "noilai_test.jsonl").exists()), None)

pytestmark = pytest.mark.skipif(REL is None, reason="no built release")


@pytest.fixture(scope="module")
def release():
    m = json.loads((REL / "manifest.json").read_text())
    dev, test, core = (load_items(REL / f"noilai_{k}.jsonl") for k in ("dev", "test", "core"))
    return m, dev, test, core


@pytest.fixture(scope="module")
def canary():
    """The release's canary: header record of the test file (design 4.6); the manifests must agree with it."""
    c = read_header(REL / "noilai_test.jsonl")["canary"]
    assert c.startswith("NOILAI-CANARY-") and release_canary(REL) == c
    return c


def test_counts_match_manifest(release, canary):
    m, dev, test, core = release
    assert len(dev) + len(test) == m["n_items"]
    counts = Counter(f"{it['task']}-{it['variant']}-{it['split']}" for it in dev + test)
    assert counts == Counter(m["counts"])
    assert Counter(f"{it['task']}-{it['variant']}" for it in core) == Counter(m["core_counts"])
    assert all(it["in_core"] for it in core) and {it["item_id"] for it in core} <= {it["item_id"] for it in test}
    # the public manifest names the canary by its SHA-256 (design 4.6); a manifest written before the public/private
    # split (v0.2) still carries the GUID and must equal the header's
    assert "canary" in m or "canary_sha256" in m
    if "canary" in m:
        assert m["canary"] == canary
    if "canary_sha256" in m:
        assert m["canary_sha256"] == hashlib.sha256(canary.encode()).hexdigest()
        assert "seed" not in m and "canary" not in m and "seed" not in m.get("generator_args", {})
    priv = REL / "manifest_private.json"
    if priv.exists():
        d = json.loads(priv.read_text())
        assert d["canary"] == canary and isinstance(d["seed"], int) and d["content_sha256"] == m["content_sha256"]


def test_content_hash_recomputes_from_the_item_files(release):
    """Design 4.6: `content_sha256` is the identity of the release (items without the canary fields, in item_id order,
    as the generator wrote them); it must recompute from the dev + test files, not merely be copied."""
    m, dev, test, _core = release
    items = sorted(dev + test, key=lambda it: it["item_id"])
    assert len(items) >= 100                                                        # non-vacuity
    assert content_hash(items) == m["content_sha256"]
    assert content_hash(items[1:]) != m["content_sha256"]                            # the check can fail


def test_sample_files_match_their_recorded_hashes(release):
    """scripts/sample_items.py records `samples.<name>.sha256`; the file on disk must hash to it (the verify gate
    compares the same digest before a run) and its counts must match the recorded ones."""
    m = release[0]
    samples = m.get("samples") or {}
    if not samples:
        pytest.skip("no sampled files recorded (scripts/sample_items.py not run on this release)")
    assert set(samples) >= {"main"}
    for name, rec in samples.items():
        path = REL / rec["file"]
        assert path.exists(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"], name
        items = load_items(path)
        assert len(items) == rec["n_items"] > 0, name
        key = (lambda it: f"{it['task']}-{it['variant']}") if name == "main" else (lambda it: it["variant"])
        assert dict(Counter(key(it) for it in items)) == rec["counts"], name
        assert read_header(path)["canary"] == read_header(REL / "noilai_test.jsonl")["canary"]
    if "main" in samples:
        per_cell = samples["main"]["per_cell"]
        assert all(v == per_cell for k, v in samples["main"]["counts"].items()), samples["main"]["counts"]


def _item_texts(it: dict) -> list[str]:
    """Every well-formed text of an item: input, T1 gold, T2 readings, T3 correct output and the `yes` candidate
    (the misspelled `spelling` twin of a T3 `no` item is not a word and is left out)."""
    texts = [it["input"]]
    if it["task"] == "T1":
        texts.append(it["gold"][0])
    elif it["task"] == "T2":
        texts.extend(g["output"] for g in it["gold"])
    else:
        texts.append(it["correct_output"])
        if it["gold"] == "yes":
            texts.append(it["candidate"])
    return texts


def test_no_item_text_uses_a_reserved_syllable_or_pair(release, request):
    """Design 4.1 / 4.4 filter 6 (tests-health-1): no text of any released item shares a syllable with a demonstration
    pair or equals an attested two-syllable window (`load_reserved()`). A release built before the reserved-OUTPUT
    screen (v0.2: no `attested_strings_sha256` in its manifest) is expected to fail until the v0.3 rebuild."""
    m, dev, test, _core = release
    if "attested_strings_sha256" not in m:
        request.applymarker(pytest.mark.xfail(strict=True, reason="release built before the reserved-output screen "
                                                                  "(design 12.30: rebuilt as v0.3); remove once rebuilt"))
    reserved = load_reserved()
    assert reserved.syllables and reserved.pairs
    bad = []
    n_words = 0
    for it in dev + test:
        for t in _item_texts(it):
            words = [spell(try_parse(w, strict=False).syllable) for w in t.split()]
            n_words += len(words)
            if any(w in reserved.syllables for w in words) or (len(words) == 2 and frozenset(words) in reserved.pairs):
                bad.append((it["item_id"], t))
    assert n_words > 10_000                                                          # non-vacuity
    assert not bad, f"{len(bad)} item texts use a reserved syllable or pair, e.g. {bad[:5]}"


def test_every_item_is_well_formed(release, canary):
    m, dev, test, _core = release
    inv = L.load_inventory()
    ids = set()
    for it in dev + test:
        assert it["item_id"] not in ids
        ids.add(it["item_id"])
        if it["split"] == "test":
            assert it["canary"] == canary and it["do_not_train"] and it["evaluation_only"]
        else:
            assert "canary" not in it and not it["vulgar"]
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
            assert convert_placement(t, m["placement_style"]) == t, (it["item_id"], t)
            for w in t.split():
                p = try_parse(w, strict=True)
                assert p is not None and L.emit(p.syllable, m["placement_style"], inv) == w, (it["item_id"], w)
                assert inv.is_legal(p.syllable, "onset_rime"), (it["item_id"], w)
        for w in it["input"].split():        # qu- is excluded from INPUTS only
            p = try_parse(w, strict=True)
            assert not (p.syllable.onset == "c" and p.syllable.glide), (it["item_id"], w)
        assert it["variant"] == it["variant_labels"][0]


def test_t3_pairs_are_complete(release):
    _m, dev, test, _core = release
    by_id = {it["item_id"]: it for it in dev + test if it["task"] == "T3"}
    for it in by_id.values():
        mate = by_id[it["pair_item_id"]]
        assert mate["pair_item_id"] == it["item_id"] and mate["gold"] != it["gold"] and mate["input"] == it["input"]
        assert mate["split"] == it["split"] and mate["in_core"] == it["in_core"]


def test_attested_file(release):
    rows = load_items(REL / "attested.jsonl")
    assert rows and all(r["task"] == "attested" for r in rows)
    assert sum(1 for r in rows if r["rule_matches_attested"]) >= 10
    assert all(("positions" in r and "exactness" in r and "eligible_h6" in r) for r in rows)
