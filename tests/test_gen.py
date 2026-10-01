"""Tests for the generator: legality, determinism, splits, T2/T3 invariants (v0.2 design)."""
import json
from collections import Counter, defaultdict

import pytest

from noilai.gen import variants as V
from noilai.gen.generate import (
    MARGINAL_KEEP,
    MARGINAL_MIN_TYPES,
    MARGINAL_RIMES,
    QUOTA_FEATURES,
    TWIN_TYPES,
    BasePair,
    Generator,
    VulgarLexicon,
    attested_strings_hash,
    c2_affected,
    canary_sha256,
    load_items,
    public_manifest,
    read_header,
    release_canary,
    write_release,
)
from noilai.vi import lexicon as L
from noilai.vi.reencode import canonical_text, convert_placement, count_placement_changes
from noilai.vi.syllable import STOP_CODAS, Syllable, spell, try_parse

CANARY = "NOILAI-CANARY-00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module")
def build():
    g = Generator(seed=11)
    b = g.build(n_lexicon=150, n_pseudo=80, per_cell_t1=60, per_cell_t2=30, per_cell_t3=25, core_per_cell=10, canary=CANARY)
    return g, b


def _syl(d):
    return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])


def _texts(it):
    """Every well-formed text of an item (the misspelled `spelling` twin of a T3 `no` item is left out)."""
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


def _shown(it):
    """The text the prompt shows: input + gold (T1) or input + candidate (T3)."""
    return it["input"] + " " + (it["gold"][0] if it["task"] == "T1" else it["candidate"])


def _canon_words(text):
    return [spell(try_parse(w, strict=False).syllable) for w in text.split()]


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
    _, b = build
    for it in (x for x in b["items"] if x["task"] == "T1"):
        a, c = (_syl(d) for d in it["input_syllables"])
        labels = it["variant_labels"]
        assert it["variant"] == labels[0] and it["variant"] in V.VARIANTS
        for v in labels:
            x, y = V.apply(v, a, c)
            assert canonical_text(it["gold"][0]) == canonical_text(f"{spell(x)} {spell(y)}")
        assert canonical_text(it["gold"][0]) not in (canonical_text(it["input"]), canonical_text(" ".join(reversed(it["input"].split()))))
        assert set(it["strata"]) >= {"variant_labels", "c2_affected", "same_onset", "same_tone", "same_rime", "degenerate", "attested_overlap"}


def test_no_duplicate_output_strings_per_base_pair_and_variant_cells_are_v1_v4(build):
    _, b = build
    seen = Counter()
    for it in (x for x in b["items"] if x["task"] == "T1"):
        seen[(it["base_pair_id"], canonical_text(it["gold"][0]))] += 1
    assert max(seen.values()) == 1
    assert {it["variant"] for it in b["items"]} <= set(V.VARIANTS)


def test_t2_gold_contains_the_base_pair_and_only_lexical_readings(build):
    g, b = build
    t2 = [x for x in b["items"] if x["task"] == "T2"]
    assert t2
    for it in t2:
        assert it["source"] == "lexicon" and it["gold_validated"] is False
        outs = [canonical_text(r["output"]) for r in it["gold"]]
        assert len(outs) == len(set(outs)) == it["strata"]["n_readings"]
        # design 3.4 / 5.2: neither the input nor its plain two-word reversal is ever a gold reading
        assert canonical_text(it["input"]) not in outs, it["item_id"]
        assert canonical_text(" ".join(reversed(it["input"].split()))) not in outs, it["item_id"]
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
    _, b = build
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
    _, b = build
    split_of = {}
    for it in b["items"]:
        # design 4.6: every item of one base pair is in ONE split, flagged items included
        split_of.setdefault(it["base_pair_id"], it["split"])
        assert split_of[it["base_pair_id"]] == it["split"], it["base_pair_id"]
        if it["vulgar"]:
            assert it["split"] == "test" and not it["in_core"] and it["vulgar_reason"]
        if it["in_core"]:
            assert it["split"] == "test"
        if it["split"] == "test":
            assert it["canary"] == CANARY and it["do_not_train"] and it["evaluation_only"]
        else:
            assert "canary" not in it
    assert any(it["vulgar"] for it in b["items"])            # the rule above was exercised
    assert b["dev_base_pairs_moved_for_vulgar"] >= 1
    core = Counter((it["task"], it["variant"]) for it in b["items"] if it["in_core"])
    assert all(v == 10 for v in core.values()) and len(core) == 12


def test_reserved_pairs_and_syllables_are_excluded(build):
    """Design 4.1 / 4.4 filter 6: no item text (input, T1 gold, T2 input and readings, T3
    candidates) shares a syllable with a demonstration pair or equals an attested window."""
    g, b = build
    n_words = 0
    for it in b["items"]:
        texts = _texts(it) + ([it["candidate"]] if it["task"] == "T3" and it["gold"] == "no" else [])
        for t in texts:
            words = _canon_words(t)
            n_words += len(words)
            for w in words:
                assert w not in g.reserved.syllables, (it["item_id"], t, w)
            if len(words) == 2:
                assert frozenset(words) not in g.reserved.pairs, (it["item_id"], t)
    assert n_words > 1000                              # non-vacuity
    assert frozenset(("mèo", "cái")) in g.reserved.pairs and frozenset(("mài", "kéo")) in g.reserved.pairs
    assert "trung" in g.reserved.syllables            # demonstration pair (prompts/demos.yaml)
    assert any(k.startswith("out:") and k.endswith(":reserved") for k in b["drops"])      # the output screen fired


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
    kw = {"n_lexicon": 40, "n_pseudo": 20, "per_cell_t1": 15, "per_cell_t2": 8, "per_cell_t3": 6, "core_per_cell": 4, "canary": CANARY}
    a = Generator(seed=5).build(**kw)
    b = Generator(seed=5).build(**kw)
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
    c = Generator(seed=5).build(**{**kw, "canary": None})
    assert c["canary"].startswith("NOILAI-CANARY-") and c["canary"] != CANARY


def test_generator_seed_is_required_and_an_int():
    """Design 4.6 (item 51): no default build seed; a caller must pass one."""
    with pytest.raises(TypeError):
        Generator()                                   # the missing argument is the point
    with pytest.raises(TypeError):
        Generator(seed="11")
    with pytest.raises(TypeError):
        Generator(seed=True)
    import inspect
    assert inspect.signature(Generator.__init__).parameters["seed"].default is inspect.Parameter.empty


def test_write_release_splits_the_public_and_private_manifests(build, tmp_path):
    """Design 4.6: manifest.json (tracked) carries neither `seed` nor `canary` (nor either inside `generator_args`) and
    names the canary by its SHA-256; manifest_private.json (git-ignored) carries everything."""
    import hashlib
    _, b = build
    extra = {"seed": 11, "generator_args": {"seed": 11, "canary": CANARY, "n_pseudo": 80}, "git_commit": "abc"}
    full = write_release(b, tmp_path, manifest_extra=extra)
    pub = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    priv = json.loads((tmp_path / "manifest_private.json").read_text(encoding="utf-8"))
    assert full == priv and priv["seed"] == 11 and priv["canary"] == CANARY and priv["generator_args"]["seed"] == 11
    assert "seed" not in pub and "canary" not in pub and "seed" not in pub["generator_args"] and "canary" not in pub["generator_args"]
    assert pub["generator_args"]["n_pseudo"] == 80 and pub["git_commit"] == "abc" and pub["private_manifest"] == "manifest_private.json"
    assert pub["canary_sha256"] == priv["canary_sha256"] == hashlib.sha256(CANARY.encode()).hexdigest() == canary_sha256(CANARY)
    assert CANARY not in (tmp_path / "manifest.json").read_text(encoding="utf-8") and "11" not in json.dumps(pub.get("seed"))
    assert {k for k in priv if k not in ("seed", "canary")} <= set(pub) | {"canary_sha256"}   # nothing else is dropped
    assert pub["content_sha256"] == priv["content_sha256"] == b["content_sha256"]
    assert public_manifest(priv) == pub
    # the release's canary is recoverable without the private file (header record), and prefers it when present
    assert release_canary(tmp_path) == CANARY
    (tmp_path / "manifest_private.json").unlink()
    assert release_canary(tmp_path) == CANARY == read_header(tmp_path / "noilai_test.jsonl")["canary"]


def test_write_and_load_release_with_header(build, tmp_path):
    _, b = build
    m = write_release(b, tmp_path)
    dev, test, core = (load_items(tmp_path / f"noilai_{k}.jsonl") for k in ("dev", "test", "core"))
    assert len(dev) + len(test) == m["n_items"] == len(b["items"])
    assert all(it["in_core"] for it in core) and len(core) == sum(m["core_counts"].values())
    assert read_header(tmp_path / "noilai_test.jsonl")["canary"] == CANARY and read_header(tmp_path / "noilai_dev.jsonl") is None
    assert "NEVER APPEAR IN TRAINING CORPORA" in read_header(tmp_path / "noilai_core.jsonl")["_header"]
    assert m["placement_style"] == "old" and "drops" in m and m["resource_sha256"]["vulgar_lexicon.tsv"]


def test_iy_emission_rules_and_zero_onset_exclusion(build):
    _, b = build
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
    # design 4.2: the five strata in the priority order; 50 x n_pseudo attempts at most
    assert tuple(q["features"]) == QUOTA_FEATURES == ("tone_class", "stop_coda", "spelling_trigger", "glide", "zero_onset")
    assert q["n_pseudo_target"] == 80 and q["cap_attempts"] == 50 * 80 and q["attempts"] <= q["cap_attempts"]
    assert q["n_pseudo"] == 80 - sum(q["shortfall"].values())
    assert set(q["shortfall"]) == set(q["target_share"])
    # every FILLED stratum is within 5 points of its target share; closed quotas are exempt and counted
    n_checked = n_closed = 0
    for k, target in q["target_share"].items():
        if q["shortfall"][k] > 0:
            n_closed += 1
            continue
        n_checked += 1
        assert abs(q["achieved_share"].get(k, 0.0) - target) <= 0.05, k
    assert n_checked >= 5 and n_checked + n_closed == len(q["target_share"])
    assert all(len(k.split("|")) == len(QUOTA_FEATURES) for k in q["target_share"])
    m = write_release(b, tmp_path)
    assert len(m["content_sha256"]) == 64 and m["pseudo_quota"]["n_pseudo"] == 80 and m["n_marginal_rimes"] >= 13
    assert m["marginal_rimes"] == g.marginal_rime_counts and len(m["marginal_rimes"]) == m["n_marginal_rimes"]
    assert m["degenerate_counts"] == b["degenerate_counts"] and m["dev_base_pairs_moved_for_vulgar"] == b["dev_base_pairs_moved_for_vulgar"]
    # the content hash ignores the canary: two builds with different canaries share it
    kw = {"n_lexicon": 40, "n_pseudo": 20, "per_cell_t1": 15, "per_cell_t2": 8, "per_cell_t3": 6, "core_per_cell": 4}
    a = Generator(seed=5).build(**kw, canary=CANARY)
    c = Generator(seed=5).build(**kw, canary="NOILAI-CANARY-00000000-0000-0000-0000-000000000002")
    assert a["content_sha256"] == c["content_sha256"]


# ------------------------------------------------------------- fixes of the 30 Sep 2026 review round
def test_t2_readings_never_include_the_input_or_its_reversal_on_a_collapsing_pair():
    """Design 3.4: on a same-onset pair V4 IS the plain reversal, and the reversed order of that
    output is the input itself; neither may be a T2 gold reading (a copy/reversal task).
    v0.2 item T2-V1-000010 (`chán chưa`, gold with `chán chưa` and `chứa chan`) is the case."""
    g = Generator(seed=11)
    x, y = try_parse("chán").syllable, try_parse("chưa").syllable
    assert x.onset == y.onset and ("chan", "chứa") in g.lex_pairs           # 'chan chứa' is the lexical base pair
    assert ("chán", "chưa") in g.lex_pairs                                   # the input itself is lexical: the trap
    readings = g.lexical_readings(x, y)
    outs = {canonical_text(r["output"]) for r in readings}
    assert "chan chứa" in outs
    assert canonical_text("chán chưa") not in outs and canonical_text("chưa chán") not in outs
    for r in readings:
        assert canonical_text(r["output"]) not in (canonical_text("chán chưa"), canonical_text("chưa chán"))


def test_vulgar_screen_is_pair_only_on_inputs_and_per_syllable_on_outputs():
    """Design 11.5 / item 42: `chim cu` and `mênh mông` are ordinary inputs; `dái chó` is a taboo pair."""
    vl = VulgarLexicon()
    assert vl.reason("chim cu", scope="input") is None and vl.reason("mênh mông", scope="input") is None
    assert vl.reason("chim cu") == "syllable:cu" and vl.reason("mênh mông") == "syllable:mông"
    assert vl.reason("dái chó", scope="input") == "pair:dái chó" and vl.reason("chó dái", scope="input") == "pair:dái chó"
    with pytest.raises(ValueError):
        vl.reason("chim cu", scope="gold")
    # through the generator: the syllable-blocklisted input pair survives, the taboo pair is dropped per rule
    g = Generator(seed=1, words=["chim cu", "dái chó", "mèo mun", "con gà"])
    pairs = g.base_pairs(n_lexicon=10, n_pseudo=0)
    kept = {(spell(p.a), spell(p.b)) for p in pairs}
    assert ("chim", "cu") in kept and ("dái", "chó") not in kept
    assert g.drops["base:vulgar_input:pair:dái chó"] == 1 and not any(k.startswith("base:vulgar_input:syllable") for k in g.drops)


def test_vulgar_drops_are_counted_per_blocklist_entry(build):
    _, b = build
    keys = [k for k in b["drops"] if "vulgar" in k]
    assert keys and all(":syllable:" in k or ":pair:" in k for k in keys)      # per rule, never a bare total
    assert "T1:vulgar_flagged" not in b["drops"] and "base:vulgar_input" not in b["drops"]
    flagged = [it for it in b["items"] if it["vulgar"]]
    assert flagged and all(it["vulgar_reason"].startswith(("syllable:", "pair:")) for it in flagged)


def test_c2_affected_describes_the_text_the_prompt_shows(build):
    """Design 6.3 test: the shown text changes under placement_new iff strata.c2_affected
    (T1: input + gold; T3: input + candidate), spelling twins included."""
    g, b = build
    n = Counter()
    for it in b["items"]:
        if it["task"] == "T2":
            continue
        shown = _shown(it)
        assert (count_placement_changes(shown) > 0) == it["strata"]["c2_affected"], (it["item_id"], shown)
        n[it["strata"]["c2_affected"]] += 1
    assert n[True] >= 5 and n[False] >= 5
    # the `no` twin of a T3 pair recomputes the covariates from its own candidate: on `hoa bướm`
    # the V2 gold `bươm hóa` is affected while `other_variant` twins such as `hướm boa` are not
    a, c = try_parse("hoa").syllable, try_parse("bướm").syllable
    bp = BasePair("bp-test", a, c, "lexicon", 1, 1)
    t1 = {it["variant"]: it for it in g.t1_items(bp)}["V2"]
    assert canonical_text(t1["gold"][0]) == canonical_text("bươm hóa") and t1["strata"]["c2_affected"]
    flags = set()
    for seed in range(40):
        g.rng.seed(seed)
        pr = g.t3_pair(t1, "other_variant")
        if pr is None:
            continue
        yes, no = pr
        assert yes["strata"]["c2_affected"] is True and yes["strata"]["iy_forms"] == t1["strata"]["iy_forms"]
        cand_s = [try_parse(w, strict=False).syllable for w in no["candidate"].split()]
        assert no["strata"]["c2_affected"] == (c2_affected([a, c]) or c2_affected(cand_s))
        assert no["strata"]["iy_forms"] == [L.iy_form(w) for w in (no["input"] + " " + no["candidate"]).split()]
        flags.add(no["strata"]["c2_affected"])
    assert flags == {True, False}                     # the recomputation changed at least one twin's flag


def test_marginal_rimes_follow_the_dictionary_type_count(build):
    """Design 2.3 C10: excluded iff < 4 vi-DauMoi.dic syllable types or in the loan set; uơ/ưi
    kept regardless; regular rimes such as oăc (hoặc) and oeo (khoeo) are never excluded."""
    g, b = build
    dic = Counter()
    for w in L.load_hunspell_syllables("new"):
        p = try_parse(w, strict=True)
        if p is not None and not (p.syllable.coda in STOP_CODAS and p.syllable.tone not in (2, 5)):
            dic[p.syllable.rime] += 1
    assert len(dic) >= 150
    for r in ("w+ă+c", "w+e+w", "w+ê+nh", "w+ơ", "ư+j", "a", "iê+ng"):
        assert r not in g.marginal_rimes, r
    assert MARGINAL_KEEP.isdisjoint(g.marginal_rimes)
    for r in g.marginal_rimes:
        assert r in MARGINAL_RIMES or dic[r] < MARGINAL_MIN_TYPES, (r, dic[r])
    for r, k in dic.items():
        if k < MARGINAL_MIN_TYPES and r not in MARGINAL_KEEP:
            assert r in g.marginal_rimes, (r, k)
    assert g.marginal_rime_counts == {r: dic.get(r, 0) for r in sorted(g.marginal_rimes)}
    assert b["marginal_rimes"] == g.marginal_rime_counts and "qu- included" in b["marginal_rime_convention"]
    # no generated base syllable carries a marginal rime; regular oăc syllables are reachable
    assert all(s.rime not in g.marginal_rimes for s in g.base_syllables)
    assert any(s.rime == "w+ă+c" for s in g.base_syllables)


def test_degenerate_and_attested_overlap_covariates_and_frozen_attested_hash(build, tmp_path):
    _, b = build
    n_deg = Counter()
    for it in b["items"]:
        s = it["strata"]
        assert s["degenerate"] == (s["same_tone"] or s["same_onset"] or s["same_rime"])
        assert s["attested_overlap"] is False           # by construction at build time (design 4.1)
        n_deg[f"{it['task']}-{it['variant']}"] += s["degenerate"]
    assert {k: v for k, v in n_deg.items() if v} == b["degenerate_counts"] and sum(n_deg.values()) > 0
    # the attested string hash ignores glosses/verification and changes with a string (item 57)
    h = attested_strings_hash()
    assert h and len(h) == 64 and b["attested_strings_sha256"] == h
    import csv
    with open(ROOT_SEED, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    fields = list(rows[0].keys())

    def write(rows_, path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows_)
    edited = [dict(r) for r in rows]
    edited[0]["gloss_input"] = "another gloss"
    edited[0]["verified_by"] = "someone"
    write(edited, tmp_path / "a.tsv")
    assert attested_strings_hash(tmp_path / "a.tsv") == h
    edited[0]["output"] = "xxx yyy"
    write(edited, tmp_path / "b.tsv")
    assert attested_strings_hash(tmp_path / "b.tsv") != h


ROOT_SEED = L.ROOT / "data" / "attested_seed.tsv"


def test_spelled_is_the_stored_word(build):
    _, b = build
    n = 0
    for it in b["items"]:
        assert [d["spelled"] for d in it["input_syllables"]] == it["input"].split(), it["item_id"]
        if it["task"] == "T1":
            assert [d["spelled"] for d in it["gold_syllables"]] == it["gold"][0].split(), it["item_id"]
        n += sum(convert_placement(d["spelled"], "new") != d["spelled"] for d in it["input_syllables"])
    assert n > 0                                       # some stored words differ from the new-style spelling


def test_a_base_pair_with_a_flagged_item_is_kept_whole_in_the_test_split():
    """Design 4.6 + 11.5: `cù tê` has a vulgar V2 output (`tề cu`) and clean V4/T2 items; with every
    base pair assigned to dev (dev_frac = 1) the whole pair still lands in test, so the public
    dev file never carries the input of a gated item. [NATIVE-CHECK: test data quoted from Viet74K]"""
    g = Generator(seed=1, words=["cù tê", "mèo mun", "con gà", "bàn ghế"])
    b = g.build(n_lexicon=10, n_pseudo=0, per_cell_t1=10, per_cell_t2=10, per_cell_t3=10, dev_frac=1.0,
                core_per_cell=0, canary=CANARY)
    by_bp = defaultdict(list)
    for it in b["items"]:
        by_bp[it["base_pair_id"]].append(it)
    mixed = [its for its in by_bp.values() if {it["vulgar"] for it in its} == {True, False}]
    assert len(mixed) == 1 and any(it["input"] == "cù tê" for it in mixed[0])
    assert all(it["split"] == "test" for it in mixed[0])
    assert all(it["split"] == "dev" for its in by_bp.values() if its is not mixed[0] for it in its)
    assert b["dev_base_pairs_moved_for_vulgar"] == 1
