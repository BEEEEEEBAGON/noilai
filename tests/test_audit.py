"""Tokenizer audit tests: metrics on a locally trained toy tokenizer, the three-valued census
on fake adapters, the item-audit covariates, and, when the downloaded Gemma 3 model is
present, sanity checks on the real thing."""
import json
import sys
from pathlib import Path

import pytest

from noilai.audit.tokenizers import (
    CENSUS_VERDICTS,
    SentencePieceAdapter,
    Token,
    TokenizerAdapter,
    audit_syllable,
    census_probe_set,
    linguistic_boundaries,
    normalization_census,
)
from noilai.vi import unicode as U
from noilai.vi.syllable import parse

ROOT = Path(__file__).resolve().parents[1]
GEMMA3 = ROOT / "data" / "external" / "gemma3_tokenizer.model"
sys.path.insert(0, str(ROOT / "scripts"))


class FakeAdapter(TokenizerAdapter):
    """Splits at fixed character offsets given per text; used to check the metrics."""
    name = "fake"

    def __init__(self, splits):
        self.splits = splits

    def encode(self, text):
        cuts = [0] + sorted(self.splits.get(text, [])) + [len(text)]
        return [Token(text=text[a:b], start=a, end=b, id=i) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]


class CharAdapter(TokenizerAdapter):
    """One token per code point, ids = code points: a byte-exact pass-through tokenizer.
    `transform` emulates an engine normalizer applied before tokenization."""
    name = "chars"

    def __init__(self, transform=None):
        self.transform = transform or (lambda s: s)

    def encode(self, text):
        t = self.transform(text)
        return [Token(text=ch, start=i, end=i + 1, id=ord(ch)) for i, ch in enumerate(t)]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


SAMPLES = ["tiếng việt", "hoà bình", "rất đẹp", "người"]


def test_linguistic_boundaries():
    p = parse("mài")
    assert linguistic_boundaries(p, "mài") == {"onset_rime": 1, "glide_nucleus": None, "nucleus_coda": 2}
    p = parse("hoàn")
    assert linguistic_boundaries(p, "hoàn") == {"onset_rime": 1, "glide_nucleus": 2, "nucleus_coda": 3}
    p = parse("quốc")
    assert linguistic_boundaries(p, "quốc") == {"onset_rime": 1, "glide_nucleus": 2, "nucleus_coda": 3}
    p = parse("ăn")
    assert linguistic_boundaries(p, "ăn") == {"onset_rime": None, "glide_nucleus": None, "nucleus_coda": 1}
    p = parse("giếng")
    assert linguistic_boundaries(p, "giếng") == {"onset_rime": 2, "glide_nucleus": None, "nucleus_coda": 3}


def test_alignment_metric_on_fake_splits():
    # ' mài' split as ' m' + 'ài'  -> boundary at onset|rime: aligned, onset_rime_split
    a = audit_syllable(FakeAdapter({" mài": [2]}), "mài")
    assert a.n_tokens == 2 and a.boundary_alignment == 1.0 and a.onset_rime_split
    # ' mài' split as ' mà' + 'i'  -> boundary at nucleus|coda: aligned but not onset split
    a = audit_syllable(FakeAdapter({" mài": [3]}), "mài")
    assert a.boundary_alignment == 1.0 and not a.onset_rime_split
    # ' hoàn' split as ' ho' + 'àn' -> glide|nucleus boundary: aligned
    a = audit_syllable(FakeAdapter({" hoàn": [3]}), "hoàn")
    assert a.boundary_alignment == 1.0
    # ' thương' split ' th' + 'ươ' + 'ng' -> both aligned
    a = audit_syllable(FakeAdapter({" thương": [3, 5]}), "thương")
    assert a.n_tokens == 3 and a.boundary_alignment == 1.0
    # ' thương' split ' t' + 'hương' -> inside the onset: misaligned
    a = audit_syllable(FakeAdapter({" thương": [2]}), "thương")
    assert a.boundary_alignment == 0.0
    # single token
    a = audit_syllable(FakeAdapter({}), "mài")
    assert a.n_tokens == 1 and a.single_token and a.boundary_alignment == 1.0


def test_nfd_tone_isolation_detected():
    import unicodedata
    s = " " + unicodedata.normalize("NFD", "mài")   # ' m a ̀ i' -> 5 chars
    a = audit_syllable(FakeAdapter({s: [3, 4]}), "mài", encoding="nfd")   # ' ma' + '̀' + 'i'
    assert a.tone_isolated and a.n_tokens == 3
    # a boundary between base and combining mark is never a linguistic boundary
    assert a.boundary_alignment < 1.0


# ------------------------------------------------------------------ the three-valued census (design 6.2)
def test_census_passes_through_normalizes_and_corrupts_are_told_apart():
    passthrough = normalization_census(CharAdapter(), SAMPLES)
    assert passthrough["verdict_nfd"] == "passes_through" and passthrough["verdict_pc"] == "passes_through"
    assert passthrough["normalizes_nfd"] is False and passthrough["nfd_roundtrip_ok_frac"] == 1.0 and passthrough["nfd_same_ids_frac"] == 0.0
    normalizing = normalization_census(CharAdapter(U.nfc), SAMPLES)
    assert normalizing["verdict_nfd"] == "normalizes" and normalizing["verdict_pc"] == "normalizes" and normalizing["normalizes_nfd"] is True
    # huggingface/tokenizers #2334: an nmt_nfkc-style normalizer that DELETES the combining tone marks of NFD text
    dropping = CharAdapter(lambda s: "".join(ch for ch in s if ch not in U.TONE_MARK_SET))
    corrupting = normalization_census(dropping, SAMPLES)
    assert corrupting["verdict_nfd"] == "corrupts" and corrupting["verdict"] == "corrupts"
    assert corrupting["normalizes_nfd"] is False                       # the two-valued field alone could not tell C1 from C3
    assert corrupting["nfd_roundtrip_ok_frac"] == 0.0 and corrupting["nfd_roundtrip_failures"][0]["input"] == "tiếng việt"
    assert U.nfc(corrupting["nfd_roundtrip_failures"][0]["decoded"]) == "tiêng viêt"
    assert corrupting["verdict_pc"] == "corrupts"                      # the PC form carries combining tone marks too
    for c in (passthrough, normalizing, corrupting):
        assert c["verdict_nfd"] in CENSUS_VERDICTS and c["n_samples"] == 4 and c["forms"] == ["nfd", "pc"]
    assert normalization_census(CharAdapter(), [])["verdict_nfd"] is None


def test_census_probe_set_is_500_words_plus_500_items_seeded():
    words = [f"a{i} b{i}" for i in range(700)] + ["single"] * 20
    items = [f"c{i} d{i}" for i in range(600)]
    s1 = census_probe_set(words, items, seed=0)
    s2 = census_probe_set(words, items, seed=0)
    assert len(s1) == 1000 and s1 == s2 and sum(1 for x in s1 if x.startswith("a")) == 500 and "single" not in s1
    assert census_probe_set(words, items, seed=1) != s1
    assert len(census_probe_set(words[:10], items[:3])) == 13     # smaller pools: as many as exist


# ------------------------------------------------------------------ item audit covariates (design 4.3, 8.4)
def test_phrase_stats_emits_align_among_split_and_the_identifiability_count():
    from audit_items import flatten_for_e2, phrase_stats

    from noilai.stats import e2

    # every syllable a single token: alignment among split syllables is undefined -> None; legacy mean = 1.0
    st = phrase_stats(FakeAdapter({}), "mài thương", "nfc")
    assert st["align_among_split"] is None and st["n_split"] == 0 and st["n_misaligned_split"] == 0
    assert st["boundary_alignment_mean"] == 1.0 and st["all_single_token"] and st["tokens_mean"] == 1
    assert st["tone_isolated"] is False and st["byte_fallback"] is False
    # ' mài' single, ' thương' split inside the onset: align_among_split is the SPLIT syllable's 0.0, not the mean 0.5
    st2 = phrase_stats(FakeAdapter({" thương": [2]}), "mài thương", "nfc")
    assert st2["align_among_split"] == 0.0 and st2["boundary_alignment_mean"] == 0.5
    assert st2["n_split"] == 1 and st2["n_misaligned_split"] == 1 and st2["alignment_per_syllable"] == [1.0, 0.0]
    # the flattened row passes e2.prepare_covariates and align_w is 0 whenever split == 0
    import pandas as pd
    rows = []
    for k, (adapter, phrase) in enumerate(((FakeAdapter({}), "mài thương"), (FakeAdapter({" thương": [2]}), "mài thương"),
                                            (FakeAdapter({" mài": [2], " thương": [3, 5]}), "mài thương"))):
        row = {"item_id": f"T1-V1-{k:06d}", "tokenizer": "fake", "encoding": "nfc", "input": phrase_stats(adapter, phrase, "nfc")}
        rows.append({**flatten_for_e2(row), "model": "m1", "correct": 1})
    d = e2.prepare_covariates(pd.DataFrame(rows))
    assert list(d["split"]) == [0, 1, 1] and d.loc[d["split"] == 0, "align_w"].eq(0).all()
    assert list(d["align_w"]) == [0.0, 0.0, 1.0] and list(d["n_misaligned_split"]) == [0, 1, 0]


def test_consistency_gate_compares_token_ids_and_fails_on_any_disagreement():
    from check_tokenizer_consistency import compare_adapters

    strings = [" mài", " thương", U.nfd(" mài")]
    same = compare_adapters(CharAdapter(), CharAdapter(), strings)
    assert same["ids_equal"] and same["agreement"] == 1.0 and same["disagreements"] == []

    class Shifted(CharAdapter):        # same pieces, re-indexed vocabulary: pieces agree, ids do not
        def encode(self, text):
            return [Token(t.text, t.start, t.end, t.id + 1) for t in super().encode(text)]

    shifted = compare_adapters(CharAdapter(), Shifted(), strings)
    assert not shifted["ids_equal"] and shifted["agreement"] == 0.0
    assert shifted["disagreements"][0]["hf_pieces"] == shifted["disagreements"][0]["spm_pieces"]
    one_off = compare_adapters(CharAdapter(), CharAdapter(lambda s: s.replace("ơ", "o")), strings)
    assert not one_off["ids_equal"] and one_off["n_agree"] == 2      # 99.9 %-style tolerance would have passed this


@pytest.mark.skipif(not GEMMA3.exists(), reason="Gemma 3 tokenizer not downloaded")
def test_gemma3_tokenizer_sanity():
    ad = SentencePieceAdapter(GEMMA3, "gemma3")
    info = ad.normalizer_info()
    assert info["vocab_size"] == 262144
    a = audit_syllable(ad, "mèo")
    assert a is not None and a.n_tokens >= 1
    census = normalization_census(ad, ["tiếng việt", "hoà bình", "người"])
    # design 6.2 / 7.2: Gemma 3 has the identity normalizer and PASSES THROUGH (all three samples carry diacritics,
    # so no NFD string may share its ids with the NFC string; every decode round-trips)
    assert census["n_samples"] == 3 and census["normalizes_nfd"] is False and census["nfd_same_ids_frac"] < 0.1
    assert census["verdict_nfd"] == "passes_through" and census["verdict_pc"] == "passes_through"
    assert census["nfd_longer_frac"] > 0 and census["nfd_roundtrip_ok_frac"] == 1.0 and census["normalizer"] == "identity"
    assert ad.decode(ad.ids("hoà bình")) == "hoà bình"


@pytest.mark.skipif(not GEMMA3.exists(), reason="Gemma 3 tokenizer not downloaded")
def test_audit_items_rows_carry_the_e2_columns_on_the_release_core(tmp_path):
    import subprocess

    core = ROOT / "data" / "release" / "v0.3" / "noilai_core.jsonl"
    if not core.exists():
        pytest.skip("no v0.3 core file")
    out = tmp_path / "items.jsonl"
    subprocess.run([sys.executable, "scripts/audit_items.py", "--items", str(core), "--spm", f"{GEMMA3}:g3", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    nfc = [r for r in rows if r["encoding"] == "nfc"]
    assert all("align_among_split" in r["input"] and "n_misaligned_split" in r["input"] for r in rows)
    single = [r for r in nfc if r["input"]["all_single_token"]]
    assert single and all(r["input"]["align_among_split"] is None for r in single)
    split = [r for r in nfc if r["input"]["n_split"] > 0]
    assert split and any(r["input"]["align_among_split"] < r["input"]["boundary_alignment_mean"] for r in split
                         if r["input"]["n_split"] < len(r["input"]["tokens_per_syllable"]))
    assert any(r["input"]["tone_isolated"] for r in rows if r["encoding"] == "nfd")


# ------------------------------------------------------------------ inventory summary emitters (design 8.4; RL-2026-09-30-06)
def test_audit_inventory_emits_alignment_among_split_and_n_split():
    from noilai.audit.tokenizers import audit_inventory

    # ' mài' single token (1.0 in the all-syllable mean, absent among split), ' thương' split inside the onset (0.0)
    res = audit_inventory(FakeAdapter({" thương": [2]}), ["mài", "thương"], encodings=("nfc",))
    s = res["summary"]["nfc"]
    assert s["n"] == 2 and s["boundary_alignment_mean"] == 0.5 and s["boundary_alignment_among_split_mean"] == 0.0 and s["n_split"] == 1
    # nothing split: the among-split mean is undefined, not 1.0
    s2 = audit_inventory(FakeAdapter({}), ["mài", "thương"], encodings=("nfc",))["summary"]["nfc"]
    assert s2["boundary_alignment_among_split_mean"] is None and s2["n_split"] == 0 and s2["boundary_alignment_mean"] == 1.0


@pytest.mark.skipif(not (ROOT / "data" / "audit" / "gemma3.json").exists(), reason="no committed Gemma 3 audit")
def test_committed_gemma3_audit_carries_the_verdicts_and_the_among_split_fields():
    """The committed audit was regenerated by scripts/audit_tokenizers.py on the 1,000-string probe set (design 6.2);
    its rows CSV reproduces the among-split summary field."""
    import csv
    from statistics import mean

    d = json.loads((ROOT / "data" / "audit" / "gemma3.json").read_text(encoding="utf-8"))
    c = d["normalization_census"]
    assert c["verdict_nfd"] in CENSUS_VERDICTS and c["verdict_pc"] in CENSUS_VERDICTS and c["n_samples"] == 1000
    assert c["probe_set"]["n_words"] == 500 and c["probe_set"]["n_items"] == 500 and not c["probe_set"]["items_file"].startswith("/")
    for enc in ("nfc", "nfd"):
        s = d["summary"][enc]
        assert s["boundary_alignment_among_split_mean"] < s["boundary_alignment_mean"] and 0 < s["n_split"] < s["n"]
    with (ROOT / "data" / "audit" / "gemma3_rows.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = [float(r["boundary_alignment"]) for r in rows if r["encoding"] == "nfc" and r["single_token"] == "False"]
    assert len(split) == d["summary"]["nfc"]["n_split"] and mean(split) == pytest.approx(d["summary"]["nfc"]["boundary_alignment_among_split_mean"])


# ------------------------------------------------------------------ scripts/reconcile_counts.py emitters (design 2.1, 2.3, 2.4, 4.1, 6.2, 9.1)
def _run_script(*args, cwd=ROOT):
    import subprocess

    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return r


@pytest.fixture(scope="module")
def counts(tmp_path_factory):
    out = tmp_path_factory.mktemp("counts") / "counts.json"
    _run_script("scripts/reconcile_counts.py", "--out", str(out), "--release", str(ROOT / "data" / "release" / "v0.3"))
    return json.loads(out.read_text(encoding="utf-8"))


def test_reconcile_counts_emits_the_design_document_tables(counts):
    h = counts["hunspell"]
    # 2.1: the WRITTEN onset table (c 254 / k 91 / g 146 / gh 26 differ from the re-spelled `onset_spelling` by the loans ka, gen)
    ow = h["onset_written"]
    assert sum(ow.values()) == h["parsable"] == 6595
    assert {k: ow[k] for k in ("", "c", "k", "g", "gh", "qu", "gi", "t")} == {"": 291, "c": 254, "k": 91, "g": 146, "gh": 26, "qu": 127, "gi": 157, "t": 360}
    assert ow["c"] + ow["k"] == counts["distributions_new_file"]["onset_spelling"]["c"] + counts["distributions_new_file"]["onset_spelling"]["k"]
    # 2.3 C11: stop codas take only sắc or nặng, up to the three blacklisted exceptions gip têt xit
    ct, cols = h["coda_tone"], h["coda_tone_columns"]
    assert cols == ["ngang", "huyen", "sac", "hoi", "nga", "nang"]
    assert [ct["c"][t] for t in cols] == [0, 0, 197, 0, 0, 145] and [ct["ch"][t] for t in cols] == [0, 0, 62, 0, 0, 55]
    assert [ct["p"][t] for t in cols] == [1, 0, 155, 0, 0, 126] and [ct["t"][t] for t in cols] == [2, 0, 279, 0, 0, 229]
    assert sum(sum(r.values()) for r in ct.values()) == 6595 and "open" in ct
    # 2.3: 162 orthographic rime spellings, the y-doublets and the qu-only rimes folded as the document says
    ro = h["rimes_orthographic"]
    assert h["n_rimes_orthographic"] == len(ro) == 162 == counts["inventory"]["extended_rimes"]
    assert {"y", "ynh", "yt", "yêm", "yên", "yêng", "yêt", "yêu", "in", "iêng", "ôc", "âc"} <= set(ro) and "uôc" in ro and "uc" in ro
    assert not any(r.startswith("q") for r in ro) and all(r == r.lower() for r in ro)
    # 2.4: byte lengths by tone; hỏi and nặng are 3 bytes on every vowel, sắc is 2 bytes on y (ý) and 3 on ỳ ỷ ỹ ỵ
    bl = counts["byte_length"]
    assert set(bl) == {"ngang", "huyen", "sac", "hoi", "nga", "nang"}
    assert set(bl["hoi"]["letters"].values()) == {3} and set(bl["nang"]["letters"].values()) == {3}
    assert bl["ngang"]["letters"]["a"] == 1 and bl["ngang"]["letters"]["ă"] == 2
    assert bl["sac"]["letters"]["y"] == 2 and bl["huyen"]["letters"]["y"] == 3 and bl["nga"]["letters"]["y"] == 3 and bl["nga"]["letters"]["e"] == 3
    assert bl["huyen"]["two_byte"] == ["a", "e", "i", "o", "u"] and bl["nga"]["two_byte"] == ["a", "i", "o", "u"]
    # 4.1: two-syllable entries vs distinct canonical pairs (49,103 / 47,535)
    w = counts["wordlist"]
    assert w["two_syllable_entries"] == w["two_syllable_pairs"] == 49103 and w["distinct_canonical_pairs"] == 47535
    # 6.2 / 9.1: per tokenizer the three-valued verdicts and the single-token share by parsed tone
    if "gemma3" in counts["tokenizer_audit"]:
        g = counts["tokenizer_audit"]["gemma3"]
        assert g["verdict_nfd"] in CENSUS_VERDICTS and g["verdict_pc"] in CENSUS_VERDICTS and g["normalizes_nfd"] is False
        assert g["census_n_samples"] == 1000 and "boundary_alignment_among_split_mean" in g["summary"]["nfc"]
        st = g["single_token_by_tone"]["nfc"]
        assert set(st) == {"ngang", "huyen", "sac", "hoi", "nga", "nang"} and sum(v["n"] for v in st.values()) >= 6590
        assert st["ngang"]["single_token_frac"] > max(v["single_token_frac"] for k, v in st.items() if k != "ngang") + 0.15


def test_reconcile_counts_release_strata_block_and_its_absence(counts, tmp_path):
    rs = counts["release_strata"]
    assert rs["status"] == "computed" and rs["release"] == "data/release/v0.3" and rs["n_items"] == 10000
    cells = rs["cells"]
    assert set(cells) == {f"{t}-{v}" for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4")}
    for c in cells.values():
        assert c["degenerate"] == c["degenerate_share"] * c["n_items"] and 0 <= c["c2_affected_test"] <= c["c2_affected"] <= c["n_items"]
        assert c["degenerate"] >= max(c["same_tone"], c["same_onset"], c["same_rime"])
        assert c["output_lexical_share"] is not None and c["n_test"] <= c["n_items"]
    assert sum(c["c2_affected_test"] for c in cells.values()) == rs["c2_affected_test_total"] > 0
    assert cells["T1-V1"]["same_tone"] > cells["T1-V2"]["same_tone"]           # V2 drops equal-tone pairs as reversals (3.4)
    assert rs["manifest"]["content_sha256"] and "seed" not in json.dumps(rs)
    # absent directory: a note, never a guess
    out = tmp_path / "c.json"
    _run_script("scripts/reconcile_counts.py", "--out", str(out), "--release", str(tmp_path / "nowhere"))
    assert json.loads(out.read_text(encoding="utf-8"))["release_strata"]["status"] == "not_computed"
    out2 = tmp_path / "c2.json"
    _run_script("scripts/reconcile_counts.py", "--out", str(out2))
    assert "pass --release" in json.loads(out2.read_text(encoding="utf-8"))["release_strata"]["note"]


# ------------------------------------------------------------------ scripts/count_placement.py per-file counts (design 6.3)
def test_count_placement_emits_per_file_counts(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text(json.dumps({"premise": "Hòa bình.", "choice1": "Thuý đi.", "choice2": "Nhà cao.", "idx": 0}, ensure_ascii=False) + "\n"
                 + json.dumps({"premise": "Trời mưa.", "choice1": "Đường ướt.", "choice2": "Trời nắng.", "idx": 1}, ensure_ascii=False) + "\n"
                 + "\n", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("khỏe mạnh\nhello world\n", encoding="utf-8")
    out = tmp_path / "pl.json"
    _run_script("scripts/count_placement.py", str(a), str(b), "--out", str(out))
    rep = json.loads(out.read_text(encoding="utf-8"))
    fa, fb = rep["per_file"][str(a)], rep["per_file"][str(b)]
    assert fa == {"n_items": 2, "n_texts": 6, "n_word_tokens": 12, "n_syllable_tokens": 12, "affected_tokens": 2, "affected_items": 1,
                  "old": 1, "new": 1}
    assert fb["n_items"] == 2 and fb["n_word_tokens"] == 4 and fb["n_syllable_tokens"] == 2 and fb["affected_tokens"] == 1 and fb["affected_items"] == 1
    assert rep["n_items"] == 4 and rep["affected_tokens"] == 3 and rep["affected_items"] == 2 and rep["old"] == 2 and rep["new"] == 1
    assert rep["n_syllable_tokens"] == 14 < rep["n_word_tokens"] == 16 and set(rep["definitions"]) >= {"n_items", "affected_items"}


@pytest.mark.skipif(not (ROOT / "data" / "external" / "xcopa_test_vi.jsonl").exists(), reason="XCOPA-vi not downloaded")
def test_committed_placement_xcopa_files_reproduce_from_the_script(tmp_path):
    """data/audit/placement_xcopa*.json are written by scripts/count_placement.py, never hand-edited: a fresh run
    reproduces them, and the test file's block carries the numbers the design document cites (43 of 500 items)."""
    committed = json.loads((ROOT / "data" / "audit" / "placement_xcopa.json").read_text(encoding="utf-8"))
    out = tmp_path / "pl.json"
    _run_script("scripts/count_placement.py", "data/external/xcopa_test_vi.jsonl", "data/external/xcopa_val_vi.jsonl", "--out", str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == committed
    test = committed["per_file"]["data/external/xcopa_test_vi.jsonl"]
    assert test["n_items"] == 500 and test["affected_items"] == 43 and test["affected_tokens"] == 53
    assert test["n_syllable_tokens"] < test["n_word_tokens"] == 10137
    if (ROOT / "data" / "audit" / "placement_xcopa_test.json").exists():
        only = json.loads((ROOT / "data" / "audit" / "placement_xcopa_test.json").read_text(encoding="utf-8"))
        assert only["per_file"] == {"data/external/xcopa_test_vi.jsonl": test} and only["affected_items"] == 43


# ------------------------------------------------------------------ scripts/make_validation_forms.py (design 10.1 AC1; 10.2 baseline design)
def _synthetic_items(per_cell: int = 25) -> list[dict]:
    items = []
    for t in ("T1", "T2", "T3"):
        for v in ("V1", "V2", "V3", "V4"):
            for i in range(per_cell):
                base = {"task": t, "variant": v, "input": f"a{i} b{i}", "vulgar": i % 11 == 10, "split": "test", "canary": "x"}
                if t == "T1":
                    items.append({**base, "item_id": f"T1-{v}-{i:06d}", "gold": [f"b{i} a{i}"]})
                elif t == "T2":
                    items.append({**base, "item_id": f"T2-{v}-{i:06d}", "gold": [{"output": f"b{i} a{i}"}]})
                else:
                    yes_id, no_id = f"T3-{v}-{2*i:06d}", f"T3-{v}-{2*i+1:06d}"
                    twin = "spelling" if i % 7 == 6 else "other_variant"
                    items.append({**base, "item_id": yes_id, "gold": "yes", "candidate": f"b{i} a{i}", "pair_item_id": no_id, "twin_type": None})
                    items.append({**base, "item_id": no_id, "gold": "no", "candidate": f"c{i} a{i}", "pair_item_id": yes_id, "twin_type": twin})
    return items


def _coverage(out: Path):
    import csv
    from collections import Counter

    forms = sorted(out.glob("baseline_form_*.csv"))
    rows = []
    for f in forms:
        with f.open(encoding="utf-8") as fh:
            rows.append(list(csv.DictReader(fh)))
    return forms, rows, Counter(r["item_id"] for fm in rows for r in fm)


def test_baseline_forms_follow_the_246_item_design(tmp_path):
    """DD 10.2: 20 forms x 30 items; 6 anchors on every form; every non-anchor item on exactly two forms; 246 distinct
    items; the rater graph connected (cyclic double coverage: 140 form pairs share one item, 50 share two)."""
    items_path = tmp_path / "main.jsonl"
    items_path.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in _synthetic_items()) + "\n", encoding="utf-8")
    out = tmp_path / "human"
    r = _run_script("scripts/make_validation_forms.py", "baseline", "--items", str(items_path), "--out", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    forms, rows, cover = _coverage(out)
    assert len(forms) == 20 and all(len(fm) == 30 and len({r["item_id"] for r in fm}) == 30 for fm in rows)
    manifest = json.loads((out / "baseline_manifest.json").read_text(encoding="utf-8"))
    anchors = set(manifest["anchor_ids"])
    assert len(anchors) == 6 and all(cover[a] == 20 for a in anchors)
    assert {a.split("-")[0] + "-" + a.split("-")[1] for a in anchors} == {"T1-V1", "T1-V2", "T1-V3", "T1-V4", "T2-V1", "T3-V1"}
    others = {k: v for k, v in cover.items() if k not in anchors}
    assert len(others) == 240 and set(others.values()) == {2} and len(cover) == 246
    assert info["shared_items_per_form_pair"] == {"1": 140, "2": 50} and info["rater_graph_connected"] is True
    assert json.loads((out / "human_items.json").read_text(encoding="utf-8")) == sorted(cover)
    # eligibility: no vulgar item, no spelling twin, never both members of a twin pair, T3 half yes / half no per cell
    by_id = {it["item_id"]: it for it in _synthetic_items()}
    chosen = [by_id[k] for k in cover]
    assert not any(it["vulgar"] for it in chosen) and not any(it.get("twin_type") == "spelling" for it in chosen)
    t3 = [it for it in chosen if it["task"] == "T3" and it["item_id"] not in anchors]
    assert len(t3) == 80 and not any(it["pair_item_id"] in anchors for it in t3)        # never the twin of an anchor
    for v in ("V1", "V2", "V3", "V4"):
        cell = [it for it in t3 if it["variant"] == v]
        assert len(cell) == 20 and sum(it["gold"] == "yes" for it in cell) == 10
        assert not any(it["pair_item_id"] in {x["item_id"] for x in cell} for it in cell)
    from collections import Counter
    per_cell = Counter((it["task"], it["variant"]) for it in chosen if it["item_id"] not in anchors)
    assert set(per_cell.values()) == {20}
    # the seed is the public sampling seed, and the same seed reproduces the same forms
    assert info["seed"] == 20261102
    out2 = tmp_path / "human2"
    _run_script("scripts/make_validation_forms.py", "baseline", "--items", str(items_path), "--out", str(out2))
    assert [f.read_text(encoding="utf-8") for f in sorted(out2.glob("*.csv"))] == [f.read_text(encoding="utf-8") for f in forms]
    # a smaller design keeps the invariants (4 forms x 24 = 6 anchors + 36 items on exactly two forms)
    out3 = tmp_path / "small"
    _run_script("scripts/make_validation_forms.py", "baseline", "--items", str(items_path), "--out", str(out3), "--n-forms", "4", "--per-form", "24")
    _forms3, rows3, cover3 = _coverage(out3)
    assert len(rows3) == 4 and all(len(fm) == 24 for fm in rows3) and sorted(set(cover3.values())) == [2, 4] and len(cover3) == 42
    # sizes that do not fit are refused, not rounded
    import subprocess
    bad = subprocess.run([sys.executable, "scripts/make_validation_forms.py", "baseline", "--items", str(items_path), "--out", str(tmp_path / "bad"),
                          "--n-forms", "20", "--per-form", "40"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert bad.returncode != 0 and "cells" in bad.stderr


def test_validation_score_reports_ac1_and_marginals(tmp_path):
    import csv

    ret = tmp_path / "returned"
    ret.mkdir()
    fields = ["item_id", "task", "variant", "input", "candidate", "question_vi", "correct", "spelling", "lexical", "offensive", "comment"]
    for coder in ("A", "B"):
        with open(ret / f"validation_form_{coder}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(50):
                # 96% prevalence, coder B disagrees on two items: alpha low, AC1 high (design 10.1)
                correct = "no" if i < 2 else "yes"
                if coder == "B" and i in (2, 3):
                    correct = "no"
                w.writerow({"item_id": f"T1-V1-{i:06d}", "task": "T1", "variant": "V1", "input": "x", "candidate": "y", "question_vi": "q",
                            "correct": correct, "spelling": "yes", "lexical": "no", "offensive": "no", "comment": ""})
    out = tmp_path / "val"
    _run_script("scripts/make_validation_forms.py", "score", "--out", str(out), "--returned", str(ret / "*.csv"))
    rep = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    c = rep["correct"]
    assert {"alpha", "alpha_ci", "percent_agreement", "n_ratings", "ac1", "ac1_ci", "marginals"} <= set(c)
    assert c["percent_agreement"] == 0.96 and c["ac1"] > 0.9 > c["alpha"] and c["ac1_ci"][0] <= c["ac1"] <= c["ac1_ci"][1]
    assert c["marginals"] == {"no": 0.06, "yes": 0.94} and rep["spelling"]["ac1"] == 1.0 and rep["spelling"]["marginals"] == {"yes": 1.0}
