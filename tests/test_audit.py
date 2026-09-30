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

    core = ROOT / "data" / "release" / "v0.2" / "noilai_core.jsonl"
    if not core.exists():
        pytest.skip("no v0.2 core file")
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
