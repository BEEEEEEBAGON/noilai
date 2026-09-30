"""Tokenizer audit tests: metrics on a locally trained toy tokenizer and, when the
downloaded Gemma 3 model is present, sanity checks on the real thing."""
from pathlib import Path

import pytest

from noilai.audit.tokenizers import (
    SentencePieceAdapter,
    Token,
    TokenizerAdapter,
    audit_syllable,
    linguistic_boundaries,
    normalization_census,
)
from noilai.vi.syllable import parse

ROOT = Path(__file__).resolve().parents[1]
GEMMA3 = ROOT / "data" / "external" / "gemma3_tokenizer.model"


class FakeAdapter(TokenizerAdapter):
    """Splits at fixed character offsets given per text; used to check the metrics."""
    name = "fake"

    def __init__(self, splits):
        self.splits = splits

    def encode(self, text):
        cuts = [0] + sorted(self.splits.get(text, [])) + [len(text)]
        return [Token(text=text[a:b], start=a, end=b, id=i) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]


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


@pytest.mark.skipif(not GEMMA3.exists(), reason="Gemma 3 tokenizer not downloaded")
def test_gemma3_tokenizer_sanity():
    ad = SentencePieceAdapter(GEMMA3, "gemma3")
    info = ad.normalizer_info()
    assert info["vocab_size"] == 262144
    a = audit_syllable(ad, "mèo")
    assert a is not None and a.n_tokens >= 1
    census = normalization_census(ad, ["tiếng việt", "hoà bình", "người"])
    assert census["n_samples"] == 3 and census["normalizes_nfd"] in (True, False)
