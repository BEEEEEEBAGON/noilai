"""Tokenizer audit: how production tokenizers split Vietnamese syllables.

For every syllable we measure
  n_tokens            tokens whose span overlaps the syllable (a token that is only the
                      leading space is not counted)
  single_token        n_tokens == 1
  boundary_alignment  share of token boundaries strictly inside the syllable that fall on a
                      linguistic boundary (onset|rime, glide|nucleus, nucleus|coda); 1.0 when
                      there is no internal boundary (STAD-style; see docs/DESIGN_DECISIONS.md)
  onset_rime_split    a boundary falls exactly between onset and rime
  tone_isolated       under NFD: the combining tone mark is in a token of its own
  byte_fallback       any token is a byte-fallback piece (<0xNN>)
and for the tokenizer as a whole the THREE-VALUED normalization census (design 6.2, 12.8),
per re-encoded form (NFD; PC = partial Windows-1258 decomposition):
  verdict_<form>      'normalizes'     ids identical to NFC on every sample (the arm is a
                                       "0 by construction" row),
                      'passes_through' ids differ and decoding the ids returns the input
                                       exactly (NFC(decoded.strip()) == NFC(input.strip())),
                      'corrupts'       decoding does not return the input on some sample
                                       (e.g. an HF Precompiled nmt_nfkc normalizer deleting
                                       NFD tone marks, huggingface/tokenizers #2334): the arm
                                       is refused on that engine.
  normalizes_nfd      legacy two-valued field (= verdict_nfd == 'normalizes'), kept for readers
  normalizer_spec     SentencePiece normalizer name when available

Two adapters: SentencePiece model files (works offline; Gemma) and Hugging Face fast
tokenizers (needs the hub or a local snapshot; everything else).
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from ..vi import unicode as U
from ..vi.syllable import Parse, try_parse


@dataclass
class Token:
    text: str      # piece text as the tokenizer sees it (may include a leading '▁' / 'Ġ' / space marker)
    start: int     # char offset in the (normalized) input
    end: int
    id: int


class TokenizerAdapter:
    name: str

    def encode(self, text: str) -> list[Token]:  # pragma: no cover - interface
        raise NotImplementedError

    def ids(self, text: str) -> list[int]:
        return [t.id for t in self.encode(text)]

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover - interface
        """Detokenize ids to text WITHOUT clean-up (design 6.2: the round trip is
        NFC(decoded.strip()) == NFC(input.strip()))."""
        raise NotImplementedError

    def normalizer_info(self) -> dict:
        return {}


class SentencePieceAdapter(TokenizerAdapter):
    def __init__(self, model_path: str | Path, name: str | None = None):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.name = name or Path(model_path).stem
        self.model_path = str(model_path)

    def encode(self, text: str) -> list[Token]:
        try:
            proto = self.sp.encode(text, return_type="proto")          # sentencepiece >= 0.2.1
        except (TypeError, ValueError):
            proto = self.sp.encode(text, out_type="immutable_proto")   # older releases
        # offsets in the proto are BYTE offsets into the normalized text; map to chars
        norm = proto.text if hasattr(proto, "text") and proto.text else text
        b2c = _byte_to_char_map(norm)
        out = []
        for p in proto.pieces:
            out.append(Token(text=p.piece, start=b2c[p.begin], end=b2c[p.end], id=p.id))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(list(ids))

    def normalizer_info(self) -> dict:
        try:
            from sentencepiece import sentencepiece_model_pb2 as pb

            m = pb.ModelProto()
            m.ParseFromString(open(self.model_path, "rb").read())
            return {
                "type": "sentencepiece",
                "model_type": pb.TrainerSpec.ModelType.Name(m.trainer_spec.model_type),
                "vocab_size": m.trainer_spec.vocab_size,
                "normalizer": m.normalizer_spec.name,
                "byte_fallback": m.trainer_spec.byte_fallback,
                "add_dummy_prefix": m.normalizer_spec.add_dummy_prefix,
                "remove_extra_whitespaces": m.normalizer_spec.remove_extra_whitespaces,
                "split_by_unicode_script": m.trainer_spec.split_by_unicode_script,
            }
        except Exception as e:
            return {"type": "sentencepiece", "error": str(e)}


class HFAdapter(TokenizerAdapter):
    def __init__(self, name_or_path: str, name: str | None = None, **kw):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path, **kw)
        self.name = name or name_or_path.replace("/", "__")

    def encode(self, text: str) -> list[Token]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        toks = self.tok.convert_ids_to_tokens(enc["input_ids"])
        return [Token(text=t, start=s, end=e, id=i) for t, (s, e), i in zip(toks, enc["offset_mapping"], enc["input_ids"])]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def normalizer_info(self) -> dict:
        info = {"type": "huggingface", "class": type(self.tok).__name__, "vocab_size": len(self.tok)}
        try:
            backend = self.tok.backend_tokenizer
            info["normalizer"] = json.loads(backend.to_str())["normalizer"]
            info["pre_tokenizer"] = json.loads(backend.to_str())["pre_tokenizer"]
            info["model_type"] = json.loads(backend.to_str())["model"]["type"]
        except Exception as e:
            info["error"] = str(e)
        return info


def _byte_to_char_map(text: str) -> list[int]:
    m = []
    for ci, ch in enumerate(text):
        m.extend([ci] * len(ch.encode("utf-8")))
    m.append(len(text))
    return m


# ------------------------------------------------------------------ syllable-level metrics
def linguistic_boundaries(parse: Parse, surface: str) -> dict[str, int | None]:
    """Character offsets (within `surface`, NFC) of the onset|rime, glide|nucleus and
    nucleus|coda boundaries, or None when the component is empty."""
    s = parse.syllable
    onset_sp = parse.onset_spelling
    if onset_sp == "gi" and parse.rime_spelling and not parse.rime_spelling.startswith("i"):
        onset_len = 2  # contracted gi: written rime begins after 'gi'
    else:
        onset_len = len(onset_sp)
    if onset_sp == "qu":
        onset_len = 1  # the glide u belongs to the rime for alignment purposes (q|u...)
    n = len(surface)
    coda_len = 0
    if s.coda:
        coda_len = {"j": 1, "w": 1}.get(s.coda, len(s.coda))
    glide_len = 1 if (s.glide and not (onset_sp == "gi")) else 0
    b_onset = onset_len if onset_len else None
    b_glide = onset_len + glide_len if s.glide else None
    b_coda = n - coda_len if coda_len else None
    return {"onset_rime": b_onset, "glide_nucleus": b_glide, "nucleus_coda": b_coda}


@dataclass
class SyllableAudit:
    syllable: str
    encoding: str
    n_tokens: int
    tokens: list[str]
    boundaries: list[int]
    single_token: bool
    boundary_alignment: float
    onset_rime_split: bool
    tone_isolated: bool
    byte_fallback: bool


def audit_syllable(adapter: TokenizerAdapter, syllable: str, encoding: str = "nfc", context: str = " ") -> SyllableAudit | None:
    """Tokenize `context + syllable` (the leading space puts the syllable in word-initial
    position as it appears in running text) and measure the split of the syllable."""
    p = try_parse(syllable, strict=False)
    if p is None:
        return None
    surf_nfc = U.nfc(syllable)
    surf = U.nfd(surf_nfc) if encoding == "nfd" else surf_nfc
    text = context + surf
    toks = adapter.encode(text)
    off = len(context)
    # tokens overlapping the syllable span, ignoring a token that is only the space marker
    inside = [t for t in toks if t.end > off and t.start < len(text) and not (t.end <= off)]
    inside = [t for t in inside if t.end > off]
    n_tokens = len(inside)
    # internal boundaries in NFC-letter units for alignment (NFD offsets are mapped back)
    bounds = sorted({t.start for t in inside if off < t.start < len(text)})
    bounds_nfc = [_nfd_to_nfc_offset(surf, b - off) for b in bounds] if encoding == "nfd" else [b - off for b in bounds]
    ling = linguistic_boundaries(p, surf_nfc)
    ling_set = {v for v in ling.values() if v is not None}
    if bounds_nfc:
        aligned = sum(1 for b in bounds_nfc if b in ling_set)
        alignment = aligned / len(bounds_nfc)
    else:
        alignment = 1.0
    onset_rime_split = ling["onset_rime"] is not None and ling["onset_rime"] in bounds_nfc
    tone_isolated = False
    if encoding == "nfd":
        for t in inside:
            piece = text[t.start:t.end]
            if piece and all(ch in U.TONE_MARK_SET for ch in piece):
                tone_isolated = True
    byte_fallback = any(t.text.startswith("<0x") and t.text.endswith(">") for t in inside)
    return SyllableAudit(syllable=surf_nfc, encoding=encoding, n_tokens=n_tokens,
                         tokens=[t.text for t in inside], boundaries=bounds_nfc, single_token=n_tokens == 1,
                         boundary_alignment=alignment, onset_rime_split=onset_rime_split,
                         tone_isolated=tone_isolated, byte_fallback=byte_fallback)


def _nfd_to_nfc_offset(surf_nfd: str, k: int) -> float:
    """Map a char offset in an NFD string to the NFC letter index; a boundary inside a
    letter (between base and combining mark) maps to a half offset, which never equals a
    linguistic boundary."""
    letters = 0
    for i, ch in enumerate(surf_nfd):
        if i == k:
            return letters if not unicodedata.combining(ch) else letters - 0.5
        if not unicodedata.combining(ch):
            letters += 1 if i else 1
    return letters


CENSUS_FORMS = {"nfd": U.nfd, "pc": U.to_nfd_partial_windows1258}
CENSUS_VERDICTS = ("normalizes", "passes_through", "corrupts")


def roundtrip_ok(adapter: TokenizerAdapter, text: str, ids: Sequence[int]) -> bool:
    """Design 6.2: decoding the ids returns the input string exactly, i.e.
    NFC(decoded.strip()) == NFC(input.strip())."""
    return U.nfc(adapter.decode(ids).strip()) == U.nfc(text.strip())


def census_verdict(same_ids: int, roundtrip_failures: int, n: int) -> str | None:
    if n == 0:
        return None
    if same_ids == n:
        return "normalizes"
    return "corrupts" if roundtrip_failures > 0 else "passes_through"


def normalization_census(adapter: TokenizerAdapter, samples: Sequence[str], forms: dict | None = None,
                         max_examples: int = 10) -> dict:
    """Three-valued census per re-encoded form (design 6.2): for every sample compare the ids
    of the form with the ids of the NFC form and check the decode round trip of the form.
    Emits `verdict_<form>` in {'normalizes', 'passes_through', 'corrupts'}, the id-equality
    and round-trip fractions, the first failing examples, the legacy `normalizes_nfd`, and
    `verdict` (the NFD verdict, the C1 arm's)."""
    forms = forms if forms is not None else CENSUS_FORMS
    n = len(samples)
    out: dict = {"n_samples": n, "forms": sorted(forms)}
    nfc_ids = [adapter.ids(U.nfc(s)) for s in samples]
    for form, fn in forms.items():
        same_ids = longer = byte_fb = rt_fail = 0
        failures = []
        for s, a in zip(samples, nfc_ids):
            text = fn(s)
            toks = adapter.encode(text)
            b = [t.id for t in toks]
            same_ids += a == b
            longer += len(b) > len(a)
            byte_fb += any(t.text.startswith("<0x") for t in toks)
            if not roundtrip_ok(adapter, text, b):
                rt_fail += 1
                if len(failures) < max_examples:
                    failures.append({"input": s, "decoded": adapter.decode(b)})
        out[f"{form}_same_ids_frac"] = same_ids / n if n else None
        out[f"{form}_longer_frac"] = longer / n if n else None
        out[f"{form}_byte_fallback_frac"] = byte_fb / n if n else None
        out[f"{form}_roundtrip_ok_frac"] = (n - rt_fail) / n if n else None
        out[f"{form}_roundtrip_failures"] = failures
        out[f"verdict_{form}"] = census_verdict(same_ids, rt_fail, n)
    out["normalizes_nfd"] = (out.get("verdict_nfd") == "normalizes") if n and "nfd" in forms else None
    out["verdict"] = out.get("verdict_nfd")
    out["roundtrip_rule"] = "NFC(decoded.strip()) == NFC(input.strip()), clean_up_tokenization_spaces=False (design 6.2)"
    out.update(adapter.normalizer_info())
    return out


def census_probe_set(words: Sequence[str], item_texts: Sequence[str], n_words: int = 500, n_items: int = 500,
                     seed: int = 0) -> list[str]:
    """The fixed 1,000-string census probe set of design 6.2: 500 multi-syllable words and
    500 NóiLái item inputs, drawn with a fixed seed (fewer when the pools are smaller)."""
    import random

    rng = random.Random(seed)
    multi = sorted({w for w in words if " " in w})
    items = sorted(set(item_texts))
    pick_w = rng.sample(multi, min(n_words, len(multi)))
    pick_i = rng.sample(items, min(n_items, len(items)))
    return pick_w + pick_i


def audit_inventory(adapter: TokenizerAdapter, syllables: Iterable[str], encodings=("nfc", "nfd")) -> dict:
    rows: list[dict] = []
    for syl in syllables:
        for enc in encodings:
            a = audit_syllable(adapter, syl, enc)
            if a:
                rows.append(asdict(a))
    summary = {}
    for enc in encodings:
        sub = [r for r in rows if r["encoding"] == enc]
        if not sub:
            continue
        summary[enc] = {
            "n": len(sub),
            "tokens_per_syllable_mean": mean(r["n_tokens"] for r in sub),
            "single_token_frac": mean(r["single_token"] for r in sub),
            "boundary_alignment_mean": mean(r["boundary_alignment"] for r in sub),
            "onset_rime_split_frac": mean(r["onset_rime_split"] for r in sub),
            "tone_isolated_frac": mean(r["tone_isolated"] for r in sub),
            "byte_fallback_frac": mean(r["byte_fallback"] for r in sub),
            "n_tokens_hist": _hist(r["n_tokens"] for r in sub),
        }
    return {"tokenizer": adapter.name, "summary": summary, "rows": rows}


def _hist(values: Iterable[int]) -> dict[str, int]:
    h: dict[str, int] = {}
    for v in values:
        h[str(v)] = h.get(str(v), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def audit_phrase_tokens(adapter: TokenizerAdapter, phrase: str) -> dict:
    """Token count per syllable of a multi-syllable phrase in running-text position."""
    out = []
    for w in phrase.split():
        a = audit_syllable(adapter, w, "nfc")
        out.append(a.n_tokens if a else None)
    return {"phrase": phrase, "tokens_per_syllable": out, "total": sum(x for x in out if x)}
