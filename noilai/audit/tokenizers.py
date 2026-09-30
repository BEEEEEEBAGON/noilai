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
and for the tokenizer as a whole
  normalizes_nfd      encode(NFD(x)) == encode(NFC(x)) for Vietnamese x (the census that
                      decides whether the C1 arm can have an effect by construction)
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
        except Exception as e:  # noqa: BLE001
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

    def normalizer_info(self) -> dict:
        info = {"type": "huggingface", "class": type(self.tok).__name__, "vocab_size": len(self.tok)}
        try:
            backend = self.tok.backend_tokenizer
            info["normalizer"] = json.loads(backend.to_str())["normalizer"]
            info["pre_tokenizer"] = json.loads(backend.to_str())["pre_tokenizer"]
            info["model_type"] = json.loads(backend.to_str())["model"]["type"]
        except Exception as e:  # noqa: BLE001
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


def normalization_census(adapter: TokenizerAdapter, samples: Sequence[str]) -> dict:
    """Does the tokenizer treat NFD input like NFC input?"""
    same_ids = 0
    nfd_longer = 0
    byte_fallback_nfd = 0
    for s in samples:
        a = adapter.ids(U.nfc(s))
        b_toks = adapter.encode(U.nfd(s))
        b = [t.id for t in b_toks]
        same_ids += a == b
        nfd_longer += len(b) > len(a)
        byte_fallback_nfd += any(t.text.startswith("<0x") for t in b_toks)
    n = len(samples)
    return {
        "n_samples": n,
        "nfd_same_ids_frac": same_ids / n if n else None,
        "normalizes_nfd": (same_ids == n) if n else None,
        "nfd_longer_frac": nfd_longer / n if n else None,
        "nfd_byte_fallback_frac": byte_fallback_nfd / n if n else None,
        **adapter.normalizer_info(),
    }


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
