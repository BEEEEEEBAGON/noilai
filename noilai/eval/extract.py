"""Extract the answer from a model completion.

Every template asks for the answer on its own line as "Đáp án: <answer>". Models deviate:
markdown bold (**Đáp án:** mài kéo), quotes, a trailing period, NFD text, "Dap an" without
diacritics, "Đáp án" followed by the answer on the next line, an explanation after the
answer, or several answer lines (the last one wins: models correct themselves in the last
line). When no marker is present, the last non-empty line is used if it looks like an answer
(exactly two Vietnamese syllables for T1/T2, a Có/Không/yes/no token for T3).

`extract_answer(raw, task) -> (answer | None, method)` where method is one of
  marker, marker_next_line, marker_truncated, fallback_last_line, fallback_yesno, none.
The extractor finds the intended answer; it never repairs it. Correctness is decided by
noilai.eval.score.
"""
from __future__ import annotations

import re

from ..vi import unicode as U
from ..vi.syllable import try_parse

# Folded (lowercase, diacritics removed) forms of the marker. "answer" is accepted for the
# English-instruction ablation, whose templates still ask for "Đáp án:".
_MARKER = re.compile(r"(?:dap\s*an|answer)\s*[:：]", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+")
_EDGE_JUNK = " \t.,;:!?\"'“”‘’()[]{}«»*_`~-–—"
_PUNCT_CUT = re.compile(r"[.,;:!?()\[\]{}\"“”«»]|\s[-–—]\s|\s(?:hoặc|hay|or)\s", re.IGNORECASE)

# Exact Vietnamese forms, the English words, and the diacritic-less ASCII forms a model may
# type. A token WITH other diacritics (cô, cò, cỏ) is not an answer: folding it to "co"
# would read a false yes.
YES_TOKENS = {"có": "yes", "yes": "yes", "co": "yes"}
NO_TOKENS = {"không": "no", "no": "no", "khong": "no"}


def fold(s: str) -> str:
    """Lowercase, NFC, all diacritics removed (đ -> d); same length as the NFC input for
    Vietnamese text, so offsets can be mapped back."""
    return U.strip_diacritics(U.nfc(s), keep_d=False).lower()


def _fold_with_offsets(s: str) -> tuple[str, list[int]]:
    folded_chars: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(s):
        f = fold(ch)
        if not f:            # a lone combining mark (non-Vietnamese) folds to nothing
            continue
        for fc in f:
            folded_chars.append(fc)
            offsets.append(i)
    offsets.append(len(s))
    return "".join(folded_chars), offsets


def clean_answer(text: str) -> str:
    """Remove markdown emphasis, quotes and edge punctuation; collapse whitespace."""
    t = U.nfc(text).replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(_EDGE_JUNK)
    return t


def is_two_syllables(text: str) -> bool:
    words = text.split()
    return len(words) == 2 and all(try_parse(w, strict=False) is not None for w in words)


def yesno_value(token: str) -> str | None:
    """'yes' / 'no' for one token (Có, Không, yes, no, and ASCII co / khong), else None."""
    low = U.nfc(token).lower()
    if low in YES_TOKENS:
        return YES_TOKENS[low]
    if low in NO_TOKENS:
        return NO_TOKENS[low]
    return None


def yesno_token(text: str) -> str | None:
    """The first Có/Không/yes/no token of `text` (original spelling), or None."""
    for m in _WORD.finditer(U.nfc(text)):
        if yesno_value(m.group(0)) is not None:
            return m.group(0)
    return None


def _truncate_phrase(ans: str) -> tuple[str, str]:
    """Reduce a longer marker answer to the two-syllable phrase it most likely intends."""
    if is_two_syllables(ans):
        return ans, "marker"
    head = _PUNCT_CUT.split(ans, maxsplit=1)[0].strip(_EDGE_JUNK)
    if head and is_two_syllables(head):
        return head, "marker_truncated"
    words = head.split() if head else ans.split()
    # "cụm từ gốc là tiền đâu" -> the two words after the last "là"
    if "là" in words:
        k = len(words) - 1 - words[::-1].index("là")
        tail = " ".join(words[k + 1: k + 3])
        if is_two_syllables(tail):
            return tail, "marker_truncated"
    if len(words) > 2 and is_two_syllables(" ".join(words[:2])):
        return " ".join(words[:2]), "marker_truncated"
    return ans, "marker"


def extract_answer(raw: str | None, task: str) -> tuple[str | None, str]:
    """See the module docstring. `task` is T1, T2, T3 or XCOPA."""
    if not raw or not raw.strip():
        return None, "none"
    text = U.nfc(raw)
    lines = text.split("\n")
    # --- marker: the last line that carries one
    for li in range(len(lines) - 1, -1, -1):
        line = lines[li]
        folded, offsets = _fold_with_offsets(line)
        hits = list(_MARKER.finditer(folded))
        if not hits:
            continue
        m = hits[-1]
        after = line[offsets[m.end()]:]
        ans = clean_answer(after)
        method = "marker"
        if not ans:
            # answer on the next non-empty line
            for nxt in lines[li + 1:]:
                if nxt.strip():
                    ans = clean_answer(nxt)
                    method = "marker_next_line"
                    break
            if not ans:
                # a marker with nothing usable after it: the model gave no answer. Never fall
                # through to the last-line fallback, which would pick the marker line itself
                # ("Đáp án" is two syllables).
                return None, "marker_empty"
        return _finish(ans, task, method)
    # --- no marker: last non-empty line
    for line in reversed(lines):
        if not line.strip():
            continue
        ans = clean_answer(line)
        if task in ("T1", "T2"):
            if is_two_syllables(ans):
                return ans, "fallback_last_line"
            return None, "none"
        if task == "T3":
            tok = yesno_token(ans)
            return (tok, "fallback_yesno") if tok else (None, "none")
        if task == "XCOPA":
            m = re.search(r"(?<!\d)([12])(?!\d)", ans)
            return (m.group(1), "fallback_last_line") if m else (None, "none")
        return (ans, "fallback_last_line") if ans else (None, "none")
    return None, "none"


def _finish(ans: str, task: str, method: str) -> tuple[str | None, str]:
    if task == "T3":
        tok = yesno_token(ans)
        return (tok, method) if tok else (ans, method)
    if task == "XCOPA":
        m = re.match(r"^\W*([12])(?!\d)", ans)
        if m:
            return m.group(1), method
        m = re.search(r"(?<!\d)([12])(?!\d)", ans)
        return (m.group(1), method) if m else (ans, method)
    if task in ("T1", "T2"):
        ans2, method2 = _truncate_phrase(ans)
        return ans2, (method2 if method == "marker" else method)
    return ans, method


def t3_label(answer: str | None) -> str | None:
    """Map an extracted T3 answer to 'yes' / 'no' (Có, Không, yes, no; diacritic-tolerant)."""
    if not answer:
        return None
    tok = yesno_token(answer)
    return None if tok is None else yesno_value(tok)
