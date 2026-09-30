"""Extract the answer from a model completion (DESIGN_DECISIONS 5.5 / 5.6; this module is canonical).

Every template asks for the answer on its own line as "Đáp án: <answer>". Models deviate:
markdown bold (**Đáp án:** mài kéo, **Đáp án**: mài kéo), "Đáp án là: ...", a dash instead of
the colon, quotes, a trailing period, NFD text, "Dap an" without diacritics, the answer on
the next line, an explanation after the answer, a leading variant label ("V1: mài kéo"), an
"input -> output" restatement, a thinking block, or several answer lines (the last one wins:
models correct themselves in the last line). When no marker is present, "Trả lời:" /
"Kết quả:" are accepted as fallback markers, and otherwise the last non-empty line is used
if it looks like an answer.

Steps (5.5):
  1. NFC.
  2. `strip_thinking`: remove <think>…</think>, <thinking>…</thinking> and
     <|channel>…<channel|> blocks; `n_thinking_chars` is recorded per row (MUST be 0 in
     main runs). An unclosed block swallows the rest of the output (truncated inside the
     thinking): the answer is then None.
  3. Marker: on the diacritic-folded line, the LAST match of
     (đáp án | answer) [emphasis]* (là)? (: | ： | - | – | —), or "đáp án là" alone; the capture
     is the rest of the line; if empty, the next non-empty line that is not itself an
     explanation (marker_next_line). Emphasis, quotes, a leading variant label and trailing
     punctuation are stripped.
  4. Task validation. T1/T2: a quoted span is preferred, the right-hand side of an arrow
     is taken, "cụm từ gốc là X Y" yields "X Y", a trailing gloss is cut. A hedge between two
     different candidate phrases ("X Y hoặc Y X") is unparseable (method `hedged`).
     T3: negation precedence (a capture whose first token is không/no/sai is "no"); otherwise
     the first yes/no token of the leading clause decides ("Câu này không đúng" is "no",
     "Câu này đúng" is "yes"); an explicit alternative joining both classes ("Có hoặc Không")
     or a leading "có thể" ("maybe") is unparseable. XCOPA: a single 1 or 2; both is a hedge.
  5. Fallbacks: a fallback marker (fallback_marker); ≤ 5 words → last line
     (fallback_last_line); T3 without a marker → first yes/no token anywhere (fallback_yesno).
  6. No lowercasing here (`canonical_text` casefolds).

`extract_answer(raw, task) -> (answer | None, method)`; `extract(raw, task)` returns the full
record (`answer, method, n_thinking_chars, thinking_unclosed, n_marker_lines, hedged`).
`method` is one of METHODS. The extractor finds the intended answer; it never repairs it.
Correctness is decided by noilai.eval.score.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..vi import unicode as U
from ..vi.syllable import try_parse

METHODS = ("marker", "marker_next_line", "marker_truncated", "marker_empty", "fallback_marker",
           "fallback_last_line", "fallback_yesno", "hedged", "none")

# Folded (lowercase, diacritics removed) forms. "answer" is accepted for the English-instruction
# ablation, whose templates still ask for "Đáp án:". Between the marker and the separator we
# allow emphasis characters and spaces (**Đáp án**:) and an optional "là"; the separator is a
# colon, a fullwidth colon or a dash. "đáp án là" without a colon is also a marker.
_EMPH = r"[\s*_`~]*"
_MARKER = re.compile(r"(?<![a-z])(?:dap\s*an|answer)" + _EMPH + r"(?:(?:la" + _EMPH + r")?[:：]|(?:la" + _EMPH
                     + r")?[-–—](?!>)|la(?=\s))", re.IGNORECASE)
# Fallback markers need the colon so that ordinary sentences ("kết quả đúng phải...") do not match.
_FALLBACK_MARKER = re.compile(r"(?<![a-z])(?:tra\s*loi|ket\s*qua)" + _EMPH + r"(?:la" + _EMPH + r")?[:：]",
                              re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+")
_EDGE_JUNK = " \t.,;:!?\"'“”‘’()[]{}«»*_`~-–—"
_PUNCT_CUT = re.compile(r"[.,;:!?()\[\]{}\"“”«»]|\s[-–—]\s|\s(?:hoặc|hay|or)\s", re.IGNORECASE)
_HEDGE_SPLIT = re.compile(r"\s(?:hoặc|hay|or)\s|\s/\s", re.IGNORECASE)
_ARROW = re.compile(r"\s*(?:->|=>|→|⇒|➜|⟶)\s*")
_QUOTED = re.compile(r"[\"“”‘’'«»](.+?)[\"“”‘’'«»]")
_LEADING_LABEL = re.compile(r"^\(?\s*(?:v\s*[1-6]|kiểu\s*[1-6]|kieu\s*[1-6]|type\s*[1-6]|variant\s*[1-6])\s*\)?"
                            r"\s*[:.\-–—)]\s*", re.IGNORECASE)
_THINK_BLOCKS = (
    re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|channel>.*?<channel\|>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<\|channel\|>(?:analysis|thought|thinking)<\|message\|>.*?<\|end\|>", re.IGNORECASE | re.DOTALL),
)
_THINK_OPEN = re.compile(r"<think>|<thinking>|<\|channel>|<\|channel\|>(?:analysis|thought|thinking)<\|message\|>",
                         re.IGNORECASE)
# Words that open a preamble ("cụm từ gốc là ...", "kết quả là ...") rather than an answer.
_PREAMBLE_WORDS = frozenset({"cụm", "kết", "câu", "đáp", "cách", "phép", "từ", "nói", "kiểu", "the", "phrase",
                             "answer", "result", "original", "output"})

# Exact Vietnamese forms, the English words, and the diacritic-less ASCII forms a model may type
# (only where the ASCII form is not itself a common word with another meaning). A token WITH other
# diacritics (cô, cò, cỏ) is not an answer: folding it to "co" would read a false yes.
YES_TOKENS = {"có": "yes", "yes": "yes", "co": "yes", "đúng": "yes", "true": "yes", "valid": "yes"}
NO_TOKENS = {"không": "no", "no": "no", "khong": "no", "sai": "no", "false": "no", "invalid": "no"}
YES_BIGRAMS = {("hợp", "lệ"): "yes", ("hop", "le"): "yes", ("chính", "xác"): "yes"}
_NEGATION_FIRST = frozenset({"không", "khong", "no", "sai", "false", "invalid"})


@dataclass(frozen=True)
class Extraction:
    answer: str | None
    method: str
    n_thinking_chars: int = 0
    thinking_unclosed: bool = False
    n_marker_lines: int = 0
    hedged: bool = False

    def as_row(self) -> dict:
        return asdict(self)


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


# ------------------------------------------------------------------ thinking blocks
def strip_thinking(raw: str) -> tuple[str, int, bool]:
    """(text without thinking blocks, n_thinking_chars, unclosed). An unclosed opening tag
    swallows everything after it (the output was cut inside the thinking)."""
    text = U.nfc(raw)
    n = 0
    for pat in _THINK_BLOCKS:
        def repl(m: re.Match) -> str:
            nonlocal n
            n += len(m.group(0))
            return " "
        text = pat.sub(repl, text)
    unclosed = False
    m = _THINK_OPEN.search(text)
    if m:
        unclosed = True
        n += len(text) - m.start()
        text = text[: m.start()]
    return text, n, unclosed


# ------------------------------------------------------------------ cleaning
def strip_leading_label(text: str) -> str:
    """Remove a leading variant label such as 'V1:', 'Kiểu 2 -', '(V3)'."""
    return _LEADING_LABEL.sub("", text, count=1).strip()


def clean_answer(text: str) -> str:
    """Remove markdown emphasis, quotes and edge punctuation, a leading variant label; collapse
    whitespace."""
    t = U.nfc(text).replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(_EDGE_JUNK)
    t = strip_leading_label(t)
    return t.strip(_EDGE_JUNK)


def is_two_syllables(text: str) -> bool:
    words = text.split()
    return len(words) == 2 and all(try_parse(w, strict=False) is not None for w in words)


def _phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(_EDGE_JUNK)


def is_explanation_line(line: str) -> bool:
    """A 'next line' that is itself a labelled explanation ('Giải thích trước: đổi vần.') rather
    than an answer: it contains a colon and more than two words."""
    t = clean_answer(line)
    return ":" in line and len(t.split()) > 2 and not is_two_syllables(t)


# ------------------------------------------------------------------ yes / no
def yesno_value(token: str) -> str | None:
    """'yes' / 'no' for one token (Có, Đúng, Không, Sai, yes, no, ASCII co / khong), else None."""
    low = U.nfc(token).lower()
    if low in YES_TOKENS:
        return YES_TOKENS[low]
    if low in NO_TOKENS:
        return NO_TOKENS[low]
    return None


def _yesno_tokens(text: str) -> list[tuple[str, str]]:
    """[(token as written, class)] in order of appearance; the bigram 'hợp lệ' counts as one."""
    words = [m.group(0) for m in _WORD.finditer(U.nfc(text))]
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(words):
        w = words[i]
        low = w.lower()
        if i + 1 < len(words) and (low, words[i + 1].lower()) in YES_BIGRAMS:
            out.append((w + " " + words[i + 1], YES_BIGRAMS[(low, words[i + 1].lower())]))
            i += 2
            continue
        v = yesno_value(w)
        if v is not None:
            out.append((w, v))
        i += 1
    return out


def yesno_token(text: str) -> str | None:
    """The first Có/Đúng/Không/Sai/yes/no token of `text` (original spelling), or None."""
    toks = _yesno_tokens(text)
    return toks[0][0] if toks else None


def _leading_clause(text: str) -> str:
    return re.split(r"[.,;:!?()\[\]\n]", U.nfc(text), maxsplit=1)[0]


def t3_label(answer: str | None) -> str | None:
    """Map an extracted T3 answer to 'yes' / 'no' with negation precedence (5.5 step 4):
    a leading không/no/sai decides 'no' ('không đúng', 'không hợp lệ'); 'có thể' ("maybe")
    opening the answer is not an answer (None); otherwise the FIRST yes/no token of the leading
    clause decides ('Câu này không đúng' -> no, 'Câu này đúng' -> yes), except that an explicit
    alternative joining both classes ('Có hoặc Không', 'Có / Không') is None."""
    if not answer:
        return None
    clause = _leading_clause(answer)
    words = [m.group(0).lower() for m in _WORD.finditer(clause)]
    if not words:
        return None
    if words[0] in _NEGATION_FIRST:
        return "no"
    if words[0] == "có" and len(words) > 1 and words[1] == "thể":
        return None
    toks = _yesno_tokens(clause)
    if not toks:
        return None
    classes = {c for _, c in toks}
    if len(classes) > 1 and _HEDGE_SPLIT.search(clause):
        return None                                  # 'Có hoặc Không': a hedge, not an answer
    return toks[0][1]


def is_hedge(ans: str) -> bool:
    """Two or more different two-syllable candidates joined by hoặc / hay / or / slash."""
    parts = [_phrase(p) for p in _HEDGE_SPLIT.split(ans)]
    parts = [p for p in parts if p]
    if len(parts) < 2 or not all(is_two_syllables(p) for p in parts):
        return False
    return len({p.lower() for p in parts}) > 1


# ------------------------------------------------------------------ T1 / T2 phrase reduction
def _is_la(word: str) -> bool:
    return U.nfc(word).lower().strip(_EDGE_JUNK) == "là"


def _truncate_phrase(ans: str) -> tuple[str, str]:
    """Reduce a longer marker answer to the two-syllable phrase it most likely intends."""
    if is_two_syllables(ans):
        return ans, "marker"
    # "input -> output": the right-hand side is the answer
    m = _ARROW.search(ans)
    if m:
        rhs = _phrase(ans[m.end():])
        if is_two_syllables(rhs):
            return rhs, "marker_truncated"
        ans = rhs or ans
    # a quoted span that is a phrase: the last one wins ('Cụm từ gốc là "tiền đâu"')
    quoted = [_phrase(q) for q in _QUOTED.findall(ans)]
    quoted = [q for q in quoted if is_two_syllables(q)]
    if quoted:
        return quoted[-1], "marker_truncated"
    words = ans.split()
    head2 = _phrase(" ".join(words[:2]))
    # "X Y là kết quả" (answer, then a gloss) vs "cụm từ gốc là X Y" (preamble, then the answer)
    if (len(words) > 2 and _is_la(words[2]) and is_two_syllables(head2)
            and U.nfc(words[0]).lower() not in _PREAMBLE_WORDS):
        return head2, "marker_truncated"
    la = [i for i, w in enumerate(words) if _is_la(w)]
    if la:
        k = la[-1]
        tail = _phrase(" ".join(words[k + 1: k + 3]))
        if is_two_syllables(tail):
            return tail, "marker_truncated"
    head = _phrase(_PUNCT_CUT.split(ans, maxsplit=1)[0])
    if head and is_two_syllables(head):
        return head, "marker_truncated"
    hw = head.split() if head else words
    if len(hw) > 2 and is_two_syllables(_phrase(" ".join(hw[:2]))):
        return _phrase(" ".join(hw[:2])), "marker_truncated"
    return ans, "marker"


# ------------------------------------------------------------------ main entry points
def extract(raw: str | None, task: str) -> Extraction:
    """See the module docstring. `task` is T1, T2, T3, attested (scored as T1) or XCOPA."""
    if not raw or not raw.strip():
        return Extraction(None, "none")
    text, n_think, unclosed = strip_thinking(raw)
    if not text.strip():
        return Extraction(None, "none", n_think, unclosed)
    lines = text.split("\n")
    n_marker_lines = sum(1 for ln in lines if _MARKER.search(fold(ln)))
    task = "T1" if task == "attested" else task

    def finish(ans: str, method: str) -> Extraction:
        answer, method2, hedged = _finish(ans, task, method)
        return Extraction(answer, method2, n_think, unclosed, n_marker_lines, hedged)

    for pattern, empty_method, line_method in ((_MARKER, "marker", "marker"), (_FALLBACK_MARKER, "fallback_marker",
                                                                                "fallback_marker")):
        for li in range(len(lines) - 1, -1, -1):
            line = lines[li]
            folded, offsets = _fold_with_offsets(line)
            hits = list(pattern.finditer(folded))
            if not hits:
                continue
            m = hits[-1]
            after = line[offsets[m.end()]:]
            ans = clean_answer(after)
            method = line_method
            if not ans:
                # the answer on the next non-empty line that is not itself an explanation
                for nxt in lines[li + 1:]:
                    if nxt.strip() and not is_explanation_line(nxt):
                        ans = clean_answer(nxt)
                        method = "marker_next_line" if pattern is _MARKER else "fallback_marker"
                        break
                if not ans:
                    # a marker with nothing usable after it: the model gave no answer. Never fall
                    # through to the last-line fallback, which would pick the marker line itself
                    # ("Đáp án" is two syllables).
                    return Extraction(None, "marker_empty" if pattern is _MARKER else "none", n_think, unclosed,
                                      n_marker_lines)
            return finish(ans, method)
    # --- no marker: last non-empty line
    n_words = len(text.split())
    for line in reversed(lines):
        if not line.strip():
            continue
        ans = clean_answer(line)
        if task in ("T1", "T2"):
            if is_hedge(ans):
                return Extraction(None, "hedged", n_think, unclosed, 0, True)
            if is_two_syllables(ans):
                return Extraction(ans, "fallback_last_line", n_think, unclosed)
            if n_words <= 5:
                ans2, meth = _truncate_phrase(ans)
                if meth == "marker_truncated":
                    return Extraction(ans2, "fallback_last_line", n_think, unclosed)
            return Extraction(None, "none", n_think, unclosed)
        if task == "T3":
            tok = yesno_token(text)          # first yes/no token anywhere in the de-thought output
            return Extraction(tok, "fallback_yesno", n_think, unclosed) if tok else Extraction(None, "none",
                                                                                                n_think, unclosed)
        if task == "XCOPA":
            digits = re.findall(r"(?<!\d)([12])(?!\d)", ans)
            if len(set(digits)) > 1:
                return Extraction(None, "hedged", n_think, unclosed, 0, True)
            return Extraction(digits[0], "fallback_last_line", n_think, unclosed) if digits else Extraction(
                None, "none", n_think, unclosed)
        return Extraction(ans, "fallback_last_line", n_think, unclosed) if ans else Extraction(None, "none",
                                                                                                n_think, unclosed)
    return Extraction(None, "none", n_think, unclosed)


def extract_answer(raw: str | None, task: str) -> tuple[str | None, str]:
    """(answer | None, method): the two-field view of `extract`."""
    e = extract(raw, task)
    return e.answer, e.method


def _finish(ans: str, task: str, method: str) -> tuple[str | None, str, bool]:
    if task == "T3":
        tok = yesno_token(_leading_clause(ans)) or yesno_token(ans)
        label = t3_label(ans)
        if label is None and tok is not None:
            # an explicit alternative ("Có hoặc Không") or "có thể": a hedge, kept as text for the log
            return ans, "hedged", True
        return (tok, method) + (False,) if tok else (ans, method, False)
    if task == "XCOPA":
        digits = re.findall(r"(?<!\d)([12])(?!\d)", ans)
        if len(set(digits)) > 1:
            return None, "hedged", True
        m = re.match(r"^\W*([12])(?!\d)", ans)
        if m:
            return m.group(1), method, False
        return (digits[0], method, False) if digits else (ans, method, False)
    if task in ("T1", "T2"):
        if is_hedge(ans):
            return None, "hedged", True
        ans2, method2 = _truncate_phrase(ans)
        return ans2, (method2 if method == "marker" else method), False
    return ans, method, False
