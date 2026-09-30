"""Checks on the paper skeleton and the study documents.

pdflatex is not available on the build machine, so these tests stand in for a compile
check: every \\input resolves, every \\cite key exists in the bibliography main.tex names,
braces balance in every source file, the mandatory Limitations section exists, TODO occurs
only inside \\placeholder{...}, every \\ref has a \\label, no \\label follows a \\section*
(it would resolve to the preceding numbered section), no cite key occurs in both bibliography
files, the Vietnamese accent macros of the placeholder bibliography render to the intended
strings, the review version carries no identifying information, and the generator scripts
under paper/ run and reproduce the committed tables, figure note and data-statement block.
The study documents are checked for the parts the plan requires, the arm-scope wording is
checked against the harness, and the human-baseline assignment design against its arithmetic.
The last section holds the documents to the design document and to the code (DESIGN_DECISIONS 1,
8.8 item 47, 12.43, 4.6): hypothesis-table identity, DATA_FORMAT vs the scorer/extractor/runner,
every quoted figure recomputed from its file, no build seed anywhere, no superseded statement.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = ROOT / "paper"
DOCS = ROOT / "docs"
MAIN = PAPER / "main.tex"
PY = sys.executable

_CITE = re.compile(r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|citeposs|nocite)\*?"
                   r"(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
_INPUT = re.compile(r"\\(?:input|include)\{([^}]*)\}")
_BIBFILES = re.compile(r"\\bibliography\{([^}]*)\}")
_BIBKEY = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
_REF = re.compile(r"\\(?:ref|autoref|pageref|eqref)\{([^}]*)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")
# a \label right after \section*{...}: LaTeX's starred section does not step the counter, so the
# label resolves to the preceding numbered section (the defect that sent "Section~\ref{sec:ethics}"
# to the Discussion)
_STARRED_LABEL = re.compile(r"\\section\*\{[^}]*\}\s*\\label\{([^}]*)\}")


# ------------------------------------------------------------------ helpers
def strip_comments(tex: str) -> str:
    out = []
    for line in tex.split("\n"):
        i = 0
        while True:
            j = line.find("%", i)
            if j < 0:
                break
            if j > 0 and line[j - 1] == "\\":
                i = j + 1
                continue
            line = line[:j]
            break
        out.append(line)
    return "\n".join(out)


def resolve_input(name: str) -> Path:
    p = PAPER / name
    if p.suffix == "":
        p = p.with_suffix(".tex")
    return p


def tex_sources() -> list[Path]:
    """main.tex and every file reachable from it through \\input / \\include."""
    seen: list[Path] = []
    stack = [MAIN]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.append(f)
        if not f.exists():
            continue
        for name in _INPUT.findall(strip_comments(f.read_text(encoding="utf-8"))):
            stack.append(resolve_input(name))
    return seen


def bib_files() -> list[Path]:
    names = []
    for m in _BIBFILES.findall(strip_comments(MAIN.read_text(encoding="utf-8"))):
        names.extend(n.strip() for n in m.split(",") if n.strip())
    return [PAPER / (n if n.endswith(".bib") else n + ".bib") for n in names]


def remove_placeholders(tex: str) -> str:
    """Delete every \\placeholder{...} span, honouring nested braces."""
    out = []
    i = 0
    key = "\\placeholder{"
    while True:
        j = tex.find(key, i)
        if j < 0:
            out.append(tex[i:])
            break
        out.append(tex[i:j])
        depth = 0
        k = j + len(key) - 1
        while k < len(tex):
            ch = tex[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k + 1
    return "".join(out)


def brace_balance(tex: str) -> int:
    depth = 0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], cwd=ROOT, text=True, capture_output=True, check=True)


# ------------------------------------------------------------------ structure
def test_main_is_acl_review_with_placeholder_macro():
    tex = MAIN.read_text(encoding="utf-8")
    body = strip_comments(tex)
    assert r"\usepackage[review]{acl}" in body
    assert r"\newcommand{\placeholder}[1]" in body
    assert r"\textcolor{red}" in body                      # red in review mode
    assert r"\ifacl@finalcopy" in body and r"\makeatletter" in body
    assert r"\begin{abstract}" in body and r"\placeholder{" in body.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    assert r"\appendix" in body
    assert "Tones Hidden in Tokens" in body


def test_every_input_resolves_to_a_file():
    missing = [f for f in tex_sources() if not f.exists()]
    assert not missing, f"unresolved \\input: {[str(m.relative_to(PAPER)) for m in missing]}"
    names = {f.name for f in tex_sources()}
    for required in ("sec_intro.tex", "sec_background.tex", "sec_benchmark.tex", "sec_setup.tex", "sec_results.tex",
                     "sec_counterfactuals.tex", "sec_tone.tex", "sec_discussion.tex", "sec_limitations.tex",
                     "sec_ethics.tex", "appendix.tex", "table1_counts.tex", "fig1_tokens.tex", "fig1_tokens_note.tex",
                     "rules_placement.tex", "prompt_examples.tex"):
        assert required in names, required


def test_braces_balance_in_every_source_file():
    bad = {}
    for f in tex_sources() + bib_files():
        d = brace_balance(strip_comments(f.read_text(encoding="utf-8")))
        if d != 0:
            bad[str(f.relative_to(PAPER))] = d
    assert not bad, f"unbalanced braces: {bad}"


def test_every_cite_key_exists_in_the_bibliography():
    bibs = bib_files()
    assert bibs, "main.tex names no \\bibliography"
    keys: set[str] = set()
    for b in bibs:
        assert b.exists(), f"missing bibliography file {b.name}"
        keys |= set(_BIBKEY.findall(b.read_text(encoding="utf-8")))
    assert keys
    cited: set[str] = set()
    for f in tex_sources():
        for grp in _CITE.findall(strip_comments(f.read_text(encoding="utf-8"))):
            cited |= {k.strip() for k in grp.split(",") if k.strip()}
    assert cited, "no citations found"
    missing = sorted(cited - keys)
    assert not missing, f"cite keys absent from {[b.name for b in bibs]}: {missing}"


def test_placeholder_bibliography_marks_every_entry_unverified():
    for b in bib_files():
        if "PLACEHOLDER" not in b.name:
            continue
        text = b.read_text(encoding="utf-8")
        entries = text.split("\n@")[1:]
        assert len(entries) >= 20
        for e in entries:
            assert "UNVERIFIED" in e, f"entry without the UNVERIFIED mark: {e[:60]}"
        # BibTeX must stay ASCII: no raw non-ASCII letters in the .bib
        assert text.isascii(), "non-ASCII characters in the .bib (use accent macros)"


def test_limitations_section_exists_and_is_unnumbered():
    found = [f for f in tex_sources() if r"\section*{Limitations}" in strip_comments(f.read_text(encoding="utf-8"))]
    assert found, "no \\section*{Limitations}"
    for f in tex_sources():
        assert r"\section{Limitations}" not in strip_comments(f.read_text(encoding="utf-8")), "Limitations must be unnumbered"


def test_ethics_and_reproducibility_present():
    joined = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())
    assert r"\section*{Ethical Considerations}" in joined
    assert r"\paragraph{Reproducibility.}" in joined
    assert "review board" in joined and "18" in joined


def test_no_todo_outside_placeholder():
    offenders = {}
    for f in tex_sources():
        body = remove_placeholders(strip_comments(f.read_text(encoding="utf-8")))
        hits = [m.start() for m in re.finditer(r"\bTODO\b|\bFIXME\b|\bXXX\b", body)]
        if hits:
            offenders[str(f.relative_to(PAPER))] = [body[max(0, h - 30):h + 30] for h in hits]
    assert not offenders, offenders


def test_remove_placeholders_handles_nested_braces():
    s = r"a \placeholder{\hyp{1}: TODO \textbf{x}} b \placeholder{TODO} c"
    assert remove_placeholders(s) == "a  b  c"


def test_every_ref_has_a_label():
    refs: set[str] = set()
    labels: set[str] = set()
    for f in tex_sources():
        body = strip_comments(f.read_text(encoding="utf-8"))
        refs |= set(_REF.findall(body))
        labels |= set(_LABEL.findall(body))
    missing = sorted(refs - labels)
    assert not missing, f"\\ref without \\label: {missing}"
    dup = [x for x in labels if sum(strip_comments(f.read_text(encoding="utf-8")).count(rf"\label{{{x}}}") for f in tex_sources()) > 1]
    assert not dup, f"duplicate labels: {dup}"


def test_no_label_follows_a_starred_section_and_unnumbered_sections_are_referred_to_by_name():
    offenders = {}
    for f in tex_sources():
        body = strip_comments(f.read_text(encoding="utf-8"))
        hits = _STARRED_LABEL.findall(body)
        if hits:
            offenders[f.name] = hits
    assert not offenders, f"\\label after \\section* resolves to the preceding numbered section: {offenders}"
    joined = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())
    for lab in ("sec:ethics", "sec:limitations"):
        assert rf"\ref{{{lab}}}" not in joined, f"{lab} would point at the wrong section"
    assert "Ethical Considerations section" in joined      # sec_background and sec_benchmark refer to it by name


def test_no_cite_key_occurs_in_two_bibliography_files():
    """The workflow moves verified entries from references_PLACEHOLDER.bib into references.bib;
    a key left in both is BibTeX's 'Repeated entry' error."""
    keys_by_file = {b.name: _BIBKEY.findall(b.read_text(encoding="utf-8")) for b in bib_files()}
    for name, keys in keys_by_file.items():
        dup = [k for k, n in Counter(keys).items() if n > 1]
        assert not dup, f"duplicate keys inside {name}: {dup}"
    names = list(keys_by_file)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sorted(set(keys_by_file[a]) & set(keys_by_file[b]))
            assert not both, f"keys in both {a} and {b}: {both}"


# ------------------------------------------------------------------ bibliography accents
_BIB_NAMED = {"\\uhorn": "\u01b0", "\\ohorn": "\u01a1", "\\UHORN": "\u01af", "\\OHORN": "\u01a0", "\\DJ": "\u0110", "\\dj": "\u0111"}
_BIB_COMB = {"'": "\u0301", "`": "\u0300", "~": "\u0303", "^": "\u0302", '"': "\u0308", "u": "\u0306", "v": "\u030c",
             "h": "\u0309", "d": "\u0323", "=": "\u0304", ".": "\u0307", "c": "\u0327", "H": "\u030b", "r": "\u030a"}
_BIB_ACCENT = re.compile(r"\\([\'`~^\"uvhdc=.Hr])\s*(?:\{([^{}])\}|([A-Za-z\u00c0-\u024f\u1e00-\u1eff]))")


def render_bib_accents(s: str) -> str:
    """Render BibTeX accent macros (T5/vntex conventions, stacked forms such as {\\'\\^e} and
    {\\d{\\^e}} included) to NFC text; braces are dropped at the end."""
    for macro, ch in _BIB_NAMED.items():
        s = re.sub(re.escape(macro) + r"(?![A-Za-z])", ch, s)
    prev = None
    while prev != s:
        prev = s
        s = _BIB_ACCENT.sub(lambda m: unicodedata.normalize("NFC", (m.group(2) or m.group(3)) + _BIB_COMB[m.group(1)]), s)
    return unicodedata.normalize("NFC", s.replace("{", "").replace("}", ""))


# the intended Vietnamese strings of the placeholder bibliography (a checked mapping, typed in NFC)
EXPECTED_BIB_VI = {
    "le-ho-1990-thu-choi-chu": {"title": "Thú chơi chữ", "author": "Lê, Trung Hoa and Hồ, Lê",
                                "publisher": "Nhà xuất bản Trẻ"},
    "doan-1977-nguam": {"author": "Đoàn, Thiện Thuật", "title": "Ngữ âm tiếng Việt",
                        "publisher": "Nhà xuất bản Đại học và Trung học chuyên nghiệp", "address": "Hà Nội"},
    "nguyen-1997-vietnamese": {"author": "Nguyễn, Đình-Hoà", "title": "Vietnamese: Tiếng Việt không son phấn"},
    "nguyen-van-hiep-noi-lai": {"author": "Nguyễn, Văn Hiệp", "title": "Nói lái trong ngôn ngữ và văn học Việt Nam"},
    "hunspell-vi": {"author": "Hồ, Ngọc Đức and Németh, László"},
    "vlstudies-2016-noilai": {"title": "Nói lái: Vietnamese spoonerism"},
}


def bib_entry_fields(text: str, key: str) -> dict[str, str]:
    m = re.search(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,(.*?)\n\}", text, re.DOTALL)
    assert m, f"entry {key} not found"
    fields = {}
    for ln in m.group(1).splitlines():
        fm = re.match(r"\s*(\w+)\s*=\s*\{(.*)\},?\s*$", ln)
        if fm:
            fields[fm.group(1).lower()] = fm.group(2)
    return fields


def test_render_bib_accents_handles_stacked_vietnamese_forms():
    assert render_bib_accents(r"{\'\^e}{\d{\^e}}{\~\uhorn}{\`\^o}{\h{a}}{\u{a}}{\DJ}{\'\^a}") == "ếệữồảăĐấ"
    assert render_bib_accents(r"Glava{\v{s}} Vuli{\'c} Kram{\'a}r") == "Glavaš Vulić Kramár"
    assert render_bib_accents(r"ch{\~u}") == "chũ"                      # the defect: a tilde on a plain u is not chữ


def test_placeholder_bibliography_vietnamese_fields_render_to_the_intended_strings():
    bibs = [b for b in bib_files() if "PLACEHOLDER" in b.name]
    assert bibs
    text = bibs[0].read_text(encoding="utf-8")
    bad = {}
    for key, expected in EXPECTED_BIB_VI.items():
        fields = bib_entry_fields(text, key)
        for fld, want in expected.items():
            got = render_bib_accents(fields[fld])
            if got != unicodedata.normalize("NFC", want):
                bad[(key, fld)] = (got, want)
            assert "\\" not in got, f"unrendered macro in {key}.{fld}: {got}"
    assert not bad, f"accent macros render to the wrong Vietnamese: {bad}"


def test_hypotheses_h1_to_h6_are_all_stated():
    joined = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())
    for i in range(1, 7):
        assert rf"\hyp{{{i}}}" in joined, f"H{i} missing from the paper"
    assert r"\hyp{3b}" in joined, "H3b (tone-isolation DiD) missing from the paper"
    assert "Pre-registered" in joined or "pre-registered" in joined
    # design-document wording that must not regress to the founding plan's
    assert "Northern style" not in joined                       # variant codes are region-neutral
    # DESIGN_DECISIONS 12.7: mediation is gone. Checked per sentence, so that a new "X% of the effect is mediated
    # by token count" cannot hide behind a "not estimate" elsewhere in the paper (the earlier global disjunction
    # was vacuous on the current sources: \label{fig:mediation} and "not estimate" both occurred).
    body = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", remove_placeholders(joined))
    allowed = ("not estimate", "do not estimate", "not a mediator", "rather than a mediator", "resisted calling it mediation")
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if re.search(r"\bmediat", sentence, re.IGNORECASE):                 # "intermediate" is not mediation
            assert any(a in sentence for a in allowed), f"mediation language outside the disclaimer: {sentence.strip()[:200]}"
    assert r"\label{fig:mediation}" not in joined and r"\ref{fig:mediation}" not in joined


def test_review_version_has_no_identifying_information():
    body = strip_comments(MAIN.read_text(encoding="utf-8"))
    assert r"\author{Anonymous ACL submission}" in body
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", body), "e-mail address in main.tex"
    joined = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())
    assert "github.com/" not in joined.lower(), "a repository URL would de-anonymize the submission"


def test_page_budget_comments_present():
    tex = MAIN.read_text(encoding="utf-8")
    for sec in ("Introduction", "Background", "Setup", "Results", "Orthographic counterfactuals", "Where tone lives",
                "Discussion"):
        assert sec in tex, f"page-budget comment for {sec} missing"


# ------------------------------------------------------------------ generators
def test_gen_rule_tables_runs_and_emits_69_placement_pairs(tmp_path):
    out = tmp_path / "tables"
    r = run("paper/gen_rule_tables.py", "--out", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["n_placement_pairs"] == 69
    for name in ("rules_components.tex", "rules_onsets.tex", "rules_spelling.tex", "rules_placement.tex", "rules_variants.tex"):
        assert (out / name).exists(), name
        assert brace_balance(strip_comments((out / name).read_text(encoding="utf-8"))) == 0
    placement = (out / "rules_placement.tex").read_text(encoding="utf-8")
    assert "% 69 pairs" in placement
    rows = [ln for ln in placement.splitlines() if ln.endswith(r"\\") and "&" in ln and "New" not in ln]
    pairs = sum(1 for ln in rows for cell in ln.rstrip("\\").split("&") if cell.strip()) // 2
    assert pairs == 69
    variants = (out / "rules_variants.tex").read_text(encoding="utf-8")
    for ex in ("mèo cái", "mài kéo", "đầu tiên", "tiền đâu", "bí mật", "bị mất", "bật mí"):
        assert ex in variants
    if "V5" in getattr(__import__("noilai.gen.variants", fromlist=["ALL_VARIANTS"]), "ALL_VARIANTS", ()):
        assert "giải pháp" in variants and "phải giáp" in variants and r"\var{6}" in variants
    # the committed tables are what the script produces (modulo the timestamp line)
    for name in ("rules_components.tex", "rules_onsets.tex", "rules_spelling.tex", "rules_placement.tex", "rules_variants.tex"):
        a = [ln for ln in (out / name).read_text(encoding="utf-8").splitlines() if not ln.startswith("% GENERATED")]
        b = [ln for ln in (PAPER / "tables" / name).read_text(encoding="utf-8").splitlines() if not ln.startswith("% GENERATED")]
        assert a == b, f"committed {name} differs from a fresh generation; re-run paper/gen_rule_tables.py"
    spelling = (out / "rules_spelling.tex").read_text(encoding="utf-8")
    assert "khuya" in spelling and "quya" not in spelling
    # the released i/y form comes from lexicon.emit, shown next to the speller's default
    from noilai.vi import lexicon as L
    from noilai.vi.syllable import spell, try_parse
    sylls = [try_parse(w, strict=False).syllable for w in ("lí", "kĩ", "mĩ", "sĩ", "tỉ")]
    emitted = ", ".join(L.emit(x) for x in sylls)
    assert f"& {emitted} (speller: {', '.join(spell(x) for x in sylls)})" in spelling
    assert "i/y emission (release)" in spelling and "lexicon.emit" in spelling
    assert "noilai.vi.lexicon.emit" in spelling.split(r"\caption")[1]      # the caption names the emission path
    # onset examples are attested syllables, not engine-legal non-words
    onsets = (out / "rules_onsets.tex").read_text(encoding="utf-8")
    inv = L.load_inventory()
    for ln in onsets.splitlines():
        if not (ln.startswith(r"\texttt{") and ln.endswith(r"\\") and ln.count("&") == 2) or "(" in ln:
            continue                                                  # header, rules, or a row with a rule note
        for ex in ln.rsplit("&", 1)[1].rstrip("\\").split(","):
            ex = ex.strip()
            assert inv.is_attested(try_parse(ex, strict=False).syllable), f"unattested onset example {ex!r}"
    for non_word in ("pồng", "khồng", "dồng"):                      # engine-legal, unattested: the old fixed template
        assert non_word not in onsets


def test_gen_table1_marks_smoke_builds(tmp_path):
    manifest = {
        "n_items": 24, "n_base_pairs": 10, "seed": 7, "git_commit": "abc123", "git_dirty": False,
        "counts": {f"{t}-{v}-{s}": 1 for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4") for s in ("dev", "test")},
        "core_counts": {f"{t}-{v}": 1 for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4")},
        "cell_pool_sizes": {f"{t}-{v}": 5 for t in ("T1", "T2") for v in ("V1", "V2", "V3", "V4")},
        "generator_args": {"n_lexicon": 3, "n_pseudo": 2, "per_cell_t1": 1, "per_cell_t2": 1, "per_cell_t3": 1,
                           "core_per_cell": 1, "dev_frac": 0.2},
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "table1_counts.tex"
    r = run("paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out))
    assert json.loads(r.stdout.strip().splitlines()[-1])["smoke"] is True
    tex = out.read_text(encoding="utf-8")
    assert "% smoke build; regenerate" in tex
    assert r"\placeholder{smoke build; regenerate" in tex
    assert r"\label{tab:counts}" in tex and "Generated & & 12 & 12 & 12 & 24" in tex
    assert brace_balance(strip_comments(tex)) == 0
    # a full-size manifest is not marked
    full = dict(manifest, n_items=10000, generator_args={"n_lexicon": 1500, "n_pseudo": 1000, "per_cell_t1": 1000,
                                                         "per_cell_t2": 500, "per_cell_t3": 500, "core_per_cell": 125})
    mpath.write_text(json.dumps(full), encoding="utf-8")
    r = run("paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out))
    assert json.loads(r.stdout.strip().splitlines()[-1])["smoke"] is False
    assert "smoke build; regenerate" not in out.read_text(encoding="utf-8")


def test_gen_table1_release_flag_and_note(tmp_path):
    """A full-size build whose filters left a cell short is smoke by the size heuristic unless
    --release declares it the frozen release; the header records which rule decided; --note
    marks the header and the caption alike; --smoke and --release contradict each other."""
    manifest = {
        "n_items": 9950, "n_base_pairs": 2500, "seed": 7, "git_commit": "abc123", "git_dirty": False,
        "counts": {f"{t}-{v}-{s}": 1 for t in ("T1", "T2", "T3") for v in ("V1", "V2", "V3", "V4") for s in ("dev", "test")},
        "core_counts": {}, "cell_pool_sizes": {},
        "generator_args": {"n_lexicon": 1500, "n_pseudo": 1000, "per_cell_t1": 1000, "per_cell_t2": 500, "per_cell_t3": 500,
                           "core_per_cell": 125, "dev_frac": 0.2},
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "t.tex"
    r = run("paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["smoke"] is True and "< 10,000" in info["reason"]
    assert "% smoke decision: " in out.read_text(encoding="utf-8")
    r = run("paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out), "--release", "--note", "v9 counts; superseded")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["smoke"] is False and "--release" in info["reason"]
    tex = out.read_text(encoding="utf-8")
    assert "% smoke decision: generator arguments match the plan; declared the frozen release by --release" in tex
    assert "smoke build; regenerate" not in tex
    assert "% note: v9 counts; superseded" in tex and r"\placeholder{v9 counts; superseded}" in tex
    assert brace_balance(strip_comments(tex)) == 0
    bad = subprocess.run([PY, "paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out), "--smoke", "--release"],
                         cwd=ROOT, text=True, capture_output=True, check=False)
    assert bad.returncode != 0 and "contradict" in bad.stderr
    bad = subprocess.run([PY, "paper/gen_table1.py", "--manifest", str(mpath), "--out", str(out), "--note", "a{b}"],
                         cwd=ROOT, text=True, capture_output=True, check=False)
    assert bad.returncode != 0


def test_committed_table1_is_consistently_marked():
    tex = (PAPER / "tables" / "table1_counts.tex").read_text(encoding="utf-8")
    assert "% GENERATED by paper/gen_table1.py" in tex
    assert "% smoke decision: " in tex, "the header must record which rule decided the smoke marking"
    header_smoke = "% smoke build; regenerate" in tex
    caption_smoke = r"\placeholder{smoke build; regenerate" in tex
    assert header_smoke == caption_smoke, "smoke marking must appear in both the header comment and the caption"
    note = re.search(r"^% note: (.*)$", tex, re.MULTILINE)
    if note:
        assert rf"\placeholder{{{note.group(1)}}}" in tex, "a --note must appear in the caption as well as the header"
    # the committed table is built from a manifest inside the repository, never from a scratch path
    src = re.search(r"^% source manifest: (.*)$", tex, re.MULTILINE).group(1)
    assert not Path(src).is_absolute() and (ROOT / src).exists(), src


def test_gen_fig1_reproduces_committed_token_boundaries(tmp_path):
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    out = tmp_path / "fig1_tokens.tex"
    r = run("paper/gen_fig1_tokens.py", "--spm", str(spm), "--out", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["input"] == "mèo cái" and info["output"] == "mài kéo"
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    nfc_out = side["rows"]["output"]["nfc"]
    assert nfc_out[0]["onset_rime_split"] is True         # ▁m | ài
    assert side["normalizer_info"]["normalizer"] == "identity"
    fresh = [ln for ln in out.read_text(encoding="utf-8").splitlines() if not ln.startswith("%")]
    committed = [ln for ln in (PAPER / "figures" / "fig1_tokens.tex").read_text(encoding="utf-8").splitlines() if not ln.startswith("%")]
    assert fresh == committed, "committed fig1_tokens.tex differs from a fresh generation; re-run paper/gen_fig1_tokens.py"
    assert r"\tok{\tokspace{}m}\tok{ài}" in "\n".join(committed)
    assert brace_balance("\n".join(committed)) == 0
    # the caption's data sentences are generated from the same rows and cannot contradict them
    note_fresh = [ln for ln in (tmp_path / "fig1_tokens_note.tex").read_text(encoding="utf-8").splitlines() if not ln.startswith("%")]
    note_committed_path = PAPER / "figures" / "fig1_tokens_note.tex"
    note_committed = [ln for ln in note_committed_path.read_text(encoding="utf-8").splitlines() if not ln.startswith("%")]
    assert note_fresh == note_committed, "committed fig1_tokens_note.tex differs from a fresh generation"
    note = side["note"]
    assert note in "\n".join(note_committed) and r"\newcommand{\figtokensnote}" in "\n".join(note_committed)
    nfd_rows = side["rows"]["input"]["nfd"] + side["rows"]["output"]["nfd"]
    for r in nfd_rows:
        assert rf"\vi{{{r['syllable']}}}" in note
    for al in {r["boundary_alignment"] for r in nfd_rows}:
        assert f"to {al:.2f} for" in note
    assert any(r["tone_isolated"] for r in nfd_rows) == ("becomes a token of its own" in note)
    intro = strip_comments((PAPER / "sec_intro.tex").read_text(encoding="utf-8"))
    assert r"\input{figures/fig1_tokens_note}" in intro
    caption = intro.split(r"\caption{", 1)[1]
    assert r"\figtokensnote{}" in caption
    assert "no internal boundary is linguistic" not in intro          # the contradicted claim


def test_gen_prompt_appendix_renders_three_tasks(tmp_path):
    out = tmp_path / "prompt_examples.tex"
    r = run("paper/gen_prompt_appendix.py", "--out", str(out))
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert [t for t, _ in info["examples"]] == ["T1", "T2", "T3"]
    side = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    for ex in side["examples"]:
        assert "Đáp án:" in ex["content"] and len(ex["prompt_hash"]) == 64
    tex = out.read_text(encoding="utf-8")
    assert tex.count(r"\begin{quote}") == 3 and brace_balance(strip_comments(tex)) == 0
    assert "mèo cái" in tex and "mài céo" in tex
    # the committed appendix was rendered from the same prompt files
    committed = json.loads((PAPER / "tables" / "prompt_examples.json").read_text(encoding="utf-8"))
    assert committed["prompt_file_sha256"] == side["prompt_file_sha256"], \
        "prompts changed since the appendix was rendered; re-run paper/gen_prompt_appendix.py"


# ------------------------------------------------------------------ study documents
def _doc(name: str) -> str:
    p = DOCS / name
    assert p.exists(), f"missing {p}"
    return p.read_text(encoding="utf-8")


def test_preregistration_has_the_required_parts():
    t = _doc("PREREGISTRATION.md")
    assert "2026-09-30" in t
    for i in range(1, 7):
        assert f"**H{i}**" in t, f"H{i} missing"
    assert "**H3b**" in t
    for phrase in ("PREREG_COMMIT", "Gate 1", "Holm", "McNemar", "base pair", "random-effects", "Mundlak",
                   "Exclusion rules", "Stopping rules", "freeze", "Primary", "Secondary", "DEVIATIONS.md",
                   "not** estimated", "go/no-go", "Amendments", "census", "pass"):
        assert phrase in t, phrase
    # the Gate 1 rule is numeric (DESIGN_DECISIONS 8.5) and the E3 scope rule is stated with its test
    assert "≥ 15 points" in t and "≥ 5 points" in t and "mid-p < 0.05" in t
    # DESIGN_DECISIONS 8.8: two stages, nothing registered yet
    for phrase in ("Stage 1", "Stage 2", "PREREG_COMMIT_STAGE1", "PREREG_COMMIT_STAGE2", "before any test-split run", "v0.3"):
        assert phrase in t, phrase
    assert "Date registered:" not in t and "built at the registration commit" not in t
    # statements the design document reversed (DD 8.1-8.4, 9.1, 9.3, 4.5, 5.7) must not be registered
    for bad in ("digits 1–6", "two-way (base pair × model) cluster-robust SEs", "= 6 tests",
                "decodable layer for tone is earlier", "Never cluster by base pair × model", "corpus independent of the tokenizers",
                "standardized \\|β(align_w)\\| > \\|β(tps_w)\\|. |", "paired mid-p McNemar on aggregated outcomes",
                "every test T1 item with `c2_affected = true`", "analysed on the 43 affected items only", "Steering is exploratory",
                "bí mà → bì má", "5 control-label seeds", "DerSimonian–Laird or REML on the per-model paired effects"):
        assert bad not in t, bad


def test_data_statement_has_bender_friedman_fields():
    t = _doc("DATA_STATEMENT.md")
    for field in ("Curation rationale", "Language variety", "Speaker demographic", "Annotator demographic",
                  "Speech situation", "Text characteristics", "Recording quality", "Provenance"):
        assert field in t, field
    assert "vi-VN" in t and "CC BY 4.0" in t and "CC BY-NC-ND 4.0" in t


def test_consent_form_is_bilingual_adults_only_and_flags_the_parental_variant():
    t = _doc("CONSENT_FORM.md")
    assert "18" in t and "Tiếng Việt" in t and "English" in t
    for vi in ("tự nguyện", "rút", "không được trả tiền", "đồng ý"):
        assert vi in t.lower() or vi in t, vi
    for en in ("voluntary", "withdraw", "unpaid", "consent"):
        assert en in t.lower(), en
    assert "[EMAIL]" in t and "[NAME]" in t                # contact placeholders
    assert "NOT USED" in t and "arental" in t              # parental-permission variant flagged
    assert "NATIVE-CHECK" in t
    assert "18–29" in t and "30–49" in t                    # DESIGN_DECISIONS 10.1 age bands
    assert "30 nói lái" in t                                # the 30-item baseline form


def test_validator_instructions_cover_the_four_judgments():
    t = _doc("VALIDATOR_INSTRUCTIONS.md")
    for col in ("`correct`", "`spelling`", "`lexical`", "`offensive`"):
        assert col in t, col
    for vi in ("chính tả", "thô tục", "từ", "Không chắc"):
        assert vi in t, vi
    assert "Krippendorff" in t and "AC1" in t and "disagree" in t.lower() and "NATIVE-CHECK" in t
    assert "mài céo" in t and "mài kéo" in t
    assert re.search(r"6[–-]8 (giờ|hours)", t)
    assert "V5" in t and "V6" in t                      # the six-type taxonomy is explained to validators


def test_human_baseline_form_design():
    t = _doc("HUMAN_BASELINE_FORM.md")
    # DESIGN_DECISIONS 10.2: 20 respondents x 30 items, 6 anchors, 240 double-judged, 246 distinct
    assert "30 items" in t and "20 minutes" in t and "246" in t and "240" in t and "anchor" in t.lower()
    assert "same p0 prompt" in t and "mean-human" in t.lower()
    assert "score_outputs" in t and "NATIVE-CHECK" in t
    assert "dictionary" in t.lower() and "exclu" in t.lower()


def test_checklist_covers_a1_to_e2_with_honest_no_on_d4():
    t = _doc("CHECKLIST_DRAFT.md")
    for item in ("A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4",
                 "D1", "D2", "D3", "D4", "D5", "E1", "E2"):
        assert re.search(rf"\|\s*{item}\s*\|", t), f"checklist item {item} missing"
    d4 = next(ln for ln in t.splitlines() if re.match(r"\|\s*D4\s*\|", ln))
    assert "**No**" in d4
    assert "UNCERTAIN: verify" in t


def test_ai_use_log_first_entry():
    t = _doc("AI_USE_LOG.md")
    assert "Claude Code" in t and "E1" in t and "Acknowledgements" in t and "2026-09-30" in t
    assert "rule engine" in t


# ------------------------------------------------------------------ claims that must stay protocol
def test_procedures_not_yet_run_are_not_written_as_results():
    """Outside \\placeholder{...}, nothing describes an unrun procedure as done."""
    joined = "\n".join(remove_placeholders(strip_comments(f.read_text(encoding="utf-8"))) for f in tex_sources())
    for phrase in ("The pilot measured", "measured design effect", "too readily", "before any model was run",
                   "took part", "were recruited", "free tiers were used", "were pre-registered in a dated commit before any model",
                   # DESIGN_DECISIONS 8.8 (nothing is registered yet; stage 2 follows the co-author's edits), 7.1, 11.2
                   "were pre-registered in a dated commit", "were fixed in a dated commit", "was frozen at Gate",
                   "on 30 September 2026 (commit", "was sent to hosted APIs", "sent only the 1,500-item core",
                   "were queried through free tiers", "are now potentially in future training data",
                   "treated as potentially contaminated"):
        assert phrase not in joined, phrase
    ethics = strip_comments((PAPER / "sec_ethics.tex").read_text(encoding="utf-8")).rstrip()
    assert ethics.endswith("}") and r"\placeholder{Re-read every paragraph above" in ethics.rsplit(r"\paragraph", 1)[1]
    limitations = strip_comments((PAPER / "sec_limitations.tex").read_text(encoding="utf-8"))
    assert r"\placeholder{Re-read every paragraph above" in limitations
    # DESIGN_DECISIONS 8.8: the pilot runs on development items, so the registration precedes the test-split runs
    assert "before any test-split run" in joined


# ------------------------------------------------------------------ arm scope: paper == pre-registration == harness
def _example_t1_item() -> dict:
    from noilai.gen import variants as V
    from noilai.gen.generate import syl_dict
    from noilai.vi.syllable import spell, try_parse
    a, b = (try_parse(w).syllable for w in ("mèo", "cái"))
    x, y = V.apply("V1", a, b)
    return {"item_id": "example-T1", "task": "T1", "variant": "V1", "input": f"{spell(a)} {spell(b)}",
            "input_syllables": [syl_dict(a), syl_dict(b)], "gold": [f"{spell(x)} {spell(y)}"]}


def test_arm_scope_default_is_whole_prompt_in_the_harness_the_paper_and_the_preregistration():
    from noilai import constants
    from noilai.eval import prompts as P
    from noilai.vi import unicode as U
    assert P.DEFAULT_ARM_SCOPE == constants.ARM_SCOPE_PRIMARY == "whole_prompt"
    item = _example_t1_item()
    instruction = "Nói lái là một cách chơi chữ"                     # from prompts/noilai.yaml concepts.vi.definition
    assert U.nfc(instruction) != U.nfd(instruction)
    whole = P.render(item, arm="nfd")[0]["content"]                  # default scope
    assert U.nfd(instruction) in whole and U.nfc(instruction) not in whole, "default scope must re-encode the instruction"
    assert U.nfd(item["input"]) in whole
    item_only = P.render(item, arm="nfd", arm_scope="item")[0]["content"]
    assert U.nfc(instruction) in item_only, "item scope must leave the instruction in NFC"
    assert U.nfd(item["input"]) in item_only and U.nfc(item["input"]) not in item_only
    assert P.prompt_id(arm_scope="item").endswith("-item") and not P.prompt_id().endswith("-item")
    stripped = P.render(item, arm="strip_tones")[0]["content"]        # strip arms: item-only under both scopes
    assert U.nfc(instruction) in stripped and U.strip_tones(item["input"]) in stripped
    # the documents say what the code does
    setup = strip_comments((PAPER / "sec_setup.tex").read_text(encoding="utf-8"))
    assert "applied to the whole prompt" in setup and "item text alone is a secondary condition" in setup
    assert "Arms apply to the item text by default" not in setup
    prereg = _doc("PREREGISTRATION.md")
    sec85 = prereg.split("### 8.5")[1].split("### 8.6")[0]
    assert "`arm_scope: whole_prompt`" in sec85 and "primary" in sec85
    assert "`arm_scope: item`" in sec85 and "secondary" in sec85
    assert "Arms are applied to the item text by default" not in prereg


# ------------------------------------------------------------------ human baseline: the assignment arithmetic
def test_human_baseline_assignment_is_a_connected_double_coverage_design():
    """240 items dealt twice to 20 respondents, the second copy with the block offset
    1 + (k // 20) % 19: nobody sees an item twice, everyone judges 24 distinct items, and every
    one of the 190 rater pairs shares one or two items (140 and 50), so the rater graph is
    complete; a fixed offset would give 10 disjoint dyads."""
    from noilai import constants as C
    n_items, n_resp = C.HUMAN_BASELINE_DOUBLE_CODED, C.HUMAN_BASELINE_PEOPLE
    per_person = C.HUMAN_BASELINE_ITEMS_PER_PERSON - C.HUMAN_BASELINE_ANCHORS
    assert (n_items, n_resp, per_person) == (240, 20, 24)
    forms: dict[int, set[int]] = defaultdict(set)
    for k in range(n_items):
        first = k % n_resp
        second = (k % n_resp + 1 + (k // n_resp) % (n_resp - 1)) % n_resp
        assert first != second
        forms[first].add(k)
        forms[second].add(k)
    assert all(len(forms[r]) == per_person for r in range(n_resp))
    assert Counter(v for k in range(n_items) for v in [sum(k in forms[r] for r in range(n_resp))]) == {2: n_items}
    shared = Counter(len(forms[i] & forms[j]) for i in range(n_resp) for j in range(i + 1, n_resp))
    assert shared == {1: 140, 2: 50}, shared
    assert sum(shared.values()) == n_resp * (n_resp - 1) // 2 and 0 not in shared
    # the fixed offset of the earlier draft is the disconnected design the doc now warns against
    dyads: dict[int, set[int]] = defaultdict(set)
    for k in range(n_items):
        dyads[k % n_resp].add(k)
        dyads[(k + 10) % n_resp].add(k)
    shared_fixed = Counter(len(dyads[i] & dyads[j]) for i in range(n_resp) for j in range(i + 1, n_resp))
    assert shared_fixed == {0: 180, 24: 10}
    t = _doc("HUMAN_BASELINE_FORM.md")
    assert "offset = 1 + (k // N_RESP) % (N_RESP - 1)" in t and "`offset = 1 + (k // 20) % 19`" in t
    assert "shared_items_per_rater_pair" in t and "{1: 140, 2: 50}" in t
    assert "data/release/v0.2/noilai_main.jsonl" in t and "noilai_core.jsonl" not in t.split("```bash")[1].split("```")[0]
    assert "offset of 10" not in t and "Assignment: a balanced incomplete block" not in t
    assert "140 pairs one, 50 pairs two" in _doc("PREREGISTRATION.md")


def test_human_baseline_snippet_runs_on_the_release(tmp_path):
    """The snippet in the doc, executed against the committed main sample, writes 20 forms of 30
    items with the documented coverage (skipped when the release files are not on disk)."""
    items = ROOT / "data" / "release" / "v0.2" / "noilai_main.jsonl"
    if not items.exists():
        pytest.skip("data/release/v0.2/noilai_main.jsonl not built")
    doc = _doc("HUMAN_BASELINE_FORM.md")
    m = re.search(r"```bash\n\S+python - <<'EOF'\n(.*?)\nEOF\n```", doc, re.DOTALL)
    assert m, "the doc's python snippet is not where the test expects it"
    code = m.group(1).replace('Path("data/human")', f"Path({str(tmp_path / 'human')!r})")
    assert str(tmp_path) in code
    r = subprocess.run([PY, "-"], input=code, cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr
    summary = r.stdout.strip().splitlines()[-1]
    assert "'forms': 20" in summary and "'items': 246" in summary and "'anchors_seen_by': 20" in summary
    assert "'others': [2]" in summary and "'shared_items_per_rater_pair': {1: 140, 2: 50}" in summary
    forms = sorted((tmp_path / "human").glob("baseline_form_*.csv"))
    assert len(forms) == 20
    assert all(len(f.read_text(encoding="utf-8").splitlines()) == 31 for f in forms)
    ids = json.loads((tmp_path / "human" / "human_items.json").read_text(encoding="utf-8"))
    assert len(ids) == 246


# ------------------------------------------------------------------ the worked patching pair
def test_worked_patching_pair_in_the_paper_passes_the_alignment_filter():
    spm = ROOT / "data" / "external" / "gemma3_tokenizer.model"
    if not spm.exists():
        pytest.skip("no Gemma 3 tokenizer downloaded")
    from noilai.audit.tokenizers import SentencePieceAdapter
    from noilai.gen import variants as V
    from noilai.probe.pairs import PATCH_PROMPT
    from noilai.vi import unicode as U
    from noilai.vi.syllable import spell, try_parse
    tone = strip_comments((PAPER / "sec_tone.tex").read_text(encoding="utf-8"))
    clean, corrupt, partner = "tử", "tự", "công"
    for s in (f"{partner} {clean}", f"{partner} {corrupt}"):
        assert rf"\vi{{{s}}}" in tone
    outs = {}
    for t in (clean, corrupt):
        o = V.apply("V3", try_parse(partner).syllable, try_parse(t).syllable)
        outs[t] = f"{spell(o[0])} {spell(o[1])}"
        assert rf"\vi{{{outs[t]}}}" in tone
    assert outs == {"tử": "cổng tư", "tự": "cộng tư"}
    ad = SentencePieceAdapter(spm, name="gemma3")
    def pieces(text: str) -> list[str]:
        return [x.text for x in ad.encode(text)]

    def ids(text: str) -> list[int]:
        return [x.id for x in ad.encode(text)]

    pc, pk = PATCH_PROMPT.format(a=partner, b=clean), PATCH_PROMPT.format(a=partner, b=corrupt)
    assert len(ids(pc)) == len(ids(pk)), "the two prompts must tokenize to the same length"
    diff = [i for i, (x, y) in enumerate(zip(ids(pc), ids(pk))) if x != y]
    assert len(diff) == 1 and pieces(pc)[diff[0]] == "\u2581" + clean and pieces(pk)[diff[0]] == "\u2581" + corrupt
    for s in (" " + clean, " " + corrupt, " cổng", " cộng"):
        assert len(pieces(s)) == 1, s
    assert U.strip_tones("cổng") == U.strip_tones("cộng")           # readout differs in the tone mark alone
    # the earlier example (bí mà / bí mạ) fails this very filter: ' mạ' is two pieces
    assert len(pieces(" mà")) != len(pieces(" mạ"))
    assert "bí mà" not in tone


# ------------------------------------------------------------------ data statement facts
def test_data_statement_release_facts_are_generated_and_current():
    manifest = ROOT / "data" / "release" / "v0.2" / "manifest.json"
    if not manifest.exists():
        pytest.skip("no v0.2 manifest")
    r = subprocess.run([PY, "paper/gen_data_statement_facts.py", "--manifest", str(manifest), "--check"],
                       cwd=ROOT, text=True, capture_output=True, check=False)
    assert r.returncode == 0, "docs/DATA_STATEMENT.md release facts are stale; re-run paper/gen_data_statement_facts.py"
    t = _doc("DATA_STATEMENT.md")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    block = t.split("<!-- BEGIN GENERATED release facts")[1].split("<!-- END GENERATED release facts -->")[0]
    assert f"`placement_style: {m['placement_style']}`" in block
    assert f"{int(m['n_items']):,} generated items" in block and m["content_sha256"] in block
    attested = manifest.parent / "attested.jsonl"
    if attested.exists():
        rows = [json.loads(ln) for ln in attested.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert f"Attested seed ({len(rows)} rows" in block
        assert f"{sum(1 for x in rows if x.get('exact'))} exact reproductions" in block
    for stale in ("22-row", "v0.2 pending", "16 exact reproductions", "mộng mơ → mờ mông"):
        assert stale not in t, stale
    assert "generated block of §0" in t


# ------------------------------------------------------------------ the design document's wording holds in the paper
def _tex_joined() -> str:
    return "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())


def test_superseded_design_wording_is_absent_from_the_paper():
    """Statements the design document replaced (DESIGN_DECISIONS 1, 4.2, 4.6, 8.1-8.4, 9.1-9.4, 11.1, 11.2, 11.5,
    section 14) must not survive in any .tex source: the paper says what the design and the code do."""
    joined = _tex_joined()
    for bad in ("standardized difference", "standardized alignment coefficient", "two-way (base pair, model)", "two-way clustered",
                "six intervention effects", "family of six", "base-pair-aggregated outcomes",          # DD 8.1-8.4
                "reads a digit", "decodable at an earlier layer", "Gemma Scope~2 sparse autoencoders are optional",
                "steering flip", "Patching and steering",                                              # DD 9.1, 9.3, 9.4
                "every model output and score file", "shared privately", "treated as potentially contaminated",   # DD 11.1, 11.2
                "weights that equalize",                                                               # DD 4.2
                "canary line", r"resource hashes; the canary GUID \placeholder{printed",                 # DD 4.6 (item 44)
                "with seed, counts", "with the seed, item counts",                                       # DD 4.6 (item 51)
                "on inputs, outputs, readings and candidates", "to inputs, gold answers, readings and candidates",  # DD 11.5
                "10--24 points",                                                                       # DD 14 item 45
                "rests on 43 items", "on the 43 affected XCOPA items with the detectable effect",       # DD 5.7 / 6.3
                "apply to any diacritic-heavy Latin script.", "the parser and re-encoder transfer",     # DD 14 item 35
                "GUID is drawn outside the seeded generator and printed"):
        assert bad not in joined, bad
    # steering is named once, as future work, in the Discussion and nowhere else
    for f in tex_sources():
        body = strip_comments(f.read_text(encoding="utf-8")).lower()
        if f.name != "sec_discussion.tex":
            assert "steering" not in body, f"steering outside the future-work sentence: {f.name}"
    assert "left to future work" in strip_comments((PAPER / "sec_discussion.tex").read_text(encoding="utf-8"))
    # what replaced them
    setup = strip_comments((PAPER / "sec_setup.tex").read_text(encoding="utf-8"))
    assert "fourteen cells" in setup and "nested likelihood-ratio test" in setup and "Hartung--Knapp" in setup
    assert "base-pair cluster bootstrap" in setup and "labelled as ignoring clustering" in setup
    assert r"\texttt{t3\_yesno\_lp}" in setup and r"\texttt{t3\_pair\_lp}" in setup and "length-normalized per token" in setup
    results = strip_comments((PAPER / "sec_results.tex").read_text(encoding="utf-8"))
    assert "nested likelihood-ratio test" in results and "300 misaligned split syllables" in results
    assert "descriptive case study, not a tested hypothesis" in results and "at least 100 native-verified exact" in results
    tone = strip_comments((PAPER / "sec_tone.tex").read_text(encoding="utf-8"))
    assert "reads the tone's name" in tone and "fewer than 50 pairs survive" in tone and "localization study" in tone
    assert "No claim of earlier decodability under NFD" in tone
    cf = strip_comments((PAPER / "sec_counterfactuals.tex").read_text(encoding="utf-8"))
    assert cf.index(r"\hyp{4} (\emph{the rarer convention costs}) leads") < cf.index(r"\hyp{3} (\emph{a crossover})")
    assert "distribution shift" in cf and "per-family table is primary" in cf and "fourteen cells" in cf
    bench = strip_comments((PAPER / "sec_benchmark.tex").read_text(encoding="utf-8"))
    assert "quota-sampled" in bench and "only when the input pair itself is a taboo phrase" in bench
    assert "The build seed is not published" in bench and "three-valued tag" in bench
    ethics = strip_comments((PAPER / "sec_ethics.tex").read_text(encoding="utf-8"))
    assert "released only inside the gated repository" in ethics and "never receives test items" in ethics


def test_canary_guid_is_printed_only_as_its_sha256():
    """DESIGN_DECISIONS 4.6 (item 44): a sentence that says the GUID is printed must say SHA-256 in the same sentence."""
    joined = _tex_joined()
    hits = 0
    for sentence in re.split(r"(?<=[.!?])\s+", joined):
        if "GUID" in sentence and "print" in sentence:
            hits += 1
            assert "SHA-256" in sentence, sentence[:200]
    assert hits >= 2                                          # sec_benchmark and appendix (non-vacuity)


def test_numbers_attributed_to_unchecked_sources_stay_inside_placeholders():
    r"""DESIGN_DECISIONS 14 item 45: no attributed number in prose until checked at page level. The two figures
    docs/RELATED_WORK_VERIFICATION.md lists as unchecked (Bean et al. 16% of 445; Ghosh & Jyothi's drops) may
    appear only inside \placeholder{...}."""
    joined = _tex_joined()
    outside = remove_placeholders(joined)
    assert r"16\%" in joined and "445" in outside                   # the checked count stays, the unchecked share is wrapped
    assert r"16\% of 445" not in outside and r"Only 16\%" not in outside
    assert "10--24" in joined and "10--24" not in outside
    rwv = _doc("RELATED_WORK_VERIFICATION.md")
    assert "16.0% not checked" in rwv and "ghosh-jyothi-2026-noncanonical" in rwv


def test_contribution_five_claims_script_generality_for_the_re_encoder_only():
    intro = strip_comments((PAPER / "sec_intro.tex").read_text(encoding="utf-8"))
    item = next(ln for ln in intro.splitlines() if "Reusable tooling" in ln)
    clauses = re.split(r";", item)
    general = [c for c in clauses if "any diacritic-heavy Latin script" in c]
    assert len(general) == 1 and "parser" not in general[0] and "spelling-rule" not in general[0], item
    assert "parser" in item and "not script-general" in item
    lim = strip_comments((PAPER / "sec_limitations.tex").read_text(encoding="utf-8"))
    assert "whereas the parser, speller and placement arms" in lim


def test_attested_counts_in_the_prose_are_generated_macros():
    """paper/tables/attested_facts.tex is written by gen_data_statement_facts.py from attested.jsonl; the prose
    quotes its macros and never types 22/16 or 34/28 (DESIGN_DECISIONS 4.7, 12.43)."""
    main = strip_comments(MAIN.read_text(encoding="utf-8"))
    assert r"\input{tables/attested_facts}" in main
    facts = (PAPER / "tables" / "attested_facts.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(\d+)\}", facts))
    assert set(macros) == {"nAttested", "nAttestedExact", "nAttestedApprox", "nAttestedHsix", "nAttestedVerified"}
    bench = strip_comments((PAPER / "sec_benchmark.tex").read_text(encoding="utf-8"))
    for m in ("nAttested", "nAttestedExact", "nAttestedApprox", "nAttestedVerified"):
        assert "\\" + m + "{}" in bench, m
    assert r"\placeholder{22} rows" not in bench and r"\placeholder{16} reproduced" not in bench
    attested = ROOT / "data" / "release" / "v0.2" / "attested.jsonl"
    if attested.exists():
        rows = [json.loads(ln) for ln in attested.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert int(macros["nAttested"]) == len(rows) >= 20
        assert int(macros["nAttestedExact"]) == sum(1 for r in rows if r.get("exact"))
        assert int(macros["nAttestedApprox"]) == sum(1 for r in rows if not r.get("rule_matches_attested"))
        assert int(macros["nAttested"]) == int(macros["nAttestedExact"]) + int(macros["nAttestedApprox"])


# ================================================================== the study documents against the design document and the code
# DESIGN_DECISIONS.md is the binding specification and the code is the reference for what the harness does. These tests
# hold the documents to both: the pre-registration's hypothesis table is identical to DESIGN_DECISIONS section 1 (item 47);
# DATA_FORMAT.md names the classes, extraction methods and arm names the code emits; every figure the documents quote from
# a repository file reproduces from that file (DESIGN_DECISIONS 12.43); the build seed appears in no published document
# (4.6); superseded statements the red team retired do not come back. Every check has a non-vacuity guard.
DD = DOCS / "DESIGN_DECISIONS.md"
PREREG = DOCS / "PREREGISTRATION.md"
MANIFEST = ROOT / "data" / "release" / "v0.2" / "manifest.json"
COUNTS = ROOT / "data" / "audit" / "counts.json"
ROWS = ROOT / "data" / "audit" / "gemma3_rows.csv"


def _read(p: Path) -> str:
    assert p.exists(), p
    return p.read_text(encoding="utf-8")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ hypothesis table identity (DD 1, 8.8; item 47)
def _hyp_rows(text: str) -> dict[str, str]:
    rows = {}
    for ln in text.splitlines():
        m = re.match(r"^\|\s*\*\*(H\d[a-z]?)\*\*", ln)
        if m:
            assert m.group(1) not in rows, f"duplicate row {m.group(1)}"
            rows[m.group(1)] = _norm(ln)
    return rows


def test_hypothesis_tables_of_the_design_document_and_the_preregistration_are_identical():
    dd, pr = _hyp_rows(_read(DD)), _hyp_rows(_read(PREREG))
    assert set(dd) == set(pr) == {"H1", "H2", "H3", "H3b", "H4", "H5", "H6"}     # non-vacuity: all seven rows
    diffs = {k: (dd[k], pr[k]) for k in dd if dd[k] != pr[k]}
    assert not diffs, "PREREGISTRATION §2 differs from DESIGN_DECISIONS §1 on: " + "; ".join(
        f"{k}: DD={a[:160]!r} PREREG={b[:160]!r}" for k, (a, b) in diffs.items())
    # the earlier defects (item 47; the paper-docs review) cannot return
    for stale in ("standardized \\|β(align_w)\\| > \\|β(tps_w)\\|. |", "Either inequality fails", "δ_sel = 0.10", "3,000"):
        assert stale not in _read(PREREG).split("## 3.")[0], stale
    assert "tests/test_paper.py" in _read(DD) and "tests/test_paper.py" in _read(PREREG)


# ------------------------------------------------------------------ DATA_FORMAT.md says what the code emits
def test_data_format_names_the_scorers_classes_and_the_extractors_methods():
    from noilai.eval import extract, score
    t = _read(DOCS / "DATA_FORMAT.md")
    scores = t.split("## Scores")[1]
    classes = set(re.findall(r"`([a-z_]+)`", scores.split("`error_class`")[1].split("`extraction_method`")[0]))
    methods = set(re.findall(r"`([a-z_]+)`", scores.split("`extraction_method`")[1].split("`component_errors`")[0]))
    assert classes == set(score.ERROR_CLASSES), (classes ^ set(score.ERROR_CLASSES))
    assert methods == set(extract.METHODS), (methods ^ set(extract.METHODS))
    assert len(classes) >= 12 and len(methods) == 9


def test_data_format_arm_names_are_the_runners_names_and_the_aliases_are_the_preregistered_ones():
    from noilai import constants
    from noilai.eval import prompts as P
    from noilai.vi import reencode as R
    t = _read(DOCS / "DATA_FORMAT.md")
    row = next(ln for ln in t.splitlines() if ln.startswith("| `arm` |"))
    named = re.findall(r"`([a-z0-9_]+)`", row)
    canonical = {a for a in named if a in set(R.ARMS)}
    campaign = {P.normalize_arm(a) for a in constants.ARMS}                 # the pre-registered arms, canonical names
    assert campaign <= canonical <= set(R.ARMS), (campaign - canonical, canonical - set(R.ARMS))
    assert not ({"base", "pc"} & set(R.ARMS)) and {"base", "pc"} <= set(named)
    for alias, target in P.ARM_ALIASES.items():
        assert alias in named and target in named, (alias, target)
    assert "never appear in rows" in row
    pid = next(ln for ln in t.splitlines() if ln.startswith("| `prompt_id` |"))
    assert P.prompt_id() in pid and "-item" in pid and "`p0`, `p1`, `p2`) and shot count" not in pid


def test_data_format_documents_the_three_valued_exactness_tag_and_the_release_uses_it():
    t = _read(DOCS / "DATA_FORMAT.md")
    assert "approx(merger:" in t and "approx(substitution:" in t and "approx(<merger>)" not in t
    attested = ROOT / "data" / "release" / "v0.2" / "attested.jsonl"
    if not attested.exists():
        pytest.skip("no v0.2 attested.jsonl")
    values = {json.loads(ln)["exactness"] for ln in attested.read_text(encoding="utf-8").splitlines() if ln.strip()}
    assert len(values) >= 3 and "exact" in values
    kinds = set()
    for v in values:
        m = re.fullmatch(r"exact|approx\((merger|substitution): .+\)", v)
        assert m, v
        if m.group(1):
            kinds.add(m.group(1))
    assert kinds == {"merger", "substitution"}


# ------------------------------------------------------------------ superseded statements stay out of the documents
def test_superseded_statements_are_absent_from_the_documents():
    absent = {
        "DATA_FORMAT.md": ("preferred after", "never redistributed", "standard new-style spelling", "records seed,"),
        "IMPLEMENTATION_DECISIONS.md": ("never redistributed", "prefers the attested `y` form after h k l m t s", "`xit`, `hoc`)",
                                        "Of the 22 seed rows"),
        "DATA_STATEMENT.md": ("predates the red-team amendments", "dev-build seed is published", "was sent to hosted APIs",
                              "is printed in the paper's appendix. The", "*hoc*", "rejects 32 entries",
                              "applied at generation to inputs, gold"),
        "DESIGN_DECISIONS.md": ("built at commit 975be27", "already matches section 1 (verified", "the file is untracked",
                                "92% longer", "clean `bí mà` → `bì má`", "49% of the time", "α = 0.84",
                                "noilai_main4200", "GUID printed in the appendix (4.6)", "generator commit and seed",
                                "Stage 1 (the v0.2 freeze commit)", "today 0 of 22", "the separator is required",
                                "To implement (absent from the code today)", "lacks `lenient_only`",
                                "rename `hiraoka-okazaki-2025-spelling`"),
        "VALIDATOR_INSTRUCTIONS.md": ("mờ mông",),
        "RISKS.md": ("92% longer",),
        "RESULTS_LOG.md": ("Counts below are from the first v0.2 build and\nare superseded",),
        "DEVIATIONS.md": ("to be frozen in a dated commit before Gate 1",),
        "WEEK1_CHECKLIST.md": ("compute_log.py add",),
        "CHECKLIST_DRAFT.md": ("to inputs, gold, T2 readings and T3 candidates in either order",),
    }
    n = 0
    for name, phrases in absent.items():
        t = _read(DOCS / name)
        for ph in phrases:
            n += 1
            assert ph not in t, f"{name}: {ph!r}"
    assert n >= 30
    for name in ("DATA_FORMAT.md", "IMPLEMENTATION_DECISIONS.md", "DATA_STATEMENT.md"):
        assert "consulted at build time" in _read(DOCS / name) or name == "DATA_FORMAT.md"


def test_build_seed_appears_in_no_document_or_paper_source():
    """DESIGN_DECISIONS 4.6 (item 51): the build seed is not published. The private release manifest is the
    only place it may live (it is not covered here; redacting it is the release scripts' job)."""
    if not MANIFEST.exists():
        pytest.skip("no v0.2 manifest")
    seed = str(json.load(MANIFEST.open(encoding="utf-8"))["seed"])
    assert seed.isdigit() and len(seed) >= 6
    files = sorted(DOCS.glob("*.md")) + sorted(PAPER.glob("*.tex")) + sorted((PAPER / "tables").glob("*.tex")) + [PAPER / "README.md"]
    assert len(files) >= 25
    hits = [str(f.relative_to(ROOT)) for f in files if seed in f.read_text(encoding="utf-8")]
    assert not hits, f"the build seed is printed in {hits}"
    ds = _read(DOCS / "DATA_STATEMENT.md")
    assert "**The build seed is not published**" in ds and "one seeded stream" in ds.split("## 0.")[1].split("## A.")[0]


# ------------------------------------------------------------------ figures reproduce from the named files (DD 12.43)
def _manifest():
    if not MANIFEST.exists():
        pytest.skip("no v0.2 manifest")
    return json.load(MANIFEST.open(encoding="utf-8"))


def test_results_log_v02_entry_matches_the_committed_manifest_and_attested_file():
    m = _manifest()
    log = _norm(_read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-05]")[1].split("## ")[0])
    split, task = defaultdict(int), defaultdict(int)
    for k, n in m["counts"].items():
        t, _v, s = k.split("-")
        split[s] += n
        task[t] += n
    d = m["drops"]
    identity = sum(v for k, v in d.items() if k.endswith(":identity"))
    reversal = sum(v for k, v in d.items() if k.endswith(":plain_reversal"))
    illegal = sum(v for k, v in d.items() if k.endswith(":illegal"))
    expected = [f"{int(m['n_items']):,} items (T1 {task['T1']:,}, T2 {task['T2']:,}, T3 {task['T3']:,})",
                f"dev {split['dev']:,} / test {split['test']:,}", f"core {sum(m['core_counts'].values()):,}",
                f"Vulgar-flagged items {sum(m['vulgar_counts'].values())}", f"C2-affected items {sum(m['c2_affected_counts'].values())}",
                f"taboo phrase {d['base:vulgar_input']} (`base:vulgar_input`)", f"identity {identity:,}, plain reversal {reversal:,}",
                f"illegal {illegal:,}", f"excluded {m['n_marginal_rimes']} (`n_marginal_rimes`)",
                m["content_sha256"], m["git_commit"][:7]]
    for e in expected:
        assert e in log, e
    attested = MANIFEST.parent / "attested.jsonl"
    if attested.exists():
        rows = [json.loads(ln) for ln in attested.read_text(encoding="utf-8").splitlines() if ln.strip()]
        exact = sum(1 for r in rows if r.get("exact"))
        approx = [r["exactness"] for r in rows if not r.get("rule_matches_attested")]
        subst = sum(1 for e in approx if e.startswith("approx(substitution"))
        assert f"{len(rows)} rows, {exact} reproduced exactly, {len(approx)} approximate ({subst} substitution, {len(approx) - subst} merger)" in log
        assert f"{sum(1 for r in rows if r.get('eligible_h6'))} H6-eligible" in log
        assert f"{sum(1 for r in rows if r.get('n_syllables') == 3)} three-syllable" in log
    # the superseded first-build paragraph is labelled and no longer claims the committed manifest as its source
    entry3 = _read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-03]")[1].split("[RL-2026-09-30-04]")[0]
    assert "**Superseded (first v0.2 build, not the committed release).**" in entry3
    assert "Source: `data/release/v0.2/manifest.json`" not in entry3


def _audit_rows():
    if not ROWS.exists():
        pytest.skip("no gemma3_rows.csv")
    with ROWS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _among_split(rows, enc):
    vals = [float(r["boundary_alignment"]) for r in rows if r["encoding"] == enc and r["single_token"] not in ("True", "true", "1")]
    assert len(vals) > 1000
    return float(np.mean(vals)), len(vals)


def test_alignment_among_split_syllables_is_what_the_documents_quote():
    rows = _audit_rows()
    nfc, n_nfc = _among_split(rows, "nfc")
    nfd, n_nfd = _among_split(rows, "nfd")
    c = json.load(COUNTS.open(encoding="utf-8"))["tokenizer_audit"]["gemma3"]["summary"]
    all_nfc, all_nfd = c["nfc"]["boundary_alignment_mean"], c["nfd"]["boundary_alignment_mean"]
    assert nfc < all_nfc and nfd < all_nfd                     # the all-syllable mean counts single tokens as 1.0
    log = _norm(_read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-06]")[1])
    assert f"NFC {nfc:.3f} over {n_nfc:,} split syllables" in log and f"NFD {nfd:.3f} over {n_nfd:,}" in log
    assert f"{all_nfc:.2f} / {all_nfd:.2f}" in log
    setup = (PAPER / "sec_setup.tex").read_text(encoding="utf-8")
    assert f"alignment {nfc:.2f} among split syllables ({all_nfc:.2f} over all syllables" in setup
    assert f"{nfd:.2f} ({all_nfd:.2f})" in setup
    reading = _read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-01]")[1].split("[RL-2026-09-30-02]")[0]
    assert f"(alignment {nfc:.2f} among split syllables; {all_nfc:.2f} over all syllables" in reading


def test_nfd_length_figures_in_the_h3_row_reproduce():
    g = json.load((ROOT / "data" / "audit" / "gemma3.json").open(encoding="utf-8"))
    c = json.load(COUNTS.open(encoding="utf-8"))["tokenizer_audit"]["gemma3"]["summary"]
    longer = g["normalization_census"]["nfd_longer_frac"]
    ratio = c["nfd"]["tokens_per_syllable_mean"] / c["nfc"]["tokens_per_syllable_mean"] - 1
    phrase = (f"longer for {round(longer * 100)}% of words and carries {round(ratio * 100)}% more tokens per syllable, "
              f"{c['nfc']['tokens_per_syllable_mean']:.2f} → {c['nfd']['tokens_per_syllable_mean']:.2f}")
    for name in ("DESIGN_DECISIONS.md", "PREREGISTRATION.md"):
        assert phrase in _read(DOCS / name), (name, phrase)
    assert f"longer for {round(longer * 100)}% of words" in _read(DOCS / "RISKS.md")
    assert f"longer for {round(longer * 100)}\\% of words, carries {round(ratio * 100)}\\% more tokens" in (PAPER / "sec_counterfactuals.tex").read_text(encoding="utf-8")
    log = _norm(_read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-06]")[1])
    assert f"{longer * 100:.1f}% of the 500 census words" in log and f"{ratio * 100:.1f}% above NFC" in log


def test_single_token_share_by_tone_reproduces_from_the_audit_rows():
    from noilai.vi.syllable import try_parse
    names = ["ngang", "huyền", "sắc", "hỏi", "ngã", "nặng"]
    by_tone = defaultdict(list)
    for r in _audit_rows():
        if r["encoding"] != "nfc":
            continue
        p = try_parse(r["syllable"], strict=False)
        if p is None:
            continue
        by_tone[names[p.syllable.tone]].append(r["single_token"] in ("True", "true", "1"))
    assert set(by_tone) == set(names) and all(len(v) > 300 for v in by_tone.values())
    share = {k: float(np.mean(v)) for k, v in by_tone.items()}
    log = _norm(_read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-06]")[1])
    for k, v in share.items():
        assert f"{k} {v * 100:.1f}%" in log, (k, v)
    others = [v for k, v in share.items() if k != "ngang"]
    phrase = f"single tokens {round(share['ngang'] * 100)}% of the time vs {round(min(others) * 100)}–{round(max(others) * 100)}% for the other tones"
    assert phrase in _read(DD), phrase


def test_agreement_illustration_reproduces_with_the_repository_functions():
    """DD 10.1's illustration is simulated, not typed: two coders, 96% prevalence, independent symmetric error."""
    from noilai.stats.agreement import gwet_ac1, krippendorff_alpha_nominal, percent_agreement
    rng = np.random.default_rng(0)
    n = 200_000
    got = {}
    for err in (0.01, 0.02):
        truth = rng.random(n) < 0.96
        a = truth ^ (rng.random(n) < err)
        b = truth ^ (rng.random(n) < err)
        ratings = [(i, "A", int(a[i])) for i in range(n)] + [(i, "B", int(b[i])) for i in range(n)]
        got[err] = (percent_agreement(ratings), krippendorff_alpha_nominal(ratings), gwet_ac1(ratings))
    log = _norm(_read(DOCS / "RESULTS_LOG.md").split("[RL-2026-09-30-06]")[1])
    for err, (raw, alpha, ac1) in got.items():
        assert f"{int(err * 100)}% error each → raw agreement {raw:.3f}, Krippendorff's α {alpha:.3f}, Gwet's AC1 {ac1:.3f}" in log, (err, raw, alpha, ac1)
    dd = _read(DD)
    raw1, a1, ac1 = got[0.01]
    raw2, a2, ac2 = got[0.02]
    assert f"raw agreement {round(raw1 * 100)}% but α ≈ {a1:.2f} and AC1 ≈ {ac1:.2f}" in dd
    assert f"raw {round(raw2 * 100)}%, α ≈ {a2:.2f}, AC1 ≈ {ac2:.2f}" in dd
    assert got[0.02][1] < 0.7 < got[0.01][1] < 0.8                  # the point of the illustration


def test_design_document_rime_list_covers_exactly_the_parsers_rimes():
    """DD 2.3: 162 orthographic spellings = 154 parser rimes + 8 y-doublets, and the 8 qu-only glide rimes
    folded into their glide-less spellings; the parser's 162 rimes are counts.json inventory.rimes."""
    from noilai.vi.syllable import try_parse
    rimes = set(json.load(COUNTS.open(encoding="utf-8"))["inventory"]["rimes"])
    para = next(ln for ln in _read(DD).splitlines() if ln.startswith("**162 distinct toneless orthographic rimes**"))
    m = re.search(r"removed here: `(.*?)`\.", para)
    raw = re.sub(r"\([^)]*\)", "", m.group(1).split("| (w,")[0])          # the (w, ô, c) note is not a spelling
    tokens = [t for t in raw.replace("|", " ").split() if re.fullmatch(r"[a-zăâêôơưđ]+", t)]
    assert len(tokens) == 162

    def key(s):
        return ("w+" if s.glide else "") + s.nucleus + ("+" + s.coda if s.coda else "")
    keys = defaultdict(list)
    for t in tokens:
        p = None
        for onset in ("", "h", "t", "c"):
            p = try_parse(onset + t, strict=False)
            if p:
                break
        assert p, t
        keys[key(p.syllable)].append(t)
    doublets = {k: v for k, v in keys.items() if len(v) > 1}
    assert len(doublets) == 8 and all(any(x.startswith("y") for x in v) for v in doublets.values())
    folded = set(re.findall(r"w\+[a-zăâêôơư]+\+[a-z]+", para))
    assert len(folded) == 8 and folded <= rimes and not (folded & set(keys))
    assert set(keys) | folded == rimes, (sorted(rimes - set(keys) - folded), sorted(set(keys) - rimes))
    assert f"{len(keys)} parser rimes" in para and f"{len(tokens)} spellings" in para


def test_data_statement_names_only_hunspell_entries_the_reconciliation_script_lists():
    c = json.load(COUNTS.open(encoding="utf-8"))["hunspell"]
    listed = set(c["unparsed"]) | set(c["rejected_phonotactics"]) | set(c["nonstandard"])
    ds = _read(DOCS / "DATA_STATEMENT.md")
    sentence = ds.split("The parser leaves")[1].split("all are listed")[0]
    named = set(re.findall(r"\*([^*]+)\*", sentence))
    assert named, "no italic entries named"
    assert named <= listed, named - listed
    assert set(c["unparsed"]) <= named and set(c["rejected_phonotactics"]) <= named
    assert f"{len(c['unparsed'])} distinct entries" in ds and f"({len(c['unparsed']) * 2} lines across the two placement files" in ds
    block = ds.split("<!-- BEGIN GENERATED release facts")[1].split("<!-- END GENERATED")[0]
    m = _manifest()
    assert f"generator commit `{m['git_commit'][:7]}`" in block and ("dirty tree" in block) == bool(m.get("git_dirty"))
    assert f"commit {m['git_commit'][:7]}" in ds.split("## H.")[1]


# ------------------------------------------------------------------ files and CLI flags the documents name exist
def test_design_document_sub_sample_names_are_run_plan_keys_and_the_checklist_names_real_subcommands():
    import yaml
    plan = yaml.safe_load((ROOT / "configs" / "run_plan.yaml").read_text(encoding="utf-8"))
    keys = set(plan["item_files"])
    sec45 = _read(DD).split("### 4.5")[1].split("### 4.6")[0]
    names = set(re.findall(r"`((?:noilai|pilot)_[a-z0-9_]+)\.jsonl`", sec45))
    assert names >= {"noilai_main", "noilai_core", "noilai_c2", "noilai_api_para300", "pilot_t1_200", "pilot_xcopa_200"}
    missing = {n for n in names if n not in keys}
    assert missing == {"noilai_api_dev"} and "not implemented" in sec45.split("noilai_api_dev.jsonl")[1][:200]
    assert "noilai_main4200" not in sec45
    help_text = subprocess.run([sys.executable, "scripts/compute_log.py", "--help"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    subs = set(re.search(r"\{([a-z,]+)\}", help_text).group(1).split(","))
    named = set(re.findall(r"scripts/compute_log\.py (\w+)", _read(DOCS / "WEEK1_CHECKLIST.md")))
    assert named and named <= subs, (named, subs)
