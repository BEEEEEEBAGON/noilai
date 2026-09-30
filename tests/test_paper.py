"""Checks on the paper skeleton and the study documents.

pdflatex is not available on the build machine, so these tests stand in for a compile
check: every \\input resolves, every \\cite key exists in the bibliography main.tex names,
braces balance in every source file, the mandatory Limitations section exists, TODO occurs
only inside \\placeholder{...}, every \\ref has a \\label, the review version carries no
identifying information, and the generator scripts under paper/ run and reproduce the
committed tables. The study documents are checked for the parts the plan requires.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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
                     "sec_ethics.tex", "appendix.tex", "table1_counts.tex", "fig1_tokens.tex", "rules_placement.tex",
                     "prompt_examples.tex"):
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


def test_hypotheses_h1_to_h6_are_all_stated():
    joined = "\n".join(strip_comments(f.read_text(encoding="utf-8")) for f in tex_sources())
    for i in range(1, 7):
        assert rf"\hyp{{{i}}}" in joined, f"H{i} missing from the paper"
    assert r"\hyp{3b}" in joined, "H3b (tone-isolation DiD) missing from the paper"
    assert "Pre-registered" in joined or "pre-registered" in joined
    # design-document wording that must not regress to the founding plan's
    assert "Northern style" not in joined                       # variant codes are region-neutral
    assert "mediat" not in joined.lower() or "not estimate" in joined or "do not estimate" in joined


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
    for name in ("rules_placement.tex", "rules_variants.tex", "rules_onsets.tex"):
        a = [ln for ln in (out / name).read_text(encoding="utf-8").splitlines() if not ln.startswith("% GENERATED")]
        b = [ln for ln in (PAPER / "tables" / name).read_text(encoding="utf-8").splitlines() if not ln.startswith("% GENERATED")]
        assert a == b, f"committed {name} differs from a fresh generation; re-run paper/gen_rule_tables.py"


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


def test_committed_table1_is_consistently_marked():
    tex = (PAPER / "tables" / "table1_counts.tex").read_text(encoding="utf-8")
    assert "% GENERATED by paper/gen_table1.py" in tex
    header_smoke = "% smoke build; regenerate" in tex
    caption_smoke = r"\placeholder{smoke build; regenerate" in tex
    assert header_smoke == caption_smoke, "smoke marking must appear in both the header comment and the caption"


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
    d4 = [ln for ln in t.splitlines() if re.match(r"\|\s*D4\s*\|", ln)][0]
    assert "**No**" in d4
    assert "UNCERTAIN: verify" in t


def test_ai_use_log_first_entry():
    t = _doc("AI_USE_LOG.md")
    assert "Claude Code" in t and "E1" in t and "Acknowledgements" in t and "2026-09-30" in t
    assert "rule engine" in t
