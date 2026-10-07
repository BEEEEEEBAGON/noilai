"""scripts/ingest_sheets.py on synthetic returned sheets built from the items of a NON-frozen build: the long and
the wide (Google Forms) baseline layouts, scoring through the models' scorer (strict / lenient / tolerant), the
pre-registered exclusions (too few answers, self-reported tool use, a failed instruction check, a duplicate response;
PREREGISTRATION section 5 item 7), the two-way person x item bootstrap, the agreement block, the dropping of name
and e-mail columns, the validation sheets scored through make_validation_forms.cmd_score with their checks, the
template, the refusal to write under paper/ and the Makefile target."""
import csv
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
IS = importlib.import_module("ingest_sheets")

PY = sys.executable
SMOKE_DEV = Path("/tmp/claude-0/-home-user-noilai/7ed5d88b-f91c-57ec-b5a3-07ddcf2f1a1c/scratchpad/smoke_build/v0.3-smoke/noilai_dev.jsonl")
N_BOOT = 100          # enough for the interval checks below; the script's default is constants.BOOTSTRAP_B
EMAIL = "someone@example.org"
NAME = "Nguyễn Văn A"


def _run(*args, check=True):
    return subprocess.run([PY, "scripts/ingest_sheets.py", *args], cwd=ROOT, text=True, capture_output=True, check=check)


@pytest.fixture(scope="module")
def item_file(tmp_path_factory) -> Path:
    """The local NON-frozen smoke build's dev file when present, else a small seeded build in a temporary
    directory (never under data/release/)."""
    if SMOKE_DEV.exists():
        return SMOKE_DEV
    out = tmp_path_factory.mktemp("rel")
    subprocess.run([PY, "scripts/build_data.py", "--out", str(out), "--seed", "3", "--n-lexicon", "120", "--n-pseudo", "60",
                    "--per-cell-t1", "40", "--per-cell-t2", "20", "--per-cell-t3", "15", "--core-per-cell", "8"],
                   cwd=ROOT, text=True, capture_output=True, check=True)
    return out / "noilai_dev.jsonl"


@pytest.fixture(scope="module")
def six(item_file) -> list[dict]:
    """Six items of a form: three T1 (V1, V2, V3), one T2, a T3 'yes' and a T3 'no'."""
    from noilai.gen.generate import load_items

    items = load_items(item_file)
    picked = []
    for task, variant, gold in (("T1", "V1", None), ("T1", "V2", None), ("T1", "V3", None), ("T2", "V1", None),
                                ("T3", "V1", "yes"), ("T3", "V2", "no")):
        picked.append(next(it for it in items if it["task"] == task and it["variant"] == variant
                           and (gold is None or it["gold"] == gold)))
    return picked


def _gold(it: dict) -> str:
    if it["task"] == "T1":
        return it["gold"][0]
    if it["task"] == "T2":
        return it["gold"][0]["output"]
    return "Có" if it["gold"] == "yes" else "Không"


def _swap(text: str) -> str:
    a, b = text.split()
    return f"{b} {a}"


def _write_long(path: Path, form: str, answers: list[tuple[str, str]], tool_use: str = "không", check: str = "Có",
                region: str = "", respondent: str = "", email_column: bool = False) -> None:
    fields = ["form_id", "respondent", "item_id", "answer"] + (["email"] if email_column else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        rows = [*answers, ("tool_use", tool_use), ("instruction_check", check), ("region", region)]
        for item_id, ans in rows:
            w.writerow({"form_id": form, "respondent": respondent, "item_id": item_id, "answer": ans,
                        **({"email": EMAIL} if email_column else {})})


def _write_wide(path: Path, items: list[dict], responses: list[dict], extra_item_header: str | None = None) -> None:
    """A Google Forms export: one row per response, item questions carrying the item id in their header."""
    header = ["Timestamp", "Email Address", "Họ và tên", "Bạn từ 18 tuổi trở lên?", "Vùng miền bạn lớn lên",
              *[f"[{it['item_id']}] câu {k}" for k, it in enumerate(items, 1)],
              "[instruction_check] Câu kiểm tra: hãy trả lời Có",
              "Bạn có dùng từ điển, công cụ tìm kiếm hay trợ lý AI cho câu nào không?", "Góp ý"]
    if extra_item_header:
        header.insert(5, extra_item_header)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in responses:
            row = ["2026/10/07 10:00:00", EMAIL, NAME, "Có", r.get("region", "Bắc")]
            if extra_item_header:
                row.append("x")
            row += r["answers"] + [r.get("check", "Có"), r.get("tool", "Không"), r.get("comment", "")]
            w.writerow(row)


def _no_personal_data(out: Path) -> None:
    for p in out.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            assert EMAIL not in text and NAME not in text and "2026/10/07" not in text, p


# ------------------------------------------------------------------ baseline sheets, long layout
@pytest.fixture(scope="module")
def long_run(tmp_path_factory, item_file, six):
    """Four long sheets: form 01 all right; form 02 with a swapped T1 (lenient only), a blank T1 and a wrong T3;
    form 03 answers 2 of 6 (too few); form 04 reports tool use."""
    tmp = tmp_path_factory.mktemp("long")
    ret = tmp / "returned"
    ret.mkdir()
    _write_long(ret / "baseline_form_01.csv", "1", [(it["item_id"], _gold(it)) for it in six], region="Miền Nam",
                email_column=True)
    ans2 = [_swap(_gold(six[0])), "", _gold(six[2]), _gold(six[3]), "Không", _gold(six[5])]
    _write_long(ret / "baseline_form_02.csv", "", list(zip([it["item_id"] for it in six], ans2)), tool_use="no",
                respondent=NAME)                                   # a name as respondent id: replaced, reported
    _write_long(ret / "baseline_form_03.csv", "3", [(it["item_id"], _gold(it) if k < 2 else "") for k, it in enumerate(six)])
    _write_long(ret / "baseline_form_04.csv", "4", [(it["item_id"], _gold(it)) for it in six], tool_use="Có")
    design = tmp / "human_items.json"
    design.write_text(json.dumps([it["item_id"] for it in six[:5]]), encoding="utf-8")      # the sixth is outside the design
    out = tmp / "out"
    r = _run("--items", str(item_file), "--baseline-returned", str(ret / "*.csv"), "--out", str(out), "--validation-out",
             str(tmp / "val"), "--min-answered", "3", "--instruction-check-answer", "Có", "--n-boot", str(N_BOOT),
             "--human-items", str(design))
    rows = [json.loads(ln) for ln in (out / "baseline_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
    report = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    return {"out": out, "stdout": r.stdout, "rows": rows, "summary": summary, "report": report}


def test_long_sheets_are_scored_with_the_models_scorer(long_run, six):
    """HUMAN_BASELINE_FORM.md section 4: the same scorer as the models; strict / lenient / tolerant per row (DD 10.2
    item 32); a blank answer is wrong; the form number is what ledger.py ingest looks for."""
    rows = long_run["rows"]
    assert len(rows) == 12 and {r["form"] for r in rows} == {"01", "02"}
    assert set(rows[0]) == {"form", "respondent", "item_id", "task", "variant", "answer", "extracted", "strict", "lenient",
                            "tolerant", "error_class", "region"}
    by = {(r["form"], r["item_id"]): r for r in rows}
    assert all(by[("01", it["item_id"])]["strict"] for it in six)
    swapped = by[("02", six[0]["item_id"])]
    assert swapped["strict"] is False and swapped["lenient"] is True and swapped["tolerant"] is True
    assert swapped["error_class"] == "lenient_only"
    blank = by[("02", six[1]["item_id"])]
    assert blank["answer"] == "" and blank["extracted"] is None and blank["strict"] is False and blank["error_class"] == "unparseable"
    assert by[("02", six[4]["item_id"])]["error_class"] == "wrong" and by[("02", six[5]["item_id"])]["strict"] is True
    assert {r["task"] for r in rows} == {"T1", "T2", "T3"}
    assert all(r["region"] == "Nam" for r in rows if r["form"] == "01")


def test_long_sheets_exclusions_and_summary(long_run, six):
    """PREREGISTRATION section 5 item 7: a form with too few answers and a respondent reporting tool use are excluded
    and counted; mean-human = the mean over items of the item's mean judgment with a two-way bootstrap; any-human is
    the ceiling; alpha between the two raters on correctness and on the answer (DD 10.2 / PREREGISTRATION 8.11)."""
    s = long_run["summary"]
    assert s["n_respondents_received"] == 4 and s["n_respondents_included"] == 2 and s["forms_included"] == ["01", "02"]
    assert s["excluded"]["n"] == 2 and s["excluded"]["by_reason"] == {"too_few_answers": 1, "tool_use": 1}
    reasons = {e["form"]: e["reasons"] for e in s["excluded"]["respondents"]}
    assert reasons == {"03": ["too_few_answers"], "04": ["tool_use"]}
    assert s["tool_use"] == {"yes": 1, "no": 3} and s["instruction_check"] == {"passed": 4}
    assert s["items"]["raters_per_item"] == {"2": 6} and s["items"]["n_double_judged"] == 6
    assert s["items"]["not_in_human_design"] == [six[5]["item_id"]] and s["items"]["design_items_unanswered"] == 0
    overall = s["accuracy"]["overall"]
    # strict: form 01 right on 6, form 02 on 3 -> three items at 0.5 and three at 1.0 -> mean over items 0.75
    assert overall["strict"]["mean_human"]["estimate"] == pytest.approx(0.75)
    assert overall["lenient"]["mean_human"]["estimate"] == pytest.approx(5 / 6)
    assert overall["tolerant"]["mean_human"]["estimate"] == pytest.approx(5 / 6)
    assert overall["strict"]["any_human"]["estimate"] == 1.0
    mh = overall["strict"]["mean_human"]
    assert mh["lo"] <= mh["estimate"] <= mh["hi"] and mh["n_boot"] == N_BOOT and mh["n_persons"] == 2 and mh["n_items"] == 6
    assert mh["method"] == "two_way_person_item_bootstrap_mean"
    assert set(s["accuracy"]["by_task"]) == {"T1", "T2", "T3"} and "T1-V1" in s["accuracy"]["by_task_variant"]
    assert s["accuracy"]["by_task"]["T2"]["strict"]["mean_human"]["estimate"] == 1.0
    ag = s["agreement"]["double_judged"]
    assert ag["correct_strict"]["n_items"] == 6 and ag["correct_strict"]["n_ratings"] == 12
    assert ag["correct_strict"]["percent_agreement"] == 0.5 and -1.0 <= ag["correct_strict"]["alpha"] <= 1.0
    assert ag["correct_strict"]["alpha_ci"][0] <= ag["correct_strict"]["alpha"] <= ag["correct_strict"]["alpha_ci"][1]
    assert ag["answer"]["n_items"] == 6 and -1.0 <= ag["answer"]["alpha"] <= 1.0
    assert s["error_classes"]["T1"] == {"correct": 4, "lenient_only": 1, "unparseable": 1}
    assert s["by_region"]["Nam"]["n_respondents"] == 1 and s["by_region"]["unreported"]["n_respondents"] == 1
    assert "mean-human strict accuracy 0.750" in long_run["stdout"]


def test_long_sheets_drop_personal_columns_and_name_like_respondent_ids(long_run):
    """No name, e-mail or timestamp reaches experiments/human/: the e-mail column is dropped and listed by header
    only; a respondent id that is not a letter or a number is replaced by the form number and reported."""
    _no_personal_data(long_run["out"])
    files = {f["file"]: f for f in long_run["report"]["baseline"]}
    assert files["baseline_form_01.csv"]["dropped_columns"] == ["email"] and files["baseline_form_01.csv"]["layout"] == "long"
    assert all(re.fullmatch(r"\d{2}", r["respondent"]) for r in long_run["rows"])
    checks = [p["check"] for p in files["baseline_form_02.csv"]["problems"]]
    assert checks == ["respondent_id_replaced"] and long_run["report"]["n_problems"] == {"error": 0, "warning": 1}
    assert long_run["report"]["exclusions"]["by_reason"] == {"too_few_answers": 1, "tool_use": 1}


# ------------------------------------------------------------------ baseline sheets, wide (Google Forms) layout
def test_wide_google_forms_export(tmp_path, item_file, six):
    """One row per respondent, the item id in the question header; e-mail and name columns dropped; the instruction
    check and the tool-use question read from their headers; a failed check and a second response on the same form
    excluded and counted; an item-shaped header that is not a release item flagged."""
    ret = tmp_path / "returned"
    ret.mkdir()
    right = [_gold(it) for it in six]
    _write_wide(ret / "Phiếu 05 (Responses).csv", six, [{"answers": right, "region": "Miền Bắc"}])
    _write_wide(ret / "baseline_form_06.csv", six, [{"answers": right, "check": "Không"},            # failed check
                                                     {"answers": right, "check": "Có"}])             # then the kept one
    _write_wide(ret / "baseline_form_07.csv", six, [{"answers": right}, {"answers": right}],         # a duplicate response
                extra_item_header="[T1-V1-999999] câu lạ")
    out = tmp_path / "out"
    r = _run("--items", str(item_file), "--baseline-returned", str(ret / "*.csv"), "--out", str(out), "--validation-out",
             str(tmp_path / "val"), "--min-answered", "3", "--instruction-check-answer", "Có", "--n-boot", str(N_BOOT))
    rows = [json.loads(ln) for ln in (out / "baseline_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    s = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
    rep = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    _no_personal_data(out)
    assert len(rows) == 18 and {r["form"] for r in rows} == {"05", "06", "07"} and all(r["strict"] for r in rows)
    assert {r["respondent"] for r in rows} == {"05", "06-2", "07-1"}
    assert all(r["region"] == "Bắc" for r in rows if r["form"] == "05")
    assert s["excluded"]["by_reason"] == {"duplicate_form_response": 1, "failed_instruction_check": 1}
    assert s["instruction_check"] == {"passed": 4, "failed": 1} and s["tool_use"] == {"no": 5}
    assert s["items"]["raters_per_item"] == {"3": 6}
    files = {f["file"]: f for f in rep["baseline"]}
    assert files["Phiếu 05 (Responses).csv"]["layout"] == "wide"
    assert files["Phiếu 05 (Responses).csv"]["dropped_columns"] == ["Email Address", "Họ và tên"]
    assert [p["check"] for p in files["baseline_form_07.csv"]["problems"]] == ["unknown_item_column"]
    assert "unknown_item_column" in r.stdout and rep["n_problems"]["error"] == 1
    # --fail-on-problems turns the flagged header into a non-zero exit
    bad = _run("--items", str(item_file), "--baseline-returned", str(ret / "baseline_form_07.csv"), "--out", str(tmp_path / "o2"),
               "--validation-out", str(tmp_path / "v2"), "--min-answered", "3", "--n-boot", "10", "--fail-on-problems", check=False)
    assert bad.returncode == 1


def test_instruction_check_as_a_release_item_is_scored_by_the_scorer(tmp_path, item_file, six):
    """HUMAN_BASELINE_FORM.md section 1: the check is an anchor-style item whose answer is in the instructions; when
    its id is a release item and no --instruction-check-answer is given, the scorer decides; its row is never a
    judgment."""
    ret = tmp_path / "returned"
    ret.mkdir()
    check = six[5]
    body = [(it["item_id"], _gold(it)) for it in six[:5]]
    with open(ret / "baseline_form_08.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "answer"])
        w.writerows([*body, (check["item_id"], _gold(check)), ("tool_use", "no")])
    with open(ret / "baseline_form_09.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "answer"])
        w.writerows([*body, (check["item_id"], "Có" if _gold(check) == "Không" else "Không"), ("tool_use", "no")])
    out = tmp_path / "out"
    _run("--items", str(item_file), "--baseline-returned", str(ret / "*.csv"), "--out", str(out), "--validation-out",
         str(tmp_path / "val"), "--min-answered", "3", "--instruction-check-id", check["item_id"], "--n-boot", "10")
    rows = [json.loads(ln) for ln in (out / "baseline_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    s = json.loads((out / "baseline_summary.json").read_text(encoding="utf-8"))
    assert {r["form"] for r in rows} == {"08"} and len(rows) == 5 and check["item_id"] not in {r["item_id"] for r in rows}
    assert s["instruction_check"] == {"passed": 1, "failed": 1}
    assert s["excluded"]["by_reason"] == {"failed_instruction_check": 1}


# ------------------------------------------------------------------ validation sheets
def test_validation_sheets_are_checked_and_scored_through_cmd_score(tmp_path, item_file):
    """Two coders; an invalid label, a duplicate rating and an unknown item id are reported and set aside; the e-mail
    column is dropped; the report is cmd_score's (alpha, AC1, marginals, generator precision) plus `coders` for
    ledger.py ingest, written to <out>/validation_report.json and, cmd_score's own copy, to --validation-out."""
    from noilai.gen.generate import load_items

    items = [it for it in load_items(item_file) if it["task"] == "T1"][:40]
    n = len(items)                       # 40 on the local smoke build; CI's small seeded build has fewer T1 dev items
    assert n >= 6
    ret = tmp_path / "returned"
    ret.mkdir()
    fields = ["item_id", "task", "variant", "input", "candidate", "question_vi", "correct", "spelling", "lexical", "offensive",
              "comment", "validator_email"]
    for coder in ("A", "B"):
        with open(ret / f"validation_form_{coder}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, it in enumerate(items):
                correct = "no" if i < 2 else "yes"
                if coder == "B" and i in (2, 3):
                    correct = "không"
                if coder == "A" and i == 5:
                    correct = "maybe"
                w.writerow({"item_id": it["item_id"], "task": "T1", "variant": it["variant"], "input": it["input"],
                            "candidate": it["gold"][0], "question_vi": "q", "correct": correct, "spelling": "yes",
                            "lexical": "no", "offensive": "no", "comment": "", "validator_email": EMAIL})
            if coder == "A":
                w.writerow({"item_id": items[0]["item_id"], "correct": "yes", "spelling": "yes", "lexical": "no", "offensive": "no"})
                w.writerow({"item_id": "T1-V1-999999", "correct": "yes", "spelling": "yes", "lexical": "no", "offensive": "no"})
    out, vout = tmp_path / "out", tmp_path / "val"
    r = _run("--items", str(item_file), "--validation-returned", str(ret / "*.csv"), "--out", str(out), "--validation-out", str(vout))
    rep = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert rep["coders"] == ["A", "B"] and rep["n_ratings_per_coder"] == {"A": n, "B": n}
    assert {"alpha", "alpha_ci", "percent_agreement", "n_ratings", "ac1", "ac1_ci", "marginals"} <= set(rep["correct"])
    assert rep["correct"]["n_ratings"] == 2 * n - 1 and rep["spelling"]["ac1"] == 1.0    # the blanked 'maybe' is unrated
    assert rep["generator_precision"]["n_items"] == n - 2                                 # two items without a majority
    assert [p["check"] for p in rep["problems"]] == ["invalid_label", "duplicate_rating", "unknown_item_id"]
    fa = next(f for f in rep["files"] if f["file"] == "validation_form_A.csv")
    assert fa["dropped_columns"] == ["validator_email"] and fa["n_unrated"] == {"correct": 1} and fa["n_rated"]["correct"] == n - 1
    assert (vout / "validation_report.json").exists() and rep["scored_by"].endswith("cmd_score")
    _no_personal_data(out)
    _no_personal_data(vout)
    assert "invalid_label" in r.stdout and "duplicate_rating" in r.stdout and "unknown_item_id" in r.stdout
    assert (out / "ingest_report.json").exists() and not (out / "baseline_scores.jsonl").exists()


# ------------------------------------------------------------------ helpers, template, refusals, Makefile
def test_two_way_bootstrap_resamples_persons_and_items():
    """Mean-human = the mean over items of the item mean; a person who is always right and one always wrong give 0.5
    with replicates at 0, 0.5 and 1 (the person dimension is resampled, not only the items); any-human is 1."""
    persons = ["A"] * 10 + ["B"] * 10
    items = [f"i{k}" for k in range(10)] * 2
    correct = [True] * 10 + [False] * 10
    ci = IS.two_way_bootstrap(correct, persons, items, n_boot=400, seed=1)
    assert ci["estimate"] == 0.5 and ci["lo"] == 0.0 and ci["hi"] == 1.0 and ci["n_persons"] == 2 and ci["n_items"] == 10
    any_ = IS.two_way_bootstrap(correct, persons, items, n_boot=50, seed=1, stat="any")
    assert any_["estimate"] == 1.0 and any_["method"].endswith("_any")
    # unequal raters per item: the item means are averaged, so 20 raters on one item weigh as much as two on another
    ci2 = IS.two_way_bootstrap([True] * 20 + [False, False], [f"p{k}" for k in range(20)] + ["p0", "p1"],
                               ["anchor"] * 20 + ["x", "x"], n_boot=20, seed=0)
    assert ci2["estimate"] == 0.5


def test_small_helpers():
    assert IS.form_id_from_name(Path("baseline_form_03.csv")) == "03" and IS.form_id_from_name(Path("Phiếu 12 (Responses).csv")) == "12"
    assert IS.form_id_from_name(Path("sheet.csv")) == "sheet" and IS.normalize_form_id("form 7", "x") == "07"
    assert IS.parse_yesno("Có") is True and IS.parse_yesno("không.") is False and IS.parse_yesno("maybe") is None
    assert IS.region_band("ngoài Việt Nam") == "ngoài Việt Nam" and IS.region_band("Miền Nam") == "Nam" and IS.region_band("") is None
    problems = []
    assert IS.safe_respondent("B", "01", problems, "w") == "B" and IS.safe_respondent("12a", "01", problems, "w") == "12a"
    assert IS.safe_respondent(NAME, "01", problems, "w") == "01" and IS.safe_respondent(EMAIL, "01", problems, "w") == "01"
    assert len(problems) == 2 and all(p["check"] == "respondent_id_replaced" for p in problems)
    header = ["Timestamp", "Email Address", "[T1-V1-000001] q", "[instruction_check] q", "Bạn có dùng từ điển?", "Vùng miền"]
    roles = IS.classify_headers(header, {"T1-V1-000001"}, "instruction_check", "tool_use")
    assert [r for r, _ in roles] == ["other", "other", "item", "check", "tool", "region"] and roles[2][1] == "T1-V1-000001"
    assert IS.personal_columns(header, [["t", EMAIL, "a", "b", "c", "d"]]) == [1]


def test_item_question_headers_quoting_a_name_like_word_are_not_dropped():
    """A Google Forms header quotes the item's input, and the lexicon has "cung tên" and "tên lửa": a header carrying
    an item id is a question column, dropped only when its values look like e-mails; a bare name header is dropped."""
    header = ["Timestamp", "Họ và tên", "[T1-V1-000001] Nói lái kiểu V1 của “cung tên” là gì?", "[T2-V1-000002] “tên lửa”"]
    assert IS.personal_columns(header, [["t", NAME, "nêt cung", "lửa tên"]]) == [1]
    assert IS.personal_columns(header, [["t", NAME, EMAIL, "lửa tên"]]) == [1, 2]
    roles = IS.classify_headers(header, {"T1-V1-000001", "T2-V1-000002"}, "instruction_check", "tool_use")
    assert [r for r, _ in roles] == ["other", "other", "item", "item"]


def test_blank_instruction_check_is_missing_not_failed():
    """A skipped check (a blank cell in a wide export, the untouched template row in a long sheet) is counted as
    missing and does not exclude the respondent; an answered wrong one does (PREREGISTRATION section 5 item 7)."""
    answers = {f"T1-V1-{k:06d}": "a b" for k in range(4)}
    blank = IS.Respondent("01", "01", "f", dict(answers), tool_use=False, instruction_check="")
    none = IS.Respondent("02", "02", "f", dict(answers), tool_use=False, instruction_check=None)
    wrong = IS.Respondent("03", "03", "f", dict(answers), tool_use=False, instruction_check="Không")
    kept, excluded, tally = IS.apply_exclusions([blank, none, wrong], min_answered=3, check_expected="Có", check_item=None)
    assert [R.form for R in kept] == ["01", "02"] and [e["reasons"] for e in excluded] == [["failed_instruction_check"]]
    assert tally["instruction_check"] == {"missing": 2, "failed": 1}


def test_template_prints_an_empty_long_sheet():
    r = _run("--template")
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "form_id,respondent,item_id,answer"
    assert [ln.split(",")[2] for ln in lines[1:]] == ["tool_use", "instruction_check", "region"]
    assert all(ln.endswith(",") for ln in lines[1:])


def test_refuses_paper_and_handles_no_sheets_and_missing_items(tmp_path, item_file):
    r = _run("--baseline-returned", str(tmp_path / "*.csv"), "--out", str(ROOT / "paper" / "x"), check=False)
    assert r.returncode == 1 and "refusing to write under paper/" in r.stderr and not (ROOT / "paper" / "x").exists()
    r = _run("--items", str(item_file), "--baseline-returned", str(tmp_path / "none" / "*.csv"), "--out", str(tmp_path / "o"))
    assert r.returncode == 0 and "nothing to ingest" in r.stdout and not (tmp_path / "o").exists()
    (tmp_path / "s.csv").write_text("item_id,answer\n", encoding="utf-8")
    r = _run("--items", str(tmp_path / "missing.jsonl"), "--baseline-returned", str(tmp_path / "s.csv"), "--out", str(tmp_path / "o"),
             check=False)
    assert r.returncode == 1 and "not found" in r.stderr


def test_makefile_target_runs_the_script_over_both_returned_globs():
    mk = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^ingest-sheets:", mk, re.MULTILINE) and re.search(r"^\.PHONY:.*\bingest-sheets\b", mk, re.MULTILINE)
    target = mk[mk.index("ingest-sheets:"):]
    assert "scripts/ingest_sheets.py" in target
    assert "'data/validation/returned/*.csv'" in target and "'data/human/returned/*.csv'" in target
    assert "--items $(RELEASE)/noilai_test.jsonl $(RELEASE)/noilai_dev.jsonl" in target and "human_items.json" in target
    if __import__("shutil").which("make"):
        r = subprocess.run(["make", "-n", "ingest-sheets"], cwd=ROOT, text=True, capture_output=True, check=False)
        assert r.returncode == 0 and "ingest_sheets.py" in r.stdout and "--baseline-returned" in r.stdout
