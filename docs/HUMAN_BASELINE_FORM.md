# Human-baseline form: design (20 respondents × 30 items, about 30–45 minutes each)

Purpose: a human level of performance on the same items, scored by the same code, for Table 2 of the paper. Design per `docs/DESIGN_DECISIONS.md` §10.2 (which supersedes the founding plan's "40 items from the core"; amended 1 October 2026, `docs/DEVIATIONS.md`): **20 adult native speakers, 30 items each; 6 anchor items seen by everyone and 24 items assigned so that every non-anchor item is seen by exactly two people → 6 + 240 = 246 distinct items**, 20 per task × variant cell, all drawn from the **open-model main sample** so that every model is scored on exactly these 246 items. Each form also carries one instruction-check row, a 10-item natural-competence block and two closing questions. Volunteers, unpaid; consent: `docs/CONSENT_FORM.md` (role "human-baseline respondent"). Vietnamese wording is marked **[NATIVE-CHECK]**. The step-by-step run (commands, exclusions, analysis, timeline) is `docs/gate1/HUMAN_BASELINE_PROTOCOL.md`; this file is the design and the form layout.

## 1. Item sampling and assignment

- **Source:** the open-model main sample (`data/release/v0.3/noilai_main.jsonl`, the run plan's seeded 4,200-item stratification of the test split; `make baseline` reads `$(RELEASE)/noilai_main.jsonl` with `RELEASE = data/release/v0.3`), restricted to items with `vulgar = false` and excluding spelling twins (they leave the T3 headline as well). The file opens with a canary header record, which the builder skips. There is no fallback to the core: the main sample is built before the baseline, and the human items must be inside it.
- **Anchors (6):** one T1 item per generated variant (V1–V4), one T2 item, one T3 item; seen by every respondent; they measure between-person agreement on identical material.
- **Double-judged items (240):** 20 per cell (T1 × 4, T2 × 4, T3 × 4 = 12 cells; T3 as yes/no items, half yes and half no, never both members of a twin pair, and never the twin of an anchor); each assigned to exactly two respondents. With 20 respondents × 24 slots = 480 slots = 240 × 2. **Assignment: a cyclic double-coverage design.** Shuffle the 240 items (seeded) and read them as 12 blocks of 20. The first copy of item `k` goes to respondent `k % 20` (round robin); the second copy goes to respondent `(k % 20 + offset) % 20` with a **block-specific offset** `offset = 1 + (k // 20) % 19`, which runs 1, 2, …, 12 over the 12 blocks. Consequences, checked by the snippet's coverage printout and by `tests/test_paper.py`: no respondent sees an item twice; every respondent judges exactly 24 distinct items; and every one of the 190 rater pairs shares at least one non-anchor item (140 pairs share one, 50 pairs share two; 240/190 ≈ 1.3 on average), so the rater graph is complete. A fixed offset (say 10 for every block) would instead produce 10 disjoint dyads each sharing all 24 of their items, with 180 rater pairs sharing nothing; the nominal α over the 240 double-judged items and the two-way (person, item) bootstrap would then rest on a disconnected design. The design is not a balanced incomplete block in the strict sense (pairs share one *or* two items), which is why it is called double-coverage here.
- **Order:** each form's 30 items in a seeded random order (T1, T2, T3 interleaved), with one **instruction-check row** `CHECK-01` inserted at a random position after the first five (it asks the respondent to write the two words *đã đọc* [NATIVE-CHECK]; a wrong answer excludes the form). The 10-item natural-competence block and the two closing rows follow the 31 rows (§2).
- **Coverage check:** the builder prints one JSON line (`forms`, `per_form`, `anchors`, `per_cell`, `double_judged`, `distinct_items`, `min_appearances`, `max_appearances`, `anchors_seen_by`, `others`, `shared_items_per_form_pair`, `rater_graph_connected`, `natural_block_items`, ...) and writes it, with every item's number of appearances, to `data/human/baseline_manifest.json`. `make baseline` fails and deletes the forms unless `forms` = 20, `per_form` = 30, `distinct_items` = 246, `min_appearances` = 2 and `max_appearances` = 20; it prints the line back as a Python dict, not JSON. It does not test `anchors_seen_by` = 20, `others` = [2] (anchors on all 20 forms, every other item on exactly 2), `rater_graph_connected` or `natural_block_items` = 10; the author checks those by eye (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §4 step 1). Pending Makefile change: test all nine and print JSON.

**What runs:** `make baseline`, i.e. `scripts/make_validation_forms.py baseline --items data/release/v0.3/noilai_main.jsonl --attested data/release/v0.3/attested.jsonl --out data/human --n-forms 20 --per-form 30 --seed 20261102` (the assignment is `baseline_design()` in that script). The snippet below is kept only as the readable statement of the assignment; it is not what builds the forms, and its CSV layout (`item_id, task, variant, prompt_vi, answer`) is the earlier one. The script's columns are in §4.

```bash
/home/user/venv-noilai/bin/python - <<'EOF'
# [reference implementation; the script's baseline_design() is the one that runs]
import csv, json, random
from collections import defaultdict
from pathlib import Path
SEED, N_RESP, PER_CELL = 20261102, 20, 20   # a public sampling seed; never the (withheld) build seed, DESIGN_DECISIONS 4.6
rng = random.Random(SEED)
ITEMS = "data/release/v0.3/noilai_main.jsonl"      # the run plan's open-model main sample (v0.3 at the freeze)
rows = [json.loads(l) for l in open(ITEMS, encoding="utf-8") if l.strip()]
items = [it for it in rows if "task" in it]        # skip the canary header record
items = [it for it in items if not it.get("vulgar") and it.get("twin_type") != "spelling"]
by_cell = defaultdict(list)
for it in items: by_cell[(it["task"], it["variant"])].append(it)
anchors, pool = [], []
for cell in sorted(by_cell):
    c = by_cell[cell]; rng.shuffle(c)
    if cell[0] == "T1" or cell in (("T2", "V1"), ("T3", "V1")):
        anchors.append(c.pop())
        c = [x for x in c if x.get("pair_item_id") != anchors[-1]["item_id"]]   # never the twin of an anchor
    if cell[0] == "T3":  # 10 yes + 10 no
        yes = [x for x in c if x["gold"] == "yes"][:PER_CELL // 2]
        no = [x for x in c if x["gold"] == "no" and x["pair_item_id"] not in {y["item_id"] for y in yes}][:PER_CELL // 2]
        pool += yes + no
    else:
        pool += c[:PER_CELL]
assert len(anchors) == 6 and len(pool) == 240, (len(anchors), len(pool))
rng.shuffle(pool)
forms = [list(anchors) for _ in range(N_RESP)]
for k, it in enumerate(pool):                       # first copy: round robin
    forms[k % N_RESP].append(it)
for k, it in enumerate(pool):                       # second copy: block-specific cyclic offset 1..12
    offset = 1 + (k // N_RESP) % (N_RESP - 1)       # never 0 (same person) and never a fixed dyad
    forms[(k % N_RESP + offset) % N_RESP].append(it)
out = Path("data/human"); out.mkdir(parents=True, exist_ok=True)
def q(it):
    if it["task"] == "T1": return f"Nói lái kiểu {it['variant']} của “{it['input']}” là gì?"
    if it["task"] == "T2": return f"“{it['input']}” là cách nói lái của cụm từ nào?"
    return f"“{it['candidate']}” có phải là nói lái kiểu {it['variant']} của “{it['input']}” không? (Có/Không)"
for i, fm in enumerate(forms, 1):
    assert len(fm) == 30 and len({x["item_id"] for x in fm}) == 30
    rng.shuffle(fm)
    with open(out / f"baseline_form_{i:02d}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "task", "variant", "prompt_vi", "answer"]); w.writeheader()
        for it in fm: w.writerow({"item_id": it["item_id"], "task": it["task"], "variant": it["variant"], "prompt_vi": q(it), "answer": ""})
cover = defaultdict(int)
for fm in forms:
    for it in fm: cover[it["item_id"]] += 1
anchor_ids = {a["item_id"] for a in anchors}
seen = [{x["item_id"] for x in fm} - anchor_ids for fm in forms]
shared = defaultdict(int)                            # coverage printout: non-anchor items shared per rater pair
for i in range(N_RESP):
    for j in range(i + 1, N_RESP):
        shared[len(seen[i] & seen[j])] += 1
assert 0 not in shared, "a rater pair shares no item: the rater graph is disconnected"
print({"forms": N_RESP, "items": len(cover), "anchors_seen_by": min(cover[a["item_id"]] for a in anchors),
       "others": sorted(set(v for k, v in cover.items() if k not in anchor_ids)),
       "shared_items_per_rater_pair": dict(sorted(shared.items()))})     # expected {1: 140, 2: 50}
(out / "human_items.json").write_text(json.dumps(sorted(cover), ensure_ascii=False))   # the 246 ids every model is scored on
EOF
```

## 2. Form layout (Google Forms built by the Apps Script; e-mail collection **off**)

The forms are built by an Apps Script, not by hand: `python scripts/make_validation_forms.py google-form --dir data/human` writes `data/human/build_forms.gs`, which the author pastes into script.google.com in the author's own Google account and runs (`buildAll`; one form per `baseline_form_<nn>.csv`; the log lists each form's link). Each form, titled "NóiLái - phiếu <nn>", has a progress bar, the introduction of §3 as its description, and these pages:

1. **Consent page** — one required question with **three boxes** (`CONSENT_BOXES_VI`: 18 or older; I have read the information sheet and agree, voluntary, unpaid, can stop at any time; I agree to storage and aggregate, anonymous publication); the form cannot be submitted unless all three are ticked. The full consent text (`docs/CONSENT_FORM.md`) is sent with the link as a PDF.

   **Gap (pending script change).** The consent question has no help text and the form sets no confirmation message (`buildOne` calls neither `setHelpText` on it nor `setConfirmationMessage`). A respondent who never opens the PDF therefore sees no contact, no withdrawal route and no statement that no ethics board reviewed the study, and after submitting sees Google's default message, not their form number. Until the script sets them, the personal message that carries the link states the form number (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §4 step 6); withdrawal is also possible by name (`docs/gate1/RECRUITMENT.md` §4). Proposed strings, `<nn>` = the form number, `[EMAIL]` filled as in the consent form:
   - help text under "Đồng ý tham gia" [NATIVE-CHECK]: `Tóm tắt: tình nguyện, không trả tiền; phiếu không ghi tên; nghiên cứu chưa được hội đồng đạo đức xét duyệt; muốn rút lui, báo mã phiếu <nn> tới [EMAIL] trước 22/11/2026.` (English: "Summary: voluntary, unpaid; the form records no name; the study has not been reviewed by an ethics board; to withdraw, send form number <nn> to [EMAIL] before 22 November 2026.")
   - confirmation message [NATIVE-CHECK]: `Cảm ơn bạn. Mã phiếu của bạn: <nn>. Muốn rút lui, báo mã này tới [EMAIL] trước 22/11/2026.` (English: "Thank you. Your form number: <nn>. To withdraw, send this number to [EMAIL] before 22 November 2026.")
2. **Demographics (coarse; age band required, the rest optional):** age band (18–29 / 30–49 / 50+); region grown up in (Bắc / Trung / Nam / Ngoài Việt Nam); years lived in Vietnam (< 5 / 5–15 / > 15); raised abroad (Có / Không). Nothing else.
3. **Questions, 43 per form, numbered by position** (the number at the start of each title is what maps an answer back to its item):
   - positions 1–31: the 30 model items and the instruction-check row `CHECK-01`. Each model item's **help text is the exact p0 user message the models receive** — the same p0 prompt text, the same three demonstrations, the item and the `Đáp án:` line, rendered by `noilai.eval.prompts.render(item, paraphrase="p0", shots=3, arm="nfc")` (for T1/T3 the instruction names and defines the kind, for T2 it withholds it, exactly as for the models); its hash is the row's `prompt_hash`. The title is a short question (`baseline_prompt`) [NATIVE-CHECK]: T1 `Nói lái kiểu <V> của “<input>” là gì?` and T2 `“<input>” là cách nói lái của cụm từ nào?` with a short-text answer; T3 `“<candidate>” có phải là nói lái kiểu <V> của “<input>” không? (Có/Không)` with Có / Không. None of these is required: a blank counts as wrong.
   - positions 32–41: the **natural-competence block**, 10 attested, non-vulgar two-syllable nói lái drawn at random from the attested file (`data/release/v0.3/attested.jsonl`, seed 20261102), the same 10 on every form, asked as `“<form>” là cách nói lái của cụm từ nào?` with no model prompt. The draw does not select for familiarity or native verification and does not de-duplicate originals: a build from the scratch release drew three textbook illustrations of the same original (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §9, open decision 5).
   - positions 42–43: the **closing questions**, required, Có / Không: `TOOLS` (did you use a dictionary, a search engine or an AI assistant for any item?) and `WAS_VALIDATOR` (were you a validator of this project?). "Có" to either excludes the form and is counted.
4. **Free comment** (optional).

Google Forms records only the submission time, so completion time is not measured; the time figure in §5 is a planning estimate. The markers `[NATIVE-CHECK]` inside the script's strings are removed from `build_forms.gs` after the native check and before the script is run (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §4 step 3).

## 3. Introduction shown to respondents (the form's description) **[NATIVE-CHECK]**

The text that runs is `FORM_INTRO_VI` in `scripts/make_validation_forms.py`; the English is the reference for meaning.

**Tiếng Việt.** Cảm ơn bạn đã tham gia. Bạn sẽ làm 30 câu nói lái, giống hệt các câu mà các mô hình ngôn ngữ nhận được (mỗi câu có phần hướng dẫn và ba ví dụ mẫu; phần hướng dẫn lặp lại ở mỗi câu, bạn chỉ cần đọc kỹ ở vài câu đầu), sau đó 10 câu giải nói lái quen thuộc và hai câu hỏi cuối. Xin không dùng từ điển, công cụ tìm kiếm hay trợ lý AI; nếu không biết, hãy để trống hoặc đoán. Thời gian khoảng 30–45 phút.

**English.** Thank you for taking part. You will do 30 nói lái questions, identical to the questions the language models receive (each has an instruction and three worked examples; the instruction repeats on every question, so you only need to read it carefully in the first few), then 10 familiar nói lái to decode and two closing questions. Please do not use a dictionary, a search engine or an AI assistant; if you do not know, leave the answer blank or guess. Time: about 30–45 minutes.

Note: the words *quen thuộc* / *familiar* overstate how the 10 natural-block items are chosen (§2, item 3: drawn at random from the attested file). Until the selection changes (open decision 5 of `docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §9), drop *quen thuộc* from `FORM_INTRO_VI` in the script (pending script change), and in `docs/CONSENT_FORM.md` §2 and `docs/gate1/RECRUITMENT.md` message (b).

The instruction-check row reads, in English: "Check item: please write exactly the two words *đã đọc* in this item's answer box." The earlier framing text of this section (a separate instructions page with the p0 text) is superseded: the p0 text is now on every item.

## 4. Scoring

- **Form CSV columns** (`data/human/baseline_form_<nn>.csv`, one row per question): `form, position, item_id, task, variant, block, prompt_model, prompt_hash, question_short, answer`. `block` is `main` (the 30 model items), `check`, `natural` or `closing`; `prompt_model` and `prompt_hash` are filled for the 30 model items only.
- **Returned answers:** the Google Forms responses are downloaded as `data/human/responses/responses_form_<nn>.csv` and converted by `python scripts/make_validation_forms.py import-responses --dir data/human` into `data/human/returned/baseline_form_<nn>_r<k>.csv` (the form's own columns with `answer` filled) and `returned/demographics.csv`.
- **Scoring:** `python scripts/make_validation_forms.py score-baseline --items data/release/v0.3/noilai_main.jsonl --dir data/human --returned 'data/human/returned/baseline_form_*.csv'` converts each `main` row to an output row `{item_id, task, arm: "base", prompt_id: "human-<form>", raw: answer, answer: answer, extraction_method: "human"}` and scores it with `noilai.eval.score.score_outputs(items, outputs)` — the **same function** that scores the models: strict T1 (structures in order; either placement and *lí/lý* accepted; misspellings and regional homophones not repaired, but binned), T2 membership in the gold set, T3 Có/Không against the gold. A blank answer counts as wrong. The natural block is scored per form (the answer's two syllables equal the attested original's, in either order). Output: `data/human/report/human_scores.jsonl` and `human_baseline_report.json` (point estimates; the intervals and α below are the analysis step, `docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §6).
- **Aggregates (pre-registered, `docs/PREREGISTRATION.md` §8.11):** item-level human accuracy = mean of the item's two judgments; **mean-human accuracy** per task and variant is the comparator, with a 95% interval from a two-way (person, item) cluster bootstrap; "any human correct" is a ceiling reported separately; results by output lexicality and by region (hỏi/ngã V3 items and -n/-ng items separately for Southern speakers); nominal Krippendorff's α between the two raters over the 240 double-judged items on the produced answer and on correctness; the human error taxonomy from the same scorer. Both α and the two-way bootstrap assume a connected rater–item design, which the cyclic double-coverage assignment of §1 guarantees (every rater pair shares one or two items); the coverage printout that documents it is kept with the forms' manifest.
- **Strict, lenient and doublet/homophone-tolerant** scores are all reported (DD 10.2): strict = `correct`; lenient = `correct_lenient` (the gold syllables in either order); tolerant = strict or `error_class` ∈ {`homophone`, `doublet`}.
- **Exclusions (pre-registered, §5.7):** a form with fewer than 15 of 30 items answered; a respondent who reports using a dictionary, search engine or AI assistant (`TOOLS` = Có); a failed instruction-check item (`CHECK-01`). Added by the 1 October 2026 amendment (`docs/DEVIATIONS.md`): a respondent who answers `WAS_VALIDATOR` = Có. `score-baseline` applies all four and records one reason per excluded form. All counts reported.
- **What is compared with the models:** every model is scored on exactly the 246 human items (`data/human/human_items.json`); the human row of Table 2 is the mean-human accuracy with n raters and 246 items; at ±4–5 points it is a reference band and no cell-level human comparison is claimed (Limitations). If more volunteers materialize, items get a third rater before the item set widens.

## 5. Logistics

- 20 forms, links sent individually so that each respondent gets a different form; one form per person. Validators never receive a link.
- Recruitment: personal invitation from the author's own network (`docs/gate1/RECRUITMENT.md`, message (b)); adults only; no dependency relation; no payment; acknowledgement offered.
- Time: about 30–45 minutes in all. DD 10.2 planned about 20 minutes for the 30 items alone; the forms now add the natural block, repeat the full model prompt on every item and add the check and closing rows, so the planning figure is 30–45 minutes. Completion time is not measured (Google Forms records only the submission time).
- Data handling: responses downloaded once as CSV (no linked response Sheet), no e-mail collection, stored under `data/human/` (git-ignored), never sent to any AI service, released in aggregate with the paper; the 20 Google Forms are deleted once the import has been checked; the form-number-to-person mapping is held separately and deleted after publication; `returned/` and `report/` are kept privately until the retention date of `docs/CONSENT_FORM.md` §5 and never released (`docs/gate1/HUMAN_BASELINE_PROTOCOL.md` §4 step 11, §7).
