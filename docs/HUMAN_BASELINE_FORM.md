# Human-baseline form: design (20 respondents × 30 items, about 20 minutes each)

Purpose: a human level of performance on the same items, scored by the same code, for Table 2 of the paper. Design per `docs/DESIGN_DECISIONS.md` §10.2 (which supersedes the founding plan's "40 items from the core"): **20 adult native speakers, 30 items each; 6 anchor items seen by everyone and 24 items assigned so that every non-anchor item is seen by exactly two people → 6 + 240 = 246 distinct items**, about 20 per task × variant cell, all drawn from the **open-model main sample** so that every model is scored on exactly these 246 items. Volunteers, unpaid; consent: `docs/CONSENT_FORM.md` (role "human-baseline respondent"). Vietnamese wording is marked **[NATIVE-CHECK]**.

## 1. Item sampling and assignment

- **Source:** the open-model main sample (`data/release/<version>/noilai_main.jsonl`, the run plan's seeded 4,200-item stratification of the test split; `v0.2` today, the v0.3 rebuild at the freeze), restricted to items with `vulgar = false` and excluding spelling twins (they leave the T3 headline as well). The file opens with a canary header record, which the snippet skips. If the main sample does not yet exist when forms are built, use the core; record which.
- **Anchors (6):** one T1 item per generated variant (V1–V4), one T2 item, one T3 item; seen by every respondent; they measure between-person agreement on identical material.
- **Double-judged items (240):** 20 per cell (T1 × 4, T2 × 4, T3 × 4 = 12 cells; T3 as yes/no items, half yes and half no, never both members of a twin pair, and never the twin of an anchor); each assigned to exactly two respondents. With 20 respondents × 24 slots = 480 slots = 240 × 2. **Assignment: a cyclic double-coverage design.** Shuffle the 240 items (seeded) and read them as 12 blocks of 20. The first copy of item `k` goes to respondent `k % 20` (round robin); the second copy goes to respondent `(k % 20 + offset) % 20` with a **block-specific offset** `offset = 1 + (k // 20) % 19`, which runs 1, 2, …, 12 over the 12 blocks. Consequences, checked by the snippet's coverage printout and by `tests/test_paper.py`: no respondent sees an item twice; every respondent judges exactly 24 distinct items; and every one of the 190 rater pairs shares at least one non-anchor item (140 pairs share one, 50 pairs share two; 240/190 ≈ 1.3 on average), so the rater graph is complete. A fixed offset (say 10 for every block) would instead produce 10 disjoint dyads each sharing all 24 of their items, with 180 rater pairs sharing nothing; the nominal α over the 240 double-judged items and the two-way (person, item) bootstrap would then rest on a disconnected design. The design is not a balanced incomplete block in the strict sense (pairs share one *or* two items), which is why it is called double-coverage here.
- **Order:** each form's 30 items in a seeded random order (T1, T2, T3 interleaved), one **instruction-check item** inserted (an anchor-style item whose answer is given in the instructions; a wrong answer excludes the form).
- **Coverage check:** the builder prints, per item, the number of respondents (must be 20 for anchors, 2 for the rest) and, per cell, the number of items (20).

The current `scripts/make_validation_forms.py baseline` subcommand produces a different design (cyclic 40-per-form forms without anchors or exact double coverage); it must be extended or replaced before the forms are sent — see the open question in the session report. The assignment above is small enough to build with the snippet below, which writes one CSV per respondent in the same column layout the script uses (`item_id, task, variant, prompt_vi, answer`), so the existing scoring path applies.

```bash
/home/user/venv-noilai/bin/python - <<'EOF'
# [to be moved into scripts/make_validation_forms.py by its owner]
import csv, json, random
from collections import defaultdict
from pathlib import Path
SEED, N_RESP, PER_CELL = 20261102, 20, 20   # a public sampling seed; never the (withheld) build seed, DESIGN_DECISIONS 4.6
rng = random.Random(SEED)
ITEMS = "data/release/v0.2/noilai_main.jsonl"      # the run plan's open-model main sample (v0.3 at the freeze)
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

## 2. Form layout (Google Forms; one page per block; e-mail collection **off**)

1. **Consent page** — the checkboxes of `docs/CONSENT_FORM.md`; the form cannot be submitted with the "18 or older" box unticked.
2. **Demographics (coarse; age band required, the rest optional):** age band (18–29 / 30–49 / 50+); region grown up in (Bắc / Trung / Nam / ngoài Việt Nam); years lived in Vietnam (< 5 / 5–15 / > 15); raised abroad (yes/no). Nothing else.
3. **Instructions page** — **the same p0 prompt text and the same three demonstrations the models receive** (`prompts/noilai.yaml`, `prompts/demos.yaml`, rendered by `noilai.eval.prompts.render` for each task; the T1/T3 instruction names and defines the variant, the T2 instruction withholds it, exactly as for the models). The form therefore changes when the prompts change; the prompt-file hashes are recorded in the form's manifest.
4. **Items, 30 questions in three formats:**
   - T1: *Nói lái kiểu V1 của "mèo cái" là gì?* → short text answer.
   - T2: *"mài kéo" là cách nói lái của cụm từ nào?* → short text answer (any valid original is accepted by the scorer).
   - T3: *"mài céo" có phải là nói lái kiểu V1 của "mèo cái" không?* → Có / Không.
5. **Closing page:** "Did you use a dictionary, a search engine or an AI assistant for any item?" (yes/no; honesty requested; *yes* excludes the form and is counted); free comment; completion time from the form's timestamps.

## 3. Instructions shown to respondents (framing text around the model prompt) **[NATIVE-CHECK]**

**Tiếng Việt.** Bạn sẽ làm 30 câu nói lái trong khoảng 20 phút. Với mỗi câu, bạn đọc phần hướng dẫn (giống hệt phần hướng dẫn mà các mô hình ngôn ngữ nhận được, kể cả ba ví dụ mẫu) rồi viết câu trả lời. Với câu hỏi giải nói lái, kiểu nói lái không được cho biết: hãy tìm cụm từ gốc có nghĩa. Khi viết kết quả, hãy viết đúng chính tả (ví dụ *c* ghép với vần *eo* viết là *keo*). Xin không dùng từ điển, công cụ tìm kiếm hay trợ lý AI; nếu không biết, hãy để trống hoặc đoán. Không có câu trả lời nào được dùng để đánh giá cá nhân bạn. Một câu trong phiếu là câu kiểm tra xem bạn đã đọc hướng dẫn chưa.

**English.** You will do 30 nói lái items in about 20 minutes. For each item you read the instruction (identical to the instruction the language models receive, including its three worked examples) and write your answer. For decoding questions the kind of nói lái is not given: find the meaningful original phrase. Spell the result correctly (*c* + *eo* is written *keo*). Please do not use a dictionary, a search engine or an AI assistant; if you do not know, leave the answer blank or guess. Nothing here evaluates you personally. One item checks that you have read the instructions.

## 4. Scoring

- Export each form to CSV (`item_id, task, variant, prompt_vi, answer`, one row per item, plus the form number). Convert to output rows `{item_id, task, variant, arm: "base", prompt_id: "human-form-<nn>", raw: answer, answer: answer}` and score with `noilai.eval.score.score_outputs(items, outputs)` — the **same function** that scores the models: strict T1 (structures in order; either placement and *lí/lý* accepted; misspellings and regional homophones not repaired, but binned), T2 membership in the gold set, T3 Có/Không against the gold. A blank answer counts as wrong.
- **Aggregates (pre-registered, `docs/PREREGISTRATION.md` §8.11):** item-level human accuracy = mean of the item's two judgments; **mean-human accuracy** per task and variant is the comparator, with a 95% interval from a two-way (person, item) cluster bootstrap; "any human correct" is a ceiling reported separately; results by output lexicality and by region (hỏi/ngã V3 items and -n/-ng items separately for Southern speakers); nominal Krippendorff's α between the two raters over the 240 double-judged items on the produced answer and on correctness; the human error taxonomy from the same scorer. Both α and the two-way bootstrap assume a connected rater–item design, which the cyclic double-coverage assignment of §1 guarantees (every rater pair shares one or two items); the coverage printout that documents it is kept with the forms' manifest.
- **Exclusions (pre-registered, §5.7):** a form with fewer than 15 of 30 items answered; a respondent who reports using a dictionary, search engine or AI assistant; a failed instruction-check item. All counts reported.
- **What is compared with the models:** every model is scored on exactly the 246 human items (`data/human/human_items.json`); the human row of Table 2 is the mean-human accuracy with n raters and 246 items; at ±4–5 points it is a reference band and no cell-level human comparison is claimed (Limitations). If more volunteers materialize, items get a third rater before the item set widens.

## 5. Logistics

- 20 forms, links sent individually so that each respondent gets a different form; one form per person.
- Recruitment: personal invitation from the author's community; adults only; no dependency relation; no payment; acknowledgement offered.
- Time: about 20 minutes; the median completion time is reported next to the accuracy.
- Data handling: responses downloaded once, no e-mail collection, stored under `data/human/` (git-ignored), released in aggregate with the paper; the form-number-to-person mapping is held separately and deleted after publication.
