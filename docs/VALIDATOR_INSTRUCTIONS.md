# Validator instructions / Hướng dẫn cho người kiểm định (bilingual)

> **[NATIVE-CHECK]** The Vietnamese text must be read by a native speaker before it is sent: the terminology (*phụ âm đầu, vần, thanh điệu, âm đệm, âm chính, âm cuối*), the question wordings, and every example. The English is the reference for meaning.
>
> Protocol: `docs/DESIGN_DECISIONS.md` §10.1. These instructions accompany the spreadsheet produced by `scripts/make_validation_forms.py sample` (columns `item_id, task, variant, input, candidate, question_vi, correct, spelling, lexical, offensive, comment`) **plus** the three supplementary sheets described in §9 below (T2 readings, the 20-pair variant-production item, the 20-pair *qu-* check), which the script does not yet produce. They are released with the data (checklist item D1).

---

## Tiếng Việt

### 1. Việc bạn làm

Bạn nhận một bảng tính (Google Sheets hoặc CSV) gồm khoảng **700 mục** được chọn ngẫu nhiên theo tầng từ bộ dữ liệu (nên mọi loại mục đều có mặt theo tỉ lệ đã định). Mỗi mục là một câu nói lái do chương trình sinh ra, kèm câu trả lời mà chương trình cho là đúng. Với **mỗi mục**, bạn điền **bốn cột** bằng một trong ba giá trị: **Có**, **Không**, **Không chắc** (viết `yes` / `no` / `unsure` cũng được). Cột `comment` để ghi chú tự do (không bắt buộc). Ngoài ra có ba bảng phụ ngắn (mục 9).

Xin **không** dùng trợ lý AI, và **không** bàn về các mục với hai người kiểm định khác trước khi nộp; từ điển được phép dùng cho câu hỏi "có phải là từ thật không", nhưng câu hỏi đó hỏi cảm nhận của bạn là người bản ngữ, không hỏi từ điển. Làm theo từng buổi ngắn (tối đa khoảng một giờ).

### 2. Nhắc lại: cấu tạo âm tiết và các kiểu nói lái

Mỗi âm tiết tiếng Việt gồm **phụ âm đầu** (có thể không có), **vần** (âm đệm nếu có + âm chính + âm cuối nếu có) và **thanh điệu** (ngang, huyền, sắc, hỏi, ngã, nặng). Ví dụ: *mèo* = m + eo + huyền; *cái* = c + ai + sắc.

Truyền thống kể **sáu** kiểu nói lái; bộ dữ liệu sinh mục cho bốn kiểu đầu và dùng hai kiểu còn lại khi chấm câu giải mã và khi tạo câu sai:

| Kiểu | Đổi giữa hai âm tiết | Giữ nguyên tại chỗ | Ví dụ |
|---|---|---|---|
| **V1** | vần | phụ âm đầu, thanh điệu | *mèo cái → mài kéo* |
| **V2** | cả phụ âm đầu và vần (đổi chỗ hai âm tiết) | thanh điệu | *đầu tiên → tiền đâu* |
| **V3** | thanh điệu | phụ âm đầu, vần | *bí mật → bị mất* |
| **V4** | vần và thanh điệu | phụ âm đầu | *bí mật → bật mí* |
| V5 | phụ âm đầu | vần, thanh điệu | *giải pháp → phải giáp* |
| V6 | phụ âm đầu và thanh điệu | vần | (= V1 đọc theo thứ tự ngược) |

Các mã kiểu **không gắn với vùng miền** (tài liệu mâu thuẫn nhau về "lái Bắc", "lái Nam"); vì vậy chúng tôi hỏi bạn ở mục 9 xem bạn tự nhiên nói lái theo kiểu nào. Khi ghép lại, chính tả phải đúng quy tắc: /k/ viết **k** trước e, ê, i; **q** trước âm đệm u; **c** ở các trường hợp khác (*c + eo → keo*, không phải *ceo*); **gh**, **ngh** trước e, ê, i; dấu thanh đặt trên âm chính (cả *hoà* và *hòa* đều được).

### 3. Ba loại mục

- **T1 (biến đổi).** *Nói lái kiểu V1 của "mèo cái" là "mài kéo"?* — `candidate` là câu trả lời của chương trình.
- **T2 (giải mã).** *"mài kéo" là cách nói lái của "mèo cái"?* — `candidate` là cụm từ gốc mà chương trình cho là đúng. Một câu nói lái có thể có nhiều cụm gốc hợp lệ; ở bảng chính bạn chỉ xét cụm được hiển thị; bảng phụ T2 (mục 9) hỏi bạn chấp nhận những cụm gốc nào.
- **T3 (phán đoán).** *"mài céo" có phải là nói lái kiểu V1 của "mèo cái" không? (đáp án hệ thống: no)* — chương trình đưa ra một **phán đoán** (yes/no); bạn xét xem phán đoán đó có đúng không.

### 4. Bốn câu hỏi và cách trả lời

| Cột | Câu hỏi | Trả lời **Có** khi | Trả lời **Không** khi |
|---|---|---|---|
| `correct` | Chương trình có **đúng** không? | T1/T2: `candidate` đúng là kết quả (hoặc cụm gốc) theo kiểu đã nêu. T3: phán đoán yes/no của hệ thống đúng. | Sai vần, sai thanh, sai phụ âm đầu, sai thứ tự, hoặc đúng theo một kiểu khác. |
| `spelling` | `candidate` có **đúng chính tả** không? | Đúng quy tắc c/k/q, g/gh, ng/ngh, i/y; dấu thanh đặt hợp lệ (*hoà/hòa*, *lí/lý* đều được). | Có lỗi chính tả (ví dụ *mài céo*). Ở mục T3 có `candidate` cố ý sai chính tả: `spelling` = Không và `correct` = Có (vì hệ thống nói "no" là đúng). |
| `lexical` | **Cả đầu vào và** `candidate` có phải **từ/cụm từ thật** trong tiếng Việt không? Ghi hai giá trị, ví dụ `Có/Không` (đầu vào/kết quả). | Là từ hoặc cụm từ có nghĩa mà người Việt dùng (kể cả phương ngữ, tiếng lóng), theo cảm nhận của bạn, không cần có trong từ điển. | Chỉ là chuỗi âm tiết hợp lệ nhưng không có nghĩa (*trinh bùng*). Âm tiết đứng riêng có nghĩa nhưng cụm không có nghĩa → Không. |
| `offensive` | Mục này (đầu vào **hoặc** `candidate`) có **cách hiểu thô tục, xúc phạm** không? | Có nghĩa thô tục, tục tĩu, xúc phạm, kể cả khi phải "nói lái ngược" mới thấy. | Không có cách hiểu nào như vậy. |

**Không chắc** dùng khi bạn thật sự không quyết định được (ví dụ một từ phương ngữ bạn không rõ). Xin ghi lý do vào `comment`.

### 5. Ví dụ đã làm sẵn

| Mục | `correct` | `spelling` | `lexical` (vào/ra) | `offensive` | Ghi chú |
|---|---|---|---|---|---|
| T1 V1: *mèo cái → mài kéo* | Có | Có | Có/Có | Không | c → k trước e: đúng |
| T1 V1: *mèo cái → mài céo* | Không | Không | Có/Không | Không | sai chính tả |
| T1 V3: *hoạ sĩ → hoã sị* | Có | Có | Có/Không | Không | đúng quy tắc; không phải từ thật; *hõa sị* cũng chấp nhận |
| T1 V4: *thầy giáo → tháo giầy* | Có | Có | Có/Có | Không | dạng dân gian *tháo giày* cũng đúng (ây/ay); ghi vào `comment` |
| T2: *tiền đâu → đầu tiên* | Có | Có | Có/Có | Không | |
| T3 V1: *mèo cái* / "mài kéo" (hệ thống: yes) | Có | Có | Có/Có | Không | |
| T3 V1: *mèo cái* / "mái kèo" (hệ thống: no) | Có | Có | Có/Có | Không | *mái kèo* là kết quả V4, không phải V1; là từ thật |
| T3 V1: *mèo cái* / "mài céo" (hệ thống: no) | Có | Không | Có/Không | Không | hệ thống đúng khi nói "no" |
| bất kỳ: *mộng mơ → mờ mông* | (theo quy tắc) | Có | Có/Có | **Có** | có cách hiểu thô tục → đánh dấu |

### 6. Khi nghi ngờ có lỗi hệ thống

Nếu bạn thấy **nhiều mục cùng một kiểu sai** (ví dụ mọi mục có *gi* đều sai), hãy ghi `RULE?` vào `comment` của một mục bất kỳ và mô tả ngắn. Đó là thông tin quý nhất bạn có thể cho chúng tôi: một lỗi quy tắc được xác nhận sẽ được sửa và toàn bộ bộ dữ liệu được sinh lại **trước khi** bất kỳ mô hình nào được chạy.

### 7. Thời gian

Khoảng **30–40 giây một mục**; 700 mục ≈ **6–8 giờ**, cộng khoảng một giờ cho ba bảng phụ, chia thành nhiều buổi trong ba tuần (19/10–8/11/2026). Không cần làm hết một lần; xin nộp phần đã làm mỗi tuần.

### 8. Xử lý bất đồng và cách chúng tôi báo cáo mức đồng thuận

Mỗi mục được hai người kiểm định xét; 200 mục được cả ba người xét để đo mức đồng thuận. Chúng tôi báo cáo hệ số Krippendorff's α **cùng với** tỉ lệ đồng ý thô, phân bố các câu trả lời và hệ số Gwet's AC1 (vì khi hầu hết mục đều đúng, α thấp dù mọi người gần như luôn đồng ý). Nếu hai người bất đồng ở cột `correct`, mục được chuyển cho người thứ ba; nếu vẫn chưa thống nhất, tác giả quyết định và **ghi lý do** vào nhật ký được công bố. Bất đồng theo hệ thống (một loại mục mà mọi người đều bác) dẫn tới rà soát bảng quy tắc. Không ai bị coi là "sai".

### 9. Ba bảng phụ

- **T2 – các cụm gốc chấp nhận được** (mọi mục T2 trong tập lõi và mọi ví dụ dân gian): với mỗi câu nói lái, chương trình liệt kê các cụm gốc nó cho là hợp lệ; bạn đánh dấu từng cụm là *chấp nhận* / *không chấp nhận*, và **thêm** cụm gốc nào bạn nghĩ ra mà chưa có trong danh sách.
- **"Bạn nói lái kiểu nào?"** (20 cặp từ): với mỗi cặp, bạn viết câu nói lái bạn **tự nhiên** nghĩ ra trước tiên, không nhìn bảng kiểu. Bảng này cho biết kiểu nào là kiểu tự phát ở vùng của bạn; xin ghi vùng bạn lớn lên (Bắc/Trung/Nam) ở đầu bảng.
- **Kiểm tra quy ước** (20 cặp có *qu-*, một số mục về vị trí dấu và *i/y*): bạn cho biết kết quả nói lái nào đúng theo cảm nhận của bạn (ví dụ *quả hồng* → *cổng hoà* hay *quồng hả*?) và cách viết nào bạn thấy tự nhiên (*hoà* hay *hòa*; *lí* hay *lý*).

---

## English

### 1. Your task

You receive a spreadsheet (Google Sheets or CSV) of about **700 items**, drawn as a stratified random sample of the dataset (every kind of item is present in a known proportion). Each item is a nói lái produced by the program together with the answer the program believes is correct. For **every item** you fill **four columns** with one of three values: **Yes**, **No**, **Unsure** (`yes` / `no` / `unsure`; the Vietnamese `Có` / `Không` are accepted). The `comment` column is free text (optional). There are also three short supplementary sheets (§9).

Please do **not** use an AI assistant, and do **not** discuss items with the other two validators before you submit. A dictionary is allowed for the "real word" question, but that question asks for your judgment as a native speaker, not for dictionary membership. Work in short sessions (about an hour at most).

### 2. Reminder: syllable structure and the kinds of nói lái

A Vietnamese syllable is an **onset** (initial consonant, possibly absent), a **rime** (glide if any + main vowel + final if any) and a **tone** (ngang, huyền, sắc, hỏi, ngã, nặng). *mèo* = m + eo + huyền; *cái* = c + ai + sắc.

The tradition counts **six** kinds; the dataset generates items for the first four and uses the other two when scoring decoding answers and when building wrong candidates:

| Kind | Swapped between the two syllables | Kept in place | Example |
|---|---|---|---|
| **V1** | rimes | onsets, tones | *mèo cái → mài kéo* |
| **V2** | onset + rime (the syllables trade places) | tones | *đầu tiên → tiền đâu* |
| **V3** | tones | onsets, rimes | *bí mật → bị mất* |
| **V4** | rimes and tones | onsets | *bí mật → bật mí* |
| V5 | onsets | rimes, tones | *giải pháp → phải giáp* |
| V6 | onsets and tones | rimes | (= V1 read in the other order) |

The kind codes carry **no regional label** (sources contradict one another about "Northern" and "Southern" nói lái); §9 therefore asks which kind you produce spontaneously. Reassembled syllables must follow the spelling rules: /k/ is **k** before e, ê, i; **q** before the glide u; **c** elsewhere (*c + eo → keo*, not *ceo*); **gh**, **ngh** before e, ê, i; the tone mark sits on the main vowel (both *hoà* and *hòa* are fine).

### 3. Three item types

- **T1 (transformation).** *The V1 nói lái of "mèo cái" is "mài kéo"?* — `candidate` is the program's output.
- **T2 (decoding).** *"mài kéo" is the nói lái of "mèo cái"?* — `candidate` is the original phrase the program proposes. A nói lái may have several valid originals; in the main sheet judge only the one shown; the T2 supplementary sheet (§9) asks which originals you accept.
- **T3 (validity).** *Is "mài céo" the V1 nói lái of "mèo cái"? (system answer: no)* — the program has made a **yes/no verdict**; you judge whether the verdict is right.

### 4. The four questions

| Column | Question | Answer **Yes** when | Answer **No** when |
|---|---|---|---|
| `correct` | Is the program **right**? | T1/T2: `candidate` is the correct output (or original) under the named kind. T3: the system's yes/no verdict is right. | Wrong rime, tone, onset or order, or correct under a different kind. |
| `spelling` | Is `candidate` **spelled correctly**? | Standard spelling (c/k/q, g/gh, ng/ngh, i/y); a valid tone-mark position (*hoà/hòa*, *lí/lý* all fine). | A spelling error (e.g. *mài céo*). For a T3 item whose `candidate` is deliberately misspelled, `spelling` = No and `correct` = Yes (the system rightly says "no"). |
| `lexical` | Are **the input and** `candidate` **real words or phrases**? Give two values, e.g. `Yes/No` (input/candidate). | A meaningful word or phrase that Vietnamese speakers use (dialect and slang included), in your judgment; dictionary membership is not required. | A legal but meaningless string of syllables (*trinh bùng*). Meaningful syllables that do not form a phrase → No. |
| `offensive` | Does the item (input **or** `candidate`) have a **vulgar or offensive reading**? | Any obscene, vulgar or insulting reading, including one reached by decoding the nói lái. | No such reading. |

Use **Unsure** only when you genuinely cannot decide (a dialect word you do not know, say) and give the reason in `comment`.

### 5. Worked examples

| Item | `correct` | `spelling` | `lexical` (in/out) | `offensive` | Note |
|---|---|---|---|---|---|
| T1 V1: *mèo cái → mài kéo* | Yes | Yes | Yes/Yes | No | c → k before e: right |
| T1 V1: *mèo cái → mài céo* | No | No | Yes/No | No | misspelled |
| T1 V3: *hoạ sĩ → hoã sị* | Yes | Yes | Yes/No | No | rule-correct; not a word; *hõa sị* equally acceptable |
| T1 V4: *thầy giáo → tháo giầy* | Yes | Yes | Yes/Yes | No | the folk form *tháo giày* (ay for ây) is also right; note it in `comment` |
| T2: *tiền đâu → đầu tiên* | Yes | Yes | Yes/Yes | No | |
| T3 V1: *mèo cái* / "mài kéo" (system: yes) | Yes | Yes | Yes/Yes | No | |
| T3 V1: *mèo cái* / "mái kèo" (system: no) | Yes | Yes | Yes/Yes | No | *mái kèo* is the V4 output, not V1; it is a real word |
| T3 V1: *mèo cái* / "mài céo" (system: no) | Yes | No | Yes/No | No | the system is right to say "no" |
| any: *mộng mơ → mờ mông* | (by rule) | Yes | Yes/Yes | **Yes** | has a vulgar reading → flag it |

### 6. If you suspect a systematic error

If **many items fail in the same way** (every item with *gi*, say), write `RULE?` in the `comment` of any one of them with a short description. This is the most valuable thing you can give us: a confirmed rule bug is fixed and the whole dataset is regenerated **before** any model is run.

### 7. Time

About **30–40 seconds per item**; 700 items ≈ **6–8 hours**, plus about an hour for the three supplementary sheets, in sessions over three weeks (19 October – 8 November 2026). You need not finish in one go; please return what you have done each week.

### 8. How disagreements are resolved, and how agreement is reported

Every item is judged by two validators; 200 items are judged by all three so that agreement can be measured. We report Krippendorff's α **together with** raw agreement, the distribution of answers and Gwet's AC1 (when almost every item is correct, α is low even though raters almost always agree). If the two validators disagree on `correct`, the item goes to the third; if disagreement remains, the author adjudicates and **logs the reason** in a published adjudication file. Systematic disagreement (a class of items everyone rejects) triggers a review of the rule tables. Nobody is "wrong".

### 9. Three supplementary sheets

- **T2 – acceptable originals** (every core T2 item and every attested example): for each nói lái the program lists the originals it considers valid; mark each *accept* / *reject*, and **add** any original you can think of that is missing.
- **"Which kind do you produce?"** (20 word pairs): for each pair write the nói lái you **spontaneously** think of first, without looking at the table of kinds. This tells us which kind is the spontaneous one in your region; give the region you grew up in (North/Centre/South) at the top of the sheet.
- **Convention checks** (20 pairs containing *qu-*, plus a few items on tone-mark position and *i/y*): say which nói lái result feels right to you (e.g. *quả hồng* → *cổng hoà* or *quồng hả*?) and which spelling feels natural (*hoà* or *hòa*; *lí* or *lý*).

---

### Notes for the author (not sent to validators)

- The sample: `make validation` (1,000 items, 200 overlap, validators A B C) is a stratified sample; the stratum weights needed for the per-cell generator-precision estimate (DD §10.1) must be written to `validation_manifest.json` (owner: `scripts/make_validation_forms.py`). Each validator sees (200 + 800 × 2 / 3) ≈ 733 items.
- The three supplementary sheets and the two-valued `lexical` column are **not** produced by the current script; they are listed in the session report as open items for its owner.
- Scoring: `scripts/make_validation_forms.py score --out data/validation --returned 'data/validation/returned/*.csv'` → `validation_report.json` with α and its bootstrap CI per question, percent agreement, and generator precision (majority vote on `correct`); raw agreement, marginals and Gwet's AC1 (and MASI/Jaccard α for the T2 sets) are to be added to `noilai.stats.agreement` (open item).
- Adjudication log: `data/validation/adjudication.tsv` (`item_id, judgments, decision, reason, date`), created at the first disagreement; released with the data.
- Demographics recorded by the form (consent form §5): age band 18–29 / 30–49 / 50+, region of upbringing, years in Vietnam, raised abroad; the letter-to-person key is held separately and deleted after publication.
- The "system answer" shown for T3 items is the gold; a validator answering `correct = No` on a T3 item is saying the *gold* is wrong. Make sure the Google Form wording preserves this.
