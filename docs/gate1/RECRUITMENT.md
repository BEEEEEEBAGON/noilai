# Recruitment messages: validators and human-baseline respondents (Gate 1)

Status: draft, 1 October 2026. Two separate messages, each in Vietnamese and English, sent to two separate groups of
people: (a) 2–3 **validators** for the Gate 1 packet (`docs/gate1/VALIDATION_PROTOCOL.md`), (b) about 20
**human-baseline respondents** for the baseline forms (`make baseline`), who are never the same people as the
validators. Consent text: `docs/CONSENT_FORM.md`. Timeline: recruit by 12 October 2026; calibration 13–15 October;
Gate 1 on 18 October; Parts B–E 19 October – 1 November; human baseline 2–8 November; results freeze 22 November.

- **[NATIVE-CHECK]** Every Vietnamese passage is checked by a native speaker before it is sent. The English text is the
  reference for meaning. Remove every `[NATIVE-CHECK]` marker before sending.
- The Vietnamese messages use *tôi* / *bạn*. Change the address terms to fit each recipient (*em*, *anh*, *chị*, *cô*,
  *chú*, ...); the content stays the same.
- Replace `[NAME]` (and `[anh/chị/bạn]`) before sending. Replace `[ADULT NAME]` and `[ADULT EMAIL]` with the adult
  second contact of `docs/CONSENT_FORM.md` §9; that line is required if the author is under 18, otherwise delete it.
  Each message is under 300 tokens per language (counts at the end of this file).

## 1. How to send

[NATIVE-CHECK] the Vietnamese phrases quoted in this section ("Có", "Tôi đã đọc phiếu thông tin ...") match the form.

1. **Personal invitations only.** Send each message as a direct message or an e-mail from the author's own account, to
   a person the sender knows to be an adult. Do not post it in public: no social-media posts, group chats, forums or
   pages. Public posts can reach minors and people whose age and relation to the author cannot be checked.
2. **Accounts.** Messages, the Google Sheets workbooks and the Google Forms (`data/human/build_forms.gs`, run in
   script.google.com) use the author's own existing accounts or an adult collaborator's. No account is ever opened with
   a misstated age; if a service requires an age the author does not meet, an adult collaborator does that step.
3. **Who not to invite.** Nobody under 18. Nobody in a dependency relation with the author (no one the author grades,
   supervises or employs). Keep the age and dependency lines in every message, even to people the author knows well.
4. **Two groups, never mixed.** Message (a) goes to validators, message (b) to different people. Never ask a validator
   to also take the baseline form, and never send a validator a baseline link. The baseline form asks the question
   (`WAS_VALIDATOR`), and a form answered "Có" there is excluded from scoring.
5. **Validators' regions.** Aim for one validator who grew up in the North, one in the Centre, one in the South. When a
   validator agrees, give them a letter (A, B, C) and record the letter's region in `data/validation/validators.json`
   (git-ignored), e.g. `{"A": {"region": "N"}}`.
6. **Private list, kept off the repository.** Record who was invited, how to contact them, who agreed, the letter or form
   number given, and the date the consent came back, in a private note or sheet in the author's own account, outside the
   repository folder (`docs/CONSENT_FORM.md` §5 (c) describes it to participants).
   It is never committed and never pasted into an AI tool. It is the letter/form-to-person key: keep it separate from the
   answers and delete it after publication. Run every command that reads `data/validation/` or `data/human/` in a plain
   terminal, never inside an AI coding session. Never paste its output, `adjudication.tsv` or any report into a chat
   assistant; only aggregates, copied into `docs/RESULTS_LOG.md` by hand, leave those folders.
7. **After a yes.** Validators: send `docs/CONSENT_FORM.md`, the information and consent form (with `[NAME]`, `[EMAIL]`,
   the §5 retention date and, if needed, the §9 second contact filled in and the markers removed), as a PDF and keep
   the returned copy with the private list, not with the answers; then share their workbook.
   Baseline respondents: send the same consent form as a PDF together with the link, because the required consent boxes
   at the start of the Google Form refer to it ("Tôi đã đọc phiếu thông tin ..."); send each person the link of one form
   only (one form number per person) and ask them not to forward it.
8. **Acknowledgement.** Offer it, default anonymous. Validators tick the optional box on the consent form. Baseline
   forms carry no name, so a respondent who wants to be thanked by name says so in a separate message, which goes into
   the private list.

## 2. Message (a): validators

### 2.1 Tiếng Việt [NATIVE-CHECK]

Chào [anh/chị/bạn],

Tôi là [NAME]. Tôi đang viết một bài báo khoa học: các mô hình ngôn ngữ lớn (loại AI đứng sau chatbot) có nói lái được
không? Chương trình của tôi sinh câu nói lái theo quy tắc chính tả; tôi cần người bản ngữ kiểm tra.

Tôi tìm 2–3 người kiểm định, tốt nhất mỗi miền Bắc, Trung, Nam một người. Trên Google Sheets, bạn đánh
dấu từng hàng Có / Không / Không chắc: câu nói lái đúng không, đúng chính tả không, có thô tục không. Mất khoảng
4–5,5 giờ, cộng thời gian đọc hướng dẫn và 1–1,5 giờ tuỳ chọn, chia nhiều buổi, 13/10–1/11/2026 (nếu có ba người,
có thể thêm vài hàng trong 2–5/11). Xin đừng dùng trợ lý AI hay bàn với người kiểm định khác. [NATIVE-CHECK]

Đây là việc tình nguyện, không thù lao, chỉ dành cho người từ 18 tuổi trở lên, không phụ thuộc vào tôi (tôi không chấm
bài, quản lý hay trả lương cho bạn). Bạn có thể dừng bất cứ lúc nào. Bài làm không ghi tên, chỉ ghi chữ cái A, B hoặc
C. Có vài câu thô tục (một câu ở vòng hiệu chỉnh được cố ý đưa vào, vài câu sưu tầm, và có thể vài câu do chương trình
sinh ra); bạn có thể bỏ qua. Nếu muốn, bạn sẽ được nêu tên trong lời cảm ơn của bài báo. [NATIVE-CHECK]

Nếu đồng ý, xin trả lời trước 12/10 và cho biết bạn lớn lên ở miền nào. Tôi sẽ gửi phiếu thông tin và đồng ý cùng
hướng dẫn. Xin đừng chuyển tiếp tin nhắn này; nếu bạn biết ai phù hợp, hãy cho tôi biết để tôi liên hệ riêng. Người
liên hệ thứ hai: [ADULT NAME], [ADULT EMAIL]. [NATIVE-CHECK]

### 2.2 English

Hello [name],

I am [NAME]. I am writing a research paper on whether large language models (the AI behind chatbots) can do nói lái.
A program I wrote generates nói lái from Vietnamese spelling rules; I need native speakers to check its answers.

I am looking for 2–3 validators, ideally one who grew up in each of the North, the Centre and the South. In a Google
Sheets workbook you mark each row Có / Không / Không chắc (yes / no / unsure): is the nói lái correct, is it spelled
correctly, is it vulgar. Time: about 4–5.5 hours, plus time to read the instructions and an optional 1–1.5 hours, in
sessions between 13 October and 1 November 2026 (with three validators, a few more rows may follow on 2–5 November).
Please do not use an AI assistant or discuss the items with other validators.

This is unpaid volunteer work, open only to adults (18 or older) who do not depend on me (I do not grade, supervise or
employ you). You can stop at any time. Your answers carry only a letter (A, B or C), no name. A few items are vulgar
(one calibration row on purpose, some collected items, and possibly some generated ones); you may skip them. If you
wish, I will thank you by name in the paper.

If you agree, reply by 12 October and tell me which region you grew up in. I will send the information and consent
form and the instructions. Please do not forward this message; if you know someone suitable, tell me and I will contact
them myself. Second contact: [ADULT NAME], [ADULT EMAIL].

## 3. Message (b): human-baseline respondents

### 3.1 Tiếng Việt [NATIVE-CHECK]

Chào [anh/chị/bạn],

Tôi là [NAME]. Tôi đang viết một bài báo khoa học: các mô hình ngôn ngữ lớn (loại AI đứng sau chatbot) có nói lái được
không? Để so sánh, tôi cần biết người bản ngữ làm những câu hỏi đó ra sao.

Tôi tìm khoảng 20 người, mỗi người làm một phiếu Google Form, khoảng 30–45 phút, trong 2–8/11/2026.
Phiếu có 30 câu nói lái (tự nói lái, giải, hoặc phán đoán đúng/sai) kèm đúng hướng dẫn và ví dụ mà mô hình nhận được,
thêm 10 câu nói lái quen thuộc để giải. Xin làm một mình, không dùng từ điển, công cụ tìm kiếm hay trợ lý AI. Các câu
đã biết là thô tục đã được loại khỏi phiếu; nếu vẫn gặp câu làm bạn khó chịu, hãy bỏ qua. [NATIVE-CHECK]

Đây là việc tình nguyện, không thù lao, chỉ dành cho người từ 18 tuổi trở lên, không phụ thuộc vào tôi (tôi không chấm
bài, quản lý hay trả lương cho bạn) và không phải người kiểm định của nghiên cứu này. Bạn có thể dừng bất cứ lúc nào.
Phiếu không hỏi tên hay e-mail, chỉ có mã số. Nếu muốn được nêu tên trong lời cảm ơn của bài báo, hãy nhắn riêng cho
tôi. [NATIVE-CHECK]

Nếu đồng ý, xin trả lời trước 12/10. Đầu tháng 11 tôi sẽ gửi riêng link phiếu cho bạn; xin đừng chuyển cho người
khác. Xin cũng đừng chuyển tiếp tin nhắn này; nếu bạn biết ai phù hợp, hãy cho tôi biết để tôi liên hệ riêng. Người
liên hệ thứ hai: [ADULT NAME], [ADULT EMAIL]. [NATIVE-CHECK]

### 3.2 English

Hello [name],

I am [NAME]. I am writing a research paper on whether large language models (the AI behind chatbots) can do nói lái.
To compare them with people, I need to know how native speakers do on the same questions.

I am looking for about 20 people to fill in one Google Form each, once, in about 30–45 minutes, between 2 and
8 November 2026. The form has 30 nói lái questions (produce, decode or judge one) with the same instructions and
examples the models receive, plus 10 familiar nói lái to decode. Please work alone, without a dictionary, a search
engine or an AI assistant. Items known to be vulgar have been removed from the form; if one still bothers you, skip it.

This is unpaid volunteer work, open only to adults (18 or older) who do not depend on me (I do not grade, supervise or
employ you) and who are not validators in this study. You can stop at any time. The form asks for no name or e-mail;
it has only a form number. If you would like to be thanked by name in the paper, tell me in a separate message.

If you agree, reply by 12 October. In early November I will send you your own link to the form; please do not forward
it. Please do not forward this message either; if you know someone suitable, tell me and I will contact them myself.
Second contact: [ADULT NAME], [ADULT EMAIL].

## 4. Answers to likely questions

Send these with the message, or use them to answer replies. They apply to both roles unless a line says otherwise.

### 4.1 Tiếng Việt [NATIVE-CHECK]

**Vì sao không trả tiền?** Dự án không có kinh phí: không có tài trợ, không có tổ chức nào chi trả, nên tôi không thể
trả thù lao. Bài báo ghi rõ rằng mọi người tham gia đều là tình nguyện viên không được trả tiền.

**Dữ liệu nào được lưu?** Câu trả lời của bạn và vài thông tin nhân khẩu học khái quát ghi trong phiếu thông tin và
đồng ý (nhóm tuổi, vùng bạn lớn lên, số năm sống ở Việt Nam, bạn có lớn lên ở nước ngoài không). Tên, e-mail, ngày sinh
và địa chỉ không được lưu cùng câu trả lời. Tách riêng khỏi câu trả lời, tôi giữ một danh sách riêng: ai được mời, cách
liên hệ, chữ cái hoặc mã số phiếu đã gán, ngày nhận lại phiếu đồng ý; danh sách này được xoá sau khi bài báo được công
bố. Các bảng trả lời thô và danh sách riêng không nằm trong kho mã nguồn của dự án và không bao giờ được gửi tới dịch vụ
AI nào. Kết quả được công bố ẩn danh. [NATIVE-CHECK]

**Tôi có thể rút lui không?** Có. Bạn có thể dừng bất cứ lúc nào, bỏ qua bất kỳ câu nào, không cần nêu lý do. Nếu muốn
rút phần đã nộp, hãy báo cho tôi tên, chữ cái hoặc mã số phiếu của bạn trước ngày chốt số liệu (22/11/2026). Sau ngày đó các
con số tổng hợp đã công bố không thể thay đổi, nhưng dữ liệu thô của bạn vẫn được xoá khi bạn yêu cầu.

**Vì sao không có hội đồng đạo đức xét duyệt?** Tôi không thuộc một tổ chức có hội đồng đạo đức nghiên cứu, nên không
thể xin xét duyệt; bài báo nói rõ điều này. Rủi ro được đánh giá là tối thiểu: đây là nhận xét ẩn danh về các câu chơi
chữ, chỉ dành cho người từ 18 tuổi trở lên, có phiếu đồng ý, và bạn có thể dừng bất cứ lúc nào.

**Tên tôi có xuất hiện không?** Không xuất hiện cùng câu trả lời. Người kiểm định chỉ được gọi là A, B, C; người làm
phiếu so sánh chỉ có mã số phiếu. Bảng đối chiếu giữa chữ cái hoặc mã số và người được giữ riêng và xoá sau khi bài báo
được công bố. Tên bạn chỉ xuất hiện trong phần Lời cảm ơn, và chỉ khi bạn yêu cầu.

**Mất bao lâu?** Người kiểm định: phần chính (A–D) khoảng 4–5,5 giờ, phần E tuỳ chọn khoảng 1–1,5 giờ, cộng thời gian
đọc hướng dẫn, chia thành nhiều buổi từ 13/10 đến 1/11/2026; nếu có ba người kiểm định, thời gian của mỗi người gần với
mức thấp, và có thể thêm vài hàng cần ý kiến thứ ba trong 2–5/11/2026. Người làm phiếu so sánh: khoảng 30–45 phút, một
lần, trong khoảng 2–8/11/2026. [NATIVE-CHECK]

### 4.2 English

**Why is it unpaid?** The project has no budget: no grant and no institution paying for it, so I cannot pay
participants. The paper states plainly that all participants were unpaid volunteers.

**What data is kept?** Your answers and the few coarse demographic fields listed in the information and consent form
(age band, the region you grew up in, years lived in Vietnam, whether you were raised abroad). Your name, e-mail, date of
birth and address are not stored with your answers. Separately from the answers, I keep a private list: who was
invited, how to contact them, the letter or form number given and the date consent came back; it is deleted after
publication. The raw answer sheets and the private list are not put into the project's code repository and are never
sent to any AI service. Results are published anonymously.

**Can I withdraw?** Yes. You can stop at any time and skip any item, without giving a reason. To withdraw answers you
have already sent, tell me your name, letter or form number before the results freeze (22 November 2026). After that, the
published aggregate numbers cannot change, but your raw data are still deleted on request.

**Why was there no ethics board review?** I do not belong to an institution with a research ethics board, so there is
no board I can apply to; the paper says so. The risk is judged to be minimal: anonymous judgments about wordplay, adults
only, written consent, and you can stop at any time.

**Will my name appear?** Not with your answers. Validators are known only as A, B or C; baseline respondents only by a
form number. The key that links letters and form numbers to people is kept separately and deleted after publication.
Your name appears only in the Acknowledgements, and only if you ask for it.

**How long does it take?** Validators: about 4–5.5 hours for the main parts (A–D) and about 1–1.5 hours for the optional
Part E, plus time to read the instructions, in sessions between 13 October and 1 November 2026; with three validators,
each person is near the lower figure, and a few rows that need a third opinion may follow on 2–5 November 2026.
Baseline respondents: about 30–45 minutes, once, between 2 and 8 November 2026.

## 5. Word counts

Whitespace-separated tokens per message body (greeting to last line, `wc -w`, with the `[NATIVE-CHECK]` markers removed
as they are before sending; Vietnamese counted by syllable, which is stricter than counting words):

| message | Tiếng Việt | English |
|---|---|---|
| (a) validators | 292 | 276 |
| (b) baseline respondents | 265 | 248 |
