# Hướng dẫn cho người kiểm định / Validator instructions (Gate 1, bilingual)

> **What this is.** The instructions sent with each validator's workbook `validation_<V>.xlsx` (Parts A–E). Draft of
> 1 October 2026, written for the amended validation. The procedure, sample sizes, scoring and adjudication are in
> `docs/gate1/VALIDATION_PROTOCOL.md`; the sheet columns are `SHEETS` in `scripts/make_validation_forms.py`; the per-row
> question wordings are `question_b` in `noilai/validation.py`. Send the Vietnamese and English halves; never send the
> last section ("Notes for the author").
>
> **[NATIVE-CHECK]** Every Vietnamese sentence must be read by a native speaker before it is sent: the terminology
> (*phụ âm đầu, vần, âm đệm, âm chính, âm cuối, thanh điệu*), the question wordings, the tables and every example. The
> English half is the reference for meaning. Remove every `[NATIVE-CHECK]` marker before sending (check:
> `grep -c "NATIVE-CHECK"` on the sent copy prints 0).
>
> **Examples.** Every nói lái form below was computed by the rule engine (`noilai.gen.variants.apply`, spelled by
> `noilai.validation.spell_pair` in the old tone-mark style the engine prints). The inputs are the three demonstration
> pairs of `prompts/demos.yaml` (*trung bình, làm chủ, vô hình*) and two ordinary phrases, *củ sen* and *xoa dầu*, which
> are not two-syllable entries of Viet74K (`noilai.vi.lexicon.lexical_pairs()`), not calibration phrases and not
> attested rows. The wrong forms come from the helpers that build the calibration and control rows (`_wrong_tone`,
> `_misspell_k`); the new-style form from `noilai.vi.reencode.convert_placement`; the wrong T2 original was checked with
> `noilai.gen.variants.identify_any_order` (no kind, either order). No calibration item and no attested row appears in
> this file. The `lexical_*` and `offensive` cells of the worked examples are the author's judgments, not engine output.

---

## Tiếng Việt

### 1. Việc bạn làm và thời gian [NATIVE-CHECK]

**Việc đầu tiên, trước khi đọc tiếp:** mở trang `D1_production` (20 hàng, khoảng 10 phút) và viết câu nói lái bạn tự
nhiên nghĩ ra trước nhất cho mỗi cụm; không có đáp án đúng hay sai. Mỗi hàng có một cụm ở cột `phrase`; viết câu nói
lái của bạn vào `your_noilai`, như khi nói chuyện. Nếu nghĩ ra hai cách, ghi cách nghĩ ra trước vào `your_noilai` và
cách kia vào `comment`. [NATIVE-CHECK]

Bạn nhận một bảng tính (tệp Excel; tác giả nhập vào Google Sheets cho bạn, các ô trả lời có danh sách chọn). Bảng
gồm năm phần, mỗi phần nằm ở một trang (sheet) riêng:

| Phần | Trang | Nội dung | Số hàng của bạn (khi có 3 người kiểm định / 2 người) | Thời gian ước tính |
|---|---|---|---|---|
| A | `A_calibration` | vòng hiệu chỉnh (mục 3) | 16 / 16 | khoảng 10 phút |
| B | `B_items` | các mục do chương trình sinh ra (mục 4–5) | 292 / 408 | khoảng 2,8 / 4,0 giờ |
| C | `C_attested` | các câu nói lái có nguồn (mục 6) | 119 / 119 | khoảng 50 phút |
| D | `D1_production`, `D2_qu`, `D3_conventions` | ba bảng phụ (mục 7) | 54 / 54 | khoảng 30 phút |
| E | `E_t2gold` | các cụm gốc của mục giải mã (mục 8), **không bắt buộc** | 200 / 275 | khoảng 1,1 / 1,5 giờ |

Tổng cộng khoảng **5,4 giờ** mỗi người nếu có ba người kiểm định, **6,9 giờ** nếu có hai người; riêng phần A–D khoảng
4,3 / 5,4 giờ. Các con số dựa trên mức ước tính 35 giây một hàng ở phần A và B, 25 giây ở phần C, 30 giây ở phần D
và 20 giây ở phần E. Khi gửi bảng, tác giả báo con số đúng cho bảng của bạn.

**Thứ tự ưu tiên** nếu không đủ thời gian: A → B → C → D → E. Phần E không bắt buộc. Bạn không cần làm hết một lần:
hãy chia ra làm nhiều buổi ngắn và **mỗi tuần gửi lại phần đã làm**, kể cả khi chưa xong.

Xin tuân theo các quy tắc sau:

- **Không dùng trợ lý AI** cho bất kỳ hàng nào.
- **Không bàn** về các hàng với những người kiểm định khác trước khi nộp. Nếu hướng dẫn chưa rõ, hãy hỏi tác giả.
- **Không chia sẻ, chụp màn hình hay đăng** bảng tính hoặc bất kỳ hàng nào, và không bàn về chúng với bất kỳ ai (kể cả
  người làm phiếu so sánh) cho đến khi bài báo được công bố, vì bảng có các mục của tập kiểm tra.
- **Từ điển chỉ được dùng** cho câu hỏi "có phải là từ hoặc cụm từ thật không" (cột `lexical_input`,
  `lexical_candidate`), và ngay cả khi đó, điều được hỏi vẫn là cảm nhận của bạn với tư cách người bản ngữ, chứ không
  phải từ đó có trong từ điển hay không. Với các câu hỏi khác, xin đừng tra cứu.
- Trong bảng tính, bạn chỉ được ký hiệu bằng một chữ cái (A, B hoặc C). Tác giả sẽ hỏi riêng bạn bốn câu: nhóm tuổi
  (18–29 / 30–49 / 50 trở lên), vùng bạn lớn lên (Bắc, Trung, Nam hay ngoài Việt Nam), số năm sống ở Việt Nam (dưới 5
  / 5–15 / trên 15), và bạn có lớn lên ở nước ngoài không; các thông tin này không nằm trong bảng tính.
- Nếu có người gửi cho bạn đường link của phiếu so sánh (một phiếu Google Form ngắn của cùng dự án), xin **đừng làm
  phiếu đó**: người kiểm định không làm phiếu so sánh.

**Cách điền.** Ở mỗi ô trả lời, hãy chọn **Có**, **Không** hoặc **Không chắc**. Xin chọn trong danh sách; nếu gõ, chỉ
gõ Có / Không / Không chắc hoặc yes / no / unsure. Chỉ dùng "Không chắc" khi bạn thật sự không quyết định được (ví dụ
một từ địa phương bạn không biết), và ghi lý do vào `comment`. Ô để trống nghĩa là bạn bỏ qua. Trong con số đồng thuận chính, "Không chắc" được tính như ô trống, vì vậy
nếu bạn đã nghiêng về một phía thì hãy chọn phía đó. Bản CSV không có danh sách chọn; bản Google Sheets có.

### 2. Cấu tạo âm tiết và sáu kiểu nói lái [NATIVE-CHECK]

Mỗi âm tiết tiếng Việt gồm **phụ âm đầu** (có thể không có), **vần** (âm đệm nếu có + âm chính + âm cuối nếu có) và
**thanh điệu** (ngang, huyền, sắc, hỏi, ngã, nặng). Ví dụ: *trung* = tr + ung + ngang; *bình* = b + inh + huyền;
*chủ* = ch + u + hỏi; *xoa* = x + oa (âm đệm o, âm chính a) + ngang.

Nói lái đổi chỗ một số thành phần giữa hai âm tiết. Dự án đặt tên sáu kiểu là V1–V6. Các mã này **không gắn với vùng
miền** (các tài liệu nói khác nhau về "lái kiểu Bắc", "lái kiểu Nam"). Mọi kết quả trong bảng do chương trình tính và
in dấu theo kiểu cũ; phần lớn không phải là từ có nghĩa, điều đó là bình thường.

| Kiểu | Đổi giữa hai âm tiết | Giữ nguyên tại chỗ | *trung bình* | *làm chủ* | *vô hình* |
|---|---|---|---|---|---|
| **V1** | vần | phụ âm đầu, thanh | trinh bùng | lù chảm | vinh hồ |
| **V2** | phụ âm đầu và vần (tức là đổi chỗ hai âm tiết) | thanh | binh trùng | chù lảm | hinh vồ |
| **V3** | thanh | phụ âm đầu, vần | trùng binh | lảm chù | vồ hinh |
| **V4** | vần và thanh | phụ âm đầu | trình bung | lủ chàm | vình hô |
| V5 | phụ âm đầu | vần, thanh | bung trình | chàm lủ | hô vình |
| V6 | phụ âm đầu và thanh | vần | bùng trinh | chảm lù | hồ vinh |

Như bảng cho thấy, V6 là kết quả V1 đọc theo thứ tự ngược lại, V5 là V4 đọc ngược và V3 là V2 đọc ngược. Các hàng T1 và
T3 chỉ dùng V1–V4; V5 và V6 chỉ có mặt trong cách phân loại và trong câu hỏi T2 ("một kiểu bất kỳ").

**Chính tả khi ghép lại.** Âm /k/ viết **k** trước e, ê, i, y; viết **q** khi có âm đệm u (*qu*); viết **c** ở các
trường hợp khác (*c* + *en* + hỏi → *kẻn*, không phải *cẻn*). Trước e, ê, i viết **gh** và **ngh** thay cho g và ng.
Theo kiểu mới, dấu thanh luôn đặt trên âm chính (*hoà*); chương trình in theo kiểu cũ, đặt dấu trên âm đệm o hoặc u
khi vần oa, oe, uy không có âm cuối, trừ sau q (*hòa*). Cả hai kiểu đều đúng. Sau phụ âm, viết i hay y (*lí* / *lý*)
đều đúng.

### 3. Phần A: vòng hiệu chỉnh [NATIVE-CHECK]

Phần A có 16 hàng, cùng các cột và cùng cách trả lời như phần B (mục 4). Chương trình tính các hàng này từ những cặp
từ **không có** trong bộ dữ liệu, và có một số hàng cố ý sai. Mục đích là kiểm tra xem hướng dẫn đã rõ chưa trước khi
bắt đầu phần chính; các hàng A không bao giờ được dùng trong bất kỳ con số nào của bài báo.

Xin làm phần A trong các ngày 13–15/10/2026 và gửi lại ngay. Tác giả so câu trả lời của bạn với đáp án và **gửi lại
cho bạn đáp án kèm lời giải thích** cho những hàng bạn trả lời khác. Nếu dưới 80% câu trả lời ở cột `correct` khớp với
đáp án, tác giả sẽ trao đổi ngắn với bạn (gọi điện hoặc nhắn tin) và gửi một bộ hiệu chỉnh thứ hai cùng dạng. **Không ai
bị loại** vì vòng hiệu chỉnh: những chỗ khác đáp án cho tác giả biết phần nào của hướng dẫn cần viết rõ hơn.

### 4. Phần B: các mục do chương trình sinh ra [NATIVE-CHECK]

Mỗi hàng có: `row_id` (B-0001, …), `task` (T1, T2 hoặc T3), `kind` (V1–V4; để trống ở hàng T2), `input`,
`candidate`, `system_verdict` (chỉ có ở hàng T3, ghi `yes` hoặc `no`; câu hỏi tiếng Việt ghi là Có / Không), và câu hỏi
viết sẵn bằng tiếng Việt (`question_vi`) và tiếng Anh (`question_en`).

**Ba loại hàng**, với câu hỏi đúng như trong bảng:

- **T1 (biến đổi).** *Nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “X” là “Y”?* — X là `input`, Y là
  `candidate`. Câu hỏi: Y có đúng là kết quả của kiểu được nêu, theo đúng thứ tự, và viết đúng chính tả không?
- **T2 (giải mã).** *“X” là cách nói lái (một kiểu bất kỳ) của “Y”?* — X (`input`) là câu nói lái, Y (`candidate`) là
  cụm gốc mà chương trình đề xuất. Câu hỏi: X có phải là nói lái của Y theo một kiểu bất kỳ trong sáu kiểu, theo thứ tự
  nào cũng được, không? Một câu nói lái có thể có nhiều cụm gốc; ở đây chỉ xét cụm được hiển thị.
- **T3 (phán đoán).** *“Y” có phải là nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “X” không? Hệ thống trả
  lời: Có.* (hoặc *Không.*) — chương trình đã đưa ra một phán đoán; câu hỏi là **phán đoán đó có đúng không**. Trả lời
  "Có" là đúng khi Y đúng là kết quả của kiểu được nêu, đúng thứ tự, đúng chính tả; trả lời "Không" là đúng trong mọi
  trường hợp còn lại (kiểu khác, sai thanh, sai chính tả, …).

Lời giải thích đi kèm tên kiểu trong câu hỏi: V1 đổi vần, giữ phụ âm đầu và thanh; V2 đổi chỗ hai âm tiết, giữ thanh;
V3 đổi thanh, giữ phụ âm đầu và vần; V4 đổi vần và thanh, giữ phụ âm đầu. Ở V2, "giữ thanh" nghĩa là thanh giữ nguyên
vị trí: hai âm tiết đổi chỗ, còn thanh thứ nhất vẫn ở âm tiết thứ nhất và thanh thứ hai vẫn ở âm tiết thứ hai (*trung
bình* → *binh trùng*).

**Một số hàng cố ý sai** (ứng viên sai, hoặc phán đoán của hệ thống sai) để kiểm tra sự chú ý. Hãy trả lời mỗi hàng
như mọi hàng khác. Nếu bạn bỏ sót nhiều hàng như vậy, tác giả sẽ cùng bạn xem lại hướng dẫn; câu trả lời của bạn không
bị loại.

**Các cột bạn điền** (giá trị: Có / Không / Không chắc, trừ `dialect` và `comment`):

| Cột | Câu hỏi | Trả lời **Có** khi | Trả lời **Không** khi |
|---|---|---|---|
| `correct` | Chương trình có đúng không? | T1: `candidate` là kết quả của kiểu được nêu, đúng thứ tự, đúng chính tả. T2: `input` là nói lái (kiểu bất kỳ, thứ tự nào cũng được) của `candidate`. T3: phán đoán Có/Không của hệ thống đúng. | T1: sai vần, sai thanh, sai phụ âm đầu, sai thứ tự, sai chính tả, hoặc đúng theo một kiểu khác. T2: không kiểu nào biến `candidate` thành `input`. T3: phán đoán của hệ thống sai. |
| `spelling` | `candidate` có viết đúng chính tả không? | Đúng quy tắc c/k/q, g/gh, ng/ngh; dấu đặt kiểu cũ hay kiểu mới đều được; i hay y sau phụ âm đều được. | Có lỗi chính tả (ví dụ *cẻn su*). |
| `lexical_input` | `input` có phải là từ hoặc cụm từ thật không? | Là từ hoặc cụm từ có nghĩa mà người Việt dùng (kể cả từ địa phương, tiếng lóng), theo cảm nhận của bạn; không cần có trong từ điển. | Chỉ là chuỗi âm tiết đọc được nhưng không có nghĩa (*trinh bùng*). Từng âm tiết có nghĩa nhưng ghép lại không thành cụm → Không. |
| `lexical_candidate` | `candidate` có phải là từ hoặc cụm từ thật không? | như trên | như trên |
| `offensive` | `input` hoặc `candidate` có cách hiểu thô tục, tục tĩu hay xúc phạm không? | Có một cách hiểu như vậy, kể cả cách hiểu chỉ hiện ra khi nói lái ngược lại. Chỉ cần chọn Có; xin đừng viết cách hiểu thô tục ra trong `comment`. | Không có cách hiểu nào như vậy. |
| `dialect` | Câu trả lời `correct` của bạn có phụ thuộc vào cách phát âm của vùng bạn không? | Chọn hiện tượng tương ứng (bảng dưới). | Chọn `none` (hoặc để trống). |
| `comment` | Ghi chú tự do (không bắt buộc) | Lý do khi chọn "Không chắc"; `RULE?` khi nghi có lỗi hệ thống (mục 9). | |

**Các quy ước** (vòng hiệu chỉnh kiểm tra phần lớn các điểm này):

- Hai cách đặt dấu (*hòa* / *hoà*) đều đúng. Viết i hay y sau phụ âm (*lí* / *lý*) đều đúng.
- Viết sai c/k, g/gh, ng/ngh làm cho ứng viên T1 **sai**: `correct` = Không và `spelling` = Không.
- Ở hàng T3 có ứng viên sai chính tả và hệ thống trả lời Không: hệ thống đúng, nên `correct` = Có, `spelling` = Không.
- *ay* / *ây* (*dày* / *dầy*) là hai cách viết khác nhau đối với chương trình. Quy tắc làm việc trên chữ viết, nên kết
  quả có thể là dạng *ây* trong khi cách viết quen thuộc là dạng *ay* (hoặc ngược lại). Nếu `candidate` đúng là kết quả
  của quy tắc thì `correct` = Có và `spelling` = Có; nếu bạn viết hay nói theo dạng kia, ghi vào `comment`.
- T1 được xét **theo đúng thứ tự**: kết quả đúng nhưng đọc ngược lại là một kiểu khác, nên `correct` = Không.
- Một kết quả đúng quy tắc nhưng có cách hiểu thô tục: `correct` = Có **và** `offensive` = Có. Hai câu hỏi tách biệt.
- `correct` chấm theo **chữ viết chuẩn**. Nếu giọng vùng bạn không phân biệt hai âm hay hai thanh nào đó (ví dụ hỏi và
  ngã) và điều đó làm bạn phân vân, hãy chọn hiện tượng đó ở cột `dialect`.

**Giá trị của cột `dialect`** (bảng các hiện tượng gộp âm theo vùng của dự án). Cột này không hỏi bạn từ đâu đến; nó
hỏi câu trả lời của bạn **ở hàng này** có dựa vào cách phát âm vùng miền không. Mỗi ô chỉ chọn được một giá trị; nếu
có hơn một hiện tượng, chọn một và ghi các hiện tượng khác vào `comment`.

| Giá trị | Nghĩa | Vùng | Ví dụ |
|---|---|---|---|
| `none` | câu trả lời không phụ thuộc vào giọng vùng | | |
| `d=gi` | d và gi phát âm như nhau | mọi vùng | *dày* / *giày* |
| `d/gi=r` | r phát âm như d / gi | Bắc | *rao* / *dao* / *giao* |
| `ch=tr` | ch và tr phát âm như nhau | Bắc | *tranh* / *chanh* |
| `s=x` | s và x phát âm như nhau | Bắc | *xa* / *sa* |
| `v=d/gi` | v (đầu âm tiết) phát âm như d / gi | Nam, Trung | *vô* / *dô* |
| `final n=ng` | âm cuối -n và -ng như nhau | Nam, Trung | *tan* / *tang* |
| `final t=c` | âm cuối -t và -c như nhau | Nam, Trung | *mắt* / *mắc* |
| `hỏi=ngã` | thanh hỏi và thanh ngã như nhau | Nam, Trung | *củ* / *cũ* |
| `other` | hiện tượng khác; mô tả trong `comment` | | |

### 5. Ví dụ đã làm sẵn cho phần B [NATIVE-CHECK]

Các ô `lexical_*` và `offensive` dưới đây là cảm nhận của tác giả; trong bảng thật, đó là cảm nhận của bạn. Cột
`dialect` của mọi ví dụ là `none`.

| # | Hàng | `correct` | `spelling` | `lexical_input` | `lexical_candidate` | `offensive` | Vì sao |
|---|---|---|---|---|---|---|---|
| 1 | T1, V1: *trung bình* → *trinh bùng* | Có | Có | Có | Không | Không | đúng kết quả V1; không phải từ thật, điều đó bình thường |
| 2 | T1, V1: *trung bình* → *trình bung* | Không | Có | Có | Không | Không | đây là kết quả V4 (đổi cả thanh), không phải V1 |
| 3 | T1, V1: *trung bình* → *bùng trinh* | Không | Có | Có | Không | Không | kết quả V1 đọc ngược lại (tức là V6); T1 xét đúng thứ tự |
| 4 | T1, V3: *làm chủ* → *lạm chù* | Không | Có | Có | Không | Không | sai thanh ở âm tiết đầu; kết quả V3 là *lảm chù* |
| 5 | T1, V1: *củ sen* → *cẻn su* | Không | Không | Có | Không | Không | /k/ trước e phải viết k: kết quả đúng là *kẻn su* |
| 6 | T1, V3: *xoa dầu* → *xoà dâu* | Có | Có | Có | Không | Không | chương trình in *xòa dâu* (dấu kiểu cũ); *xoà dâu* (kiểu mới) cũng đúng |
| 7 | T2: *lù chảm* / cụm gốc *làm chủ* | Có | Có | Không | Có | Không | *lù chảm* là kết quả V1 của *làm chủ* |
| 8 | T2: *trinh bùng* / cụm gốc *trúng bình* | Không | Có | Không | Không | Không | không kiểu nào, theo thứ tự nào, biến *trúng bình* thành *trinh bùng* (cụm gốc đúng là *trung bình*) |
| 9 | T3, V1: *vô hình* / *vinh hồ*; hệ thống trả lời: Có | Có | Có | Có | Không | Không | *vinh hồ* đúng là kết quả V1, nên "Có" là đúng |
| 10 | T3, V1: *làm chủ* / *lủ chàm*; hệ thống trả lời: Không | Có | Có | Có | Không | Không | *lủ chàm* là kết quả V4, không phải V1, nên "Không" là đúng |
| 11 | T3, V2: *trung bình* / *binh trùng*; hệ thống trả lời: Không | Không | Có | Có | Không | Không | *binh trùng* đúng là kết quả V2, nên hệ thống sai |
| 12 | T3, V1: *củ sen* / *cẻn su*; hệ thống trả lời: Không | Có | Không | Có | Không | Không | ứng viên sai chính tả (phải là *kẻn su*), nên "Không" là đúng |

Trong bảng tính, hàng 1 hiện ra với câu hỏi: *Nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “trung bình” là
“trinh bùng”?*; hàng 7: *“lù chảm” là cách nói lái (một kiểu bất kỳ) của “làm chủ”?*; hàng 11: *“binh trùng” có phải là
nói lái kiểu V2 (đổi chỗ hai âm tiết, giữ thanh) của “trung bình” không? Hệ thống trả lời: Không.*

### 6. Phần C: các câu nói lái có nguồn [NATIVE-CHECK]

Phần C có 119 hàng (C-001, …); mọi người kiểm định đều làm toàn bộ. Mỗi hàng gồm `input`, `output` và `source` (nguồn:
sách, báo, trang web, hoặc "folk" là truyền miệng; một số nguồn ghi "[UNCERTAIN]": tác giả chưa kiểm tra lại nguồn
đó). Một phần các hàng lấy từ danh sách có sẵn của dự án, phần còn lại tìm được trên mạng; chưa có hàng nào được người
bản ngữ xác nhận. Một số hàng có ba âm tiết. Hai cụm có thể theo thứ tự bất kỳ (cụm nghĩa gốc
trước hoặc câu nói lái trước). Nguồn chỉ để tham khảo: bạn không cần mở đường dẫn; xin trả lời theo hiểu biết của bạn.

| Cột | Câu hỏi | Ghi chú |
|---|---|---|
| `known` | Bạn đã từng nghe hoặc đọc câu nói lái này chưa? | Có / Không / Không chắc |
| `valid` | Một người Việt có chấp nhận cụm thứ hai là nói lái của cụm thứ nhất không? | Có một số hàng "bẻ" quy tắc một chút để có nghĩa (đổi một nguyên âm hay một thanh), hoặc chỉ đúng theo giọng một vùng. Điều đó không sao: nếu người nói vẫn chấp nhận thì trả lời Có, và ghi hiện tượng vào `dialect` hoặc mô tả trong `comment`. Trả lời Không khi không ai coi cụm này là nói lái của cụm kia. |
| `spelling_ok` | Cả hai cụm có viết đúng chính tả không? | Hai cách đặt dấu và i/y đều đúng như ở phần B. Nếu Không, ghi cách viết đúng vào `comment`. |
| `your_form` | Nếu bạn biết câu này ở một dạng khác, hoặc bạn sẽ nói nó khác đi, hãy viết dạng của bạn. | Để trống nếu dạng trong bảng đúng như bạn biết. |
| `offensive` | như phần B | |
| `dialect` | như phần B: chọn hiện tượng nếu câu nói lái chỉ đúng theo giọng một vùng | |
| `comment` | ghi chú tự do | |

**Lưu ý:** một vài hàng **có nội dung thô tục**, vì đó là những câu chơi chữ tục có chủ ý trong thơ ca dân gian và văn
học (ví dụ thơ Hồ Xuân Hương). Bạn có thể bỏ qua các hàng đó: chỉ chọn Có ở cột `offensive` và để trống các cột còn lại
(nếu muốn, ghi "bỏ qua" vào `comment`). Xin đừng viết cách hiểu thô tục ra trong `comment` hay `your_form`.

### 7. Phần D: ba bảng phụ [NATIVE-CHECK]

- **`D1_production`** (20 hàng; cột `phrase`, `your_noilai`, `comment`): đây là việc đầu tiên ở mục 1, làm trước khi
  đọc các mục khác. Với mỗi cụm, viết vào `your_noilai` câu nói lái mà bạn **tự nhiên nghĩ ra trước nhất**, như khi
  nói chuyện; không có đáp án đúng hay sai. Bảng này cho biết người ở mỗi vùng tự nhiên nói lái theo kiểu nào. Nếu nghĩ
  ra hai cách, ghi cách bạn nghĩ ra trước vào `your_noilai` và cách kia vào `comment`.
- **`D2_qu`** (20 hàng; cột `phrase`, `kind`, `option_A`, `option_B`, `choice`, `your_form`, `comment`). Mỗi cụm có
  một âm tiết bắt đầu bằng *qu-*. Với kiểu ghi ở `kind`, có hai kết quả có thể là `option_A` và `option_B`; chúng khác
  nhau ở chỗ chữ *u* của *qu-* đi theo vần hay ở lại với phụ âm đầu. Ở `choice`, chọn **A**, **B**, **cả hai** hoặc
  **không cái nào** theo cảm nhận của bạn. Ở `your_form`, viết dạng bạn sẽ nói (bắt buộc khi chọn "không cái nào").
- **`D3_conventions`** (14 câu hỏi, chỉ bằng tiếng Việt; cột `topic`, `question_vi`, `options`, `answer`, `comment`).
  Câu hỏi về cách viết mà bạn thấy tự nhiên (vị trí dấu, i/y, ay/ây, một số vần hiếm). Ghi câu trả lời vào `answer`,
  chọn trong `options`; với câu hỏi về nhiều chữ, trả lời cho từng chữ.

### 8. Phần E: các cụm gốc của mục giải mã (không bắt buộc) [NATIVE-CHECK]

Mỗi hàng gồm một câu nói lái (`noilai_form`) và tối đa ba cụm gốc (`reading_1`, `reading_2`, `reading_3`) mà chương
trình tìm thấy trong từ điển. Với mỗi cụm, ở ô `accept_` tương ứng, chọn **Có** nếu cụm đó là từ hoặc cụm từ thật và
`noilai_form` là nói lái của nó (kiểu bất kỳ, thứ tự nào cũng được); chọn **Không** nếu không. Ô `reading` để trống
nghĩa là không có cụm thứ hai hoặc thứ ba. Nếu **ngay lập tức** nghĩ ra một cụm gốc khác mà danh sách còn thiếu, viết
nó vào `missing_reading` (một cụm; nếu nhiều hơn, ghi thêm vào `comment`). Đừng mất công tìm: chỉ thêm khi cụm đó tự
nhiên nảy ra trong đầu. Hầu hết các hàng chỉ do một người làm; 50 hàng do mọi người cùng làm để đo mức đồng thuận.

### 9. Khi nghi có lỗi hệ thống: `RULE?` [NATIVE-CHECK]

Nếu bạn thấy **nhiều hàng sai theo cùng một kiểu** (ví dụ mọi hàng có *gi* đều sai, hoặc mọi hàng có một vần nào đó),
hãy ghi `RULE?` vào `comment` của một hàng như thế, kèm một câu mô tả ngắn. Tác giả đối chiếu từng ghi chú `RULE?` với
bảng quy tắc. Một lỗi được xác nhận sẽ được sửa và toàn bộ dữ liệu được sinh lại **trước khi** bất kỳ mô hình nào được
chạy trên tập kiểm tra. Đây là thông tin quý nhất bạn có thể đóng góp.

### 10. Khi người kiểm định không đồng ý, và cách báo cáo mức đồng thuận [NATIVE-CHECK]

Mỗi hàng phần B do hai người kiểm định làm (nếu có ba người thì 60 hàng do cả ba người làm; nếu chỉ có hai người thì
cả hai làm mọi hàng). Câu trả lời cuối cùng ở cột `correct` được quyết định như sau:

1. Mọi câu trả lời rõ ràng (Có hoặc Không, ít nhất hai câu) giống nhau → lấy câu trả lời đó.
2. Hai người trả lời khác nhau → người kiểm định thứ ba, người chưa thấy hàng này, làm hàng đó mà không biết hai người
   kia trả lời gì; đa số trong ba người quyết định.
3. Vẫn chưa ngã ngũ, hoặc chỉ có hai người kiểm định → tác giả quyết định dựa trên bảng quy tắc và ghi quyết định cùng
   lý do vào nhật ký phân xử; số quyết định của tác giả được báo cáo trong bài báo.
4. Có ít hơn hai câu trả lời rõ ràng (do "Không chắc" hoặc ô trống) → hàng đó chưa có kết luận: nó vẫn ở trong bộ dữ
   liệu nhưng không được tính vào con số độ chính xác của chương trình.
5. Tác giả không bao giờ sửa kết luận mà những người kiểm định đã đồng ý với nhau.

Chỉ cần **một** người chọn `offensive` = Có là mục đó bị đánh dấu thô tục; mục bị đánh dấu không được đưa vào các câu
hỏi gửi cho dịch vụ AI trực tuyến (qua API), cũng không được đưa vào phiếu so sánh. Mọi hiện tượng ở cột `dialect` đều
được ghi lại. Ở
phần C, một hàng được xem là **đã được người bản ngữ xác nhận** khi ít nhất hai người trả lời `valid` = Có và không ai
trả lời Không.

Mức đồng thuận được báo cáo cho từng cột bằng hệ số Krippendorff's α và Gwet's AC1, mỗi hệ số kèm khoảng tin cậy
("Không chắc" tính như ô trống; α được tính thêm một lần coi nó là một lựa chọn riêng), **cùng với** tỉ lệ đồng ý thô
và phân bố các câu trả lời. Khi gần như mọi mục đều đúng, α có thể thấp dù mọi người hầu như luôn đồng ý; vì
thế các con số kia được báo cáo bên cạnh. Tỉ lệ bắt được các hàng cố ý sai cũng được báo cáo cho từng người, chỉ theo
chữ cái. **Không ai bị coi là "sai"**: bất đồng cho biết chỗ nào phụ thuộc vào vùng miền, chỗ nào thật sự mơ hồ, hoặc
chỗ nào hướng dẫn chưa rõ.

### 11. Thời gian và lịch [NATIVE-CHECK]

Khoảng 5,4 giờ mỗi người khi có ba người kiểm định, 6,9 giờ khi có hai người (phần A–D khoảng 4,3 / 5,4 giờ; phần E
khoảng 1,1 / 1,5 giờ), chia thành nhiều buổi tùy bạn.

| Ngày (2026) | Việc |
|---|---|
| chậm nhất ngày 12/10 | bạn đồng ý tham gia và gửi lại phiếu đồng ý đã đánh dấu; tác giả chia sẻ bảng tính với bạn |
| 13–15/10 | phần A (vòng hiệu chỉnh), gửi lại ngay |
| 15–17/10 | tác giả gửi đáp án và lời giải thích; trao đổi thêm nếu cần |
| 18/10 | mốc Gate 1: bắt đầu phần chính |
| 19/10–1/11 | phần B, C, D (và E nếu bạn có thời gian); gửi lại phần đã làm mỗi tuần |
| 2–5/11 | tác giả tính kết quả; nếu có ba người kiểm định, bạn có thể nhận một bảng nhỏ gồm các hàng mà hai người kia trả lời khác nhau |
| 22/11 | chốt kết quả; đến hết ngày này bạn vẫn có thể rút khỏi nghiên cứu và câu trả lời của bạn sẽ không được dùng |

---

## English

### 1. What you do and how long it takes [NATIVE-CHECK]

**First, before reading on:** open `D1_production` (20 rows, about 10 minutes) and write the nói lái that first comes to
mind for each phrase; there is no right or wrong answer. Each row has a phrase in `phrase`; write your nói lái in
`your_noilai`, as you would say it in conversation. If two forms come to mind, put the first in `your_noilai` and the
other in `comment`.

You receive a spreadsheet (an Excel file, which the author imports into Google Sheets for you; the answer cells have
dropdowns). It has five parts, each on its own sheet:

| Part | Sheet | Content | Your rows (with 3 validators / with 2) | Estimated time |
|---|---|---|---|---|
| A | `A_calibration` | calibration round (§3) | 16 / 16 | about 10 min |
| B | `B_items` | items generated by the program (§4–5) | 292 / 408 | about 2.8 / 4.0 h |
| C | `C_attested` | attested nói lái with sources (§6) | 119 / 119 | about 50 min |
| D | `D1_production`, `D2_qu`, `D3_conventions` | three supplementary sheets (§7) | 54 / 54 | about 30 min |
| E | `E_t2gold` | the originals of the decoding items (§8), **optional** | 200 / 275 | about 1.1 / 1.5 h |

In all about **5.4 hours** each with three validators, **6.9 hours** with two; Parts A–D alone about 4.3 / 5.4 hours.
These figures use planning rates of 35 seconds per row in Parts A and B, 25 seconds in Part C, 30 seconds in Part D
and 20 seconds in Part E. The author tells you the figure for your own sheet when sending it.

**Priority** if time runs short: A → B → C → D → E. Part E is optional. You need not finish in one go: work in short
sessions and **return what you have each week**, even if it is not finished.

Please keep to these rules:

- **No AI assistant** for any row.
- **No discussion** of rows with the other validators before you submit. If something in the instructions is unclear,
  ask the author.
- **Do not share, screenshot or post** the workbook or any of its rows, and do not discuss them with anyone (including
  people taking the human-baseline form) until the paper is published: the workbook contains test items.
- **A dictionary is allowed only** for the real-word question (`lexical_input`, `lexical_candidate`), and even there the
  question asks for your judgment as a native speaker, not for dictionary membership. Do not look anything up for the
  other questions.
- In the sheet you are identified only by a letter (A, B or C). The author asks you separately four coarse questions:
  your age band (18–29 / 30–49 / 50+), the region you grew up in (North, Centre, South or outside Vietnam), years lived
  in Vietnam (< 5 / 5–15 / > 15) and whether you were raised abroad. They are not part of the sheet.
- If anyone sends you a link to the project's human-baseline form (a short Google Form), please **do not fill it
  in**: validators do not take the human-baseline form.

**How to fill in.** Every answer cell takes **Có** (yes), **Không** (no) or **Không chắc** (unsure). Please pick from
the dropdown; if you type, use only Có / Không / Không chắc or yes / no / unsure. Use "Không chắc" only when you genuinely cannot decide (a regional word you do not know, say) and
give the reason in `comment`. An empty cell means you skipped the row. In the main agreement figure "Không chắc" counts
like an empty cell, so if you lean one way, choose that answer. The CSV copies have no dropdowns; the Google Sheets copy
does.

### 2. Syllable structure and the six kinds of nói lái [NATIVE-CHECK]

A Vietnamese syllable is an **onset** (initial consonant, possibly absent), a **rime** (glide if any + main vowel +
final if any) and a **tone** (ngang, huyền, sắc, hỏi, ngã, nặng). For example: *trung* = tr + ung + ngang; *bình* = b +
inh + huyền; *chủ* = ch + u + hỏi; *xoa* = x + oa (glide o, main vowel a) + ngang. [NATIVE-CHECK]

Nói lái swaps some components between two syllables. The project names six kinds V1–V6. The codes carry **no regional
label** (sources disagree about "Northern" and "Southern" nói lái). Every output in the table was computed by the
program and is printed in the old tone-mark style; most are not words, which is expected. [NATIVE-CHECK]

| Kind | Swapped between the two syllables | Kept in place | *trung bình* | *làm chủ* | *vô hình* |
|---|---|---|---|---|---|
| **V1** | rimes | onsets, tones | trinh bùng | lù chảm | vinh hồ |
| **V2** | onsets and rimes (the syllables trade places) | tones | binh trùng | chù lảm | hinh vồ |
| **V3** | tones | onsets, rimes | trùng binh | lảm chù | vồ hinh |
| **V4** | rimes and tones | onsets | trình bung | lủ chàm | vình hô |
| V5 | onsets | rimes, tones | bung trình | chàm lủ | hô vình |
| V6 | onsets and tones | rimes | bùng trinh | chảm lù | hồ vinh |

As the table shows, V6 is the V1 output read in the other order, V5 is V4 reversed and V3 is V2 reversed. T1 and T3
rows use only V1–V4; V5 and V6 appear only in the taxonomy and in the T2 question ("of any kind").

**Spelling when syllables are reassembled.** /k/ is written **k** before e, ê, i, y; **q** with the glide u (*qu*);
**c** elsewhere (*c* + *en* + hỏi → *kẻn*, not *cẻn*). Before e, ê, i, g and ng are written **gh** and **ngh**. In
the new style the tone mark always sits on the main vowel (*hoà*); the program prints the old style, which puts it on
the glide o or u in the rimes oa, oe, uy without a final, except after q (*hòa*). Both are correct. After a consonant,
i or y (*lí* / *lý*) are both correct.

### 3. Part A: calibration round

Part A has 16 rows with the same columns and the same answers as Part B (§4). The program computed them from word pairs
that are **not** in the dataset, and some rows are deliberately wrong. Its purpose is to check that the instructions are
clear before the main work starts; Part A rows are never used in any number in the paper.

Please do Part A on 13–15 October 2026 and return it at once. The author compares your answers with the key and **sends
you the key with an explanation** for every row where your answer differs. If fewer than 80% of your `correct` answers
match the key, the author has a short call or exchange of messages with you and sends a second calibration set of the
same form. **Nobody is excluded** on calibration: the differences tell the author which parts of the instructions need
to be clearer.

### 4. Part B: items generated by the program [NATIVE-CHECK]

Each row has `row_id` (B-0001, …), `task` (T1, T2 or T3), `kind` (V1–V4; empty on T2 rows), `input`, `candidate`,
`system_verdict` (T3 rows only, `yes` or `no`; the Vietnamese question writes it Có / Không), and the question written
out in Vietnamese (`question_vi`) and English (`question_en`).

**Three item types**, with the question exactly as the sheet phrases it [NATIVE-CHECK]:

- **T1 (transformation).** *Nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “X” là “Y”?* ("Is “Y” the V1 nói
  lái (swap rimes, keep onsets and tones) of “X”?") X is `input`, Y is `candidate`. The question: is Y the output of the
  named kind, in the named order, correctly spelled?
- **T2 (decoding).** *“X” là cách nói lái (một kiểu bất kỳ) của “Y”?* ("Is “X” a nói lái (of any kind) of “Y”?") X
  (`input`) is the nói lái form, Y (`candidate`) is the original the program proposes. The question: is X a nói lái of Y
  under any of the six kinds, in either order? A nói lái can have several originals; judge only the one shown.
- **T3 (judgment).** *“Y” có phải là nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “X” không? Hệ thống trả
  lời: Có.* (or *Không.*) ("Is “Y” the V1 nói lái (swap rimes, keep onsets and tones) of “X”? The system answers:
  yes / no.") The program has given a verdict; the question is **whether that verdict is right**. "Có" is right when Y
  is the output of the named kind, in the named order, correctly spelled; "Không" is right in every other case (another
  kind, a wrong tone, a misspelling, …).

The gloss after the kind name in the question: V1 *đổi vần, giữ phụ âm đầu và thanh* (swap rimes, keep onsets and
tones); V2 *đổi chỗ hai âm tiết, giữ thanh* (swap whole syllables, keep tones in place); V3 *đổi thanh, giữ phụ âm đầu
và vần* (swap tones, keep onsets and rimes); V4 *đổi vần và thanh, giữ phụ âm đầu* (swap rimes with their tones, keep
onsets). In V2, "giữ thanh" means the tones stay in position: the two syllables trade places while the first tone stays
on the first syllable and the second tone on the second (*trung bình* → *binh trùng*).

**Some rows are deliberately wrong** (a wrong candidate, or a wrong system verdict) to check attention. Answer each of
them like any other row. If you miss many of them, the author goes over the instructions with you again; your answers
are not discarded.

**The columns you fill in** (values Có / Không / Không chắc, except `dialect` and `comment`):

| Column | Question | Answer **Có** (yes) when | Answer **Không** (no) when |
|---|---|---|---|
| `correct` | Is the program right? | T1: `candidate` is the named kind's output, in the named order, correctly spelled. T2: `input` is a nói lái (any kind, either order) of `candidate`. T3: the system's Có/Không verdict is right. | T1: wrong rime, tone or onset, wrong order, a misspelling, or right under a different kind. T2: no kind turns `candidate` into `input`. T3: the system's verdict is wrong. |
| `spelling` | Is `candidate` spelled correctly? | Standard c/k/q, g/gh, ng/ngh; old- or new-style mark placement; i or y after a consonant. | A spelling error (e.g. *cẻn su*). |
| `lexical_input` | Is `input` a real word or phrase? | A meaningful word or phrase Vietnamese speakers use (regional words and slang included), in your judgment; dictionary membership is not required. | A pronounceable but meaningless string (*trinh bùng*). Meaningful syllables that do not make a phrase → Không. |
| `lexical_candidate` | Is `candidate` a real word or phrase? | as above | as above |
| `offensive` | Does `input` or `candidate` have a vulgar, obscene or insulting reading? | It has such a reading, including one that only appears when you swap the nói lái back. Choosing Có is enough; please do not write the vulgar reading out in `comment`. | No such reading. |
| `dialect` | Does your `correct` answer depend on how your region pronounces something? | Choose the merger (table below). | Choose `none` (or leave empty). |
| `comment` | Free text (optional) | The reason for a "Không chắc"; `RULE?` when you suspect a systematic error (§9). | |

**Conventions** (the calibration round checks most of these):

- Both tone-mark placements (*hòa* / *hoà*) are correct. i or y after a consonant (*lí* / *lý*) are both correct.
- A c/k, g/gh or ng/ngh misspelling makes a T1 candidate **wrong**: `correct` = Không and `spelling` = Không.
- A T3 row with a misspelled candidate and the system answering Không: the system is right, so `correct` = Có and
  `spelling` = Không.
- *ay* / *ây* (*dày* / *dầy*) are different spellings for the program. The rule works on the letters, so the output
  can have *ây* where the familiar spelling has *ay* (or the reverse). If `candidate` is the rule output, `correct` = Có
  and `spelling` = Có; if you would write or say the other form, put it in `comment`.
- T1 is judged **in the named order**: a right output read the other way round is another kind, so `correct` = Không.
- A rule-correct output with a vulgar reading: `correct` = Có **and** `offensive` = Có. The two questions are separate.
- `correct` is judged on the **standard spelling**. If your regional pronunciation does not distinguish two sounds or
  tones (hỏi and ngã, say) and that makes you hesitate, choose that merger in `dialect`.

**Values of the `dialect` column** (the project's table of regional mergers). The column does not ask where you are
from; it asks whether your answer **on this row** depends on regional pronunciation. A cell takes one value; if more
than one merger applies, choose one and name the others in `comment`. [NATIVE-CHECK]

| Value | Meaning | Region | Example |
|---|---|---|---|
| `none` | the answer does not depend on regional pronunciation | | |
| `d=gi` | d and gi sound the same | everywhere | *dày* / *giày* |
| `d/gi=r` | r sounds like d / gi | North | *rao* / *dao* / *giao* |
| `ch=tr` | ch and tr sound the same | North | *tranh* / *chanh* |
| `s=x` | s and x sound the same | North | *xa* / *sa* |
| `v=d/gi` | onset v sounds like d / gi | South, Centre | *vô* / *dô* |
| `final n=ng` | final -n and -ng sound the same | South, Centre | *tan* / *tang* |
| `final t=c` | final -t and -c sound the same | South, Centre | *mắt* / *mắc* |
| `hỏi=ngã` | the hỏi and ngã tones sound the same | South, Centre | *củ* / *cũ* |
| `other` | another merger; describe it in `comment` | | |

### 5. Worked examples for Part B [NATIVE-CHECK]

The `lexical_*` and `offensive` cells below are the author's judgments; in your sheet they are yours. The `dialect`
cell of every example is `none`. [NATIVE-CHECK]

| # | Row | `correct` | `spelling` | `lexical_input` | `lexical_candidate` | `offensive` | Why |
|---|---|---|---|---|---|---|---|
| 1 | T1, V1: *trung bình* → *trinh bùng* | Có | Có | Có | Không | Không | the V1 output; not a word, which is expected |
| 2 | T1, V1: *trung bình* → *trình bung* | Không | Có | Có | Không | Không | this is the V4 output (tones move too), not V1 |
| 3 | T1, V1: *trung bình* → *bùng trinh* | Không | Có | Có | Không | Không | the V1 output read the other way round (i.e. V6); T1 is judged in the named order |
| 4 | T1, V3: *làm chủ* → *lạm chù* | Không | Có | Có | Không | Không | wrong tone on the first syllable; the V3 output is *lảm chù* |
| 5 | T1, V1: *củ sen* → *cẻn su* | Không | Không | Có | Không | Không | /k/ before e is written k: the right output is *kẻn su* |
| 6 | T1, V3: *xoa dầu* → *xoà dâu* | Có | Có | Có | Không | Không | the program prints *xòa dâu* (old style); *xoà dâu* (new style) is equally right |
| 7 | T2: *lù chảm* / original *làm chủ* | Có | Có | Không | Có | Không | *lù chảm* is the V1 output of *làm chủ* |
| 8 | T2: *trinh bùng* / original *trúng bình* | Không | Có | Không | Không | Không | no kind, in either order, turns *trúng bình* into *trinh bùng* (the original is *trung bình*) |
| 9 | T3, V1: *vô hình* / *vinh hồ*; system answers Có | Có | Có | Có | Không | Không | *vinh hồ* is the V1 output, so "Có" is right |
| 10 | T3, V1: *làm chủ* / *lủ chàm*; system answers Không | Có | Có | Có | Không | Không | *lủ chàm* is the V4 output, not V1, so "Không" is right |
| 11 | T3, V2: *trung bình* / *binh trùng*; system answers Không | Không | Có | Có | Không | Không | *binh trùng* is the V2 output, so the system is wrong |
| 12 | T3, V1: *củ sen* / *cẻn su*; system answers Không | Có | Không | Có | Không | Không | the candidate is misspelled (it should be *kẻn su*), so "Không" is right |

In the sheet, row 1 appears with the question *Nói lái kiểu V1 (đổi vần, giữ phụ âm đầu và thanh) của “trung bình” là
“trinh bùng”?*; row 7 with *“lù chảm” là cách nói lái (một kiểu bất kỳ) của “làm chủ”?*; row 11 with *“binh trùng” có
phải là nói lái kiểu V2 (đổi chỗ hai âm tiết, giữ thanh) của “trung bình” không? Hệ thống trả lời: Không.*

### 6. Part C: attested nói lái with sources [NATIVE-CHECK]

Part C has 119 rows (C-001, …); every validator judges all of them. Each row has `input`, `output` and `source` (a
book, newspaper, web page, or "folk" for oral tradition; a source marked "[UNCERTAIN]" has not yet been re-checked by
the author). Some rows come from the project's existing list and the rest were found on the web; no row has been checked
by a native speaker yet. Some rows have three syllables. The two phrases may come in either
order (the plain meaning first or the nói lái first). The source is for information only: you need not open the link;
please answer from your own knowledge.

| Column | Question | Notes |
|---|---|---|
| `known` | Have you heard or read this nói lái before? | Có / Không / Không chắc |
| `valid` | Would a Vietnamese speaker accept the second phrase as a nói lái of the first? | Some rows bend the rule a little for meaning (a vowel or a tone changed), or work only in one region's pronunciation. That is fine: if speakers accept it, answer Có and name the merger in `dialect` or describe the change in `comment`. Answer Không when nobody would take one as a nói lái of the other. |
| `spelling_ok` | Are both phrases spelled correctly? | Both mark placements and i/y are correct, as in Part B. If Không, write the right spelling in `comment`. |
| `your_form` | If you know this nói lái in another form, or would say it differently, write your form. | Leave it empty if the form shown is the one you know. |
| `offensive` | as in Part B | |
| `dialect` | as in Part B: choose the merger if the nói lái only works in one region's pronunciation | |
| `comment` | free text | |

**Warning:** a few rows **have vulgar content**, because they are deliberate obscene puns from folk poetry and
literature (e.g. the poems of Hồ Xuân Hương). You may skip them: choose Có in `offensive` only and leave the other cells
empty (and, if you like, write "bỏ qua" (skipped) in `comment`). Please do not write the vulgar reading out in
`comment` or `your_form`. [NATIVE-CHECK]

### 7. Part D: three supplementary sheets [NATIVE-CHECK]

- **`D1_production`** (20 rows; columns `phrase`, `your_noilai`, `comment`): this is the first task in §1, done before
  reading the other sections. For each phrase, write in `your_noilai` the nói lái you **spontaneously produce first**,
  as you would in conversation; there is no right or wrong answer. This sheet tells us which kind speakers in each
  region produce spontaneously. If two forms come to mind, put the first in `your_noilai` and the other in `comment`.
- **`D2_qu`** (20 rows; columns `phrase`, `kind`, `option_A`, `option_B`, `choice`, `your_form`, `comment`). Each phrase
  has a syllable beginning with *qu-*. Under the kind in `kind` there are two possible outputs, `option_A` and
  `option_B`; they differ in whether the *u* of *qu-* moves with the rime or stays with the onset. In `choice`, pick
  **A**, **B**, **cả hai** (both) or **không cái nào** (neither), by your own sense of it. In `your_form`, write the
  form you would say (required when you choose "không cái nào"). [NATIVE-CHECK]
- **`D3_conventions`** (14 questions, in Vietnamese only; columns `topic`, `question_vi`, `options`, `answer`,
  `comment`). Questions about the spelling you find natural (tone-mark position, i/y, ay/ây, a few rare rimes). Write
  your answer in `answer`, choosing from `options`; for a question about several words, answer for each word.

### 8. Part E: the originals of the decoding items (optional) [NATIVE-CHECK]

Each row has a nói lái form (`noilai_form`) and up to three originals (`reading_1`, `reading_2`, `reading_3`) that the
program found in the dictionary. For each original, in the matching `accept_` cell, choose **Có** if it is a real word
or phrase and `noilai_form` is a nói lái of it (any kind, either order); choose **Không** otherwise. An empty `reading`
cell means there is no second or third original. If another original that the list misses comes to mind **at once**,
write it in `missing_reading` (one phrase; any more go in `comment`). Do not search for one: add it only if it comes to
you unprompted. Most rows go to one validator only; 50 rows go to everyone so that agreement can be measured.

### 9. If you suspect a systematic error: `RULE?`

If you see **many rows failing in the same way** (every row with *gi*, say, or every row with a particular rime), write
`RULE?` in the `comment` of one such row with a one-sentence description. The author checks every `RULE?` note against
the rule tables. A confirmed error is fixed and the whole dataset is regenerated **before** any model is run on the test
set. This is the most valuable thing you can give us.

### 10. Disagreements, and how agreement is reported [NATIVE-CHECK]

Every Part B row is judged by two validators (with three validators, 60 rows are judged by all three; with two, both
judge every row). The final `correct` label is decided as follows:

1. All definite answers (Có or Không, at least two of them) agree → that answer.
2. Two validators disagree → the third validator, who has not seen the row, judges it without seeing the other answers;
   the majority of the three decides.
3. Still split, or only two validators → the author decides against the rule tables and writes the decision and the
   reason into the adjudication log; the number of author decisions is reported in the paper.
4. Fewer than two definite answers ("Không chắc" or empty) → the row stays unresolved: it remains in the dataset but is
   left out of the program's precision figure.
5. The author never overrides validators who agree.

**One** validator's `offensive` = Có flags the item; a flagged item is kept out of the prompts sent to online AI
services (APIs) and out of the human-baseline form. Every merger named in `dialect` is recorded. In Part C, a row counts as
**native-verified** when at least two validators answer `valid` = Có and none answers Không.

Agreement is reported for each column as Krippendorff's α and Gwet's AC1, each with a confidence interval ("Không
chắc" treated as an empty cell; α is also computed once with it as a category of its own), **together with** raw
agreement and the distribution of answers. When almost every item is correct, α can be low even though validators almost always agree,
which is why the other figures are reported beside it. Each validator's rate of catching the deliberately wrong rows is
also reported, by letter only. **Nobody is "wrong"**: a disagreement shows where regional pronunciation matters, where
an item is genuinely ambiguous, or where the instructions are unclear.

### 11. Time and timeline

About 5.4 hours each with three validators, 6.9 hours with two (Parts A–D about 4.3 / 5.4 hours; Part E about 1.1 /
1.5 hours), in as many sessions as you like.

| Date (2026) | Step |
|---|---|
| by 12 Oct | you agree to take part and return the ticked consent form; the author shares your workbook with you |
| 13–15 Oct | Part A (calibration); return it at once |
| 15–17 Oct | the author sends the key and the explanations; a short exchange if needed |
| 18 Oct | Gate 1: the main work starts |
| 19 Oct – 1 Nov | Parts B, C, D (and E if you have time); return what you have each week |
| 2–5 Nov | the author scores the sheets; with three validators you may receive a small sheet of rows on which the other two disagreed |
| 22 Nov | results are frozen; up to and including this date you can withdraw, and your answers will then not be used |

---

## Notes for the author (not sent)

Procedure, sizes and the adjudication rule: `docs/gate1/VALIDATION_PROTOCOL.md` (§3 packet, §5 timeline, §6
agreement and adjudication). Commands, from the repository root:

- **Build the packet:** `make validation VALIDATORS="A B C"` (or `VALIDATORS="A B"`). It writes
  `data/validation/validation_<V>.xlsx` per validator (sheets `A_calibration`, `B_items`, `C_attested`, `D1_production`,
  `D2_qu`, `D3_conventions`, `E_t2gold`), CSV copies `<sheet>_<V>.csv`, and the author-only files that are **never
  sent**: `A_calibration_key.json`, `B_key.json`, `C_rows.json` (the Part C rows, needed by `make validation-score`:
  without it Part C is not scored and `attested_verified.tsv` is not written), `D_engine.json`, `E_key.json`,
  `validation_manifest.json` (the manifest's `keys_never_sent` lists the same six). Import each workbook into Google Sheets with File > Import (keeps the dropdowns) and share
  it with one validator. Send the per-validator hours from `validation_manifest.json` (`per_validator.hours_estimate`)
  with the sheet (§1 promises it).
- **Validators' coarse demographics:** §1 tells validators the author asks four questions separately (as consent §5
  states). Record them in `data/validation/validators.json`, e.g.
  `{"A": {"region": "N", "age_band": "30-49", "years_vn": ">15", "raised_abroad": "no"}}`, region codes `N` / `C` /
  `S` / `abroad` (the scorer reads only `region`). Report validators' age bands pooled, never per letter.
- **Score:** put returned workbooks or CSVs in `data/validation/returned/` and run `make validation-score`
  (= `python scripts/make_validation_forms.py score --dir data/validation --returned 'data/validation/returned/*'`);
  the report goes to `data/validation/report/validation_report.json`. The `calibration` block gives each validator's
  matches and the rows to discuss with `explanation_vi` / `explanation_en` from `A_calibration_key.json`: send those
  back after Part A. Fewer than 80% `correct` matches (`VALIDATION_CALIBRATION_PASS`): re-brief and send the second
  calibration set (see the next bullet). Control catch rate under 75% (`VALIDATION_CONTROL_CATCH_MIN`) after the first
  weekly return: re-brief and report; never exclude or down-weight.
  **File names:** the scorer takes the validator letter from the file name, so name every returned file exactly
  `validation_<V>.xlsx` (Google Sheets: File > Download > Microsoft Excel) or `<sheet>_<V>.csv` /
  `B_adjudicate_<V>.csv`. Each weekly return replaces the previous file for that letter, so never keep two files per
  letter. Rename the browser's `validation_A (1).xlsx` and Google's `validation_A - B_items.csv` before scoring: the
  first is counted as an extra validator `A (1)` and the second is ignored without a warning. Check that
  `validation_report.json` → `B.validators` lists only A, B, C.
  **Off-list labels:** the workbook's dropdowns are set to reject other values (after the Google Sheets import, check
  under Data > Data validation that "Reject the input" is on [UNCERTAIN: verify how the import maps it]); `norm_label`
  stops the whole scoring run on any value outside its lists (e.g. `ko`, `hok`, `có lẽ`), so check returned answer
  columns for such values and correct them with the validator before scoring.
- **Second calibration set:** for a validator below 80%,
  `python scripts/make_validation_forms.py calibration2 --dir data/validation --validators B --release data/release/v0.3`
  writes `calibration2_B.xlsx` (sheet `A2_calibration`, CSV copy `A2_calibration_B.csv`) and the key
  `A2_calibration_key.json`: the same specs on the next inputs, the round-1 and release phrases excluded; a spec with no
  other input is left out rather than repeated (spec 10, the single vulgar input, always; 14 rows on the bare spec
  lists). Put the returned file in `data/validation/returned/` and run `make validation-score`: the report's
  `calibration_round2` block scores it like round 1.
- **Third-validator sheets (three validators):** run once per letter,
  `for V in A B C; do python scripts/make_validation_forms.py adjudication-sheet --dir data/validation --to $V; done`.
  Each `B_adjudicate_<V>.csv` holds the rows split between the other two validators, with blank judgments (with the
  rotating pairs A–B, B–C, C–A, a single `--to C` covers only the A–B splits). Send it to validator V only. Returned
  files go into `data/validation/returned/` under the same name, then rescore. Remaining splits: fill `decision`
  (yes/no), `reason` and `date` in `data/validation/adjudication.tsv` and rescore; the scorer never lets a decision
  override agreeing validators. `adjudication.tsv` also holds each validator's judgments and comments, so it is not
  released (§10 tells validators only the count of author decisions is reported).
- **Offensive flags:** §10 tells validators that a flagged item is kept out of API prompts and out of the
  human-baseline form. `make validation-score` writes `data/audit/validator_flags.json` (the flagged items' ids and
  texts, no validator letters; commit it): the API screen of `noilai/eval/run.py` drops every item it names from every
  API run (counted in the run manifest's `api_safety`), and `make baseline` leaves those items and attested rows off the
  forms (`--exclude-flags`). Re-run `make validation-score` after every batch of returns, before `make baseline` and
  before the first API run.
- **Vulgar rows and an under-18 author:** §4 and §6 ask validators not to write vulgar readings out. If the author is
  under 18, the adult co-contact reads the Part C comments on flagged rows.
- **Before sending, check the two non-demonstration example phrases against the built release** (the release
  `.jsonl` files and the build seed were not available when this file was written). If any is found, replace that
  example with another engine-computed one:
  `python -c "import glob; from noilai.gen.generate import load_items; from noilai.validation import release_phrase_keys, phrase_key; k = release_phrase_keys([i for f in glob.glob('data/release/v0.3/noilai_*.jsonl') for i in load_items(f)]); print(len(k), {p: phrase_key(p) in k for p in ('củ sen', 'kẻn su', 'cẻn su', 'xoa dầu', 'xòa dâu', 'xoà dâu', 'trúng bình', 'lạm chù')})"`
  (the first number must be non-zero, or the release was not found).
- **Quoted sheet text:** §4 and §5 quote `question_b` and `VARIANT_GLOSS_VI` (`noilai/validation.py`) verbatim. If
  either changes (a review suggested the V2 gloss "đổi chỗ hai âm tiết, thanh giữ nguyên vị trí" and the T1/T2 frame
  "... có phải là ... không?"), update every quote here and drop the V2 clarifying sentence in §4, then rebuild the
  packet.
- **D1 order:** §1 asks validators to do D1 first, before reading on (about 10 minutes at 30 s per row), so that the
  production answers are not shaped by the §2 table, Part A or Part B. This sets the order of work only; the priority
  A > B > C > D > E for what to drop is unchanged.
- **Changes after calibration:** version this file and log the change in the protocol (§5 says any change to the
  instructions is versioned and logged there).
