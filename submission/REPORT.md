# Lab 21 — Evaluation Report

**Họ tên**: Từ Hoàng Giang  **MSSV**: 2A202602363  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 (14.6 GB khả dụng, fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

- **Thời gian chạy từng stage**: NB1: 12 s, NB2: 357 s, NB3: 550 s, NB4: 1558 s, NB6: 1378 s (nguồn: log Colab do người dùng cung cấp; NB5 không ghi lại thời gian riêng).

`max_length` dùng khi huấn luyện là 1024 — giá trị cố định của tier T4. NB1 đo độ dài token của toàn bộ 250 mẫu sau chat template: p95 = 98, dài nhất = 101, và gợi ý `max_length` = 256. Không mẫu nào chạm ngưỡng 1024 nên không có mẫu bị cắt; với batch 1 trên mỗi thiết bị, giá trị trần lớn hơn không làm tăng bộ nhớ thực tế. Tôi giữ 1024 thay vì hạ xuống 256 để giữ nguyên cấu hình tier dùng chung cho cả bốn run, tránh thêm một biến thay đổi vào phép so sánh.

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*

Có. `results/template_check.json` cho thấy template của Qwen3.5 giữ nguyên khối `<think>…</think>` có nội dung (verdict: "reasoning preserved — safe to train on traces"). Với corpus mặc định, câu trả lời huấn luyện là JSON trần nên khối `<think>` trong phần được giám sát là rỗng; vì vậy ba chế độ mask không phải `everything` cho cùng một mask và tôi dùng `assistant-only`.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `True` |
| Câu hỏi KHÔNG nằm trong loss | `True` |

Dán 3–5 dòng đầu của đoạn được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3727 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1163 |
| (c) LoRA fine-tune | 0.970 | 0.522 | 1.000 | 1670 |

**(b) có thật sự mạnh hơn (a) không?** Có. Điểm target tăng từ 0.000 lên 0.765, tỷ lệ format đạt chuẩn tăng từ 0.000 lên 1.000 và độ trễ giảm mạnh từ 3727 ms xuống 1163 ms.

Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Không sửa — SHA của `OPTIMIZED_PROMPT` khớp hoàn toàn với bản gốc trong labkit (xác nhận bởi kiểm tra `baseline (b) prompt unmodified ok`), đảm bảo mốc so sánh hoàn toàn khách quan.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency (ms) | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | **0.97** | 1.0 | 1670.2 | 489.5 | 8.78 |
| `attn_only` | q,v (attn-only) | 283 | 32,456,704 | 0.0001 | 0.5373 | **0.97** | 1.0 | 1049.2 | 328.5 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.00** | 0.0 | 6293.0 | 489.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.94** | 1.0 | 2098.0 | 565.1 | 3.86 |

> Ghi chú: Cột `train loss (NB4)` là giá trị train loss trung bình trong suốt 30 bước huấn luyện (`result.training_loss` của Trainer), không phải giá trị loss tức thời tại bước cuối cùng.
>
> Biến duy nhất mỗi run thay đổi so với `correct` (Rubric 2.3):
> - `attn_only`: Đổi vị trí gắn adapter từ toàn bộ các khối tuyến tính văn bản (`text-linear`) sang chỉ các ma trận attention (`q,v`), đồng thời tăng rank lên r=283 (alpha=566) để khớp chính xác ngân sách tham số (~32.46M tham số, chênh lệch chỉ ~0.03%).
> - `wrong_lr`: Đổi tốc độ học (learning rate) từ 1e-4 xuống 1e-5 (thang đo của full fine-tuning), giữ nguyên mọi tham số kiến trúc.
> - `qlora`: Đổi mô hình gốc sang nạp 4-bit (`load_in_4bit=True` với BitsAndBytes NF4), giữ nguyên rank 16 và LR 1e-4.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`attn_only` có cùng ngân sách tham số huấn luyện với `correct` (32,456,704 so với 32,464,896, sai lệch chỉ ~0.03%) và đạt kết quả hòa tuyệt đối trên tập target (cả hai đều đạt 0.97). Tuy nhiên, trên tập huấn luyện, `attn_only` lại có train loss trung bình thấp hơn đáng kể (0.5373 so với 0.6265 của `correct`). Do đó, nếu xếp hạng theo train loss, ta sẽ chọn sai mô hình vượt trội; thứ tự theo target (`correct` = `attn_only` > `qlora` > `wrong_lr`) khác biệt so với thứ tự theo loss (`attn_only` < `correct` < `qlora` < `wrong_lr`). Kết quả này cho thấy trên tác vụ hẹp với 50 mẫu đánh giá, khi đã chuẩn hóa ngân sách tham số, vị trí gắn adapter không tạo ra sự chênh lệch chất lượng target đo được, trong khi `attn_only` còn có lợi thế sinh nhanh hơn rõ rệt (1049.2 ms so với 1670.2 ms). Dù vậy, đây là kết quả trên một lượt chạy đơn lẻ (seed 42) nên không khẳng định vị trí gắn adapter luôn vô nghĩa trên mọi tác vụ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ thay đổi duy nhất learning rate từ 1e-4 xuống 1e-5, nhưng train loss trung bình dừng ở mức 1.5702, cao hơn rất nhiều so với 0.6265 của `correct`. Hậu quả trực tiếp là mô hình hoàn toàn thất bại trên tập đánh giá với target đạt 0.00 và format đạt 0.00, kèm theo độ trễ sinh tăng vọt lên 6293.0 ms do mô hình sinh văn bản tự do không theo khuôn khổ JSON. Nếu chỉ nhìn vào loss cao mà không biết thông tin về learning rate, người làm mô hình rất dễ kết luận sai lầm rằng tập dữ liệu quá phức tạp, bài toán khó học hoặc kiến trúc LoRA không thể hội tụ. Thực chất, nguyên nhân cốt lõi chỉ là 30 bước tối ưu hóa với learning rate quá nhỏ (thang đo dành cho full fine-tuning) chưa đủ biên độ cập nhật để đưa các trọng số adapter thích nghi với định dạng đầu ra mong muốn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

Cấu hình `qlora` tiết kiệm khoảng 56.0% bộ nhớ VRAM đỉnh (từ 8.78 GB của `correct` giảm xuống chỉ còn 3.86 GB, tính theo `(8.78 - 3.86) / 8.78`). Sự tiết kiệm bộ nhớ đáng kể này phải đánh đổi bằng việc chất lượng target giảm nhẹ từ 0.97 xuống 0.94 (-0.03), thời gian huấn luyện kéo dài hơn (565.1 s so với 489.5 s) và độ trễ khi sinh chậm hơn (2098.0 ms so với 1670.2 ms) (nhiều khả năng do chi phí giải lượng tử hoá trọng số 4-bit ở mỗi bước — lab không đo riêng phần này). Kết quả đo đạc thực tế cho thấy cái giá phải trả về độ chính xác là rất nhỏ trên tác vụ trích xuất JSON hẹp này, do đó số liệu chỉ ủng hộ một phần khuyến nghị lý thuyết "không dùng QLoRA cho dòng Qwen3.5". Trong thực tế triển khai, nếu phần cứng chỉ có GPU dung lượng nhỏ (dưới 8 GB), QLoRA vẫn là một giải pháp đánh đổi rất khả thi và hiệu quả.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.269` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ):
Mô hình fine-tune đạt mức cải thiện vượt trội trên tập đích chuyên biệt với target delta đạt +0.205 (nâng độ chính xác từ 0.765 của baseline b lên 0.970) cùng tỷ lệ format chuẩn JSON tuyệt đối đạt 1.000. Điều này xác nhận rằng chat template và cơ chế assistant-only loss mask đã được thiết lập hoàn toàn chính xác theo đúng thiết kế của pipeline.

Tuy nhiên, cổng hồi quy đưa ra phán quyết FAILED do năng lực tổng quát của mô hình bị suy giảm nghiêm trọng vượt quá ngưỡng dung sai: điểm regression đo bằng recall từ khóa trên 15 câu tri thức phổ thông đã sụt giảm từ 0.791 xuống 0.522 (tương ứng regression delta là -0.269, vi phạm ngưỡng khắt khe -0.020). Đây là biểu hiện rõ rệt của hiện tượng quên thảm họa (catastrophic forgetting) khi mô hình bị ép học một phân phối cấu trúc hẹp. Do pipeline đánh giá không lưu câu trả lời chi tiết từng mẫu của bài test regression, chúng ta đặt giả thuyết cần kiểm chứng rằng mô hình đã bị thiên kiến sinh JSON ngay cả với các câu hỏi ngôn ngữ tự nhiên thông thường. Chỉ số `valid_trace_rate = 0.0` là hoàn toàn tự nhiên vì dữ liệu huấn luyện mặc định sử dụng câu trả lời JSON trần, không chứa chuỗi suy luận nên khối `<think>` hoàn toàn rỗng. Để khắc phục triệt để hiện tượng này mà không nới lỏng cổng đánh giá hay can thiệp vào prompt chuẩn, phương án kỹ thuật chuẩn mực là trộn thêm 1% đến 5% dữ liệu văn bản phổ thông (replay data) trong quá trình huấn luyện theo đúng khuyến nghị tại slide bài giảng.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | i | Ticket (rút gọn) | Nhãn đúng | FT dự đoán | ft_score | Nhận xét |
|---|---|---|---|---|---|---|
| 1 | 48 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sen`… | 1.00 | ✅ FT thắng (đúng cả 4 trường) |
| 2 | 49 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sent`… | 1.00 | ✅ FT thắng (đúng cả 4 trường) |
| 3 | 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment":`… | 0.75 | ❌ **FT thua** (sai `urgency`: đoán `"trung_binh"` thay vì `"thap"`) |
| 4 | 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sen`… | 0.75 | ❌ **FT thua** (sai `urgency`: đoán `"trung_binh"` thay vì `"thap"`) |
| 5 | 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment"`… | 0.75 | ❌ **FT thua** (sai `urgency`: đoán `"trung_binh"` thay vì `"thap"`) |

Ghi chú: pipeline chỉ lưu dự đoán theo từng mẫu của bản fine-tune (`results/qualitative.json`); baseline (b) chỉ được lưu dưới dạng điểm tổng hợp trong `results/baselines_frozen.json`. Vì vậy bảng so sánh mỗi dự đoán với nhãn đúng, không với câu trả lời của (b). "Thua" ở đây nghĩa là bản fine-tune sai ít nhất một trong bốn trường. Do `ft_pred` trong `results/qualitative.json` bị cắt ở độ dài 90 ký tự, chuỗi dự đoán được hiển thị đúng nguyên văn từ file kèm dấu ba chấm.

**Có mẫu chung nào ở các ca FT thua không?**
Có. Toàn bộ 6 ca bị trừ điểm trong tập đánh giá 50 mẫu (đều có `ft_score = 0.75`, tương ứng đúng 3/4 trường) đều xuất hiện một lỗi sai duy nhất: mô hình fine-tune dự đoán đúng hoàn toàn `intent`, `product` và `sentiment`, nhưng đều phân loại sai nhãn `urgency` thành `"trung_binh"` trong khi nhãn chuẩn là `"thap"`. Kiểm tra tập huấn luyện `data/split/train.jsonl` cho thấy phân bố nhãn `urgency` gồm 86 mẫu `"thap"`, 72 mẫu `"cao"` và 67 mẫu `"trung_binh"`, do đó lỗi không xuất phát từ việc mất cân bằng thiếu mẫu `"thap"`. Trên tập đánh giá có 18 ticket mang nhãn `urgency = "thap"`; bản fine-tune đoán đúng 12 và đoán thành `"trung_binh"` ở 6 ticket còn lại. Dữ liệu hiện có không cho biết vì sao đúng 6 ticket này bị đẩy lên mức trung bình. Một giả thuyết cần kiểm là cách diễn đạt trong ticket (ví dụ lời nhờ lịch sự, khiếu nại nhẹ) làm ranh giới giữa `thap` và `trung_binh` khó phân biệt; kiểm được bằng cách so cách diễn đạt của 6 ticket này với 12 ticket `thap` được đoán đúng.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):**
Dựa trên các bằng chứng thực nghiệm thu thập được, bản LoRA fine-tune này tuyệt đối KHÔNG nên được triển khai trực tiếp vào môi trường vận hành thực tế ở thời điểm hiện tại. Mặc dù mô hình mang lại bước nhảy vọt về độ chính xác trên tác vụ chuyên biệt với target delta đạt +0.205 (nâng độ chính xác lên 0.970) cùng khả năng tuân thủ định dạng JSON hoàn hảo (format = 1.000), việc vi phạm nghiêm trọng cổng an toàn hồi quy (regression delta tụt -0.269, vượt xa ngưỡng dung sai -0.020) cho thấy mô hình đã mắc phải hội chứng quên thảm họa và bị suy thoái năng lực ngôn ngữ tổng quát. Trong một kịch bản ứng dụng trợ lý chăm sóc khách hàng thực tế, việc suy giảm tri thức nền tảng có thể dẫn tới những phản hồi sai lệch nghiêm trọng đối với các tình huống giao tiếp thông thường nằm ngoài phân phối hẹp.

Qua các thí nghiệm đối chứng có kiểm soát trong lab, đòn bẩy kỹ thuật thực sự quyết định hiệu năng theo thứ tự tác động đo được là: **Learning Rate** (yếu tố sống còn: LR 1e-4 đạt target 0.97 trong khi LR 1e-5 chỉ đạt 0.00 do không đủ bước để thích nghi) > **Bảo tồn tri thức tổng quát** (suy giảm 0.269 nếu thiếu dữ liệu duy trì) > **Lượng tử hóa mô hình nền** (QLoRA 4-bit chỉ giảm nhẹ 0.03 target nhưng tiết kiệm đến 56% VRAM) > **Vị trí gắn adapter khi chuẩn hóa ngân sách** (khi khớp số lượng tham số, `attn_only` và `text-linear` hòa nhau ở mức 0.97 target). Đề xuất bước đi tiếp theo là giữ nguyên cấu hình LoRA 1e-4, bổ sung 1% đến 5% dữ liệu replay tổng quát để vượt qua cổng hồi quy trước khi tiến hành thử nghiệm A/B trên người dùng thật.

**Ba điều tôi học được** (cụ thể, không generic):
1. Loss thấp không có ý nghĩa model tốt hơn
Khi bắt đầu với bài học tôi nghĩ train loss càng thấp thì model sẽ tốt lên. Ở NB4, attn_only có train loss thấp nhất (0.5373 so với 0.6265 của correct), với 50 mẫu target ở NB5 thì bằng nhau đều là 0.97. Nếu chọn dựa trên loss sẽ có thể chọn sai người thắng. Nên so sánh cấu hình bằng điểm trên tác vụ thật và chỉ so khi tham số khớp với nhau
2. Chiến thắng ở bài chính thì chưa chắc nên deploy sẽ tốt
Tôi đã chạy thử với 8 mẫu cho fine-tune kết quả thua prompt tốt (0.5 so với 0.688, lần chạy thử 8 mẫu, không lưu trong `results/`), nên có kết luận ban đầu có thể là "không cần fine-tune". Lần chạy đầy đủ thì lại cho kết quả ngược lại: target tăng từ 0.765 lên 0.970, nhưng điểm kiến thức chung đã giảm từ 0.791 về 0.522, cổng hồi quy báo FAILED. Rút ra được hai bài học: Thứ nhất: kết quả trên số lượng mẫu nhỏ (8 mẫu) chưa chắc đủ mạnh hay đáng tin, thứ hai: một bản  fine-tune nên đo được cả cái nó mất (hay bị giảm chỉ số) chứ không phải chỉ tính cái nó được (tăng chỉ số nào)
3. Một vài con số cấu hình có thể ảnh hưởng dây chuyền lên toàn bộ
Ở đây "wrong_lr" chỉ khác "correct" đúng "learning rate" (1e-5 thay vì 1e-4) sau khi chạy 30 bước kết quả trả ra target 0.00 và format 0.00, không trả ra được 1 JSON nào. Loss vẫn giảm (trung bình 1.5702), nếu chỉ nhìn đường loss, tôi đã có thể nghĩ nhầm là "dữ liệu khó" thay vì "Learning rate sai thang đo". Với LoRA, "learning rate" phải khoảng gấp 10 lần full fine-tune. Ngoài ra mask cần quyết định model học gì. NB1 cho thấy chỉ 41% token (39/94) được tính loss, đúng bằng phần câu trả lời JSON. Nếu không che, model sẽ học cả việc viết lại ticket của khách. Điều tôi nhớ là lab bắt chứng minh mask bằng cách giải mã ngược, không tin vào cấu hình

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ sửa lý do dẫn đến phán quyết FAILED:
Trộn thêm khoảng 1-5% dữ liệu kiến thức phổ thông vào 225 mẫu đã train giữ nguyên cấu hình "correct" như đã nói ở 3 điều học được (LR 1e-4, r=16, 30 bước), rồi đo lại với cả 4 nhóm. Với mục tiêu là "regression" không tụt quá 0.02 so với mốc 0.791, đảm bảo target trên 0.765. Đi cùng với đó xem 6 ticket bị đoán nhầm urgency (thap thành trung_binh)
và so với 12 ticket "thap" đoán đúng, để biết lỗi ở cách diễn đạt ticket hay ở dữ liệu train.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/gianghf08102k/lab21-qwen35-triage-vi

**Chi tiết B1 (NB6):**
- Kết quả kiểm tra merge (`results/merge_check.json`): `before_merge` = 0.97 → `after_merge` = 0.97 (Δ = 0.0, sai số cho phép 0.01, n = 50 mẫu). Trọng số sau khi hòa nhập hoàn toàn bảo toàn độ chính xác.
- Kiểm tra hot-swap đa adapter: `['correct', 'attn_only', 'qlora']` theo `results/hotswap_log.txt`.

Phần merge chạy trong NB6 và ghi `results/merge_check.json`. Phần hot-swap của NB6 bị lỗi `offload_dir` khi nạp base lần thứ hai trong cùng tiến trình (bộ nhớ GPU của model đã merge chưa được giải phóng, nên 4 layer cuối bị đẩy sang CPU). Tôi chạy lại đúng đoạn mã hot-swap của NB6 trong một tiến trình Python mới trên cùng phiên Colab; kết quả nằm ở `results/hotswap_log.txt`: ba adapter `correct`, `attn_only`, `qlora` được nạp trên cùng một base và lần lượt trả JSON hợp lệ cho cùng một ticket.
