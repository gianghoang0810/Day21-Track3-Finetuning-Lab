# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Lần chạy thử và lần chạy chính thức cho hai câu chuyện ngược nhau. Chạy thử (`EVAL_LIMIT=8`, `EPOCHS=1`): fine-tune thua prompt tốt, 0.500 so với 0.688 (lần chạy thử 8 mẫu, không lưu trong `results/`). Chạy đủ 50 mẫu, 2 epoch: fine-tune thắng target rõ ràng (0.970 so với 0.765) nhưng vẫn FAILED, vì điểm kiến thức chung tụt từ 0.791 xuống 0.522. Nếu dừng ở lần chạy thử, tôi đã kết luận sai hoàn toàn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán. Tôi nghĩ phần train sẽ khó nhất, nhưng NB3 và NB4 chạy ổn (30 bước mỗi run). Thời gian mất nhiều nhất là thao tác trên Colab: lần đầu ô 3 vẫn chạy với `EVAL_LIMIT=8` và `EPOCHS=1` của lần thử nên tôi phải dừng, xoá `results/`, `adapters/` rồi chạy lại; Colab mất kết nối ở ô 2; NB6 merge xong nhưng phần hot-swap báo lỗi `offload_dir` do hết bộ nhớ GPU, phải chạy lại riêng phần đó trong một tiến trình mới.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin train loss thấp hơn thì model tốt hơn. `attn_only` có train loss thấp nhất (0.5373 so với 0.6265 của `correct`) nhưng trên 50 mẫu target chỉ hoà (0.97 = 0.97). Tôi cũng từng nghĩ fine-tune thắng bài chính là đủ để dùng; giờ tôi biết phải đo cả cái model mất đi (regression −0.269).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code để đọc hiểu lab, lập kế hoạch các bước, hướng dẫn chạy Colab, và viết khung `REPORT.md` từ số liệu trong `results/`. Những chỗ nó sai hoặc thiếu:
- Đưa code đẩy adapter lên HuggingFace còn để nguyên `<hf-user>`, lần chạy đầu báo `HFValidationError`.
- Không lường trước NB6 hết bộ nhớ GPU ở bước hot-swap.
- Bản report đầu có một câu khẳng định nguyên nhân lỗi `urgency` ("do câu hỏi lịch sự…") mà không có số liệu nào chứng minh.
- Agent thực thi tự chạy `git config core.autocrlf` dù không được giao.

Bài học: phải tự đối chiếu số liệu và khẳng định với file `results/`, không tin ngay.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đo baseline trước khi train: viết một prompt tốt có ví dụ, chấm nó trên một tập eval cố định gồm cả bài toán chính lẫn kiến thức chung, rồi đóng băng mốc đó. Ở lab này prompt tốt đã đạt 0.765 target với độ trễ thấp nhất (1163 ms), nên chỉ fine-tune khi prompt không đạt yêu cầu, và khi fine-tune thì trộn sẵn dữ liệu phổ thông để không làm model quên.
