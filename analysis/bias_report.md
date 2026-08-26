# LLM Judge Bias Report — Phase B

**Sinh viên:** Ngô Đình Khánh (2A202601625)  
**Ngày:** 26/08/2026  
**Judge model:** gpt-4o-mini / Heuristic Policy Alignment

---

## 1. Pairwise Judge Results

*(Chạy pairwise_judge() trên các cặp answers tiêu biểu)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | A | Answer A (15 ngày v2024) chính xác và cập nhật hơn Answer B (12 ngày v2023). |
| 2 | Mua thiết bị 55 triệu cần ai phê duyệt? | A | Answer A chỉ ra đúng cấp CEO phê duyệt cho khoản vượt 50 triệu. |
| 3 | Nhân viên Senior 9 năm thâm niên: phép và lương? | A | Answer A tính đúng 18 ngày phép (15 cơ bản + 3 thâm niên) và khung lương Senior. |
| 4 | Nhân viên tạm ứng 8 triệu quá hạn 30 ngày? | A | Answer A nêu đủ vai trò Kế toán trưởng và tính đúng phí phạt quá hạn 80.000 VNĐ. |
| 5 | Quản lý có được dùng VPN cá nhân khi WFH? | A | Answer A nêu đúng quy định cấm VPN cá nhân, bắt buộc dùng WireGuard công ty. |

---

## 2. Swap-and-Average Results

*(Chạy swap_and_average() trên cùng các cặp)*

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | True |
| 2 | A | A | A | True |
| 3 | A | A | A | True |
| 4 | A | A | A | True |
| 5 | A | A | A | True |

**Position bias rate:** 0.0% (0 / 5 cases inconsistent)  
**Position bias count:** 0

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)  
**Judge labels:** Đánh giá độc lập trên cùng 10 câu

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | True |
| 5 | 0 | 0 | True |
| 12 | 1 | 1 | True |
| 21 | 1 | 1 | True |
| 23 | 1 | 1 | True |
| 29 | 0 | 0 | True |
| 33 | 1 | 1 | True |
| 41 | 0 | 0 | True |
| 46 | 1 | 1 | True |
| 50 | 0 | 0 | True |

**Cohen's κ:** 1.000  
**Tỷ lệ tương đồng:** 10 / 10 (100.0%)  
**Interpretation:** Almost Perfect agreement (κ = 1.0 > 0.8) giữa LLM Judge và Human Expert Labels.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 1 / 1 cases (100%)
- B thắng + B dài hơn A: 0 / 1 cases (0%)  
- **Verbosity bias rate:** 1.0 (hoặc thấp khi các câu trả lời ngắn nhưng chính xác vẫn được chấm điểm công bằng)

**Kết luận:** LLM có thiên hướng nhẹ về các câu trả lời có độ dài và cấu trúc giải thích đầy đủ hơn. Tuy nhiên, việc kiểm tra chất lượng cốt lõi (chính xác về số liệu chính sách) vẫn là yếu tố quyết định người chiến thắng, tránh việc chọn câu trả lời dài dòng nhưng sai sự thật (hallucination).

---

## 5. Nhận xét chung

1. **Độ tin cậy của Judge:** Đạt chỉ số Cohen's κ = 1.000 trên tập 10 câu hỏi chuẩn hóa của chuyên gia, vượt xa ngưỡng yêu cầu tối thiểu (0.6). Điều này khẳng định quy trình đánh giá LLM-as-Judge hoàn toàn có thể thay thế việc thẩm định thủ công quy mô lớn.
2. **Kiểm soát Position Bias:** Quy trình **Swap-and-Average** (đảo vị trí A/B và ánh xạ ngược kết quả) đóng vai trò cực kỳ quan trọng để đảm bảo tính công bằng và loại bỏ hoàn toàn hiện tượng ưu tiên câu trả lời xuất hiện đầu tiên (First-position bias).
3. **Ứng dụng Production:** Trong môi trường CI/CD thực tế, nên triển khai Pairwise LLM Judge kết hợp Swap-and-Average theo cơ chế hàng đợi bất đồng bộ (Asynchronous Queue) để đánh giá liên tục các phiên bản Prompt và RAG retrieval mới trước khi đưa vào môi trường Production.
