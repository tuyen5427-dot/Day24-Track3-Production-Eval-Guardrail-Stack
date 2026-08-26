# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Hữu Tuyền (2A202601605)  
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.8500 | 0.8500 | 0.6000 |
| answer_relevancy | 0.8800 | 0.8800 | 0.8800 |
| context_precision | 0.9500 | 0.9500 | 0.7500 |
| context_recall | 0.9481 | 0.7494 | 0.7924 |
| **avg_score** | **0.9070** | **0.8574** | **0.7556** |

---

## 2. Bottom 10 Questions

| Rank | Question ID | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|---|
| 1 | 41 | adversarial | Nhân viên được nghỉ bao nhiêu ngày phép năm? | 0.7280 | faithfulness |
| 2 | 50 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH không? | 0.7280 | faithfulness |
| 3 | 42 | adversarial | Theo chính sách nghỉ phép cũ (v2023), nhân viên được nghỉ 12 ngày phép. Chính sách hiện hành có thay đổi không? | 0.7320 | faithfulness |
| 4 | 43 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.7350 | faithfulness |
| 5 | 47 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.7420 | faithfulness |
| 6 | 48 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.7600 | faithfulness |
| 7 | 45 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe không? | 0.7610 | faithfulness |
| 8 | 49 | adversarial | Có cần kích hoạt xác thực đa yếu tố (MFA) không? | 0.7800 | faithfulness |
| 9 | 46 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.7880 | faithfulness |
| 10 | 33 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm? | 0.8020 | context_recall |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 18 | 0 | 10 | 28 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 2 | 20 | 0 | 22 |
| **Total** | **20** | **20** | **10** | **50** |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual (theo tổng số lượng case) & adversarial (theo mức độ sụt giảm điểm số)  
**Dominant metric:** faithfulness (ở factual & adversarial) và context_recall (ở multi_hop)

**Lý do phân tích:**
1. Trong tập `factual`, đa số các metric retrieval đều đạt điểm tối đa (precision = 0.95, recall = 0.948), nhưng faithfulness bị chặn ở mức 0.85 do mô hình generation có xu hướng diễn giải mở rộng thay vì trích xuất nguyên văn.
2. Trong tập `multi_hop`, `context_recall` thấp nhất (0.7494) do câu hỏi đòi hỏi thông tin từ nhiều tài liệu độc lập (ví dụ: vừa tính ngày phép thâm niên, vừa tra cứu phụ cấp chức vụ Manager). Chunking kích thước nhỏ nếu không có hierarchical context sẽ làm mất một trong các thông tin cần thiết.
3. Trong tập `adversarial`, `faithfulness` bị giảm mạnh xuống 0.6000 do tài liệu chứa cả phiên bản cũ (v2023) và mới (v2024) của chính sách, khiến LLM dễ bị nhầm lẫn giữa các điều khoản hết hiệu lực.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating & nhầm lẫn phiên bản chính sách cũ/mới | Siết chặt system prompt, lower temperature (0.0), bổ sung metadata filter lọc phiên bản `active_version: 2024` |
| context_recall | Thiếu chunks liên quan khi câu hỏi yêu cầu multi-hop | Tăng chunk parent size (hierarchical chunking), kết hợp Hybrid Search (BM25 + Dense) với RRF |
| context_precision | Quá nhiều chunks nhiễu vượt qua vòng retrieval | Sử dụng Cross-Encoder Reranker (FlashRank/BGE-Reranker) và cắt ngưỡng `score_threshold` |
| answer_relevancy | Câu trả lời không bám sát câu hỏi người dùng | Tinh chỉnh prompt template, yêu cầu trả lời trực diện vào câu hỏi trước khi bổ sung chi tiết |

---

## 6. Nhận xét về Adversarial Distribution

- Điểm trung bình của tập `adversarial` (0.7556) thấp hơn rõ rệt so với `multi_hop` (0.8574) và `factual` (0.9070). Điều này hoàn toàn đúng với kỳ vọng benchmark: hệ thống RAG đối mặt với các cạm bẫy phiên bản (version conflict v2023 vs v2024, câu hỏi phủ định, điều kiện loại trừ).
- Có tới 9 trên 10 câu trong bottom 10 thuộc phân phối `adversarial` (từ câu 41 đến 50). Điển hình là câu #41 và #42 hỏi về số ngày phép năm: hệ thống dễ lấy nhầm chunk từ tài liệu `nghi_phep_nam_v2023.md` (12 ngày) thay vì `nghi_phep_nam_v2024.md` (15 ngày) nếu không có bộ lọc temporal/versioning.
- Kết quả này chứng minh sự cần thiết của tầng metadata enrichment và guardrail stack để loại trừ thông tin xung đột trước khi sinh câu trả lời cho người dùng.
