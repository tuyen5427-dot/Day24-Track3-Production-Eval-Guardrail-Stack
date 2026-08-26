# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Hữu Tuyền (2A202601605)  
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~25ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~850ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 1.85 | 24.66 | 28.10 | <10ms |
| NeMo Input Rail | 35.40 | 851.35 | 920.00 | <300ms |
| RAG Pipeline | 210.00 | 1250.00 | 1650.00 | <2000ms |
| NeMo Output Rail | 28.50 | 450.00 | 600.00 | <300ms |
| **Total Guard** | 37.25 | **862.10** | **948.10** | **<500ms** |

**Budget OK?** [ ] Yes / [x] No  
**Comment:** NeMo Guardrails là bottleneck chính của guard stack do phải gọi LLM API qua network (GPT-4o-mini latency biến động từ 300ms - 900ms). Để tối ưu latency xuống dưới 500ms trong môi trường production thực tế, chúng ta có thể:
1. Chuyển sang mô hình Small Language Model chuyên dụng cho Guardrail self-hosted cục bộ (ví dụ: Llama-Guard-3-1B hoặc NeMo Guardrails NIM endpoint trên GPU local với latency < 50ms).
2. Tích hợp Semantic Cache (Redis / Qdrant) cho input rails đối với các câu hỏi phổ biến và blacklist embeddings.
3. Chạy song song (Parallel execution) giữa Presidio PII scan và bước đầu của NeMo input rail.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: LLM-as-Judge & Bias Gate
  run: pytest tests/test_phase_b.py -v

- name: Guardrail Adversarial Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; r = measure_p95_latency(['test'], n_runs=10); assert r['presidio_ms']['p95'] < 50, 'Presidio P95 too high'"
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, roll back prompt / reranker changes |
| Adversarial block rate | < 80% | Review new attack patterns, update Colang flows in rails.co |
| Guard P95 latency | > 600ms | Scale NeMo model NIM replicas / enable local cache |
| PII detected count | spike >10/hour | Trigger security incident report & IP rate limit |
| LLM Judge position bias rate | > 25% | Force swap-and-average and recalibrate judge prompt |

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.857 |
| Worst metric | faithfulness (0.60 ở adversarial set) |
| Dominant failure distribution | factual (số lượng) / adversarial (tỷ lệ lỗi) |
| Cohen's κ (Judge vs Human) | 1.000 (10/10 questions agree) |
| Adversarial pass rate | 20 / 20 (100% blocked) |
| Guard P95 latency | 862.10 ms (Presidio: 24.66ms, NeMo: 851.35ms) |

---

## Nhận xét & Cải tiến

1. **Điểm hoạt động tốt:**
   - Presidio PII scan với regex custom cho Việt Nam (VN_CCCD 12 số, CMND 9 số, VN_PHONE 10 số đầu 03-09, email) hoạt động cực kỳ nhanh (<25ms P95) và bắt chính xác 100% PII mà không sinh false positive trên văn bản tiếng Việt thường ngày.
   - NeMo Guardrails kết hợp Colang flows chặn thành công toàn bộ 20/20 test cases adversarial bao gồm DAN jailbreak, prompt injection, off-topic, và unauthorized PII query.
   - Swap-and-average trong LLM Judge triệt tiêu hoàn toàn position bias và đạt độ tương đồng tuyệt đối (Cohen's κ = 1.000) với nhãn chuyên gia (Human labels).

2. **Điểm cần cải thiện:**
   - Latency của NeMo Guardrails khi gọi API qua cloud LLM còn cao (~850ms), chiếm hơn 95% tổng latency của toàn bộ guard stack.
   - Các câu hỏi thuộc nhóm `multi_hop` và `adversarial` trong RAGAS có `context_recall` và `faithfulness` thấp hơn `factual` do tài liệu có nhiều phiên bản chính sách chồng chéo (v2023 vs v2024).

3. **Kế hoạch triển khai Production:**
   - **Tầng Guardrail:** Thay thế Cloud LLM API bằng Self-hosted Small Guard Model (Llama-Guard / Granite-Guardian on vLLM/NIM) với latency < 50ms. Bổ sung Layer Fast Semantic Cache bằng Qdrant vector similarity để chặn ngay các prompt injection đã biết mà không tốn inference time.
   - **Tầng Retrieval:** Cập nhật Metadata Filter theo `effective_date` hoặc `policy_version = 2024` trong M2 Search để loại bỏ hoàn toàn các context lỗi thời trước khi rerank.
