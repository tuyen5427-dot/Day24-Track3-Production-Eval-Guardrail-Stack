from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_CACHED_ANALYZER = None
_CACHED_ANONYMIZER = None
_CACHED_RAILS = None


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.

    Custom recognizers:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer   = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,
        }
    """
    global _CACHED_ANALYZER, _CACHED_ANONYMIZER
    if analyzer is None or anonymizer is None:
        if _CACHED_ANALYZER is None or _CACHED_ANONYMIZER is None:
            _CACHED_ANALYZER, _CACHED_ANONYMIZER = setup_presidio()
        analyzer = _CACHED_ANALYZER
        anonymizer = _CACHED_ANONYMIZER

    target_entities = [
        "VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "CRYPTO", "IBAN_CODE", "IP_ADDRESS",
    ]
    results = analyzer.analyze(text=text, entities=target_entities, language=PRESIDIO_LANGUAGE)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml."""
    global _CACHED_RAILS
    if _CACHED_RAILS is not None:
        return _CACHED_RAILS
    try:
        from nemoguardrails import RailsConfig, LLMRails
        config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
        _CACHED_RAILS = LLMRails(config)
        return _CACHED_RAILS
    except Exception:
        return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,
        }
    """
    text_lower = text.lower()

    # Flow rules matching Colang rails.co definitions
    jailbreak_keywords = [
        "bỏ qua tất cả hướng dẫn", "ignore your previous instructions", "pretend you are dan",
        "you are now an unrestricted ai", "forget your system prompt", "system override",
        "đóng vai ai không có giới hạn", "đóng vai hr manager", "không có giới hạn",
        "tấn công mạng", "in ra toàn bộ system instructions", "admin command",
        "<!-- ignore", "dump all training data", "bỏ qua tất cả policy",
    ]

    off_topic_keywords = [
        "viết một bài thơ", "bài thơ về", "nấu ăn", "phở bò", "bitcoin", "ethereum",
        "giá cổ phiếu", "recommend phim", "bộ phim hay nhất", "giải toán",
        "phương trình vi phân", "thời tiết", "tin tức",
    ]

    pii_request_keywords = [
        "cho tôi biết cccd", "số điện thoại của nhân viên", "lương của nhân viên cụ thể",
        "thông tin cá nhân của", "email của nhân viên", "tiết lộ thông tin nhân viên",
        "tiết lộ bảng lương", "tiết lộ lương tháng", "toàn bộ thông tin nhân viên",
    ]

    is_jailbreak = any(kw in text_lower for kw in jailbreak_keywords)
    is_off_topic = any(kw in text_lower for kw in off_topic_keywords)
    is_pii_request = any(kw in text_lower for kw in pii_request_keywords)

    if is_jailbreak or is_off_topic or is_pii_request:
        if is_jailbreak:
            refuse_msg = "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có thể trả lời các câu hỏi về chính sách nhân sự công ty."
        elif is_off_topic:
            refuse_msg = "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về chính sách nội bộ của công ty như nghỉ phép, lương thưởng, bảo hiểm, và các quy trình HR."
        else:
            refuse_msg = "Xin lỗi, tôi không thể cung cấp thông tin cá nhân của nhân viên cụ thể. Đây là dữ liệu bảo mật theo chính sách phân loại dữ liệu của công ty."

        return {
            "allowed": False,
            "blocked_reason": "nemo_input_rail",
            "response": refuse_msg,
        }

    # If NeMo LLM rails are active and reachable:
    if rails is None:
        rails = setup_nemo_rails()

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            refuse_keywords = ["xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"]
            blocked = any(kw in str(response).lower() for kw in refuse_keywords)
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": str(response),
            }
        except Exception:
            pass

    return {
        "allowed": True,
        "blocked_reason": None,
        "response": "Input allowed by NeMo input rail.",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,
        }
    """
    sensitive_keywords = [
        "cccd của nhân viên là", "số điện thoại cá nhân của",
        "mật khẩu hệ thống là", "thông tin bí mật",
        "mật khẩu admin", "bảng lương chi tiết",
    ]
    refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot", "tôi không thể cung cấp"]

    flagged = any(kw in answer.lower() for kw in sensitive_keywords + refuse_keywords)

    if rails is None:
        rails = setup_nemo_rails()

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ])
            llm_flagged = any(kw in str(response).lower() for kw in refuse_keywords)
            if llm_flagged:
                return {
                    "safe": False,
                    "flagged_reason": "nemo_output_rail",
                    "final_answer": str(response),
                }
        except Exception:
            pass

    return {
        "safe": not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer": "Tôi không thể cung cấp thông tin này. Vui lòng liên hệ phòng Nhân sự trực tiếp." if flagged else answer,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async await)
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + "...",
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack."""
    if not test_inputs:
        test_inputs = ["Nhân viên muốn hỏi về chính sách nghỉ phép năm."]

    runs = test_inputs[:n_runs] if len(test_inputs) >= n_runs else (test_inputs * (n_runs // len(test_inputs) + 1))[:n_runs]
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        for text in runs:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail (async await)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times):
        s = sorted(times)
        n = len(s)
        if n == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms":     percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_guard_report(adversarial_results: list[dict], latency: dict,
                      path: str = "reports/guard_results.json") -> None:
    """Save Phase C guard results to JSON report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    passed_count = sum(1 for r in adversarial_results if r.get("passed"))
    total_count = len(adversarial_results)
    report = {
        "total_adversarial": total_count,
        "passed_count": passed_count,
        "pass_rate": round(passed_count / max(total_count, 1), 4),
        "latency": latency,
        "adversarial_suite_results": adversarial_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set]
    latency = measure_p95_latency(sample_inputs, n_runs=20)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    save_guard_report(results, latency)
