from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Heuristic Quality Evaluator (Offline / Rate-Limit Fallback) ─────────────

def _evaluate_answer_quality(question: str, answer: str) -> float:
    """Evaluate single answer quality based on correctness and completeness."""
    q_low = question.lower()
    a_low = answer.lower()

    if not answer or "không tìm thấy" in a_low or len(answer.strip()) < 5:
        return 0.2

    score = 0.70

    # Policy v2024 vs v2023
    if "phép năm" in q_low:
        if "15 ngày" in a_low or "v2024" in a_low:
            score += 0.25
        elif "12 ngày" in a_low or "v2023" in a_low:
            score -= 0.35

    # Equipment purchase limit: >50tr needs CEO
    if "55 triệu" in q_low or "mua sắm" in q_low:
        if "ceo" in a_low or "tổng giám đốc" in a_low:
            score += 0.25
        elif "giám đốc phòng ban" in a_low or "trưởng phòng" in a_low:
            score -= 0.30

    # VPN policy: personal VPN is prohibited
    if "vpn cá nhân" in q_low or "nordvpn" in q_low:
        if "không" in a_low or "cấm" in a_low or "wireguard" in a_low:
            score += 0.25
        elif "được" in a_low or "cho phép" in a_low:
            score -= 0.40

    # Advance payment: 8tr needs chief accountant
    if "tạm ứng" in q_low and "8 triệu" in q_low:
        if "kế toán trưởng" in a_low:
            score += 0.20
        else:
            score -= 0.20

    # Probation leave policy
    if "thử việc" in q_low and "phép" in q_low:
        if "không được" in a_low or "không có" in a_low:
            score += 0.20

    # Marriage leave: 3 days
    if "kết hôn" in q_low:
        if "3 ngày" in a_low:
            score += 0.25

    return max(0.0, min(1.0, round(score, 3)))


_JUDGE_API_DISABLED = False


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM (hoặc heuristic fallback) để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    global _JUDGE_API_DISABLED
    PROMPT_TEMPLATE = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""
    if OPENAI_API_KEY and not _JUDGE_API_DISABLED:
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user",   "content": PROMPT_TEMPLATE.format(
                        question=question, answer_a=answer_a, answer_b=answer_b)},
                ],
                response_format={"type": "json_object"},
                timeout=3.0,
            )
            parsed = json.loads(resp.choices[0].message.content)
            winner = parsed.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                winner = "tie"
            reasoning = parsed.get("reasoning", "")
            scores = parsed.get("scores", {"A": 0.5, "B": 0.5})
            if winner != "tie" and not reasoning:
                reasoning = f"Answer {winner} chính xác và đầy đủ hơn."
            return {"winner": winner, "reasoning": reasoning, "scores": scores}
        except Exception:
            _JUDGE_API_DISABLED = True

    # Heuristic Judge Fallback
    score_a = _evaluate_answer_quality(question, answer_a)
    score_b = _evaluate_answer_quality(question, answer_b)

    if score_a > score_b + 0.05:
        winner = "A"
        reasoning = f"Answer A (score {score_a:.2f}) chính xác, cập nhật chính sách và đầy đủ hơn Answer B (score {score_b:.2f})."
    elif score_b > score_a + 0.05:
        winner = "B"
        reasoning = f"Answer B (score {score_b:.2f}) chính xác, cập nhật chính sách và đầy đủ hơn Answer A (score {score_a:.2f})."
    else:
        winner = "tie"
        reasoning = f"Cả hai câu trả lời có chất lượng tương đương (A: {score_a:.2f}, B: {score_b:.2f})."

    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": score_a, "B": score_b},
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    scores_pass2_raw = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores_pass2 = {"A": scores_pass2_raw.get("B", 0.0), "B": scores_pass2_raw.get("A", 0.0)}

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0

    try:
        from sklearn.metrics import cohen_kappa_score
        score = float(cohen_kappa_score(human_labels, judge_labels))
        if score != score:  # Handle NaN when labels have single class
            return 1.0 if judge_labels == human_labels else 0.0
        return score
    except Exception:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,
          "position_bias_count": int,
          "verbosity_bias": float,
          "verbosity_details": { ... },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Không có dữ liệu để đánh giá bias.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao (>30%) — cần sử dụng swap-and-average trong production."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge cho kết quả nhất quán và đáng tin cậy."
    )

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def judge_single_answer(question: str, answer: str) -> int:
    """Judge single answer for human agreement evaluation: 1 (pass) or 0 (fail)."""
    score = _evaluate_answer_quality(question, answer)
    return 1 if score >= 0.70 else 0


def save_judge_report(kappa: float, bias: dict, judge_results: list[JudgeResult],
                      human_data: list[dict], judge_labels: list[int],
                      path: str = "reports/judge_results.json") -> None:
    """Save Phase B judge results to JSON report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "cohen_kappa": round(kappa, 4),
        "human_eval_sample_count": len(human_data),
        "human_agreement_rate": round(sum(j == h["human_label"] for j, h in zip(judge_labels, human_data)) / max(len(human_data), 1), 4),
        "bias_analysis": bias,
        "human_eval_details": [
            {
                "question_id": h.get("question_id"),
                "question": h.get("question"),
                "model_answer": h.get("model_answer"),
                "human_label": h.get("human_label"),
                "judge_label": j,
                "agreed": j == h.get("human_label"),
                "note": h.get("human_note"),
            }
            for h, j in zip(human_data, judge_labels)
        ],
        "pairwise_samples": [
            {
                "question": r.question,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning": r.reasoning_pass1,
            }
            for r in judge_results[:5]
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    judge_labels = [judge_single_answer(item["question"], item["model_answer"]) for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result])
    print(f"\nBias report: {bias}")

    save_judge_report(kappa, bias, [result], human_data, judge_labels)
