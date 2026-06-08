import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    ALLOW_HOSTED_LLM_EXPLANATIONS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

try:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENROUTER_API_KEY or "none", base_url=OPENROUTER_BASE_URL)
    _LLM_READY = bool(OPENROUTER_API_KEY and ALLOW_HOSTED_LLM_EXPLANATIONS)
except ImportError:
    _LLM_READY = False


def _rule_based(rec: Dict, sim: float) -> str:
    parts = []
    yoe   = rec.get("yoe", 0)
    title = rec.get("title", "")
    n_sk  = rec.get("n_matched_skills", 0)
    gh    = rec.get("github", -1)
    nd    = rec.get("notice_days", 90)
    rr    = rec.get("response_rate", 0)
    shipped = rec.get("shipped_system", 0)
    aml_years = rec.get("applied_ml_years", 0)
    availability = rec.get("availability", 0)

    parts.append(f"{yoe}yr {title}")
    if aml_years >= 4:
        parts.append(f"{aml_years:.1f}yr applied ML evidence")
    if shipped >= 80:
        parts.append("strong shipped retrieval/ranking-system evidence")
    elif shipped >= 55:
        parts.append("credible production ML/search evidence")
    if n_sk >= 5:
        parts.append(f"covers {n_sk} JD capability areas")
    if gh > 60:
        parts.append(f"active GitHub ({gh:.0f}/100)")
    if nd <= 30:
        parts.append("immediate/short-notice joiner")
    if rr >= 0.7:
        parts.append("highly responsive to recruiters")
    elif availability >= 75:
        parts.append("strong platform availability signals")
    parts.append(f"semantic match {sim:.2f}")

    return "; ".join(parts) + "."


def _llm_reason(rec: Dict, sim: float) -> str:
    prompt = (
        "You are a senior technical recruiter writing a one-sentence hiring note.\n"
        f"Role: Senior AI Engineer (embeddings, vector DBs, ranking/retrieval, Python)\n"
        f"Candidate: {rec.get('yoe')}yr {rec.get('title')}, "
        f"{rec.get('n_matched_skills')} core skill matches, "
        f"semantic fit {sim:.2f}, "
        f"GitHub {rec.get('github')}, "
        f"notice {rec.get('notice_days')}d, "
        f"response rate {rec.get('response_rate', 0):.0%}.\n"
        f"Summary: {rec.get('summary', '')}\n\n"
        "Write one concise sentence (max 30 words) explaining why this candidate fits. "
        "Be specific. No fluff."
    )
    try:
        resp = _client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _rule_based(rec, sim)


def explain(rec: Dict, sim: float) -> str:
    if _LLM_READY:
        return _llm_reason(rec, sim)
    return _rule_based(rec, sim)


def explain_batch(records: List[Dict], sims: Dict[str, float]) -> Dict[str, str]:
    return {
        rec["candidate_id"]: explain(rec, sims.get(rec["candidate_id"], 0.0))
        for rec in records
    }
