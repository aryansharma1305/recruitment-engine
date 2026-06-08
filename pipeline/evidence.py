import re
from collections import Counter
from datetime import datetime
from typing import Dict, List

from pipeline.loader import safe_career, safe_profile, safe_signals, safe_skills


CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "ltimindtree",
}

TECH_TITLE_TERMS = {
    "ai", "ml", "machine learning", "nlp", "data scientist", "data engineer",
    "software", "backend", "platform", "search", "ranking", "recommendation",
    "applied scientist", "research engineer", "cloud",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower().strip())


def profile_text(candidate: Dict) -> str:
    profile = safe_profile(candidate)
    skills = safe_skills(candidate)
    return norm(
        " ".join(
            [
                profile.get("current_title", ""),
                profile.get("headline", ""),
                profile.get("summary", ""),
                profile.get("current_industry", ""),
                " ".join(s.get("name", "") for s in skills),
            ]
        )
    )


def career_text(candidate: Dict) -> str:
    return norm(
        " ".join(
            " ".join(
                [
                    j.get("title", ""),
                    j.get("company", ""),
                    j.get("industry", ""),
                    j.get("company_size", ""),
                    j.get("description", ""),
                ]
            )
            for j in safe_career(candidate)
        )
    )


def all_text(candidate: Dict) -> str:
    return norm(profile_text(candidate) + " " + career_text(candidate))


def has_any(text: str, terms: List[str]) -> bool:
    return any(term in text for term in terms)


def count_any(text: str, terms: List[str]) -> int:
    return sum(1 for term in terms if term in text)


def company_mix(candidate: Dict) -> Dict[str, float]:
    career = safe_career(candidate)
    if not career:
        return {"consulting_ratio": 0.0, "product_ratio": 0.0, "startup_ratio": 0.0}

    consulting = 0
    product = 0
    startup = 0
    for job in career:
        blob = norm(" ".join([job.get("company", ""), job.get("industry", ""), job.get("description", "")]))
        if any(firm in blob for firm in CONSULTING_FIRMS) or "consulting" in blob or "it services" in blob:
            consulting += 1
        if has_any(blob, ["product", "saas", "platform", "marketplace", "internet", "software", "consumer tech"]):
            product += 1
        if has_any(blob, ["startup", "founding", "series a", "early-stage", "early stage"]):
            startup += 1

    n = len(career)
    return {
        "consulting_ratio": consulting / n,
        "product_ratio": product / n,
        "startup_ratio": startup / n,
    }


def title_switching(candidate: Dict) -> Dict[str, float]:
    career = safe_career(candidate)
    durations = [float(j.get("duration_months", 0) or 0) for j in career]
    short_roles = sum(1 for d in durations if 0 < d < 18)
    role_titles = [norm(j.get("title", "")) for j in career if j.get("title")]
    title_families = Counter()
    for title in role_titles:
        if "manager" in title:
            title_families["manager"] += 1
        elif "engineer" in title:
            title_families["engineer"] += 1
        elif "scientist" in title:
            title_families["scientist"] += 1
        elif "analyst" in title:
            title_families["analyst"] += 1
        else:
            title_families[title[:20]] += 1

    return {
        "short_role_ratio": short_roles / len(career) if career else 0.0,
        "title_family_count": float(len(title_families)),
    }


def applied_ml_years(candidate: Dict) -> float:
    months = 0.0
    title_terms = [
        "machine learning", "ml engineer", "ai engineer", "nlp", "data scientist",
        "applied scientist", "research engineer", "search engineer", "ranking engineer",
        "recommendation", "recommender",
    ]
    core_terms = [
        "machine learning", "nlp", "retrieval", "ranking", "recommendation",
        "recommender", "embedding", "llm", "rag", "semantic search", "vector search",
    ]
    ownership_terms = [
        "built", "deployed", "shipped", "owned", "implemented", "trained", "inference",
        "model", "pipeline", "system", "service", "index", "evaluation", "production",
    ]
    for job in safe_career(candidate):
        title = norm(job.get("title", ""))
        desc = norm(job.get("description", ""))
        title_hit = has_any(title, title_terms)
        desc_core_hits = count_any(desc, core_terms)
        desc_ownership_hits = count_any(desc, ownership_terms)
        if title_hit or (desc_core_hits >= 2 and desc_ownership_hits >= 2):
            months += float(job.get("duration_months", 0) or 0)
    return months / 12.0


def shipped_system_score(candidate: Dict) -> float:
    text = career_text(candidate)
    delivery_terms = [
        "shipped", "launched", "deployed", "production", "real users", "at scale",
        "scaled", "owned", "built", "maintained", "improved", "optimized",
    ]
    system_terms = [
        "ranking", "recommendation", "recommender", "search", "retrieval", "embedding",
        "semantic", "vector", "index", "nlp", "matching", "candidate matching",
    ]
    eval_terms = ["ndcg", "mrr", "map", "a/b", "ab test", "experiment", "offline", "online", "evaluation"]

    delivery = count_any(text, delivery_terms)
    systems = count_any(text, system_terms)
    evals = count_any(text, eval_terms)

    if systems == 0 or delivery == 0:
        return 0.0
    score = min(65.0, systems * 12.0) + min(25.0, delivery * 5.0) + min(10.0, evals * 4.0)
    return min(100.0, score)


def availability_score(candidate: Dict) -> float:
    sig = safe_signals(candidate)
    today = datetime.now().date()
    try:
        days = (today - datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()).days
    except Exception:
        days = 365

    recency = 100 if days <= 7 else 85 if days <= 30 else 55 if days <= 90 else 25 if days <= 180 else 5
    response = float(sig.get("recruiter_response_rate", 0) or 0) * 100
    response_time = float(sig.get("avg_response_time_hours", 999) or 999)
    speed = 100 if response_time <= 12 else 85 if response_time <= 24 else 65 if response_time <= 72 else 35 if response_time <= 168 else 12
    open_to_work = 100 if sig.get("open_to_work_flag") else 35
    saved = min(100.0, float(sig.get("saved_by_recruiters_30d", 0) or 0) * 12)
    offer = sig.get("offer_acceptance_rate", -1)
    offer_score = 55.0 if offer == -1 else float(offer or 0) * 100

    return round(
        recency * 0.25
        + response * 0.25
        + speed * 0.15
        + open_to_work * 0.15
        + saved * 0.10
        + offer_score * 0.10,
        2,
    )


def current_title_is_technical(candidate: Dict) -> bool:
    title = norm(safe_profile(candidate).get("current_title", ""))
    return any(term in title for term in TECH_TITLE_TERMS)
