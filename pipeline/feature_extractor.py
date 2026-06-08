import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    JD_CORE_SKILLS, JD_SKILL_BUCKETS, SKILL_ALIASES, TITLE_SCORES, CONSULTING_FIRMS,
    RELEVANT_EDU_FIELDS, EXP_MIN, EXP_IDEAL, EXP_MAX, BEHAVIORAL_W,
)
from pipeline.loader import safe_skills, safe_career, safe_education, safe_signals, safe_profile
from pipeline.evidence import (
    all_text, applied_ml_years, availability_score, career_text, company_mix,
    current_title_is_technical, shipped_system_score, title_switching,
)
from pipeline.text_utils import norm as _norm

try:
    from rapidfuzz import fuzz, process as rfprocess
    _FUZZY = True
except ImportError:
    _FUZZY = False

_PROF_W = {"expert": 1.0, "advanced": 0.80, "intermediate": 0.55, "beginner": 0.25}


def _canonical(name: str) -> str:
    n = _norm(name)
    return SKILL_ALIASES.get(n, n)


def _title_score(title: str) -> float:
    t = _norm(title)
    if t in TITLE_SCORES:
        return float(TITLE_SCORES[t])
    for k, v in TITLE_SCORES.items():
        if k in t:
            return float(v)
    if _FUZZY:
        hit = rfprocess.extractOne(t, list(TITLE_SCORES.keys()), scorer=fuzz.ratio)
        if hit and hit[1] >= 75:
            return float(TITLE_SCORES[hit[0]]) * hit[1] / 100
    return 20.0


def _candidate_text(candidate: Dict) -> str:
    return all_text(candidate)


def _term_matches(term: str, candidate_terms: Dict[str, float], text: str) -> float:
    term = _canonical(term)
    if term in candidate_terms:
        return candidate_terms[term]
    if term in text:
        return 0.25
    if _FUZZY and candidate_terms:
        hit = rfprocess.extractOne(term, list(candidate_terms.keys()), scorer=fuzz.ratio)
        if hit and hit[1] >= 86:
            return candidate_terms[hit[0]] * hit[1] / 100 * 0.85
    return 0.0


def skill_match(candidate: Dict) -> Tuple[float, int]:
    skills = safe_skills(candidate)
    signals = safe_signals(candidate)
    assessments = {_canonical(k): float(v) for k, v in signals.get("skill_assessment_scores", {}).items()}

    cmap: Dict[str, float] = {}
    for s in skills:
        key = _canonical(s.get("name", ""))
        w = _PROF_W.get(s.get("proficiency", "beginner"), 0.25)
        months = float(s.get("duration_months", 0) or 0)
        if months >= 36:
            w = min(1.0, w + 0.08)
        elif months >= 18:
            w = min(1.0, w + 0.04)
        if key not in cmap or w > cmap[key]:
            cmap[key] = w

    text = _candidate_text(candidate)
    career_blob = career_text(candidate)
    total, matched_buckets = 0.0, 0

    for bucket in JD_SKILL_BUCKETS.values():
        terms = [_canonical(t) for t in bucket["terms"]]
        best = max(_term_matches(t, cmap, text) for t in terms)
        if any(t in career_blob for t in terms):
            best = max(best, 0.48)

        for assessed_name, assessed_score in assessments.items():
            if any(_term_matches(t, {assessed_name: 1.0}, assessed_name) for t in terms):
                best = min(1.0, best + assessed_score / 100 * 0.12)

        if best >= 0.45:
            matched_buckets += 1
        total += best * float(bucket["weight"])

    score = min(100.0, total * 100)
    return round(score, 2), matched_buckets


def experience(candidate: Dict) -> float:
    profile = safe_profile(candidate)
    career  = safe_career(candidate)
    yoe = float(profile.get("years_of_experience", 0))

    if yoe < EXP_MIN:
        score = max(0.0, (yoe / EXP_MIN) * 38)
    elif yoe <= EXP_IDEAL:
        score = 38 + (yoe - EXP_MIN) / (EXP_IDEAL - EXP_MIN) * 47
    elif yoe <= EXP_MAX:
        score = 85 - (yoe - EXP_IDEAL) / (EXP_MAX - EXP_IDEAL) * 18
    else:
        score = max(48.0, 67 - (yoe - EXP_MAX) * 2.5)

    companies = [_norm(j.get("company", "")) for j in career]
    if companies and all(any(f in c for f in CONSULTING_FIRMS) for c in companies):
        score *= 0.68

    aml_years = applied_ml_years(candidate)
    if aml_years >= 4:
        score = min(100.0, score + 8)
    elif aml_years < 1 and yoe >= 5:
        score *= 0.82

    return round(min(100.0, score), 2)


def _desc_score(career: List[Dict]) -> float:
    ai_kw = [
        "machine learning", "deep learning", "neural", "embedding", "vector",
        "ranking", "recommendation", "retrieval", "nlp", "language model",
        "transformer", "training", "inference", "pytorch", "tensorflow",
        "deploy", "mlops", "search", "semantic", "rag", "fine-tun", "llm",
        "a/b", "evaluation", "production ml", "feature store",
    ]
    blob = " ".join(_norm(j.get("description", "")) for j in career)
    if not blob:
        return 0.0
    hits = sum(1 for kw in ai_kw if kw in blob)
    return min(100.0, hits / len(ai_kw) * 160)


def career_relevance(candidate: Dict) -> float:
    profile = safe_profile(candidate)
    career  = safe_career(candidate)
    curr_score = _title_score(profile.get("current_title", ""))

    past_scores = []
    for i, job in enumerate(career):
        w = 1.0 / (i + 1)
        past_scores.append(_title_score(job.get("title", "")) * w)
    past_avg = sum(past_scores) / len(past_scores) if past_scores else 0.0

    desc = _desc_score(career)

    shipped = shipped_system_score(candidate)
    aml_years = applied_ml_years(candidate)

    base = curr_score * 0.32 + past_avg * 0.22 + desc * 0.20 + shipped * 0.20 + min(100.0, aml_years / 5 * 100) * 0.06
    if len(career) >= 2:
        sorted_c = sorted(career, key=lambda j: j.get("start_date", "2000-01-01"))
        if _title_score(sorted_c[-1].get("title", "")) > _title_score(sorted_c[0].get("title", "")):
            base = min(100.0, base + 5)
    return round(min(100.0, base), 2)


def product_impact(candidate: Dict) -> float:
    career = safe_career(candidate)
    text = _candidate_text(candidate)
    delivery_terms = ["shipped", "launched", "deployed", "production", "real users", "scale", "a/b", "experiment"]
    technical_terms = ["ranking", "recommendation", "recommender", "search", "retrieval", "embedding", "nlp", "ml", "machine learning"]
    research_only_terms = ["research intern", "academic lab", "publication only", "survey paper"]
    framework_demo_terms = [
        "langchain tutorial", "chatbot demo", "prompt engineering bootcamp",
        "online courses", "side projects", "ai enthusiast", "grow my ai capabilities",
    ]

    career_blob = career_text(candidate)
    delivery_hits = sum(1 for term in delivery_terms if term in career_blob)
    technical_hits = sum(1 for term in technical_terms if term in career_blob)
    score = min(100.0, delivery_hits / 4 * 45 + technical_hits / 5 * 45)
    score = max(score, shipped_system_score(candidate))

    if technical_hits == 0:
        score = min(score, 30.0)
    elif delivery_hits == 0:
        score = min(score, 55.0)

    product_roles = 0
    consulting_roles = 0
    for job in career:
        blob = _norm(" ".join([job.get("industry", ""), job.get("company", ""), job.get("description", "")]))
        if any(firm in blob for firm in CONSULTING_FIRMS):
            consulting_roles += 1
        if any(term in blob for term in ("product", "saas", "platform", "marketplace", "internet", "software")):
            product_roles += 1

    if product_roles:
        score = min(100.0, score + min(20.0, product_roles * 6))
    if not current_title_is_technical(candidate):
        score *= 0.55
    if career and consulting_roles == len(career):
        score *= 0.55
    if any(term in text for term in research_only_terms):
        score *= 0.70
    if any(term in text for term in framework_demo_terms):
        score *= 0.55
    return round(score, 2)


def location_logistics(candidate: Dict) -> float:
    profile = safe_profile(candidate)
    sig = safe_signals(candidate)
    location = _norm(" ".join([profile.get("location", ""), profile.get("country", "")]))

    preferred = ["pune", "noida"]
    acceptable = ["hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram", "bangalore", "bengaluru"]

    if any(city in location for city in preferred):
        loc = 100.0
    elif any(city in location for city in acceptable):
        loc = 82.0
    elif profile.get("country", "").lower() == "india" and sig.get("willing_to_relocate"):
        loc = 72.0
    elif profile.get("country", "").lower() == "india":
        loc = 55.0
    else:
        loc = 25.0

    notice = sig.get("notice_period_days", 90)
    notice_score = 100 if notice <= 30 else 70 if notice <= 60 else 42 if notice <= 90 else 18
    return round(loc * 0.65 + notice_score * 0.35, 2)


def fit_penalty(candidate: Dict) -> float:
    profile = safe_profile(candidate)
    career = safe_career(candidate)
    text = _candidate_text(candidate)
    title = _norm(profile.get("current_title", ""))
    penalty = 0.0

    non_technical_titles = {
        "project manager", "marketing manager", "hr manager", "accountant",
        "graphic designer", "content writer", "sales executive", "customer support",
        "operations manager", "business analyst",
    }
    if any(t in title for t in non_technical_titles):
        penalty += 0.34
    elif not current_title_is_technical(candidate):
        penalty += 0.08

    learning_only_terms = [
        "ai enthusiast", "self-learner", "self learner", "online courses",
        "side projects", "played with", "experimenting with langchain",
        "grow my ai capabilities", "building competence on the ml side",
        "chatgpt", "emerging ai capabilities", "curious about how ai tools",
        "productivity and content creation", "transitioning toward ai",
    ]
    if any(term in text for term in learning_only_terms):
        penalty += 0.22

    cv_speech_robotics_terms = ["computer vision", "image classification", "speech recognition", "robotics"]
    ir_terms = ["retrieval", "ranking", "recommendation", "search", "nlp", "embedding"]
    if any(term in text for term in cv_speech_robotics_terms) and not any(term in text for term in ir_terms):
        penalty += 0.14

    companies = [_norm(j.get("company", "")) for j in career]
    if companies and all(any(f in c for f in CONSULTING_FIRMS) for c in companies):
        penalty += 0.18

    mix = company_mix(candidate)
    if mix["consulting_ratio"] >= 0.80 and mix["product_ratio"] < 0.20:
        penalty += 0.10

    switching = title_switching(candidate)
    if switching["short_role_ratio"] >= 0.55 and switching["title_family_count"] >= 3:
        penalty += 0.10

    if applied_ml_years(candidate) < 1.0 and _title_score(title) < 75:
        penalty += 0.18

    yoe = float(profile.get("years_of_experience", 0) or 0)
    if yoe > 12 and _title_score(title) < 65:
        penalty += 0.08

    return round(min(0.75, penalty), 3)


def education(candidate: Dict) -> float:
    tiers = {"tier_1": 100, "tier_2": 80, "tier_3": 60, "tier_4": 40, "unknown": 45}
    deg_bonus = {"phd": 15, "ph.d": 15, "m.tech": 8, "m.e.": 8, "m.sc": 6, "msc": 6}

    best = 0.0
    for edu in safe_education(candidate):
        base = float(tiers.get(edu.get("tier", "unknown"), 45))
        field = _norm(edu.get("field_of_study", ""))
        deg   = _norm(edu.get("degree", ""))
        if any(f in field for f in RELEVANT_EDU_FIELDS):
            base += 15
        for k, v in deg_bonus.items():
            if k in deg:
                base += v
                break
        best = max(best, min(100.0, base))
    return round(best if best > 0 else 30.0, 2)


def behavioral(candidate: Dict) -> float:
    sig   = safe_signals(candidate)
    today = datetime.now().date()
    comp  = {}

    comp["open_to_work"]   = 100.0 if sig.get("open_to_work_flag") else 20.0

    gh = sig.get("github_activity_score", -1)
    comp["github_activity"] = float(gh) if gh != -1 else 15.0

    comp["response_rate"]       = sig.get("recruiter_response_rate", 0) * 100
    comp["interview_completion"] = sig.get("interview_completion_rate", 0) * 100
    comp["profile_completeness"] = sig.get("profile_completeness_score", 0)

    try:
        days = (today - datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()).days
        comp["recency"] = 100 if days <= 7 else 85 if days <= 30 else 60 if days <= 90 else 30 if days <= 180 else 5
    except Exception:
        comp["recency"] = 30.0

    v = sum([
        50 * int(bool(sig.get("verified_email"))),
        30 * int(bool(sig.get("verified_phone"))),
        20 * int(bool(sig.get("linkedin_connected"))),
    ])
    comp["verification"] = float(v)

    notice = sig.get("notice_period_days", 90)
    comp["notice_period"] = 100 if notice <= 15 else 85 if notice <= 30 else 60 if notice <= 60 else 30 if notice <= 90 else 10

    base = sum(comp.get(k, 0) * w for k, w in BEHAVIORAL_W.items())
    total = base * 0.65 + availability_score(candidate) * 0.35
    return round(min(100.0, total), 2)


def extract(candidate: Dict) -> Dict:
    profile  = safe_profile(candidate)
    sig      = safe_signals(candidate)
    sk, n_sk = skill_match(candidate)

    return {
        "candidate_id":    candidate.get("candidate_id", ""),
        "name":            profile.get("anonymized_name", ""),
        "title":           profile.get("current_title", ""),
        "yoe":             profile.get("years_of_experience", 0),
        "country":         profile.get("country", ""),
        "skill_match":     sk,
        "experience":      experience(candidate),
        "career_relevance": career_relevance(candidate),
        "product_impact":   product_impact(candidate),
        "applied_ml_years":  round(applied_ml_years(candidate), 2),
        "shipped_system":    round(shipped_system_score(candidate), 2),
        "availability":     availability_score(candidate),
        "location_logistics": location_logistics(candidate),
        "fit_penalty":      fit_penalty(candidate),
        "education":       education(candidate),
        "behavioral":      behavioral(candidate),
        "n_matched_skills": n_sk,
        "github":          sig.get("github_activity_score", -1),
        "open_to_work":    sig.get("open_to_work_flag", False),
        "notice_days":     sig.get("notice_period_days", 90),
        "response_rate":   sig.get("recruiter_response_rate", 0),
        "summary":         profile.get("summary", "")[:400],
    }
