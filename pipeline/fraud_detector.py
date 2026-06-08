import re
import sys
import os
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.loader import safe_profile, safe_skills, safe_career, safe_education, safe_signals
from pipeline.evidence import applied_ml_years, current_title_is_technical, shipped_system_score

_CSUITE = {"cto", "ceo", "coo", "cmo", "chief", "vp of", "vice president", "director of"}
_CONSULT = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra"}
_AI_KEYWORDS = {
    "rag", "llm", "vector", "embedding", "embeddings", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "langchain", "openai", "fine-tuning", "lora", "qlora",
    "transformer", "nlp", "ranking", "recommendation", "semantic search",
}
_NON_TECH_TITLES = {
    "marketing manager", "hr manager", "accountant", "graphic designer", "content writer",
    "sales executive", "operations manager", "customer support", "project manager",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _career_months(career: List[Dict]) -> float:
    return sum(int(j.get("duration_months", 0)) for j in career) / 12.0


def _date_overlaps(career: List[Dict]) -> int:
    periods = []
    for job in career:
        try:
            s = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
            e = (datetime.strptime(job["end_date"], "%Y-%m-%d").date()
                 if job.get("end_date") else datetime.now().date())
            periods.append((s, e))
        except Exception:
            continue

    count = 0
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            overlap_start = max(periods[i][0], periods[j][0])
            overlap_end   = min(periods[i][1], periods[j][1])
            if overlap_end > overlap_start:
                months = (overlap_end.year - overlap_start.year) * 12 + (overlap_end.month - overlap_start.month)
                if months > 3:
                    count += 1
    return count


def _candidate_blob(candidate: Dict) -> str:
    profile = safe_profile(candidate)
    skills = safe_skills(candidate)
    career = safe_career(candidate)
    return _norm(
        " ".join(
            [
                profile.get("current_title", ""),
                profile.get("headline", ""),
                profile.get("summary", ""),
                " ".join(s.get("name", "") for s in skills),
                " ".join(j.get("description", "") for j in career),
            ]
        )
    )


def detect(candidate: Dict) -> Tuple[float, List[str]]:
    flags   = []
    penalty = 0.0

    profile  = safe_profile(candidate)
    career   = safe_career(candidate)
    skills   = safe_skills(candidate)
    sig      = safe_signals(candidate)
    edu_list = safe_education(candidate)

    yoe   = float(profile.get("years_of_experience", 0))
    title = _norm(profile.get("current_title", ""))
    blob  = _candidate_blob(candidate)

    career_yrs = _career_months(career)
    if yoe > career_yrs + 3.0:
        flags.append(f"Claims {yoe}yr exp; career timeline is {career_yrs:.1f}yr")
        penalty += 0.30
    if career_yrs > yoe + 4.0 and yoe > 0:
        flags.append(f"Career timeline {career_yrs:.1f}yr materially exceeds claimed {yoe}yr")
        penalty += 0.12

    n_skills = len(skills)
    if n_skills > 50:
        flags.append(f"Implausible skill count: {n_skills}")
        penalty += 0.22
    elif n_skills > 38:
        penalty += 0.08

    expert_zero = [
        s.get("name", "")
        for s in skills
        if _norm(s.get("proficiency", "")) == "expert" and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= 10:
        flags.append(f"{len(expert_zero)} expert skills with zero months used")
        penalty += 0.40
    elif len(expert_zero) >= 5:
        flags.append(f"{len(expert_zero)} expert skills with zero months used")
        penalty += 0.22

    ai_skill_names = [_norm(s.get("name", "")) for s in skills if any(k in _norm(s.get("name", "")) for k in _AI_KEYWORDS)]
    if len(ai_skill_names) >= 10 and not current_title_is_technical(candidate):
        flags.append("AI keyword-heavy skills on non-technical current title")
        penalty += 0.28
    elif len(ai_skill_names) >= 14 and shipped_system_score(candidate) < 20:
        flags.append("Many AI skills but little shipped-system evidence")
        penalty += 0.18

    if any(t in title for t in _CSUITE) and yoe < 5:
        flags.append(f"C-suite title with {yoe}yr experience")
        penalty += 0.35

    for edu in edu_list:
        sy, ey = edu.get("start_year", 0), edu.get("end_year", 0)
        if sy and ey and sy > ey:
            flags.append(f"Education end year {ey} precedes start {sy}")
            penalty += 0.25
            break
        if sy and ey and (ey - sy) > 10:
            flags.append(f"Degree duration {ey - sy} years")
            penalty += 0.10

    overlaps = _date_overlaps(career)
    if overlaps > 1:
        flags.append(f"{overlaps} overlapping employment periods")
        penalty += 0.20

    ev, ph, li = sig.get("verified_email", False), sig.get("verified_phone", False), sig.get("linkedin_connected", False)
    if not ev and not ph and not li:
        flags.append("No verification signals (email/phone/LinkedIn all unverified)")
        penalty += 0.20

    comp = sig.get("profile_completeness_score", 100)
    endr = sig.get("endorsements_received", 0)
    if comp < 20 and endr > 100:
        flags.append(f"Completeness {comp}% but {endr} endorsements")
        penalty += 0.22

    senior_kw = {"senior", "lead", "principal", "staff", "head of", "manager"}
    if any(k in title for k in senior_kw) and yoe < 2:
        flags.append(f"Senior-level title with {yoe}yr experience")
        penalty += 0.18

    if any(t in title for t in _NON_TECH_TITLES) and len(ai_skill_names) >= 6:
        flags.append("Non-technical title with many AI keywords")
        penalty += 0.22

    curr_jobs = [j for j in career if j.get("is_current")]
    if curr_jobs:
        curr_dur = curr_jobs[0].get("duration_months", 0)
        if curr_dur > yoe * 12 + 12:
            flags.append(f"Current role duration ({curr_dur}mo) exceeds claimed experience")
            penalty += 0.20

    total_skill_mo = sum(s.get("duration_months", 0) for s in skills)
    total_career_mo = career_yrs * 12
    if total_skill_mo > total_career_mo * 10 and total_career_mo > 0:
        flags.append(f"Aggregate skill months ({total_skill_mo}) implausible vs career ({int(total_career_mo)}mo)")
        penalty += 0.15

    if applied_ml_years(candidate) < 0.75 and len(ai_skill_names) >= 8:
        flags.append("AI skills claimed without matching AI/ML career history")
        penalty += 0.18

    learning_only = {
        "online courses", "side projects", "played with", "ai enthusiast", "langchain tutorial",
        "chatgpt", "emerging ai capabilities", "productivity and content creation",
        "transitioning toward ai", "curious about how ai tools",
    }
    if any(term in blob for term in learning_only) and len(ai_skill_names) >= 6 and applied_ml_years(candidate) < 1.5:
        flags.append("AI appears to be recent learning rather than production experience")
        penalty += 0.20
    elif any(term in blob for term in learning_only) and any(t in title for t in _NON_TECH_TITLES):
        flags.append("Non-technical profile with AI-learning language")
        penalty += 0.18

    companies = [_norm(j.get("company", "")) for j in career]
    if companies and all(any(f in c for f in _CONSULT) for c in companies):
        flags.append("Entire career at pure consulting firms (explicit JD disqualifier)")
        penalty += 0.25

    descriptions = [_norm(j.get("description", ""))[:180] for j in career if j.get("description")]
    repeated_desc = sum(1 for _, n in Counter(descriptions).items() if n >= 3)
    if repeated_desc:
        flags.append("Repeated career descriptions across multiple roles")
        penalty += 0.10

    try:
        days_inactive = (datetime.now().date() - datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d").date()).days
    except Exception:
        days_inactive = 365
    if days_inactive > 180 and sig.get("recruiter_response_rate", 0) < 0.10:
        flags.append("Stale profile with very low recruiter response")
        penalty += 0.12

    return round(min(1.0, penalty), 3), flags
