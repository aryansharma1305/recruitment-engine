import os
from typing import Dict, Set

from config import JD_SKILL_BUCKETS


def read_job_description(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        from docx import Document

        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def jd_terms() -> Set[str]:
    terms: Set[str] = set()
    for bucket in JD_SKILL_BUCKETS.values():
        terms.update(bucket["terms"])
    return terms


def summarize_jd(jd_text: str) -> Dict[str, str]:
    lowered = jd_text.lower()
    return {
        "raw_text": jd_text,
        "role": "Senior AI Engineer",
        "must_have": (
            "production embeddings/retrieval, vector or hybrid search, strong Python, "
            "ranking evaluation, and applied ML systems shipped to users"
        ),
        "nice_to_have": "LLM fine-tuning, learning-to-rank, HR-tech or marketplace experience",
        "negative_signals": (
            "pure research without deployment, keyword-only AI projects, pure consulting career, "
            "title chasing, framework demos without systems depth"
        ),
        "location": "Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR acceptable; relocation useful",
        "mentions_ranking": str("ranking" in lowered or "retrieval" in lowered),
    }
