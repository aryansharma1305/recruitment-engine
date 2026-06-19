import re
import io
from typing import Dict, List, Optional
from datetime import datetime

try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False

try:
    from docx import Document as DocxDocument
    _DOCX = True
except ImportError:
    _DOCX = False


_SECTION_HEADERS = {
    "experience": re.compile(
        r"^(work experience|experience|employment|career|professional experience|work history)[\s:]*$",
        re.I
    ),
    "education": re.compile(
        r"^(education|academic background|qualifications|educational background)[\s:]*$",
        re.I
    ),
    "skills": re.compile(
        r"^(skills|technical skills|core skills|key skills|competencies|technologies)[\s:]*$",
        re.I
    ),
    "summary": re.compile(
        r"^(summary|profile|about|objective|about me|professional summary)[\s:]*$",
        re.I
    ),
}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_TECH_SKILLS = {
    "python", "pytorch", "tensorflow", "keras", "numpy", "pandas", "scikit-learn",
    "sklearn", "sql", "spark", "kafka", "docker", "kubernetes", "aws", "gcp", "azure",
    "fastapi", "flask", "django", "redis", "mongodb", "postgresql", "elasticsearch",
    "opensearch", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "langchain", "llm", "rag", "embedding", "embeddings", "transformers", "huggingface",
    "bert", "gpt", "llama", "mistral", "fine-tuning", "lora", "qlora", "peft",
    "pytorch lightning", "xgboost", "lightgbm", "catboost", "mlops", "mlflow",
    "wandb", "airflow", "dbt", "hadoop", "hive", "cassandra", "neo4j",
    "ranking", "recommendation", "retrieval", "nlp", "computer vision", "opencv",
    "java", "scala", "go", "rust", "javascript", "typescript", "react", "node.js",
    "graphql", "rest api", "grpc", "git", "linux", "bash", "terraform", "ci/cd",
    "machine learning", "deep learning", "neural network", "reinforcement learning",
    "vector database", "semantic search", "a/b testing", "feature engineering",
    "data pipeline", "etl", "feature store", "model serving", "bm25",
}

_TITLE_PATTERNS = [
    re.compile(r"\b(senior|staff|principal|lead|junior|sr\.?|jr\.?)\s+", re.I),
    re.compile(r"\b(engineer|scientist|researcher|analyst|architect|developer|manager|director|vp|head)\b", re.I),
]

_DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,\.]+(\d{4})\b"
    r"|(\d{4})\b",
    re.I
)

_PRESENT_RE = re.compile(r"\b(present|current|now|ongoing)\b", re.I)


def _extract_text_from_pdf(content: bytes) -> str:
    if not _PYMUPDF:
        return ""
    doc = fitz.open(stream=content, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_text_from_docx(content: bytes) -> str:
    if not _DOCX:
        return ""
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(content: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return _extract_text_from_pdf(content)
    if ext in ("docx", "doc"):
        return _extract_text_from_docx(content)
    return content.decode("utf-8", errors="ignore")


def _split_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {
        "header": [], "summary": [], "experience": [],
        "education": [], "skills": [], "other": []
    }
    current = "header"
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        matched = None
        for section, pattern in _SECTION_HEADERS.items():
            if pattern.match(stripped):
                matched = section
                break
        if matched:
            current = matched
        else:
            sections[current].append(stripped)
    return sections


def _parse_name(header_lines: List[str]) -> str:
    for line in header_lines[:4]:
        words = line.strip().split()
        if 1 < len(words) <= 5 and all(w[0].isupper() for w in words if w):
            if not any(c in line for c in ["@", "http", "+", "|"]):
                return line.strip()
    return "Candidate"


def _parse_contact(header_lines: List[str]) -> Dict:
    email = ""
    for line in header_lines:
        m = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", line)
        if m:
            email = m.group()
            break
    return {"email": email}


def _parse_skills_section(lines: List[str]) -> List[Dict]:
    skills = []
    for line in lines:
        parts = re.split(r"[,|•·\t]+", line)
        for part in parts:
            name = part.strip().strip("-").strip()
            if 1 < len(name) < 60:
                name_lower = name.lower()
                proficiency = "intermediate"
                if name_lower in _TECH_SKILLS:
                    proficiency = "advanced"
                skills.append({
                    "name": name,
                    "proficiency": proficiency,
                    "endorsements": 0,
                    "duration_months": 0,
                })
    return skills


def _parse_date_range(text: str):
    matches = list(_DATE_RE.finditer(text))
    has_present = bool(_PRESENT_RE.search(text))

    dates = []
    for m in matches:
        if m.group(1):
            month = _MONTH_MAP.get(m.group(1)[:3].lower(), 1)
            year = int(m.group(2))
            dates.append(datetime(year, month, 1))
        elif m.group(3):
            year = int(m.group(3))
            if 1980 <= year <= datetime.now().year:
                dates.append(datetime(year, 1, 1))

    dates.sort()
    if len(dates) >= 2:
        start = dates[0]
        end = None if has_present else dates[-1]
    elif len(dates) == 1:
        start = dates[0]
        end = None if has_present else None
    else:
        return None, None, True

    is_current = end is None
    return start, end, is_current


def _parse_experience_section(lines: List[str]) -> List[Dict]:
    jobs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        has_date = bool(_DATE_RE.search(line)) or bool(_PRESENT_RE.search(line))

        title = ""
        company = ""

        if has_date:
            start, end, is_current = _parse_date_range(line)
            title_part = re.sub(r"[\|•·].*", "", line).strip()
            title_part = _DATE_RE.sub("", title_part).strip()
            title_part = _PRESENT_RE.sub("", title_part).strip(" -–—|·•")

            if i + 1 < len(lines) and not bool(_DATE_RE.search(lines[i + 1])):
                company = lines[i + 1].strip()
                i += 1
            elif "|" in line or "," in line:
                parts = re.split(r"[|,]", title_part, maxsplit=1)
                title = parts[0].strip()
                company = parts[1].strip() if len(parts) > 1 else ""
            else:
                title = title_part
        elif i + 1 < len(lines) and bool(_DATE_RE.search(lines[i + 1])):
            title = line.strip()
            start, end, is_current = _parse_date_range(lines[i + 1])
            if i + 2 < len(lines) and not bool(_DATE_RE.search(lines[i + 2])):
                company = lines[i + 2].strip()
                i += 2
            else:
                i += 1
        else:
            i += 1
            continue

        desc_lines = []
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if bool(_DATE_RE.search(next_line)) or bool(_PRESENT_RE.search(next_line)):
                break
            if any(p.match(_SECTION_HEADERS.get(s, re.compile("$")).pattern or "$")
                   for s in _SECTION_HEADERS):
                break
            desc_lines.append(next_line)
            i += 1

        if not start:
            continue

        duration = 0
        if start and end:
            duration = max(1, (end.year - start.year) * 12 + (end.month - start.month))
        elif start and is_current:
            duration = max(1, (datetime.now().year - start.year) * 12 + (datetime.now().month - start.month))

        jobs.append({
            "company": company or "Unknown Company",
            "title": title or "Engineer",
            "start_date": start.strftime("%Y-%m-%d") if start else "2020-01-01",
            "end_date": end.strftime("%Y-%m-%d") if end else None,
            "duration_months": duration,
            "is_current": is_current,
            "industry": "",
            "company_size": "",
            "description": " ".join(desc_lines),
        })

    return jobs


def _parse_education_section(lines: List[str]) -> List[Dict]:
    edu = []
    i = 0
    while i < len(lines):
        line = lines[i]
        degree_kw = re.compile(
            r"\b(b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc|b\.?sc|m\.?sc|phd|ph\.d|"
            r"bachelor|master|doctorate|mba|b\.?a|m\.?a|bca|mca)\b", re.I
        )
        if degree_kw.search(line):
            institution = ""
            field = ""
            years = list(re.findall(r"\b(19|20)\d{2}\b", line))
            if not years and i + 1 < len(lines):
                years = list(re.findall(r"\b(19|20)\d{2}\b", lines[i + 1]))
                institution = lines[i + 1].strip()
                i += 1

            degree_match = degree_kw.search(line)
            degree = degree_match.group() if degree_match else "Degree"
            field_match = re.search(r"(?:in|of)\s+([A-Za-z\s]+)", line)
            if field_match:
                field = field_match.group(1).strip()

            start_year = int(years[0]) if years else 0
            end_year = int(years[1]) if len(years) > 1 else (start_year + 4 if start_year else 0)

            edu.append({
                "institution": institution or "University",
                "degree": degree,
                "field_of_study": field or "Engineering",
                "start_year": start_year,
                "end_year": end_year,
                "grade": "",
                "tier": "unknown",
            })
        i += 1
    return edu


def _estimate_yoe(jobs: List[Dict]) -> float:
    total = sum(j.get("duration_months", 0) for j in jobs)
    return round(total / 12.0, 1)


def _infer_current_title(jobs: List[Dict]) -> str:
    for job in jobs:
        if job.get("is_current"):
            return job.get("title", "")
    return jobs[0].get("title", "Engineer") if jobs else "Engineer"


def _infer_current_company(jobs: List[Dict]) -> str:
    for job in jobs:
        if job.get("is_current"):
            return job.get("company", "")
    return jobs[0].get("company", "") if jobs else ""


def parse_resume(content: bytes, filename: str, candidate_index: int) -> Dict:
    raw_text = extract_text(content, filename)
    lines = [l.strip() for l in raw_text.splitlines()]

    sections = _split_sections(lines)

    name = _parse_name(sections["header"])
    jobs = _parse_experience_section(sections["experience"])
    edu = _parse_education_section(sections["education"])
    raw_skills = _parse_skills_section(sections["skills"])

    # Fallback: mine skills from full text if none found
    if len(raw_skills) < 3:
        full_lower = raw_text.lower()
        for skill in _TECH_SKILLS:
            if skill in full_lower and not any(s["name"].lower() == skill for s in raw_skills):
                raw_skills.append({
                    "name": skill.title(),
                    "proficiency": "intermediate",
                    "endorsements": 0,
                    "duration_months": 0,
                })

    summary_text = " ".join(sections["summary"][:10])
    yoe = _estimate_yoe(jobs)
    current_title = _infer_current_title(jobs)
    current_company = _infer_current_company(jobs)

    cid = f"UPLOAD_{candidate_index:04d}"

    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": name,
            "headline": current_title,
            "summary": summary_text,
            "location": "",
            "country": "",
            "years_of_experience": yoe,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": "",
            "current_industry": "",
        },
        "career_history": jobs,
        "education": edu,
        "skills": raw_skills,
        "certifications": [],
        "redrob_signals": {
            "open_to_work_flag": True,
            "github_activity_score": -1,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 48,
            "interview_completion_rate": 0.7,
            "offer_acceptance_rate": -1,
            "last_active_date": datetime.now().strftime("%Y-%m-%d"),
            "saved_by_recruiters_30d": 0,
            "profile_completeness_score": 60,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
            "notice_period_days": 60,
            "willing_to_relocate": True,
            "skill_assessment_scores": {},
            "endorsements_received": 0,
        },
        "_source_file": filename,
    }
