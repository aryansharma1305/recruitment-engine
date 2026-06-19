import re
from typing import Dict, List, Set

_KNOWN_SKILLS: List[str] = [
    # Core ML/AI
    "python", "pytorch", "tensorflow", "keras", "scikit-learn", "numpy", "pandas",
    "machine learning", "deep learning", "neural network", "reinforcement learning",
    "nlp", "natural language processing", "computer vision", "transformers",
    "huggingface", "bert", "gpt", "llama", "mistral", "llm", "large language model",
    "fine-tuning", "lora", "qlora", "peft", "rlhf",
    # Retrieval / Search / Ranking
    "embedding", "embeddings", "sentence-transformers", "bge", "e5",
    "vector database", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "elasticsearch", "opensearch", "bm25", "hybrid search", "semantic search",
    "dense retrieval", "sparse retrieval", "retrieval augmented generation", "rag",
    "ranking", "reranking", "learning to rank", "recommendation", "recommender",
    "ndcg", "mrr", "a/b testing", "online evaluation", "offline evaluation",
    # Engineering
    "mlops", "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "ci/cd",
    "airflow", "kafka", "spark", "redis", "postgresql", "mongodb", "sql",
    "fastapi", "flask", "django", "grpc", "rest api", "graphql",
    "feature store", "mlflow", "wandb", "model deployment", "model serving",
    "xgboost", "lightgbm", "catboost",
    # Product / HR tech
    "ats", "hrtech", "recruiting", "talent acquisition", "applicant tracking",
    "product management", "roadmap", "user research", "wireframing", "figma",
    # General Engineering
    "java", "scala", "go", "rust", "javascript", "typescript", "react", "node.js",
    "git", "linux", "bash", "data structures", "system design", "distributed systems",
]

_EXPERIENCE_RE = re.compile(
    r"(\d+)\+?\s*(?:to|-)\s*(\d+)?\s*years?|(\d+)\+?\s*years?",
    re.I
)

_TITLE_PATTERNS = re.compile(
    r"\b(senior|staff|principal|lead|junior|jr|sr|mid[- ]level|entry[- ]level)?\s*"
    r"(software|ml|ai|machine learning|data|nlp|backend|frontend|full.?stack|"
    r"platform|devops|sre|security|product|ux|ui|research|applied|cloud)?\s*"
    r"(engineer|scientist|developer|analyst|architect|researcher|manager|"
    r"director|designer|consultant|specialist|lead|head)\b",
    re.I
)

_LOCATION_RE = re.compile(
    r"\b(bangalore|bengaluru|mumbai|delhi|hyderabad|pune|noida|chennai|"
    r"gurgaon|gurugram|india|remote|hybrid|us|usa|uk|singapore)\b",
    re.I
)


def _extract_min_experience(text: str) -> float:
    for m in _EXPERIENCE_RE.finditer(text):
        if m.group(1):
            return float(m.group(1))
        if m.group(3):
            return float(m.group(3))
    return 0.0


def _extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found: List[str] = []
    seen: Set[str] = set()
    for skill in _KNOWN_SKILLS:
        if skill in text_lower and skill not in seen:
            found.append(skill)
            seen.add(skill)
    return found


def _extract_preferred_titles(text: str) -> List[str]:
    return [m.group(0).strip() for m in _TITLE_PATTERNS.finditer(text)][:6]


def _extract_locations(text: str) -> List[str]:
    return list({m.group(0).lower() for m in _LOCATION_RE.finditer(text)})


def _build_skill_buckets(skills: List[str]) -> Dict:
    retrieval_terms = {
        "embedding", "embeddings", "sentence-transformers", "bge", "e5",
        "dense retrieval", "semantic search", "retrieval augmented generation", "rag"
    }
    vector_terms = {
        "vector database", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "chroma", "elasticsearch", "opensearch", "hybrid search", "bm25"
    }
    ranking_terms = {
        "ranking", "reranking", "learning to rank", "recommendation", "recommender",
        "ndcg", "mrr", "a/b testing"
    }
    ml_terms = {
        "python", "pytorch", "tensorflow", "scikit-learn", "machine learning", "deep learning"
    }
    llm_terms = {
        "llm", "large language model", "fine-tuning", "lora", "qlora", "peft",
        "transformers", "huggingface", "nlp", "natural language processing"
    }
    prod_terms = {
        "mlops", "docker", "kubernetes", "aws", "gcp", "azure", "model deployment", "feature store"
    }

    skill_set = set(s.lower() for s in skills)

    def _weight_for(bucket_terms):
        overlap = len(skill_set & bucket_terms)
        return {"terms": bucket_terms, "weight": round(max(0.05, min(0.25, overlap * 0.05 + 0.10)), 2)}

    return {
        "retrieval_embeddings": _weight_for(retrieval_terms),
        "vector_search":        _weight_for(vector_terms),
        "ranking_eval":         _weight_for(ranking_terms),
        "python_ml":            _weight_for(ml_terms),
        "llm_nlp":              _weight_for(llm_terms),
        "production_ml":        _weight_for(prod_terms),
        "product_context": {
            "terms": {"product", "saas", "platform", "marketplace", "internet", "software"},
            "weight": 0.05
        },
    }


def _build_embed_text(jd_text: str, skills: List[str], titles: List[str], locations: List[str]) -> str:
    parts = [jd_text[:800]]
    if titles:
        parts.append("Preferred roles: " + ", ".join(titles[:4]) + ".")
    if skills:
        parts.append("Required skills: " + ", ".join(skills[:20]) + ".")
    if locations:
        parts.append("Locations: " + ", ".join(locations) + ".")
    return " ".join(parts)


def parse_jd(jd_text: str) -> Dict:
    skills    = _extract_skills(jd_text)
    titles    = _extract_preferred_titles(jd_text)
    locations = _extract_locations(jd_text)
    min_exp   = _extract_min_experience(jd_text)

    return {
        "jd_core_skills":   skills,
        "jd_skill_buckets": _build_skill_buckets(skills),
        "jd_embed_text":    _build_embed_text(jd_text, skills, titles, locations),
        "exp_min":          min_exp or 2.0,
        "exp_ideal":        (min_exp or 2.0) + 3.0,
        "exp_max":          (min_exp or 2.0) + 9.0,
        "preferred_locations": locations,
        "preferred_titles": titles,
    }
