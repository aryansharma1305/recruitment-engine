import sys
import os
import numpy as np
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMBED_MODEL, EMBED_BATCH, JD_EMBED_TEXT
from pipeline.loader import safe_profile, safe_skills, safe_career, safe_signals

try:
    from sentence_transformers import SentenceTransformer
    _model: Optional[SentenceTransformer] = None

    def _get_model() -> SentenceTransformer:
        global _model
        if _model is None:
            _model = SentenceTransformer(EMBED_MODEL)
        return _model

except ImportError:
    SentenceTransformer = None
    _model = None


def _build_text(candidate: Dict) -> str:
    profile = safe_profile(candidate)
    skills  = safe_skills(candidate)
    career  = safe_career(candidate)

    skill_text   = ", ".join(s.get("name", "") for s in skills)
    career_text  = " | ".join(
        (
            f"{j.get('title', '')} at {j.get('company', '')} "
            f"({j.get('industry', '')}, {j.get('company_size', '')}): "
            f"{j.get('description', '')[:450]}"
        )
        for j in career[:5]
    )
    signal_text = (
        f"github activity {safe_signals(candidate).get('github_activity_score', -1)}. "
        f"open to work {safe_signals(candidate).get('open_to_work_flag', False)}."
    )
    return (
        f"Current role: {profile.get('current_title', '')}. "
        f"Career evidence: {career_text}. "
        f"Skills: {skill_text}. "
        f"Headline: {profile.get('headline', '')}. "
        f"Summary: {profile.get('summary', '')[:260]}. "
        f"{signal_text}"
    )[:2600]


def _encode(texts: List[str]) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    model = _get_model()
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        out.append(vecs)
    return np.vstack(out).astype("float32")


def embed_jd(jd_text: str = JD_EMBED_TEXT) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    model = _get_model()
    query = f"Represent this sentence for searching relevant passages: {jd_text}"
    return model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")


def cosine_scores(candidates: List[Dict], jd_emb: np.ndarray) -> Dict[str, float]:
    texts = [_build_text(c) for c in candidates]
    vecs  = _encode(texts)
    jd    = jd_emb[0] if jd_emb.ndim == 2 else jd_emb
    sims  = (vecs @ jd).tolist()
    return {c.get("candidate_id", ""): float(s) for c, s in zip(candidates, sims)}


def local_tfidf_scores(candidates: List[Dict], jd_text: str) -> Dict[str, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    texts = [_build_text(c) for c in candidates]
    docs = [jd_text] + texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if len(candidates) >= 500 else 1,
        max_features=60000,
        sublinear_tf=True,
        norm="l2",
    )
    mat = vectorizer.fit_transform(docs)
    sims = linear_kernel(mat[0:1], mat[1:]).ravel()
    if sims.size == 0:
        return {}
    lo, hi = float(sims.min()), float(sims.max())
    if hi > lo:
        sims = (sims - lo) / (hi - lo)
    return {c.get("candidate_id", ""): float(s) for c, s in zip(candidates, sims)}


def local_embedding_scores(candidates: List[Dict], jd_text: str) -> Dict[str, float]:
    jd_emb = embed_jd(jd_text)
    return cosine_scores(candidates, jd_emb)
