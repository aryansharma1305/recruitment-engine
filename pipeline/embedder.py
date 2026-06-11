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
    skills = safe_skills(candidate)
    career = safe_career(candidate)
    signals = safe_signals(candidate)

    skill_text = ", ".join(
        sorted(
            set(
                s.get("name", "").lower()
                for s in skills
                if s.get("name")
            )
        )
    )

    career_text = " | ".join(
        (
            f"TITLE: {j.get('title', '')} "
            f"COMPANY: {j.get('company', '')} "
            f"INDUSTRY: {j.get('industry', '')} "
            f"DESCRIPTION: {j.get('description', '')[:700]}"
        )
        for j in career[:8]
    )

    signal_text = (
        f"github activity {signals.get('github_activity_score', 0)}. "
        f"open to work {signals.get('open_to_work_flag', False)}."
    )

    return (
        f"CURRENT_TITLE: {profile.get('current_title', '')}. "
        f"HEADLINE: {profile.get('headline', '')}. "
        f"SUMMARY: {profile.get('summary', '')[:800]}. "
        f"SKILLS: {skill_text}. "
        f"CAREER_HISTORY: {career_text}. "
        f"{signal_text}"
    )[:4000]


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

def normalize_score_dict(scores: Dict[str, float]) -> Dict[str, float]:

    if not scores:
        return scores

    values = list(scores.values())

    lo = min(values)
    hi = max(values)

    if hi <= lo:
        return {k: 0.5 for k in scores}

    return {
        k: (v - lo) / (hi - lo)
        for k, v in scores.items()
    }


def skill_overlap_boost(candidate: Dict) -> float:

    skills = {
        s.get("name", "").lower()
        for s in safe_skills(candidate)
        if s.get("name")
    }

    matches = sum(
        1
        for skill in JD_CORE_SKILLS
        if skill.lower() in skills
    )

    return min(matches / 20.0, 1.0)


def embed_jd(jd_text: str = JD_EMBED_TEXT) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    model = _get_model()
    query = f"""
Senior AI Engineer candidate search.

Required skills and experience:

{jd_text}

Find candidates who demonstrate:

- Production AI systems
- Retrieval systems
- Ranking systems
- Recommendation systems
- Vector databases
- Embeddings
- NLP
- Applied machine learning

Return semantically relevant candidates.
"""
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

    embed_scores = cosine_scores(candidates, jd_emb)

    tfidf_scores = local_tfidf_scores(candidates, jd_text)

    embed_scores = normalize_score_dict(embed_scores)
    tfidf_scores = normalize_score_dict(tfidf_scores)

    hybrid_scores = {}

    for candidate in candidates:

        cid = candidate.get("candidate_id", "")

        semantic = embed_scores.get(cid, 0.0)
        lexical = tfidf_scores.get(cid, 0.0)

        skill_boost = skill_overlap_boost(candidate)

        hybrid_scores[cid] = (
            0.75 * semantic +
            0.15 * lexical +
            0.10 * skill_boost
        )

    return hybrid_scores
