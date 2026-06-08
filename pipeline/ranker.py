from typing import Dict, List, Tuple
from config import WEIGHTS, FRAUD_CUTOFF


def score(features: Dict, semantic_sim: float, graph_boost: float = 0.0) -> float:
    raw = (
        features["skill_match"]      * WEIGHTS["skill_match"]
        + semantic_sim * 100          * WEIGHTS["semantic_sim"]
        + features["career_relevance"] * WEIGHTS["career_relevance"]
        + features["experience"]      * WEIGHTS["experience"]
        + features["behavioral"]      * WEIGHTS["behavioral"]
        + features.get("product_impact", 0.0) * WEIGHTS["product_impact"]
        + features.get("shipped_system", 0.0) * WEIGHTS["shipped_system"]
        + min(100.0, features.get("applied_ml_years", 0.0) / 5 * 100) * WEIGHTS["applied_ml_years"]
        + features.get("availability", 0.0) * WEIGHTS["availability"]
        + features.get("location_logistics", 0.0) * WEIGHTS["location_logistics"]
        + features["education"]       * WEIGHTS["education"]
        + features["certification"]   * WEIGHTS["certification"]
    )
    fraud = features.get("fraud_score", 0.0)
    if fraud >= FRAUD_CUTOFF:
        return 0.0
    fit_penalty = features.get("fit_penalty", 0.0)
    penalised = raw * (1.0 - fraud * 0.45) * (1.0 - fit_penalty)
    boosted   = min(100.0, penalised + min(graph_boost * 100, 4.0))
    return round(boosted, 4)


def rank(
    records: List[Dict],
    semantic_sims: Dict[str, float],
    graph_boosts: Dict[str, float],
) -> List[Tuple[str, float]]:
    scored = []
    for rec in records:
        cid   = rec["candidate_id"]
        sim   = semantic_sims.get(cid, 0.0)
        boost = graph_boosts.get(cid, 0.0)
        s     = score(rec, sim, boost)
        if s > 0:
            scored.append((cid, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored
