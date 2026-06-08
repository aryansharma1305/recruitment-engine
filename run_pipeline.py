import os
import sys
import csv
import time
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CANDIDATES_FILE, SAMPLE_FILE, JD_FILE, OUTPUT_CSV, TOP500_CSV,
    TOP_N_PREFILTER, TOP_N_RANK, FINAL_N,
    FRAUD_CUTOFF, JD_CORE_SKILLS, OUTPUT_DIR,
)
from pipeline.loader import stream, load_json_array, cid as get_cid
from pipeline.jd_loader import read_job_description, summarize_jd
from pipeline.feature_extractor import extract
from pipeline.fraud_detector import detect
from pipeline.embedder import local_tfidf_scores, local_embedding_scores
from pipeline.graph_rag import CandidateGraph
from pipeline.ranker import rank
from pipeline.explainer import explain_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="Run on sample_candidates.json (fast test)")
    p.add_argument("--candidates-file", default=CANDIDATES_FILE, help="Path to candidates.jsonl")
    p.add_argument("--sample-file", default=SAMPLE_FILE, help="Path to sample_candidates.json")
    p.add_argument("--jd-file", default=JD_FILE, help="Path to job description .docx or .txt")
    p.add_argument("--output-csv", default=OUTPUT_CSV, help="Submission CSV path")
    p.add_argument("--top-k", type=int, default=TOP_N_PREFILTER, help="Candidates to keep after semantic prefilter")
    p.add_argument("--semantic-backend", choices=["tfidf", "embedding", "none"], default="tfidf")
    p.add_argument("--use-graph", action="store_true", help="Enable experimental graph expansion/boosting")
    return p.parse_args()


def load_candidates(use_sample: bool, candidates_file: str, sample_file: str):
    if use_sample:
        return load_json_array(sample_file)
    return list(stream(candidates_file))


def phase1_extract_and_fraud(candidates):
    print(f"\n[Phase 1] Extracting features + fraud detection ({len(candidates)} candidates)")
    features    = {}
    fraud_info  = {}
    fraud_scores = {}

    for c in tqdm(candidates, desc="Extracting", unit="cand"):
        cid = get_cid(c)
        features[cid]     = extract(c)
        fscore, flags      = detect(c)
        fraud_info[cid]    = flags
        fraud_scores[cid]  = fscore
        features[cid]["fraud_score"] = fscore

    clean     = sum(1 for s in fraud_scores.values() if s < FRAUD_CUTOFF)
    flagged   = len(fraud_scores) - clean
    print(f"  Clean: {clean} | Flagged/removed: {flagged}")
    return features, fraud_scores, fraud_info


def phase2_semantic_score(candidates, jd_text, backend, top_k):
    print(f"\n[Phase 2] Local semantic scoring (backend={backend}, top_k={top_k})")
    if backend == "none":
        sims = {get_cid(c): 0.0 for c in candidates}
    elif backend == "embedding":
        sims = local_embedding_scores(candidates, jd_text)
    else:
        sims = local_tfidf_scores(candidates, jd_text)

    ranked = sorted(sims.items(), key=lambda x: (-x[1], x[0]))
    if top_k and top_k < len(ranked):
        ranked = ranked[:top_k]
    print(f"  Scored {len(sims)} candidates; retained {len(ranked)} semantic matches.")
    return ranked


def phase3_build_graph(candidates, fraud_scores, seed_ids):
    print(f"\n[Phase 3] Building GraphRAG knowledge graph")
    graph = CandidateGraph()
    for c in tqdm(candidates, desc="Graph build", unit="node"):
        cid = get_cid(c)
        graph.add_candidate(c, fraud_score=fraud_scores.get(cid, 0.0))
    graph.build_coworker_edges()

    stats = graph.stats()
    print(f"  Nodes → candidates:{stats.get('candidate', 0)} | skills:{stats.get('skill', 0)} | companies:{stats.get('company', 0)}")

    jd_skills = set(JD_CORE_SKILLS)
    boosts = graph.multihop_expand(seed_ids, jd_skills, max_hops=2)
    print(f"  Multi-hop discovered {len(boosts)} additional candidates via 2-hop expansion")
    return boosts


def phase4_rank(features, semantic_results, graph_boosts, top_n):
    print(f"\n[Phase 4] Ranking (top {top_n})")
    sims     = {cid: sim for cid, sim in semantic_results}
    all_recs = list(features.values())
    ranked   = rank(all_recs, sims, graph_boosts)
    top      = ranked[:top_n]
    print(f"  Ranked {len(ranked)} candidates → selected top {len(top)}")
    return top, sims


def phase5_explain_and_write(top_ranked, features, sims, fraud_info, output_csv, top_n=FINAL_N):
    print(f"\n[Phase 5] Generating explanations for top {top_n}")
    top_cids  = [cid for cid, _ in top_ranked[:top_n]]
    top_recs  = [features[cid] for cid in top_cids if cid in features]
    top_sims  = {cid: sims.get(cid, 0.0) for cid in top_cids}

    reasons = explain_batch(top_recs, top_sims)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows_to_write = min(top_n, len(top_ranked))
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (candidate_id, score) in enumerate(top_ranked[:rows_to_write], 1):
            reason = reasons.get(candidate_id, "")
            display_score = max(0.0, score / 100 - rank_pos * 0.0000001)
            writer.writerow([candidate_id, rank_pos, f"{display_score:.6f}", reason])

    print(f"  Wrote {rows_to_write} rows → {output_csv}")


def main():
    args = parse_args()
    t0   = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidates  = load_candidates(args.sample, args.candidates_file, args.sample_file)
    jd_text = read_job_description(args.jd_file)
    jd_summary = summarize_jd(jd_text)
    print(f"\n[JD] Loaded {args.jd_file}")
    print(f"  Role: {jd_summary['role']}")
    print(f"  Must-have: {jd_summary['must_have']}")

    features, fraud_scores, fraud_info = phase1_extract_and_fraud(candidates)

    semantic_results = phase2_semantic_score(candidates, jd_text, args.semantic_backend, args.top_k)

    if args.use_graph:
        seed_ids = {cid for cid, _ in semantic_results[:500]}
        graph_boosts = phase3_build_graph(candidates, fraud_scores, seed_ids)
    else:
        print("\n[Phase 3] Graph expansion skipped (use --use-graph to enable experimental boost)")
        graph_boosts = {}

    top_ranked, sims = phase4_rank(features, semantic_results, graph_boosts, top_n=TOP_N_RANK)

    # Save top 500 checkpoint
    with open(TOP500_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "score"])
        for cid, s in top_ranked:
            writer.writerow([cid, round(s / 100, 4)])

    phase5_explain_and_write(top_ranked, features, sims, fraud_info, args.output_csv)

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s — output: {args.output_csv}")


if __name__ == "__main__":
    main()
