import sys
import os
import uuid
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.resume_parser import parse_resume
from api.jd_parser import parse_jd

import config as default_config
from pipeline.feature_extractor import extract as extract_features
from pipeline.fraud_detector import detect as detect_fraud
from pipeline.embedder import embed_candidates, embed_jd, cosine_sim_batch
from pipeline.ranker import score as compute_score, rank as rank_candidates
from pipeline.explainer import explain_batch

app = FastAPI(title="Recruitment Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: Dict[str, Dict] = {}

MAX_RESUMES = 50


def _apply_jd_config(jd_parsed: Dict):
    default_config.JD_CORE_SKILLS    = jd_parsed["jd_core_skills"]
    default_config.JD_SKILL_BUCKETS  = jd_parsed["jd_skill_buckets"]
    default_config.JD_EMBED_TEXT     = jd_parsed["jd_embed_text"]
    default_config.EXP_MIN           = jd_parsed["exp_min"]
    default_config.EXP_IDEAL         = jd_parsed["exp_ideal"]
    default_config.EXP_MAX           = jd_parsed["exp_max"]


def _run_pipeline(candidates: List[Dict], jd_parsed: Dict, job_id: str):
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["progress"] = 0
    total = len(candidates)

    try:
        _apply_jd_config(jd_parsed)

        # Phase 1 — Extract features + fraud
        records = []
        for i, cand in enumerate(candidates):
            fraud_score, flags = detect_fraud(cand)
            if fraud_score >= default_config.FRAUD_CUTOFF:
                _jobs[job_id]["progress"] = int((i + 1) / total * 30)
                continue
            feats = extract_features(cand)
            feats["fraud_score"] = fraud_score
            feats["fraud_flags"] = flags
            feats["_raw"] = cand
            records.append(feats)
            _jobs[job_id]["progress"] = int((i + 1) / total * 30)

        if not records:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["results"] = []
            return

        _jobs[job_id]["progress"] = 30

        # Phase 2 — Semantic similarity
        cand_texts = [
            " ".join([
                r.get("title", ""),
                r.get("summary", ""),
                " ".join(
                    s.get("name", "")
                    for s in r["_raw"].get("skills", [])
                ),
            ])
            for r in records
        ]

        jd_emb  = embed_jd(jd_parsed["jd_embed_text"])
        cand_embs = embed_candidates(cand_texts)
        sims = cosine_sim_batch(jd_emb, cand_embs)

        _jobs[job_id]["progress"] = 70

        # Phase 3 — Score + rank
        sim_map = {r["candidate_id"]: float(sims[i]) for i, r in enumerate(records)}
        scored = rank_candidates(records, sim_map)
        top = scored[:min(len(scored), default_config.FINAL_N)]

        _jobs[job_id]["progress"] = 85

        # Phase 4 — Explain top candidates
        to_explain = top[:min(len(top), default_config.TOP_N_EXPLAIN)]
        explanations = explain_batch(to_explain, [r["_raw"] for r in to_explain], sim_map)

        _jobs[job_id]["progress"] = 95

        # Build results
        results = []
        for i, rec in enumerate(top):
            cid = rec["candidate_id"]
            raw = rec["_raw"]
            prof = raw.get("profile", {})
            results.append({
                "rank":       i + 1,
                "score":      rec.get("final_score", 0.0),
                "reasoning":  explanations.get(cid, ""),
                "profile_data": raw,
                "score_breakdown": {
                    "skill_match":      rec.get("skill_match", 0),
                    "career_relevance": rec.get("career_relevance", 0),
                    "experience":       rec.get("experience", 0),
                    "behavioral":       rec.get("behavioral", 0),
                    "shipped_system":   rec.get("shipped_system", 0),
                    "applied_ml_years": rec.get("applied_ml_years", 0),
                    "semantic_sim":     round(sim_map.get(cid, 0) * 100, 1),
                    "fraud_score":      rec.get("fraud_score", 0),
                },
                "fraud_flags": rec.get("fraud_flags", []),
                "source_file": raw.get("_source_file", ""),
            })

        _jobs[job_id]["status"]   = "done"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["results"]  = results

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(exc)
        raise


@app.post("/api/rank")
async def rank_resumes(
    background_tasks: BackgroundTasks,
    jd_text: str = Form(...),
    resumes: List[UploadFile] = File(...),
):
    if len(resumes) > MAX_RESUMES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_RESUMES} resumes allowed per run.")
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    # Parse JD
    jd_parsed = parse_jd(jd_text)

    # Parse resumes
    candidates = []
    errors = []
    for i, upload in enumerate(resumes):
        try:
            content = await upload.read()
            cand = parse_resume(content, upload.filename or f"resume_{i}.pdf", i)
            candidates.append(cand)
        except Exception as e:
            errors.append({"file": upload.filename, "error": str(e)})

    if not candidates:
        raise HTTPException(status_code=422, detail="No resumes could be parsed. Check file formats.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":     "queued",
        "progress":   0,
        "results":    None,
        "error":      None,
        "errors":     errors,
        "created_at": datetime.now().isoformat(),
        "n_resumes":  len(candidates),
        "jd_skills":  jd_parsed["jd_core_skills"][:12],
    }

    background_tasks.add_task(_run_pipeline, candidates, jd_parsed, job_id)

    return {"job_id": job_id, "n_resumes": len(candidates), "parse_errors": errors}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job["progress"],
        "n_resumes": job.get("n_resumes", 0),
        "results":  job["results"] if job["status"] == "done" else None,
        "error":    job.get("error"),
        "jd_skills": job.get("jd_skills", []),
    }


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    _jobs.pop(job_id, None)
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
