import sys
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.loader import safe_skills, safe_career, safe_signals, safe_profile
from pipeline.text_utils import norm as _norm
from config import FRAUD_CUTOFF, JD_CORE_SKILLS, SKILL_ALIASES

try:
    from rapidfuzz import fuzz, process as rfprocess
    _FUZZY = True
except ImportError:
    _FUZZY = False


_JD_SKILLS_NORM = {_norm(s) for s in JD_CORE_SKILLS}


def _canonical(name: str) -> str:
    n = _norm(name)
    return SKILL_ALIASES.get(n, n)


class CandidateGraph:
    """
    Knowledge graph over candidates, skills, and companies.

    Node types:
      - candidate:{id}
      - skill:{canonical_name}
      - company:{name}

    Edge types:
      - HAS_SKILL   candidate → skill   (weight = proficiency score)
      - WORKED_AT   candidate → company (weight = tenure months / 12)
      - COWORKER    candidate → candidate (via shared company, inferred)
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def add_candidate(self, candidate: Dict, fraud_score: float = 0.0):
        cid     = candidate.get("candidate_id", "")
        profile = safe_profile(candidate)
        sig     = safe_signals(candidate)

        cnode = f"candidate:{cid}"
        self.G.add_node(
            cnode,
            type="candidate",
            cid=cid,
            title=profile.get("current_title", ""),
            yoe=profile.get("years_of_experience", 0),
            github=sig.get("github_activity_score", -1),
            fraud=fraud_score,
        )

        prof_w = {"expert": 1.0, "advanced": 0.80, "intermediate": 0.55, "beginner": 0.25}

        for skill in safe_skills(candidate):
            canonical = _canonical(skill.get("name", ""))
            snode = f"skill:{canonical}"
            if not self.G.has_node(snode):
                self.G.add_node(snode, type="skill", name=canonical)
            w = prof_w.get(skill.get("proficiency", "beginner"), 0.25)
            self.G.add_edge(cnode, snode, rel="HAS_SKILL", weight=w)

        for job in safe_career(candidate):
            company = _norm(job.get("company", ""))
            if not company:
                continue
            conode = f"company:{company}"
            if not self.G.has_node(conode):
                self.G.add_node(conode, type="company", name=company)
            tenure = job.get("duration_months", 0) / 12.0
            self.G.add_edge(cnode, conode, rel="WORKED_AT", weight=min(tenure, 5.0))

    def build_coworker_edges(self, max_company_degree: int = 150):
        company_to_candidates: Dict[str, List[str]] = defaultdict(list)
        for node, data in self.G.nodes(data=True):
            if data.get("type") == "candidate":
                for _, co, edata in self.G.out_edges(node, data=True):
                    if edata.get("rel") == "WORKED_AT":
                        company_to_candidates[co].append(node)

        for co, cnodes in company_to_candidates.items():
            if len(cnodes) > max_company_degree:
                continue
            for i in range(len(cnodes)):
                for j in range(i + 1, len(cnodes)):
                    if not self.G.has_edge(cnodes[i], cnodes[j]):
                        self.G.add_edge(cnodes[i], cnodes[j], rel="COWORKER", weight=0.3)
                    if not self.G.has_edge(cnodes[j], cnodes[i]):
                        self.G.add_edge(cnodes[j], cnodes[i], rel="COWORKER", weight=0.3)

    def multihop_expand(
        self,
        seed_candidates: Set[str],
        jd_skills: Set[str],
        max_hops: int = 2,
    ) -> Dict[str, float]:
        """
        Multi-hop graph traversal starting from seed candidates.

        Hop 1: seed candidates → their skills → other candidates sharing those skills
        Hop 2: expanded candidates → their companies → more candidates at those companies
               with relevant skills

        Returns: {candidate_id: relevance_boost} for non-seed candidates discovered
        """
        discovered: Dict[str, float] = {}

        seed_nodes = {f"candidate:{cid}" for cid in seed_candidates}

        # Hop 1 — skill-based expansion
        hop1_candidates: Set[str] = set()
        for cnode in seed_nodes:
            if not self.G.has_node(cnode):
                continue
            for _, snode, edata in self.G.out_edges(cnode, data=True):
                if edata.get("rel") != "HAS_SKILL":
                    continue
                skill_name = self.G.nodes[snode].get("name", "")
                if not self._skill_relevant(skill_name, jd_skills):
                    continue
                for other, _, inv_edata in self.G.in_edges(snode, data=True):
                    if inv_edata.get("rel") != "HAS_SKILL":
                        continue
                    if other in seed_nodes or other == cnode:
                        continue
                    prof_w = inv_edata.get("weight", 0.25)
                    boost = prof_w * 0.15
                    other_cid = self.G.nodes[other].get("cid", "")
                    if other_cid:
                        discovered[other_cid] = max(discovered.get(other_cid, 0.0), boost)
                        hop1_candidates.add(other)

        if max_hops < 2:
            return discovered

        # Hop 2 — company-based expansion from hop-1 candidates
        for cnode in hop1_candidates:
            if not self.G.has_node(cnode):
                continue
            for _, conode, edata in self.G.out_edges(cnode, data=True):
                if edata.get("rel") != "WORKED_AT":
                    continue
                for other, _, inv_edata in self.G.in_edges(conode, data=True):
                    if inv_edata.get("rel") != "WORKED_AT":
                        continue
                    if other in seed_nodes or other in hop1_candidates:
                        continue
                    other_data = self.G.nodes.get(other, {})
                    if other_data.get("fraud", 1.0) >= FRAUD_CUTOFF:
                        continue
                    other_cid = other_data.get("cid", "")
                    if other_cid and self._candidate_has_any_jd_skill(other, jd_skills):
                        boost = 0.06
                        discovered[other_cid] = max(discovered.get(other_cid, 0.0), boost)

        return discovered

    def _skill_relevant(self, skill_name: str, jd_skills: Set[str]) -> bool:
        if skill_name in jd_skills:
            return True
        if _FUZZY:
            hit = rfprocess.extractOne(skill_name, list(jd_skills), scorer=fuzz.ratio)
            return bool(hit and hit[1] >= 78)
        return False

    def _candidate_has_any_jd_skill(self, cnode: str, jd_skills: Set[str]) -> bool:
        for _, snode, edata in self.G.out_edges(cnode, data=True):
            if edata.get("rel") != "HAS_SKILL":
                continue
            sname = self.G.nodes[snode].get("name", "")
            if self._skill_relevant(sname, jd_skills):
                return True
        return False

    def stats(self) -> Dict[str, int]:
        types: Dict[str, int] = defaultdict(int)
        for _, data in self.G.nodes(data=True):
            types[data.get("type", "unknown")] += 1
        return dict(types)
