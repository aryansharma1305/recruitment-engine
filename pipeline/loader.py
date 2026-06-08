import json
import os
from typing import Generator, Dict, Any, List


def stream(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_profile(c: Dict)      -> Dict:  return c.get("profile", {})
def safe_skills(c: Dict)       -> List:  return c.get("skills", [])
def safe_career(c: Dict)       -> List:  return c.get("career_history", [])
def safe_education(c: Dict)    -> List:  return c.get("education", [])
def safe_certs(c: Dict)        -> List:  return c.get("certifications", [])
def safe_signals(c: Dict)      -> Dict:  return c.get("redrob_signals", {})
def cid(c: Dict)               -> str:   return c.get("candidate_id", "UNKNOWN")
