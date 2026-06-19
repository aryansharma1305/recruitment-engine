import csv
import json
import os

SUBMISSION_FILE = "output/submission.csv"
CANDIDATES_FILE = "/Users/gugloo/Downloads/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUTPUT_DIR = "frontend/public"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top_candidates.json")

def main():
    if not os.path.exists(SUBMISSION_FILE):
        print(f"Error: {SUBMISSION_FILE} not found.")
        return

    # Load top 100
    top_candidates = {}
    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            top_candidates[cid] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"],
                "profile_data": None
            }

    print(f"Loaded {len(top_candidates)} candidates from {SUBMISSION_FILE}")

    # Extract full profile data
    found = 0
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in top_candidates:
                top_candidates[cid]["profile_data"] = cand
                found += 1
                if found == len(top_candidates):
                    break

    print(f"Matched {found} profiles from {CANDIDATES_FILE}")

    # Sort by rank
    sorted_candidates = sorted(top_candidates.values(), key=lambda x: x["rank"])

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_candidates, f, indent=2)

    print(f"Successfully exported {len(sorted_candidates)} candidates to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
