"""
Convert SummEval data into our annotation format as a fallback gold standard.
Used if fresh human annotation is incomplete.

SummEval structure:
  { id, text, machine_summaries[16], coherence[16], consistency[16],
    fluency[16], relevance[16], ... }
  Scores are already averaged across annotators (float).

Output:
  data/annotations/gold_standard_summeval.json  -- same format as our gold_standard.json
  data/annotations/human_annotations_summeval.json -- simulated per-annotator format

NOTE: SummEval scores are on a 1–5 scale for coherence/relevance/fluency,
but consistency is 1–5 as well. We normalize to the same format.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ANN_DIR = DATA_DIR / "annotations"
ANN_DIR.mkdir(exist_ok=True)

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def main():
    # Load SummEval
    with open(DATA_DIR / "raw" / "summeval_human.json") as f:
        summeval = json.load(f)

    # Load our 50 selected articles to know which IDs to use
    with open(DATA_DIR / "raw" / "articles_50.json") as f:
        our_articles = json.load(f)
    our_ids = {a["id"] for a in our_articles}

    # SummEval indices correspond to CNN/DM test indices
    gold_records = []
    for idx, record in enumerate(summeval):
        if idx not in our_ids:
            continue

        # SummEval has 16 machine summaries; take first 2 (index 0 and 1)
        # These will be our "system A" and "system B" fallback
        for sys_idx in range(2):
            summary = record["machine_summaries"][sys_idx]
            record_id = f"{idx}_sys{sys_idx}"

            # Scores are averaged across annotators
            gold_records.append({
                "id": record_id,
                "article_id": idx,
                "system_idx": sys_idx,
                "summary": summary,
                "coherence_gold": record["coherence"][sys_idx],
                "consistency_gold": record["consistency"][sys_idx],
                "fluency_gold": record["fluency"][sys_idx],
                "relevance_gold": record["relevance"][sys_idx],
                "source": "summeval",
            })

    # Save as gold standard
    gold_path = ANN_DIR / "gold_standard_summeval.json"
    with open(gold_path, "w") as f:
        json.dump(gold_records, f, indent=2)
    print(f"Saved {len(gold_records)} SummEval gold records to {gold_path}")


if __name__ == "__main__":
    main()
