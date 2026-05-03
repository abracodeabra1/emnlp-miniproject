"""
Merge raw per-annotator annotation files into a single human_annotations.json.
Run after all 3 annotators have finished.

Input:  data/annotations/raw/{ann1,ann2,ann3}_annotations.json
Output: data/annotations/human_annotations.json
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ANN_DIR = DATA_DIR / "annotations"
RAW_DIR = ANN_DIR / "raw"

ANNOTATORS = ["ann1", "ann2", "ann3"]
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]


def main():
    all_records = []
    for ann in ANNOTATORS:
        path = RAW_DIR / f"{ann}_annotations.json"
        if not path.exists():
            print(f"WARNING: {path} not found — skipping {ann}")
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            all_records.append({
                "id": r["id"],
                "pair_id": r["pair_id"],
                "summary_key": r["summary_key"],
                "annotator": ann,
                "coherence": r["coherence"],
                "consistency": r["consistency"],
                "fluency": r["fluency"],
                "relevance": r["relevance"],
            })

    out_path = ANN_DIR / "human_annotations.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Merged {len(all_records)} annotation records to {out_path}")

    # Quick stats
    import pandas as pd
    df = pd.DataFrame(all_records)
    print(f"\nAnnotators: {df['annotator'].unique()}")
    print(f"Unique items: {df['id'].nunique()}")
    for dim in DIMENSIONS:
        print(f"  {dim}: mean={df[dim].mean():.2f}, std={df[dim].std():.2f}")


if __name__ == "__main__":
    main()
