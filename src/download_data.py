"""
Download SummEval data and 50 CNN/DM source articles.
SummEval: https://github.com/Yale-LILY/SummEval
CNN/DailyMail via HuggingFace datasets.

Run: python src/download_data.py
Output:
  data/raw/articles.json       -- 50 CNN/DM articles
  data/raw/summeval_human.json -- SummEval human annotations (fallback)
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_ARTICLES = 50

# SummEval uses CNN/DM test split articles 0–99
SUMMEVAL_ARTICLE_IDS = list(range(100))


def download_cnn_dm_articles():
    """Pull the 100 SummEval CNN/DM source articles."""
    print("Loading CNN/DailyMail test split...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    random.seed(SEED)
    # SummEval uses articles at specific indices — use first 100 to match
    indices = list(range(100))
    selected = [{"id": i, "article": ds[i]["article"], "highlights": ds[i]["highlights"]} for i in indices]

    out_path = RAW_DIR / "articles_100.json"
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved {len(selected)} articles to {out_path}")

    # Select our 50 for the study (diverse subset: every other article)
    random.seed(SEED)
    subset_indices = sorted(random.sample(range(100), NUM_ARTICLES))
    subset = [selected[i] for i in subset_indices]
    subset_path = RAW_DIR / "articles_50.json"
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved {len(subset)} study articles to {subset_path}")

    return subset


def download_summeval_annotations():
    """
    Pull SummEval human annotations from HuggingFace.
    Used as fallback gold standard if fresh annotation is incomplete.
    """
    print("Loading SummEval annotations...")
    try:
        ds = load_dataset("mteb/summeval", split="test")
        records = [dict(row) for row in ds]
        out_path = RAW_DIR / "summeval_human.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} SummEval annotation records to {out_path}")
    except Exception as e:
        print(f"Could not load SummEval from HuggingFace ({e}). Skipping.")


if __name__ == "__main__":
    download_cnn_dm_articles()
    download_summeval_annotations()
    print("Done.")
