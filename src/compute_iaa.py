"""
Compute inter-annotator agreement (IAA) on human annotation data.
Metrics: Krippendorff's Alpha (ordinal), Cohen's Kappa (pairwise).

Input:  data/annotations/human_annotations.json
Output: results/iaa_report.json  + printed summary

Annotation file format (list of records):
  {
    "id": "pair_001",
    "article_id": 5,
    "summary_id": "A",
    "annotator": "ann1",   // ann1 | ann2 | ann3
    "coherence": 4,
    "consistency": 2,
    "fluency": 5,
    "relevance": 3
  }
"""

import json
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
ANNOTATORS = ["ann1", "ann2", "ann3"]


def load_annotations():
    path = DATA_DIR / "annotations" / "human_annotations.json"
    with open(path) as f:
        return json.load(f)


def build_matrix(df: pd.DataFrame, dimension: str):
    """
    Build annotator × item matrix for krippendorff.
    Rows = annotators, Columns = items (summary ids).
    """
    pivot = df.pivot_table(index="annotator", columns="id", values=dimension, aggfunc="first")
    return pivot.reindex(ANNOTATORS).values.astype(float)


def compute_krippendorff(matrix: np.ndarray) -> float:
    return krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")


def compute_kappa_pairs(df: pd.DataFrame, dimension: str) -> dict:
    kappas = {}
    for a1, a2 in combinations(ANNOTATORS, 2):
        df1 = df[df["annotator"] == a1].set_index("id")[dimension]
        df2 = df[df["annotator"] == a2].set_index("id")[dimension]
        common = df1.index.intersection(df2.index)
        if len(common) < 5:
            continue
        kappas[f"{a1}_vs_{a2}"] = cohen_kappa_score(
            df1.loc[common].values, df2.loc[common].values, weights="quadratic"
        )
    return kappas


def main():
    annotations = load_annotations()
    df = pd.DataFrame(annotations)

    report = {}
    print("\n=== Inter-Annotator Agreement ===\n")

    for dim in DIMENSIONS:
        matrix = build_matrix(df, dim)
        alpha = compute_krippendorff(matrix)
        kappas = compute_kappa_pairs(df, dim)
        mean_kappa = np.mean(list(kappas.values())) if kappas else float("nan")

        report[dim] = {
            "krippendorff_alpha": round(alpha, 4),
            "cohen_kappa_pairs": {k: round(v, 4) for k, v in kappas.items()},
            "mean_cohen_kappa": round(mean_kappa, 4),
        }

        print(f"{dim.upper()}")
        print(f"  Krippendorff α = {alpha:.4f}  (>0.4 = moderate, >0.6 = substantial)")
        for pair, k in kappas.items():
            print(f"  Cohen κ [{pair}] = {k:.4f}")
        print()

    out_path = RESULTS_DIR / "iaa_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved IAA report to {out_path}")

    # Build gold standard: mean score per item per dimension across annotators
    gold = (
        df.groupby("id")[DIMENSIONS]
        .mean()
        .reset_index()
        .rename(columns={d: f"{d}_gold" for d in DIMENSIONS})
    )
    gold_path = DATA_DIR / "annotations" / "gold_standard.json"
    gold.to_json(gold_path, orient="records", indent=2)
    print(f"Saved gold standard to {gold_path}")


if __name__ == "__main__":
    main()
