"""
Build taxonomy of LLM judge failure modes.
Identifies examples where LLM judges diverge from human gold and categorizes why.

Output:
  results/failure_taxonomy.json
  results/figures/failure_breakdown.pdf
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

JUDGES = ["gemini", "prometheus", "judgelm", "llama"]
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

FAILURE_THRESHOLD = 1.5  # judge score differs from human gold by >= this


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_gold():
    with open(DATA_DIR / "annotations" / "gold_standard.json") as f:
        data = json.load(f)
    # Index by the compound id produced by compute_iaa.py: "<pair_id>_<summary_key>"
    return pd.DataFrame(data).set_index("id")


def _add_gold_id(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the merge key matching gold_standard.json's 'id' field."""
    df = df.copy()
    df["gold_id"] = df["article_id"].astype(str) + "_" + df["summary_key"]
    return df



def categorize_failure(row: pd.Series, gold: pd.DataFrame) -> list:
    """Return list of failure modes for this judge result."""
    failures = []
    gold_id = row.get("gold_id")
    dim = row["dimension"]
    judge_score = row["score"]

    if gold_id not in gold.index or pd.isna(judge_score):
        return failures

    human_score = gold.loc[gold_id, f"{dim}_gold"]
    delta = judge_score - human_score

    if abs(delta) < FAILURE_THRESHOLD:
        return failures  # Not a failure

    # Failure: judge disagrees with human
    # Check failure modes:

    # 1. Over-trusting fluency: judge rates high, human rates consistency low
    if dim == "fluency" and delta > 0 and gold_id in gold.index:
        cons_human = gold.loc[gold_id, "consistency_gold"]
        if cons_human <= 2:
            failures.append("over_trusting_fluency")

    # 2. Ignoring factual errors: judge rates consistency high, human rates low
    if dim == "consistency" and delta > 0:
        failures.append("ignoring_factual_errors")

    # 3. Over-penalizing poor writing when content is correct
    if dim == "fluency" and delta < 0 and gold_id in gold.index:
        cons_human = gold.loc[gold_id, "consistency_gold"]
        if cons_human >= 4:
            failures.append("penalizing_poor_style_over_accuracy")

    # 4. Generic high score (dimension conflation): all dims same score
    if delta > 0:
        failures.append("score_inflation")
    else:
        failures.append("score_deflation")

    return failures


def analyze_position_reversals(pairwise_df: pd.DataFrame) -> dict:
    """Count cases where judge picks A→B but B→A (position-dependent reversal)."""
    reversals = {}
    # Filter to with_reference=False to prevent duplicate article_ids in the indexed DataFrame
    std = pairwise_df[
        (pairwise_df["variant"] == "standard") & (~pairwise_df["with_reference"])
    ]

    for judge in JUDGES:
        j = std[std["judge"] == judge]
        ab = j[j["order"] == "AB"].set_index("article_id")
        ba = j[j["order"] == "BA"].set_index("article_id")
        common = ab.index.intersection(ba.index)

        rev = 0
        for art_id in common:
            ab_w = ab.loc[art_id, "winner"]
            ba_w = ba.loc[art_id, "winner"]
            if ab_w and ba_w and ab_w == ba_w:
                rev += 1  # Same position-letter wins = position reversal

        reversals[judge] = {
            "position_reversals": rev,
            "total": len(common),
            "reversal_rate": round(rev / len(common), 3) if len(common) > 0 else 0,
        }
    return reversals


def analyze_dimension_conflation(direct_df: pd.DataFrame) -> dict:
    """
    Identify examples where a judge assigns nearly identical scores to all 4 dimensions.
    Indicates the judge is not distinguishing between dimensions.
    """
    conflation = {}
    subset = direct_df[(direct_df["variant"] == "standard") & (~direct_df["with_reference"])]

    for judge in JUDGES:
        j = subset[subset["judge"] == judge]
        conflated = 0
        total = 0

        for art_id in j["article_id"].unique():
            for model in ["llama", "mistral"]:
                scores = []
                for dim in DIMENSIONS:
                    row = j[(j["article_id"] == art_id) & (j["dimension"] == dim) & (j["summary_model"] == model)]
                    if not row.empty and not pd.isna(row.iloc[0]["score"]):
                        scores.append(row.iloc[0]["score"])
                if len(scores) == 4:
                    total += 1
                    if np.std(scores) < 0.5:
                        conflated += 1

        conflation[judge] = {
            "conflated": conflated,
            "total": total,
            "conflation_rate": round(conflated / total, 3) if total > 0 else 0,
        }
    return conflation


def main():
    direct = pd.DataFrame(load_jsonl(RESULTS_DIR / "direct_scores.jsonl"))
    pairwise = pd.DataFrame(load_jsonl(RESULTS_DIR / "pairwise_scores.jsonl"))
    gold = load_gold()

    # Failure mode counts per judge
    failure_counts = {j: {} for j in JUDGES}
    std = _add_gold_id(direct[(direct["variant"] == "standard") & (~direct["with_reference"])])

    for judge in JUDGES:
        j_data = std[std["judge"] == judge]
        for _, row in j_data.iterrows():
            modes = categorize_failure(row, gold)
            for m in modes:
                failure_counts[judge][m] = failure_counts[judge].get(m, 0) + 1

    position_reversals = analyze_position_reversals(pairwise)
    dim_conflation = analyze_dimension_conflation(direct)

    # Add to failure counts
    for judge in JUDGES:
        if judge in position_reversals:
            failure_counts[judge]["position_dependent_reversal"] = position_reversals[judge]["position_reversals"]
        if judge in dim_conflation:
            failure_counts[judge]["dimension_conflation"] = dim_conflation[judge]["conflated"]

    taxonomy = {
        "failure_modes": {
            "over_trusting_fluency": "Judge gives high score when output is fluent but factually inconsistent",
            "ignoring_factual_errors": "Judge rates consistency high despite clear hallucinations vs source",
            "penalizing_poor_style_over_accuracy": "Judge penalizes correct-but-awkward summaries too harshly",
            "score_inflation": "Judge systematically scores above human gold",
            "score_deflation": "Judge systematically scores below human gold",
            "position_dependent_reversal": "Judge picks different winner depending on A/B ordering",
            "dimension_conflation": "Judge gives nearly identical scores across all 4 dimensions",
        },
        "failure_counts_by_judge": failure_counts,
        "position_reversals": position_reversals,
        "dimension_conflation": dim_conflation,
    }

    out_path = RESULTS_DIR / "failure_taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved failure taxonomy to {out_path}")

    print("\n=== FAILURE MODE COUNTS ===")
    for judge, counts in failure_counts.items():
        print(f"\n{judge.upper()}")
        for mode, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {n}")

    # Stacked bar plot
    try:
        import matplotlib.pyplot as plt

        modes = list({m for j in failure_counts.values() for m in j})
        judges = JUDGES
        data = {m: [failure_counts[j].get(m, 0) for j in judges] for m in modes}
        df = pd.DataFrame(data, index=judges)
        ax = df.plot(kind="bar", stacked=True, figsize=(10, 5))
        ax.set_title("Failure Modes by Judge")
        ax.set_xlabel("Judge")
        ax.set_ylabel("Count")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig_dir = RESULTS_DIR / "figures"
        fig_dir.mkdir(exist_ok=True)
        plt.savefig(fig_dir / "failure_breakdown.pdf", dpi=150)
        plt.savefig(fig_dir / "failure_breakdown.png", dpi=150)
        plt.close()
        print("Saved failure breakdown plot.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
