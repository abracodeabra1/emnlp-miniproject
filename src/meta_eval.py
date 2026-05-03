"""
Meta-evaluation: measure how well each LLM judge correlates with human gold,
and quantify biases (position, verbosity, reference, self-preference).

Input:
  results/direct_scores.jsonl
  results/pairwise_scores.jsonl
  results/rubric_scores.jsonl
  data/annotations/gold_standard.json
  data/system_outputs/all_pairs.json

Output:
  results/meta_eval_report.json
  results/figures/  (correlation heatmaps, bias plots)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
# "gemini" / bare "nvidia" for older JSONL; new NVIDIA runs use nvidia/<NIM model id>.
JUDGES = [
    "nvidia/minimaxai/minimax-m2.7",
    "nvidia/moonshotai/kimi-k2-thinking",
    "nvidia",
    "gemini",
    "prometheus",
    "judgelm",
    "llama",
]


# ─── Loaders ──────────────────────────────────────────────────────────────────

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
    return pd.DataFrame(data).set_index("id")


def load_pairs():
    with open(DATA_DIR / "system_outputs" / "all_pairs.json") as f:
        data = json.load(f)
    return {p["id"]: p for p in data}


# ─── Part 1: Correlation with human gold ──────────────────────────────────────

def _add_gold_id(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the key that matches gold_standard.json's 'id' field.

    compute_iaa.py groups annotations by the id set in annotate_app.py:
      id = f"{pair_id}_{summary_key}"  →  e.g. "0_summary_A"
    article_id in judge results is the integer pair id; summary_key is "summary_A"/"summary_B".
    """
    df = df.copy()
    df["gold_id"] = df["article_id"].astype(str) + "_" + df["summary_key"]
    return df


def correlation_analysis(direct_df: pd.DataFrame, gold: pd.DataFrame) -> dict:
    report = {}
    # Use standard-no-ref as canonical judge scores
    subset = direct_df[(direct_df["variant"] == "standard") & (~direct_df["with_reference"])]
    subset = _add_gold_id(subset)

    for judge in JUDGES:
        report[judge] = {}
        j_data = subset[subset["judge"] == judge]

        for dim in DIMENSIONS:
            dim_data = j_data[j_data["dimension"] == dim].copy()
            dim_data = dim_data.dropna(subset=["score"])
            merged = dim_data.merge(
                gold[[f"{dim}_gold"]].rename(columns={f"{dim}_gold": "human_score"}),
                left_on="gold_id", right_index=True, how="inner"
            )
            if len(merged) < 5:
                continue
            rho, p_rho = spearmanr(merged["score"], merged["human_score"])
            tau, p_tau = kendalltau(merged["score"], merged["human_score"])
            report[judge][dim] = {
                "spearman_rho": round(rho, 4),
                "spearman_p": round(p_rho, 4),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(p_tau, 4),
                "n": len(merged),
            }
    return report


# ─── Part 2: Position bias ─────────────────────────────────────────────────────

def position_bias_analysis(pairwise_df: pd.DataFrame) -> dict:
    """
    For each judge: compare win rate of the first-presented summary across orderings.
    If unbiased, first-position win rate ≈ 0.5.
    """
    report = {}
    # Filter to with_reference=False to avoid duplicate article_ids from the two ref conditions,
    # which would cause .loc[art_id] to return a DataFrame instead of a scalar.
    std_data = pairwise_df[
        (pairwise_df["variant"] == "standard") & (~pairwise_df["with_reference"])
    ]

    for judge in JUDGES:
        j_data = std_data[std_data["judge"] == judge]

        # Match AB and BA results for same article
        ab = j_data[j_data["order"] == "AB"].set_index("article_id")
        ba = j_data[j_data["order"] == "BA"].set_index("article_id")
        common = ab.index.intersection(ba.index)

        if len(common) == 0:
            continue

        # Position-consistent = judge always picks same model regardless of order
        consistent = 0
        first_wins = 0
        for art_id in common:
            ab_winner = ab.loc[art_id, "winner"]   # 'A' means llama won
            ba_winner = ba.loc[art_id, "winner"]   # 'A' now means qwen (order flipped)

            if ab_winner is None or ba_winner is None:
                continue

            # If ab_winner==A and ba_winner==B, judge consistently picks llama (position-independent)
            if (ab_winner == "A" and ba_winner == "B") or (ab_winner == "B" and ba_winner == "A"):
                consistent += 1

            # First-position win: winner=='A' (first-presented)
            if ab_winner == "A":
                first_wins += 1

        n = len(common)
        report[judge] = {
            "n_pairs": n,
            "position_consistent_rate": round(consistent / n, 4) if n > 0 else None,
            "first_position_win_rate": round(first_wins / n, 4) if n > 0 else None,
            "position_bias_score": round(abs(first_wins / n - 0.5) * 2, 4) if n > 0 else None,
        }
    return report


# ─── Part 3: Verbosity bias ────────────────────────────────────────────────────

def verbosity_bias_analysis(direct_df: pd.DataFrame, pairs: dict) -> dict:
    """Correlate judge scores with summary word count."""
    report = {}
    subset = direct_df[(direct_df["variant"] == "standard") & (~direct_df["with_reference"])]

    for judge in JUDGES:
        j_data = subset[subset["judge"] == judge].copy()
        dim_results = {}

        for dim in DIMENSIONS:
            dim_data = j_data[j_data["dimension"] == dim].dropna(subset=["score"])

            # Get word count for each summary
            lengths = []
            scores = []
            for _, row in dim_data.iterrows():
                art_id = row["article_id"]
                if art_id not in pairs:
                    continue
                pair = pairs[art_id]
                model = row.get("summary_model", "")
                summary = pair["summary_A"] if model == pair["model_A"] else pair["summary_B"]
                lengths.append(len(summary.split()))
                scores.append(row["score"])

            if len(lengths) < 5:
                continue
            rho, p = spearmanr(lengths, scores)
            dim_results[dim] = {"spearman_rho": round(rho, 4), "p": round(p, 4), "n": len(lengths)}

        report[judge] = dim_results
    return report


# ─── Part 4: Reference bias ────────────────────────────────────────────────────

def reference_bias_analysis(direct_df: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Compare correlation with/without reference for each judge."""
    report = {}
    std = _add_gold_id(direct_df[direct_df["variant"] == "standard"])

    for judge in JUDGES:
        j_data = std[std["judge"] == judge]
        report[judge] = {}
        for dim in DIMENSIONS:
            dim_data = j_data[j_data["dimension"] == dim]
            for use_ref in [False, True]:
                subset = dim_data[dim_data["with_reference"] == use_ref].dropna(subset=["score"])
                merged = subset.merge(
                    gold[[f"{dim}_gold"]].rename(columns={f"{dim}_gold": "human_score"}),
                    left_on="gold_id", right_index=True, how="inner"
                )
                if len(merged) < 5:
                    continue
                rho, _ = spearmanr(merged["score"], merged["human_score"])
                key = "with_ref" if use_ref else "no_ref"
                if dim not in report[judge]:
                    report[judge][dim] = {}
                report[judge][dim][key] = round(rho, 4)
    return report


# ─── Part 5: Self-preference bias ──────────────────────────────────────────────

def self_preference_analysis(pairwise_df: pd.DataFrame) -> dict:
    """
    Llama judge (llama) vs other judges: does llama prefer llama-generated summaries?
    """
    report = {}
    std = pairwise_df[(pairwise_df["variant"] == "standard") & (pairwise_df["order"] == "AB")]

    for judge in JUDGES:
        j_data = std[std["judge"] == judge]
        # Count how often judge picks llama vs qwen outputs
        llama_wins = 0
        total = 0
        for _, row in j_data.iterrows():
            winner = row.get("winner")
            if winner is None:
                continue
            model_a = row.get("model_A", "")
            model_b = row.get("model_B", "")
            winning_model = model_a if winner == "A" else model_b
            if winning_model == "llama":
                llama_wins += 1
            total += 1
        if total > 0:
            report[judge] = {
                "llama_win_rate": round(llama_wins / total, 4),
                "n": total,
            }
    return report


# ─── Part 6: Prompt sensitivity ────────────────────────────────────────────────

def prompt_sensitivity_analysis(direct_df: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Std deviation of Spearman r across 3 prompt variants per judge per dimension."""
    report = {}
    df_with_id = _add_gold_id(direct_df)
    for judge in JUDGES:
        j_data = df_with_id[df_with_id["judge"] == judge]
        report[judge] = {}
        for dim in DIMENSIONS:
            dim_data = j_data[j_data["dimension"] == dim]
            rhos = []
            variants_used = []
            for variant in ["minimal", "standard", "cot"]:
                v_data = dim_data[
                    (dim_data["variant"] == variant) & (~dim_data["with_reference"])
                ].dropna(subset=["score"])
                merged = v_data.merge(
                    gold[[f"{dim}_gold"]].rename(columns={f"{dim}_gold": "human_score"}),
                    left_on="gold_id", right_index=True, how="inner"
                )
                if len(merged) >= 5:
                    rho, _ = spearmanr(merged["score"], merged["human_score"])
                    rhos.append(rho)
                    variants_used.append(variant)
            if rhos:
                report[judge][dim] = {
                    "rhos_by_variant": dict(zip(variants_used, [round(r, 4) for r in rhos])),
                    "std_across_variants": round(float(np.std(rhos)), 4),
                }
    return report


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_report: dict):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = {}
        for judge in JUDGES:
            data[judge] = {}
            for dim in DIMENSIONS:
                val = corr_report.get(judge, {}).get(dim, {}).get("spearman_rho", float("nan"))
                data[judge][dim] = val

        df = pd.DataFrame(data).T
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(df, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="RdYlGn", ax=ax)
        ax.set_title("Spearman ρ: LLM Judge vs Human Gold (by dimension)")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Judge")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "correlation_heatmap.pdf", dpi=150)
        plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()
        print("Saved correlation heatmap.")
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots.")


def plot_position_bias(pos_report: dict):
    try:
        import matplotlib.pyplot as plt

        judges = [j for j in JUDGES if j in pos_report]
        bias_scores = [pos_report[j].get("position_bias_score", 0) for j in judges]

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(judges, bias_scores, color=["#e74c3c" if b > 0.1 else "#2ecc71" for b in bias_scores])
        ax.axhline(0.1, color="gray", linestyle="--", label="threshold (0.1)")
        ax.set_ylabel("Position Bias Score (0=unbiased, 1=fully biased)")
        ax.set_title("Position Bias by Judge")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "position_bias.pdf", dpi=150)
        plt.savefig(FIG_DIR / "position_bias.png", dpi=150)
        plt.close()
        print("Saved position bias plot.")
    except ImportError:
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    direct = pd.DataFrame(load_jsonl(RESULTS_DIR / "direct_scores.jsonl"))
    pairwise = pd.DataFrame(load_jsonl(RESULTS_DIR / "pairwise_scores.jsonl"))
    gold = load_gold()
    pairs = load_pairs()

    print("Computing correlations...")
    corr = correlation_analysis(direct, gold)

    print("Analyzing position bias...")
    pos_bias = position_bias_analysis(pairwise)

    print("Analyzing verbosity bias...")
    verb_bias = verbosity_bias_analysis(direct, pairs)

    print("Analyzing reference bias...")
    ref_bias = reference_bias_analysis(direct, gold)

    print("Analyzing self-preference...")
    self_pref = self_preference_analysis(pairwise)

    print("Analyzing prompt sensitivity...")
    prompt_sens = prompt_sensitivity_analysis(direct, gold)

    report = {
        "correlation_with_human": corr,
        "position_bias": pos_bias,
        "verbosity_bias": verb_bias,
        "reference_bias": ref_bias,
        "self_preference": self_pref,
        "prompt_sensitivity": prompt_sens,
    }

    out_path = RESULTS_DIR / "meta_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved meta-eval report to {out_path}")

    plot_correlation_heatmap(corr)
    plot_position_bias(pos_bias)

    # Print summary to console
    print("\n=== CORRELATION SUMMARY (Spearman ρ, standard prompt, no ref) ===")
    for judge in JUDGES:
        print(f"\n{judge.upper()}")
        for dim in DIMENSIONS:
            val = corr.get(judge, {}).get(dim, {}).get("spearman_rho", "N/A")
            print(f"  {dim}: {val}")

    print("\n=== POSITION BIAS ===")
    for judge, stats in pos_bias.items():
        print(f"  {judge}: bias_score={stats.get('position_bias_score')}, first_pos_win={stats.get('first_position_win_rate')}")


if __name__ == "__main__":
    main()
