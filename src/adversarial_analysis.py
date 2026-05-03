"""
Analyze how LLM judges perform on adversarial examples.
Tests whether judges prefer surface quality (fluency) over correctness.

Input:
  results/direct_scores.jsonl  (contains adversarial entries)
  results/pairwise_scores.jsonl
  data/adversarial/adversarial_examples.json

Output:
  results/adversarial_report.json
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"

JUDGES = ["gemini", "prometheus", "judgelm", "llama"]
ADV_TYPES = ["fluent_inconsistent", "correct_poorly_written", "meaning_shift"]


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_adversarial():
    with open(DATA_DIR / "adversarial" / "adversarial_examples.json") as f:
        return json.load(f)


def analyze_direct_scores(direct_df: pd.DataFrame) -> dict:
    """
    For each adversarial type and judge: compare judge scores on
    good_summary vs adversarial_summary.
    Hypothesis: judges rate fluent_inconsistent adversarial summaries
    as high or higher than good summaries (failure mode).
    """
    report = {}
    # Adversarial direct rows are tagged with article_id starting "adv_" and dimension="consistency"
    adv_data = direct_df[
        (direct_df["dimension"] == "consistency") &
        (direct_df["article_id"].astype(str).str.startswith("adv_", na=False))
    ]

    for judge in JUDGES:
        j_data = adv_data[adv_data["judge"] == judge]
        report[judge] = {}

        for adv_type in ADV_TYPES:
            type_data = j_data[j_data["adversarial_type"] == adv_type]
            good_rows = type_data[type_data["is_adversarial"] == False][["article_id", "score"]].dropna()
            bad_rows = type_data[type_data["is_adversarial"] == True][["article_id", "score"]].dropna()

            # Align by article_id so Wilcoxon compares matched pairs
            merged = good_rows.merge(bad_rows, on="article_id", suffixes=("_good", "_bad"))
            if len(merged) < 3:
                continue

            good_scores = merged["score_good"]
            bad_scores = merged["score_bad"]

            good_mean = good_scores.mean()
            bad_mean = bad_scores.mean()

            try:
                stat, p = wilcoxon(good_scores.values, bad_scores.values)
            except Exception:
                stat, p = None, None

            report[judge][adv_type] = {
                "good_summary_mean_consistency": round(good_mean, 3),
                "adversarial_summary_mean_consistency": round(bad_mean, 3),
                "delta": round(bad_mean - good_mean, 3),  # positive = judge prefers adversarial (failure)
                "judge_fooled_rate": round((bad_scores >= good_scores).mean(), 3),
                "wilcoxon_p": round(p, 4) if p is not None else None,
                "n": len(merged),
            }
    return report


def analyze_pairwise(pairwise_df: pd.DataFrame) -> dict:
    """
    In pairwise adversarial tests (good vs adversarial): what fraction of time
    does the judge pick the adversarial summary?
    """
    report = {}
    adv_pairs = pairwise_df[pairwise_df["order"] == "good_vs_adv"]

    for judge in JUDGES:
        j_data = adv_pairs[adv_pairs["judge"] == judge]
        report[judge] = {}

        for adv_type in ADV_TYPES:
            type_data = j_data[j_data["adversarial_type"] == adv_type]
            total = len(type_data)
            if total == 0:
                continue
            # Winner=='B' means judge chose adversarial (it was placed second)
            adv_wins = (type_data["winner"] == "B").sum()
            report[judge][adv_type] = {
                "adversarial_win_rate": round(adv_wins / total, 3),
                "n": total,
            }
    return report


def print_summary(report: dict, section: str):
    print(f"\n=== {section} ===")
    for judge, dims in report.items():
        print(f"\n{judge.upper()}")
        for adv_type, stats in dims.items():
            print(f"  {adv_type}: {stats}")


def main():
    direct = pd.DataFrame(load_jsonl(RESULTS_DIR / "direct_scores.jsonl"))
    pairwise = pd.DataFrame(load_jsonl(RESULTS_DIR / "pairwise_scores.jsonl"))

    direct_report = analyze_direct_scores(direct)
    pairwise_report = analyze_pairwise(pairwise)

    report = {
        "direct_score_adversarial": direct_report,
        "pairwise_adversarial": pairwise_report,
    }

    out_path = RESULTS_DIR / "adversarial_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved adversarial report to {out_path}")

    print_summary(direct_report, "Direct Scoring: Adversarial Consistency Scores")
    print_summary(pairwise_report, "Pairwise: Adversarial Win Rates")


if __name__ == "__main__":
    main()
