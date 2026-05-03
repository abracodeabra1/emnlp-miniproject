#!/usr/bin/env python3
"""Generate PDF figures for the course report (reads results/*.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_correlation_two_judges():
    with open(RESULTS / "meta_eval_report.json") as f:
        meta = json.load(f)
    dims = ["coherence", "consistency", "fluency", "relevance"]
    labels = ["Coherence", "Consistency", "Fluency", "Relevance"]
    judges = [("prometheus", "GPT-OSS-20B"), ("judgelm", "Qwen3-32B")]
    x = np.arange(len(dims))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (key, name) in enumerate(judges):
        rhos = [meta["correlation_with_human"][key][d]["spearman_rho"] for d in dims]
        ax.bar(x + (i - 0.5) * width, rhos, width, label=name)
    ax.set_ylabel("Spearman $\\rho$ (vs.\ human)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Judge–human rank correlation by dimension")
    fig.tight_layout()
    fig.savefig(FIG / "corr_by_dimension_two_judges.pdf", dpi=150)
    fig.savefig(FIG / "corr_by_dimension_two_judges.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG / "corr_by_dimension_two_judges.pdf")


def fig_adversarial_pairwise():
    with open(RESULTS / "adversarial_report.json") as f:
        adv = json.load(f)
    pw = adv["pairwise_adversarial"]
    types = ["fluent_inconsistent", "correct_poorly_written", "meaning_shift"]
    type_labels = ["Fluent\ninconsistent", "Correct,\npoor style", "Meaning\nshift"]
    judges = [("prometheus", "GPT-OSS-20B"), ("judgelm", "Qwen3-32B")]
    x = np.arange(len(types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (key, name) in enumerate(judges):
        rates = [pw[key][t]["adversarial_win_rate"] for t in types]
        ax.bar(x + (i - 0.5) * width, rates, width, label=name)
    ax.set_ylabel("Adversarial win rate (pairwise)")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Head-to-head: adversarial vs.\ good summary (by probe type)")
    fig.tight_layout()
    fig.savefig(FIG / "adversarial_pairwise_winrate.pdf", dpi=150)
    fig.savefig(FIG / "adversarial_pairwise_winrate.png", dpi=150)
    plt.close(fig)
    print("Wrote", FIG / "adversarial_pairwise_winrate.pdf")


if __name__ == "__main__":
    fig_correlation_two_judges()
    fig_adversarial_pairwise()
