"""
Master experiment runner. Executes all LLM judge experiments and saves results.

Groq logical judge ids (see src/judge.py): prometheus → openai/gpt-oss-20b,
judgelm → qwen/qwen3-32b, llama → llama-3.3-70b-versatile. NVIDIA NIM judges
use `nvidia/<NIM model id>` per configured default or env overrides.

Experiments:
  A) Direct scoring: all judges × all pairs × all dimensions × 3 prompt variants × with/without ref
  B) Pairwise: all judges × all pairs × 3 variants × A/B + B/A orderings
  C) Rubric-based: all judges × all pairs × 2 variants × with/without ref

Usage:
  python src/run_experiments.py --judges nvidia prometheus judgelm llama
  python src/run_experiments.py --judges nvidia --mode direct  # run one judge/mode
  python src/run_experiments.py --judges nvidia --mode all --clear  # fresh run

Results saved to:
  results/direct_scores.jsonl
  results/pairwise_scores.jsonl
  results/rubric_scores.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Running as `python src/run_experiments.py` puts `src/` on sys.path, not the repo root;
# `from src.judge import ...` needs the parent of `src/` on the path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "data"
RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALL_JUDGES = ["nvidia", "prometheus", "judgelm", "llama"]
ALL_VARIANTS = ["minimal", "standard", "cot"]
PAIRWISE_VARIANTS = ["minimal", "standard", "cot"]


def load_pairs():
    with open(DATA_DIR / "system_outputs" / "all_pairs.json") as f:
        return json.load(f)


def load_adversarial():
    path = DATA_DIR / "adversarial" / "adversarial_examples.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def append_result(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_direct(judges, pairs, adversarial):
    from src.judge import Judge
    from src.prompts import DIMENSIONS

    out_path = RESULTS_DIR / "direct_scores.jsonl"
    print(f"\n=== Direct Scoring → {out_path} ===")

    for judge_name in judges:
        judge = Judge(judge_name)
        for pair in tqdm(pairs, desc=f"{judge_name}/direct"):
            for summary_key, model_key in [("summary_A", "model_A"), ("summary_B", "model_B")]:
                for variant in ALL_VARIANTS:
                    for use_ref in [False, True]:
                        ref = pair["reference"] if use_ref else None
                        for dim in DIMENSIONS:
                            result = judge.direct_score(
                                pair["article"], pair[summary_key], dim, variant, ref
                            )
                            result.update({
                                "article_id": pair["id"],
                                "summary_key": summary_key,
                                "summary_model": pair[model_key],
                                "with_reference": use_ref,
                            })
                            append_result(out_path, result)

    # Adversarial: score good vs adversarial summaries
    for judge_name in judges:
        judge = Judge(judge_name)
        for adv in tqdm(adversarial, desc=f"{judge_name}/direct-adv"):
            for summary_key in ["good_summary", "adversarial_summary"]:
                result = judge.direct_score(adv["article"], adv[summary_key], "consistency", "standard")
                result.update({
                    "article_id": adv["id"],
                    "summary_type": summary_key,
                    "adversarial_type": adv["adversarial_type"],
                    "is_adversarial": summary_key == "adversarial_summary",
                })
                append_result(out_path, result)


def run_pairwise(judges, pairs, adversarial):
    from src.judge import Judge

    out_path = RESULTS_DIR / "pairwise_scores.jsonl"
    print(f"\n=== Pairwise Comparison → {out_path} ===")

    for judge_name in judges:
        judge = Judge(judge_name)
        for pair in tqdm(pairs, desc=f"{judge_name}/pairwise"):
            for variant in PAIRWISE_VARIANTS:
                for use_ref in [False, True]:
                    ref = pair["reference"] if use_ref else None
                    # Order A→B
                    r_ab = judge.pairwise(pair["article"], pair["summary_A"], pair["summary_B"], variant, ref)
                    r_ab.update({"article_id": pair["id"], "order": "AB",
                                 "model_A": pair["model_A"], "model_B": pair["model_B"],
                                 "with_reference": use_ref})
                    append_result(out_path, r_ab)

                    # Order B→A (position bias test)
                    r_ba = judge.pairwise(pair["article"], pair["summary_B"], pair["summary_A"], variant, ref)
                    r_ba.update({"article_id": pair["id"], "order": "BA",
                                 "model_A": pair["model_B"], "model_B": pair["model_A"],
                                 "with_reference": use_ref})
                    append_result(out_path, r_ba)

    # Adversarial pairwise: good vs adversarial
    for judge_name in judges:
        judge = Judge(judge_name)
        for adv in tqdm(adversarial, desc=f"{judge_name}/pairwise-adv"):
            r = judge.pairwise(adv["article"], adv["good_summary"], adv["adversarial_summary"], "cot")
            r.update({"article_id": adv["id"], "order": "good_vs_adv",
                      "adversarial_type": adv["adversarial_type"]})
            append_result(out_path, r)


def run_rubric(judges, pairs):
    from src.judge import Judge

    out_path = RESULTS_DIR / "rubric_scores.jsonl"
    print(f"\n=== Rubric-Based Evaluation → {out_path} ===")

    for judge_name in judges:
        judge = Judge(judge_name)
        for pair in tqdm(pairs, desc=f"{judge_name}/rubric"):
            for summary_key, model_key in [("summary_A", "model_A"), ("summary_B", "model_B")]:
                for variant in ["standard", "cot"]:
                    for use_ref in [False, True]:
                        ref = pair["reference"] if use_ref else None
                        result = judge.rubric(pair["article"], pair[summary_key], variant, ref)
                        result.update({
                            "article_id": pair["id"],
                            "summary_key": summary_key,
                            "summary_model": pair[model_key],
                            "with_reference": use_ref,
                        })
                        append_result(out_path, result)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM judge experiments: NVIDIA NIM (--judges nvidia) and/or Groq "
            "(prometheus=gpt-oss-20b, judgelm=qwen3-32b, llama=70b). "
            "See src/judge.py for API model ids."
        ),
    )
    parser.add_argument("--judges", nargs="+", default=ALL_JUDGES, choices=ALL_JUDGES)
    parser.add_argument("--mode", choices=["direct", "pairwise", "rubric", "all"], default="all")
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete existing result JSONL files before running to avoid duplicate entries on re-run",
    )
    args = parser.parse_args()

    from src.judge import expand_judge_names

    judges = expand_judge_names(args.judges)
    print(f"Judges (expanded): {judges}")

    if args.clear:
        for fname in ["direct_scores.jsonl", "pairwise_scores.jsonl", "rubric_scores.jsonl"]:
            p = RESULTS_DIR / fname
            if p.exists():
                p.unlink()
                print(f"Cleared {p}")

    pairs = load_pairs()
    adversarial = load_adversarial()
    print(f"Loaded {len(pairs)} pairs, {len(adversarial)} adversarial examples")

    if args.mode in ("direct", "all"):
        run_direct(judges, pairs, adversarial)
    if args.mode in ("pairwise", "all"):
        run_pairwise(judges, pairs, adversarial)
    if args.mode in ("rubric", "all"):
        run_rubric(judges, pairs)

    print("\nAll experiments complete. Results in results/")


if __name__ == "__main__":
    main()
