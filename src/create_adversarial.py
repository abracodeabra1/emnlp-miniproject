"""
Create 10 adversarial examples in 3 categories:
  1. Fluent but inconsistent (4 examples) -- hallucinated facts
  2. Correct but poorly written (3 examples) -- extractive + errors
  3. Paraphrase with meaning shift (3 examples) -- subtle claim inversion

Uses Gemini 2.5 flash to generate adversarial variants of existing good summaries.

Usage:
  python src/create_adversarial.py

Output:
  data/adversarial/adversarial_examples.json
"""

import json
import os
import random
from pathlib import Path
from google import genai

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 123
random.seed(SEED)

client = genai.Client()  # reads GEMINI_API_KEY env var


PROMPTS = {
    "fluent_inconsistent": (
        "Given this news article and its summary, rewrite the summary so that it:\n"
        "- Sounds fluent, well-written, and confident\n"
        "- Contains 2–3 subtle factual errors (wrong numbers, names, or events that differ from the article)\n"
        "- Is roughly the same length as the original\n"
        "Do NOT change the style or tone — it should look like a plausible high-quality summary.\n\n"
        "Article:\n{article}\n\nOriginal summary:\n{summary}\n\nAdversarial summary:"
    ),
    "correct_poorly_written": (
        "Given this news article and its summary, rewrite the summary so that it:\n"
        "- Is factually accurate and contains the same key information\n"
        "- Has poor writing quality: choppy sentences, awkward phrasing, some grammatical errors, "
        "repetition, and lacks cohesion between sentences\n"
        "- Is roughly the same length\n\n"
        "Article:\n{article}\n\nOriginal summary:\n{summary}\n\nPoorly-written accurate summary:"
    ),
    "meaning_shift": (
        "Given this news article and its summary, rewrite the summary so that it:\n"
        "- Reads almost identically to the original (paraphrase style)\n"
        "- But makes one subtle change that inverts or significantly alters the meaning of a key claim "
        "(e.g., changes the subject, flips a causal direction, or changes what was achieved vs. failed)\n"
        "- The change should NOT be detectable at a glance — only careful reading vs the source reveals it\n\n"
        "Article:\n{article}\n\nOriginal summary:\n{summary}\n\nMeaning-shifted paraphrase:"
    ),
}

COUNTS = {
    "fluent_inconsistent": 4,
    "correct_poorly_written": 3,
    "meaning_shift": 3,
}


def load_good_summaries():
    """Load the Llama summaries as base for adversarial generation."""
    path = DATA_DIR / "system_outputs" / "llama_summaries.json"
    with open(path) as f:
        return json.load(f)


def generate_adversarial(article: str, summary: str, adv_type: str) -> str:
    prompt = PROMPTS[adv_type].format(article=article[:2500], summary=summary)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


def main():
    summaries = load_good_summaries()

    # Sample without replacement across categories
    pool = random.sample(summaries, sum(COUNTS.values()))
    idx = 0

    examples = []
    for adv_type, count in COUNTS.items():
        for _ in range(count):
            entry = pool[idx]
            idx += 1
            print(f"Generating {adv_type} for article {entry['id']}...")
            adv_summary = generate_adversarial(entry["article"], entry["summary"], adv_type)
            examples.append({
                "id": f"adv_{entry['id']}_{adv_type}",
                "source_article_id": entry["id"],
                "article": entry["article"],
                "reference": entry["reference"],
                "good_summary": entry["summary"],         # correct, well-written
                "adversarial_summary": adv_summary,       # the adversarial variant
                "adversarial_type": adv_type,
                "model_for_good": "llama",
            })

    out_path = OUT_DIR / "adversarial_examples.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} adversarial examples to {out_path}")


if __name__ == "__main__":
    main()
