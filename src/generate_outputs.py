"""
Generate summaries for 50 CNN/DM articles using two system models (~1B, fp16).
Designed to run on a machine with CUDA GPUs.

Usage:
  # With vLLM (preferred, GPU machine):
  python src/generate_outputs.py --backend vllm

  # With HuggingFace transformers (fallback):
  python src/generate_outputs.py --backend transformers

Output:
  data/system_outputs/llama_summaries.json
  data/system_outputs/qwen_summaries.json
  data/system_outputs/all_pairs.json   -- combined for downstream use
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "system_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}

SUMMARIZATION_PROMPT = (
    "You are a professional news summarizer. "
    "Write a concise, factually accurate summary of the following news article in 3–5 sentences. "
    "Include only information explicitly stated in the article.\n\n"
    "Article:\n{article}\n\nSummary:"
)

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3  # slight diversity but mostly deterministic

VLLM_MAX_MODEL_LEN = 8192


def load_articles():
    path = RAW_DIR / "articles_50.json"
    with open(path) as f:
        return json.load(f)


def generate_vllm(model_name: str, articles: list, label: str):
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name} with vLLM (fp16)...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=VLLM_MAX_MODEL_LEN,
    )
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS)

    prompts = [SUMMARIZATION_PROMPT.format(article=a["article"][:3000]) for a in articles]
    outputs = llm.generate(prompts, sampling)

    results = []
    for article, out in zip(articles, outputs):
        results.append({
            "id": article["id"],
            "article": article["article"],
            "reference": article["highlights"],
            "summary": out.outputs[0].text.strip(),
            "model": label,
        })

    return results


def _build_pipeline(model_name: str):
    """Load a text-generation pipeline in fp16 (CUDA / MPS / CPU)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  {model_name}: fp16")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_transformers(model_name: str, articles: list, label: str):
    print(f"Loading {model_name} with transformers...")
    pipe = _build_pipeline(model_name)

    results = []
    for article in articles:
        prompt = SUMMARIZATION_PROMPT.format(article=article["article"][:3000])
        out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=True)
        generated = out[0]["generated_text"][len(prompt):].strip()
        results.append({
            "id": article["id"],
            "article": article["article"],
            "reference": article["highlights"],
            "summary": generated,
            "model": label,
        })
        print(f"  [{label}] Article {article['id']} done")

    return results


def save(results: list, label: str):
    path = OUT_DIR / f"{label}_summaries.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} summaries to {path}")


def build_pairs(llama_results: list, qwen_results: list):
    """Merge into pairwise format keyed by article id."""
    qwen_by_id = {r["id"]: r for r in qwen_results}
    pairs = []
    for r in llama_results:
        art_id = r["id"]
        if art_id in qwen_by_id:
            pairs.append({
                "id": art_id,
                "article": r["article"],
                "reference": r["reference"],
                "summary_A": r["summary"],
                "model_A": "llama",
                "summary_B": qwen_by_id[art_id]["summary"],
                "model_B": "qwen",
            })
    path = OUT_DIR / "all_pairs.json"
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="transformers")
    parser.add_argument("--models", nargs="+", choices=["llama", "qwen", "both"], default=["both"])
    args = parser.parse_args()

    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    gen = generate_vllm if args.backend == "vllm" else generate_transformers
    to_run = ["llama", "qwen"] if "both" in args.models else args.models

    all_results = {}
    for label in to_run:
        results = gen(MODELS[label], articles, label)
        save(results, label)
        all_results[label] = results

    if "llama" in all_results and "qwen" in all_results:
        build_pairs(all_results["llama"], all_results["qwen"])

    print("Done generating summaries.")


if __name__ == "__main__":
    main()
