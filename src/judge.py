"""
LLM Judge runner. Supports four judges:
  - prometheus  : prometheus-eval/prometheus-7b-v2.0 (local, specialized)
  - judgelm     : BAAI/JudgeLM-7B-v1.0 (local, specialized)
  - llama       : meta-llama/Meta-Llama-3.1-8B-Instruct (local, zero-shot general)
  - gemini      : gemini-2.5-flash via Google Gemini API (frontier)

Usage:
  from src.judge import Judge
  judge = Judge("gemini")
  score = judge.direct_score(article, summary, dimension="coherence", variant="standard")
  winner = judge.pairwise(article, summary_a, summary_b, variant="cot")
  scores = judge.rubric(article, summary, variant="standard")
"""

import json
import os
import re
import time
from typing import Optional

from src.prompts import (
    DIMENSIONS,
    build_direct,
    build_pairwise,
    build_rubric,
)

JUDGE_MODELS = {
    "prometheus": "prometheus-eval/prometheus-7b-v2.0",
    "judgelm": "BAAI/JudgeLM-7B-v1.0",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def _extract_score_1_5(text: str) -> Optional[int]:
    """Parse a 1–5 integer score from model output."""
    # Look for "Score: N" pattern first
    m = re.search(r"[Ss]core[:\s]+([1-5])", text)
    if m:
        return int(m.group(1))
    # Fallback: last standalone digit 1–5
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        return int(digits[-1])
    return None


def _extract_winner(text: str) -> Optional[str]:
    """Parse 'A' or 'B' winner from pairwise output."""
    m = re.search(r"[Ww]inner[:\s]+([AB])", text)
    if m:
        return m.group(1)
    # Last standalone A or B
    tokens = re.findall(r"\b([AB])\b", text)
    if tokens:
        return tokens[-1]
    return None


def _extract_rubric_scores(text: str) -> dict:
    """Parse multi-dimension rubric output into {dim: score} dict."""
    scores = {}
    for dim in DIMENSIONS:
        m = re.search(rf"{dim.capitalize()}[:\s]+([1-5])", text, re.IGNORECASE)
        if m:
            scores[dim] = int(m.group(1))
    return scores


class _LocalJudge:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"Loading {self.model_name}...")

        if torch.cuda.is_available():
            # INT4 quantization via bitsandbytes: halves VRAM vs fp16
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                return_full_text=False,
            )
            print(f"  {self.model_name}: loaded in INT4 (bitsandbytes)")
        else:
            # Apple Silicon MPS or CPU — bitsandbytes requires CUDA; use fp16
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                return_full_text=False,
            )
            print(f"  {self.model_name}: loaded in fp16 (MPS/CPU)")

    def generate(self, prompt: str) -> str:
        self._load()
        out = self._pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
        )
        return out[0]["generated_text"].strip()


class _GeminiJudge:
    def __init__(self):
        from google import genai
        from google.genai import types as genai_types
        self._types = genai_types
        self.client = genai.Client()  # reads GEMINI_API_KEY

    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=512,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


class Judge:
    def __init__(self, judge_name: str):
        """
        judge_name: one of 'prometheus', 'judgelm', 'llama', 'gemini'
        """
        self.name = judge_name
        if judge_name == "gemini":
            self._backend = _GeminiJudge()
        elif judge_name in JUDGE_MODELS:
            self._backend = _LocalJudge(JUDGE_MODELS[judge_name])
        else:
            raise ValueError(f"Unknown judge: {judge_name}")

    def _call(self, prompt: str) -> str:
        return self._backend.generate(prompt)

    def direct_score(
        self,
        article: str,
        summary: str,
        dimension: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_direct(article, summary, dimension, variant, reference)
        raw = self._call(prompt)
        score = _extract_score_1_5(raw)
        return {"score": score, "raw": raw, "judge": self.name,
                "dimension": dimension, "variant": variant, "mode": "direct"}

    def direct_score_all_dims(
        self,
        article: str,
        summary: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        results = {}
        for dim in DIMENSIONS:
            results[dim] = self.direct_score(article, summary, dim, variant, reference)
        return results

    def pairwise(
        self,
        article: str,
        summary_a: str,
        summary_b: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_pairwise(article, summary_a, summary_b, variant, reference)
        raw = self._call(prompt)
        winner = _extract_winner(raw)
        return {"winner": winner, "raw": raw, "judge": self.name,
                "variant": variant, "mode": "pairwise"}

    def rubric(
        self,
        article: str,
        summary: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_rubric(article, summary, variant, reference)
        raw = self._call(prompt)
        scores = _extract_rubric_scores(raw)
        return {"scores": scores, "raw": raw, "judge": self.name,
                "variant": variant, "mode": "rubric"}
