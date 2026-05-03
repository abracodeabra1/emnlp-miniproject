"""
All prompt templates for LLM judge experiments.

Three evaluation modes × three wording variants × with/without reference.
"""

# ─── Dimension definitions (used in rubric prompts) ───────────────────────────

DIMENSION_RUBRICS = {
    "coherence": (
        "Coherence (1–5): Does the summary have a clear, logical structure? "
        "Does it read as a unified whole, with sentences that flow naturally together? "
        "1=completely incoherent, 5=perfectly coherent and well-organized."
    ),
    "consistency": (
        "Consistency (1–5): Are all facts in the summary supported by the source article? "
        "Penalize any information that contradicts or is not present in the article (hallucinations). "
        "1=multiple factual errors, 5=fully consistent with no hallucinations."
    ),
    "fluency": (
        "Fluency (1–5): Is the language grammatically correct, natural, and easy to read? "
        "1=unreadable, 5=perfect grammar and natural phrasing."
    ),
    "relevance": (
        "Relevance (1–5): Does the summary focus on the most important information from the article? "
        "Penalize summaries that include trivial details or omit key events. "
        "1=completely irrelevant, 5=covers all key points with no padding."
    ),
}

DIMENSIONS = list(DIMENSION_RUBRICS.keys())

# ─── Template builders ────────────────────────────────────────────────────────

def _ref_block(reference: str | None) -> str:
    if reference:
        return f"\nReference summary (for calibration only):\n{reference}\n"
    return ""


# ── Mode A: Direct Scoring ─────────────────────────────────────────────────────

DIRECT_MINIMAL = (
    "Rate the following summary on {dimension} from 1 to 5.\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "Score (1-5):"
)

DIRECT_STANDARD = (
    "You are an expert evaluator of text summaries.\n"
    "Rate the following summary on the dimension of {dimension}.\n\n"
    "{rubric}\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "Score (integer 1–5):"
)

DIRECT_COT = (
    "You are an expert evaluator of text summaries.\n"
    "Evaluate the following summary on the dimension of {dimension}.\n\n"
    "{rubric}\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "First, explain your reasoning step by step. "
    "Then on a new line write: Score: <integer 1–5>"
)


# ── Mode B: Pairwise Comparison ────────────────────────────────────────────────

PAIRWISE_MINIMAL = (
    "Which summary is better overall?\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary A:\n{summary_a}\n\n"
    "Summary B:\n{summary_b}\n\n"
    "Answer with only 'A' or 'B':"
)

PAIRWISE_STANDARD = (
    "You are an expert evaluator of text summaries.\n"
    "Compare the two summaries below and decide which is better overall quality "
    "(considering coherence, consistency with the article, fluency, and relevance).\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary A:\n{summary_a}\n\n"
    "Summary B:\n{summary_b}\n\n"
    "Which summary is better? Answer with only 'A' or 'B':"
)

PAIRWISE_COT = (
    "You are an expert evaluator of text summaries.\n"
    "Compare the two summaries below across all quality dimensions "
    "(coherence, factual consistency, fluency, relevance).\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary A:\n{summary_a}\n\n"
    "Summary B:\n{summary_b}\n\n"
    "Think step by step about the strengths and weaknesses of each summary. "
    "Then on a final line write: Winner: A  or  Winner: B"
)


# ── Mode C: Rubric-Based (all 4 dimensions at once) ────────────────────────────

RUBRIC_STANDARD = (
    "You are an expert evaluator of text summaries.\n"
    "Score the following summary on each of the four dimensions below.\n\n"
    "{rubrics}\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "Provide your scores in this exact format:\n"
    "Coherence: <1-5>\n"
    "Consistency: <1-5>\n"
    "Fluency: <1-5>\n"
    "Relevance: <1-5>"
)

RUBRIC_COT = (
    "You are an expert evaluator of text summaries.\n"
    "Score the following summary on each of the four dimensions below.\n\n"
    "{rubrics}\n\n"
    "{ref}"
    "Article:\n{article}\n\n"
    "Summary:\n{summary}\n\n"
    "For each dimension, first explain your reasoning, then give a score.\n"
    "Format:\n"
    "Coherence reasoning: ...\nCoherence: <1-5>\n"
    "Consistency reasoning: ...\nConsistency: <1-5>\n"
    "Fluency reasoning: ...\nFluency: <1-5>\n"
    "Relevance reasoning: ...\nRelevance: <1-5>"
)


# ─── Public interface ─────────────────────────────────────────────────────────

ALL_RUBRICS_TEXT = "\n".join(DIMENSION_RUBRICS.values())


def build_direct(
    article: str,
    summary: str,
    dimension: str,
    variant: str = "standard",   # minimal | standard | cot
    reference: str | None = None,
) -> str:
    ref = _ref_block(reference)
    rubric = DIMENSION_RUBRICS[dimension]
    tmpl = {"minimal": DIRECT_MINIMAL, "standard": DIRECT_STANDARD, "cot": DIRECT_COT}[variant]
    return tmpl.format(
        dimension=dimension,
        rubric=rubric,
        ref=ref,
        article=article[:3000],
        summary=summary,
    )


def build_pairwise(
    article: str,
    summary_a: str,
    summary_b: str,
    variant: str = "standard",
    reference: str | None = None,
) -> str:
    ref = _ref_block(reference)
    tmpl = {"minimal": PAIRWISE_MINIMAL, "standard": PAIRWISE_STANDARD, "cot": PAIRWISE_COT}[variant]
    return tmpl.format(ref=ref, article=article[:3000], summary_a=summary_a, summary_b=summary_b)


def build_rubric(
    article: str,
    summary: str,
    variant: str = "standard",   # standard | cot
    reference: str | None = None,
) -> str:
    ref = _ref_block(reference)
    tmpl = {"standard": RUBRIC_STANDARD, "cot": RUBRIC_COT}[variant]
    return tmpl.format(rubrics=ALL_RUBRICS_TEXT, ref=ref, article=article[:3000], summary=summary)
