# Miniproject: LLM as a Judge

**Course: Evaluation Methods in NLP- 2026**

---

## Objective

This project goes beyond using LLMs as evaluators. You will treat the LLM itself as an **object of study**:

- When does it behave like a reliable evaluator?
- When does it fail systematically?
- Can LLM-based evaluation be trusted in research?

---

## Part 1: Task Setup (Controlled Design)

Select one task:

- Machine Translation
- Summarization
- Question Answering
- Dialogue / Instruction Following

Requirements:

- 40–60 examples
- At least 2 system outputs per input
- Include challenging cases (ambiguity, hallucination, noisy inputs)

---

## Part 2: Human Evaluation (Gold Standard)

- Design a clear annotation rubric
- Minimum 3 annotators
- Compute agreement:
  - Cohen's Kappa or Krippendorff's Alpha

Output: A clean human gold ranking or scoring.

---

## Part 3: LLM as Judge

Use at least one LLM and implement:

1. Direct scoring
2. Pairwise comparison
3. Rubric-based structured evaluation

### Prompt Sensitivity Study

Vary:

- Prompt wording
- Order of candidates (A/B swap)
- With vs without reference

---

## Part 4: Meta-Evaluation

Evaluate the evaluator.

### Metrics

- Spearman / Kendall correlation
- Pairwise agreement
- Consistency across prompts

### Bias Analysis

Analyze:

- Position bias
- Verbosity bias
- Reference bias
- Self-preference bias (if applicable)

---

## Part 5: Adversarial Testing

Create at least 10 adversarial examples:

- Fluent but incorrect outputs
- Correct but poorly written outputs
- Paraphrase vs meaning shift

Test whether LLM prefers surface quality over correctness.

---

## Part 6: Failure Analysis

Develop a taxonomy of failure modes, such as:

- Over-trusting fluency
- Ignoring factual errors
- Inconsistent reasoning

---

## Part 7: Critical Questions

Answer clearly:

1. When does LLM evaluation align with humans?
2. Where does it fail and why?
3. Is it internally consistent?
4. How sensitive is it to prompts?
5. Can it replace human evaluation?

---

## Part 8: Extension (Choose One)

- Compare multiple LLM judges
- Cross-lingual evaluation
- Self-evaluation vs external evaluation
- Chain-of-thought vs no reasoning
- Score calibration

---

## Deliverables

- Report (6–8 pages, EMNLP style)
- Appendix (prompts, examples, guidelines)
- Code (reproducible pipeline)

---

## Important Note

If your conclusion is simply that "LLMs correlate well with humans" without deeper analysis, the assignment is incomplete.

---

## Timeline

**Submission Deadline: [3rd May 2026], No more extension!**
