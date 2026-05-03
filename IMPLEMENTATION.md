# Implementation Summary: LLM as a Judge

**Course**: Evaluation Methods in NLP 2026  
**Deadline**: May 3, 2026  
**Task**: Summarization (CNN/DailyMail via SummEval corpus)

---

## Research Design

### Central Question
Do LLM judges fail uniformly, or is failure **dimension-specific**? The hypothesis is that judges correlate well with humans on surface dimensions (fluency, coherence) but fail on semantic ones (consistency/faithfulness), because consistency requires fact-checking against a source document rather than surface-level pattern matching.

### System Models (what gets judged)
| Model | HuggingFace ID |
|---|---|
| Llama-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` |

Both run in **fp16** (transformers or vLLM). They are small, comparable instruct models for a lightweight pipeline.

### Judge Models (Part 3 + Part 8 extension)
All judges below are **API-only** (no local model weights).

| Judge id | Backend | Model ID | Notes |
|---|---|---|---|
| `nvidia` | [NVIDIA NIM](https://build.nvidia.com/models) (OpenAI-compatible) | Default **two** models: `minimaxai/minimax-m2.7`, `moonshotai/kimi-k2-thinking` (override with `NVIDIA_JUDGE_MODELS` comma-separated, or a single `NVIDIA_JUDGE_MODEL`) | Set `NVIDIA_API_KEY`; optional `NVIDIA_MAX_REQUESTS_PER_MINUTE` (default 30). JSONL `judge` is `nvidia/<model id>`. |
| `prometheus` | Groq | `llama-3.1-8b-instant` | Set `GROQ_API_KEY` |
| `judgelm` | Groq | `openai/gpt-oss-20b` | Set `GROQ_API_KEY` |
| `llama` | Groq | `llama-3.3-70b-versatile` | Set `GROQ_API_KEY`; strong general judge vs. small system Llama |

Groq’s free/developer tier is **request- and token-limited** (on the order of ~1k requests/day depending on plan); scale `run_experiments.py` batches or split runs across days if you hit 429s (the client retries with backoff).

The NVIDIA judge uses a **sliding-window requests-per-minute** limiter (default 30 RPM) before each call, plus retries on 429/503. Tune `NVIDIA_MAX_REQUESTS_PER_MINUTE` to match your NVIDIA account limits.

### Evaluation Dimensions
All models are scored on four dimensions (1–5 Likert scale), matching SummEval:

| Dimension | What it tests |
|---|---|
| **Coherence** | Logical structure and flow |
| **Consistency** | Factual accuracy vs. source (hallucination detection) |
| **Fluency** | Grammar and naturalness |
| **Relevance** | Coverage of key information |

---

## Project Structure

```
miniproject/
├── data/
│   ├── raw/
│   │   ├── articles_50.json          # 50 CNN/DM source articles (selected)
│   │   ├── articles_100.json         # Full 100-article pool
│   │   └── summeval_human.json       # SummEval fallback annotations
│   ├── system_outputs/
│   │   ├── llama_summaries.json      # Llama-3.2-1B outputs
│   │   ├── qwen_summaries.json       # Qwen2.5-1.5B outputs
│   │   └── all_pairs.json            # Merged pairwise format
│   ├── adversarial/
│   │   └── adversarial_examples.json # 10 adversarial examples (3 types)
│   └── annotations/
│       ├── raw/                      # Per-annotator files (ann1/2/3)
│       ├── human_annotations.json    # Merged annotation data
│       ├── gold_standard.json        # Mean scores per item (our annotations)
│       └── gold_standard_summeval.json # SummEval fallback gold
├── src/
│   ├── download_data.py
│   ├── generate_outputs.py
│   ├── create_adversarial.py
│   ├── annotate_app.py
│   ├── merge_annotations.py
│   ├── compute_iaa.py
│   ├── build_summeval_fallback.py
│   ├── prompts.py
│   ├── judge.py
│   ├── run_experiments.py
│   ├── meta_eval.py
│   ├── adversarial_analysis.py
│   └── failure_analysis.py
├── notebooks/                        # Exploratory analysis
├── results/
│   ├── direct_scores.jsonl
│   ├── pairwise_scores.jsonl
│   ├── rubric_scores.jsonl
│   ├── iaa_report.json
│   ├── meta_eval_report.json
│   ├── adversarial_report.json
│   ├── failure_taxonomy.json
│   └── figures/
├── report/                           # EMNLP-style LaTeX paper
├── requirements.txt
└── run_all.sh
```

---

## Execution Pipeline

### Prerequisites
```bash
conda activate fixed_res
export HF_HOME=/scratch/abraham/transformers_cache
export HF_DATASETS_CACHE=/scratch/abraham/transformers_cache
export GEMINI_API_KEY=...     # adversarial generation only (`create_adversarial.py`)
export NVIDIA_API_KEY=...    # frontier judge (`--judges nvidia`); see https://build.nvidia.com/models
export GROQ_API_KEY=...       # prometheus / judgelm / llama judges
# On GPU machine (optional, for faster system output generation): pip install vllm
```

Keep `HF_HOME` and `HF_DATASETS_CACHE` at `/scratch/abraham/transformers_cache` (do not point them at a repo-local `.cache`); the account needs write access to that directory.

### Step 1 — Download Data (done)
```bash
python src/download_data.py
```
Downloads 50 CNN/DM articles and SummEval fallback annotations.

### Step 2 — Generate System Outputs
```bash
# Preferred (GPU machine with vLLM):
python src/generate_outputs.py --backend vllm

# Fallback (HuggingFace transformers):
python src/generate_outputs.py --backend transformers
```
Generates summaries for 50 articles × 2 models = 100 summaries. Saves to `data/system_outputs/`.

### Step 3 — Create Adversarial Examples
```bash
python src/create_adversarial.py
```
Uses GPT-4o to create 10 adversarial summaries in 3 categories (see Part 5 below). Saves to `data/adversarial/`.

### Step 4 — Human Annotation
Each of the 3 annotators runs the Flask app independently:
```bash
ANNOTATOR_ID=ann1 python src/annotate_app.py   # you — http://localhost:5000
ANNOTATOR_ID=ann2 python src/annotate_app.py   # colleague 1
ANNOTATOR_ID=ann3 python src/annotate_app.py   # colleague 2
```
The app presents source article + summary side-by-side with 4 sliders (1–5). Annotations are saved incrementally to `data/annotations/raw/`. Each annotator rates ~110 summaries (~3–4 hours).

After all 3 annotators finish:
```bash
python src/merge_annotations.py   # combine files
python src/compute_iaa.py         # Krippendorff α + Cohen κ → results/iaa_report.json
```

**Fallback**: If annotation cannot be completed, run:
```bash
python src/build_summeval_fallback.py
# Then point gold_standard path to gold_standard_summeval.json
```

### Step 5 — Run Judge Experiments
```bash
# NVIDIA frontier judge + Groq judges (API keys only; no local judge weights):
python src/run_experiments.py --judges nvidia prometheus judgelm llama --mode all

# Or run a single mode:
python src/run_experiments.py --judges llama --mode direct
```

### Step 6 — Meta-Evaluation
```bash
python src/meta_eval.py
```
Outputs `results/meta_eval_report.json` and two figures.

### Step 7 — Adversarial + Failure Analysis
```bash
python src/adversarial_analysis.py
python src/failure_analysis.py
```

### Full pipeline (non-interactive):
```bash
bash run_all.sh
```

---

## Experiments in Detail

### Part 3: LLM as Judge — Three Modes

**Mode A — Direct Scoring**: Each judge scores a single summary per dimension.  
**Mode B — Pairwise Comparison**: Judge picks the better of two summaries (run A→B and B→A for position bias).  
**Mode C — Rubric-Based**: Judge scores all 4 dimensions in one call with explicit criteria.

### Prompt Sensitivity Study
Each mode is run with 3 prompt variants and 2 reference conditions:

| Variant | Description |
|---|---|
| `minimal` | One-sentence instruction, no criteria |
| `standard` | Brief rubric per dimension |
| `cot` | Full criteria + "think step by step" before scoring |
| `with_reference` | Include CNN/DM highlights as anchor |
| `no_reference` | No reference provided |

**Total inference calls**: thousands of HTTP calls to **Gemini** and **Groq** (no local judge inference).

### Part 4: Bias Analysis

| Bias | Method |
|---|---|
| **Position bias** | Compare A→B vs B→A win rates; χ² test. Measure `position_bias_score = 2·|first_win_rate − 0.5|` |
| **Verbosity bias** | Spearman r between summary word count and judge score per dimension |
| **Reference bias** | Δ Spearman r (with ref vs without ref) per judge per dimension |
| **Self-preference** | Llama judge win rate for Llama outputs vs Qwen outputs; compare to other judges |
| **Prompt sensitivity** | Std deviation of Spearman r across 3 prompt variants |

### Part 5: Adversarial Examples

10 examples across 3 types (generated via GPT-4o from good Llama summaries):

| Type | Count | Description |
|---|---|---|
| `fluent_inconsistent` | 4 | Sounds fluent and confident; contains 2–3 subtle factual errors |
| `correct_poorly_written` | 3 | Accurate content; choppy sentences, grammatical errors |
| `meaning_shift` | 3 | Near-paraphrase of a good summary that inverts a key claim |

**Test**: Do judges prefer `fluent_inconsistent` over `correct_poorly_written`? (Expected: yes, especially for smaller judges.)

### Part 6: Failure Taxonomy

| Mode | Operationalization |
|---|---|
| Over-trusting fluency | Fluency score ≥ 4 but human consistency ≤ 2 |
| Ignoring factual errors | LLM consistency score > human consistency score by ≥ 1.5 |
| Penalizing poor style | Fluency score << human score when consistency is high |
| Position-dependent reversal | Judge picks A over B, but also picks A (same original model) when order is B→A |
| Dimension conflation | Std deviation of all 4 dimension scores < 0.5 |

---

## Key Papers to Cite

| Paper | Relevance |
|---|---|
| Zheng et al. 2023 — MT-Bench ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) | Foundational biases: position, verbosity, self-enhancement |
| Wang et al. 2024 — ACL ([arXiv:2305.17926](https://arxiv.org/abs/2305.17926)) | Position bias methodology; our BPC analysis replicates this |
| Liu et al. 2023 — G-Eval, EMNLP ([arXiv:2303.16634](https://arxiv.org/abs/2303.16634)) | Same task (summarization), CoT evaluation; direct comparison point |
| Fabbri et al. 2021 — SummEval, TACL | Dataset; use their IAA as a benchmark for our annotation quality |
| Kim et al. 2024 — Prometheus-2 ([arXiv:2405.01535](https://arxiv.org/abs/2405.01535)) | Primary specialized judge model |
| Zeng et al. 2023 — JudgeLM, ICLR 2025 | Second specialized judge model |
| Dubois et al. 2024 — AlpacaEval 2 | Length-bias correction methodology |
| Survey ([arXiv:2411.15594](https://arxiv.org/abs/2411.15594)) | Background and bias taxonomy |

---

## Expected Key Findings

1. **Dimension gap**: Spearman ρ with human gold will be notably higher on `fluency`/`coherence` than `consistency` across all judges, because consistency requires source-grounded fact verification.
2. **Position bias**: All judges show measurable position bias; smaller open-source judges (JudgeLM, Prometheus) show stronger bias than GPT-4o.
3. **Self-preference**: Llama judge will show elevated win rate for Llama-generated summaries vs other judges.
4. **CoT helps consistency**: The `cot` prompt variant will improve Spearman ρ on `consistency` most (rubric + reasoning forces the judge to compare to source).
5. **Adversarial vulnerability**: Judges will prefer `fluent_inconsistent` over `correct_poorly_written` at a measurable rate, especially for `judgelm` and `llama` (general) judges.

---

## 6-Day Timeline

| Day | Date | Milestone |
|---|---|---|
| 1 | Apr 27 | ✅ Env setup, data download, all source files written |
| 2 | Apr 28 | Generate system outputs + adversarial examples; deploy annotation app |
| 3 | Apr 29 | Complete annotation; compute IAA; run GPT-4o experiments |
| 4 | Apr 30 | Run open-source judge experiments; prompt sensitivity analysis |
| 5 | May 1 | Meta-eval, adversarial analysis, failure taxonomy; start report |
| 6 | May 2 | Finish report (6–8 pages, EMNLP style); clean code + appendix |
| 7 | May 3 | Final review + submit |
