# Execution Plan

**Course**: Evaluation Methods in NLP 2026  
**Task**: LLM as a Judge — Summarization (CNN/DailyMail)  
**Deadline**: May 3, 2026

---

## Environment Setup (do once)

```bash
conda activate fixed_res
pip install -r requirements.txt

export HF_HOME=/scratch/abraham/transformers_cache
export HF_DATASETS_CACHE=/scratch/abraham/transformers_cache

# Gemini: adversarial generation only. NVIDIA: frontier judge (NIM). Groq: prometheus=gpt-oss-20b, judgelm=qwen3-32b, llama=70b.
export GEMINI_API_KEY=AIza...
export NVIDIA_API_KEY=nvapi-...   # https://build.nvidia.com/models — optional: NVIDIA_MAX_REQUESTS_PER_MINUTE=30
export GROQ_API_KEY=gsk_...   # https://console.groq.com/ — judges: prometheus (openai/gpt-oss-20b), judgelm (qwen/qwen3-32b), llama
```

---

## Step 1 — Download data ✅ (already done)

```bash
python src/download_data.py
```

**Produces**: `data/raw/articles_50.json`, `data/raw/summeval_human.json`  
**Status**: Complete — files already exist.

---

## Step 2 — Generate system outputs

```bash
# On Apple Silicon Mac or CPU (slower):
python src/generate_outputs.py --backend transformers

# On a CUDA GPU server (faster):
python src/generate_outputs.py --backend vllm
```

**Needs**: GPU or Apple Silicon MPS (Llama-3.2-1B + Qwen2.5-1.5B in fp16 need only a few GB)  
**Produces**:
- `data/system_outputs/llama_summaries.json`
- `data/system_outputs/qwen_summaries.json`
- `data/system_outputs/all_pairs.json`

---

## Step 3 — Create adversarial examples

```bash
python src/create_adversarial.py
```

**Needs**: `llama_summaries.json` (Step 2), `GEMINI_API_KEY`  
**Produces**: `data/adversarial/adversarial_examples.json` (10 examples across 3 types)

---

## Steps 4a + 4b — Run in PARALLEL

Steps 4a and 4b are independent of each other. Start both at the same time after Step 3.

### 4a — Human annotation

Each of the 3 annotators runs the Flask app separately and rates summaries in their browser. The app covers all 50 article pairs (100 summaries) plus 20 adversarial examples = ~120 items per annotator.

```bash
# Terminal 1 — Annotator 1 (you):
ANNOTATOR_ID=ann1 python src/annotate_app.py
# Open http://localhost:5000

# Terminal 2 — Annotator 2 (colleague):
ANNOTATOR_ID=ann2 python src/annotate_app.py

# Terminal 3 — Annotator 3 (colleague):
ANNOTATOR_ID=ann3 python src/annotate_app.py
```

After **all 3 annotators** finish:

```bash
python src/merge_annotations.py   # → data/annotations/human_annotations.json
python src/compute_iaa.py         # → results/iaa_report.json
                                  # → data/annotations/gold_standard.json
```

**Produces**: `data/annotations/gold_standard.json` (mean scores across annotators — the human gold)

---

### 4b — LLM judge experiments (start immediately in a separate terminal)

```bash
# NVIDIA frontier judge (needs NVIDIA_API_KEY; client throttles ~30 RPM by default):
python src/run_experiments.py --judges nvidia --mode all --clear

# Groq judges (needs GROQ_API_KEY) — no local GPU weights:
python src/run_experiments.py --judges prometheus judgelm llama --mode all

# Full sweep (NVIDIA + Groq):
python src/run_experiments.py --judges nvidia prometheus judgelm llama --mode all --clear
```

**Needs**: `all_pairs.json` (Step 2), `adversarial_examples.json` (Step 3), `NVIDIA_API_KEY`, `GROQ_API_KEY`  
**Produces**:
- `results/direct_scores.jsonl` (~2,400 entries per NVIDIA model × 2 default models unless you change `NVIDIA_JUDGE_MODELS`)
- `results/pairwise_scores.jsonl` (~600 entries)
- `results/rubric_scores.jsonl` (~400 entries)

> **Note**: Full runs issue thousands of API calls across NVIDIA NIM and Groq. The NVIDIA judge enforces a client-side requests-per-minute cap (see `NVIDIA_MAX_REQUESTS_PER_MINUTE`). Use `--clear` only on first run to avoid duplicates. If interrupted, rerun **without** `--clear` to append remaining entries. Groq free tier may return 429: the judge client retries with exponential backoff; spread work across days if limits persist.

---

## Step 5 — Meta-evaluation

```bash
python src/meta_eval.py
```

**Needs**: Steps 4a AND 4b complete (gold standard + all result JSONL files)  
**Produces**:
- `results/meta_eval_report.json` (correlations, all bias analyses)
- `results/figures/correlation_heatmap.pdf/.png`
- `results/figures/position_bias.pdf/.png`

---

## Step 6 — Adversarial analysis

```bash
python src/adversarial_analysis.py
```

**Needs**: Step 4b complete (result JSONL files)  
**Produces**: `results/adversarial_report.json`

---

## Step 7 — Failure analysis

```bash
python src/failure_analysis.py
```

**Needs**: Steps 4a AND 4b complete  
**Produces**:
- `results/failure_taxonomy.json`
- `results/figures/failure_breakdown.pdf/.png`

---

## Dependency Graph

```
download_data (Step 1 — done)
       │
generate_outputs (Step 2)
       │
create_adversarial (Step 3)
       │
       ├──────────────────────────────┐
       │                              │
  [4a] Human annotation               [4b] Judge experiments
  annotate_app × 3                     run_experiments.py
  merge_annotations.py                 (NVIDIA + Groq judges)
  compute_iaa.py
       │                              │
  gold_standard.json         direct/pairwise/rubric_scores.jsonl
       │                              │
       └──────────────┬───────────────┘
                      │
             ┌────────┼────────┐
             │        │        │
         Step 5    Step 6   Step 7
        meta_eval  adv_analysis  failure_analysis
```

---

## Output Files Summary

| File | Produced by | Used by |
|------|------------|---------|
| `data/system_outputs/all_pairs.json` | Step 2 | Steps 3, 4a, 4b |
| `data/adversarial/adversarial_examples.json` | Step 3 | Steps 4a, 4b |
| `data/annotations/gold_standard.json` | Step 4a | Steps 5, 7 |
| `results/iaa_report.json` | Step 4a | Report |
| `results/direct_scores.jsonl` | Step 4b | Steps 5, 6, 7 |
| `results/pairwise_scores.jsonl` | Step 4b | Steps 5, 6, 7 |
| `results/rubric_scores.jsonl` | Step 4b | Step 5 |
| `results/meta_eval_report.json` | Step 5 | Report |
| `results/adversarial_report.json` | Step 6 | Report |
| `results/failure_taxonomy.json` | Step 7 | Report |
| `results/figures/` | Steps 5, 7 | Report |

---

## Quick-run script

```bash
bash run_all.sh
```

The script pauses at the annotation step and resumes once you press Enter.
