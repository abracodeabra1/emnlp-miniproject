#!/usr/bin/env bash
# Master pipeline: run all steps in order.
#
# Required environment variable:
#   GEMINI_API_KEY  — for both adversarial generation and LLM judge experiments
#
# Usage: bash run_all.sh
#
# NOTE: Steps 4a (human annotation) and 4b (judge experiments) can run in
# PARALLEL after step 3 completes. See comments below.

set -e
cd "$(dirname "$0")"

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set. Export it before running."
    exit 1
fi

# ── Step 1: Download data (already done if data/raw/ is populated) ────────────
echo "=== Step 1: Download data ==="
python src/download_data.py

# ── Step 2: Generate system outputs (Llama + Mistral summaries) ───────────────
echo "=== Step 2: Generate system outputs (Llama + Mistral) ==="
# Change --backend to vllm if running on a CUDA GPU server
python src/generate_outputs.py --backend transformers

# ── Step 3: Create adversarial examples ───────────────────────────────────────
echo "=== Step 3: Create adversarial examples ==="
python src/create_adversarial.py

# ── Steps 4a + 4b run in PARALLEL ─────────────────────────────────────────────
#
#   4a — Human annotation (you + 2 colleagues):
#        Each annotator runs the Flask app independently and rates summaries in
#        their browser. After all 3 finish, run merge + IAA.
#
#   4b — LLM judge experiments (can start immediately in a separate terminal):
#        These do NOT depend on human annotations — start them now to save time.
#
echo ""
echo "=== Steps 4a + 4b: Run IN PARALLEL ==="
echo ""
echo "  4a — Human annotation (open 3 separate terminals):"
echo "         ANNOTATOR_ID=ann1 python src/annotate_app.py  → http://localhost:5000"
echo "         ANNOTATOR_ID=ann2 python src/annotate_app.py"
echo "         ANNOTATOR_ID=ann3 python src/annotate_app.py"
echo "       After all 3 finish:"
echo "         python src/merge_annotations.py"
echo "         python src/compute_iaa.py"
echo ""
echo "  4b — LLM judge experiments (run in a separate terminal NOW):"
echo "         python src/run_experiments.py --judges gemini --mode all --clear"
echo "         # If local judges available (GPU / Apple Silicon MPS):"
echo "         python src/run_experiments.py --judges prometheus judgelm llama --mode all"
echo ""
read -p "Press Enter once BOTH annotation (4a) AND judge experiments (4b) are complete..."

# ── Step 5: Meta-evaluation ───────────────────────────────────────────────────
echo "=== Step 5: Meta-evaluation ==="
python src/meta_eval.py

# ── Step 6: Adversarial analysis ──────────────────────────────────────────────
echo "=== Step 6: Adversarial analysis ==="
python src/adversarial_analysis.py

# ── Step 7: Failure analysis ──────────────────────────────────────────────────
echo "=== Step 7: Failure analysis ==="
python src/failure_analysis.py

echo ""
echo "All done. Results are in results/"
echo "Next: open notebooks/ for visualizations and write the report."
