#!/bin/bash
# Full reasoning-gym evaluation (101 datasets, 10 samples each)
# Assumes vLLM is already running on port 8000

set -e
cd /workspace/reasoning-gym

MODEL="${1:-Qwen/Qwen3-4B}"
OUTPUT_TAG="${2:-baseline}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"

echo "============================================"
echo "Full Reasoning-Gym Evaluation"
echo "============================================"
echo "Model:      $MODEL"
echo "Output tag: $OUTPUT_TAG"
echo "URL:        $VLLM_URL"
echo "Datasets:   101"
echo "Samples:    10 per dataset"
echo "Total:      1010 samples"
echo "============================================"

# Check vLLM is running
if ! curl -s "$VLLM_URL/models" > /dev/null; then
    echo "ERROR: vLLM not running at $VLLM_URL"
    echo "Start it with: cd /workspace/vllm && ./vllm.sh Qwen/Qwen3-4B"
    exit 1
fi

echo "vLLM is running, starting eval..."
echo "Started at: $(date)"
echo ""

source .venv/bin/activate

python eval/eval.py \
    --config experiments/full-eval.yaml \
    --base-url "$VLLM_URL" \
    --model "$MODEL" \
    --api-key dummy \
    --max-concurrent 64 \
    --max-datasets 20 \
    --output-dir "experiments/results/${OUTPUT_TAG}"

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Finished at: $(date)"
echo "Results: experiments/results/${OUTPUT_TAG}/"
echo "============================================"
