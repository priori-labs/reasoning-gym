#!/bin/bash
# Quick test of reasoning-gym eval (5 datasets, 1 sample each)
# Assumes vLLM is already running on port 8000

set -e
cd /workspace/reasoning-gym

MODEL="${1:-Qwen/Qwen3-4B}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"

echo "============================================"
echo "Quick Test: Reasoning-Gym Eval"
echo "============================================"
echo "Model: $MODEL"
echo "URL:   $VLLM_URL"
echo "============================================"

# Check vLLM is running
if ! curl -s "$VLLM_URL/models" > /dev/null; then
    echo "ERROR: vLLM not running at $VLLM_URL"
    echo "Start it with: cd /workspace/vllm && ./vllm.sh Qwen/Qwen3-4B"
    exit 1
fi

echo "vLLM is running, starting eval..."

source .venv/bin/activate

python eval/eval.py \
    --config experiments/quick-test.yaml \
    --base-url "$VLLM_URL" \
    --model "$MODEL" \
    --api-key dummy \
    --max-concurrent 5 \
    --max-datasets 5

echo ""
echo "Done! Check results in experiments/results/"
