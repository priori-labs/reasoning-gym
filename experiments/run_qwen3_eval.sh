#!/bin/bash
# Eval Qwen3-8B on reasoning-gym tasks
#
# Usage:
#   ./run_qwen3_eval.sh [base|finetuned] [--quick]
#
# Examples:
#   ./run_qwen3_eval.sh base           # Eval base Qwen3-8B
#   ./run_qwen3_eval.sh finetuned      # Eval your CPT fine-tuned model
#   ./run_qwen3_eval.sh base --quick   # Quick test (1 sample per dataset)

set -e

MODEL_TYPE="${1:-base}"
shift 2>/dev/null || true

# Model paths
BASE_MODEL="Qwen/Qwen3-8B"
FINETUNED_MODEL="/workspace/core-reasoning/saves/qwen3-8b-cpt-merged-exp4"

# Select model based on argument
if [ "$MODEL_TYPE" = "finetuned" ] || [ "$MODEL_TYPE" = "ft" ]; then
    MODEL_PATH="$FINETUNED_MODEL"
    MODEL_NAME="$FINETUNED_MODEL"  # Must match exactly what vLLM serves
    echo "Using fine-tuned model: $MODEL_PATH"
else
    MODEL_PATH="$BASE_MODEL"
    MODEL_NAME="$BASE_MODEL"  # Must match exactly what vLLM serves
    echo "Using base model: $MODEL_PATH"
fi

# GPU settings (adjust based on your hardware)
# H100 80GB: MAX_CONCURRENT=256, MAX_DATASETS=50
# A100 40GB: MAX_CONCURRENT=128, MAX_DATASETS=30
# RTX 4090:  MAX_CONCURRENT=64,  MAX_DATASETS=20
export MAX_CONCURRENT="${MAX_CONCURRENT:-256}"
export MAX_DATASETS="${MAX_DATASETS:-50}"
export VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"

VLLM_PORT=8000

echo "============================================"
echo "Reasoning-Gym Evaluation for Qwen3-8B"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Concurrency: $MAX_CONCURRENT"
echo "Max datasets: $MAX_DATASETS"
echo "============================================"

# Check if vLLM is already running
if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
    echo "vLLM server already running at $VLLM_URL"
    VLLM_ALREADY_RUNNING=true
else
    VLLM_ALREADY_RUNNING=false
    echo "Starting vLLM server..."

    # Start vLLM in background
    vllm serve "$MODEL_PATH" \
        --port $VLLM_PORT \
        --tensor-parallel-size 1 \
        --max-model-len 40960 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        2>&1 | tee /tmp/vllm_server.log &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for server to be ready
    echo "Waiting for vLLM server to start..."
    for i in {1..120}; do
        if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
            echo "vLLM server ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: vLLM server failed to start. Check /tmp/vllm_server.log"
            exit 1
        fi
        sleep 2
    done
fi

# Run evaluation
cd /workspace/reasoning-gym

echo ""
echo "Starting evaluation..."
echo ""

./experiments/run_local_vllm.sh "$MODEL_NAME" "$@"

# Results location
RESULTS_DIR="experiments/results"
LATEST_RESULT=$(ls -td "$RESULTS_DIR"/*/ 2>/dev/null | head -1)

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Results: $LATEST_RESULT"
echo "============================================"

# Optionally stop vLLM if we started it
if [ "$VLLM_ALREADY_RUNNING" = false ] && [ -n "$VLLM_PID" ]; then
    echo ""
    read -p "Stop vLLM server? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $VLLM_PID 2>/dev/null || true
        echo "vLLM server stopped."
    fi
fi
