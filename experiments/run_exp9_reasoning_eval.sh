#!/bin/bash
# =============================================================================
# Exp9 Reasoning-Gym Evaluation
# =============================================================================
# Compare Qwen3-4B Instruct vs Exp9 SFT-LoRA adapter on reasoning-gym tasks
#
# Purpose: Test if SFT training improved reasoning capabilities
#
# Usage:
#   ./experiments/run_exp9_reasoning_eval.sh [options]
#
# Options:
#   --baseline-only    Only run baseline eval
#   --adapter-only     Only run adapter eval (assumes vLLM running)
#   --samples N        Samples per dataset (default: 50)
#   --quick            Use 5 samples (quick test)
#
# =============================================================================

set -e

SCRIPT_DIR="$(dirname "$0")"
cd /workspace/reasoning-gym

# Configuration
SAMPLES_PER_DATASET=50
VLLM_PORT=8000
VLLM_URL="http://localhost:8000/v1"

# Models
BASE_MODEL="Qwen/Qwen3-4B"
ADAPTER_PATH="/workspace/core-reasoning/saves/qwen3-4b-instruct-sft-lora-exp9"
ADAPTER_NAME="exp9-sft-lora"

# GPU settings for A40 48GB
export MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
export MAX_DATASETS="${MAX_DATASETS:-20}"

# Flags
RUN_BASELINE=true
RUN_ADAPTER=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-only)
            RUN_ADAPTER=false
            shift
            ;;
        --adapter-only)
            RUN_BASELINE=false
            shift
            ;;
        --samples)
            SAMPLES_PER_DATASET="$2"
            shift 2
            ;;
        --quick)
            SAMPLES_PER_DATASET=5
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "Exp9 Reasoning-Gym Evaluation"
echo "============================================"
echo "Samples per dataset: $SAMPLES_PER_DATASET"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Max datasets: $MAX_DATASETS"
echo ""
echo "Models:"
echo "  Base:    $BASE_MODEL"
echo "  Adapter: $ADAPTER_PATH"
echo ""
echo "Run baseline: $RUN_BASELINE"
echo "Run adapter:  $RUN_ADAPTER"
echo "============================================"

# Create temp config with specified sample size
TEMP_CONFIG=$(mktemp).yaml
sed "s/default_size: [0-9]*/default_size: $SAMPLES_PER_DATASET/" \
    "$SCRIPT_DIR/qwen3-4b-exp9.yaml" > "$TEMP_CONFIG"

echo "Created temp config with default_size: $SAMPLES_PER_DATASET"

# Function to kill vLLM server
kill_vllm() {
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
}

# Function to start vLLM server
start_vllm() {
    local WITH_LORA="$1"

    kill_vllm

    if [ "$WITH_LORA" = "true" ]; then
        echo "Starting vLLM with LoRA adapter..."
        uv run vllm serve "$BASE_MODEL" \
            --port $VLLM_PORT \
            --tensor-parallel-size 1 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            --enable-lora \
            --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
            2>&1 | tee /tmp/vllm_exp9_lora.log &
    else
        echo "Starting vLLM without adapter (baseline)..."
        uv run vllm serve "$BASE_MODEL" \
            --port $VLLM_PORT \
            --tensor-parallel-size 1 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            2>&1 | tee /tmp/vllm_exp9_baseline.log &
    fi

    VLLM_PID=$!

    # Wait for server
    echo "Waiting for vLLM server..."
    for i in {1..120}; do
        if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
            echo "vLLM server ready!"
            return 0
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: vLLM server failed to start"
            cat /tmp/vllm_exp9_*.log
            return 1
        fi
        sleep 2
    done
}

# Function to run evaluation
run_eval() {
    local MODEL_NAME="$1"
    local OUTPUT_NAME="$2"

    echo ""
    echo "============================================"
    echo "Evaluating: $OUTPUT_NAME"
    echo "Model: $MODEL_NAME"
    echo "============================================"

    uv run python "$SCRIPT_DIR/../eval/eval.py" \
        --config "$TEMP_CONFIG" \
        --base-url "$VLLM_URL" \
        --model "$MODEL_NAME" \
        --api-key dummy \
        --max-concurrent "$MAX_CONCURRENT" \
        --max-datasets "$MAX_DATASETS" \
        --output-dir "experiments/results/exp9_${OUTPUT_NAME}"
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo ""
echo "Starting evaluations at $TIMESTAMP"

# Run baseline evaluation
if [ "$RUN_BASELINE" = "true" ]; then
    start_vllm "false"
    run_eval "$BASE_MODEL" "baseline_${TIMESTAMP}"
fi

# Run adapter evaluation
if [ "$RUN_ADAPTER" = "true" ]; then
    start_vllm "true"
    run_eval "$ADAPTER_NAME" "adapter_${TIMESTAMP}"
fi

# Cleanup
rm -f "$TEMP_CONFIG"
kill_vllm

echo ""
echo "============================================"
echo "Exp9 Reasoning-Gym Evaluation Complete!"
echo "============================================"
echo "Results saved to: experiments/results/exp9_*"
echo ""
echo "To compare results:"
echo "  ls experiments/results/exp9_*/"
echo "  cat experiments/results/exp9_*/summary.json"
echo "============================================"
