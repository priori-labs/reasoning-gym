#!/bin/bash
# =============================================================================
# Exp6d: In-Domain Puzzle Evaluation
# =============================================================================
# Compare Qwen3-8B-Instruct vs CPT-grafted model (Exp6b) on reasoning-gym tasks
#
# Purpose: Test if CPT model improved on puzzle types it was trained on
#
# Usage:
#   ./experiments/run_exp6d_puzzle_eval.sh [--samples N] [--quick]
#
# Examples:
#   ./experiments/run_exp6d_puzzle_eval.sh              # 10 samples per dataset
#   ./experiments/run_exp6d_puzzle_eval.sh --samples 50 # 50 samples per dataset
#   ./experiments/run_exp6d_puzzle_eval.sh --quick      # 1 sample (test run)

set -e

SCRIPT_DIR="$(dirname "$0")"
cd /workspace/reasoning-gym

# Default settings
SAMPLES_PER_DATASET=10
VLLM_PORT=8000
VLLM_URL="http://localhost:8000/v1"

# Models to compare
INSTRUCT_MODEL="Qwen/Qwen3-8B"
GRAFTED_MODEL="/workspace/core-reasoning/saves/qwen3-8b-shadow-ft-exp6b"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES_PER_DATASET="$2"
            shift 2
            ;;
        --quick)
            SAMPLES_PER_DATASET=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# GPU settings for A40 48GB
export MAX_CONCURRENT="${MAX_CONCURRENT:-128}"
export MAX_DATASETS="${MAX_DATASETS:-30}"

echo "============================================"
echo "Exp6d: In-Domain Puzzle Evaluation"
echo "============================================"
echo "Samples per dataset: $SAMPLES_PER_DATASET"
echo "Total tasks: 101 datasets × $SAMPLES_PER_DATASET = $((101 * SAMPLES_PER_DATASET))"
echo ""
echo "Models:"
echo "  1. Instruct: $INSTRUCT_MODEL"
echo "  2. Grafted:  $GRAFTED_MODEL"
echo "============================================"

# Create temp config with specified sample size
TEMP_CONFIG=$(mktemp).yaml
sed "s/default_size: [0-9]*/default_size: $SAMPLES_PER_DATASET/" \
    "$SCRIPT_DIR/qwen3-8b.yaml" > "$TEMP_CONFIG"

echo "Created temp config with default_size: $SAMPLES_PER_DATASET"

# Function to run eval for a model
run_eval() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"

    echo ""
    echo "============================================"
    echo "Evaluating: $MODEL_NAME"
    echo "Path: $MODEL_PATH"
    echo "============================================"

    # Check if vLLM is running with the right model
    CURRENT_MODEL=$(curl -s "$VLLM_URL/models" 2>/dev/null | uv run python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")

    if [[ "$CURRENT_MODEL" != "$MODEL_PATH" ]]; then
        echo "Starting vLLM server for $MODEL_PATH..."

        # Kill any existing vLLM
        pkill -f "vllm serve" 2>/dev/null || true
        sleep 2

        # Start vLLM
        uv run vllm serve "$MODEL_PATH" \
            --port $VLLM_PORT \
            --tensor-parallel-size 1 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.90 \
            --enable-prefix-caching \
            2>&1 | tee /tmp/vllm_${MODEL_NAME}.log &

        VLLM_PID=$!

        # Wait for server
        echo "Waiting for vLLM server..."
        for i in {1..120}; do
            if curl -s "$VLLM_URL/models" > /dev/null 2>&1; then
                echo "vLLM server ready!"
                break
            fi
            if [ $i -eq 120 ]; then
                echo "ERROR: vLLM server failed to start"
                cat /tmp/vllm_${MODEL_NAME}.log
                exit 1
            fi
            sleep 2
        done
    else
        echo "vLLM already serving $MODEL_PATH"
    fi

    # Run evaluation
    uv run python "$SCRIPT_DIR/../eval/eval.py" \
        --config "$TEMP_CONFIG" \
        --base-url "$VLLM_URL" \
        --model "$MODEL_PATH" \
        --api-key dummy \
        --max-concurrent "$MAX_CONCURRENT" \
        --max-datasets "$MAX_DATASETS"
}

# Run evaluations
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "Starting evaluations at $TIMESTAMP"
echo ""

# Eval 1: Instruct baseline
run_eval "$INSTRUCT_MODEL" "instruct"

# Eval 2: Grafted CPT model
run_eval "$GRAFTED_MODEL" "grafted"

# Cleanup
rm -f "$TEMP_CONFIG"
pkill -f "vllm serve" 2>/dev/null || true

echo ""
echo "============================================"
echo "Exp6d Evaluation Complete!"
echo "============================================"
echo "Results saved to: experiments/results/"
echo ""
echo "Compare results:"
echo "  ls -la experiments/results/"
echo ""
echo "Update CPT_EXPERIMENT_LOG.md with results!"
echo "============================================"
