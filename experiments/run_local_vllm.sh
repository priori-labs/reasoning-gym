#!/bin/bash
# Run evaluation against a local vLLM server
#
# Usage:
#   ./experiments/run_local_vllm.sh <model-name> [additional args]
#
# Example:
#   ./experiments/run_local_vllm.sh Qwen/Qwen2.5-7B-Instruct
#   ./experiments/run_local_vllm.sh Qwen/Qwen2.5-7B-Instruct --quick

SCRIPT_DIR="$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Usage: $0 <model-name> [additional args]"
    echo "Example: $0 Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

MODEL_NAME="$1"
shift

# Default settings optimized for H100 (80GB) with 8B model
# Adjust these based on your GPU and model size:
#   - H100 80GB + 8B model: --max-concurrent 256 --max-datasets 50
#   - A100 40GB + 8B model: --max-concurrent 128 --max-datasets 30
#   - RTX 4090 24GB + 8B model: --max-concurrent 64 --max-datasets 20
MAX_CONCURRENT="${MAX_CONCURRENT:-256}"
MAX_DATASETS="${MAX_DATASETS:-50}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"

exec "$SCRIPT_DIR/run_eval.sh" \
    --config "$SCRIPT_DIR/qwen3-8b.yaml" \
    --base-url "$VLLM_URL" \
    --model "$MODEL_NAME" \
    --api-key dummy \
    --max-concurrent "$MAX_CONCURRENT" \
    --max-datasets "$MAX_DATASETS" \
    "$@"
