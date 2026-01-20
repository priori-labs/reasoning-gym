#!/bin/bash
# Reasoning-gym evaluation for Exp13 (Instruct + CPT)
# Compares baseline Qwen3-1.7B vs Exp13 CPT model on puzzle tasks

set -e
cd /workspace/reasoning-gym

MODEL="Qwen/Qwen3-1.7B"
ADAPTER_PATH="/workspace/core-reasoning/saves/qwen3-1.7b-cpt-lora-exp13"
VLLM_PORT=8000
SAMPLES_PER_DATASET=10

echo "============================================"
echo "Exp13 Puzzle Evaluation (reasoning-gym)"
echo "============================================"
echo "Model:   $MODEL"
echo "Adapter: $ADAPTER_PATH"
echo "Samples: $SAMPLES_PER_DATASET per dataset"
echo "============================================"

# Activate venv
source .venv/bin/activate

# Function to start vLLM
start_vllm() {
    local use_adapter=$1
    local tag=$2

    # Kill any existing vLLM
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3

    if [ "$use_adapter" = true ]; then
        echo "Starting vLLM with adapter..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --dtype bfloat16 \
            --trust-remote-code \
            --enable-lora \
            --lora-modules "exp13=$ADAPTER_PATH" \
            --max-lora-rank 64 \
            --port $VLLM_PORT \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --max-model-len 4096 \
            > /tmp/vllm_${tag}.log 2>&1 &
    else
        echo "Starting vLLM baseline..."
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --dtype bfloat16 \
            --trust-remote-code \
            --port $VLLM_PORT \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --max-model-len 4096 \
            > /tmp/vllm_${tag}.log 2>&1 &
    fi

    # Wait for vLLM to be ready
    echo "Waiting for vLLM to start..."
    for i in {1..60}; do
        if curl -s "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            echo "vLLM ready!"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: vLLM failed to start"
    cat /tmp/vllm_${tag}.log
    exit 1
}

# Function to run evaluation
run_eval() {
    local tag=$1
    local model_name=$2

    echo ""
    echo "Running evaluation: $tag"
    echo "Model name: $model_name"
    echo "Started at: $(date)"

    python eval/eval.py \
        --config experiments/full-eval.yaml \
        --base-url "http://localhost:$VLLM_PORT/v1" \
        --model "$model_name" \
        --api-key dummy \
        --max-concurrent 32 \
        --default-size $SAMPLES_PER_DATASET \
        --output-dir "experiments/results/exp13_puzzle_eval/${tag}"

    echo "Finished: $tag at $(date)"
}

# Create output directory
mkdir -p experiments/results/exp13_puzzle_eval

# ===========================================
# Baseline evaluation
# ===========================================
echo ""
echo "========== BASELINE EVALUATION =========="
start_vllm false "baseline"
run_eval "baseline" "$MODEL"

# ===========================================
# Exp13 CPT evaluation
# ===========================================
echo ""
echo "========== EXP13 CPT EVALUATION =========="
start_vllm true "exp13"
# When using LoRA with vLLM, specify the adapter name
run_eval "exp13" "exp13"

# ===========================================
# Cleanup and compare
# ===========================================
pkill -f "vllm.entrypoints" 2>/dev/null || true

echo ""
echo "============================================"
echo "Comparing results..."
echo "============================================"

python << 'PYEOF'
import json
from pathlib import Path
from collections import defaultdict

def load_results(results_dir):
    results = {}
    for f in Path(results_dir).glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            dataset = f.stem
            passed = sum(1 for r in data.get('results', []) if r.get('correct', False))
            total = len(data.get('results', []))
            results[dataset] = {'passed': passed, 'total': total, 'rate': passed/total if total > 0 else 0}
    return results

baseline = load_results("experiments/results/exp13_puzzle_eval/baseline")
exp13 = load_results("experiments/results/exp13_puzzle_eval/exp13")

print(f"\n{'='*70}")
print("REASONING-GYM RESULTS: Baseline vs Exp13 CPT")
print(f"{'='*70}")

# Aggregate by category (from config)
categories = {
    'arc': ['arc_1d', 'arc_agi', 'rearc'],
    'games': ['boxnet', 'countdown', 'futoshiki', 'knight_swap', 'maze', 'mini_sudoku', 'n_queens', 'rush_hour', 'sokoban', 'sudoku', 'tower_of_hanoi'],
    'logic': ['aiw', 'circuit_logic', 'knights_knaves', 'propositional_logic', 'self_reference', 'syllogism', 'zebra_puzzles'],
    'cognition': ['color_cube_rotation', 'modulo_grid', 'needle_haystack', 'number_sequence', 'rectangle_count'],
}

all_datasets = set(baseline.keys()) | set(exp13.keys())

print(f"\n{'Dataset':<30} {'Baseline':>12} {'Exp13':>12} {'Delta':>12}")
print("-" * 70)

total_base = 0
total_exp13 = 0
total_n = 0
gains = []
losses = []

for ds in sorted(all_datasets):
    b = baseline.get(ds, {'passed': 0, 'total': 0, 'rate': 0})
    e = exp13.get(ds, {'passed': 0, 'total': 0, 'rate': 0})

    base_rate = b['rate'] * 100
    exp_rate = e['rate'] * 100
    delta = exp_rate - base_rate

    total_base += b['passed']
    total_exp13 += e['passed']
    total_n += max(b['total'], e['total'])

    marker = ""
    if delta >= 10:
        marker = " ↑↑"
        gains.append((ds, delta))
    elif delta <= -10:
        marker = " ↓↓"
        losses.append((ds, delta))

    print(f"{ds:<30} {base_rate:>11.1f}% {exp_rate:>11.1f}% {delta:>+11.1f}%{marker}")

print("-" * 70)
overall_base = total_base / total_n * 100 if total_n > 0 else 0
overall_exp = total_exp13 / total_n * 100 if total_n > 0 else 0
print(f"{'OVERALL':<30} {overall_base:>11.1f}% {overall_exp:>11.1f}% {overall_exp - overall_base:>+11.1f}%")

print(f"\n{'='*70}")
print(f"Total tasks: {total_n}")
print(f"Baseline passed: {total_base} ({overall_base:.1f}%)")
print(f"Exp13 passed: {total_exp13} ({overall_exp:.1f}%)")
print(f"{'='*70}")

if gains:
    print("\n*** BIGGEST GAINS (≥10%) ***")
    for ds, delta in sorted(gains, key=lambda x: -x[1])[:10]:
        print(f"  {ds}: {delta:+.1f}%")

if losses:
    print("\n*** BIGGEST LOSSES (≤-10%) ***")
    for ds, delta in sorted(losses, key=lambda x: x[1])[:10]:
        print(f"  {ds}: {delta:+.1f}%")
PYEOF

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: experiments/results/exp13_puzzle_eval/"
