# Experiments

Scripts and configs for running reasoning-gym evaluations.

## Quick Start

### Running with OpenRouter (cloud)

```bash
export OPENROUTER_API_KEY=your-key
./experiments/run_eval.sh --config experiments/qwen3-8b.yaml
```

### Running with local vLLM

1. Start vLLM server:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

2. Run evaluation:
```bash
./experiments/run_local_vllm.sh Qwen/Qwen2.5-7B-Instruct
```

For a quick test (1 sample per dataset):
```bash
./experiments/run_local_vllm.sh Qwen/Qwen2.5-7B-Instruct --quick
```

## Concurrency Settings

The eval system has two concurrency controls:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-concurrent` | 10 | Max concurrent API calls |
| `--max-datasets` | unlimited | Max concurrent dataset evaluations |

### Recommended settings by GPU

| GPU | VRAM | 8B Model | 70B Model |
|-----|------|----------|-----------|
| H100 | 80GB | `--max-concurrent 256 --max-datasets 50` | `--max-concurrent 64 --max-datasets 20` |
| A100 | 40GB | `--max-concurrent 128 --max-datasets 30` | `--max-concurrent 32 --max-datasets 10` |
| RTX 4090 | 24GB | `--max-concurrent 64 --max-datasets 20` | N/A (won't fit) |

Override defaults via environment variables:
```bash
MAX_CONCURRENT=128 MAX_DATASETS=30 ./experiments/run_local_vllm.sh Qwen/Qwen2.5-7B-Instruct
```

## Eval Size

The `qwen3-8b.yaml` config runs:
- **101 datasets** across 11 categories
- **50 samples per dataset** (default_size: 50)
- **5,050 total examples**

Use `--quick` for fast testing (1 sample per dataset = 101 examples).

## Config Files

| Config | Description |
|--------|-------------|
| `qwen3-8b.yaml` | Full eval suite, 50 samples/dataset |
| `grok-4.1-fast.yaml` | Same suite for Grok model |

## Output

Results are saved to `experiments/results/<model>_<timestamp>/`:
- `summary.json` - Scores per dataset
- `checkpoint.json` - Resume checkpoint
- `<category>/<dataset>.json` - Per-dataset results
