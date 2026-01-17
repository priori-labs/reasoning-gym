#!/usr/bin/env python
"""
Test script for synthetic data generation.

Runs a small test (10 samples) and displays results for inspection.

Usage:
    python -m synthetic_data.test_run

Reads OPENROUTER_API_KEY from .env file in synthetic_data/ directory.
"""

import asyncio
import os
import sys
from pathlib import Path

from .db import DatabaseManager
from .generator import GenerationConfig, SyntheticDataGenerator


def load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value


# Load .env file from the synthetic_data directory
load_env_file(Path(__file__).parent / ".env")


def print_response(response: dict, index: int) -> None:
    """Print a response in a readable format."""
    print(f"\n{'='*80}")
    print(f"RESPONSE {index + 1}")
    print(f"{'='*80}")
    print(f"Dataset: {response['dataset_name']}")
    print(f"Entry Index: {response['entry_idx']}")
    print(f"Template: {response['prompt_template_id']}")
    print(f"Temperature: {response['temperature']}")
    print(f"Score: {response['score']} {'(PASSED)' if response['passed'] else '(FAILED)'}")
    print(f"\n--- Question ---")
    print(response['question'][:500] + "..." if len(response['question']) > 500 else response['question'])
    print(f"\n--- Expected Answer ---")
    print(response['expected_answer'])
    print(f"\n--- Extracted Answer ---")
    print(response['extracted_answer'])
    # Show reasoning if present (from models that expose thinking)
    if response.get('reasoning'):
        print(f"\n--- Model Reasoning ---")
        reasoning = response['reasoning']
        print(reasoning[:2000] + "..." if len(reasoning) > 2000 else reasoning)
    print(f"\n--- Full Response ---")
    full_resp = response['full_response'] or "(No response)"
    print(full_resp[:1000] + "..." if len(full_resp) > 1000 else full_resp)
    if response.get('error'):
        print(f"\n--- Error ---")
        print(response['error'])
    print(f"\n--- Stats ---")
    print(f"Response time: {response.get('response_time_ms', 'N/A')}ms")
    print(f"Input tokens: {response.get('input_tokens', 'N/A')}")
    print(f"Output tokens: {response.get('output_tokens', 'N/A')}")
    if response.get('reasoning_tokens'):
        print(f"Reasoning tokens: {response.get('reasoning_tokens')}")
    if response.get('cost') is not None:
        print(f"Cost: ${response.get('cost'):.6f}")


async def main():
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY=your-api-key")
        return 1

    # Load test config
    config_path = Path(__file__).parent / "configs" / "test.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config from: {config_path}")
    config = GenerationConfig.from_yaml(config_path)
    config.api_key = api_key

    # Use test database in current directory
    db_path = Path("test_generation.db")
    config.db_path = db_path

    # Remove old test database if exists
    if db_path.exists():
        print(f"Removing old test database: {db_path}")
        db_path.unlink()

    # Create database and generator
    db = DatabaseManager(db_path)
    generator = SyntheticDataGenerator(config, db)

    # Initialize run
    print(f"\nInitializing test run...")
    print(f"Model: {config.model}")
    print(f"Datasets: {config.datasets}")
    print(f"Dataset size: {config.dataset_size}")
    print(f"Templates: {config.prompt_templates}")
    print(f"Temperatures: {config.temperatures}")

    run_id = generator.initialize_run()
    print(f"Run ID: {run_id}")

    # Get task count
    task_counts = db.get_task_counts(run_id)
    total_tasks = sum(task_counts.values())
    print(f"Total tasks: {total_tasks}")

    # Run generation with progress
    print(f"\nStarting generation...")

    def progress_callback(completed: int, failed: int, total: int):
        print(f"  Progress: {completed + failed}/{total} (completed: {completed}, failed: {failed})")

    try:
        stats = await generator.run(run_id, progress_callback)
    except KeyboardInterrupt:
        print("\nInterrupted!")
        db.update_run_status(run_id, "paused")
        return 130

    # Print statistics
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total responses: {stats['total_responses']}")
    print(f"Passed: {stats['passed_responses']}")
    print(f"Failed: {stats['total_responses'] - stats['passed_responses']}")
    print(f"Errors: {stats['error_responses']}")
    if stats['total_responses'] > 0:
        print(f"Pass rate: {stats['pass_rate']:.1%}")
        if stats.get('avg_score') is not None:
            print(f"Average score: {stats['avg_score']:.3f}")
    if stats.get('total_cost'):
        print(f"Total cost: ${stats['total_cost']:.6f}")

    # Display all responses
    print(f"\n{'='*80}")
    print("DETAILED RESPONSES")
    print(f"{'='*80}")

    responses = db.get_all_responses(run_id)
    for i, response in enumerate(responses):
        print_response(response, i)

    # Export to JSONL for inspection
    export_path = Path("test_output.jsonl")
    print(f"\n{'='*80}")
    print(f"Exporting to: {export_path}")
    print(f"{'='*80}")

    from .export import export_to_jsonl
    count = export_to_jsonl(db, run_id, export_path, format="cpt", passed_only=False)
    print(f"Exported {count} records")

    print(f"\nTest complete! Files created:")
    print(f"  - {db_path} (SQLite database)")
    print(f"  - {export_path} (JSONL export)")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
