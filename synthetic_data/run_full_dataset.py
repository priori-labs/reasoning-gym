#!/usr/bin/env python3
"""
Run OpenRouter model evaluation on all 105 reasoning-gym datasets.

Usage:
    # Standard run (all datasets, 100 entries each)
    python -m synthetic_data.run_full_dataset

    # With custom config
    python -m synthetic_data.run_full_dataset --config path/to/config.yaml

    # Resume interrupted run
    python -m synthetic_data.run_full_dataset --resume <run-id>

    # Quick test (5 entries per dataset)
    python -m synthetic_data.run_full_dataset --test

    # Export results
    python -m synthetic_data.run_full_dataset --export <run-id> --output results.jsonl
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def load_env_file():
    """Load .env file from project root."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value


def main():
    # Load environment variables from project root .env
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Run OpenRouter model on full reasoning-gym dataset"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start a new generation run")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("synthetic_data/configs/full_dataset.yaml"),
        help="Path to config file",
    )
    run_parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only 5 entries per dataset",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        help="Override model (e.g., 'openai/gpt-4o')",
    )

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted run")
    resume_parser.add_argument("run_id", help="Run ID to resume")
    resume_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=100,
        help="Max concurrent requests (default: 100)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show run statistics")
    stats_parser.add_argument("run_id", help="Run ID to show stats for")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export run results")
    export_parser.add_argument("run_id", help="Run ID to export")
    export_parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output file path (.jsonl or .parquet)",
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["cpt", "sft", "raw"],
        default="cpt",
        help="Export format",
    )
    export_parser.add_argument(
        "--passed-only",
        action="store_true",
        help="Only export passed responses",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all runs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Import after arg parsing to speed up --help
    from tqdm import tqdm

    from .db import DatabaseManager
    from .export import export_to_jsonl, export_to_parquet
    from .generator import GenerationConfig, create_generator

    if args.command == "run":
        config = GenerationConfig.from_yaml(args.config)

        # Apply overrides
        if args.test:
            config.dataset_size = 5
            print("TEST MODE: Using 5 entries per dataset")

        if args.model:
            config.model = args.model
            print(f"Using model: {config.model}")

        # Verify API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY environment variable not set")
            print("Set it with: export OPENROUTER_API_KEY=your-key-here")
            sys.exit(1)
        config.api_key = api_key

        # Create generator
        generator, db = create_generator(config)
        run_id = generator.initialize_run()

        task_counts = db.get_task_counts(run_id)
        total_tasks = sum(task_counts.values())

        print(f"\nStarting run: {run_id}")
        print(f"Model: {config.model}")
        print(f"Datasets: {len(config.datasets) or 'all'}")
        print(f"Total tasks: {total_tasks}")
        print(f"Output: {config.db_path}\n")

        # Progress bar
        pbar = tqdm(total=total_tasks, desc="Processing", unit="task")

        def on_progress(completed, failed, total):
            pbar.n = completed + failed
            pbar.set_postfix(passed=completed, failed=failed)
            pbar.refresh()

        def on_task(result):
            status = "✓" if result.passed else "✗" if result.success else "!"
            cost_str = f"${result.cost:.4f}" if result.cost else "N/A"
            tqdm.write(
                f"  {status} {result.dataset_name}[{result.entry_idx}] "
                f"tokens={result.input_tokens}+{result.output_tokens} "
                f"cost={cost_str}"
            )

        try:
            stats = asyncio.run(
                generator.run(run_id, progress_callback=on_progress, task_callback=on_task)
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted! Resume with:")
            print(f"  python -m synthetic_data.run_full_dataset resume {run_id}")
            sys.exit(0)
        finally:
            pbar.close()

        # Print summary
        print("\n" + "=" * 60)
        print("RUN COMPLETE")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"Total: {stats.get('total_responses', 0)}")
        print(f"Passed: {stats.get('passed_count', 0)} ({stats.get('pass_rate', 0):.1%})")
        print(f"Failed: {stats.get('failed_count', 0)}")
        print(f"Total cost: ${stats.get('total_cost', 0):.4f}")
        print(f"\nExport with:")
        print(f"  python -m synthetic_data.run_full_dataset export {run_id} -o results.jsonl")

    elif args.command == "resume":
        # Find db path from existing run
        # Try default locations
        for db_path in [
            Path("synthetic_data/outputs/full_dataset.db"),
            Path("synthetic_data/outputs/generation.db"),
        ]:
            if db_path.exists():
                db = DatabaseManager(db_path)
                run = db.get_run(args.run_id)
                if run:
                    break
        else:
            print(f"ERROR: Run {args.run_id} not found")
            sys.exit(1)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)

        import json as json_module
        config = GenerationConfig.from_dict(json_module.loads(run["config_json"]))
        config.api_key = api_key
        config.max_concurrent = args.concurrency

        generator, db = create_generator(config, db_path)

        task_counts = db.get_task_counts(args.run_id)
        pending = task_counts.get("pending", 0) + task_counts.get("in_progress", 0)
        completed = task_counts.get("completed", 0)

        print(f"Resuming run: {args.run_id}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Completed: {completed}, Pending: {pending}")

        pbar = tqdm(total=pending + completed, initial=completed, desc="Processing")

        def on_progress(c, f, t):
            pbar.n = c + f
            pbar.refresh()

        def on_task(result):
            status = "✓" if result.passed else "✗" if result.success else "!"
            cost_str = f"${result.cost:.4f}" if result.cost else "N/A"
            tqdm.write(
                f"  {status} {result.dataset_name}[{result.entry_idx}] "
                f"tokens={result.input_tokens}+{result.output_tokens} "
                f"cost={cost_str}"
            )

        stats = asyncio.run(generator.run(args.run_id, progress_callback=on_progress, task_callback=on_task))
        pbar.close()

        print(f"\nCompleted! Pass rate: {stats.get('pass_rate', 0):.1%}")

    elif args.command == "stats":
        db = DatabaseManager(Path("synthetic_data/outputs/full_dataset.db"))
        stats = db.get_run_statistics(args.run_id)

        if not stats:
            print(f"Run {args.run_id} not found")
            sys.exit(1)

        print(f"Run: {args.run_id}")
        print(f"Status: {stats.get('status', 'unknown')}")
        print(f"Total: {stats.get('total_responses', 0)}")
        print(f"Passed: {stats.get('passed_count', 0)} ({stats.get('pass_rate', 0):.1%})")
        print(f"Cost: ${stats.get('total_cost', 0):.4f}")

    elif args.command == "export":
        db = DatabaseManager(Path("synthetic_data/outputs/full_dataset.db"))

        if str(args.output).endswith(".parquet"):
            count = export_to_parquet(
                db=db,
                run_id=args.run_id,
                output_path=args.output,
                format=args.format,
                passed_only=args.passed_only,
            )
        else:
            count = export_to_jsonl(
                db=db,
                run_id=args.run_id,
                output_path=args.output,
                format=args.format,
                passed_only=args.passed_only,
            )

        print(f"Exported {count} responses to {args.output}")

    elif args.command == "list":
        db = DatabaseManager(Path("synthetic_data/outputs/full_dataset.db"))
        runs = db.list_runs()

        if not runs:
            print("No runs found")
        else:
            print(f"{'Run ID':<40} {'Status':<12} {'Started'}")
            print("-" * 70)
            for run in runs:
                print(f"{run['run_id']:<40} {run['status']:<12} {run['started_at']}")


if __name__ == "__main__":
    main()
