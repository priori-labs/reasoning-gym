#!/usr/bin/env python
"""
CLI entry point for synthetic data generation.

Usage:
    python -m synthetic_data.generate run --config configs/default.yaml
    python -m synthetic_data.generate resume --run-id <uuid>
    python -m synthetic_data.generate stats --run-id <uuid>
    python -m synthetic_data.generate export --run-id <uuid> --output data.jsonl --format cpt
    python -m synthetic_data.generate list
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .db import DatabaseManager
from .export import export_to_jsonl, export_to_parquet
from .generator import GenerationConfig, SyntheticDataGenerator, TaskResult, create_generator


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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_api_key(args: argparse.Namespace) -> Optional[str]:
    """Get API key from args or environment."""
    if args.api_key:
        return args.api_key
    return os.getenv("OPENROUTER_API_KEY")


def cmd_run(args: argparse.Namespace) -> int:
    """Run generation with config file."""
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    config = GenerationConfig.from_yaml(config_path)

    # Override with CLI args
    api_key = get_api_key(args)
    if not api_key:
        logger.error(
            "API key not provided. Set OPENROUTER_API_KEY environment variable "
            "or use --api-key"
        )
        return 1
    config.api_key = api_key

    if args.model:
        config.model = args.model
    if args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    if args.db_path:
        config.db_path = Path(args.db_path)
    if args.datasets:
        config.datasets = args.datasets

    # Create generator
    generator, db = create_generator(config)

    # Initialize run
    run_id = generator.initialize_run()
    logger.info(f"Starting run: {run_id}")

    # Tracking stats for display
    total_cost = 0.0
    passed_count = 0
    total_tokens = 0

    # Progress bar
    pbar = None

    def progress_callback(completed: int, failed: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Generating", unit="tasks", ncols=120)
            pbar.update(completed + failed)
        else:
            pbar.n = completed + failed
            # Update postfix with running stats
            pbar.set_postfix_str(
                f"✓{passed_count} ${total_cost:.4f} {total_tokens//1000}k tok"
            )
            pbar.refresh()

    def task_callback(result: TaskResult):
        nonlocal total_cost, passed_count, total_tokens
        if result.cost:
            total_cost += result.cost
        if result.passed:
            passed_count += 1
        total_tokens += result.input_tokens + result.output_tokens

        # Concise per-task log line
        status = "✓" if result.passed else "✗" if result.success else "!"
        cost_str = f"${result.cost:.4f}" if result.cost else "$-.----"
        tokens = result.input_tokens + result.output_tokens
        reasoning = f" r:{result.reasoning_tokens}" if result.reasoning_tokens else ""

        print(
            f"  {status} {result.dataset_name}[{result.entry_idx}] "
            f"| {tokens:,} tok{reasoning} | {result.response_time_ms}ms | {cost_str}",
            flush=True
        )

    # Run generation
    try:
        stats = asyncio.run(generator.run(run_id, progress_callback, task_callback))
        if pbar:
            pbar.close()

        print_stats(stats)
        logger.info(f"Run completed: {run_id}")
        return 0

    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        logger.info("Interrupted. Run can be resumed with:")
        logger.info(f"  python -m synthetic_data.generate resume --run-id {run_id}")
        db.update_run_status(run_id, "paused")
        return 130

    except Exception as e:
        if pbar:
            pbar.close()
        logger.error(f"Error: {e}")
        return 1


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume an interrupted run."""
    run_id = args.run_id

    # Get API key
    api_key = get_api_key(args)
    if not api_key:
        logger.error(
            "API key not provided. Set OPENROUTER_API_KEY environment variable "
            "or use --api-key"
        )
        return 1

    # Load database
    db_path = Path(args.db_path) if args.db_path else Path("synthetic_data/outputs/generation.db")
    db = DatabaseManager(db_path)

    # Get run info
    run = db.get_run(run_id)
    if not run:
        logger.error(f"Run not found: {run_id}")
        return 1

    # Load config from run
    config = GenerationConfig.from_dict(json.loads(run["config_json"]))
    config.api_key = api_key
    config.db_path = db_path

    # Create generator
    generator = SyntheticDataGenerator(config, db)

    logger.info(f"Resuming run: {run_id}")

    # Tracking stats for display
    total_cost = 0.0
    passed_count = 0
    total_tokens = 0

    # Progress bar
    pbar = None

    def progress_callback(completed: int, failed: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Generating", unit="tasks", ncols=120)
            pbar.update(completed + failed)
        else:
            pbar.n = completed + failed
            pbar.set_postfix_str(
                f"✓{passed_count} ${total_cost:.4f} {total_tokens//1000}k tok"
            )
            pbar.refresh()

    def task_callback(result: TaskResult):
        nonlocal total_cost, passed_count, total_tokens
        if result.cost:
            total_cost += result.cost
        if result.passed:
            passed_count += 1
        total_tokens += result.input_tokens + result.output_tokens

        status = "✓" if result.passed else "✗" if result.success else "!"
        cost_str = f"${result.cost:.4f}" if result.cost else "$-.----"
        tokens = result.input_tokens + result.output_tokens
        reasoning = f" r:{result.reasoning_tokens}" if result.reasoning_tokens else ""

        print(
            f"  {status} {result.dataset_name}[{result.entry_idx}] "
            f"| {tokens:,} tok{reasoning} | {result.response_time_ms}ms | {cost_str}",
            flush=True
        )

    # Run generation
    try:
        stats = asyncio.run(generator.run(run_id, progress_callback, task_callback))
        if pbar:
            pbar.close()

        print_stats(stats)
        logger.info(f"Run completed: {run_id}")
        return 0

    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        logger.info("Interrupted. Run can be resumed again with:")
        logger.info(f"  python -m synthetic_data.generate resume --run-id {run_id}")
        db.update_run_status(run_id, "paused")
        return 130

    except Exception as e:
        if pbar:
            pbar.close()
        logger.error(f"Error: {e}")
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Show statistics for a run."""
    run_id = args.run_id

    # Load database
    db_path = Path(args.db_path) if args.db_path else Path("synthetic_data/outputs/generation.db")
    db = DatabaseManager(db_path)

    # Get run info
    run = db.get_run(run_id)
    if not run:
        logger.error(f"Run not found: {run_id}")
        return 1

    # Get statistics
    stats = db.get_run_statistics(run_id)
    print_stats(stats, run)

    return 0


def print_stats(stats: dict, run: Optional[dict] = None) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)

    if run:
        print(f"\nRun ID: {stats['run_id']}")
        print(f"Model: {run['model']}")
        print(f"Status: {run['status']}")
        print(f"Started: {run['started_at']}")
        if run.get("completed_at"):
            print(f"Completed: {run['completed_at']}")

    print(f"\n--- Task Progress ---")
    task_counts = stats.get("task_counts", {})
    total_tasks = sum(task_counts.values())
    print(f"Total tasks: {total_tasks}")
    for status, count in sorted(task_counts.items()):
        pct = (count / total_tasks * 100) if total_tasks else 0
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\n--- Response Summary ---")
    print(f"Total responses: {stats['total_responses']}")
    print(f"Passed (score=1.0): {stats['passed_responses']}")
    print(f"Errors: {stats['error_responses']}")
    if stats["total_responses"]:
        print(f"Pass rate: {stats['pass_rate']:.1%}")
        if stats.get("avg_score") is not None:
            print(f"Average score: {stats['avg_score']:.3f}")
        if stats.get("avg_response_time_ms") is not None:
            print(f"Avg response time: {stats['avg_response_time_ms']:.0f}ms")

    print(f"\n--- Token Usage ---")
    print(f"Input tokens: {stats['total_input_tokens']:,}")
    print(f"Output tokens: {stats['total_output_tokens']:,}")
    print(f"Total tokens: {stats['total_input_tokens'] + stats['total_output_tokens']:,}")

    if stats.get("total_cost"):
        print(f"\n--- Cost ---")
        cost = stats['total_cost']
        print(f"Total cost: ${cost:.6f}" if cost < 0.01 else f"Total cost: ${cost:.4f}")

    if stats.get("by_template"):
        print(f"\n--- By Prompt Template ---")
        for row in stats["by_template"]:
            pass_rate = row["passed"] / row["total"] if row["total"] else 0
            print(
                f"  {row['prompt_template_id']}: {row['total']} total, "
                f"{row['passed']} passed ({pass_rate:.1%})"
            )

    if stats.get("by_temperature"):
        print(f"\n--- By Temperature ---")
        for row in stats["by_temperature"]:
            pass_rate = row["passed"] / row["total"] if row["total"] else 0
            print(
                f"  {row['temperature']}: {row['total']} total, "
                f"{row['passed']} passed ({pass_rate:.1%})"
            )

    if stats.get("by_dataset") and len(stats["by_dataset"]) <= 20:
        print(f"\n--- By Dataset (showing first 20) ---")
        for row in stats["by_dataset"][:20]:
            pass_rate = row["passed"] / row["total"] if row["total"] else 0
            print(
                f"  {row['dataset_name']}: {row['total']} total, "
                f"{row['passed']} passed ({pass_rate:.1%})"
            )

    print("=" * 60 + "\n")


def cmd_export(args: argparse.Namespace) -> int:
    """Export responses to file."""
    run_id = args.run_id
    output_path = Path(args.output)
    format = args.format

    # Load database
    db_path = Path(args.db_path) if args.db_path else Path("synthetic_data/outputs/generation.db")
    db = DatabaseManager(db_path)

    # Get run info
    run = db.get_run(run_id)
    if not run:
        logger.error(f"Run not found: {run_id}")
        return 1

    # Determine export function based on output file extension
    passed_only = not args.include_failed

    if output_path.suffix == ".parquet":
        count = export_to_parquet(db, run_id, output_path, format, passed_only)
    else:
        count = export_to_jsonl(db, run_id, output_path, format, passed_only)

    logger.info(f"Exported {count} records to {output_path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all runs."""
    # Load database
    db_path = Path(args.db_path) if args.db_path else Path("synthetic_data/outputs/generation.db")

    if not db_path.exists():
        logger.info("No database found. No runs to list.")
        return 0

    db = DatabaseManager(db_path)

    # Get all runs
    status_filter = args.status if hasattr(args, "status") and args.status else None
    runs = db.list_runs(status_filter)

    if not runs:
        logger.info("No runs found.")
        return 0

    print("\n" + "=" * 100)
    print("GENERATION RUNS")
    print("=" * 100)
    print(f"{'Run ID':<40} {'Model':<30} {'Status':<12} {'Started':<20}")
    print("-" * 100)

    for run in runs:
        print(
            f"{run['run_id']:<40} {run['model'][:28]:<30} "
            f"{run['status']:<12} {run['started_at']:<20}"
        )

    print("=" * 100 + "\n")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic data generation for reasoning tasks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--db-path", help="Path to SQLite database")
    common_parser.add_argument("--api-key", help="OpenRouter API key")
    common_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # run command
    run_parser = subparsers.add_parser(
        "run", parents=[common_parser], help="Start a new generation run"
    )
    run_parser.add_argument(
        "--config", required=True, help="Path to configuration YAML file"
    )
    run_parser.add_argument("--model", help="Override model in config")
    run_parser.add_argument(
        "--max-concurrent", type=int, help="Maximum concurrent API calls"
    )
    run_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to process (space-separated)",
    )

    # resume command
    resume_parser = subparsers.add_parser(
        "resume", parents=[common_parser], help="Resume an interrupted run"
    )
    resume_parser.add_argument("--run-id", required=True, help="Run ID to resume")

    # stats command
    stats_parser = subparsers.add_parser(
        "stats", parents=[common_parser], help="Show statistics for a run"
    )
    stats_parser.add_argument("--run-id", required=True, help="Run ID to show stats for")

    # export command
    export_parser = subparsers.add_parser(
        "export", parents=[common_parser], help="Export responses to file"
    )
    export_parser.add_argument("--run-id", required=True, help="Run ID to export")
    export_parser.add_argument(
        "--output", "-o", required=True, help="Output file path (.jsonl or .parquet)"
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["cpt", "sft", "raw"],
        default="cpt",
        help="Export format (default: cpt)",
    )
    export_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed responses (default: only passed)",
    )

    # list command
    list_parser = subparsers.add_parser(
        "list", parents=[common_parser], help="List all runs"
    )
    list_parser.add_argument(
        "--status",
        choices=["running", "completed", "paused"],
        help="Filter by status",
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "resume":
        return cmd_resume(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "list":
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
