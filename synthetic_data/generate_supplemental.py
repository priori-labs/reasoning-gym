#!/usr/bin/env python
"""
Supplemental generation script for re-testing failed samples with a stronger teacher model.

This script:
1. Reads failed/missing entries from a source database (e.g., arc_full.db)
2. Generates new responses using a stronger "teacher" model
3. Stores results in a new supplemental database WITHOUT scoring against expected answers
   (the teacher model's response becomes the new reference)

Usage:
    python -m synthetic_data.generate_supplemental \
        --source-db outputs/arc_full.db \
        --output-db outputs/arc_full_supplemental.db \
        --model anthropic/claude-sonnet-4-20250514 \
        --include-failed \
        --include-missing

    # Or use a config file:
    python -m synthetic_data.generate_supplemental \
        --config configs/supplemental.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

import reasoning_gym
from reasoning_gym.factory import DATASETS
from reasoning_gym.utils import extract_answer

from .db import DatabaseManager, TaskRecord
from .prompts import PROMPTS, TEMPERATURES, get_prompt_template

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


# Load .env file
load_env_file(Path(__file__).parent / ".env")


# =============================================================================
# Coverage Analysis
# =============================================================================

@dataclass
class CoverageInfo:
    """Information about what entries need to be generated."""
    dataset_name: str
    total_entries: int
    passed_indices: set[int]
    failed_indices: set[int]
    missing_indices: set[int]  # Entries that were never tested


def get_coverage_from_source_db(
    source_db_path: Path,
    datasets: list[str],
    dataset_size: int = 100,
) -> dict[str, CoverageInfo]:
    """
    Analyze the source database to determine which entries passed, failed, or are missing.

    Args:
        source_db_path: Path to the source database (e.g., arc_full.db)
        datasets: List of dataset names to analyze
        dataset_size: Expected number of entries per dataset

    Returns:
        Dict mapping dataset name to CoverageInfo
    """
    conn = sqlite3.connect(source_db_path)
    conn.row_factory = sqlite3.Row

    coverage = {}

    for dataset_name in datasets:
        # Query the responses table for this dataset
        cursor = conn.execute("""
            SELECT entry_idx, passed
            FROM responses
            WHERE dataset_name = ?
        """, (dataset_name,))

        rows = cursor.fetchall()

        passed_indices = set()
        failed_indices = set()
        tested_indices = set()

        for row in rows:
            entry_idx = row["entry_idx"]
            passed = row["passed"]
            tested_indices.add(entry_idx)

            if passed:
                passed_indices.add(entry_idx)
            else:
                failed_indices.add(entry_idx)

        # Calculate missing indices (never tested)
        all_indices = set(range(dataset_size))
        missing_indices = all_indices - tested_indices

        coverage[dataset_name] = CoverageInfo(
            dataset_name=dataset_name,
            total_entries=dataset_size,
            passed_indices=passed_indices,
            failed_indices=failed_indices,
            missing_indices=missing_indices,
        )

    conn.close()
    return coverage


def print_coverage_summary(coverage: dict[str, CoverageInfo]) -> None:
    """Print a summary of coverage information."""
    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Passed':>8} {'Failed':>8} {'Missing':>8} {'To Gen':>8}")
    print("-" * 70)

    total_passed = 0
    total_failed = 0
    total_missing = 0
    total_to_gen = 0

    for dataset_name, info in sorted(coverage.items()):
        to_gen = len(info.failed_indices) + len(info.missing_indices)
        print(
            f"{dataset_name:<25} "
            f"{len(info.passed_indices):>8} "
            f"{len(info.failed_indices):>8} "
            f"{len(info.missing_indices):>8} "
            f"{to_gen:>8}"
        )
        total_passed += len(info.passed_indices)
        total_failed += len(info.failed_indices)
        total_missing += len(info.missing_indices)
        total_to_gen += to_gen

    print("-" * 70)
    print(
        f"{'TOTAL':<25} "
        f"{total_passed:>8} "
        f"{total_failed:>8} "
        f"{total_missing:>8} "
        f"{total_to_gen:>8}"
    )
    print("=" * 70 + "\n")


# =============================================================================
# Supplemental Generation Config
# =============================================================================

@dataclass
class SupplementalConfig:
    """Configuration for supplemental generation."""

    # Source database (to read failed/missing entries)
    source_db_path: Path = field(default_factory=lambda: Path("synthetic_data/outputs/arc_full.db"))

    # Output database
    output_db_path: Path = field(default_factory=lambda: Path("synthetic_data/outputs/arc_full_supplemental.db"))

    # Model settings (use a stronger teacher model)
    model: str = "anthropic/claude-sonnet-4-20250514"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None

    # Generation settings
    max_tokens: int = 8192  # Higher for teacher model
    timeout: int = 300

    # Concurrency settings
    max_concurrent: int = 20  # Lower for expensive model
    batch_size: int = 50

    # Dataset settings
    datasets: list[str] = field(default_factory=list)
    dataset_size: int = 100
    dataset_seed: int = 42

    # What to include in supplemental generation
    include_failed: bool = True
    include_missing: bool = True

    # Variation settings (simpler for teacher - just one good template)
    prompt_template: str = "chain_of_thought"
    temperature: float = 0.3  # Lower for more consistent teacher output

    # Retry settings
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0

    # Testing
    task_limit: Optional[int] = None  # Limit number of tasks (for testing)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "source_db_path": str(self.source_db_path),
            "output_db_path": str(self.output_db_path),
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
            "datasets": self.datasets,
            "dataset_size": self.dataset_size,
            "dataset_seed": self.dataset_seed,
            "include_failed": self.include_failed,
            "include_missing": self.include_missing,
            "prompt_template": self.prompt_template,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_factor": self.backoff_factor,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "SupplementalConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if "source_db_path" in data:
            data["source_db_path"] = Path(data["source_db_path"])
        if "output_db_path" in data:
            data["output_db_path"] = Path(data["output_db_path"])

        return cls(**data)


# =============================================================================
# Task Result
# =============================================================================

@dataclass
class SupplementalTaskResult:
    """Result of processing a single supplemental task."""
    task_id: int
    dataset_name: str
    entry_idx: int
    success: bool
    passed: bool = False  # Whether teacher got the correct answer (only passed are stored)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    cost: float | None = None
    response_time_ms: int = 0
    error: str | None = None


# =============================================================================
# Supplemental Generator
# =============================================================================

class SupplementalGenerator:
    """Generates supplemental responses for failed/missing entries using a teacher model."""

    def __init__(self, config: SupplementalConfig, db: DatabaseManager):
        self.config = config
        self.db = db

        # Set up API client
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

        # Cache for loaded datasets
        self._dataset_cache: dict[str, Any] = {}

        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    def _get_dataset(self, name: str) -> Any:
        """Get or create a dataset instance."""
        if name not in self._dataset_cache:
            self._dataset_cache[name] = reasoning_gym.create_dataset(
                name,
                size=self.config.dataset_size,
                seed=self.config.dataset_seed,
            )
        return self._dataset_cache[name]

    def initialize_run(
        self,
        coverage: dict[str, CoverageInfo],
        run_id: Optional[str] = None,
    ) -> str:
        """
        Initialize a supplemental generation run.

        Only creates tasks for failed/missing entries based on coverage analysis.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        # Create run record
        self.db.create_run(
            run_id=run_id,
            model=self.config.model,
            config=self.config.to_dict(),
        )

        # Build task list from coverage
        tasks = []
        for dataset_name, info in coverage.items():
            indices_to_generate = set()

            if self.config.include_failed:
                indices_to_generate.update(info.failed_indices)
            if self.config.include_missing:
                indices_to_generate.update(info.missing_indices)

            for entry_idx in sorted(indices_to_generate):
                tasks.append((
                    dataset_name,
                    entry_idx,
                    self.config.prompt_template,
                    self.config.temperature,
                ))

                # Check limit
                if self.config.task_limit and len(tasks) >= self.config.task_limit:
                    break

            # Check limit at outer loop too
            if self.config.task_limit and len(tasks) >= self.config.task_limit:
                break

        # Add tasks to queue
        num_added = self.db.add_tasks(run_id, tasks)
        logger.info(f"Created supplemental run {run_id} with {num_added} tasks")

        return run_id

    def _extract_reasoning(self, completion: Any) -> tuple[str | None, int | None]:
        """Extract reasoning content and token count from completion."""
        reasoning = None
        reasoning_tokens = None

        if completion.choices:
            message = completion.choices[0].message
            if hasattr(message, "reasoning") and message.reasoning:
                reasoning = message.reasoning
            elif hasattr(message, "thinking") and message.thinking:
                reasoning = message.thinking

        if completion.usage:
            if hasattr(completion.usage, "reasoning_tokens"):
                reasoning_tokens = completion.usage.reasoning_tokens
            elif hasattr(completion.usage, "completion_tokens_details"):
                details = completion.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    reasoning_tokens = details.reasoning_tokens

        return reasoning, reasoning_tokens

    def _extract_cost(self, completion: Any) -> float | None:
        """Extract cost from completion response."""
        if hasattr(completion, "cost") and completion.cost is not None:
            return float(completion.cost)
        if hasattr(completion, "usage") and completion.usage:
            if hasattr(completion.usage, "cost") and completion.usage.cost is not None:
                return float(completion.usage.cost)
        return None

    async def _get_response_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> tuple[str, str | None, int, int, int, int | None, float | None]:
        """Get a response from the model with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                async with self.semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=self.config.max_tokens,
                        temperature=temperature,
                    )

                response_time_ms = int((time.time() - start_time) * 1000)
                response_text = completion.choices[0].message.content

                input_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(completion.usage, "completion_tokens", 0) or 0

                reasoning, reasoning_tokens = self._extract_reasoning(completion)
                cost = self._extract_cost(completion)

                return response_text, reasoning, response_time_ms, input_tokens, output_tokens, reasoning_tokens, cost

            except Exception as e:
                delay = min(
                    self.config.max_delay,
                    self.config.base_delay * (self.config.backoff_factor ** attempt),
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        raise Exception(f"Failed after {self.config.max_retries} attempts")

    async def _process_task(
        self,
        run_id: str,
        task: TaskRecord,
        coverage: dict[str, CoverageInfo],
    ) -> SupplementalTaskResult:
        """Process a single supplemental task."""
        # Mark task as in progress
        self.db.update_task_status(task.task_id, "in_progress")

        try:
            # Load dataset and get entry
            dataset = self._get_dataset(task.dataset_name)
            entry = dataset[task.entry_idx]

            question = entry["question"]
            expected_answer = str(entry["answer"])
            entry_metadata = entry.get("metadata")

            # Get system prompt
            system_prompt = get_prompt_template(task.prompt_template_id)

            # Get teacher model response
            response_text, reasoning, response_time_ms, input_tokens, output_tokens, reasoning_tokens, cost = (
                await self._get_response_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=question,
                    temperature=task.temperature,
                )
            )

            # Extract answer from teacher's response
            extracted_answer = extract_answer(response_text)

            # Score the teacher's response against the expected answer
            score = dataset.score_answer(answer=extracted_answer, entry=entry)
            passed = score == 1.0

            # Debug: log first few failures to understand why
            if not passed and self.completed_tasks < 5:
                logger.info(f"DEBUG {task.dataset_name}[{task.entry_idx}] score={score}")
                logger.info(f"  Expected: {expected_answer[:200]}...")
                logger.info(f"  Extracted: {str(extracted_answer)[:200] if extracted_answer else 'None'}...")
                # Show end of response to see if answer tags are present
                logger.info(f"  Response end (last 500 chars): ...{response_text[-500:] if response_text else 'None'}")

            # Only store responses where the teacher got it correct
            # Discard incorrect responses - we only want high-quality training data
            if passed:
                augmented_metadata = entry_metadata.copy() if entry_metadata else {}
                augmented_metadata["teacher_model"] = self.config.model

                self.db.add_response(
                    task_id=task.task_id,
                    run_id=run_id,
                    dataset_name=task.dataset_name,
                    entry_idx=task.entry_idx,
                    question=question,
                    expected_answer=expected_answer,
                    entry_metadata=augmented_metadata,
                    prompt_template_id=task.prompt_template_id,
                    system_prompt=system_prompt,
                    temperature=task.temperature,
                    model=self.config.model,
                    full_response=response_text,
                    reasoning=reasoning,
                    extracted_answer=extracted_answer,
                    score=score,
                    response_time_ms=response_time_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cost=cost,
                )

            # Mark task as completed
            self.db.update_task_status(task.task_id, "completed")
            self.completed_tasks += 1

            return SupplementalTaskResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                entry_idx=task.entry_idx,
                success=True,
                passed=passed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost=cost,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Task failed for {task.dataset_name}[{task.entry_idx}]: {error_msg}")

            # Mark task as failed (don't store error responses - only want successful correct answers)
            self.db.update_task_status(task.task_id, "failed")
            self.failed_tasks += 1

            return SupplementalTaskResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                entry_idx=task.entry_idx,
                success=False,
                passed=False,
                error=error_msg,
            )

    async def run(
        self,
        run_id: str,
        coverage: dict[str, CoverageInfo],
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        task_callback: Optional[Callable[[SupplementalTaskResult], None]] = None,
    ) -> dict[str, Any]:
        """Run the supplemental generation process."""
        # Reset in_progress tasks from previous interrupted run
        reset_count = self.db.reset_in_progress_tasks(run_id)
        if reset_count > 0:
            logger.info(f"Reset {reset_count} in_progress tasks to pending")

        # Get initial task counts
        task_counts = self.db.get_task_counts(run_id)
        self.total_tasks = sum(task_counts.values())
        self.completed_tasks = task_counts.get("completed", 0)
        self.failed_tasks = task_counts.get("failed", 0)

        logger.info(
            f"Starting supplemental generation: {task_counts.get('pending', 0)} pending, "
            f"{self.completed_tasks} completed, {self.failed_tasks} failed"
        )

        # Process tasks
        while True:
            pending_tasks = self.db.get_pending_tasks(run_id, limit=self.config.batch_size)

            if not pending_tasks:
                break

            task_coros = [self._process_task(run_id, task, coverage) for task in pending_tasks]

            for coro in asyncio.as_completed(task_coros):
                try:
                    result = await coro
                    if task_callback and isinstance(result, SupplementalTaskResult):
                        task_callback(result)
                    if progress_callback:
                        progress_callback(
                            self.completed_tasks,
                            self.failed_tasks,
                            self.total_tasks,
                        )
                except Exception as e:
                    logger.error(f"Task error: {e}")

        # Update run status
        self.db.update_run_status(run_id, "completed")

        return self.db.get_run_statistics(run_id)


# =============================================================================
# CLI
# =============================================================================

def print_stats(stats: dict, run: Optional[dict] = None) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 60)
    print("SUPPLEMENTAL GENERATION STATISTICS")
    print("=" * 60)

    if run:
        print(f"\nRun ID: {stats['run_id']}")
        print(f"Teacher Model: {run['model']}")
        print(f"Status: {run['status']}")

    print(f"\n--- Task Progress ---")
    task_counts = stats.get("task_counts", {})
    total_tasks = sum(task_counts.values())
    print(f"Total tasks: {total_tasks}")
    for status, count in sorted(task_counts.items()):
        pct = (count / total_tasks * 100) if total_tasks else 0
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\n--- Response Summary ---")
    print(f"Responses stored (teacher correct): {stats['passed_responses']}")
    print(f"Responses discarded (teacher incorrect): {stats['total_responses'] - stats['passed_responses']}")
    if stats["total_responses"]:
        print(f"Teacher success rate: {stats['pass_rate']:.1%}")

    print(f"\n--- Token Usage ---")
    print(f"Input tokens: {stats['total_input_tokens']:,}")
    print(f"Output tokens: {stats['total_output_tokens']:,}")

    if stats.get("total_cost"):
        print(f"\n--- Cost ---")
        cost = stats['total_cost']
        print(f"Total cost: ${cost:.4f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate supplemental data for failed/missing entries using a teacher model"
    )

    # Config file option
    parser.add_argument("--config", help="Path to YAML config file")

    # Source and output database
    parser.add_argument(
        "--source-db",
        default="synthetic_data/outputs/arc_full.db",
        help="Source database to read failed/missing entries from",
    )
    parser.add_argument(
        "--output-db",
        default="synthetic_data/outputs/arc_full_supplemental.db",
        help="Output database for supplemental generations",
    )

    # Model settings
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4-20250514",
        help="Teacher model to use (default: Claude Sonnet 4)",
    )
    parser.add_argument("--api-key", help="OpenRouter API key")

    # What to include (by default, both failed and missing are included)
    parser.add_argument(
        "--no-failed",
        action="store_true",
        help="Exclude entries that failed in original run",
    )
    parser.add_argument(
        "--no-missing",
        action="store_true",
        help="Exclude entries that were never tested",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only include failed entries (shorthand for --no-missing)",
    )

    # Generation settings
    parser.add_argument("--max-concurrent", type=int, default=20, help="Max concurrent API calls")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for teacher model")
    parser.add_argument("--prompt-template", default="chain_of_thought", help="Prompt template to use")

    # Dataset settings
    parser.add_argument("--dataset-size", type=int, default=100, help="Expected entries per dataset")
    parser.add_argument("--dataset-seed", type=int, default=42, help="Seed for dataset generation")

    # Other options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without running")
    parser.add_argument("--resume", help="Resume a previous run by run_id")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process (for testing)")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine what to include (defaults: both failed and missing)
    include_failed = not args.no_failed
    include_missing = not args.no_missing and not args.only_failed

    # Load config
    if args.config:
        config = SupplementalConfig.from_yaml(Path(args.config))
        # CLI flags can override config file
        if args.no_failed:
            config.include_failed = False
        if args.no_missing or args.only_failed:
            config.include_missing = False
    else:
        config = SupplementalConfig(
            source_db_path=Path(args.source_db),
            output_db_path=Path(args.output_db),
            model=args.model,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            prompt_template=args.prompt_template,
            dataset_size=args.dataset_size,
            dataset_seed=args.dataset_seed,
            include_failed=include_failed,
            include_missing=include_missing,
        )

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "API key not provided. Set OPENROUTER_API_KEY environment variable or use --api-key"
        )
        return 1
    config.api_key = api_key

    # Apply task limit if specified
    if args.limit:
        config.task_limit = args.limit

    # Define the 25 ARC-like datasets from arc_full config (used as default)
    ARC_DATASETS = [
        "arc_agi", "rearc", "arc_1d", "manipulate_matrix", "rotate_matrix",
        "game_of_life", "spiral_matrix", "binary_matrix", "pool_matrix",
        "rotten_oranges", "modulo_grid", "rectangle_count", "largest_island",
        "kakurasu", "futoshiki", "maze", "sokoban", "sudoku", "mini_sudoku",
        "rush_hour", "knight_swap", "color_cube_rotation", "graph_color",
        "n_queens", "tower_of_hanoi",
    ]
    # Use ARC_DATASETS if no datasets specified in config
    if not config.datasets:
        config.datasets = ARC_DATASETS

    # Verify source database exists
    if not config.source_db_path.exists():
        logger.error(f"Source database not found: {config.source_db_path}")
        return 1

    # Analyze coverage
    logger.info(f"Analyzing coverage from {config.source_db_path}...")
    coverage = get_coverage_from_source_db(
        config.source_db_path,
        config.datasets,
        config.dataset_size,
    )

    # Print coverage summary
    print_coverage_summary(coverage)

    # Calculate what will be generated
    total_to_generate = 0
    for info in coverage.values():
        if config.include_failed:
            total_to_generate += len(info.failed_indices)
        if config.include_missing:
            total_to_generate += len(info.missing_indices)

    print(f"Will generate {total_to_generate} supplemental entries")
    print(f"  Include failed: {config.include_failed}")
    print(f"  Include missing: {config.include_missing}")
    print(f"  Teacher model: {config.model}")
    print(f"  Output database: {config.output_db_path}")
    print()

    if args.dry_run:
        logger.info("Dry run complete. Use without --dry-run to execute.")
        return 0

    # Create output database and generator
    config.output_db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(config.output_db_path)
    generator = SupplementalGenerator(config, db)

    # Initialize or resume run
    if args.resume:
        run_id = args.resume
        run = db.get_run(run_id)
        if not run:
            logger.error(f"Run not found: {run_id}")
            return 1
        logger.info(f"Resuming run: {run_id}")
        # When resuming, tasks are already in the database - don't check total_to_generate
    else:
        if total_to_generate == 0:
            logger.info("No entries to generate. Exiting.")
            return 0
        run_id = generator.initialize_run(coverage)
        logger.info(f"Starting run: {run_id}")

    # Progress tracking
    total_cost = 0.0
    teacher_correct = 0
    total_tokens = 0
    pbar = None

    def progress_callback(completed: int, failed: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Generating", unit="tasks", ncols=120)
            pbar.update(completed + failed)
        else:
            pbar.n = completed + failed
            pbar.set_postfix_str(
                f"teacher_ok:{teacher_correct} ${total_cost:.4f} {total_tokens//1000}k tok"
            )
            pbar.refresh()

    def task_callback(result: SupplementalTaskResult):
        nonlocal total_cost, teacher_correct, total_tokens
        if result.cost:
            total_cost += result.cost
        total_tokens += result.input_tokens + result.output_tokens
        if result.passed:
            teacher_correct += 1

        # Status: ✓ = passed (stored), ✗ = wrong answer (discarded), ! = error
        if result.error:
            status = "!"
        elif result.passed:
            status = "✓"
        else:
            status = "✗"

        outcome = "stored" if result.passed else "discarded" if result.success else "error"
        cost_str = f"${result.cost:.4f}" if result.cost else "$-.----"
        tokens = result.input_tokens + result.output_tokens
        reasoning = f" r:{result.reasoning_tokens}" if result.reasoning_tokens else ""

        tqdm.write(
            f"  {status} {result.dataset_name}[{result.entry_idx}] [{outcome}] "
            f"| {tokens:,} tok{reasoning} | {result.response_time_ms}ms | {cost_str}"
        )

    # Run generation
    try:
        stats = asyncio.run(generator.run(run_id, coverage, progress_callback, task_callback))
        if pbar:
            pbar.close()

        print_stats(stats, db.get_run(run_id))
        logger.info(f"Run completed: {run_id}")
        return 0

    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        logger.info("Interrupted. Run can be resumed with:")
        logger.info(f"  python -m synthetic_data.generate_supplemental --resume {run_id}")
        db.update_run_status(run_id, "paused")
        return 130

    except Exception as e:
        if pbar:
            pbar.close()
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
