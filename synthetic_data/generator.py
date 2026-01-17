"""
Core async generation logic for synthetic data generation.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from openai import AsyncOpenAI

import reasoning_gym
from reasoning_gym.factory import DATASETS
from reasoning_gym.utils import extract_answer

from .db import DatabaseManager, TaskRecord
from .prompts import PROMPTS, TEMPERATURES, get_prompt_template, iter_variations

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of processing a single task."""

    task_id: int
    dataset_name: str
    entry_idx: int
    success: bool
    passed: bool = False
    score: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    cost: float | None = None
    response_time_ms: int = 0
    error: str | None = None


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""

    # Model settings
    model: str = "anthropic/claude-3.5-sonnet"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None

    # Generation settings
    max_tokens: int = 4096
    timeout: int = 300

    # Concurrency settings
    max_concurrent: int = 50
    batch_size: int = 100

    # Dataset settings
    datasets: list[str] = field(default_factory=list)  # Empty = all datasets
    dataset_size: int = 50  # Number of entries per dataset
    dataset_seed: int = 42

    # Variation settings
    prompt_templates: list[str] = field(default_factory=list)  # Empty = all templates
    temperatures: list[float] = field(default_factory=list)  # Empty = default temps
    random_variation: bool = False  # If True, randomly assign template/temp per task

    # Output settings
    db_path: Path = field(default_factory=lambda: Path("generation.db"))
    store_failures: bool = True

    # Retry settings
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
            "datasets": self.datasets,
            "dataset_size": self.dataset_size,
            "dataset_seed": self.dataset_seed,
            "prompt_templates": self.prompt_templates,
            "temperatures": self.temperatures,
            "random_variation": self.random_variation,
            "db_path": str(self.db_path),
            "store_failures": self.store_failures,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "backoff_factor": self.backoff_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        """Create config from dictionary."""
        if "db_path" in data:
            data["db_path"] = Path(data["db_path"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class SyntheticDataGenerator:
    """Generates synthetic reasoning traces using LLM APIs."""

    def __init__(self, config: GenerationConfig, db: DatabaseManager):
        """Initialize the generator.

        Args:
            config: Generation configuration
            db: Database manager instance
        """
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

    def _get_datasets_to_process(self) -> list[str]:
        """Get list of dataset names to process.

        Returns:
            List of dataset names
        """
        if self.config.datasets:
            # Validate specified datasets exist
            for name in self.config.datasets:
                if name not in DATASETS:
                    raise ValueError(f"Unknown dataset: {name}")
            return self.config.datasets
        else:
            # Use all registered datasets
            return list(DATASETS.keys())

    def _get_dataset(self, name: str) -> Any:
        """Get or create a dataset instance.

        Args:
            name: Dataset name

        Returns:
            Dataset instance
        """
        if name not in self._dataset_cache:
            self._dataset_cache[name] = reasoning_gym.create_dataset(
                name,
                size=self.config.dataset_size,
                seed=self.config.dataset_seed,
            )
        return self._dataset_cache[name]

    def initialize_run(self, run_id: Optional[str] = None) -> str:
        """Initialize a new generation run.

        Creates the run record and populates the task queue with all variations.

        Args:
            run_id: Optional run ID (generated if not provided)

        Returns:
            Run ID
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        # Create run record
        self.db.create_run(
            run_id=run_id,
            model=self.config.model,
            config=self.config.to_dict(),
        )

        # Get datasets and variations
        datasets = self._get_datasets_to_process()
        templates = self.config.prompt_templates or list(PROMPTS.keys())
        temps = self.config.temperatures or TEMPERATURES

        # Seed random for reproducibility
        rng = random.Random(self.config.dataset_seed)

        # Build task list
        tasks = []
        for dataset_name in datasets:
            dataset = self._get_dataset(dataset_name)
            num_entries = len(dataset)

            for entry_idx in range(num_entries):
                if self.config.random_variation:
                    # Randomly select one template and temperature per task
                    template_id = rng.choice(templates)
                    temp = rng.choice(temps)
                    tasks.append((dataset_name, entry_idx, template_id, temp))
                else:
                    # Full cross-product of all templates and temperatures
                    for template_id in templates:
                        for temp in temps:
                            tasks.append((dataset_name, entry_idx, template_id, temp))

        # Add tasks to queue
        num_added = self.db.add_tasks(run_id, tasks)
        logger.info(f"Created run {run_id} with {num_added} tasks")

        return run_id

    def _extract_reasoning(self, completion: Any) -> tuple[str | None, int | None]:
        """Extract reasoning content and token count from completion.

        Args:
            completion: The API completion response

        Returns:
            Tuple of (reasoning_text, reasoning_tokens)
        """
        reasoning = None
        reasoning_tokens = None

        # Try to get reasoning from message
        if completion.choices:
            message = completion.choices[0].message
            # OpenRouter may expose reasoning as 'reasoning' or 'thinking'
            if hasattr(message, "reasoning") and message.reasoning:
                reasoning = message.reasoning
            elif hasattr(message, "thinking") and message.thinking:
                reasoning = message.thinking

        # Try to get reasoning tokens from usage
        if completion.usage:
            if hasattr(completion.usage, "reasoning_tokens"):
                reasoning_tokens = completion.usage.reasoning_tokens
            elif hasattr(completion.usage, "completion_tokens_details"):
                details = completion.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    reasoning_tokens = details.reasoning_tokens

        return reasoning, reasoning_tokens

    def _extract_cost(self, completion: Any) -> float | None:
        """Extract cost from completion response.

        Args:
            completion: The API completion response

        Returns:
            Cost in USD or None if not available
        """
        # OpenRouter may expose cost in different locations
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
        """Get a response from the model with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt (the question)
            temperature: Temperature setting

        Returns:
            Tuple of (response_text, reasoning, response_time_ms, input_tokens, output_tokens, reasoning_tokens, cost)

        Raises:
            Exception: If all retries fail
        """
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

                # Extract token usage
                input_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(completion.usage, "completion_tokens", 0) or 0

                # Extract reasoning
                reasoning, reasoning_tokens = self._extract_reasoning(completion)

                # Extract cost
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

    async def _process_task(self, run_id: str, task: TaskRecord) -> TaskResult:
        """Process a single task.

        Args:
            run_id: Run identifier
            task: Task to process

        Returns:
            TaskResult with status and metrics
        """
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

            # Get model response
            response_text, reasoning, response_time_ms, input_tokens, output_tokens, reasoning_tokens, cost = (
                await self._get_response_with_retry(
                    system_prompt=system_prompt,
                    user_prompt=question,
                    temperature=task.temperature,
                )
            )

            # Extract answer and score
            extracted_answer = extract_answer(response_text)
            score = dataset.score_answer(answer=extracted_answer, entry=entry)

            # Store response
            self.db.add_response(
                task_id=task.task_id,
                run_id=run_id,
                dataset_name=task.dataset_name,
                entry_idx=task.entry_idx,
                question=question,
                expected_answer=expected_answer,
                entry_metadata=entry_metadata,
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

            return TaskResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                entry_idx=task.entry_idx,
                success=True,
                passed=score == 1.0,
                score=score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cost=cost,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            error_msg = str(e)

            if self.config.store_failures:
                # Store failed attempt
                try:
                    dataset = self._get_dataset(task.dataset_name)
                    entry = dataset[task.entry_idx]
                    system_prompt = get_prompt_template(task.prompt_template_id)

                    self.db.add_response(
                        task_id=task.task_id,
                        run_id=run_id,
                        dataset_name=task.dataset_name,
                        entry_idx=task.entry_idx,
                        question=entry["question"],
                        expected_answer=str(entry["answer"]),
                        entry_metadata=entry.get("metadata"),
                        prompt_template_id=task.prompt_template_id,
                        system_prompt=system_prompt,
                        temperature=task.temperature,
                        model=self.config.model,
                        error=error_msg,
                    )
                except Exception as store_error:
                    logger.error(f"Failed to store error record: {store_error}")

            # Mark task as failed
            self.db.update_task_status(task.task_id, "failed")
            self.failed_tasks += 1

            return TaskResult(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                entry_idx=task.entry_idx,
                success=False,
                error=error_msg,
            )

    async def run(
        self,
        run_id: str,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        task_callback: Optional[Callable[[TaskResult], None]] = None,
    ) -> dict[str, Any]:
        """Run the generation process.

        Args:
            run_id: Run identifier
            progress_callback: Optional callback for progress updates (completed, failed, total)
            task_callback: Optional callback for each task result (called immediately on completion)

        Returns:
            Statistics dict
        """
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
            f"Starting generation: {task_counts.get('pending', 0)} pending, "
            f"{self.completed_tasks} completed, {self.failed_tasks} failed"
        )

        # Process tasks with streaming results (log as each completes)
        while True:
            pending_tasks = self.db.get_pending_tasks(
                run_id, limit=self.config.batch_size
            )

            if not pending_tasks:
                break

            # Create task coroutines
            task_coros = [self._process_task(run_id, task) for task in pending_tasks]

            # Process with as_completed to get results as they finish
            for coro in asyncio.as_completed(task_coros):
                try:
                    result = await coro
                    if task_callback and isinstance(result, TaskResult):
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

        # Return statistics
        return self.db.get_run_statistics(run_id)

    async def resume(
        self,
        run_id: str,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, Any]:
        """Resume a paused or interrupted run.

        Args:
            run_id: Run identifier
            progress_callback: Optional callback for progress updates

        Returns:
            Statistics dict
        """
        run = self.db.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Load config from run
        config_data = json.loads(run["config_json"])
        self.config = GenerationConfig.from_dict(config_data)

        # Update run status to running
        self.db.update_run_status(run_id, "running")

        # Continue with run
        return await self.run(run_id, progress_callback)


def create_generator(
    config: GenerationConfig,
    db_path: Optional[Path] = None,
) -> tuple[SyntheticDataGenerator, DatabaseManager]:
    """Create a generator and database manager.

    Args:
        config: Generation configuration
        db_path: Optional database path (uses config.db_path if not provided)

    Returns:
        Tuple of (generator, database_manager)
    """
    db = DatabaseManager(db_path or config.db_path)
    generator = SyntheticDataGenerator(config, db)
    return generator, db
