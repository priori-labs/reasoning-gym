"""
SQLite database schema and operations for synthetic data generation.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


@dataclass
class TaskRecord:
    """Represents a task in the queue."""

    task_id: int
    run_id: str
    dataset_name: str
    entry_idx: int
    prompt_template_id: str
    temperature: float
    status: str


@dataclass
class ResponseRecord:
    """Represents a generated response."""

    response_id: int
    task_id: int
    run_id: str
    dataset_name: str
    entry_idx: int
    question: str
    expected_answer: str
    entry_metadata_json: Optional[str]
    prompt_template_id: str
    system_prompt: str
    temperature: float
    full_response: Optional[str]
    extracted_answer: Optional[str]
    score: Optional[float]
    passed: bool
    model: str
    response_time_ms: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    error: Optional[str]
    created_at: str


SCHEMA = """
-- Generation runs
CREATE TABLE IF NOT EXISTS generation_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT DEFAULT 'running'  -- running, completed, paused
);

-- Task queue (enables resumability)
CREATE TABLE IF NOT EXISTS task_queue (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    entry_idx INTEGER NOT NULL,
    prompt_template_id TEXT NOT NULL,
    temperature REAL NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed
    UNIQUE(run_id, dataset_name, entry_idx, prompt_template_id, temperature)
);

-- Generated responses
CREATE TABLE IF NOT EXISTS responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    entry_idx INTEGER NOT NULL,
    question TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    entry_metadata_json TEXT,
    prompt_template_id TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    temperature REAL NOT NULL,
    full_response TEXT,
    reasoning TEXT,              -- Model's reasoning/thinking content
    extracted_answer TEXT,
    score REAL,
    passed BOOLEAN GENERATED ALWAYS AS (score = 1.0) STORED,
    model TEXT NOT NULL,
    response_time_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,    -- Tokens used for reasoning
    cost REAL,                   -- Cost in USD from OpenRouter
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_responses_passed ON responses(passed);
CREATE INDEX IF NOT EXISTS idx_responses_run ON responses(run_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(run_id, status);
CREATE INDEX IF NOT EXISTS idx_task_queue_run ON task_queue(run_id);
"""


class DatabaseManager:
    """Manages SQLite database operations for synthetic data generation."""

    def __init__(self, db_path: Path):
        """Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ========== Run Management ==========

    def create_run(self, run_id: str, model: str, config: dict[str, Any]) -> None:
        """Create a new generation run.

        Args:
            run_id: Unique identifier for the run
            model: Model name being used
            config: Configuration dictionary
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO generation_runs (run_id, model, config_json, status)
                VALUES (?, ?, ?, 'running')
                """,
                (run_id, model, json.dumps(config)),
            )

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run information.

        Args:
            run_id: Run identifier

        Returns:
            Run info dict or None if not found
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM generation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None

    def update_run_status(self, run_id: str, status: str) -> None:
        """Update run status.

        Args:
            run_id: Run identifier
            status: New status (running, completed, paused)
        """
        with self._get_connection() as conn:
            if status == "completed":
                conn.execute(
                    """
                    UPDATE generation_runs
                    SET status = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE run_id = ?
                    """,
                    (status, run_id),
                )
            else:
                conn.execute(
                    "UPDATE generation_runs SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )

    def list_runs(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """List all runs, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of run info dicts
        """
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM generation_runs WHERE status = ? ORDER BY started_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM generation_runs ORDER BY started_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    # ========== Task Queue Management ==========

    def add_tasks(
        self,
        run_id: str,
        tasks: list[tuple[str, int, str, float]],
    ) -> int:
        """Add tasks to the queue.

        Args:
            run_id: Run identifier
            tasks: List of (dataset_name, entry_idx, prompt_template_id, temperature) tuples

        Returns:
            Number of tasks added
        """
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO task_queue
                (run_id, dataset_name, entry_idx, prompt_template_id, temperature)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(run_id, *task) for task in tasks],
            )
            return conn.total_changes

    def get_pending_tasks(
        self, run_id: str, limit: Optional[int] = None
    ) -> list[TaskRecord]:
        """Get pending tasks from the queue.

        Args:
            run_id: Run identifier
            limit: Maximum number of tasks to return

        Returns:
            List of TaskRecord objects
        """
        with self._get_connection() as conn:
            query = """
                SELECT task_id, run_id, dataset_name, entry_idx, prompt_template_id, temperature, status
                FROM task_queue
                WHERE run_id = ? AND status = 'pending'
                ORDER BY task_id
            """
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query, (run_id,)).fetchall()
            return [TaskRecord(**dict(row)) for row in rows]

    def update_task_status(self, task_id: int, status: str) -> None:
        """Update task status.

        Args:
            task_id: Task identifier
            status: New status (pending, in_progress, completed, failed)
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE task_queue SET status = ? WHERE task_id = ?",
                (status, task_id),
            )

    def update_tasks_status_bulk(self, task_ids: list[int], status: str) -> None:
        """Update status for multiple tasks.

        Args:
            task_ids: List of task identifiers
            status: New status
        """
        if not task_ids:
            return
        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(task_ids))
            conn.execute(
                f"UPDATE task_queue SET status = ? WHERE task_id IN ({placeholders})",
                [status] + task_ids,
            )

    def reset_in_progress_tasks(self, run_id: str) -> int:
        """Reset in_progress tasks to pending (for resume after crash).

        Args:
            run_id: Run identifier

        Returns:
            Number of tasks reset
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE task_queue
                SET status = 'pending'
                WHERE run_id = ? AND status = 'in_progress'
                """,
                (run_id,),
            )
            return conn.total_changes

    def get_task_counts(self, run_id: str) -> dict[str, int]:
        """Get task counts by status.

        Args:
            run_id: Run identifier

        Returns:
            Dict mapping status to count
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM task_queue
                WHERE run_id = ?
                GROUP BY status
                """,
                (run_id,),
            ).fetchall()
            return {row["status"]: row["count"] for row in rows}

    # ========== Response Management ==========

    def add_response(
        self,
        task_id: int,
        run_id: str,
        dataset_name: str,
        entry_idx: int,
        question: str,
        expected_answer: str,
        entry_metadata: Optional[dict[str, Any]],
        prompt_template_id: str,
        system_prompt: str,
        temperature: float,
        model: str,
        full_response: Optional[str] = None,
        reasoning: Optional[str] = None,
        extracted_answer: Optional[str] = None,
        score: Optional[float] = None,
        response_time_ms: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        error: Optional[str] = None,
    ) -> int:
        """Add a response to the database.

        Args:
            task_id: Task identifier
            run_id: Run identifier
            dataset_name: Name of the dataset
            entry_idx: Index of the entry in the dataset
            question: The question/prompt
            expected_answer: The expected answer
            entry_metadata: Optional metadata for the entry
            prompt_template_id: ID of the prompt template used
            system_prompt: The system prompt used
            temperature: Temperature setting used
            model: Model name
            full_response: Full model response
            reasoning: Model's reasoning/thinking content
            extracted_answer: Extracted answer from response
            score: Score from evaluation
            response_time_ms: Response time in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            reasoning_tokens: Tokens used for reasoning
            cost: Cost in USD from OpenRouter
            error: Error message if failed

        Returns:
            Response ID
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO responses (
                    task_id, run_id, dataset_name, entry_idx, question, expected_answer,
                    entry_metadata_json, prompt_template_id, system_prompt, temperature,
                    full_response, reasoning, extracted_answer, score, model,
                    response_time_ms, input_tokens, output_tokens, reasoning_tokens, cost, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    dataset_name,
                    entry_idx,
                    question,
                    expected_answer,
                    json.dumps(entry_metadata) if entry_metadata else None,
                    prompt_template_id,
                    system_prompt,
                    temperature,
                    full_response,
                    reasoning,
                    extracted_answer,
                    score,
                    model,
                    response_time_ms,
                    input_tokens,
                    output_tokens,
                    reasoning_tokens,
                    cost,
                    error,
                ),
            )
            return cursor.lastrowid

    def get_passed_responses(
        self, run_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get responses that passed (score == 1.0).

        Args:
            run_id: Run identifier
            limit: Maximum number of responses
            offset: Number of responses to skip

        Returns:
            List of response dicts
        """
        with self._get_connection() as conn:
            query = """
                SELECT * FROM responses
                WHERE run_id = ? AND passed = 1
                ORDER BY response_id
            """
            params = [run_id]
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_all_responses(
        self, run_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get all responses for a run.

        Args:
            run_id: Run identifier
            limit: Maximum number of responses
            offset: Number of responses to skip

        Returns:
            List of response dicts
        """
        with self._get_connection() as conn:
            query = """
                SELECT * FROM responses
                WHERE run_id = ?
                ORDER BY response_id
            """
            params = [run_id]
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def count_responses(self, run_id: str, passed_only: bool = False) -> int:
        """Count responses for a run.

        Args:
            run_id: Run identifier
            passed_only: If True, only count passed responses

        Returns:
            Number of responses
        """
        with self._get_connection() as conn:
            if passed_only:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM responses WHERE run_id = ? AND passed = 1",
                    (run_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM responses WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            return row["count"]

    # ========== Statistics ==========

    def get_run_statistics(self, run_id: str) -> dict[str, Any]:
        """Get comprehensive statistics for a run.

        Args:
            run_id: Run identifier

        Returns:
            Dict with statistics
        """
        with self._get_connection() as conn:
            # Task counts
            task_counts = self.get_task_counts(run_id)

            # Response stats
            response_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_responses,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_responses,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_responses,
                    AVG(score) as avg_score,
                    AVG(response_time_ms) as avg_response_time_ms,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(cost) as total_cost
                FROM responses
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            # Dataset breakdown
            dataset_stats = conn.execute(
                """
                SELECT
                    dataset_name,
                    COUNT(*) as total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                    AVG(score) as avg_score
                FROM responses
                WHERE run_id = ?
                GROUP BY dataset_name
                ORDER BY dataset_name
                """,
                (run_id,),
            ).fetchall()

            # Prompt template breakdown
            template_stats = conn.execute(
                """
                SELECT
                    prompt_template_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                    AVG(score) as avg_score
                FROM responses
                WHERE run_id = ?
                GROUP BY prompt_template_id
                ORDER BY prompt_template_id
                """,
                (run_id,),
            ).fetchall()

            # Temperature breakdown
            temp_stats = conn.execute(
                """
                SELECT
                    temperature,
                    COUNT(*) as total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed,
                    AVG(score) as avg_score
                FROM responses
                WHERE run_id = ?
                GROUP BY temperature
                ORDER BY temperature
                """,
                (run_id,),
            ).fetchall()

            return {
                "run_id": run_id,
                "task_counts": task_counts,
                "total_responses": response_stats["total_responses"] or 0,
                "passed_responses": response_stats["passed_responses"] or 0,
                "error_responses": response_stats["error_responses"] or 0,
                "pass_rate": (
                    response_stats["passed_responses"] / response_stats["total_responses"]
                    if response_stats["total_responses"]
                    else 0
                ),
                "avg_score": response_stats["avg_score"],
                "avg_response_time_ms": response_stats["avg_response_time_ms"],
                "total_input_tokens": response_stats["total_input_tokens"] or 0,
                "total_output_tokens": response_stats["total_output_tokens"] or 0,
                "total_cost": response_stats["total_cost"] or 0.0,
                "by_dataset": [dict(row) for row in dataset_stats],
                "by_template": [dict(row) for row in template_stats],
                "by_temperature": [dict(row) for row in temp_stats],
            }
