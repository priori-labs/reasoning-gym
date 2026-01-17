"""
Synthetic data generation pipeline for Continued Pre-Training (CPT).

This module provides tools for generating synthetic reasoning traces using
teacher models via OpenRouter, with systematic variation for maximum dataset diversity.

Usage:
    python -m synthetic_data.generate run --config configs/default.yaml
    python -m synthetic_data.generate resume --run-id <uuid>
    python -m synthetic_data.generate stats --run-id <uuid>
    python -m synthetic_data.generate export --run-id <uuid> --output data.jsonl

See configs/default.yaml for configuration options.
"""

from .db import DatabaseManager
from .export import export_to_jsonl, export_to_parquet
from .generator import GenerationConfig, SyntheticDataGenerator, TaskResult, create_generator
from .prompts import PROMPTS, TEMPERATURES, get_all_variations, get_prompt_template

__all__ = [
    "DatabaseManager",
    "GenerationConfig",
    "SyntheticDataGenerator",
    "TaskResult",
    "create_generator",
    "export_to_jsonl",
    "export_to_parquet",
    "PROMPTS",
    "TEMPERATURES",
    "get_prompt_template",
    "get_all_variations",
]
