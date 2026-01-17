"""
Data augmentation for synthetic reasoning traces.

Transforms raw generation output into multiple training formats.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from .db import DatabaseManager


FormatType = Literal["internal", "explicit", "combined"]


@dataclass
class AugmentedSample:
    """A single augmented training sample."""

    text: str  # The full training text
    format_type: FormatType
    source_response_id: int
    dataset_name: str
    metadata: dict[str, Any]


def strip_answer_tags(text: str) -> tuple[str, str | None]:
    """Remove <answer> tags and extract the answer.

    Args:
        text: Text potentially containing <answer></answer> tags

    Returns:
        Tuple of (text_without_tags, extracted_answer)
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        # Remove the answer tags from text
        text_clean = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return text_clean, answer

    return text, None


def format_internal(
    question: str,
    reasoning: str | None,
    extracted_answer: str | None,
) -> str | None:
    """Format: prompt → internal reasoning → answer.

    Args:
        question: The problem statement
        reasoning: Model's internal thinking
        extracted_answer: The final answer

    Returns:
        Formatted text or None if missing required fields
    """
    if not reasoning or not extracted_answer:
        return None

    return f"""Problem: {question}

<reasoning>
{reasoning.strip()}
</reasoning>

<answer>{extracted_answer}</answer>"""


def format_explicit(
    question: str,
    full_response: str | None,
) -> str | None:
    """Format: prompt → explicit step-by-step → answer.

    Args:
        question: The problem statement
        full_response: Model's explicit reasoning with answer

    Returns:
        Formatted text or None if missing required fields
    """
    if not full_response:
        return None

    # full_response already contains the answer tags
    return f"""Problem: {question}

{full_response.strip()}"""


def format_combined(
    question: str,
    reasoning: str | None,
    full_response: str | None,
) -> str | None:
    """Format: prompt → internal reasoning → explicit reasoning → answer.

    Args:
        question: The problem statement
        reasoning: Model's internal thinking
        full_response: Model's explicit reasoning with answer

    Returns:
        Formatted text or None if missing required fields
    """
    if not reasoning or not full_response:
        return None

    # Strip answer from full_response since we'll have it at the end
    explicit_text, answer = strip_answer_tags(full_response)

    if not answer:
        # If no answer tags found, use full response as-is
        return f"""Problem: {question}

<internal_reasoning>
{reasoning.strip()}
</internal_reasoning>

<explicit_reasoning>
{full_response.strip()}
</explicit_reasoning>"""

    return f"""Problem: {question}

<internal_reasoning>
{reasoning.strip()}
</internal_reasoning>

<explicit_reasoning>
{explicit_text.strip()}
</explicit_reasoning>

<answer>{answer}</answer>"""


FORMATTERS = {
    "internal": format_internal,
    "explicit": format_explicit,
    "combined": format_combined,
}


def augment_response(
    response: dict[str, Any],
    formats: list[FormatType] | None = None,
) -> list[AugmentedSample]:
    """Augment a single response into multiple formats.

    Args:
        response: Response record from database
        formats: List of format types to generate (default: all)

    Returns:
        List of AugmentedSample objects
    """
    formats = formats or ["internal", "explicit", "combined"]
    samples = []

    question = response["question"]
    reasoning = response.get("reasoning")
    full_response = response.get("full_response")
    extracted_answer = response.get("extracted_answer")

    base_metadata = {
        "dataset": response["dataset_name"],
        "entry_idx": response["entry_idx"],
        "prompt_template": response["prompt_template_id"],
        "temperature": response["temperature"],
        "model": response["model"],
        "score": response["score"],
        "expected_answer": response["expected_answer"],
    }

    for fmt in formats:
        if fmt == "internal":
            text = format_internal(question, reasoning, extracted_answer)
        elif fmt == "explicit":
            text = format_explicit(question, full_response)
        elif fmt == "combined":
            text = format_combined(question, reasoning, full_response)
        else:
            continue

        if text:
            samples.append(AugmentedSample(
                text=text,
                format_type=fmt,
                source_response_id=response["response_id"],
                dataset_name=response["dataset_name"],
                metadata={**base_metadata, "format": fmt},
            ))

    return samples


def iter_augmented(
    db: DatabaseManager,
    run_id: str,
    formats: list[FormatType] | None = None,
    passed_only: bool = True,
    batch_size: int = 1000,
) -> Iterator[AugmentedSample]:
    """Iterate over augmented samples from a run.

    Args:
        db: Database manager
        run_id: Run identifier
        formats: List of format types to generate
        passed_only: Only include passed responses
        batch_size: Batch size for database queries

    Yields:
        AugmentedSample objects
    """
    offset = 0
    while True:
        if passed_only:
            responses = db.get_passed_responses(run_id, limit=batch_size, offset=offset)
        else:
            responses = db.get_all_responses(run_id, limit=batch_size, offset=offset)

        if not responses:
            break

        for response in responses:
            for sample in augment_response(response, formats):
                yield sample

        offset += batch_size


def export_augmented(
    db: DatabaseManager,
    run_id: str,
    output_path: Path,
    formats: list[FormatType] | None = None,
    passed_only: bool = True,
) -> dict[str, int]:
    """Export augmented samples to JSONL.

    Args:
        db: Database manager
        run_id: Run identifier
        output_path: Output file path
        formats: List of format types to generate
        passed_only: Only include passed responses

    Returns:
        Dict with counts per format type
    """
    counts = {"internal": 0, "explicit": 0, "combined": 0, "total": 0}

    with open(output_path, "w") as f:
        for sample in iter_augmented(db, run_id, formats, passed_only):
            record = {
                "text": sample.text,
                "format": sample.format_type,
                "metadata": sample.metadata,
            }
            f.write(json.dumps(record) + "\n")
            counts[sample.format_type] += 1
            counts["total"] += 1

    return counts


def preview_augmented(
    db: DatabaseManager,
    run_id: str,
    formats: list[FormatType] | None = None,
    n_samples: int = 3,
) -> None:
    """Preview augmented samples.

    Args:
        db: Database manager
        run_id: Run identifier
        formats: List of format types to generate
        n_samples: Number of samples to preview per format
    """
    formats = formats or ["internal", "explicit", "combined"]
    seen = {fmt: 0 for fmt in formats}

    for sample in iter_augmented(db, run_id, formats, passed_only=True):
        if seen[sample.format_type] >= n_samples:
            continue

        print(f"\n{'='*80}")
        print(f"FORMAT: {sample.format_type.upper()} | Dataset: {sample.dataset_name}")
        print(f"{'='*80}")
        print(sample.text[:2000])
        if len(sample.text) > 2000:
            print(f"\n... [{len(sample.text) - 2000} more chars]")
        print()

        seen[sample.format_type] += 1

        if all(v >= n_samples for v in seen.values()):
            break
