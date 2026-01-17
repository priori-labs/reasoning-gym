"""
Export functionality for synthetic data generation.
"""

import json
from pathlib import Path
from typing import Any, Iterator, Literal

from .db import DatabaseManager


ExportFormat = Literal["cpt", "sft", "raw"]


def format_cpt(response: dict[str, Any]) -> dict[str, Any]:
    """Format response for Continued Pre-Training (CPT).

    Output format:
    {"text": "Problem: {question}\n\n{full_response}", "reasoning": ..., "metadata": {...}}

    Args:
        response: Response record from database

    Returns:
        Formatted record for CPT
    """
    text = f"Problem: {response['question']}\n\n{response['full_response']}"

    metadata = {
        "dataset": response["dataset_name"],
        "entry_idx": response["entry_idx"],
        "prompt_template": response["prompt_template_id"],
        "temperature": response["temperature"],
        "model": response["model"],
        "score": response["score"],
        "expected_answer": response["expected_answer"],
        "extracted_answer": response["extracted_answer"],
    }

    # Include entry metadata if present
    if response.get("entry_metadata_json"):
        try:
            entry_meta = json.loads(response["entry_metadata_json"])
            metadata["entry_metadata"] = entry_meta
        except json.JSONDecodeError:
            pass

    result = {
        "text": text,
        "metadata": metadata,
    }

    # Include reasoning if present (from models that expose thinking)
    if response.get("reasoning"):
        result["reasoning"] = response["reasoning"]
        if response.get("reasoning_tokens"):
            metadata["reasoning_tokens"] = response["reasoning_tokens"]

    # Include cost if present
    if response.get("cost") is not None:
        metadata["cost"] = response["cost"]

    return result


def format_sft(response: dict[str, Any]) -> dict[str, Any]:
    """Format response for Supervised Fine-Tuning (SFT).

    Output format:
    {"instruction": question, "response": full_response, "reasoning": ..., "system_prompt": ...}

    Args:
        response: Response record from database

    Returns:
        Formatted record for SFT
    """
    result = {
        "instruction": response["question"],
        "response": response["full_response"],
        "system_prompt": response["system_prompt"],
        "metadata": {
            "dataset": response["dataset_name"],
            "entry_idx": response["entry_idx"],
            "prompt_template": response["prompt_template_id"],
            "temperature": response["temperature"],
            "model": response["model"],
            "score": response["score"],
            "expected_answer": response["expected_answer"],
            "extracted_answer": response["extracted_answer"],
        },
    }

    # Include reasoning if present (from models that expose thinking)
    if response.get("reasoning"):
        result["reasoning"] = response["reasoning"]
        if response.get("reasoning_tokens"):
            result["metadata"]["reasoning_tokens"] = response["reasoning_tokens"]

    # Include cost if present
    if response.get("cost") is not None:
        result["metadata"]["cost"] = response["cost"]

    return result


def format_raw(response: dict[str, Any]) -> dict[str, Any]:
    """Format response as raw database record.

    Args:
        response: Response record from database

    Returns:
        Full response record
    """
    # Parse entry_metadata_json if present
    result = dict(response)
    if result.get("entry_metadata_json"):
        try:
            result["entry_metadata"] = json.loads(result["entry_metadata_json"])
            del result["entry_metadata_json"]
        except json.JSONDecodeError:
            pass
    return result


FORMATTERS = {
    "cpt": format_cpt,
    "sft": format_sft,
    "raw": format_raw,
}


def iter_export(
    db: DatabaseManager,
    run_id: str,
    format: ExportFormat = "cpt",
    passed_only: bool = True,
    batch_size: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Iterate over formatted export records.

    Args:
        db: Database manager
        run_id: Run identifier
        format: Export format (cpt, sft, raw)
        passed_only: If True, only export passed responses
        batch_size: Batch size for database queries

    Yields:
        Formatted records
    """
    formatter = FORMATTERS.get(format)
    if not formatter:
        raise ValueError(f"Unknown export format: {format}")

    offset = 0
    while True:
        if passed_only:
            responses = db.get_passed_responses(run_id, limit=batch_size, offset=offset)
        else:
            responses = db.get_all_responses(run_id, limit=batch_size, offset=offset)

        if not responses:
            break

        for response in responses:
            # Skip responses without full_response (errors)
            if format != "raw" and not response.get("full_response"):
                continue
            yield formatter(response)

        offset += batch_size


def export_to_jsonl(
    db: DatabaseManager,
    run_id: str,
    output_path: Path,
    format: ExportFormat = "cpt",
    passed_only: bool = True,
) -> int:
    """Export responses to JSONL file.

    Args:
        db: Database manager
        run_id: Run identifier
        output_path: Output file path
        format: Export format (cpt, sft, raw)
        passed_only: If True, only export passed responses

    Returns:
        Number of records exported
    """
    count = 0
    with open(output_path, "w") as f:
        for record in iter_export(db, run_id, format, passed_only):
            f.write(json.dumps(record) + "\n")
            count += 1

    return count


def export_to_parquet(
    db: DatabaseManager,
    run_id: str,
    output_path: Path,
    format: ExportFormat = "cpt",
    passed_only: bool = True,
) -> int:
    """Export responses to Parquet file.

    Requires pyarrow to be installed.

    Args:
        db: Database manager
        run_id: Run identifier
        output_path: Output file path
        format: Export format (cpt, sft, raw)
        passed_only: If True, only export passed responses

    Returns:
        Number of records exported
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for Parquet export. Install with: pip install pyarrow")

    records = list(iter_export(db, run_id, format, passed_only))
    if not records:
        return 0

    # Flatten nested dicts for parquet
    flat_records = []
    for record in records:
        flat = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat[f"{key}_{k}"] = v
            else:
                flat[key] = value
        flat_records.append(flat)

    table = pa.Table.from_pylist(flat_records)
    pq.write_table(table, output_path)

    return len(records)
