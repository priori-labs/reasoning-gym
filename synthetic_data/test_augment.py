#!/usr/bin/env python
"""
Test script for data augmentation.

Usage:
    python -m synthetic_data.test_augment --db-path arc_deepseek.db --run-id <run_id>
    python -m synthetic_data.test_augment --db-path arc_deepseek.db --latest
"""

import argparse
from pathlib import Path

from .augment import export_augmented, preview_augmented
from .db import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Test data augmentation")
    parser.add_argument("--db-path", required=True, help="Path to database")
    parser.add_argument("--run-id", help="Run ID to process")
    parser.add_argument("--latest", action="store_true", help="Use latest run")
    parser.add_argument("--preview", type=int, default=1, help="Preview N samples per format")
    parser.add_argument("--export", help="Export to JSONL file")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["internal", "explicit", "combined"],
        default=["internal", "explicit", "combined"],
        help="Formats to generate",
    )

    args = parser.parse_args()

    db = DatabaseManager(Path(args.db_path))

    # Get run ID
    if args.latest:
        runs = db.list_runs()
        if not runs:
            print("No runs found")
            return 1
        run_id = runs[0]["run_id"]
        print(f"Using latest run: {run_id}")
    elif args.run_id:
        run_id = args.run_id
    else:
        print("Error: Must specify --run-id or --latest")
        return 1

    # Preview samples
    if args.preview:
        print(f"\nPreviewing {args.preview} sample(s) per format...")
        preview_augmented(db, run_id, args.formats, args.preview)

    # Export if requested
    if args.export:
        print(f"\nExporting to {args.export}...")
        counts = export_augmented(
            db, run_id, Path(args.export), args.formats, passed_only=True
        )
        print(f"Exported: {counts}")

    return 0


if __name__ == "__main__":
    exit(main())
