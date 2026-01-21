#!/usr/bin/env python3
"""Re-score existing evaluation results with updated extraction/scoring logic.

Usage:
    python eval/rescore.py experiments/results/exp13_puzzle/cpt/exp13_20260120_194549/

This will:
1. Re-extract answers from stored model responses
2. Re-score using the updated scoring logic
3. Save new summary alongside the original
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning_gym.utils import extract_answer
from reasoning_gym.factory import create_dataset


def rescore_dataset(result_file: Path, verbose: bool = False) -> dict:
    """Re-extract answers and optionally re-score a single dataset result file.

    Note: For datasets with custom scorers that require metadata, we can only
    re-extract answers, not re-score (since metadata may not be saved).
    """
    with open(result_file) as f:
        data = json.load(f)

    dataset_name = data.get('name', result_file.stem)
    config = data.get('config', {})

    # Check if metadata is available for rescoring
    has_metadata = any(result.get('metadata') for result in data.get('results', []))

    # Try to create the dataset for scoring
    dataset = None
    try:
        dataset = create_dataset(dataset_name, seed=config.get('seed', 42), size=config.get('size', 10))
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not create dataset {dataset_name}: {e}")

    # Track extraction improvements
    extraction_improved = 0
    extraction_unchanged = 0

    new_results = []
    total_best_score = 0
    total_mean_score = 0

    for i, result in enumerate(data.get('results', [])):
        original_answer = result.get('best_model_answer')
        original_score = result.get('best_score', 0)

        # Re-extract from best response
        best_response = result.get('best_full_model_response', '')
        new_answer = extract_answer(best_response)

        # Track extraction changes
        if original_answer is None and new_answer is not None:
            extraction_improved += 1
            if verbose:
                print(f"  Entry {i}: Extraction improved - now got: {str(new_answer)[:50]}")
        else:
            extraction_unchanged += 1

        # Try to re-score if we have the dataset and metadata
        new_score = original_score  # Default to original score
        if dataset is not None and has_metadata:
            entry = {
                'question': result.get('question', ''),
                'answer': result.get('expected_answer', ''),
                'metadata': result.get('metadata', {}),
            }
            try:
                new_score = dataset.score_answer(answer=new_answer, entry=entry)
            except Exception:
                pass  # Keep original score

        new_result = {
            'question': result.get('question'),
            'expected_answer': result.get('expected_answer'),
            'best_model_answer': new_answer,
            'best_full_model_response': best_response,
            'best_score': new_score,
            'mean_score': result.get('mean_score', 0),
            'original_best_score': original_score,
            'original_best_answer': original_answer,
            'completions': result.get('completions', []),
        }

        if result.get('metadata'):
            new_result['metadata'] = result['metadata']

        new_results.append(new_result)
        total_best_score += new_score
        total_mean_score += result.get('mean_score', 0)

    # Update summary
    n = len(new_results)
    data['results'] = new_results
    data['average_best_score'] = total_best_score / n if n > 0 else 0
    data['average_mean_score'] = total_mean_score / n if n > 0 else 0
    data['original_average_best_score'] = sum(r.get('original_best_score', 0) for r in new_results) / n if n else 0
    data['rescored'] = True
    data['extraction_improved'] = extraction_improved

    if extraction_improved > 0:
        print(f"    Extraction improved for {extraction_improved} entries")

    return data


def rescore_directory(results_dir: Path, verbose: bool = False):
    """Re-score all results in a directory."""
    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    # Find all category subdirectories
    categories = [d for d in results_dir.iterdir() if d.is_dir()]

    all_scores = {}

    for category_dir in categories:
        print(f"\nProcessing category: {category_dir.name}")

        for result_file in category_dir.glob('*.json'):
            print(f"  Re-scoring: {result_file.name}")

            try:
                new_data = rescore_dataset(result_file, verbose=verbose)

                # Save rescored results
                output_file = result_file.with_suffix('.rescored.json')
                with open(output_file, 'w') as f:
                    json.dump(new_data, f, indent=2)

                # Track scores
                dataset_name = new_data.get('name', result_file.stem)
                original = new_data.get('original_average_best_score', 0)
                rescored = new_data.get('average_best_score', 0)

                if original != rescored:
                    print(f"    Score changed: {original:.1%} -> {rescored:.1%} ({rescored - original:+.1%})")

                all_scores[dataset_name] = {
                    'original': original,
                    'rescored': rescored,
                    'delta': rescored - original,
                }
            except Exception as e:
                print(f"    Error: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESCORE SUMMARY")
    print("=" * 60)

    improved = [(k, v) for k, v in all_scores.items() if v['delta'] > 0.01]
    degraded = [(k, v) for k, v in all_scores.items() if v['delta'] < -0.01]

    if improved:
        print(f"\nImproved ({len(improved)} datasets):")
        for name, scores in sorted(improved, key=lambda x: -x[1]['delta'])[:10]:
            print(f"  {name}: {scores['original']:.1%} -> {scores['rescored']:.1%} ({scores['delta']:+.1%})")

    if degraded:
        print(f"\nDegraded ({len(degraded)} datasets):")
        for name, scores in sorted(degraded, key=lambda x: x[1]['delta'])[:10]:
            print(f"  {name}: {scores['original']:.1%} -> {scores['rescored']:.1%} ({scores['delta']:+.1%})")

    total_original = sum(v['original'] for v in all_scores.values())
    total_rescored = sum(v['rescored'] for v in all_scores.values())
    n = len(all_scores)

    print(f"\nOverall: {total_original/n:.1%} -> {total_rescored/n:.1%} ({(total_rescored-total_original)/n:+.1%})")


def main():
    parser = argparse.ArgumentParser(description='Re-score evaluation results with updated extraction/scoring')
    parser.add_argument('results_dir', type=Path, help='Directory containing evaluation results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show per-entry score changes')

    args = parser.parse_args()
    rescore_directory(args.results_dir, verbose=args.verbose)


if __name__ == '__main__':
    main()
