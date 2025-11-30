#!/usr/bin/env python3
"""
Analyze DLC-Bench evaluation results and compare with paper benchmarks.
"""

import json
import argparse
from typing import Dict, List, Tuple
import sys

def load_results(eval_file: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(eval_file) as f:
        return json.load(f)

def calculate_statistics(scores: Dict[str, float]) -> Dict:
    """Calculate detailed statistics from scores."""
    scores_list = list(scores.values())

    # Filter out None values
    valid_scores = [s for s in scores_list if s is not None]

    if not valid_scores:
        return {
            'mean': 0.0,
            'min': 0.0,
            'max': 0.0,
            'num_samples': 0,
            'num_zero': 0,
            'num_negative': 0,
            'num_positive': 0,
        }

    return {
        'mean': sum(valid_scores) / len(valid_scores),
        'min': min(valid_scores),
        'max': max(valid_scores),
        'num_samples': len(valid_scores),
        'num_zero': sum(1 for s in valid_scores if s == 0),
        'num_negative': sum(1 for s in valid_scores if s < 0),
        'num_positive': sum(1 for s in valid_scores if s > 0),
    }

def convert_to_percentage(score: float) -> float:
    """Convert score from [-1, 1] range to [0, 100] percentage."""
    # DLC-Bench scores range from -1 to 1
    # Paper reports as percentage (0-100)
    # Formula: (score + 1) / 2 * 100
    return (score + 1) / 2 * 100

def analyze_results(eval_file: str, paper_pos: float = 52.3, paper_neg: float = 82.2) -> None:
    """Analyze evaluation results and compare with paper."""

    print("=" * 80)
    print("DLC-BENCH EVALUATION RESULTS ANALYSIS")
    print("=" * 80)

    # Load results
    results = load_results(eval_file)

    # Extract scores
    avg_pos = results.get('avg_pos', 0)
    avg_neg = results.get('avg_neg', 0)

    # Get individual scores (exclude metadata keys)
    individual_scores = {k: v for k, v in results.items()
                        if k not in ['avg_pos', 'avg_neg'] and isinstance(v, dict)}

    # Extract score components
    scores_pos = {}
    scores_neg = {}
    scores_overall = {}

    for ann_id, data in individual_scores.items():
        if 'score_pos' in data:
            scores_pos[ann_id] = data['score_pos']
        if 'score_neg' in data:
            scores_neg[ann_id] = data['score_neg']
        if 'score' in data:
            scores_overall[ann_id] = data['score']

    # Calculate statistics
    pos_stats = calculate_statistics(scores_pos)
    neg_stats = calculate_statistics(scores_neg)
    overall_stats = calculate_statistics(scores_overall)

    # Convert to percentages
    avg_pos_pct = convert_to_percentage(avg_pos)
    avg_neg_pct = convert_to_percentage(avg_neg)
    avg_overall_pct = (avg_pos_pct + avg_neg_pct) / 2

    print(f"\nEvaluation file: {eval_file}")
    print(f"Total annotations: {len(individual_scores)}")

    # Table 1: Main Results Comparison
    print("\n" + "=" * 80)
    print("TABLE 1: MAIN RESULTS (Percentage Scores)")
    print("=" * 80)
    print(f"{'Metric':<30} {'Your Results':>15} {'Paper (DAM 3B)':>15} {'Difference':>15}")
    print("-" * 80)
    print(f"{'Positive Score':<30} {avg_pos_pct:>14.1f}% {paper_pos:>14.1f}% {avg_pos_pct - paper_pos:>+14.1f}%")
    print(f"{'Negative Score':<30} {avg_neg_pct:>14.1f}% {paper_neg:>14.1f}% {avg_neg_pct - paper_neg:>+14.1f}%")
    print(f"{'Overall Score':<30} {avg_overall_pct:>14.1f}% {(paper_pos + paper_neg)/2:>14.1f}% {avg_overall_pct - (paper_pos + paper_neg)/2:>+14.1f}%")
    print("=" * 80)

    # Table 2: Raw Score Distribution
    print("\n" + "=" * 80)
    print("TABLE 2: RAW SCORES ([-1, 1] range)")
    print("=" * 80)
    print(f"{'Metric':<30} {'Your Score':>15}")
    print("-" * 80)
    print(f"{'Average Positive Score':<30} {avg_pos:>15.3f}")
    print(f"{'Average Negative Score':<30} {avg_neg:>15.3f}")
    print(f"{'Overall Score':<30} {(avg_pos + avg_neg)/2:>15.3f}")
    print("=" * 80)

    # Table 3: Score Distribution Analysis
    print("\n" + "=" * 80)
    print("TABLE 3: SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Category':<25} {'Positive':>12} {'Negative':>12} {'Overall':>12}")
    print("-" * 80)
    print(f"{'Total Samples':<25} {pos_stats['num_samples']:>12} {neg_stats['num_samples']:>12} {overall_stats['num_samples']:>12}")
    print(f"{'Zero Scores':<25} {pos_stats['num_zero']:>12} {neg_stats['num_zero']:>12} {overall_stats['num_zero']:>12}")
    print(f"{'Negative Scores':<25} {pos_stats['num_negative']:>12} {neg_stats['num_negative']:>12} {overall_stats['num_negative']:>12}")
    print(f"{'Positive Scores':<25} {pos_stats['num_positive']:>12} {neg_stats['num_positive']:>12} {overall_stats['num_positive']:>12}")
    print(f"{'Min Score':<25} {pos_stats['min']:>12.3f} {neg_stats['min']:>12.3f} {overall_stats['min']:>12.3f}")
    print(f"{'Max Score':<25} {pos_stats['max']:>12.3f} {neg_stats['max']:>12.3f} {overall_stats['max']:>12.3f}")
    print("=" * 80)

    # Analysis and interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    pos_diff = avg_pos_pct - paper_pos
    neg_diff = avg_neg_pct - paper_neg
    overall_diff = avg_overall_pct - (paper_pos + paper_neg)/2

    print(f"\nScore Differences from Paper:")
    print(f"   Positive: {pos_diff:+.1f}% {'(LOWER)' if pos_diff < 0 else '(HIGHER)'}")
    print(f"   Negative: {neg_diff:+.1f}% {'(LOWER)' if neg_diff < 0 else '(HIGHER)'}")
    print(f"   Overall:  {overall_diff:+.1f}% {'(LOWER)' if overall_diff < 0 else '(HIGHER)'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze DLC-Bench evaluation results')
    parser.add_argument('--eval-file', type=str, required=True,
                       help='Path to evaluation results JSON file')
    parser.add_argument('--paper-pos', type=float, default=52.3,
                       help='Paper reported positive score (default: 52.3 for DAM 3B)')
    parser.add_argument('--paper-neg', type=float, default=82.2,
                       help='Paper reported negative score (default: 82.2 for DAM 3B)')

    args = parser.parse_args()

    try:
        analyze_results(args.eval_file, args.paper_pos, args.paper_neg)
    except FileNotFoundError:
        print(f"Error: File not found: {args.eval_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
