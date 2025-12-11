#!/usr/bin/env python3
"""
Evaluate DAM predictions on Ref-L4 dataset with GPT summarization
1. Matches IDs from generated captions with ground truth
2. Uses GPT to summarize both captions
3. Calculates metrics on the summarized versions
"""

import json
import os
import argparse
from openai import OpenAI
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from tqdm import tqdm


def load_predictions(pred_file):
    """Load DAM predictions (format: {image_id: caption})"""
    print(f"Loading predictions from {pred_file}...")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    print(f"✓ Loaded {len(predictions)} predictions")
    return predictions


def load_ground_truth(gt_file):
    """Load ground truth captions (format: {image_id: [caption1, caption2, ...]})"""
    print(f"Loading ground truth from {gt_file}...")
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    print(f"✓ Loaded {len(ground_truth)} ground truth entries")
    return ground_truth


def summarize_with_gpt(client, caption, model="gpt-4o-mini"):
    """
    Summarize a caption using GPT

    Args:
        client: OpenAI client
        caption: Caption text to summarize
        model: GPT model to use

    Returns:
        Summarized caption
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes image captions. "
                               "Provide a concise, clear summary that captures the key visual elements. "
                               "Keep the summary under 50 words while preserving important details."
                },
                {
                    "role": "user",
                    "content": f"Summarize this image caption:\n\n{caption}"
                }
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing with GPT: {e}")
        return caption  # Return original if summarization fails


def process_captions_with_gpt(predictions, ground_truth, api_key, model="gpt-4o-mini", skip_summarization=False):
    """
    Process predictions and ground truth through GPT for summarization

    Args:
        predictions: dict of {image_id: generated_caption}
        ground_truth: dict of {image_id: [gt_caption1, gt_caption2, ...]}
        api_key: OpenAI API key
        model: GPT model to use
        skip_summarization: If True, skip GPT summarization step

    Returns:
        tuple of (processed_predictions, processed_ground_truth)
    """
    if skip_summarization:
        print("\n⚠️  Skipping GPT summarization step")
        # Convert to evaluation format without summarization
        processed_preds = {}
        processed_gts = {}

        common_ids = set(predictions.keys()) & set(ground_truth.keys())

        for img_id in common_ids:
            processed_preds[img_id] = [predictions[img_id]]
            processed_gts[img_id] = ground_truth[img_id] if isinstance(ground_truth[img_id], list) else [ground_truth[img_id]]

        print(f"✓ Prepared {len(common_ids)} samples without summarization")
        return processed_preds, processed_gts

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 80)
    print("Summarizing captions with GPT...")
    print("=" * 80)

    processed_preds = {}
    processed_gts = {}

    # Find common image IDs
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    print(f"\nFound {len(common_ids)} common image IDs")

    # Process each image
    for img_id in tqdm(common_ids, desc="Processing captions"):
        # Summarize prediction
        pred_caption = predictions[img_id]
        summarized_pred = summarize_with_gpt(client, pred_caption, model)
        processed_preds[img_id] = [summarized_pred]

        # Summarize ground truth (can be multiple)
        gt_captions = ground_truth[img_id]
        if not isinstance(gt_captions, list):
            gt_captions = [gt_captions]

        summarized_gts = []
        for gt_caption in gt_captions:
            summarized_gt = summarize_with_gpt(client, gt_caption, model)
            summarized_gts.append(summarized_gt)

        processed_gts[img_id] = summarized_gts

    print(f"\n✓ Processed {len(processed_preds)} samples")
    return processed_preds, processed_gts


def evaluate_metrics(gts, res):
    """
    Compute standard captioning metrics

    Args:
        gts: dict of {image_id: [ref1, ref2, ...]}
        res: dict of {image_id: [prediction]}

    Returns:
        dict of metric scores
    """
    print("\n" + "=" * 80)
    print("Computing Metrics...")
    print("=" * 80)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
    ]

    # Try to include SPICE if available
    try:
        scorers.append((Spice(), "SPICE"))
    except:
        print("⚠️  SPICE not available, skipping...")

    results = {}

    for scorer, method in scorers:
        print(f"\nComputing {method}...")
        try:
            score, scores = scorer.compute_score(gts, res)

            if isinstance(method, list):
                # BLEU returns multiple scores
                for m, s in zip(method, score):
                    results[m] = s
                    print(f"  {m}: {s:.4f}")
            else:
                results[method] = score
                print(f"  {method}: {score:.4f}")
        except Exception as e:
            print(f"  Error computing {method}: {e}")
            if isinstance(method, list):
                for m in method:
                    results[m] = 0.0
            else:
                results[method] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate DAM predictions with GPT summarization'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='results/full_dataset/ref_l4_dam_captions.json',
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='data/ref-l4/ground_truth.json',
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/full_dataset/ref_l4_dam_gpt_eval_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env variable)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='GPT model to use for summarization'
    )
    parser.add_argument(
        '--skip_gpt',
        action='store_true',
        help='Skip GPT summarization and evaluate directly'
    )
    parser.add_argument(
        '--save_processed',
        type=str,
        default=None,
        help='Save processed (summarized) captions to this file'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key and not args.skip_gpt:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api_key")
        return

    print("=" * 80)
    print("DAM Ref-L4 Evaluation with GPT Summarization")
    print("=" * 80)
    print(f"\nPredictions: {args.predictions}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"Output: {args.output}")
    print(f"GPT Model: {args.model}")
    if args.skip_gpt:
        print("Mode: Direct evaluation (no GPT summarization)")
    else:
        print("Mode: With GPT summarization")

    # Load data
    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)

    # Process captions through GPT
    processed_preds, processed_gts = process_captions_with_gpt(
        predictions,
        ground_truth,
        api_key,
        args.model,
        skip_summarization=args.skip_gpt
    )

    # Save processed captions if requested
    if args.save_processed:
        print(f"\nSaving processed captions to {args.save_processed}...")
        os.makedirs(os.path.dirname(args.save_processed), exist_ok=True)
        with open(args.save_processed, 'w') as f:
            json.dump({
                'predictions': processed_preds,
                'ground_truth': processed_gts
            }, f, indent=2)
        print(f"✓ Saved processed captions")

    # Compute metrics
    results = evaluate_metrics(processed_gts, processed_preds)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\nCaptioning Metrics:")
    print(f"  BLEU-1      : {results.get('BLEU-1', 0.0):.4f}")
    print(f"  BLEU-2      : {results.get('BLEU-2', 0.0):.4f}")
    print(f"  BLEU-3      : {results.get('BLEU-3', 0.0):.4f}")
    print(f"  BLEU-4      : {results.get('BLEU-4', 0.0):.4f}")
    print(f"  METEOR      : {results.get('METEOR', 0.0):.4f}")
    print(f"  ROUGE-L     : {results.get('ROUGE-L', 0.0):.4f}")
    print(f"  CIDEr       : {results.get('CIDEr', 0.0):.4f}")
    if 'SPICE' in results:
        print(f"  SPICE       : {results.get('SPICE', 0.0):.4f}")

    print("\n" + "=" * 80)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'metrics': results,
            'config': {
                'predictions': args.predictions,
                'ground_truth': args.ground_truth,
                'gpt_model': args.model,
                'skip_gpt': args.skip_gpt,
                'num_samples': len(processed_preds)
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
