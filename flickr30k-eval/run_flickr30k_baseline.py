#!/usr/bin/env python3
"""
Run DAM inference on Flickr30k Entities 100-sample baseline - Version 2
Generates very short reference phrases (1-5 words) matching Flickr30k style
"""

import sys
import os

# Add DAM to path
dam_path = "/home/saksham/Desktop/GenAI/FocusDAM/external/describe-anything"
sys.path.insert(0, dam_path)

import json
from PIL import Image
from tqdm import tqdm
import numpy as np

# Import DAM utilities
sys.path.insert(0, os.path.join(dam_path, "evaluation"))
import dam_utils

print("=" * 80)
print("Flickr30k Entities Baseline v2 - 100 Sample Subset")
print("Ultra-Short Reference Phrase Generation (1-5 words)")
print("=" * 80)

# Configuration
MODEL_PATH = "/home/saksham/Desktop/GenAI/Kiru-Project/models/DAM-3B"
DATA_ROOT = "/home/saksham/Desktop/GenAI/Kiru-Project/data/flickr30k"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATIONS_FILE = os.path.join(DATA_ROOT, "entities_annotations.json")
OUTPUT_FILE = "/home/saksham/Desktop/GenAI/Kiru-Project/results/flickr30k/baseline_100_dam3b_v2_predictions.json"

# Number of samples to process (100 for baseline subset)
NUM_SAMPLES = 100

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Annotations: {ANNOTATIONS_FILE}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Samples: {NUM_SAMPLES}")

# Initialize DAM model with ultra-short phrase constraint
print("\nInitializing DAM model...")
print("Prompt: Generate ultra-short phrases (1-5 words max)")
try:
    dam_utils.init(
        model_path=MODEL_PATH,
        sep=", ",
        conv_mode="v1",
        # Simplified query for shorter outputs
        query="<image> Short phrase (1-5 words):",
        temperature=0.1,  # Lower temperature for more focused outputs
        top_p=None,
        num_beams=1,
        max_new_tokens=8,  # Strict limit for short phrases
        crop_mode="full+focal_crop",
        no_concat_images=False,
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nMake sure you have:")
    print("  1. Downloaded DAM-3B model to models/DAM-3B")
    print("  2. Installed required packages")
    print("  3. FocusDAM repository at /home/saksham/Desktop/GenAI/FocusDAM")
    sys.exit(1)

# Load annotations
print("\nLoading Flickr30k Entities annotations...")
try:
    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)
    print(f"✓ Loaded {len(annotations)} phrase-bbox annotations")
except Exception as e:
    print(f"✗ Error loading annotations: {e}")
    sys.exit(1)

# Use first 100 samples
NUM_SAMPLES = min(NUM_SAMPLES, len(annotations))
annotations = annotations[:NUM_SAMPLES]
print(f"✓ Using first {NUM_SAMPLES} samples for baseline")

# Run inference
print(f"\nRunning inference on {NUM_SAMPLES} phrase-bbox pairs...")
print("Generating ultra-short phrases (1-5 words)...")
predictions = {}
errors = []

for i in tqdm(range(NUM_SAMPLES)):
    ann = annotations[i]
    ann_id = ann['ann_id']

    try:
        # Load image
        img_path = os.path.join(IMAGES_DIR, ann['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Convert bbox to mask
        bbox = ann['bbox']
        mask = np.zeros((ann['height'], ann['width']), dtype=np.uint8)
        x, y, w, h = [int(v) for v in bbox]
        mask[y:y+h, x:x+w] = 255
        mask = Image.fromarray(mask)

        # Generate description
        outputs, info = dam_utils.get_description(img, mask)

        # Post-process: Keep only first sentence/phrase if it's too long
        if '.' in outputs:
            outputs = outputs.split('.')[0].strip()
        if ',' in outputs and len(outputs.split()) > 6:
            outputs = outputs.split(',')[0].strip()

        # Limit to first 5 words as a safety measure
        words = outputs.split()
        if len(words) > 5:
            outputs = ' '.join(words[:5])

        # Store prediction
        predictions[ann_id] = outputs

        # Print first few examples
        if i < 10:
            print(f"\n--- Sample {i+1} ---")
            print(f"Image: {ann['file_name']}")
            print(f"GT ({len(ann['phrase'].split())} words): {ann['phrase']}")
            print(f"Pred ({len(outputs.split())} words): {outputs}")

    except Exception as e:
        error_msg = f"Error on {ann_id}: {e}"
        print(f"\n{error_msg}")
        predictions[ann_id] = f"ERROR: {str(e)}"
        errors.append((ann_id, str(e)))

# Save predictions
print(f"\nSaving predictions to {OUTPUT_FILE}...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"✓ Saved {len(predictions)} predictions")

if errors:
    print(f"\n⚠ {len(errors)} errors occurred")

# Print statistics
pred_lengths = [len(p.split()) for p in predictions.values() if not p.startswith('ERROR')]
print("\n" + "=" * 80)
print("Prediction Statistics")
print("=" * 80)
print(f"Total samples: {NUM_SAMPLES}")
print(f"Successful predictions: {len(pred_lengths)}")
print(f"Avg prediction length: {sum(pred_lengths)/len(pred_lengths):.1f} words")
print(f"Min/Max length: {min(pred_lengths)}/{max(pred_lengths)} words")

print("\n" + "=" * 80)
print("Inference complete!")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Run evaluation:")
print(f"     python eval_flickr30k_entities.py \\")
print(f"       --predictions {OUTPUT_FILE} \\")
print(f"       --annotations {DATA_ROOT}/entities_groundtruth.json \\")
print(f"       --output results/flickr30k/baseline_100_v2_evaluation.json")
