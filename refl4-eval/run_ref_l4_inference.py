#!/usr/bin/env python3
"""
Run DAM inference on Ref-L4 dataset using the describe-anything code
"""

import sys
import os

# Add DAM to path
dam_path = "/home/saksham/Desktop/GenAI/FocusDAM/external/describe-anything"
sys.path.insert(0, dam_path)

import json
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
import numpy as np

# Import DAM utilities
sys.path.insert(0, os.path.join(dam_path, "evaluation"))
import dam_utils

print("=" * 80)
print("DAM Inference on Ref-L4 Dataset")
print("=" * 80)

# Configuration
MODEL_PATH = "/home/saksham/Desktop/GenAI/Kiru-Project/models/DAM-3B"
DATA_ROOT = "/home/saksham/Desktop/GenAI/Kiru-Project/data/ref-l4"
OUTPUT_FILE = "/home/saksham/Desktop/GenAI/Kiru-Project/results/full_dataset/ref_l4_dam_captions.json"

# Number of samples to process (None = all samples)
NUM_SAMPLES = 3000
START_INDEX = 1112  # Skip already processed samples

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Dataset: {DATA_ROOT}")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Samples: {NUM_SAMPLES}")

# Initialize DAM model
print("\nInitializing DAM model...")
try:
    dam_utils.init(
        model_path=MODEL_PATH,
        sep=", ",
        conv_mode="v1",
        query="<image>\nDescribe the masked region in detail.",
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
        crop_mode="full+focal_crop",
        no_concat_images=False,
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nMake sure you have:")
    print("  1. Downloaded DAM-3B model to models/DAM-3B")
    print("  2. Installed required packages: torch, transformers, etc.")
    sys.exit(1)

# Load dataset
print("\nLoading Ref-L4 dataset...")
try:
    dataset = load_from_disk(f'{DATA_ROOT}/dataset')
    val_data = dataset['val']
    print(f"✓ Loaded {len(val_data)} validation samples")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)

# Determine number of samples to process
if NUM_SAMPLES is None:
    end_index = len(val_data)
else:
    end_index = min(START_INDEX + NUM_SAMPLES, len(val_data))

# Run inference
print(f"\nRunning inference from sample {START_INDEX} to {end_index} ({end_index - START_INDEX} samples)...")
print(f"Saving progress every 100 samples to avoid data loss...")

# Load existing predictions if they exist
predictions = {}
if os.path.exists(OUTPUT_FILE):
    print(f"Loading existing predictions from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'r') as f:
        predictions = json.load(f)
    print(f"✓ Loaded {len(predictions)} existing predictions")

# Create output directory
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

for i in tqdm(range(START_INDEX, end_index)):
    sample = val_data[i]

    try:
        # Load image
        img_path = os.path.join(DATA_ROOT, sample['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Convert bbox to mask
        # Ref-L4 bbox format: [x, y, width, height]
        bbox = sample['bbox']
        mask = np.zeros((sample['height'], sample['width']), dtype=np.uint8)
        x, y, w, h = [int(v) for v in bbox]
        mask[y:y+h, x:x+w] = 255
        mask = Image.fromarray(mask)

        # Generate description
        outputs, info = dam_utils.get_description(img, mask)

        # Store prediction
        image_id = sample['image_id']
        predictions[image_id] = outputs

        # Print first few examples
        if i < 3:
            print(f"\n--- Sample {i+1} ---")
            print(f"Image: {sample['file_name']}")
            print(f"Ground truth: {sample['caption'][:100]}...")
            print(f"Prediction: {outputs[:100]}...")

        # Save progress every 100 samples
        if (i + 1) % 100 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(predictions, f, indent=2)

    except Exception as e:
        print(f"\nError on sample {i} ({sample['image_id']}): {e}")
        predictions[sample['image_id']] = f"ERROR: {str(e)}"

# Save predictions
print(f"\nSaving predictions to {OUTPUT_FILE}...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"✓ Saved {len(predictions)} predictions")

print("\n" + "=" * 80)
print("Inference complete!")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Run evaluation: python eval_ref_l4.py --predictions {OUTPUT_FILE}")
print(f"  2. Compare with Table 4 baseline scores")