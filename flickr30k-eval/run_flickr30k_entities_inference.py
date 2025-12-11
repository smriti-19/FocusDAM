#!/usr/bin/env python3

# Run DAM inference on Flickr30k Entities dataset (phrase-level grounding)
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

print("DAM Inference on Flickr30k Entities Dataset") 

# Configuration
MODEL_PATH = "/home/saksham/Desktop/GenAI/Kiru-Project/models/DAM-3B"
DATA_ROOT = "/home/saksham/Desktop/GenAI/Kiru-Project/data/flickr30k"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATIONS_FILE = os.path.join(DATA_ROOT, "entities_annotations.json")
OUTPUT_FILE = "/home/saksham/Desktop/GenAI/Kiru-Project/results/flickr30k/shortphraseentities_dam3b_predictions.json"

# Number of samples to process (None for all)
NUM_SAMPLES = None

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Annotations: {ANNOTATIONS_FILE}")
print(f"  Output: {OUTPUT_FILE}")

# Initialize DAM model
print("\nInitializing DAM model...")
try:
    dam_utils.init(
        model_path=MODEL_PATH,
        sep=", ",
        conv_mode="v1",
        query="<image>\nDescribe this image in detail.",  # Standard captioning query
        temperature=0.2,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
        crop_mode="full+focal_crop",  # Same as Ref-L4 for compatibility
        no_concat_images=False,  # Same as Ref-L4
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nMake sure you have:")
    print("  1. Downloaded DAM-3B model to models/DAM-3B")
    print("  2. Installed required packages")
    print("  3. FocusDAM repository at /home/saksham/Desktop/GenAI/FocusDAM")
    sys.exit(1)

# Load annotations
print("\nLoading Flickr30k Entities annotations.")
try:
    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} phrase-bbox annotations")
except Exception as e:
    print(f"Error loading annotations: {e}")
    print("\nRun: python prepare_flickr30k_entities.py --split test")
    sys.exit(1)

# Determine how many to process
if NUM_SAMPLES is None:
    NUM_SAMPLES = len(annotations)
else:
    NUM_SAMPLES = min(NUM_SAMPLES, len(annotations))

# Run inference
print(f"\nRunning inference on {NUM_SAMPLES} phrase-bbox pairs...")
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
        # Bbox format: [x, y, width, height]
        bbox = ann['bbox']
        mask = np.zeros((ann['height'], ann['width']), dtype=np.uint8)
        x, y, w, h = [int(v) for v in bbox]
        mask[y:y+h, x:x+w] = 255
        mask = Image.fromarray(mask)

        # Generate description
        outputs, info = dam_utils.get_description(img, mask)

        # Store prediction
        predictions[ann_id] = outputs

        # Print first few examples
        if i < 3:
            print(f"\n--- Sample {i+1} ---")
            print(f"Image: {ann['file_name']}")
            print(f"Phrase (GT): {ann['phrase'][:80]}...")
            print(f"Prediction: {outputs[:80]}...")
            print(f"BBox: {bbox}")

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

print(f"Saved {len(predictions)} predictions")

if errors:
    print(f"\n {len(errors)} errors occurred:")
    for ann_id, err in errors[:5]:
        print(f"  - {ann_id}: {err}")

    # Save error log
    error_file = OUTPUT_FILE.replace('.json', '_errors.json')
    with open(error_file, 'w') as f:
        json.dump([{'ann_id': ann_id, 'error': err} for ann_id, err in errors], f, indent=2)
    print(f"Error log saved to {error_file}")


print("Inference complete")
print(f"\nNext steps:")
print(f"  1. Run evaluation:")
print(f"     python eval_flickr30k_subset.py \\")
print(f"       --predictions {OUTPUT_FILE} \\")
print(f"       --annotations {DATA_ROOT}/entities_groundtruth.json \\")
print(f"       --output results/flickr30k/entities_evaluation_results.json")
print(f"  2. Compare with Table 3 baseline scores")
