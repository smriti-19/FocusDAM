# Convert Ref-L4 dataset to COCO format for DAM's get_model_outputs.py script

import json
import os
from datasets import load_from_disk
from tqdm import tqdm

print("Loading Ref-L4 dataset...")
dataset = load_from_disk('data/ref-l4/dataset')
val_data = dataset['val']

print(f"Processing {len(val_data)} validation samples...")

# COCO format structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Track unique images and create image entries
image_dict = {}
for sample in tqdm(val_data):
    img_id = sample['image_id']

    if img_id not in image_dict:
        image_dict[img_id] = {
            "id": img_id,
            "file_name": sample['file_name'],
            "height": sample['height'],
            "width": sample['width']
        }

# Add images to COCO format
coco_format["images"] = list(image_dict.values())

# Add annotations
for sample in tqdm(val_data):
    # Convert bbox from [x, y, width, height] to COCO format
    x, y, w, h = sample['bbox']

    annotation = {
        "id": sample['id'],
        "image_id": sample['image_id'],
        "category_id": sample.get('ori_category_id', 'unknown'),
        "bbox": [x, y, w, h],
        "area": sample.get('bbox_area', w * h),
        "segmentation": [],  # Ref-L4 doesn't have segmentation, will be generated from bbox
        "iscrowd": 0,
        "caption": sample['caption']  # Custom field for ground truth
    }

    coco_format["annotations"].append(annotation)

# Add dummy category (not really used for captioning)
coco_format["categories"] = [{"id": "default", "name": "object"}]

# Save to file
output_file = 'data/ref-l4/annotations.json'
print(f"\nSaving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(coco_format, f, indent=2)

print(f"✓ Saved {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")

# Also save ground truth in evaluation format
print("\nCreating ground truth reference file for evaluation...")
ground_truth = {}
for sample in val_data:
    img_id = sample['image_id']
    if img_id not in ground_truth:
        ground_truth[img_id] = []
    ground_truth[img_id].append(sample['caption'])

gt_file = 'data/ref-l4/ground_truth.json'
with open(gt_file, 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f" Saved ground truth to {gt_file}")
print("\nDataset preparation complete!")
print(f"  Images: {len(coco_format['images'])}")
print(f"  Annotations: {len(coco_format['annotations'])}")
print(f"  Location: data/ref-l4/")