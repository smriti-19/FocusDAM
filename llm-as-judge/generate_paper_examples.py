#!/usr/bin/env python3
"""
Generate random examples from DLC-Bench for paper figures.
Creates visualizations showing: original image, mask, Q&A pairs, and metadata.
"""

import os
import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_mask_from_annotation(coco, ann_id):
    """Extract binary mask from COCO annotation."""
    ann = coco.anns[ann_id]
    mask = coco.annToMask(ann)
    return mask

def create_mask_overlay(image, mask, alpha=0.5):
    """Create an overlay visualization of the mask on the image."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Create colored mask (red)
    img_array = np.array(image)
    overlay = img_array.copy()

    # Apply red color to masked region
    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array([255, 0, 0]) * alpha

    return Image.fromarray(overlay.astype(np.uint8))

def get_questions_for_annotation(qa_data, ann_id):
    """Get all Q&A pairs for a specific annotation."""
    ann_id_str = str(ann_id)
    if ann_id_str in qa_data:
        return qa_data[ann_id_str]
    return []

def visualize_example(image_path, mask, questions, ann_info, class_name, output_path):
    """
    Create a comprehensive visualization showing:
    - Original image
    - Masked overlay
    - Q&A pairs
    - Metadata
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Create mask overlay
    mask_overlay = create_mask_overlay(image, mask)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Mask overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask_overlay)
    ax2.set_title('Masked Region (Red)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Metadata
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    metadata_text = f"""
Image Metadata:
━━━━━━━━━━━━━━━━━━━━━━━━
Image ID: {ann_info['image_id']}
Annotation ID: {ann_info['id']}
Object Class: {class_name}
Image Size: {image.size[0]} × {image.size[1]} px
Bbox: {ann_info['bbox']}
Area: {ann_info['area']:.0f} px²
    """.strip()

    ax3.text(0.05, 0.95, metadata_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Q&A pairs
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Select one question from each type if available
    qa_types = {'recognition': None, 'positive': None, 'negative': None}
    for q in questions:
        q_type = q['type']
        if qa_types[q_type] is None:
            qa_types[q_type] = q

    qa_text = "Sample Q&A Pairs:\n" + "━" * 50 + "\n\n"

    for q_type, qa in qa_types.items():
        if qa:
            qa_text += f"[{q_type.upper()}]\n"
            qa_text += f"Q: {qa['question']}\n"
            qa_text += f"Options:\n"

            # Find correct answer (choice with score = 1)
            correct_idx = None
            for i, choice in enumerate(qa['choices'], 1):
                option_text, score = choice
                marker = "✓" if score == 1 else " "
                if score == 1:
                    correct_idx = i
                    correct_text = option_text
                qa_text += f"  {marker} {i}. {option_text}\n"

            if correct_idx:
                qa_text += f"Answer: {correct_idx}. {correct_text}\n\n"
            else:
                qa_text += "\n"

    ax4.text(0.05, 0.95, qa_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Main title
    fig.suptitle(f'DLC-Bench Example: {class_name} (Annotation {ann_info["id"]})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization: {output_path}")

def print_example_info(ann_id, class_name, image_id, questions, dam_caption=None):
    """Print example information in a formatted way."""
    print(f"Example: Annotation {ann_id}")
    print(f"Image ID: {image_id}")
    print(f"Object Class: {class_name}")
    print(f"Total Questions: {len(questions)}")

    if dam_caption:
        print(f"DAM Description:")
        print(f"  {dam_caption}")

    print()

    # Group by question type
    by_type = {'recognition': [], 'positive': [], 'negative': []}
    for q in questions:
        by_type[q['type']].append(q)

    for q_type, qs in by_type.items():
        if qs:
            print(f"\n[{q_type.upper()}] Questions: {len(qs)}")
            # Show first question as example
            q = qs[0]
            print(f"  Q: {q['question']}")
            print(f"  Options:")

            # Find correct answer (choice with score = 1)
            correct_idx = None
            for i, choice in enumerate(q['choices'], 1):
                option_text, score = choice
                marker = "→" if score == 1 else " "
                if score == 1:
                    correct_idx = i
                print(f"    {marker} {i}. {option_text}")

            if correct_idx:
                print(f"  Correct Answer: {correct_idx}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate random examples from DLC-Bench for paper figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    parser.add_argument('--data-root', type=str, default='DLC-Bench',
                       help='Path to DLC-Bench dataset directory')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of random examples to generate')
    parser.add_argument('--annotation-ids', type=int, nargs='+',
                       help='Specific annotation IDs to visualize (overrides --num-examples)')
    parser.add_argument('--output-dir', type=str, default='paper_examples',
                       help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-json', action='store_true',
                       help='Save example data as JSON')
    parser.add_argument('--dam-captions', type=str, default=None,
                       help='Path to DAM captions JSON file (e.g., model_outputs_cache/dam_local_captions.json)')

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load DLC-Bench data
    print(f"Loading DLC-Bench data from {args.data_root}...")
    ann_file = os.path.join(args.data_root, 'annotations.json')
    class_file = os.path.join(args.data_root, 'class_names.json')
    qa_file = os.path.join(args.data_root, 'qa.json')

    coco = COCO(ann_file)

    with open(class_file) as f:
        class_names = json.load(f)

    with open(qa_file) as f:
        qa_data = json.load(f)

    # Load DAM captions if provided
    dam_captions = None
    if args.dam_captions:
        if os.path.exists(args.dam_captions):
            print(f"Loading DAM captions from {args.dam_captions}...")
            with open(args.dam_captions) as f:
                dam_captions = json.load(f)
            print(f"Loaded {len(dam_captions)} DAM captions")
        else:
            print(f"Warning: DAM captions file not found: {args.dam_captions}")

    # Get annotation IDs
    if args.annotation_ids:
        ann_ids = args.annotation_ids
        print(f"Using specified annotation IDs: {ann_ids}")
    else:
        all_ann_ids = list(coco.anns.keys())
        ann_ids = random.sample(all_ann_ids, min(args.num_examples, len(all_ann_ids)))
        print(f"Randomly selected {len(ann_ids)} annotations")

    # Process each example
    examples_data = []

    for ann_id in ann_ids:
        # Get annotation info
        ann = coco.anns[ann_id]
        img_id = ann['image_id']
        img_info = coco.loadImgs(img_id)[0]

        # Get class name
        class_name = class_names.get(str(ann_id), 'unknown')

        # Get mask
        mask = get_mask_from_annotation(coco, ann_id)

        # Get questions
        questions = get_questions_for_annotation(qa_data, ann_id)

        # Get DAM caption if available
        dam_caption = None
        if dam_captions:
            dam_caption = dam_captions.get(str(ann_id))

        # Print info
        print_example_info(ann_id, class_name, img_id, questions, dam_caption)

        # Create visualization
        img_path = os.path.join(args.data_root, 'images', img_info['file_name'])
        output_path = output_dir / f'example_ann{ann_id}_{class_name.replace(" ", "_")}.png'

        visualize_example(img_path, mask, questions, ann, class_name, output_path)

        # Save original image
        original_output_path = output_dir / f'original_ann{ann_id}_{class_name.replace(" ", "_")}.jpg'
        image = Image.open(img_path).convert('RGB')
        image.save(original_output_path, quality=95)
        print(f"✓ Saved original image: {original_output_path}")

        # Save data for JSON export
        if args.save_json:
            example_data = {
                'annotation_id': ann_id,
                'image_id': img_id,
                'image_file': img_info['file_name'],
                'class_name': class_name,
                'bbox': ann['bbox'],
                'area': ann['area'],
                'questions': questions
            }
            examples_data.append(example_data)

    # Save JSON if requested
    if args.save_json:
        json_path = output_dir / 'examples_data.json'
        with open(json_path, 'w') as f:
            json.dump(examples_data, f, indent=2)
        print(f"\n✓ Saved example data: {json_path}")

    print(f"\n")
    print(f"Success, examples in: {output_dir.absolute()}")

if __name__ == '__main__':
    main()
