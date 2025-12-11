import os
import sys
import json
from tqdm import tqdm

# add flickr30k_entities utils to path
sys.path.insert(0, 'data/flickr30k/flickr30k_entities')
from flickr30k_entities_utils import get_sentence_data, get_annotations

def prepare_flickr30k_entities(
    entities_dir='data/flickr30k/flickr30k_entities',
    images_dir='data/flickr30k/images',
    output_file='data/flickr30k/entities_annotations.json',
    split='test',
    max_samples=None
):
    """
    Extract phrase-bbox pairs from Flickr30k Entities

    Args:
        entities_dir: path to flickr30k_entities repo
        images_dir: path to downloaded images
        output_file: where to save processed annotations
        split: 'test', 'val', or 'train'
        max_samples: max number of images to process (None for all)
    """

    # Load split file
    split_file = os.path.join(entities_dir, f'{split}.txt')
    with open(split_file, 'r') as f:
        image_ids = [line.strip().split('.')[0] for line in f.readlines()]

    print(f"\n{split} split has {len(image_ids)} images")

    if max_samples:
        image_ids = image_ids[:max_samples]
        print(f"Processing first {max_samples} images")

    # Process annotations
    annotations = []
    skipped = 0

    for img_id in tqdm(image_ids, desc="Processing"):
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(images_dir, img_filename)

        # Skip if image not downloaded
        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Load sentence annotations
        sentence_file = os.path.join(entities_dir, 'Sentences', f'{img_id}.txt')
        if not os.path.exists(sentence_file):
            continue

        sentences_data = get_sentence_data(sentence_file)

        # Load bbox annotations
        bbox_file = os.path.join(entities_dir, 'Annotations', f'{img_id}.xml')
        if not os.path.exists(bbox_file):
            continue

        bbox_data = get_annotations(bbox_file)

        # Extract phrase-bbox pairs
        for sent_idx, sent_data in enumerate(sentences_data):
            for phrase_info in sent_data['phrases']:
                phrase_id = phrase_info['phrase_id']
                phrase_text = phrase_info['phrase']

                # Get bounding boxes for this phrase
                if phrase_id in bbox_data['boxes']:
                    bboxes = bbox_data['boxes'][phrase_id]

                    # Use first bbox (can have multiple)
                    if bboxes:
                        bbox = bboxes[0]  # [xmin, ymin, xmax, ymax]

                        # Convert to [x, y, width, height] format (like Ref-L4)
                        x, y, xmax, ymax = bbox
                        w = xmax - x
                        h = ymax - y

                        # Create annotation entry
                        ann_id = f"{img_id}_{sent_idx}_{phrase_id}"
                        annotations.append({
                            'ann_id': ann_id,
                            'image_id': img_id,
                            'file_name': img_filename,
                            'sentence_id': sent_idx,
                            'phrase_id': phrase_id,
                            'phrase': phrase_text,
                            'bbox': [x, y, w, h],
                            'bbox_format': 'xywh',
                            'full_sentence': sent_data['sentence'],
                            'width': bbox_data['width'],
                            'height': bbox_data['height']
                        })

    print(f"\n Processed {len(image_ids)} images")
    print(f" Extracted {len(annotations)} phrase-bbox pairs")
    if skipped > 0:
        print(f" Skipped {skipped} images")

    # Save annotations
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"\n Saved annotations to: {output_file}")

    # Also save ground truth in simple format for evaluation
    gt_file = output_file.replace('annotations', 'groundtruth')
    gt_dict = {ann['ann_id']: ann['phrase'] for ann in annotations}
    with open(gt_file, 'w') as f:
        json.dump(gt_dict, f, indent=2)

    print(f"Saved ground truth to: {gt_file}")

    # Print statistics
    print("Dataset Statistics")
    print(f"Total phrase-bbox pairs: {len(annotations)}")
    print(f"Unique images: {len(set(ann['image_id'] for ann in annotations))}")
    print(f"Avg phrases per image: {len(annotations) / len(set(ann['image_id'] for ann in annotations)):.1f}")

    # Sample annotations
    print("Sample Annotations")
    for i, ann in enumerate(annotations[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Image: {ann['file_name']}")
        print(f"  Phrase: {ann['phrase']}")
        print(f"  BBox: {ann['bbox']}")
        print(f"  Full sentence: {ann['full_sentence']}")

    return annotations


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare Flickr30k Entities for evaluation'
    )
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'],
                        help='Dataset split to prepare')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max images to process (None for all available)')

    args = parser.parse_args()

    prepare_flickr30k_entities(
        split=args.split,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
