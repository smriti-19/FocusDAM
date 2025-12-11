#!/usr/bin/env python3
"""
Generate captions for DLC-Bench using local DAM-3B model.
Compatible with eval_model_outputs_gpt4o.py for evaluation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Add the DAM repository to Python path if needed
DAM_REPO_PATH = "/home/saksham/Desktop/GenAI/FocusDAM/external/describe-anything"
if os.path.exists(DAM_REPO_PATH):
    sys.path.insert(0, DAM_REPO_PATH)

import torch

def load_dam_model(model_path: str, device: str = "cuda"):
    """
    Load DAM model using the official DAM utilities.

    Args:
        model_path: Path to DAM model checkpoint
        device: Device to load model on

    Returns:
        Loaded model, tokenizer, and other components
    """
    print(f"Loading DAM model from {model_path}...")

    try:
        # Import DAM utilities
        from dam import load_pretrained_model, get_model_name_from_path, disable_torch_init

        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            model_name,
            model_base=None
        )

        model.config.image_processor = image_processor

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        print(f"✓ Model loaded successfully on {device}")
        return model, tokenizer, image_processor

    except ImportError as e:
        print(f"Error: Could not import DAM utilities: {e}")
        print("\nMake sure the DAM repository is available at:")
        print(f"  {DAM_REPO_PATH}")
        print("\nOr update DAM_REPO_PATH in this script.")
        sys.exit(1)

def load_dlc_bench_data(data_root: str):
    """Load DLC-Bench annotations and class names."""
    from pycocotools.coco import COCO

    coco = COCO(os.path.join(data_root, 'annotations.json'))

    with open(os.path.join(data_root, 'class_names.json')) as f:
        class_names = json.load(f)

    return coco, class_names

def get_mask_from_annotation(coco, ann_id: int) -> Image.Image:
    """Get binary mask from COCO annotation."""
    anns = coco.loadAnns([ann_id])
    mask_np = coco.annToMask(anns[0]) * 255
    return Image.fromarray(mask_np.astype(np.uint8))

def generate_caption_dam(model, tokenizer, image_processor, image: Image.Image,
                        mask: Image.Image, query: str, crop_mode: str,
                        temperature: float, max_new_tokens: int,
                        no_concat_images: bool = False) -> str:
    """
    Generate caption using DAM model.

    Args:
        model: DAM model
        tokenizer: Model tokenizer
        image_processor: Image processor
        image: Original RGB image
        mask: Binary mask
        query: Query prompt
        crop_mode: Cropping mode (e.g., "full+focal_crop")
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        no_concat_images: Whether to concatenate images

    Returns:
        Generated caption string
    """
    try:
        # Import DAM utilities for inference
        from dam import DescribeAnythingModel, process_image, tokenizer_image_token
        from dam import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from dam import SeparatorStyle, conv_templates
        from dam import KeywordsStoppingCriteria

        # Setup conversation
        conv_mode = "v1"  # or "llama_3" depending on your model

        if DEFAULT_IMAGE_TOKEN not in query:
            raise ValueError("Query must contain <image> token")

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process image with mask
        crop_image = DescribeAnythingModel.crop_image
        mask_np = (np.asarray(mask) / 255).astype(np.uint8)

        # Handle crop mode
        if "+" not in crop_mode:
            crop_mode_1, crop_mode_2 = crop_mode, None
        else:
            crop_mode_1, crop_mode_2 = crop_mode.split("+")

        # Process first image
        images_tensor, image_info = process_image(
            image,
            model.config,
            None,
            pil_preprocess_fn=lambda pil_img: crop_image(image, mask_np=mask_np, crop_mode=crop_mode_1)
        )
        images_tensor = images_tensor[None].to(model.device, dtype=torch.float16)

        mask_np_processed = image_info["mask_np"]
        mask_pil = Image.fromarray(mask_np_processed * 255)

        masks_tensor = process_image(mask_pil, model.config, None)
        masks_tensor = masks_tensor[None].to(model.device, dtype=torch.float16)

        images_tensor = torch.cat((images_tensor, masks_tensor[:, :1, ...]), dim=1)

        # Process second image if crop_mode has "+"
        if crop_mode_2 is not None:
            images_tensor2, image_info2 = process_image(
                image,
                model.config,
                None,
                pil_preprocess_fn=lambda pil_img: crop_image(pil_img, mask_np=mask_np, crop_mode=crop_mode_2)
            )
            images_tensor2 = images_tensor2[None].to(model.device, dtype=torch.float16)

            mask_np2 = image_info2["mask_np"]
            mask_pil2 = Image.fromarray(mask_np2 * 255)

            masks_tensor2 = process_image(mask_pil2, model.config, None)
            masks_tensor2 = masks_tensor2[None].to(model.device, dtype=torch.float16)

            images_tensor2 = torch.cat((images_tensor2, masks_tensor2[:, :1, ...]), dim=1)
        else:
            images_tensor2 = None

        # Prepare input IDs
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        # Setup stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Prepare images for generation
        if images_tensor2 is not None:
            if no_concat_images:
                images = [images_tensor, images_tensor2]
            else:
                images = [torch.cat((images_tensor, images_tensor2), dim=1)]
        else:
            images = [images_tensor]

        # Generate caption
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        return outputs

    except Exception as e:
        print(f"Error in caption generation: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate captions for DLC-Bench using local DAM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to DAM model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for captions')
    parser.add_argument('--data-root', type=str, default='DLC-Bench',
                       help='Path to DLC-Bench dataset directory')
    parser.add_argument('--query', type=str, default='<image>\nDescribe the masked region in detail.',
                       help='Query prompt template')
    parser.add_argument('--crop-mode', type=str, default='full+focal_crop',
                       help='Crop mode (full, focal_crop, full+focal_crop)')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--no-concat-images', action='store_true',
                       help='Do not concatenate images in channel dimension')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run model on')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of annotations to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print generated captions')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("\nAvailable models in /home/saksham/Desktop/GenAI/Kiru-Project/models/:")
        models_dir = "/home/saksham/Desktop/GenAI/Kiru-Project/models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                print(f"  - {item}")
        sys.exit(1)

    # Load DAM model
    model, tokenizer, image_processor = load_dam_model(args.model_path, args.device)

    # Load DLC-Bench data
    print(f"Loading DLC-Bench data from {args.data_root}")
    coco, class_names = load_dlc_bench_data(args.data_root)

    # Get all annotation IDs
    ann_ids = list(coco.anns.keys())
    if args.limit:
        ann_ids = ann_ids[:args.limit]

    print(f"Total annotations to process: {len(ann_ids)}")
    print(f"Model: {args.model_path}")
    print(f"Query: {args.query}")
    print(f"Crop mode: {args.crop_mode}")
    print(f"Output file: {args.output}")

    # Load existing captions if resuming
    captions = {}
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output) as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} existing captions")

    # Generate captions
    processed = 0
    skipped = 0
    errors = 0

    for ann_id in tqdm(ann_ids, desc="Generating captions"):
        ann_id_str = str(ann_id)

        # Skip if already processed
        if ann_id_str in captions:
            skipped += 1
            continue

        try:
            # Get annotation info
            ann = coco.anns[ann_id]
            img_id = ann['image_id']
            img_info = coco.loadImgs(img_id)[0]

            # Load image
            img_path = os.path.join(args.data_root, 'images', img_info['file_name'])
            image = Image.open(img_path).convert('RGB')

            # Get mask
            mask = get_mask_from_annotation(coco, ann_id)

            # Generate caption
            caption = generate_caption_dam(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=image,
                mask=mask,
                query=args.query,
                crop_mode=args.crop_mode,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                no_concat_images=args.no_concat_images
            )

            # Store caption
            captions[ann_id_str] = caption
            processed += 1

            if args.verbose:
                class_name = class_names.get(ann_id_str, 'unknown')
                print(f"\nAnnotation {ann_id_str} ({class_name}):")
                print(f"  {caption}")

            # Save periodically (every 10 captions)
            if processed % 10 == 0:
                with open(args.output, 'w') as f:
                    json.dump(captions, f, indent=2)

        except Exception as e:
            print(f"\nError processing annotation {ann_id}: {e}")
            captions[ann_id_str] = f"Error: {str(e)}"
            errors += 1

    # Final save
    with open(args.output, 'w') as f:
        json.dump(captions, f, indent=2)
    print("Success")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total captions: {len(captions)}")
    print(f"Output saved to: {args.output}")
