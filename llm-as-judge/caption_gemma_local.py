#!/usr/bin/env python3
"""
Generate captions for DLC-Bench using local Gemma model from HuggingFace.
Compatible with eval_model_outputs_gpt4o.py for evaluation.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login

def create_masked_visualization(image: Image.Image, mask: Image.Image, mode: str = 'highlight') -> Image.Image:
    """
    Create visualization of masked region (same as caption_gpt.py for consistency).

    Args:
        image: Original RGB image
        mask: Binary mask (white=region of interest)
        mode: Visualization mode (highlight, overlay, masked_only, side_by_side)

    Returns:
        Visualized image
    """
    import scipy.ndimage

    # Convert to numpy arrays
    img_array = np.array(image)
    mask_array = np.array(mask.convert('L')) > 128

    if mode == 'highlight':
        # Highlight the masked region with a colored border
        # Dilate mask to create border
        dilated = scipy.ndimage.binary_dilation(mask_array, iterations=5)
        border = dilated & ~mask_array

        # Create output image
        output = img_array.copy()
        output[border] = [255, 0, 0]  # Red border

        return Image.fromarray(output)

    elif mode == 'overlay':
        # Semi-transparent overlay on masked region
        output = img_array.copy().astype(float)
        overlay_color = np.array([255, 255, 0])  # Yellow
        alpha = 0.3

        output[mask_array] = (1 - alpha) * output[mask_array] + alpha * overlay_color

        return Image.fromarray(output.astype(np.uint8))

    elif mode == 'masked_only':
        # Show only the masked region, rest is black
        output = np.zeros_like(img_array)
        output[mask_array] = img_array[mask_array]

        return Image.fromarray(output)

    elif mode == 'side_by_side':
        # Show original and masked version side by side
        masked = np.zeros_like(img_array)
        masked[mask_array] = img_array[mask_array]

        output = np.concatenate([img_array, masked], axis=1)
        return Image.fromarray(output)

    else:
        raise ValueError(f"Unknown visualization_mode: {mode}")

def get_mask_from_annotation(coco, ann_id):
    """Extract binary mask from COCO annotation."""
    ann = coco.anns[ann_id]
    mask = coco.annToMask(ann)
    return Image.fromarray((mask * 255).astype(np.uint8))

def load_dlc_bench_data(data_root: str):
    """Load DLC-Bench annotations and class names."""
    from pycocotools.coco import COCO

    ann_file = os.path.join(data_root, 'annotations.json')
    class_file = os.path.join(data_root, 'class_names.json')

    coco = COCO(ann_file)

    with open(class_file) as f:
        class_names = json.load(f)

    return coco, class_names

def generate_caption_gemma(model, processor, image: Image.Image, mask: Image.Image,
                          prompt_template: str, visualization_mode: str,
                          max_new_tokens: int, device: str) -> str:
    """
    Generate caption using local Gemma model.

    Args:
        model: Gemma model
        processor: Gemma processor
        image: Original image
        mask: Binary mask
        prompt_template: Prompt template to use
        visualization_mode: How to visualize the mask
        max_new_tokens: Maximum new tokens to generate
        device: Device to run on (cuda/cpu)

    Returns:
        Generated caption string
    """
    # Create visualization (same as GPT to ensure consistency)
    vis_image = create_masked_visualization(image, mask, visualization_mode)

    # Gemma expects a specific prompt format with <start_of_image> token
    # Use a simpler completion-style prompt that works better with Gemma
    # Instead of the full template, use a simple descriptive prompt
    gemma_prompt = "<start_of_image> The highlighted object in this image is"

    # Process inputs
    model_inputs = processor(text=gemma_prompt, images=vis_image, return_tensors="pt")

    # Move to device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generation = generation[0][input_len:]

    caption = processor.decode(generation, skip_special_tokens=True)
    caption = caption.strip()
    caption = caption.replace('<end_of_turn>', '').replace('<start_of_turn>', '').strip()

    return caption

# Prompt templates (same as caption_gpt.py for consistency)
PROMPT_TEMPLATES = {
    'dam_style': """Describe the highlighted object in detail.
Focus on:
- What the object is
- Its appearance (colors, textures, materials)
- Its position and orientation
- Any distinguishing features
Be specific and concise.""",

    'default': "Describe the highlighted object in the image in detail.",

    'simple': "What is the highlighted object?",

    'very_detailed': """Provide a comprehensive description of the highlighted object.
Include:
1. Object identification and category
2. Physical attributes (size, shape, color, texture, material)
3. Condition and state
4. Position and orientation in the scene
5. Relationship to other objects
6. Any text, logos, or distinctive markings
7. Notable features or characteristics
Be precise and observational. Describe what you see, not what you infer."""
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate captions for DLC-Bench using local Gemma model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate captions using Gemma 3 4B (normal)
  python caption_gemma_local.py \\
    --model google/gemma-3-4b-pt \\
    --output gemma_3_4b_captions.json

  # Use 4-bit quantization to save VRAM (recommended for 8GB GPUs)
  python caption_gemma_local.py \\
    --model google/gemma-3-4b-pt \\
    --output gemma_3_4b_captions.json \\
    --load-in-4bit

  # Use 8-bit quantization (balanced memory/quality)
  python caption_gemma_local.py \\
    --model google/gemma-3-4b-pt \\
    --output gemma_3_4b_captions.json \\
    --load-in-8bit

  # Quick test (10 samples)
  python caption_gemma_local.py \\
    --model google/gemma-3-4b-pt \\
    --output test_gemma.json \\
    --limit 10 \\
    --verbose

Memory Requirements:
  - 4B model (FP16): ~8GB VRAM
  - 4B model (8-bit): ~4GB VRAM
  - 4B model (4-bit): ~2-3GB VRAM
  - 9B model (FP16): ~18GB VRAM
  - 9B model (4-bit): ~5-6GB VRAM

Supported models:
  Gemma 3 (requires transformers >= 4.48.0):
  - google/gemma-3-4b-pt (4B parameters)
  - google/gemma-3-9b-pt (9B parameters)

  PaliGemma (vision-language, works with older transformers):
  - google/paligemma-3b-pt-224 (3B parameters, 224px)
  - google/paligemma-3b-pt-448 (3B parameters, 448px, better quality)
  - google/paligemma-3b-pt-896 (3B parameters, 896px, best quality)

Note: Requires HuggingFace authentication for Gemma models.
Run: huggingface-cli login

For quantization, install: pip install bitsandbytes
For Gemma 3 support: pip install --upgrade transformers
        """)

    parser.add_argument('--model', type=str, default='google/gemma-3-4b-pt',
                       help='Gemma model from HuggingFace (default: google/gemma-3-4b-pt)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for captions')
    parser.add_argument('--data-root', type=str, default='DLC-Bench',
                       help='Path to DLC-Bench dataset directory')
    parser.add_argument('--prompt-template', type=str, default='dam_style',
                       choices=list(PROMPT_TEMPLATES.keys()),
                       help='Prompt template to use')
    parser.add_argument('--visualization', type=str, default='highlight',
                       choices=['highlight', 'overlay', 'masked_only', 'side_by_side'],
                       help='How to visualize the masked region')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                       help='Maximum new tokens to generate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detected if not specified)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of annotations to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print generated captions')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace token (if not already logged in)')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit quantization (saves VRAM)')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit quantization (saves more VRAM)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (default: 1)')

    args = parser.parse_args()

    # Login to HuggingFace if token provided
    if args.hf_token:
        print("Logging in to HuggingFace...")
        login(token=args.hf_token, new_session=False)

    # Auto-detect device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {args.device}")

    # Set memory optimization environment variables
    if args.device == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Enabled CUDA memory fragmentation optimization")

    # Load Gemma model and processor
    print(f"Loading Gemma model: {args.model}")

    # Prepare model loading arguments
    model_kwargs = {}

    if args.device == 'cuda':
        if args.load_in_4bit:
            print("Loading in 4-bit quantization (requires bitsandbytes)")
            model_kwargs.update({
                'load_in_4bit': True,
                'device_map': 'auto',
                'torch_dtype': torch.float16,
            })
        elif args.load_in_8bit:
            print("Loading in 8-bit quantization (requires bitsandbytes)")
            model_kwargs.update({
                'load_in_8bit': True,
                'device_map': 'auto',
                'torch_dtype': torch.float16,
            })
        else:
            model_kwargs.update({
                'torch_dtype': torch.float16,
                'device_map': 'auto',
                'low_cpu_mem_usage': True,
            })
    else:
        model_kwargs.update({
            'torch_dtype': torch.float32,
            'low_cpu_mem_usage': True,
        })

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            **model_kwargs
        )

        if args.device == 'cpu' and not (args.load_in_4bit or args.load_in_8bit):
            model = model.to(args.device)

        model.eval()

        # Print memory usage
        if args.device == 'cuda':
            print(f"Model loaded successfully!")
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print("Model loaded successfully on CPU!")

    except ValueError as e:
        if "gemma3" in str(e).lower():
            print(f"Error: Gemma 3 models require transformers >= 4.48.0")
            print(f"Your version: {__import__('transformers').__version__}")
            print("\nOptions:")
            print("1. Upgrade transformers: pip install --upgrade transformers")
            print("2. Use Gemma 2 models instead:")
            print("   - google/paligemma-3b-pt-224")
            print("   - google/paligemma-3b-pt-448")
            print("   - google/paligemma-3b-pt-896")
            raise
        else:
            print(f"Error loading model: {e}")
            print("\nMake sure you have:")
            print("1. Accepted Gemma license on HuggingFace")
            print("2. Logged in with: huggingface-cli login")
            if args.load_in_4bit or args.load_in_8bit:
                print("3. Installed bitsandbytes: pip install bitsandbytes")
            raise
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have:")
        print("1. Accepted Gemma license on HuggingFace")
        print("2. Logged in with: huggingface-cli login")
        if args.load_in_4bit or args.load_in_8bit:
            print("3. Installed bitsandbytes: pip install bitsandbytes")
        raise

    # Load DLC-Bench data
    print(f"Loading DLC-Bench data from {args.data_root}")
    try:
        coco, class_names = load_dlc_bench_data(args.data_root)
    except Exception as e:
        print(f"Error loading DLC-Bench data: {e}")
        print("Make sure scipy is installed: pip install scipy")
        raise

    # Get all annotation IDs
    ann_ids = list(coco.anns.keys())
    if args.limit:
        ann_ids = ann_ids[:args.limit]

    print(f"Total annotations to process: {len(ann_ids)}")
    print(f"Model: {args.model}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Visualization mode: {args.visualization}")
    print(f"Output file: {args.output}")

    # Load existing captions if resuming
    captions = {}
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output) as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} existing captions")

    # Get prompt template
    prompt_template = PROMPT_TEMPLATES[args.prompt_template]

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
            caption = generate_caption_gemma(
                model=model,
                processor=processor,
                image=image,
                mask=mask,
                prompt_template=prompt_template,
                visualization_mode=args.visualization,
                max_new_tokens=args.max_new_tokens,
                device=args.device
            )

            # Store caption
            captions[ann_id_str] = caption
            processed += 1

            if args.verbose:
                class_name = class_names.get(ann_id_str, 'unknown')
                print(f"\nAnnotation {ann_id_str} ({class_name}):")
                print(f"  {caption}")

            # Clear CUDA cache periodically to avoid memory fragmentation
            if args.device == 'cuda' and processed % 5 == 0:
                torch.cuda.empty_cache()

            # Save periodically (every 10 images)
            if processed % 10 == 0:
                with open(args.output, 'w') as f:
                    json.dump(captions, f, indent=2)

                if args.device == 'cuda' and args.verbose:
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        except Exception as e:
            print(f"\nError processing annotation {ann_id_str}: {e}")
            errors += 1
            # Clear cache after error
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            continue

    # Final save
    print(f"\nSaving final results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(captions, f, indent=2)

    print(f"\nDone!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total captions: {len(captions)}")
