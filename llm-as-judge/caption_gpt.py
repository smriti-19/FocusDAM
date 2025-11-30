#!/usr/bin/env python3
"""
Generate captions for DLC-Bench using vision models (GPT/Claude/Qwen/Gemini).
"""

import os
import json
import argparse
import base64
import time
import requests
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env.local')

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_masked_visualization(image: Image.Image, mask: Image.Image,
                                visualization_mode: str = "highlight") -> Image.Image:
    """
    Create visualization of masked region.

    Args:
        image: Original RGB image
        mask: Binary mask (0 or 255)
        visualization_mode:
            - "highlight": Show full image with masked region highlighted
            - "overlay": Show mask overlay on image
            - "side_by_side": Show image and mask side by side
            - "masked_only": Show only the masked region (rest blacked out)
    """
    img_np = np.array(image.convert('RGB'))
    mask_np = np.array(mask.convert('L'))

    # Normalize mask to 0-1
    mask_binary = (mask_np > 127).astype(np.uint8)

    if visualization_mode == "highlight":
        result = img_np.copy().astype(np.float32)
        result[mask_binary == 0] = result[mask_binary == 0] * 0.3

        from scipy import ndimage
        border = ndimage.binary_dilation(mask_binary, iterations=3) & ~mask_binary.astype(bool)
        result[border] = [255, 0, 0]

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    elif visualization_mode == "overlay":
        result = img_np.copy()
        overlay = np.zeros_like(img_np)
        overlay[mask_binary == 1] = [255, 0, 0]
        result = (result * 0.7 + overlay * 0.3).astype(np.uint8)
        return Image.fromarray(result)

    elif visualization_mode == "masked_only":
        result = img_np.copy()
        result[mask_binary == 0] = [0, 0, 0]
        return Image.fromarray(result)

    elif visualization_mode == "side_by_side":
        mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
        result = np.concatenate([img_np, mask_rgb], axis=1)
        return Image.fromarray(result)

    else:
        raise ValueError(f"Unknown visualization_mode: {visualization_mode}")

def generate_caption_gpt(client: OpenAI, image: Image.Image, mask: Image.Image,
                        model: str, prompt_template: str, visualization_mode: str,
                        temperature: float, max_tokens: int) -> str:
    """Generate caption using GPT Vision API or compatible APIs."""
    vis_image = create_masked_visualization(image, mask, visualization_mode)
    image_b64 = encode_image_to_base64(vis_image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_template
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()

def generate_caption_claude(api_key: str, image: Image.Image, mask: Image.Image,
                           model: str, prompt_template: str, visualization_mode: str,
                           temperature: float, max_tokens: int) -> str:
    """Generate caption using Anthropic Claude REST API."""
    vis_image = create_masked_visualization(image, mask, visualization_mode)
    image_b64 = encode_image_to_base64(vis_image)

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_template
                    }
                ]
            }
        ]
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "content" in result and len(result["content"]) > 0:
                for content_block in result["content"]:
                    if content_block.get("type") == "text":
                        return content_block["text"].strip()

            raise ValueError(f"Unexpected Claude API response structure: {result}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"Claude API rate limit exceeded after {max_retries} retries")
            else:
                raise RuntimeError(f"Claude API HTTP error: {e.response.status_code} - {e.response.text}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Claude API request failed: {str(e)}")

def generate_caption_gemini(api_key: str, image: Image.Image, mask: Image.Image,
                           model: str, prompt_template: str, visualization_mode: str,
                           temperature: float, max_tokens: int) -> str:
    """Generate caption using Google Gemini REST API."""
    vis_image = create_masked_visualization(image, mask, visualization_mode)
    image_b64 = encode_image_to_base64(vis_image)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64
                    }
                },
                {"text": prompt_template}
            ]
        }],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"].strip()

            raise ValueError(f"Unexpected Gemini API response structure: {result}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"Gemini API rate limit exceeded after {max_retries} retries")
            else:
                raise RuntimeError(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Gemini API request failed: {str(e)}")

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

# Prompt templates
PROMPT_TEMPLATES = {
    "default": """Describe the highlighted/masked region in the image in detail.
Focus ONLY on the object(s) within the highlighted area, not the entire image.

Provide a detailed description including:
- The type of object
- Its color(s)
- Its shape and size
- Its material/texture
- Any distinctive features or patterns
- Its orientation or pose

Be specific and accurate. Do not describe objects outside the highlighted region.""",

    "dam_style": """Describe the masked region in detail.

The masked/highlighted area shows a specific object or region. Describe ONLY what is visible in that highlighted area.

Include details about:
- What the object is
- Its appearance (colors, shapes, textures)
- Its structure and composition
- Any notable features

Be concise but thorough.""",

    "simple": """Describe the object in the highlighted/marked region in detail. Focus only on what's inside the highlighted area.""",

    "very_detailed": """You are looking at an image where a specific region has been highlighted or masked.

Your task: Provide an extremely detailed description of ONLY the object(s) visible in the highlighted/masked region. Ignore everything else in the image.

Include:
1. Object identification (what is it?)
2. Color(s) - be specific about shades
3. Shape and geometric properties
4. Material and texture
5. Size relative to the visible area
6. Any text, patterns, or markings
7. Orientation and position
8. Any distinctive or notable features

Be precise and observational. Describe what you see, not what you infer."""
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate captions for DLC-Bench using vision models (GPT/Qwen/Gemini/Claude)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate captions using GPT-4o
  python caption_gpt.py --model gpt-4o --output gpt4o_captions.json

  # Generate with GPT-4o-mini (cheaper)
  python caption_gpt.py --model gpt-4o-mini --output gpt4o_mini_captions.json

  # Generate with Claude Sonnet 4.5
  python caption_gpt.py --model claude-sonnet-4-5-20250929 --output claude_sonnet_4_5_captions.json

  # Generate with Qwen3-VL-Plus (Alibaba DashScope)
  python caption_gpt.py --model qwen3-vl-plus --output qwen3_vl_plus_captions.json

  # Generate with Gemini 2.5 Flash (Google)
  python caption_gpt.py --model gemini-2.5-flash --output gemini_2_5_flash_captions.json

  # Use different prompt template
  python caption_gpt.py --model gpt-4o --prompt-template very_detailed

  # Use different visualization
  python caption_gpt.py --model gpt-4o --visualization masked_only

Available prompt templates: default, dam_style, simple, very_detailed
Available visualizations: highlight, overlay, masked_only, side_by_side

Supported models:
  - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo (requires OPENAI_API_KEY)
  - Anthropic: claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022 (requires ANTHROPIC_API_KEY)
  - Alibaba: qwen3-vl-plus, qwen-vl-max (requires ALIBABA_API_KEY)
  - Google: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash (requires GEMINI_API_KEY)
        """)

    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Vision model (gpt-4o, claude-sonnet-4-5-20250929, qwen3-vl-plus, gemini-2.5-flash, etc.)')
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
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--max-tokens', type=int, default=300,
                       help='Maximum tokens to generate')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key file (overrides .env.local)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of annotations to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print generated captions')

    args = parser.parse_args()

    # Determine which API to use based on model
    is_claude_model = args.model.startswith('claude')
    is_gemini_model = args.model.startswith('gemini')
    is_alibaba_model = args.model.startswith('qwen')

    if is_claude_model:
        # Load Anthropic API key for Claude models
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if args.api_key:
            with open(args.api_key) as f:
                api_key = f.read().strip()

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env.local or --api-key not provided")

        print(f"Using Anthropic Claude REST API")
        print(f"API key loaded (ends with: ...{api_key[-4:]})")

        # Store API key (will be used directly in REST calls)
        claude_api_key = api_key
        client = None  # No client object needed for REST API

    elif is_gemini_model:
        # Load Google API key for Gemini models
        api_key = os.getenv('GEMINI_API_KEY')
        if args.api_key:
            with open(args.api_key) as f:
                api_key = f.read().strip()

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env.local or --api-key not provided")

        print(f"Using Google Gemini REST API")
        print(f"API key loaded (ends with: ...{api_key[-4:]})")

        # Store API key (will be used directly in REST calls)
        gemini_api_key = api_key
        client = None  # No client object needed for REST API

    elif is_alibaba_model:
        # Load Alibaba API key for Qwen models
        api_key = os.getenv('ALIBABA_API_KEY')
        if args.api_key:
            with open(args.api_key) as f:
                api_key = f.read().strip()

        if not api_key:
            raise ValueError("ALIBABA_API_KEY not found in .env.local or --api-key not provided")

        print(f"Using Alibaba DashScope API")
        print(f"API key loaded (ends with: ...{api_key[-4:]})")

        # Initialize OpenAI client with Alibaba DashScope base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    else:
        # Load OpenAI API key for GPT models
        api_key = os.getenv('OPENAI_API_KEY')
        if args.api_key:
            with open(args.api_key) as f:
                api_key = f.read().strip()

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env.local or --api-key not provided")

        print(f"Using OpenAI API")
        print(f"API key loaded (ends with: ...{api_key[-4:]})")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

    # Load DLC-Bench data
    print(f"Loading DLC-Bench data from {args.data_root}")
    try:
        import scipy.ndimage
    except ImportError:
        print("Warning: scipy not installed. Installing for mask visualization...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])
        import scipy.ndimage

    coco, class_names = load_dlc_bench_data(args.data_root)

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

            # Generate caption using appropriate function based on model
            if is_claude_model:
                caption = generate_caption_claude(
                    api_key=claude_api_key,
                    image=image,
                    mask=mask,
                    model=args.model,
                    prompt_template=prompt_template,
                    visualization_mode=args.visualization,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
            elif is_gemini_model:
                caption = generate_caption_gemini(
                    api_key=gemini_api_key,
                    image=image,
                    mask=mask,
                    model=args.model,
                    prompt_template=prompt_template,
                    visualization_mode=args.visualization,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
            else:
                # Use GPT-compatible function (works for OpenAI and Alibaba)
                caption = generate_caption_gpt(
                    client=client,
                    image=image,
                    mask=mask,
                    model=args.model,
                    prompt_template=prompt_template,
                    visualization_mode=args.visualization,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
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

    print("\n" + "="*80)
    print("CAPTION GENERATION COMPLETE")
    print("="*80)
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total captions: {len(captions)}")
    print(f"Output saved to: {args.output}")
    print("="*80)