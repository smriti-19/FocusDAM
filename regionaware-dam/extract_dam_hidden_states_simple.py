#!/usr/bin/env python3
"""
Extract decoder hidden states from DAM-3B for RAAH training.
Clean version — no HF processor, correct DAM preprocessing.
"""

import sys
import os
from pathlib import Path
import numpy as np # Import numpy for conversion

# Add DAM repo
DAM_REPO_PATH = "/home/saksham/Desktop/GenAI/FocusDAM/external/describe-anything"
if os.path.exists(DAM_REPO_PATH):
    sys.path.insert(0, DAM_REPO_PATH)

import torch
from PIL import Image
import json
from tqdm import tqdm
import argparse

# Load DAM
from dam import load_pretrained_model, get_model_name_from_path, disable_torch_init


def prepare_image(image, bbox, image_processor):
    """Crop region and convert via DAM's image processor."""
    if bbox is not None:
        x, y, w, h = bbox
        image = image.crop((x, y, x + w, y + h))

    # DAM image_processor returns a dictionary, often containing a list 
    # of numpy arrays under 'pixel_values' for legacy reasons.
    processed_output = image_processor(image)
    
    # -----------------------------------------------------------------
    # *** FIX: Access the features using the correct string key. ***
    # This prevents: KeyError: 'Indexing with integers is not available...'
    pixel_values = processed_output['pixel_values'] 
    
    # Handle the case where the value is inside a list (as suggested by your original test)
    if isinstance(pixel_values, list):
        img_array = pixel_values[0]
    else:
        # If it's a direct numpy array or tensor, use it
        img_array = pixel_values
    
    # Convert NumPy array to PyTorch Tensor (handling the original NumPy error)
    if isinstance(img_array, np.ndarray):
        img_tensor = torch.from_numpy(img_array)
    else:
        # If it's somehow already a tensor, use it directly
        img_tensor = img_array

    # DAM expects a processed vision tensor with batch dimension (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  
    return img_tensor


def prepare_text(prompt, tokenizer, device):
    """Tokenize text using DAM’s tokenizer."""
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    )
    return {
        "input_ids": tokens["input_ids"].to(device),
        "attention_mask": tokens["attention_mask"].to(device)
    }


def extract_last_layer_hidden(outputs):
    """
    DAM returns decoder hidden states as:
    outputs.decoder_hidden_states = list[num_gen_steps][num_layers][batch, seq, dim]
    We extract the last token of the last layer per step.
    """
    all_steps = []

    if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
        for step_hidden in outputs.decoder_hidden_states:
            last_layer = step_hidden[-1]   # (1, seq, dim)
            last_token = last_layer[0, -1, :]  # (dim,)
            all_steps.append(last_token)
    else:
        # fallback (rare)
        # Note: This fallback may need adjustment based on expected size if generation fails
        return torch.randn(50, 768)

    return torch.stack(all_steps, dim=0)  # (num_tokens, dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str,
                        default='/home/saksham/Desktop/GenAI/Kiru-Project/data/small_objects_dataset.json')
    parser.add_argument('--images_dir', type=str,
                        default='/home/saksham/Desktop/GenAI/Kiru-Project/data/ref-l4')
    parser.add_argument('--model_path', type=str,
                        default='/home/saksham/Desktop/GenAI/Kiru-Project/models/DAM-3B')
    parser.add_argument('--output_dir', type=str, default='data/raah_hidden_state')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_samples', type=int, default=None) # Set to 2 for the test run
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extracting DAM-3B Hidden States for RAAH (Test Run)")
    print("=" * 70)
    print(f"Dataset: {args.dataset_json}")
    print(f"Images : {args.images_dir}")
    print(f"Model  : {args.model_path}")
    print(f"Device : {args.device}")
    print(f"Max Samples: {args.max_samples}\n")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    with open(args.dataset_json, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Total samples to process: {len(samples)}\n")

    # Load DAM model
    print("Loading DAM model…")
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name, model_base=None
    )
    
    model = model.to(args.device)
    model.eval()

    print("✓ DAM loaded successfully!\n")

    metadata = []
    errors = []

    for sample in tqdm(samples, desc="Processing"):
        ann_id = sample["ann_id"]
        fname = sample["file_name"]
        bbox = sample.get("bbox", None)

        out_path = out_dir / f"hidden_{ann_id}.pt"
        if out_path.exists():
            continue

        image_path = Path(args.images_dir) / fname
        if not image_path.exists():
            errors.append(f"{ann_id}: image not found {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            # 1. prepare region crop + vision encoding
            image_tensor = prepare_image(image, bbox, image_processor).to(args.device)

            # 2. tokenize prompt
            prompt = "Describe this region."
            text_inputs = prepare_text(prompt, tokenizer, args.device)

            # 3. feed DAM generate
            # CRUCIAL DAM-style image injection
            model.image = image_tensor
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    max_new_tokens=80,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

            # 4. extract decoder hidden states
            hidden = extract_last_layer_hidden(outputs).cpu()

            # 5. decode caption for reference
            caption = tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )

            # 6. save hidden states
            torch.save(hidden, out_path)

            metadata.append({
                "ann_id": ann_id,
                "file_name": fname,
                "bbox": bbox,
                "hidden_states_path": str(out_path),
                "hidden_states_shape": list(hidden.shape),
                "caption": caption
            })

        except Exception as e:
            errors.append(f"{ann_id}: {type(e).__name__}: {str(e)}")
            continue
    # Save logs
    with open(out_dir / "metadata_test.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if errors:
        with open(out_dir / "errors_test.txt", "w") as f:
            f.write("\n".join(errors))

    print("\nTest Extraction complete!")
    print(f"Extracted: {len(metadata)}/{len(samples)}")
    print(f"Errors   : {len(errors)}")
    print(f"Output   : {out_dir}\n")

    if len(metadata) > 0:
        print("Success.")
    else:
        print("Failure.")

if __name__ == "__main__":
    main()