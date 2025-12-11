"""
Extract DAM decoder hidden states for RAAH training

This script runs DAM inference and saves the final decoder hidden states
for each sample in the small objects dataset.
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def extract_hidden_states_dam(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    bbox: list,
    device: str = 'cuda'
):
    """
    Run DAM inference and extract decoder hidden states

    Args:
        model: DAM model
        processor: DAM processor
        tokenizer: DAM tokenizer
        image: PIL Image
        bbox: Bounding box [x, y, width, height]
        device: Device to run on

    Returns:
        hidden_states: [seq_len, hidden_dim] tensor
        caption: Generated caption string
    """
    # Prepare input
    prompt = "Describe this region in detail."

    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    # Run inference with output_hidden_states=True
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # Extract caption
    caption = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Extract hidden states
    if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
        # Stack hidden states from last layer across all generation steps
        # This is model-specific - adjust as needed
        last_layer_states = []
        for step_hidden_states in outputs.decoder_hidden_states:
            # step_hidden_states is a tuple of (num_layers,) tensors
            # Each tensor is [batch_size, seq_len, hidden_dim]
            last_layer = step_hidden_states[-1]  # Last layer
            last_layer_states.append(last_layer[0, -1, :])  # Last token, first batch item

        # Stack all tokens: [num_tokens, hidden_dim]
        hidden_states = torch.stack(last_layer_states, dim=0)

    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        # Alternative: use encoder-decoder hidden states
        # Adjust based on actual DAM output structure
        last_layer_states = []
        for step_hidden_states in outputs.hidden_states:
            last_layer = step_hidden_states[-1]
            last_layer_states.append(last_layer[0, -1, :])

        hidden_states = torch.stack(last_layer_states, dim=0)

    else:
        raise ValueError(
            "Could not extract hidden states from model output. "
            "Make sure the model supports output_hidden_states=True"
        )

    return hidden_states.cpu(), caption


def main():
    parser = argparse.ArgumentParser(description="Extract DAM hidden states for RAAH training")
    parser.add_argument('--dataset_json', type=str, default='data/small_objects_dataset.json')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images (e.g., data/ref-l4/)')
    parser.add_argument('--model_path', type=str, default='models/DAM-3B',
                        help='Path to DAM model')
    parser.add_argument('--output_dir', type=str, default='data/raah_hidden_states',
                        help='Directory to save hidden states')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process in batches (not implemented yet)')

    args = parser.parse_args()

    print("=" * 80)
    print("Extracting DAM Hidden States for RAAH Training")
    print("=" * 80)
    print(f"\nDataset: {args.dataset_json}")
    print(f"Images directory: {args.images_dir}")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    with open(args.dataset_json, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Found {len(samples)} samples\n")

    # Load DAM model
    print("Loading DAM model...")
    print("This may take a few minutes...\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32
        ).to(args.device)
        model.eval()
        print("Model loaded successfully!\n")

    except Exception as e:
        print(f"Error loading model: {e}\n")
        return

    # Extract hidden states
    print("Extracting hidden states...")
    metadata = []

    for sample in tqdm(samples, desc="Processing samples"):
        ann_id = sample['ann_id']
        image_id = sample['image_id']
        file_name = sample['file_name']
        bbox = sample['bbox']

        # Load image
        image_path = Path(args.images_dir) / file_name

        if not image_path.exists():
            print(f"\nWarning: Image not found: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert('RGB')

            # Extract hidden states
            hidden_states, caption = extract_hidden_states_dam(
                model, processor, tokenizer, image, bbox, args.device
            )

            # Save hidden states
            output_path = output_dir / f"hidden_{ann_id}.pt"
            torch.save(hidden_states, output_path)

            # Record metadata
            metadata.append({
                'ann_id': ann_id,
                'image_id': image_id,
                'file_name': file_name,
                'hidden_states_path': str(output_path),
                'hidden_states_shape': list(hidden_states.shape),
                'generated_caption': caption
            })

        except Exception as e:
            print(f"\nError processing sample {ann_id}: {e}")
            continue

    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Extraction complete!")


if __name__ == "__main__":
    main()
