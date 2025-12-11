# Region-Aware Attribute Head (RAAH)

A lightweight classifier that recovers fine-grained attributes (colors, materials, textures, shapes) from DAM's decoder hidden states to produce richer and more accurate localized descriptions.

## Overview

RAAH addresses the limitation where DAM observes visual details internally but fails to verbalize them in captions. By adding a small MLP classifier on top of DAM's decoder, RAAH can:

- Predict fine-grained attributes that DAM "sees" but doesn't mention
- Augment captions with missing details (e.g., "white", "wooden", "round")
- Improve attribute recall without modifying DAM's architecture

## Installation

```bash
# Install dependencies
pip install torch torchvision transformers Pillow tqdm

cd /home/saksham/Desktop/GenAI/Kiru-Project
```

## Quick Start

### 1. Build Attribute Vocabulary

```bash
python raah/attribute_vocabulary.py
```

This creates `raah/attribute_vocab.json` from the small objects dataset with:
- **Color**: 19 classes (white, red, black, tan, blue, ...)
- **Material**: 12 classes (wood, metal, glass, ...)
- **Shape**: 10 classes (round, rectangular, circular, ...)
- **Texture**: 11 classes (shiny, glossy, striped, ...)

### 2. Train RAAH (Two Options)

#### Option A: With Real DAM Hidden States

First, extract hidden states from DAM:

```bash
python raah/extract_dam_hidden_states.py \
  --dataset_json data/small_objects_dataset.json \
  --images_dir /home/saksham/Desktop/GenAI/Kiru-Project/data/ref-l4
  --model_path models/DAM-3B \
  --output_dir data/raah_hidden_states
```

Then train RAAH:

```bash
python3 raah/train_raah.py \
  --dataset_json data/small_objects_dataset.json \
  --hidden_states_dir data/raah_hidden_state \
  --output_dir raah/checkpoints/test \
  --batch_size 32 \
  --num_epochs 20 \
  --lr 1e-3
```

#### Option B: With Random Hidden States (For Testing)

Skip hidden state extraction and train directly with random features:

```bash
python raah/train_raah.py \
  --dataset_json data/small_objects_dataset.json \
  --output_dir raah/checkpoints \
  --batch_size 32 \
  --num_epochs 20 \
  --lr 1e-3
```

**Training Args:**
- `--hidden_dim`: DAM decoder dimension (default: 768)
- `--hidden_size`: RAAH MLP hidden size (default: 512)
- `--pooling_type`: Region pooling method (`mean`, `max`, `last`, `attention`)
- `--dropout`: Dropout rate (default: 0.1)

### 3. Run Inference

```python
from raah import RAAHInferencePipeline
import torch

# Load pipeline
pipeline = RAAHInferencePipeline(
    raah_checkpoint_path='raah/checkpoints/best_model.pt',
    vocab_path='raah/attribute_vocab.json',
    confidence_threshold=0.5,
    top_k=3
)

# DAM generates caption and hidden states
dam_caption = "A washing machine in the corner."
dam_hidden_states = torch.randn(50, 768)  # [seq_len, hidden_dim]

# Augment with RAAH
augmented_caption = pipeline(dam_hidden_states, dam_caption)
print(augmented_caption)
# → "A washing machine in the corner. Additional attributes: white, metal, rectangular."
```

## Dataset

The training uses `data/small_objects_dataset.json` which contains:
- **2,908 samples** with ground-truth attributes
- Small objects (< 5% image area) from Objects365
- Weak labels automatically extracted from reference captions

**Sample structure:**
```json
{
  "ann_id": 4,
  "image_id": "o365_546894",
  "file_name": "objects365_v1_00546894.jpg",
  "bbox": [135.56, 194.31, 74.94, 114.74],
  "caption": "The white washing machine in the room corner...",
  "attributes_found": [["color", "white"]],
  "area_fraction": 0.024,
  "attribute_score": 1
}
```

## Evaluation Metrics

RAAH tracks per-attribute and overall metrics:

- **Precision**: % of predicted attributes that are correct
- **Recall**: % of ground-truth attributes that are predicted
- **F1 Score**: Harmonic mean of precision and recall

Example output:
```
Epoch 20/20
  Train Loss: 0.0823 | Val Loss: 0.0945
  Train F1: 0.7245 | Val F1: 0.6892

  Per-attribute F1 scores (Val):
    color: 0.7234
    material: 0.6512
    shape: 0.6845
    texture: 0.5876
```

## Integration with DAM Pipeline

To integrate RAAH into your DAM inference pipeline:

```python
# 1. Run DAM with hidden state extraction
dam_outputs = dam_model.generate(
    **inputs,
    max_new_tokens=200,
    output_hidden_states=True,
    return_dict_in_generate=True
)

# 2. Extract hidden states (last layer, all tokens)
hidden_states = extract_decoder_hidden_states(dam_outputs)

# 3. Generate caption
caption = tokenizer.decode(dam_outputs.sequences[0])

# 4. Augment with RAAH
augmented_caption = raah_pipeline(hidden_states, caption)
```

## File Structure

```
raah/
├── __init__.py                      # Package init
├── README.md                        # This file
├── attribute_vocabulary.py          # Vocabulary builder
├── raah_model.py                    # RAAH model architecture
├── raah_dataset.py                  # Dataset classes
├── train_raah.py                    # Training script
├── raah_inference.py                # Inference pipeline
├── extract_dam_hidden_states.py     # DAM hidden state extractor
├── attribute_vocab.json             # Generated vocabulary
└── checkpoints/
    ├── best_model.pt                # Best model checkpoint
    └── training_history.json        # Training logs
```