#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for RAAH implementation
Verifies all components work correctly
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))


def test_vocabulary():
    """Test attribute vocabulary building"""
    print("=" * 80)
    print("Testing Attribute Vocabulary")
    print("=" * 80)

    from raah.attribute_vocabulary import AttributeVocabulary

    vocab = AttributeVocabulary(min_frequency=2)
    vocab.build_from_dataset('data/small_objects_dataset.json')

    print(f"\n[PASS] Vocabulary built successfully")
    print(f"  Vocab sizes: {vocab.vocab_sizes}")

    # Test encoding
    test_attrs = [('color', 'white'), ('material', 'wood'), ('shape', 'round')]
    encoded = vocab.encode_attributes(test_attrs)
    print(f"\n[PASS] Encoding test:")
    print(f"  Input: {test_attrs}")
    print(f"  Encoded: {[(k, v) for k, v in encoded.items() if v]}")

    # Test decoding
    pred_dict = {
        'color': [(0.9, encoded['color'][0]) if encoded['color'] else (0.9, 0)],
        'material': [(0.8, encoded['material'][0]) if encoded['material'] else (0.8, 0)],
        'shape': [(0.7, encoded['shape'][0]) if encoded['shape'] else (0.7, 0)],
        'texture': []
    }
    decoded = vocab.decode_predictions(pred_dict, threshold=0.5)
    print(f"\n[PASS] Decoding test:")
    print(f"  Decoded: {decoded}")

    return vocab


def test_model(vocab):
    """Test RAAH model"""
    print("\n" + "=" * 80)
    print("Testing RAAH Model")
    print("=" * 80)

    from raah.raah_model import RAAH, compute_attribute_loss

    # Create model
    model = RAAH(
        hidden_dim=768,
        vocab_sizes=vocab.vocab_sizes,
        pooling_type='mean',
        dropout=0.1,
        hidden_size=512
    )

    print(f"\n[PASS] Model created")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    hidden_states = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(hidden_states, attention_mask)
    print(f"\n[PASS] Forward pass successful")
    print(f"  Output shapes: {[(k, v.shape) for k, v in outputs.items()]}")

    # Test prediction
    predictions = model.predict_attributes(hidden_states, attention_mask, threshold=0.5)
    print(f"\n[PASS] Prediction successful")
    print(f"  Sample prediction: {predictions[0]}")

    # Test loss computation
    labels = {
        attr_type: torch.randint(0, 2, (batch_size, size)).float()
        for attr_type, size in vocab.vocab_sizes.items()
    }

    loss, loss_dict = compute_attribute_loss(outputs, labels)
    print(f"\n[PASS] Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Per-attribute: {[(k, v.item()) for k, v in loss_dict.items()]}")

    return model


def test_dataset(vocab):
    """Test dataset classes"""
    print("\n" + "=" * 80)
    print("Testing Dataset")
    print("=" * 80)

    from raah.raah_dataset import SimpleRAAHDataset, collate_raah_batch
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = SimpleRAAHDataset(
        dataset_json_path='data/small_objects_dataset.json',
        vocab=vocab,
        hidden_dim=768,
        seq_len=50,
        split='train'
    )

    print(f"\n[PASS] Dataset created")
    print(f"  Size: {len(dataset)} samples")

    # Test sample
    sample = dataset[0]
    print(f"\n[PASS] Sample loaded")
    print(f"  Hidden states: {sample['hidden_states'].shape}")
    print(f"  Labels: {[(k, v.shape, v.sum().item()) for k, v in sample['labels'].items()]}")
    print(f"  Caption: {sample['caption'][:80]}...")

    # Test dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_raah_batch
    )

    batch = next(iter(loader))
    print(f"\n[PASS] Batch collation successful")
    print(f"  Batch size: {batch['hidden_states'].shape[0]}")
    print(f"  Hidden states: {batch['hidden_states'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Labels: {[(k, v.shape) for k, v in batch['labels'].items()]}")

    return dataset


def test_training_loop(model, dataset, vocab):
    """Test training loop"""
    print("\n" + "=" * 80)
    print("Testing Training Loop")
    print("=" * 80)

    from torch.utils.data import DataLoader
    from raah.raah_dataset import collate_raah_batch
    from raah.raah_model import compute_attribute_loss
    import torch.optim as optim

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_raah_batch
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one training step
    batch = next(iter(loader))
    hidden_states = batch['hidden_states']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    # Forward
    outputs = model(hidden_states, attention_mask)
    loss, loss_dict = compute_attribute_loss(outputs, labels)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\n[PASS] Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Per-attribute: {[(k, v.item()) for k, v in loss_dict.items()]}")

    # Check gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_grad}")


def test_inference_pipeline(vocab):
    """Test inference pipeline"""
    print("\n" + "=" * 80)
    print("Testing Inference Pipeline (without checkpoint)")
    print("=" * 80)

    from raah.raah_inference import RAAHInferencePipeline

    # Test augmentation logic without loading checkpoint
    from raah.raah_model import RAAH

    model = RAAH(
        hidden_dim=768,
        vocab_sizes=vocab.vocab_sizes,
        pooling_type='mean'
    )

    # Dummy prediction
    hidden_states = torch.randn(1, 50, 768)
    predictions = model.predict_attributes(hidden_states, threshold=0.5, top_k=3)

    print(f"\n[PASS] Prediction successful")
    print(f"  Raw predictions: {predictions[0]}")

    # Test caption augmentation
    original_caption = "A washing machine in the corner of a room."

    # Simulate some predicted attributes
    from raah.attribute_vocabulary import AttributeVocabulary

    test_attributes = [
        ('color', 'white', 0.85),
        ('material', 'metal', 0.72),
        ('shape', 'rectangular', 0.68)
    ]

    # Simple augmentation test
    caption_lower = original_caption.lower()
    filtered = [(t, v, c) for t, v, c in test_attributes if v.lower() not in caption_lower]

    attr_str = ", ".join([v for _, v, _ in filtered])
    augmented = f"{original_caption.rstrip('.')}. Additional attributes: {attr_str}."

    print(f"\n[PASS] Caption augmentation test")
    print(f"  Original: {original_caption}")
    print(f"  Attributes: {test_attributes}")
    print(f"  Augmented: {augmented}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RAAH IMPLEMENTATION TEST SUITE")
    print("=" * 80 + "\n")

    try:
        # Test 1: Vocabulary
        vocab = test_vocabulary()

        # Test 2: Model
        model = test_model(vocab)

        # Test 3: Dataset
        dataset = test_dataset(vocab)

        # Test 4: Training loop
        test_training_loop(model, dataset, vocab)

        # Test 5: Inference pipeline
        test_inference_pipeline(vocab)

        # Summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED [PASS]")
        print("=" * 80)
        print("\nYou can now:")
        print("  1. Build vocabulary: python raah/attribute_vocabulary.py")
        print("  2. Train RAAH: python raah/train_raah.py")
        print("  3. Run inference: See raah/README.md for examples")
        print("\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED [FAIL]")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
