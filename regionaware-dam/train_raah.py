"""
Training script for Region-Aware Attribute Head (RAAH)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from glob import glob
import argparse
from tqdm import tqdm
import json
import numpy as np
from typing import Dict

from raah.attribute_vocabulary import AttributeVocabulary
from raah.raah_model import RAAH, compute_attribute_loss
from raah.raah_dataset import SimpleRAAHDataset, RAAHDataset, collate_raah_batch


def compute_metrics(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for attribute predictions

    Args:
        logits: Dict of attribute_type -> [batch_size, num_classes]
        labels: Dict of attribute_type -> [batch_size, num_classes]
        threshold: Probability threshold for positive prediction

    Returns:
        Dict with metrics per attribute type and overall
    """
    metrics = {}

    all_tp, all_fp, all_fn = 0, 0, 0

    for attr_type in logits.keys():
        if attr_type not in labels:
            continue

        # Get predictions
        probs = torch.sigmoid(logits[attr_type])
        preds = (probs >= threshold).float()

        # True positives, false positives, false negatives
        tp = ((preds == 1) & (labels[attr_type] == 1)).sum().item()
        fp = ((preds == 1) & (labels[attr_type] == 0)).sum().item()
        fn = ((preds == 0) & (labels[attr_type] == 1)).sum().item()

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'{attr_type}_precision'] = precision
        metrics[f'{attr_type}_recall'] = recall
        metrics[f'{attr_type}_f1'] = f1

        all_tp += tp
        all_fp += fp
        all_fn += fn

    # Overall metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0.0

    metrics['overall_precision'] = overall_precision
    metrics['overall_recall'] = overall_recall
    metrics['overall_f1'] = overall_f1

    return metrics


def train_epoch(
    model: RAAH,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    all_losses = {attr_type: 0.0 for attr_type in model.classifier.attribute_types}

    all_logits = {attr_type: [] for attr_type in model.classifier.attribute_types}
    all_labels = {attr_type: [] for attr_type in model.classifier.attribute_types}

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        hidden_states = batch['hidden_states'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # Forward pass
        logits = model(hidden_states, attention_mask)

        # Compute loss
        loss, loss_dict = compute_attribute_loss(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for attr_type, attr_loss in loss_dict.items():
            all_losses[attr_type] += attr_loss.item()

        # Store predictions for metrics
        for attr_type in logits.keys():
            all_logits[attr_type].append(logits[attr_type].detach())
            all_labels[attr_type].append(labels[attr_type].detach())

        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    avg_losses = {k: v / len(dataloader) for k, v in all_losses.items()}

    # Concatenate all predictions
    all_logits = {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
    all_labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}

    metrics = compute_metrics(all_logits, all_labels)

    return {
        'loss': avg_loss,
        **avg_losses,
        **metrics
    }


@torch.no_grad()
def evaluate(
    model: RAAH,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    total_loss = 0.0
    all_losses = {attr_type: 0.0 for attr_type in model.classifier.attribute_types}

    all_logits = {attr_type: [] for attr_type in model.classifier.attribute_types}
    all_labels = {attr_type: [] for attr_type in model.classifier.attribute_types}

    progress_bar = tqdm(dataloader, desc="Validation")

    for batch in progress_bar:
        # Move to device
        hidden_states = batch['hidden_states'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        # Forward pass
        logits = model(hidden_states, attention_mask)

        # Compute loss
        loss, loss_dict = compute_attribute_loss(logits, labels)

        # Accumulate metrics
        total_loss += loss.item()
        for attr_type, attr_loss in loss_dict.items():
            all_losses[attr_type] += attr_loss.item()

        # Store predictions
        for attr_type in logits.keys():
            all_logits[attr_type].append(logits[attr_type])
            all_labels[attr_type].append(labels[attr_type])

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    avg_losses = {k: v / len(dataloader) for k, v in all_losses.items()}

    all_logits = {k: torch.cat(v, dim=0) for k, v in all_logits.items()}
    all_labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}

    metrics = compute_metrics(all_logits, all_labels)

    return {
        'loss': avg_loss,
        **avg_losses,
        **metrics
    }

def filter_dataset_by_hidden_states(
    dataset_json_path: str,
    hidden_states_dir: str,
    split: str
) -> list:
    """
    Loads the full dataset JSON, checks if the hidden state file exists 
    for each sample, and returns a filtered list of sample metadata.
    """
    # Load full dataset metadata
    with open(dataset_json_path, "r") as f:
        data = json.load(f)
    
    # Filter by split (train/val)
    samples = [s for s in data["samples"] if s["split"] == split]
    
    # Create a set of ann_ids for which hidden states exist
    hidden_states_path = Path(hidden_states_dir)
    
    # Find all existing hidden_*.pt files and extract their ann_id (e.g., 'hidden_4.pt' -> 4)
    existing_ann_ids = set()
    for pt_file in hidden_states_path.glob("hidden_*.pt"):
        try:
            # Assumes file name is 'hidden_{ann_id}.pt'
            ann_id = int(pt_file.stem.split('_')[1])
            existing_ann_ids.add(ann_id)
        except:
            # Skip files that don't match the expected naming convention
            continue

    print(f"Found {len(existing_ann_ids)} hidden state files in {hidden_states_dir}.")

    # Filter the dataset samples to only include those with existing hidden states
    filtered_samples = [s for s in samples if s.get("ann_id") in existing_ann_ids]
    
    missing_count = len(samples) - len(filtered_samples)
    if missing_count > 0:
        print(f"Warning: Filtered out {missing_count} samples from '{split}' split because the .pt file was missing.")
    
    return filtered_samples

def main():
    parser = argparse.ArgumentParser(description="Train RAAH")
    parser.add_argument('--dataset_json', type=str, default='data/small_objects_dataset.json')
    parser.add_argument('--hidden_states_dir', type=str, default=None,
                        help='Directory with pre-computed DAM hidden states (if None, uses random)')
    parser.add_argument('--output_dir', type=str, default='raah/checkpoints')
    parser.add_argument('--vocab_path', type=str, default='raah/attribute_vocab.json')
    parser.add_argument('--hidden_dim', type=int, default=768, help='DAM decoder hidden dim')
    parser.add_argument('--hidden_size', type=int, default=512, help='RAAH hidden layer size')
    parser.add_argument('--pooling_type', type=str, default='mean',
                        choices=['mean', 'max', 'last', 'attention'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training Region-Aware Attribute Head (RAAH)")
    print("=" * 80)
    print(f"\nDevice: {args.device}")
    print(f"Output directory: {args.output_dir}\n")

    # Load or build vocabulary
    if Path(args.vocab_path).exists():
        print(f"Loading vocabulary from {args.vocab_path}")
        vocab = AttributeVocabulary.load(args.vocab_path)
    else:
        print(f"Building vocabulary from {args.dataset_json}")
        vocab = AttributeVocabulary(min_frequency=2)
        vocab.build_from_dataset(args.dataset_json)
        vocab.save(args.vocab_path)

    # Create datasets
    if args.hidden_states_dir is not None:
        print(f"\nUsing pre-computed hidden states from: {args.hidden_states_dir}")
        train_dataset = RAAHDataset(
            hidden_states_dir=args.hidden_states_dir,
            dataset_json_path=args.dataset_json,
            vocab=vocab,
            split='train'
        )
        val_dataset = RAAHDataset(
            hidden_states_dir=args.hidden_states_dir,
            dataset_json_path=args.dataset_json,
            vocab=vocab,
            split='val'
        )
    else:
        print("\nUsing random hidden states (for testing)")
        train_dataset = SimpleRAAHDataset(
            dataset_json_path=args.dataset_json,
            vocab=vocab,
            hidden_dim=args.hidden_dim,
            split='train'
        )
        val_dataset = SimpleRAAHDataset(
            dataset_json_path=args.dataset_json,
            vocab=vocab,
            hidden_dim=args.hidden_dim,
            split='val'
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_raah_batch,
        pin_memory=True if args.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_raah_batch,
        pin_memory=True if args.device == 'cuda' else False
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating RAAH model...")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Pooling type: {args.pooling_type}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Vocab sizes: {vocab.vocab_sizes}")

    model = RAAH(
        hidden_dim=args.hidden_dim,
        vocab_sizes=vocab.vocab_sizes,
        pooling_type=args.pooling_type,
        dropout=args.dropout,
        hidden_size=args.hidden_size
    ).to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, args.device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train F1: {train_metrics['overall_f1']:.4f} | Val F1: {val_metrics['overall_f1']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Per-attribute metrics
        print("\n  Per-attribute F1 scores (Val):")
        for attr_type in vocab.attribute_types:
            f1_key = f'{attr_type}_f1'
            if f1_key in val_metrics:
                print(f"    {attr_type}: {val_metrics[f1_key]:.4f}")

        # Save best model
        if val_metrics['overall_f1'] > best_val_f1:
            best_val_f1 = val_metrics['overall_f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'vocab_sizes': vocab.vocab_sizes,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"\n  ✓ Saved best model (F1: {best_val_f1:.4f})")

        # Save history
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': scheduler.get_last_lr()[0]
        })

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
