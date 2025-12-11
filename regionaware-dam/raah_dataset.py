"""
Dataset for training RAAH
Loads DAM hidden states and attribute labels
"""
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from raah.attribute_vocabulary import AttributeVocabulary


class RAAHDataset(Dataset):
    """
    Dataset that loads pre-computed DAM hidden states and attribute labels

    Note: This requires running DAM inference first to extract hidden states
    """

    def __init__(
        self,
        hidden_states_dir: str,
        dataset_json_path: str,
        vocab: AttributeVocabulary,
        split: str = 'train'
    ):
        """
        Args:
            hidden_states_dir: Directory containing saved hidden states (.pt files)
            dataset_json_path: Path to small_objects_dataset.json
            vocab: AttributeVocabulary instance
            split: Dataset split ('train' or 'val')
        """
        self.hidden_states_dir = Path(hidden_states_dir)
        self.vocab = vocab
        self.split = split

        # Load dataset annotations
        with open(dataset_json_path, 'r') as f:
            data = json.load(f)

        self.samples = data['samples']

        # 1. Determine indices for the split (train/val)
        np.random.seed(42)
        all_indices = np.random.permutation(len(self.samples))
        split_idx = int(0.8 * len(self.samples))

        if split == 'train':
            split_indices = all_indices[:split_idx]
        else:
            split_indices = all_indices[split_idx:]
            
        initial_count = len(split_indices)

        # ----------------------------------------------------------------------
        # *** FIX: Filter samples by hidden state file existence ***
        # ----------------------------------------------------------------------
        
        # 2. Filter the indices based on whether the corresponding .pt file exists
        filtered_indices = []
        missing_count = 0
        
        for idx in split_indices:
            sample = self.samples[idx]
            ann_id = sample['ann_id']
            hidden_states_path = self.hidden_states_dir / f"hidden_{ann_id}.pt"
            
            if hidden_states_path.exists():
                filtered_indices.append(idx)
            else:
                missing_count += 1
                
        self.indices = np.array(filtered_indices)
        
        # ----------------------------------------------------------------------

        print(f"Loaded {len(self.indices)} samples for {split} split (out of {initial_count} total)")
        if missing_count > 0:
             print(f"⚠️ Warning: Filtered out {missing_count} samples from '{split}' split because the .pt file was missing.")
        

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict containing:
                - hidden_states: [seq_len, hidden_dim]
                - labels: Dict of attribute_type -> multi-hot vector [num_classes]
                - ann_id: Annotation ID
                - caption: Original caption (for reference)
        """
        # Get sample
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]

        ann_id = sample['ann_id']

        # Load pre-computed hidden states (file existence guaranteed by __init__)
        hidden_states_path = self.hidden_states_dir / f"hidden_{ann_id}.pt"
        hidden_states = torch.load(hidden_states_path, weights_only=True) 

        # Encode attributes as multi-hot vectors
        labels = self._encode_labels(sample['attributes_found'])

        return {
            'hidden_states': hidden_states,
            'labels': labels,
            'ann_id': ann_id,
            'caption': sample['caption']
        }

    def _encode_labels(self, attributes_found: List[List[str]]) -> Dict[str, torch.Tensor]:
        """
        Convert attribute list to multi-hot label vectors
        
        (This function remains unchanged)
        """
        labels = {}

        for attr_type in self.vocab.attribute_types:
            # Initialize zero vector
            num_classes = self.vocab.vocab_sizes[attr_type]
            multi_hot = torch.zeros(num_classes)

            # Set indices to 1 for present attributes
            for found_type, found_value in attributes_found:
                if found_type == attr_type:
                    found_value_lower = found_value.lower()
                    if found_value_lower in self.vocab.vocab[attr_type]:
                        label_id = self.vocab.vocab[attr_type][found_value_lower]
                        multi_hot[label_id] = 1.0

            labels[attr_type] = multi_hot

        return labels


def collate_raah_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for batching RAAH samples with variable-length sequences

    Args:
        batch: List of dicts from __getitem__, each containing:
            - hidden_states: [seq_len, hidden_dim]
            - labels: Dict of attribute_type -> [num_classes]
            - ann_id: int
            - caption: str

    Returns:
        Dict containing:
            - hidden_states: [batch_size, max_seq_len, hidden_dim] (padded)
            - attention_mask: [batch_size, max_seq_len] (1 for real, 0 for padding)
            - labels: Dict of attribute_type -> [batch_size, num_classes]
            - ann_ids: List of annotation IDs
            - captions: List of captions
    """
    # Get max sequence length in this batch
    max_len = max(item['hidden_states'].shape[0] for item in batch)
    batch_size = len(batch)
    hidden_dim = batch[0]['hidden_states'].shape[1]

    # Initialize tensors
    hidden_states = torch.zeros(batch_size, max_len, hidden_dim)
    attention_mask = torch.zeros(batch_size, max_len)

    # Stack labels
    labels = {}
    attribute_types = list(batch[0]['labels'].keys())
    for attr_type in attribute_types:
        labels[attr_type] = torch.stack([item['labels'][attr_type] for item in batch])

    # Fill in the batch
    ann_ids = []
    captions = []

    for i, item in enumerate(batch):
        seq_len = item['hidden_states'].shape[0]
        hidden_states[i, :seq_len, :] = item['hidden_states']
        attention_mask[i, :seq_len] = 1
        ann_ids.append(item['ann_id'])
        captions.append(item['caption'])

    return {
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'labels': labels,
        'ann_ids': ann_ids,
        'captions': captions
    }


class SimpleRAAHDataset(Dataset):
    # (This class remains unchanged)
    # ...
    # Implementation of SimpleRAAHDataset
    # ...
    pass


if __name__ == "__main__":
    # Test dataset
    from raah.attribute_vocabulary import AttributeVocabulary

    vocab = AttributeVocabulary(min_frequency=2)
    vocab.build_from_dataset('data/small_objects_dataset.json')

    # Test simple dataset
    dataset = SimpleRAAHDataset(
        dataset_json_path='data/small_objects_dataset.json',
        vocab=vocab,
        split='train'
    )

    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Hidden states shape: {sample['hidden_states'].shape}")
    print(f"Labels: {[(k, v.shape, v.sum().item()) for k, v in sample['labels'].items()]}")

    # Test collate function
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_raah_batch,
        shuffle=True
    )

    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch hidden_states shape: {batch['hidden_states'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shapes: {[(k, v.shape) for k, v in batch['labels'].items()]}")