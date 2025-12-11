"""
Attribute Vocabulary Builder for RAAH
Extracts and manages attribute labels from dataset
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class AttributeVocabulary:
    """Manages attribute vocabulary and label encoding"""

    def __init__(self, min_frequency: int = 2):
        """
        Args:
            min_frequency: Minimum frequency for an attribute value to be included
        """
        self.min_frequency = min_frequency
        self.attribute_types = ['color', 'material', 'shape', 'texture']

        # Maps: attribute_type -> {value -> label_id}
        self.vocab: Dict[str, Dict[str, int]] = {}

        # Reverse maps: attribute_type -> {label_id -> value}
        self.idx_to_value: Dict[str, Dict[int, str]] = {}

        # Sizes for each attribute type
        self.vocab_sizes: Dict[str, int] = {}

    def build_from_dataset(self, dataset_path: str) -> None:
        """Build vocabulary from small objects dataset"""
        print(f"Building attribute vocabulary from {dataset_path}")

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        samples = data['samples']

        # Count attribute values by type
        attribute_counters = {
            attr_type: Counter()
            for attr_type in self.attribute_types
        }

        for sample in samples:
            if 'attributes_found' in sample and sample['attributes_found']:
                for attr_type, attr_value in sample['attributes_found']:
                    if attr_type in attribute_counters:
                        attribute_counters[attr_type][attr_value.lower()] += 1

        # Build vocabulary for each attribute type
        for attr_type in self.attribute_types:
            # Filter by minimum frequency
            filtered_values = [
                value for value, count in attribute_counters[attr_type].items()
                if count >= self.min_frequency
            ]

            # Sort for consistent ordering
            filtered_values.sort()

            # Create value -> id mapping (start from 0, reserve no label)
            self.vocab[attr_type] = {
                value: idx for idx, value in enumerate(filtered_values)
            }

            # Create id -> value mapping
            self.idx_to_value[attr_type] = {
                idx: value for value, idx in self.vocab[attr_type].items()
            }

            # Store vocabulary size (number of classes)
            self.vocab_sizes[attr_type] = len(filtered_values)

            print(f"  {attr_type}: {self.vocab_sizes[attr_type]} unique values "
                  f"(from {len(attribute_counters[attr_type])} total)")

    def encode_attributes(self, attributes_found: List[Tuple[str, str]]) -> Dict[str, List[int]]:
        """
        Encode attributes into multi-hot vectors

        Args:
            attributes_found: List of (attribute_type, attribute_value) tuples

        Returns:
            Dict mapping attribute_type to list of label indices
        """
        encoded = {attr_type: [] for attr_type in self.attribute_types}

        for attr_type, attr_value in attributes_found:
            if attr_type in self.vocab:
                attr_value_lower = attr_value.lower()
                if attr_value_lower in self.vocab[attr_type]:
                    label_id = self.vocab[attr_type][attr_value_lower]
                    encoded[attr_type].append(label_id)

        return encoded

    def decode_predictions(
        self,
        predictions: Dict[str, List[int]],
        threshold: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Decode predictions back to attribute strings

        Args:
            predictions: Dict of attribute_type -> list of (score, label_id) tuples
            threshold: Minimum confidence threshold

        Returns:
            List of (attribute_type, attribute_value, confidence) tuples
        """
        decoded = []

        for attr_type, pred_list in predictions.items():
            if attr_type in self.idx_to_value:
                for score, label_id in pred_list:
                    if score >= threshold and label_id in self.idx_to_value[attr_type]:
                        value = self.idx_to_value[attr_type][label_id]
                        decoded.append((attr_type, value, score))

        return decoded

    def save(self, save_path: str) -> None:
        """Save vocabulary to disk"""
        data = {
            'vocab': self.vocab,
            'idx_to_value': self.idx_to_value,
            'vocab_sizes': self.vocab_sizes,
            'attribute_types': self.attribute_types,
            'min_frequency': self.min_frequency
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Vocabulary saved to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> 'AttributeVocabulary':
        """Load vocabulary from disk"""
        with open(load_path, 'r') as f:
            data = json.load(f)

        vocab_obj = cls(min_frequency=data['min_frequency'])
        vocab_obj.vocab = {k: v for k, v in data['vocab'].items()}
        vocab_obj.idx_to_value = {
            k: {int(idx): v for idx, v in mapping.items()}
            for k, mapping in data['idx_to_value'].items()
        }
        vocab_obj.vocab_sizes = data['vocab_sizes']
        vocab_obj.attribute_types = data['attribute_types']

        print(f"Vocabulary loaded from {load_path}")
        for attr_type, size in vocab_obj.vocab_sizes.items():
            print(f"  {attr_type}: {size} classes")

        return vocab_obj


if __name__ == "__main__":
    # Build and save vocabulary
    vocab = AttributeVocabulary(min_frequency=2)
    vocab.build_from_dataset('data/small_objects_dataset.json')
    vocab.save('raah/attribute_vocab.json')

    # Test encoding
    test_attrs = [('color', 'white'), ('material', 'wood'), ('shape', 'round')]
    encoded = vocab.encode_attributes(test_attrs)
    print("\nTest encoding:")
    print(f"  Input: {test_attrs}")
    print(f"  Encoded: {encoded}")
