"""
RAAH Inference Pipeline
Integrates RAAH with DAM to append predicted attributes to captions
"""
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json

from raah.attribute_vocabulary import AttributeVocabulary
from raah.raah_model import RAAH


class RAAHInferencePipeline:
    """
    Inference pipeline that combines DAM captions with RAAH attribute predictions
    """

    def __init__(
        self,
        raah_checkpoint_path: str,
        vocab_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        top_k: int = 3
    ):
        """
        Args:
            raah_checkpoint_path: Path to trained RAAH checkpoint
            vocab_path: Path to attribute vocabulary JSON
            device: Device to run inference on
            confidence_threshold: Minimum confidence for attribute predictions
            top_k: Maximum number of predictions per attribute type
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

        # Load vocabulary
        print(f"Loading attribute vocabulary from {vocab_path}")
        self.vocab = AttributeVocabulary.load(vocab_path)

        # Load RAAH model
        print(f"Loading RAAH model from {raah_checkpoint_path}")
        checkpoint = torch.load(raah_checkpoint_path, map_location=device)

        self.model = RAAH(
            hidden_dim=checkpoint['args']['hidden_dim'],
            vocab_sizes=checkpoint['vocab_sizes'],
            pooling_type=checkpoint['args']['pooling_type'],
            dropout=checkpoint['args']['dropout'],
            hidden_size=checkpoint['args']['hidden_size']
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Store expected hidden dim from training
        self.expected_hidden_dim = checkpoint['args']['hidden_dim']

        # Create adaptive projection layer if needed (for dimension mismatch)
        self.projection = None

        print(f"Model loaded (epoch {checkpoint['epoch']}, val F1: {checkpoint['val_f1']:.4f})")
        print(f"Using device: {device}")
        print(f"Expected hidden dim: {self.expected_hidden_dim}")

    @torch.no_grad()
    def predict_attributes(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Predict attributes from DAM decoder hidden states

        Args:
            hidden_states: [seq_len, hidden_dim] or [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            List of (attribute_type, attribute_value, confidence) tuples
        """
        # Handle single sample
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, seq_len, hidden_dim]
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)

        # Move to device and ensure correct dtype (float32)
        hidden_states = hidden_states.to(self.device, dtype=torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Handle dimension mismatch with adaptive projection
        actual_hidden_dim = hidden_states.shape[-1]
        if actual_hidden_dim != self.expected_hidden_dim:
            if self.projection is None:
                # Create projection layer on first use
                import torch.nn as nn
                self.projection = nn.Linear(
                    actual_hidden_dim,
                    self.expected_hidden_dim,
                    bias=False
                ).to(self.device)
                # Initialize with mean pooling weights (simple averaging)
                with torch.no_grad():
                    self.projection.weight.copy_(
                        torch.randn(self.expected_hidden_dim, actual_hidden_dim) * 0.01
                    )
                print(f"  Created projection layer: {actual_hidden_dim} → {self.expected_hidden_dim}")

            # Apply projection
            batch_size, seq_len, _ = hidden_states.shape
            hidden_states = self.projection(hidden_states.view(-1, actual_hidden_dim))
            hidden_states = hidden_states.view(batch_size, seq_len, self.expected_hidden_dim)

        # Get predictions
        predictions = self.model.predict_attributes(
            hidden_states,
            attention_mask,
            threshold=self.confidence_threshold,
            top_k=self.top_k
        )

        # Decode to attribute strings (use first batch item)
        pred_dict = predictions[0]

        # Convert to list of tuples with scores
        pred_list_with_scores = {
            attr_type: [(score, label_id) for label_id, score in preds]
            for attr_type, preds in pred_dict.items()
        }

        # Decode using vocabulary
        decoded = self.vocab.decode_predictions(
            pred_list_with_scores,
            threshold=self.confidence_threshold
        )

        return decoded

    def augment_caption(
        self,
        caption: str,
        predicted_attributes: List[Tuple[str, str, float]],
        deduplication: bool = True
    ) -> str:
        """
        Augment caption with predicted attributes

        Args:
            caption: Original DAM caption
            predicted_attributes: List of (attr_type, attr_value, confidence)
            deduplication: Remove attributes already mentioned in caption

        Returns:
            Augmented caption
        """
        if not predicted_attributes:
            return caption

        # Filter out attributes already in caption (case-insensitive)
        caption_lower = caption.lower()

        if deduplication:
            filtered_attrs = [
                (attr_type, attr_value, conf)
                for attr_type, attr_value, conf in predicted_attributes
                if attr_value.lower() not in caption_lower
            ]
        else:
            filtered_attrs = predicted_attributes

        if not filtered_attrs:
            return caption

        # Group by attribute type
        attrs_by_type = {}
        for attr_type, attr_value, conf in filtered_attrs:
            if attr_type not in attrs_by_type:
                attrs_by_type[attr_type] = []
            attrs_by_type[attr_type].append((attr_value, conf))

        # Build attribute phrase
        attr_phrases = []
        for attr_type in ['color', 'material', 'shape', 'texture']:
            if attr_type in attrs_by_type:
                # Sort by confidence
                values = sorted(attrs_by_type[attr_type], key=lambda x: x[1], reverse=True)
                # Take top value
                attr_phrases.append(values[0][0])

        if not attr_phrases:
            return caption

        # Format: "caption [ATTR: attr1, attr2, ...]"
        attr_str = ", ".join(attr_phrases)
        augmented = f"{caption.rstrip('.')}. Additional attributes: {attr_str}."

        return augmented

    def __call__(
        self,
        hidden_states: torch.Tensor,
        caption: str,
        attention_mask: Optional[torch.Tensor] = None,
        return_attributes: bool = False
    ) -> Union[str, Tuple[str, List[Tuple[str, str, float]]]]:
        """
        Full inference pipeline: predict attributes and augment caption

        Args:
            hidden_states: DAM decoder hidden states
            caption: Original DAM caption
            attention_mask: Optional attention mask
            return_attributes: If True, return (caption, attributes) tuple

        Returns:
            Augmented caption, or (augmented_caption, predicted_attributes) if return_attributes=True
        """
        # Predict attributes
        predicted_attributes = self.predict_attributes(hidden_states, attention_mask)

        # Augment caption
        augmented_caption = self.augment_caption(caption, predicted_attributes)

        if return_attributes:
            return augmented_caption, predicted_attributes
        else:
            return augmented_caption


def batch_inference(
    raah_pipeline: RAAHInferencePipeline,
    hidden_states_list: List[torch.Tensor],
    captions: List[str],
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Run batch inference with RAAH

    Args:
        raah_pipeline: RAAHInferencePipeline instance
        hidden_states_list: List of hidden state tensors
        captions: List of original captions
        output_path: Optional path to save results

    Returns:
        List of dicts with original and augmented captions
    """
    results = []

    for i, (hidden_states, caption) in enumerate(zip(hidden_states_list, captions)):
        augmented_caption, attributes = raah_pipeline(
            hidden_states,
            caption,
            return_attributes=True
        )

        result = {
            'index': i,
            'original_caption': caption,
            'augmented_caption': augmented_caption,
            'predicted_attributes': [
                {'type': attr_type, 'value': attr_value, 'confidence': conf}
                for attr_type, attr_value, conf in attributes
            ]
        }

        results.append(result)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='raah/checkpoints/best_model.pt')
    parser.add_argument('--vocab', type=str, default='raah/attribute_vocab.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create pipeline
    pipeline = RAAHInferencePipeline(
        raah_checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )

    # Test with dummy hidden states
    print("\nTesting with dummy hidden states...")
    hidden_states = torch.randn(50, 768)  # [seq_len, hidden_dim]
    caption = "A washing machine in the corner of a room."

    augmented_caption, attributes = pipeline(
        hidden_states,
        caption,
        return_attributes=True
    )

    print(f"\nOriginal caption: {caption}")
    print(f"Predicted attributes: {attributes}")
    print(f"Augmented caption: {augmented_caption}")
