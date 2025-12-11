"""
Region-Aware Attribute Head (RAAH) Model
Lightweight classifier on top of DAM decoder to recover fine-grained attributes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class RegionPooling(nn.Module):
    """Pool decoder hidden states into region-level representation"""

    def __init__(self, pooling_type: str = 'mean'):
        """
        Args:
            pooling_type: Type of pooling ('mean', 'max', 'last', 'attention')
        """
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Learnable attention pooling
            self.attention = nn.Linear(1, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool hidden states into a single vector

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] - optional mask for valid tokens

        Returns:
            pooled: [batch_size, hidden_dim]
        """
        if self.pooling_type == 'mean':
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling_type == 'max':
            if attention_mask is not None:
                # Set padding to -inf before max pooling
                hidden_states = hidden_states.clone()
                hidden_states[attention_mask == 0] = float('-inf')
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling_type == 'last':
            # Use last non-padding token
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]

        elif self.pooling_type == 'attention':
            # Learnable attention weights
            attention_scores = self.attention(hidden_states.transpose(1, 2)).transpose(1, 2)
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = (hidden_states * attention_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return pooled


class AttributeClassifier(nn.Module):
    """Multi-head attribute classifier"""

    def __init__(
        self,
        hidden_dim: int,
        vocab_sizes: Dict[str, int],
        dropout: float = 0.1,
        hidden_size: int = 512
    ):
        """
        Args:
            hidden_dim: Dimension of DAM decoder hidden states
            vocab_sizes: Dict mapping attribute_type to number of classes
            dropout: Dropout probability
            hidden_size: Size of intermediate hidden layer
        """
        super().__init__()
        self.attribute_types = list(vocab_sizes.keys())
        self.vocab_sizes = vocab_sizes

        # Shared feature transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Separate classification heads for each attribute type
        self.classifiers = nn.ModuleDict({
            attr_type: nn.Linear(hidden_size, vocab_sizes[attr_type])
            for attr_type in self.attribute_types
        })

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify attributes

        Args:
            pooled_features: [batch_size, hidden_dim]

        Returns:
            Dict mapping attribute_type to logits [batch_size, num_classes]
        """
        # Shared transformation
        shared_features = self.shared_transform(pooled_features)

        # Classify each attribute type
        outputs = {}
        for attr_type in self.attribute_types:
            outputs[attr_type] = self.classifiers[attr_type](shared_features)

        return outputs


class RAAH(nn.Module):
    """
    Region-Aware Attribute Head

    Combines region pooling and attribute classification
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_sizes: Dict[str, int],
        pooling_type: str = 'mean',
        dropout: float = 0.1,
        hidden_size: int = 512
    ):
        """
        Args:
            hidden_dim: Dimension of DAM decoder hidden states
            vocab_sizes: Dict mapping attribute_type to number of classes
            pooling_type: Type of pooling for region representation
            dropout: Dropout probability
            hidden_size: Size of intermediate hidden layer
        """
        super().__init__()

        self.pooling = RegionPooling(pooling_type=pooling_type)
        self.classifier = AttributeClassifier(
            hidden_dim=hidden_dim,
            vocab_sizes=vocab_sizes,
            dropout=dropout,
            hidden_size=hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] - DAM decoder hidden states
            attention_mask: [batch_size, seq_len] - optional attention mask

        Returns:
            Dict mapping attribute_type to logits [batch_size, num_classes]
        """
        # Pool hidden states into region representation
        pooled = self.pooling(hidden_states, attention_mask)

        # Classify attributes
        outputs = self.classifier(pooled)

        return outputs

    def predict_attributes(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        top_k: int = 3
    ) -> List[Dict[str, List[Tuple[int, float]]]]:
        """
        Predict attributes with confidence scores

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            threshold: Minimum confidence threshold
            top_k: Maximum number of predictions per attribute type

        Returns:
            List of dicts (one per batch item) mapping attribute_type to
            list of (label_id, confidence) tuples
        """
        self.eval()
        with torch.no_grad():
            # Get logits
            logits = self.forward(hidden_states, attention_mask)

            batch_size = hidden_states.size(0)
            predictions = []

            for i in range(batch_size):
                sample_pred = {}

                for attr_type, attr_logits in logits.items():
                    # Get probabilities
                    probs = torch.sigmoid(attr_logits[i])

                    # Get top-k predictions above threshold
                    top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(0)))

                    # Filter by threshold
                    valid_preds = [
                        (idx.item(), prob.item())
                        for idx, prob in zip(top_indices, top_probs)
                        if prob.item() >= threshold
                    ]

                    sample_pred[attr_type] = valid_preds

                predictions.append(sample_pred)

        return predictions


def compute_attribute_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    pos_weight: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-label classification loss for attributes

    Args:
        logits: Dict mapping attribute_type to logits [batch_size, num_classes]
        labels: Dict mapping attribute_type to binary labels [batch_size, num_classes]
        pos_weight: Optional positive class weights for imbalanced data

    Returns:
        total_loss: Combined loss across all attribute types
        loss_dict: Individual losses for each attribute type
    """
    loss_dict = {}
    total_loss = 0.0

    for attr_type in logits.keys():
        if attr_type in labels:
            # Binary cross-entropy loss for multi-label classification
            if pos_weight is not None and attr_type in pos_weight:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight[attr_type])
            else:
                criterion = nn.BCEWithLogitsLoss()

            loss = criterion(logits[attr_type], labels[attr_type].float())
            loss_dict[attr_type] = loss
            total_loss = total_loss + loss

    # Average across attribute types
    total_loss = total_loss / len(logits)

    return total_loss, loss_dict


if __name__ == "__main__":
    # Test RAAH model
    batch_size = 4
    seq_len = 50
    hidden_dim = 768

    # Mock decoder hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    # Example vocabulary sizes
    vocab_sizes = {
        'color': 19,
        'material': 12,
        'shape': 10,
        'texture': 11
    }

    # Create model
    model = RAAH(
        hidden_dim=hidden_dim,
        vocab_sizes=vocab_sizes,
        pooling_type='mean'
    )

    # Forward pass
    outputs = model(hidden_states, attention_mask)
    print("Output logits:")
    for attr_type, logits in outputs.items():
        print(f"  {attr_type}: {logits.shape}")

    # Test prediction
    predictions = model.predict_attributes(hidden_states, attention_mask)
    print(f"\nPredictions for batch item 0:")
    for attr_type, preds in predictions[0].items():
        print(f"  {attr_type}: {preds}")
