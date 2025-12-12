"""
locality_guard.py

Scaffolding for the LocalityGuard extension on top of the
Describe Anything Model (DAM).

LocalityGuard is an inference-time mechanism that encourages
the model to rely on visual evidence inside the target region
while still allowing global context for coherence.

This file is currently self-contained and does not depend on
the rest of the DAM codebase, so it can be imported and wired
in later from the generation script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import LogitsProcessor


@dataclass
class AttentionSnapshot:
    """
    Lightweight container for a single attention map.

    Args:
        attn: Tensor of shape [num_heads, num_queries, num_keys].
              For DAM, keys correspond to visual tokens.
        mask_hw: Optional spatial mask in [H, W] that can be
                 used to compute region vs non-region attention.
    """
    attn: torch.Tensor
    mask_hw: Optional[torch.Tensor] = None


class AttentionTracker:
    """
    Simple tracker for cross-attention maps during generation.

    Typical usage:

        tracker = AttentionTracker()
        # register a forward hook on the cross-attn module and
        # call tracker.record(attn_weights) inside the hook
        ...
        latest = tracker.latest()

    For now this is just a placeholder. Hook registration and
    exact tensor shapes will be filled in once the DAM layers
    are inspected.
    """

    def __init__(self) -> None:
        self._history: List[AttentionSnapshot] = []

    def clear(self) -> None:
        """Drop all stored attention maps."""
        self._history.clear()

    def record(self, attn: torch.Tensor, mask_hw: Optional[torch.Tensor] = None) -> None:
        """
        Store a new attention map.

        Args:
            attn: Attention weights, expected shape [B, num_heads, Q, K]
                  or [num_heads, Q, K]. Only the last step will be used
                  by the LocalityGuard processor.
            mask_hw: Optional binary mask over spatial positions.
        """
        if attn is None:
            return

        # Detach to avoid keeping autograd history on the tracker side.
        attn = attn.detach().cpu()
        if mask_hw is not None:
            mask_hw = mask_hw.detach().cpu()

        self._history.append(AttentionSnapshot(attn=attn, mask_hw=mask_hw))

    def latest(self) -> Optional[AttentionSnapshot]:
        """Return the most recent snapshot, if any."""
        if not self._history:
            return None
        return self._history[-1]


class LocalityGuardProcessor(LogitsProcessor):
    """
    Phase-1 LocalityGuard stub.

    This class is designed to plug into the HuggingFace `generate`
    API as a `LogitsProcessor`. At every decoding step it receives
    the current input ids and the logits for the next token and
    may modify the logits before sampling.

    In later phases this processor will:
      * Read the latest cross-attention map from an AttentionTracker.
      * Compare attention inside vs outside the target region.
      * Apply a penalty proportional to "leakage" outside the region.

    For now it is intentionally implemented as a no-op so that
    integration with DAM can be tested without changing behaviour.
    """

    def __init__(
        self,
        lambda_penalty: float = 0.0,
        tracker: Optional[AttentionTracker] = None,
    ) -> None:
        super().__init__()
        self.lambda_penalty = float(lambda_penalty)
        self.tracker = tracker

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: [batch_size, seq_len] token ids generated so far.
            scores:    [batch_size, vocab_size] logits for the next token.

        Returns:
            Modified scores tensor (currently unchanged).
        """

        # Placeholder: this is where the locality penalty will go.
        # Example sketch for later:
        #
        #   snapshot = self.tracker.latest()
        #   if snapshot is not None:
        #       leakage = compute_leakage(snapshot.attn, snapshot.mask_hw)
        #       scores = scores - self.lambda_penalty * leakage
        #
        # For Phase 1 we just pass scores through unchanged.
        return scores


def create_default_locality_guard(lambda_penalty: float = 0.0) -> LocalityGuardProcessor:
    """
    Convenience constructor that creates a tracker and a processor
    at once. This is what the generation script will likely call:

        tracker, lg = create_default_locality_guard(lambda_penalty=0.5)
        # register hooks that call tracker.record(...)
        # pass `lg` into the logits_processor list of `model.generate`
    """
    tracker = AttentionTracker()
    processor = LocalityGuardProcessor(lambda_penalty=lambda_penalty, tracker=tracker)
    return processor
