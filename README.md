# FocusDAM

# DAM + LocalityGuard Extension

This repository hosts our course project on **Describe Anything Model (DAM)** and an inference-time extension called **LocalityGuard**.

The high-level goal is:

> Encourage DAM to rely more on visual evidence *inside* a selected region,
> while still using the full image for global coherence.

The project is organized around two roles:
- **Baseline:** Reproduce DAM-3B results on DLC-Bench with the official NVIDIA codebase.
- **Extension (LocalityGuard):** Add a decoding-time locality penalty on top of the baseline.

## Current status

- ✅ Repository created and linked to my working environment.
- ✅ Initial scaffolding for `LocalityGuardProcessor` and `AttentionTracker` is in place  
  (see `locality_guard.py`).
- ⏳ Next steps: wire the processor into the DAM generation call (`model.generate`) and
  register attention hooks on the cross-attention layers.

The `locality_guard.py` file currently defines:

- `AttentionTracker`: a small utility to store cross-attention maps during generation.
- `LocalityGuardProcessor`: a `transformers.LogitsProcessor` subclass that will be plugged
  into the HuggingFace `generate` API.
- `create_default_locality_guard(...)`: convenience helper to construct both the tracker
  and the processor from one call.

At this stage, `LocalityGuardProcessor` is intentionally a **no-op**, so we can test the
integration with DAM without changing its behaviour. The actual locality penalty logic
will be added once the cross-attention tensor shapes are confirmed from the DAM model.

## Planned next steps

1. Integrate `LocalityGuardProcessor` into the DAM inference path
   (inside the function that builds `generation_kwargs` for `model.generate`).
2. Register forward hooks on the cross-attention modules to populate `AttentionTracker`.
3. Implement a simple leakage metric:
   - compare attention inside vs outside the binary mask for the selected region;
   - apply a penalty proportional to leakage.
4. Run ablations on DLC-Bench and measure:
   - caption quality (CIDEr / SPICE)
   - region grounding quality (ORL or related metrics).

Once the LocalityGuard path is stable, I will open it up for teammates to add:
- multi-scale variants,
- alternative penalty functions,
- and LoRA-style fine-tuning experiments if time permits.
