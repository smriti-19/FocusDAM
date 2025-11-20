# FocusDAM


FocusDAM is our course project on extending the **Describe Anything Model (DAM)** with region-aware and robustness improvements.

The goal is to start from the official DAM-3B baseline and add inference-time and architecture-level tweaks that make the model:

- Rely more on evidence inside the selected region
- Behave consistently across scales and crops
- Produce cleaner, more focused captions that are still globally coherent

---

## Project components

We are organizing the work into four main pieces:

### 1. Baseline replication (DAM-3B on DLC-Bench)

- Reproduce DAM-3B results using the official `NVlabs/describe-anything` code.
- Run on DLC-Bench and log the same metrics as the paper:
  - Captioning: CIDEr, SPICE, BLEU, LLM-judge
  - Grounding / region quality: ORL or equivalent
- This serves as the “control” system for all our ablations.

**Outputs:**

- A stable evaluation script for DLC-Bench
- Saved prediction files for each variant:
  - `DAM-3B-baseline`
  - `DAM-3B-LocalityGuard`
  - `DAM-3B-MultiScale`
  - `DAM-3B-Rephraser`

---

### 2. LocalityGuard (decode-time locality penalty)

LocalityGuard is an inference-time extension that nudges the model to ground its words inside the selected region.

High-level idea:

- Keep DAM’s full-image context so captions stay coherent.
- Track cross-attention during decoding and estimate how much attention “leaks” outside the region mask.
- Penalize logits when leakage is high, especially for content words.

Current scaffolding:

- `locality_guard.py`
  - `AttentionTracker`: stores cross-attention maps over time.
  - `LocalityGuardProcessor`: subclass of `transformers.LogitsProcessor` that will be plugged into `model.generate`.
  - `create_default_locality_guard(...)`: helper to build tracker + processor together.

Planned implementation steps:

1. Wire `LocalityGuardProcessor` into the DAM generation call in `get_description_from_prompt_iterator` (via `logits_processor`).
2. Register forward hooks on the cross-attention layers to populate `AttentionTracker`.
3. Implement a simple leakage metric (attention inside vs outside the region mask).
4. Apply a penalty proportional to leakage and evaluate its effect on both caption quality and region correctness.

---

### 3. Lite multi-scale attention

The second extension focuses on improving robustness to object size and framing.

Planned design:

- Use multiple crops / scales (e.g. full image + focal crop) that DAM already supports in its `prompt_mode`.
- Experiment with:
  - Reweighting visual tokens from different scales during decoding.
  - Simple ensembling of logits from full vs focal views.
- Measure whether multi-scale variants help:
  - Small objects inside large scenes
  - Partially occluded or thin regions

This will likely live in a separate module, e.g. `multiscale_focus.py`, and hook into the same generation path as LocalityGuard.

---

### 4. Rephraser / caption polishing

The third extension focuses on improving the **form** of the caption without changing the underlying evidence:

- Take the raw DAM caption and optionally:
  - Remove redundant context that is clearly outside the marked region.
  - Clarify object type, attributes, or spatial relations.
- Implemented either as:
  - A lightweight prompt to DAM itself, or
  - A small rephraser model (LoRA or instruction-tuned variant).

Planned evaluation:

- Human or LLM-judge ratings for:
  - Faithfulness to the region
  - Clarity and conciseness
  - Whether important regional details are preserved

---

## Current status

- Repo created: **FocusDAM**
- Initial scaffolding for LocalityGuard is in place:
  - `AttentionTracker`
  - `LocalityGuardProcessor`
- Next immediate tasks:
  - Plug LocalityGuard into the DAM `generate` call using `logits_processor`
  - Add a test script that runs DAM + LocalityGuard on a few DLC-Bench examples

---

## Roadmap

Rough order of operations:

1. **Week 1–2**
   - Finish DAM-3B baseline replication on DLC-Bench.
   - Integrate LocalityGuard as a no-op `LogitsProcessor` and verify it runs per token.
2. **Week 3**
   - Add attention hooks, implement leakage penalty, run ablations.
   - Implement simplest multi-scale variant (two views, logit interpolation).
3. **Week 4**
   - Add the rephraser step and evaluate captions with and without rephrasing.
   - Collect qualitative examples and finalize figures / tables for the poster.

---

## How to contribute

Typical workflow:

```bash
git clone https://github.com/smriti-19/FocusDAM.git
cd FocusDAM

# create your own branch
git checkout -b <feature-name>

# edit code...

git add .
git commit -m "Describe your change here"
git push origin <feature-name>
