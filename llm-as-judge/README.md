# DLC-Bench Evaluation Suite

Complete toolkit for benchmarking vision-language models on DLC-Bench (Detailed Localized Captioning) using LLM-as-a-judge evaluation.

---

## Quick Setup

```bash
# Clone or navigate to the evaluation directory
cd evaluation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install openai anthropic tqdm inflect python-dotenv scipy pycocotools pillow requests transformers torch huggingface_hub bitsandbytes matplotlib
```

### Download DLC-Bench Dataset

Choose one of the following methods:

**Method 1: Using the download script (recommended)**
```bash
python3 download_dlc_bench.py
```

**Method 2: Manual git clone**
```bash
git lfs install
git clone https://huggingface.co/datasets/nvidia/DLC-Bench
```

**Method 3: Using bash script**
```bash
chmod +x download_dlc_bench.sh
./download_dlc_bench.sh
```

### API Keys Configuration

Create `.env.local` file with your API keys:

```bash
# For OpenAI models (GPT-4o, GPT-4o-mini, etc.)
OPENAI_API_KEY=sk-proj-...

# For Anthropic models (Claude Sonnet 4.5, etc.)
ANTHROPIC_API_KEY=sk-ant-...

# For Alibaba models (Qwen3-VL-Plus, etc.)
ALIBABA_API_KEY=sk-...

# For Google models (Gemini 2.5 Flash, etc.)
GEMINI_API_KEY=...
```

---

## Complete Evaluation Workflow

### Step 1: Generate Captions

Choose one of the following methods to generate captions for DLC-Bench:

#### Option A: API-based Vision Models

**GPT-4o / GPT-4o-mini (OpenAI)**
```bash
python caption_gpt.py \
    --model gpt-4o-mini \
    --output model_outputs_cache/gpt4o_mini_captions.json \
    --visualization highlight \
    --prompt-template dam_style
```

**Claude Sonnet 4.5 (Anthropic)**
```bash
python caption_gpt.py \
    --model claude-sonnet-4-5-20250929 \
    --output model_outputs_cache/claude_sonnet_4_5_captions.json
```

**Qwen3-VL-Plus (Alibaba DashScope)**
```bash
python caption_gpt.py \
    --model qwen3-vl-plus \
    --output model_outputs_cache/qwen3_vl_plus_captions.json
```

**Gemini 2.5 Flash (Google)**
```bash
python caption_gpt.py \
    --model gemini-2.5-flash \
    --output model_outputs_cache/gemini_2_5_flash_captions.json
```

**Available options:**
- `--model`: Vision model name
- `--visualization`: highlight (default), masked_only, overlay, side_by_side
- `--prompt-template`: dam_style (default), default, simple, very_detailed
- `--resume`: Resume interrupted runs
- `--limit N`: Process only first N annotations (for testing)
- `--verbose`: Print captions as they are generated

#### Option B: Local DAM Model

```bash
python caption_dam_local.py \
    --model-path /path/to/DAM-3B \
    --output model_outputs_cache/dam_local_captions.json \
    --crop-mode full+focal_crop \
    --temperature 0.2
```

**Key options:**
- `--crop-mode`: full+focal_crop (default), full, focal_crop
- `--temperature`: 0.2 (default, same as paper)
- `--device`: cuda (default), cpu

#### Option C: Local Gemma Model

```bash
python caption_gemma_local.py \
    --model google/gemma-3-4b-pt \
    --output model_outputs_cache/gemma_3_4b_captions.json \
    --load-in-4bit
```

**Memory optimization:**
- `--load-in-4bit`: ~2-3GB VRAM (recommended for 8GB GPUs)
- `--load-in-8bit`: ~4GB VRAM
- No flag: Full precision (~8GB+ VRAM)

**Note:** Requires HuggingFace access. Login with `huggingface-cli login`

### Step 2: Evaluate Captions with LLM Judge

Evaluate the generated captions using GPT models as judges:

```bash
python eval_model_outputs_gpt4o.py \
    --pred model_outputs_cache/gpt4o_mini_captions.json \
    --model gpt-4o-mini
```

**Output:** Creates `model_outputs_cache/gpt4o_mini_captions_eval_gpt-4o-mini.json`

**Judge model options:**
- `gpt-4o-mini`: Fast and cheap (~$0.08 per 100 annotations)
- `gpt-4o`: More accurate but expensive (~$1.50 per 100 annotations)
- `gpt-3.5-turbo`: Cheapest (~$0.03 per 100 annotations)

**Additional options:**
- `--api-call-limit N`: Limit API calls (default: 2000)
- `--default-prediction TEXT`: Fallback for missing captions

### Step 3: Analyze Results

```bash
python analyze_results.py \
    --eval-file model_outputs_cache/gpt4o_mini_captions_eval_gpt-4o-mini.json
```

**Output includes:**
- Overall scores (Positive, Negative, Overall)
- Comparison with paper benchmarks
- Score distribution analysis
- Recognition failure analysis
- Top/bottom performing examples

---

## Example: Complete Benchmark Pipeline

Here's a complete example comparing GPT-4o-mini against local DAM:

```bash
# Generate captions
# Generate captions with LLMs (for comparison) [OpenAI, Anthropic, Qwen, Gemini]
python caption_gpt.py \
    --model gpt-4o-mini \
    --output model_outputs_cache/<captions_file>.json

# OR Generate captions with DAM locally
python caption_dam_local.py \
    --model-path /path/to/DAM-3B \
    --output model_outputs_cache/<captions_file>.json

# Evaluate with LLM as a judge
# 1. Generate answers for the captions by the judge model
python eval_model_outputs_gpt4o.py \
    --pred model_outputs_cache/<captions_file>.json \
    --model gpt-4o-mini

# 2. Convert the scores to percentages and generate the analysis
python analyze_results.py \
    --eval-file model_outputs_cache/<captions_file>_eval_<judge_model>.json
```

---

## Understanding LLM-as-a-Judge Evaluation

### How It Works

The evaluation uses GPT models to judge caption quality through multiple-choice questions:

1. **Recognition Question**: Does the caption correctly identify the object?
2. **Positive Questions** (~4 per instance): Does the caption mention expected attributes (color, shape, etc.)?
3. **Negative Questions** (~5 per instance): Does the caption avoid hallucinating non-existent attributes?

### Score Calculation

**Positive Score** (measures completeness):
- Tests if caption includes correct attributes
- Question scores: 1 (correct), 0.5 (partial), 0 (missing), -1 (incorrect)
- Average across all positive questions

**Negative Score** (measures hallucination avoidance):
- Tests if caption avoids mentioning non-existent attributes
- Question scores: 1 (not mentioned - good), 0 (partial), -1 (hallucinated - bad)
- Average across all negative questions

**Overall Score**:
- `(Positive Score + Negative Score) / 2`

**Recognition Failure**:
- If object class is wrong, all scores capped at 0
- Critical for overall performance