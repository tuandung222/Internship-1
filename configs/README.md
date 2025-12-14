# CoRGI Pipeline Configurations

## Recommended Configs (V2 Pipeline)

| Config | Description | Models | Use Case |
|--------|-------------|--------|----------|
| **`qwen_only_v2.yaml`** | Single Qwen model | Qwen3-VL-4B | Fast, minimal VRAM |
| **`qwen_florence2_smolvlm2_v2.yaml`** | Multi-model | Qwen + Florence + SmolVLM | Best quality |
| **`default_v2.yaml`** | Default V2 | Varies | General use |

## Quick Start

```bash
# Simple (single model)
python inference.py --config configs/qwen_only_v2.yaml --image test.jpg --question "..."

# Best quality (multi-model)
python inference.py --config configs/qwen_florence2_smolvlm2_v2.yaml --image test.jpg --question "..."
```

## Config Structure (V2)

```yaml
# Phase 1+2 MERGED: Reasoning + Grounding
reasoning:
  model:
    model_type: qwen_instruct
    model_id: Qwen/Qwen3-VL-4B-Instruct
    device: cuda:0
    use_v2_prompt: true  # IMPORTANT: Enable V2 prompts

# Grounding (reuse reasoning model)
grounding:
  reuse_reasoning: true

# Phase 3: Evidence Extraction (smart routing)
captioning:
  model:
    model_type: composite  # OR: qwen_instruct for single model
  ocr:
    model_type: florence2  # Text extraction
  caption:
    model_type: smolvlm2   # Object description

# Phase 4: Synthesis
synthesis:
  reuse_reasoning: true

# Pipeline settings
pipeline:
  use_v2: true
  max_reasoning_steps: 6
  max_regions_per_step: 1
```

## Legacy Configs

V1 pipeline configs are in `legacy/` folder for backward compatibility.

```bash
# Use V1 pipeline
python inference.py --config configs/legacy/qwen_only.yaml --image test.jpg --question "..."
```

## Model Requirements

| Model | VRAM | Speed |
|-------|------|-------|
| Qwen3-VL-2B | ~6GB | Fast |
| Qwen3-VL-4B | ~10GB | Medium |
| Florence-2-base | ~2GB | Very Fast |
| SmolVLM2-500M | ~1GB | Very Fast |

## Notes

- All configs use `cuda:5` by default - change `device` as needed
- Set `enable_compile: true` for PyTorch 2.0+ speedup
- Use `torch_dtype: bfloat16` for modern GPUs, `float16` for older GPUs
