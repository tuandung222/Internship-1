# Qwen3-VL Cookbook Alignment

This project mirrors the official Qwen3-VL cookbook patterns (see `../Qwen3-VL/cookbooks`) when running the CoRGI pipeline with the real model.

## Key Parallels
- **Model + Processor loading**: We rely on `AutoModelForImageTextToText` and `AutoProcessor` exactly as described in the main README and cookbook notebooks such as `think_with_images.ipynb`.
- **Chat template**: `Qwen3VLClient` uses `processor.apply_chat_template(..., add_generation_prompt=True)` before calling `generate`, which matches the recommended multi-turn messaging flow.
- **Image transport**: Both the pipeline and demo scripts accept PIL images and ensure conversion to RGB prior to inference, mirroring cookbook utilities that normalize channels.
- **Max tokens & decoding**: Default `max_new_tokens=512` and `temperature=0.2` align with cookbook demos favouring deterministic outputs for evaluation.
- **Single-model pipeline**: All stages (reasoning, ROI extraction, answer synthesis) are executed by the same Qwen3-VL instance, following the cookbook philosophy of leveraging the model’s intrinsic grounding capability without external detectors.

## Practical Tips for Local Inference
- Use the `pytorch` Conda env with the latest `transformers` (>=4.45) to access `AutoModelForImageTextToText` support, as advised in the cookbook README.
- When VRAM is limited, switch to `Qwen/Qwen3-VL-4B-Instruct` via `--model-id` or `CORGI_QWEN_MODEL` environment variable—no other code changes needed.
- The integration test (`corgi_tests/test_integration_qwen.py`) and demo (`examples/demo_qwen_corgi.py`) download the official demo image if `CORGI_DEMO_IMAGE` is not supplied, matching cookbook notebooks that reference the same asset URL.
- For reproducibility, set `HF_HOME` (or use the cookbook’s `snapshot_download`) to manage local caches and avoid repeated downloads.
- The `Qwen/Qwen3-VL-8B-Thinking` checkpoint often emits free-form “thinking” text instead of JSON; the pipeline now falls back to parsing those narratives for step and ROI extraction, and strips `<think>…</think>` scaffolding from final answers.

These notes ensure our CoRGI adaptation stays consistent with the official Qwen workflow while keeping the codebase modular for experimentation.
