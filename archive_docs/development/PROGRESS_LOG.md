# CoRGI Custom Demo — Progress Log

> Keep this log short and chronological. Newest updates at the top.

## 2024-10-22
- Reproduced the CoRGI pipeline failure with the real `Qwen/Qwen3-VL-8B-Thinking` checkpoint and traced it to reasoning outputs that only use ordinal step words.
- Taught the text parser to normalize “First/Second step” style markers into numeric indices, refreshed the unit tests to cover the new heuristic, and reran the demo/end-to-end pipeline successfully.
- Tidied Qwen generation settings to avoid unused temperature flags when running deterministically.
- Validated ROI extraction on a vision-heavy prompt against the real model and hardened prompts so responses stay in structured JSON without verbose preambles.
- Added meta-comment pruning so thinking-mode rambles (e.g., redundant “Step 3” reflections) are dropped while preserving genuine reasoning; confirmed with the official demo image that only meaningful steps remain.
- Authored a metadata-rich `README.md` (with Hugging Face Space front matter) so the deployed Space renders without configuration errors.
- Updated `app.py` to fall back to `demo.queue()` when `concurrency_count` is unsupported, fixing the runtime error seen on Spaces.
- Added ZeroGPU support: cached model/processor globals live on CUDA when available, a `@spaces.GPU`-decorated executor handles pipeline runs, and requirements now include the `spaces` SDK.
- Introduced structured logging for the app (`app.py`) and pipeline execution to trace model loads, cache hits, and Gradio lifecycle events on Spaces.
- Reworked the Gradio UI to show per-step panels with annotated evidence galleries, giving each CoRGI reasoning step its own window alongside the final synthesized answer.
- Preloaded the default Qwen3-VL model/tokenizer at import so Spaces load the GPU weights before serving requests.
- Switched inference to bfloat16, tightened defaults (max steps/regions = 3), added per-stage timers, and moved the @spaces.GPU decorator down to the raw `_chat` call so each generation stays within the 120 s ZeroGPU budget.

## 2024-10-21
- Updated default checkpoints to `Qwen/Qwen3-VL-8B-Thinking` and verified CLI/Gradio/test coverage.
- Exercised the real model to capture thinking-style outputs; added parser fallbacks for textual reasoning/ROI responses and stripped `<think>` tags from answer synthesis.
- Extended unit test suite (reasoning, ROI, client helpers) to cover the new parsing paths and ran `pytest` successfully.

## 2024-10-20
- Added optional integration test (`corgi_tests/test_integration_qwen.py`) gated by `CORGI_RUN_QWEN_INTEGRATION` for running the real Qwen3-VL model on the official demo asset.
- Created runnable example script (`examples/demo_qwen_corgi.py`) to reproduce the Hugging Face demo prompt locally with structured pipeline logging.
- Published Hugging Face Space harness (`app.py`) and deployment helper (`scripts/push_space.sh`) including requirements for ZeroGPU tier.
- Documented cookbook alignment and inference tips (`QWEN_INFERENCE_NOTES.md`).
- Added CLI runner (`corgi.cli`) with formatting helpers plus JSON export; authored matching unittest coverage.
- Implemented Gradio demo harness (`corgi.gradio_app`) with markdown reporting and helper utilities for dependency injection.
- Expanded unit test suite (CLI + Gradio) and ran `pytest corgi_tests` successfully (1 skip when gradio missing).
- Initialized structured project plan and progress log scaffolding.
- Assessed existing modules (`corgi.pipeline`, `corgi.qwen_client`, parsers, tests) to identify pending demo features (CLI + Gradio).
- Confirmed Qwen3-VL will be the single backbone for reasoning, ROI verification, and answer synthesis.

<!-- Template for future updates:
## YYYY-MM-DD
- Summary of change / milestone.
- Follow-up actions.
-->
