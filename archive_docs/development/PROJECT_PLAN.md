# CoRGI Custom Demo — Project Plan

## Context
- **Objective**: ship a runnable CoRGI demo (CLI + Gradio) powered entirely by Qwen3-VL for structured reasoning, ROI evidence extraction, and answer synthesis.
- **Scope**: stay within the `corgi_custom` package, reuse Qwen3-VL cookbooks where possible, keep dependency footprint minimal (no extra detectors/rerankers).
- **Environment**: Conda env `pytorch`, default VLM `Qwen/Qwen3-VL-8B-Thinking`.

## Milestones
| Status | Milestone | Notes |
| --- | --- | --- |
| ✅ | Core pipeline skeleton (dataclasses, parsers, Qwen client wrappers) | Already merged in repo. |
| ✅ | Project documentation & progress tracking scaffolding | Plan + progress log committed. |
| ✅ | CLI runner that prints step-by-step pipeline output | Supports overrides + JSON export. |
| ✅ | Gradio demo mirroring CLI functionality | Blocks UI with markdown report messaging. |
| ✅ | Automated tests for new modules | CLI + Gradio helpers covered with unit tests. |
| ✅ | HF Space deployment automation | Bash script + app harness for zerogpu Spaces. |
| 🟡 | Final verification (unit tests, smoke instructions) | Document how to run `pytest` and the demos. |

## Work Breakdown Structure
1. **Docs & Tracking**  
   - [x] Finalize plan and progress log templates.  
   - [x] Document environment setup expectations.
2. **Pipeline UX**  
   - [x] Implement CLI entrypoint (`corgi.cli:main`).  
   - [x] Provide structured stdout for steps/evidence/answer.  
   - [x] Allow optional JSON dump for downstream tooling.
3. **Interactive Demo**  
   - [x] Build Gradio app harness (image upload + question textbox).  
   - [ ] Stream progress (optional) and display textual reasoning/evidence.  
   - [x] Handle model loading errors gracefully.
4. **Testing & Tooling**  
   - [x] Add fixture-friendly helpers to avoid heavy model loads in tests.  
   - [x] Write unit tests for CLI argument parsing + formatting.  
   - [ ] Add regression test for pipeline serialization.
5. **Docs & Hand-off**  
   - [ ] Update README/demo instructions.  
   - [ ] Provide sample command sequences for CLI/Gradio.  
   - [ ] Capture open risks & future enhancements.
6. **Deployment & Ops**  
   - [x] Add Hugging Face Space entrypoint (`app.py`).  
   - [x] Write deployment helper script (`scripts/push_space.sh`).  
   - [ ] Add automated checklists/logs for Space updates.

## Risks & Mitigations
- **Model loading latency / VRAM** → expose config knobs and mention 4B fallback.
- **Parsing drift from Qwen outputs** → keep parser tolerant; add debug flag to dump raw responses.
- **Test runtime** → mock Qwen client via fixtures; avoid loading real model in unit tests.

## Progress Tracking
- Refer to `PROGRESS_LOG.md` for dated status updates.
- Update milestone table whenever a deliverable completes.
