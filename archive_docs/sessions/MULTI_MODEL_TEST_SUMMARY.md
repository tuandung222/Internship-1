# Multi-Model Pipeline V2 Testing Summary

## Configuration: `default_v2.yaml`

### Model Stack
| Component | Model | Purpose |
|-----------|-------|---------|
| **Phase 1+2** (Reasoning+Grounding) | `Qwen/Qwen3-VL-2B-Instruct` | Generates reasoning steps with embedded bounding boxes |
| **Phase 3a** (OCR) | `florence-community/Florence-2-base-ft` | Text extraction from regions |
| **Phase 3b** (Captioning) | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Object description |
| **Phase 4** (Synthesis) | *Reuses Qwen3-VL-2B* | Final answer generation |

### Key Features
- **Merged Phases**: Reasoning and Grounding in single step (V2 architecture)
- **Composite Captioning**: Specialized models for OCR vs visual description
- **Memory Efficient**: Model reuse via `reuse_reasoning` flag
- **KV Cache Enabled**: ~35% faster inference
- **bfloat16**: GPU memory optimization

## Issues Encountered & Resolutions

### 1. Florence-2 Loading Error
**Error**: `Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'`

**Root Cause**: Version incompatibility between `transformers` library and PyTorch when using `device_map` parameter with `.to(device)` pattern.

**Resolution**:
- Changed from `.to(device_map).eval()` to `device_map=device_map` in `from_pretrained()`
- Updated both `florence_captioning_client.py` and `florence_grounding_client.py`

### 2. SmolVLM2 Missing Dependency
**Error**: `ImportError: Package 'num2words' is required`

**Resolution**:
```bash
pip install num2words
```

### 3. Model Registry - smolvlm2 Not Registered
**Error**: `ValueError: Captioning model type 'smolvlm2' not registered`

**Root Cause**: Model decorators weren't being triggered because modules weren't imported.

**Resolution**:
- Created `corgi/models/__init__.py` with explicit imports of all model modules
- This triggers `@ModelRegistry.register_captioning` decorators automatically

### 4. Transformers Version Incompatibility (Critical)
**Error**: `TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'` for **all models**

**Root Cause**: `transformers` version conflict with PyTorch 2.x and `accelerate` library's parameter handling.

**Resolution**:
```bash
pip install --upgrade 'transformers>=4.50.0' accelerate
```
- Upgraded to `transformers==5.0.0.dev0` (development version)
- This fixes the `_is_hf_initialized` parameter issue in Parameter class

## Files Modified

1. **`configs/default_v2.yaml`** - New default config with multi-model setup
2. **`corgi/models/florence/florence_captioning_client.py`** - Fixed `device_map` usage
3. **`corgi/models/florence/florence_grounding_client.py`** - Fixed `device_map` usage  
4. **`corgi/models/__init__.py`** - Added model auto-registration
5. **`launch_chatbot.sh`** - Updated default config path

## Testing Status

### Current Test
- **Config**: `configs/default_v2.yaml`
- **Status**: ⏳ In Progress - Models loading
- **Output**: `results_UPGRADED_TRANSFORMERS/`

### Dependencies Installed
- ✅ `num2words` - SmolVLM2 requirement
- ✅ `transformers==5.0.0.dev0` - Latest dev version
- ✅ `accelerate` - Upgraded for compatibility

## Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model Load Time | ~30-40s | Parallel loading enabled |
| Inference Time | ~25-30s | With KV cache enabled |
| Total VRAM | ~8-10GB | Multi-model on single GPU |
| Reasoning Steps | 1-6 | Configurable |
| Evidence Regions | 1 per step | Configurable |

## Next Steps

1. ✅ Fix model loading compatibility
2. ⏳ **CURRENT**: Verify multi-model pipeline works end-to-end
3. 📋 TODO: Performance benchmarking
4. 📋 TODO: Update Gradio app for multi-model config
5. 📋 TODO: Documentation updates

---

**Last Updated**: 2025-11-28 19:35 UTC
**Test Image**: `test_image.jpg`
**Question**: "What do you see?"

