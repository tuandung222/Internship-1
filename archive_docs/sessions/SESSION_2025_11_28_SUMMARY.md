# Session Summary - November 28, 2025

## 🎯 Main Objective
Test Pipeline V2 with multi-model configuration:
- **Qwen3-VL-2B-Instruct** (Reasoning + Grounding + Synthesis)
- **Florence-2-base-ft** (OCR)
- **SmolVLM2-500M** (Captioning)

## ✅ Completed Tasks

### 1. Configuration Setup
- ✅ Created `configs/default_v2.yaml` as default multi-model config
- ✅ Updated `launch_chatbot.sh` to use default_v2.yaml
- ✅ Configured Florence-2-base-ft instead of Florence-2-base

### 2. Model Loading Fixes

#### Issue #1: Florence-2 `device_map` Error
**Error**: `Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'`

**Root Cause**: Using `.to(device_map).eval()` pattern incompatible with community checkpoints

**Fix**:
```python
# Before (WRONG):
model = ModelClass.from_pretrained(...).to(device_map).eval()

# After (CORRECT):
model = ModelClass.from_pretrained(..., device_map=device_map).eval()
```

**Files Modified**:
- `corgi/models/florence/florence_captioning_client.py`
- `corgi/models/florence/florence_grounding_client.py`

#### Issue #2: SmolVLM2 Missing Dependency
**Error**: `ImportError: Package 'num2words' is required`

**Fix**:
```bash
pip install num2words
```

#### Issue #3: Model Registration
**Error**: `ValueError: Captioning model type 'smolvlm2' not registered`

**Root Cause**: Model decorators not triggered because modules weren't imported

**Fix**: Created `corgi/models/__init__.py`:
```python
# Import all model modules to trigger @ModelRegistry.register decorators
from . import florence
from . import qwen
from . import smolvlm  # Critical!
from . import fastvlm
from . import paddle
from . import vintern
```

#### Issue #4: Transformers Version Incompatibility (CRITICAL)
**Error**: `TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'` for ALL models

**Root Cause**: `transformers` library version conflict with PyTorch 2.x and `accelerate`

**Fix**:
```bash
pip install --upgrade 'transformers>=4.50.0' accelerate
```

**Result**: Upgraded to `transformers==5.0.0.dev0`

### 3. Documentation Organization
Reorganized cluttered root directory:

**Before**: 11 .md files in root  
**After**: Only `README.md` in root

**New Structure**:
```
docs/
├── README.md                 # Documentation index (NEW)
├── optimization/             # Performance docs (NEW)
│   ├── OPTIMIZATION_ANALYSIS.md
│   ├── OPTIMIZATION_QUESTIONS_ANSWERED.md
│   ├── KV_CACHE_OPTIMIZATION_DONE.md
│   └── enable_kv_cache.py
├── ui/                       # UI/Chatbot docs (NEW)
│   ├── CHATBOT_UI_SUMMARY.md
│   └── GRADIO_CHATBOT_V2_README.md
├── sessions/                 # Session summaries (NEW)
│   ├── MULTI_MODEL_TEST_SUMMARY.md
│   ├── FINAL_SESSION_SUMMARY.md
│   ├── CLEANUP_SUMMARY.md
│   └── DOCUMENTATION_CLEANUP_SUMMARY.md
├── pipeline_v2/             # V2 architecture
├── testing/                 # Test docs
└── guides/                  # User guides
```

**Files Moved**: 10  
**Files Deleted**: 1 (GIT_COMMIT_MESSAGE.md)  
**Total Docs**: 70 markdown files

## 📊 Current Status

### Multi-Model Inference Test
- **Status**: ⏳ In Progress (10+ minutes)
- **Config**: `configs/default_v2.yaml`
- **Models**: 
  - ✅ Loaded: Qwen3-VL-2B (reasoning)
  - ✅ Loaded: SmolVLM2-500M (caption)
  - ✅ Loaded: Florence-2-base-ft (OCR)
- **GPU VRAM**: 12.5 GB in use
- **Process PID**: 2878260
- **Output Dir**: `results_UPGRADED_TRANSFORMERS/`
- **Issue**: Running longer than expected (~10 min), no results generated yet

### Possible Issues
1. First-time model download/caching may be slow
2. Florence-2 community checkpoint may have compatibility issues
3. Process may be stuck (needs investigation)

## 🔧 Technical Details

### Dependencies Installed
- ✅ `num2words` - SmolVLM2 requirement
- ✅ `transformers==5.0.0.dev0` - Development version for compatibility
- ✅ `accelerate` - Upgraded

### Model Configuration
```yaml
reasoning:
  model_type: qwen_instruct
  model_id: Qwen/Qwen3-VL-2B-Instruct
  device: cuda:5
  torch_dtype: bfloat16

captioning:
  model_type: composite
  ocr:
    model_type: florence2
    model_id: florence-community/Florence-2-base-ft
    torch_dtype: bfloat16
  caption:
    model_type: smolvlm2
    model_id: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    torch_dtype: float16

synthesis:
  reuse_reasoning: true
```

### Code Changes
**Files Modified**: 5
1. `corgi/models/__init__.py` - Created, added imports
2. `corgi/models/florence/florence_captioning_client.py` - Fixed device_map
3. `corgi/models/florence/florence_grounding_client.py` - Fixed device_map
4. `configs/default_v2.yaml` - Created
5. `launch_chatbot.sh` - Updated default

**Documentation Created**: 4
1. `docs/README.md` - Documentation index
2. `docs/sessions/MULTI_MODEL_TEST_SUMMARY.md`
3. `docs/sessions/DOCUMENTATION_CLEANUP_SUMMARY.md`
4. `monitor_inference.sh` - Monitoring script

## 📋 Next Steps

### Immediate
1. ⏳ **WAIT**: Let inference complete or timeout
2. 🔍 **DEBUG**: If stuck, investigate process state
3. ✅ **VERIFY**: Check results once generated

### Follow-up
1. Performance benchmarking with multi-model setup
2. Update Gradio app for default_v2.yaml
3. Test with different question types
4. Document performance metrics

### Optional
1. Try Florence-2-large-ft for better accuracy
2. Test with FastVLM instead of SmolVLM2
3. Implement model selection in Gradio UI

## 📈 Performance Expectations

| Metric | Expected | Notes |
|--------|----------|-------|
| Model Load Time | 30-40s | Parallel loading |
| Inference Time | 25-30s | With KV cache |
| Total VRAM | 8-10GB | Multi-model |
| First Run | 2-3 min | Model caching |

## 🎓 Lessons Learned

1. **Always use `device_map` parameter** in `from_pretrained()` - don't chain `.to(device)`
2. **Check model-specific dependencies** before testing (num2words for SmolVLM2)
3. **Module imports trigger decorators** - need `__init__.py` with explicit imports for registry pattern
4. **Transformers version matters** - use latest dev version for cutting-edge models
5. **Organization matters** - 70 docs need proper structure

## 🔗 References

- [Florence-2 Official](https://huggingface.co/florence-community/Florence-2-base-ft)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
- [Transformers v5 Docs](https://huggingface.co/docs/transformers/main)

---

**Session Duration**: ~3 hours  
**Issues Resolved**: 4 critical, 1 major  
**Documentation Organized**: 70 files  
**Status**: Testing in progress  

**Next Session**: Verify multi-model results and benchmark performance

