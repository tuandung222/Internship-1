# CoRGI Documentation

Complete documentation for the CoRGI (Chain of Reasoning with Grounded Insights) project.

## 📚 Documentation Structure

### Core Documentation
- **[INFERENCE_README.md](INFERENCE_README.md)** - Inference scripts usage guide

### Pipeline V2
- **[pipeline_v2/](pipeline_v2/)** - Pipeline V2 architecture and testing
  - `ARCHITECTURE_REVIEW_V2.md` - V1 vs V2 comparison
  - `PIPELINE_V2_SUMMARY.md` - V2 implementation details
  - `V2_TEST_PROGRESS.md` - Testing progress
  - `TEST_SESSION_SUMMARY.md` - Test results

### Optimization
- **[optimization/](optimization/)** - Performance optimization docs
  - `OPTIMIZATION_ANALYSIS.md` - Comprehensive optimization analysis
  - `OPTIMIZATION_QUESTIONS_ANSWERED.md` - KV Cache, bfloat16, image encoding
  - `KV_CACHE_OPTIMIZATION_DONE.md` - KV Cache implementation
  - `enable_kv_cache.py` - Automation script

### UI/Chatbot
- **[ui/](ui/)** - Gradio UI documentation
  - `CHATBOT_UI_SUMMARY.md` - Chatbot implementation details
  - `GRADIO_CHATBOT_V2_README.md` - Usage guide

### Testing
- **[testing/](testing/)** - Testing documentation
  - `TESTING_COMPLETE.md` - Test results
  - `REAL_PIPELINE_TEST_IMPLEMENTATION.md` - Test implementation
  - `README.md` - Testing guide

### Session Summaries
- **[sessions/](sessions/)** - Development session summaries
  - `MULTI_MODEL_TEST_SUMMARY.md` - Multi-model testing
  - `FINAL_SESSION_SUMMARY.md` - Overall progress
  - `CLEANUP_SUMMARY.md` - Project organization

### Guides
- **[guides/](guides/)** - User guides
  - `GETTING_STARTED_WITH_TESTING.md`
  - `QUICK_START.md`

### Other Categories
- **[architecture/](architecture/)** - Architecture documentation
- **[bugfixes/](bugfixes/)** - Bug fixes and resolutions
- **[development/](development/)** - Development notes
- **[florence2/](florence2/)** - Florence-2 specific docs
- **[history/](history/)** - Historical documentation
- **[paper/](paper/)** - Research paper references

## 🚀 Quick Links

### Getting Started
1. [Main README](../README.md) - Project overview
2. [Quick Start Guide](guides/QUICK_START.md)
3. [Inference Guide](INFERENCE_README.md)

### For Developers
1. [Pipeline V2 Architecture](pipeline_v2/ARCHITECTURE_REVIEW_V2.md)
2. [Optimization Guide](optimization/OPTIMIZATION_ANALYSIS.md)
3. [Testing Guide](testing/README.md)

### For Users
1. [Chatbot UI Guide](ui/GRADIO_CHATBOT_V2_README.md)
2. [Getting Started with Testing](guides/GETTING_STARTED_WITH_TESTING.md)

## 📊 Key Metrics & Results

- **Pipeline V2**: 35% faster with KV Cache
- **Model Stack**: Qwen-2B + Florence-2-base + SmolVLM2-500M
- **VRAM Usage**: ~8-10GB for multi-model setup
- **Inference Time**: ~25-30s per query

## 🔗 External Resources

- [Hugging Face Space](https://huggingface.co/spaces/your-space)
- [GitHub Repository](https://github.com/your-repo)
- [Research Paper](paper/)

---

**Last Updated**: 2025-11-28
**CoRGI Version**: 2.0
