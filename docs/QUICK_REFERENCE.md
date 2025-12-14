# CoRGI Quick Reference

## 🚀 Quick Start

### Inference (CLI)

```bash
# Default (V2 pipeline, auto-detect from config)
python inference.py --image photo.jpg --question "What is this?" --output results/

# Explicit pipeline version
python inference.py --pipeline v2 --image photo.jpg --question "What is this?"
python inference.py --pipeline v1 --image photo.jpg --question "What is this?"

# Batch processing
python inference.py --batch questions.txt --output batch_results/

# With custom config
python inference.py --config configs/qwen_florence2_smolvlm2_v2.yaml --image photo.jpg --question "..."
```

### Gradio UI (Web Interface)

```bash
# Default (V2 pipeline, standard mode)
python app_unified.py

# V2 with chatbot streaming UI
python app_unified.py --mode chatbot

# V1 pipeline
python app_unified.py --pipeline v1

# Custom port and share link
python app_unified.py --port 7861 --share

# HuggingFace Spaces mode
python app_unified.py --spaces
```

---

## 📁 File Structure

### Entry Points (Use These)

| File | Purpose |
|------|---------|
| `inference.py` | CLI inference (V1 + V2 unified) |
| `app_unified.py` | Gradio UI (V1 + V2, standard + chatbot) |

### Deprecated (Backward Compat Only)

| File | Replacement |
|------|-------------|
| `inference_v2.py` | `python inference.py --pipeline v2` |
| `app_v2.py` | `python app_unified.py --pipeline v2` |
| `app_qwen_only.py` | `python app_unified.py` |
| `app.py` | `python app_unified.py` |

---

## ⚙️ Configs

### Recommended

| Config | Description |
|--------|-------------|
| `configs/qwen_only_v2.yaml` | V2, single Qwen model (fast) |
| `configs/qwen_florence2_smolvlm2_v2.yaml` | V2, multi-model (best quality) |

### Legacy (V1)

| Config | Description |
|--------|-------------|
| `configs/legacy/qwen_only.yaml` | V1, single model |
| `configs/legacy/florence_qwen.yaml` | V1, multi-model |

---

## 🔧 Pipeline Versions

### V2 (Recommended)
- Merged Phase 1+2 (faster)
- Smart routing: OCR OR Caption
- Inline bounding boxes from model

### V1 (Legacy)
- Separate 3 stages
- Always OCR + Caption
- External grounding model

---

## 📊 Output Files

After inference, results are saved to:

```
output_dir/
├── results.json      # Full results (JSON)
├── summary.txt       # Human-readable report
├── images/
│   └── original.jpg  # Input image
├── visualizations/
│   └── annotated.jpg # Image with bboxes
└── evidence/
    └── *.jpg         # Cropped evidence regions
```

---

## 🔍 Examples

### Single Image Analysis

```bash
python inference.py \
    --image test_image.jpg \
    --question "How many people are in this image?" \
    --output results/people_count/
```

### Document Understanding

```bash
python inference.py \
    --image invoice.png \
    --question "What is the total amount?" \
    --config configs/qwen_florence2_smolvlm2_v2.yaml \
    --output results/invoice/
```

### Interactive Demo

```bash
# Launch web UI
python app_unified.py --port 7860

# Open browser: http://localhost:7860
```

---

## 🔄 Streaming API

For real-time, progressive display of pipeline results:

### Using run_streaming()

```python
from corgi.core import CoRGIPipelineV2, StreamEventType
from corgi.models.factory import VLMClientFactory

# Create pipeline
client = VLMClientFactory.create_from_config(config)
pipeline = CoRGIPipelineV2(vlm_client=client)

# Stream execution
for event in pipeline.run_streaming(image, question):
    if event.type == StreamEventType.STEP:
        print(f"Step: {event.data['statement']}")
    elif event.type == StreamEventType.EVIDENCE:
        print(f"Evidence: {event.data}")
    elif event.type == StreamEventType.ANSWER:
        print(f"Answer: {event.data['answer']}")
```

### Event Types

| Event | Description |
|-------|-------------|
| `PIPELINE_START` | Pipeline execution begins |
| `PHASE_START/END` | Phase lifecycle |
| `COT_TEXT` | Chain-of-thought generated |
| `STEP` | Reasoning step with optional bbox |
| `BBOX` | Bounding box from fallback grounding |
| `EVIDENCE` | OCR or caption extracted |
| `ANSWER` | Final answer generated |
| `KEY_EVIDENCE` | Key evidence item |
| `WARNING/ERROR` | Non-fatal warnings or errors |
| `PIPELINE_END` | Pipeline execution complete |

### Chatbot Mode

```bash
# Run streaming chatbot UI
python app_unified.py --mode chatbot
python gradio_chatbot_v2.py  # Direct launch
```
