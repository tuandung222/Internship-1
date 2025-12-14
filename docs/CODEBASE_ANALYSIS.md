# Phân tích Mã nguồn CoRGI Implementation

**Tài liệu tham khảo:** [CoRGI Paper](./paper/corgi_paper_original.md) | [Paper Review](./paper/corgi_paper_review.md)

---

## 📋 Tổng quan

### Mục đích của các File Python Script

| File | Mục đích | Pipeline Version |
|------|----------|-----------------|
| `app_qwen_only.py` | Gradio entrypoint chỉ dùng Qwen (HuggingFace Spaces) | V1 |
| `app_v2.py` | Gradio entrypoint cho Pipeline V2 | V2 |
| `app.py` | Gradio entrypoint với multi-model config (Qwen + PaddleOCR + FastVLM) | V1 |
| `inference.py` | CLI batch inference script | V1 |
| `inference_v2.py` | CLI batch inference script | V2 |
| `gradio_chatbot_v2.py` | Gradio chatbot-style UI với streaming | V2 |

---

## 🔄 Hai Kiểu Pipeline

### Pipeline V1 (Legacy) - `pipeline.py`

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE V1 (pipeline.py) - 3 Stages RIÊNG BIỆT                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Image + Question                                                             │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  STAGE 1: Structured Reasoning      │  ← VLM Call #1 (Qwen)                │
│  │  structured_reasoning()             │                                      │
│  │  Output: List[ReasoningStep]        │                                      │
│  │    - statement                      │                                      │
│  │    - needs_vision: bool             │                                      │
│  │    - need_ocr: bool                 │                                      │
│  └─────────────────────────────────────┘                                      │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  STAGE 2: Visual Grounding          │  ← VLM Call #2 (per step!)           │
│  │  extract_step_evidence()            │                                      │
│  │  OR extract_all_steps_evidence()    │                                      │
│  │  Output: List[GroundedEvidence]     │                                      │
│  │    - bbox + description + ocr_text  │                                      │
│  └─────────────────────────────────────┘                                      │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  STAGE 3: Answer Synthesis          │  ← VLM Call #3 (Qwen)                │
│  │  synthesize_answer()                │                                      │
│  │  Output: answer, key_evidence       │                                      │
│  └─────────────────────────────────────┘                                      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Đặc điểm:**
- **3 stages riêng biệt** → nhiều VLM calls
- **Grounding tách biệt** → có thể dùng model chuyên biệt (Florence-2)
- **Evidence extraction đồng nhất** → cả OCR và Caption cho mọi step

---

### Pipeline V2 (Current) - `pipeline_v2.py`

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE V2 (pipeline_v2.py) - MERGED Phase 1+2 + Smart Routing              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Image + Question                                                             │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  PHASE 1+2 MERGED: Reasoning +      │  ← VLM Call #1 (Qwen)                │
│  │  Grounding IN ONE CALL              │                                      │
│  │  structured_reasoning_v2()          │                                      │
│  │                                     │                                      │
│  │  Output: (cot_text, List[ReasoningStepV2])                                 │
│  │    - statement                      │                                      │
│  │    - need_object_captioning: bool   │  ← EXPLICIT flag                     │
│  │    - need_text_ocr: bool            │  ← EXPLICIT flag                     │
│  │    - bbox: Optional[List[float]]    │  ← INLINE bbox!                      │
│  └─────────────────────────────────────┘                                      │
│        │                                                                      │
│        ▼ (fallback_grounding nếu thiếu bbox)                                  │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  PHASE 3: Smart Evidence Routing    │  ← VLM Calls (per step, parallel)    │
│  │                                     │                                      │
│  │  IF need_object_captioning:         │                                      │
│  │    → SmolVLM2.caption_region()      │  (Object description)                │
│  │                                     │                                      │
│  │  ELIF need_text_ocr:                │                                      │
│  │    → Florence-2.ocr_region()        │  (Text extraction)                   │
│  │                                     │                                      │
│  │  ELSE: Skip (pure reasoning)        │                                      │
│  └─────────────────────────────────────┘                                      │
│        │                                                                      │
│        ▼                                                                      │
│  ┌─────────────────────────────────────┐                                      │
│  │  PHASE 4: Answer Synthesis          │  ← VLM Call (reuse Qwen)             │
│  │  synthesize_answer()                │                                      │
│  └─────────────────────────────────────┘                                      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Đặc điểm:**
- **Merged Phase 1+2** → giảm 35% latency
- **Smart Routing** → OCR HOẶC Caption (không cả hai)
- **Inline bbox** → model tự output bbox trong reasoning
- **Evidence type flags** → explicit decision

---

## 📊 So sánh với Paper Gốc

### Paper CoRGI (3 Stages)

| Stage | Component | Description |
|-------|-----------|-------------|
| **Stage 1** | Reasoning Chain Generation | VLM tạo reasoning steps |
| **Stage 2** | VEVM (Visual Evidence Verification Module) | 3 sub-modules: Relevance Classifier + RoI Selector (Grounding DINO) + VLM Evidence Extractor |
| **Stage 3** | Final Answer Synthesis | VLM synthesize với evidence |

### Implementation So sánh

| Aspect | Paper | V1 Implementation | V2 Implementation |
|--------|-------|-------------------|-------------------|
| **Reasoning** | VLM (Qwen/LLaVA/Gemma) | Qwen3-VL | Qwen3-VL |
| **Relevance Classifier** | Trained MLP classifier | Implicit via `needs_vision` flag | Implicit via `need_object/need_text` flags |
| **RoI Selection** | Grounding DINO | VLM grounding (per-step) hoặc Florence-2 | Inline từ Qwen + fallback grounding |
| **Evidence Extraction** | VLM captioning | VLM captioning + OCR | Smart routing: SmolVLM2 OR Florence-2 |
| **Importance Scoring** | Sigmoid mapping (0-100%) | ❌ Chưa implement | ❌ Chưa implement |

---

## 🔧 Models và Components

### V1 Pipeline (`inference.py`, `app.py`, `app_qwen_only.py`)

```yaml
# Cấu hình ví dụ: qwen_paddleocr_fastvlm.yaml
reasoning:
  model: Qwen3-VL-2B-Instruct   # Phase 1: Reasoning
  
grounding:
  model: Qwen3-VL-2B            # Phase 2: Grounding (reuse hoặc model riêng)
  # HOẶC: Florence-2 (chuyên biệt cho grounding)

captioning:
  model: FastVLM-1.5B           # Phase 2: Evidence extraction - Caption
  
ocr:
  model: PaddleOCR-VL           # Phase 2: Evidence extraction - OCR

synthesis:
  model: Qwen3-VL-2B            # Phase 3: Synthesis (reuse reasoning model)
```

### V2 Pipeline (`inference_v2.py`, `app_v2.py`, `gradio_chatbot_v2.py`)

```yaml
# Cấu hình ví dụ: qwen_florence2_smolvlm2_v2.yaml
reasoning:
  model: Qwen3-VL-4B-Instruct   # Phase 1+2 MERGED: Reasoning + Grounding
  use_v2_prompt: true
  
grounding:
  reuse_reasoning: true          # Reuse reasoning model cho fallback

captioning:
  composite: true
  ocr: Florence-2-large-ft       # Smart routing: Text evidence
  caption: SmolVLM2-1.7B         # Smart routing: Object evidence

synthesis:
  reuse_reasoning: true          # Reuse reasoning model
```

---

## 🎯 Phân tích Chi tiết từng File

### 1. `app_qwen_only.py` (37 lines)
**Mục đích:** HuggingFace Spaces entrypoint đơn giản nhất

```python
# Key points:
- Import build_demo từ corgi.ui.gradio_app
- Sử dụng DEFAULT_QWEN_CONFIG
- Queue với concurrency_count=1 cho Spaces
```

**Ưu điểm:** Đơn giản, dễ deploy
**Hạn chế:** Chỉ dùng Qwen, không multi-model

---

### 2. `app_v2.py` (53 lines)
**Mục đích:** V2 Pipeline Gradio app

```python
# Key points:
- config_filter="v2" - chỉ hiện V2 configs
- Port 7861 (khác V1 là 7860)
- Title và description cho V2 features
```

**Ưu điểm:** Clean, focus vào V2
**Hạn chế:** Chưa có streaming như `gradio_chatbot_v2.py`

---

### 3. `app.py` (84 lines)
**Mục đích:** Multi-model entrypoint với config fallback

```python
# Priority order:
1. qwen_paddleocr_fastvlm.yaml  # New pipeline
2. qwen_vintern.yaml            # Legacy
3. florence_qwen_spaces.yaml    # Fallback

# Key points:
- huggingface_hub version check
- Auto-upgrade if needed
- Flexible config selection
```

**Ưu điểm:** Robust với nhiều fallback options
**Hạn chế:** Phức tạp, nhiều conditional logic

---

### 4. `inference.py` (550 lines)
**Mục đích:** CLI batch inference cho V1 Pipeline

```python
# Features:
- Single image hoặc batch processing
- annotate_image_with_evidence()  # Vẽ bbox lên ảnh
- save_evidence_crops()           # Cắt và lưu evidence regions
- save_detailed_results()         # JSON output
- save_summary_report()           # Human-readable report

# Usage:
python inference.py --image test.jpg --question "..." --output results/
python inference.py --batch questions.txt --output results/
```

**Ưu điểm:** Comprehensive output, well-structured
**Hạn chế:** V1 only, code duplication với inference_v2.py

---

### 5. `inference_v2.py` (623 lines)
**Mục đích:** CLI batch inference cho V2 Pipeline

```python
# V2-specific features:
- annotate_image_with_evidence_v2()  # Color-coded by evidence type
  - Green: Object evidence
  - Red: Text evidence
- V2 stats tracking:
  - bbox_from_phase1_count
  - object_evidence_count
  - text_evidence_count
```

**Ưu điểm:** V2 features, better tracking
**Hạn chế:** ~70% code duplicate với inference.py

---

### 6. `gradio_chatbot_v2.py` (453 lines)
**Mục đích:** Streaming chatbot-style UI

```python
# Features:
- stream_pipeline_execution() - Generator yielding (chat_history, image)
- Phase-by-phase streaming:
  1. Show reasoning CoT
  2. Show steps with bbox
  3. Stream evidence extraction per step
  4. Show final answer
- Progressive bbox visualization
```

**Ưu điểm:** Best UX, real-time feedback
**Hạn chế:** Trực tiếp gọi internal methods (_vlm.structured_reasoning_v2), bypasses pipeline

---

## ⚠️ Phức tạp và Nhập nhằng

### 1. **Code Duplication (Cao)**

```
inference.py vs inference_v2.py:
├── setup_output_dir()           # 100% giống nhau
├── annotate_image_*()           # ~80% giống nhau
├── save_evidence_crops*()       # ~80% giống nhau
├── save_detailed_results*()     # ~60% giống nhau
├── save_summary_report*()       # ~60% giống nhau
└── batch_inference*()           # ~70% giống nhau
```

**Vấn đề:** Khó maintain, dễ diverge

---

### 2. **Inconsistent Naming**

| Concept | V1 | V2 |
|---------|----|----|
| Evidence | `GroundedEvidence` | `GroundedEvidenceV2` |
| Step | `ReasoningStep` | `ReasoningStepV2` |
| Result | `PipelineResult` | `PipelineResultV2` |
| OCR flag | `need_ocr` | `need_text_ocr` |
| Caption flag | `needs_vision` | `need_object_captioning` |

---

### 3. **Entrypoint Overlap**

```
app.py         → Multi-model V1 (HuggingFace Spaces)
app_qwen_only.py → Single-model V1 (HuggingFace Spaces)
app_v2.py      → V2 (local)
gradio_chatbot_v2.py → V2 with streaming (local)
```

**Vấn đề:** 4 entrypoints khác nhau, khó biết nên dùng cái nào

---

### 4. **gradio_chatbot_v2.py Bypasses Pipeline**

```python
# Trực tiếp gọi internal methods:
cot_text, steps = pipeline._vlm.structured_reasoning_v2(...)
caption = pipeline._vlm.caption_region(...)
ocr_text = pipeline._vlm.ocr_region(...)
```

**Vấn đề:** Không đi qua pipeline, logic duplicate, dễ out-of-sync

---

### 5. **Config Complexity**

```
configs/
├── default.yaml                    # V1
├── default_v2.yaml                 # V2
├── qwen_only.yaml                  # V1, single model
├── qwen_only_v2.yaml               # V2, single model
├── florence_qwen.yaml              # V1, multi-model
├── florence_qwen_spaces.yaml       # V1, spaces-optimized
├── qwen_florence2_smolvlm2_v2.yaml # V2, multi-model
├── qwen_paddleocr_fastvlm.yaml     # V1, newest multi-model
├── qwen_paddleocr_smolvlm2.yaml    # V1 variant
├── qwen_vintern.yaml               # V1, legacy
└── ... 
```

**Vấn đề:** Quá nhiều configs, khó biết nên dùng cái nào

---

## 🛠️ Hướng Refactor & Reorganize

### 1. **Consolidate Entrypoints**

```python
# BEFORE: 4 files
app.py, app_qwen_only.py, app_v2.py, gradio_chatbot_v2.py

# AFTER: 1 unified file với CLI options
app.py --mode standard|chatbot --pipeline v1|v2 --config <config>
```

---

### 2. **Unify Inference Scripts**

```python
# BEFORE: 2 files (inference.py, inference_v2.py)

# AFTER: 1 file với auto-detect
inference.py --config configs/qwen_only_v2.yaml  # Auto V2
inference.py --config configs/qwen_only.yaml     # Auto V1

# Shared utilities
corgi/utils/inference_helpers.py:
  - setup_output_dir()
  - annotate_image()
  - save_results()
```

---

### 3. **Streaming via Pipeline (Not Bypass)**

```python
# BEFORE (gradio_chatbot_v2.py):
cot_text, steps = pipeline._vlm.structured_reasoning_v2(...)

# AFTER: Pipeline exposes streaming API
class CoRGIPipelineV2:
    def run_streaming(self, image, question, max_steps, max_regions):
        """Generator that yields intermediate results."""
        yield PipelineEvent(type="phase1_start", data=None)
        cot_text, steps = self._run_phase1_2_merged(...)
        yield PipelineEvent(type="phase1_complete", data={"cot": cot_text, "steps": steps})
        
        for evidence in self._run_phase3_streaming(...):
            yield PipelineEvent(type="evidence", data=evidence)
        
        answer, key_evidence, explanation = self._run_phase4_synthesis(...)
        yield PipelineEvent(type="complete", data={"answer": answer, ...})
```

---

### 4. **Simplify Configs**

```
configs/
├── qwen_single.yaml          # One Qwen model for everything (V2)
├── qwen_multi_model.yaml     # Qwen + Florence + SmolVLM (V2)
├── legacy/                   # V1 configs for backward compat
│   ├── qwen_only.yaml
│   └── florence_qwen.yaml
└── README.md                 # Config documentation
```

---

### 5. **Type Unification**

```python
# Option A: Unified types with version field
@dataclass
class ReasoningStep:
    index: int
    statement: str
    needs_vision: bool
    need_ocr: bool = False
    need_object_captioning: bool = False
    need_text_ocr: bool = False
    bbox: Optional[List[float]] = None
    
    @property
    def has_bbox(self) -> bool:
        return self.bbox is not None

# Option B: Keep separate but with shared base
class ReasoningStepBase(Protocol):
    index: int
    statement: str
    needs_vision: bool
```

---

### 6. **Implement Missing Paper Features**

```python
# Từ paper: Importance Scoring
class RelevanceClassifier:
    """Trained MLP để classify step relevance."""
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def classify(self, step_text: str) -> tuple[bool, float]:
        """Returns (is_visual, importance_score)."""
        logit = self.model(step_text)
        sigmoid = torch.sigmoid(logit)
        is_visual = sigmoid > THRESHOLD
        importance = piecewise_mapping(sigmoid) if is_visual else 0.0
        return is_visual, importance

# Usage in VEVM
for step in steps:
    is_visual, importance = classifier.classify(step.statement)
    if not is_visual:
        continue  # Skip non-visual steps
    # Extract evidence with importance prefix
    evidence = f"importance: {importance:.0%}% | {extracted_text}"
```

---

## 📁 Proposed Directory Structure

```
corgi_custom/
├── app.py                          # Unified Gradio entrypoint
├── inference.py                    # Unified CLI inference
├── corgi/
│   ├── core/
│   │   ├── pipeline.py             # Unified pipeline (V1/V2 via config)
│   │   ├── types.py                # Unified types
│   │   ├── streaming.py            # Streaming support
│   │   └── config.py
│   ├── models/
│   │   ├── base.py                 # Base VLM client protocol
│   │   ├── qwen/
│   │   ├── florence/
│   │   ├── smolvlm/
│   │   └── factory.py
│   ├── verification/               # NEW: Paper's VEVM components
│   │   ├── relevance_classifier.py
│   │   ├── roi_selector.py
│   │   └── evidence_extractor.py
│   ├── ui/
│   │   ├── gradio_app.py           # Unified Gradio UI
│   │   └── streaming_handler.py    # Chatbot streaming logic
│   └── utils/
│       ├── inference_helpers.py    # Shared inference utilities
│       └── ...
├── configs/
│   ├── default.yaml                # Recommended config
│   ├── minimal.yaml                # Single model, fast
│   ├── full.yaml                   # Multi-model, best quality
│   └── legacy/
└── docs/
```

---

## ✅ Summary

### Điểm mạnh của Implementation

1. **Modular architecture** - Dễ swap models
2. **V2 optimization** - Merged phases, smart routing
3. **Comprehensive output** - JSON, visualization, reports
4. **Multiple UI options** - Standard và streaming chatbot

### Điểm cần cải thiện

1. **Code duplication** - inference.py vs inference_v2.py (~60% overlap)
2. **Entrypoint fragmentation** - 4 app files
3. **Config sprawl** - 10+ config files
4. **Missing paper features** - Relevance classifier, importance scoring
5. **Bypass anti-pattern** - gradio_chatbot_v2 bypasses pipeline

### Ưu tiên Refactor

| Priority | Task | Impact |
|----------|------|--------|
| 🔴 High | Unify inference.py + inference_v2.py | Reduce maintenance |
| 🔴 High | Add streaming API to pipeline | Clean architecture |
| 🟡 Medium | Consolidate entrypoints | Better UX |
| 🟡 Medium | Simplify configs | Reduce confusion |
| 🟢 Low | Implement importance scoring | Match paper |
| 🟢 Low | Train relevance classifier | Better accuracy |
