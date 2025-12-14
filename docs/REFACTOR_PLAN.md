# 🔧 Kế hoạch Refactor CoRGI Implementation

**Ngày tạo:** December 2024  
**Mục tiêu:** Giảm code duplication, cải thiện architecture, dễ maintain

---

## 📋 Tổng quan

### Trạng thái hiện tại
- 4 entrypoints (`app.py`, `app_qwen_only.py`, `app_v2.py`, `gradio_chatbot_v2.py`)
- 2 inference scripts gần giống nhau (~60% overlap)
- 2 pipeline versions (V1, V2) với types riêng biệt
- 10+ config files
- `gradio_chatbot_v2.py` bypass pipeline

### Mục tiêu sau refactor
- 1 unified entrypoint với CLI options
- 1 inference script thống nhất
- Pipeline với streaming API
- Config đơn giản hóa
- Clean architecture

---

## 🚀 Phase 1: Quick Wins (1-2 ngày)

### Task 1.1: Tạo shared inference utilities
**File:** `corgi/utils/inference_helpers.py`

```python
# Trích xuất functions chung từ inference.py và inference_v2.py

def setup_output_dir(output_dir: Path, create_subdirs: bool = True) -> dict:
    """Create output directory structure."""
    ...

def annotate_image_with_bboxes(
    image: Image.Image,
    bboxes: List[dict],  # {"bbox": [...], "label": "...", "color": "...", "type": "object|text"}
    output_path: Path,
) -> None:
    """Universal bbox annotation (works for V1 and V2)."""
    ...

def save_evidence_crops(
    image: Image.Image,
    evidences: List[dict],  # Generic evidence format
    evidence_dir: Path,
    prefix: str = "evidence",
) -> None:
    """Save cropped evidence regions."""
    ...

def save_results_json(
    result: dict,  # Generic result dict
    output_path: Path,
    pipeline_version: str = "v2",
) -> None:
    """Save JSON results."""
    ...

def save_summary_report(
    image_path: Path,
    question: str,
    result: dict,
    output_path: Path,
    pipeline_version: str = "v2",
) -> None:
    """Save human-readable report."""
    ...
```

**Action:**
1. [ ] Tạo file `corgi/utils/inference_helpers.py`
2. [ ] Copy functions từ `inference_v2.py` (version mới hơn)
3. [ ] Generalize để work với cả V1 và V2
4. [ ] Update `inference.py` và `inference_v2.py` để import từ đây

---

### Task 1.2: Cleanup entrypoints
**Giữ lại:**
- `app.py` → Unified entrypoint (rename từ `app_v2.py`)
- `inference.py` → Unified CLI

**Archive (move to `archive/`):**
- `app_qwen_only.py`
- `app_v2.py` (sau khi merge vào `app.py`)
- `inference_v2.py` (sau khi merge)

**Action:**
1. [ ] Tạo folder `archive/legacy_entrypoints/`
2. [ ] Move `app_qwen_only.py` vào archive
3. [ ] Merge logic từ `app_v2.py` vào `app.py`
4. [ ] Merge `inference_v2.py` vào `inference.py` với flag `--pipeline v1|v2`

---

### Task 1.3: Simplify configs
**Giữ lại (rename):**
```
configs/
├── default.yaml              # ← từ qwen_only_v2.yaml (recommended)
├── multi_model.yaml          # ← từ qwen_florence2_smolvlm2_v2.yaml
├── minimal.yaml              # ← từ qwen_only.yaml (single model, fast)
└── legacy/                   # Archive old configs
    ├── qwen_only.yaml
    ├── florence_qwen.yaml
    └── ...
```

**Action:**
1. [ ] Tạo `configs/legacy/`
2. [ ] Move configs V1 vào legacy
3. [ ] Rename V2 configs thành tên đơn giản
4. [ ] Update README với config recommendations

---

## 🏗️ Phase 2: Architecture Improvements (3-5 ngày)

### Task 2.1: Add Streaming API to Pipeline

**File:** `corgi/core/streaming.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator

class EventType(Enum):
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    STEP_GENERATED = "step_generated"
    EVIDENCE_EXTRACTED = "evidence_extracted"
    ANSWER_READY = "answer_ready"
    ERROR = "error"

@dataclass
class PipelineEvent:
    type: EventType
    phase: str  # "reasoning", "grounding", "evidence", "synthesis"
    data: Any
    step_index: Optional[int] = None
    progress: float = 0.0  # 0.0 to 1.0
```

**File:** `corgi/core/pipeline_v2.py` (update)

```python
class CoRGIPipelineV2:
    def run_streaming(
        self,
        image: Image.Image,
        question: str,
        max_steps: int = 6,
        max_regions: int = 1,
    ) -> Generator[PipelineEvent, None, PipelineResultV2]:
        """
        Streaming pipeline execution.
        
        Yields PipelineEvent for each milestone.
        Returns final PipelineResultV2.
        """
        yield PipelineEvent(EventType.PHASE_START, "reasoning", None)
        
        cot_text, steps = self._run_phase1_2_merged(...)
        yield PipelineEvent(EventType.PHASE_COMPLETE, "reasoning", {
            "cot_text": cot_text,
            "steps": [s.to_dict() for s in steps]
        })
        
        for i, step in enumerate(steps):
            if not step.needs_vision:
                continue
            evidence = self._extract_single_evidence(image, step)
            yield PipelineEvent(
                EventType.EVIDENCE_EXTRACTED, 
                "evidence",
                evidence.to_dict(),
                step_index=step.index,
                progress=(i + 1) / len(steps)
            )
        
        answer, key_evidence, explanation = self._run_phase4_synthesis(...)
        yield PipelineEvent(EventType.ANSWER_READY, "synthesis", {
            "answer": answer,
            "explanation": explanation,
            "key_evidence": [k.to_dict() for k in key_evidence]
        })
```

**Action:**
1. [ ] Tạo `corgi/core/streaming.py` với event types
2. [ ] Add `run_streaming()` method vào `CoRGIPipelineV2`
3. [ ] Update `gradio_chatbot_v2.py` để dùng streaming API thay vì bypass

---

### Task 2.2: Refactor gradio_chatbot_v2.py

**Before:**
```python
# Bypass pipeline, trực tiếp gọi internal methods
cot_text, steps = pipeline._vlm.structured_reasoning_v2(...)
caption = pipeline._vlm.caption_region(...)
```

**After:**
```python
# Use streaming API
def stream_pipeline_execution(image, question, max_steps, max_regions):
    for event in pipeline.run_streaming(image, question, max_steps, max_regions):
        if event.type == EventType.PHASE_COMPLETE and event.phase == "reasoning":
            yield format_reasoning_message(event.data)
        elif event.type == EventType.EVIDENCE_EXTRACTED:
            yield format_evidence_message(event.data)
        elif event.type == EventType.ANSWER_READY:
            yield format_answer_message(event.data)
```

**Action:**
1. [ ] Refactor `stream_pipeline_execution()` để dùng `pipeline.run_streaming()`
2. [ ] Remove trực tiếp calls đến `_vlm.*`
3. [ ] Test streaming UI

---

### Task 2.3: Unify Types (Optional)

**Option A: Backward compatible wrappers**
```python
# corgi/core/types.py
@dataclass
class ReasoningStep:
    index: int
    statement: str
    needs_vision: bool
    need_ocr: bool = False
    # V2 fields with defaults
    need_object_captioning: bool = False
    need_text_ocr: bool = False
    bbox: Optional[List[float]] = None
    
    @property
    def has_bbox(self) -> bool:
        return self.bbox is not None
    
    @property
    def evidence_type(self) -> str:
        if self.need_object_captioning:
            return "object"
        elif self.need_text_ocr:
            return "text"
        return "none"
```

**Action:**
1. [ ] Merge `ReasoningStepV2` fields vào `ReasoningStep`
2. [ ] Add backward compat properties
3. [ ] Update V1 pipeline để work với unified type

---

## 📦 Phase 3: Cleanup & Documentation (1-2 ngày)

### Task 3.1: Archive legacy files

```bash
mkdir -p archive/legacy_entrypoints
mkdir -p archive/legacy_inference
mkdir -p configs/legacy

# Move files
mv app_qwen_only.py archive/legacy_entrypoints/
mv inference_v2.py archive/legacy_inference/  # After merge
mv configs/qwen_vintern.yaml configs/legacy/
mv configs/florence_qwen*.yaml configs/legacy/
# etc.
```

### Task 3.2: Update documentation

1. [ ] Update `README.md` với simplified usage
2. [ ] Update `docs/USAGE_GUIDE.md`
3. [ ] Add migration guide cho users của V1

### Task 3.3: Add tests

1. [ ] Test `inference_helpers.py` functions
2. [ ] Test streaming API
3. [ ] Integration test cho unified entrypoint

---

## 📅 Timeline

```
Week 1:
├── Day 1-2: Phase 1 (Quick Wins)
│   ├── Task 1.1: inference_helpers.py
│   ├── Task 1.2: Cleanup entrypoints
│   └── Task 1.3: Simplify configs

Week 2:
├── Day 3-5: Phase 2 (Architecture)
│   ├── Task 2.1: Streaming API
│   ├── Task 2.2: Refactor chatbot
│   └── Task 2.3: Unify types (optional)

├── Day 6-7: Phase 3 (Cleanup)
│   ├── Task 3.1: Archive files
│   ├── Task 3.2: Update docs
│   └── Task 3.3: Add tests
```

---

## ✅ Checklist

### Phase 1: Quick Wins
- [ ] `corgi/utils/inference_helpers.py` created
- [ ] `inference.py` unified (V1 + V2)
- [ ] Entrypoints consolidated
- [ ] Configs simplified

### Phase 2: Architecture
- [ ] `corgi/core/streaming.py` created
- [ ] `CoRGIPipelineV2.run_streaming()` implemented
- [ ] `gradio_chatbot_v2.py` uses streaming API
- [ ] Types unified (optional)

### Phase 3: Cleanup
- [ ] Legacy files archived
- [ ] Documentation updated
- [ ] Tests added

---

## 🎯 Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Entrypoint files | 4 | 1-2 |
| Inference scripts | 2 | 1 |
| Config files | 10+ | 3-4 + legacy |
| Code duplication | ~60% | <10% |
| Streaming bypass | Yes | No |

---

## 🚨 Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Break existing workflows | Keep legacy files in `archive/` |
| HuggingFace Spaces compatibility | Test trên Spaces trước khi deploy |
| Performance regression | Benchmark trước/sau |
| Missing edge cases | Comprehensive testing |

---

## 📝 Notes

- **Ưu tiên stability** - Giữ backward compat khi có thể
- **Incremental changes** - Commit từng task riêng
- **Test early** - Chạy tests sau mỗi task
- **Document changes** - Update docs song song với code
