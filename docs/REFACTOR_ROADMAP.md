# 🗺️ Lộ trình Refactor CoRGI - Chi tiết từng bước

**Nguyên tắc:** Dễ → Khó, Ít rủi ro → Nhiều rủi ro, Test ngay sau mỗi bước

---

## 📊 Ma trận Ưu tiên

| Task | Độ khó | Rủi ro | Impact | Thứ tự |
|------|--------|--------|--------|--------|
| Tạo `inference_helpers.py` | 🟢 Thấp | 🟢 Thấp | 🔴 Cao | **#1** |
| Cleanup configs | 🟢 Thấp | 🟢 Thấp | 🟡 TB | **#2** |
| Archive legacy entrypoints | 🟢 Thấp | 🟢 Thấp | 🟡 TB | **#3** |
| Merge inference scripts | 🟡 TB | 🟡 TB | 🔴 Cao | **#4** |
| Merge app entrypoints | 🟡 TB | 🟡 TB | 🟡 TB | **#5** |
| Add streaming API | 🔴 Cao | 🟡 TB | 🔴 Cao | **#6** |
| Refactor chatbot | 🔴 Cao | 🔴 Cao | 🔴 Cao | **#7** |

---

## 🚦 Sprint 1: Zero-Risk Changes (Ngày 1)

### Step 1.1: Tạo `inference_helpers.py` ✅

**Rủi ro:** 🟢 Không (chỉ tạo file mới, không sửa file cũ)

**Thực hiện:**
```bash
# Tạo file mới
touch corgi/utils/inference_helpers.py
```

**Nội dung:** Extract shared functions từ `inference_v2.py`

**Test ngay:**
```bash
# Test import
python -c "from corgi.utils.inference_helpers import setup_output_dir; print('OK')"

# Test function
python -c "
from corgi.utils.inference_helpers import setup_output_dir
from pathlib import Path
paths = setup_output_dir(Path('/tmp/test_corgi_refactor'))
print('Created:', list(paths.keys()))
import shutil; shutil.rmtree('/tmp/test_corgi_refactor')
print('PASSED')
"
```

---

### Step 1.2: Tổ chức lại Configs ✅

**Rủi ro:** 🟢 Không (chỉ move files, giữ nguyên nội dung)

**Thực hiện:**
```bash
# Tạo folder legacy
mkdir -p configs/legacy

# Move V1 configs vào legacy (KHÔNG xóa, chỉ copy)
cp configs/qwen_vintern.yaml configs/legacy/
cp configs/florence_qwen.yaml configs/legacy/
cp configs/florence_qwen_spaces.yaml configs/legacy/
cp configs/qwen_paddleocr_fastvlm.yaml configs/legacy/
cp configs/qwen_paddleocr_smolvlm2.yaml configs/legacy/

# Tạo symlinks cho configs chính
ln -sf qwen_only_v2.yaml configs/default.yaml
ln -sf qwen_florence2_smolvlm2_v2.yaml configs/multi_model.yaml
```

**Test ngay:**
```bash
# Verify symlinks work
python -c "
from corgi.core.config import load_config
config = load_config('configs/default.yaml')
print('default.yaml loaded:', config is not None)
"
```

---

### Step 1.3: Archive Legacy Entrypoints ✅

**Rủi ro:** 🟢 Không (chỉ move, giữ nguyên chức năng)

**Thực hiện:**
```bash
# Tạo archive folder
mkdir -p archive/legacy_entrypoints

# Copy (không move) để giữ backward compat
cp app_qwen_only.py archive/legacy_entrypoints/
```

**Test ngay:**
```bash
# Verify original still works
python -c "from app_qwen_only import demo; print('app_qwen_only.py OK')"

# Verify archive copy exists
ls -la archive/legacy_entrypoints/app_qwen_only.py
```

---

## 🚦 Sprint 2: Low-Risk Refactoring (Ngày 2-3)

### Step 2.1: Update `inference.py` để dùng helpers ✅

**Rủi ro:** 🟡 Thấp (sửa file nhưng có test)

**Thực hiện:**
1. Backup file gốc
2. Import từ `inference_helpers.py`
3. Xóa duplicate code
4. Test

```bash
# Backup
cp inference.py archive/inference_v1_backup.py
```

**Test ngay:**
```bash
# Dry run (không cần GPU)
python inference.py --help

# Test với mock (nếu có)
python -c "
from inference import setup_output_dir, save_summary_report
print('Imports OK')
"

# Full test (cần GPU)
python inference.py \
  --image test_image.jpg \
  --question 'What is in this image?' \
  --config configs/default.yaml \
  --output /tmp/test_inference_v1 \
  --no-crops --no-visualization

# Verify output
ls -la /tmp/test_inference_v1/
cat /tmp/test_inference_v1/summary.txt
```

---

### Step 2.2: Update `inference_v2.py` để dùng helpers ✅

**Rủi ro:** 🟡 Thấp

**Thực hiện:** Tương tự Step 2.1

**Test ngay:**
```bash
# Backup
cp inference_v2.py archive/inference_v2_backup.py

# Test
python inference_v2.py \
  --image test_image.jpg \
  --question 'What is in this image?' \
  --config configs/default.yaml \
  --output /tmp/test_inference_v2

# Compare outputs
diff /tmp/test_inference_v1/summary.txt /tmp/test_inference_v2/summary_v2.txt
```

---

### Step 2.3: Merge inference scripts (Optional) ✅

**Rủi ro:** 🟡 Trung bình

**Thực hiện:**
```python
# inference.py - Unified version
def main():
    parser = argparse.ArgumentParser()
    # ... existing args ...
    parser.add_argument(
        "--pipeline",
        choices=["v1", "v2", "auto"],
        default="auto",
        help="Pipeline version (auto = detect from config)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect pipeline version from config
    if args.pipeline == "auto":
        args.pipeline = "v2" if "v2" in str(args.config) else "v1"
    
    if args.pipeline == "v2":
        from corgi.core.pipeline_v2 import CoRGIPipelineV2 as Pipeline
    else:
        from corgi.core.pipeline import CoRGIPipeline as Pipeline
```

**Test ngay:**
```bash
# Test V1 mode
python inference.py --pipeline v1 --image test_image.jpg --question "..." --output /tmp/test_v1

# Test V2 mode  
python inference.py --pipeline v2 --image test_image.jpg --question "..." --output /tmp/test_v2

# Test auto mode với V2 config
python inference.py --config configs/qwen_only_v2.yaml --image test_image.jpg --question "..." --output /tmp/test_auto
```

---

## 🚦 Sprint 3: Medium-Risk Changes (Ngày 4-5)

### Step 3.1: Tạo Unified App Entrypoint ✅

**Rủi ro:** 🟡 Trung bình

**Thực hiện:** Tạo `app_unified.py` mới (không sửa files cũ)

```python
# app_unified.py
"""
Unified Gradio entrypoint for CoRGI.

Usage:
    python app_unified.py                    # Default V2
    python app_unified.py --pipeline v1      # V1 mode
    python app_unified.py --mode chatbot     # Streaming chatbot
    python app_unified.py --config custom.yaml
"""

import argparse
from corgi.ui.gradio_app import build_demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["v1", "v2"], default="v2")
    parser.add_argument("--mode", choices=["standard", "chatbot"], default="standard")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    
    args = parser.parse_args()
    
    # Select config based on pipeline
    if args.config is None:
        args.config = "configs/default.yaml" if args.pipeline == "v2" else "configs/legacy/qwen_only.yaml"
    
    if args.mode == "chatbot":
        from gradio_chatbot_v2 import demo
    else:
        demo = build_demo(
            default_config=args.config,
            config_filter=args.pipeline,
        )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )

if __name__ == "__main__":
    main()
```

**Test ngay:**
```bash
# Test mỗi mode
python app_unified.py --help
python app_unified.py --pipeline v2 --port 7860 &
sleep 5 && curl http://localhost:7860 && kill %1

python app_unified.py --pipeline v1 --port 7861 &
sleep 5 && curl http://localhost:7861 && kill %1
```

---

### Step 3.2: Update README với hướng dẫn mới ✅

**Rủi ro:** 🟢 Không

**Test:** Review manually

---

## 🚦 Sprint 4: Higher-Risk Architecture (Ngày 6-8)

### Step 4.1: Tạo Streaming API ✅

**Rủi ro:** 🟡 Trung bình (tạo file mới, không sửa pipeline cũ)

**Thực hiện:** Tạo `corgi/core/streaming.py`

**Test ngay:**
```python
# test_streaming.py
from corgi.core.streaming import EventType, PipelineEvent

# Test event creation
event = PipelineEvent(
    type=EventType.PHASE_START,
    phase="reasoning",
    data=None,
    progress=0.0
)
assert event.type == EventType.PHASE_START
print("PASSED: Event creation")

# Test serialization
assert event.to_dict()["type"] == "phase_start"
print("PASSED: Serialization")
```

---

### Step 4.2: Add `run_streaming()` to Pipeline ✅

**Rủi ro:** 🟡 Trung bình (thêm method mới, không sửa `run()` cũ)

**Test ngay:**
```python
# test_pipeline_streaming.py
from PIL import Image
from corgi.core.pipeline_v2 import CoRGIPipelineV2
from corgi.core.streaming import EventType

# Load pipeline (need GPU)
# ... setup code ...

# Test streaming
events = list(pipeline.run_streaming(image, question))
assert any(e.type == EventType.PHASE_START for e in events)
assert any(e.type == EventType.ANSWER_READY for e in events)
print(f"PASSED: Got {len(events)} events")
```

---

### Step 4.3: Refactor `gradio_chatbot_v2.py` ✅

**Rủi ro:** 🔴 Cao (sửa UI code)

**Chuẩn bị:**
```bash
# Backup
cp gradio_chatbot_v2.py archive/gradio_chatbot_v2_backup.py
```

**Test ngay:**
```bash
# Launch và test manually
python gradio_chatbot_v2.py --config configs/default.yaml

# Test với browser
# 1. Upload image
# 2. Ask question
# 3. Verify streaming works
# 4. Check console for errors
```

---

## 📋 Checklist Tổng hợp

### Sprint 1: Zero-Risk ✅
- [ ] Step 1.1: `inference_helpers.py` created
  - [ ] Test: Import OK
  - [ ] Test: `setup_output_dir()` works
- [ ] Step 1.2: Configs organized
  - [ ] Test: `default.yaml` loads
  - [ ] Test: Legacy configs still work
- [ ] Step 1.3: Legacy entrypoints archived
  - [ ] Test: Original files still work
  - [ ] Test: Archive copies exist

### Sprint 2: Low-Risk ✅
- [ ] Step 2.1: `inference.py` uses helpers
  - [ ] Test: CLI help works
  - [ ] Test: Full inference works
- [ ] Step 2.2: `inference_v2.py` uses helpers
  - [ ] Test: CLI help works
  - [ ] Test: Full inference works
- [ ] Step 2.3: Unified inference (optional)
  - [ ] Test: `--pipeline v1` works
  - [ ] Test: `--pipeline v2` works
  - [ ] Test: Auto-detect works

### Sprint 3: Medium-Risk ✅
- [ ] Step 3.1: `app_unified.py` created
  - [ ] Test: V1 mode launches
  - [ ] Test: V2 mode launches
  - [ ] Test: Chatbot mode launches
- [ ] Step 3.2: README updated
  - [ ] Test: Instructions are clear

### Sprint 4: Higher-Risk ✅
- [ ] Step 4.1: `streaming.py` created
  - [ ] Test: Event types work
  - [ ] Test: Serialization works
- [ ] Step 4.2: `run_streaming()` added
  - [ ] Test: Events generated correctly
  - [ ] Test: Final result same as `run()`
- [ ] Step 4.3: Chatbot refactored
  - [ ] Test: Streaming works in UI
  - [ ] Test: No regressions

---

## 🔄 Rollback Plan

Mỗi step đều có rollback:

| Step | Rollback Command |
|------|------------------|
| 1.1 | `rm corgi/utils/inference_helpers.py` |
| 1.2 | `rm -rf configs/legacy; rm configs/default.yaml configs/multi_model.yaml` |
| 1.3 | `rm -rf archive/legacy_entrypoints` |
| 2.1 | `cp archive/inference_v1_backup.py inference.py` |
| 2.2 | `cp archive/inference_v2_backup.py inference_v2.py` |
| 3.1 | `rm app_unified.py` |
| 4.3 | `cp archive/gradio_chatbot_v2_backup.py gradio_chatbot_v2.py` |

---

## 🚨 Stop Conditions

Dừng refactor nếu:
1. ❌ Bất kỳ test nào fail
2. ❌ Pipeline inference cho kết quả khác trước
3. ❌ Gradio UI không launch được
4. ❌ HuggingFace Spaces deploy fail

---

## 📅 Timeline Chi tiết

```
Ngày 1 (Sprint 1):
├── 09:00-10:00: Step 1.1 + Test
├── 10:00-11:00: Step 1.2 + Test
└── 11:00-12:00: Step 1.3 + Test

Ngày 2-3 (Sprint 2):
├── Day 2 AM: Step 2.1 + Test
├── Day 2 PM: Step 2.2 + Test
└── Day 3: Step 2.3 + Test

Ngày 4-5 (Sprint 3):
├── Day 4: Step 3.1 + Test
└── Day 5: Step 3.2 + Review

Ngày 6-8 (Sprint 4):
├── Day 6: Step 4.1 + Test
├── Day 7: Step 4.2 + Test
└── Day 8: Step 4.3 + Full Integration Test
```

---

## ✅ Bắt đầu với Step 1.1

Sẵn sàng bắt đầu? Confirm và tôi sẽ thực hiện **Step 1.1: Tạo `inference_helpers.py`**
