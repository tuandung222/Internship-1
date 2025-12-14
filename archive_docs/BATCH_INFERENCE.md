# CoRGI Batch Inference Examples

## Format của batch file

Tạo file `batch_input.txt` với format:
```
# Mỗi dòng: đường_dẫn_ảnh|câu_hỏi
/path/to/image1.jpg|What is in this image?
/path/to/image2.jpg|How many people are there?
/path/to/image3.jpg|What color is the car?
```

## Single Image Inference

```bash
# Basic usage
python inference.py \
    --image tests/test_images/sample.jpg \
    --question "What is happening in the image?" \
    --output results/single_test

# With specific config
python inference.py \
    --image sample.jpg \
    --question "Describe the scene" \
    --config configs/qwen_paddleocr_fastvlm.yaml \
    --output results/test1

# Production mode (faster)
CORGI_LOG_LEVEL=WARNING python inference.py \
    --image sample.jpg \
    --question "What objects are visible?" \
    --output results/test2
```

## Batch Inference

```bash
# Process multiple images from file
python inference.py \
    --batch images.txt \
    --output results/batch_001

# With production optimizations
CORGI_LOG_LEVEL=WARNING python inference.py \
    --batch images.txt \
    --config configs/qwen_only.yaml \
    --output results/batch_production

# Skip crops and visualizations (faster)
python inference.py \
    --batch images.txt \
    --no-crops \
    --no-visualization \
    --output results/batch_fast
```

## Output Structure

Sau khi chạy, kết quả sẽ được lưu trong structure sau:

```
results/
├── images/
│   └── original.jpg              # Ảnh gốc
├── evidence/
│   ├── evidence_step0_region0.jpg  # Crop của từng evidence
│   ├── evidence_step0_region1.jpg
│   └── ...
├── visualizations/
│   └── annotated.jpg             # Ảnh với bounding boxes
├── logs/
│   └── (future: log files)
├── results.json                   # Kết quả đầy đủ (JSON)
└── summary.txt                    # Báo cáo dễ đọc (text)
```

### results.json

Chứa toàn bộ thông tin chi tiết:
```json
{
  "metadata": {
    "timestamp": "2025-11-23T20:15:00",
    "total_duration_ms": 4523.5
  },
  "question": "What is in the image?",
  "answer": "The image shows...",
  "reasoning_steps": [...],
  "evidence": [...],
  "key_evidence": [...],
  "timings": [...]
}
```

### summary.txt

Báo cáo dễ đọc:
```
================================================================================
CoRGI Pipeline Inference Report
================================================================================

Image: /path/to/image.jpg
Question: What is in the image?
Total Duration: 4.52s

--------------------------------------------------------------------------------
FINAL ANSWER
--------------------------------------------------------------------------------
The image shows a busy street with several cars and pedestrians...

--------------------------------------------------------------------------------
REASONING STEPS (3 steps)
--------------------------------------------------------------------------------
Step 0: Identify the main objects in the scene
  - Needs Vision: True
  - Need OCR: False
...
```

## Batch Processing

### Tạo batch file

```bash
# Tạo danh sách ảnh tự động
find images/ -name "*.jpg" | while read img; do
    echo "$img|What is in this image?" >> batch.txt
done
```

### Batch file với câu hỏi khác nhau

```
# batch_diverse.txt
images/street.jpg|How many cars are there?
images/person.jpg|What is the person wearing?
images/food.jpg|What type of food is this?
images/document.jpg|What does the text say?
```

### Processing batch

```bash
python inference.py --batch batch_diverse.txt --output results/batch_diverse
```

### Batch output structure

```
results/batch_diverse/
├── result_0001_street/
│   ├── images/
│   ├── evidence/
│   ├── visualizations/
│   ├── results.json
│   └── summary.txt
├── result_0002_person/
│   └── ...
├── result_0003_food/
│   └── ...
└── batch_summary.json  # Tổng kết toàn bộ batch
```

## Advanced Usage

### Custom pipeline parameters

```bash
python inference.py \
    --image sample.jpg \
    --question "..." \
    --max-steps 8 \      # Tăng số bước reasoning
    --max-regions 10 \   # Tăng số vùng evidence
    --output results/detailed
```

### Pipeline với optimizations

```bash
# Set environment variables
export CORGI_LOG_LEVEL=WARNING
export CORGI_MAX_IMAGE_SIZE=1024

# Run inference
python inference.py --batch images.txt --output results/optimized
```

### Processing rất nhiều ảnh

```bash
# Chia batch thành chunks nhỏ hơn
split -l 100 large_batch.txt batch_chunk_

# Process từng chunk
for chunk in batch_chunk_*; do
    python inference.py \
        --batch $chunk \
        --output results/$(basename $chunk) \
        --no-crops  # Skip crops để tiết kiệm disk
done
```

## Performance Tips

1. **Sử dụng production logging:**
   ```bash
   CORGI_LOG_LEVEL=WARNING python inference.py ...
   ```

2. **Skip unnecessary outputs:**
   ```bash
   python inference.py --no-crops --no-visualization ...
   ```

3. **Batch processing:**
   - Tốt hơn là chạy 1 batch file lớn thay vì nhiều single images
   - Pipeline chỉ load 1 lần

4. **GPU memory:**
   - Nếu OOM, giảm `CORGI_MAX_IMAGE_SIZE`:
   ```bash
   CORGI_MAX_IMAGE_SIZE=768 python inference.py ...
   ```

## Integration với Scripts Khác

### Python API

```python
from inference import run_inference
from corgi.core.pipeline import CoRGIPipeline
from corgi.core.config import CoRGiConfig
from corgi.models.factory import VLMClientFactory
from pathlib import Path

# Load pipeline
config = CoRGiConfig.from_yaml("configs/qwen_only.yaml")
client = VLMClientFactory.create_from_config(config)
pipeline = CoRGIPipeline(vlm_client=client)

# Run inference
result = run_inference(
    image_path=Path("image.jpg"),
    question="What is this?",
    pipeline=pipeline,
    output_dir=Path("results/test"),
)

print(f"Answer: {result.answer}")
```

### Bash automation

```bash
#!/bin/bash
# process_all.sh

for img in images/*.jpg; do
    output_dir="results/$(basename $img .jpg)"
    python inference.py \
        --image "$img" \
        --question "Describe everything you see" \
        --output "$output_dir"
done
```

## Troubleshooting

### Error: Config file not found
```bash
# Kiểm tra config path
ls configs/

# Dùng absolute path
python inference.py --config /absolute/path/to/config.yaml ...
```

### Error: Image not found (batch mode)
```bash
# Kiểm tra paths trong batch file
cat batch.txt

# Sử dụng absolute paths
sed -i 's|images/|/absolute/path/images/|g' batch.txt
```

### Slow inference
```bash
# Enable optimizations
export CORGI_LOG_LEVEL=WARNING
export CORGI_DISABLE_COMPILE=0

# Reduce image size
export CORGI_MAX_IMAGE_SIZE=768
```
