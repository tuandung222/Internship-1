# CoRGI Usage Guide

## Overview

CoRGI (Chain-of-Reasoning with Grounded Image regions) is a visual reasoning pipeline that:
1. **Generates structured reasoning steps** for visual questions
2. **Extracts ROI evidence** using grounding for vision-dependent steps
3. **Synthesizes final answers** grounded in visual evidence

## Quick Start

### 1. Using the CLI

```bash
conda activate pytorch
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Basic usage
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image path/to/image.jpg \
    --question "Your visual question here" \
    --max-steps 3 \
    --max-regions 3

# With JSON output
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image path/to/image.jpg \
    --question "Your question" \
    --json-out output.json

# Using different model
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image path/to/image.jpg \
    --question "Your question" \
    --model-id "Qwen/Qwen3-VL-4B-Instruct"
```

### 2. Using Python API

```python
from PIL import Image
from corgi.pipeline import CoRGIPipeline
from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig

# Load image
image = Image.open("path/to/image.jpg").convert("RGB")

# Create pipeline
config = QwenGenerationConfig(model_id="Qwen/Qwen3-VL-8B-Thinking")
client = Qwen3VLClient(config)
pipeline = CoRGIPipeline(vlm_client=client)

# Run inference
result = pipeline.run(
    image=image,
    question="Your visual question",
    max_steps=3,
    max_regions=3
)

# Access results
print("Answer:", result.answer)
for step in result.steps:
    print(f"Step {step.index}: {step.statement}")
    print(f"  Needs vision: {step.needs_vision}")

for evidence in result.evidence:
    print(f"Evidence for step {evidence.step_index}:")
    print(f"  BBox: {evidence.bbox}")
    print(f"  Description: {evidence.description}")
```

### 3. Using Gradio Interface

```bash
conda activate pytorch
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Launch local Gradio server
PYTHONPATH=$(pwd) python app.py
```

Then open http://localhost:7860 in your browser.

## Configuration Options

### Model Selection

- **Default**: `Qwen/Qwen3-VL-8B-Thinking` (recommended for quality)
- **Alternative**: `Qwen/Qwen3-VL-4B-Instruct` (faster, less VRAM)

Set via `--model-id` (CLI) or `QwenGenerationConfig(model_id=...)` (API).

### Pipeline Parameters

- **max_steps** (default: 3): Maximum reasoning steps to generate
  - Increase for complex reasoning (max 6)
  - Decrease for faster inference (min 1)

- **max_regions** (default: 3): Maximum ROI regions per visual step
  - Increase for detailed visual grounding (max 6)
  - Decrease for faster processing (min 1)

### Generation Parameters

```python
config = QwenGenerationConfig(
    model_id="Qwen/Qwen3-VL-8B-Thinking",
    max_new_tokens=512,        # Max tokens to generate
    temperature=None,           # Deterministic (greedy) decoding
    do_sample=False,           # Disable sampling for reproducibility
)
```

## Example Use Cases

### 1. Counting Objects

```bash
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image examples/dining_table.png \
    --question "How many plates are on the table?" \
    --max-steps 3
```

### 2. Visual Verification

```bash
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image examples/photo.jpg \
    --question "Is anyone wearing a white watch?" \
    --max-steps 2 \
    --max-regions 2
```

### 3. Spatial Reasoning

```bash
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image examples/scene.jpg \
    --question "What is to the left of the TV?" \
    --max-steps 3 \
    --max-regions 4
```

## Understanding Output

### CLI Output Format

```
Question: How many people are in the image?
-- Steps --
[1] Count visible people in the image (needs vision: yes; reason: Visual counting required)
[2] Verify the count (needs vision: yes; reason: Double-check accuracy)
-- Evidence --
Step 1 | bbox=(0.12, 0.34, 0.45, 0.89) | desc: person standing
Step 1 | bbox=(0.56, 0.23, 0.78, 0.91) | desc: person sitting
-- Answer --
Answer: There are 2 people in the image.
```

### Python API Result

```python
result = pipeline.run(...)

# Access components
result.question          # Original question
result.steps            # List[ReasoningStep]
result.evidence         # List[GroundedEvidence]
result.answer           # Final answer string
result.reasoning_log    # Prompt/response for reasoning
result.grounding_logs   # List of ROI extraction logs
result.answer_log       # Prompt/response for synthesis
result.timings          # Performance metrics
result.total_duration_ms  # Total time in milliseconds
```

### JSON Output

```bash
PYTHONPATH=$(pwd) python -m corgi.cli \
    --image image.jpg \
    --question "Your question" \
    --json-out result.json
```

Output structure:
```json
{
  "question": "Your question",
  "steps": [
    {
      "index": 1,
      "statement": "Reasoning step text",
      "needs_vision": true,
      "reason": "Why vision is needed"
    }
  ],
  "evidence": [
    {
      "step_index": 1,
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "description": "Visual evidence",
      "confidence": 0.95
    }
  ],
  "answer": "Final answer",
  "total_duration_ms": 45123.45,
  "timings": [...],
  "reasoning_log": {...},
  "grounding_logs": [...],
  "answer_log": {...}
}
```

## Performance Tips

### Speed Optimization

1. **Use smaller model**: `Qwen/Qwen3-VL-4B-Instruct` is 2x faster
2. **Reduce max_steps**: Use 2-3 steps for simpler questions
3. **Reduce max_regions**: Use 1-2 regions per step
4. **Use GPU**: Ensure CUDA is available

### Quality Optimization

1. **Use thinking model**: `Qwen/Qwen3-VL-8B-Thinking` for better reasoning
2. **Increase max_steps**: Use 4-6 steps for complex reasoning
3. **Increase max_regions**: Use 3-6 regions for detailed grounding
4. **Use bfloat16**: Already enabled by default

## Troubleshooting

### Common Issues

**Q: Statements appear truncated**
- A: Known issue with thinking-mode outputs. Final answer is still correct.
- Workaround: Model sometimes outputs verbose thinking instead of clean JSON.

**Q: Empty evidence list**
- A: Model determined no visual verification needed, or no relevant regions found.
- Solution: Rephrase question to be more specific.

**Q: Slow inference**
- A: Qwen3-VL models are large (8B/4B parameters).
- Solution: Use GPU, reduce max_steps/regions, or use smaller model.

**Q: CUDA out of memory**
- A: Model requires ~16GB VRAM for 8B model.
- Solution: Use `Qwen/Qwen3-VL-4B-Instruct` or increase `--max-shard-size`.

## Advanced Usage

### Custom Prompts

You can customize prompts by modifying the client:

```python
from corgi.qwen_client import DEFAULT_REASONING_PROMPT

# Customize reasoning prompt
custom_prompt = """
Your custom prompt here with {max_steps} placeholder.
Question: {question}
"""

# Modify in qwen_client.py before creating client
```

### Batch Processing

```python
import json
from pathlib import Path
from PIL import Image

images_dir = Path("images/")
questions = [
    "Question 1",
    "Question 2",
    # ...
]

results = []
for img_path, question in zip(images_dir.glob("*.jpg"), questions):
    image = Image.open(img_path).convert("RGB")
    result = pipeline.run(image=image, question=question)
    results.append({
        "image": str(img_path),
        "question": question,
        "answer": result.answer,
        "steps": len(result.steps),
        "evidence": len(result.evidence),
    })

with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integration Testing

```python
import os
os.environ["CORGI_RUN_QWEN_INTEGRATION"] = "1"

import pytest
pytest.main(["-v", "corgi_tests/test_integration_qwen.py"])
```

## Citation

If you use CoRGI in your research, please cite:

```bibtex
@misc{corgi2024,
  title={CoRGI: Chain-of-Reasoning with Grounded Image regions},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/spaces/tuandunghcmut/corgi-qwen3-vl-demo}}
}
```

## Resources

- **Hugging Face Space**: https://huggingface.co/spaces/tuandunghcmut/corgi-qwen3-vl-demo
- **Qwen3-VL Repository**: https://github.com/QwenLM/Qwen3-VL
- **Project Plan**: See PROJECT_PLAN.md
- **Progress Log**: See PROGRESS_LOG.md
- **Inference Notes**: See QWEN_INFERENCE_NOTES.md

