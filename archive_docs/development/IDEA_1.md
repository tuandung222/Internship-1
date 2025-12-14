# CoRGI Prototype (3-Stage) — Qwen-VL × GroundingDINO × Text Reranker

Repo mẫu triển khai pipeline CoRGI 3-stage:

* **Stage 1**: Qwen-VL sinh *reasoning chain* từ ảnh + câu hỏi.
* **Stage 2 (VEVM)**: phân loại mức cần thị giác → Grounding DINO chọn RoI → Qwen-VL mô tả RoI → **Text cross-encoder re-ranker** so khớp *step ↔ mô tả RoI* → xác minh bằng chứng.
* **Stage 3**: Tổng hợp đáp án dựa trên câu hỏi, bước suy luận và bằng chứng đã kiểm chứng.

---

## Cấu trúc thư mục

```
📦 corgi-prototype
├─ 📁 corgi
│  ├─ __init__.py
│  ├─ pipeline.py
│  ├─ vlm_qwen.py
│  ├─ detector_grounding_dino.py
│  ├─ reranker_text.py
│  ├─ relevance.py
│  └─ utils.py
├─ 📁 scripts
│  └─ corgi_demo.py
├─ 📁 configs
│  └─ config.example.yaml
├─ README.md
├─ requirements.txt
└─ LICENSE
```

---

## Cài đặt

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (khuyến nghị) đăng nhập HF nếu cần model private
# huggingface-cli login
```

> GPU khuyến nghị: 16–24GB VRAM cho `Qwen/Qwen2.5-VL-7B-Instruct`. Có thể đổi sang bản nhỏ hơn nếu cần.

---

## Chạy thử

```bash
python -m scripts.corgi_demo \
  --image ./your_image.jpg \
  --question "What is the person holding?" \
  --out result.json
```

Kết quả `result.json` gồm: `steps` (chuỗi lý luận), `evidence` (bằng chứng từng bước: box, tag, mô tả, điểm), và `answer`.

---

## Nâng cấp nhanh

* Thay model Qwen-VL khác (ví dụ kích thước nhỏ hơn) bằng cách chỉnh `vlm_model_id` trong `CoRGIPipeline`.
* Điều chỉnh `alpha` (trọng số kết hợp điểm re-rank text và điểm detection) trong `pipeline.py`.
* Bật cache mô tả RoI theo (ảnh, box) nếu muốn giảm chi phí (điểm gợi ý trong code).

---

## Mã nguồn

### `requirements.txt`

```txt
# Core
torch>=2.2
transformers>=4.43
accelerate
pillow
opencv-python
numpy
einops

# Grounding DINO (HF zero-shot OD)
# transformers đã đủ, dùng AutoModelForZeroShotObjectDetection

# Text reranker (cross-encoder, đa ngôn ngữ)
FlagEmbedding>=1.2.10
```

### `LICENSE`

```text
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `README.md`

````markdown
# CoRGI Prototype (Qwen-VL × GroundingDINO × Text Reranker)

## Overview
Pipeline 3-stage theo CoRGI: Reasoning → Visual Evidence Verification (VEVM) → Answer Synthesis.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

## Quickstart

```bash
python -m scripts.corgi_demo --image ./your_image.jpg --question "What is the person holding?" --out result.json
```

## Notes

* Đổi model Qwen-VL nếu thiếu VRAM.
* Điều chỉnh `alpha` trong `pipeline.py` để cân bằng text rerank vs detection score.
* Nếu làm OCR: thêm PaddleOCR/Tesseract và gộp text OCR vào mô tả RoI trước khi re-rank.

````

### `configs/config.example.yaml`
```yaml
# Example config (chưa được load tự động — tham khảo để chỉnh tay trong code)
vlm_model_id: Qwen/Qwen2.5-VL-7B-Instruct
gdino_model_id: IDEA-Research/grounding-dino-base
max_steps: 4
roi_per_step: 4
alpha_fuse: 0.8  # trọng số ưu tiên điểm text rerank
box_threshold: 0.25
text_threshold: 0.25
````

### `corgi/__init__.py`

```python
__all__ = [
    "CoRGIPipeline",
    "QwenVL",
    "GroundingDINO",
    "TextReranker",
    "RelevanceClassifier",
]

from .pipeline import CoRGIPipeline
from .vlm_qwen import QwenVL
from .detector_grounding_dino import GroundingDINO
from .reranker_text import TextReranker
from .relevance import RelevanceClassifier
```

### `corgi/utils.py`

```python
from typing import List
from PIL import Image


def crop_box(pil_img: Image.Image, box_xyxy: List[float], pad_ratio: float = 0.03) -> Image.Image:
    w, h = pil_img.size
    x1, y1, x2, y2 = box_xyxy
    pw, ph = pad_ratio * w, pad_ratio * h
    x1 = int(max(0, x1 - pw)); y1 = int(max(0, y1 - ph))
    x2 = int(min(w, x2 + pw)); y2 = int(min(h, y2 + ph))
    return pil_img.crop((x1, y1, x2, y2))


def minmax(xs):
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]
```

### `corgi/vlm_qwen.py`

```python
from typing import List, Any, Dict
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class QwenVL:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    @torch.inference_mode()
    def _chat(self, image: Image.Image | None, prompt: str, max_new_tokens=256) -> str:
        msgs = [{"role": "user", "content": []}]
        if image is not None:
            msgs[0]["content"].append({"type": "image", "image": image})
        msgs[0]["content"].append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        if image is not None:
            inputs["images"] = [image]
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return text.split("assistant\n")[-1].strip()

    def generate_reasoning_steps(self, image: Image.Image, question: str, num_steps: int = 4) -> List[str]:
        sys = (
            "You are a meticulous vision-language reasoner. "
            f"Given the image and the question, produce {num_steps} numbered reasoning steps ONLY, no final answer. "
            "Each step must be a short sentence grounded on observable cues or necessary world knowledge."
        )
        text = f"{sys}\nQuestion: {question}\nFormat:\n1) ...\n2) ...\n3) ...\n4) ..."
        out = self._chat(image, text, max_new_tokens=256)
        steps = []
        for line in out.splitlines():
            line = line.strip()
            if len(line) < 3:
                continue
            if line[0].isdigit():
                line = line.split(")", 1)[-1] if ")" in line[:4] else line.split(".", 1)[-1]
            steps.append(line.strip(" -:"))
        return [s for s in steps if s][:num_steps]

    def describe_roi(self, crop: Image.Image, step_text: str) -> str:
        prompt = (
            "Describe precisely what is visible in this region in 1-2 sentences, "
            "focusing only on directly observable details relevant to: "
            f"\"{step_text}\".\nAvoid speculation."
        )
        return self._chat(crop, prompt, max_new_tokens=128)

    def synthesize_answer(self, image: Image.Image, question: str,
                          steps: List[str], evidences: List[Dict[str, Any]]) -> str:
        ev_lines = []
        for i, ev in enumerate(evidences, 1):
            desc = ev.get("description", "")
            box = ev.get("box", None)
            tag = ev.get("tag", "")
            ev_lines.append(f"[EV{i}] tag={tag} box={box} :: {desc}")
        context = "\n".join([f"Step {i}: {s}" for i, s in enumerate(steps, 1)]) + "\n" + "\n".join(ev_lines)
        prompt = (
            "You are a careful solver. Using the verified visual evidence, answer the question.\n"
            "Rules: Prefer direct observations from [EV*]. If evidence contradicts a step, trust evidence.\n"
            f"Question: {question}\nReasoning steps and evidence:\n{context}\n"
            "Final answer (one sentence):"
        )
        return self._chat(image, prompt, max_new_tokens=128)
```

### `corgi/detector_grounding_dino.py`

```python
from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import AutoProcessor as HFProcessor, AutoModelForZeroShotObjectDetection


class GroundingDINO:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base",
                 box_thresh: float = 0.25, text_thresh: float = 0.25):
        self.processor = HFProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.inference_mode()
    def detect(self, image: Image.Image, phrases: List[str], topk: int = 5) -> List[Dict[str, Any]]:
        inputs = self.processor(images=image, text=phrases, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh,
            target_sizes=[image.size[::-1]]
        )[0]
        dets = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            dets.append({
                "score": float(score.item()),
                "tag": phrases[label],
                "box": [float(x) for x in box.tolist()],
            })
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:topk]
        return dets
```

### `corgi/reranker_text.py`

```python
from typing import List, Tuple
import torch

try:
    from FlagEmbedding import FlagReranker  # BAAI/bge-reranker-*
except Exception as e:
    FlagReranker = None


class TextReranker:
    """
    Cross-encoder text re-ranker: chấm điểm mức liên quan giữa (reasoning step) và (mô tả ROI).
    Mặc định dùng BAAI/bge-reranker-v2-m3 (đa ngôn ngữ, tốt cho tiếng Việt).
    """
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool | None = None):
        if FlagReranker is None:
            raise ImportError("Please `pip install FlagEmbedding` to use TextReranker.")
        if use_fp16 is None:
            use_fp16 = torch.cuda.is_available()
        self.reranker = FlagReranker(model_id, use_fp16=use_fp16)

    @torch.inference_mode()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            return [float(scores)]
        return [float(s) for s in scores]
```

### `corgi/relevance.py`

```python
class RelevanceClassifier:
    """
    Heuristic gate xác định bước nào cần kiểm chứng thị giác.
    Có thể thay bằng MLP nhẹ nếu cần huấn luyện.
    """
    VISUAL_HINTS = [
        "color", "left", "right", "top", "bottom", "near", "holding", "wearing", "text", "logo",
        "number", "behind", "front", "background", "foreground", "shape", "size", "count", "read",
    ]

    def needs_vision(self, step_text: str) -> float:
        s = step_text.lower()
        base = 0.2
        hits = sum(1 for k in self.VISUAL_HINTS if k in s)
        return min(1.0, base + 0.2 * hits)  # 0.2..1.0
```

### `corgi/pipeline.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image

from .vlm_qwen import QwenVL
from .detector_grounding_dino import GroundingDINO
from .reranker_text import TextReranker
from .relevance import RelevanceClassifier
from .utils import crop_box, minmax


@dataclass
class Evidence:
    step_idx: int
    tag: str
    box: List[float]
    score: float
    description: str


class CoRGIPipeline:
    """
    3-Stage CoRGI:
      - Stage 1: Qwen-VL sinh reasoning chain
      - Stage 2 (VEVM): gate → GroundingDINO → mô tả RoI → text re-rank → chọn evidence
      - Stage 3: tổng hợp đáp án bằng Qwen-VL
    """

    def __init__(self,
                 vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 gdino_model_id: str = "IDEA-Research/grounding-dino-base",
                 alpha_fuse: float = 0.8,
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.25):
        self.vlm = QwenVL(vlm_model_id)
        self.det = GroundingDINO(gdino_model_id, box_thresh=box_threshold, text_thresh=text_threshold)
        self.rel = RelevanceClassifier()
        self.rerank = TextReranker("BAAI/bge-reranker-v2-m3")
        self.alpha = alpha_fuse

    def _extract_tags(self, step_text: str, image: Image.Image) -> List[str]:
        ask = (
            "List 3-6 short object phrases from the image that would help verify this statement:\n"
            f"\"{step_text}\"\n"
            "Return as a comma-separated list, no numbering, lowercase."
        )
        out = self.vlm._chat(image, ask, max_new_tokens=64)
        tags = [t.strip().strip(".") for t in out.split(",")]
        tags = [t for t in tags if 1 <= len(t.split()) <= 5 and t not in {"image", "photo", "scene"}]
        return tags[:6] if tags else ["person", "object", "text"]

    def run(self, image: Image.Image, question: str, max_steps: int = 4, roi_per_step: int = 4) -> Dict[str, Any]:
        # Stage 1: chain of thought steps (no answer)
        steps = self.vlm.generate_reasoning_steps(image, question, num_steps=max_steps)

        evidences: List[Evidence] = []
        for i, step in enumerate(steps, 1):
            rel_w = self.rel.needs_vision(step)
            if rel_w < 0.4:
                continue

            # Stage 2.a — detect with tags
            tags = self._extract_tags(step, image)
            dets = self.det.detect(image, tags, topk=roi_per_step * 3)

            # Stage 2.b — describe RoIs
            cand = []  # [(det, crop, desc)]
            for d in dets:
                crop = crop_box(image, d["box"])
                desc = self.vlm.describe_roi(crop, step)
                cand.append((d, crop, desc))

            # Stage 2.c — text re-rank (step ↔ desc)
            pairs = [(step, desc) for _, _, desc in cand]
            txt_scores = self.rerank.score_pairs(pairs) if pairs else []

            # Stage 2.d — fuse with detection score
            det_scores = [c[0]["score"] for c in cand]
            det_scores_n = minmax(det_scores)
            fused = [self.alpha * ts + (1 - self.alpha) * ds for ts, ds in zip(txt_scores, det_scores_n)]

            # Stage 2.e — select top-k evidences
            order = sorted(range(len(cand)), key=lambda idx: fused[idx], reverse=True)[:roi_per_step]
            for idx in order:
                d, _, desc = cand[idx]
                evidences.append(Evidence(
                    step_idx=i, tag=d["tag"], box=d["box"], score=float(fused[idx]), description=desc
                ))

        evid_json = [e.__dict__ for e in evidences]
        final_answer = self.vlm.synthesize_answer(image, question, steps, evid_json)
        return {"question": question, "steps": steps, "evidence": evid_json, "answer": final_answer}
```

### `scripts/corgi_demo.py`

```python
import argparse, json
from PIL import Image
from corgi.pipeline import CoRGIPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default="corgi_result.json")
    parser.add_argument("--vlm", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--per_step", type=int, default=4, help="RoI per step")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--box_thresh", type=float, default=0.25)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    pipe = CoRGIPipeline(
        vlm_model_id=args.vlm,
        gdino_model_id=args.gdino,
        alpha_fuse=args.alpha,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
    )
    result = pipe.run(img, args.question, max_steps=args.max_steps, roi_per_step=args.per_step)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Final answer:\n", result["answer"])
    print(f"Saved details to {args.out}")


if __name__ == "__main__":
    main()
```

---

## Gợi ý commit đầu tiên

```bash
echo "# CoRGI Prototype" > README.md
git init && git add . && git commit -m "init: corgi 3-stage prototype (Qwen-VL × GroundingDINO × text reranker)"
```

## Troubleshooting

* **CUDA OOM**: hạ `--vlm` xuống model nhỏ hơn hoặc giảm `--max_steps`, `--per_step`.
* **FlagEmbedding not found**: `pip install FlagEmbedding`.
* **Grounding DINO chậm / ít box**: tăng `--box_thresh` thấp hơn, thêm tag giàu thông tin.
* **Kết quả lan man**: tăng `alpha` để ưu tiên text re-rank, hoặc siết prompt trong `vlm_qwen.describe_roi`.
