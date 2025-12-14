# CoRGI — Tóm tắt bài báo & cách hiện thực trong repo

> **Mục tiêu**: Tóm tắt ngắn gọn ý tưởng CoRGI (Verified Chain-of-Thought Reasoning with Visual Grounding) và mô tả cách ta đã triển khai 3-stage bằng **Qwen‑VL × GroundingDINO × Text Reranker**.

---

## TL;DR

* **Vấn đề**: VLM sinh lý luận nhiều bước (*reasoning chain*) có thể **hallucinate** nếu không ràng buộc bằng chứng thị giác.
* **Ý tưởng CoRGI**: tách suy luận thành 3 giai đoạn. Mỗi bước suy luận đều có **bằng chứng hình ảnh đã xác minh** trước khi tổng hợp đáp án cuối.
* **Cách làm của repo**:

  1. **Qwen‑VL** sinh *reasoning steps* →
  2. **VEVM**: (gate) → **Grounding DINO** chọn **RoI** theo *tags* → **Qwen‑VL** mô tả RoI → **Text cross‑encoder re‑ranker** so khớp *step ↔ mô tả RoI* → lưu **evidence** →
  3. **Qwen‑VL** tổng hợp đáp án dựa trên *question + steps + evidences*.

---

## 1) CoRGI: khung 3 giai đoạn

```
[Image, Question]
      │
      ▼
(1) Reasoning Chain Generation (VLM)  →  r1, r2, …, rn  (chưa trả lời)
      │
      ▼
(2) Visual Evidence Verification Module (VEVM)
    • Relevance gate: bước nào cần nhìn ảnh?
    • RoI selection: tìm vùng liên quan (detector/grounding)
    • Evidence extraction/checking: mô tả vùng (text) + xác minh
      │
      ▼
(3) Answer Synthesis  ←  Question + Steps + Verified Evidences
```

**Lợi ích**

* **Giảm ảo giác**: mỗi bước phải bám vào vật thể/vùng cụ thể.
* **Gợi giải thích**: log được *step ↔ RoI ↔ mô tả* phục vụ audit.
* **Module hoá**: thay thế từng khối (detector, reranker, VLM) dễ dàng.

---

## 2) Mapping sang repo của bạn

* **Stage 1 — Reasoning**
  File: `corgi/vlm_qwen.py` → `QwenVL.generate_reasoning_steps()`
  Prompt sinh danh sách bước dạng `1) … 2) … 3) …`.

* **Stage 2 — VEVM**

  * **Gate**: `corgi/relevance.py` (heuristic), có thể thay bằng MLP nhẹ.
  * **Tag extractor**: `CoRGIPipeline._extract_tags()` (hỏi Qwen‑VL để lấy vài cụm danh từ/obj).
  * **RoI selection**: `corgi/detector_grounding_dino.py` dùng **GroundingDINO** (zero‑shot OD + grounding theo câu/tag).
  * **Evidence extraction**: `QwenVL.describe_roi()` mô tả chi tiết vùng theo *step*.
  * **Text re‑ranker**: `corgi/reranker_text.py` dùng **BAAI/bge‑reranker‑v2‑m3** (cross‑encoder) để chấm **(step, mô tả RoI)** → chọn top‑k.
  * **Fusion**: `pipeline.py` hợp nhất `text_score` (ưu tiên) với `det_score` bằng hệ số `alpha_fuse`.

* **Stage 3 — Synthesis**
  File: `corgi/vlm_qwen.py` → `QwenVL.synthesize_answer()`
  Đầu vào: Question + Steps + danh sách Evidence `[tag, box, desc, score]`.

* **Orchestrator**: `corgi/pipeline.py` → lớp `CoRGIPipeline` chạy full 3‑stage.

* **Demo CLI**: `scripts/corgi_demo.py`.

---

## 3) Prompt & cấu hình quan trọng

### 3.1 Prompt mẫu

* **Sinh bước suy luận**

```text
You are a meticulous vision-language reasoner.
Given the image and the question, produce N numbered reasoning steps ONLY, no final answer.
Each step must be short and grounded on observable cues or necessary world knowledge.
```

* **Gợi tag cho detector**

```text
List 3–6 short object phrases from the image that would help verify this statement: "{step}"
Return as a comma-separated list, no numbering, lowercase.
```

* **Mô tả RoI**

```text
Describe precisely what is visible in this region in 1–2 sentences, focusing only on details relevant to: "{step}".
Avoid speculation.
```

* **Tổng hợp đáp án**

```text
Using the verified visual evidence [EV*], answer the question in one sentence.
Prefer direct observations from [EV*]. If conflict arises, trust evidence.
```

### 3.2 Các nút chỉnh (knobs)

* `max_steps`: số bước reasoning (mặc định 4).
* `roi_per_step`: số RoI giữ lại sau re‑rank (mặc định 4).
* `alpha_fuse`: trọng số ưu tiên điểm text re‑rank (0.8 khuyến nghị).
* `box_threshold`, `text_threshold` của GroundingDINO.
* Chọn model **Qwen‑VL** phù hợp VRAM; có thể thay bằng VLM khác.

---

## 4) Vì sao dùng **text re‑ranking** cho “classifier”?

* Phù hợp mục tiêu *“đoạn text reasoning có liên quan vùng ảnh không”*: so khớp **(step)** với **(mô tả RoI)** thuần văn bản.
* Cross‑encoder (bge‑reranker‑v2‑m3) thường chính xác hơn khi số candidate nhỏ.
* Đa ngôn ngữ → tốt cho tiếng Việt/Anh.

> Lưu ý: để re‑rank được, ta **mô tả RoI trước** (VLM → text) rồi mới chấm điểm.

---

## 5) Đánh giá & kiểm thử

### 5.1 Mức model/pipeline

* **Ablation**:

  * Tắt gate (luôn kiểm chứng) vs bật gate.
  * CLIP image‑text vs Text re‑ranker.
  * Tags từ VLM vs danh sách cố định.
  * Thay detector (GroundingDINO) bằng OWL‑ViT/YOLO‑World để so sánh.
* **Sensitivity**: quét `alpha_fuse ∈ {0.5, 0.6, …, 0.9}`.

### 5.2 Mức bài toán

* **Độ chính xác VQA**: exact match / VQA accuracy.
* **Rationale sufficiency**: tỉ lệ câu trả lời có ít nhất 1 EV phù hợp.
* **Calibration**: có thể báo **ECE** cho xác suất “answer confidence” (nếu bạn sinh probability hoặc quy đổi score hợp lý).
* **Speed/Cost**: số lần gọi VLM / ảnh.

### 5.3 Quan sát định tính

* Log ảnh crop + hộp box + mô tả + điểm re‑rank để audit.
* Lưu `result.json` kèm các [EV*] liên kết đến hình crop.

---

## 6) Giới hạn & mẹo thực chiến

* **Tag nghèo nàn** → RoI lạc đề: tăng `roi_per_step`, thêm prompt ví dụ tag, hoặc bổ sung module NER/phrase mining.
* **Nhiễu văn bản** (biển hiệu, chữ nhỏ): thêm OCR (PaddleOCR/Tesseract) rồi nối vào mô tả RoI trước khi re‑rank.
* **Nhiều đối tượng tương tự**: bật *non‑max suppression per tag* hoặc nhóm theo `tag` rồi chọn top theo fused score.
* **Chi phí**: cache mô tả RoI theo (img_id, box), hoặc giảm `max_steps`.

---

## 7) Lộ trình mở rộng

* Thay **heuristic gate** bằng **MLP nhẹ** (nhị phân *needs‑vision?*) huấn luyện trên cặp *(step, label)*.
* Huấn luyện **tagger** (trích noun‑phrase từ step + từ ảnh) thay vì chỉ hỏi VLM.
* Tích hợp **structured evidence** (quan hệ, layout) cho tài liệu/biểu đồ.
* Kết hợp **uncertainty**: bỏ phiếu nhiều mô tả RoI và tính độ tin cậy.
* Xuất **report HTML** hiển thị ảnh + box + lời giải (phục vụ demo/đồ án).

---

## 8) Cách chạy nhanh trong repo

```bash
python -m scripts.corgi_demo \
  --image ./your_image.jpg \
  --question "What is the person holding?" \
  --max_steps 4 --per_step 4 --alpha 0.8 \
  --out result.json
```

Kết quả gồm:

* `steps`: chuỗi reasoning từng bước.
* `evidence`: `[tag, box, description, score]` cho mỗi bước đã xác minh.
* `answer`: câu trả lời cuối, ưu tiên bằng chứng [EV*].

---

## 9) Liên hệ tới đồ án/ứng dụng

* **DocVQA / ChartVQA**: thêm OCR + layout grounding.
* **Retail/Checkout**: đếm món, đối chiếu nhãn/giá.
* **Safety/Compliance**: phát hiện vật thể nguy hiểm, xác minh thuộc tính (mũ bảo hộ, áo phản quang…).
* **AR/Robotics**: cần bằng chứng thị giác đáng tin trong chuỗi thao tác.

---

**Kết luận**: CoRGI cung cấp một *khuôn* để buộc chuỗi suy luận của VLM phải dựa trên bằng chứng nhìn thấy được. Bản triển khai của bạn giữ đúng tinh thần này với Qwen‑VL (reasoning + mô tả), GroundingDINO (RoI), và text re‑ranker (xác minh liên quan), cho kết quả minh bạch, dễ kiểm thử và dễ mở rộng.
