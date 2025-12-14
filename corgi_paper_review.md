# CoRGI Paper Review

**Paper:** CoRGI: Verified Chain-of-Thought Reasoning with Post-hoc Visual Grounding  
**Authors:** Shixin Yi, Lin Shang  
**Link:** https://arxiv.org/abs/2508.00378  
**Reviewer:** CoRGI Implementation Team  
**Date:** December 2024

---

## 📋 Tổng quan

### Mục tiêu của Paper

Paper giải quyết vấn đề **hallucination** (ảo giác) trong các Vision-Language Models (VLMs) khi thực hiện multimodal reasoning. Các VLMs hiện tại thường tạo ra các giải thích nghe có vẻ hợp lý nhưng không thực sự dựa trên nội dung hình ảnh.

### Đóng góp chính

1. **Xác định vấn đề "Single-Look Bias"** - Mô hình chỉ nhìn ảnh một lần, sau đó reasoning dựa hoàn toàn trên language model
2. **Đề xuất CoRGI Framework** - Một pipeline 3 giai đoạn để verify từng bước reasoning với visual evidence
3. **Thiết kế VEVM Module** - Visual Evidence Verification Module với các thành phần modular

---

## 🔍 Phân tích Chi tiết

### 1. Vấn đề được giải quyết

#### Single-Look Bias

```
┌─────────────────────────────────────────────────────────────────┐
│  TRADITIONAL VLM ARCHITECTURE (Problematic)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image ──┬──► Visual Encoder ──► Fixed Representation          │
│          │                              │                       │
│          │                              ▼                       │
│          │                    Language Model (LLM)              │
│          │                              │                       │
│          │         ┌────────────────────┴─────────────────┐     │
│          │         │  Step 1 → Step 2 → Step 3 → Answer   │     │
│          │         │  (Autoregressive, never re-consults  │     │
│          │         │   the image!)                        │     │
│          │         └──────────────────────────────────────┘     │
│          │                                                      │
│          └──► ❌ Image không được "nhìn lại" trong reasoning    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Hậu quả:**
- Reasoning có thể fluent về mặt ngôn ngữ nhưng không faithful với visual content
- Hallucination: mô hình "bịa" thông tin không có trong ảnh
- Các bước reasoning drift away khỏi thực tế visual

#### Tại sao không dùng Iterative Grounding?

| Approach | Pros | Cons |
|----------|------|------|
| Iterative Grounding | Real-time verification | Đòi hỏi architecture redesign, expensive training |
| **Post-hoc Verification (CoRGI)** | Lightweight, modular, compatible với VLMs hiện có | Sequential, không real-time correction |

---

### 2. Phương pháp: CoRGI Pipeline

#### Stage 1: Reasoning Chain Generation

**Mục đích:** Tạo chuỗi reasoning đa bước

```
Input: Image I + Question Q
       ↓
VLM (e.g., Qwen-2.5VL 7B)
       ↓
Output: R = {r₁, r₂, ..., rₙ}
```

**Nhận xét:**
- ✅ Sử dụng foundation VLM mạnh để tạo reasoning plan
- ✅ Mỗi step là một logical assertion
- ⚠️ Ở stage này, reasoning vẫn có thể bị hallucination (sẽ được verify ở Stage 2)

#### Stage 2: Visual Evidence Verification (VEVM)

**Đây là phần core của framework, gồm 3 sub-modules:**

##### 2.1 Relevance Classification

```python
# Pseudo-code
class RelevanceClassifier(nn.Module):
    def __init__(self):
        self.mlp = MLP(hidden_dim=...)
    
    def forward(self, reasoning_step):
        logit = self.mlp(reasoning_step)
        sigmoid_value = torch.sigmoid(logit)
        
        if sigmoid_value < threshold:
            return "non_visual", 0  # Bypass
        else:
            importance = piecewise_mapping(sigmoid_value)
            return "visual", importance  # e.g., "importance: 75%"
```

**Ý nghĩa:**
- Không phải tất cả reasoning steps đều cần visual verification
- Một số steps thuần túy abstract reasoning (e.g., "Based on the previous observations...")
- Classifier quyết định **IF** và **HOW MUCH** cần verify

##### 2.2 RoI Selection

**Model:** Grounding DINO (zero-shot object detection)

```
Input: Reasoning step text (e.g., "The person is wearing a red shirt")
       ↓
Grounding DINO
       ↓
Output: Bounding boxes cho regions liên quan
```

**Ưu điểm:**
- Zero-shot: không cần train thêm
- Dynamically identify regions based on text
- Spatial precision cho evidence extraction

##### 2.3 VLM-based Visual Evidence Extraction

```
Input: 
  - RoI crops từ image
  - Reasoning step (để condition)
       ↓
VLM as "Fact Checker"
       ↓
Output: Textual description E = {e₁, e₂, ..., eₙ}
```

**Key insight:** Dùng VLM hiện có thay vì train model mới → practical và scalable

#### Stage 3: Final Answer Synthesis

```
Prompt = {
    "Question": Q,
    "Reasoning Chain": R,
    "Visual Evidence": E (với importance scores)
}
       ↓
VLM
       ↓
Final Answer (grounded)
```

**Lợi ích:**
- Model có cả "thoughts" VÀ "evidence"
- Reduce hallucination tendency
- More robust conclusions

---

### 3. Kết quả Thực nghiệm

#### 3.1 Datasets và Models

| Benchmark | Focus |
|-----------|-------|
| VCR | Visual Commonsense Reasoning |
| ScienceQA | Scientific knowledge |
| MMMU | Multi-discipline exam problems |
| MathVista | Math-based reasoning |
| HallusionBench | Hallucination stress testing |

**VLM Backbones:** Qwen-2.5VL-7B, LLaVA-1.6-7B, Gemma3-12B

#### 3.2 Performance Analysis

##### Improvement Summary

| Model | Best Improvement | Dataset |
|-------|-----------------|---------|
| LLaVA-1.6 | **+12.9 points** | VCR QA→R |
| Qwen-2.5VL | **+8.4 points** | VCR Q→AR |
| Gemma3-12B | **+8.3 points** | VCR Q→AR |

##### Key Observations

1. **Weaker models benefit more:** LLaVA-1.6 có built-in grounding yếu hơn → gains lớn nhất từ post-hoc verification

2. **Strong models still benefit:** Ngay cả Qwen-2.5VL (strong) vẫn có gains, cho thấy mọi VLM đều có unsupported reasoning steps

3. **Generalization:** CoRGI works across diverse tasks (science, math, commonsense, hallucination testing)

#### 3.3 Ablation Study Insights

```
Full CoRGI = Best Performance
     ↓
Remove Relevance Classifier → ↓ 1.9-2.4 points
Remove RoI Selection → ↓ 1.3-2.5 points  
Remove Reasoning Conditioning → ↓ 0.2-2.2 points
Remove All Visual Evidence → ↓ 1.7-2.0 points (= CoT baseline)
```

**Kết luận:** Mỗi component đều cần thiết, có synergistic effect khi combine

---

### 4. So sánh với Implementation của chúng ta

#### 4.1 Tương đồng

| Paper CoRGI | Our Implementation |
|-------------|-------------------|
| 3-stage pipeline | ✅ V2 pipeline với merged Phase 1+2 |
| Visual grounding từ text | ✅ Qwen/Florence for grounding |
| Evidence extraction with VLM | ✅ SmolVLM2/Florence OCR |
| Synthesis with all evidence | ✅ Phase 4 synthesis |

#### 4.2 Khác biệt và Cải tiến

| Aspect | Paper | Our V2 Implementation |
|--------|-------|----------------------|
| **Reasoning + Grounding** | 2 separate stages | Merged vào 1 call (faster) |
| **Evidence Type** | VLM decides implicitly | Explicit `need_object` / `need_text` flags |
| **Relevance Classifier** | Trained MLP | Rule-based hoặc implicit từ structured output |
| **RoI Selector** | Grounding DINO | Qwen built-in grounding HOẶC Florence |
| **Latency** | ~10s+ estimated | ~6.3s (37% faster) |
| **Memory** | Multiple models | Reuse reasoning model (67% less VRAM) |

#### 4.3 Các cải tiến tiềm năng từ Paper

1. **Importance Scoring:** Paper dùng importance percentage từ classifier → có thể add vào synthesis prompt

2. **Explicit Relevance Classification:** Train một classifier nhẹ để filter non-visual steps

3. **Step-level Verification:** Verify từng step chi tiết hơn thay vì batch

---

## 💡 Insights và Lessons Learned

### Điểm mạnh của Paper

1. **Practical approach:** Không cần retrain VLM, chỉ add verification layer
2. **Modular design:** Có thể swap components easily
3. **Comprehensive evaluation:** 5 benchmarks, 3 VLMs, ablation studies
4. **Clear problem definition:** "Single-look bias" là một framing tốt

### Điểm yếu và Limitations

1. **Sequential nature:** Errors early in chain không thể recover
2. **Dependency on initial CoT:** Garbage in → garbage out
3. **Latency overhead:** Extra VLM calls cho verification
4. **No real-time correction:** Post-hoc, not iterative

### Gợi ý cho Future Work (từ Paper)

1. **RL for iterative refinement:** Real-time error correction
2. **RAG integration:** Ground reasoning trong external knowledge
3. **Lightweight verifiers:** Distilled models cho efficiency

---

## 🔧 Recommendations cho Implementation

### Short-term (Có thể làm ngay)

1. **Add importance scoring** trong prompt synthesis:
   ```
   Evidence 1 (importance: 85%): "The person is wearing a red shirt"
   Evidence 2 (importance: 45%): "Background shows a park"
   ```

2. **Implement step-level bypass** cho pure reasoning steps:
   - Nếu step không chứa visual references → skip evidence extraction

3. **Better error handling** khi grounding fails:
   - Fallback to full-image evidence

### Medium-term (Cần thiết kế thêm)

1. **Train lightweight relevance classifier:**
   - Input: reasoning step text
   - Output: visual_relevance_score [0, 1]
   - Data: Label từ manual annotation hoặc VLM self-assessment

2. **Add confidence calibration:**
   - Track accuracy của evidence vs final answer correctness
   - Adjust importance weights accordingly

### Long-term (Research direction)

1. **Iterative verification:**
   - After each step, verify và potentially revise
   - Requires streaming architecture

2. **Multi-modal RAG:**
   - Retrieve relevant images/documents for comparison
   - Cross-reference với external knowledge

---

## 📊 Summary Table

| Criterion | Score (1-5) | Comments |
|-----------|-------------|----------|
| **Novelty** | 4/5 | Good framing of single-look bias; post-hoc verification is practical |
| **Technical Soundness** | 4/5 | Well-designed ablations; clear methodology |
| **Reproducibility** | 4/5 | Details provided; uses public VLMs |
| **Impact** | 4/5 | Practical framework; immediate applicability |
| **Writing Quality** | 4/5 | Clear structure; good visualizations |

**Overall Assessment:** Paper đề xuất một framework practical và effective cho việc improve multimodal reasoning. Approach post-hoc verification là một trade-off hợp lý giữa performance và complexity. Implementation của chúng ta đã capture được essence của paper và có một số optimizations thêm (merged stages, smart routing).

---

## 📚 References cho Deep Dive

1. **Chain-of-Thought:** Wei et al., 2022 - Foundational paper on CoT prompting
2. **Grounding DINO:** Liu et al., 2024 - Open-set object detection dùng cho RoI selection
3. **Visual CoT:** Shao et al., 2024 - Dataset với bounding box annotations cho visual reasoning
4. **LLaVA-CoT:** Xu et al., 2024 - Structured multi-stage reasoning

---

*Review completed: December 2024*
