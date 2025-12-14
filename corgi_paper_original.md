# CoRGI: Verified Chain-of-Thought Reasoning with Post-hoc Visual Grounding

**Authors:** Shixin Yi, Lin Shang

**Paper:** https://arxiv.org/abs/2508.00378

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
   - [Vision–Language Models](#21-visionlanguage-models)
   - [Chain-of-Thought Reasoning in VLMs](#22-chain-of-thought-reasoning-in-vlms)
   - [Thinking with Images: Reasoning via Visual Grounding](#23-thinking-with-images-reasoning-via-visual-grounding)
3. [Method](#3-method)
   - [Stage 1: Reasoning Chain Generation](#31-stage-1-reasoning-chain-generation)
   - [Stage 2: Visual Evidence Verification](#32-stage-2-visual-evidence-verification)
   - [Stage 3: Final Answer Synthesis](#33-stage-3-final-answer-synthesis)
4. [Experiments and Results](#4-experiments-and-results)
   - [Experimental Setup](#41-experimental-setup)
   - [Main Result: The Efficacy of Visual Verification](#42-main-result-the-efficacy-of-visual-verification)
   - [Design Choices in Visual Evidence Verification within VEVM](#43-design-choices-in-visual-evidence-verification-within-vevm)
   - [Qualitative Analysis and Further Experiments](#44-qualitative-analysis-and-further-experiments)
5. [Conclusion](#5-conclusion)
6. [References](#6-references)

---

## Abstract

Multimodal reasoning with vision–language models (VLMs) often suffers from **hallucinations**, as models tend to generate explanations after only a superficial inspection of the image. We present **CoRGI (Chain of Reasoning with Grounded Insights)**, a framework that enhances reasoning reliability through post-hoc verification of chain-of-thought outputs.

Given a VLM-generated rationale, CoRGI:
1. Decomposes it into step-wise statements
2. Grounds each step in visual evidence
3. Filters or corrects unsupported claims before producing the final answer

Experiments on five challenging benchmarks—**VCR**, **ScienceQA**, **MMMU**, **MathVista**, and **HallusionBench**—demonstrate that CoRGI consistently improves both answer accuracy and explanation faithfulness across multiple VLM backbones, including **Qwen-2.5VL**, **LLaVA-1.6**, and **Gemma3-12B**.

**Keywords:** Vision-Language Models, Chain-of-Thought Reasoning, Visual Grounding, Multimodal Verification, Visual Commonsense Reasoning

---

## 1. Introduction

Recent advances in vision–language models (VLMs) have enabled strong performance on multimodal tasks such as visual question answering, captioning, and commonsense reasoning. A common way to further enhance reasoning capability is to apply **Chain‑of‑Thought (CoT) prompting** (Wei et al. 2022), which elicits multi‑step intermediate reasoning in natural language. While CoT can improve interpretability and logical structure, we observe that its explanations often **drift away from the image content**. The generated reasoning may sound fluent but is not visually faithful, producing hallucinations.

### The Single-Look Bias Problem

We argue that a major underlying factor is what we call the **single‑look bias**. This bias is rooted in their architecture:

- The model's interaction with the visual input is confined to an initial stage
- A single, static representation of the image is formed through encoding and feature alignment
- The subsequent reasoning process is performed autoregressively by a large language model (LLM)
- The LLM relies on this fixed visual representation and its internal language priors, **never re‑consulting the image**

As a result, while the generated steps may be linguistically fluent, they are often detached from the actual visual evidence, leading to hallucinations.

### Why Not Iterative Grounding?

A natural way to address the single‑look bias would be **iterative grounding**, where the model revisits the image during each reasoning step and conditions the next token generation on new visual evidence. However:

- Requires tight cross‑modal integration
- Needs architectural redesign
- Demands expensive training
- Can be prohibitively costly in practice

### The CoRGI Solution

We propose a pragmatic alternative: **CoRGI (Chain of Reasoning with Grounded Insights)**, a framework that introduces **post‑hoc visual verification** into the CoT process.

Key design principles:
- Instead of altering the autoregressive decoding loop, CoRGI first lets a VLM generate a full CoT chain
- In a separate stage, each reasoning step is checked against the image by extracting region‑specific evidence
- This design is **lightweight**, **modular**, and **compatible with existing VLMs**
- Requires only a small relevance classifier and off‑the‑shelf detectors

We deliberately use the term **verification** in a precise sense: CoRGI performs post‑hoc, step‑wise visual evidence checking, not real‑time iterative correction.

### CoRGI Three-Stage Pipeline

1. **Reasoning Chain Generation:** A powerful VLM first generates a multi-step reasoning chain based on the input image and question.

2. **Visual Evidence Verification:** For each reasoning step, a dedicated Visual Evidence Verification Module (VEVM) determines whether the step requires visual verification, locates relevant Regions of Interest (RoIs), and queries a visual-language model to describe the grounded visual evidence.

3. **Answer Synthesis with Verified Evidence:** The VLM is finally prompted with the original question, the generated reasoning steps, and the corresponding visual evidence, enabling it to synthesize a final, better-grounded answer.

### Main Contributions

1. We identify the **single‑look bias** as a core challenge in VLM-based multimodal reasoning, and introduce CoRGI, a general framework for visual evidence verification in reasoning chains.

2. We design a modular **Visual Evidence Verification Module (VEVM)** that supports step-wise verification using minimal training and off-the-shelf components.

3. We validate CoRGI across five multimodal reasoning benchmarks (VCR, ScienceQA, MMMU, MathVista, HallusionBench) and three recent VLM backbones (Qwen-2.5VL-7B, LLaVA-1.6-7B, Gemma3-12B). Results show consistent accuracy gains and reduced hallucinations.

---

## 2. Related Work

### 2.1 Vision–Language Models

Vision–Language Models (VLMs) underpin modern multimodal reasoning:

| Model | Key Innovation |
|-------|----------------|
| **CLIP** (Radford et al. 2021) | Joint image–text embedding via contrastive pretraining |
| **Flamingo** (Alayrac et al. 2022) | Perceiver-based resampler for few-shot multimodal prompting |
| **BLIP‑2** (Li et al. 2023) | Querying Transformer linking frozen visual encoder to LLM |
| **InstructBLIP** (Dai et al. 2023) | Instruction tuning on BLIP‑2 for visual instruction-following |
| **Qwen‑2VL** (Wang et al. 2024) | High-performance multilingual VLM for Chinese–English reasoning |

### 2.2 Chain-of-Thought Reasoning in VLMs

| Work | Contribution |
|------|--------------|
| **Wei et al. 2022** | Original CoT prompting for LLMs |
| **Multimodal-CoT** (Zhang et al. 2023) | Two-stage fine-tuning for vision-language CoT |
| **LLaVA‑CoT** (Xu et al. 2024) | Structured multi-stage reasoning with stage-level beam search |

### 2.3 Thinking with Images: Reasoning via Visual Grounding

| Work | Approach |
|------|----------|
| **OpenAI's Thinking with Images** (2024) | GPT-4V's multi-hop visual reasoning |
| **Visual Sketchpad** (Hu et al. 2024) | Interactive visual sketchpad for drawing and reasoning |
| **VisualToolAgent (VisTA)** (Huang et al. 2025) | RL for dynamic visual tool integration |
| **VLM-R3** (Jiang et al. 2025) | RL for deciding when/where to attend to visual evidence |
| **Visual CoT** (Shao et al. 2024) | Dataset of 373k QA pairs with bounding box annotations |

---

## 3. Method

![CoRGI Pipeline](pipeline_diagram.png)
*Figure 2: An illustration of the three-stage CoRGI pipeline.*

### 3.1 Stage 1: Reasoning Chain Generation

The initial stage generates a high-level reasoning plan:

- **Input:** Image `I` and question `Q`
- **Output:** Multi-step textual reasoning chain `R = {r₁, r₂, ..., rₙ}`
- **Model:** Pre-trained foundation VLM (e.g., Qwen-2.5VL 7B)

Each `rᵢ` is a natural language sentence representing a logical assertion or a line of thought intended to incrementally lead to the final answer.

### 3.2 Stage 2: Visual Evidence Verification

This is the **core of the CoRGI framework**. The purpose is to validate each reasoning step `rᵢ` by grounding it in factual visual evidence.

The **Visual Evidence Verification Module (VEVM)** is designed as a modular system with three sub-processes:

#### 3.2.1 Relevance Classification (Deciding if and how much to look)

Not all reasoning steps require direct visual verification; some are more about abstract reasoning than visual grounding.

**Implementation:**
- Each reasoning step `rᵢ` passes through a lightweight **MLP classifier** ('RelevanceClassifier')
- Outputs a logit with dual purpose:
  - **Gating Mechanism:** If sigmoid value below threshold → step bypassed (non-visual)
  - **Importance Weighting:** If relevant → sigmoid converted to importance score (e.g., "importance: 75%")

#### 3.2.2 RoI Selection (Deciding where to look)

Once a step is deemed visually relevant:

- **Model:** **Grounding DINO** (Liu et al. 2024b)
- **Purpose:** Zero-shot identification of relevant image regions
- **Input:** Reasoning step's textual content
- **Output:** Regions of Interest (RoIs)

#### 3.2.3 VLM-based Visual Evidence Extraction (Describing what is seen)

With RoIs identified:

- **Model:** Pre-trained VLM (e.g., Qwen-2.5VL 7B) as "fact checker"
- **Task:** Provide concise, grounded descriptions of visual content within each RoI
- **Conditioning:** Descriptions conditioned on current reasoning step
- **Fallback:** If no RoIs selected → applied to full image
- **Output:** Textual descriptions `E = {e₁, e₂, ..., eₙ}`

### 3.3 Stage 3: Final Answer Synthesis

All generated information is aggregated for the final decision:

**Input to VLM:**
- Original Question `Q`
- Generated Reasoning Chain `R`
- Extracted Visual Evidence `E` (each prefixed with importance score)

**Output:** Final, well-grounded answer

By providing the model with not just its own "thoughts" but also the "evidence" supporting those thoughts, we reduce its tendency to hallucinate and guide it towards a more robust conclusion.

---

## 4. Experiments and Results

### 4.1 Experimental Setup

#### Datasets

| Dataset | Focus Area |
|---------|------------|
| **VCR** (Zellers et al. 2019) | Visual Commonsense Reasoning |
| **ScienceQA** (Lu et al. 2022) | Scientific knowledge |
| **MMMU** (Yue et al. 2024) | Multi-discipline exam-style problems |
| **MathVista** (Lu et al. 2024) | Math-based reasoning |
| **HallusionBench** (Guan et al. 2024) | Hallucination-sensitive evaluation |

#### Base Models

- **Qwen-2.5VL-7B** (Bai et al. 2025)
- **LLaVA-1.6-7B** (Liu et al. 2024a)
- **Gemma3-12B** (Team et al. 2025)

#### Baselines

| Setting | Description |
|---------|-------------|
| **Raw VLM** | Direct answer without intermediate reasoning |
| **+CoT** | Reasoning chain generation, then answer (no verification) |
| **+CoRGI (Ours)** | Full framework with VEVM verification |

### 4.2 Main Result: The Efficacy of Visual Verification

#### Table 1: Performance on Five Multimodal Reasoning Benchmarks

| Model | VCR Q→A | VCR QA→R | VCR Q→AR | ScienceQA | MMMU | MathVista | HallusionBench |
|-------|---------|----------|----------|-----------|------|-----------|----------------|
| **Qwen-2.5VL (Raw)** | 63.0 | 60.9 | 32.6 | 81.2 | 46.6 | 61.3 | 60.3 |
| +CoT | 61.3 | 59.9 | 39.2 | 82.7 | 47.3 | 60.2 | 57.2 |
| **+CoRGI (Ours)** | **63.3** (+0.3) | **61.6** (+0.7) | **41.0** (+8.4) | **82.8** (+1.6) | **48.7** (+2.1) | **62.2** (+0.9) | **61.2** (+0.9) |
| **LLaVA-1.6 (Raw)** | 45.1 | 37.1 | 11.6 | 58.2 | 32.2 | 26.5 | 44.6 |
| +CoT | 50.7 | 50.7 | 19.3 | 63.4 | 32.3 | 27.1 | 45.0 |
| **+CoRGI (Ours)** | **52.5** (+7.4) | **53.0** (+12.9) | **21.0** (+9.4) | **63.9** (+5.7) | **34.3** (+2.1) | **29.3** (+2.8) | **48.2** (+3.6) |
| **Gemma3-12B (Raw)** | 55.9 | 52.2 | 26.4 | 75.1 | 44.5 | 34.9 | 58.0 |
| +CoT | 52.7 | 52.3 | 34.2 | 74.6 | 45.0 | 36.0 | 58.4 |
| **+CoRGI (Ours)** | **56.0** (+0.1) | **56.5** (+4.3) | **34.7** (+8.3) | **77.5** (+2.4) | **45.8** (+1.3) | **39.8** (+4.9) | **59.3** (+1.3) |

#### Key Observations

1. **CoRGI provides improvements over both direct VLM outputs and CoT prompting across all datasets and backbones.**

2. **Gains are especially pronounced on LLaVA-1.6:**
   - VCR QA→R accuracy: **+12.9 points**
   - Other benchmarks: **+5–9 point gains**
   - Models with weaker built-in grounding benefit substantially from post-hoc verification

3. **Consistent gains on stronger backbones:**
   - Qwen's Q→AR accuracy: **+8.4 points**
   - Gemma's QA→R: **+4.3 points**
   - Even well-performing VLMs still produce unsupported reasoning steps

4. **Generalization across diverse tasks** including science QA, math reasoning, and hallucination stress testing

### 4.3 Design Choices in Visual Evidence Verification within VEVM

#### 4.3.1 Exploratory Attempt: End-to-End Generation Model (EGM)

An early exploration of end-to-end visual evidence generation:

**Architecture:**
- **Image Encoder:** CLIP-ViT (frozen)
- **Text Encoder:** DistilBERT (frozen)
- **Decoder:** GPT-2 (frozen initially)
- **Fusion:** Multi-stage attention (RoI-to-RoI self-attention + RoI-to-context cross-attention)

**Model Size:** 380M parameters (342M frozen, 38M trainable fusion modules)

**Training:** Two-stage strategy on 200K training examples

**Failure Modes:**
1. **Step Repetition:** Model generated shallow rephrasings without visual grounding
2. **Incorrect Visual Grounding:** Referenced irrelevant RoIs or described non-existent scenes

**Conclusion:** Limited training data insufficient for learning grounded generation → motivated VLM-based approach.

#### 4.3.2 Component-wise Ablation of VEVM

**Table 2: Ablation Results on Qwen-2.5VL-7B**

| Ablation Setting | VCR Q→A | VCR QA→R | VCR Q→AR | ScienceQA | MMMU |
|------------------|---------|----------|----------|-----------|------|
| **Full CoRGI** | 63.3 | 61.6 | 41.0 | 82.8 | 48.7 |
| w/o Relevance Classifier | 61.4 (↓1.9) | 59.2 (↓2.4) | 40.9 (↓0.1) | 82.4 (↓0.4) | 47.9 (↓0.8) |
| w/o RoI Selection | 62.0 (↓1.3) | 59.1 (↓2.5) | 40.8 (↓0.2) | 82.7 (↓0.1) | 48.4 (↓0.3) |
| w/o Reasoning Conditioning | 61.1 (↓2.2) | 61.4 (↓0.2) | 40.4 (↓0.6) | 82.2 (↓0.6) | 47.9 (↓0.8) |
| w/o Visual Evidence (+CoT) | 61.3 (↓2.0) | 59.9 (↓1.7) | 39.2 (↓1.8) | 82.7 (↓0.1) | 47.3 (↓1.4) |

**Key Findings:**

- **Relevance classification** avoids unnecessary verification for non-visual steps
- **RoI selection** ensures spatial precision
- **Reasoning-conditioned generation** improves semantic focus
- **Disabling any component leads to performance degradation**

### 4.4 Qualitative Analysis and Further Experiments

The paper demonstrates zero-shot generalization capability of CoRGI on VQA-v2 dataset, showing:

- Successfully generates coherent reasoning chains
- Produces factually grounded visual evidence for unseen datasets
- Highlights versatility of the pipeline structure

---

## 5. Conclusion

### Summary

We introduced **CoRGI**, a framework for enhancing multimodal reasoning through **post-hoc verification** of chain-of-thought outputs. Unlike raw VLMs that often rely on a single pass of visual inspection, CoRGI revisits the image to verify and ground individual reasoning steps, thereby:

- **Reducing hallucinations**
- **Improving faithfulness**

Across diverse benchmarks and multiple VLM backbones, CoRGI demonstrates consistent improvements over both direct VLM answers and CoT prompting.

### Limitations

1. **Sequential, post-hoc manner:** CoRGI does not revise the reasoning chain itself. Before the entire reasoning chain is complete, errors cannot be identified and corrected in real-time. Initial mistakes can compound, leading to cascading flawed reasoning.

2. **Sensitivity to initial CoT quality:** If the reasoning chain heads in the wrong direction due to missing commonsense or task-specific knowledge, even accurate visual grounding may not recover the correct answer.

3. **Latency:** The current evidence extraction depends on large external VLMs, which can introduce latency.

### Future Directions

1. **Real-time reasoning correction** through reinforcement learning policies for iterative refinement

2. **External knowledge integration** using Retrieval-Augmented Generation (RAG) to ground CoT in structured facts from knowledge graphs or textual corpora

3. **Lightweight verification modules** trained via knowledge distillation for end-to-end self-verification capabilities

---

## 6. References

1. Alayrac, J.-B., et al. (2022). Flamingo: a visual language model for few-shot learning. *NeurIPS*, 35: 23716–23736.

2. Bai, S., et al. (2025). Qwen2.5-VL technical report. *arXiv preprint arXiv:2502.13923*.

3. Dai, W., et al. (2023). InstructBLIP: Towards general-purpose vision-language models with instruction tuning. *NeurIPS*, 36: 49250–49267.

4. Guan, T., et al. (2024). HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. *CVPR*, 14375–14385.

5. Hu, Y., et al. (2024). Visual Sketchpad: Sketching as a visual chain of thought for multimodal language models. *NeurIPS*, 37: 139348–139379.

6. Huang, Z., et al. (2025). VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection. *arXiv preprint arXiv:2505.20289*.

7. Jiang, C., et al. (2025). VLM-R3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought. *arXiv preprint arXiv:2505.16192*.

8. Li, J., et al. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML*, 19730–19742.

9. Liu, H., et al. (2024a). Improved baselines with visual instruction tuning. *CVPR*, 26296–26306.

10. Liu, S., et al. (2024b). Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. *ECCV*, 38–55.

11. Lu, P., et al. (2022). Learn to explain: Multimodal reasoning via thought chains for science question answering. *NeurIPS*, 35: 2507–2521.

12. Lu, P., et al. (2024). MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. *ICLR*.

13. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*, 8748–8763.

14. Shao, H., et al. (2024). Visual CoT: Unleashing chain-of-thought reasoning in multi-modal language models. *CoRR*.

15. Wang, P., et al. (2024). Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*.

16. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*, 35: 24824–24837.

17. Xu, G., et al. (2024). LLaVA-CoT: Let vision language models reason step-by-step. *arXiv preprint arXiv:2411.10440*.

18. Yue, X., et al. (2024). MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. *CVPR*.

19. Zellers, R., et al. (2019). From Recognition to Cognition: Visual Commonsense Reasoning. *CVPR*.

20. Zhang, Z., et al. (2023). Multimodal chain-of-thought reasoning in language models. *arXiv preprint arXiv:2302.00923*.

---

*This document is a markdown conversion of the original HTML paper from [arXiv:2508.00378v2](https://arxiv.org/html/2508.00378v2)*
