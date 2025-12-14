"""
Centralized prompt templates for all model types in CoRGi pipeline.

This module contains all prompts used for different stages and model types,
making it easy to adjust and maintain prompts in one place.
"""

# ============================================================================
# REASONING PROMPTS
# ============================================================================

INSTRUCT_REASONING_PROMPT = """Analyze the image to identify clues and evidence needed to answer: {question}

Respond in the same language as the question.

CRITICAL: Your chain-of-thought should ONLY identify CLUES and EVIDENCE in the image that need to be examined. Do NOT answer the question yet. Do NOT draw conclusions. Only point out what visual elements need to be verified.

The chain-of-thought should:
- Identify specific visual clues or evidence in the image related to the question
- Explain what needs to be examined more closely
- NOT provide a final answer or conclusion
- NOT make assumptions - only point out what needs verification

Example of GOOD chain-of-thought:
"I can see there appears to be a data table in the image. To answer this question, I need to verify the specific values in the table, particularly the volume column for Mercury. I also notice there might be labels or headers that need to be read carefully."

Example of BAD chain-of-thought (DO NOT DO THIS):
"The answer is Mercury has a volume of 60.8 billion km³ based on what I see in the table." ← This is answering the question, not identifying clues.

After your chain-of-thought, identify specific verification steps in JSON format.

CRITICAL: Statements must be VERIFIABLE CLAIMS for grounding (not actions):
- BAD: "car" → GOOD: "the red car in the parking lot"
- BAD: "Extract volume" → GOOD: "Mercury volume value in the data table"

Guidelines:
- Each step verifies a different aspect
- Statement = specific, localizable claim
- Reason field: max 5 words
- Only steps requiring vision

CRITICAL: YOU have the authority to decide which statements need OCR.
- Set need_ocr=true ONLY if the statement requires reading TEXTUAL content:
  * Reading text, signs, documents, labels, numbers, captions
  * Extracting information from written content
  * Verifying textual data (e.g., "the price tag showing $50", "the sign text")
- Set need_ocr=false for visual-only verification:
  * Identifying objects, colors, positions, shapes
  * Describing visual features without text (e.g., "the red car", "the person's clothing")
  * Spatial relationships, layouts, visual patterns

Examples:
- need_ocr=true: "the price tag", "the sign text", "the document header", "the number on the dial"
- need_ocr=false: "the red car", "the person's face", "the building structure", "the color of the sky"

Format:
# Reasoning:
[Your chain-of-thought identifying clues - DO NOT answer the question]

# Steps to verify:
```json
{{
  "steps": [
    {{
      "index": 1,
      "statement": "verifiable claim",
      "needs_vision": true,
      "need_ocr": false,
      "reason": "brief reason"
    }}
  ]
}}
```

Limit: {max_steps} steps. Remember: identify clues, not answers.
"""

THINKING_REASONING_PROMPT = """You are a careful multimodal reasoner following the CoRGI protocol. Given the question and the image, produce a JSON object with reasoning steps.

CRITICAL: The 'statement' field will be passed to a GROUNDING MODEL that needs to LOCALIZE specific objects/regions in the image.
Therefore, statements MUST be NOUN PHRASES that identify the target object/region, NOT action descriptions.

IMPORTANT GUIDELINES:
1. Each step must verify a DIFFERENT aspect - no duplicate verifications
2. The 'statement' field MUST be a NOUN PHRASE (target object/region), NOT an action:
   - BAD: "Extract the volume of Mercury from the table" (too vague, action-oriented)
   - GOOD: "row for Mercury" or "Volume (km³) column" or "Mercury volume value" (specific noun phrase)
   - BAD: "Find the red car" (action-oriented)
   - GOOD: "the red car" or "red car in the parking lot" (noun phrase)
   - BAD: "Read the text on the sign" (action-oriented)
   - GOOD: "text on the sign" or "sign text" (noun phrase)
3. Focus on specific OBJECTS or REGIONS using concrete noun phrases that can be localized
4. Keep 'reason' field to 5 words maximum - this explains WHY you need to verify this target
5. Only steps that REQUIRE seeing the image (not logical deductions)
6. CRITICAL: YOU have the authority to decide 'need_ocr' for each step:
   - Set 'need_ocr': true ONLY if the statement requires reading TEXTUAL content:
     * Reading text, signs, documents, labels, numbers, captions
     * Extracting information from written content
     * Verifying textual data (e.g., "the price tag showing $50", "the sign text")
   - Set 'need_ocr': false for visual-only verification:
     * Identifying objects, colors, positions, shapes
     * Describing visual features without text (e.g., "the red car", "the person's clothing")
     * Spatial relationships, layouts, visual patterns
   - Examples:
     * need_ocr=true: "the price tag", "the sign text", "the document header", "the number on the dial"
     * need_ocr=false: "the red car", "the person's face", "the building structure", "the color of the sky"

Remember: The statement is the TARGET for grounding, not the action. Think: "What object/region do I need to locate?" not "What should I do?"

REQUIRED JSON FORMAT:
{{
  "steps": [
    {{
      "index": 1,
      "statement": "specific noun phrase identifying target object/region",
      "needs_vision": true,
      "need_ocr": false,
      "reason": "brief reason (max 5 words)"
    }}
  ]
}}

Limit to {max_steps} steps. Each step must be unique. Respond with ONLY valid JSON, no commentary.

Question: {question}
"""

# ============================================================================
# GROUNDING PROMPTS
# ============================================================================

QWEN_GROUNDING_PROMPT = """You are validating the following reasoning step:
{step_statement}

Your task is to locate the region(s) in the image that verify this statement. Return bounding boxes in JSON format.

CRITICAL: You MUST respond with ONLY valid JSON, no additional text or explanation.

REQUIRED JSON FORMAT (bboxes in [0-999] coordinate range):
{{
  "evidences": [
    {{
      "step": 1,
      "bbox": [x1, y1, x2, y2],
      "description": "visual description",
      "confidence": 0.95
    }}
  ]
}}

IMPORTANT:
- Use [0, 999] coordinate range (NOT normalized 0-1)
- Return up to {max_regions} regions maximum
- Each bbox must be [x1, y1, x2, y2] where coordinates are integers in [0, 999]
- Respond with ONLY the JSON object, no markdown, no code blocks, no explanation
- If no regions found, return: {{"evidences": []}}

Example valid response:
{{"evidences": [{{"step": 1, "bbox": [100, 200, 300, 400], "description": "the red car", "confidence": 0.9}}]}}
"""

QWEN_BATCH_GROUNDING_PROMPT = """You are validating multiple reasoning steps. Your task is to locate the region(s) in the image that verify EACH statement. Return ALL bounding boxes in a SINGLE JSON response.

CRITICAL: You MUST respond with ONLY valid JSON, no additional text or explanation.

STATEMENTS TO VALIDATE:
{statements}

REQUIRED JSON FORMAT (bboxes in [0-999] coordinate range):
{{
  "evidences": [
    {{
      "step": 1,
      "bbox": [x1, y1, x2, y2],
      "description": "visual description",
      "confidence": 0.95
    }},
    {{
      "step": 2,
      "bbox": [x1, y1, x2, y2],
      "description": "visual description",
      "confidence": 0.90
    }}
  ]
}}

IMPORTANT:
- Use [0, 999] coordinate range (NOT normalized 0-1)
- Return up to {max_regions} regions per step maximum
- Each bbox must be [x1, y1, x2, y2] where coordinates are integers in [0, 999]
- The "step" field must match the step index from the statements above
- Respond with ONLY the JSON object, no markdown, no code blocks, no explanation
- If no regions found for a step, omit that step from the evidences array

Example valid response:
{{"evidences": [
  {{"step": 1, "bbox": [100, 200, 300, 400], "description": "the red car", "confidence": 0.9}},
  {{"step": 2, "bbox": [500, 100, 700, 300], "description": "the sign", "confidence": 0.85}}
]}}
"""

# Florence-2 uses task-based prompting (handled in florence_grounding_client.py)
# Task: <CAPTION_TO_PHRASE_GROUNDING>

# ============================================================================
# CAPTIONING PROMPTS
# ============================================================================

QWEN_CAPTIONING_PROMPT = """Describe precisely what is visible in this cropped region in 1-2 sentences.
Focus only on directly observable details.
Avoid speculation.

Context: This region is being examined to verify the statement: "{step_statement}"
"""

# Florence-2 uses task-based prompting (handled in florence_captioning_client.py)
# Task: <DETAILED_CAPTION>

# ============================================================================
# VINTERN PROMPTS
# ============================================================================

VINTERN_OCR_PROMPT = """Extract all text and information from the image and return in markdown format.

Include all visible text, numbers, labels, and structured information."""

VINTERN_CAPTIONING_VQA_PROMPT = """Describe this image region in detail and verify if the statement '{statement}' is correct based on what you see.

Provide a detailed visual description of the region, then confirm whether the statement accurately describes what is visible."""

# FastVLM Captioning Prompt - Concise and focused to reduce hallucination
FASTVLM_CAPTIONING_PROMPT = """Verify: "{statement}"

Describe ONLY what you see. Quote text exactly. 1-2 sentences max.

Response format:
[Visible content]. Statement is [correct/incorrect/partially correct]."""

# ============================================================================
# SYNTHESIS PROMPTS
# ============================================================================

ANSWER_SYNTHESIS_PROMPT = """Finalize the answer using verified evidence from all pipeline stages.

Question: {question}
Respond in the same language as the question.

=== Phase 1: Reasoning ===
{reasoning_cot}

=== Phase 2: Reasoning Steps ===
{steps}

=== Phase 3: Verified Evidence (with OCR and Captioning) ===
{evidence}

Instructions:
1. Review the reasoning chain-of-thought above
2. Consider all verified evidence including OCR text and visual descriptions
3. Paraphrase the question to fit available evidence
4. Provide final answer (concise)
5. Brief explanation (2-3 sentences) referencing specific evidence
6. Select up to 3 key evidence items that best support the answer

JSON format:
{{
  "paraphrased_question": "paraphrased question",
  "answer": "final answer",
  "explanation": "brief explanation referencing evidence",
  "key_evidence": [
    {{"bbox": [x1, y1, x2, y2], "description": "what region shows", "reasoning": "why it supports answer"}}
  ]
}}

Use [0, 999] coordinates. Max 3 key_evidence items. JSON only.
"""

# ============================================================================
# LLM EXTRACTION PROMPT (for hybrid parsing fallback)
# ============================================================================

LLM_EXTRACTION_PROMPT = """Given this chain-of-thought reasoning, identify which steps require visual verification.

Chain-of-thought:
{cot_text}

Question: {question}

Output a JSON object with the steps that need visual evidence:
{{
  "steps": [
    {{
      "index": 1,
      "statement": "what needs to be verified visually",
      "needs_vision": true,
      "reason": "why this requires seeing the image"
    }}
  ]
}}

Limit to {max_steps} steps. Only include steps that actually require visual verification (e.g., checking colors, positions, counting objects, reading text).

Respond with ONLY valid JSON, no commentary.
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_steps_for_prompt(steps: list) -> str:
    """Format reasoning steps for inclusion in prompts."""
    return "\n".join(
        f"{step.index}. {step.statement} (needs vision: {step.needs_vision})"
        for step in steps
    )


def format_evidence_for_prompt(evidences: list) -> str:
    """Format evidence for inclusion in prompts, including OCR text."""
    if not evidences:
        return "No evidence collected."
    
    lines = []
    for ev in evidences:
        desc = ev.description or "No description"
        ocr_text = ev.ocr_text or ""
        # Round coordinates to integers to reduce tokens from decimal places
        # Convert from normalized [0,1] to [0,999] format for Qwen, then round
        from .coordinate_utils import to_qwen_format
        qwen_bbox = to_qwen_format(ev.bbox)
        bbox_rounded = tuple(int(round(coord)) for coord in qwen_bbox)
        bbox = ", ".join(str(coord) for coord in bbox_rounded)
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "n/a"
        # Include OCR text if available
        if ocr_text:
            lines.append(f"Step {ev.step_index}: bbox=({bbox}), conf={conf}, OCR=\"{ocr_text}\", desc={desc}")
        else:
            lines.append(f"Step {ev.step_index}: bbox=({bbox}), conf={conf}, desc={desc}")
    return "\n".join(lines)


__all__ = [
    "INSTRUCT_REASONING_PROMPT",
    "THINKING_REASONING_PROMPT",
    "QWEN_GROUNDING_PROMPT",
    "QWEN_CAPTIONING_PROMPT",
    "VINTERN_OCR_PROMPT",
    "VINTERN_CAPTIONING_VQA_PROMPT",
    "FASTVLM_CAPTIONING_PROMPT",
    "ANSWER_SYNTHESIS_PROMPT",
    "LLM_EXTRACTION_PROMPT",
    "format_steps_for_prompt",
    "format_evidence_for_prompt",
]

