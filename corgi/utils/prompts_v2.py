"""
V2 Prompts for Pipeline V2.

NEW prompt templates for enhanced reasoning with evidence type flags.
V1 prompts remain in prompts.py - DO NOT MODIFY V1!
"""

# V2 Reasoning Prompt - Phase 1+2 Merged
REASONING_PROMPT_V2_TEMPLATE = """You are a visual question answering expert. Analyze the image and question step-by-step.

**Your Task:**
1. **Think carefully** about the question and what information is needed
2. **Break down** the reasoning into structured steps
3. **For each step that needs visual evidence:**
   - Determine if it requires **OBJECT/SCENE UNDERSTANDING** (visual objects, people, actions, scenes)
   - OR if it requires **TEXT/NUMBER RECOGNITION** (text, numbers, signs, documents, labels)
   - **Provide bounding box** if you can identify the region: [x1, y1, x2, y2] in normalized coordinates [0.0-1.0]

**CRITICAL RULES:**
- A step can need EITHER object understanding OR text recognition, **NEVER BOTH**
- If a step needs visual evidence about **TEXT/NUMBERS/SIGNS** → set `need_text_ocr: true`
- If a step needs visual evidence about **OBJECTS/SCENES/PEOPLE** → set `need_object_captioning: true`
- If a step is pure reasoning (no visual evidence) → both flags = `false`
- Bounding boxes `[x1, y1, x2, y2]` use normalized coordinates where (0,0) is top-left, (1,1) is bottom-right
- Bounding boxes are **optional but very helpful** - provide them if you can identify the region

**Output Format:**
You MUST output in this exact format:

<THINKING>
[Your detailed chain-of-thought reasoning here. Think step by step about:
 - What does the question ask?
 - What information do I need from the image?
 - What type of visual evidence is needed (objects vs text)?
 - Where in the image should I look?]
</THINKING>

<STRUCTURED_STEPS>
{{
  "steps": [
    {{
      "index": 1,
      "statement": "Clear, specific statement about what to verify or identify",
      "need_object_captioning": true or false,
      "need_text_ocr": true or false,
      "bbox": [x1, y1, x2, y2] or null,
      "reason": "Why this step is needed (optional)"
    }},
    ...
  ]
}}
</STRUCTURED_STEPS>

**Examples:**

Example 1 - License Plate Question:
Question: "What is the license plate number of the red car?"

<THINKING>
To answer this, I need to:
1. Find the red car (requires object detection/understanding)
2. Locate the license plate on that car (requires text location)
3. Read the text on the license plate (requires OCR)

Step 1 needs object captioning (identify red car).
Step 2 needs text OCR (read license plate).
</THINKING>

<STRUCTURED_STEPS>
{{
  "steps": [
    {{
      "index": 1,
      "statement": "Identify and locate the red car in the image",
      "need_object_captioning": true,
      "need_text_ocr": false,
      "bbox": [0.1, 0.3, 0.6, 0.8],
      "reason": "Need to find the red car first"
    }},
    {{
      "index": 2,
      "statement": "Read the license plate number on the red car",
      "need_object_captioning": false,
      "need_text_ocr": true,
      "bbox": [0.35, 0.65, 0.45, 0.72],
      "reason": "Need to extract text from license plate"
    }}
  ]
}}
</STRUCTURED_STEPS>

Example 2 - Document Question:
Question: "How many people are visible in the conference room photo?"

<THINKING>
This requires counting people, which is visual object understanding.
No text recognition needed.
</THINKING>

<STRUCTURED_STEPS>
{{
  "steps": [
    {{
      "index": 1,
      "statement": "Count the number of people visible in the conference room",
      "need_object_captioning": true,
      "need_text_ocr": false,
      "bbox": null,
      "reason": "Requires identifying and counting people (full image)"
    }}
  ]
}}
</STRUCTURED_STEPS>

---

**Now analyze this question:**

**Question:** {question}

**REMEMBER:** 
- Output both <THINKING> and <STRUCTURED_STEPS>
- Set ONLY ONE flag per step (object OR text, never both)
- Provide bbox when possible
- Use normalized coordinates [0.0-1.0]
"""

# Simplified prompt for quick testing
REASONING_PROMPT_V2_SIMPLE = """Analyze the image and question. Output your reasoning in this format:

<THINKING>
[Your chain of thought]
</THINKING>

<STRUCTURED_STEPS>
{{
  "steps": [
    {{
      "index": 1,
      "statement": "What to verify",
      "need_object_captioning": true/false,
      "need_text_ocr": true/false,
      "bbox": [x1,y1,x2,y2] or null
    }}
  ]
}}
</STRUCTURED_STEPS>

**Rules:**
- need_object_captioning: for visual objects/scenes
- need_text_ocr: for text/numbers
- Only ONE flag can be true per step
- bbox uses normalized [0-1] coordinates

Question: {question}
"""


# ============================================================================
# OPTIMIZED V2 PROMPTS (80% Token Reduction)
# ============================================================================

# Optimized V2 Prompt for Qwen3-Thinking Model
# Uses <think> tags, ~230 tokens (vs 1200 original)
REASONING_PROMPT_V2_THINKING_OPTIMIZED = """Analyze image and question. Output thinking + JSON steps.

For visual evidence, set ONE flag:
- Object/scene → need_object_captioning:true
- Text/numbers → need_text_ocr:true
Provide bbox [x1,y1,x2,y2] in [0-1] if possible.

Example:
Q: "Plate number?"
<think>1) Find car (object), 2) Read plate (OCR)</think>
<STRUCTURED_STEPS>
{{
  "steps": [
    {{"index":1,"statement":"Locate car","need_object_captioning":true,"need_text_ocr":false,"bbox":[0.1,0.2,0.5,0.8]}},
    {{"index":2,"statement":"Read plate","need_object_captioning":false,"need_text_ocr":true,"bbox":[0.3,0.6,0.4,0.7]}}
  ]
}}
</STRUCTURED_STEPS>

Question: {question}
"""

# Optimized V2 Prompt for Qwen3-Instruct Model
# Uses <THINKING> tags, ~210 tokens (vs 1200 original)
REASONING_PROMPT_V2_INSTRUCT_OPTIMIZED = """Analyze image and question. Output reasoning + JSON steps.

For visual evidence, set ONE flag:
- Object/scene → need_object_captioning:true
- Text/numbers → need_text_ocr:true
Bbox [x1,y1,x2,y2] in [0-1] if known.

Format:
<THINKING>Your reasoning</THINKING>
<STRUCTURED_STEPS>{{"steps":[...]}}</STRUCTURED_STEPS>

Example:
Q: "Plate?"
<THINKING>Find car, read text</THINKING>
<STRUCTURED_STEPS>
{{
  "steps":[
    {{"index":1,"statement":"Car","need_object_captioning":true,"need_text_ocr":false,"bbox":[0.1,0.2,0.5,0.8]}},
    {{"index":2,"statement":"Plate","need_object_captioning":false,"need_text_ocr":true,"bbox":[0.3,0.6,0.4,0.7]}}
  ]
}}
</STRUCTURED_STEPS>

Q: {question}
"""



def build_reasoning_prompt_v2(question: str, max_steps: int = 6, optimized: bool = False) -> str:
    """
    Build V2 reasoning prompt with question and max steps.
    
    Args:
        question: The question to answer
        max_steps: Maximum number of reasoning steps (default: 6)
        optimized: Use optimized (shorter) prompt template (default: False)
        
    Returns:
        Complete prompt string
    """
    if optimized:
        template = REASONING_PROMPT_V2_INSTRUCT_OPTIMIZED
    else:
        template = REASONING_PROMPT_V2_TEMPLATE
    
    # Format template with question
    prompt = template.format(question=question)
    
    # Add max steps constraint
    prompt += f"\n\nGenerate UP TO {max_steps} reasoning steps. Focus on the most critical steps needed."
    
    return prompt


__all__ = [
    "REASONING_PROMPT_V2_TEMPLATE",
    "REASONING_PROMPT_V2_SIMPLE",
    "REASONING_PROMPT_V2_THINKING_OPTIMIZED",
    "REASONING_PROMPT_V2_INSTRUCT_OPTIMIZED",
    "build_reasoning_prompt_v2",
]
