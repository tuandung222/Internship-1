# Structured Answer Update

**Date**: October 29, 2025  
**Feature**: Answer synthesis with key evidence (bbox + reasoning)  
**Model**: Qwen/Qwen3-VL-4B-Thinking

---

## 🎯 What Changed

### 1. Structured Answer Format

**Before**: Simple text answer
```python
answer: str  # Just text
```

**After**: Answer + Key Evidence with Reasoning
```python
answer: str  # Text answer
key_evidence: List[KeyEvidence]  # Bounding boxes with reasoning
```

**KeyEvidence Structure**:
```python
@dataclass
class KeyEvidence:
    bbox: BBox              # (x1, y1, x2, y2) normalized
    description: str        # What this region shows
    reasoning: str          # Why this supports the answer
```

---

### 2. Updated Answer Synthesis Prompt

Model now outputs structured JSON:

```json
{
  "answer": "Final answer sentence",
  "key_evidence": [
    {
      "bbox": [x1, y1, x2, y2],
      "description": "What this region shows",
      "reasoning": "Why this supports the answer"
    }
  ]
}
```

**Example Output**:
```json
{
  "answer": "There are zero small cakes visible on the table.",
  "key_evidence": [
    {
      "bbox": [0.5, 0.675, 0.695, 0.972],
      "description": "The table with a sewing machine and other items",
      "reasoning": "The table contains a sewing machine, papers, and miscellaneous objects but no small cakes, confirming the count is zero."
    }
  ]
}
```

---

## 📊 Test Results

```bash
python test_structured_answer.py
```

**Output**:
```
📝 Reasoning Steps (3):
  [1] The table in the room contains a sewing machine...
  [2] No small cakes are visible on the table.
  [3] The count of small cakes on the table is zero.

🔍 Visual Evidence (0):

💡 Final Answer:
  There are zero small cakes visible on the table.

🎯 Key Evidence with Reasoning (1):
  Evidence 1:
    BBox: (0.5, 0.675, 0.695, 0.972)
    Description: The table with a sewing machine and other items
    Reasoning: The table contains a sewing machine, papers, and 
               miscellaneous objects but no small cakes, 
               confirming the count is zero.

✅ Test completed! (78.0s total)
```

---

## 🔧 Implementation Details

### Files Changed

1. **`corgi/types.py`**
   - Added `KeyEvidence` dataclass
   - Updated `PipelineResult` to include `key_evidence` field

2. **`corgi/qwen_client.py`**
   - Updated `DEFAULT_ANSWER_PROMPT` with JSON format
   - Changed `synthesize_answer()` return type: `str` → `tuple[str, List[KeyEvidence]]`
   - Added `_parse_answer_response()` for structured parsing

3. **`corgi/pipeline.py`**
   - Updated `SupportsQwenClient` protocol
   - Modified `PipelineResult.to_json()` to serialize key_evidence
   - Updated pipeline to unpack answer tuple

4. **`corgi/parsers.py`**
   - Improved `_load_first_json()` with better fallback handling
   - Added regex-based JSON extraction as fallback

5. **`test_structured_answer.py`** (NEW)
   - Comprehensive test for structured answer feature

---

## 💡 Benefits

### For Users
- ✅ **Visual Grounding**: See which regions support the answer
- ✅ **Explainability**: Understand WHY model gave that answer
- ✅ **Transparency**: Clear evidence chain

### For Developers
- ✅ **Structured Output**: Easy to parse and visualize
- ✅ **Rich Information**: Bbox + description + reasoning
- ✅ **Backward Compatible**: Falls back to text-only if JSON fails

---

## 🐛 Known Issues & Fixes

### Issue 1: Empty JSON Response

**Problem**: Model sometimes returns empty response  
**Error**: `ValueError: Expecting value: line 1 column 1 (char 0)`

**Fix**: Improved parser with fallbacks
```python
# Check for empty response
if not response or not response.strip():
    logger.error("Empty response from model!")
    
# Fallback to text parsing if JSON fails
except Exception as e:
    logger.warning(f"Failed to parse JSON, falling back: {e}")
    return cleaned_text, []
```

### Issue 2: Thinking Tags in Response

**Problem**: Model includes `<think>...</think>` tags  
**Fix**: Strip thinking content before parsing
```python
def _strip_think_content(text: str) -> str:
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    return cleaned.strip()
```

---

## 📝 Usage Examples

### Python API

```python
from corgi.pipeline import CoRGIPipeline
from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig

# Create pipeline
config = QwenGenerationConfig()
client = Qwen3VLClient(config)
pipeline = CoRGIPipeline(vlm_client=client)

# Run
result = pipeline.run(image, question, max_steps=3, max_regions=3)

# Access answer
print(result.answer)

# Access key evidence
for ke in result.key_evidence:
    print(f"BBox: {ke.bbox}")
    print(f"Description: {ke.description}")
    print(f"Reasoning: {ke.reasoning}")
```

### JSON Export

```python
# Export to JSON
json_data = result.to_json()

# Structure:
{
  "answer": "...",
  "key_evidence": [
    {
      "bbox": [x1, y1, x2, y2],
      "description": "...",
      "reasoning": "..."
    }
  ],
  "steps": [...],
  "evidence": [...],
  ...
}
```

---

## 🧪 Testing

```bash
# Test structured answer
python test_structured_answer.py

# Test full pipeline
PYTHONPATH=$(pwd) python examples/demo_qwen_corgi.py
```

---

## 🚀 Future Enhancements

1. **Multiple Evidence per Step**: Support multiple key evidences
2. **Confidence Scores**: Add confidence to key evidence
3. **Visual Highlighting**: Auto-highlight key regions in UI
4. **Evidence Aggregation**: Combine related evidences

---

## ✅ Status

**Implementation**: ✅ Complete  
**Testing**: ✅ Passed  
**Documentation**: ✅ Done  
**Ready**: ✅ For deployment

---

## 📚 Related Files

- **Implementation**: `corgi/qwen_client.py`, `corgi/types.py`
- **Pipeline**: `corgi/pipeline.py`
- **Testing**: `test_structured_answer.py`
- **Examples**: `examples/demo_qwen_corgi.py`

---

**Summary**: Answer synthesis now returns structured evidence with bounding boxes and reasoning, making the model's conclusions transparent and verifiable! 🎉

