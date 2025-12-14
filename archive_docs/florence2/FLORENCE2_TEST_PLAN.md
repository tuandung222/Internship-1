# Florence-2 + Qwen3-VL Testing Plan

**Date:** November 8, 2025  
**Goal:** Test CoRGi pipeline with Qwen3-VL-4B for reasoning/synthesis + Florence-2 for grounding/captioning  
**GPU Available:** GPU 6, 7 (free) - A100 80GB each

---

## Current Status

✅ **Qwen-only configuration:** Working perfectly  
⚠️ **Florence-2:** Has compatibility issues with transformers library

**Error:**
```
AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
```

---

## Strategy: 3-Phase Approach

### Phase 1: Fix Florence-2 Compatibility (PRIORITY)

**Approach A: Downgrade Transformers (RECOMMENDED)**

Florence-2 works well with older transformers versions.

```bash
# Test with transformers 4.44.0
pip install transformers==4.44.0

# Run test
python test_real_pipeline.py --config configs/default.yaml --save-viz
```

**Approach B: Force Eager Attention**

Modify Florence clients to explicitly use eager attention.

```python
# In florence_grounding_client.py and florence_captioning_client.py
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    attn_implementation="eager",  # Force eager attention
)
```

**Approach C: Patch Florence-2 Model Code**

If using cached model, we can patch the model file directly.

```bash
# Location of Florence-2 model code
~/.cache/huggingface/modules/transformers_modules/microsoft/Florence_hyphen_2_hyphen_large/
```

Add `_supports_sdpa = False` to the model class.

---

### Phase 2: Create Florence-2 Test Configuration

**File:** `configs/test_florence2.yaml`

```yaml
# CoRGi Pipeline with Florence-2 for Grounding/Captioning

reasoning:
  model:
    model_id: "Qwen/Qwen3-VL-4B-Instruct"
    model_type: "qwen_instruct"
    device: "cuda:6"  # Use free GPU 6
    torch_dtype: "auto"
    enable_compile: false
    enable_flash_attn: false
  max_steps: 3
  max_new_tokens: 512
  extraction_method: "hybrid"

grounding:
  model:
    model_id: "microsoft/Florence-2-large"
    model_type: "florence2"
    device: "cuda:6"  # Same GPU as reasoning
    torch_dtype: "auto"
    enable_compile: false
    enable_flash_attn: false
  max_regions: 3
  max_new_tokens: 128

captioning:
  model:
    model_id: "microsoft/Florence-2-large"
    model_type: "florence2"
    device: "cuda:6"  # Reuse Florence-2 from grounding
    torch_dtype: "auto"
    enable_compile: false
    enable_flash_attn: false
  max_new_tokens: 128

synthesis:
  model:
    model_id: "Qwen/Qwen3-VL-4B-Instruct"
    model_type: "qwen_instruct"
    device: "cuda:6"  # Reuse Qwen from reasoning
    torch_dtype: "auto"
    enable_compile: false
    enable_flash_attn: false
  max_new_tokens: 256
```

---

### Phase 3: Comprehensive Testing & Comparison

#### Test 1: Basic Functionality Test

```bash
# Test Florence-2 grounding/captioning
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --save-viz \
    --output-dir test_results/florence2_basic
```

**Success Criteria:**
- ✅ Pipeline completes without errors
- ✅ Florence-2 extracts bounding boxes
- ✅ Florence-2 generates region captions
- ✅ Bboxes in [0,1] format
- ✅ Final answer is accurate

#### Test 2: Side-by-Side Comparison

Test the same image/question with both configurations:

```bash
# Test 1: Qwen-only
python test_real_pipeline.py \
    --config configs/test_qwen_only.yaml \
    --save-viz \
    --output-dir test_results/comparison/qwen_only

# Test 2: Qwen + Florence-2
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --save-viz \
    --output-dir test_results/comparison/qwen_florence
```

**Compare:**
- Execution time
- Memory usage
- Bbox accuracy
- Caption quality
- Answer quality

#### Test 3: Multiple Test Cases

Create diverse test cases:

```bash
# Test Case 1: People counting
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --question "How many people are in the image?" \
    --save-viz

# Test Case 2: Object detection
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --question "What objects are visible in the image?" \
    --save-viz

# Test Case 3: Spatial relationships
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --question "What is the person holding?" \
    --save-viz

# Test Case 4: Attribute verification
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --question "What color is the person's shirt?" \
    --save-viz
```

#### Test 4: Performance Benchmarking

Create a batch test script:

```python
# batch_test_florence.py
test_cases = [
    ("demo.jpg", "How many people?"),
    ("demo.jpg", "What objects?"),
    ("demo.jpg", "What is the person wearing?"),
    # ... more cases
]

results = []
for image, question in test_cases:
    result = run_pipeline(image, question, "configs/test_florence2.yaml")
    results.append(result)

# Generate comparison report
create_comparison_report(results)
```

---

## Expected Results

### Florence-2 Advantages

1. **Specialized for Grounding**
   - Trained specifically for phrase grounding
   - Should have better bbox precision
   - Faster inference for grounding tasks

2. **Better Region Captions**
   - Optimized for dense captioning
   - More detailed region descriptions
   - Better spatial understanding

### Performance Expectations

**Qwen-Only:**
```
Total Time: ~34s
- Reasoning: ~10s
- Grounding: ~8s (via Qwen adapter)
- Captioning: ~4s (via Qwen adapter)
- Synthesis: ~12s
Memory: ~1.2GB
```

**Qwen + Florence-2:**
```
Total Time: ~25-30s (expected)
- Reasoning: ~10s (Qwen)
- Grounding: ~3-5s (Florence-2, faster)
- Captioning: ~2-3s (Florence-2, faster)
- Synthesis: ~12s (Qwen)
Memory: ~2.5GB (two models)
```

---

## Implementation Steps

### Step 1: Fix Florence-2 Loading (Choose One)

**Option A: Transformers Downgrade (Easiest)**
```bash
conda activate pytorch
pip install transformers==4.44.0
```

**Option B: Modify Client Code**
```python
# In florence_grounding_client.py line ~71
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    attn_implementation="eager",  # ADD THIS
)
```

**Option C: Patch Model (Advanced)**
```bash
# Find model file
find ~/.cache/huggingface -name "modeling_florence2.py"

# Edit and add to Florence2ForConditionalGeneration class:
_supports_sdpa = False
```

### Step 2: Create Test Configuration

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom

# Create Florence-2 config
cat > configs/test_florence2.yaml << 'EOF'
# [Content from Phase 2 above]
EOF
```

### Step 3: Run Initial Test

```bash
# Test Florence-2 loading first
python -c "
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-large',
    trust_remote_code=True,
    attn_implementation='eager'
)
print('Florence-2 loaded successfully!')
"

# If successful, run full pipeline
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --save-viz
```

### Step 4: Run Comparison Tests

```bash
# Create comparison directory
mkdir -p test_results/comparison

# Run both configurations
./run_comparison_test.sh
```

### Step 5: Analyze Results

```bash
# Check output files
ls test_results/comparison/

# Compare performance
python -c "
import json
qwen = json.load(open('test_results/comparison/qwen_only/results.json'))
florence = json.load(open('test_results/comparison/qwen_florence/results.json'))

print('Qwen-only time:', qwen['timings']['Pipeline Execution'])
print('Florence-2 time:', florence['timings']['Pipeline Execution'])
"
```

---

## Validation Checklist

### ✅ Florence-2 Loading
- [ ] Model loads without errors
- [ ] Processor initializes correctly
- [ ] Model moves to correct GPU (cuda:6)

### ✅ Grounding Quality
- [ ] Bboxes extracted for all steps
- [ ] Bboxes in [0,1] normalized format
- [ ] Bboxes are accurate (visual inspection)
- [ ] Confidence scores present

### ✅ Captioning Quality
- [ ] Captions generated for all regions
- [ ] Captions are detailed and relevant
- [ ] Better than Qwen adapter captions

### ✅ Performance
- [ ] Faster than Qwen-only (expected)
- [ ] Memory usage acceptable (<3GB)
- [ ] No GPU conflicts

### ✅ Pipeline Integration
- [ ] All stages complete successfully
- [ ] Coordinate conversion working
- [ ] Final answer is accurate
- [ ] JSON/Markdown reports generated

---

## Troubleshooting

### Issue: Florence-2 Still Fails

**Try:**
```bash
# Clear cache and reinstall
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Florence*
pip install --upgrade --force-reinstall transformers==4.44.0
```

### Issue: CUDA Out of Memory

**Try:**
```bash
# Use GPU 7 instead
# Edit config: device: "cuda:7"

# Or reduce model precision
# Edit config: torch_dtype: "float16"
```

### Issue: Slow Performance

**Try:**
```bash
# Enable compile for Florence-2
# Edit config: enable_compile: true

# Or reduce token limits
# Edit config: max_new_tokens: 64
```

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Florence-2 Loads | ✅ Success | ⏳ |
| Pipeline Completes | ✅ Success | ⏳ |
| Bbox Validation | 100% valid | ⏳ |
| Faster than Qwen-only | Yes | ⏳ |
| Caption Quality | Better | ⏳ |
| Memory Usage | <3GB | ⏳ |

---

## Next Steps After Success

1. **Benchmark Suite**
   - Create 20+ test cases
   - Measure accuracy metrics
   - Compare Qwen vs Florence-2

2. **Optimization**
   - Enable Flash Attention (if compatible)
   - Enable torch.compile
   - Batch processing

3. **Documentation**
   - Update README with Florence-2 setup
   - Add Florence-2 examples
   - Performance comparison table

4. **Integration**
   - Make Florence-2 the default for grounding/captioning
   - Update all examples
   - Deploy to production

---

## Quick Commands Reference

```bash
# Fix Florence-2
pip install transformers==4.44.0

# Test loading
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True, attn_implementation='eager')"

# Run test
python test_real_pipeline.py --config configs/test_florence2.yaml --save-viz

# Compare
python compare_results.py test_results/comparison/
```

---

**Status:** Ready to implement  
**Estimated Time:** 30-60 minutes  
**Risk:** Low (Qwen-only fallback available)

