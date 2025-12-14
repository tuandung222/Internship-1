# Getting Started with Real Pipeline Testing

**Quick Start Guide for Testing CoRGi with Real Models**

## What's New?

A comprehensive test script (`test_real_pipeline.py`) has been created to test the CoRGi pipeline with real Qwen3-VL-4B-Instruct + Florence-2 models. This guide will help you get started quickly.

## Prerequisites

### 1. Install Additional Dependencies

```bash
pip install rich psutil
```

These provide:
- `rich`: Beautiful console output with progress bars and tables
- `psutil`: Memory usage monitoring

### 2. Ensure Models Will Be Downloaded

First run will download models automatically (~10GB total):
- Qwen/Qwen3-VL-4B-Instruct (~8GB)
- microsoft/Florence-2-large (~2GB)

Make sure you have:
- ✅ Stable internet connection
- ✅ At least 15GB free disk space
- ✅ GPU with sufficient VRAM (recommended: 16GB+)

## Quick Start - 3 Simple Commands

### 1. Basic Test (Recommended First Run)

```bash
cd /home/dungvpt/workspace/corgi_implementation/corgi_custom
python test_real_pipeline.py
```

This will:
- Load default configuration from `configs/default.yaml`
- Download models if needed (first run only)
- Fetch demo image from Qwen's official demo
- Run the pipeline with the question: "How many people are there in the image? Is there any one who is wearing a white watch?"
- Display results in real-time
- Save JSON and Markdown reports to `test_results/`

**Expected time:** 
- First run: 5-10 minutes (model download)
- Subsequent runs: 30-60 seconds (model loading + inference)

### 2. Test with Visualization

```bash
python test_real_pipeline.py --save-viz
```

Additionally saves an annotated image showing bounding boxes drawn on the original image.

### 3. Test with Custom Image

```bash
python test_real_pipeline.py --image path/to/your/image.jpg --question "Your question here?" --save-viz
```

## Understanding the Output

### Console Output

You'll see a beautifully formatted console output like:

```
╔══════════════════════════════════════════════════════════╗
║  CoRGi Pipeline Test - Real Models                      ║
║  Qwen3-VL-4B-Instruct + Florence-2                      ║
╚══════════════════════════════════════════════════════════╝

[1/5] Loading Configuration...
✓ Config loaded from: configs/default.yaml (0.12s)

[2/5] Loading Models...
⠋ Loading VLM models... (elapsed: 18s)
✓ Reasoning: Qwen/Qwen3-VL-4B-Instruct
✓ Grounding: microsoft/Florence-2-large
✓ Captioning: microsoft/Florence-2-large
✓ Synthesis: Qwen/Qwen3-VL-4B-Instruct

[3/5] Preparing Input...
Question: How many people are there in the image? ...

[4/5] Running Pipeline...
⠋ Processing... (elapsed: 11s)
✓ Pipeline completed in 11.23s

[Tables showing reasoning steps, evidence, and results]

╔══════════════════════════════════════════════════════════╗
║  FINAL ANSWER                                            ║
╠══════════════════════════════════════════════════════════╣
║  There are 2 people in the image. Yes, one person is    ║
║  wearing a white watch on their left wrist.              ║
╚══════════════════════════════════════════════════════════╝

[Performance metrics table]

[5/5] Validation and Saving Results...
✓ All bboxes validated successfully!
✓ Results saved to test_results/
```

### Output Files

Check the `test_results/` directory for:

1. **JSON file**: `pipeline_results_YYYYMMDD_HHMMSS.json`
   - Complete structured output
   - Perfect for programmatic analysis
   - Contains all timings and validation results

2. **Markdown report**: `pipeline_report_YYYYMMDD_HHMMSS.md`
   - Human-readable formatted report
   - Tables and sections
   - Easy to share or review

3. **Annotated image** (if `--save-viz` used): `annotated_YYYYMMDD_HHMMSS.png`
   - Original image with bboxes drawn
   - Color-coded by reasoning step
   - Labeled with step numbers

## What to Look For

### 1. Coordinate Validation

The script automatically validates all bounding boxes:

✅ **Good output:**
```
✓ All bboxes validated successfully!
```

⚠️ **Issues found:**
```
⚠ Bbox validation issues found:
  - Evidence 2 bbox out of range: [1.2, 0.5, 0.8, 0.9]
```

All bboxes should be in `[0, 1]` normalized format.

### 2. Performance Metrics

Look at the performance summary to identify bottlenecks:

```
Performance Summary:
┌──────────────────────┬────────────────┬────────────┐
│ Stage                │ Duration (s)   │ Percentage │
├──────────────────────┼────────────────┼────────────┤
│ Model Loading        │ 18.45          │ 60.2%      │  ← Usually the slowest
│ Pipeline Execution   │ 11.23          │ 36.7%      │
│ Config Loading       │ 0.12           │ 0.4%       │
│ Image Loading        │ 0.23           │ 0.7%       │
└──────────────────────┴────────────────┴────────────┘
```

### 3. Reasoning Quality

Check if the reasoning steps are:
- Non-duplicate (each step verifies different aspect)
- Object-focused (specific noun phrases)
- Appropriate for visual verification

### 4. Evidence Quality

Check if the grounded evidence:
- Has valid bboxes in [0, 1] range
- Has meaningful descriptions
- Relates to the reasoning steps

## Common Issues and Solutions

### Issue: Out of Memory

**Solution 1:** Disable torch.compile in config:
```yaml
# In configs/default.yaml
enable_compile: false
```

**Solution 2:** Use smaller batch size or reduce max_new_tokens

### Issue: Models Download Slowly

**Solution:** This is normal for first run. Models are ~10GB total. Be patient!

### Issue: Import Error for 'rich'

**Solution:** 
```bash
pip install rich psutil
```

### Issue: CUDA Out of Memory

**Solution:** You need a GPU with at least 16GB VRAM. Consider:
- Closing other GPU processes
- Using a smaller model
- Running on a machine with more VRAM

## Advanced Usage

### Test Different Configurations

Create custom config files:

```bash
# Copy default config
cp configs/default.yaml configs/my_config.yaml

# Edit my_config.yaml to change models, parameters, etc.

# Run with custom config
python test_real_pipeline.py --config configs/my_config.yaml
```

### Batch Testing Multiple Images

```bash
# Test all images in a directory
for img in test_images/*.jpg; do
    echo "Testing: $img"
    python test_real_pipeline.py --image "$img" --save-viz --no-progress
done
```

### Analyze Results Programmatically

```python
import json
import glob

# Load all test results
results = []
for file in glob.glob('test_results/pipeline_results_*.json'):
    with open(file) as f:
        results.append(json.load(f))

# Calculate average performance
avg_time = sum(r['timings']['Pipeline Execution'] for r in results) / len(results)
print(f"Average pipeline time: {avg_time:.2f}s")

# Check validation status
all_valid = all(r['coordinate_validation']['all_valid'] for r in results)
print(f"All tests passed validation: {all_valid}")
```

## What's Been Tested

The implementation has been validated through:

1. ✅ **Unit tests**: All coordinate conversion utilities
2. ✅ **Integration tests**: Full pipeline with mocked models
3. ✅ **Script validation**: Help output and argument parsing
4. ✅ **Code quality**: Linting (minor warnings for uninstalled dependencies)

## Next Steps

1. **Run Your First Test**
   ```bash
   python test_real_pipeline.py
   ```

2. **Review the Output**
   - Check console output for any issues
   - Open the Markdown report in `test_results/`
   - If you used `--save-viz`, view the annotated image

3. **Verify Coordinates**
   - Ensure all bboxes passed validation
   - Check JSON output for detailed validation results

4. **Optimize if Needed**
   - Review performance metrics
   - Adjust config settings if certain stages are slow
   - Re-run to compare

5. **Test with Your Own Data**
   ```bash
   python test_real_pipeline.py --image your_image.jpg --question "Your question?" --save-viz
   ```

## Documentation

For more details, see:
- **`TEST_REAL_PIPELINE_README.md`**: Complete usage guide
- **`REAL_PIPELINE_TEST_IMPLEMENTATION.md`**: Technical implementation details
- **`PLAN_COMPLETION_SUMMARY.md`**: Summary of what was implemented

## Support

If you encounter issues:
1. Check the troubleshooting section in `TEST_REAL_PIPELINE_README.md`
2. Review the validation output for specific error messages
3. Check GPU/memory availability
4. Ensure all dependencies are installed

## Summary

You now have a professional testing framework that:
- ✅ Tests the full pipeline with real models
- ✅ Provides real-time progress monitoring
- ✅ Validates all outputs automatically
- ✅ Generates comprehensive reports
- ✅ Supports custom configurations
- ✅ Handles errors gracefully

**Start testing now:**
```bash
python test_real_pipeline.py
```

Good luck! 🚀


