# 🚀 Quick Start Guide - CoRGi Pipeline Testing

**Status:** ✅ Ready to use  
**Last Updated:** November 8, 2025

---

## TL;DR - Run in 30 Seconds

```bash
# 1. Activate environment
conda activate pytorch

# 2. Run test
python test_real_pipeline.py --config configs/test_qwen_only.yaml --save-viz

# 3. Check results
ls test_results/
```

**That's it!** 🎉

---

## What You Get

- ✅ Complete pipeline execution (~34 seconds)
- ✅ JSON results with all data
- ✅ Markdown report (human-readable)
- ✅ Bbox visualization image
- ✅ Performance metrics
- ✅ Coordinate validation

---

## Example Output

**Question:**
> "How many people are there in the image? Is there any one who is wearing a white watch?"

**Answer:**
> "There is one person in the image, and she is wearing a white watch."

**Files Generated:**
```
test_results/
├── pipeline_results_20251108_011111.json   # Structured data
├── pipeline_report_20251108_011111.md      # Human report
└── annotated_20251108_011111.png           # Visual bboxes
```

---

## Try Your Own Image

```bash
python test_real_pipeline.py \
    --config configs/test_qwen_only.yaml \
    --image /path/to/your/image.jpg \
    --question "What objects are in the image?" \
    --save-viz
```

---

## Performance

- **Model:** Qwen3-VL-4B-Instruct
- **Total Time:** ~34 seconds
- **Memory:** ~1.2 GB
- **GPU:** Optional (works on CPU)

---

## Configuration Files

- `configs/test_qwen_only.yaml` - **Recommended** (working)
- `configs/default.yaml` - Uses Florence-2 (has issues)

---

## Documentation

- **BUG_FIX_SUMMARY.md** - What bugs were fixed
- **TESTING_COMPLETE.md** - Full technical details
- **TEST_REAL_PIPELINE_README.md** - Complete usage guide
- **This file** - Quick start

---

## Troubleshooting

### Issue: Out of memory
**Solution:** Reduce `max_steps` and `max_regions` in config

### Issue: Model download slow
**Solution:** First run downloads models (~2GB), be patient

### Issue: CUDA out of memory
**Solution:** Will fallback to CPU automatically

---

## Need Help?

1. Check `BUG_FIX_SUMMARY.md` for known issues
2. Check `TESTING_COMPLETE.md` for full details
3. Check `test_results/*.md` for example outputs

---

**Happy Testing!** 🎉

