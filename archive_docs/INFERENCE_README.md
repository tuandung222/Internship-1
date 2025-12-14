# CoRGI Batch Inference Script

## Quick Start

```bash
# Single image
python inference.py --image sample.jpg --question "What is this?" --output results/

# Batch processing
python inference.py --batch batch_example.txt --output results/

# Production mode (optimized)
CORGI_LOG_LEVEL=WARNING python inference.py --image sample.jpg --question "..." --output results/
```

## Features

✅ **No UI Required**: Run inference from command line
✅ **Batch Processing**: Process multiple images from a file
✅ **Complete Results**: Saves everything - answer, evidence, visualizations, JSON
✅ **Organized Output**: Results saved in structured folders
✅ **Production Ready**: Works with all config files and optimizations

## Output Structure

```
results/
├── images/
│   └── original.jpg              # Original image
├── evidence/
│   ├── evidence_step0_region0.jpg  # Individual evidence crops
│   └── ...
├── visualizations/
│   └── annotated.jpg             # Image with bounding boxes
├── results.json                   # Complete pipeline results (JSON)
└── summary.txt                    # Human-readable report
```

## See Full Documentation

- [BATCH_INFERENCE.md](docs/BATCH_INFERENCE.md) - Complete usage guide with examples
- [batch_example.txt](batch_example.txt) - Example batch input file

## Created Files

1. **inference.py** - Main script (executable)
2. **docs/BATCH_INFERENCE.md** - Complete documentation
3. **batch_example.txt** - Example batch file format
