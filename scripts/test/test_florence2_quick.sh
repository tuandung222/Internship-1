#!/bin/bash
# Quick test script for Florence-2 + Qwen3-VL pipeline

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Florence-2 + Qwen3-VL Pipeline Test                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

echo "Step 1: Check if Florence-2 can be loaded..."
python -c "
import sys
try:
    from transformers import AutoModelForCausalLM
    print('Testing Florence-2 loading with eager attention...')
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Florence-2-large',
        trust_remote_code=True,
        attn_implementation='eager',
        torch_dtype='auto'
    )
    print('✅ Florence-2 loaded successfully!')
    sys.exit(0)
except Exception as e:
    print(f'❌ Florence-2 loading failed: {e}')
    print('')
    print('Possible fixes:')
    print('1. pip install transformers==4.44.0')
    print('2. Check FLORENCE2_TEST_PLAN.md for more solutions')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Florence-2 loading failed. Please fix the issue first."
    exit 1
fi

echo ""
echo "Step 2: Running full pipeline test..."
python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --save-viz \
    --output-dir test_results/florence2_test

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Test Complete!                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: test_results/florence2_test/"
echo ""
echo "Check the output files:"
echo "  - pipeline_results_*.json  (structured data)"
echo "  - pipeline_report_*.md     (human-readable report)"
echo "  - annotated_*.png          (visualization)"
echo ""

