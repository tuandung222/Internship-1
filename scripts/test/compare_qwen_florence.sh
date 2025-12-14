#!/bin/bash
# Compare Qwen-only vs Qwen+Florence-2 configurations

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Qwen-only vs Qwen+Florence-2 Comparison Test                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

# Create comparison directory
mkdir -p test_results/comparison

# Test question
QUESTION="How many people are there in the image? Is there any one who is wearing a white watch?"

echo "Test Question: $QUESTION"
echo ""

# Test 1: Qwen-only
echo "═══════════════════════════════════════════════════════════════"
echo "Test 1/2: Qwen-only Configuration"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python test_real_pipeline.py \
    --config configs/test_qwen_only.yaml \
    --question "$QUESTION" \
    --save-viz \
    --output-dir test_results/comparison/qwen_only

echo ""
echo "✅ Qwen-only test complete"
echo ""

# Test 2: Qwen + Florence-2
echo "═══════════════════════════════════════════════════════════════"
echo "Test 2/2: Qwen + Florence-2 Configuration"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python test_real_pipeline.py \
    --config configs/test_florence2.yaml \
    --question "$QUESTION" \
    --save-viz \
    --output-dir test_results/comparison/qwen_florence

echo ""
echo "✅ Qwen+Florence-2 test complete"
echo ""

# Generate comparison report
echo "═══════════════════════════════════════════════════════════════"
echo "Generating Comparison Report"
echo "═══════════════════════════════════════════════════════════════"
echo ""

python -c "
import json
from pathlib import Path

# Load results
qwen_file = Path('test_results/comparison/qwen_only').glob('pipeline_results_*.json')
florence_file = Path('test_results/comparison/qwen_florence').glob('pipeline_results_*.json')

qwen_path = list(qwen_file)[0]
florence_path = list(florence_file)[0]

qwen = json.load(open(qwen_path))
florence = json.load(open(florence_path))

print('╔════════════════════════════════════════════════════════════════╗')
print('║  Performance Comparison                                        ║')
print('╚════════════════════════════════════════════════════════════════╝')
print()

print(f'Configuration       | Qwen-only | Qwen+Florence-2')
print(f'--------------------|-----------|-----------------')

# Timing comparison
qwen_time = qwen['timings']['Pipeline Execution']
florence_time = florence['timings']['Pipeline Execution']
speedup = (qwen_time / florence_time - 1) * 100

print(f'Pipeline Time       | {qwen_time:6.2f}s  | {florence_time:6.2f}s ({speedup:+.1f}%)')

# Total time
qwen_total = sum(qwen['timings'].values())
florence_total = sum(florence['timings'].values())
total_speedup = (qwen_total / florence_total - 1) * 100

print(f'Total Time          | {qwen_total:6.2f}s  | {florence_total:6.2f}s ({total_speedup:+.1f}%)')

# Steps and evidence
qwen_steps = len(qwen['results']['steps'])
florence_steps = len(florence['results']['steps'])

qwen_evidence = len(qwen['results']['evidence'])
florence_evidence = len(florence['results']['evidence'])

print(f'Reasoning Steps     | {qwen_steps:6d}    | {florence_steps:6d}')
print(f'Visual Evidence     | {qwen_evidence:6d}    | {florence_evidence:6d}')

print()
print('Output Files:')
print(f'  Qwen-only:        test_results/comparison/qwen_only/')
print(f'  Qwen+Florence-2:  test_results/comparison/qwen_florence/')
print()
"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Comparison Complete!                                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Review reports in test_results/comparison/"
echo "  2. Compare visualizations (annotated_*.png)"
echo "  3. Check caption quality in markdown reports"
echo ""

