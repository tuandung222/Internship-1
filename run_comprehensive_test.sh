#!/bin/bash
# =============================================================================
# Comprehensive CoRGI Pipeline V2 Test
# =============================================================================
# This script runs inference on 3 test cases to exercise ALL components:
# - Reasoning + Grounding (Qwen VL)
# - OCR Evidence (Florence-2)  
# - Caption Evidence (SmolVLM2)
# - Synthesis (Qwen VL)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/qwen_florence2_smolvlm2_v2.yaml"
OUTPUT_BASE="results/comprehensive_test"

echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}   CoRGI COMPREHENSIVE PIPELINE TEST${NC}"
echo -e "${BLUE}=============================================================${NC}"
echo ""
echo "Config: $CONFIG"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# =============================================================================
# Test Case 1: OCR-focused (Chart with data)
# Triggers: Reasoning + Grounding + OCR (Florence-2) + Synthesis
# =============================================================================
echo -e "${YELLOW}=============================================================${NC}"
echo -e "${YELLOW}TEST 1: OCR-Focused (Chart Analysis)${NC}"
echo -e "${YELLOW}Expected: Florence-2 OCR should be triggered${NC}"
echo -e "${YELLOW}=============================================================${NC}"
echo ""

IMAGE1="test_images/chart_gdp.jpg"
QUESTION1="What are the top 3 countries by GDP shown in this chart? Read the exact numbers from the chart."
OUTPUT1="$OUTPUT_BASE/test1_ocr_chart"

echo "Image: $IMAGE1"
echo "Question: $QUESTION1"
echo ""

python inference_traced.py \
    --image "$IMAGE1" \
    --question "$QUESTION1" \
    --config "$CONFIG" \
    --output "$OUTPUT1" \
    --max-steps 3 \
    2>&1 | tee "$OUTPUT_BASE/test1_log.txt"

echo ""
echo -e "${GREEN}✓ Test 1 completed. Results in: $OUTPUT1${NC}"
echo ""

# =============================================================================
# Test Case 2: Visual-focused (Animals)
# Triggers: Reasoning + Grounding + Caption (SmolVLM2) + Synthesis
# =============================================================================
echo -e "${YELLOW}=============================================================${NC}"
echo -e "${YELLOW}TEST 2: Visual-Focused (Animal Scene)${NC}"
echo -e "${YELLOW}Expected: SmolVLM2 Caption should be triggered${NC}"
echo -e "${YELLOW}=============================================================${NC}"
echo ""

IMAGE2="test_images/dog_cat.jpg"
QUESTION2="Describe the interaction between the animals in this image. What breed of dog is shown? Is the cat relaxed or alert?"
OUTPUT2="$OUTPUT_BASE/test2_visual_animals"

echo "Image: $IMAGE2"
echo "Question: $QUESTION2"
echo ""

python inference_traced.py \
    --image "$IMAGE2" \
    --question "$QUESTION2" \
    --config "$CONFIG" \
    --output "$OUTPUT2" \
    --max-steps 3 \
    2>&1 | tee "$OUTPUT_BASE/test2_log.txt"

echo ""
echo -e "${GREEN}✓ Test 2 completed. Results in: $OUTPUT2${NC}"
echo ""

# =============================================================================
# Test Case 3: Mixed (Food with description)
# Triggers: Reasoning + Grounding + BOTH OCR and Caption + Synthesis
# =============================================================================
echo -e "${YELLOW}=============================================================${NC}"
echo -e "${YELLOW}TEST 3: Mixed Content (Food Analysis)${NC}"
echo -e "${YELLOW}Expected: Both OCR and Caption may be triggered${NC}"
echo -e "${YELLOW}=============================================================${NC}"
echo ""

IMAGE3="test_images/burger.jpg"
QUESTION3="Describe this food item in detail. What ingredients can you identify? Is this a healthy meal?"
OUTPUT3="$OUTPUT_BASE/test3_mixed_food"

echo "Image: $IMAGE3"
echo "Question: $QUESTION3"
echo ""

python inference_traced.py \
    --image "$IMAGE3" \
    --question "$QUESTION3" \
    --config "$CONFIG" \
    --output "$OUTPUT3" \
    --max-steps 3 \
    2>&1 | tee "$OUTPUT_BASE/test3_log.txt"

echo ""
echo -e "${GREEN}✓ Test 3 completed. Results in: $OUTPUT3${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}=============================================================${NC}"
echo -e "${BLUE}   TEST SUMMARY${NC}"
echo -e "${BLUE}=============================================================${NC}"
echo ""
echo "All test results saved to: $OUTPUT_BASE/"
echo ""
echo "Test outputs:"
echo "  📊 $OUTPUT1/trace_report.html  - OCR test"
echo "  🐕 $OUTPUT2/trace_report.html  - Visual test"
echo "  🍔 $OUTPUT3/trace_report.html  - Mixed test"
echo ""
echo "Logs:"
echo "  📝 $OUTPUT_BASE/test1_log.txt"
echo "  📝 $OUTPUT_BASE/test2_log.txt"
echo "  📝 $OUTPUT_BASE/test3_log.txt"
echo ""

# Verify components were called
echo -e "${YELLOW}Checking component usage in traces...${NC}"
echo ""

for trace_file in "$OUTPUT1/trace.json" "$OUTPUT2/trace.json" "$OUTPUT3/trace.json"; do
    if [ -f "$trace_file" ]; then
        test_name=$(dirname "$trace_file" | xargs basename)
        echo "=== $test_name ==="
        
        # Check for OCR
        if grep -q '"ocr"' "$trace_file" 2>/dev/null; then
            echo -e "  ${GREEN}✓ OCR (Florence-2) was called${NC}"
        else
            echo -e "  ${YELLOW}○ OCR was NOT called${NC}"
        fi
        
        # Check for Caption
        if grep -q '"caption"' "$trace_file" 2>/dev/null; then
            echo -e "  ${GREEN}✓ Caption (SmolVLM2) was called${NC}"
        else
            echo -e "  ${YELLOW}○ Caption was NOT called${NC}"
        fi
        
        # Check for synthesis
        if grep -q '"synthesis"' "$trace_file" 2>/dev/null; then
            echo -e "  ${GREEN}✓ Synthesis was completed${NC}"
        else
            echo -e "  ${RED}✗ Synthesis NOT found${NC}"
        fi
        
        echo ""
    fi
done

echo -e "${GREEN}=============================================================${NC}"
echo -e "${GREEN}   ALL TESTS COMPLETE!${NC}"
echo -e "${GREEN}=============================================================${NC}"
echo ""
echo "To view detailed traces, open the HTML reports in your browser:"
echo "  firefox $OUTPUT_BASE/test*/trace_report.html"
echo ""
