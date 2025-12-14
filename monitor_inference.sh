#!/bin/bash

# Monitor inference progress in real-time
LOG_FILE="$1"
[[ -z "$LOG_FILE" ]] && LOG_FILE="logs/inference_multi_model.log"

echo "🔍 Monitoring: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== CoRGI Inference Monitor ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo "================================"
    echo ""
    
    if [[ -f "$LOG_FILE" ]]; then
        # Show key events
        echo "📊 Latest Events:"
        tail -50 "$LOG_FILE" | grep -E "(INFO|ERROR|✓|Phase|loaded|SUCCESS|complete)" | tail -10
        echo ""
        
        # Check phase status
        if grep -q "Phase 1" "$LOG_FILE"; then
            echo "✓ Phase 1: Reasoning+Grounding Started"
        fi
        if grep -q "Phase 3" "$LOG_FILE"; then
            echo "✓ Phase 3: Evidence Extraction Started"
        fi
        if grep -q "Phase 4" "$LOG_FILE"; then
            echo "✓ Phase 4: Synthesis Started"
        fi
        if grep -q "SUCCESS" "$LOG_FILE"; then
            echo "✅ INFERENCE COMPLETE!"
            break
        fi
        if grep -q "ERROR" "$LOG_FILE"; then
            echo "❌ ERROR DETECTED - Check log for details"
        fi
    else
        echo "⏳ Waiting for log file..."
    fi
    
    sleep 5
done

