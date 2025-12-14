#!/bin/bash
# Quick launch script for CoRGI V2 Chatbot

echo "🤖 CoRGI V2 - Streaming Chatbot Launcher"
echo "========================================"
echo ""

# Check if config is provided (default: multi-model setup)
CONFIG="${1:-configs/qwen_florence2_smolvlm2_v2.yaml}"
PORT="${2:-7860}"

echo "📄 Config: $CONFIG"
echo "🔌 Port: $PORT"
echo ""

# Check if file exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -1 configs/*_v2.yaml
    exit 1
fi

echo "🚀 Launching Gradio Chatbot..."
echo "   Open browser at: http://localhost:$PORT"
echo ""

# Launch with Python
python gradio_chatbot_v2.py \
    --config "$CONFIG" \
    --server-port "$PORT"

