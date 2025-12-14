"""
⚠️ DEPRECATED: Use app_unified.py instead.

    python app_unified.py --pipeline v2

This file is kept for backward compatibility.

---

CoRGI Pipeline V2 - Gradio Application (DEPRECATED)

Enhanced Gradio UI for Pipeline V2 with:
- Qwen3-Thinking support
- Optimized prompts (80% token reduction)
- Smart evidence routing
- V2-specific stats
"""

import warnings
warnings.warn(
    "app_v2.py is deprecated. Use 'python app_unified.py --pipeline v2' instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys

# Set production logging if not already set
if "CORGI_LOG_LEVEL" not in os.environ:
    os.environ["CORGI_LOG_LEVEL"] = "INFO"

from corgi.utils.production_logging import configure_production_logging

configure_production_logging()

from corgi.ui.gradio_app import build_demo


def main():
    """Launch Gradio app for Pipeline V2."""

    # Build demo with V2 config filter
    demo = build_demo(
        config_filter="v2",  # Only show V2 configs
        title="CoRGI Pipeline V2 🚀",
        description="""
        **Pipeline V2 Features:**
        - ⚡ 30-40% faster (merged Phase 1+2)
        - 🎯 Smart routing (OCR OR Caption, not both)
        - 🧠 Qwen3-Thinking support
        - 📊 80% fewer prompt tokens
        - ✨ Better quality (Florence-2 + SmolVLM2)
        """,
    )

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from V1 (7860)
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
