"""
⚠️ DEPRECATED: Use app_unified.py instead.

    python app_unified.py --config configs/qwen_only_v2.yaml

This file is kept for backward compatibility with HuggingFace Spaces.

---

Hugging Face Spaces entrypoint for the CoRGI Qwen3-VL demo (DEPRECATED).
"""

import warnings
warnings.warn(
    "app_qwen_only.py is deprecated. Use 'python app_unified.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

from __future__ import annotations

import logging
import os
from pathlib import Path

from corgi.ui.gradio_app import build_demo, DEFAULT_QWEN_CONFIG


def _configure_logging() -> logging.Logger:
    level = os.getenv("CORGI_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("corgi.app")


logger = _configure_logging()
logger.info("Initializing Gradio demo build with Qwen-only configuration.")

# Build demo with Qwen-only config
demo = build_demo(default_config=str(DEFAULT_QWEN_CONFIG))
logger.info("Gradio Blocks created; configuring queue.")
try:  # Gradio >=4.29 supports concurrency_count
    demo = demo.queue(concurrency_count=1)
    logger.info("Queue configured with concurrency_count=1.")
except TypeError:
    logger.warning("concurrency_count unsupported; falling back to default queue().")
    demo = demo.queue()

if __name__ == "__main__":
    logger.info("Launching Gradio demo.")
    demo.launch()
