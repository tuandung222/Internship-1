"""Hugging Face Spaces entrypoint for the CoRGI Qwen3-VL + Florence-2 demo."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure huggingface_hub is at least version 1.0.0 before importing anything
try:
    import huggingface_hub
    import spaces
    from packaging import version
    if version.parse(huggingface_hub.__version__) < version.parse("1.0.0"):
        print(f"Upgrading huggingface_hub from {huggingface_hub.__version__} to >=1.0.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub>=1.0.0,<2.0.0"])
        # Reload module after upgrade
        import importlib
        importlib.reload(huggingface_hub)
except ImportError:
    # If huggingface_hub is not installed, install it
    # print("Installing huggingface_hub>=1.0.0...")
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=1.0.0,<2.0.0"])
    # import huggingface_hub
    pass
except Exception as e:
    print(f"Warning: Could not check/upgrade huggingface_hub: {e}")
    # Continue anyway - requirements.txt should handle it

from corgi.ui.gradio_app import build_demo


def _configure_logging() -> logging.Logger:
    level = os.getenv("CORGI_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("corgi.app")


logger = _configure_logging()
logger.info("Initializing Gradio demo build with Qwen3-VL-2B + PaddleOCR-VL + FastVLM-1.5B configuration.")

# Priority order for config files:
# 1. New pipeline config (Qwen + PaddleOCR + FastVLM) - preferred
# 2. Legacy Vintern config (for backward compatibility)
# 3. Spaces config (fallback)
config_dir = Path(__file__).parent / "configs"
new_pipeline_config = config_dir / "qwen_paddleocr_fastvlm.yaml"
legacy_vintern_config = config_dir / "qwen_vintern.yaml"
spaces_config = config_dir / "florence_qwen_spaces.yaml"

if new_pipeline_config.exists():
    logger.info(f"Using new pipeline config: {new_pipeline_config}")
    logger.info("Pipeline: Qwen3-VL-2B (Reasoning/Grounding/Synthesis) + PaddleOCR-VL (OCR) + FastVLM-1.5B (Captioning+VQA)")
    default_config = str(new_pipeline_config)
elif legacy_vintern_config.exists():
    logger.info(f"Using legacy Vintern config: {legacy_vintern_config}")
    logger.info("Pipeline: Qwen3-VL-2B + Vintern-1B-v3.5 (deprecated)")
    default_config = str(legacy_vintern_config)
elif spaces_config.exists():
    logger.info(f"Using Spaces-optimized config: {spaces_config}")
    default_config = str(spaces_config)
else:
    logger.warning("No config file found, using default qwen_only.yaml")
    default_config = str(config_dir / "qwen_only.yaml")

# Build demo with pipeline config
demo = build_demo(default_config=default_config)
logger.info("Gradio Blocks created; configuring queue.")
try:  # Gradio >=4.29 supports concurrency_count
    demo = demo.queue(concurrency_count=1)
    logger.info("Queue configured with concurrency_count=1.")
except TypeError:
    logger.warning("concurrency_count unsupported; falling back to default queue().")
    demo = demo.queue()

if __name__ == "__main__":
    logger.info("Launching Gradio demo (Qwen3-VL-2B + PaddleOCR-VL + FastVLM-1.5B).")
    demo.launch()
