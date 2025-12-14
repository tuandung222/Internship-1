#!/usr/bin/env python3
"""
CoRGI Unified Gradio Application

A single entrypoint for all CoRGI Gradio interfaces.

Modes:
    - standard: Traditional form-based UI with final results
    - chatbot: Streaming chatbot-style UI with step-by-step execution

Usage:
    # Default: V2 pipeline, standard mode
    python app_unified.py

    # V2 with chatbot streaming
    python app_unified.py --mode chatbot

    # V1 pipeline
    python app_unified.py --pipeline v1

    # Custom config
    python app_unified.py --config configs/qwen_florence2_smolvlm2_v2.yaml

    # HuggingFace Spaces deployment
    python app_unified.py --spaces

    # Custom port and share
    python app_unified.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# Logging Configuration
# =============================================================================


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("corgi.app")


# =============================================================================
# Config Resolution
# =============================================================================

# Default config paths
CONFIG_DIR = Path(__file__).parent / "configs"

# Recommended configs for each pipeline version
# Default: Multi-model V2 (Qwen + Florence-2 + SmolVLM2)
DEFAULT_CONFIGS = {
    "v2": CONFIG_DIR / "qwen_florence2_smolvlm2_v2.yaml",
    "v1": CONFIG_DIR / "legacy" / "qwen_only.yaml",
}

# Fallback configs (in priority order)
FALLBACK_CONFIGS = [
    CONFIG_DIR / "qwen_florence2_smolvlm2_v2.yaml",  # Multi-model V2 (preferred)
    CONFIG_DIR / "qwen_only_v2.yaml",                 # Single-model V2
    CONFIG_DIR / "default_v2.yaml",
    CONFIG_DIR / "qwen_only.yaml",
]


def resolve_config(
    config_path: Optional[Path],
    pipeline_version: str,
) -> Path:
    """
    Resolve the config file path.
    
    Args:
        config_path: Explicit config path (if provided)
        pipeline_version: "v1" or "v2"
        
    Returns:
        Resolved config file path
    """
    # Use explicit config if provided
    if config_path and config_path.exists():
        return config_path
    
    # Use default for pipeline version
    default = DEFAULT_CONFIGS.get(pipeline_version)
    if default and default.exists():
        return default
    
    # Fallback search
    for fallback in FALLBACK_CONFIGS:
        if fallback.exists():
            return fallback
    
    raise FileNotFoundError(
        f"No config file found. Tried: {config_path}, {default}, {FALLBACK_CONFIGS}"
    )


def detect_pipeline_version(config_path: Path) -> str:
    """Detect pipeline version from config file."""
    config_name = config_path.name.lower()
    
    if "v2" in config_name or "_v2" in config_name:
        return "v2"
    
    # Check config content
    try:
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        if config_data.get("pipeline", {}).get("use_v2", False):
            return "v2"
    except Exception:
        pass
    
    return "v1"


# =============================================================================
# App Builders
# =============================================================================


def build_standard_app(
    config_path: Path,
    pipeline_version: str,
):
    """
    Build standard Gradio app (form-based UI).
    
    Args:
        config_path: Path to config file
        pipeline_version: "v1" or "v2"
        
    Returns:
        Gradio Blocks demo
    """
    from corgi.ui.gradio_app import build_demo
    
    # Build demo with the specified config
    # Note: build_demo only accepts default_config parameter
    demo = build_demo(default_config=str(config_path))
    
    return demo


def build_chatbot_app(
    config_path: Path,
    pipeline_version: str,
):
    """
    Build chatbot-style Gradio app (streaming UI).
    
    Uses the refactored gradio_chatbot_v2.py which leverages the streaming API.
    
    Args:
        config_path: Path to config file
        pipeline_version: "v1" or "v2"
        
    Returns:
        Gradio Blocks demo
    """
    # Import the chatbot module (uses streaming API internally)
    import gradio_chatbot_v2
    
    # Set the default config
    gradio_chatbot_v2.current_config = str(config_path)
    
    # Return the pre-built demo
    return gradio_chatbot_v2.demo


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CoRGI Unified Gradio Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (V2 pipeline, standard mode)
    python app_unified.py

    # Chatbot mode with streaming
    python app_unified.py --mode chatbot

    # V1 pipeline
    python app_unified.py --pipeline v1

    # Custom config and port
    python app_unified.py --config configs/qwen_florence2_smolvlm2_v2.yaml --port 7861

    # HuggingFace Spaces deployment
    python app_unified.py --spaces
        """,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["standard", "chatbot"],
        default="standard",
        help="UI mode: standard (form-based) or chatbot (streaming). Default: standard",
    )
    
    # Pipeline version
    parser.add_argument(
        "--pipeline",
        choices=["v1", "v2", "auto"],
        default="auto",
        help="Pipeline version: v1, v2, or auto (detect from config). Default: auto",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to pipeline config YAML. Default: auto-select based on pipeline version",
    )
    
    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port. Default: 7860",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host. Default: 0.0.0.0",
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    
    # HuggingFace Spaces mode
    parser.add_argument(
        "--spaces",
        action="store_true",
        help="Enable HuggingFace Spaces optimizations (queue, etc.)",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Default: INFO",
    )
    
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="Skip model warm-up (faster start but first inference will be slower)",
    )
    
    parser.add_argument(
        "--sequential-loading",
        action="store_true",
        help="Load models sequentially instead of parallel (more stable but slower)",
    )
    
    return parser.parse_args()


def warm_up_models(
    config_path: Path,
    pipeline_version: str,
    logger,
    sequential_loading: bool = False,
) -> None:
    """
    Warm up models on CUDA before launching the app.
    
    This ensures models are loaded and ready, avoiding cold start latency.
    
    Args:
        config_path: Path to config file
        pipeline_version: "v1" or "v2"
        logger: Logger instance
        sequential_loading: Load models sequentially (more stable)
    """
    from corgi.utils.warm_up import warm_up_pipeline, WarmUpConfig, verify_cuda_ready
    
    # Verify CUDA first
    if not verify_cuda_ready():
        logger.warning("CUDA not available, skipping warm-up")
        return
    
    # Configure warm-up
    warm_up_config = WarmUpConfig(
        run_dummy_inference=True,  # Run dummy inference to warm CUDA kernels
        dummy_image_size=(224, 224),  # Small image for fast warm-up
        clear_cache_after=True,
        sequential_loading=sequential_loading,
    )
    
    try:
        # Warm up the pipeline
        use_v2 = pipeline_version == "v2"
        _pipeline = warm_up_pipeline(
            config_path=config_path,
            warm_up_config=warm_up_config,
            use_v2=use_v2,
        )
        
        # Store in global cache so Gradio can reuse
        # (The gradio_app module has its own caching mechanism)
        logger.info("✓ Models warmed up and ready for inference")
        
    except Exception as e:
        logger.error(f"Warm-up failed: {e}")
        logger.warning("Continuing without warm-up (first inference will be slower)")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    os.environ["CORGI_LOG_LEVEL"] = args.log_level
    logger = configure_logging(args.log_level)
    
    # Resolve config
    try:
        # Determine pipeline version
        if args.pipeline == "auto":
            if args.config:
                pipeline_version = detect_pipeline_version(args.config)
            else:
                pipeline_version = "v2"  # Default to V2
        else:
            pipeline_version = args.pipeline
        
        config_path = resolve_config(args.config, pipeline_version)
        logger.info(f"Using config: {config_path}")
        logger.info(f"Pipeline version: {pipeline_version.upper()}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # =========================================================================
    # WARM UP MODELS ON CUDA
    # =========================================================================
    if not args.no_warm_up:
        logger.info("")
        logger.info("Warming up models on CUDA...")
        warm_up_models(
            config_path=config_path,
            pipeline_version=pipeline_version,
            logger=logger,
            sequential_loading=args.sequential_loading,
        )
        logger.info("")
    else:
        logger.info("Skipping model warm-up (--no-warm-up)")
        logger.info("")
    
    # Build app
    logger.info(f"Building {args.mode} UI...")
    
    if args.mode == "chatbot":
        demo = build_chatbot_app(config_path, pipeline_version)
    else:
        demo = build_standard_app(config_path, pipeline_version)
    
    # Configure queue for Spaces
    if args.spaces:
        logger.info("Configuring for HuggingFace Spaces...")
        try:
            demo = demo.queue(concurrency_count=1)
        except TypeError:
            demo = demo.queue()
    
    # Launch
    logger.info("=" * 60)
    logger.info(f"CoRGI {pipeline_version.upper()} - {args.mode.capitalize()} Mode")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path.name}")
    logger.info(f"URL: http://{args.host}:{args.port}")
    if args.share:
        logger.info("Public link will be generated...")
    logger.info("=" * 60)
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


# =============================================================================
# Module-level demo for HuggingFace Spaces import
# =============================================================================

# When imported (e.g., by Spaces), we don't auto-build demo
# The caller should use build_standard_app() or build_chatbot_app() explicitly
# This avoids import-time errors and gives more control

demo = None  # Will be set by caller if needed


if __name__ == "__main__":
    main()
