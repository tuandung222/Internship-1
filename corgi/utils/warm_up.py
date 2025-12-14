"""
Model Warm-up Utilities

Preload models onto CUDA and run dummy inference to ensure they're ready.

Usage:
    from corgi.utils.warm_up import warm_up_pipeline, WarmUpConfig
    
    # Warm up with default settings
    pipeline = warm_up_pipeline("configs/qwen_florence2_smolvlm2_v2.yaml")
    
    # Warm up with custom settings
    config = WarmUpConfig(run_dummy_inference=True, dummy_question="test")
    pipeline = warm_up_pipeline("configs/...", config)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger("corgi.warm_up")


@dataclass
class WarmUpConfig:
    """Configuration for model warm-up."""
    
    # Whether to run a dummy inference to fully initialize CUDA kernels
    run_dummy_inference: bool = True
    
    # Dummy image size (smaller = faster warm-up)
    dummy_image_size: Tuple[int, int] = (224, 224)
    
    # Dummy question for warm-up inference
    dummy_question: str = "What is in this image?"
    
    # Whether to clear CUDA cache after warm-up
    clear_cache_after: bool = True
    
    # Disable parallel loading (more stable but slower)
    sequential_loading: bool = False
    
    # Timeout for warm-up (seconds)
    timeout: float = 120.0


def create_dummy_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a simple dummy image for warm-up inference."""
    # Create a simple gradient image (fast to process)
    import numpy as np
    
    width, height = size
    # Create RGB gradient
    r = np.linspace(0, 255, width, dtype=np.uint8)
    g = np.linspace(0, 255, height, dtype=np.uint8)
    
    r_grid, g_grid = np.meshgrid(r, g)
    b_grid = np.full((height, width), 128, dtype=np.uint8)
    
    rgb = np.stack([r_grid, g_grid, b_grid], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def warm_up_pipeline(
    config_path: str | Path,
    warm_up_config: Optional[WarmUpConfig] = None,
    use_v2: bool = True,
) -> Any:
    """
    Warm up pipeline by loading models and optionally running dummy inference.
    
    Args:
        config_path: Path to pipeline config YAML
        warm_up_config: Warm-up configuration (default: WarmUpConfig())
        use_v2: Whether to use V2 pipeline (default: True)
        
    Returns:
        Initialized pipeline ready for inference
    """
    from ..core.config import load_config
    from ..models.factory import VLMClientFactory
    
    if warm_up_config is None:
        warm_up_config = WarmUpConfig()
    
    config_path = Path(config_path)
    logger.info("=" * 60)
    logger.info("🔥 WARMING UP PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Pipeline: {'V2' if use_v2 else 'V1'}")
    logger.info(f"Dummy inference: {warm_up_config.run_dummy_inference}")
    
    total_start = time.monotonic()
    
    # Step 1: Load config
    logger.info("[1/4] Loading configuration...")
    config = load_config(str(config_path))
    
    # Step 2: Load models
    logger.info("[2/4] Loading models onto CUDA...")
    load_start = time.monotonic()
    
    parallel_loading = not warm_up_config.sequential_loading
    
    try:
        client = VLMClientFactory.create_from_config(
            config,
            parallel_loading=parallel_loading,
        )
        load_duration = time.monotonic() - load_start
        logger.info(f"✓ Models loaded in {load_duration:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    # Step 3: Create pipeline
    logger.info("[3/4] Creating pipeline...")
    
    if use_v2:
        from ..core.pipeline_v2 import CoRGIPipelineV2
        pipeline = CoRGIPipelineV2(vlm_client=client)
    else:
        from ..core.pipeline import CoRGIPipeline
        pipeline = CoRGIPipeline(vlm_client=client)
    
    # Step 4: Dummy inference (optional but recommended)
    if warm_up_config.run_dummy_inference:
        logger.info("[4/4] Running dummy inference to warm CUDA kernels...")
        
        try:
            dummy_image = create_dummy_image(warm_up_config.dummy_image_size)
            
            inference_start = time.monotonic()
            
            # Run inference with minimal steps
            _ = pipeline.run(
                image=dummy_image,
                question=warm_up_config.dummy_question,
                max_steps=1,
                max_regions=1,
            )
            
            inference_duration = time.monotonic() - inference_start
            logger.info(f"✓ Dummy inference completed in {inference_duration:.2f}s")
            
        except Exception as e:
            logger.warning(f"Dummy inference failed (non-fatal): {e}")
            # Continue anyway - models are still loaded
    else:
        logger.info("[4/4] Skipping dummy inference (disabled)")
    
    # Clear CUDA cache
    if warm_up_config.clear_cache_after and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✓ CUDA cache cleared")
    
    total_duration = time.monotonic() - total_start
    
    logger.info("=" * 60)
    logger.info(f"🚀 PIPELINE READY (total: {total_duration:.2f}s)")
    logger.info("=" * 60)
    
    # Log GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return pipeline


def warm_up_models_only(
    config_path: str | Path,
    sequential: bool = False,
) -> Any:
    """
    Warm up models only (no pipeline, no dummy inference).
    
    Useful for checking if models can be loaded without errors.
    
    Args:
        config_path: Path to pipeline config YAML
        sequential: Load models sequentially (more stable)
        
    Returns:
        VLM client with loaded models
    """
    from ..core.config import load_config
    from ..models.factory import VLMClientFactory
    
    logger.info("Loading models...")
    config = load_config(str(config_path))
    
    client = VLMClientFactory.create_from_config(
        config,
        parallel_loading=not sequential,
    )
    
    logger.info("✓ Models loaded and ready")
    return client


def verify_cuda_ready() -> bool:
    """
    Verify CUDA is available and functional.
    
    Returns:
        True if CUDA is ready, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available!")
        return False
    
    try:
        # Quick CUDA test
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA available: {device_count} GPU(s)")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {name} ({memory:.1f}GB)")
        
        # Test tensor allocation
        test_tensor = torch.tensor([1.0], device="cuda:0")
        del test_tensor
        torch.cuda.empty_cache()
        
        logger.info("✓ CUDA is functional")
        return True
        
    except Exception as e:
        logger.error(f"CUDA verification failed: {e}")
        return False


__all__ = [
    "WarmUpConfig",
    "warm_up_pipeline",
    "warm_up_models_only",
    "verify_cuda_ready",
    "create_dummy_image",
]
