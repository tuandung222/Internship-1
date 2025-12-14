"""
PaddleOCR-VL Client.

Specialized client for OCR using PaddlePaddle/PaddleOCR-VL model.
Supports tasks: 'ocr', 'table', 'chart', 'formula'.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import time
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ...core.config import ModelConfig
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    import spaces  # type: ignore
except ImportError:  # pragma: no cover
    spaces = None


def _gpu_decorator(duration: int = 120):
    """Decorator for GPU execution on HF Spaces."""
    if spaces is None:
        return lambda fn: fn
    return spaces.GPU(duration=duration)


# Task prompts mapping
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}
_PROCESSOR_CACHE: dict[str, AutoProcessor] = {}


def _load_paddleocr_backend(
    config: ModelConfig, task: str = "ocr"
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """
    Load PaddleOCR-VL model and processor.

    Args:
        config: Model configuration
        task: OCR task type ('ocr', 'table', 'chart', 'formula')

    Returns:
        Tuple of (model, processor)
    """
    model_id = config.model_id
    cache_key = f"{model_id}_{task}"

    if cache_key not in _MODEL_CACHE:
        import time

        load_start = time.time()
        logger.info(f"Loading PaddleOCR-VL model: {model_id} (task: {task})...")

        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Using bfloat16 (hardware supported)")
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
            logger.info("Using float16 (bfloat16 not supported on this GPU)")
        else:
            torch_dtype = torch.float32
            logger.info("Using float32 (CPU mode)")

        # Override if specified in config
        if config.torch_dtype and config.torch_dtype != "auto":
            torch_dtype = getattr(torch, config.torch_dtype, torch_dtype)

        # Get device from config
        device_map = config.device
        if not device_map:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")

        # Handle CUDA_VISIBLE_DEVICES: if set, PyTorch maps visible GPUs to cuda:0, cuda:1, etc.
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and device_map.startswith("cuda:"):
            # Map device to cuda:0 if CUDA_VISIBLE_DEVICES is set
            device_map = "cuda:0"
            logger.info(
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected, mapping to {device_map}"
            )

        # Patch flash_attn import to avoid flash-attn2 conflict
        # PaddleOCR-VL's modeling code tries to import flash_attn, which conflicts with flash-attn2
        # We'll patch transformers.utils.import_utils to bypass flash_attn checks, then use kernels-community/flash-attn3
        import sys
        import types
        import importlib.util

        # Patch transformers.utils.import_utils to bypass flash_attn availability check
        # This prevents the ValueError: flash_attn.__spec__ is None error
        original_is_package_available = None
        original_is_flash_attn_2_available = None
        patched = False

        try:
            from transformers.utils import import_utils

            original_is_package_available = import_utils._is_package_available
            original_is_flash_attn_2_available = getattr(
                import_utils, "is_flash_attn_2_available", None
            )

            def patched_is_package_available(pkg_name: str) -> bool:
                """Patched version that returns False for flash_attn to bypass checks."""
                if pkg_name == "flash_attn":
                    return False  # Pretend flash_attn is not available
                return original_is_package_available(pkg_name)

            def patched_is_flash_attn_2_available() -> bool:
                """Patched version that always returns False."""
                return False

            # Apply patches
            import_utils._is_package_available = patched_is_package_available
            if original_is_flash_attn_2_available:
                import_utils.is_flash_attn_2_available = (
                    patched_is_flash_attn_2_available
                )

            patched = True
            logger.debug(
                "Patched transformers.utils.import_utils to bypass flash_attn checks"
            )
        except Exception as e:
            logger.warning(f"Could not patch transformers.utils.import_utils: {e}")
            # Continue anyway, will try to load without patch

        try:
            # PaddleOCR-VL has compatibility issues with flash-attn3 (cu_seqlens_q errors)
            # Use SDPA (scaled_dot_product_attention) instead for optimized inference
            model_load_start = time.time()
            try:
                # Try SDPA first (faster than standard attention, compatible with PaddleOCR-VL)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map=device_map if device_map != "cpu" else None,
                    attn_implementation="sdpa",
                )
                logger.info(
                    f"✓ SDPA (scaled_dot_product_attention) enabled for {model_id}"
                )
            except Exception as e:
                logger.warning(
                    f"SDPA not available ({e}), falling back to standard attention"
                )
                # Fallback: use standard attention (most compatible)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map=device_map if device_map != "cpu" else None,
                    attn_implementation="eager",  # Explicitly use standard attention
                )

            model = model.eval()

            if device_map == "cpu":
                model = model.to("cpu")

            model_load_time = time.time() - model_load_start
            logger.info(f"Model weights loaded in {model_load_time:.2f}s")
            logger.info(f"✓ PaddleOCR-VL model loaded on {device_map}")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL model: {e}", exc_info=True)
            raise
        finally:
            # Restore original functions if they were patched
            if patched:
                try:
                    from transformers.utils import import_utils

                    if original_is_package_available is not None:
                        import_utils._is_package_available = (
                            original_is_package_available
                        )
                    if original_is_flash_attn_2_available is not None:
                        import_utils.is_flash_attn_2_available = (
                            original_is_flash_attn_2_available
                        )
                    logger.debug(
                        "Restored original transformers.utils.import_utils functions"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not restore transformers.utils.import_utils: {e}"
                    )

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"✓ PaddleOCR-VL processor loaded")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL processor: {e}", exc_info=True)
            raise

        load_duration = time.time() - load_start
        logger.info(f"✓ PaddleOCR-VL model fully loaded in {load_duration:.2f}s total")

        _MODEL_CACHE[cache_key] = model
        _PROCESSOR_CACHE[cache_key] = processor

    return _MODEL_CACHE[cache_key], _PROCESSOR_CACHE[cache_key]


@ModelRegistry.register_captioning("paddleocr")
class PaddleOCRClient:
    """
    Client for PaddleOCR-VL model for OCR tasks.

    Supports tasks: 'ocr', 'table', 'chart', 'formula'.
    """

    def __init__(
        self,
        config: ModelConfig,
        task: str = "ocr",
        image_logger=None,
        output_tracer=None,
    ) -> None:
        """
        Initialize PaddleOCR-VL client.

        Args:
            config: Model configuration
            task: OCR task type ('ocr', 'table', 'chart', 'formula')
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        if task not in PROMPTS:
            raise ValueError(
                f"Invalid task: {task}. Must be one of {list(PROMPTS.keys())}"
            )

        self.config = config
        self.task = task
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Load model in GPU context immediately (required for HF Spaces)
        self._model, self._processor = _load_paddleocr_backend_in_gpu_context(
            config, task
        )

    @staticmethod
    def _crop_region(
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
    ) -> Image.Image:
        """Crop image to bounding box region with 25% extension on each side."""
        x1, y1, x2, y2 = bbox

        # Extend bbox by 25% on each dimension (12.5% on each side)
        width = x2 - x1
        height = y2 - y1
        extend_x = width * 0.125  # 12.5% of width on each side = 25% total
        extend_y = height * 0.125  # 12.5% of height on each side = 25% total

        # Apply extension
        x1_extended = max(0.0, x1 - extend_x)
        y1_extended = max(0.0, y1 - extend_y)
        x2_extended = min(1.0, x2 + extend_x)
        y2_extended = min(1.0, y2 + extend_y)

        w, h = image.size

        # Convert normalized to pixel coordinates
        left = int(x1_extended * w)
        top = int(y1_extended * h)
        right = int(x2_extended * w)
        bottom = int(y2_extended * h)

        # Ensure valid coordinates
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))

        return image.crop((left, top, right, bottom))

    @_gpu_decorator(duration=120)
    def _chat(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Run OCR inference with PaddleOCR-VL model.

        Args:
            image: Input image
            prompt: Text prompt (task prompt like "OCR:")
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        try:
            # Build messages in PaddleOCR-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Decode
            outputs = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Clean response: Extract only Assistant's answer from chat template
            # Format is typically: "User: OCR:\nAssistant: <actual_text>"
            if "Assistant:" in outputs:
                # Extract everything after "Assistant:"
                outputs = outputs.split("Assistant:", 1)[1].strip()

            return outputs.strip()

        except Exception as e:
            logger.error(f"PaddleOCR-VL inference failed: {e}", exc_info=True)
            return ""

    @_gpu_decorator(duration=120)
    def ocr_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
    ) -> str:
        """
        Extract OCR text from a specific region.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging

        Returns:
            Extracted text, or empty string if no text found
        """
        start_time = time.time()
        try:
            # Log original image before crop
            if self.image_logger:
                self.image_logger.log_image(
                    image=image,
                    stage="ocr",
                    step_index=step_index,
                    bbox_index=bbox_index,
                    image_type="original",
                    bbox=bbox,
                    metadata={"before_crop": True, "task": self.task},
                )

            # Crop region
            original_size = image.size
            cropped = self._crop_region(image, bbox)
            cropped_size = cropped.size
            crop_ratio = (cropped_size[0] * cropped_size[1]) / (
                original_size[0] * original_size[1]
            )

            logger.info(
                f"[PaddleOCR {self.task}] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(
                f"[PaddleOCR {self.task}] Crop ratio: {crop_ratio:.2%} of original"
            )

            # Limit resolution for cropped regions (max 1024x1024)
            max_dimension = 1024
            if max(cropped_size) > max_dimension:
                # Maintain aspect ratio while limiting size
                if cropped_size[0] > cropped_size[1]:
                    new_width = max_dimension
                    new_height = int(cropped_size[1] * max_dimension / cropped_size[0])
                else:
                    new_height = max_dimension
                    new_width = int(cropped_size[0] * max_dimension / cropped_size[1])
                cropped = cropped.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                logger.debug(
                    f"[PaddleOCR {self.task}] Resized cropped image from {cropped_size} to {cropped.size} to limit resolution"
                )

            # Log cropped image
            if self.image_logger:
                self.image_logger.log_cropped_image(
                    original=image,
                    cropped=cropped,
                    bbox=bbox,
                    stage="ocr",
                    step_index=step_index,
                    bbox_index=bbox_index,
                    metadata={
                        "crop_ratio": crop_ratio,
                        "original_size": original_size,
                        "cropped_size": cropped.size,
                        "task": self.task,
                    },
                )

            # Run OCR with task-specific prompt
            prompt = PROMPTS[self.task]
            raw_output = self._chat(cropped, prompt, max_new_tokens=1024)
            ocr_text = raw_output.strip() if raw_output else ""

            # Log OCR result to console
            logger.info(
                f'[PaddleOCR {self.task}] Step {step_index or 0}, BBox {bbox_index or 0}: OCR Result = "{ocr_text}"'
            )

            duration_ms = (time.time() - start_time) * 1000.0

            # Log output
            if self.output_tracer:
                self.output_tracer.trace_captioning(
                    raw_output=raw_output,
                    description=ocr_text,
                    model_name=self.config.model_id,
                    step_index=step_index or 0,
                    bbox_index=bbox_index or 0,
                    bbox=bbox,
                    model_config={
                        "model_id": self.config.model_id,
                        "torch_dtype": str(self.config.torch_dtype),
                        "device": self.config.device,
                        "task": self.task,
                    },
                    intermediate_states=[
                        {
                            "stage": "crop",
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                            "crop_ratio": crop_ratio,
                        },
                        {
                            "stage": "generation",
                            "max_new_tokens": 1024,
                            "task": self.task,
                        },
                    ],
                    duration_ms=duration_ms,
                    metadata={"task": self.task},
                )

            logger.debug(
                f"[PaddleOCR {self.task}] OCR completed in {duration_ms:.1f}ms"
            )
            return ocr_text

        except Exception as e:
            logger.error(f"PaddleOCR-VL OCR failed: {e}", exc_info=True)
            return ""


def _load_paddleocr_backend_in_gpu_context(config: ModelConfig, task: str = "ocr"):
    """
    Load PaddleOCR-VL backend in GPU context (for HF Spaces).

    This wrapper ensures model loading happens in the correct GPU context.
    """
    if spaces is not None:
        # On HF Spaces, use GPU decorator
        @spaces.GPU(duration=300)
        def load():
            return _load_paddleocr_backend(config, task)

        return load()
    else:
        # Local execution
        return _load_paddleocr_backend(config, task)


__all__ = ["PaddleOCRClient"]
