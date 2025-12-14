"""
Florence-2 OCR Client.

Specialized client for OCR using microsoft/Florence-2-base-ft model.
Superior to PaddleOCR for document understanding and text recognition.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import time
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

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


_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}
_PROCESSOR_CACHE: dict[str, AutoProcessor] = {}


def _load_florence2_backend(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """
    Load Florence-2 model and processor.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, processor)
    """
    model_id = config.model_id

    if model_id not in _MODEL_CACHE:
        load_start = time.time()
        logger.info(f"Loading Florence-2 model: {model_id}...")

        # Determine dtype
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            logger.info("Using float16 for Florence-2")
        else:
            torch_dtype = torch.float32
            logger.info("Using float32 (CPU mode)")

        # Override if specified in config
        if config.torch_dtype and config.torch_dtype != "auto":
            torch_dtype = getattr(torch, config.torch_dtype, torch_dtype)

        # Get device from config
        target_device = config.device or "cpu"
        if not target_device:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")

        # Handle CUDA_VISIBLE_DEVICES remapping
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and target_device.startswith("cuda:"):
            target_device = "cuda:0"
            logger.info(
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected, mapping to {target_device}"
            )

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"✓ Florence-2 processor loaded")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 processor: {e}", exc_info=True)
            raise

        # Load model
        device_map = {"": target_device} if target_device != "cpu" else None

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            ).eval()
            logger.info("✓ Florence-2 model loaded")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}", exc_info=True)
            raise

        # Enable torch.compile if requested
        if config.enable_compile:
            try:
                logger.info("Compiling Florence-2 model with torch.compile...")
                compile_start = time.time()
                model = torch.compile(model, mode="reduce-overhead")
                compile_duration = time.time() - compile_start
                logger.info(f"✓ Florence-2 compiled in {compile_duration:.2f}s")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        load_duration = time.time() - load_start
        logger.info(
            f"✓ Florence-2 model fully loaded in {load_duration:.2f}s total on {target_device}"
        )

        _MODEL_CACHE[model_id] = model
        _PROCESSOR_CACHE[model_id] = processor

    return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]


@ModelRegistry.register_captioning("florence2")
class Florence2OCRClient:
    """
    Client for Florence-2 OCR.

    Florence-2 supports various vision tasks via task prompts:
    - <OCR>: Extract all text from image
    - <OCR_WITH_REGION>: OCR with bounding boxes
    - <CAPTION>: Generate image caption
    """

    def __init__(
        self,
        config: ModelConfig,
        image_logger=None,
        output_tracer=None,
    ) -> None:
        """
        Initialize Florence-2 OCR client.

        Args:
            config: Model configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        self.config = config
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Load model
        self._model, self._processor = _load_florence2_backend_in_gpu_context(config)

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
        extend_x = width * 0.125
        extend_y = height * 0.125

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
    def _run_ocr(
        self,
        image: Image.Image,
        task_prompt: str = "<OCR>",
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Run OCR inference with Florence-2 model.

        Args:
            image: Input image
            task_prompt: Florence-2 task prompt (e.g., "<OCR>", "<OCR_WITH_REGION>")
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        try:
            # Prepare inputs
            inputs = self._processor(
                text=task_prompt, images=image, return_tensors="pt"
            ).to(self._model.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )

            # Decode
            generated_text = self._processor.batch_decode(
                outputs, skip_special_tokens=False
            )[0]

            # Parse Florence-2 output
            ocr_text = self._parse_florence2_output(generated_text, task_prompt)

            return ocr_text

        except Exception as e:
            logger.error(f"Florence-2 OCR failed: {e}", exc_info=True)
            return ""

    def _parse_florence2_output(self, raw_output: str, task_prompt: str) -> str:
        """
        Parse Florence-2 output format.

        Florence-2 outputs in format: "<TASK>content</TASK>"
        For <OCR>, output is: "<OCR>extracted text</OCR>"
        """
        # Extract content between task tags
        task_name = task_prompt.strip("<>")
        pattern = f"<{task_name}>(.*?)</{task_name}>"
        match = re.search(pattern, raw_output, re.DOTALL)

        if match:
            content = match.group(1).strip()
            return content
        else:
            # Fallback: return everything after task prompt
            logger.warning(f"Could not parse Florence-2 output format, using fallback")
            if task_prompt in raw_output:
                return raw_output.split(task_prompt, 1)[1].strip()
            return raw_output.strip()

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
                    metadata={"before_crop": True, "model": "florence2"},
                )

            # Crop region
            original_size = image.size
            cropped = self._crop_region(image, bbox)
            cropped_size = cropped.size

            logger.info(
                f"[Florence-2 OCR] Cropping image: original={original_size}, "
                f"bbox={bbox}, cropped={cropped_size}"
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
                        "original_size": original_size,
                        "cropped_size": cropped_size,
                        "model": "florence2",
                    },
                )

            # Run OCR
            raw_output = self._run_ocr(
                cropped, task_prompt="<OCR>", max_new_tokens=1024
            )
            ocr_text = raw_output.strip() if raw_output else ""

            # Log OCR result
            logger.info(
                f"[Florence-2 OCR] Step {step_index or 0}, BBox {bbox_index or 0}: "
                f'OCR Result = "{ocr_text}"'
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
                        "task": "ocr",
                    },
                    intermediate_states=[
                        {
                            "stage": "crop",
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                        },
                        {
                            "stage": "generation",
                            "max_new_tokens": 1024,
                            "task": "<OCR>",
                        },
                    ],
                    duration_ms=duration_ms,
                    metadata={"model": "florence2"},
                )

            logger.debug(f"[Florence-2 OCR] OCR completed in {duration_ms:.1f}ms")
            return ocr_text

        except Exception as e:
            logger.error(f"Florence-2 OCR failed: {e}", exc_info=True)
            return ""


def _load_florence2_backend_in_gpu_context(config: ModelConfig):
    """
    Load Florence-2 backend in GPU context (for HF Spaces).

    This wrapper ensures model loading happens in the correct GPU context.
    """
    if spaces is None:
        return _load_florence2_backend(config)

    @spaces.GPU(duration=120)
    def load():
        return _load_florence2_backend(config)

    return load()


__all__ = ["Florence2OCRClient"]
