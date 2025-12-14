"""
SmolVLM2 Client.

Specialized client for captioning and VQA using HuggingFaceTB/SmolVLM2-500M-Video-Instruct model.
This is a smaller, faster alternative to FastVLM with better instruction following.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import time
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from ...core.config import ModelConfig
from ..registry import ModelRegistry
from ...utils.prompts import FASTVLM_CAPTIONING_PROMPT

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


_MODEL_CACHE: dict[str, AutoModelForImageTextToText] = {}
_PROCESSOR_CACHE: dict[str, AutoProcessor] = {}


def _load_smolvlm2_backend(
    config: ModelConfig,
) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load SmolVLM2 model and processor.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, processor)
    """
    model_id = config.model_id

    if model_id not in _MODEL_CACHE:
        import time

        load_start = time.time()
        logger.info(f"Loading SmolVLM2 model: {model_id}...")

        # Determine dtype
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            logger.info("Using bfloat16 for SmolVLM2")
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
            logger.info(f"✓ SmolVLM2 processor loaded")
        except Exception as e:
            logger.error(f"Failed to load SmolVLM2 processor: {e}", exc_info=True)
            raise

        # Load model
        device_map = {"": target_device} if target_device != "cpu" else None

        try:
            # Try with flash attention 2 first
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
            ).eval()
            logger.info("✓ SmolVLM2 loaded with Flash Attention 2")
        except Exception as e:
            logger.warning(
                f"Flash Attention 2 not available ({e}), using default attention"
            )
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                ).eval()
                logger.info("✓ SmolVLM2 loaded with default attention")
            except Exception as e2:
                logger.error(f"Failed to load SmolVLM2 model: {e2}", exc_info=True)
                raise

        # Enable torch.compile if requested
        if config.enable_compile:
            try:
                logger.info("Compiling SmolVLM2 model with torch.compile...")
                compile_start = time.time()
                model = torch.compile(model, mode="reduce-overhead")
                compile_duration = time.time() - compile_start
                logger.info(f"✓ SmolVLM2 compiled in {compile_duration:.2f}s")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        load_duration = time.time() - load_start
        logger.info(
            f"✓ SmolVLM2 model fully loaded in {load_duration:.2f}s total on {target_device}"
        )

        _MODEL_CACHE[model_id] = model
        _PROCESSOR_CACHE[model_id] = processor

    return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]


@ModelRegistry.register_captioning("smolvlm2")
class SmolVLM2CaptioningClient:
    """
    Client for SmolVLM2-500M model for captioning and VQA.

    Smaller and faster alternative to FastVLM with better instruction following.
    """

    def __init__(
        self,
        config: ModelConfig,
        image_logger=None,
        output_tracer=None,
    ) -> None:
        """
        Initialize SmolVLM2 client.

        Args:
            config: Model configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        self.config = config
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Load model
        self._model, self._processor = _load_smolvlm2_backend_in_gpu_context(config)

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
    def _chat(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Run captioning/VQA inference with SmolVLM2 model.

        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        try:
            # Build messages in SmolVLM2 format (simplified compared to FastVLM!)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image},  # Can pass PIL image directly
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template - processor handles everything!
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device, dtype=self._model.dtype)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for consistent results
                )

            # Decode
            response = self._processor.batch_decode(outputs, skip_special_tokens=True)[
                0
            ]

            # Clean response: Extract only the actual answer from chat template
            # Format may be: "User: <prompt>\nAssistant: <actual_answer>"
            if "Assistant:" in response:
                response = response.split("Assistant:", 1)[1].strip()

            return response.strip()

        except Exception as e:
            logger.error(f"SmolVLM2 inference failed: {e}", exc_info=True)
            return ""

    @_gpu_decorator(duration=120)
    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> str:
        """
        Generate caption for a specific region with VQA (statement verification).

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging
            statement: Optional statement to verify (for VQA mode)

        Returns:
            Generated caption or VQA response
        """
        start_time = time.time()
        try:
            # Log original image before crop
            if self.image_logger:
                self.image_logger.log_image(
                    image=image,
                    stage="captioning",
                    step_index=step_index,
                    bbox_index=bbox_index,
                    image_type="original",
                    bbox=bbox,
                    metadata={"before_crop": True},
                )

            # Crop region
            original_size = image.size
            cropped = self._crop_region(image, bbox)
            cropped_size = cropped.size

            logger.info(
                f"[SmolVLM2] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )

            # Log cropped image
            if self.image_logger:
                self.image_logger.log_cropped_image(
                    original=image,
                    cropped=cropped,
                    bbox=bbox,
                    stage="captioning",
                    step_index=step_index,
                    bbox_index=bbox_index,
                    metadata={
                        "original_size": original_size,
                        "cropped_size": cropped_size,
                    },
                )

            # Build prompt based on mode
            if statement:
                # VQA mode - verify statement with detailed explanation
                prompt = f"""Examine this image region carefully and verify the following statement:
"{statement}"

Provide a detailed analysis:
1. Describe what you actually see in detail
2. Compare it with the statement
3. Conclude whether the statement is correct, incorrect, or partially correct
4. Explain your reasoning"""
                max_tokens = 256  # More tokens for detailed VQA
            else:
                # Captioning mode - super detailed
                prompt = FASTVLM_CAPTIONING_PROMPT
                max_tokens = 512  # More tokens for detailed captioning

            # Run inference with appropriate token limit
            raw_output = self._chat(cropped, prompt, max_new_tokens=max_tokens)
            caption = raw_output.strip() if raw_output else ""

            # Log result
            logger.info(
                f'[SmolVLM2] Step {step_index or 0}, BBox {bbox_index or 0}: Caption = "{caption}"'
            )

            duration_ms = (time.time() - start_time) * 1000.0

            # Log output
            if self.output_tracer:
                self.output_tracer.trace_captioning(
                    raw_output=raw_output,
                    description=caption,
                    model_name=self.config.model_id,
                    step_index=step_index or 0,
                    bbox_index=bbox_index or 0,
                    bbox=bbox,
                    model_config={
                        "model_id": self.config.model_id,
                        "torch_dtype": str(self.config.torch_dtype),
                        "device": self.config.device,
                    },
                    intermediate_states=[
                        {
                            "stage": "crop",
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                        },
                        {
                            "stage": "generation",
                            "max_new_tokens": 128,
                            "vqa_mode": statement is not None,
                        },
                    ],
                    duration_ms=duration_ms,
                    metadata={"statement": statement} if statement else {},
                )

            logger.debug(f"[SmolVLM2] Captioning completed in {duration_ms:.1f}ms")
            return caption

        except Exception as e:
            logger.error(f"SmolVLM2 captioning failed: {e}", exc_info=True)
            return ""


def _load_smolvlm2_backend_in_gpu_context(config: ModelConfig):
    """
    Load SmolVLM2 backend in GPU context (for HF Spaces).

    This wrapper ensures model loading happens in the correct GPU context.
    """
    if spaces is None:
        return _load_smolvlm2_backend(config)

    @spaces.GPU(duration=120)
    def load():
        return _load_smolvlm2_backend(config)

    return load()


__all__ = ["SmolVLM2CaptioningClient"]
