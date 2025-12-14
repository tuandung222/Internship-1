"""
FastVLM Client.

Specialized client for captioning and VQA using apple/FastVLM-0.5B model.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging
import time
import os

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# Image token index for FastVLM
IMAGE_TOKEN_INDEX = -200


_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}
_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def _load_fastvlm_backend(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load FastVLM model and tokenizer.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    model_id = config.model_id

    if model_id not in _MODEL_CACHE:
        import time

        load_start = time.time()
        logger.info(f"Loading FastVLM model: {model_id}...")

        # Determine dtype
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            logger.info("Using float16 for FastVLM")
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

        # Handle CUDA_VISIBLE_DEVICES remapping when a specific GPU index is hidden
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and target_device.startswith("cuda:"):
            target_device = "cuda:0"
            logger.info(
                f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} detected, mapping to {target_device}"
            )

        # Check for Flash Attention 3 support (auto-enable if available)
        use_flash_attn3 = False
        try:
            import flash_attn

            if hasattr(flash_attn, "__version__"):
                version = flash_attn.__version__
                if version.startswith("3.") or int(version.split(".")[0]) >= 3:
                    use_flash_attn3 = True
                    logger.info(
                        f"Flash Attention 3 detected (version {version}), will attempt to enable"
                    )
        except ImportError:
            logger.debug("Flash Attention not available, using default attention")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"✓ FastVLM tokenizer loaded")
        except Exception as e:
            logger.error(f"Failed to load FastVLM tokenizer: {e}", exc_info=True)
            raise

        # Load model
        # Strategy: Load directly to target device via device_map to avoid meta tensors,
        # disabling low_cpu_mem_usage to prevent lazy/empty parameter materialization.
        device_map = {"": target_device} if target_device != "cpu" else None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            ).eval()
        except Exception as e:
            logger.error(f"Failed to load FastVLM model: {e}", exc_info=True)
            raise

        load_duration = time.time() - load_start
        logger.info(
            f"✓ FastVLM model fully loaded in {load_duration:.2f}s total on {target_device}"
        )

        _MODEL_CACHE[model_id] = model
        _TOKENIZER_CACHE[model_id] = tokenizer

    return _MODEL_CACHE[model_id], _TOKENIZER_CACHE[model_id]


@ModelRegistry.register_captioning("fastvlm")
class FastVLMCaptioningClient:
    """
    Client for FastVLM-0.5B model for captioning and VQA.
    """

    def __init__(
        self,
        config: ModelConfig,
        image_logger=None,
        output_tracer=None,
    ) -> None:
        """
        Initialize FastVLM client.

        Args:
            config: Model configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        self.config = config
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Load model in GPU context immediately (required for HF Spaces)
        self._model, self._tokenizer = _load_fastvlm_backend_in_gpu_context(config)

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
        max_new_tokens: int = 128,
    ) -> str:
        """
        Run captioning/VQA inference with FastVLM model.

        Args:
            image: Input image
            prompt: Text prompt (with <image> placeholder)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        try:
            # Build chat template and render to string (not tokens) so we can place <image> exactly
            messages = [{"role": "user", "content": prompt}]

            rendered = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            # Split around <image> placeholder
            if "<image>" not in rendered:
                # If no <image> token, add it at the beginning
                rendered = "<image>\n" + rendered

            pre, post = rendered.split("<image>", 1)

            # Tokenize the text *around* the image token (no extra specials!)
            pre_ids = self._tokenizer(
                pre, return_tensors="pt", add_special_tokens=False
            ).input_ids
            post_ids = self._tokenizer(
                post, return_tensors="pt", add_special_tokens=False
            ).input_ids

            # Splice in the IMAGE token id (-200) at the placeholder position
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(
                self._model.device
            )
            attention_mask = torch.ones_like(input_ids, device=self._model.device)

            # Preprocess image via the model's own processor
            try:
                # Try to use vision tower's image processor
                if hasattr(self._model, "get_vision_tower"):
                    vision_tower = self._model.get_vision_tower()
                    if hasattr(vision_tower, "image_processor"):
                        px = vision_tower.image_processor(
                            images=image, return_tensors="pt"
                        )["pixel_values"]
                        px = px.to(self._model.device, dtype=self._model.dtype)
                    else:
                        raise AttributeError("Vision tower has no image_processor")
                else:
                    raise AttributeError("Model has no get_vision_tower method")
            except (AttributeError, Exception) as e:
                logger.error(f"Failed to use vision tower image processor: {e}")
                # Fallback: use basic image preprocessing
                from torchvision.transforms import Compose, Resize, ToTensor, Normalize

                transform = Compose(
                    [
                        Resize((224, 224)),  # Default size
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                px = (
                    transform(image)
                    .unsqueeze(0)
                    .to(self._model.device, dtype=self._model.dtype)
                )

            # Generate
            input_len = input_ids.shape[1]  # Store input length for decoding
            with torch.no_grad():
                out = self._model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=max_new_tokens,
                )

            # Decode response (skip input tokens)
            response = self._tokenizer.decode(
                out[0][input_len:], skip_special_tokens=True
            )

            # Clean response: Extract only the actual answer from chat template
            # Format may be: "User: <prompt>\nAssistant: <actual_answer>"
            if "Assistant:" in response:
                response = response.split("Assistant:", 1)[1].strip()

            return response.strip()

        except Exception as e:
            logger.error(f"FastVLM inference failed: {e}", exc_info=True)
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
            statement: Optional statement to verify (for VQA component)
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging

        Returns:
            Textual description with statement verification
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
                    metadata={"before_crop": True, "statement": statement},
                )

            # Crop region
            original_size = image.size
            cropped = self._crop_region(image, bbox)
            cropped_size = cropped.size
            crop_ratio = (cropped_size[0] * cropped_size[1]) / (
                original_size[0] * original_size[1]
            )

            logger.info(
                f"[FastVLM Captioning] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(
                f"[FastVLM Captioning] Crop ratio: {crop_ratio:.2%} of original"
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
                    f"[FastVLM Captioning] Resized cropped image from {cropped_size} to {cropped.size} to limit resolution"
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
                        "crop_ratio": crop_ratio,
                        "original_size": original_size,
                        "cropped_size": cropped.size,
                        "statement": statement,
                    },
                )

            # Build VQA prompt with statement verification using focused prompt
            if statement:
                prompt = FASTVLM_CAPTIONING_PROMPT.format(statement=statement)
            else:
                prompt = "<image>\nDescribe this image region concisely. Focus only on directly visible elements (text, numbers, objects). Keep description to 1-2 sentences."

            # Reduce max_new_tokens to prevent verbose/hallucinated responses
            # 200 tokens should be enough for a concise description
            raw_output = self._chat(cropped, prompt, max_new_tokens=200)
            caption = raw_output.strip() if raw_output else "Unable to caption region"

            # Post-process: Truncate if too long and remove hallucinated content
            # Limit to ~300 characters to keep descriptions concise
            if len(caption) > 300:
                # Try to find a natural break point (sentence end)
                truncated = caption[:300]
                last_period = truncated.rfind(".")
                last_newline = truncated.rfind("\n")
                break_point = max(last_period, last_newline)
                if (
                    break_point > 150
                ):  # Only truncate if we have a reasonable break point
                    caption = caption[: break_point + 1]
                else:
                    caption = caption[:300] + "..."
                logger.debug(
                    f"[FastVLM Captioning] Truncated description from {len(raw_output)} to {len(caption)} chars"
                )

            # Log Captioning result to console
            logger.info(
                f"[FastVLM Captioning] Step {step_index or 0}, BBox {bbox_index or 0}, Statement: \"{statement or 'N/A'}\""
            )
            logger.info(
                f'[FastVLM Captioning] Step {step_index or 0}, BBox {bbox_index or 0}: Caption Result = "{caption}"'
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
                            "crop_ratio": crop_ratio,
                        },
                        {
                            "stage": "generation",
                            "max_new_tokens": 200,
                            "statement": statement,
                        },
                    ],
                    duration_ms=duration_ms,
                    metadata={"statement": statement},
                )

            logger.debug(
                f"[FastVLM Captioning] Captioning completed in {duration_ms:.1f}ms"
            )
            return caption

        except Exception as e:
            logger.error(f"FastVLM captioning failed: {e}", exc_info=True)
            return "Unable to caption region"


def _load_fastvlm_backend_in_gpu_context(config: ModelConfig):
    """
    Load FastVLM backend in GPU context (for HF Spaces).

    This wrapper ensures model loading happens in the correct GPU context.
    """
    if spaces is not None:
        # On HF Spaces, use GPU decorator
        @spaces.GPU(duration=300)
        def load():
            return _load_fastvlm_backend(config)

        return load()
    else:
        # Local execution
        return _load_fastvlm_backend(config)


__all__ = ["FastVLMCaptioningClient"]
