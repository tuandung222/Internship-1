"""
Vintern-1B-v3.5 Captioning Client.

DEPRECATED: This client is deprecated in favor of PaddleOCR-VL (OCR) and FastVLM-0.5B (Captioning+VQA).
Kept for backward compatibility only.

Specialized client for image region OCR and captioning using Vintern-1B-v3.5.
This component focuses on OCR (markdown format) and captioning with VQA (statement verification).
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from ...core.config import ModelConfig
from ...utils.prompts import VINTERN_OCR_PROMPT, VINTERN_CAPTIONING_VQA_PROMPT

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


# Image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    """Build image transform for Vintern model."""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
):
    """Dynamically preprocess image for Vintern model."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_for_vintern(
    image: Image.Image, input_size: int = 448, max_num: int = 6
) -> torch.Tensor:
    """Load and preprocess image for Vintern model."""
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


_VINTERN_MODEL_CACHE: dict[str, AutoModel] = {}
_VINTERN_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


@_gpu_decorator(duration=120)
def _load_vintern_backend_in_gpu_context(
    config: ModelConfig,
) -> Tuple[AutoModel, AutoTokenizer]:
    """Load Vintern model in GPU context (for Hugging Face Spaces compatibility)."""
    return _load_vintern_backend(config)


def _load_vintern_backend(config: ModelConfig) -> Tuple[AutoModel, AutoTokenizer]:
    """Load Vintern-1B-v3.5 model with optimizations."""
    model_id = config.model_id

    if model_id not in _VINTERN_MODEL_CACHE:
        logger.info(f"Loading Vintern-1B-v3.5 model: {model_id}")

        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Vintern: Using bfloat16")
        else:
            torch_dtype = torch.bfloat16
            logger.warning("Vintern: bf16 not supported, but using bfloat16 anyway")

        # Override if specified in config
        if config.torch_dtype and config.torch_dtype != "auto":
            torch_dtype = getattr(torch, config.torch_dtype, torch_dtype)

        # Get device from config
        device_map = config.device
        if not device_map:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")

        # Handle CUDA_VISIBLE_DEVICES
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and device_map.startswith("cuda:"):
            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                requested_device_id = int(device_map.split(":")[1])
                if requested_device_id != 0:
                    logger.warning(
                        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} is set. "
                        f"Config specifies '{device_map}', but only 'cuda:0' is visible to PyTorch. "
                        f"Automatically mapping to 'cuda:0'."
                    )
                    device_map = "cuda:0"

        # Validate device
        if device_map.startswith("cuda:"):
            try:
                device_id = int(device_map.split(":")[1])
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    if device_id >= num_gpus:
                        available_devices = ", ".join(
                            [f"cuda:{i}" for i in range(num_gpus)]
                        )
                        raise ValueError(
                            f"Invalid device '{device_map}': Only {num_gpus} GPU(s) available. "
                            f"Available devices: {available_devices if num_gpus > 0 else 'none'}. "
                            f"{'Note: CUDA_VISIBLE_DEVICES is set, so only cuda:0 is available.' if cuda_visible_devices else ''} "
                            f"Please update your config file to use a valid device."
                        )
                    try:
                        torch.cuda.device(device_id)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        raise ValueError(
                            f"Cannot access device '{device_map}': {e}. "
                            f"Please check if the GPU is available and not in use by another process."
                        ) from e
                else:
                    raise ValueError(
                        f"CUDA is not available, but device '{device_map}' was specified. "
                        f"Please use 'cpu' or ensure CUDA is properly installed."
                    )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Could not validate device '{device_map}': {e}")

        # Clear CUDA cache before loading (use requested device if possible)
        if torch.cuda.is_available() and device_map.startswith("cuda:"):
            device_id = int(device_map.split(":")[1])
            try:
                torch.cuda.set_device(device_id)
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                logger.info(
                    f"GPU memory before load on {device_map}: {free_bytes / 1024**3:.2f} GB free "
                    f"({total_bytes / 1024**3:.2f} GB total)"
                )
            except Exception:
                total_mem = torch.cuda.get_device_properties(device_id).total_memory
                allocated = torch.cuda.memory_allocated(device_id)
                free_memory = total_mem - allocated
                logger.info(
                    f"GPU memory before load on {device_map}: {free_memory / 1024**3:.2f} GB free "
                    f"({total_mem / 1024**3:.2f} GB total reported)"
                )

        # Load model following official Vintern demo pattern:
        # Load to CPU with low_cpu_mem_usage=True but NO device_map, then move to GPU
        try:
            if device_map.startswith("cuda:"):
                # Load to CPU first (no device_map), then move to GPU
                # This matches the official Vintern demo pattern
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_flash_attn=False,  # Explicitly disable flash attention
                ).eval()
                # Move to GPU manually
                device_id = int(device_map.split(":")[1])
                model = model.to(f"cuda:{device_id}")
            else:
                # For CPU, load normally
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_flash_attn=False,
                ).eval()
                if device_map != "cpu":
                    model = model.to(device_map)
            logger.info(f"✓ Vintern model loaded on {device_map}")
        except RuntimeError as e:
            error_msg = str(e)
            if (
                "out of memory" in error_msg.lower()
                or "cuda" in error_msg.lower()
                and "memory" in error_msg.lower()
            ):
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.error(
                        f"CUDA Out of Memory when loading Vintern model.\n"
                        f"GPU Memory Status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total.\n"
                        f"Please free up GPU memory or use a different GPU device."
                    )
                    raise RuntimeError(
                        f"CUDA Out of Memory when loading Vintern model. "
                        f"GPU has {total:.2f} GB total, {allocated:.2f} GB allocated. "
                        f"Please free up GPU memory or use a different GPU device. "
                        f"Original error: {error_msg}"
                    ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to load Vintern model: {e}")
            raise

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=False
            )
            logger.info("✓ Vintern tokenizer loaded")
        except Exception as e:
            logger.error(f"Failed to load Vintern tokenizer: {e}")
            raise

        _VINTERN_MODEL_CACHE[model_id] = model
        _VINTERN_TOKENIZER_CACHE[model_id] = tokenizer

    return _VINTERN_MODEL_CACHE[model_id], _VINTERN_TOKENIZER_CACHE[model_id]


# Import at module level to avoid circular imports
from ..registry import ModelRegistry


@ModelRegistry.register_captioning("vintern")
class VinternCaptioningClient:
    """
    Vintern-1B-v3.5 client specialized for region OCR and captioning.

    This client generates OCR text (in markdown format) and captions with VQA
    (statement verification) for specific image regions using Vintern-1B-v3.5.
    """

    def __init__(
        self,
        config: ModelConfig,
        image_logger=None,
        output_tracer=None,
        image_size: int = 448,
        max_num_patches: int = 6,
    ):
        """
        Initialize Vintern captioning client.

        Model is loaded in GPU context immediately for Hugging Face Spaces compatibility.

        Args:
            config: Model configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
            image_size: Image input size for preprocessing (default: 448)
            max_num_patches: Maximum number of image patches (default: 6)
        """
        self.config = config
        self.image_size = image_size
        self.max_num_patches = max_num_patches
        # Load model in GPU context immediately (required for HF Spaces)
        self._model, self._tokenizer = _load_vintern_backend_in_gpu_context(config)
        self.image_logger = image_logger
        self.output_tracer = output_tracer

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
        Run chat inference with Vintern model.

        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Model response text
        """
        try:
            # Preprocess image
            pixel_values = load_image_for_vintern(
                image, input_size=self.image_size, max_num=self.max_num_patches
            )

            # Move to device
            device = self.config.device
            if device.startswith("cuda:"):
                pixel_values = pixel_values.to(torch.bfloat16).to(device)

            # Generation config
            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
                repetition_penalty=2.5,
            )

            # Format prompt with image
            question = "<image>\n" + prompt

            # Run inference
            response, _ = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True,
            )

            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"Vintern chat failed: {e}", exc_info=True)
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
        Extract OCR text from a specific region in markdown format.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging

        Returns:
            Extracted text in markdown format, or empty string if no text found
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
                    metadata={"before_crop": True},
                )

            # Crop region
            original_size = image.size
            cropped = self._crop_region(image, bbox)
            cropped_size = cropped.size
            crop_ratio = (cropped_size[0] * cropped_size[1]) / (
                original_size[0] * original_size[1]
            )

            logger.info(
                f"[Vintern OCR] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(f"[Vintern OCR] Crop ratio: {crop_ratio:.2%} of original")

            # OPTIMIZATION: Resize cropped image to reduce number of patches
            # Limit max dimension to 2x image_size to ensure max_num_patches is respected
            max_dimension = self.image_size * 2
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
                    f"[Vintern OCR] Resized cropped image from {cropped_size} to {cropped.size} to reduce patches"
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
                        "cropped_size": cropped.size,  # Use resized size
                    },
                )

            # Run OCR with markdown prompt
            prompt = VINTERN_OCR_PROMPT
            raw_output = self._chat(cropped, prompt, max_new_tokens=700)
            ocr_text = raw_output.strip() if raw_output else ""

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
                    },
                    intermediate_states=[
                        {
                            "stage": "crop",
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                            "crop_ratio": crop_ratio,
                        },
                        {"stage": "generation", "max_new_tokens": 700},
                    ],
                    duration_ms=duration_ms,
                    metadata={"ocr_mode": True, "markdown_format": True},
                )

            logger.debug(f"Vintern OCR: {ocr_text[:80] if ocr_text else '(empty)'}")
            return ocr_text
        except Exception as e:
            logger.error(f"Vintern OCR failed: {e}", exc_info=True)
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
                f"[Vintern Captioning] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(
                f"[Vintern Captioning] Crop ratio: {crop_ratio:.2%} of original"
            )

            # OPTIMIZATION: Resize cropped image to reduce number of patches
            # Limit max dimension to 2x image_size to ensure max_num_patches is respected
            max_dimension = self.image_size * 2
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
                    f"[Vintern Captioning] Resized cropped image from {cropped_size} to {cropped.size} to reduce patches"
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
                        "cropped_size": cropped.size,  # Use resized size
                        "statement": statement,
                    },
                )

            # Run captioning with VQA prompt
            prompt = VINTERN_CAPTIONING_VQA_PROMPT.format(
                statement=statement or "the region"
            )
            raw_output = self._chat(cropped, prompt, max_new_tokens=700)
            caption = raw_output.strip() if raw_output else "Unable to caption region"

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
                        {"stage": "generation", "max_new_tokens": 700},
                    ],
                    duration_ms=duration_ms,
                    metadata={"statement": statement, "vqa_mode": True},
                )

            logger.debug(f"Vintern Captioning: {caption[:80]}")
            return caption
        except Exception as e:
            logger.error(f"Vintern Captioning failed: {e}", exc_info=True)
            return "Caption generation failed"

    @_gpu_decorator(duration=120)
    def ocr_and_caption_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract OCR text and generate captions for multiple regions in batch.

        Processes both OCR and Captioning tasks in parallel for each region.

        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            statements: Optional list of statements to verify (for VQA component)
            step_index: Optional step index for logging

        Returns:
            Tuple of (ocr_texts: List[str], captions: List[str])
        """
        if not bboxes:
            return [], []

        start_time = time.time()
        logger.info(
            f"[Vintern Unified Batch] Processing {len(bboxes)} regions (OCR + Captioning)"
        )

        # Use the same statement for all regions (from the reasoning step)
        ocr_texts = []
        captions = []

        def process_region(
            bbox_idx: int, bbox: Tuple[float, float, float, float]
        ) -> Tuple[str, str]:
            """Process a single region with both OCR and Captioning."""
            ocr_text = ""
            caption = ""

            try:
                # Log original image before crop
                if self.image_logger:
                    self.image_logger.log_image(
                        image=image,
                        stage="ocr_captioning",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        image_type="original",
                        bbox=bbox,
                        metadata={
                            "before_crop": True,
                            "batch_index": bbox_idx,
                            "unified_batch": True,
                            "statement": statement,
                        },
                    )

                # Crop region once (shared by both tasks)
                original_size = image.size
                cropped = self._crop_region(image, bbox)
                cropped_size = cropped.size

                # Log cropped image
                if self.image_logger:
                    self.image_logger.log_cropped_image(
                        original=image,
                        cropped=cropped,
                        bbox=bbox,
                        stage="ocr_captioning",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        metadata={
                            "batch_index": bbox_idx,
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                            "unified_batch": True,
                            "statement": statement,
                        },
                    )

                # Run both tasks in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit OCR task
                    ocr_future = executor.submit(
                        self._chat, cropped, VINTERN_OCR_PROMPT, 700
                    )
                    # Submit Captioning task (use statement from step if available)
                    caption_prompt = VINTERN_CAPTIONING_VQA_PROMPT.format(
                        statement=statement or "the region"
                    )
                    caption_future = executor.submit(
                        self._chat, cropped, caption_prompt, 700
                    )

                    # Wait for OCR result
                    try:
                        ocr_result = ocr_future.result()
                        ocr_text = ocr_result.strip() if ocr_result else ""
                    except Exception as e:
                        logger.warning(f"OCR failed for region {bbox_idx}: {e}")
                        ocr_text = ""

                    # Wait for Captioning result
                    try:
                        caption_result = caption_future.result()
                        caption = (
                            caption_result.strip()
                            if caption_result
                            else "Unable to caption region"
                        )
                    except Exception as e:
                        logger.warning(f"Captioning failed for region {bbox_idx}: {e}")
                        caption = "Caption generation failed"

                # Log outputs
                if self.output_tracer:
                    # Log OCR
                    self.output_tracer.trace_captioning(
                        raw_output=ocr_text,
                        description=ocr_text,
                        model_name=self.config.model_id,
                        step_index=step_index or 0,
                        bbox_index=bbox_idx,
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
                            {"stage": "generation", "max_new_tokens": 700},
                        ],
                        duration_ms=(time.time() - start_time) * 1000.0 / len(bboxes),
                        metadata={
                            "task": "OCR",
                            "batch_index": bbox_idx,
                            "unified_batch": True,
                            "markdown_format": True,
                        },
                    )
                    # Log Captioning
                    self.output_tracer.trace_captioning(
                        raw_output=caption,
                        description=caption,
                        model_name=self.config.model_id,
                        step_index=step_index or 0,
                        bbox_index=bbox_idx,
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
                            {"stage": "generation", "max_new_tokens": 700},
                        ],
                        duration_ms=(time.time() - start_time) * 1000.0 / len(bboxes),
                        metadata={
                            "task": "Captioning+VQA",
                            "batch_index": bbox_idx,
                            "unified_batch": True,
                            "statement": statement,
                        },
                    )

            except Exception as e:
                logger.error(
                    f"Unified batch processing failed for bbox {bbox_idx}: {e}",
                    exc_info=True,
                )
                ocr_text = ""
                caption = "Processing failed"

            return ocr_text, caption

        # Process all regions
        for bbox_idx, bbox in enumerate(bboxes):
            ocr_text, caption = process_region(bbox_idx, bbox)
            ocr_texts.append(ocr_text)
            captions.append(caption)

        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(
            f"[Vintern Unified Batch] Completed in {duration_ms:.2f}ms ({duration_ms/len(bboxes):.2f}ms per region)"
        )

        return ocr_texts, captions


__all__ = ["VinternCaptioningClient"]
