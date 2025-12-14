"""
Florence-2 Captioning Client.

Specialized client for image region captioning using Florence-2.
This component focuses on generating textual descriptions for specific image regions.

IMPORTANT: Florence-2 REQUIRES SDPA (Scaled Dot Product Attention) attention implementation.
This is enforced at model loading time - if SDPA is not available, loading will fail.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Florence2ForConditionalGeneration
    import transformers

    TRANSFORMERS_VERSION = transformers.__version__
    HAS_FLORENCE2 = True
except ImportError:
    HAS_FLORENCE2 = False
    TRANSFORMERS_VERSION = None

from ...core.config import ModelConfig

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


_FLORENCE_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}
_FLORENCE_PROCESSOR_CACHE: dict[str, AutoProcessor] = {}


@_gpu_decorator(duration=120)
def _load_florence_backend_in_gpu_context(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence-2 model in GPU context (for Hugging Face Spaces compatibility)."""
    return _load_florence_backend(config)


def _load_florence_backend(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence-2 model with optimizations."""
    model_id = config.model_id

    if model_id not in _FLORENCE_MODEL_CACHE:
        logger.info(f"Loading Florence-2 captioning model: {model_id}")

        # Determine dtype - Florence-2 requires bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Florence-2 Captioning: Using bfloat16")
        else:
            # Fallback to bfloat16 anyway - Florence-2 is designed for it
            torch_dtype = torch.bfloat16
            logger.warning(
                "Florence-2 Captioning: bf16 not supported, but using bfloat16 anyway (required by Florence-2)"
            )

        # Override if specified in config
        if config.torch_dtype and config.torch_dtype != "auto":
            torch_dtype = getattr(torch, config.torch_dtype, torch_dtype)

        # Get device from config (required, no fallback)
        device_map = config.device
        if not device_map:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")

        # Handle CUDA_VISIBLE_DEVICES: if set, PyTorch maps visible GPUs to cuda:0, cuda:1, etc.
        import os

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices and device_map.startswith("cuda:"):
            # When CUDA_VISIBLE_DEVICES is set, only cuda:0 is available (the first visible GPU)
            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                requested_device_id = int(device_map.split(":")[1])
                if requested_device_id != 0:
                    logger.warning(
                        f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} is set. "
                        f"Config specifies '{device_map}', but only 'cuda:0' is visible to PyTorch. "
                        f"Automatically mapping to 'cuda:0'."
                    )
                    device_map = "cuda:0"

        # Validate device exists
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
                    # Test device access
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

        # Choose model class
        # IMPORTANT: Must use AutoModelForCausalLM with trust_remote_code=True to get the
        # correct model implementation that matches the processor from remote code.
        # Using native Florence2ForConditionalGeneration causes "image tokens do not match"
        # errors because the native model expects image tokens but the remote processor
        # doesn't add them.
        ModelClass = AutoModelForCausalLM
        logger.info(f"Using AutoModelForCausalLM (remote code) for {model_id}")

        # Clear CUDA cache before loading to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.cuda.is_initialized():
                free_memory = torch.cuda.get_device_properties(
                    0
                ).total_memory - torch.cuda.memory_allocated(0)
                logger.info(
                    f"GPU memory before load: {free_memory / 1024**3:.2f} GB free"
                )

        # Load model following official Florence-2 documentation
        # https://huggingface.co/florence-community/Florence-2-base-ft
        # Use device_map directly in from_pretrained (NOT .to() after)
        # Note: attn_implementation='eager' is used to avoid SDPA compatibility issues
        # with newer transformers versions (4.57+)
        try:
            model = ModelClass.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,  # Direct device mapping in from_pretrained
                trust_remote_code=True,
                attn_implementation='eager',  # Avoid SDPA issues with remote code
            ).eval()
            logger.info(f"Florence-2 Captioning loaded on {device_map}")
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a CUDA OOM error
            if (
                "out of memory" in error_msg.lower()
                or "cuda" in error_msg.lower()
                and "memory" in error_msg.lower()
            ):
                # Get GPU memory info for better error message
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.error(
                        f"CUDA Out of Memory when loading Florence-2 Captioning model.\n"
                        f"GPU Memory Status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total.\n"
                        f"Please free up GPU memory or use a different GPU device."
                    )
                    raise RuntimeError(
                        f"CUDA Out of Memory when loading Florence-2 Captioning model. "
                        f"GPU has {total:.2f} GB total, {allocated:.2f} GB allocated. "
                        f"Please free up GPU memory or use a different GPU device. "
                        f"Original error: {error_msg}"
                    ) from e
            # Check if it's an SDPA-specific error
            if "sdpa" in error_msg.lower() or "attention" in error_msg.lower():
                logger.error(
                    f"Florence-2 Captioning SDPA not available ({e}), this is required for Florence-2!"
                )
                raise RuntimeError(
                    f"Florence-2 requires SDPA attention, but it's not available: {e}"
                ) from e
            # Re-raise other RuntimeErrors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to load Florence-2 Captioning model: {e}")
            raise

        # Try to enable Torch Compile
        if config.enable_compile:
            import os

            if os.environ.get("CORGI_DISABLE_COMPILE", "0") != "1":
                try:
                    logger.info("Compiling Florence-2 Captioning model...")
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("✓ Florence-2 Captioning torch compile enabled")
                except Exception as e:
                    logger.warning(f"Florence-2 Captioning compile failed ({e})")

        # Load processor (official Florence-2 method)
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("✓ Florence-2 Processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 Processor: {e}")
            raise

        _FLORENCE_MODEL_CACHE[model_id] = model
        _FLORENCE_PROCESSOR_CACHE[model_id] = processor
        logger.info(f"✓ Florence-2 Captioning model loaded successfully")

    return _FLORENCE_MODEL_CACHE[model_id], _FLORENCE_PROCESSOR_CACHE[model_id]


class Florence2CaptioningClient:
    """
    Florence-2 client specialized for region captioning.

    This client generates textual descriptions for specific image regions
    using Florence-2's DETAILED_CAPTION task.
    """

    def __init__(self, config: ModelConfig, image_logger=None, output_tracer=None):
        """
        Initialize Florence-2 captioning client.

        Model is loaded in GPU context immediately for Hugging Face Spaces compatibility.

        Args:
            config: Model configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        self.config = config
        # Load model in GPU context immediately (required for HF Spaces)
        self._model, self._processor = _load_florence_backend_in_gpu_context(config)
        self.image_logger = image_logger
        self.output_tracer = output_tracer

    def _run_task(
        self,
        task_prompt: str,
        image: Image.Image,
        text_input: Optional[str] = None,
    ) -> dict:
        """Run Florence-2 task and return parsed results."""
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.debug(
            f"[Florence-2 Captioning] Image: mode={image.mode}, size={image.size}"
        )

        # Build prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        logger.debug(f"[Florence-2 Captioning] Prompt: {prompt[:100]}")

        # Prepare inputs
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        logger.debug(
            f"[Florence-2 Captioning] Processor output keys: {list(inputs.keys())}"
        )
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            logger.debug(
                f"[Florence-2 Captioning] pixel_values: shape={pv.shape if pv is not None else None}, dtype={pv.dtype if pv is not None else None}"
            )

        # Move to device from config and ensure correct dtype
        device = torch.device(self.config.device)
        inputs = {
            k: (
                v.to(device=device, dtype=torch.bfloat16)
                if k == "pixel_values" and isinstance(v, torch.Tensor)
                else v.to(device) if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in inputs.items()
        }

        # Generate with autocast for bfloat16
        # Note: Using greedy decoding (num_beams=1) to avoid compatibility issues
        # with beam search in remote code + transformers 4.57+
        device_type = "cuda" if "cuda" in self.config.device else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1,  # Greedy decoding - avoid beam search issues
                do_sample=False,  # Greedy
            )

        # Decode
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process
        parsed_answer = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        return parsed_answer, generated_text

    @_gpu_decorator(duration=120)
    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
    ) -> str:
        """
        Generate caption for a specific region.

        Crops the region and uses <MORE_DETAILED_CAPTION> task.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging

        Returns:
            Textual description of the region
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
            crop_ratio = (cropped_size[0] * cropped_size[1]) / (
                original_size[0] * original_size[1]
            )

            logger.info(
                f"[Florence-2 Captioning] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(
                f"[Florence-2 Captioning] Crop ratio: {crop_ratio:.2%} of original"
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
                        "cropped_size": cropped_size,
                    },
                )

            # Caption the cropped region with MORE_DETAILED_CAPTION for richer descriptions
            task = "<MORE_DETAILED_CAPTION>"
            results, raw_output = self._run_task(task, cropped)

            duration_ms = (time.time() - start_time) * 1000.0

            # Log raw and parsed outputs
            if self.output_tracer:
                caption = (
                    results.get(task, "Unable to caption region")
                    if isinstance(results, dict)
                    else "Unable to caption region"
                )
                self.output_tracer.trace_captioning(
                    raw_output=raw_output,
                    description=caption if isinstance(caption, str) else str(caption),
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
                        {"stage": "generation", "task": task, "max_new_tokens": 1024},
                    ],
                    duration_ms=duration_ms,
                    metadata={"task_prompt": task},
                )

            if isinstance(results, dict) and task in results:
                caption = results[task]
                logger.debug(f"Florence-2 Captioning: {caption[:80]}")
                return caption
            else:
                logger.warning(
                    f"No caption in Florence-2 Captioning response: {results}"
                )
                return "Unable to caption region"
        except Exception as e:
            logger.error(f"Florence-2 Captioning failed: {e}", exc_info=True)
            return "Caption generation failed"

    @_gpu_decorator(duration=120)
    def caption_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> List[str]:
        """
        Generate captions for multiple regions in batch.

        Processes multiple cropped regions efficiently.

        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging

        Returns:
            List of captions corresponding to each bbox
        """
        if not bboxes:
            return []

        start_time = time.time()
        logger.info(f"[Florence-2 Captioning] Batch captioning {len(bboxes)} regions")

        captions = []
        task = "<MORE_DETAILED_CAPTION>"

        # Process all regions
        for bbox_idx, bbox in enumerate(bboxes):
            try:
                # Log original image before crop
                if self.image_logger:
                    self.image_logger.log_image(
                        image=image,
                        stage="captioning",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        image_type="original",
                        bbox=bbox,
                        metadata={"before_crop": True, "batch_index": bbox_idx},
                    )

                # Crop region
                original_size = image.size
                cropped = self._crop_region(image, bbox)
                cropped_size = cropped.size

                # Log cropped image
                if self.image_logger:
                    self.image_logger.log_cropped_image(
                        original=image,
                        cropped=cropped,
                        bbox=bbox,
                        stage="captioning",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        metadata={
                            "batch_index": bbox_idx,
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                        },
                    )

                # Caption the cropped region
                results, raw_output = self._run_task(task, cropped)

                # Extract caption
                if isinstance(results, dict) and task in results:
                    caption = results[task]
                    captions.append(caption)

                    # Log output
                    if self.output_tracer:
                        self.output_tracer.trace_captioning(
                            raw_output=raw_output,
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
                                {"stage": "generation", "task": task},
                            ],
                            duration_ms=(time.time() - start_time)
                            * 1000.0
                            / len(bboxes),
                            metadata={"task_prompt": task, "batch_index": bbox_idx},
                        )
                else:
                    logger.warning(
                        f"No caption in Florence-2 response for bbox {bbox_idx}: {results}"
                    )
                    captions.append("Unable to caption region")

            except Exception as e:
                logger.error(
                    f"Florence-2 Captioning failed for bbox {bbox_idx}: {e}",
                    exc_info=True,
                )
                captions.append("Caption generation failed")

        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(
            f"[Florence-2 Captioning] Batch captioning completed in {duration_ms:.2f}ms ({duration_ms/len(bboxes):.2f}ms per region)"
        )

        return captions

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

        Crops the region and uses <OCR> task.
        Returns concatenated text from OCR results.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging
            bbox_index: Optional bbox index for logging

        Returns:
            Extracted text from OCR, or empty string if no text found
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
                f"[Florence-2 OCR] Cropping image: original={original_size}, bbox={bbox}, cropped={cropped_size}"
            )
            logger.debug(f"[Florence-2 OCR] Crop ratio: {crop_ratio:.2%} of original")

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
                        "cropped_size": cropped_size,
                    },
                )

            # Run OCR task on cropped region
            task = "<OCR>"
            results, raw_output = self._run_task(task, cropped)

            duration_ms = (time.time() - start_time) * 1000.0

            # Parse OCR result - <OCR> returns plain text string, not dict
            ocr_text = ""
            if isinstance(results, dict) and task in results:
                ocr_text = results[task]
                if not isinstance(ocr_text, str):
                    ocr_text = str(ocr_text)
            elif isinstance(results, str):
                ocr_text = results
            else:
                logger.warning(f"Unexpected OCR result format: {results}")
                ocr_text = ""

            # Log raw and parsed outputs
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
                        {"stage": "generation", "task": task, "max_new_tokens": 1024},
                    ],
                    duration_ms=duration_ms,
                    metadata={"task_prompt": task, "ocr_mode": True},
                )

            logger.debug(f"Florence-2 OCR: {ocr_text[:80] if ocr_text else '(empty)'}")
            return ocr_text.strip() if ocr_text else ""
        except Exception as e:
            logger.error(f"Florence-2 OCR failed: {e}", exc_info=True)
            return ""

    @_gpu_decorator(duration=120)
    def ocr_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> List[str]:
        """
        Extract OCR text from multiple regions in batch.

        Processes multiple cropped regions efficiently.

        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging

        Returns:
            List of OCR text corresponding to each bbox
        """
        if not bboxes:
            return []

        start_time = time.time()
        logger.info(f"[Florence-2 OCR] Batch OCR {len(bboxes)} regions")

        ocr_texts = []
        task = "<OCR>"

        # Process all regions
        for bbox_idx, bbox in enumerate(bboxes):
            try:
                # Log original image before crop
                if self.image_logger:
                    self.image_logger.log_image(
                        image=image,
                        stage="ocr",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        image_type="original",
                        bbox=bbox,
                        metadata={"before_crop": True, "batch_index": bbox_idx},
                    )

                # Crop region
                original_size = image.size
                cropped = self._crop_region(image, bbox)
                cropped_size = cropped.size

                # Log cropped image
                if self.image_logger:
                    self.image_logger.log_cropped_image(
                        original=image,
                        cropped=cropped,
                        bbox=bbox,
                        stage="ocr",
                        step_index=step_index,
                        bbox_index=bbox_idx,
                        metadata={
                            "batch_index": bbox_idx,
                            "original_size": original_size,
                            "cropped_size": cropped_size,
                        },
                    )

                # Run OCR task
                results, raw_output = self._run_task(task, cropped)

                # Parse OCR result
                ocr_text = ""
                if isinstance(results, dict) and task in results:
                    ocr_text = results[task]
                    if not isinstance(ocr_text, str):
                        ocr_text = str(ocr_text)
                elif isinstance(results, str):
                    ocr_text = results
                else:
                    logger.warning(f"Unexpected OCR result format: {results}")
                    ocr_text = ""

                ocr_texts.append(ocr_text.strip() if ocr_text else "")

                # Log output
                if self.output_tracer:
                    self.output_tracer.trace_captioning(
                        raw_output=raw_output,
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
                            {
                                "stage": "generation",
                                "task": task,
                                "max_new_tokens": 1024,
                            },
                        ],
                        duration_ms=(time.time() - start_time) * 1000.0,
                        metadata={
                            "task_prompt": task,
                            "ocr_mode": True,
                            "batch_index": bbox_idx,
                        },
                    )

            except Exception as e:
                logger.error(f"Florence-2 OCR failed for bbox {bbox_idx}: {e}")
                ocr_texts.append("")

        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(f"[Florence-2 OCR] Batch OCR completed in {duration_ms:.1f}ms")
        return ocr_texts

    @_gpu_decorator(duration=120)
    def ocr_and_caption_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract OCR text and generate captions for multiple regions in batch.

        Processes both OCR and Captioning tasks in parallel for each region.
        More efficient than separate batch calls since each region is cropped once.

        Args:
            image: Input image
            bboxes: List of bounding boxes (x1, y1, x2, y2) in normalized coordinates [0, 1]
            step_index: Optional step index for logging

        Returns:
            Tuple of (ocr_texts: List[str], captions: List[str])
        """
        if not bboxes:
            return [], []

        start_time = time.time()
        logger.info(
            f"[Florence-2 Unified Batch] Processing {len(bboxes)} regions (OCR + Captioning)"
        )

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
                        },
                    )

                # Run both tasks in parallel using ThreadPoolExecutor
                ocr_raw = ""
                caption_raw = ""

                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit OCR task
                    ocr_future = executor.submit(self._run_task, "<OCR>", cropped)
                    # Submit Captioning task
                    caption_future = executor.submit(
                        self._run_task, "<MORE_DETAILED_CAPTION>", cropped
                    )

                    # Wait for OCR result
                    try:
                        ocr_results, ocr_raw = ocr_future.result()
                        # Parse OCR result
                        if isinstance(ocr_results, dict) and "<OCR>" in ocr_results:
                            ocr_text = ocr_results["<OCR>"]
                            if not isinstance(ocr_text, str):
                                ocr_text = str(ocr_text)
                        elif isinstance(ocr_results, str):
                            ocr_text = ocr_results
                        else:
                            logger.warning(
                                f"Unexpected OCR result format for bbox {bbox_idx}: {ocr_results}"
                            )
                            ocr_text = ""
                        ocr_text = ocr_text.strip() if ocr_text else ""
                    except Exception as e:
                        logger.warning(f"OCR failed for region {bbox_idx}: {e}")
                        ocr_text = ""

                    # Wait for Captioning result
                    try:
                        caption_results, caption_raw = caption_future.result()
                        # Parse Captioning result
                        if (
                            isinstance(caption_results, dict)
                            and "<MORE_DETAILED_CAPTION>" in caption_results
                        ):
                            caption = caption_results["<MORE_DETAILED_CAPTION>"]
                            if not isinstance(caption, str):
                                caption = str(caption)
                        else:
                            logger.warning(
                                f"No caption in Florence-2 response for bbox {bbox_idx}: {caption_results}"
                            )
                            caption = "Unable to caption region"
                    except Exception as e:
                        logger.warning(f"Captioning failed for region {bbox_idx}: {e}")
                        caption = "Caption generation failed"

                # Log outputs
                if self.output_tracer:
                    # Log OCR
                    self.output_tracer.trace_captioning(
                        raw_output=ocr_raw,
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
                            {
                                "stage": "generation",
                                "task": "<OCR>",
                                "max_new_tokens": 1024,
                            },
                        ],
                        duration_ms=(time.time() - start_time) * 1000.0 / len(bboxes),
                        metadata={
                            "task_prompt": "<OCR>",
                            "batch_index": bbox_idx,
                            "unified_batch": True,
                        },
                    )
                    # Log Captioning
                    self.output_tracer.trace_captioning(
                        raw_output=caption_raw,
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
                            {
                                "stage": "generation",
                                "task": "<MORE_DETAILED_CAPTION>",
                                "max_new_tokens": 1024,
                            },
                        ],
                        duration_ms=(time.time() - start_time) * 1000.0 / len(bboxes),
                        metadata={
                            "task_prompt": "<MORE_DETAILED_CAPTION>",
                            "batch_index": bbox_idx,
                            "unified_batch": True,
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
            f"[Florence-2 Unified Batch] Completed in {duration_ms:.2f}ms ({duration_ms/len(bboxes):.2f}ms per region)"
        )

        return ocr_texts, captions

    # Minimum crop size for Florence-2 (to avoid image feature mismatch errors)
    MIN_CROP_SIZE = 128

    @staticmethod
    def _crop_region(
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        min_size: int = 128,
    ) -> Image.Image:
        """
        Crop image to bounding box region with 25% extension on each side.
        
        Also ensures the crop is at least min_size x min_size pixels,
        resizing if necessary to avoid Florence-2 image feature errors.
        """
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

        cropped = image.crop((left, top, right, bottom))
        
        # Ensure minimum size to avoid Florence-2 image feature errors
        crop_w, crop_h = cropped.size
        if crop_w < min_size or crop_h < min_size:
            # Scale up maintaining aspect ratio, ensuring both dims >= min_size
            scale = max(min_size / crop_w, min_size / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.debug(
                f"Resized small crop from ({crop_w}, {crop_h}) to ({new_w}, {new_h})"
            )
        
        return cropped


__all__ = ["Florence2CaptioningClient"]
