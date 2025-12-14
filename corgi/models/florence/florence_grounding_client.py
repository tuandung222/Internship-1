"""
Florence-2 Grounding Client.

Specialized client for visual grounding (phrase-to-region) using Florence-2.
This component focuses on extracting bounding boxes for regions relevant to textual statements.

IMPORTANT: Florence-2 REQUIRES SDPA (Scaled Dot Product Attention) attention implementation.
This is enforced at model loading time - if SDPA is not available, loading will fail.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import logging
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Florence2ForConditionalGeneration

    HAS_FLORENCE2 = True
except ImportError:
    HAS_FLORENCE2 = False

from ...core.config import ModelConfig
from ...utils.coordinate_utils import non_maximum_suppression

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
        import time

        load_start = time.time()
        logger.info(
            f"Loading Florence-2 Grounding model: {model_id} (this may take 10-15 seconds)..."
        )

        # Determine dtype - Florence-2 requires bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Florence-2 Grounding: Using bfloat16")
        else:
            # Fallback to bfloat16 anyway - Florence-2 is designed for it
            torch_dtype = torch.bfloat16
            logger.warning(
                "Florence-2 Grounding: bf16 not supported, but using bfloat16 anyway (required by Florence-2)"
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

        # Choose model class (prefer Florence2ForConditionalGeneration for florence-community models)
        ModelClass = AutoModelForCausalLM
        if HAS_FLORENCE2 and (
            "florence-community" in model_id.lower() or "Florence-2" in model_id
        ):
            ModelClass = Florence2ForConditionalGeneration
            logger.info(f"Using Florence2ForConditionalGeneration for {model_id}")

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
        model_load_start = time.time()
        try:
            model = ModelClass.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,  # Direct device mapping in from_pretrained
                trust_remote_code=True,
            ).eval()
            logger.info(f"Florence-2 Grounding loaded on {device_map}")
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
                        f"CUDA Out of Memory when loading Florence-2 Grounding model.\n"
                        f"GPU Memory Status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total.\n"
                        f"Please free up GPU memory or use a different GPU device."
                    )
                    raise RuntimeError(
                        f"CUDA Out of Memory when loading Florence-2 Grounding model. "
                        f"GPU has {total:.2f} GB total, {allocated:.2f} GB allocated. "
                        f"Please free up GPU memory or use a different GPU device. "
                        f"Original error: {error_msg}"
                    ) from e
            # Check if it's an SDPA-specific error
            if "sdpa" in error_msg.lower() or "attention" in error_msg.lower():
                logger.error(
                    f"Florence-2 Grounding SDPA not available ({e}), this is required for Florence-2!"
                )
                raise RuntimeError(
                    f"Florence-2 requires SDPA attention, but it's not available: {e}"
                ) from e
            # Re-raise other RuntimeErrors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to load Florence-2 Grounding model: {e}")
            raise
        model_load_time = time.time() - model_load_start
        logger.info(f"Model weights loaded in {model_load_time:.2f}s")

        # Try to enable Torch Compile
        if config.enable_compile:
            import os

            if os.environ.get("CORGI_DISABLE_COMPILE", "0") != "1":
                compile_start = time.time()
                try:
                    logger.info(
                        "Compiling Florence-2 Grounding model with torch.compile..."
                    )
                    model = torch.compile(model, mode="reduce-overhead")
                    compile_time = time.time() - compile_start
                    logger.info(
                        f"✓ Florence-2 Grounding torch compile enabled (took {compile_time:.2f}s)"
                    )
                except Exception as e:
                    logger.warning(f"Florence-2 Grounding compile failed ({e})")
        else:
            logger.info(
                "✓ Torch compile DISABLED for Florence-2 Grounding (enable_compile=false in config)"
            )

        processor_start = time.time()
        # Load processor (official Florence-2 method)
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            processor_time = time.time() - processor_start
            logger.info(f"✓ Florence-2 Processor loaded successfully in {processor_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 Processor: {e}")
            raise

        _FLORENCE_MODEL_CACHE[model_id] = model
        _FLORENCE_PROCESSOR_CACHE[model_id] = processor
        total_load_time = time.time() - load_start
        logger.info(
            f"✓ Florence-2 Grounding model fully loaded in {total_load_time:.2f}s total"
        )

    return _FLORENCE_MODEL_CACHE[model_id], _FLORENCE_PROCESSOR_CACHE[model_id]


class Florence2GroundingClient:
    """
    Florence-2 client specialized for phrase grounding.

    This client extracts bounding boxes for regions relevant to textual statements
    using Florence-2's CAPTION_TO_PHRASE_GROUNDING task.
    """

    def __init__(self, config: ModelConfig, image_logger=None, output_tracer=None):
        """
        Initialize Florence-2 grounding client.

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
            f"[Florence-2 Grounding] Image: mode={image.mode}, size={image.size}"
        )

        # Build prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        logger.debug(f"[Florence-2 Grounding] Prompt: {prompt[:100]}")

        # Prepare inputs
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        logger.debug(
            f"[Florence-2 Grounding] Processor output keys: {list(inputs.keys())}"
        )
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            logger.debug(
                f"[Florence-2 Grounding] pixel_values: shape={pv.shape if pv is not None else None}, dtype={pv.dtype if pv is not None else None}"
            )

        # Move to device from config
        device = torch.device(self.config.device)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate with autocast for bfloat16 and nucleus sampling
        device_type = "cuda" if "cuda" in self.config.device else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=True,  # Enable nucleus sampling
                temperature=0.25,  # Low temperature for accuracy
                top_p=0.9,  # Nucleus sampling threshold
                top_k=50,  # Top-k filtering
                use_cache=True,  # Enable KV cache (tested with transformers >= 4.58)
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
    def extract_regions(
        self,
        image: Image.Image,
        statement: str,
        max_regions: int = 3,
        step_index: Optional[int] = None,
        nms_enabled: bool = True,
        nms_iou_threshold: float = 0.5,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extract bounding boxes for regions relevant to the statement.

        Uses <CAPTION_TO_PHRASE_GROUNDING> task for phrase grounding.

        Args:
            image: Input image
            statement: Reasoning statement to ground
            max_regions: Maximum number of regions to return
            step_index: Optional step index for logging

        Returns:
            List of bounding boxes as tuples (x1, y1, x2, y2) in normalized coordinates [0, 1]
        """
        start_time = time.time()
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        try:
            logger.debug(
                f"[Florence-2 Grounding] extract_regions called with statement: {statement[:100]}"
            )
            logger.debug(
                f"[Florence-2 Grounding] Image: {image.size if image else 'None'}"
            )

            # Log input image
            if self.image_logger:
                self.image_logger.log_image(
                    image=image,
                    stage="grounding",
                    step_index=step_index,
                    image_type="input",
                    metadata={"statement": statement, "max_regions": max_regions},
                )

            results, raw_output = self._run_task(task, image, statement)

            logger.debug(
                f"[Florence-2 Grounding] _run_task returned: {type(results)}, keys: {results.keys() if isinstance(results, dict) else 'None'}"
            )

            duration_ms = (time.time() - start_time) * 1000.0

            # Extract bboxes
            normalized_bboxes = []
            intermediate_states = []

            if (
                isinstance(results, dict)
                and task in results
                and "bboxes" in results[task]
            ):
                bboxes = results[task]["bboxes"]
                logger.debug(
                    f"[Florence-2 Grounding] Found {len(bboxes)} bboxes before NMS: {bboxes}"
                )

                # Apply NMS to remove overlapping bboxes (before normalization for efficiency)
                if nms_enabled and len(bboxes) > 1:
                    # Normalize bboxes temporarily for NMS (NMS works on normalized coordinates)
                    w, h = image.size
                    normalized_for_nms = [
                        (x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in bboxes
                    ]
                    filtered_normalized = non_maximum_suppression(
                        normalized_for_nms, iou_threshold=nms_iou_threshold
                    )
                    # Convert back to pixel coordinates for consistency
                    bboxes = [
                        (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
                        for x1, y1, x2, y2 in filtered_normalized
                    ]
                    logger.info(
                        f"[Florence-2 Grounding] NMS reduced {len(results[task]['bboxes'])} bboxes to {len(bboxes)} (IoU threshold: {nms_iou_threshold})"
                    )

                # Normalize to 0-1 range
                w, h = image.size
                for idx, bbox in enumerate(bboxes[:max_regions]):
                    x1, y1, x2, y2 = bbox
                    normalized = (x1 / w, y1 / h, x2 / w, y2 / h)
                    normalized_bboxes.append(normalized)

                    # Log bbox overlay image
                    if self.image_logger:
                        self.image_logger.log_image(
                            image=image,
                            stage="grounding",
                            step_index=step_index,
                            bbox_index=idx,
                            image_type="bbox_overlay",
                            bbox=normalized,
                            metadata={
                                "statement": statement,
                                "original_bbox": list(bbox),
                            },
                        )

                    intermediate_states.append(
                        {
                            "stage": "normalization",
                            "bbox_index": idx,
                            "original_bbox": list(bbox),
                            "normalized_bbox": list(normalized),
                            "image_size": (w, h),
                        }
                    )

                logger.debug(
                    f"Florence-2 Grounding extracted {len(normalized_bboxes)} regions for: {statement[:50]}"
                )
            else:
                logger.warning(
                    f"No bboxes found in Florence-2 Grounding response. Task in results: {task in results if isinstance(results, dict) else False}. Results: {results}"
                )

            # Log raw and parsed outputs
            if self.output_tracer:
                self.output_tracer.trace_grounding(
                    raw_output=raw_output,
                    bboxes=normalized_bboxes,
                    model_name=self.config.model_id,
                    statement=statement,
                    step_index=step_index or 0,
                    model_config={
                        "model_id": self.config.model_id,
                        "torch_dtype": str(self.config.torch_dtype),
                        "device": self.config.device,
                    },
                    intermediate_states=intermediate_states,
                    duration_ms=duration_ms,
                    metadata={"task_prompt": task, "max_regions": max_regions},
                )

            return normalized_bboxes

        except Exception as e:
            logger.error(
                f"Florence-2 Grounding region extraction failed: {e}", exc_info=True
            )
            return []


__all__ = ["Florence2GroundingClient"]
