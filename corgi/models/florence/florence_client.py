"""
Florence-2 Large client for fast ROI extraction and captioning.

This is a convenience wrapper that combines the specialized Florence-2
grounding and captioning clients for backward compatibility.

For new code, consider using Florence2GroundingClient and Florence2CaptioningClient
directly for more fine-grained control.

Model: microsoft/Florence-2-large
"""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image

from ...core.config import ModelConfig
from .florence_grounding_client import Florence2GroundingClient
from .florence_captioning_client import Florence2CaptioningClient


class Florence2Client:
    """
    Convenience wrapper for Florence-2 Large model.

    Provides fast ROI extraction and captioning for CoRGI pipeline by
    combining the specialized grounding and captioning clients.

    For backward compatibility with existing code.
    """

    def __init__(
        self, model_id: str = "microsoft/Florence-2-large", device: str = "cuda:7"
    ):
        """
        Initialize Florence-2 client.

        Args:
            model_id: HuggingFace model ID for Florence-2
            device: Device to use (e.g., 'cuda:7'). Must be specified.
        """
        self.model_id = model_id

        if not device:
            raise ValueError("device must be specified (e.g., 'cuda:7')")

        # Create a config with device
        config = ModelConfig(
            model_id=model_id,
            model_type="florence2",
            device=device,
            enable_compile=True,
            enable_flash_attn=True,
        )

        # Initialize specialized clients
        # Note: They will share the same model cache due to model_id
        self.grounding_client = Florence2GroundingClient(config)
        self.captioning_client = Florence2CaptioningClient(config)

    def extract_regions(
        self,
        image: Image.Image,
        statement: str,
        max_regions: int = 3,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extract bounding boxes for regions relevant to the statement.

        Delegates to Florence2GroundingClient.

        Args:
            image: Input image
            statement: Reasoning statement to ground
            max_regions: Maximum number of regions to return

        Returns:
            List of bounding boxes as tuples (x1, y1, x2, y2) in normalized coordinates
        """
        return self.grounding_client.extract_regions(image, statement, max_regions)

    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
    ) -> str:
        """
        Generate caption for a specific region.

        Delegates to Florence2CaptioningClient.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates

        Returns:
            Textual description of the region
        """
        return self.captioning_client.caption_region(image, bbox)


__all__ = ["Florence2Client"]


def _load_florence_backend(
    model_id: str = "microsoft/Florence-2-large",
) -> tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Florence-2 model with optimizations."""
    if model_id not in _FLORENCE_MODEL_CACHE:
        logger.info(f"Loading Florence-2 model: {model_id}")

        # Determine dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Florence-2: Using bfloat16")
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
            logger.info("Florence-2: Using float16")
        else:
            torch_dtype = torch.float32
            logger.info("Florence-2: Using float32 (CPU mode)")

        # Device must be passed as parameter, no hardcode
        raise ValueError(
            "_load_florence_backend requires device parameter. Use Florence2GroundingClient or Florence2CaptioningClient with ModelConfig instead."
        )

        # OPTIMIZATION: Torch Compile for Florence-2
        if os.environ.get("CORGI_DISABLE_COMPILE", "0") != "1":
            try:
                logger.info("Compiling Florence-2 model...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("✓ Florence-2 torch compile enabled")
            except Exception as e:
                logger.warning(f"Florence-2 compile failed ({e})")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        _FLORENCE_MODEL_CACHE[model_id] = model
        _FLORENCE_PROCESSOR_CACHE[model_id] = processor
        logger.info(f"✓ Florence-2 model loaded successfully")

    return _FLORENCE_MODEL_CACHE[model_id], _FLORENCE_PROCESSOR_CACHE[model_id]


# Legacy class - use the first Florence2Client above instead
# This class is kept for backward compatibility but requires device parameter
class Florence2ClientLegacy:
    """
    Legacy client for Florence-2 Large model.

    DEPRECATED: Use Florence2Client with device parameter instead.
    """

    def __init__(
        self, model_id: str = "microsoft/Florence-2-large", device: str = "cuda:7"
    ):
        if not device:
            raise ValueError("device must be specified (e.g., 'cuda:7')")
        self.model_id = model_id
        # Use specialized clients instead of legacy _load_florence_backend
        config = ModelConfig(
            model_id=model_id,
            model_type="florence2",
            device=device,
            enable_compile=True,
            enable_flash_attn=True,
        )
        from .florence_grounding_client import Florence2GroundingClient
        from .florence_captioning_client import Florence2CaptioningClient

        self.grounding_client = Florence2GroundingClient(config)
        self.captioning_client = Florence2CaptioningClient(config)
        self._model = self.grounding_client._model
        self._processor = self.grounding_client._processor

    def _run_task(
        self,
        task_prompt: str,
        image: Image.Image,
        text_input: Optional[str] = None,
    ) -> dict:
        """Run Florence-2 task and return parsed results."""
        # Build prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        # Prepare inputs
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        # Move to device from config (use grounding client's config)
        device = torch.device(self.grounding_client.config.device)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate with autocast for bfloat16
        device_type = "cuda" if "cuda" in self.grounding_client.config.device else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,  # Sufficient for most tasks
                early_stopping=False,
                do_sample=False,
                num_beams=1,  # Greedy = faster
            )

        # Decode
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process
        parsed_answer = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        return parsed_answer

    def extract_regions(
        self,
        image: Image.Image,
        statement: str,
        max_regions: int = 3,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extract bounding boxes for regions relevant to the statement.

        Uses <CAPTION_TO_PHRASE_GROUNDING> task for phrase grounding.

        Args:
            image: Input image
            statement: Reasoning statement to ground
            max_regions: Maximum number of regions to return

        Returns:
            List of bounding boxes as tuples (x1, y1, x2, y2) in normalized coordinates
        """
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        try:
            results = self._run_task(task, image, statement)

            # Extract bboxes
            if task in results and "bboxes" in results[task]:
                bboxes = results[task]["bboxes"]
                # Normalize to 0-1 range
                w, h = image.size
                normalized_bboxes = []
                for bbox in bboxes[:max_regions]:
                    x1, y1, x2, y2 = bbox
                    normalized_bboxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
                logger.debug(
                    f"Florence-2 extracted {len(normalized_bboxes)} regions for: {statement[:50]}"
                )
                return normalized_bboxes
            else:
                logger.warning(f"No bboxes found in Florence-2 response: {results}")
                return []
        except Exception as e:
            logger.error(f"Florence-2 region extraction failed: {e}")
            return []

    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
    ) -> str:
        """
        Generate caption for a specific region.

        Crops the region and uses <DETAILED_CAPTION> task.

        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) in normalized coordinates

        Returns:
            Textual description of the region
        """
        try:
            # Crop region
            cropped = self._crop_region(image, bbox)

            # Caption the cropped region
            task = "<DETAILED_CAPTION>"
            results = self._run_task(task, cropped)

            if task in results:
                caption = results[task]
                logger.debug(f"Florence-2 caption: {caption[:80]}")
                return caption
            else:
                logger.warning(f"No caption in Florence-2 response: {results}")
                return "Unable to caption region"
        except Exception as e:
            logger.error(f"Florence-2 captioning failed: {e}")
            return "Caption generation failed"

    def caption_dense_regions(
        self,
        image: Image.Image,
        max_regions: int = 5,
    ) -> List[Tuple[Tuple[float, float, float, float], str]]:
        """
        Generate dense region captions.

        Uses <DENSE_REGION_CAPTION> to get multiple labeled regions.

        Args:
            image: Input image
            max_regions: Maximum regions to return

        Returns:
            List of (bbox, caption) tuples
        """
        task = "<DENSE_REGION_CAPTION>"
        try:
            results = self._run_task(task, image)

            if task in results:
                data = results[task]
                bboxes = data.get("bboxes", [])
                labels = data.get("labels", [])

                # Normalize and combine
                w, h = image.size
                region_captions = []
                for bbox, label in zip(bboxes[:max_regions], labels[:max_regions]):
                    x1, y1, x2, y2 = bbox
                    normalized_bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                    region_captions.append((normalized_bbox, label))

                logger.debug(
                    f"Florence-2 dense captions: {len(region_captions)} regions"
                )
                return region_captions
            else:
                logger.warning(f"No dense captions in Florence-2 response: {results}")
                return []
        except Exception as e:
            logger.error(f"Florence-2 dense captioning failed: {e}")
            return []

    @staticmethod
    def _crop_region(
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
    ) -> Image.Image:
        """Crop image to bounding box region."""
        x1, y1, x2, y2 = bbox
        w, h = image.size

        # Convert normalized to pixel coordinates
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)

        # Ensure valid coordinates
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))

        return image.crop((left, top, right, bottom))


__all__ = ["Florence2Client"]
