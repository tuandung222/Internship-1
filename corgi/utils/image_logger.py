"""
Image logging infrastructure for comprehensive pipeline tracing.

Handles saving images at all pipeline stages with organized directory structure
and comprehensive metadata tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..core.trace_types import ImageLogEntry

logger = logging.getLogger(__name__)


class ImageLogger:
    """Handles image saving and metadata tracking for pipeline stages."""

    def __init__(self, output_dir: Path, enabled: bool = True):
        """
        Initialize image logger.

        Args:
            output_dir: Base output directory for saving images
            enabled: Whether logging is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.entries: List[ImageLogEntry] = []

        # Create base directories
        if self.enabled:
            self.images_dir = self.output_dir / "images"
            self.images_dir.mkdir(parents=True, exist_ok=True)

            # Create stage subdirectories
            for stage in [
                "original",
                "reasoning",
                "grounding",
                "captioning",
                "synthesis",
            ]:
                (self.images_dir / stage).mkdir(exist_ok=True)

    def log_image(
        self,
        image: Image.Image,
        stage: str,
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
        image_type: str = "input",
        bbox: Optional[Tuple[float, float, float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log an image with metadata.

        Args:
            image: PIL Image to save
            stage: Pipeline stage ('reasoning', 'grounding', 'captioning', 'synthesis', 'original')
            step_index: Step index (for multi-step stages)
            bbox_index: Bbox index (for multi-bbox stages)
            image_type: Type of image ('input', 'original', 'cropped', 'bbox_overlay')
            bbox: Bounding box coordinates if applicable
            metadata: Additional metadata to store

        Returns:
            Path to saved image file, or None if logging disabled
        """
        if not self.enabled:
            return None

        try:
            # Generate file path
            file_path = self.get_image_path(stage, step_index, bbox_index, image_type)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            image.save(file_path, format="PNG")
            logger.debug(f"Saved image: {file_path}")

            # Create log entry
            entry = ImageLogEntry(
                stage=stage,
                step_index=step_index,
                bbox_index=bbox_index,
                image_type=image_type,
                file_path=str(file_path.relative_to(self.output_dir)),
                original_size=image.size,
                bbox=bbox,
                metadata=metadata or {},
            )
            self.entries.append(entry)

            # Save metadata JSON
            metadata_path = file_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to log image: {e}", exc_info=True)
            return None

    def log_cropped_image(
        self,
        original: Image.Image,
        cropped: Image.Image,
        bbox: Tuple[float, float, float, float],
        stage: str,
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Log both original and cropped images with bbox overlay.

        Args:
            original: Original full image
            cropped: Cropped region image
            bbox: Bounding box coordinates (normalized [0,1])
            stage: Pipeline stage
            step_index: Step index
            bbox_index: Bbox index
            metadata: Additional metadata

        Returns:
            Tuple of (original_image_path, cropped_image_path)
        """
        if not self.enabled:
            return None, None

        try:
            # Calculate crop ratio
            orig_w, orig_h = original.size
            crop_w, crop_h = cropped.size
            crop_ratio = (crop_w * crop_h) / (orig_w * orig_h)

            # Log original image with bbox overlay
            original_with_bbox = self._draw_bbox_overlay(original, bbox)
            orig_path = self.log_image(
                original_with_bbox,
                stage=stage,
                step_index=step_index,
                bbox_index=bbox_index,
                image_type="original",
                bbox=bbox,
                metadata={**(metadata or {}), "crop_ratio": crop_ratio},
            )

            # Log cropped image
            crop_metadata = {
                **(metadata or {}),
                "crop_ratio": crop_ratio,
                "original_size": original.size,
                "cropped_size": cropped.size,
            }
            crop_path = self.log_image(
                cropped,
                stage=stage,
                step_index=step_index,
                bbox_index=bbox_index,
                image_type="cropped",
                bbox=bbox,
                metadata=crop_metadata,
            )

            # Update entries with crop info
            for entry in self.entries[-2:]:
                if entry.image_type == "original":
                    entry.cropped_size = cropped.size
                    entry.crop_ratio = crop_ratio
                elif entry.image_type == "cropped":
                    entry.original_size = original.size
                    entry.cropped_size = cropped.size
                    entry.crop_ratio = crop_ratio

            return orig_path, crop_path

        except Exception as e:
            logger.error(f"Failed to log cropped image: {e}", exc_info=True)
            return None, None

    def _draw_bbox_overlay(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        color: str = "red",
        width: int = 3,
    ) -> Image.Image:
        """Draw bounding box overlay on image."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        w, h = image.size
        x1, y1, x2, y2 = bbox

        # Convert normalized to pixel coordinates
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)

        # Draw rectangle
        draw.rectangle(
            [(left, top), (right, bottom)],
            outline=color,
            width=width,
        )

        return img_copy

    def get_image_path(
        self,
        stage: str,
        step_index: Optional[int],
        bbox_index: Optional[int],
        image_type: str,
    ) -> Path:
        """
        Generate organized file path for image.

        Args:
            stage: Pipeline stage
            step_index: Step index (None for original)
            bbox_index: Bbox index (None if not applicable)
            image_type: Type of image

        Returns:
            Path object for image file
        """
        if stage == "original":
            filename = "input_image.png"
            return self.images_dir / stage / filename

        # Stage-specific directory
        stage_dir = self.images_dir / stage

        if step_index is not None:
            step_dir = stage_dir / f"step_{step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)

            if bbox_index is not None:
                filename = f"bbox_{bbox_index}_{image_type}.png"
                return step_dir / filename
            else:
                filename = f"{image_type}_image.png"
                return step_dir / filename
        else:
            filename = f"{image_type}_image.png"
            return stage_dir / filename

    def get_all_entries(self) -> List[ImageLogEntry]:
        """Get all logged image entries."""
        return self.entries.copy()

    def save_metadata_summary(self, file_path: Optional[Path] = None) -> None:
        """Save summary of all image metadata to JSON."""
        if not self.enabled:
            return

        if file_path is None:
            file_path = self.output_dir / "metadata" / "image_metadata.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_images": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved image metadata summary: {file_path}")


__all__ = ["ImageLogger"]
