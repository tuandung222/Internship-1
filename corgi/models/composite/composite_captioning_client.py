"""
Composite Captioning Client.

Combines PaddleOCR for OCR and FastVLM for Captioning+VQA.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import logging

from PIL import Image

from ...core.config import ModelConfig

logger = logging.getLogger(__name__)


class CompositeCaptioningClient:
    """
    Composite client that combines PaddleOCR (OCR) and FastVLM (Captioning+VQA).

    This allows using different models for OCR and Captioning tasks.
    """

    def __init__(
        self,
        ocr_client,  # PaddleOCRClient
        captioning_client,  # FastVLMCaptioningClient
        image_logger=None,
        output_tracer=None,
    ) -> None:
        """
        Initialize composite captioning client.

        Args:
            ocr_client: PaddleOCR client for OCR tasks
            captioning_client: FastVLM client for Captioning+VQA tasks
            image_logger: Optional ImageLogger instance
            output_tracer: Optional OutputTracer instance
        """
        self.ocr_client = ocr_client
        self.captioning_client = captioning_client
        self.image_logger = image_logger
        self.output_tracer = output_tracer

    def ocr_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
    ) -> str:
        """Extract OCR text from region using PaddleOCR."""
        return self.ocr_client.ocr_region(image, bbox, step_index, bbox_index)

    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        bbox_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> str:
        """Generate caption for region using FastVLM (with VQA if statement provided)."""
        return self.captioning_client.caption_region(
            image, bbox, step_index, bbox_index, statement
        )

    def ocr_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
    ) -> List[str]:
        """Extract OCR text from multiple regions using PaddleOCR."""
        if hasattr(self.ocr_client, "ocr_regions_batch"):
            return self.ocr_client.ocr_regions_batch(image, bboxes, step_index)
        else:
            # Fallback to sequential calls
            return [
                self.ocr_client.ocr_region(image, bbox, step_index, i)
                for i, bbox in enumerate(bboxes)
            ]

    def caption_regions_batch(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        step_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> List[str]:
        """Generate captions for multiple regions using FastVLM."""
        if hasattr(self.captioning_client, "caption_regions_batch"):
            return self.captioning_client.caption_regions_batch(
                image, bboxes, step_index, statement
            )
        else:
            # Fallback to sequential calls
            return [
                self.captioning_client.caption_region(
                    image, bbox, step_index, i, statement
                )
                for i, bbox in enumerate(bboxes)
            ]


__all__ = ["CompositeCaptioningClient"]
