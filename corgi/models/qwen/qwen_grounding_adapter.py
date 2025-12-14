"""
Qwen Grounding Adapter.

Adapter to use Qwen models for visual grounding (region extraction).
This wraps the existing Qwen-based ROI extraction logic to implement
the grounding protocol.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Union
import logging

from PIL import Image

from ...utils.prompts import QWEN_GROUNDING_PROMPT, QWEN_BATCH_GROUNDING_PROMPT
from ...utils.parsers import parse_roi_evidence
from ...utils.coordinate_utils import from_qwen_format

logger = logging.getLogger(__name__)


class QwenGroundingAdapter:
    """
    Adapter to use Qwen for visual grounding.

    This adapter wraps a Qwen client (Instruct or Thinking) and implements
    the grounding protocol by using Qwen's ROI extraction capabilities.
    """

    def __init__(self, qwen_client):
        """
        Initialize Qwen grounding adapter.

        Args:
            qwen_client: A Qwen client (Qwen3VLInstructClient or Qwen3VLThinkingClient)
        """
        self.client = qwen_client

    def extract_regions(
        self,
        image: Image.Image,
        statement: str = None,
        max_regions: int = 3,
        statements: List[Tuple[int, str]] = None,
    ) -> Union[
        List[Tuple[float, float, float, float]],
        Dict[int, List[Tuple[float, float, float, float]]],
    ]:
        """
        Extract bounding boxes for regions relevant to the statement(s).

        Uses Qwen's ROI extraction capability with JSON-based grounding prompts.
        Supports both single statement (backward compatible) and batch mode.

        Args:
            image: Input image
            statement: Single reasoning statement to ground (for backward compatibility)
            max_regions: Maximum number of regions to return per step
            statements: List of (step_index, statement) pairs for batch grounding

        Returns:
            - If single statement: List of bounding boxes as tuples (x1, y1, x2, y2) in normalized coordinates [0, 1]
            - If batch mode: Dict mapping step_index -> List[bboxes] for that step
        """
        # Determine if batch mode or single statement mode
        if statements is not None and len(statements) > 0:
            # Batch mode: extract all bboxes in one inference
            return self._extract_regions_batch(image, statements, max_regions)
        elif statement is not None:
            # Single statement mode (backward compatible)
            return self._extract_regions_single(image, statement, max_regions)
        else:
            logger.error("Either 'statement' or 'statements' must be provided")
            return []

    def _extract_regions_single(
        self,
        image: Image.Image,
        statement: str,
        max_regions: int = 3,
    ) -> List[Tuple[float, float, float, float]]:
        """Extract regions for a single statement (backward compatible)."""
        prompt = QWEN_GROUNDING_PROMPT.format(
            step_statement=statement,
            max_regions=max_regions,
        )

        try:
            # Use the Qwen client's _chat method
            response = self.client._chat(image=image, prompt=prompt, max_new_tokens=128)

            if not response or not response.strip():
                logger.warning(
                    f"Empty response from Qwen grounding for statement: {statement[:50]}"
                )
                return []

            # Log raw response for debugging
            logger.debug(
                f"Qwen grounding raw response (first 500 chars): {response[:500]}"
            )

            # Parse the ROI evidence (parse with bbox_format="qwen" to convert from [0, 999] to [0, 1])
            try:
                evidences = parse_roi_evidence(
                    response, default_step_index=1, bbox_format="qwen"
                )
            except ValueError as e:
                # Log the full response if parsing fails
                logger.warning(f"Failed to parse Qwen grounding response: {e}")
                logger.warning(f"Full response (first 1000 chars): {response[:1000]}")
                return []

            # Extract bboxes (already normalized to [0, 1])
            bboxes = [ev.bbox for ev in evidences[:max_regions]]

            if bboxes:
                logger.info(
                    f"Qwen grounding extracted {len(bboxes)} regions for: {statement[:50]}"
                )
            else:
                logger.warning(
                    f"Qwen grounding extracted 0 regions for: {statement[:50]}"
                )

            return bboxes

        except Exception as e:
            logger.error(f"Qwen grounding extraction failed: {e}", exc_info=True)
            return []

    def _extract_regions_batch(
        self,
        image: Image.Image,
        statements: List[Tuple[int, str]],
        max_regions: int = 3,
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Extract regions for multiple statements in a single inference.

        Args:
            image: Input image
            statements: List of (step_index, statement) pairs
            max_regions: Maximum number of regions to return per step

        Returns:
            Dict mapping step_index -> List[bboxes] for that step
        """
        # Build statements list for prompt
        statements_text = "\n".join(
            [f"Step {step_idx}: {stmt}" for step_idx, stmt in statements]
        )

        prompt = QWEN_BATCH_GROUNDING_PROMPT.format(
            statements=statements_text,
            max_regions=max_regions,
        )

        try:
            # Single inference call for all statements
            response = self.client._chat(image=image, prompt=prompt, max_new_tokens=512)

            if not response or not response.strip():
                logger.warning(
                    f"Empty response from Qwen batch grounding for {len(statements)} statements"
                )
                return {step_idx: [] for step_idx, _ in statements}

            # Log raw response for debugging
            logger.debug(
                f"Qwen batch grounding raw response (first 1000 chars): {response[:1000]}"
            )

            # Parse the ROI evidence (parse with bbox_format="qwen" to convert from [0, 999] to [0, 1])
            try:
                evidences = parse_roi_evidence(
                    response, default_step_index=None, bbox_format="qwen"
                )
            except ValueError as e:
                # Log the full response if parsing fails
                logger.warning(f"Failed to parse Qwen batch grounding response: {e}")
                logger.warning(f"Full response (first 2000 chars): {response[:2000]}")
                return {step_idx: [] for step_idx, _ in statements}

            # Group bboxes by step index
            result: Dict[int, List[Tuple[float, float, float, float]]] = {
                step_idx: [] for step_idx, _ in statements
            }

            for ev in evidences:
                step_idx = ev.step_index
                if step_idx in result:
                    result[step_idx].append(ev.bbox)
                else:
                    logger.warning(
                        f"Found bbox for unknown step index {step_idx}, ignoring"
                    )

            # Limit to max_regions per step
            for step_idx in result:
                result[step_idx] = result[step_idx][:max_regions]

            # Log results
            total_bboxes = sum(len(bboxes) for bboxes in result.values())
            logger.info(
                f"Qwen batch grounding extracted {total_bboxes} total regions across {len(statements)} steps"
            )
            for step_idx, bboxes in result.items():
                if bboxes:
                    logger.debug(f"  Step {step_idx}: {len(bboxes)} regions")

            return result

        except Exception as e:
            logger.error(f"Qwen batch grounding extraction failed: {e}", exc_info=True)
            return {step_idx: [] for step_idx, _ in statements}


__all__ = ["QwenGroundingAdapter"]


__all__ = ["QwenGroundingAdapter"]
