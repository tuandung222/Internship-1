"""
V2 Parsers for Pipeline V2.

NEW parsers for enhanced JSON format with evidence type flags.
V1 parsers remain in parsers.py - DO NOT MODIFY V1!
"""

from __future__ import annotations

import json
import re
import logging
from typing import List, Tuple

from ..core.types_v2 import ReasoningStepV2

logger = logging.getLogger(__name__)


def parse_structured_reasoning_v2(response: str) -> Tuple[str, List[ReasoningStepV2]]:
    """
    Parse V2 reasoning response with enhanced JSON format.

    Expected format:
        <THINKING>
        [Chain of thought reasoning]
        </THINKING>

        <STRUCTURED_STEPS>
        {
          "steps": [
            {
              "index": 1,
              "statement": "...",
              "need_object_captioning": true/false,
              "need_text_ocr": true/false,
              "bbox": [x1, y1, x2, y2] or null,
              "reason": "..." (optional)
            },
            ...
          ]
        }
        </STRUCTURED_STEPS>

    Args:
        response: Raw model response

    Returns:
        Tuple of (cot_text, list of ReasoningStepV2)

    Raises:
        ValueError: If response format is invalid
    """
    # Extract CoT (Chain of Thought)
    cot_match = re.search(
        r"<THINKING>(.*?)</THINKING>", response, re.DOTALL | re.IGNORECASE
    )
    cot_text = cot_match.group(1).strip() if cot_match else ""

    if not cot_text:
        logger.warning("No <THINKING> section found in V2 response")

    # Extract JSON
    json_match = re.search(
        r"<STRUCTURED_STEPS>(.*?)</STRUCTURED_STEPS>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not json_match:
        raise ValueError("No <STRUCTURED_STEPS> found in V2 response")

    json_text = json_match.group(1).strip()

    # Parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse V2 JSON: {e}")
        logger.error(f"JSON text: {json_text[:500]}")
        raise ValueError(f"Invalid JSON in <STRUCTURED_STEPS>: {e}")

    if "steps" not in data:
        raise ValueError("Missing 'steps' key in V2 JSON")

    # Parse steps
    steps = []
    for i, step_dict in enumerate(data["steps"]):
        try:
            # Validate required fields
            if "index" not in step_dict:
                logger.warning(f"Step {i}: Missing 'index', using {i+1}")
                step_dict["index"] = i + 1

            if "statement" not in step_dict:
                raise ValueError(
                    f"Step {step_dict.get('index', i+1)}: Missing 'statement'"
                )

            # Get flags with defaults
            need_obj = step_dict.get("need_object_captioning", False)
            need_ocr = step_dict.get("need_text_ocr", False)

            # Auto-fix: if both flags are set, prefer OCR (keep text flag)
            if need_obj and need_ocr:
                logger.warning(
                    f"Step {step_dict['index']}: Both flags True (mutually exclusive), "
                    "keeping need_text_ocr=True, setting need_object_captioning=False"
                )
                need_obj = False

            # Parse bbox if present
            bbox = step_dict.get("bbox")
            if bbox is not None:
                # Validate bbox format
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning(
                        f"Step {step_dict['index']}: Invalid bbox format {bbox}, ignoring"
                    )
                    bbox = None
                elif not all(isinstance(x, (int, float)) for x in bbox):
                    logger.warning(
                        f"Step {step_dict['index']}: bbox contains non-numeric values, ignoring"
                    )
                    bbox = None

            # Create step (validation happens in __post_init__)
            step = ReasoningStepV2(
                index=step_dict["index"],
                statement=step_dict["statement"],
                need_object_captioning=need_obj,
                need_text_ocr=need_ocr,
                bbox=bbox,
                reason=step_dict.get("reason"),
            )
            steps.append(step)

        except Exception as e:
            logger.error(f"Failed to parse step {i}: {e}")
            logger.error(f"Step data: {step_dict}")
            # Continue with next step instead of failing entire parse
            continue

    if not steps:
        raise ValueError("No valid steps parsed from V2 response")

    logger.info(f"Parsed {len(steps)} V2 reasoning steps from response")

    # Log statistics
    vision_steps = sum(1 for s in steps if s.needs_vision)
    bbox_steps = sum(1 for s in steps if s.has_bbox)
    obj_steps = sum(1 for s in steps if s.need_object_captioning)
    ocr_steps = sum(1 for s in steps if s.need_text_ocr)

    logger.info(
        f"V2 steps: {vision_steps} vision ({obj_steps} object, {ocr_steps} text), "
        f"{bbox_steps} with bbox"
    )

    return cot_text, steps


def parse_v2_fallback_to_v1(response: str) -> Tuple[str, List[ReasoningStepV2]]:
    """
    Fallback parser: Try V2 first, if fails, parse as V1 and convert to V2.

    This ensures backward compatibility with V1-style responses.
    """
    try:
        return parse_structured_reasoning_v2(response)
    except Exception as e:
        logger.warning(f"V2 parsing failed ({e}), attempting V1 fallback")

        # Import V1 parser
        from ..utils.parsers import parse_structured_reasoning

        try:
            cot, v1_steps = parse_structured_reasoning(response)

            # Convert V1 steps to V2 format
            # V1 has: needs_vision, need_ocr
            # V2 needs: need_object_captioning, need_text_ocr
            v2_steps = []
            for v1_step in v1_steps:
                # Default: if needs_vision but no OCR flag, assume object captioning
                if hasattr(v1_step, "need_ocr") and v1_step.need_ocr:
                    need_obj = False
                    need_ocr = True
                elif hasattr(v1_step, "needs_vision") and v1_step.needs_vision:
                    need_obj = True
                    need_ocr = False
                else:
                    need_obj = False
                    need_ocr = False

                v2_step = ReasoningStepV2(
                    index=v1_step.index,
                    statement=v1_step.statement,
                    need_object_captioning=need_obj,
                    need_text_ocr=need_ocr,
                    bbox=None,  # V1 doesn't have bbox
                    reason=v1_step.reason if hasattr(v1_step, "reason") else None,
                )
                v2_steps.append(v2_step)

            logger.info(f"Successfully converted {len(v2_steps)} V1 steps to V2 format")
            return cot, v2_steps

        except Exception as e2:
            logger.error(f"V1 fallback also failed: {e2}")
            raise ValueError(f"Both V2 and V1 parsers failed") from e


def parse_v2_thinking(response: str) -> Tuple[str, List[ReasoningStepV2]]:
    """
    Parse Qwen3-Thinking model response with <think> tags.

    Thinking model uses <think> instead of <THINKING>.

    Expected format:
        <think>
        [Internal reasoning]
        </think>

        <STRUCTURED_STEPS>
        {
          "steps": [...]
        }
        </STRUCTURED_STEPS>

    Args:
        response: Raw model response

    Returns:
        Tuple of (thinking_text, list of ReasoningStepV2)

    Raises:
        ValueError: If response format is invalid
    """
    # Extract <think> content (Thinking model)
    think_match = re.search(
        r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE
    )
    thinking_text = think_match.group(1).strip() if think_match else ""

    if not thinking_text:
        logger.warning("No <think> section found in Thinking model response")

    # Extract <STRUCTURED_STEPS>
    json_match = re.search(
        r"<STRUCTURED_STEPS>(.*?)</STRUCTURED_STEPS>",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if not json_match:
        raise ValueError("No <STRUCTURED_STEPS> found in Thinking model response")

    json_text = json_match.group(1).strip()

    # Parse JSON (same logic as V2)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Thinking model JSON: {e}")
        logger.error(f"JSON text: {json_text[:500]}")
        raise ValueError(f"Invalid JSON in <STRUCTURED_STEPS>: {e}")

    if "steps" not in data:
        raise ValueError("Missing 'steps' key in Thinking model JSON")

    # Parse steps (reuse V2 logic)
    steps = []
    for i, step_dict in enumerate(data["steps"]):
        try:
            if "index" not in step_dict:
                logger.warning(f"Thinking step {i}: Missing 'index', using {i+1}")
                step_dict["index"] = i + 1

            if "statement" not in step_dict:
                raise ValueError(
                    f"Thinking step {step_dict.get('index', i+1)}: Missing 'statement'"
                )

            need_obj = step_dict.get("need_object_captioning", False)
            need_ocr = step_dict.get("need_text_ocr", False)

            # Auto-fix mutual exclusion
            if need_obj and need_ocr:
                logger.warning(
                    f"Thinking step {step_dict['index']}: Both flags True, "
                    "setting need_object_captioning=False"
                )
                need_obj = False

            # Parse bbox
            bbox = step_dict.get("bbox")
            if bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning(
                        f"Thinking step {step_dict['index']}: Invalid bbox, ignoring"
                    )
                    bbox = None
                elif not all(isinstance(x, (int, float)) for x in bbox):
                    logger.warning(
                        f"Thinking step {step_dict['index']}: Non-numeric bbox, ignoring"
                    )
                    bbox = None

            step = ReasoningStepV2(
                index=step_dict["index"],
                statement=step_dict["statement"],
                need_object_captioning=need_obj,
                need_text_ocr=need_ocr,
                bbox=bbox,
                reason=step_dict.get("reason"),
            )
            steps.append(step)

        except Exception as e:
            logger.error(f"Failed to parse Thinking step {i}: {e}")
            logger.error(f"Step data: {step_dict}")
            continue

    if not steps:
        raise ValueError("No valid steps parsed from Thinking model response")

    logger.info(f"Parsed {len(steps)} Thinking model steps")

    # Log statistics
    vision_steps = sum(1 for s in steps if s.needs_vision)
    bbox_steps = sum(1 for s in steps if s.has_bbox)
    obj_steps = sum(1 for s in steps if s.need_object_captioning)
    ocr_steps = sum(1 for s in steps if s.need_text_ocr)

    logger.info(
        f"Thinking model: {vision_steps} vision ({obj_steps} object, {ocr_steps} text), "
        f"{bbox_steps} with bbox"
    )

    return thinking_text, steps


__all__ = [
    "parse_structured_reasoning_v2",
    "parse_v2_fallback_to_v1",
    "parse_v2_thinking",
]
