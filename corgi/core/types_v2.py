"""
V2 Data Models for Pipeline V2.

NEW data structures for enhanced reasoning with evidence type flags.
V1 types remain in types.py - DO NOT MODIFY V1!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStepV2:
    """
    V2 Reasoning Step with evidence type discrimination.

    Key changes from V1:
    - Added need_object_captioning flag
    - Added need_text_ocr flag
    - Added optional bbox from Phase 1
    - Enforces mutual exclusion between flags
    """

    index: int
    statement: str
    need_object_captioning: bool  # NEW: Requires visual object/scene understanding
    need_text_ocr: bool  # NEW: Requires text/number recognition
    bbox: Optional[List[float]] = None  # NEW: Optional bbox from Phase 1 [x1,y1,x2,y2]
    reason: Optional[str] = None

    def __post_init__(self):
        """Validate mutual exclusion constraint."""
        if self.need_object_captioning and self.need_text_ocr:
            logger.warning(
                f"Step {self.index}: Both need_object_captioning and need_text_ocr "
                f"are True (mutually exclusive). Forcing need_object_captioning=False."
            )
            # Auto-fix: Prefer OCR if both are set
            self.need_object_captioning = False

        # Validate bbox format if provided
        if self.bbox is not None:
            if not isinstance(self.bbox, list) or len(self.bbox) != 4:
                raise ValueError(
                    f"Step {self.index}: bbox must be a list of 4 floats [x1,y1,x2,y2], "
                    f"got {self.bbox}"
                )
            if not all(0.0 <= coord <= 1.0 for coord in self.bbox):
                logger.warning(
                    f"Step {self.index}: bbox coordinates should be normalized [0,1], "
                    f"got {self.bbox}"
                )

    @property
    def needs_vision(self) -> bool:
        """Backward compatibility: needs vision if either flag is True."""
        return self.need_object_captioning or self.need_text_ocr

    @property
    def evidence_type(self) -> str:
        """
        Return evidence type: 'object', 'text', or 'none'.

        Used for routing in Phase 3.
        """
        if self.need_object_captioning:
            return "object"
        elif self.need_text_ocr:
            return "text"
        else:
            return "none"

    @property
    def has_bbox(self) -> bool:
        """Check if bbox is provided from Phase 1 (skip grounding)."""
        return self.bbox is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "statement": self.statement,
            "need_object_captioning": self.need_object_captioning,
            "need_text_ocr": self.need_text_ocr,
            "bbox": self.bbox,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningStepV2":
        """Create from dictionary (for deserialization)."""
        return cls(
            index=data["index"],
            statement=data["statement"],
            need_object_captioning=data.get("need_object_captioning", False),
            need_text_ocr=data.get("need_text_ocr", False),
            bbox=data.get("bbox"),
            reason=data.get("reason"),
        )


@dataclass
class GroundedEvidenceV2:
    """
    V2 Grounded Evidence with explicit evidence type.

    Key changes from V1:
    - Added evidence_type field ('object' or 'text')
    - description is None for text evidence
    - ocr_text is None for object evidence
    """

    step_index: int
    statement: str
    bbox: Tuple[float, float, float, float]
    evidence_type: str  # NEW: 'object' or 'text'
    description: Optional[str] = None  # Caption for objects
    ocr_text: Optional[str] = None  # OCR for text
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate evidence type consistency."""
        if self.evidence_type not in ["object", "text"]:
            raise ValueError(
                f"evidence_type must be 'object' or 'text', got {self.evidence_type}"
            )

        # Enforce type-specific fields
        if self.evidence_type == "object":
            if self.description is None:
                logger.warning(
                    f"Step {self.step_index}: evidence_type='object' but description is None"
                )
            if self.ocr_text is not None:
                logger.warning(
                    f"Step {self.step_index}: evidence_type='object' should not have ocr_text, "
                    "setting to None"
                )
                self.ocr_text = None

        elif self.evidence_type == "text":
            if self.ocr_text is None:
                logger.warning(
                    f"Step {self.step_index}: evidence_type='text' but ocr_text is None"
                )
            if self.description is not None:
                logger.warning(
                    f"Step {self.step_index}: evidence_type='text' should not have description, "
                    "setting to None"
                )
                self.description = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_index": self.step_index,
            "statement": self.statement,
            "bbox": list(self.bbox),
            "evidence_type": self.evidence_type,
            "description": self.description,
            "ocr_text": self.ocr_text,
            "confidence": self.confidence,
        }


__all__ = ["ReasoningStepV2", "GroundedEvidenceV2"]
