"""
Pydantic schemas for structured outputs with grammar-constrained decoding.

These schemas are used with the Outlines library to force the model to generate
valid JSON that conforms to our expected structure, eliminating parsing errors.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ReasoningStepSchema(BaseModel):
    """Schema for a single reasoning step."""
    
    index: int = Field(..., description="1-based step index")
    statement: str = Field(..., description="Concise reasoning statement")
    needs_vision: bool = Field(..., description="Whether visual verification is required")
    reason: Optional[str] = Field(None, description="Short explanation for needs_vision flag")
    need_ocr: bool = Field(False, description="Whether OCR is needed for this step")


class ReasoningStepsSchema(BaseModel):
    """Schema for structured reasoning output."""
    
    steps: List[ReasoningStepSchema] = Field(..., description="List of reasoning steps")


class ROIEvidenceItemSchema(BaseModel):
    """Schema for a single region of interest evidence item."""
    
    step: int = Field(..., description="Step index this evidence supports")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2], normalized 0-1")
    description: str = Field(..., description="Visual description of the region")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")


class ROIEvidenceSchema(BaseModel):
    """Schema for ROI extraction output."""
    
    evidences: List[ROIEvidenceItemSchema] = Field(..., description="List of visual evidence regions")


class KeyEvidenceItemSchema(BaseModel):
    """Schema for a single key evidence item with reasoning."""
    
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2], normalized 0-1")
    description: str = Field(..., description="What this region shows")
    reasoning: str = Field(..., description="Why this evidence supports the answer")


class AnswerSynthesisSchema(BaseModel):
    """Schema for answer synthesis with structured evidence."""
    
    key_evidence: List[KeyEvidenceItemSchema] = Field(
        default_factory=list,
        description="Key visual evidence supporting the answer (maximum 2 items)"
    )
    explanation: str = Field(..., description="Brief explanation connecting key evidence to the answer")
    answer: str = Field(..., description="Final answer to the question (or single letter for multiple choice)")


__all__ = [
    "ReasoningStepSchema",
    "ReasoningStepsSchema",
    "ROIEvidenceItemSchema",
    "ROIEvidenceSchema",
    "KeyEvidenceItemSchema",
    "AnswerSynthesisSchema",
]

