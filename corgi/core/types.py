from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BBoxFormat(Enum):
    """Format for bounding box coordinates."""
    NORMALIZED_0_1 = "normalized"  # [0, 1] range (Florence-2, internal standard)
    QWEN_0_999 = "qwen"            # [0, 999] range (Qwen3-VL)
    PIXEL = "pixel"                 # Absolute pixel coordinates


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ReasoningStep:
    """Represents a single structured reasoning step."""

    index: int
    statement: str
    needs_vision: bool
    reason: Optional[str] = None
    need_ocr: bool = False


@dataclass(frozen=True)
class GroundedEvidence:
    """Evidence item grounded to a region of interest in the image."""

    step_index: int
    bbox: BBox
    description: Optional[str] = None
    ocr_text: Optional[str] = None
    confidence: Optional[float] = None
    raw_source: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class KeyEvidence:
    """Key evidence with reasoning for final answer."""
    
    bbox: BBox
    description: str
    reasoning: str


@dataclass(frozen=True)
class PromptLog:
    """Capture the prompt/response pair used at a given pipeline stage."""

    prompt: str
    response: Optional[str] = None
    step_index: Optional[int] = None
    stage: Optional[str] = None


@dataclass(frozen=True)
class StageTiming:
    """Timing metadata for a pipeline stage or sub-step."""

    name: str
    duration_ms: float
    step_index: Optional[int] = None


def steps_to_serializable(steps: List[ReasoningStep]) -> List[Dict[str, object]]:
    """Helper to convert steps into JSON-friendly dictionaries."""

    return [
        {
            "index": step.index,
            "statement": step.statement,
            "needs_vision": step.needs_vision,
            "need_ocr": step.need_ocr,
            **({"reason": step.reason} if step.reason is not None else {}),
        }
        for step in steps
    ]


def evidences_to_serializable(evidences: List[GroundedEvidence]) -> List[Dict[str, object]]:
    """Helper to convert evidences into JSON-friendly dictionaries."""

    serializable: List[Dict[str, object]] = []
    for ev in evidences:
        item: Dict[str, object] = {
            "step_index": ev.step_index,
            "bbox": list(ev.bbox),
        }
        if ev.description is not None:
            item["description"] = ev.description
        if ev.ocr_text is not None:
            item["ocr_text"] = ev.ocr_text
        if ev.confidence is not None:
            item["confidence"] = ev.confidence
        if ev.raw_source is not None:
            item["raw_source"] = ev.raw_source
        serializable.append(item)
    return serializable


def prompt_logs_to_serializable(logs: List[PromptLog]) -> List[Dict[str, object]]:
    """Convert prompt logs into JSON-friendly structures."""

    serializable: List[Dict[str, object]] = []
    for log in logs:
        item: Dict[str, object] = {"prompt": log.prompt}
        if log.response is not None:
            item["response"] = log.response
        if log.step_index is not None:
            item["step_index"] = log.step_index
        if log.stage is not None:
            item["stage"] = log.stage
        serializable.append(item)
    return serializable


def stage_timings_to_serializable(timings: List[StageTiming]) -> List[Dict[str, object]]:
    serializable: List[Dict[str, object]] = []
    for timing in timings:
        item: Dict[str, object] = {
            "name": timing.name,
            "duration_ms": timing.duration_ms,
        }
        if timing.step_index is not None:
            item["step_index"] = timing.step_index
        serializable.append(item)
    return serializable