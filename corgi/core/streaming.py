"""
CoRGI Streaming API

Provides a generator-based interface for streaming pipeline execution results.
Each step of the pipeline yields events that consumers can process progressively.

Usage:
    from corgi.core.streaming import StreamingPipelineExecutor, StreamEventType

    executor = StreamingPipelineExecutor(pipeline)

    for event in executor.run(image, question):
        if event.type == StreamEventType.PHASE_START:
            print(f"Starting: {event.phase}")
        elif event.type == StreamEventType.STEP:
            print(f"Step {event.step_index}: {event.data}")
        elif event.type == StreamEventType.EVIDENCE:
            print(f"Evidence: {event.data}")
        elif event.type == StreamEventType.ANSWER:
            print(f"Answer: {event.data['answer']}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .pipeline_v2 import CoRGIPipelineV2, PipelineResultV2
    from .types_v2 import ReasoningStepV2, GroundedEvidenceV2


class StreamEventType(Enum):
    """Types of streaming events emitted during pipeline execution."""

    # Lifecycle events
    PIPELINE_START = auto()
    PIPELINE_END = auto()

    # Phase events
    PHASE_START = auto()
    PHASE_END = auto()

    # Intermediate results
    COT_TEXT = auto()  # Chain-of-thought text generated
    STEP = auto()  # Reasoning step generated
    BBOX = auto()  # Bounding box generated (from Phase 1 or fallback)
    EVIDENCE = auto()  # Evidence extracted (OCR or caption)

    # Final results
    ANSWER = auto()  # Final answer
    KEY_EVIDENCE = auto()  # Key evidence items

    # Error/warning
    WARNING = auto()
    ERROR = auto()


@dataclass
class StreamEvent:
    """
    A single streaming event from pipeline execution.

    Attributes:
        type: The type of event
        phase: Current phase name (e.g., "reasoning", "evidence", "synthesis")
        step_index: Step index (if applicable)
        data: Event-specific data payload
        timestamp: Event timestamp (seconds since pipeline start)
        duration_ms: Duration of operation (if applicable)
    """

    type: StreamEventType
    phase: Optional[str] = None
    step_index: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "type": self.type.name,
            "phase": self.phase,
            "step_index": self.step_index,
            "data": self.data,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class StreamingPipelineExecutor:
    """
    Executes pipeline with streaming events.

    Wraps a CoRGIPipelineV2 instance and provides a generator interface
    that yields StreamEvent objects as the pipeline progresses.
    """

    def __init__(self, pipeline: "CoRGIPipelineV2"):
        """
        Initialize streaming executor.

        Args:
            pipeline: A CoRGIPipelineV2 instance (must be initialized with VLM client)
        """
        self._pipeline = pipeline
        self._start_time: float = 0.0

    def _elapsed(self) -> float:
        """Get elapsed time since pipeline start."""
        return time.monotonic() - self._start_time

    def _emit(
        self,
        event_type: StreamEventType,
        phase: Optional[str] = None,
        step_index: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> StreamEvent:
        """Create and return a stream event."""
        return StreamEvent(
            type=event_type,
            phase=phase,
            step_index=step_index,
            data=data or {},
            timestamp=self._elapsed(),
            duration_ms=duration_ms,
        )

    def run(
        self,
        image: Image.Image,
        question: str,
        max_steps: int = 6,
        max_regions: int = 1,
    ) -> Generator[StreamEvent, None, "PipelineResultV2"]:
        """
        Run pipeline with streaming events.

        Yields StreamEvent objects as the pipeline executes.
        The final return value is the complete PipelineResultV2.

        Args:
            image: Input image
            question: Question to answer
            max_steps: Maximum reasoning steps
            max_regions: Maximum regions per step (fallback grounding)

        Yields:
            StreamEvent objects for each pipeline step

        Returns:
            PipelineResultV2 (accessible via generator.send() or returned at end)
        """
        from .pipeline_v2 import PipelineResultV2
        from .types_v2 import GroundedEvidenceV2
        from .types import StageTiming, KeyEvidence

        self._start_time = time.monotonic()
        vlm = self._pipeline._vlm

        # Reset logs
        vlm.reset_logs()

        # === PIPELINE START ===
        yield self._emit(
            StreamEventType.PIPELINE_START,
            data={
                "question": question,
                "max_steps": max_steps,
                "pipeline_version": "v2",
            },
        )

        timings: List[StageTiming] = []
        all_steps = []
        all_evidences = []

        # === PHASE 1+2: Reasoning + Grounding (Merged) ===
        yield self._emit(
            StreamEventType.PHASE_START,
            phase="reasoning_grounding",
            data={"description": "Structured reasoning with inline grounding"},
        )

        phase1_start = time.monotonic()
        cot_text, steps = vlm.structured_reasoning_v2(image, question, max_steps)
        phase1_duration = (time.monotonic() - phase1_start) * 1000.0

        timings.append(StageTiming(name="phase1_2_merged", duration_ms=phase1_duration))

        # Emit CoT text
        yield self._emit(
            StreamEventType.COT_TEXT,
            phase="reasoning_grounding",
            data={"cot_text": cot_text},
            duration_ms=phase1_duration,
        )

        # Emit each step
        for step in steps:
            yield self._emit(
                StreamEventType.STEP,
                phase="reasoning_grounding",
                step_index=step.index,
                data={
                    "statement": step.statement,
                    "need_object_captioning": step.need_object_captioning,
                    "need_text_ocr": step.need_text_ocr,
                    "has_bbox": step.has_bbox,
                    "bbox": step.bbox if step.has_bbox else None,
                },
            )

        # Fallback grounding for steps missing bbox
        updated_steps = []
        for step in steps:
            if step.needs_vision and not step.has_bbox:
                # Need fallback grounding
                yield self._emit(
                    StreamEventType.PHASE_START,
                    phase="fallback_grounding",
                    step_index=step.index,
                    data={"reason": "Step needs vision but has no bbox"},
                )

                grounding_start = time.monotonic()
                bboxes = vlm.extract_bboxes_fallback(image, step.statement)
                grounding_duration = (time.monotonic() - grounding_start) * 1000.0

                if bboxes:
                    # Create updated step with bbox
                    from .types_v2 import ReasoningStepV2

                    updated_step = ReasoningStepV2(
                        index=step.index,
                        statement=step.statement,
                        need_object_captioning=step.need_object_captioning,
                        need_text_ocr=step.need_text_ocr,
                        bbox=list(bboxes[0]),
                        reason=step.reason,
                    )
                    updated_steps.append(updated_step)

                    yield self._emit(
                        StreamEventType.BBOX,
                        phase="fallback_grounding",
                        step_index=step.index,
                        data={"bbox": list(bboxes[0]), "source": "fallback"},
                        duration_ms=grounding_duration,
                    )
                else:
                    updated_steps.append(step)
                    yield self._emit(
                        StreamEventType.WARNING,
                        phase="fallback_grounding",
                        step_index=step.index,
                        data={"message": "No bbox found, step will be skipped"},
                    )

                timings.append(
                    StageTiming(
                        name=f"fallback_grounding_step_{step.index}",
                        duration_ms=grounding_duration,
                        step_index=step.index,
                    )
                )
            else:
                updated_steps.append(step)

        all_steps = updated_steps

        yield self._emit(
            StreamEventType.PHASE_END,
            phase="reasoning_grounding",
            data={
                "steps_count": len(all_steps),
                "steps_with_bbox": sum(1 for s in all_steps if s.has_bbox),
            },
            duration_ms=phase1_duration,
        )

        # === PHASE 3: Smart Evidence Routing ===
        yield self._emit(
            StreamEventType.PHASE_START,
            phase="evidence_extraction",
            data={"description": "Smart routing to OCR or Caption"},
        )

        phase3_start = time.monotonic()

        for step in all_steps:
            if not step.needs_vision or not step.has_bbox:
                continue

            bbox_tuple = tuple(step.bbox)

            if step.need_object_captioning:
                # Object captioning
                cap_start = time.monotonic()
                caption = vlm.caption_region(
                    image, bbox_tuple, step_index=step.index, statement=step.statement
                )
                cap_duration = (time.monotonic() - cap_start) * 1000.0

                evidence = GroundedEvidenceV2(
                    step_index=step.index,
                    statement=step.statement,
                    bbox=bbox_tuple,
                    evidence_type="object",
                    description=caption,
                    ocr_text=None,
                    confidence=0.95,
                )
                all_evidences.append(evidence)

                yield self._emit(
                    StreamEventType.EVIDENCE,
                    phase="evidence_extraction",
                    step_index=step.index,
                    data={
                        "evidence_type": "object",
                        "bbox": list(bbox_tuple),
                        "description": caption,
                    },
                    duration_ms=cap_duration,
                )

                timings.append(
                    StageTiming(
                        name=f"caption_step_{step.index}",
                        duration_ms=cap_duration,
                        step_index=step.index,
                    )
                )

            elif step.need_text_ocr:
                # OCR
                ocr_start = time.monotonic()
                ocr_text = vlm.ocr_region(image, bbox_tuple, step_index=step.index)
                ocr_duration = (time.monotonic() - ocr_start) * 1000.0

                evidence = GroundedEvidenceV2(
                    step_index=step.index,
                    statement=step.statement,
                    bbox=bbox_tuple,
                    evidence_type="text",
                    description=None,
                    ocr_text=ocr_text,
                    confidence=0.95,
                )
                all_evidences.append(evidence)

                yield self._emit(
                    StreamEventType.EVIDENCE,
                    phase="evidence_extraction",
                    step_index=step.index,
                    data={
                        "evidence_type": "text",
                        "bbox": list(bbox_tuple),
                        "ocr_text": ocr_text,
                    },
                    duration_ms=ocr_duration,
                )

                timings.append(
                    StageTiming(
                        name=f"ocr_step_{step.index}",
                        duration_ms=ocr_duration,
                        step_index=step.index,
                    )
                )

        phase3_duration = (time.monotonic() - phase3_start) * 1000.0

        yield self._emit(
            StreamEventType.PHASE_END,
            phase="evidence_extraction",
            data={
                "evidence_count": len(all_evidences),
                "object_count": sum(1 for e in all_evidences if e.evidence_type == "object"),
                "text_count": sum(1 for e in all_evidences if e.evidence_type == "text"),
            },
            duration_ms=phase3_duration,
        )

        # === PHASE 4: Synthesis ===
        yield self._emit(
            StreamEventType.PHASE_START,
            phase="synthesis",
            data={"description": "Generating final answer from evidence"},
        )

        phase4_start = time.monotonic()
        answer, key_evidence, explanation = vlm.synthesize_answer(
            image, question, all_steps, all_evidences
        )
        phase4_duration = (time.monotonic() - phase4_start) * 1000.0

        timings.append(StageTiming(name="answer_synthesis", duration_ms=phase4_duration))

        # Emit answer
        yield self._emit(
            StreamEventType.ANSWER,
            phase="synthesis",
            data={
                "answer": answer,
                "explanation": explanation,
            },
            duration_ms=phase4_duration,
        )

        # Emit key evidence
        for i, ke in enumerate(key_evidence):
            yield self._emit(
                StreamEventType.KEY_EVIDENCE,
                phase="synthesis",
                step_index=i,
                data={
                    "bbox": list(ke.bbox),
                    "description": ke.description,
                    "reasoning": ke.reasoning,
                },
            )

        yield self._emit(
            StreamEventType.PHASE_END,
            phase="synthesis",
            data={"key_evidence_count": len(key_evidence)},
            duration_ms=phase4_duration,
        )

        # === PIPELINE END ===
        total_duration = (time.monotonic() - self._start_time) * 1000.0
        timings.append(StageTiming(name="total_pipeline_v2", duration_ms=total_duration))

        # Get paraphrased question if available
        paraphrased_question = None
        if hasattr(vlm, "_paraphrased_question"):
            paraphrased_question = vlm._paraphrased_question

        # Build final result
        result = PipelineResultV2(
            question=question,
            steps=all_steps,
            evidence=all_evidences,
            answer=answer,
            key_evidence=key_evidence,
            explanation=explanation,
            paraphrased_question=paraphrased_question,
            cot_text=cot_text,
            reasoning_log=vlm.reasoning_log,
            grounding_logs=list(vlm.grounding_logs),
            answer_log=vlm.answer_log,
            timings=timings,
            total_duration_ms=total_duration,
            bbox_from_phase1_count=sum(1 for s in steps if s.has_bbox),
            object_evidence_count=sum(1 for e in all_evidences if e.evidence_type == "object"),
            text_evidence_count=sum(1 for e in all_evidences if e.evidence_type == "text"),
        )

        yield self._emit(
            StreamEventType.PIPELINE_END,
            data={
                "total_duration_ms": total_duration,
                "steps_count": len(all_steps),
                "evidence_count": len(all_evidences),
                "answer": answer,
            },
        )

        return result


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "StreamingPipelineExecutor",
]
