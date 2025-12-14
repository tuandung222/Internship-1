"""
CoRGI Pipeline V2.

V2 pipeline with merged Phase 1+2 and smart evidence routing.
This is a NEW implementation - V1 pipeline.py remains unchanged for backward compatibility.

Key V2 Features:
- Phase 1+2 MERGED: Single Qwen call for reasoning + grounding
- Evidence type discrimination: need_object_captioning vs need_text_ocr
- Smart routing: OCR OR Caption (not both) based on flags
- Optional bbox from Phase 1 (skip grounding if present)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol
import time
import logging

from PIL import Image

from ..utils.image_logger import ImageLogger
from ..utils.output_tracer import OutputTracer

# V2 types
from .types_v2 import ReasoningStepV2, GroundedEvidenceV2

# V1 types for compatibility
from .types import (
    KeyEvidence,
    PromptLog,
    StageTiming,
    stage_timings_to_serializable,
)

logger = logging.getLogger(__name__)


class SupportsVLMClientV2(Protocol):
    """
    Protocol for V2 VLM client.

    Key difference from V1: structured_reasoning_v2 returns steps WITH optional bboxes.
    """

    def structured_reasoning_v2(
        self, image: Image.Image, question: str, max_steps: int
    ) -> tuple[str, List[ReasoningStepV2]]:
        """
        V2 reasoning: Returns (cot_text, steps with optional bboxes and evidence type flags).
        """
        ...

    def extract_bboxes_fallback(
        self,
        image: Image.Image,
        statement: str,
    ) -> List[tuple[float, float, float, float]]:
        """
        Fallback grounding if Phase 1 didn't provide bbox.
        """
        ...

    def ocr_region(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        step_index: Optional[int] = None,
    ) -> str:
        """OCR for text evidence."""
        ...

    def caption_region(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
        step_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> str:
        """Caption for object evidence."""
        ...

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStepV2],
        evidences: List[GroundedEvidenceV2],
    ) -> tuple[str, List[KeyEvidence], Optional[str]]:
        """Synthesize final answer."""
        ...

    def reset_logs(self) -> None: ...

    reasoning_log: Optional[PromptLog]
    grounding_logs: List[PromptLog]
    answer_log: Optional[PromptLog]


@dataclass(frozen=True)
class PipelineResultV2:
    """
    V2 Pipeline Result.

    Enhanced with evidence type tracking.
    """

    question: str
    steps: List[ReasoningStepV2]
    evidence: List[GroundedEvidenceV2]
    answer: str
    key_evidence: List[KeyEvidence] = field(default_factory=list)
    explanation: Optional[str] = None
    paraphrased_question: Optional[str] = None
    cot_text: Optional[str] = None

    # Logs
    reasoning_log: Optional[PromptLog] = None
    grounding_logs: List[PromptLog] = field(default_factory=list)
    answer_log: Optional[PromptLog] = None

    # Timings
    timings: List[StageTiming] = field(default_factory=list)
    total_duration_ms: float = 0.0

    # V2 specific stats
    bbox_from_phase1_count: int = 0  # How many steps had bbox from Phase 1
    object_evidence_count: int = 0  # Object captioning calls
    text_evidence_count: int = 0  # OCR calls

    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "question": self.question,
            "cot_text": self.cot_text,
            "steps": [step.to_dict() for step in self.steps],
            "evidence": [ev.to_dict() for ev in self.evidence],
            "answer": self.answer,
            "key_evidence": [
                {
                    "bbox": list(ke.bbox),
                    "description": ke.description,
                    "reasoning": ke.reasoning,
                }
                for ke in self.key_evidence
            ],
            "explanation": self.explanation,
            "paraphrased_question": self.paraphrased_question,
            "total_duration_ms": self.total_duration_ms,
            "timings": stage_timings_to_serializable(self.timings),
            # V2 stats
            "bbox_from_phase1_count": self.bbox_from_phase1_count,
            "object_evidence_count": self.object_evidence_count,
            "text_evidence_count": self.text_evidence_count,
        }


class CoRGIPipelineV2:
    """
    V2 CoRGI Pipeline.

    Major changes from V1:
    1. Phase 1+2 merged: single call for reasoning + bboxes
    2. Smart routing: object captioning OR OCR (not both)
    3. Evidence type discrimination via flags
    """

    def __init__(
        self,
        vlm_client: SupportsVLMClientV2,
        image_logger=None,
        output_tracer=None,
    ):
        """
        Initialize V2 pipeline.

        Args:
            vlm_client: V2 VLM client with support for merged reasoning+grounding
            image_logger: Optional ImageLogger
            output_tracer: Optional OutputTracer
        """
        if vlm_client is None:
            raise ValueError("A VLM client instance must be provided.")
        self._vlm = vlm_client
        self.image_logger = image_logger
        self.output_tracer = output_tracer

    def run_streaming(
        self,
        image: "Image.Image",
        question: str,
        max_steps: int = 6,
        max_regions: int = 1,
    ):
        """
        Run pipeline with streaming events.

        This is a convenience wrapper around StreamingPipelineExecutor.
        Use this for progressive/real-time display of results.

        Args:
            image: Input image
            question: Question to answer
            max_steps: Max reasoning steps
            max_regions: Max regions per step (fallback grounding)

        Yields:
            StreamEvent objects for each pipeline step

        Example:
            for event in pipeline.run_streaming(image, question):
                if event.type == StreamEventType.STEP:
                    print(f"Step: {event.data['statement']}")
                elif event.type == StreamEventType.ANSWER:
                    print(f"Answer: {event.data['answer']}")
        """
        from .streaming import StreamingPipelineExecutor

        executor = StreamingPipelineExecutor(self)
        return executor.run(image, question, max_steps, max_regions)

    def run(
        self,
        image: Image.Image,
        question: str,
        max_steps: int = 3,
        max_regions: int = 1,  # V2: typically 1 bbox per step from Phase 1
    ) -> PipelineResultV2:
        """
        Run V2 pipeline.

        Flow:
        1. Phase 1+2 MERGED: Reasoning + Grounding (single call)
        2. Phase 3: Smart Evidence Description (routing by type)
        3. Phase 4: Synthesis

        Args:
            image: Input image
            question: Question to answer
            max_steps: Max reasoning steps
            max_regions: Max regions per step (fallback grounding)

        Returns:
            PipelineResultV2
        """
        self._vlm.reset_logs()
        timings: List[StageTiming] = []
        total_start = time.monotonic()

        # Log original input image
        if self.image_logger:
            self.image_logger.log_image(
                image=image,
                stage="original",
                step_index=None,
                image_type="input",
                metadata={
                    "question": question,
                    "max_steps": max_steps,
                    "pipeline_version": "v2",
                },
            )

        # --- PHASE 1+2 MERGED: Structured Reasoning + Grounding ---
        cot_text, steps = self._run_phase1_2_merged(image, question, max_steps, timings)

        # --- PHASE 3: Smart Evidence Description ---
        evidences = self._run_phase3_smart_routing(image, steps, timings)

        # --- PHASE 4: Synthesis ---
        answer, key_evidence, explanation = self._run_phase4_synthesis(
            image, question, steps, evidences, timings
        )

        # Calculate stats
        bbox_from_phase1 = sum(1 for s in steps if s.has_bbox)
        object_count = sum(1 for e in evidences if e.evidence_type == "object")
        text_count = sum(1 for e in evidences if e.evidence_type == "text")

        # Get paraphrased_question if available
        paraphrased_question = None
        if hasattr(self._vlm, "_paraphrased_question"):
            paraphrased_question = self._vlm._paraphrased_question

        total_duration = (time.monotonic() - total_start) * 1000.0
        timings.append(
            StageTiming(name="total_pipeline_v2", duration_ms=total_duration)
        )

        return PipelineResultV2(
            question=question,
            steps=steps,
            evidence=evidences,
            answer=answer,
            key_evidence=key_evidence,
            explanation=explanation,
            paraphrased_question=paraphrased_question,
            cot_text=cot_text,
            reasoning_log=self._vlm.reasoning_log,
            grounding_logs=list(self._vlm.grounding_logs),
            answer_log=self._vlm.answer_log,
            timings=timings,
            total_duration_ms=total_duration,
            bbox_from_phase1_count=bbox_from_phase1,
            object_evidence_count=object_count,
            text_evidence_count=text_count,
        )

    def _run_phase1_2_merged(
        self,
        image: Image.Image,
        question: str,
        max_steps: int,
        timings: List[StageTiming],
    ) -> tuple[str, List[ReasoningStepV2]]:
        """
        Phase 1+2 MERGED: Structured reasoning + grounding in single call.

        Model generates:
        - CoT text
        - JSON with steps containing:
          * statement
          * need_object_captioning / need_text_ocr flags
          * optional bbox [x1,y1,x2,y2]

        Returns:
            (cot_text, list of ReasoningStepV2)
        """
        logger.info("[V2] Phase 1+2 MERGED: Structured reasoning + grounding")

        stage = None
        if self.output_tracer:
            stage = self.output_tracer.start_stage(
                "reasoning_grounding_merged_v2", "reasoning"
            )

        start = time.monotonic()
        cot_text, steps = self._vlm.structured_reasoning_v2(
            image=image, question=question, max_steps=max_steps
        )
        duration = (time.monotonic() - start) * 1000.0

        timings.append(StageTiming(name="phase1_2_merged", duration_ms=duration))

        if self.output_tracer and stage:
            self.output_tracer.end_stage(stage)

        # Log stats
        bbox_provided = sum(1 for s in steps if s.has_bbox)
        logger.info(
            f"[V2] Phase 1+2 result: {len(steps)} steps, "
            f"{bbox_provided} with bbox from model"
        )

        # Fallback grounding for steps missing bbox
        steps = self._fallback_grounding_if_needed(image, steps, timings)

        return cot_text, steps

    def _fallback_grounding_if_needed(
        self,
        image: Image.Image,
        steps: List[ReasoningStepV2],
        timings: List[StageTiming],
    ) -> List[ReasoningStepV2]:
        """
        Fallback grounding: If step needs vision but has no bbox, call grounding.

        This handles cases where model didn't provide bbox in Phase 1.
        Creates new ReasoningStepV2 instances with bbox added.
        """
        missing_bbox = [s for s in steps if s.needs_vision and not s.has_bbox]

        if not missing_bbox:
            return steps  # All steps have bbox, no fallback needed

        logger.info(
            f"[V2] Fallback grounding for {len(missing_bbox)} steps missing bbox"
        )

        # Create new steps list with bboxes filled in
        updated_steps = []

        for step in steps:
            if step.needs_vision and not step.has_bbox:
                # Need fallback grounding
                logger.info(
                    f"[V2] Step {step.index}: Fallback grounding for '{step.statement[:50]}...'"
                )

                start = time.monotonic()
                bboxes = self._vlm.extract_bboxes_fallback(image, step.statement)
                duration = (time.monotonic() - start) * 1000.0

                timings.append(
                    StageTiming(
                        name=f"fallback_grounding_step_{step.index}",
                        duration_ms=duration,
                        step_index=step.index,
                    )
                )

                if bboxes:
                    # Create new step with bbox
                    updated_step = ReasoningStepV2(
                        index=step.index,
                        statement=step.statement,
                        need_object_captioning=step.need_object_captioning,
                        need_text_ocr=step.need_text_ocr,
                        bbox=list(bboxes[0]),  # Use first bbox
                        reason=step.reason,
                    )
                    updated_steps.append(updated_step)
                    logger.info(
                        f"[V2] Step {step.index}: Fallback grounding found bbox {bboxes[0]}"
                    )
                else:
                    # No bbox found, keep original step (will be skipped in Phase 3)
                    updated_steps.append(step)
                    logger.warning(
                        f"[V2] Step {step.index}: Fallback grounding found no bbox, "
                        "step will be skipped in evidence description"
                    )
            else:
                # Step already has bbox or doesn't need vision
                updated_steps.append(step)

        return updated_steps

    def _run_phase3_smart_routing(
        self,
        image: Image.Image,
        steps: List[ReasoningStepV2],
        timings: List[StageTiming],
    ) -> List[GroundedEvidenceV2]:
        """
        Phase 3: Smart Evidence Description with routing.

        Routes each step to appropriate module based on evidence type:
        - need_object_captioning=True → SmolVLM2 caption
        - need_text_ocr=True → Florence-2 OCR
        - Both False → Skip

        Returns:
            List of GroundedEvidenceV2
        """
        logger.info("[V2] Phase 3: Smart evidence routing")

        evidences: List[GroundedEvidenceV2] = []

        for step in steps:
            if not step.needs_vision or not step.has_bbox:
                continue

            bbox_tuple = tuple(step.bbox)

            # Route based on evidence type
            if step.need_object_captioning:
                # Object captioning path
                logger.info(f"[V2] Step {step.index}: Routing to CAPTIONING (object)")
                start = time.monotonic()

                caption = self._vlm.caption_region(
                    image, bbox_tuple, step_index=step.index, statement=step.statement
                )

                duration = (time.monotonic() - start) * 1000.0
                timings.append(
                    StageTiming(
                        name=f"caption_step_{step.index}",
                        duration_ms=duration,
                        step_index=step.index,
                    )
                )

                evidence = GroundedEvidenceV2(
                    step_index=step.index,
                    statement=step.statement,
                    bbox=bbox_tuple,
                    evidence_type="object",
                    description=caption,
                    ocr_text=None,
                    confidence=0.95,
                )
                evidences.append(evidence)

            elif step.need_text_ocr:
                # Text OCR path
                logger.info(f"[V2] Step {step.index}: Routing to OCR (text)")
                start = time.monotonic()

                ocr_text = self._vlm.ocr_region(
                    image, bbox_tuple, step_index=step.index
                )

                duration = (time.monotonic() - start) * 1000.0
                timings.append(
                    StageTiming(
                        name=f"ocr_step_{step.index}",
                        duration_ms=duration,
                        step_index=step.index,
                    )
                )

                evidence = GroundedEvidenceV2(
                    step_index=step.index,
                    statement=step.statement,
                    bbox=bbox_tuple,
                    evidence_type="text",
                    description=None,
                    ocr_text=ocr_text,
                    confidence=0.95,
                )
                evidences.append(evidence)

            else:
                # No visual evidence needed (shouldn't happen if needs_vision=True)
                logger.warning(
                    f"[V2] Step {step.index}: needs_vision=True but no flags set, skipping"
                )

        logger.info(
            f"[V2] Phase 3 complete: {len(evidences)} evidence items "
            f"({sum(1 for e in evidences if e.evidence_type=='object')} object, "
            f"{sum(1 for e in evidences if e.evidence_type=='text')} text)"
        )

        return evidences

    def _run_phase4_synthesis(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStepV2],
        evidences: List[GroundedEvidenceV2],
        timings: List[StageTiming],
    ) -> tuple[str, List[KeyEvidence], Optional[str]]:
        """
        Phase 4: Answer synthesis.

        Same as V1 but uses V2 types.
        """
        logger.info("[V2] Phase 4: Answer synthesis")

        stage = None
        if self.output_tracer:
            stage = self.output_tracer.start_stage("answer_synthesis_v2", "synthesis")

        start = time.monotonic()
        answer, key_evidence, explanation = self._vlm.synthesize_answer(
            image=image, question=question, steps=steps, evidences=evidences
        )
        duration = (time.monotonic() - start) * 1000.0

        timings.append(StageTiming(name="answer_synthesis", duration_ms=duration))

        if self.output_tracer and stage:
            self.output_tracer.end_stage(stage)

        logger.info(f"[V2] Synthesis complete: {len(key_evidence)} key evidence items")

        return answer, key_evidence, explanation


__all__ = [
    "CoRGIPipelineV2",
    "PipelineResultV2",
    "SupportsVLMClientV2",
]
