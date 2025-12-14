from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import logging
import time

from PIL import Image

from ..utils.image_logger import ImageLogger
from ..utils.output_tracer import OutputTracer

from .types import (
    GroundedEvidence,
    KeyEvidence,
    PromptLog,
    ReasoningStep,
    StageTiming,
    evidences_to_serializable,
    prompt_logs_to_serializable,
    stage_timings_to_serializable,
    steps_to_serializable,
)

logger = logging.getLogger(__name__)

class SupportsVLMClient(Protocol):
    """
    Protocol describing the methods required from a VLM client.

    This is a generic protocol that any VLM client (Qwen, Florence, Composite)
    must implement to be used in the CoRGI pipeline.
    """

    def structured_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> List[ReasoningStep]: ...

    def extract_step_evidence(
        self,
        image: Image.Image,
        question: str,
        step: ReasoningStep,
        max_regions: int,
    ) -> List[GroundedEvidence]: ...

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        evidences: List[GroundedEvidence],
    ) -> tuple[str, List[KeyEvidence], Optional[str]]: ...

    def reset_logs(self) -> None: ...

    reasoning_log: Optional[PromptLog]
    grounding_logs: List[PromptLog]
    answer_log: Optional[PromptLog]


# Backward compatibility alias
SupportsQwenClient = SupportsVLMClient


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated output of the CoRGI pipeline."""

    question: str
    steps: List[ReasoningStep]
    evidence: List[GroundedEvidence]
    answer: str
    key_evidence: List[KeyEvidence] = field(default_factory=list)
    explanation: Optional[str] = None
    paraphrased_question: Optional[str] = (
        None  # Paraphrased question from synthesis stage
    )
    cot_text: Optional[str] = None  # Full Chain of Thought text from model
    reasoning_log: Optional[PromptLog] = None
    grounding_logs: List[PromptLog] = field(default_factory=list)
    answer_log: Optional[PromptLog] = None
    timings: List[StageTiming] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def to_json(self) -> dict:
        payload = {
            "question": self.question,
            "cot_text": self.cot_text,  # Full Chain of Thought text
            "steps": steps_to_serializable(self.steps),
            "evidence": evidences_to_serializable(self.evidence),
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
        }
        reasoning_entries = (
            prompt_logs_to_serializable([self.reasoning_log])
            if self.reasoning_log
            else []
        )
        if reasoning_entries:
            payload["reasoning_log"] = reasoning_entries[0]

        payload["grounding_logs"] = prompt_logs_to_serializable(self.grounding_logs)
        payload["timings"] = stage_timings_to_serializable(self.timings)

        answer_entries = (
            prompt_logs_to_serializable([self.answer_log]) if self.answer_log else []
        )
        if answer_entries:
            payload["answer_log"] = answer_entries[0]

        return payload


class CoRGIPipeline:
    """
    Orchestrates the CoRGI reasoning pipeline using a VLM client.

    The VLM client can be any implementation that satisfies the SupportsVLMClient
    protocol, including:
    - Qwen3VLThinkingClient
    - Qwen3VLInstructClient
    - CompositeVLMClient (mix and match different models)
    """

    def __init__(
        self,
        vlm_client: SupportsVLMClient,
        image_logger=None,
        output_tracer=None,
    ):
        """
        Initialize CoRGI pipeline with a VLM client.

        Args:
            vlm_client: VLM client implementing SupportsVLMClient protocol
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing

        Raises:
            ValueError: If vlm_client is None
        """
        if vlm_client is None:
            raise ValueError("A VLM client instance must be provided.")
        self._vlm = vlm_client
        self.image_logger = image_logger
        self.output_tracer = output_tracer

    def run(
        self,
        image: Image.Image,
        question: str,
        max_steps: int = 3,
        max_regions: int = 3,
    ) -> PipelineResult:
        self._vlm.reset_logs()
        if self.output_tracer is not None and hasattr(self.output_tracer, "reset"):
            try:
                self.output_tracer.reset()
            except Exception:
                pass
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
                    "max_regions": max_regions,
                },
            )

        # Start reasoning stage trace
        reasoning_stage = None
        if self.output_tracer:
            reasoning_stage = self.output_tracer.start_stage(
                "structured_reasoning", "reasoning"
            )

        reasoning_start = time.monotonic()
        steps = self._vlm.structured_reasoning(
            image=image, question=question, max_steps=max_steps
        )
        reasoning_duration = (time.monotonic() - reasoning_start) * 1000.0
        timings.append(
            StageTiming(name="structured_reasoning", duration_ms=reasoning_duration)
        )

        # Get CoT text if available (from Instruct models)
        cot_text = None
        if hasattr(self._vlm, "_cot_text"):
            cot_text = self._vlm._cot_text

        if self.output_tracer and reasoning_stage:
            self.output_tracer.end_stage(reasoning_stage)

        # Batch grounding: extract evidence for all steps in a single call
        vision_steps = [step for step in steps if step.needs_vision]
        evidences: List[GroundedEvidence] = []

        if vision_steps:
            # Check if VLM client supports batch grounding
            if hasattr(self._vlm, "extract_all_steps_evidence"):
                # Use batch grounding (single inference for all steps)
                stage_name = "roi_batch_all_steps"
                grounding_stage = None
                if self.output_tracer:
                    grounding_stage = self.output_tracer.start_stage(
                        stage_name, "grounding"
                    )

                grounding_start = time.monotonic()
                all_evidences = self._vlm.extract_all_steps_evidence(
                    image=image,
                    question=question,
                    steps=vision_steps,
                    max_regions=max_regions,
                )
                grounding_duration = (time.monotonic() - grounding_start) * 1000.0
                timings.append(
                    StageTiming(name=stage_name, duration_ms=grounding_duration)
                )

                if self.output_tracer and grounding_stage:
                    self.output_tracer.end_stage(grounding_stage)

                # Limit to max_regions per step
                for step in vision_steps:
                    step_evs = [
                        ev for ev in all_evidences if ev.step_index == step.index
                    ]
                    evidences.extend(step_evs[:max_regions])
            else:
                # Fallback to per-step extraction (backward compatibility)
                for step in vision_steps:
                    stage_name = f"roi_step_{step.index}"

                    # Start grounding stage trace
                    grounding_stage = None
                    if self.output_tracer:
                        grounding_stage = self.output_tracer.start_stage(
                            stage_name, "grounding", step.index
                        )

                    grounding_start = time.monotonic()
                    step_evs = self._vlm.extract_step_evidence(
                        image=image,
                        question=question,
                        step=step,
                        max_regions=max_regions,
                    )
                    grounding_duration = (time.monotonic() - grounding_start) * 1000.0
                    timings.append(
                        StageTiming(
                            name=stage_name,
                            duration_ms=grounding_duration,
                            step_index=step.index,
                        )
                    )

                    if self.output_tracer and grounding_stage:
                        self.output_tracer.end_stage(grounding_stage)

                    if not step_evs:
                        continue
                    evidences.extend(step_evs[:max_regions])

        # Start synthesis stage trace
        synthesis_stage = None
        if self.output_tracer:
            synthesis_stage = self.output_tracer.start_stage(
                "answer_synthesis", "synthesis"
            )

        answer_start = time.monotonic()
        logger.info("[Pipeline] Stage 4: answer_synthesis starting")
        answer, key_evidence, explanation = self._vlm.synthesize_answer(
            image=image, question=question, steps=steps, evidences=evidences
        )
        answer_duration = (time.monotonic() - answer_start) * 1000.0
        logger.info(
            "[Pipeline] Stage 4: answer_synthesis completed in %.2fs",
            answer_duration / 1000.0,
        )
        timings.append(
            StageTiming(name="answer_synthesis", duration_ms=answer_duration)
        )

        # Get paraphrased_question if available
        paraphrased_question = None
        if hasattr(self._vlm, "_paraphrased_question"):
            paraphrased_question = self._vlm._paraphrased_question

        if self.output_tracer and synthesis_stage:
            self.output_tracer.end_stage(synthesis_stage)

        total_duration = (time.monotonic() - total_start) * 1000.0
        timings.append(StageTiming(name="total_pipeline", duration_ms=total_duration))
        return PipelineResult(
            question=question,
            steps=steps,
            evidence=evidences,
            answer=answer,
            key_evidence=key_evidence,
            explanation=explanation,
            paraphrased_question=paraphrased_question,
            cot_text=cot_text,  # Full Chain of Thought text
            reasoning_log=self._vlm.reasoning_log,
            grounding_logs=list(self._vlm.grounding_logs),
            answer_log=self._vlm.answer_log,
            timings=timings,
            total_duration_ms=total_duration,
        )


__all__ = [
    "CoRGIPipeline",
    "PipelineResult",
    "SupportsVLMClient",
    "SupportsQwenClient",  # Backward compatibility
]
