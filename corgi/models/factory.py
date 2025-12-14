"""
VLM Client Factory for CoRGi Pipeline.

Provides factory methods to create composite VLM clients from configuration,
allowing flexible composition of different models for each pipeline stage.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Callable
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from ..core.config import CoRGiConfig, ModelConfig
from .registry import ModelRegistry
from ..core.types import GroundedEvidence, KeyEvidence, PromptLog, ReasoningStep, BBox
from ..utils.coordinate_utils import non_maximum_suppression
from .qwen.qwen_instruct_client import Qwen3VLInstructClient
from .qwen.qwen_thinking_client import Qwen3VLThinkingClient
from .florence.florence_grounding_client import Florence2GroundingClient
from .florence.florence_captioning_client import Florence2CaptioningClient
from .qwen.qwen_grounding_adapter import QwenGroundingAdapter
from .qwen.qwen_captioning_adapter import QwenCaptioningAdapter

# Import new clients to trigger registration
from .paddle.paddleocr_client import PaddleOCRClient  # noqa: F401
from .fastvlm.fastvlm_client import FastVLMCaptioningClient  # noqa: F401
from .composite.composite_captioning_client import CompositeCaptioningClient

# Import Vintern client (deprecated, kept for backward compatibility)
from .vintern.vintern_client import VinternCaptioningClient  # noqa: F401

logger = logging.getLogger(__name__)


# Register models with the registry
ModelRegistry.register_reasoning("qwen_instruct")(Qwen3VLInstructClient)
ModelRegistry.register_reasoning("qwen_thinking")(Qwen3VLThinkingClient)

# For synthesis, Qwen models are used directly
ModelRegistry.register_synthesis("qwen_instruct")(Qwen3VLInstructClient)
ModelRegistry.register_synthesis("qwen_thinking")(Qwen3VLThinkingClient)

# Register Florence-2 models
ModelRegistry.register_grounding("florence2")(Florence2GroundingClient)
ModelRegistry.register_captioning("florence2")(Florence2CaptioningClient)

# Register Composite model
ModelRegistry.register_captioning("composite")(CompositeCaptioningClient)


class CompositeVLMClient:
    """
    Composite VLM client that delegates to specialized models.

    This client implements the full pipeline protocol by composing different
    models for reasoning, grounding, captioning, and synthesis stages.
    """

    def __init__(
        self,
        reasoning,
        grounding,
        captioning,
        synthesis,
        config: CoRGiConfig,
        image_logger=None,
        output_tracer=None,
    ):
        """
        Initialize composite client.

        Args:
            reasoning: Reasoning model instance
            grounding: Grounding model instance
            captioning: Captioning model instance
            synthesis: Synthesis model instance
            config: Full pipeline configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
        """
        self.reasoning = reasoning
        self.grounding = grounding
        self.captioning = captioning
        self.synthesis = synthesis
        self.config = config
        self.image_logger = image_logger
        self.output_tracer = output_tracer
        self.batch_evidence_enabled = True
        self.reset_logs()

    def set_batch_evidence_enabled(self, enabled: bool) -> None:
        """Enable/disable batch evidence extraction (OCR + captioning)."""
        self.batch_evidence_enabled = bool(enabled)

    def reset_logs(self) -> None:
        """Reset prompt/response logs."""
        self._reasoning_log: Optional[PromptLog] = None
        self._cot_text: Optional[str] = None  # Chain of Thought text
        self._grounding_logs: List[PromptLog] = []
        self._answer_log: Optional[PromptLog] = None
        self._paraphrased_question: Optional[str] = None

    @property
    def reasoning_log(self) -> Optional[PromptLog]:
        return self._reasoning_log

    @property
    def grounding_logs(self) -> List[PromptLog]:
        return list(self._grounding_logs)

    @property
    def answer_log(self) -> Optional[PromptLog]:
        return self._answer_log

    def structured_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> List[ReasoningStep]:
        """
        Generate structured reasoning steps.

        Delegates to reasoning model. Handles both Instruct and Thinking models.

        Args:
            image: Input image
            question: Question to answer
            max_steps: Maximum number of reasoning steps

        Returns:
            List of ReasoningStep objects
        """
        # Check if reasoning model has generate_reasoning (Instruct model)
        if hasattr(self.reasoning, "generate_reasoning"):
            cot_text, steps = self.reasoning.generate_reasoning(
                image, question, max_steps
            )
            # Prefer the underlying model's full prompt/response log if available.
            model_log = getattr(self.reasoning, "reasoning_log", None)
            if model_log is not None:
                self._reasoning_log = model_log
            else:
                # Fallback: at least store question + CoT text.
                self._reasoning_log = PromptLog(
                    prompt=f"Question: {question}", response=cot_text, stage="reasoning"
                )
            self._cot_text = cot_text  # Store for pipeline access
            return steps
        else:
            # Thinking model - uses structured_reasoning directly
            steps = self.reasoning.structured_reasoning(image, question, max_steps)
            # Copy log from reasoning model
            if hasattr(self.reasoning, "reasoning_log"):
                self._reasoning_log = self.reasoning.reasoning_log
            self._cot_text = None  # Thinking models don't have separate CoT text
            return steps

    def structured_reasoning_v2(
        self, image: Image.Image, question: str, max_steps: int
    ) -> Tuple[str, List]:
        """
        Generate V2 structured reasoning steps (with integrated grounding).

        Delegates to reasoning model's V2 method if available, otherwise falls back to V1.

        Args:
            image: Input image
            question: Question to answer
            max_steps: Maximum number of reasoning steps

        Returns:
            Tuple of (cot_text, List[ReasoningStepV2])
        """
        # Check if reasoning model has V2 method
        if hasattr(self.reasoning, "structured_reasoning_v2"):
            cot_text, steps = self.reasoning.structured_reasoning_v2(
                image, question, max_steps
            )
            # Prefer the underlying model's full prompt/response log if available.
            model_log = getattr(self.reasoning, "reasoning_log", None)
            if model_log is not None:
                self._reasoning_log = model_log
            else:
                self._reasoning_log = PromptLog(
                    prompt=f"Question: {question}", response=cot_text, stage="reasoning_v2"
                )
            self._cot_text = cot_text
            return cot_text, steps
        else:
            # Fallback to V1 and convert
            logger.warning(
                "Reasoning model does not support V2, falling back to V1 → V2 conversion"
            )
            steps_v1 = self.structured_reasoning(image, question, max_steps)
            
            # Convert V1 ReasoningStep to V2 ReasoningStepV2
            from ..core.types_v2 import ReasoningStepV2
            steps_v2 = []
            for step in steps_v1:
                step_v2 = ReasoningStepV2(
                    index=step.index,
                    statement=step.statement,
                    need_object_captioning=step.needs_vision and not step.need_ocr,
                    need_text_ocr=step.need_ocr,
                    bbox=None,  # V1 doesn't have bboxes
                    reason=step.reason,
                )
                steps_v2.append(step_v2)
            
            return self._cot_text or "", steps_v2

    def extract_bboxes_fallback(
        self,
        image: Image.Image,
        statement: str,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Fallback grounding if Phase 1 didn't provide bbox.
        
        Args:
            image: Input image
            statement: Statement to ground
            
        Returns:
            List of bboxes
        """
        if hasattr(self.grounding, "extract_regions"):
            bboxes = self.grounding.extract_regions(image, statement, max_regions=1)
            return bboxes if isinstance(bboxes, list) else []
        else:
            logger.warning("Grounding model doesn't support extract_regions")
            return []

    def ocr_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
    ) -> str:
        """OCR for text evidence (V2 pipeline)."""
        if hasattr(self.captioning, "ocr_region"):
            return self.captioning.ocr_region(image, bbox, step_index)
        else:
            logger.warning("Captioning model doesn't support ocr_region")
            return ""

    def caption_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        step_index: Optional[int] = None,
        statement: Optional[str] = None,
    ) -> str:
        """Caption for object evidence (V2 pipeline)."""
        if hasattr(self.captioning, "caption_region"):
            # Check if method accepts statement parameter
            import inspect
            sig = inspect.signature(self.captioning.caption_region)
            if "statement" in sig.parameters:
                return self.captioning.caption_region(
                    image, bbox, step_index, statement=statement
                )
            else:
                return self.captioning.caption_region(image, bbox, step_index)
        else:
            logger.warning("Captioning model doesn't support caption_region")
            return ""

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List,
        evidences: List,
    ) -> Tuple[str, List[KeyEvidence], Optional[str]]:
        """
        Synthesize final answer from V2 evidences.
        
        Args:
            image: Input image
            question: Original question
            steps: List of ReasoningStepV2
            evidences: List of GroundedEvidenceV2
            
        Returns:
            Tuple of (answer, key_evidences, explanation)
        """
        # Delegate to synthesis model
        if hasattr(self.synthesis, "synthesize_answer"):
            answer, key_evidence, explanation = self.synthesis.synthesize_answer(
                image, question, steps, evidences
            )

            # Copy prompt/response logs from synthesis model for UI/debugging.
            if hasattr(self.synthesis, "answer_log"):
                self._answer_log = self.synthesis.answer_log

            # Copy paraphrased_question if available.
            if hasattr(self.synthesis, "_paraphrased_question"):
                self._paraphrased_question = self.synthesis._paraphrased_question
            else:
                self._paraphrased_question = None

            return answer, key_evidence, explanation
        else:
            # Fallback: use basic synthesis
            logger.warning("Using basic synthesis (no dedicated synthesis model)")
            answer = f"Based on {len(evidences)} evidence regions, analysis is in progress."
            return answer, [], None

    def _run_ocr_and_caption_parallel(
        self,
        image: Image.Image,
        bbox: BBox,
        step_index: int,
        bbox_index: int,
        step: Optional[ReasoningStep] = None,
    ) -> Tuple[str, str]:
        """
        Run OCR and Captioning in parallel for a single region.

        Args:
            image: Input image
            bbox: Bounding box coordinates
            step_index: Index of the reasoning step
            bbox_index: Index of the bounding box within the step
            step: Optional reasoning step (for statement in VQA and need_ocr flag)

        Returns:
            Tuple of (ocr_text, caption)
        """
        ocr_text = ""
        caption = ""

        # Check if OCR is needed based on step.need_ocr flag
        need_ocr = step.need_ocr if step else False

        # Get statement from step if available (for Vintern VQA)
        statement = step.statement if step else None

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit OCR task only if need_ocr is True
            ocr_future = None
            if need_ocr and hasattr(self.captioning, "ocr_region"):
                ocr_future = executor.submit(
                    self.captioning.ocr_region, image, bbox, step_index, bbox_index
                )
            elif not need_ocr:
                logger.debug(
                    f"Skipping OCR for step {step_index}, bbox {bbox_index}: need_ocr=False"
                )

            # Submit Captioning task (always run captioning)
            caption_future = None
            if hasattr(self.captioning, "caption_region"):
                # Check if method accepts statement parameter (e.g., Vintern for VQA)
                import inspect

                sig = inspect.signature(self.captioning.caption_region)
                if "statement" in sig.parameters:
                    caption_future = executor.submit(
                        self.captioning.caption_region,
                        image,
                        bbox,
                        step_index,
                        bbox_index,
                        statement=statement,
                    )
                else:
                    caption_future = executor.submit(
                        self.captioning.caption_region,
                        image,
                        bbox,
                        step_index,
                        bbox_index,
                    )

            # Wait for results
            if ocr_future:
                try:
                    ocr_text = ocr_future.result() or ""
                except Exception as e:
                    logger.warning(f"OCR failed for region {bbox}: {e}")
                    ocr_text = ""
            # If OCR was skipped, ocr_text remains empty string

            if caption_future:
                try:
                    caption = caption_future.result() or ""
                except Exception as e:
                    logger.warning(f"Captioning failed for region {bbox}: {e}")
                    caption = ""

        return ocr_text, caption

    def _run_ocr_and_caption_batch_parallel(
        self,
        image: Image.Image,
        bboxes: List[BBox],
        step_index: int,
        step: Optional[ReasoningStep] = None,
    ) -> List[Tuple[str, str]]:
        """
        Run OCR and Captioning batches in parallel.

        Prefers unified batch method if available (more efficient for Florence-2).
        Falls back to separate batch calls if unified method not available.

        Args:
            image: Input image
            bboxes: List of bounding box coordinates
            step_index: Index of the reasoning step
            step: Optional reasoning step (for statement in VQA and need_ocr flag)

        Returns:
            List of tuples: [(ocr_text, caption), ...]
        """
        # Check if OCR is needed based on step.need_ocr flag
        need_ocr = step.need_ocr if step else False

        # Check if captioning client has unified batch method (e.g., Florence-2, Vintern)
        if hasattr(self.captioning, "ocr_and_caption_regions_batch"):
            try:
                logger.info(f"Using unified batch method for OCR + Captioning")
                # Check if method accepts statement parameter (e.g., Vintern for VQA)
                import inspect

                sig = inspect.signature(self.captioning.ocr_and_caption_regions_batch)
                if "statement" in sig.parameters and step is not None:
                    # Pass statement for VQA component (Vintern)
                    ocr_texts, captions = self.captioning.ocr_and_caption_regions_batch(
                        image, bboxes, step_index, statement=step.statement
                    )
                else:
                    ocr_texts, captions = self.captioning.ocr_and_caption_regions_batch(
                        image, bboxes, step_index
                    )
                # Ensure lengths match
                if len(ocr_texts) != len(bboxes) or len(captions) != len(bboxes):
                    logger.warning(
                        f"Unified batch returned mismatched lengths: OCR={len(ocr_texts)}, Captions={len(captions)}, Expected={len(bboxes)}"
                    )
                    ocr_texts = (ocr_texts + [""] * len(bboxes))[: len(bboxes)]
                    captions = (captions + [""] * len(bboxes))[: len(bboxes)]

                # If need_ocr is False, clear all OCR texts
                if not need_ocr:
                    logger.debug(
                        f"Skipping OCR for step {step_index} (batch): need_ocr=False"
                    )
                    ocr_texts = [""] * len(bboxes)

                return list(zip(ocr_texts, captions))
            except Exception as e:
                logger.warning(
                    f"Unified batch method failed, falling back to separate batches: {e}"
                )
                # Fall through to separate batch calls

        # Fallback to separate batch calls
        ocr_texts = [""] * len(bboxes)
        captions = [""] * len(bboxes)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit OCR batch task only if need_ocr is True
            ocr_future = None
            if need_ocr and hasattr(self.captioning, "ocr_regions_batch"):
                ocr_future = executor.submit(
                    self.captioning.ocr_regions_batch, image, bboxes, step_index
                )
            elif not need_ocr:
                logger.debug(
                    f"Skipping OCR batch for step {step_index}: need_ocr=False"
                )

            # Submit Captioning batch task (always run captioning)
            caption_future = None
            if hasattr(self.captioning, "caption_regions_batch"):
                caption_future = executor.submit(
                    self.captioning.caption_regions_batch, image, bboxes, step_index
                )

            # Wait for results
            if ocr_future:
                try:
                    ocr_texts = ocr_future.result() or [""] * len(bboxes)
                    if len(ocr_texts) != len(bboxes):
                        logger.warning(
                            f"OCR batch returned {len(ocr_texts)} results, expected {len(bboxes)}"
                        )
                        ocr_texts = (ocr_texts + [""] * len(bboxes))[: len(bboxes)]
                except Exception as e:
                    logger.warning(f"OCR batch failed: {e}")
                    ocr_texts = [""] * len(bboxes)
            # If OCR was skipped, ocr_texts remains empty strings

            if caption_future:
                try:
                    captions = caption_future.result() or [""] * len(bboxes)
                    if len(captions) != len(bboxes):
                        logger.warning(
                            f"Captioning batch returned {len(captions)} results, expected {len(bboxes)}"
                        )
                        captions = (captions + [""] * len(bboxes))[: len(bboxes)]
                except Exception as e:
                    logger.warning(f"Captioning batch failed: {e}")
                    captions = [""] * len(bboxes)

        return list(zip(ocr_texts, captions))

    def extract_all_steps_evidence(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        max_regions: int,
    ) -> List[GroundedEvidence]:
        """
        Extract visual evidence for all reasoning steps in a single batch grounding call.

        This method collects all statements from all steps and calls grounding once,
        then distributes the results back to corresponding steps.

        Args:
            image: Input image
            question: Original question
            steps: List of reasoning steps to verify
            max_regions: Maximum regions to extract per step

        Returns:
            List of GroundedEvidence objects for all steps
        """
        try:
            # Filter steps that need vision
            vision_steps = [step for step in steps if step.needs_vision]
            if not vision_steps:
                logger.info("No steps require vision, skipping grounding")
                return []

            # Get NMS config from grounding config
            nms_enabled = getattr(self.config.grounding, "nms_enabled", True)
            nms_iou_threshold = getattr(self.config.grounding, "nms_iou_threshold", 0.5)

            # Collect all statements with step indices for batch grounding
            statements = [(step.index, step.statement) for step in vision_steps]

            # Use grounding model to get bboxes in batch mode (single inference)
            bboxes_by_step: Dict[int, List[Tuple[float, float, float, float]]] = {}
            if hasattr(self.grounding, "extract_regions"):
                # Check if extract_regions supports batch mode (statements parameter)
                import inspect

                sig = inspect.signature(self.grounding.extract_regions)
                params = sig.parameters

                if "statements" in params:
                    # Batch mode: single inference for all steps
                    logger.info(
                        f"Using batch grounding for {len(statements)} steps (single inference)"
                    )
                    result = self.grounding.extract_regions(
                        image=image,
                        statements=statements,
                        max_regions=max_regions,
                    )
                    # Result is a dict mapping step_index -> List[bboxes]
                    if isinstance(result, dict):
                        bboxes_by_step = result
                    else:
                        # Fallback: treat as single step result
                        logger.warning(
                            "Batch grounding returned non-dict, treating as single step"
                        )
                        if vision_steps:
                            bboxes_by_step = {
                                vision_steps[0].index: (
                                    result if isinstance(result, list) else []
                                )
                            }
                else:
                    # Fallback to per-step extraction (backward compatibility)
                    logger.info(
                        f"Grounding model doesn't support batch mode, using per-step extraction"
                    )
                    for step in vision_steps:
                        kwargs = {}
                        if "nms_enabled" in params:
                            kwargs["nms_enabled"] = nms_enabled
                        if "nms_iou_threshold" in params:
                            kwargs["nms_iou_threshold"] = nms_iou_threshold
                        if "step_index" in params:
                            kwargs["step_index"] = step.index

                        bboxes = self.grounding.extract_regions(
                            image, step.statement, max_regions, **kwargs
                        )
                        bboxes_by_step[step.index] = (
                            bboxes if isinstance(bboxes, list) else []
                        )
            else:
                logger.warning("Grounding model doesn't have extract_regions method")
                return []

            # Process bboxes for each step
            all_evidences: List[GroundedEvidence] = []

            # Fast path: batch captioning across all steps (single captioning generate call).
            # This is especially useful when max_regions is small (e.g., 1) but there are
            # many steps, which would otherwise trigger many small sequential caption calls.
            try:
                if self.batch_evidence_enabled and hasattr(
                    self.captioning, "caption_regions_batch"
                ):
                    flat_regions: List[Tuple[ReasoningStep, BBox]] = []
                    for step in vision_steps:
                        step_bboxes = bboxes_by_step.get(step.index, [])
                        for bbox in step_bboxes:
                            flat_regions.append((step, bbox))

                    if len(flat_regions) > 1:
                        flat_bboxes = [bbox for _, bbox in flat_regions]

                        # Only run OCR for steps that request it.
                        ocr_indices = [
                            idx
                            for idx, (step, _) in enumerate(flat_regions)
                            if getattr(step, "need_ocr", False)
                        ]

                        captions: List[str] = [""] * len(flat_bboxes)
                        ocr_texts: List[str] = [""] * len(flat_bboxes)

                        with ThreadPoolExecutor(max_workers=2) as executor:
                            caption_future = executor.submit(
                                self.captioning.caption_regions_batch,
                                image,
                                flat_bboxes,
                                0,  # batch across steps
                            )
                            ocr_future = None
                            if (
                                ocr_indices
                                and hasattr(self.captioning, "ocr_regions_batch")
                            ):
                                ocr_bboxes = [flat_bboxes[i] for i in ocr_indices]
                                ocr_future = executor.submit(
                                    self.captioning.ocr_regions_batch,
                                    image,
                                    ocr_bboxes,
                                    0,  # batch across steps
                                )

                            captions = caption_future.result() or [""] * len(flat_bboxes)
                            if len(captions) != len(flat_bboxes):
                                logger.warning(
                                    "Captioning batch returned %d results, expected %d",
                                    len(captions),
                                    len(flat_bboxes),
                                )
                                captions = (captions + [""] * len(flat_bboxes))[: len(flat_bboxes)]

                            if ocr_future:
                                ocr_results = ocr_future.result() or [""] * len(ocr_indices)
                                if len(ocr_results) != len(ocr_indices):
                                    logger.warning(
                                        "OCR batch returned %d results, expected %d",
                                        len(ocr_results),
                                        len(ocr_indices),
                                    )
                                    ocr_results = (ocr_results + [""] * len(ocr_indices))[
                                        : len(ocr_indices)
                                    ]
                                for rel_idx, flat_idx in enumerate(ocr_indices):
                                    ocr_texts[flat_idx] = ocr_results[rel_idx] or ""

                        for flat_idx, ((step, bbox), caption) in enumerate(
                            zip(flat_regions, captions)
                        ):
                            all_evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description=caption or "",
                                    ocr_text=ocr_texts[flat_idx] or "",
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )

                        # Per-step logs for consistency
                        for step in vision_steps:
                            step_evidences = [
                                ev for ev in all_evidences if ev.step_index == step.index
                            ]
                            summary = (
                                f"Extracted {len(step_evidences)} evidence regions for step {step.index} "
                                f"(OCR + Captioning)"
                            )
                            self._grounding_logs.append(
                                PromptLog(
                                    prompt=step.statement,
                                    response=summary,
                                    step_index=step.index,
                                    stage="grounding_composite",
                                )
                            )

                        logger.info(
                            "Batch grounding completed: %d total evidence regions across %d steps",
                            len(all_evidences),
                            len(vision_steps),
                        )
                        return all_evidences
            except Exception as e:
                logger.warning(
                    "Cross-step batch captioning failed, falling back to per-step execution: %s",
                    e,
                )

            for step in vision_steps:
                step_bboxes = bboxes_by_step.get(step.index, [])
                if not step_bboxes:
                    logger.warning(f"No bboxes extracted for step {step.index}")
                    continue

                # Check if OCR is needed for this step
                need_ocr = step.need_ocr if step else False
                if not need_ocr:
                    logger.info(
                        f"Skipping OCR for step {step.index}: need_ocr=False (only running captioning)"
                    )
                else:
                    logger.info(
                        f"Running OCR and Captioning in parallel for {len(step_bboxes)} regions in step {step.index}"
                    )

                # Try batch parallel execution first
                if self.batch_evidence_enabled and (
                    hasattr(self.captioning, "ocr_regions_batch")
                    or hasattr(self.captioning, "caption_regions_batch")
                ):
                    try:
                        results = self._run_ocr_and_caption_batch_parallel(
                            image, step_bboxes, step.index, step=step
                        )
                        for bbox_idx, (bbox, (ocr_text, caption)) in enumerate(
                            zip(step_bboxes, results)
                        ):
                            all_evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description=caption,
                                    ocr_text=ocr_text,
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )
                    except Exception as e:
                        logger.warning(
                            f"Batch parallel execution failed for step {step.index}, falling back: {e}"
                        )
                        # Fallback to sequential parallel execution
                        for bbox_idx, bbox in enumerate(step_bboxes):
                            try:
                                ocr_text, caption = self._run_ocr_and_caption_parallel(
                                    image, bbox, step.index, bbox_idx, step=step
                                )
                                all_evidences.append(
                                    GroundedEvidence(
                                        bbox=bbox,
                                        description=caption,
                                        ocr_text=ocr_text,
                                        step_index=step.index,
                                        confidence=0.95,
                                    )
                                )
                            except Exception as e2:
                                logger.warning(
                                    f"Failed to process region {bbox} in step {step.index}: {e2}"
                                )
                                all_evidences.append(
                                    GroundedEvidence(
                                        bbox=bbox,
                                        description="",
                                        ocr_text="",
                                        step_index=step.index,
                                        confidence=0.95,
                                    )
                                )
                else:
                    # Sequential parallel execution
                    for bbox_idx, bbox in enumerate(step_bboxes):
                        try:
                            ocr_text, caption = self._run_ocr_and_caption_parallel(
                                image, bbox, step.index, bbox_idx, step=step
                            )
                            all_evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description=caption,
                                    ocr_text=ocr_text,
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to process region {bbox} in step {step.index}: {e}"
                            )
                            all_evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description="",
                                    ocr_text="",
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )

                # Log for this step
                step_evidences = [
                    ev for ev in all_evidences if ev.step_index == step.index
                ]
                summary = f"Extracted {len(step_evidences)} evidence regions for step {step.index} (OCR + Captioning)"
                self._grounding_logs.append(
                    PromptLog(
                        prompt=step.statement,
                        response=summary,
                        step_index=step.index,
                        stage="grounding_composite",
                    )
                )

            logger.info(
                f"Batch grounding completed: {len(all_evidences)} total evidence regions across {len(vision_steps)} steps"
            )
            return all_evidences

        except Exception as e:
            logger.error(f"Batch evidence extraction failed: {e}", exc_info=True)
            return []

    def extract_step_evidence(
        self,
        image: Image.Image,
        question: str,
        step: ReasoningStep,
        max_regions: int,
    ) -> List[GroundedEvidence]:
        """
        Extract visual evidence for a reasoning step.

        Always runs both OCR and Captioning in parallel for all regions.

        Args:
            image: Input image
            question: Original question
            step: Reasoning step to verify
            max_regions: Maximum regions to extract

        Returns:
            List of GroundedEvidence objects
        """
        try:
            # Get NMS config from grounding config
            nms_enabled = getattr(self.config.grounding, "nms_enabled", True)
            nms_iou_threshold = getattr(self.config.grounding, "nms_iou_threshold", 0.5)

            # Use grounding model to get bboxes (pass NMS config if grounding client supports it)
            if hasattr(self.grounding, "extract_regions"):
                # Check if extract_regions accepts nms parameters and step_index
                import inspect

                sig = inspect.signature(self.grounding.extract_regions)
                params = sig.parameters

                # Build arguments based on what the method accepts
                kwargs = {}
                if "nms_enabled" in params:
                    kwargs["nms_enabled"] = nms_enabled
                if "nms_iou_threshold" in params:
                    kwargs["nms_iou_threshold"] = nms_iou_threshold
                if "step_index" in params:
                    kwargs["step_index"] = step.index

                bboxes = self.grounding.extract_regions(
                    image, step.statement, max_regions, **kwargs
                )
            else:
                bboxes = []

            if not bboxes:
                logger.warning(f"No bboxes extracted for step {step.index}")
                return []

            # Note: NMS is already applied in florence_grounding_client.extract_regions()
            # No need to apply it again here to avoid redundant processing

            evidences = []

            # Always run both OCR and Captioning in parallel
            logger.info(
                f"Running OCR and Captioning in parallel for {len(bboxes)} regions in step {step.index}"
            )

            # Try batch parallel execution first
            if self.batch_evidence_enabled and (
                hasattr(self.captioning, "ocr_regions_batch")
                or hasattr(self.captioning, "caption_regions_batch")
            ):
                try:
                    results = self._run_ocr_and_caption_batch_parallel(
                        image, bboxes, step.index, step=step
                    )
                    for bbox_idx, (bbox, (ocr_text, caption)) in enumerate(
                        zip(bboxes, results)
                    ):
                        evidences.append(
                            GroundedEvidence(
                                bbox=bbox,
                                description=caption,  # Caption text
                                ocr_text=ocr_text,  # OCR text
                                step_index=step.index,
                                confidence=0.95,  # Default confidence
                            )
                        )
                except Exception as e:
                    logger.warning(
                        f"Batch parallel execution failed, falling back to sequential parallel: {e}"
                    )
                    # Fallback to sequential parallel execution
                    for bbox_idx, bbox in enumerate(bboxes):
                        try:
                            ocr_text, caption = self._run_ocr_and_caption_parallel(
                                image, bbox, step.index, bbox_idx, step=step
                            )
                            evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description=caption,
                                    ocr_text=ocr_text,
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )
                        except Exception as e2:
                            logger.warning(f"Failed to process region {bbox}: {e2}")
                            # Still create evidence with empty strings
                            evidences.append(
                                GroundedEvidence(
                                    bbox=bbox,
                                    description="",
                                    ocr_text="",
                                    step_index=step.index,
                                    confidence=0.95,
                                )
                            )
            else:
                # Sequential parallel execution (no batch methods available)
                for bbox_idx, bbox in enumerate(bboxes):
                    try:
                        ocr_text, caption = self._run_ocr_and_caption_parallel(
                            image, bbox, step.index, bbox_idx, step=step
                        )
                        evidences.append(
                            GroundedEvidence(
                                bbox=bbox,
                                description=caption,
                                ocr_text=ocr_text,
                                step_index=step.index,
                                confidence=0.95,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to process region {bbox}: {e}")
                        # Still create evidence with empty strings
                        evidences.append(
                            GroundedEvidence(
                                bbox=bbox,
                                description="",
                                ocr_text="",
                                step_index=step.index,
                                confidence=0.95,
                            )
                        )

            # Log for consistency
            summary = f"Extracted {len(evidences)} evidence regions for step {step.index} (OCR + Captioning)"
            self._grounding_logs.append(
                PromptLog(
                    prompt=step.statement,
                    response=summary,
                    step_index=step.index,
                    stage="grounding_composite",
                )
            )

            return evidences

        except Exception as e:
            logger.error(f"Evidence extraction failed for step {step.index}: {e}")
            return []

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        evidences: List[GroundedEvidence],
    ) -> Tuple[str, List[KeyEvidence], Optional[str]]:
        """
        Synthesize final answer with key evidence.

        Delegates to synthesis model.

        Args:
            image: Input image
            question: Original question
            steps: List of reasoning steps
            evidences: List of grounded evidence

        Returns:
            Tuple of (answer_text, key_evidence, explanation)
        """
        answer, key_evidence, explanation = self.synthesis.synthesize_answer(
            image, question, steps, evidences
        )

        # Copy log from synthesis model
        if hasattr(self.synthesis, "answer_log"):
            self._answer_log = self.synthesis.answer_log

        # Copy paraphrased_question if available
        if hasattr(self.synthesis, "_paraphrased_question"):
            self._paraphrased_question = self.synthesis._paraphrased_question
        else:
            self._paraphrased_question = None

        return answer, key_evidence, explanation


class VLMClientFactory:
    """
    Factory to create complete VLM client from configuration.

    Produces a CompositeVLMClient that implements all required protocols
    by composing specialized models for each stage.
    """

    @staticmethod
    def create_from_config(
        config: CoRGiConfig,
        image_logger=None,
        output_tracer=None,
        parallel_loading: bool = True,
    ) -> CompositeVLMClient:
        """
        Create VLM client from config with optional parallel model loading.

        Args:
            config: Complete CoRGi pipeline configuration
            image_logger: Optional ImageLogger instance for image logging
            output_tracer: Optional OutputTracer instance for output tracing
            parallel_loading: If True, load independent models in parallel (default: True)

        Returns:
            CompositeVLMClient instance
        """
        logger.info("Creating VLM client from configuration")
        load_start = time.time()
        
        # Note: Previously we force-enabled parallel loading when multiple models exist.
        # This caused "meta tensor" errors on some systems. Now we respect the user's choice.
        if config.requires_parallel_loading() and parallel_loading:
            # Parallel loading only helps when models are placed on different devices.
            # When multiple models target the same CUDA device, concurrent loading can be
            # unstable and may trigger "meta tensor" issues in practice.
            try:
                devices: List[str] = []
                if config.reasoning.model and config.reasoning.model.device:
                    devices.append(str(config.reasoning.model.device))
                if (
                    config.captioning.model
                    and config.captioning.model.model_type == "composite"
                ):
                    if config.captioning.ocr.model and config.captioning.ocr.model.device:
                        devices.append(str(config.captioning.ocr.model.device))
                    if (
                        config.captioning.caption.model
                        and config.captioning.caption.model.device
                    ):
                        devices.append(str(config.captioning.caption.model.device))
                elif config.captioning.model and config.captioning.model.device:
                    devices.append(str(config.captioning.model.device))

                devices = [d for d in devices if d]
                unique_devices = sorted(set(devices))
                if (
                    len(unique_devices) <= 1
                    and unique_devices
                    and unique_devices[0].startswith("cuda:")
                ):
                    logger.info(
                        "All models target %s; disabling parallel loading for stability.",
                        unique_devices[0],
                    )
                    parallel_loading = False
            except Exception:
                # Never block loading due to heuristic.
                pass

        if config.requires_parallel_loading() and parallel_loading:
            logger.info(
                "Multiple distinct models detected. Using parallel loading for faster startup."
            )
        elif config.requires_parallel_loading() and not parallel_loading:
            logger.info(
                "Multiple distinct models detected. Using sequential loading (slower but more stable)."
            )

        # Determine which models need to be loaded and identify reuse opportunities
        models_to_load: List[Tuple[str, str, Callable, Dict[str, Any]]] = []
        # Format: (model_name, model_type, load_function, kwargs)

        # Helper to create load function
        def make_load_fn(model_type: str, model_config: ModelConfig, stage: str):
            def load():
                if model_type == "reasoning":
                    return ModelRegistry.create_reasoning_model(model_config)
                elif model_type == "grounding":
                    return ModelRegistry.create_grounding_model(model_config)
                elif model_type == "captioning":
                    # Pass model-specific config
                    captioning_kwargs = {}
                    if model_config.model_type == "vintern":
                        captioning_kwargs = {
                            "image_size": config.captioning.image_size,
                            "max_num_patches": config.captioning.max_num_patches,
                        }
                    elif model_config.model_type == "paddleocr":
                        captioning_kwargs = {
                            "task": config.captioning.ocr_task,
                        }
                    return ModelRegistry.create_captioning_model(
                        model_config, **captioning_kwargs
                    )
                elif model_type == "synthesis":
                    return ModelRegistry.create_synthesis_model(model_config)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

            return load

        # Check reasoning model
        reasoning_model_id = config.reasoning.model.model_id
        models_to_load.append(
            (
                "reasoning",
                "reasoning",
                make_load_fn("reasoning", config.reasoning.model, "reasoning"),
                {},
            )
        )

        # Check grounding model
        # Special handling for Qwen grounding (if not reusing reasoning)
        if config.grounding.model and config.grounding.model.model_type in (
            "qwen_instruct",
            "qwen_thinking",
        ):
            # Qwen adapter - check if we can reuse reasoning
            if config.grounding.model.model_id == reasoning_model_id:
                logger.info("Grounding will reuse reasoning model (Qwen adapter)")
            else:
                models_to_load.append(
                    (
                        "grounding",
                        "reasoning",
                        make_load_fn("reasoning", config.grounding.model, "grounding"),
                        {},
                    )
                )
        else:
            # Florence-2 or other - check if same as captioning
            if (
                config.grounding.model is not None
                and config.captioning.model is not None
                and config.grounding.model.model_id == config.captioning.model.model_id
                and config.grounding.model.model_type
                == config.captioning.model.model_type
            ):
                logger.info("Grounding and captioning will share the same model")
                models_to_load.append(
                    (
                        "grounding_captioning",
                        "grounding",
                        make_load_fn("grounding", config.grounding.model, "grounding"),
                        {},
                    )
                )
            elif config.grounding.model is not None:
                models_to_load.append(
                    (
                        "grounding",
                        "grounding",
                        make_load_fn("grounding", config.grounding.model, "grounding"),
                        {},
                    )
                )

        # Check captioning model
        if config.captioning.model.model_type in ("qwen_instruct", "qwen_thinking"):
            # Qwen adapter - check reuse
            if config.captioning.model.model_id == reasoning_model_id:
                logger.info("Captioning will reuse reasoning model (Qwen adapter)")
            elif (
                config.grounding.model is not None
                and config.captioning.model.model_id == config.grounding.model.model_id
                and config.grounding.model.model_type
                in ("qwen_instruct", "qwen_thinking")
            ):
                logger.info("Captioning will reuse grounding model (Qwen adapter)")
            else:
                # Check if already added
                if not any(name == "captioning" for name, _, _, _ in models_to_load):
                    models_to_load.append(
                        (
                            "captioning",
                            "reasoning",
                            make_load_fn(
                                "reasoning", config.captioning.model, "captioning"
                            ),
                            {},
                        )
                    )
        elif config.captioning.model.model_type == "composite":
            # Composite model - needs special handling
            # Load OCR and Captioning sub-models, then create composite
            logger.info("Loading composite captioning model (OCR + Caption)")
            
            def load_composite():
                from .composite.composite_captioning_client import CompositeCaptioningClient
                
                # Load OCR client
                ocr_config = config.captioning.ocr.model
                logger.info(f"Loading OCR client: {ocr_config.model_type}")
                ocr_client = ModelRegistry.create_captioning_model(ocr_config)
                
                # Load Captioning client
                caption_config = config.captioning.caption.model
                logger.info(f"Loading Caption client: {caption_config.model_type}")
                caption_client = ModelRegistry.create_captioning_model(caption_config)
                
                # Create composite
                composite = CompositeCaptioningClient(
                    ocr_client=ocr_client,
                    captioning_client=caption_client,
                    image_logger=image_logger,
                    output_tracer=output_tracer,
                )
                return composite
            
            models_to_load.append(
                (
                    "captioning",
                    "captioning",
                    load_composite,
                    {},
                )
            )
        else:
            # Florence-2 or other - check if already added as grounding_captioning
            if not any(
                name in ("grounding_captioning", "captioning")
                for name, _, _, _ in models_to_load
            ):
                models_to_load.append(
                    (
                        "captioning",
                        "captioning",
                        make_load_fn(
                            "captioning", config.captioning.model, "captioning"
                        ),
                        {},
                    )
                )

        # Check synthesis model
        if config.synthesis.model is not None and config.synthesis.model.model_id == reasoning_model_id:
            logger.info("Synthesis will reuse reasoning model")
        elif config.synthesis.model is not None:
            models_to_load.append(
                (
                    "synthesis",
                    "synthesis",
                    make_load_fn("synthesis", config.synthesis.model, "synthesis"),
                    {},
                )
            )
        else:
            logger.info("Synthesis will reuse reasoning model (reuse_reasoning=true)")

        # Load models (parallel or sequential with fallback)
        loaded_models: Dict[str, Any] = {}
        use_parallel = parallel_loading and len(models_to_load) > 1
        parallel_failed = False

        if use_parallel:
            model_names = [name for name, _, _, _ in models_to_load]
            logger.info(
                f"Loading {len(models_to_load)} models in parallel: {model_names}"
            )
            
            try:
                with ThreadPoolExecutor(
                    max_workers=min(4, len(models_to_load))
                ) as executor:
                    futures = {
                        executor.submit(load_fn): (name, model_type)
                        for name, model_type, load_fn, kwargs in models_to_load
                    }

                    for future in as_completed(futures):
                        name, model_type = futures[future]
                        try:
                            model = future.result()
                            loaded_models[name] = model
                            logger.info(f"✓ {name} model loaded (parallel)")
                        except NotImplementedError as e:
                            # "meta tensor" error - need to fallback to sequential
                            if "meta tensor" in str(e).lower():
                                logger.warning(
                                    f"Meta tensor error loading {name}, falling back to sequential"
                                )
                                parallel_failed = True
                                break
                            raise
                        except Exception as e:
                            logger.error(f"Failed to load {name} model: {e}", exc_info=True)
                            raise
            except Exception as e:
                if "meta tensor" in str(e).lower():
                    parallel_failed = True
                    logger.warning(
                        f"Parallel loading failed with meta tensor error, falling back to sequential"
                    )
                else:
                    raise

        # Sequential loading (primary or fallback)
        if not use_parallel or parallel_failed:
            if parallel_failed:
                logger.info("Retrying with sequential loading...")
                loaded_models.clear()
            else:
                logger.info(f"Loading {len(models_to_load)} models sequentially...")
            
            for name, model_type, load_fn, kwargs in models_to_load:
                try:
                    model = load_fn()
                    loaded_models[name] = model
                    logger.info(f"✓ {name} model loaded")
                except Exception as e:
                    logger.error(f"Failed to load {name} model: {e}", exc_info=True)
                    raise

        load_time = time.time() - load_start
        load_mode = "parallel" if use_parallel and not parallel_failed else "sequential"
        logger.info(f"All models loaded in {load_time:.2f}s ({load_mode})")

        # Now construct the final models with adapters and loggers
        # Reasoning model
        reasoning_model = loaded_models.get("reasoning")
        if reasoning_model is None:
            # Should not happen, but handle gracefully
            reasoning_model = ModelRegistry.create_reasoning_model(
                config.reasoning.model
            )

        # Set loggers and extraction method
        if hasattr(reasoning_model, "image_logger"):
            reasoning_model.image_logger = image_logger
        if hasattr(reasoning_model, "output_tracer"):
            reasoning_model.output_tracer = output_tracer
        if hasattr(reasoning_model, "extraction_method"):
            reasoning_model.extraction_method = config.reasoning.extraction_method
        # Set reasoning generation config
        if hasattr(reasoning_model, "reasoning_max_tokens"):
            reasoning_model.reasoning_max_tokens = config.reasoning.max_new_tokens
        if hasattr(reasoning_model, "reasoning_do_sample"):
            reasoning_model.reasoning_do_sample = getattr(
                config.reasoning, "do_sample", False
            )

        # Grounding model
        if config.grounding.model is not None and config.grounding.model.model_type in ("qwen_instruct", "qwen_thinking"):
            if config.grounding.model.model_id == reasoning_model_id:
                grounding_base = reasoning_model
                # Verify same instance for Phase 1 (Reasoning) and Phase 2 (Grounding)
                if id(grounding_base) == id(reasoning_model):
                    logger.info(
                        "✓ Phase 1 (Reasoning) and Phase 2 (Grounding) share the same Qwen3-VL model instance"
                    )
                else:
                    logger.warning(
                        "⚠ Grounding base is not the same instance as reasoning model"
                    )
            else:
                grounding_base = loaded_models.get("grounding")
                if grounding_base is None:
                    grounding_base = ModelRegistry.create_reasoning_model(
                        config.grounding.model
                    )
                if hasattr(grounding_base, "image_logger"):
                    grounding_base.image_logger = image_logger
                if hasattr(grounding_base, "output_tracer"):
                    grounding_base.output_tracer = output_tracer
            grounding_model = QwenGroundingAdapter(grounding_base)
        elif config.grounding.model is None:
            # Reuse reasoning model (reuse_reasoning: true)
            grounding_base = reasoning_model
            grounding_model = QwenGroundingAdapter(grounding_base)
            logger.info("✓ Grounding reusing reasoning model (Qwen adapter)")
        else:
            # Check if shared with captioning
            if config.grounding.model.model_id == config.captioning.model.model_id:
                grounding_base = loaded_models.get("grounding_captioning")
            else:
                grounding_base = loaded_models.get("grounding")
            if grounding_base is None:
                grounding_base = ModelRegistry.create_grounding_model(
                    config.grounding.model
                )
            if hasattr(grounding_base, "image_logger"):
                grounding_base.image_logger = image_logger
            if hasattr(grounding_base, "output_tracer"):
                grounding_base.output_tracer = output_tracer
            grounding_model = grounding_base

        # Captioning model
        # Check if we need a composite client (separate OCR and Captioning models)
        if config.captioning.ocr_model is not None:
            # Create composite client with separate OCR and Captioning models
            logger.info(
                "Creating composite captioning client: PaddleOCR (OCR) + FastVLM (Captioning)"
            )

            # Create OCR client (PaddleOCR)
            ocr_kwargs = {}
            if config.captioning.ocr_model.model_type == "paddleocr":
                ocr_kwargs = {"task": config.captioning.ocr_task}
            ocr_client = ModelRegistry.create_captioning_model(
                config.captioning.ocr_model, **ocr_kwargs
            )
            if hasattr(ocr_client, "image_logger"):
                ocr_client.image_logger = image_logger
            if hasattr(ocr_client, "output_tracer"):
                ocr_client.output_tracer = output_tracer

            # Create Captioning client (FastVLM)
            captioning_kwargs = {}
            if config.captioning.model.model_type == "vintern":
                captioning_kwargs = {
                    "image_size": config.captioning.image_size,
                    "max_num_patches": config.captioning.max_num_patches,
                }
            captioning_client = ModelRegistry.create_captioning_model(
                config.captioning.model, **captioning_kwargs
            )
            if hasattr(captioning_client, "image_logger"):
                captioning_client.image_logger = image_logger
            if hasattr(captioning_client, "output_tracer"):
                captioning_client.output_tracer = output_tracer

            # Create composite client
            captioning_model = CompositeCaptioningClient(
                ocr_client=ocr_client,
                captioning_client=captioning_client,
                image_logger=image_logger,
                output_tracer=output_tracer,
            )
        elif config.captioning.model.model_type in ("qwen_instruct", "qwen_thinking"):
            if config.captioning.model.model_id == reasoning_model_id:
                captioning_base = reasoning_model
            elif (
                config.grounding.model is not None
                and config.captioning.model.model_id == config.grounding.model.model_id
                and config.grounding.model.model_type
                in ("qwen_instruct", "qwen_thinking")
            ):
                # Reuse grounding's base model
                if hasattr(grounding_model, "client"):
                    captioning_base = grounding_model.client
                else:
                    captioning_base = grounding_base
            else:
                captioning_base = loaded_models.get("captioning")
                if captioning_base is None:
                    captioning_base = ModelRegistry.create_reasoning_model(
                        config.captioning.model
                    )
                if hasattr(captioning_base, "image_logger"):
                    captioning_base.image_logger = image_logger
                if hasattr(captioning_base, "output_tracer"):
                    captioning_base.output_tracer = output_tracer
            captioning_model = QwenCaptioningAdapter(captioning_base)
        else:
            # Florence-2 or other captioning model
            # If grounding model exists and is not Qwen, add to load list
            if config.grounding.model and (
                config.grounding.model.model_id != reasoning_model_id
                and config.grounding.model.model_id != config.captioning.model.model_id
            ):
                # Models are shared - create separate CaptioningClient instance
                # The cache mechanism will ensure they share the same model/processor
                logger.info(
                    "Captioning will share model instance with grounding (creating separate client)"
                )
                # Pass model-specific config
                captioning_kwargs = {}
                if config.captioning.model.model_type == "vintern":
                    captioning_kwargs = {
                        "image_size": config.captioning.image_size,
                        "max_num_patches": config.captioning.max_num_patches,
                    }
                elif config.captioning.model.model_type == "paddleocr":
                    captioning_kwargs = {
                        "task": config.captioning.ocr_task,
                    }
                captioning_model = ModelRegistry.create_captioning_model(
                    config.captioning.model, **captioning_kwargs
                )
            else:
                # Different model - use loaded or create new
                captioning_base = loaded_models.get("captioning")
                if captioning_base is None:
                    # Pass model-specific config
                    captioning_kwargs = {}
                    if config.captioning.model.model_type == "vintern":
                        captioning_kwargs = {
                            "image_size": config.captioning.image_size,
                            "max_num_patches": config.captioning.max_num_patches,
                        }
                    elif config.captioning.model.model_type == "paddleocr":
                        captioning_kwargs = {
                            "task": config.captioning.ocr_task,
                        }
                    captioning_base = ModelRegistry.create_captioning_model(
                        config.captioning.model, **captioning_kwargs
                    )
                if hasattr(captioning_base, "image_logger"):
                    captioning_base.image_logger = image_logger
                if hasattr(captioning_base, "output_tracer"):
                    captioning_base.output_tracer = output_tracer
                captioning_model = captioning_base

            # Set loggers for captioning model
            if hasattr(captioning_model, "image_logger"):
                captioning_model.image_logger = image_logger
            if hasattr(captioning_model, "output_tracer"):
                captioning_model.output_tracer = output_tracer

        # Synthesis model
        if config.synthesis.model is None:
            # Reuse reasoning model (reuse_reasoning: true)
            synthesis_model = reasoning_model
            logger.info("✓ Synthesis reusing reasoning model (reuse_reasoning: true)")
        elif config.synthesis.model.model_id == reasoning_model_id:
            synthesis_model = reasoning_model
            # Verify same instance for Phase 1 (Reasoning) and Phase 4 (Synthesis)
            if id(synthesis_model) == id(reasoning_model):
                logger.info(
                    "✓ Phase 1 (Reasoning) and Phase 4 (Synthesis) share the same Qwen3-VL model instance"
                )
            else:
                logger.warning(
                    "⚠ Synthesis model is not the same instance as reasoning model"
                )
        else:
            synthesis_model = loaded_models.get("synthesis")
            if synthesis_model is None:
                synthesis_model = ModelRegistry.create_synthesis_model(
                    config.synthesis.model
                )
            if hasattr(synthesis_model, "image_logger"):
                synthesis_model.image_logger = image_logger
            if hasattr(synthesis_model, "output_tracer"):
                synthesis_model.output_tracer = output_tracer

        # Set synthesis generation config
        if hasattr(synthesis_model, "synthesis_max_tokens"):
            synthesis_model.synthesis_max_tokens = config.synthesis.max_new_tokens
        if hasattr(synthesis_model, "synthesis_do_sample"):
            synthesis_model.synthesis_do_sample = getattr(
                config.synthesis, "do_sample", False
            )

        # Create composite client
        client = CompositeVLMClient(
            reasoning=reasoning_model,
            grounding=grounding_model,
            captioning=captioning_model,
            synthesis=synthesis_model,
            config=config,
            image_logger=image_logger,
            output_tracer=output_tracer,
        )

        total_time = time.time() - load_start
        logger.info(f"✓ VLM client created successfully in {total_time:.2f}s total")
        return client


__all__ = ["VLMClientFactory", "CompositeVLMClient"]
