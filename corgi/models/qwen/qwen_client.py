"""
Backward Compatibility Shim for qwen_client.py

This module provides backward compatibility by importing from the new
qwen_thinking_client module and aliasing classes.

DEPRECATED: For new code, use:
- Qwen3VLThinkingClient for Thinking models
- Qwen3VLInstructClient for Instruct models
- VLMClientFactory.create_from_config() for configurable pipelines
"""

from __future__ import annotations

import os
import warnings
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import spaces  # type: ignore
except ImportError:  # pragma: no cover - spaces library only on HF Spaces
    spaces = None  # type: ignore

import torch
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor

from .qwen_thinking_client import (
    Qwen3VLThinkingClient,
    # Note: QwenGenerationConfig is defined in this file, not in qwen_thinking_client
)
from ...core.types import ReasoningStep, GroundedEvidence, KeyEvidence, PromptLog
from ...core.schemas import ReasoningStepsSchema
from ...utils.parsers import (
    parse_structured_reasoning,
    parse_roi_evidence,
    _normalize_bbox,
)

logger = logging.getLogger(__name__)


# OPTIMIZATION: Enhanced prompts with strict JSON schema for better structured outputs
DEFAULT_REASONING_PROMPT = (
    "You are a careful multimodal reasoner following the CoRGI protocol. "
    "Given the question and the image, produce a JSON object with reasoning steps. "
    "REQUIRED JSON FORMAT:\n"
    "{{\n"
    '  "steps": [\n'
    "    {{\n"
    '      "index": 1,\n'
    '      "statement": "concise reasoning statement",\n'
    '      "needs_vision": true,\n'
    '      "reason": "why visual verification is needed"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Limit to {max_steps} steps. Respond with ONLY valid JSON, no commentary."
)

DEFAULT_GROUNDING_PROMPT = (
    "You are validating the following reasoning step:\n"
    "{step_statement}\n\n"
    "REQUIRED JSON FORMAT:\n"
    "{{\n"
    '  "evidences": [\n'
    "    {{\n"
    '      "step": 1,\n'
    '      "bbox": [x1, y1, x2, y2],\n'
    '      "description": "visual description",\n'
    '      "confidence": 0.95\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Return up to {max_regions} regions. Use normalized coordinates (0-1). "
    "Respond with ONLY valid JSON, no commentary."
)

DEFAULT_ANSWER_PROMPT = (
    "You are finalizing the answer using verified evidence. "
    "Question: {question}\n"
    "Structured reasoning steps:\n"
    "{steps}\n"
    "Verified visual evidence (bounding boxes):\n"
    "{evidence}\n\n"
    "Please provide your final answer in this EXACT JSON format:\n"
    "{{\n"
    '  "answer": "Your final answer sentence",\n'
    '  "key_evidence": [\n'
    "    {{\n"
    '      "bbox": [x1, y1, x2, y2],\n'
    '      "description": "What this region shows",\n'
    '      "reasoning": "Why this supports the answer"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "If you cannot provide structured evidence, just give:\n"
    '{{"answer": "your answer", "key_evidence": []}}\n'
    "Do not include <think> tags. Respond with JSON only."
)


def _format_steps_for_prompt(steps: List[ReasoningStep]) -> str:
    return "\n".join(
        f"{step.index}. {step.statement} (needs vision: {step.needs_vision})"
        for step in steps
    )


def _format_evidence_for_prompt(evidences: List[GroundedEvidence]) -> str:
    if not evidences:
        return "No evidence collected."
    lines = []
    for ev in evidences:
        desc = ev.description or "No description"
        bbox = ", ".join(f"{coord:.2f}" for coord in ev.bbox)
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "n/a"
        lines.append(f"Step {ev.step_index}: bbox=({bbox}), conf={conf}, desc={desc}")
    return "\n".join(lines)


def _strip_think_content(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = cleaned.replace("<think>", "")
    return cleaned.strip()


def _parse_json_with_schema(text: str, schema_class):
    """
    Parse JSON text and validate with Pydantic schema.

    OPTIMIZATION: This provides structured validation with clear error messages,
    making outputs more reliable than regex-based parsing.
    """
    import json

    # Strip thinking content
    cleaned = _strip_think_content(text)

    # Try to extract JSON
    try:
        # Look for JSON in the response
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned[start_idx : end_idx + 1]
            data = json.loads(json_str)
            # Validate with Pydantic
            validated = schema_class(**data)
            return validated
        else:
            raise ValueError(f"No JSON object found in response: {cleaned[:200]}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Attempted to parse: {cleaned[:500]}")
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Schema validation error: {e}")
        logger.error(f"Data: {cleaned[:500]}")
        raise ValueError(f"Schema validation failed: {e}")


_MODEL_CACHE: dict[str, AutoModelForImageTextToText] = {}
_PROCESSOR_CACHE: dict[str, AutoProcessor] = {}


def _gpu_decorator(duration: int = 120):
    if spaces is None:
        return lambda fn: fn
    return spaces.GPU(duration=duration)


def _ensure_device(
    model: AutoModelForImageTextToText, device: str
) -> AutoModelForImageTextToText:
    """Ensure model is on specified device."""
    target_device = torch.device(device)
    current_device = next(model.parameters()).device
    if current_device != target_device:
        model.to(target_device)
    return model


def _load_backend(
    model_id: str, device: str
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    if model_id not in _MODEL_CACHE:
        # Get device from config (required, no fallback)
        device_map = device
        if not device_map:
            raise ValueError("device must be specified in config (e.g., 'cuda:7')")

        # Use kernels-community/flash-attn3 for optimized inference with bfloat16
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for inference
                device_map=device_map,
                attn_implementation="kernels-community/flash-attn2",
            )
            logger.info(f"✓ kernels-community/flash-attn2 enabled for {model_id}")
        except Exception as e:
            logger.warning(
                f"kernels-community/flash-attn2 not available ({e}), falling back to standard attention"
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for inference
                device_map=device_map,
            )

        model = model.eval()

        # OPTIMIZATION: Enable Torch Compile for additional 1.5-2x speedup
        # Can be disabled via CORGI_DISABLE_COMPILE=1 env var for debugging
        if os.environ.get("CORGI_DISABLE_COMPILE", "0") != "1":
            try:
                logger.info(
                    "Compiling model with torch.compile (this may take a minute)..."
                )
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("✓ Torch compile enabled")
            except Exception as e:
                logger.warning(f"Torch compile failed ({e}), using uncompiled model")
        else:
            logger.info("Torch compile disabled via CORGI_DISABLE_COMPILE")

        # Try fast processor first, fallback to slow if unavailable
        try:
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            logger.info("✓ Using fast processor")
        except Exception as e:
            logger.warning(
                f"Fast processor not available, falling back to slow processor: {e}"
            )
            processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        _MODEL_CACHE[model_id] = model
        _PROCESSOR_CACHE[model_id] = processor
    return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]


@dataclass
class QwenGenerationConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Thinking"
    device: str = "cuda:7"  # Device must be specified in config
    max_new_tokens: int = 512
    temperature: float | None = None
    do_sample: bool = False


class Qwen3VLClient:
    """Wrapper around transformers Qwen3-VL chat API for CoRGI pipeline."""

    def __init__(
        self,
        config: Optional[QwenGenerationConfig] = None,
        florence_client=None,  # Optional Florence2Client for faster ROI extraction
    ) -> None:
        self.config = config or QwenGenerationConfig()
        if not self.config.device:
            raise ValueError(
                "device must be specified in QwenGenerationConfig (e.g., 'cuda:7')"
            )
        self._model, self._processor = _load_backend(
            self.config.model_id, self.config.device
        )
        self.florence_client = (
            florence_client  # OPTIMIZATION: Use Florence-2 for ROI if available
        )
        if self.florence_client:
            logger.info("✓ Florence-2 enabled for ROI extraction and captioning")
        self.reset_logs()

    def reset_logs(self) -> None:
        self._reasoning_log: Optional[PromptLog] = None
        self._grounding_logs: List[PromptLog] = []
        self._answer_log: Optional[PromptLog] = None

    @property
    def reasoning_log(self) -> Optional[PromptLog]:
        return self._reasoning_log

    @property
    def grounding_logs(self) -> List[PromptLog]:
        return list(self._grounding_logs)

    @property
    def answer_log(self) -> Optional[PromptLog]:
        return self._answer_log

    @_gpu_decorator()
    def _chat(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=[chat_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        # OPTIMIZATION: Greedy decoding for faster inference
        # Old settings (commented for reference):
        # gen_kwargs = {
        #     "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
        #     "do_sample": self.config.do_sample,
        # }
        # if self.config.do_sample and self.config.temperature is not None:
        #     gen_kwargs["temperature"] = self.config.temperature

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": False,  # Greedy decoding = faster
            "temperature": 0.0,  # Deterministic
            "num_beams": 1,  # No beam search overhead
            "use_cache": True,  # Enable KV cache for faster inference
        }
        # Generate with autocast for bfloat16 inference
        device_type = "cuda" if "cuda" in self.config.device else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16
        ):
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        prompt_length = inputs.input_ids.shape[1]
        generated_tokens = output_ids[:, prompt_length:]
        response = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

    def structured_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> List[ReasoningStep]:
        prompt = (
            DEFAULT_REASONING_PROMPT.format(max_steps=max_steps)
            + f"\nQuestion: {question}"
        )
        # OPTIMIZATION: Reduced from 1024 to 512 tokens (sufficient for 3-6 steps)
        response = self._chat(image=image, prompt=prompt, max_new_tokens=512)
        self._reasoning_log = PromptLog(
            prompt=prompt, response=response, stage="reasoning"
        )

        # Log response for debugging
        if not response or not response.strip():
            logger.error(
                f"Empty response from model for structured reasoning! Question: {question}"
            )
            # Return minimal fallback to avoid crash
            return [
                ReasoningStep(
                    index=1,
                    statement="Unable to generate reasoning (empty model response)",
                    needs_vision=False,
                    reason=None,
                    need_ocr=False,
                )
            ]

        logger.info(f"Structured reasoning response length: {len(response)} chars")
        logger.debug(f"Response preview: {response[:200]}...")

        # OPTIMIZATION: Use Pydantic schema validation for more reliable parsing
        try:
            validated = _parse_json_with_schema(response, ReasoningStepsSchema)
            return [
                ReasoningStep(
                    index=step.index,
                    statement=step.statement,
                    needs_vision=step.needs_vision,
                    reason=step.reason,
                    need_ocr=getattr(step, "need_ocr", False),
                )
                for step in validated.steps
            ]
        except Exception as e:
            logger.warning(
                f"Pydantic validation failed, falling back to legacy parser: {e}"
            )
            # OLD IMPLEMENTATION: Fallback to regex-based parser
            try:
                return parse_structured_reasoning(response, max_steps=max_steps)
            except Exception as e2:
                logger.error(f"Both parsers failed: {e2}")
                logger.error(f"Raw response (first 500 chars): {response[:500]}")
                # Return fallback to avoid crash
                return [
                    ReasoningStep(
                        index=1,
                        statement=f"Parsing error: {str(e)[:100]}",
                        needs_vision=False,
                        reason=None,
                        need_ocr=False,
                    )
                ]

    def extract_step_evidence(
        self,
        image: Image.Image,
        question: str,
        step: ReasoningStep,
        max_regions: int,
    ) -> List[GroundedEvidence]:
        # OPTIMIZATION: Use Florence-2 for faster grounding + captioning if available
        if self.florence_client:
            try:
                logger.debug(
                    f"Using Florence-2 for step {step.index} evidence extraction"
                )
                bboxes = self.florence_client.extract_regions(
                    image, step.statement, max_regions
                )

                evidences = []
                for bbox in bboxes:
                    description = self.florence_client.caption_region(image, bbox)
                    evidences.append(
                        GroundedEvidence(
                            bbox=bbox,
                            description=description,
                            step_index=step.index,
                            confidence=0.95,  # Florence-2 doesn't provide confidence, use default
                        )
                    )

                # Log for consistency (even though we used Florence-2)
                summary = (
                    f"Florence-2: Found {len(evidences)} regions for step {step.index}"
                )
                self._grounding_logs.append(
                    PromptLog(
                        prompt=step.statement,
                        response=summary,
                        step_index=step.index,
                        stage="grounding_florence2",
                    )
                )
                logger.info(
                    f"✓ Florence-2 extracted {len(evidences)} regions for step {step.index}"
                )
                return evidences
            except Exception as e:
                logger.error(
                    f"Florence-2 extraction failed, falling back to Qwen3-VL: {e}"
                )
                # Fall through to Qwen3-VL method below

        # OLD IMPLEMENTATION: Qwen3-VL-based ROI extraction (fallback or when Florence-2 disabled)
        prompt = DEFAULT_GROUNDING_PROMPT.format(
            step_statement=step.statement,
            max_regions=max_regions,
        )
        # OPTIMIZATION: Reduced from 256 to 128 tokens (sufficient for bbox+description)
        response = self._chat(image=image, prompt=prompt, max_new_tokens=128)

        # Log response
        if not response or not response.strip():
            logger.warning(
                f"Empty response for step evidence extraction (step {step.index})"
            )

        try:
            evidences = parse_roi_evidence(response, default_step_index=step.index)
        except Exception as e:
            logger.error(f"Failed to parse ROI evidence for step {step.index}: {e}")
            logger.error(f"Raw response (first 300 chars): {response[:300]}")
            evidences = []

        self._grounding_logs.append(
            PromptLog(
                prompt=prompt,
                response=response,
                step_index=step.index,
                stage="grounding",
            )
        )
        return evidences[:max_regions]

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        evidences: List[GroundedEvidence],
    ) -> tuple[str, List[KeyEvidence], Optional[str]]:
        prompt = DEFAULT_ANSWER_PROMPT.format(
            question=question,
            steps=_format_steps_for_prompt(steps),
            evidence=_format_evidence_for_prompt(evidences),
        )
        # OPTIMIZATION: Reduced from 512 to 256 tokens (sufficient for structured answer+evidence)
        response = self._chat(image=image, prompt=prompt, max_new_tokens=256)
        self._answer_log = PromptLog(
            prompt=prompt, response=response, stage="synthesis"
        )

        # Log response
        if not response or not response.strip():
            logger.error("Empty response from model for answer synthesis!")
            return "Unable to generate answer (empty model response)", [], None

        logger.info(f"Answer synthesis response length: {len(response)} chars")
        logger.debug(f"Response preview: {response[:200]}...")

        # Parse structured answer with key evidence
        try:
            answer_text, key_evidence, explanation = self._parse_answer_response(
                response
            )
            return answer_text, key_evidence, explanation
        except Exception as e:
            logger.error(f"Failed to parse answer response: {e}")
            logger.error(f"Raw response (first 500 chars): {response[:500]}")
            # Return fallback
            cleaned = _strip_think_content(response)
            return cleaned[:500] if cleaned else "Unable to parse answer", [], None

    def _parse_answer_response(
        self, response: str
    ) -> tuple[str, List[KeyEvidence], Optional[str]]:
        """Parse answer synthesis response with structured evidence."""
        import json

        # Strip thinking tags
        cleaned = _strip_think_content(response)

        # Try to parse JSON
        try:
            # Look for JSON structure
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned[start_idx : end_idx + 1]
                data = json.loads(json_str)

                answer = data.get("answer", cleaned)
                explanation = data.get("explanation", "")
                key_evidences = []

                # Limit to 2 key evidence items
                evidence_items = data.get("key_evidence", [])[:2]

                for item in evidence_items:
                    try:
                        bbox = _normalize_bbox(item.get("bbox", []))
                        description = item.get("description", "")
                        reasoning = item.get("reasoning", "")

                        if bbox and description:
                            key_evidences.append(
                                KeyEvidence(
                                    bbox=bbox,
                                    description=description,
                                    reasoning=reasoning,
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Failed to parse key evidence item: {e}")
                        continue

                return answer, key_evidences, explanation if explanation else None
        except Exception as e:
            logger.warning(
                f"Failed to parse structured answer, falling back to text: {e}"
            )

        # Fallback: return cleaned text with empty evidence
        return cleaned, [], None


# Backward compatibility alias
# This allows existing code using Qwen3VLClient to continue working
Qwen3VLClient = Qwen3VLThinkingClient


def create_qwen_client(
    model_id: str = "Qwen/Qwen3-VL-4B-Thinking", florence_client=None
):
    """
    Legacy factory function for creating Qwen clients.

    DEPRECATED: Use VLMClientFactory.create_from_config() instead.

    Args:
        model_id: Model identifier
        florence_client: Optional Florence-2 client

    Returns:
        Qwen3VLThinkingClient instance
    """
    warnings.warn(
        "create_qwen_client() is deprecated. "
        "Use VLMClientFactory.create_from_config() or instantiate "
        "Qwen3VLInstructClient/Qwen3VLThinkingClient directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = QwenGenerationConfig(model_id=model_id)
    return Qwen3VLThinkingClient(config=config, florence_client=florence_client)
