"""
Qwen3-VL Instruct Model Client.

This client is specialized for Qwen3-VL Instruct models (not Thinking models).
It generates natural language chain-of-thought reasoning followed by structured
JSON output with steps that need visual verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import os
import time

import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    QwenVLModel = Qwen3VLForConditionalGeneration
except ImportError:
    # Fallback to Qwen2VL for older transformers
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        Qwen2VLConfig,
        AutoConfig,
    )

    QwenVLModel = Qwen2VLForConditionalGeneration

    # Patch transformers to recognize qwen3_vl as qwen2_vl
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "qwen3_vl" not in CONFIG_MAPPING:
            CONFIG_MAPPING["qwen3_vl"] = Qwen2VLConfig

        # Also patch AutoModel mapping if possible, though AutoConfig is usually the gatekeeper
    except Exception as e:
        logger.warning(f"Failed to patch transformers for Qwen3-VL: {e}")


# Model cache for reusing loaded models
_MODEL_CACHE: dict = {}
_PROCESSOR_CACHE: dict = {}


@dataclass
class ReasoningStep:
    """A single step of reasoning."""

    step_number: int
    content: str
    is_visual: bool = False
    bbox: Optional[List[int]] = None


class QwenInstructClient:
    """
    Client for Qwen Instruct models.

    This client generates hybrid output: natural language chain-of-thought
    followed by structured JSON with steps that need visual verification.
    """

    def __init__(
        self,
        config: ModelConfig,
        extraction_method: str = "hybrid",
        image_logger=None,
        output_tracer=None,
    ):
        """
        Initialize the Qwen Instruct client.

        Args:
            config: Model configuration
            extraction_method: Method for extracting structured data ("hybrid" or "json_only")
            image_logger: Optional logger for images
            output_tracer: Optional tracer for outputs
        """
        self.config = config
        self.extraction_method = extraction_method
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Load model and processor
        self._model, self._processor = self._load_backend(config)

    def _load_backend(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model and processor."""
        model_id = config.model_id
        device_map = config.device

        logger.info(f"Loading Qwen model: {model_id} on {device_map}")

        # Validate device availability
        if device_map != "cpu" and device_map != "auto":
            try:
                if torch.cuda.is_available():
                    # Check if device index is valid
                    device_idx = int(device_map.split(":")[-1])
                    if device_idx >= torch.cuda.device_count():
                        raise ValueError(
                            f"Device index {device_idx} out of range (available: {torch.cuda.device_count()})"
                        )
                    # Test memory allocation
                    try:
                        t = torch.tensor([1.0], device=device_map)
                        del t
                        torch.cuda.empty_cache()
                    except Exception as e:
                        raise ValueError(
                            f"Cannot access device '{device_map}': {e}. "
                            f"Please check if the GPU is available and not in use by another process."
                        ) from e
                else:
                    raise ValueError(
                        f"CUDA is not available, but device '{device_map}' was specified. "
                        f"Please use 'cpu' or ensure CUDA is properly installed."
                    )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Could not validate device '{device_map}': {e}")

        # Load model using official Qwen3-VL cookbook method
        # https://github.com/Qwen/Qwen3-VL/cookbooks
        model_load_start = time.time()
        
        import warnings

        try:
            # Transformers >= 4.48
            from transformers import AutoModelForImageTextToText as _AutoQwenModel

            model_class_name = "AutoModelForImageTextToText"
        except Exception:
            # Backward compatibility (deprecated in newer transformers)
            from transformers import AutoModelForVision2Seq as _AutoQwenModel

            model_class_name = "AutoModelForVision2Seq"

        def _load_model_with_auto_dtype():
            base_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
            }
            # Newer transformers prefers `dtype=...`; older uses `torch_dtype=...`.
            try:
                return _AutoQwenModel.from_pretrained(model_id, dtype="auto", **base_kwargs)
            except TypeError:
                return _AutoQwenModel.from_pretrained(
                    model_id, torch_dtype="auto", **base_kwargs
                )
        
        # Suppress common transformers warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*AutoModelForVision2Seq.*deprecated.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*torch_dtype.*deprecated.*",
            )
            warnings.filterwarnings("ignore", category=UserWarning, 
                                    message=".*Torch was not compiled with flash attention.*")
            warnings.filterwarnings("ignore", 
                                    message=".*You are using a model of type.*")
            
            model = _load_model_with_auto_dtype()
        logger.info(f"Qwen3-VL loaded on {device_map} using {model_class_name}")

        model_load_time = time.time() - model_load_start
        logger.info(f"Model weights loaded in {model_load_time:.2f}s")

        model = model.eval()

        # Try to enable Torch Compile
        if (
            config.enable_compile
            and os.environ.get("CORGI_DISABLE_COMPILE", "0") != "1"
        ):
            compile_start = time.time()
            try:
                logger.info(
                    "Compiling model with torch.compile (this may take a minute)..."
                )
                model = torch.compile(model, mode="reduce-overhead")
                compile_time = time.time() - compile_start
                logger.info(f"✓ Torch compile enabled (took {compile_time:.2f}s)")
            except Exception as e:
                logger.warning(f"Torch compile failed ({e}), using uncompiled model")
        else:
            logger.info("✓ Torch compile DISABLED (enable_compile=false in config)")

        processor_start = time.time()
        # Try fast processor first, fallback to slow if unavailable
        try:
            processor = AutoProcessor.from_pretrained(
                model_id, use_fast=True, trust_remote_code=True
            )
            logger.info("✓ Using fast processor")
        except Exception as e:
            logger.warning(
                f"Fast processor not available, falling back to slow processor: {e}"
            )
            processor = AutoProcessor.from_pretrained(
                model_id, use_fast=False, trust_remote_code=True
            )
        processor_time = time.time() - processor_start
        logger.info(f"Processor loaded in {processor_time:.2f}s")

        _MODEL_CACHE[model_id] = model
        total_load_time = time.time() - model_load_start
        logger.info(f"✓ Qwen model fully loaded in {total_load_time:.2f}s total")
        _PROCESSOR_CACHE[model_id] = processor

        return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]


class Qwen3VLInstructClient:
    """
    Client for Qwen3-VL Instruct models.

    This client generates hybrid output: natural language chain-of-thought
    followed by structured JSON with steps that need visual verification.
    """

    def __init__(
        self,
        config: ModelConfig,
        extraction_method: str = "hybrid",
        image_logger=None,
        output_tracer=None,
    ):
        """
        Initialize the Qwen Instruct client.

        Args:
            config: Model configuration
            extraction_method: Method for extracting structured data ("hybrid" or "json_only")
            image_logger: Optional logger for images
            output_tracer: Optional tracer for outputs
        """
        self.config = config
        self.extraction_method = extraction_method
        self.image_logger = image_logger
        self.output_tracer = output_tracer

        # Prompt logs (used by PipelineResult / Gradio UI)
        # Keep these names aligned with SupportsVLMClient in corgi/core/pipeline.py.
        self.reasoning_log = None
        self.answer_log = None

        # Load model and processor
        # We reuse the loading logic from QwenInstructClient but wrapped in this class
        # Ideally we should inherit, but for now we'll just instantiate a helper or duplicate logic
        # Actually, let's just use the helper function or static method if we refactored
        # But since I'm rewriting the file, I can just copy the logic or make it a mixin

        # For simplicity in this fix, I will duplicate the loading logic or use a shared helper
        # But wait, the previous code had QwenInstructClient and Qwen3VLInstructClient separate?
        # Let's check the original file content I viewed.
        # It seems QwenInstructClient was the main one, and Qwen3VLInstructClient was added recently?
        # Actually, looking at the imports in factory.py, it imports Qwen3VLInstructClient.

        self._model, self._processor = self._load_backend(config)

    def _load_backend(self, config: ModelConfig) -> Tuple[Any, Any]:
        # Reuse the same loading logic
        return QwenInstructClient._load_backend(self, config)

    def generate_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> Tuple[str, List[ReasoningStep]]:
        """
        Generate reasoning steps for the given image and question (V1 pipeline).

        Args:
            image: Input image
            question: Question to answer
            max_steps: Maximum number of reasoning steps

        Returns:
            Tuple of (cot_text, steps)
        """
        # Use V1 prompts and V1 parser
        from ...utils.prompts import INSTRUCT_REASONING_PROMPT
        from ...utils.parsers import parse_hybrid_reasoning_response
        
        # Build V1 prompt
        prompt = INSTRUCT_REASONING_PROMPT.format(question=question, max_steps=max_steps)
        
        # Generate with model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                use_cache=True,
            )
        
        # Decode
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Record prompt I/O for UI/debugging.
        try:
            from ...core.types import PromptLog

            self.reasoning_log = PromptLog(
                prompt=prompt,
                response=response,
                stage="reasoning",
            )
        except Exception:
            # Logging should never break inference.
            self.reasoning_log = None
        
        # Parse V1 response (hybrid format: CoT + JSON)
        cot_text, steps = parse_hybrid_reasoning_response(response, max_steps)
        
        return cot_text, steps
    
    def structured_reasoning_v2(
        self, image: Image.Image, question: str, max_steps: int
    ) -> Tuple[str, List]:
        """
        Generate V2 structured reasoning with integrated grounding.

        Args:
            image: Input image
            question: Question to answer
            max_steps: Maximum number of reasoning steps

        Returns:
            Tuple of (cot_text, List[ReasoningStepV2])
        """
        from ...utils.prompts_v2 import build_reasoning_prompt_v2
        from ...utils.parsers_v2 import parse_structured_reasoning_v2
        
        # Build V2 prompt
        prompt = build_reasoning_prompt_v2(question, max_steps)
        
        # Generate with model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            use_cache=True,  # ✅ Enable KV cache for 30-40% speedup
            )
        
        # Decode
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Record prompt I/O for UI/debugging.
        try:
            from ...core.types import PromptLog

            self.reasoning_log = PromptLog(
                prompt=prompt,
                response=response,
                stage="reasoning_v2",
            )
        except Exception:
            self.reasoning_log = None
        
        # Parse V2 response
        cot_text, steps = parse_structured_reasoning_v2(response)
        
        logger.info(f"✓ Generated {len(steps)} reasoning steps")
        return cot_text, steps

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List,
        evidences: List,
    ) -> Tuple[str, List, Optional[str]]:
        """
        Synthesize final answer from evidences (V2 pipeline).
        
        Args:
            image: Input image
            question: Original question
            steps: List of ReasoningStepV2
            evidences: List of GroundedEvidenceV2
            
        Returns:
            Tuple of (answer, key_evidences, explanation)
        """
        from ...core.types import KeyEvidence
        
        # Format steps and evidences for prompt
        steps_text = "\n".join(
            f"{s.index}. {s.statement}" for s in steps
        )
        
        evidences_text = "\n".join(
            f"Region {i+1}: {ev.description} (bbox: {ev.bbox})"
            for i, ev in enumerate(evidences)
        )
        
        prompt = f"""Finalize the answer using verified evidence.

Question: {question}

Reasoning steps:
{steps_text}

Verified visual evidence:
{evidences_text}

Provide your final answer in JSON format:
{{
  "answer": "Your concise final answer",
  "explanation": "Brief explanation (2-3 sentences)",
  "key_evidence": [
    {{
      "bbox": [x1, y1, x2, y2],
      "description": "What this region shows",
      "reasoning": "Why this supports the answer"
    }}
  ]
}}

Use up to 3 key evidence items. Respond with JSON only."""

        # Generate answer
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]
        
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        # Move inputs to model device
        inputs = inputs.to(self._model.device)
        
        # Generate
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        use_cache=True,  # ✅ Enable KV cache for 30-40% speedup
        )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Record synthesis prompt I/O for UI/debugging.
        try:
            from ...core.types import PromptLog

            self.answer_log = PromptLog(
                prompt=prompt,
                response=response,
                stage="answer_synthesis",
            )
        except Exception:
            self.answer_log = None
        
        # Strip thinking tags if the model emits them (defensive)
        cleaned_response = response
        if "</think>" in cleaned_response:
            cleaned_response = cleaned_response.split("</think>", 1)[-1]
        cleaned_response = cleaned_response.replace("<think>", "").strip()

        # Parse JSON response
        import json
        import re
        
        def repair_json(json_str: str) -> str:
            """Attempt to repair common JSON issues from LLM output."""
            # Remove any markdown code blocks
            json_str = re.sub(r'^```json\s*', '', json_str.strip())
            json_str = re.sub(r'\s*```$', '', json_str.strip())
            json_str = json_str.replace("…", "")
            
            # Replace single quotes with double quotes (but not within strings)
            # This is a simple heuristic - replace ' with " for property names
            json_str = re.sub(r"'([^']+)':", r'"\1":', json_str)
            json_str = re.sub(r":\s*'([^']*)'([,}\]])", r': "\1"\2', json_str)
            
            # Fix trailing commas before } or ]
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            
            # Fix missing commas between elements (common LLM error)
            # Pattern: "value" followed by newline and then "key":
            json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)

            # Pattern: ] or } or number/bool/null followed by newline and then "key":
            json_str = re.sub(
                r'(\]|\}|true|false|null|-?\d+(?:\.\d+)?)\s*\n\s*"',
                r"\1,\n\"",
                json_str,
                flags=re.IGNORECASE,
            )
            
            # Fix missing commas after } or ] followed by { or "
            json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
            json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
            json_str = re.sub(r']\s*\n\s*{', '],\n{', json_str)
            json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)

            # Same repairs without newlines (sometimes the model omits line breaks)
            json_str = re.sub(r'}\s*{', '},{', json_str)
            json_str = re.sub(r'}\s*"', '},"', json_str)
            json_str = re.sub(r']\s*{', '],{', json_str)
            json_str = re.sub(r']\s*"', '],"', json_str)
            
            # Fix newlines inside string values (common in descriptions)
            # This is tricky - we need to be careful not to break valid JSON
            # Replace literal newlines inside strings with \n
            def fix_string_newlines(match):
                content = match.group(1)
                # Replace actual newlines with escaped newlines
                content = content.replace('\n', '\\n').replace('\r', '\\r')
                return f'"{content}"'
            
            # Match strings that might have newlines (between quotes, allowing for escapes)
            # Only apply to values, not the whole JSON
            json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', fix_string_newlines, json_str)
            
            return json_str
        
        try:
            # Find JSON in response
            start_idx = cleaned_response.find("{")
            end_idx = cleaned_response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_response[start_idx:end_idx+1]
                
                # Try parsing directly first
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to repair and parse again
                    repaired_json = repair_json(json_str)
                    try:
                        data = json.loads(repaired_json)
                        logger.info("JSON repaired successfully")
                    except json.JSONDecodeError as e2:
                        # Log the problematic JSON for debugging
                        logger.warning(f"JSON repair failed: {e2}")
                        logger.debug(f"Original JSON: {json_str[:500]}...")
                        raise e2
                
                answer = data.get("answer", cleaned_response)
                explanation = data.get("explanation", None)
                key_evidences = []
                
                # Parse key evidences
                for item in data.get("key_evidence", [])[:3]:
                    try:
                        bbox = item.get("bbox", [])
                        if len(bbox) == 4:
                            key_evidences.append(
                                KeyEvidence(
                                    bbox=tuple(bbox),
                                    description=item.get("description", ""),
                                    reasoning=item.get("reasoning", ""),
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Failed to parse key evidence: {e}")
                
                return answer, key_evidences, explanation
        except Exception as e:
            logger.warning(f"Failed to parse synthesis response as JSON: {e}")
            # Try to extract answer from text as final fallback
            # Look for common patterns like "answer": "..." or just return first sentence
            answer_match = re.search(
                r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
                cleaned_response,
                flags=re.DOTALL,
            )
            if answer_match:
                answer_text = answer_match.group(1).strip()
                explanation_match = re.search(
                    r'"explanation"\s*:\s*"((?:\\.|[^"\\])*)"',
                    cleaned_response,
                    flags=re.DOTALL,
                )
                explanation_text = (
                    explanation_match.group(1).strip() if explanation_match else None
                )
                return answer_text, [], explanation_text or None
            # Fallback to first meaningful sentence
            return cleaned_response.strip().split('\n')[0][:500], [], None
    
    def _chat(
        self, image: Image.Image, prompt: str, max_new_tokens: int = 512
    ) -> str:
        """
        Low-level chat method for grounding adapter compatibility.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return response
