"""
Model Registry for CoRGI pipeline.

Provides registration and factory methods for VLM models.
"""

from __future__ import annotations

from typing import Dict, Type, Callable, Optional
import logging

from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for VLM model classes."""

    _reasoning_models: Dict[str, Type] = {}
    _grounding_models: Dict[str, Type] = {}
    _captioning_models: Dict[str, Type] = {}
    _synthesis_models: Dict[str, Type] = {}

    @classmethod
    def register_reasoning(cls, model_type: str) -> Callable[[Type], Type]:
        """Register a reasoning model class."""

        def decorator(model_class: Type) -> Type:
            cls._reasoning_models[model_type] = model_class
            logger.debug(
                f"Registered reasoning model: {model_type} -> {model_class.__name__}"
            )
            return model_class

        return decorator

    @classmethod
    def register_grounding(cls, model_type: str) -> Callable[[Type], Type]:
        """Register a grounding model class."""

        def decorator(model_class: Type) -> Type:
            cls._grounding_models[model_type] = model_class
            logger.debug(
                f"Registered grounding model: {model_type} -> {model_class.__name__}"
            )
            return model_class

        return decorator

    @classmethod
    def register_captioning(cls, model_type: str) -> Callable[[Type], Type]:
        """Register a captioning model class."""

        def decorator(model_class: Type) -> Type:
            cls._captioning_models[model_type] = model_class
            logger.debug(
                f"Registered captioning model: {model_type} -> {model_class.__name__}"
            )
            return model_class

        return decorator

    @classmethod
    def register_synthesis(cls, model_type: str) -> Callable[[Type], Type]:
        """Register a synthesis model class."""

        def decorator(model_class: Type) -> Type:
            cls._synthesis_models[model_type] = model_class
            logger.debug(
                f"Registered synthesis model: {model_type} -> {model_class.__name__}"
            )
            return model_class

        return decorator

    @classmethod
    def create_reasoning_model(cls, config: ModelConfig):
        """Create a reasoning model instance."""
        model_type = config.model_type
        if model_type not in cls._reasoning_models:
            raise ValueError(
                f"Reasoning model type '{model_type}' not registered. Available: {list(cls._reasoning_models.keys())}"
            )

        model_class = cls._reasoning_models[model_type]
        return model_class(config)

    @classmethod
    def create_grounding_model(cls, config: ModelConfig):
        """Create a grounding model instance."""
        model_type = config.model_type
        if model_type not in cls._grounding_models:
            raise ValueError(
                f"Grounding model type '{model_type}' not registered. Available: {list(cls._grounding_models.keys())}"
            )

        model_class = cls._grounding_models[model_type]
        return model_class(config)

    @classmethod
    def create_captioning_model(cls, config: ModelConfig, **kwargs):
        """Create a captioning model instance."""
        model_type = config.model_type
        if model_type not in cls._captioning_models:
            raise ValueError(
                f"Captioning model type '{model_type}' not registered. Available: {list(cls._captioning_models.keys())}"
            )

        model_class = cls._captioning_models[model_type]
        return model_class(config, **kwargs)

    @classmethod
    def create_synthesis_model(cls, config: ModelConfig):
        """Create a synthesis model instance."""
        model_type = config.model_type
        if model_type not in cls._synthesis_models:
            raise ValueError(
                f"Synthesis model type '{model_type}' not registered. Available: {list(cls._synthesis_models.keys())}"
            )

        model_class = cls._synthesis_models[model_type]
        return model_class(config)

    @staticmethod
    def _detect_model_type(model_id: str) -> str:
        """Detect model type from model ID."""
        model_id_lower = model_id.lower()

        if "qwen" in model_id_lower:
            if "instruct" in model_id_lower:
                return "qwen_instruct"
            elif "thinking" in model_id_lower:
                return "qwen_thinking"
            else:
                # Default to instruct for Qwen models
                return "qwen_instruct"
        elif "florence" in model_id_lower or "florence-2" in model_id_lower:
            return "florence2"
        elif "paddleocr" in model_id_lower or "paddlepaddle" in model_id_lower:
            return "paddleocr"
        elif "fastvlm" in model_id_lower:
            return "fastvlm"
        elif "vintern" in model_id_lower or "5cd-ai" in model_id_lower:
            return "vintern"  # Deprecated, kept for backward compatibility
        else:
            # Default fallback
            return "qwen_instruct"


__all__ = ["ModelRegistry"]
