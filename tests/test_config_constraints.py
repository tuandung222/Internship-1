"""
Unit tests for CoRGi configuration constraints and helpers.
"""

from __future__ import annotations

import pytest

from corgi.core.config import (
    CaptioningConfig,
    CoRGiConfig,
    GroundingConfig,
    ModelConfig,
    ReasoningConfig,
    SynthesisConfig,
)


def _qwen_model_config(model_id: str = "Qwen/Qwen3-VL-2B-Instruct") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        model_type="qwen_instruct",
        device="cpu",
    )


def _vintern_model_config() -> ModelConfig:
    return ModelConfig(
        model_id="5CD-AI/Vintern-1B-v3_5",
        model_type="vintern",
        device="cpu",
    )


def test_qwen_vintern_config_enforces_shared_qwen_model():
    """When captioning uses Vintern, Qwen must be shared across reasoning/grounding/synthesis."""
    config = CoRGiConfig(
        reasoning=ReasoningConfig(model=_qwen_model_config()),
        grounding=GroundingConfig(model=_qwen_model_config(), max_regions=1),
        captioning=CaptioningConfig(model=_vintern_model_config()),
        synthesis=SynthesisConfig(model=_qwen_model_config()),
    )

    # Should not raise
    config.ensure_vintern_constraints()
    assert config.requires_parallel_loading() is True


def test_qwen_vintern_config_raises_when_models_differ():
    """Mismatch in Qwen model IDs should be rejected for Vintern deployments."""
    mismatched_grounding = GroundingConfig(
        model=_qwen_model_config("Qwen/Qwen3-VL-4B-Instruct"), max_regions=1
    )
    config = CoRGiConfig(
        reasoning=ReasoningConfig(model=_qwen_model_config()),
        grounding=mismatched_grounding,
        captioning=CaptioningConfig(model=_vintern_model_config()),
        synthesis=SynthesisConfig(model=_qwen_model_config()),
    )

    with pytest.raises(ValueError):
        config.ensure_vintern_constraints()
