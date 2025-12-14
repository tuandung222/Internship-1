"""
Qwen3-VL Thinking Model Client (Stub).

This is a placeholder for the Thinking model client.
For now, it's not fully implemented but provides the interface.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import logging

from PIL import Image

from ...core.config import ModelConfig
from ...core.types import ReasoningStep, GroundedEvidence, KeyEvidence

logger = logging.getLogger(__name__)


class Qwen3VLThinkingClient:
    """
    Client for Qwen3-VL Thinking models (stub implementation).

    This is a placeholder that raises NotImplementedError.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Qwen3-VL Thinking client (stub).

        Args:
            config: Model configuration
        """
        self.config = config
        logger.warning("Qwen3VLThinkingClient is a stub implementation")

    def structured_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> List[ReasoningStep]:
        """Generate structured reasoning steps (not implemented)."""
        raise NotImplementedError("Qwen3VLThinkingClient is not fully implemented yet")

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        evidences: List[GroundedEvidence],
    ) -> Tuple[str, List[KeyEvidence], Optional[str]]:
        """Synthesize final answer (not implemented)."""
        raise NotImplementedError("Qwen3VLThinkingClient is not fully implemented yet")


__all__ = ["Qwen3VLThinkingClient"]
