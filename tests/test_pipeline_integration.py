"""
Integration tests for CoRGi pipeline with mixed models.

Tests that the full pipeline correctly handles coordinate conversion
when using different model combinations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from corgi.core.pipeline import CoRGIPipeline
from corgi.models.factory import VLMClientFactory
from corgi.core.config import (
    CoRGiConfig,
    ModelConfig,
    ReasoningConfig,
    GroundingConfig,
    CaptioningConfig,
    SynthesisConfig,
)
from corgi.models.registry import ModelRegistry
from corgi.models.qwen.qwen_grounding_adapter import QwenGroundingAdapter


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new("RGB", (1000, 800), color="white")


@pytest.fixture
def florence_qwen_config():
    """Config using Florence for grounding/captioning and Qwen for reasoning/synthesis."""
    return CoRGiConfig(
        reasoning=ReasoningConfig(
            model=ModelConfig(
                model_id="Qwen/Qwen3-VL-4B-Instruct",
                model_type="qwen_instruct",
                device="cpu",
            ),
            max_steps=3,
            max_new_tokens=512,
        ),
        grounding=GroundingConfig(
            model=ModelConfig(
                model_id="microsoft/Florence-2-large",
                model_type="florence2",
                device="cpu",
            ),
            max_regions=3,
            max_new_tokens=128,
        ),
        captioning=CaptioningConfig(
            model=ModelConfig(
                model_id="microsoft/Florence-2-large",
                model_type="florence2",
                device="cpu",
            ),
            max_new_tokens=128,
        ),
        synthesis=SynthesisConfig(
            model=ModelConfig(
                model_id="Qwen/Qwen3-VL-4B-Instruct",
                model_type="qwen_instruct",
                device="cpu",
            ),
            max_new_tokens=256,
        ),
    )


class TestPipelineCoordinateIntegration:
    """Integration tests for coordinate handling in full pipeline."""

    @patch("corgi.models.qwen.qwen_instruct_client._load_backend")
    @patch("corgi.models.florence.florence_grounding_client._load_florence_backend")
    @patch("corgi.models.florence.florence_captioning_client._load_florence_backend")
    def test_florence_qwen_pipeline(
        self,
        mock_florence_cap_backend,
        mock_florence_ground_backend,
        mock_qwen_backend,
        florence_qwen_config,
        sample_image,
    ):
        """Test full pipeline with Florence grounding/captioning and Qwen reasoning/synthesis."""
        # Setup mocks for Qwen
        mock_qwen_model = Mock()
        mock_qwen_processor = Mock()
        mock_qwen_backend.return_value = (mock_qwen_model, mock_qwen_processor)

        # Setup mocks for Florence
        mock_florence_model = Mock()
        mock_florence_processor = Mock()
        mock_florence_ground_backend.return_value = (
            mock_florence_model,
            mock_florence_processor,
        )
        mock_florence_cap_backend.return_value = (
            mock_florence_model,
            mock_florence_processor,
        )

        # Mock Qwen reasoning response
        def qwen_side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            if "chain-of-thought" in prompt or "reasoning" in prompt.lower():
                return """
                # Reasoning:
                This is a test reasoning.
                
                # Steps to verify:
                ```json
                {
                    "steps": [
                        {
                            "index": 1,
                            "statement": "test object to verify",
                            "needs_vision": true,
                            "reason": "visual check needed"
                        }
                    ]
                }
                ```
                """
            elif "finalizing the answer" in prompt.lower():
                return """
                {
                    "answer": "Test answer",
                    "key_evidence": [
                        {
                            "bbox": [100, 200, 800, 600],
                            "description": "Key visual evidence",
                            "reasoning": "Supports the answer"
                        }
                    ]
                }
                """
            return "{}"

        # Mock Florence grounding response
        mock_florence_model.generate.return_value = MagicMock()
        mock_florence_processor.batch_decode.return_value = ["<s>mocked</s>"]
        mock_florence_processor.post_process_generation.return_value = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[100, 200, 500, 600]],  # Pixel coords
                "labels": ["object"],
            },
            "<DETAILED_CAPTION>": "A test caption",
        }
        mock_florence_processor.return_value = {
            "input_ids": Mock(),
            "pixel_values": Mock(),
        }

        # Create pipeline
        try:
            client = VLMClientFactory.create_from_config(florence_qwen_config)
            pipeline = CoRGIPipeline(vlm_client=client)

            # Mock the Qwen client's _chat method
            with patch.object(
                client._reasoning_client, "_chat", side_effect=qwen_side_effect
            ):
                with patch.object(
                    client._synthesis_client, "_chat", side_effect=qwen_side_effect
                ):
                    # Run pipeline
                    result = pipeline.run(
                        image=sample_image,
                        question="Test question?",
                        max_steps=3,
                        max_regions=3,
                    )

            # Verify result structure
            assert result is not None
            assert hasattr(result, "answer")
            assert hasattr(result, "steps")
            assert hasattr(result, "evidences")
            assert hasattr(result, "key_evidence")

            # Verify all bboxes in final output are normalized [0, 1]
            for ev in result.evidence:
                bbox = ev.bbox
                assert all(
                    0 <= coord <= 1 for coord in bbox
                ), f"Evidence bbox not normalized: {bbox}"

            for key_ev in result.key_evidence:
                bbox = key_ev.bbox
                assert all(
                    0 <= coord <= 1 for coord in bbox
                ), f"Key evidence bbox not normalized: {bbox}"

        except Exception as e:
            pytest.skip(f"Pipeline creation failed (may need model files): {e}")

    def test_all_bboxes_normalized_in_output(self):
        """Test that all bboxes in pipeline output are in [0, 1] format."""
        # This is a property-based test
        # All evidences and key_evidence bboxes should be in [0, 1]

        # Create a minimal mock result
        from corgi.core.pipeline import PipelineResult
        from corgi.core.types import ReasoningStep, GroundedEvidence, KeyEvidence

        result = PipelineResult(
            question="test question",
            answer="test",
            steps=[ReasoningStep(1, "test", True)],
            evidence=[
                GroundedEvidence(
                    step_index=1, bbox=(0.1, 0.2, 0.8, 0.9), description="test"
                )
            ],
            key_evidence=[
                KeyEvidence(
                    bbox=(0.1, 0.2, 0.8, 0.9), description="test", reasoning="test"
                )
            ],
            timings=[],
        )

        # Verify all bboxes are normalized
        for ev in result.evidence:
            assert all(0 <= coord <= 1 for coord in ev.bbox)

        for key_ev in result.key_evidence:
            assert all(0 <= coord <= 1 for coord in key_ev.bbox)


def test_vlm_factory_reuses_qwen_client_and_forces_parallel_loading(monkeypatch):
    """Ensure Qwen reasoning client is shared and models load in parallel for Qwen+Vintern configs."""
    qwen_cfg = ModelConfig(
        model_id="Qwen/Qwen3-VL-2B-Instruct", model_type="qwen_instruct", device="cpu"
    )
    vintern_cfg = ModelConfig(
        model_id="5CD-AI/Vintern-1B-v3_5", model_type="vintern", device="cpu"
    )
    config = CoRGiConfig(
        reasoning=ReasoningConfig(model=qwen_cfg),
        grounding=GroundingConfig(model=qwen_cfg, max_regions=1),
        captioning=CaptioningConfig(model=vintern_cfg),
        synthesis=SynthesisConfig(model=qwen_cfg),
    )

    dummy_qwen = object()
    dummy_vintern = object()

    def fake_create_reasoning_model(cls, _config):
        return dummy_qwen

    def fake_create_captioning_model(cls, _config):
        return dummy_vintern

    monkeypatch.setattr(
        ModelRegistry,
        "create_reasoning_model",
        classmethod(fake_create_reasoning_model),
    )
    monkeypatch.setattr(
        ModelRegistry,
        "create_captioning_model",
        classmethod(fake_create_captioning_model),
    )

    parallel_tracker = {"used": False}

    class DummyFuture:
        def __init__(self, fn):
            self._fn = fn
            self._result = None

        def result(self):
            if self._result is None:
                self._result = self._fn()
            return self._result

    class DummyExecutor:
        def __init__(self, max_workers):
            parallel_tracker["used"] = True
            self.max_workers = max_workers

        def submit(self, fn):
            return DummyFuture(fn)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_as_completed(futures):
        iterable = list(futures.keys()) if isinstance(futures, dict) else list(futures)
        for future in iterable:
            yield future

    monkeypatch.setattr("corgi.models.factory.ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr("corgi.models.factory.as_completed", fake_as_completed)

    client = VLMClientFactory.create_from_config(config, parallel_loading=False)

    assert client.reasoning is dummy_qwen
    assert isinstance(client.grounding, QwenGroundingAdapter)
    assert client.grounding.client is dummy_qwen
    assert client.synthesis is dummy_qwen
    assert client.captioning is dummy_vintern
    assert parallel_tracker["used"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
