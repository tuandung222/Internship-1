import importlib.util
import unittest

from PIL import Image

from corgi.ui.gradio_app import (
    DEFAULT_MODEL_ID,
    PipelineState,
    ensure_pipeline_state,
    format_result_markdown,
    build_demo,
)
from corgi.core.pipeline import PipelineResult
from corgi.core.types import GroundedEvidence, ReasoningStep


class DummyPipeline:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def run(
        self, image: Image.Image, question: str, max_steps: int, max_regions: int
    ):  # pragma: no cover - replaced in ensure tests
        self.calls += 1
        return None


def _make_result() -> PipelineResult:
    steps = [
        ReasoningStep(
            index=1,
            statement="The man is holding a guitar.",
            needs_vision=True,
            reason="Visible instrument.",
        ),
    ]
    evidences = [
        GroundedEvidence(
            step_index=1,
            bbox=(0.1, 0.1, 0.3, 0.3),
            description="Guitar near hands",
            confidence=0.9,
        )
    ]
    return PipelineResult(
        question="What is he holding?",
        steps=steps,
        evidence=evidences,
        answer="He is holding a guitar.",
    )


class GradioHelpersTests(unittest.TestCase):
    def test_ensure_pipeline_state_reuses_existing(self):
        created = []

        def factory(model_id):
            pipeline = DummyPipeline(name=model_id)
            created.append(pipeline)
            return pipeline

        first = ensure_pipeline_state(None, None, factory)
        self.assertEqual(first.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(len(created), 1)

        second = ensure_pipeline_state(first, DEFAULT_MODEL_ID, factory)
        self.assertIs(second.pipeline, first.pipeline)
        self.assertEqual(len(created), 1)

        third = ensure_pipeline_state(second, "custom-model", factory)
        self.assertEqual(third.model_id, "custom-model")
        self.assertIsNot(third.pipeline, first.pipeline)
        self.assertEqual(len(created), 2)

    def test_format_result_markdown_contains_sections(self):
        result = _make_result()
        markdown = format_result_markdown(result)
        self.assertIn("### Answer", markdown)
        self.assertIn("He is holding a guitar.", markdown)
        self.assertIn("### Reasoning Steps", markdown)
        self.assertIn("Step 1", markdown)
        self.assertIn("### Visual Evidence", markdown)
        self.assertIn("bbox=(0.10, 0.10, 0.30, 0.30)", markdown)

    @unittest.skipUnless(
        importlib.util.find_spec("gradio") is not None, "gradio not installed"
    )
    def test_build_demo_returns_blocks(self):
        demo = build_demo(pipeline_factory=lambda _: None)
        import gradio as gr

        self.assertIsInstance(demo, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
