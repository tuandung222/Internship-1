import unittest
from typing import List

from PIL import Image

from corgi.core.pipeline import CoRGIPipeline
from corgi.core.types import GroundedEvidence, ReasoningStep


class FakeQwenClient:
    def __init__(self):
        self.extract_calls: List[int] = []
        self.answer_requested = False

    def structured_reasoning(
        self, image: Image.Image, question: str, max_steps: int
    ) -> List[ReasoningStep]:
        return [
            ReasoningStep(
                index=1,
                statement="The person holds a tennis racket.",
                needs_vision=True,
            ),
            ReasoningStep(
                index=2, statement="Players hold rackets in tennis.", needs_vision=False
            ),
            ReasoningStep(
                index=3, statement="The court surface is clay.", needs_vision=True
            ),
        ]

    def extract_step_evidence(
        self, image: Image.Image, question: str, step: ReasoningStep, max_regions: int
    ) -> List[GroundedEvidence]:
        self.extract_calls.append(step.index)
        if step.index == 1:
            return [
                GroundedEvidence(
                    step_index=1,
                    bbox=(0.1, 0.2, 0.4, 0.6),
                    description="Close-up of a racket in the player's hand.",
                    confidence=0.88,
                )
            ]
        if step.index == 3:
            return []
        raise AssertionError("Unexpected step")

    def synthesize_answer(
        self,
        image: Image.Image,
        question: str,
        steps: List[ReasoningStep],
        evidences: List[GroundedEvidence],
    ) -> str:
        self.answer_requested = True
        assert any(ev.step_index == 1 for ev in evidences)
        assert all(
            step.needs_vision or step.index not in self.extract_calls for step in steps
        )
        return "The player is holding a tennis racket on a clay court."


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (10, 10), color="white")


class PipelineFlowTests(unittest.TestCase):
    def test_pipeline_runs_full_flow(self):
        client = FakeQwenClient()
        pipeline = CoRGIPipeline(vlm_client=client)

        result = pipeline.run(
            image=_dummy_image(),
            question="What is the player holding and where are they playing?",
            max_steps=3,
            max_regions=2,
        )

        self.assertTrue(client.answer_requested)
        self.assertEqual(client.extract_calls, [1, 3])
        self.assertEqual(
            result.answer, "The player is holding a tennis racket on a clay court."
        )
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.evidence[0].bbox, (0.1, 0.2, 0.4, 0.6))
        self.assertEqual(result.evidence[0].step_index, 1)
        json_blob = result.to_json()
        self.assertIn("steps", json_blob)
        self.assertIn("evidence", json_blob)
        self.assertIn("answer", json_blob)


if __name__ == "__main__":
    unittest.main()
