import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from corgi.ui.cli import build_parser, execute_cli
from corgi.core.pipeline import PipelineResult
from corgi.core.types import GroundedEvidence, ReasoningStep


class FakePipeline:
    def __init__(self, result: PipelineResult):
        self._result = result
        self.calls = []

    def run(self, image, question: str, max_steps: int, max_regions: int):
        self.calls.append(
            {
                "question": question,
                "max_steps": max_steps,
                "max_regions": max_regions,
            }
        )
        return self._result


class CliTests(unittest.TestCase):
    def _make_image(self) -> Path:
        tmp_dir = tempfile.mkdtemp()
        img_path = Path(tmp_dir) / "dummy.png"
        Image.new("RGB", (8, 8), color="white").save(img_path)
        return img_path

    def _make_result(self, question: str) -> PipelineResult:
        steps = [
            ReasoningStep(
                index=1,
                statement="The player holds a racket.",
                needs_vision=True,
                reason="Visible equipment.",
            ),
            ReasoningStep(
                index=2, statement="They are on a tennis court.", needs_vision=True
            ),
        ]
        evidences = [
            GroundedEvidence(
                step_index=1,
                bbox=(0.1, 0.2, 0.3, 0.4),
                description="Racket near the player's hand.",
                confidence=0.82,
            )
        ]
        return PipelineResult(
            question=question,
            steps=steps,
            evidence=evidences,
            answer="The player holds a tennis racket on court.",
        )

    def test_build_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--image",
                "image.jpg",
                "--question",
                "What is the player holding?",
            ]
        )
        self.assertEqual(args.image, Path("image.jpg"))
        self.assertEqual(args.question, "What is the player holding?")
        self.assertEqual(args.max_steps, 3)
        self.assertEqual(args.max_regions, 3)
        self.assertIsNone(args.json_out)
        self.assertIsNone(args.model_id)

    def test_execute_cli_formats_output_and_writes_json(self):
        img_path = self._make_image()
        result = self._make_result(question="What is the player holding?")
        pipeline = FakePipeline(result=result)

        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "out.json"
            execute_cli(
                image_path=img_path,
                question=result.question,
                max_steps=3,
                max_regions=2,
                model_id=None,
                json_out=json_path,
                pipeline_factory=lambda _model_id: pipeline,
                output_stream=buffer,
            )

            content = buffer.getvalue()
            self.assertIn("Question: What is the player holding?", content)
            self.assertIn("[1] The player holds a racket.", content)
            self.assertIn("needs vision: yes", content)
            self.assertIn("Step 1 | bbox=(0.10, 0.20, 0.30, 0.40)", content)
            self.assertIn("Answer: The player holds a tennis racket on court.", content)

            with json_path.open() as f:
                payload = json.load(f)
            self.assertIn("steps", payload)
            self.assertIn("evidence", payload)
            self.assertEqual(payload["question"], result.question)

        self.assertEqual(len(pipeline.calls), 1)
        call = pipeline.calls[0]
        self.assertEqual(call["question"], result.question)
        self.assertEqual(call["max_steps"], 3)
        self.assertEqual(call["max_regions"], 2)


if __name__ == "__main__":
    unittest.main()
