import os
import unittest
from io import BytesIO
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image

from corgi.core.pipeline import CoRGIPipeline
from corgi.models.qwen.qwen_client import Qwen3VLClient, QwenGenerationConfig

DEMO_IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)
DEMO_QUESTION = "How many people are there in the image? Is there any one who is wearing a white watch?"


def _load_demo_image() -> Image.Image:
    local_path = os.getenv("CORGI_DEMO_IMAGE")
    if local_path:
        return Image.open(local_path).convert("RGB")
    try:
        with urlopen(DEMO_IMAGE_URL, timeout=20) as resp:
            data = resp.read()
    except Exception as exc:  # pragma: no cover - network/download errors lead to skip
        raise unittest.SkipTest(f"Unable to fetch demo image: {exc}")
    return Image.open(BytesIO(data)).convert("RGB")


@unittest.skipUnless(
    os.getenv("CORGI_RUN_QWEN_INTEGRATION") == "1",
    "Set CORGI_RUN_QWEN_INTEGRATION=1 to run the full Qwen3-VL pipeline test.",
)
class QwenRealModelIntegrationTests(unittest.TestCase):
    def test_pipeline_produces_grounded_answer(self):
        image = _load_demo_image()
        model_id = os.getenv("CORGI_QWEN_MODEL", "Qwen/Qwen3-VL-8B-Thinking")
        max_steps = int(os.getenv("CORGI_MAX_STEPS", "4"))
        max_regions = int(os.getenv("CORGI_MAX_REGIONS", "4"))

        client = Qwen3VLClient(QwenGenerationConfig(model_id=model_id))
        pipeline = CoRGIPipeline(vlm_client=client)
        result = pipeline.run(
            image=image,
            question=DEMO_QUESTION,
            max_steps=max_steps,
            max_regions=max_regions,
        )

        self.assertEqual(result.question, DEMO_QUESTION)
        self.assertGreater(
            len(result.steps), 0, "Model should return at least one reasoning step"
        )
        visual_steps = [step for step in result.steps if step.needs_vision]
        self.assertGreater(
            len(visual_steps),
            0,
            "Expected at least one step requiring visual verification",
        )
        self.assertTrue(result.answer, "Model should synthesize a non-empty answer")
        if result.evidence:
            first_ev = result.evidence[0]
            for coord in first_ev.bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)


if __name__ == "__main__":
    unittest.main()
