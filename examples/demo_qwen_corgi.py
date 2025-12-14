#!/usr/bin/env python
"""Run CoRGI pipeline on the Qwen3-VL demo image and question.

Usage:
    python examples/demo_qwen_corgi.py [--model-id Qwen/Qwen3-VL-8B-Thinking]

If the demo image cannot be downloaded automatically, set the environment
variable `CORGI_DEMO_IMAGE` to a local file path.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from corgi.pipeline import CoRGIPipeline
from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig

DEMO_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
DEMO_QUESTION = "How many people are there in the image? Is there any one who is wearing a white watch?"


def fetch_demo_image() -> Image.Image:
    if path := os.getenv("CORGI_DEMO_IMAGE"):
        return Image.open(path).convert("RGB")
    with urlopen(DEMO_IMAGE_URL) as resp:  # nosec B310 - trusted URL from official demo asset
        data = resp.read()
    return Image.open(BytesIO(data)).convert("RGB")


def format_steps(pipeline_result) -> str:
    lines = ["Reasoning steps:"]
    for step in pipeline_result.steps:
        needs = "yes" if step.needs_vision else "no"
        reason = f" (reason: {step.reason})" if step.reason else ""
        lines.append(f"  [{step.index}] {step.statement} — needs vision: {needs}{reason}")
    return "\n".join(lines)


def format_evidence(pipeline_result) -> str:
    lines = ["Visual evidence:"]
    if not pipeline_result.evidence:
        lines.append("  (no evidence returned)")
        return "\n".join(lines)
    for ev in pipeline_result.evidence:
        bbox = ", ".join(f"{coord:.2f}" for coord in ev.bbox)
        desc = ev.description or "(no description)"
        conf = f", conf={ev.confidence:.2f}" if ev.confidence is not None else ""
        lines.append(f"  Step {ev.step_index}: bbox=({bbox}), desc={desc}{conf}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CoRGI pipeline with the real Qwen3-VL model.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Thinking", help="Hugging Face model id for Qwen3-VL")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=4)
    args = parser.parse_args()

    image = fetch_demo_image()
    client = Qwen3VLClient(QwenGenerationConfig(model_id=args.model_id))
    pipeline = CoRGIPipeline(client)

    result = pipeline.run(
        image=image,
        question=DEMO_QUESTION,
        max_steps=args.max_steps,
        max_regions=args.max_regions,
    )

    print(f"Question: {DEMO_QUESTION}")
    print(format_steps(result))
    print(format_evidence(result))
    print("Answer:", result.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
