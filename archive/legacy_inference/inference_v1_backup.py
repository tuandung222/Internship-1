#!/usr/bin/env python3
"""
CoRGI Batch Inference Script

This script runs the CoRGI pipeline on images without requiring the Gradio UI.
All results (answer, evidence, reasoning steps, visualizations) are saved to an output folder.

Usage:
    # Single image
    python inference.py --image path/to/image.jpg --question "What is in the image?" --output results/

    # Batch processing
    python inference.py --batch images.txt --output results/

    # With specific config
    python inference.py --image image.jpg --question "..." --config configs/qwen_only.yaml --output results/

    # With production optimizations
    CORGI_LOG_LEVEL=WARNING python inference.py --image image.jpg --question "..." --output results/
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import CoRGI components
from corgi.core.pipeline import CoRGIPipeline, PipelineResult
from corgi.core.config import CoRGiConfig
from corgi.models.factory import VLMClientFactory
from corgi.utils.image_logger import ImageLogger
from corgi.utils.output_tracer import OutputTracer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("corgi.inference")


def setup_output_dir(output_dir: Path, create_subdirs: bool = True) -> dict:
    """Create output directory structure and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": output_dir,
        "images": output_dir / "images" if create_subdirs else output_dir,
        "evidence": output_dir / "evidence" if create_subdirs else output_dir,
        "visualizations": (
            output_dir / "visualizations" if create_subdirs else output_dir
        ),
        "logs": output_dir / "logs" if create_subdirs else output_dir,
    }

    if create_subdirs:
        for path in paths.values():
            path.mkdir(exist_ok=True)

    return paths


def annotate_image_with_evidence(
    image: Image.Image,
    result: PipelineResult,
    output_path: Path,
):
    """Create annotated image with all evidence bounding boxes."""
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size

    # Define colors for different steps
    colors = [
        (244, 67, 54, 255),  # red
        (255, 193, 7, 255),  # amber
        (76, 175, 80, 255),  # green
        (33, 150, 243, 255),  # blue
        (156, 39, 176, 255),  # purple
    ]

    # Try to load font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except:
        font = None

    # Draw all evidence
    for idx, ev in enumerate(result.evidence):
        color = colors[ev.step_index % len(colors)]

        # Convert normalized to pixel coordinates
        x1 = int(ev.bbox[0] * width)
        y1 = int(ev.bbox[1] * height)
        x2 = int(ev.bbox[2] * width)
        y2 = int(ev.bbox[3] * height)

        # Draw bounding box
        line_width = max(2, int(min(width, height) * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label
        label = f"S{ev.step_index}"
        draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)

    # Composite and save
    annotated = Image.alpha_composite(img, overlay).convert("RGB")
    annotated.save(output_path, quality=95)
    logger.info(f"Saved annotated image: {output_path}")


def save_evidence_crops(
    image: Image.Image,
    result: PipelineResult,
    evidence_dir: Path,
    prefix: str = "evidence",
):
    """Save individual crops of each evidence region."""
    width, height = image.size

    for idx, ev in enumerate(result.evidence):
        # Convert normalized to pixel coordinates
        x1 = int(ev.bbox[0] * width)
        y1 = int(ev.bbox[1] * height)
        x2 = int(ev.bbox[2] * width)
        y2 = int(ev.bbox[3] * height)

        # Ensure valid coordinates
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        # Crop and save
        crop = image.crop((x1, y1, x2, y2))
        crop_path = evidence_dir / f"{prefix}_step{ev.step_index}_region{idx}.jpg"
        crop.save(crop_path, quality=95)

    logger.info(f"Saved {len(result.evidence)} evidence crops to {evidence_dir}")


def save_detailed_results(
    result: PipelineResult,
    output_path: Path,
):
    """Save detailed JSON results with all pipeline information."""
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_duration_ms": result.total_duration_ms,
        },
        "question": result.question,
        "answer": result.answer,
        "explanation": result.explanation,
        "paraphrased_question": result.paraphrased_question,
        "reasoning_steps": [
            {
                "index": step.index,
                "statement": step.statement,
                "needs_vision": step.needs_vision,
                "need_ocr": step.need_ocr,
            }
            for step in result.steps
        ],
        "evidence": [
            {
                "step_index": ev.step_index,
                "bbox": list(ev.bbox),
                "description": ev.description,
                "ocr_text": ev.ocr_text,
                "confidence": ev.confidence,
            }
            for ev in result.evidence
        ],
        "key_evidence": [
            {
                "bbox": list(ke.bbox),
                "description": ke.description,
                "reasoning": ke.reasoning,
            }
            for ke in result.key_evidence
        ],
        "timings": [
            {
                "stage": timing.name,
                "duration_ms": timing.duration_ms,
                "step_index": timing.step_index,
            }
            for timing in result.timings
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved detailed results: {output_path}")


def save_summary_report(
    image_path: Path,
    question: str,
    result: PipelineResult,
    output_path: Path,
):
    """Save human-readable summary report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CoRGI Pipeline Inference Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {result.total_duration_ms/1000:.2f}s\n\n")

        f.write("-" * 80 + "\n")
        f.write("FINAL ANSWER\n")
        f.write("-" * 80 + "\n")
        f.write(f"{result.answer}\n\n")

        if result.explanation:
            f.write("-" * 80 + "\n")
            f.write("EXPLANATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{result.explanation}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"REASONING STEPS ({len(result.steps)} steps)\n")
        f.write("-" * 80 + "\n")
        for step in result.steps:
            f.write(f"\nStep {step.index}: {step.statement}\n")
            f.write(f"  - Needs Vision: {step.needs_vision}\n")
            f.write(f"  - Need OCR: {step.need_ocr}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write(f"EVIDENCE ({len(result.evidence)} regions)\n")
        f.write("-" * 80 + "\n")
        for idx, ev in enumerate(result.evidence):
            f.write(f"\nEvidence {idx + 1} (Step {ev.step_index}):\n")
            f.write(f"  - BBox: {ev.bbox}\n")
            if ev.description:
                f.write(f"  - Description: {ev.description}\n")
            if ev.ocr_text:
                f.write(f"  - OCR: {ev.ocr_text}\n")
            if ev.confidence:
                f.write(f"  - Confidence: {ev.confidence:.3f}\n")
        f.write("\n")

        if result.key_evidence:
            f.write("-" * 80 + "\n")
            f.write(f"KEY EVIDENCE ({len(result.key_evidence)} regions)\n")
            f.write("-" * 80 + "\n")
            for idx, ke in enumerate(result.key_evidence):
                f.write(f"\nKey Evidence {idx + 1}:\n")
                f.write(f"  - Description: {ke.description}\n")
                f.write(f"  - Reasoning: {ke.reasoning}\n")
                f.write(f"  - BBox: {ke.bbox}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for timing in result.timings:
            step_info = (
                f" (step {timing.step_index})" if timing.step_index is not None else ""
            )
            f.write(f"{timing.name}{step_info}: {timing.duration_ms/1000:.2f}s\n")

    logger.info(f"Saved summary report: {output_path}")


def run_inference(
    image_path: Path,
    question: str,
    pipeline: CoRGIPipeline,
    output_dir: Path,
    save_crops: bool = True,
    save_visualization: bool = True,
) -> PipelineResult:
    """Run inference on a single image and save all results."""
    logger.info(f"Processing: {image_path}")
    logger.info(f"Question: {question}")

    # Setup output directories
    paths = setup_output_dir(output_dir)

    # Load image
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image.size}")

    # Run pipeline
    start_time = time.time()
    result = pipeline.run(
        image=image,
        question=question,
        max_steps=6,
        max_regions=5,
    )
    inference_time = time.time() - start_time

    logger.info(f"Inference completed in {inference_time:.2f}s")
    logger.info(f"Answer: {result.answer}")

    # Save original image
    image.save(paths["images"] / "original.jpg", quality=95)

    # Save annotated image with all evidence
    if save_visualization and result.evidence:
        annotate_image_with_evidence(
            image,
            result,
            paths["visualizations"] / "annotated.jpg",
        )

    # Save individual evidence crops
    if save_crops and result.evidence:
        save_evidence_crops(
            image,
            result,
            paths["evidence"],
        )

    # Save detailed JSON results
    save_detailed_results(result, paths["root"] / "results.json")

    # Save human-readable summary
    save_summary_report(
        image_path,
        question,
        result,
        paths["root"] / "summary.txt",
    )

    logger.info(f"All results saved to: {output_dir}")
    return result


def batch_inference(
    batch_file: Path,
    pipeline: CoRGIPipeline,
    output_root: Path,
):
    """Run inference on a batch of images from a file."""
    # Read batch file (format: image_path|question)
    with open(batch_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    logger.info(f"Processing {len(lines)} images from {batch_file}")

    results = []
    for idx, line in enumerate(lines, 1):
        try:
            parts = line.split("|", 1)
            if len(parts) != 2:
                logger.warning(f"Skipping invalid line {idx}: {line}")
                continue

            image_path = Path(parts[0].strip())
            question = parts[1].strip()

            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                continue

            # Create output directory for this image
            output_dir = output_root / f"result_{idx:04d}_{image_path.stem}"

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {idx}/{len(lines)}: {image_path.name}")
            logger.info(f"{'='*80}")

            result = run_inference(
                image_path,
                question,
                pipeline,
                output_dir,
            )
            results.append(
                {
                    "index": idx,
                    "image": str(image_path),
                    "question": question,
                    "answer": result.answer,
                    "duration_ms": result.total_duration_ms,
                }
            )

        except Exception as e:
            logger.error(f"Failed to process line {idx}: {e}", exc_info=True)
            continue

    # Save batch summary
    batch_summary = output_root / "batch_summary.json"
    with open(batch_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(lines),
                "processed": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"\nBatch processing complete: {len(results)}/{len(lines)} successful")
    logger.info(f"Batch summary saved to: {batch_summary}")


def main():
    parser = argparse.ArgumentParser(
        description="CoRGI Batch Inference - Run pipeline without UI and save all results"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to single input image",
    )
    input_group.add_argument(
        "--batch",
        type=Path,
        help="Path to batch file (format: image_path|question per line)",
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (required for single image mode)",
    )

    # Pipeline configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen_only.yaml"),
        help="Path to pipeline config YAML (default: configs/qwen_only.yaml)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_results"),
        help="Output directory for results (default: inference_results)",
    )

    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Skip saving individual evidence crops",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip creating annotated visualizations",
    )

    # Pipeline parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum reasoning steps (default: 6)",
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=5,
        help="Maximum regions per step (default: 5)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.image and not args.question:
        parser.error("--question is required when using --image")

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load pipeline
    logger.info("=" * 80)
    logger.info("CoRGI Batch Inference")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info("Loading pipeline...")

    try:
        config = CoRGiConfig.from_yaml(str(args.config))
        client = VLMClientFactory.create_from_config(
            config,
            parallel_loading=True,
        )
        pipeline = CoRGIPipeline(vlm_client=client)
        logger.info("✓ Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Run inference
    try:
        if args.batch:
            # Batch mode
            batch_inference(args.batch, pipeline, args.output)
        else:
            # Single image mode
            if not args.image.exists():
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)

            run_inference(
                args.image,
                args.question,
                pipeline,
                args.output,
                save_crops=not args.no_crops,
                save_visualization=not args.no_visualization,
            )

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Inference complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
