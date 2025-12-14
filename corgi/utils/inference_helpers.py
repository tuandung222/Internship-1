"""
Shared utilities for CoRGI inference scripts.

This module provides common functions used by both inference.py (V1) and inference_v2.py (V2).
Reduces code duplication and ensures consistent behavior across pipeline versions.

Functions:
    - setup_output_dir: Create output directory structure
    - annotate_image_with_bboxes: Draw bounding boxes on image
    - save_evidence_crops: Save cropped evidence regions
    - save_results_json: Save JSON results
    - save_summary_report: Save human-readable report
    - load_font: Load font for annotations
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# =============================================================================
# Directory Setup
# =============================================================================


def setup_output_dir(output_dir: Path, create_subdirs: bool = True) -> Dict[str, Path]:
    """
    Create output directory structure and return paths.

    Args:
        output_dir: Base output directory path
        create_subdirs: Whether to create subdirectories

    Returns:
        Dictionary with paths: root, images, evidence, visualizations, logs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": output_dir,
        "images": output_dir / "images" if create_subdirs else output_dir,
        "evidence": output_dir / "evidence" if create_subdirs else output_dir,
        "visualizations": output_dir / "visualizations" if create_subdirs else output_dir,
        "logs": output_dir / "logs" if create_subdirs else output_dir,
    }

    if create_subdirs:
        for path in paths.values():
            path.mkdir(exist_ok=True)

    return paths


# =============================================================================
# Font Loading
# =============================================================================

# Font paths to try (cross-platform)
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",  # Arch Linux
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "C:\\Windows\\Fonts\\arial.ttf",  # Windows
]


def load_font(size: int = 16) -> Optional[ImageFont.FreeTypeFont]:
    """
    Load a font for image annotations.

    Args:
        size: Font size in points

    Returns:
        Font object or None if no font found
    """
    for font_path in FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return None


# =============================================================================
# Image Annotation
# =============================================================================

# Default color palette for bounding boxes
DEFAULT_COLORS = {
    "object": (76, 175, 80, 255),    # Green - object captioning
    "text": (244, 67, 54, 255),      # Red - OCR/text
    "default": (33, 150, 243, 255),  # Blue - default
    "none": (158, 158, 158, 255),    # Gray - fallback
}

# Step-based colors for V1 (indexed by step)
STEP_COLORS = [
    (244, 67, 54, 255),   # Red
    (255, 193, 7, 255),   # Amber
    (76, 175, 80, 255),   # Green
    (33, 150, 243, 255),  # Blue
    (156, 39, 176, 255),  # Purple
    (255, 87, 34, 255),   # Deep Orange
    (0, 150, 136, 255),   # Teal
]


def annotate_image_with_bboxes(
    image: Image.Image,
    bboxes: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    color_by: str = "type",  # "type" or "step"
) -> Image.Image:
    """
    Draw bounding boxes on image.

    Args:
        image: PIL Image to annotate
        bboxes: List of bbox dicts with keys:
            - bbox: [x1, y1, x2, y2] normalized coordinates
            - label: Optional label text
            - step_index: Step index (for step-based coloring)
            - evidence_type: "object", "text", or "none" (for type-based coloring)
        output_path: Optional path to save annotated image
        color_by: Coloring strategy - "type" or "step"

    Returns:
        Annotated image (RGBA converted to RGB)
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img.size

    font = load_font(16)

    for bbox_info in bboxes:
        bbox = bbox_info.get("bbox", [0, 0, 1, 1])
        label = bbox_info.get("label", "")
        step_index = bbox_info.get("step_index", 0)
        evidence_type = bbox_info.get("evidence_type", "default")

        # Select color based on strategy
        if color_by == "type":
            color = DEFAULT_COLORS.get(evidence_type, DEFAULT_COLORS["default"])
        else:  # color_by == "step"
            color = STEP_COLORS[step_index % len(STEP_COLORS)]

        # Convert normalized to pixel coordinates
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Draw bounding box
        line_width = max(2, int(min(width, height) * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label if provided
        if label:
            draw.text((x1 + 4, y1 + 4), label, fill=color, font=font)

    # Composite and convert to RGB
    annotated = Image.alpha_composite(img, overlay).convert("RGB")

    # Save if output path provided
    if output_path:
        annotated.save(output_path, quality=95)
        logger.info(f"Saved annotated image: {output_path}")

    return annotated


# =============================================================================
# Evidence Crops
# =============================================================================


def save_evidence_crops(
    image: Image.Image,
    evidences: List[Dict[str, Any]],
    evidence_dir: Path,
    prefix: str = "evidence",
) -> List[Path]:
    """
    Save cropped evidence regions.

    Args:
        image: Source image
        evidences: List of evidence dicts with keys:
            - bbox: [x1, y1, x2, y2] normalized coordinates
            - step_index: Step index
            - evidence_type: Optional type ("object", "text")
        evidence_dir: Directory to save crops
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    width, height = image.size
    saved_paths = []

    for idx, ev in enumerate(evidences):
        bbox = ev.get("bbox", [0, 0, 1, 1])
        step_index = ev.get("step_index", idx)
        evidence_type = ev.get("evidence_type", "")

        # Convert normalized to pixel coordinates
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Ensure valid coordinates
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        # Crop and save
        crop = image.crop((x1, y1, x2, y2))

        # Build filename
        type_suffix = f"_{evidence_type}" if evidence_type else ""
        crop_path = evidence_dir / f"{prefix}_step{step_index}{type_suffix}_region{idx}.jpg"
        crop.save(crop_path, quality=95)
        saved_paths.append(crop_path)

    logger.info(f"Saved {len(saved_paths)} evidence crops to {evidence_dir}")
    return saved_paths


# =============================================================================
# Result Saving
# =============================================================================


def save_results_json(
    result_data: Dict[str, Any],
    output_path: Path,
    pipeline_version: str = "v2",
) -> None:
    """
    Save results as JSON.

    Args:
        result_data: Dictionary with result data
        output_path: Path to save JSON file
        pipeline_version: Pipeline version string
    """
    # Add metadata if not present
    if "metadata" not in result_data:
        result_data["metadata"] = {}

    result_data["metadata"]["timestamp"] = datetime.now().isoformat()
    result_data["metadata"]["pipeline_version"] = pipeline_version

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON results: {output_path}")


def save_summary_report(
    image_path: Path,
    question: str,
    result_data: Dict[str, Any],
    output_path: Path,
    pipeline_version: str = "v2",
) -> None:
    """
    Save human-readable summary report.

    Args:
        image_path: Path to input image
        question: Question asked
        result_data: Dictionary with result data containing:
            - answer: Final answer
            - explanation: Optional explanation
            - cot_text: Optional chain of thought text
            - steps: List of reasoning steps
            - evidence: List of evidence items
            - key_evidence: Optional list of key evidence
            - total_duration_ms: Total processing time
            - v2_stats: Optional V2-specific stats
        output_path: Path to save report
        pipeline_version: Pipeline version string
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"CoRGI Pipeline {pipeline_version.upper()} Inference Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Image: {image_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        total_duration = result_data.get("total_duration_ms", 0)
        f.write(f"Total Duration: {total_duration / 1000:.2f}s\n\n")

        # V2 Stats (if present)
        v2_stats = result_data.get("v2_stats")
        if v2_stats:
            f.write("-" * 80 + "\n")
            f.write("PIPELINE V2 STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Bboxes from Phase 1: {v2_stats.get('bbox_from_phase1_count', 0)}\n")
            f.write(f"Object evidence: {v2_stats.get('object_evidence_count', 0)}\n")
            f.write(f"Text evidence: {v2_stats.get('text_evidence_count', 0)}\n\n")

        # Final Answer
        f.write("-" * 80 + "\n")
        f.write("FINAL ANSWER\n")
        f.write("-" * 80 + "\n")
        f.write(f"{result_data.get('answer', 'N/A')}\n\n")

        # Explanation
        explanation = result_data.get("explanation")
        if explanation:
            f.write("-" * 80 + "\n")
            f.write("EXPLANATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{explanation}\n\n")

        # Chain of Thought (V2)
        cot_text = result_data.get("cot_text")
        if cot_text:
            f.write("-" * 80 + "\n")
            f.write("CHAIN OF THOUGHT\n")
            f.write("-" * 80 + "\n")
            # Truncate if too long
            if len(cot_text) > 1000:
                f.write(f"{cot_text[:1000]}...\n\n")
            else:
                f.write(f"{cot_text}\n\n")

        # Reasoning Steps
        steps = result_data.get("steps", [])
        f.write("-" * 80 + "\n")
        f.write(f"REASONING STEPS ({len(steps)} steps)\n")
        f.write("-" * 80 + "\n")
        for step in steps:
            index = step.get("index", "?")
            statement = step.get("statement", "N/A")
            f.write(f"\nStep {index}: {statement}\n")

            # V1 flags
            if "needs_vision" in step:
                f.write(f"  - Needs Vision: {step.get('needs_vision', False)}\n")
            if "need_ocr" in step:
                f.write(f"  - Need OCR: {step.get('need_ocr', False)}\n")

            # V2 flags
            if "need_object_captioning" in step:
                f.write(f"  - need_object_captioning: {step.get('need_object_captioning', False)}\n")
            if "need_text_ocr" in step:
                f.write(f"  - need_text_ocr: {step.get('need_text_ocr', False)}\n")
            if "has_bbox" in step:
                f.write(f"  - Has bbox: {step.get('has_bbox', False)}\n")
            if step.get("bbox"):
                f.write(f"  - Bbox: {step.get('bbox')}\n")
        f.write("\n")

        # Evidence
        evidence = result_data.get("evidence", [])
        f.write("-" * 80 + "\n")
        f.write(f"EVIDENCE ({len(evidence)} regions)\n")
        f.write("-" * 80 + "\n")
        for idx, ev in enumerate(evidence):
            step_index = ev.get("step_index", "?")
            evidence_type = ev.get("evidence_type", "")
            type_info = f", Type: {evidence_type}" if evidence_type else ""

            f.write(f"\nEvidence {idx + 1} (Step {step_index}{type_info}):\n")

            if ev.get("statement"):
                f.write(f"  - Statement: {ev.get('statement')}\n")
            f.write(f"  - BBox: {ev.get('bbox', 'N/A')}\n")
            if ev.get("description"):
                f.write(f"  - Description: {ev.get('description')}\n")
            if ev.get("ocr_text"):
                f.write(f"  - OCR: {ev.get('ocr_text')}\n")
            if ev.get("confidence"):
                f.write(f"  - Confidence: {ev.get('confidence'):.3f}\n")
        f.write("\n")

        # Key Evidence
        key_evidence = result_data.get("key_evidence", [])
        if key_evidence:
            f.write("-" * 80 + "\n")
            f.write(f"KEY EVIDENCE ({len(key_evidence)} regions)\n")
            f.write("-" * 80 + "\n")
            for idx, ke in enumerate(key_evidence):
                f.write(f"\nKey Evidence {idx + 1}:\n")
                f.write(f"  - Description: {ke.get('description', 'N/A')}\n")
                f.write(f"  - Reasoning: {ke.get('reasoning', 'N/A')}\n")
                f.write(f"  - BBox: {ke.get('bbox', 'N/A')}\n")

        # Timings
        timings = result_data.get("timings", [])
        if timings:
            f.write("\n" + "-" * 80 + "\n")
            f.write("PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for timing in timings:
                name = timing.get("name", timing.get("stage", "unknown"))
                duration = timing.get("duration_ms", 0)
                step_index = timing.get("step_index")
                step_info = f" (step {step_index})" if step_index is not None else ""
                f.write(f"{name}{step_info}: {duration / 1000:.2f}s\n")

    logger.info(f"Saved summary report: {output_path}")


# =============================================================================
# Result Conversion Helpers
# =============================================================================


def pipeline_result_to_dict(result: Any, pipeline_version: str = "v2") -> Dict[str, Any]:
    """
    Convert a PipelineResult or PipelineResultV2 to a dictionary.

    Args:
        result: Pipeline result object
        pipeline_version: Version string

    Returns:
        Dictionary representation of the result
    """
    # If result already has to_json method, use it
    if hasattr(result, "to_json"):
        data = result.to_json()
    else:
        # Manual conversion
        data = {
            "question": getattr(result, "question", ""),
            "answer": getattr(result, "answer", ""),
            "explanation": getattr(result, "explanation", None),
            "cot_text": getattr(result, "cot_text", None),
            "paraphrased_question": getattr(result, "paraphrased_question", None),
            "total_duration_ms": getattr(result, "total_duration_ms", 0),
        }

        # Steps
        if hasattr(result, "steps"):
            data["steps"] = [
                {
                    "index": getattr(s, "index", i),
                    "statement": getattr(s, "statement", ""),
                    "needs_vision": getattr(s, "needs_vision", False),
                    "need_ocr": getattr(s, "need_ocr", False),
                    "need_object_captioning": getattr(s, "need_object_captioning", False),
                    "need_text_ocr": getattr(s, "need_text_ocr", False),
                    "has_bbox": getattr(s, "has_bbox", False),
                    "bbox": getattr(s, "bbox", None),
                    "evidence_type": getattr(s, "evidence_type", None),
                }
                for i, s in enumerate(result.steps)
            ]

        # Evidence
        if hasattr(result, "evidence"):
            data["evidence"] = [
                {
                    "step_index": getattr(e, "step_index", 0),
                    "statement": getattr(e, "statement", ""),
                    "bbox": list(getattr(e, "bbox", [])),
                    "evidence_type": getattr(e, "evidence_type", ""),
                    "description": getattr(e, "description", None),
                    "ocr_text": getattr(e, "ocr_text", None),
                    "confidence": getattr(e, "confidence", None),
                }
                for e in result.evidence
            ]

        # Key Evidence
        if hasattr(result, "key_evidence"):
            data["key_evidence"] = [
                {
                    "bbox": list(getattr(ke, "bbox", [])),
                    "description": getattr(ke, "description", ""),
                    "reasoning": getattr(ke, "reasoning", ""),
                }
                for ke in result.key_evidence
            ]

        # Timings
        if hasattr(result, "timings"):
            data["timings"] = [
                {
                    "name": getattr(t, "name", ""),
                    "duration_ms": getattr(t, "duration_ms", 0),
                    "step_index": getattr(t, "step_index", None),
                }
                for t in result.timings
            ]

        # V2 stats
        if pipeline_version == "v2":
            data["v2_stats"] = {
                "bbox_from_phase1_count": getattr(result, "bbox_from_phase1_count", 0),
                "object_evidence_count": getattr(result, "object_evidence_count", 0),
                "text_evidence_count": getattr(result, "text_evidence_count", 0),
            }

    return data


def evidence_to_bbox_list(evidences: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert evidence objects to bbox list format for annotate_image_with_bboxes.

    Args:
        evidences: List of evidence objects or dicts

    Returns:
        List of bbox dicts suitable for annotate_image_with_bboxes
    """
    bbox_list = []
    for ev in evidences:
        if isinstance(ev, dict):
            bbox = ev.get("bbox", [0, 0, 1, 1])
            step_index = ev.get("step_index", 0)
            evidence_type = ev.get("evidence_type", "default")
        else:
            bbox = list(getattr(ev, "bbox", [0, 0, 1, 1]))
            step_index = getattr(ev, "step_index", 0)
            evidence_type = getattr(ev, "evidence_type", "default")

        # Generate label
        type_char = evidence_type[0].upper() if evidence_type else "?"
        label = f"S{step_index}:{type_char}"

        bbox_list.append({
            "bbox": bbox,
            "label": label,
            "step_index": step_index,
            "evidence_type": evidence_type,
        })

    return bbox_list


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Directory setup
    "setup_output_dir",
    # Font
    "load_font",
    # Image annotation
    "annotate_image_with_bboxes",
    "DEFAULT_COLORS",
    "STEP_COLORS",
    # Evidence crops
    "save_evidence_crops",
    # Result saving
    "save_results_json",
    "save_summary_report",
    # Conversion helpers
    "pipeline_result_to_dict",
    "evidence_to_bbox_list",
]
