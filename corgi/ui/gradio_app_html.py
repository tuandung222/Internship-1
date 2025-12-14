from __future__ import annotations

import logging
import itertools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import spaces  # type: ignore
except ImportError:  # pragma: no cover - spaces library only on HF Spaces
    spaces = None  # type: ignore

from PIL import Image, ImageDraw
import base64
from io import BytesIO

from .cli import DEFAULT_MODEL_ID
from ..core.pipeline import CoRGIPipeline, PipelineResult
from ..core.config import CoRGiConfig
from ..models.factory import VLMClientFactory
from ..core.types import GroundedEvidence, KeyEvidence, PromptLog, ReasoningStep
from ..utils.prompts import (
    INSTRUCT_REASONING_PROMPT,
    QWEN_GROUNDING_PROMPT,
    QWEN_CAPTIONING_PROMPT,
    ANSWER_SYNTHESIS_PROMPT,
)


@dataclass
class PipelineState:
    model_id: str
    pipeline: Optional[CoRGIPipeline]


_PIPELINE_CACHE: dict[str, CoRGIPipeline] = {}
_GLOBAL_FACTORY: Callable[[Optional[str]], CoRGIPipeline] | None = None
logger = logging.getLogger("corgi.gradio_app")

# Default config paths
# Config directory lives at repository root (../configs from this file)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
DEFAULT_QWEN_CONFIG = DEFAULT_CONFIG_DIR / "qwen_only.yaml"
DEFAULT_FLORENCE_QWEN_CONFIG = DEFAULT_CONFIG_DIR / "florence_qwen.yaml"

MAX_UI_STEPS = 6
GALLERY_MAX_DIM = 768
EVIDENCE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (244, 67, 54),  # red
    (255, 193, 7),  # amber
    (76, 175, 80),  # green
    (33, 150, 243),  # blue
    (156, 39, 176),  # purple
    (255, 87, 34),  # deep orange
)

# Key evidence colors (gold/yellow tones to indicate importance)
KEY_EVIDENCE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 215, 0),  # gold
    (255, 193, 7),  # amber
    (255, 165, 0),  # orange
    (255, 223, 0),  # yellow
    (218, 165, 32),  # goldenrod
    (255, 200, 0),  # golden yellow
)

try:
    _THUMBNAIL_RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow < 9.1
    _THUMBNAIL_RESAMPLE = Image.LANCZOS  # type: ignore


def _create_pipeline_from_config_v2(
    config_path: Optional[str] = None,
    config: Optional[CoRGiConfig] = None,
    parallel_loading: bool = True,
    image_logger=None,
    output_tracer=None,
) -> CoRGIPipeline:
    """
    Create pipeline from config using VLMClientFactory.

    Args:
        config_path: Path to YAML config file (relative to configs/ or absolute)
        config: Optional CoRGiConfig object (takes precedence over config_path)
        parallel_loading: Enable parallel model loading (default: True)
        image_logger: Optional ImageLogger instance
        output_tracer: Optional OutputTracer instance

    Returns:
        CoRGIPipeline instance
    """
    if config is None:
        if config_path is None:
            config_path = str(DEFAULT_QWEN_CONFIG)

        # Handle relative paths
        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            # Try relative to configs directory first
            full_path = DEFAULT_CONFIG_DIR / config_path_obj
            if not full_path.exists():
                # Try as absolute path
                full_path = config_path_obj
            config_path = str(full_path)

        try:
            config = CoRGiConfig.from_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    # Enforce deployment constraints (e.g., Qwen sharing when using Vintern)
    try:
        config.ensure_vintern_constraints()
    except Exception as constraint_error:
        logger.error(f"Config validation failed: {constraint_error}")
        raise

    # Note: Previously force-enabled parallel loading. This caused "meta tensor" errors.
    # Now we respect the user's choice for stability.
    if config.requires_parallel_loading():
        logger.info(
            f"Multiple distinct models detected. Loading mode: {'parallel' if parallel_loading else 'sequential'}"
        )

    # Create VLM client using factory
    client = VLMClientFactory.create_from_config(
        config,
        image_logger=image_logger,
        output_tracer=output_tracer,
        parallel_loading=parallel_loading,
    )

    # Create pipeline
    return CoRGIPipeline(
        vlm_client=client,
        image_logger=image_logger,
        output_tracer=output_tracer,
    )


# Backward compatibility: keep old factory for now
def _default_factory(
    model_id: Optional[str], use_florence: bool = False
) -> CoRGIPipeline:
    """Legacy factory for backward compatibility."""
    logger.warning(
        "Using legacy _default_factory. Consider migrating to _create_pipeline_from_config_v2"
    )
    # Use default Qwen config
    return _create_pipeline_from_config_v2(
        config_path=str(DEFAULT_QWEN_CONFIG),
        parallel_loading=True,
    )


def _warm_default_pipeline() -> None:
    """Preload default pipeline."""
    cache_key = _make_cache_key(
        config_path=str(DEFAULT_QWEN_CONFIG),
        parallel_loading=True,
        batch_captioning=True,
        model_id_override=None,
    )
    if cache_key in _PIPELINE_CACHE:
        return
    try:
        logger.info("Preloading default pipeline")
        # Use regular version for warm-up (GPU context not needed for preload)
        # GPU-decorated version will be used when actually creating pipeline in _run_pipeline
        _PIPELINE_CACHE[cache_key] = _create_pipeline_from_config_v2(
            config_path=str(DEFAULT_QWEN_CONFIG),
            parallel_loading=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to preload default pipeline: %s", exc)


def _make_cache_key(
    config_path: str,
    parallel_loading: bool,
    batch_captioning: bool,
    model_id_override: Optional[str],
) -> str:
    """Generate cache key from config and optimization flags."""
    parts = [
        config_path,
        f"parallel_{parallel_loading}",
        f"batch_{batch_captioning}",
    ]
    if model_id_override:
        parts.append(f"override_{model_id_override}")
    # Create hash for shorter key
    key_str = "_".join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


_GLOBAL_FACTORY = _default_factory  # type: ignore[assignment]
# Disable automatic warm-up - let build_demo() handle it with the correct config
# _warm_default_pipeline()


def _get_pipeline(
    cache_key: str,
    factory_fn: Callable[[], CoRGIPipeline],
) -> CoRGIPipeline:
    """Get or create pipeline from cache."""
    pipeline = _PIPELINE_CACHE.get(cache_key)
    if pipeline is None:
        logger.info("Creating new pipeline with cache_key=%s", cache_key)
        pipeline = factory_fn()
        _PIPELINE_CACHE[cache_key] = pipeline
    else:
        logger.debug("Reusing cached pipeline with cache_key=%s", cache_key)
    return pipeline


# @spaces.GPU(duration=120)


def _execute_pipeline(
    image: Image.Image,
    question: str,
    max_steps: int,
    max_regions: int,
    pipeline: CoRGIPipeline,
) -> PipelineResult:
    """Execute pipeline with given parameters."""
    logger.info(
        "Executing pipeline | max_steps=%s | max_regions=%s",
        max_steps,
        max_regions,
    )
    return pipeline.run(
        image=image,
        question=question,
        max_steps=max_steps,
        max_regions=max_regions,
    )


def _group_evidence_by_step(
    evidences: List[GroundedEvidence],
) -> Dict[int, List[GroundedEvidence]]:
    grouped: Dict[int, List[GroundedEvidence]] = {}
    for ev in evidences:
        grouped.setdefault(ev.step_index, []).append(ev)
    return grouped


def _format_evidence_caption(evidence: GroundedEvidence) -> str:
    bbox_str = ", ".join(f"{coord:.2f}" for coord in evidence.bbox)
    parts = [f"Step {evidence.step_index}"]
    if evidence.description:
        parts.append(f"Caption: {evidence.description}")
    if evidence.ocr_text:
        parts.append(f"OCR: {evidence.ocr_text}")
    if evidence.confidence is not None:
        parts.append(f"Confidence: {evidence.confidence:.2f}")
    parts.append(f"BBox: ({bbox_str})")
    return "\n".join(parts)


def _crop_evidence_image(
    image: Image.Image,
    evidence: GroundedEvidence,
) -> Image.Image:
    """Crop image to evidence bounding box region."""
    x1, y1, x2, y2 = evidence.bbox
    w, h = image.size

    # Convert normalized to pixel coordinates
    left = int(x1 * w)
    top = int(y1 * h)
    right = int(x2 * w)
    bottom = int(y2 * h)

    # Ensure valid coordinates
    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    cropped = image.crop((left, top, right, bottom))

    # Optionally add border to cropped image for clarity
    from PIL import ImageDraw

    bordered = Image.new(
        "RGB", (cropped.width + 4, cropped.height + 4), (255, 255, 255)
    )
    bordered.paste(cropped, (2, 2))
    draw = ImageDraw.Draw(bordered)
    draw.rectangle(
        [0, 0, bordered.width - 1, bordered.height - 1],
        outline=(200, 200, 200),
        width=2,
    )

    return bordered


def _annotate_evidence_image(
    image: Image.Image,
    evidence: GroundedEvidence,
    color: Tuple[int, int, int],
) -> Image.Image:
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size

    x1 = max(0, min(int(evidence.bbox[0] * width), width - 1))
    y1 = max(0, min(int(evidence.bbox[1] * height), height - 1))
    x2 = max(0, min(int(evidence.bbox[2] * width), width - 1))
    y2 = max(0, min(int(evidence.bbox[3] * height), height - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    outline_width = max(2, int(min(width, height) * 0.005))
    rgba_color = color + (255,)
    fill_color = color + (64,)

    draw.rectangle(
        [x1, y1, x2, y2], fill=fill_color, outline=rgba_color, width=outline_width
    )
    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    if max(annotated.size) > GALLERY_MAX_DIM:
        annotated.thumbnail((GALLERY_MAX_DIM, GALLERY_MAX_DIM), _THUMBNAIL_RESAMPLE)
    return annotated


def _empty_ui_payload(message: str) -> Dict[str, object]:
    placeholder_prompt = f"```text\n{message}\n```"
    css = _create_css_stylesheet()
    empty_html = f'<div class="corgi-section"><p><em>{message}</em></p></div>'
    return {
        # Core payload fields used by gradio_app.py
        "input_image": None,
        "final_answer": message,
        "explanation": "",
        "paraphrased_question": "",
        "cot_text": message,
        "structured_steps": message,
        "reasoning_prompt": placeholder_prompt,
        "roi_overview_image": None,
        "roi_gallery": [],
        "grounding_prompts": placeholder_prompt,
        "evidence_table": message,
        "cropped_images_gallery": [],
        "ocr_table": message,
        "ocr_gallery": [],
        "captioning_table": message,
        "captioning_gallery": [],
        "key_evidence_overview_image": None,
        "key_evidence_gallery": [],
        "key_evidence_text": message,
        "answer_prompt": placeholder_prompt,
        "raw_outputs": message,
        "timing": message,
        # Markdown (backward compatibility)
        "answer_markdown": f"### Final Answer\n{message}",
        "chain_markdown": message,
        "chain_prompt": placeholder_prompt,
        "roi_prompt": placeholder_prompt,
        "evidence_markdown": message,
        "evidence_prompt": placeholder_prompt,
        "answer_process_markdown": message,
        "key_evidence_markdown": message,
        "timing_markdown": message,
        # HTML (new enhanced UI)
        "answer_html": css
        + f'<div class="corgi-section"><h3>Final Answer</h3><p>{message}</p></div>',
        "answer_synthesis_html": css
        + f'<div class="corgi-section"><h3>Final Answer</h3><p>{message}</p></div>',
        "chain_cot_html": css + empty_html,
        "chain_steps_html": empty_html,
        "chain_prompt_html": css + empty_html,
        "evidence_table_html": css + empty_html,
        "evidence_prompt_html": css + empty_html,
        "answer_prompt_html": css + empty_html,
        "roi_prompt_html": css + empty_html,
        # Unified markdown document
        "unified_markdown": f"# CoRGI Inference Pipeline\n\n{message}",
    }


def _annotate_overview_image(
    image: Image.Image, evidences: List[GroundedEvidence]
) -> Optional[Image.Image]:
    if not evidences:
        return None
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size

    step_colors: Dict[int, Tuple[int, int, int]] = {}
    color_cycle = itertools.cycle(EVIDENCE_COLORS)
    for ev in evidences:
        color = step_colors.setdefault(ev.step_index, next(color_cycle))
        x1 = max(0, min(int(ev.bbox[0] * width), width - 1))
        y1 = max(0, min(int(ev.bbox[1] * height), height - 1))
        x2 = max(0, min(int(ev.bbox[2] * width), width - 1))
        y2 = max(0, min(int(ev.bbox[3] * height), height - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        outline_width = max(
            3, int(min(width, height) * 0.008)
        )  # Thicker lines for better visibility
        rgba_color = color + (255,)
        fill_color = color + (60,)
        draw.rectangle([x1, y1, x2, y2], outline=rgba_color, width=outline_width)
        label = f"S{ev.step_index}"
        # Larger font for better visibility
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=max(14, int(min(width, height) * 0.02)),
            )
        except:
            font = None
        draw.text((x1 + 4, y1 + 4), label, fill=rgba_color, font=font)

    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    # For ROI overview, keep larger size (max 1200px) for better visibility of bounding boxes
    ROI_OVERVIEW_MAX_DIM = 1200
    if max(annotated.size) > ROI_OVERVIEW_MAX_DIM:
        annotated.thumbnail(
            (ROI_OVERVIEW_MAX_DIM, ROI_OVERVIEW_MAX_DIM), _THUMBNAIL_RESAMPLE
        )
    return annotated


def _annotate_key_evidence_image(
    image: Image.Image,
    key_evidence: KeyEvidence,
    color: Tuple[int, int, int],
) -> Image.Image:
    """Annotate image with a single key evidence bbox."""
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size

    x1 = max(0, min(int(key_evidence.bbox[0] * width), width - 1))
    y1 = max(0, min(int(key_evidence.bbox[1] * height), height - 1))
    x2 = max(0, min(int(key_evidence.bbox[2] * width), width - 1))
    y2 = max(0, min(int(key_evidence.bbox[3] * height), height - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    outline_width = max(
        3, int(min(width, height) * 0.006)
    )  # Slightly thicker for key evidence
    rgba_color = color + (255,)
    fill_color = color + (80,)  # Slightly more opaque fill

    draw.rectangle(
        [x1, y1, x2, y2], fill=fill_color, outline=rgba_color, width=outline_width
    )

    # Add label with short description
    label = (
        key_evidence.description[:20] + "..."
        if len(key_evidence.description) > 20
        else key_evidence.description
    )
    draw.text((x1 + 4, y1 + 4), label, fill=rgba_color)

    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    if max(annotated.size) > GALLERY_MAX_DIM:
        annotated.thumbnail((GALLERY_MAX_DIM, GALLERY_MAX_DIM), _THUMBNAIL_RESAMPLE)
    return annotated


def _annotate_key_evidence_overview(
    image: Image.Image, key_evidences: List[KeyEvidence]
) -> Optional[Image.Image]:
    """Annotate image with all key evidence bboxes."""
    if not key_evidences:
        return None
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size

    color_cycle = itertools.cycle(KEY_EVIDENCE_COLORS)
    for idx, kev in enumerate(key_evidences):
        color = next(color_cycle)
        x1 = max(0, min(int(kev.bbox[0] * width), width - 1))
        y1 = max(0, min(int(kev.bbox[1] * height), height - 1))
        x2 = max(0, min(int(kev.bbox[2] * width), width - 1))
        y2 = max(0, min(int(kev.bbox[3] * height), height - 1))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        outline_width = max(3, int(min(width, height) * 0.006))
        rgba_color = color + (255,)
        fill_color = color + (70,)
        draw.rectangle([x1, y1, x2, y2], outline=rgba_color, width=outline_width)
        label = f"Key {idx + 1}"
        draw.text((x1 + 4, y1 + 4), label, fill=rgba_color)

    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    if max(annotated.size) > GALLERY_MAX_DIM:
        annotated.thumbnail((GALLERY_MAX_DIM, GALLERY_MAX_DIM), _THUMBNAIL_RESAMPLE)
    return annotated


def _format_key_evidence_caption(key_evidence: KeyEvidence, index: int) -> str:
    """Format caption for key evidence gallery item."""
    bbox_str = ", ".join(f"{coord:.2f}" for coord in key_evidence.bbox)
    parts = [f"Key Evidence {index + 1}"]
    parts.append(f"Description: {key_evidence.description}")
    parts.append(f"Reasoning: {key_evidence.reasoning}")
    parts.append(f"BBox: ({bbox_str})")
    return "\n".join(parts)


def _create_css_stylesheet() -> str:
    """Create CSS stylesheet for UI components."""
    return """
    <style>
    .corgi-introduction {
        max-width: 100%;
        line-height: 1.6;
        margin-bottom: 30px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .corgi-introduction h1 {
        color: #007bff;
        border-bottom: 3px solid #007bff;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .corgi-introduction h2 {
        color: #333;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0e0e0;
    }
    .corgi-introduction h3 {
        color: #555;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .corgi-introduction table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .corgi-introduction table th {
        background: #007bff;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .corgi-introduction table td {
        padding: 12px;
        border-bottom: 1px solid #e0e0e0;
        vertical-align: top;
    }
    .corgi-introduction table td ul {
        margin: 8px 0;
        padding-left: 20px;
    }
    .corgi-introduction table td ul li {
        margin: 6px 0;
        line-height: 1.5;
    }
    .corgi-introduction table tr:hover {
        background: #f8f9fa;
    }
    .corgi-introduction .engineering-technique {
        margin: 20px 0;
        padding: 15px;
        background: #ffffff;
        border-left: 4px solid #007bff;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .corgi-introduction .engineering-technique h3 {
        margin-top: 0;
        color: #007bff;
    }
    .corgi-introduction .engineering-technique ul {
        margin: 10px 0;
        padding-left: 25px;
    }
    .corgi-introduction .engineering-technique ul li {
        margin: 8px 0;
        line-height: 1.6;
    }
    .corgi-introduction .prompt-template {
        margin: 20px 0;
        padding: 15px;
        background: #ffffff;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .corgi-introduction .prompt-template h3 {
        margin-top: 0;
        color: #28a745;
    }
    .corgi-introduction code {
        background: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .corgi-introduction pre {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        overflow-x: auto;
        margin: 10px 0;
    }
    .corgi-introduction pre code.language-mermaid {
        background: #fff9e6;
        border: 2px solid #ffd700;
        padding: 15px;
        display: block;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.4;
    }
    .corgi-introduction img {
        max-width: 100%;
        height: auto;
        border: 2px solid #007bff;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .corgi-unified-doc {
        max-width: 100%;
        line-height: 1.6;
    }
    .corgi-unified-doc h2 {
        margin-top: 40px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #007bff;
        color: #007bff;
        font-size: 1.5em;
        font-weight: bold;
    }
    .corgi-unified-doc h3 {
        margin-top: 25px;
        margin-bottom: 15px;
        color: #333;
        font-size: 1.2em;
        font-weight: 600;
    }
    .corgi-unified-doc img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .corgi-unified-doc table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .corgi-unified-doc table th {
        background: #007bff;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .corgi-unified-doc table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    .corgi-unified-doc table tr:hover {
        background: #f5f5f5;
    }
    .corgi-unified-doc code {
        background: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .corgi-unified-doc pre {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        overflow-x: auto;
        margin: 10px 0;
    }
    .corgi-section {
        margin: 20px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .corgi-section h3 {
        margin-top: 0;
        color: #333;
        font-size: 1.2em;
    }
    .corgi-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .corgi-table th {
        background: #007bff;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .corgi-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    .corgi-table tr:hover {
        background: #f5f5f5;
    }
    .corgi-table tr:nth-child(even) {
        background: #fafafa;
    }
    .corgi-code-block {
        background: #ffffff;
        color: #333;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        line-height: 1.6;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .corgi-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .corgi-collapsible {
        margin: 10px 0;
    }
    .corgi-collapsible-header {
        background: #007bff;
        color: white;
        padding: 10px 15px;
        cursor: pointer;
        border-radius: 5px;
        user-select: none;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .corgi-collapsible-header:hover {
        background: #0056b3;
    }
    .corgi-collapsible-content {
        display: none;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 0 0 5px 5px;
    }
    .corgi-collapsible-content.active {
        display: block;
    }
    .corgi-bbox {
        display: inline-block;
        padding: 2px 6px;
        background: #ffc107;
        color: #000;
        border-radius: 3px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .corgi-answer {
        font-size: 1.3em;
        font-weight: bold;
        color: #28a745;
        margin: 15px 0;
        padding: 15px;
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
    }
    .corgi-explanation {
        padding: 15px;
        background: #e7f3ff;
        border-left: 4px solid #007bff;
        border-radius: 5px;
        margin: 15px 0;
        line-height: 1.6;
    }
    .corgi-evidence-thumb {
        max-width: 150px;
        max-height: 150px;
        border: 2px solid #007bff;
        border-radius: 5px;
        margin: 5px;
    }
    .corgi-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px;
    }
    .badge-yes {
        background: #28a745;
        color: white;
    }
    .badge-no {
        background: #6c757d;
        color: white;
    }
    .badge-ocr {
        background: #ffc107;
        color: #000;
    }
    </style>
    <script>
    function toggleCollapsible(element) {
        const content = element.nextElementSibling;
        const icon = element.querySelector('.toggle-icon');
        if (content.classList.contains('active')) {
            content.classList.remove('active');
            icon.textContent = '▶';
        } else {
            content.classList.add('active');
            icon.textContent = '▼';
        }
    }
    </script>
    """


def _image_to_base64(image: Image.Image, max_size: int = 200) -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    # Resize if too large
    if max(image.size) > max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def _image_to_base64_markdown(image: Image.Image, max_dim: int = 1200) -> str:
    """Convert PIL Image to base64 and return markdown image syntax."""
    # Resize if too large
    if max(image.size) > max_dim:
        image = image.copy()
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def _format_cot_html(cot_text: str) -> str:
    """Format Chain of Thought text as HTML."""
    if not cot_text:
        return '<div class="corgi-section"><p><em>No Chain of Thought text available.</em></p></div>'

    escaped_text = (
        cot_text.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
    )
    return f"""
    <div class="corgi-section">
        <h3>Chain of Thought (Full Text)</h3>
        <div class="corgi-code-block">{escaped_text}</div>
    </div>
    """


def _format_structured_steps_html(steps: List[ReasoningStep]) -> str:
    """Format structured reasoning steps as HTML table."""
    if not steps:
        return '<div class="corgi-section"><p><em>No reasoning steps available.</em></p></div>'

    rows = []
    for step in steps:
        needs_vision_badge = (
            f'<span class="corgi-badge badge-yes">Yes</span>'
            if step.needs_vision
            else f'<span class="corgi-badge badge-no">No</span>'
        )
        need_ocr_badge = (
            f'<span class="corgi-badge badge-ocr">OCR</span>'
            if getattr(step, "need_ocr", False)
            else ""
        )
        reason = step.reason or "<em>N/A</em>"
        rows.append(
            f"""
        <tr>
            <td><strong>{step.index}</strong></td>
            <td>{step.statement}</td>
            <td>{needs_vision_badge}</td>
            <td>{need_ocr_badge}</td>
            <td>{reason}</td>
        </tr>
        """
        )

    return f"""
    <div class="corgi-section">
        <h3>Structured Reasoning Steps</h3>
        <table class="corgi-table">
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Statement</th>
                    <th>Needs Vision</th>
                    <th>OCR</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _format_prompt_html(log: Optional[PromptLog], title: str) -> str:
    """Format prompt as collapsible HTML."""
    if log is None:
        return f'<div class="corgi-section"><p><em>{title} prompt unavailable.</em></p></div>'

    prompt_escaped = (
        log.prompt.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
    )
    response_html = ""
    if log.response:
        response_escaped = (
            log.response.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        )
        response_html = f"""
        <div class="corgi-collapsible">
            <div class="corgi-collapsible-header" onclick="toggleCollapsible(this)">
                <span>Model Response</span>
                <span class="toggle-icon">▶</span>
            </div>
            <div class="corgi-collapsible-content">
                <div class="corgi-code-block">{response_escaped}</div>
            </div>
        </div>
        """

    return f"""
    <div class="corgi-section">
        <h3>{title} Prompt</h3>
        <div class="corgi-collapsible">
            <div class="corgi-collapsible-header" onclick="toggleCollapsible(this)">
                <span>Show Prompt</span>
                <span class="toggle-icon">▶</span>
            </div>
            <div class="corgi-collapsible-content active">
                <div class="corgi-code-block">{prompt_escaped}</div>
            </div>
        </div>
        {response_html}
    </div>
    """


def _format_prompt_markdown(log: Optional[PromptLog], title: str) -> str:
    if log is None:
        return f"**{title} Prompt**\n_Prompt unavailable._"
    lines = [f"**{title} Prompt**", "```text", log.prompt, "```"]
    if log.response:
        lines.extend(["**Model Response**", "```text", log.response, "```"])
    return "\n".join(lines)


def _format_evidence_table_html(
    evidences: List[GroundedEvidence], image: Image.Image
) -> str:
    """Format evidence as HTML table with thumbnails."""
    if not evidences:
        return '<div class="corgi-section"><p><em>No visual evidence collected.</em></p></div>'

    rows = []
    width, height = image.size

    for idx, ev in enumerate(evidences, start=1):
        # Create cropped thumbnail
        thumb_img = _crop_evidence_image(image, ev)
        thumb_base64 = _image_to_base64(thumb_img, max_size=200)

        # Calculate pixel coordinates
        x1_px = int(ev.bbox[0] * width)
        y1_px = int(ev.bbox[1] * height)
        x2_px = int(ev.bbox[2] * width)
        y2_px = int(ev.bbox[3] * height)

        bbox_norm = (
            f"[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]"
        )
        bbox_px = f"[{x1_px}, {y1_px}, {x2_px}, {y2_px}]"

        desc = ev.description or "<em>No description</em>"
        ocr_text = ev.ocr_text or "<em>No OCR text</em>"
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"

        rows.append(
            f"""
        <tr>
            <td><img src="{thumb_base64}" class="corgi-evidence-thumb" alt="Evidence {idx}" /></td>
            <td><strong>{ev.step_index}</strong></td>
            <td><span class="corgi-bbox">{bbox_norm}</span><br/><small>{bbox_px} px</small></td>
            <td>{desc}</td>
            <td><span style="color: #666; font-style: italic;">{ocr_text}</span></td>
            <td>{conf}</td>
        </tr>
        """
        )

    return f"""
    <div class="corgi-section">
        <h3>Evidence Table</h3>
        <table class="corgi-table">
            <thead>
                <tr>
                    <th>Thumbnail</th>
                    <th>Step</th>
                    <th>Bounding Box</th>
                    <th>Description (Caption)</th>
                    <th>OCR Text</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _format_key_evidence_html(
    key_evidence: List[KeyEvidence], image: Optional[Image.Image]
) -> str:
    """Format key evidence as HTML cards."""
    if not key_evidence:
        return (
            '<div class="corgi-section"><p><em>No key evidence returned.</em></p></div>'
        )

    if image is None:
        # Fallback without images
        cards = []
        for idx, kev in enumerate(key_evidence, start=1):
            bbox_norm = f"[{kev.bbox[0]:.3f}, {kev.bbox[1]:.3f}, {kev.bbox[2]:.3f}, {kev.bbox[3]:.3f}]"
            cards.append(
                f"""
            <div class="corgi-card">
                <h4>Key Evidence {idx}</h4>
                <p><strong>Description:</strong> {kev.description}</p>
                <p><strong>Reasoning:</strong> {kev.reasoning}</p>
                <p><strong>BBox (normalized):</strong> <span class="corgi-bbox">{bbox_norm}</span></p>
            </div>
            """
            )
        return f"""
        <div class="corgi-section">
            <h3>Key Evidence</h3>
            {''.join(cards)}
        </div>
        """

    cards = []
    width, height = image.size

    for idx, kev in enumerate(key_evidence, start=1):
        # Create thumbnail with bbox
        color = KEY_EVIDENCE_COLORS[idx % len(KEY_EVIDENCE_COLORS)]
        thumb_img = _annotate_key_evidence_image(image, kev, color)
        thumb_base64 = _image_to_base64(thumb_img, max_size=200)

        # Calculate pixel coordinates
        x1_px = int(kev.bbox[0] * width)
        y1_px = int(kev.bbox[1] * height)
        x2_px = int(kev.bbox[2] * width)
        y2_px = int(kev.bbox[3] * height)

        bbox_norm = f"[{kev.bbox[0]:.3f}, {kev.bbox[1]:.3f}, {kev.bbox[2]:.3f}, {kev.bbox[3]:.3f}]"
        bbox_px = f"[{x1_px}, {y1_px}, {x2_px}, {y2_px}]"

        cards.append(
            f"""
        <div class="corgi-card">
            <h4>Key Evidence {idx}</h4>
            <img src="{thumb_base64}" style="max-width: 100%; border-radius: 5px; margin: 10px 0;" alt="Key Evidence {idx}" />
            <p><strong>Description:</strong> {kev.description}</p>
            <p><strong>Reasoning:</strong> {kev.reasoning}</p>
            <p><strong>BBox (normalized):</strong> <span class="corgi-bbox">{bbox_norm}</span></p>
            <p><strong>BBox (pixels):</strong> <small>{bbox_px}</small></p>
        </div>
        """
        )

    return f"""
    <div class="corgi-section">
        <h3>Key Evidence</h3>
        {''.join(cards)}
    </div>
    """


def _format_answer_synthesis_html(
    result: PipelineResult, image: Optional[Image.Image] = None
) -> str:
    """Format answer synthesis as HTML."""
    answer_escaped = (
        (result.answer or "(no answer returned)")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("&", "&amp;")
    )
    answer_html = f'<div class="corgi-answer">{answer_escaped}</div>'

    paraphrased_html = ""
    if result.paraphrased_question:
        paraphrased_escaped = (
            result.paraphrased_question.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("&", "&amp;")
        )
        paraphrased_html = f'<div class="corgi-paraphrased"><strong>Paraphrased Question:</strong> {paraphrased_escaped}</div>'

    explanation_html = ""
    if result.explanation:
        explanation_escaped = (
            result.explanation.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("&", "&amp;")
        )
        explanation_html = f'<div class="corgi-explanation"><strong>Explanation:</strong> {explanation_escaped}</div>'

    key_evidence_html = _format_key_evidence_html(result.key_evidence or [], image)

    return f"""
    <div class="corgi-section">
        <h3>Final Answer</h3>
        {paraphrased_html}
        {answer_html}
        {explanation_html}
        {key_evidence_html}
    </div>
    """


def _format_grounding_prompts(logs: List[PromptLog]) -> str:
    """Format grounding prompts as HTML."""
    if not logs:
        return (
            '<div class="corgi-section"><p><em>No ROI prompts available.</em></p></div>'
        )

    blocks = []
    for log in logs:
        heading = (
            f"Step {log.step_index}" if log.step_index is not None else "ROI Prompt"
        )
        prompt_escaped = (
            log.prompt.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        )
        response_html = ""
        if log.response:
            response_escaped = (
                log.response.replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("&", "&amp;")
            )
            response_html = f"""
            <div class="corgi-collapsible">
                <div class="corgi-collapsible-header" onclick="toggleCollapsible(this)">
                    <span>Model Response</span>
                    <span class="toggle-icon">▶</span>
                </div>
                <div class="corgi-collapsible-content">
                    <div class="corgi-code-block">{response_escaped}</div>
                </div>
            </div>
            """

        blocks.append(
            f"""
        <div class="corgi-section">
            <h4>{heading}</h4>
            <div class="corgi-collapsible">
                <div class="corgi-collapsible-header" onclick="toggleCollapsible(this)">
                    <span>Show Prompt</span>
                    <span class="toggle-icon">▶</span>
                </div>
                <div class="corgi-collapsible-content active">
                    <div class="corgi-code-block">{prompt_escaped}</div>
                </div>
            </div>
            {response_html}
        </div>
        """
        )

    return "".join(blocks)


def _mermaid_to_image_base64(mermaid_code: str) -> Optional[str]:
    """
    Convert Mermaid diagram code to base64-encoded image using mermaid.ink API.

    Args:
        mermaid_code: Mermaid diagram code (without ```mermaid wrapper)

    Returns:
        Base64-encoded image data URI, or None if conversion fails
    """
    import base64
    import urllib.request
    import urllib.error

    try:
        # Encode Mermaid code to base64 URL-safe
        mermaid_bytes = mermaid_code.encode("utf-8")
        mermaid_b64 = base64.urlsafe_b64encode(mermaid_bytes).decode("utf-8")

        # Create mermaid.ink URL
        mermaid_url = f"https://mermaid.ink/img/{mermaid_b64}"

        # Download image
        with urllib.request.urlopen(mermaid_url, timeout=10) as response:
            image_data = response.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{image_b64}"
    except Exception as e:
        logger.warning(f"Failed to convert Mermaid to image: {e}")
        return None


def _generate_introduction_html() -> str:
    """Generate introduction HTML with Mermaid diagram and prompt templates."""
    import html

    html_parts = []

    # Title
    html_parts.append("<h1>CoRGI Pipeline Architecture</h1>")

    # Overview
    html_parts.append("<h2>Overview</h2>")
    html_parts.append(
        "<p><strong>CoRGI (Chain of Reasoning with Grounded Insights)</strong> is a framework that enhances reasoning reliability in vision-language models (VLMs) through <strong>post-hoc visual verification</strong> of chain-of-thought outputs. This implementation addresses the <strong>single-look bias</strong> problem, where VLMs generate fluent but unverified reasoning chains that drift away from actual image content, leading to hallucinations.</p>"
    )
    html_parts.append(
        "<p>Instead of requiring expensive iterative grounding or architectural redesign, CoRGI performs <strong>post-hoc verification</strong>: it first lets a VLM generate a full reasoning chain, then verifies each step against visual evidence in a separate stage. This lightweight, modular approach is compatible with existing VLMs and requires no end-to-end retraining.</p>"
    )
    html_parts.append(
        "<p>This implementation combines the strengths of <strong>Qwen3-VL-2B-Instruct</strong> and <strong>Florence-2-large-ft</strong> models to provide accurate, evidence-based answers with improved faithfulness and interpretability.</p>"
    )

    # Pipeline Architecture
    html_parts.append("<h2>Pipeline Architecture</h2>")

    # Build Mermaid diagram code
    mermaid_diagram = """graph TD
    A[Input: Image + Question] --> B[Stage 1: Chain of Thought Reasoning]
    B -->|Qwen3-VL-2B-Instruct| C[Structured Reasoning Steps]
    C --> D{Step needs vision?}
    D -->|Yes| E[Stage 2: ROI Extraction]
    D -->|No| I[Stage 4: Answer Synthesis]
    E -->|Florence-2-large-ft| F[Grounding: Extract Bounding Boxes]
    F --> G[Apply NMS: Remove Overlapping Boxes]
    G --> H[Stage 3: Evidence Description]
    H -->|Florence-2-large-ft| J[Parallel Processing]
    J --> K[OCR: Extract Text from Regions]
    J --> L[Captioning: Describe Visual Content]
    K --> M[Combined Evidence]
    L --> M
    M --> I
    I -->|Qwen3-VL-2B-Instruct| N[Final Answer + Key Evidence]
    N --> O[Output: Answer, Explanation, Key Evidence]
    
    style B fill:#e1f5ff
    style E fill:#fff4e1
    style H fill:#fff4e1
    style I fill:#e1f5ff
    style F fill:#ffe1f5
    style K fill:#ffe1f5
    style L fill:#ffe1f5"""

    # Add Mermaid code block (TEMPORARILY COMMENTED FOR HF SPACES DEPLOYMENT)
    # html_parts.append('<p><strong>Mermaid Diagram Code:</strong></p>')
    # html_parts.append('<pre><code class="language-mermaid">')
    # html_parts.append(html.escape(mermaid_diagram))
    # html_parts.append('</code></pre>')

    # Convert to image and embed as base64
    mermaid_image_base64 = _mermaid_to_image_base64(mermaid_diagram)
    if mermaid_image_base64:
        html_parts.append("<p><strong>Visual Diagram:</strong></p>")
        html_parts.append(
            f'<img src="{mermaid_image_base64}" alt="Pipeline Architecture" style="max-width: 100%; height: auto; border: 2px solid #007bff; border-radius: 8px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" />'
        )
    else:
        html_parts.append(
            "<p><em>Note: Mermaid diagram rendering is available when viewing this in a compatible Markdown viewer.</em></p>"
        )

    # Model Configuration Table
    html_parts.append("<h2>Model Configuration</h2>")
    html_parts.append("<table>")
    html_parts.append(
        "<thead><tr><th>Stage</th><th>Model</th><th>Purpose</th><th>Key Features</th></tr></thead>"
    )
    html_parts.append("<tbody>")
    html_parts.append("<tr>")
    html_parts.append("<td><strong>Stage 1: Reasoning</strong></td>")
    html_parts.append("<td>Qwen3-VL-2B-Instruct</td>")
    html_parts.append("<td>Generate structured reasoning steps</td>")
    html_parts.append(
        "<td><ul><li>Chain-of-Thought reasoning</li><li>JSON-structured output</li><li>Noun phrase extraction</li></ul></td>"
    )
    html_parts.append("</tr>")
    html_parts.append("<tr>")
    html_parts.append("<td><strong>Stage 2: Grounding</strong></td>")
    html_parts.append("<td>Florence-2-large-ft</td>")
    html_parts.append("<td>Extract bounding boxes for regions</td>")
    html_parts.append(
        "<td><ul><li>Phrase-to-region grounding</li><li>Overlapping box removal</li></ul></td>"
    )
    html_parts.append("</tr>")
    html_parts.append("<tr>")
    html_parts.append("<td><strong>Stage 3: Evidence</strong></td>")
    html_parts.append("<td>Florence-2-large-ft</td>")
    html_parts.append("<td>Describe regions (OCR + Captioning)</td>")
    html_parts.append(
        "<td><ul><li>Parallel OCR + Captioning</li><li>Region cropping</li><li>Detailed visual descriptions</li></ul></td>"
    )
    html_parts.append("</tr>")
    html_parts.append("<tr>")
    html_parts.append("<td><strong>Stage 4: Synthesis</strong></td>")
    html_parts.append("<td>Qwen3-VL-2B-Instruct</td>")
    html_parts.append("<td>Generate final answer</td>")
    html_parts.append(
        "<td><ul><li>Evidence-based reasoning</li><li>Key evidence selection</li><li>Explanation generation</li></ul></td>"
    )
    html_parts.append("</tr>")
    html_parts.append("</tbody>")
    html_parts.append("</table>")

    # Differences from Original Paper
    html_parts.append("<h2>Differences from Original Paper</h2>")
    html_parts.append(
        '<p>This implementation adapts the CoRGI framework from the <a href="https://arxiv.org/pdf/2508.00378v2" target="_blank">original paper</a> with several key modifications and optimizations:</p>'
    )

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>Model Choices</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Reasoning & Synthesis</strong>: Uses <strong>Qwen3-VL-2B-Instruct</strong> instead of Qwen-2.5VL, LLaVA1.6, or Gemma3-12B from the paper</li>"
    )
    html_parts.append(
        "<li><strong>Grounding & Evidence</strong>: Uses <strong>Florence-2-large-ft</strong> instead of Grounding DINO for region detection and evidence extraction</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>Pipeline Structure</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>4 Stages</strong>: Separates ROI extraction (Stage 2) and evidence description (Stage 3) into distinct stages, compared to the paper's 3-stage structure</li>"
    )
    html_parts.append(
        "<li><strong>Relevance Classification</strong>: Uses model-generated <code>needs_vision</code> flag directly from reasoning output instead of a separate relevance classifier</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>Optimizations & Enhancements</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>NMS (Non-Maximum Suppression)</strong>: Applies IoU-based filtering to remove overlapping bounding boxes, reducing redundant evidence regions</li>"
    )
    html_parts.append(
        "<li><strong>Parallel OCR + Captioning</strong>: Processes OCR and captioning tasks in parallel for each region, improving efficiency</li>"
    )
    html_parts.append(
        "<li><strong>No Text Reranker</strong>: Uses direct evidence matching instead of a separate text cross-encoder reranker</li>"
    )
    html_parts.append(
        "<li><strong>Modular Architecture</strong>: Fully pluggable model system allowing easy swapping of components (Qwen, Florence-2, or custom models)</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>Implementation Benefits</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Lightweight</strong>: No separate relevance classifier training required</li>"
    )
    html_parts.append(
        "<li><strong>Efficient</strong>: Optimized for Hugging Face Spaces deployment with GPU context management</li>"
    )
    html_parts.append(
        "<li><strong>Flexible</strong>: YAML-based configuration system for easy model and parameter adjustments</li>"
    )
    html_parts.append(
        "<li><strong>Production-Ready</strong>: Includes comprehensive error handling, coordinate system normalization, and performance optimizations</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    # Pipeline Flow
    html_parts.append("<h2>Pipeline Flow</h2>")
    html_parts.append("<ol>")
    html_parts.append(
        "<li><strong>Input</strong>: User provides an image and a question</li>"
    )
    html_parts.append(
        "<li><strong>Reasoning</strong>: Qwen3-VL generates structured reasoning steps with noun phrases</li>"
    )
    html_parts.append(
        "<li><strong>Grounding</strong>: Florence-2 extracts bounding boxes for each reasoning step</li>"
    )
    html_parts.append(
        "<li><strong>NMS</strong>: Overlapping boxes are filtered using IoU-based NMS</li>"
    )
    html_parts.append(
        "<li><strong>Evidence</strong>: For each region, Florence-2 runs OCR and Captioning in parallel</li>"
    )
    html_parts.append(
        "<li><strong>Synthesis</strong>: Qwen3-VL combines all evidence to generate final answer with key evidence</li>"
    )
    html_parts.append(
        "<li><strong>Output</strong>: Answer, explanation, and key evidence bounding boxes</li>"
    )
    html_parts.append("</ol>")

    # Prompt Templates
    html_parts.append("<h2>Prompt Templates</h2>")

    # Stage 1: Reasoning Prompt
    html_parts.append('<div class="prompt-template">')
    html_parts.append("<h3>Stage 1: Chain of Thought Reasoning</h3>")
    html_parts.append("<p><strong>Model</strong>: Qwen3-VL-2B-Instruct</p>")
    html_parts.append("<p><strong>Prompt Template</strong>:</p>")
    example_reasoning_prompt = INSTRUCT_REASONING_PROMPT.format(
        question="What is the color of the watch?", max_steps=3
    )
    html_parts.append('<pre><code class="language-text">')
    html_parts.append(html.escape(example_reasoning_prompt))
    html_parts.append("</code></pre>")
    html_parts.append("</div>")

    # Stage 2: Grounding Prompt
    html_parts.append('<div class="prompt-template">')
    html_parts.append("<h3>Stage 2: ROI Extraction (Grounding)</h3>")
    html_parts.append("<p><strong>Model</strong>: Florence-2-large-ft</p>")
    html_parts.append(
        "<p><strong>Task</strong>: <code>&lt;CAPTION_TO_PHRASE_GROUNDING&gt;</code></p>"
    )
    html_parts.append(
        "<p><strong>Note</strong>: Florence-2 uses task-based prompting. For Qwen-based grounding, the prompt is:</p>"
    )
    example_grounding_prompt = QWEN_GROUNDING_PROMPT.format(
        step_statement="the watch on the woman's wrist", max_regions=3
    )
    html_parts.append('<pre><code class="language-text">')
    html_parts.append(html.escape(example_grounding_prompt))
    html_parts.append("</code></pre>")
    html_parts.append("</div>")

    # Stage 3: Evidence Description Prompts
    html_parts.append('<div class="prompt-template">')
    html_parts.append("<h3>Stage 3: Evidence Description</h3>")
    html_parts.append("<p><strong>Model</strong>: Florence-2-large-ft</p>")
    html_parts.append("<p><strong>Tasks</strong>:</p>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>OCR</strong>: <code>&lt;OCR&gt;</code> - Extract text from regions</li>"
    )
    html_parts.append(
        "<li><strong>Captioning</strong>: <code>&lt;MORE_DETAILED_CAPTION&gt;</code> - Generate detailed visual descriptions</li>"
    )
    html_parts.append("</ul>")
    html_parts.append(
        "<p><strong>Note</strong>: Both tasks run in parallel for each evidence region. For Qwen-based captioning, the prompt is:</p>"
    )
    example_captioning_prompt = QWEN_CAPTIONING_PROMPT.format(
        step_statement="the watch on the woman's wrist"
    )
    html_parts.append('<pre><code class="language-text">')
    html_parts.append(html.escape(example_captioning_prompt))
    html_parts.append("</code></pre>")
    html_parts.append("</div>")

    # Stage 4: Answer Synthesis Prompt
    html_parts.append('<div class="prompt-template">')
    html_parts.append("<h3>Stage 4: Answer Synthesis</h3>")
    html_parts.append("<p><strong>Model</strong>: Qwen3-VL-2B-Instruct</p>")
    html_parts.append("<p><strong>Prompt Template</strong>:</p>")
    example_steps = "1. the watch on the woman's wrist (needs vision: True)"
    example_evidence = "Step 1: bbox=(539, 571, 568, 612), conf=0.95, desc=A person is wearing a white watch on their wrist."
    example_synthesis_prompt = ANSWER_SYNTHESIS_PROMPT.format(
        question="What is the color of the watch?",
        steps=example_steps,
        evidence=example_evidence,
    )
    html_parts.append('<pre><code class="language-text">')
    html_parts.append(html.escape(example_synthesis_prompt))
    html_parts.append("</code></pre>")
    html_parts.append("</div>")

    # Engineering Techniques
    html_parts.append("<h2>Engineering Techniques</h2>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>1. Non-Maximum Suppression (NMS)</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Purpose</strong>: Remove overlapping bounding boxes from grounding results</li>"
    )
    html_parts.append(
        "<li><strong>Method</strong>: IoU-based filtering with configurable threshold (default: 0.5)</li>"
    )
    html_parts.append(
        "<li><strong>Implementation</strong>: Applied in <code>Florence2GroundingClient.extract_regions()</code></li>"
    )
    html_parts.append(
        "<li><strong>Benefit</strong>: Reduces redundant evidence regions, improves efficiency</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>2. Parallel OCR and Captioning</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Purpose</strong>: Extract both textual and visual information from evidence regions</li>"
    )
    html_parts.append(
        "<li><strong>Method</strong>: Unified batch processing with <code>ThreadPoolExecutor</code></li>"
    )
    html_parts.append(
        "<li><strong>Implementation</strong>: <code>Florence2CaptioningClient.ocr_and_caption_regions_batch()</code></li>"
    )
    html_parts.append(
        "<li><strong>Benefit</strong>: Single image crop per region, parallel task execution</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>3. KV Cache Optimization</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Purpose</strong>: Speed up transformer inference by caching key-value states</li>"
    )
    html_parts.append(
        "<li><strong>Method</strong>: <code>use_cache=True</code> in generation parameters</li>"
    )
    html_parts.append(
        "<li><strong>Implementation</strong>: Enabled for Qwen models, disabled for Florence-2 (transformers bug workaround)</li>"
    )
    html_parts.append(
        "<li><strong>Benefit</strong>: Faster inference, especially for multi-step reasoning</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>4. Fast Image Processor</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Purpose</strong>: Accelerate image preprocessing</li>"
    )
    html_parts.append(
        "<li><strong>Method</strong>: <code>use_fast=True</code> with fallback to slow processor</li>"
    )
    html_parts.append(
        "<li><strong>Implementation</strong>: Applied in all model loading functions</li>"
    )
    html_parts.append("<li><strong>Benefit</strong>: Reduced preprocessing time</li>")
    html_parts.append("</ul>")
    html_parts.append("</div>")

    html_parts.append('<div class="engineering-technique">')
    html_parts.append("<h3>5. Coordinate System Handling</h3>")
    html_parts.append("<ul>")
    html_parts.append(
        "<li><strong>Purpose</strong>: Ensure consistent bounding box coordinates across models</li>"
    )
    html_parts.append(
        "<li><strong>Method</strong>: Automatic conversion between normalized [0,1], pixel [0,999], and absolute pixel formats</li>"
    )
    html_parts.append(
        "<li><strong>Implementation</strong>: <code>coordinate_utils.py</code> with format detection</li>"
    )
    html_parts.append(
        "<li><strong>Benefit</strong>: Seamless integration between Qwen3-VL and Florence-2</li>"
    )
    html_parts.append("</ul>")
    html_parts.append("</div>")

    return "\n".join(html_parts)


def _generate_unified_markdown_document(
    result: PipelineResult,
    image: Image.Image,
    max_slots: int = MAX_UI_STEPS,
) -> str:
    """Generate unified markdown document with all inference stages."""
    sections = []

    # Final Answer (no "Stage" prefix)
    sections.append("## Final Answer")
    sections.append(f"{result.answer or '(no answer returned)'}")
    if result.explanation:
        sections.append(f"\n**Explanation:** {result.explanation}")
    sections.append("")

    # Stage 1: Chain of Thought
    sections.append("## Stage 1: Chain of Thought")

    # Full CoT text
    if result.cot_text:
        sections.append("### Chain of Thought (Full Text)")
        sections.append("```text")
        sections.append(result.cot_text)
        sections.append("```")
        sections.append("")

    # Structured reasoning steps
    sections.append("### Structured Reasoning Steps")
    evidences_by_step = _group_evidence_by_step(result.evidence)
    for step in result.steps[:max_slots]:
        needs_vision = "Yes" if step.needs_vision else "No"
        need_ocr = "Yes" if getattr(step, "need_ocr", False) else "No"
        step_lines = [
            f"**Step {step.index}:** {step.statement}",
            f"- Needs vision: {needs_vision}",
            f"- Needs OCR: {need_ocr}",
        ]
        if step.reason:
            step_lines.append(f"- Reason: {step.reason}")
        evs = evidences_by_step.get(step.index, [])
        if evs:
            step_lines.append(f"- Visual evidence items: {len(evs)}")
        else:
            step_lines.append("- No visual evidence returned for this step.")
        sections.append("\n".join(step_lines))
        sections.append("")

    if len(result.steps) > max_slots:
        sections.append(f"_Only the first {max_slots} steps are shown._")
        sections.append("")

    # Prompt (always expanded, no collapsible)
    if result.reasoning_log:
        sections.append("### Reasoning Prompt")
        sections.append("```text")
        sections.append(result.reasoning_log.prompt)
        sections.append("```")
        if result.reasoning_log.response:
            sections.append("\n**Model Response:**")
            sections.append("```text")
            sections.append(result.reasoning_log.response)
            sections.append("```")
        sections.append("")

    # Stage 2: ROI Extraction
    sections.append("## Stage 2: ROI Extraction")

    # ROI Overview Image (base64 embedded)
    roi_overview = _annotate_overview_image(image, result.evidence)
    if roi_overview:
        roi_base64 = _image_to_base64_markdown(roi_overview, max_dim=1200)
        sections.append("### Annotated Image with All ROIs")
        sections.append(f"![ROI Overview]({roi_base64})")
        sections.append("")

    # ROI Gallery will be shown as separate component after this section
    if result.evidence:
        sections.append(f"### Evidence Gallery ({len(result.evidence)} items)")
        sections.append(
            "_See gallery component below for detailed view of each evidence region._"
        )
        sections.append("")
        sections.append("---")  # Separator before gallery
        sections.append("")

    # ROI Prompts
    if result.grounding_logs:
        sections.append("### ROI Extraction Prompts")
        for idx, log in enumerate(result.grounding_logs, start=1):
            sections.append(f"**Prompt {idx}:**")
            sections.append("```text")
            sections.append(log.prompt)
            sections.append("```")
            if log.response:
                sections.append(f"**Response {idx}:**")
                sections.append("```text")
                sections.append(log.response)
                sections.append("```")
            sections.append("")

    # Stage 3: Evidence Descriptions
    sections.append("## Stage 3: Evidence Descriptions")

    if result.evidence:
        sections.append("### Evidence Table")
        sections.append(
            "| Step | BBox (normalized) | BBox (pixels) | Description | OCR Text | Confidence |"
        )
        sections.append(
            "|:----:|:------------------:|:-------------:|:------------|:---------|:---------:|"
        )

        width, height = image.size
        for idx, ev in enumerate(result.evidence, start=1):
            x1_px = int(ev.bbox[0] * width)
            y1_px = int(ev.bbox[1] * height)
            x2_px = int(ev.bbox[2] * width)
            y2_px = int(ev.bbox[3] * height)
            # Format BBox columns with code blocks for monospace and compact display
            bbox_norm = f"`[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]`"
            bbox_px = f"`[{x1_px}, {y1_px}, {x2_px}, {y2_px}]`"

            # IMPORTANT: Escape pipe characters to prevent breaking Markdown table
            desc = (ev.description or "N/A").replace("|", "\\|")
            ocr_text = (ev.ocr_text or "N/A").replace("|", "\\|")
            conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
            # Step column: bold and centered
            sections.append(
                f"| **{ev.step_index}** | {bbox_norm} | {bbox_px} | {desc} | {ocr_text} | {conf} |"
            )
        sections.append("")
    else:
        sections.append("_No visual evidence collected._")
        sections.append("")

    # Evidence prompts
    if result.grounding_logs:
        sections.append("### Evidence Description Prompts")
        sections.append("_Same as ROI Extraction prompts above._")
        sections.append("")

    # Stage 4: Answer Synthesis
    sections.append("## Stage 4: Answer Synthesis")

    sections.append(f"**Question:** {result.question}")
    if result.paraphrased_question:
        sections.append(f"**Paraphrased Question:** {result.paraphrased_question}")
    sections.append(f"**Final Answer:** {result.answer or '(no answer returned)'}")
    if result.explanation:
        sections.append(f"**Explanation:** {result.explanation}")
    sections.append("")

    # Key Evidence Overview Image (base64 embedded)
    if result.key_evidence:
        key_evidence_overview = _annotate_key_evidence_overview(
            image, result.key_evidence
        )
        if key_evidence_overview:
            key_base64 = _image_to_base64_markdown(key_evidence_overview, max_dim=1200)
            sections.append("### Key Evidence Overview")
            sections.append(f"![Key Evidence Overview]({key_base64})")
            sections.append("")

        sections.append("### Key Evidence Details")
        for idx, kev in enumerate(result.key_evidence, start=1):
            bbox = ", ".join(f"{coord:.3f}" for coord in kev.bbox)
            sections.append(f"**Key Evidence {idx}:**")
            sections.append(f"- Description: {kev.description}")
            sections.append(f"- Reasoning: {kev.reasoning}")
            sections.append(f"- BBox (normalized): [{bbox}]")
            sections.append("")

        sections.append("### Key Evidence Gallery")
        sections.append(
            "_See gallery component below for detailed view of each key evidence region._"
        )
        sections.append("")
        sections.append("---")  # Separator before gallery
        sections.append("")

    # Answer Synthesis Prompt
    if result.answer_log:
        sections.append("### Answer Synthesis Prompt")
        sections.append("```text")
        sections.append(result.answer_log.prompt)
        sections.append("```")
        if result.answer_log.response:
            sections.append("\n**Model Response:**")
            sections.append("```text")
            sections.append(result.answer_log.response)
            sections.append("```")
        sections.append("")

    # Performance Metrics (no "Stage" prefix)
    sections.append("## Performance Metrics")

    if result.timings:
        total_entry = next(
            (t for t in result.timings if t.name == "total_pipeline"), None
        )
        if total_entry:
            sections.append(f"**Total pipeline:** {total_entry.duration_ms/1000:.2f} s")
        for timing in result.timings:
            if timing.name == "total_pipeline":
                continue
            label = timing.name.replace("_", " ").title()
            if timing.step_index is not None:
                label += f" (step {timing.step_index})"
            sections.append(f"- {label}: {timing.duration_ms/1000:.2f} s")
    else:
        sections.append("_No timing data available._")

    return "\n".join(sections)


def _format_grounding_prompts_markdown(logs: List[PromptLog]) -> str:
    """Format grounding prompts as markdown (backward compatibility)."""
    if not logs:
        return "_No ROI prompts available._"
    blocks: List[str] = []
    for log in logs:
        heading = (
            f"#### Step {log.step_index}"
            if log.step_index is not None
            else "#### ROI Prompt"
        )
        sections = [heading, "**Prompt**", "```text", log.prompt, "```"]
        if log.response:
            sections.extend(["**Model Response**", "```text", log.response, "```"])
        blocks.append("\n".join(sections))
    return "\n\n".join(blocks)


def _prepare_ui_payload(
    image: Image.Image,
    result: PipelineResult,
    max_slots: int = MAX_UI_STEPS,
) -> Dict[str, object]:
    answer_text = f"### Final Answer\n{result.answer or '(no answer returned)'}"

    # Build Chain of Thought display: Full CoT text first, then structured steps
    chain_sections: List[str] = []

    # 1. Show full Chain of Thought text if available
    if result.cot_text:
        chain_sections.append("## Chain of Thought (Full Text)")
        chain_sections.append("```text")
        chain_sections.append(result.cot_text)
        chain_sections.append("```")
        chain_sections.append("")  # Empty line separator

    # 2. Show structured reasoning steps
    chain_sections.append("## Structured Reasoning Steps")
    step_lines: List[str] = []
    evidences_by_step = _group_evidence_by_step(result.evidence)
    for step in result.steps[:max_slots]:
        lines = [
            f"**Step {step.index}:** {step.statement}",
            f"- Needs vision: {'yes' if step.needs_vision else 'no'}",
        ]
        if step.reason:
            lines.append(f"- Reason: {step.reason}")
        evs = evidences_by_step.get(step.index, [])
        if evs:
            lines.append(f"- Visual evidence items: {len(evs)}")
        else:
            lines.append("- No visual evidence returned for this step.")
        step_lines.append("\n".join(lines))
    if len(result.steps) > max_slots:
        step_lines.append(f"_Only the first {max_slots} steps are shown._")

    if step_lines:
        chain_sections.extend(step_lines)
    else:
        chain_sections.append("_No reasoning steps returned._")

    chain_markdown = "\n\n".join(chain_sections)

    roi_overview = _annotate_overview_image(image, result.evidence)
    aggregated_gallery: List[Tuple[Image.Image, str]] = []
    for idx, evidence in enumerate(result.evidence):
        color = EVIDENCE_COLORS[idx % len(EVIDENCE_COLORS)]
        annotated = _annotate_evidence_image(image, evidence, color)
        aggregated_gallery.append((annotated, _format_evidence_caption(evidence)))

    evidence_blocks: List[str] = []
    for idx, evidence in enumerate(result.evidence, start=1):
        bbox = ", ".join(f"{coord:.2f}" for coord in evidence.bbox)
        desc = evidence.description or "(no description)"
        ocr = f"- OCR: {evidence.ocr_text}" if evidence.ocr_text else ""
        conf = (
            f"Confidence: {evidence.confidence:.2f}"
            if evidence.confidence is not None
            else "Confidence: n/a"
        )
        evidence_blocks.append(
            f"**Evidence {idx} — Step {evidence.step_index}**\n- Caption: {desc}\n{ocr}\n- {conf}\n- BBox: ({bbox})"
        )
    evidence_markdown = (
        "\n\n".join(evidence_blocks)
        if evidence_blocks
        else "_No visual evidence collected._"
    )

    reasoning_prompt_md = _format_prompt_markdown(result.reasoning_log, "Reasoning")
    roi_prompt_md = _format_grounding_prompts(result.grounding_logs)
    evidence_prompt_md = (
        roi_prompt_md if result.grounding_logs else "_No ROI prompts available._"
    )
    answer_prompt_md = _format_prompt_markdown(result.answer_log, "Answer Synthesis")

    answer_process_lines = [
        f"**Question:** {result.question}",
        f"**Final Answer:** {result.answer or '(no answer returned)'}",
        f"**Steps considered:** {len(result.steps)}",
        f"**Visual evidence items:** {len(result.evidence)}",
    ]
    if result.key_evidence:
        answer_process_lines.append(
            f"**Key evidence items:** {len(result.key_evidence)}"
        )
    answer_process_markdown = "\n".join(answer_process_lines)

    # Generate key evidence visualization
    key_evidence_overview = _annotate_key_evidence_overview(
        image, result.key_evidence or []
    )
    key_evidence_gallery: List[Tuple[Image.Image, str]] = []
    if result.key_evidence:
        for idx, kev in enumerate(result.key_evidence):
            color = KEY_EVIDENCE_COLORS[idx % len(KEY_EVIDENCE_COLORS)]
            annotated = _annotate_key_evidence_image(image, kev, color)
            key_evidence_gallery.append(
                (annotated, _format_key_evidence_caption(kev, idx))
            )

    # Generate key evidence markdown
    key_evidence_blocks: List[str] = []
    if result.key_evidence:
        for idx, kev in enumerate(result.key_evidence, start=1):
            bbox = ", ".join(f"{coord:.2f}" for coord in kev.bbox)
            key_evidence_blocks.append(
                f"**Key Evidence {idx}**\n"
                f"- Description: {kev.description}\n"
                f"- Reasoning: {kev.reasoning}\n"
                f"- BBox: ({bbox})"
            )
    key_evidence_markdown = (
        "\n\n".join(key_evidence_blocks)
        if key_evidence_blocks
        else "_No key evidence returned._"
    )

    timing_lines: List[str] = []
    if result.timings:
        total_entry = next(
            (t for t in result.timings if t.name == "total_pipeline"), None
        )
        if total_entry:
            timing_lines.append(
                f"**Total pipeline:** {total_entry.duration_ms/1000:.2f} s"
            )
        for timing in result.timings:
            if timing.name == "total_pipeline":
                continue
            label = timing.name.replace("_", " ")
            if timing.step_index is not None:
                label += f" (step {timing.step_index})"
            timing_lines.append(f"- {label}: {timing.duration_ms/1000:.2f} s")
    timing_markdown = (
        "\n".join(timing_lines) if timing_lines else "_No timing data available._"
    )

    # Generate HTML versions with CSS
    css_stylesheet = _create_css_stylesheet()

    # Chain of Thought HTML
    cot_html = _format_cot_html(result.cot_text or "")
    steps_html = _format_structured_steps_html(result.steps[:max_slots])
    chain_prompt_html = _format_prompt_html(result.reasoning_log, "Reasoning")
    chain_html = css_stylesheet + cot_html + steps_html

    # Evidence HTML
    evidence_table_html = _format_evidence_table_html(result.evidence, image)
    evidence_prompt_html = _format_grounding_prompts(result.grounding_logs)

    # Answer Synthesis HTML
    answer_synthesis_html = _format_answer_synthesis_html(result, image)
    answer_prompt_html = _format_prompt_html(result.answer_log, "Answer Synthesis")

    # ROI prompts HTML
    roi_prompt_html = _format_grounding_prompts(result.grounding_logs)

    return {
        # Markdown (backward compatibility)
        "answer_markdown": answer_text,
        "chain_markdown": chain_markdown,
        "chain_prompt": reasoning_prompt_md,
        "roi_overview": roi_overview,
        "roi_gallery": aggregated_gallery,
        "roi_prompt": roi_prompt_md,
        "evidence_markdown": evidence_markdown,
        "evidence_prompt": evidence_prompt_md,
        "answer_process_markdown": answer_process_markdown,
        "answer_prompt": answer_prompt_md,
        "key_evidence_overview": key_evidence_overview,
        "key_evidence_gallery": key_evidence_gallery,
        "key_evidence_markdown": key_evidence_markdown,
        "timing_markdown": timing_markdown,
        # HTML (new enhanced UI)
        "answer_html": css_stylesheet
        + answer_synthesis_html,  # Final answer display at top
        "answer_synthesis_html": css_stylesheet
        + answer_synthesis_html,  # Answer synthesis tab
        "chain_cot_html": css_stylesheet + cot_html,
        "chain_steps_html": steps_html,
        "chain_prompt_html": css_stylesheet + chain_prompt_html,
        "evidence_table_html": css_stylesheet + evidence_table_html,
        "evidence_prompt_html": css_stylesheet + evidence_prompt_html,
        "answer_prompt_html": css_stylesheet + answer_prompt_html,
        "roi_prompt_html": css_stylesheet + roi_prompt_html,
        # Unified markdown document
        "unified_markdown": _generate_unified_markdown_document(
            result, image, max_slots
        ),
    }


if spaces is not None:

    @spaces.GPU(duration=120)  # type: ignore[attr-defined]
    def _create_pipeline_from_config_v2_gpu(
        config_path: Optional[str] = None,
        config: Optional[CoRGiConfig] = None,
        parallel_loading: bool = True,
        image_logger=None,
        output_tracer=None,
    ) -> CoRGIPipeline:
        """GPU-decorated version of _create_pipeline_from_config_v2 for HF Spaces."""
        logger.debug("Creating pipeline in GPU context.")
        return _create_pipeline_from_config_v2(
            config_path=config_path,
            config=config,
            parallel_loading=parallel_loading,
            image_logger=image_logger,
            output_tracer=output_tracer,
        )

    @spaces.GPU(duration=120)  # type: ignore[attr-defined]
    def _execute_pipeline_gpu(
        image: Image.Image,
        question: str,
        max_steps: int,
        max_regions: int,
        pipeline: CoRGIPipeline,
    ) -> PipelineResult:
        logger.debug("Running GPU-decorated pipeline.")
        return _execute_pipeline(image, question, max_steps, max_regions, pipeline)

else:
    # Fallback when spaces is not available
    def _create_pipeline_from_config_v2_gpu(
        config_path: Optional[str] = None,
        config: Optional[CoRGiConfig] = None,
        parallel_loading: bool = True,
        image_logger=None,
        output_tracer=None,
    ) -> CoRGIPipeline:
        """Fallback version when spaces is not available."""
        return _create_pipeline_from_config_v2(
            config_path=config_path,
            config=config,
            parallel_loading=parallel_loading,
            image_logger=image_logger,
            output_tracer=output_tracer,
        )

    def _execute_pipeline_gpu(
        image: Image.Image,
        question: str,
        max_steps: int,
        max_regions: int,
        pipeline: CoRGIPipeline,
    ) -> PipelineResult:
        return _execute_pipeline(image, question, max_steps, max_regions, pipeline)


def ensure_pipeline_state(
    previous: Optional[PipelineState],
    model_id: Optional[str],
    factory: Callable[[Optional[str]], CoRGIPipeline] | None = None,
) -> PipelineState:
    target_model = model_id or DEFAULT_MODEL_ID
    factory = factory or _default_factory
    if previous is not None and previous.model_id == target_model:
        return previous
    pipeline = factory(target_model)
    return PipelineState(model_id=target_model, pipeline=pipeline)


def format_result_markdown(result: PipelineResult) -> str:
    lines: list[str] = []
    lines.append("### Answer")
    lines.append(result.answer or "(no answer returned)")
    lines.append("")
    lines.append("### Reasoning Steps")
    if result.steps:
        for step in result.steps:
            needs = "yes" if step.needs_vision else "no"
            reason = f" — {step.reason}" if step.reason else ""
            lines.append(
                f"- **Step {step.index}**: {step.statement} _(needs vision: {needs})_{reason}"
            )
    else:
        lines.append("- No reasoning steps returned.")
    lines.append("")
    lines.append("### Visual Evidence")
    if result.evidence:
        for ev in result.evidence:
            bbox = ", ".join(f"{coord:.2f}" for coord in ev.bbox)
            desc = ev.description or "(no description)"
            ocr = f" — OCR: {ev.ocr_text}" if ev.ocr_text else ""
            conf = (
                f" — confidence {ev.confidence:.2f}"
                if ev.confidence is not None
                else ""
            )
            lines.append(f"- Step {ev.step_index}: bbox=({bbox}) — {desc}{ocr}{conf}")
    else:
        lines.append("- No visual evidence collected.")
    return "\n".join(lines)


def _run_pipeline(
    state: Optional[PipelineState],
    image: Image.Image | None,
    question: str,
    max_steps: int,
    max_regions: int,
    config_path: str,
    parallel_loading: bool,
    batch_captioning: bool,
    model_id_override: Optional[str],
) -> tuple[PipelineState, Dict[str, object]]:
    """
    Run pipeline with new config-based approach.

    Args:
        state: Current pipeline state
        image: Input image
        question: Question text
        max_steps: Maximum reasoning steps
        max_regions: Maximum regions per step
        config_path: Path to config YAML file
        parallel_loading: Enable parallel model loading
        batch_captioning: Enable batch captioning (not directly used, but included in cache key)
        model_id_override: Optional model ID override (not yet implemented)

    Returns:
        Tuple of (new_state, ui_payload)
    """
    if image is None:
        logger.info("Request skipped: no image provided.")
        return state or PipelineState(model_id="", pipeline=None), _empty_ui_payload(
            "Please provide an image before running the demo."
        )
    if not question.strip():
        logger.info("Request skipped: question empty.")
        return state or PipelineState(model_id="", pipeline=None), _empty_ui_payload(
            "Please enter a question before running the demo."
        )

    # Create cache key
    cache_key = _make_cache_key(
        config_path=config_path,
        parallel_loading=parallel_loading,
        batch_captioning=batch_captioning,
        model_id_override=model_id_override,
    )

    logger.info(
        "Received request | config=%s | parallel_loading=%s | batch_captioning=%s",
        config_path,
        parallel_loading,
        batch_captioning,
    )
    rgb_image = image.convert("RGB")

    try:
        # Get or create pipeline (use GPU-decorated version for HF Spaces)
        def create_pipeline():
            return _create_pipeline_from_config_v2_gpu(
                config_path=config_path,
                parallel_loading=parallel_loading,
            )

        pipeline = _get_pipeline(cache_key, create_pipeline)

        # Execute pipeline
        result = _execute_pipeline_gpu(
            image=rgb_image,
            question=question.strip(),
            max_steps=int(max_steps),
            max_regions=int(max_regions),
            pipeline=pipeline,
        )
    except FileNotFoundError as exc:
        logger.exception("Config file not found: %s", exc)
        return (
            state or PipelineState(model_id="", pipeline=None),
            _empty_ui_payload(f"Configuration error: Config file not found. {exc}"),
        )
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Pipeline execution failed: %s", exc)
        return (
            state or PipelineState(model_id="", pipeline=None),
            _empty_ui_payload(f"Pipeline error: {exc}"),
        )

    new_state = PipelineState(model_id=cache_key, pipeline=pipeline)
    payload = _prepare_ui_payload(rgb_image, result, MAX_UI_STEPS)
    return new_state, payload


def build_demo(
    pipeline_factory: Callable[[Optional[str]], CoRGIPipeline] | None = None,
    default_config: Optional[str] = None,
) -> "gradio.Blocks":
    """
    Build Gradio demo with flexible UI controls.

    Args:
        pipeline_factory: Optional legacy factory (for backward compatibility)
        default_config: Default config file path (relative to configs/ or absolute)

    Returns:
        Gradio Blocks instance
    """
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised when gradio missing
        raise RuntimeError(
            "Gradio is required to build the demo. Install gradio>=4.0."
        ) from exc

    # Set default config
    if default_config is None:
        default_config = str(DEFAULT_QWEN_CONFIG)

    # Handle legacy factory for backward compatibility
    if pipeline_factory is not None:
        global _GLOBAL_FACTORY
        _GLOBAL_FACTORY = pipeline_factory
        logger.info("Registering legacy pipeline factory %s", pipeline_factory)

    # Get available config files
    available_configs = []
    if DEFAULT_CONFIG_DIR.exists():
        for config_file in DEFAULT_CONFIG_DIR.glob("*.yaml"):
            available_configs.append(config_file.name)

    if not available_configs:
        available_configs = ["qwen_only.yaml", "florence_qwen.yaml"]

    # Determine default config name
    default_config_name = (
        Path(default_config).name
        if Path(default_config).exists()
        else available_configs[0]
    )

    # Warm up pipeline with the correct default config
    try:
        # Resolve config path to absolute path for consistency
        config_path_obj = Path(default_config)
        if not config_path_obj.is_absolute():
            # Try relative to configs directory first
            full_path = DEFAULT_CONFIG_DIR / config_path_obj
            if not full_path.exists():
                # Try as absolute path
                full_path = config_path_obj
            default_config = str(full_path)

        logger.info(f"Warming up pipeline with config: {default_config}")
        cache_key = _make_cache_key(
            config_path=default_config,
            parallel_loading=True,
            batch_captioning=True,
            model_id_override=None,
        )
        if cache_key not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[cache_key] = _create_pipeline_from_config_v2(
                config_path=default_config,
                parallel_loading=True,
            )
            logger.info("✓ Default pipeline warmed up successfully")
        else:
            logger.info("Pipeline already cached, skipping warm-up")
    except Exception as exc:
        logger.warning(f"Failed to warm up default pipeline: {exc}")

    with gr.Blocks(title="CoRGI Qwen3-VL Demo") as demo:
        state = gr.State()  # stores PipelineState

        # Introduction section at the top
        # Use HTML directly for better rendering control on Hugging Face Spaces
        introduction_html = _generate_introduction_html()
        introduction_markdown = gr.HTML(
            value=_create_css_stylesheet()
            + f'<div class="corgi-introduction">{introduction_html}</div>',
        )

        # Top section: Input controls in a compact grid
        with gr.Row():
            image_input = gr.Image(label="Input image", type="pil", scale=1)
            question_input = gr.Textbox(
                label="Question",
                placeholder="What is happening in the image?",
                lines=2,
                scale=2,
            )

        with gr.Row():
            config_dropdown = gr.Dropdown(
                label="Configuration",
                choices=available_configs,
                value=default_config_name,
                info="Select pipeline configuration",
                scale=1,
            )
            parallel_loading_checkbox = gr.Checkbox(
                label="Parallel Model Loading",
                value=True,
                info="Load models in parallel (faster startup, more memory)",
                scale=1,
            )
            batch_captioning_checkbox = gr.Checkbox(
                label="Batch Captioning",
                value=True,
                info="Process multiple captions in batch (faster)",
                scale=1,
            )

        with gr.Row():
            model_id_override_input = gr.Textbox(
                label="Model ID Override (optional)",
                placeholder="Leave blank to use config",
                info="Override model ID from config (experimental)",
                scale=2,
            )
            max_steps_slider = gr.Slider(
                label="Max reasoning steps",
                minimum=1,
                maximum=6,
                step=1,
                value=3,
                scale=1,
            )
            max_regions_slider = gr.Slider(
                label="Max regions per step",
                minimum=1,
                maximum=6,
                step=1,
                value=1,
                scale=1,
            )
            run_button = gr.Button("Run CoRGI", variant="primary", scale=1)

        # Bottom section: Single scrollable column with unified markdown document
        with gr.Column():
            unified_markdown = gr.Markdown(
                value="# CoRGI Inference Pipeline\n\nUpload an image and ask a question to begin.",
                elem_classes=["corgi-unified-doc"],
            )

            # Galleries placed after their corresponding sections in markdown
            roi_gallery = gr.Gallery(
                label="Evidence Gallery",
                columns=1,
                height=400,
                allow_preview=True,
                show_label=True,
                visible=False,  # Will be shown when there's evidence
            )

            key_evidence_gallery = gr.Gallery(
                label="Key Evidence Gallery",
                columns=1,
                height=400,
                allow_preview=True,
                visible=False,  # Will be shown when there's key evidence
            )

        def _on_submit(
            state_data,
            image,
            question,
            config_file,
            parallel_loading,
            batch_captioning,
            model_id_override,
            max_steps,
            max_regions,
        ):
            """Handle form submission."""
            pipeline_state = (
                state_data if isinstance(state_data, PipelineState) else None
            )

            # Resolve config path
            config_path_obj = DEFAULT_CONFIG_DIR / config_file
            if not config_path_obj.exists():
                # Try as absolute path
                config_path_obj = Path(config_file)
            config_path = str(config_path_obj)

            new_state, payload = _run_pipeline(
                pipeline_state,
                image,
                question,
                int(max_steps),
                int(max_regions),
                config_path,
                parallel_loading,
                batch_captioning,
                model_id_override.strip() if model_id_override else None,
            )

            # Show galleries if there's content
            roi_gallery_visible = len(payload.get("roi_gallery", [])) > 0
            key_evidence_gallery_visible = (
                len(payload.get("key_evidence_gallery", [])) > 0
            )

            return [
                new_state,
                payload["unified_markdown"],  # Unified markdown document
                payload["roi_gallery"],  # ROI gallery
                gr.update(visible=roi_gallery_visible),  # Update ROI gallery visibility
                payload["key_evidence_gallery"],  # Key evidence gallery
                gr.update(
                    visible=key_evidence_gallery_visible
                ),  # Update key evidence gallery visibility
            ]

        output_components = [
            state,
            unified_markdown,
            roi_gallery,
            roi_gallery,  # For visibility update
            key_evidence_gallery,
            key_evidence_gallery,  # For visibility update
        ]

        run_button.click(
            fn=_on_submit,
            inputs=[
                state,
                image_input,
                question_input,
                config_dropdown,
                parallel_loading_checkbox,
                batch_captioning_checkbox,
                model_id_override_input,
                max_steps_slider,
                max_regions_slider,
            ],
            outputs=output_components,
        )

    return demo


def launch_demo(
    *,
    pipeline_factory: Callable[[Optional[str]], CoRGIPipeline] | None = None,
    **launch_kwargs,
) -> None:
    demo = build_demo(pipeline_factory=pipeline_factory)
    demo.launch(**launch_kwargs)


__all__ = [
    "PipelineState",
    "ensure_pipeline_state",
    "format_result_markdown",
    "build_demo",
    "launch_demo",
    "DEFAULT_MODEL_ID",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_QWEN_CONFIG",
    "DEFAULT_FLORENCE_QWEN_CONFIG",
]
