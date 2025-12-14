"""
Gradio App for CoRGI Pipeline - Using Native Gradio Components.

This is a refactored version that uses Gradio components instead of HTML
for better compatibility and maintainability.
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import spaces  # type: ignore
except ImportError:
    spaces = None

from PIL import Image, ImageDraw
import itertools

# Import helper functions from gradio_app_html
from .gradio_app_html import (
    PipelineState,
    ensure_pipeline_state,
    format_result_markdown,
    DEFAULT_MODEL_ID,
    _PIPELINE_CACHE,
    _GLOBAL_FACTORY,
    DEFAULT_CONFIG_DIR,
    DEFAULT_QWEN_CONFIG,
    MAX_UI_STEPS,
    GALLERY_MAX_DIM,
    EVIDENCE_COLORS,
    KEY_EVIDENCE_COLORS,
    _create_pipeline_from_config_v2,
    _get_pipeline,
    _execute_pipeline,
    _group_evidence_by_step,
    _format_evidence_caption,
    _annotate_evidence_image,
    _annotate_overview_image,
    _annotate_key_evidence_image,
    _annotate_key_evidence_overview,
    _format_key_evidence_caption,
    _make_cache_key,
    _empty_ui_payload,
    _THUMBNAIL_RESAMPLE,
)

from ..core.pipeline import CoRGIPipeline, PipelineResult
from ..core.config import CoRGiConfig
from ..core.types import (
    GroundedEvidence,
    KeyEvidence,
    PromptLog,
    ReasoningStep,
    StageTiming,
)
from PIL import ImageDraw

logger = logging.getLogger("corgi.gradio_app")


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


# GPU decorators for HF Spaces
if spaces is not None:

    @spaces.GPU(duration=120)
    def _create_pipeline_from_config_v2_gpu(
        config_path: Optional[str] = None,
        config: Optional[CoRGiConfig] = None,
        parallel_loading: bool = True,
        image_logger=None,
        output_tracer=None,
    ) -> CoRGIPipeline:
        """GPU-decorated version for HF Spaces."""
        return _create_pipeline_from_config_v2(
            config_path=config_path,
            config=config,
            parallel_loading=parallel_loading,
            image_logger=image_logger,
            output_tracer=output_tracer,
        )

    @spaces.GPU(duration=120)
    def _execute_pipeline_gpu(
        image: Image.Image,
        question: str,
        max_steps: int,
        max_regions: int,
        pipeline: CoRGIPipeline,
    ) -> PipelineResult:
        """GPU-decorated pipeline execution."""
        return _execute_pipeline(image, question, max_steps, max_regions, pipeline)

else:

    def _create_pipeline_from_config_v2_gpu(*args, **kwargs):
        return _create_pipeline_from_config_v2(*args, **kwargs)

    def _execute_pipeline_gpu(*args, **kwargs):
        return _execute_pipeline(*args, **kwargs)


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
    """Run pipeline and prepare UI payload with Gradio components."""
    if image is None:
        return state or PipelineState(model_id="", pipeline=None), _empty_ui_payload(
            "Please provide an image before running the demo."
        )
    if not question.strip():
        return state or PipelineState(model_id="", pipeline=None), _empty_ui_payload(
            "Please enter a question before running the demo."
        )

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

        def create_pipeline():
            return _create_pipeline_from_config_v2_gpu(
                config_path=config_path,
                parallel_loading=parallel_loading,
            )

        pipeline = _get_pipeline(cache_key, create_pipeline)
        # Toggle batch evidence extraction on the cached pipeline.
        try:
            vlm = getattr(pipeline, "_vlm", None)
            if vlm is not None and hasattr(vlm, "set_batch_evidence_enabled"):
                vlm.set_batch_evidence_enabled(batch_captioning)
        except Exception:
            # UI toggle should never break pipeline execution.
            pass
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
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return (
            state or PipelineState(model_id="", pipeline=None),
            _empty_ui_payload(f"Pipeline error: {exc}"),
        )

    new_state = PipelineState(model_id=cache_key, pipeline=pipeline)
    payload = _prepare_ui_payload_components(rgb_image, result, MAX_UI_STEPS)
    return new_state, payload


def _prepare_ui_payload_components(
    image: Image.Image,
    result: PipelineResult,
    max_slots: int = MAX_UI_STEPS,
) -> Dict[str, object]:
    """Prepare UI payload using Gradio components instead of HTML."""

    # Prepare galleries
    roi_gallery: List[Tuple[Image.Image, str]] = []
    for idx, evidence in enumerate(result.evidence):
        color = EVIDENCE_COLORS[idx % len(EVIDENCE_COLORS)]
        annotated = _annotate_evidence_image(image, evidence, color)
        roi_gallery.append((annotated, _format_evidence_caption(evidence)))

    key_evidence_gallery: List[Tuple[Image.Image, str]] = []
    if result.key_evidence:
        for idx, kev in enumerate(result.key_evidence):
            color = KEY_EVIDENCE_COLORS[idx % len(KEY_EVIDENCE_COLORS)]
            annotated = _annotate_key_evidence_image(image, kev, color)
            key_evidence_gallery.append(
                (annotated, _format_key_evidence_caption(kev, idx))
            )

    # Prepare overview images
    roi_overview = _annotate_overview_image(image, result.evidence)
    key_evidence_overview = _annotate_key_evidence_overview(
        image, result.key_evidence or []
    )

    # Format text content using Gradio Markdown components
    final_answer_text = result.answer or "(no answer returned)"
    explanation_text = result.explanation or ""

    # Chain of Thought
    cot_text = result.cot_text or ""
    structured_steps_text = _format_structured_steps_text(
        result.steps[:max_slots], result.evidence
    )

    # Step-by-step panels (per reasoning step)
    step_panels = _prepare_step_panels_payload(
        image=image,
        steps=result.steps[:max_slots],
        evidence=result.evidence,
        max_slots=max_slots,
    )

    # Evidence table
    evidence_table_text = _format_evidence_table_text(result.evidence, image)

    # Key evidence
    key_evidence_text = _format_key_evidence_text(result.key_evidence or [])

    # Timing
    timing_text = _format_timing_text(result.timings)

    # Prompts
    reasoning_prompt_text = _format_prompt_text(result.reasoning_log, "Reasoning")
    grounding_prompts_text = _format_grounding_prompts_text(result.grounding_logs)
    answer_prompt_text = _format_prompt_text(result.answer_log, "Answer Synthesis")

    # Cropped images gallery
    cropped_images_gallery = _prepare_cropped_images_gallery(image, result.evidence)

    # Separate OCR and Captioning data
    ocr_evidence, captioning_evidence = _separate_ocr_captioning_data(
        result.evidence, image
    )
    ocr_gallery = _prepare_cropped_images_gallery(image, ocr_evidence)
    captioning_gallery = _prepare_cropped_images_gallery(image, captioning_evidence)

    # Format OCR and Captioning tables separately
    ocr_table_text = _format_evidence_table_text(ocr_evidence, image)
    captioning_table_text = _format_evidence_table_text(captioning_evidence, image)

    # Raw outputs
    raw_outputs_text = _format_raw_outputs_text(result)

    return {
        # Input image (for display at top of tabs)
        "input_image": image,
        # Final Answer
        "final_answer": final_answer_text,
        "explanation": explanation_text,
        "paraphrased_question": result.paraphrased_question or "",
        # Chain of Thought
        "cot_text": cot_text,
        "structured_steps": structured_steps_text,
        "reasoning_prompt": reasoning_prompt_text,
        # Step-by-step
        "step_panels": step_panels,
        # ROI Extraction
        "roi_overview_image": roi_overview,
        "roi_gallery": roi_gallery,
        "grounding_prompts": grounding_prompts_text,
        # Evidence
        "evidence_table": evidence_table_text,
        "cropped_images_gallery": cropped_images_gallery,
        # OCR Results
        "ocr_table": ocr_table_text,
        "ocr_gallery": ocr_gallery,
        # Captioning Results
        "captioning_table": captioning_table_text,
        "captioning_gallery": captioning_gallery,
        # Answer Synthesis
        "key_evidence_overview_image": key_evidence_overview,
        "key_evidence_gallery": key_evidence_gallery,
        "key_evidence_text": key_evidence_text,
        "answer_prompt": answer_prompt_text,
        # Raw Outputs (Debug)
        "raw_outputs": raw_outputs_text,
        # Performance
        "timing": timing_text,
    }


def _prepare_step_panels_payload(
    *,
    image: Image.Image,
    steps: List[ReasoningStep],
    evidence: List[GroundedEvidence],
    max_slots: int,
) -> List[Dict[str, object]]:
    evidences_by_step = _group_evidence_by_step(evidence)
    panels: List[Dict[str, object]] = []

    for slot_idx in range(max_slots):
        if slot_idx >= len(steps):
            panels.append({"visible": False, "markdown": "", "gallery": [], "has_gallery": False})
            continue

        step = steps[slot_idx]
        evs = evidences_by_step.get(step.index, [])

        md_lines = [
            f"**Step {step.index}:** {step.statement}",
            f"- Needs vision: {'Yes' if step.needs_vision else 'No'}",
        ]
        if hasattr(step, "need_ocr"):
            md_lines.append(f"- Needs OCR: {'Yes' if step.need_ocr else 'No'}")
        if step.reason:
            md_lines.append(f"- Reason: {step.reason}")

        if not evs:
            md_lines.append("- Evidence: _none_")

        gallery: List[Tuple[Image.Image, str]] = []
        for ev in evs[:6]:
            cropped = _crop_evidence_image(image, ev)
            bbox = ", ".join(f"{coord:.3f}" for coord in ev.bbox)
            caption_parts: List[str] = [f"bbox=({bbox})"]
            if ev.ocr_text:
                caption_parts.append(f"ocr: {ev.ocr_text.strip()[:160]}")
            if ev.description:
                caption_parts.append(f"caption: {ev.description.strip()[:200]}")
            gallery.append((cropped, " | ".join(caption_parts)))

        panels.append(
            {
                "visible": True,
                "markdown": "\n".join(md_lines),
                "gallery": gallery,
                "has_gallery": bool(gallery),
            }
        )

    return panels


def _format_structured_steps_text(
    steps: List,
    evidence: List[GroundedEvidence],
) -> str:
    """Format structured reasoning steps as text."""
    if not steps:
        return "_No reasoning steps returned._"

    evidences_by_step = _group_evidence_by_step(evidence)
    lines = []

    for step in steps:
        step_lines = [
            f"**Step {step.index}:** {step.statement}",
            f"- Needs vision: {'Yes' if step.needs_vision else 'No'}",
        ]
        # Always display need_ocr status
        if hasattr(step, "need_ocr"):
            step_lines.append(f"- Needs OCR: {'Yes' if step.need_ocr else 'No'}")
        else:
            step_lines.append("- Needs OCR: No (not specified)")
        if step.reason:
            step_lines.append(f"- Reason: {step.reason}")

        evs = evidences_by_step.get(step.index, [])
        if evs:
            step_lines.append(f"- Visual evidence items: {len(evs)}")
        else:
            step_lines.append("- No visual evidence returned for this step.")

        lines.append("\n".join(step_lines))

    return "\n\n".join(lines)


def _format_evidence_table_text(
    evidence: List[GroundedEvidence],
    image: Image.Image,
) -> str:
    """Format evidence as a table using Markdown."""
    if not evidence:
        return "_No visual evidence collected._"

    width, height = image.size
    lines = [
        "| Step | BBox (normalized) | BBox (pixels) | Description | OCR Text | Confidence |",
        "|:----:|:------------------:|:-------------:|:------------|:---------|:----------:|",
    ]

    for ev in evidence:
        x1_px = int(ev.bbox[0] * width)
        y1_px = int(ev.bbox[1] * height)
        x2_px = int(ev.bbox[2] * width)
        y2_px = int(ev.bbox[3] * height)

        bbox_norm = f"`[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]`"
        bbox_px = f"`[{x1_px}, {y1_px}, {x2_px}, {y2_px}]`"

        # IMPORTANT: Escape pipe characters to prevent breaking Markdown table
        desc = (ev.description or "N/A")[:100].replace("|", "\\|")  # Escape pipes
        ocr_text = (ev.ocr_text or "N/A")[:100].replace("|", "\\|")  # Escape pipes
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"

        # Highlight rows with OCR text (need_ocr=True)
        has_ocr = ev.ocr_text and ev.ocr_text.strip() and ev.ocr_text != "N/A"
        step_display = f"**{ev.step_index}**"
        if has_ocr:
            step_display = f"**{ev.step_index}** 🏷️"  # Add OCR indicator

        lines.append(
            f"| {step_display} | {bbox_norm} | {bbox_px} | {desc} | {ocr_text} | {conf} |"
        )

    return "\n".join(lines)


def _format_key_evidence_text(key_evidence: List[KeyEvidence]) -> str:
    """Format key evidence as text."""
    if not key_evidence:
        return "_No key evidence returned._"

    lines = []
    for idx, kev in enumerate(key_evidence, start=1):
        bbox = ", ".join(f"{coord:.3f}" for coord in kev.bbox)
        lines.append(f"**Key Evidence {idx}:**")
        lines.append(f"- Description: {kev.description}")
        lines.append(f"- Reasoning: {kev.reasoning}")
        lines.append(f"- BBox (normalized): [{bbox}]")
        lines.append("")

    return "\n".join(lines)


def _format_timing_text(timings: List) -> str:
    """Format timing information."""
    if not timings:
        return "_No timing data available._"

    lines = []
    total_entry = next((t for t in timings if t.name == "total_pipeline"), None)
    if total_entry:
        lines.append(f"**Total pipeline:** {total_entry.duration_ms/1000:.2f} s")

    for timing in timings:
        if timing.name == "total_pipeline":
            continue
        label = timing.name.replace("_", " ").title()
        if timing.step_index is not None:
            label += f" (step {timing.step_index})"
        lines.append(f"- {label}: {timing.duration_ms/1000:.2f} s")

    return "\n".join(lines)


def _format_prompt_text(log: Optional, title: str) -> str:
    """Format prompt log as text."""
    if not log or not log.prompt:
        return f"_No {title.lower()} prompt available._"

    lines = [f"### {title} Prompt", "```text", log.prompt, "```"]

    if log.response:
        lines.extend(["", "**Model Response:**", "```text", log.response, "```"])

    return "\n".join(lines)


def _format_grounding_prompts_text(logs: List) -> str:
    """Format grounding prompts as text."""
    if not logs:
        return "_No ROI prompts available._"

    lines = []
    for idx, log in enumerate(logs, start=1):
        lines.append(f"**Prompt {idx}:**")
        lines.append("```text")
        lines.append(log.prompt)
        lines.append("```")
        if log.response:
            lines.append(f"**Response {idx}:**")
            lines.append("```text")
            lines.append(log.response)
            lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _format_raw_outputs_text(result: PipelineResult) -> str:
    """Format raw outputs for debug tab."""
    lines = []

    # Reasoning stage
    if result.reasoning_log:
        lines.append("## Reasoning Stage")
        lines.append("### Prompt:")
        lines.append("```text")
        lines.append(result.reasoning_log.prompt)
        lines.append("```")
        if result.reasoning_log.response:
            lines.append("### Raw Response:")
            lines.append("```text")
            lines.append(result.reasoning_log.response)
            lines.append("```")
        lines.append("")

    # Grounding stage
    if result.grounding_logs:
        lines.append("## Grounding Stage")
        for idx, log in enumerate(result.grounding_logs, start=1):
            lines.append(f"### Step {log.step_index or idx}")
            lines.append("**Prompt:**")
            lines.append("```text")
            lines.append(log.prompt)
            lines.append("```")
            if log.response:
                lines.append("**Raw Response:**")
                lines.append("```text")
                lines.append(log.response)
                lines.append("```")
            lines.append("")

    # Evidence stage (OCR and Captioning)
    if result.evidence:
        lines.append("## Evidence Stage (OCR + Captioning)")
        for idx, ev in enumerate(result.evidence, start=1):
            lines.append(f"### Evidence {idx} (Step {ev.step_index})")
            lines.append(f"**BBox:** `{ev.bbox}`")
            if ev.ocr_text:
                lines.append("**OCR Text:**")
                lines.append("```text")
                lines.append(ev.ocr_text)
                lines.append("```")
            if ev.description:
                lines.append("**Caption:**")
                lines.append("```text")
                lines.append(ev.description)
                lines.append("```")
            lines.append("")

    # Synthesis stage
    if result.answer_log:
        lines.append("## Synthesis Stage")
        lines.append("### Prompt:")
        lines.append("```text")
        lines.append(result.answer_log.prompt)
        lines.append("```")
        if result.answer_log.response:
            lines.append("### Raw Response:")
            lines.append("```text")
            lines.append(result.answer_log.response)
            lines.append("```")
        lines.append("")

    if not lines:
        return "_No raw outputs available._"

    return "\n".join(lines)


def _prepare_cropped_images_gallery(
    image: Image.Image,
    evidence: List[GroundedEvidence],
) -> List[Tuple[Image.Image, str]]:
    """Prepare cropped images gallery for evidence regions."""
    gallery = []
    for idx, ev in enumerate(evidence, start=1):
        cropped = _crop_evidence_image(image, ev)
        caption = f"Step {ev.step_index}, Evidence {idx}"
        if ev.ocr_text:
            caption += f" | OCR: {ev.ocr_text[:50]}..."
        gallery.append((cropped, caption))
    return gallery


def _separate_ocr_captioning_data(
    evidence: List[GroundedEvidence],
    image: Image.Image,
) -> Tuple[List[GroundedEvidence], List[GroundedEvidence]]:
    """Separate evidence into OCR-only and Captioning-only lists."""
    ocr_evidence = [ev for ev in evidence if ev.ocr_text and ev.ocr_text.strip()]
    captioning_evidence = [
        ev for ev in evidence if ev.description and ev.description.strip()
    ]
    return ocr_evidence, captioning_evidence


def build_demo(
    pipeline_factory: Callable[[Optional[str]], CoRGIPipeline] | None = None,
    default_config: Optional[str] = None,
):
    """
    Build Gradio demo using native Gradio components.

    Args:
        pipeline_factory: Optional legacy factory (for backward compatibility)
        default_config: Default config file path

    Returns:
        Gradio Blocks instance
    """
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required to build the demo. Install gradio>=4.0."
        ) from exc

    if default_config is None:
        default_config = str(DEFAULT_QWEN_CONFIG)

    if pipeline_factory is not None:
        global _GLOBAL_FACTORY
        _GLOBAL_FACTORY = pipeline_factory
        logger.info("Registering legacy pipeline factory %s", pipeline_factory)

    # Get available config files
    available_configs = []
    if DEFAULT_CONFIG_DIR.exists():
        for config_file in sorted(DEFAULT_CONFIG_DIR.glob("*.yaml")):
            # Skip test configs for cleaner UI (they can still be used via path)
            if not config_file.name.startswith("test_"):
                available_configs.append(config_file.name)

    # Sort configs with preferred ones first
    def config_sort_key(name):
        # Preferred order: qwen_paddleocr_fastvlm, qwen_only, florence_qwen, then others
        if "paddleocr" in name.lower():
            return (0, name)
        elif name == "qwen_only.yaml":
            return (1, name)
        elif name == "florence_qwen.yaml":
            return (2, name)
        else:
            return (3, name)

    available_configs.sort(key=config_sort_key)

    if not available_configs:
        available_configs = ["qwen_only.yaml", "florence_qwen.yaml"]

    default_config_name = (
        Path(default_config).name
        if Path(default_config).exists()
        else available_configs[0]
    )

    # Warm up pipeline
    try:
        config_path_obj = Path(default_config)
        if not config_path_obj.is_absolute():
            full_path = DEFAULT_CONFIG_DIR / config_path_obj
            if not full_path.exists():
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
    except Exception as exc:
        logger.warning(f"Failed to warm up default pipeline: {exc}")

    demo_css = """
        .stage-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .final-answer-box {
            background: #e8f5e9;
            border: 2px solid #4caf50;
            border-radius: 8px;
            padding: 16px;
        }
    """

    # Gradio v6 moved/changed the Blocks CSS API; injecting <style> is the most compatible.
    with gr.Blocks(title="CoRGI Pipeline Demo") as demo:
        state = gr.State()

        gr.HTML(f"<style>{demo_css}</style>")

        # Header
        gr.Markdown("# CoRGI Pipeline\n\nChain of Reasoning with Grounded Insights - Scroll down to see step-by-step results")

        # =================================================================
        # INPUT SECTION
        # =================================================================
        gr.Markdown("---")
        gr.Markdown("## Input")
        
        with gr.Row():
            image_input = gr.Image(label="Input Image", type="pil", height=400)
            with gr.Column():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is happening in the image?",
                    lines=3,
                )
                run_button = gr.Button("Run CoRGI", variant="primary", size="lg")

        # Configuration (collapsible)
        with gr.Accordion("Advanced Configuration", open=False):
            with gr.Row():
                config_dropdown = gr.Dropdown(
                    label="Configuration",
                    choices=available_configs,
                    value=default_config_name,
                    info="Select pipeline configuration",
                )
                parallel_loading_checkbox = gr.Checkbox(
                    label="Parallel Model Loading",
                    value=True,
                    info="Load models in parallel (faster startup, more memory)",
                )
                batch_captioning_checkbox = gr.Checkbox(
                    label="Batch Captioning",
                    value=True,
                    info="Process multiple captions in batch (faster)",
                )
            with gr.Row():
                model_id_override_input = gr.Textbox(
                    label="Model ID Override (optional)",
                    placeholder="Leave blank to use config",
                    info="Override model ID from config (experimental)",
                )
                max_steps_slider = gr.Slider(
                    label="Max reasoning steps",
                    minimum=1,
                    maximum=6,
                    step=1,
                    value=3,
                )
                max_regions_slider = gr.Slider(
                    label="Max regions per step",
                    minimum=1,
                    maximum=6,
                    step=1,
                    value=1,
                )

        # =================================================================
        # RESULTS SECTION - Scrollable from top to bottom
        # =================================================================
        gr.Markdown("---")
        gr.Markdown("## Pipeline Results")
        
        # Hidden image displays (not needed in scrollable layout)
        input_image_final = gr.Image(visible=False)
        input_image_reasoning = gr.Image(visible=False)
        input_image_grounding = gr.Image(visible=False)
        input_image_evidence = gr.Image(visible=False)
        input_image_ocr = gr.Image(visible=False)
        input_image_captioning = gr.Image(visible=False)
        input_image_synthesis = gr.Image(visible=False)
        paraphrased_question_output = gr.Textbox(visible=False)

        # -----------------------------------------------------------------
        # STAGE 1: Reasoning
        # -----------------------------------------------------------------
        with gr.Accordion("Stage 1: Reasoning (Chain of Thought)", open=True):
            gr.Markdown("*The model generates step-by-step reasoning about the image and question.*")
            cot_output = gr.Textbox(
                label="Chain of Thought",
                lines=8,
                interactive=False,
            )
            structured_steps_output = gr.Markdown(
                label="Structured Reasoning Steps",
            )
            with gr.Accordion("View Reasoning Prompt", open=False):
                reasoning_prompt_output = gr.Markdown()

        # -----------------------------------------------------------------
        # STEP-BY-STEP VIEW (per reasoning step)
        # -----------------------------------------------------------------
        with gr.Accordion("Step-by-step View (Reasoning → Evidence)", open=True):
            gr.Markdown(
                "*This section shows each reasoning step together with its extracted evidence regions.*"
            )
            step_accordions = []
            step_markdown_outputs = []
            step_gallery_outputs = []
            for idx in range(MAX_UI_STEPS):
                with gr.Accordion(
                    f"Step {idx + 1}",
                    open=(idx == 0),
                    visible=False,
                ) as step_acc:
                    step_md = gr.Markdown()
                    step_gallery = gr.Gallery(
                        label="Evidence regions (cropped)",
                        columns=3,
                        height=240,
                        allow_preview=True,
                        visible=False,
                    )
                step_accordions.append(step_acc)
                step_markdown_outputs.append(step_md)
                step_gallery_outputs.append(step_gallery)

        # -----------------------------------------------------------------
        # STAGE 2: ROI Extraction (Grounding)
        # -----------------------------------------------------------------
        with gr.Accordion("Stage 2: ROI Extraction (Grounding)", open=True):
            gr.Markdown("*The model identifies regions of interest (bounding boxes) for each reasoning step.*")
            with gr.Row():
                roi_overview_output = gr.Image(
                    label="All ROIs Annotated",
                    type="pil",
                    height=350,
                    visible=False,
                )
            roi_gallery_output = gr.Gallery(
                label="Individual ROI Regions",
                columns=3,
                height=300,
                allow_preview=True,
                visible=False,
            )
            with gr.Accordion("View Grounding Prompts", open=False):
                grounding_prompts_output = gr.Markdown()

        # -----------------------------------------------------------------
        # STAGE 3: Evidence Extraction (OCR + Captioning)
        # -----------------------------------------------------------------
        with gr.Accordion("Stage 3: Evidence Extraction", open=True):
            gr.Markdown("*For each ROI, the model extracts text (OCR) or generates captions.*")
            
            # Evidence summary table
            evidence_table_output = gr.Markdown(
                label="Evidence Summary",
            )
            
            # Cropped regions gallery
            cropped_images_gallery_output = gr.Gallery(
                label="Cropped Evidence Regions",
                columns=4,
                height=250,
                allow_preview=True,
                visible=False,
            )
            
            # Sub-sections for OCR and Captioning
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### OCR Results")
                    ocr_table_output = gr.Markdown()
                    ocr_gallery_output = gr.Gallery(
                        label="OCR Regions",
                        columns=2,
                        height=200,
                        allow_preview=True,
                        visible=False,
                    )
                with gr.Column():
                    gr.Markdown("### Captioning Results")
                    captioning_table_output = gr.Markdown()
                    captioning_gallery_output = gr.Gallery(
                        label="Captioned Regions",
                        columns=2,
                        height=200,
                        allow_preview=True,
                        visible=False,
                    )

        # -----------------------------------------------------------------
        # STAGE 4: Answer Synthesis (FINAL)
        # -----------------------------------------------------------------
        gr.Markdown("---")
        gr.Markdown("## Final Answer")
        
        with gr.Row():
            with gr.Column(scale=2):
                final_answer_output = gr.Textbox(
                    label="Answer",
                    lines=3,
                    interactive=False,
                    elem_classes=["final-answer-box"],
                )
                explanation_output = gr.Textbox(
                    label="Explanation",
                    lines=4,
                    interactive=False,
                )
            with gr.Column(scale=1):
                key_evidence_overview_output = gr.Image(
                    label="Key Evidence Overview",
                    type="pil",
                    height=250,
                    visible=False,
                )
        
        # Key evidence details
        with gr.Accordion("Key Evidence Details", open=True):
            key_evidence_gallery_output = gr.Gallery(
                label="Key Evidence Regions",
                columns=3,
                height=250,
                allow_preview=True,
                visible=False,
            )
            key_evidence_text_output = gr.Markdown()
            
            with gr.Accordion("View Synthesis Prompt", open=False):
                answer_prompt_output = gr.Markdown()

        # -----------------------------------------------------------------
        # DEBUG & PERFORMANCE (Collapsed by default)
        # -----------------------------------------------------------------
        gr.Markdown("---")
        
        with gr.Accordion("Debug: Raw Model Outputs", open=False):
            raw_outputs_output = gr.Markdown()

        with gr.Accordion("Performance Metrics", open=False):
            timing_output = gr.Markdown()

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
            config_path_obj = DEFAULT_CONFIG_DIR / config_file
            if not config_path_obj.exists():
                config_path_obj = Path(config_file)
            config_path = str(config_path_obj)

            new_state, payload = _run_pipeline(
                state_data,
                image,
                question,
                int(max_steps),
                int(max_regions),
                config_path,
                parallel_loading,
                batch_captioning,
                model_id_override.strip() if model_id_override else None,
            )

            # Return all outputs with proper visibility handling
            has_paraphrased = bool(payload.get("paraphrased_question"))
            has_roi_overview = payload.get("roi_overview_image") is not None
            has_roi_gallery = len(payload.get("roi_gallery", [])) > 0
            has_key_overview = payload.get("key_evidence_overview_image") is not None
            has_key_gallery = len(payload.get("key_evidence_gallery", [])) > 0
            has_cropped_gallery = len(payload.get("cropped_images_gallery", [])) > 0
            has_ocr_gallery = len(payload.get("ocr_gallery", [])) > 0
            has_captioning_gallery = len(payload.get("captioning_gallery", [])) > 0
            input_image = payload.get("input_image")
            step_panels = payload.get("step_panels") or [
                {"visible": False, "markdown": "", "gallery": [], "has_gallery": False}
                for _ in range(MAX_UI_STEPS)
            ]

            outputs = [
                new_state,
                # Final Answer Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_final
                payload["final_answer"],
                payload["explanation"],
                gr.update(
                    value=payload.get("paraphrased_question", ""),
                    visible=has_paraphrased,
                ),
                # Reasoning Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_reasoning
                payload["cot_text"],
                payload["structured_steps"],
                payload["reasoning_prompt"],
                # Step-by-step panels (fixed slots)
                *list(
                    itertools.chain.from_iterable(
                        (
                            gr.update(
                                visible=panel.get("visible", False),
                                open=(i == 0 and panel.get("visible", False)),
                            ),
                            panel.get("markdown", ""),
                            gr.update(
                                value=panel.get("gallery", []),
                                visible=panel.get("has_gallery", False),
                            ),
                        )
                        for i, panel in enumerate(step_panels)
                    )
                ),
                # Grounding Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_grounding
                gr.update(
                    value=payload.get("roi_overview_image"), visible=has_roi_overview
                ),
                gr.update(value=payload["roi_gallery"], visible=has_roi_gallery),
                payload["grounding_prompts"],
                # Evidence Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_evidence
                payload["evidence_table"],
                gr.update(
                    value=payload["cropped_images_gallery"], visible=has_cropped_gallery
                ),
                # OCR Results Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_ocr
                payload["ocr_table"],
                gr.update(value=payload["ocr_gallery"], visible=has_ocr_gallery),
                # Captioning Results Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_captioning
                payload["captioning_table"],
                gr.update(
                    value=payload["captioning_gallery"], visible=has_captioning_gallery
                ),
                # Synthesis Tab
                gr.update(
                    value=input_image, visible=input_image is not None
                ),  # input_image_synthesis
                gr.update(
                    value=payload.get("key_evidence_overview_image"),
                    visible=has_key_overview,
                ),
                gr.update(
                    value=payload["key_evidence_gallery"], visible=has_key_gallery
                ),
                payload["key_evidence_text"],
                payload["answer_prompt"],
                # Raw Outputs Tab
                payload["raw_outputs"],
                # Performance Tab
                payload["timing"],
            ]

            return outputs

        # Connect button to function
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
            outputs=[
                state,
                # Final Answer Tab
                input_image_final,
                final_answer_output,
                explanation_output,
                paraphrased_question_output,
                # Reasoning Tab
                input_image_reasoning,
                cot_output,
                structured_steps_output,
                reasoning_prompt_output,
                # Step-by-step panels (accordion + markdown + gallery) x MAX_UI_STEPS
                *list(
                    itertools.chain.from_iterable(
                        (acc, md, gal)
                        for acc, md, gal in zip(
                            step_accordions, step_markdown_outputs, step_gallery_outputs
                        )
                    )
                ),
                # Grounding Tab
                input_image_grounding,
                roi_overview_output,
                roi_gallery_output,
                grounding_prompts_output,
                # Evidence Tab
                input_image_evidence,
                evidence_table_output,
                cropped_images_gallery_output,
                # OCR Results Tab
                input_image_ocr,
                ocr_table_output,
                ocr_gallery_output,
                # Captioning Results Tab
                input_image_captioning,
                captioning_table_output,
                captioning_gallery_output,
                # Synthesis Tab
                input_image_synthesis,
                key_evidence_overview_output,
                key_evidence_gallery_output,
                key_evidence_text_output,
                answer_prompt_output,
                # Raw Outputs Tab
                raw_outputs_output,
                # Performance Tab
                timing_output,
            ],
        )

    return demo


def launch_demo(
    *,
    pipeline_factory: Callable[[Optional[str]], CoRGIPipeline] | None = None,
    **launch_kwargs,
) -> None:
    """Launch the Gradio demo."""
    demo = build_demo(pipeline_factory=pipeline_factory)
    demo.launch(**launch_kwargs)


__all__ = [
    "PipelineState",
    "build_demo",
    "launch_demo",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_QWEN_CONFIG",
]
