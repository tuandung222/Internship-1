#!/usr/bin/env python3
"""
CoRGI "Streamlit-like" (scrollable) Gradio UI.

Why this file exists:
- Gradio 6 serves images via `/gradio_api/file=...` endpoints.
- In some environments, the browser ends up requesting `/file=...` or a mismatched host,
  which causes image placeholders (white/blank) in Gallery/Image components.
- This UI embeds all images as base64 `data:` URIs inside a single scrollable HTML report,
  so the report is self-contained and images reliably render.

Note:
- This file is intentionally additive. It does NOT modify `app_unified.py`.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

# Reduce TensorFlow logs if TF is present (pulled indirectly by some ML packages).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence.*")

logger = logging.getLogger("corgi.app_streamit")


@dataclass
class _UiState:
    cache_key: str = ""


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _resolve_config_path(config_value: str) -> str:
    config_path = Path(config_value)
    if config_path.is_absolute():
        return str(config_path)
    configs_dir = Path(__file__).parent / "configs"
    candidate = configs_dir / config_path
    return str(candidate if candidate.exists() else config_path)


def _available_configs() -> list[str]:
    configs_dir = Path(__file__).parent / "configs"
    if not configs_dir.exists():
        return []
    return sorted(p.name for p in configs_dir.glob("*.yaml"))


def _resize_to_max_dim(image: Image.Image, max_dim: int) -> Image.Image:
    if max_dim <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    return image.resize((new_w, new_h), resample=resample)


def _image_to_data_uri(
    image: Image.Image,
    *,
    max_dim: int = 1200,
    fmt: str = "JPEG",
    quality: int = 85,
) -> str:
    img = image.convert("RGB")
    img = _resize_to_max_dim(img, max_dim=max_dim)
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        mime = "image/jpeg"
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _h(text: str) -> str:
    return html.escape(text or "")


def _html_pre(text: str) -> str:
    return f"<pre><code>{_h(text)}</code></pre>"


def _html_details(summary: str, inner_html: str, *, open: bool = False) -> str:
    return f"<details{' open' if open else ''}><summary>{_h(summary)}</summary>{inner_html}</details>"


def _html_img(data_uri: str, *, caption: str | None = None) -> str:
    caption_html = f"<figcaption>{_h(caption)}</figcaption>" if caption else ""
    return (
        "<figure>"
        f'<img src="{_h(data_uri)}" loading="lazy" />'
        f"{caption_html}"
        "</figure>"
    )


def _crop_bbox(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = image.size

    left = int(x1 * w)
    top = int(y1 * h)
    right = int(x2 * w)
    bottom = int(y2 * h)

    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    cropped = image.crop((left, top, right, bottom)).convert("RGB")
    bordered = Image.new("RGB", (cropped.width + 4, cropped.height + 4), (255, 255, 255))
    bordered.paste(cropped, (2, 2))
    return bordered


def _render_report_html(
    image_rgb: Image.Image,
    result,
    *,
    config_path: str,
    parallel_loading: bool,
    batch_captioning: bool,
    max_steps: int,
    max_regions: int,
) -> str:
    # Import heavy helpers lazily.
    from corgi.core.types import evidences_to_serializable, prompt_logs_to_serializable, steps_to_serializable
    from corgi.ui.gradio_app_html import (
        EVIDENCE_COLORS,
        KEY_EVIDENCE_COLORS,
        _annotate_evidence_image,
        _annotate_key_evidence_image,
        _annotate_key_evidence_overview,
        _annotate_overview_image,
        _group_evidence_by_step,
    )

    style = """
    <style>
      :root { --fg:#111827; --muted:#6b7280; --bg:#ffffff; --card:#f9fafb; --border:#e5e7eb; }
      body { background: var(--bg); }
      .corgi-report { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
                      color: var(--fg); line-height: 1.5; }
      .corgi-report h2 { margin: 18px 0 8px; font-size: 18px; }
      .corgi-report h3 { margin: 14px 0 8px; font-size: 15px; }
      .corgi-report p { margin: 6px 0; }
      .corgi-report .meta { color: var(--muted); font-size: 12px; }
      .corgi-report .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin: 10px 0; }
      .corgi-report pre { overflow: auto; background: #0b1020; color: #e5e7eb; padding: 10px; border-radius: 10px; }
      .corgi-report code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
      .corgi-report details { border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; margin: 10px 0; background: #fff; }
      .corgi-report summary { cursor: pointer; font-weight: 600; }
      .corgi-report figure { margin: 10px 0; }
      .corgi-report img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 10px; background: #fff; }
      .corgi-report figcaption { color: var(--muted); font-size: 12px; margin-top: 6px; }
      .corgi-report table { border-collapse: collapse; width: 100%; }
      .corgi-report th, .corgi-report td { border: 1px solid var(--border); padding: 8px; vertical-align: top; font-size: 12px; }
      .corgi-report th { background: #f3f4f6; text-align: left; }
      .corgi-report .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }
      .corgi-report .warn { color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; padding: 8px 10px; border-radius: 10px; }
    </style>
    """

    header = (
        '<div class="corgi-report">'
        "<h2>Pipeline Report</h2>"
        f'<p class="meta">Config: <code>{_h(config_path)}</code> · parallel_loading={parallel_loading} · batch_captioning={batch_captioning} · max_steps={max_steps} · max_regions={max_regions}</p>'
        "</div>"
    )

    # ---- Input ----
    input_img_uri = _image_to_data_uri(image_rgb, max_dim=1200)
    input_section = (
        '<div class="corgi-report">'
        '<div class="card">'
        "<h2>Input</h2>"
        f"<p><span class='badge'>Question</span> {_h(result.question)}</p>"
        f"{_html_img(input_img_uri, caption='Original image (embedded)')}"
        "</div>"
        "</div>"
    )

    # ---- Stage 1: Reasoning ----
    steps_json = json.dumps(
        {"steps": steps_to_serializable(result.steps)},
        ensure_ascii=False,
        indent=2,
    )
    stage1_inner = ""
    if result.cot_text:
        stage1_inner += "<h3>Chain-of-Thought (raw text)</h3>" + _html_pre(result.cot_text)
    else:
        stage1_inner += "<p class='meta'>(No separate CoT text returned.)</p>"

    stage1_inner += "<h3>Structured Reasoning (JSON)</h3>" + _html_pre(steps_json)

    if result.reasoning_log and result.reasoning_log.prompt:
        reasoning_prompt = _html_pre(result.reasoning_log.prompt)
        reasoning_resp = _html_pre(result.reasoning_log.response or "")
        stage1_inner += _html_details("View reasoning prompt", reasoning_prompt, open=False)
        stage1_inner += _html_details("View raw reasoning model output", reasoning_resp, open=False)
    else:
        stage1_inner += "<p class='meta'>(No reasoning prompt log available.)</p>"

    stage1 = '<div class="corgi-report">' + _html_details("Stage 1: Reasoning", stage1_inner, open=True) + "</div>"

    # ---- Stage 2 + 3: Grounding + Evidence ----
    evidences_by_step = _group_evidence_by_step(result.evidence)

    roi_overview = _annotate_overview_image(image_rgb, result.evidence) if result.evidence else None
    stage2_inner = ""
    if roi_overview is not None:
        roi_uri = _image_to_data_uri(roi_overview, max_dim=1400)
        stage2_inner += "<h3>All ROIs (annotated)</h3>" + _html_img(roi_uri, caption="Stage 2 overview: all extracted regions")
    else:
        stage2_inner += "<p class='meta'>(No ROI regions extracted.)</p>"

    # Step-by-step blocks
    for step in result.steps:
        step_evs = evidences_by_step.get(step.index, [])
        needs_vision = "yes" if step.needs_vision else "no"
        need_ocr = "yes" if getattr(step, "need_ocr", False) else "no"
        step_block = (
            "<div class='card'>"
            f"<p><b>Step {step.index}:</b> {_h(step.statement)}</p>"
            f"<p class='meta'>needs_vision={needs_vision} · need_ocr={need_ocr}</p>"
        )

        if step_evs:
            step_overview = _annotate_overview_image(image_rgb, step_evs)
            if step_overview is not None:
                step_block += _html_img(
                    _image_to_data_uri(step_overview, max_dim=1200),
                    caption=f"Annotated image for step {step.index}",
                )

            for ev_idx, ev in enumerate(step_evs, start=1):
                bbox_str = ", ".join(f"{c:.3f}" for c in ev.bbox)
                crop = _crop_bbox(image_rgb, ev.bbox)
                crop_uri = _image_to_data_uri(crop, max_dim=700)
                caption = ev.description or "(no caption)"
                ocr = ev.ocr_text or ""
                conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "n/a"
                step_block += (
                    "<div class='card'>"
                    f"<p><b>Evidence {ev_idx}</b> · bbox=[{_h(bbox_str)}] · conf={_h(conf)}</p>"
                    f"{_html_img(crop_uri, caption='Cropped ROI')}"
                    f"<p><b>Caption:</b> {_h(caption)}</p>"
                )
                if ocr:
                    step_block += f"<p><b>OCR:</b> {_h(ocr)}</p>"
                step_block += "</div>"
        else:
            step_block += "<p class='meta'>(No evidence items for this step.)</p>"

        step_block += "</div>"
        stage2_inner += _html_details(f"Step {step.index}: Reasoning → Evidence", step_block, open=(step.index == 1))

    # Grounding prompts (collapsed)
    if getattr(result, "grounding_logs", None):
        grounding_logs = prompt_logs_to_serializable(result.grounding_logs)
        stage2_inner += _html_details(
            "View grounding/evidence prompts (raw)",
            _html_pre(json.dumps(grounding_logs, ensure_ascii=False, indent=2)),
            open=False,
        )

    stage2 = '<div class="corgi-report">' + _html_details("Stage 2–3: Grounding & Evidence", stage2_inner, open=True) + "</div>"

    # ---- Stage 4: Synthesis ----
    synthesis_input = {
        "question": result.question,
        "steps": steps_to_serializable(result.steps),
        "evidence": evidences_to_serializable(result.evidence),
    }
    synthesis_output = {
        "answer": result.answer,
        "explanation": result.explanation,
        "key_evidence": [
            {"bbox": list(ke.bbox), "description": ke.description, "reasoning": ke.reasoning}
            for ke in (result.key_evidence or [])
        ],
    }

    stage4_inner = "<h3>Answer</h3>"
    stage4_inner += f"<p><b>Final Answer:</b> {_h(result.answer or '(no answer returned)')}</p>"
    if result.explanation:
        stage4_inner += f"<p><b>Explanation:</b> {_h(result.explanation)}</p>"

    if result.key_evidence:
        key_overview = _annotate_key_evidence_overview(image_rgb, result.key_evidence)
        if key_overview is not None:
            stage4_inner += "<h3>Key Evidence Overview</h3>"
            stage4_inner += _html_img(
                _image_to_data_uri(key_overview, max_dim=1400),
                caption="Key evidence bboxes overlaid on the original image",
            )

        # Per key evidence detail (with crops)
        for idx, kev in enumerate(result.key_evidence, start=1):
            color = KEY_EVIDENCE_COLORS[(idx - 1) % len(KEY_EVIDENCE_COLORS)]
            annotated = _annotate_key_evidence_image(image_rgb, kev, color)
            crop = _crop_bbox(image_rgb, kev.bbox)
            stage4_inner += (
                "<div class='card'>"
                f"<p><b>Key Evidence {idx}</b></p>"
                f"<p><b>Description:</b> {_h(kev.description)}</p>"
                f"<p><b>Reasoning:</b> {_h(kev.reasoning)}</p>"
                f"{_html_img(_image_to_data_uri(annotated, max_dim=1200), caption='Annotated key evidence')}"
                f"{_html_img(_image_to_data_uri(crop, max_dim=700), caption='Cropped key evidence')}"
                "</div>"
            )
    else:
        stage4_inner += "<p class='meta'>(No key evidence returned.)</p>"

    stage4_inner += "<h3>Synthesis I/O</h3>"
    stage4_inner += _html_details(
        "View synthesis input (JSON)",
        _html_pre(json.dumps(synthesis_input, ensure_ascii=False, indent=2)),
        open=False,
    )
    stage4_inner += _html_details(
        "View synthesis output (parsed JSON)",
        _html_pre(json.dumps(synthesis_output, ensure_ascii=False, indent=2)),
        open=False,
    )

    if result.answer_log and result.answer_log.prompt:
        stage4_inner += _html_details("View synthesis prompt (raw)", _html_pre(result.answer_log.prompt), open=False)
        stage4_inner += _html_details(
            "View synthesis model output (raw)",
            _html_pre(result.answer_log.response or ""),
            open=False,
        )
    else:
        stage4_inner += "<p class='warn'>No answer synthesis prompt available (answer_log is empty).</p>"

    stage4 = '<div class="corgi-report">' + _html_details("Stage 4: Answer Synthesis", stage4_inner, open=True) + "</div>"

    # ---- Timings ----
    timings = getattr(result, "timings", None) or []
    timing_rows = ""
    for t in timings:
        name = getattr(t, "name", "")
        ms = getattr(t, "duration_ms", 0.0)
        step_idx = getattr(t, "step_index", None)
        timing_rows += (
            "<tr>"
            f"<td>{_h(name)}</td>"
            f"<td>{_h(str(step_idx) if step_idx is not None else '')}</td>"
            f"<td>{_h(f'{ms/1000.0:.2f}')}</td>"
            "</tr>"
        )
    timing_table = (
        "<table><thead><tr><th>Stage</th><th>Step</th><th>Seconds</th></tr></thead>"
        f"<tbody>{timing_rows}</tbody></table>"
        if timing_rows
        else "<p class='meta'>(No timing data.)</p>"
    )
    perf = '<div class="corgi-report">' + _html_details("Performance", timing_table, open=False) + "</div>"

    return style + header + input_section + stage1 + stage2 + stage4 + perf


def _run_stream_report(
    state: Optional[_UiState],
    image: Image.Image | None,
    question: str,
    config_name: str,
    parallel_loading: bool,
    batch_captioning: bool,
    max_steps: int,
    max_regions: int,
) -> tuple[_UiState, str]:
    # Lazy imports to keep `--help` fast.
    from corgi.ui.gradio_app_html import _create_pipeline_from_config_v2, _execute_pipeline, _get_pipeline, _make_cache_key

    if image is None:
        return state or _UiState(), "<div class='corgi-report'><p class='warn'>Please provide an image.</p></div>"
    if not question.strip():
        return state or _UiState(), "<div class='corgi-report'><p class='warn'>Please enter a question.</p></div>"

    config_path = _resolve_config_path(config_name)
    cache_key = _make_cache_key(
        config_path=config_path,
        parallel_loading=bool(parallel_loading),
        batch_captioning=bool(batch_captioning),
        model_id_override=None,
    )

    try:
        pipeline = _get_pipeline(
            cache_key,
            lambda: _create_pipeline_from_config_v2(
                config_path=config_path,
                parallel_loading=bool(parallel_loading),
            ),
        )

        # Toggle batch evidence extraction on the cached pipeline (if supported).
        try:
            vlm = getattr(pipeline, "_vlm", None)
            if vlm is not None and hasattr(vlm, "set_batch_evidence_enabled"):
                vlm.set_batch_evidence_enabled(bool(batch_captioning))
        except Exception:
            pass

        image_rgb = image.convert("RGB")
        result = _execute_pipeline(
            image=image_rgb,
            question=question.strip(),
            max_steps=int(max_steps),
            max_regions=int(max_regions),
            pipeline=pipeline,
        )

        report_html = _render_report_html(
            image_rgb,
            result,
            config_path=config_path,
            parallel_loading=bool(parallel_loading),
            batch_captioning=bool(batch_captioning),
            max_steps=int(max_steps),
            max_regions=int(max_regions),
        )
        return _UiState(cache_key=cache_key), report_html
    except Exception as exc:
        logger.exception("Stream report failed: %s", exc)
        return (
            state or _UiState(),
            f"<div class='corgi-report'><p class='warn'>Pipeline error: {_h(str(exc))}</p></div>",
        )


def _build_demo(default_config: str) -> "gradio.Blocks":
    import gradio as gr

    configs = _available_configs()
    default_config_name = Path(default_config).name if default_config else (configs[0] if configs else "")

    with gr.Blocks(title="CoRGI Stream Report") as demo:
        gr.Markdown("# CoRGI — Scrollable Pipeline Report")
        gr.Markdown(
            "Giao diện này hiển thị toàn bộ pipeline theo dạng báo cáo cuộn xuống (ảnh được nhúng base64 để tránh lỗi ảnh trắng)."
        )

        state = gr.State(value=_UiState())

        with gr.Row():
            image_input = gr.Image(label="Input image", type="pil")
            question_input = gr.Textbox(
                label="Question",
                placeholder="What is happening in the image?",
                lines=2,
            )

        with gr.Row():
            config_dropdown = gr.Dropdown(
                label="Configuration",
                choices=configs or [default_config_name],
                value=default_config_name,
            )
            parallel_loading = gr.Checkbox(label="Parallel loading", value=True)
            batch_captioning = gr.Checkbox(label="Batch evidence", value=True)

        with gr.Row():
            max_steps = gr.Slider(label="Max steps", minimum=1, maximum=6, step=1, value=3)
            max_regions = gr.Slider(label="Max regions/step", minimum=1, maximum=5, step=1, value=1)
            run_btn = gr.Button("Run", variant="primary")

        report = gr.HTML(label="Report")

        run_btn.click(
            fn=_run_stream_report,
            inputs=[
                state,
                image_input,
                question_input,
                config_dropdown,
                parallel_loading,
                batch_captioning,
                max_steps,
                max_regions,
            ],
            outputs=[state, report],
        )

    return demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoRGI Streamlit-like (scroll) Gradio UI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "qwen_florence2_smolvlm2_v2.yaml"),
        help="Path to config YAML (default: qwen_florence2_smolvlm2_v2.yaml)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--no-warm-up", action="store_true", help="Skip model warm-up")
    parser.add_argument("--sequential-loading", action="store_true", help="Warm-up with sequential model loading")
    return parser.parse_args()


def _warm_up_if_needed(args: argparse.Namespace) -> None:
    if args.no_warm_up:
        return
    try:
        from corgi.utils.warm_up import verify_cuda_ready, warm_up_pipeline, WarmUpConfig

        if not verify_cuda_ready():
            return
        warm_up_pipeline(
            config_path=args.config,
            warm_up_config=WarmUpConfig(sequential_loading=bool(args.sequential_loading)),
            use_v2=False,  # Keep consistent with the standard UI pipeline class.
        )
    except Exception as exc:
        logger.warning("Warm-up skipped due to error: %s", exc)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    _warm_up_if_needed(args)

    demo = _build_demo(default_config=args.config)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=bool(args.share),
        show_error=True,
    )


if __name__ == "__main__":
    sys.exit(main())

