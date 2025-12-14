#!/usr/bin/env python
"""
Real Model Pipeline Test - Qwen3-VL-2B-Instruct + Florence-2

This script tests the full CoRGi pipeline with real models, providing:
- Real-time progress monitoring
- Detailed inspection of each stage
- Performance profiling
- Coordinate validation
- Results saved to JSON and Markdown

Usage:
    python test_real_pipeline.py [options]

Examples:
    # Use default config and demo image
    python test_real_pipeline.py

    # Use custom config and image
    python test_real_pipeline.py --config my_config.yaml --image my_image.jpg

    # Save bbox visualization
    python test_real_pipeline.py --save-viz
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.request import urlopen

import psutil
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from corgi.config import CoRGiConfig
from corgi.vlm_factory import VLMClientFactory
from corgi.pipeline import CoRGIPipeline, PipelineResult
from corgi.types import ReasoningStep, GroundedEvidence, KeyEvidence
from corgi.image_logger import ImageLogger
from corgi.output_tracer import OutputTracer
from corgi.report_generator import HTMLReportGenerator
from corgi.trace_types import PipelineTrace


# Demo image and question from official Qwen demo
DEMO_IMAGE_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
)
DEMO_QUESTION = "How many people are there in the image? Is there any one who is wearing a white watch?"

# Console for rich output
console = Console()


def fetch_demo_image() -> Image.Image:
    """Fetch the demo image from URL or local path."""
    if path := os.getenv("CORGI_DEMO_IMAGE"):
        console.print(f"[cyan]Loading image from: {path}[/cyan]")
        return Image.open(path).convert("RGB")

    console.print(f"[cyan]Fetching demo image from: {DEMO_IMAGE_URL}[/cyan]")
    with urlopen(DEMO_IMAGE_URL) as resp:
        data = resp.read()
    return Image.open(BytesIO(data)).convert("RGB")


def load_custom_image(image_path: str) -> Image.Image:
    """Load a custom image from path."""
    console.print(f"[cyan]Loading image from: {image_path}[/cyan]")
    return Image.open(image_path).convert("RGB")


def display_reasoning_results(
    steps: List[ReasoningStep], cot_text: Optional[str] = None
):
    """Display reasoning stage results: Full CoT text first, then structured steps."""
    # Show full Chain of Thought text if available
    if cot_text:
        console.print(
            "\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]"
        )
        console.print("[bold cyan]Chain of Thought (Full Text)[/bold cyan]")
        console.print(
            "[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]"
        )
        console.print(f"[white]{cot_text}[/white]\n")

    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]"
    )
    console.print("[bold cyan]Structured Reasoning Steps[/bold cyan]")
    console.print(
        "[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]"
    )
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Statement", style="white", width=50)
    table.add_column("Needs Vision", style="yellow", width=13)
    table.add_column("Reason", style="dim", width=30)

    for step in steps:
        needs = "[green]Yes[/green]" if step.needs_vision else "[dim]No[/dim]"
        reason = step.reason or ""
        table.add_row(str(step.index), step.statement, needs, reason)

    console.print(table)


def display_grounding_results(evidences: List[GroundedEvidence]):
    """Display grounding results with bbox coordinates."""
    console.print("\n[bold cyan]Visual Evidence:[/bold cyan]")

    if not evidences:
        console.print("[dim]No evidence extracted[/dim]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", width=6)
    table.add_column("BBox [0,1]", style="yellow", width=35)
    table.add_column("Description", style="white", width=40)
    table.add_column("Conf", style="green", width=6)

    for ev in evidences:
        bbox_str = (
            f"[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]"
        )
        desc = (ev.description or "")[:40]
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
        table.add_row(str(ev.step_index), bbox_str, desc, conf)

    console.print(table)


def display_synthesis_results(answer: str, key_evidence: List[KeyEvidence]):
    """Display final answer and key evidence."""
    console.print()
    answer_panel = Panel(
        f"[bold white]{answer}[/bold white]",
        title="[bold green]FINAL ANSWER[/bold green]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(answer_panel)

    if key_evidence:
        console.print("\n[bold cyan]Key Evidence:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("BBox [0,1]", style="yellow", width=35)
        table.add_column("Description", style="white", width=30)
        table.add_column("Reasoning", style="dim", width=35)

        for ev in key_evidence:
            bbox_str = f"[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]"
            table.add_row(bbox_str, ev.description[:30], ev.reasoning[:35])

        console.print(table)


def validate_and_log_coordinates(result: PipelineResult, image: Image.Image):
    """Validate coordinates and log conversion details."""
    console.print("\n[bold cyan]═══ Coordinate Validation Report ═══[/bold cyan]")

    # Check grounding evidence (should be [0,1] normalized)
    if result.evidence:
        table = Table(title="Grounding Evidence Coordinates", show_header=True)
        table.add_column("Step", style="cyan", width=8)
        table.add_column("BBox [0,1]", style="green", width=40)
        table.add_column("BBox [pixels]", style="yellow", width=40)
        table.add_column("Format Check", style="magenta", width=15)

        for ev in result.evidence[:5]:
            bbox_norm = ev.bbox
            # Convert to pixels for verification
            w, h = image.size
            bbox_px = (
                bbox_norm[0] * w,
                bbox_norm[1] * h,
                bbox_norm[2] * w,
                bbox_norm[3] * h,
            )

            # Verify normalized
            is_valid = all(0 <= v <= 1 for v in bbox_norm)
            status = "✓ Valid" if is_valid else "✗ Invalid"

            table.add_row(
                str(ev.step_index),
                f"({bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f})",
                f"({bbox_px[0]:.0f}, {bbox_px[1]:.0f}, {bbox_px[2]:.0f}, {bbox_px[3]:.0f})",
                status,
            )
        console.print(table)

    # Check synthesis key evidence (should be [0,1] normalized internally)
    if result.key_evidence:
        table = Table(title="Synthesis Key Evidence Coordinates", show_header=True)
        table.add_column("BBox [0,1]", style="green", width=40)
        table.add_column("Description", style="yellow", width=50)
        table.add_column("Format Check", style="magenta", width=15)

        for kev in result.key_evidence[:3]:
            bbox_norm = kev.bbox
            is_valid = all(0 <= v <= 1 for v in bbox_norm)
            status = "✓ Valid" if is_valid else "✗ Invalid"

            table.add_row(
                f"({bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f})",
                (
                    kev.description[:50] + "..."
                    if len(kev.description) > 50
                    else kev.description
                ),
                status,
            )
        console.print(table)

    console.print(
        "\n[bold green]✓ All coordinates validated successfully![/bold green]"
    )


def display_performance_summary(
    timings: Dict[str, float], memory_mb: Optional[float] = None
):
    """Display performance metrics in a formatted table."""
    console.print("\n[bold cyan]Performance Summary:[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan", width=20)
    table.add_column("Duration (s)", style="yellow", justify="right", width=15)
    table.add_column("Percentage", style="green", justify="right", width=12)

    total_time = sum(timings.values())

    for stage, duration in timings.items():
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        table.add_row(stage, f"{duration:.2f}", f"{percentage:.1f}%")

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_time:.2f}[/bold]",
        "[bold]100.0%[/bold]",
        style="bold",
    )

    console.print(table)

    if memory_mb is not None:
        console.print(f"\n[cyan]Peak Memory Usage:[/cyan] {memory_mb:.1f} MB")


def validate_all_bboxes(result: PipelineResult) -> Dict[str, Any]:
    """Validate that all bboxes are in correct [0, 1] format."""
    validation = {
        "all_valid": True,
        "evidence_valid": True,
        "key_evidence_valid": True,
        "issues": [],
    }

    # Check evidence bboxes
    for i, ev in enumerate(result.evidence):
        bbox = ev.bbox
        if not all(0 <= coord <= 1 for coord in bbox):
            validation["evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(f"Evidence {i} bbox out of range: {bbox}")

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            validation["evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(f"Evidence {i} bbox invalid ordering: {bbox}")

        if any(math.isnan(c) or math.isinf(c) for c in bbox):
            validation["evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(f"Evidence {i} bbox has NaN/Inf: {bbox}")

    # Check key evidence bboxes
    for i, ev in enumerate(result.key_evidence):
        bbox = ev.bbox
        if not all(0 <= coord <= 1 for coord in bbox):
            validation["key_evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(f"Key evidence {i} bbox out of range: {bbox}")

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            validation["key_evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(
                f"Key evidence {i} bbox invalid ordering: {bbox}"
            )

        if any(math.isnan(c) or math.isinf(c) for c in bbox):
            validation["key_evidence_valid"] = False
            validation["all_valid"] = False
            validation["issues"].append(f"Key evidence {i} bbox has NaN/Inf: {bbox}")

    return validation


def visualize_bboxes(
    image: Image.Image, evidences: List[GroundedEvidence], output_path: Path
):
    """Draw bboxes on image and save."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Color palette for different steps
    colors = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#FFA07A",  # Light Salmon
        "#98D8C8",  # Mint
        "#F7DC6F",  # Yellow
    ]

    width, height = image.size

    for ev in evidences:
        # Convert normalized to pixel coordinates
        x1 = int(ev.bbox[0] * width)
        y1 = int(ev.bbox[1] * height)
        x2 = int(ev.bbox[2] * width)
        y2 = int(ev.bbox[3] * height)

        # Select color based on step index
        color = colors[(ev.step_index - 1) % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"Step {ev.step_index}"
        # Simple text without font loading
        draw.text((x1 + 5, y1 + 5), label, fill=color)

    img_copy.save(output_path)
    console.print(f"[green]✓[/green] Bbox visualization saved to: {output_path}")


def save_json_results(
    result: PipelineResult,
    config: CoRGiConfig,
    timings: Dict[str, float],
    validation: Dict[str, Any],
    output_path: Path,
    question: str,
):
    """Save results to JSON file."""
    # Manually serialize PipelineResult
    result_dict = {
        "question": result.question,
        "answer": result.answer,
        "steps": [
            {
                "index": s.index,
                "statement": s.statement,
                "needs_vision": s.needs_vision,
                "reason": s.reason,
            }
            for s in result.steps
        ],
        "evidence": [
            {
                "step_index": e.step_index,
                "bbox": list(e.bbox),
                "description": e.description,
                "confidence": e.confidence,
            }
            for e in result.evidence
        ],
        "key_evidence": (
            [
                {
                    "bbox": list(e.bbox),
                    "description": e.description,
                    "reasoning": e.reasoning,
                }
                for e in result.key_evidence
            ]
            if result.key_evidence
            else []
        ),
    }

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "reasoning_model": config.reasoning.model.model_id,
            "grounding_model": config.grounding.model.model_id,
            "captioning_model": config.captioning.model.model_id,
            "synthesis_model": config.synthesis.model.model_id,
            "max_steps": config.reasoning.max_steps,
            "max_regions": config.grounding.max_regions,
        },
        "question": question,
        "results": result_dict,
        "timings": timings,
        "coordinate_validation": validation,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"[green]✓[/green] JSON results saved to: {output_path}")


def save_markdown_report(
    result: PipelineResult,
    config: CoRGiConfig,
    timings: Dict[str, float],
    validation: Dict[str, Any],
    output_path: Path,
    question: str,
):
    """Save results to Markdown report."""
    lines = [
        "# CoRGi Pipeline Test Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Configuration",
        f"\n- **Reasoning Model:** {config.reasoning.model.model_id}",
        f"- **Grounding Model:** {config.grounding.model.model_id}",
        f"- **Captioning Model:** {config.captioning.model.model_id}",
        f"- **Synthesis Model:** {config.synthesis.model.model_id}",
        f"- **Max Steps:** {config.reasoning.max_steps}",
        f"- **Max Regions:** {config.grounding.max_regions}",
        "\n## Question",
        f"\n{question}",
        "\n## Answer",
        f"\n{result.answer}",
        "\n## Reasoning Steps",
        "\n| Index | Statement | Needs Vision | Reason |",
        "|-------|-----------|--------------|--------|",
    ]

    for step in result.steps:
        needs = "Yes" if step.needs_vision else "No"
        reason = step.reason or ""
        lines.append(f"| {step.index} | {step.statement} | {needs} | {reason} |")

    lines.extend(
        [
            "\n## Visual Evidence",
            "\n| Step | BBox [0,1] | Description | Confidence |",
            "|------|------------|-------------|------------|",
        ]
    )

    for ev in result.evidence:
        bbox_str = (
            f"[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]"
        )
        desc = ev.description or ""
        conf = f"{ev.confidence:.2f}" if ev.confidence is not None else "N/A"
        lines.append(f"| {ev.step_index} | {bbox_str} | {desc} | {conf} |")

    if result.key_evidence:
        lines.extend(
            [
                "\n## Key Evidence",
                "\n| BBox [0,1] | Description | Reasoning |",
                "|------------|-------------|-----------|",
            ]
        )

        for ev in result.key_evidence:
            bbox_str = f"[{ev.bbox[0]:.3f}, {ev.bbox[1]:.3f}, {ev.bbox[2]:.3f}, {ev.bbox[3]:.3f}]"
            lines.append(f"| {bbox_str} | {ev.description} | {ev.reasoning} |")

    lines.extend(
        [
            "\n## Performance Metrics",
            "\n| Stage | Duration (s) | Percentage |",
            "|-------|--------------|------------|",
        ]
    )

    total_time = sum(timings.values())
    for stage, duration in timings.items():
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        lines.append(f"| {stage} | {duration:.2f} | {percentage:.1f}% |")

    lines.append(f"| **TOTAL** | **{total_time:.2f}** | **100.0%** |")

    lines.extend(
        [
            "\n## Coordinate Validation",
            f"\n- **All Valid:** {validation['all_valid']}",
            f"- **Evidence Valid:** {validation['evidence_valid']}",
            f"- **Key Evidence Valid:** {validation['key_evidence_valid']}",
        ]
    )

    if validation["issues"]:
        lines.append("\n### Issues Found:\n")
        for issue in validation["issues"]:
            lines.append(f"- {issue}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[green]✓[/green] Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test CoRGi pipeline with real Qwen3-VL-2B-Instruct + Florence-2 models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--image", type=str, help="Path to test image (default: fetch demo image)"
    )
    parser.add_argument(
        "--question", type=str, help="Question to ask (default: demo question)"
    )
    parser.add_argument(
        "--save-viz", action="store_true", help="Save bbox visualization image"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable rich progress display"
    )
    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        default=True,
        help="Enable comprehensive tracing (default: True)",
    )
    parser.add_argument(
        "--disable-tracing", action="store_true", help="Disable comprehensive tracing"
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Directory for trace output (default: same as output-dir)",
    )
    parser.add_argument(
        "--parallel-loading",
        action="store_true",
        default=True,
        help="Enable parallel model loading (default: True)",
    )
    parser.add_argument(
        "--no-parallel-loading",
        action="store_true",
        help="Disable parallel model loading",
    )
    parser.add_argument(
        "--batch-captioning",
        action="store_true",
        default=True,
        help="Enable batch captioning (default: True)",
    )
    parser.add_argument(
        "--no-batch-captioning", action="store_true", help="Disable batch captioning"
    )

    args = parser.parse_args()

    # Handle tracing flag
    enable_tracing = args.enable_tracing and not args.disable_tracing

    # Handle optimization flags
    parallel_loading = args.parallel_loading and not args.no_parallel_loading
    batch_captioning = args.batch_captioning and not args.no_batch_captioning

    if parallel_loading:
        console.print("[green]✓[/green] Parallel model loading enabled")
    else:
        console.print("[yellow]⚠[/yellow] Parallel model loading disabled")

    if batch_captioning:
        console.print("[green]✓[/green] Batch captioning enabled")
    else:
        console.print("[yellow]⚠[/yellow] Batch captioning disabled")

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Initialize loggers if tracing enabled
    trace_dir = args.trace_dir or args.output_dir
    trace_dir.mkdir(exist_ok=True)

    image_logger = None
    output_tracer = None

    if enable_tracing:
        console.print(f"[cyan]Tracing enabled. Trace directory: {trace_dir}[/cyan]")
        image_logger = ImageLogger(trace_dir, enabled=True)
        output_tracer = OutputTracer(trace_dir, enabled=True)
    else:
        console.print("[dim]Tracing disabled[/dim]")

    # Display header
    header = Panel(
        "[bold white]CoRGi Pipeline Test - Real Models[/bold white]\n"
        "[cyan]Qwen3-VL-2B-Instruct + Florence-2[/cyan]\n"
        f"[dim]Parallel Loading: {'ON' if parallel_loading else 'OFF'} | "
        f"Batch Captioning: {'ON' if batch_captioning else 'OFF'}[/dim]",
        border_style="bold blue",
        padding=(1, 2),
    )
    console.print(header)

    # Track memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    timings = {}

    # Stage 1: Load configuration
    console.print("\n[bold cyan][1/5] Loading Configuration...[/bold cyan]")
    start = time.time()

    if not args.config.exists():
        console.print(f"[red]Error: Config file not found: {args.config}[/red]")
        return 1

    config = CoRGiConfig.from_yaml(args.config)
    timings["Config Loading"] = time.time() - start
    console.print(
        f"[green]✓[/green] Config loaded from: {args.config} ({timings['Config Loading']:.2f}s)"
    )

    # Stage 2: Load models
    console.print("\n[bold cyan][2/5] Loading Models...[/bold cyan]")
    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        disable=args.no_progress,
    ) as progress:
        task = progress.add_task("Loading VLM models...", total=None)

        try:
            client = VLMClientFactory.create_from_config(
                config,
                image_logger=image_logger,
                output_tracer=output_tracer,
                parallel_loading=parallel_loading,
            )
            pipeline = CoRGIPipeline(
                vlm_client=client,
                image_logger=image_logger,
                output_tracer=output_tracer,
            )
        except Exception as e:
            console.print(f"[red]Error loading models: {e}[/red]")
            raise

    timings["Model Loading"] = time.time() - start

    console.print(f"[green]✓[/green] Reasoning: {config.reasoning.model.model_id}")
    console.print(f"[green]✓[/green] Grounding: {config.grounding.model.model_id}")
    console.print(f"[green]✓[/green] Captioning: {config.captioning.model.model_id}")
    console.print(f"[green]✓[/green] Synthesis: {config.synthesis.model.model_id}")
    console.print(f"[dim]Total loading time: {timings['Model Loading']:.2f}s[/dim]")

    # Stage 3: Load image and question
    console.print("\n[bold cyan][3/5] Preparing Input...[/bold cyan]")
    start = time.time()

    if args.image:
        image = load_custom_image(args.image)
    else:
        image = fetch_demo_image()

    question = args.question or DEMO_QUESTION
    console.print(f"[cyan]Question:[/cyan] {question}")
    console.print(f"[cyan]Image size:[/cyan] {image.size}")

    timings["Image Loading"] = time.time() - start

    # Stage 4: Run pipeline
    console.print("\n[bold cyan][4/5] Running Pipeline...[/bold cyan]")
    start = time.time()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            disable=args.no_progress,
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            result = pipeline.run(
                image=image,
                question=question,
                max_steps=config.reasoning.max_steps,
                max_regions=config.grounding.max_regions,
            )
    except Exception as e:
        console.print(f"[red]Error running pipeline: {e}[/red]")
        raise

    timings["Pipeline Execution"] = time.time() - start
    console.print(
        f"[green]✓[/green] Pipeline completed in {timings['Pipeline Execution']:.2f}s"
    )

    # Get peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = peak_memory - initial_memory

    # Display results
    display_reasoning_results(result.steps, cot_text=result.cot_text)
    display_grounding_results(result.evidence)
    display_synthesis_results(result.answer, result.key_evidence)
    validate_and_log_coordinates(result, image)
    display_performance_summary(timings, memory_used)

    # Stage 5: Validation and saving
    console.print("\n[bold cyan][5/5] Validation and Saving Results...[/bold cyan]")
    validation = validate_all_bboxes(result)

    if validation["all_valid"]:
        console.print("[green]✓[/green] All bboxes validated successfully!")
    else:
        console.print("[yellow]⚠[/yellow] Bbox validation issues found:")
        for issue in validation["issues"]:
            console.print(f"  [yellow]-[/yellow] {issue}")

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_path = args.output_dir / f"pipeline_results_{timestamp}.json"
    save_json_results(result, config, timings, validation, json_path, question)

    # Save markdown report
    md_path = args.output_dir / f"pipeline_report_{timestamp}.md"
    save_markdown_report(result, config, timings, validation, md_path, question)

    # Save bbox visualization if requested
    if args.save_viz and result.evidence:
        viz_path = args.output_dir / f"annotated_{timestamp}.png"
        visualize_bboxes(image, result.evidence, viz_path)

    # Generate trace report if tracing enabled
    if enable_tracing and image_logger and output_tracer:
        console.print("\n[bold cyan]Generating Trace Report...[/bold cyan]")
        try:
            # Create pipeline trace
            pipeline_trace = PipelineTrace(
                pipeline_id=f"pipeline_{timestamp}",
                question=question,
                config={
                    "reasoning": {
                        "model_id": config.reasoning.model.model_id,
                        "model_type": config.reasoning.model.model_type,
                        "max_steps": config.reasoning.max_steps,
                    },
                    "grounding": {
                        "model_id": config.grounding.model.model_id,
                        "model_type": config.grounding.model.model_type,
                        "max_regions": config.grounding.max_regions,
                    },
                    "captioning": {
                        "model_id": config.captioning.model.model_id,
                        "model_type": config.captioning.model.model_type,
                    },
                    "synthesis": {
                        "model_id": config.synthesis.model.model_id,
                        "model_type": config.synthesis.model.model_type,
                    },
                },
                start_timestamp=datetime.utcnow().isoformat() + "Z",
                end_timestamp=datetime.utcnow().isoformat() + "Z",
                total_duration_ms=result.total_duration_ms,
                original_image_path=str(
                    trace_dir / "images" / "original" / "input_image.png"
                ),
                final_result={
                    "answer": result.answer,
                    "steps_count": len(result.steps),
                    "evidence_count": len(result.evidence),
                    "key_evidence_count": len(result.key_evidence),
                },
            )

            # Save image metadata
            image_logger.save_metadata_summary()

            # Save trace summary
            output_tracer.save_summary()

            # Generate HTML report
            report_generator = HTMLReportGenerator(trace_dir)
            html_path = report_generator.generate_report(
                pipeline_trace,
                image_logger,
                output_tracer,
            )

            # Save pipeline trace JSON
            trace_json_path = trace_dir / f"pipeline_trace_{timestamp}.json"
            pipeline_trace.save_json(trace_json_path)

            console.print(
                f"[green]✓[/green] Trace report generated: [cyan]{html_path}[/cyan]"
            )
            console.print(
                f"[green]✓[/green] Pipeline trace saved: [cyan]{trace_json_path}[/cyan]"
            )

        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to generate trace report: {e}")
            import traceback

            console.print(traceback.format_exc())

    # Final summary
    console.print("\n[bold green]✓ Test completed successfully![/bold green]")
    console.print(f"\nResults saved to: [cyan]{args.output_dir}[/cyan]")
    if enable_tracing:
        console.print(f"Trace data saved to: [cyan]{trace_dir}[/cyan]")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        exit(1)
