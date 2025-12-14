#!/usr/bin/env python
"""
VQA Dataset Inference and Report Generation Script

Processes VQA datasets from Hugging Face, runs them through the CoRGI pipeline,
and generates comprehensive HTML and Markdown reports with embedded images.

Usage:
    python inference_to_markdown.py --dataset "5CD-AI/Viet-ShareGPT-4o-Text-VQA" --config configs/florence_qwen.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import base64
import json
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
)

from corgi.config import CoRGiConfig
from corgi.vlm_factory import VLMClientFactory
from corgi.pipeline import CoRGIPipeline, PipelineResult
from corgi.image_logger import ImageLogger
from corgi.output_tracer import OutputTracer
from corgi.report_generator import HTMLReportGenerator
from corgi.trace_types import PipelineTrace

# Try to import datasets library
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not found. Install with: pip install datasets")

logger = logging.getLogger(__name__)
console = Console()


def extract_question_from_conversation(conversation: Any) -> Optional[str]:
    """
    Extract question from first turn of conversation.

    Handles different conversation formats (ShareGPT, string JSON, list, etc.).

    Args:
        conversation: Can be a list, string JSON, or dict

    Returns:
        Question string from first turn, or None if not found
    """
    if not conversation:
        return None

    # Handle string JSON format (e.g., "[{'role': 'user', 'content': '...'}]")
    if isinstance(conversation, str):
        try:
            import json

            conversation = json.loads(conversation)
        except (json.JSONDecodeError, ValueError):
            return None

    # Handle list format
    if isinstance(conversation, list):
        if not conversation:
            return None

        first_turn = conversation[0] if conversation else None
        if not first_turn or not isinstance(first_turn, dict):
            return None

        # Format 1: ShareGPT format with 'from' and 'value'
        if "from" in first_turn and "value" in first_turn:
            if first_turn.get("from") in ["user", "human", "User", "Human"]:
                return first_turn.get("value", "").strip()

        # Format 2: Role-based format with 'role' and 'content'
        if "role" in first_turn and "content" in first_turn:
            if first_turn.get("role") in ["user", "human", "User", "Human"]:
                return first_turn.get("content", "").strip()

        # Format 3: Direct 'question' or 'text' field
        if "question" in first_turn:
            return first_turn.get("question", "").strip()
        if "text" in first_turn:
            return first_turn.get("text", "").strip()
        if "value" in first_turn:
            return first_turn.get("value", "").strip()

    return None


def process_dataset_sample(sample: Dict, index: int) -> Tuple[Image.Image, str, str]:
    """
    Process a dataset sample to extract image and question.

    Args:
        sample: Dataset sample dictionary
        index: Sample index

    Returns:
        Tuple of (image, question, sample_id)
    """
    # Extract image
    if "image" in sample:
        image = sample["image"]
        if not isinstance(image, Image.Image):
            # Try to convert if it's a different format
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            else:
                raise ValueError(f"Sample {index}: 'image' field is not a PIL Image")
    else:
        raise ValueError(f"Sample {index}: No 'image' field found")

    # Extract question from conversation
    question = None
    if "conversations" in sample:
        question = extract_question_from_conversation(sample["conversations"])
    elif "conversation" in sample:
        question = extract_question_from_conversation(sample["conversation"])
    elif "question" in sample:
        question = sample["question"]
    elif "text" in sample:
        question = sample["text"]

    if not question or not question.strip():
        raise ValueError(
            f"Sample {index}: No question found in conversation or direct fields"
        )

    sample_id = f"sample_{index}"
    if "id" in sample:
        sample_id = f"sample_{sample['id']}"

    return image, question.strip(), sample_id


def load_vqa_dataset(dataset_name: str, split: str = "train") -> Any:
    """
    Load VQA dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset name (e.g., "5CD-AI/Viet-ShareGPT-4o-Text-VQA")
        split: Dataset split to load (default: "train")

    Returns:
        Dataset object
    """
    if not HAS_DATASETS:
        raise ImportError(
            "'datasets' library is required. Install with: pip install datasets"
        )

    console.print(f"[cyan]Loading dataset: {dataset_name} (split: {split})[/cyan]")
    dataset = load_dataset(dataset_name, split=split)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} samples")
    return dataset


def run_pipeline_with_tracing(
    image: Image.Image,
    question: str,
    config_path: Path,
    output_dir: Path,
    sample_id: str,
    parallel_loading: bool = True,
) -> Tuple[PipelineResult, ImageLogger, OutputTracer]:
    """
    Run pipeline with full tracing enabled.

    Args:
        image: Input image
        question: Question to answer
        config_path: Path to config YAML file
        output_dir: Output directory for this sample
        sample_id: Sample identifier
        parallel_loading: Whether to load models in parallel

    Returns:
        Tuple of (PipelineResult, ImageLogger, OutputTracer)
    """
    # Create sample-specific output directory
    sample_output_dir = output_dir / "samples" / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loggers
    image_logger = ImageLogger(sample_output_dir, enabled=True)
    output_tracer = OutputTracer(sample_output_dir, enabled=True)

    # Load config
    config = CoRGiConfig.from_yaml(config_path)

    # Create pipeline
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

    # Run pipeline
    result = pipeline.run(
        image=image,
        question=question,
        max_steps=config.reasoning.max_steps,
        max_regions=config.grounding.max_regions,
    )

    return result, image_logger, output_tracer


def generate_sample_report(
    result: PipelineResult,
    image: Image.Image,
    sample_id: str,
    output_dir: Path,
    image_logger: ImageLogger,
    output_tracer: OutputTracer,
    config: CoRGiConfig,
) -> Tuple[Path, Path]:
    """
    Generate HTML and Markdown reports for a single sample.

    Args:
        result: PipelineResult from pipeline execution
        image: Original input image
        sample_id: Sample identifier
        output_dir: Base output directory
        image_logger: ImageLogger instance
        output_tracer: OutputTracer instance
        config: Pipeline configuration

    Returns:
        Tuple of (html_path, markdown_path)
    """
    sample_output_dir = output_dir / "samples" / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline trace
    pipeline_trace = PipelineTrace(
        pipeline_id=f"pipeline_{sample_id}",
        question=result.question,
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
            sample_output_dir / "images" / "original" / "input_image.png"
        ),
        final_result={
            "answer": result.answer,
            "steps_count": len(result.steps),
            "evidence_count": len(result.evidence) if result.evidence else 0,
            "key_evidence_count": (
                len(result.key_evidence) if result.key_evidence else 0
            ),
        },
    )

    # Save metadata
    image_logger.save_metadata_summary()
    output_tracer.save_summary()

    # Generate HTML report
    html_generator = HTMLReportGenerator(sample_output_dir)
    html_path = html_generator.generate_report(
        pipeline_trace,
        image_logger,
        output_tracer,
    )

    # Generate Markdown report
    from corgi.markdown_report_generator import MarkdownReportGenerator

    markdown_generator = MarkdownReportGenerator(sample_output_dir)
    markdown_path = markdown_generator.generate_report(
        pipeline_trace,
        image_logger,
        output_tracer,
        result=result,
    )

    return html_path, markdown_path


def load_checkpoint(output_dir: Path) -> Dict[str, Any]:
    """Load checkpoint file if it exists."""
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {
        "last_processed_index": -1,
        "processed_samples": [],
        "errors": [],
    }


def save_checkpoint(output_dir: Path, checkpoint: Dict[str, Any]) -> None:
    """Save checkpoint file atomically."""
    checkpoint_path = output_dir / "checkpoint.json"
    temp_path = checkpoint_path.with_suffix(".json.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        temp_path.replace(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def process_dataset_batch(
    dataset: Any,
    config_path: Path,
    output_dir: Path,
    start_index: int = 0,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    skip_existing: bool = True,
    parallel_loading: bool = True,
    sample_indices: Optional[List[int]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Process dataset in batches with checkpoint/resume capability.

    Args:
        dataset: Hugging Face dataset
        config_path: Path to config YAML
        output_dir: Output directory
        start_index: Starting index (for resume, ignored if sample_indices provided)
        batch_size: Batch size (currently 1, for future expansion)
        max_samples: Maximum samples to process (None = all, ignored if sample_indices provided)
        skip_existing: Skip already processed samples
        parallel_loading: Whether to load models in parallel
        sample_indices: Optional list of specific indices to process (overrides start_index/max_samples)

    Returns:
        Tuple of (processed_sample_ids, errors)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = load_checkpoint(output_dir)
    processed_samples = set(checkpoint.get("processed_samples", []))
    errors = checkpoint.get("errors", [])

    # Determine indices to process
    if sample_indices is not None:
        # Process only specified indices
        indices_to_process = sorted(sample_indices)
        samples_to_process = len(indices_to_process)
    else:
        # Process sequential range
        total_samples = len(dataset)
        end_index = (
            min(start_index + max_samples, total_samples)
            if max_samples
            else total_samples
        )
        indices_to_process = list(range(start_index, end_index))
        samples_to_process = len(indices_to_process)

    if sample_indices is not None:
        console.print(
            f"[cyan]Processing {samples_to_process} specified samples (indices: {indices_to_process})[/cyan]"
        )
    else:
        console.print(
            f"[cyan]Processing {samples_to_process} samples (indices {start_index} to {start_index + samples_to_process - 1})[/cyan]"
        )
    if processed_samples:
        console.print(
            f"[yellow]Found {len(processed_samples)} already processed samples[/yellow]"
        )

    processed_ids = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing samples...", total=samples_to_process)

        for idx in indices_to_process:
            sample_id = f"sample_{idx}"

            # Skip if already processed
            if skip_existing and sample_id in processed_samples:
                progress.update(task, advance=1)
                continue

            try:
                # Process sample
                sample = dataset[idx]
                image, question, sample_id = process_dataset_sample(sample, idx)

                console.print(
                    f"\n[cyan]Processing {sample_id}: {question[:80]}...[/cyan]"
                )

                # Run pipeline
                result, image_logger, output_tracer = run_pipeline_with_tracing(
                    image=image,
                    question=question,
                    config_path=config_path,
                    output_dir=output_dir,
                    sample_id=sample_id,
                    parallel_loading=parallel_loading,
                )

                # Load config for report generation
                config = CoRGiConfig.from_yaml(config_path)

                # Generate reports
                html_path, markdown_path = generate_sample_report(
                    result=result,
                    image=image,
                    sample_id=sample_id,
                    output_dir=output_dir,
                    image_logger=image_logger,
                    output_tracer=output_tracer,
                    config=config,
                )

                processed_ids.append(sample_id)
                processed_samples.add(sample_id)

                # Update checkpoint
                checkpoint = {
                    "last_processed_index": idx,
                    "processed_samples": list(processed_samples),
                    "errors": errors,
                }
                save_checkpoint(output_dir, checkpoint)

                console.print(
                    f"[green]✓[/green] {sample_id} completed - Reports: {html_path.name}, {markdown_path.name}"
                )
                progress.update(task, advance=1)

            except Exception as e:
                error_info = {
                    "sample_id": sample_id,
                    "index": idx,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                errors.append(error_info)
                logger.error(f"Error processing {sample_id}: {e}", exc_info=True)
                console.print(f"[red]✗[/red] {sample_id} failed: {e}")

                # Update checkpoint with error
                checkpoint = {
                    "last_processed_index": idx,
                    "processed_samples": list(processed_samples),
                    "errors": errors,
                }
                save_checkpoint(output_dir, checkpoint)
                progress.update(task, advance=1)

    return processed_ids, errors


def concatenate_reports(output_dir: Path, report_type: str = "markdown") -> Path:
    """
    Concatenate all per-sample reports into a single file.

    Args:
        output_dir: Output directory containing samples subdirectory
        report_type: Type of report to concatenate ("markdown" or "html")

    Returns:
        Path to concatenated report file
    """
    samples_dir = output_dir / "samples"
    if not samples_dir.exists():
        raise ValueError(f"Samples directory not found: {samples_dir}")

    # Collect all reports
    if report_type == "markdown":
        pattern = "*.md"
        output_file = output_dir / "full_report.md"
    else:
        pattern = "*.html"
        output_file = output_dir / "full_report.html"

    report_files = sorted(samples_dir.rglob(pattern))

    if not report_files:
        console.print(f"[yellow]No {report_type} reports found to concatenate[/yellow]")
        return output_file

    console.print(
        f"[cyan]Concatenating {len(report_files)} {report_type} reports...[/cyan]"
    )

    parts = []

    if report_type == "markdown":
        # Markdown concatenation
        parts.append("# CoRGI Pipeline - Complete Dataset Report\n")
        parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        parts.append(f"**Total Samples:** {len(report_files)}\n")
        parts.append("\n---\n")

        for report_file in report_files:
            sample_id = report_file.parent.name
            parts.append(f"\n## Sample: {sample_id}\n")
            parts.append("---\n")

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Skip the header if it's a full report
                    if content.startswith("# CoRGI Pipeline Trace Report"):
                        # Find the first section after header
                        lines = content.split("\n")
                        skip_until = 0
                        for i, line in enumerate(lines):
                            if line.startswith("## "):
                                skip_until = i
                                break
                        content = "\n".join(lines[skip_until:])
                    parts.append(content)
            except Exception as e:
                parts.append(f"\n*Error loading report for {sample_id}: {e}*\n")

            parts.append("\n---\n")
    else:
        # HTML concatenation
        parts.append(
            """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CoRGI Pipeline - Complete Dataset Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        hr { margin: 40px 0; border: 2px solid #ccc; }
        .sample-header { background: #f0f0f0; padding: 20px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>CoRGI Pipeline - Complete Dataset Report</h1>
    <p><strong>Generated:</strong> """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
    <p><strong>Total Samples:</strong> """
            + str(len(report_files))
            + """</p>
    <hr>
"""
        )

        for report_file in report_files:
            sample_id = report_file.parent.name
            parts.append(
                f'<div class="sample-header"><h2>Sample: {sample_id}</h2></div>\n'
            )
            parts.append("<hr>\n")

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract body content if it's a full HTML document
                    if "<body>" in content:
                        start = content.find("<body>") + 6
                        end = content.find("</body>")
                        if end > start:
                            content = content[start:end]
                    parts.append(content)
            except Exception as e:
                parts.append(
                    f'<p style="color: red;">Error loading report for {sample_id}: {e}</p>\n'
                )

            parts.append("<hr>\n")

        parts.append("</body>\n</html>")

    # Write concatenated report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    console.print(f"[green]✓[/green] Concatenated report saved: {output_file}")
    return output_file


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Process VQA dataset through CoRGI pipeline and generate comprehensive reports"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset name (e.g., '5CD-AI/Viet-ShareGPT-4o-Text-VQA')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (default: None, will concatenate all splits)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/florence_qwen.yaml"),
        help="Path to config YAML file (default: configs/florence_qwen.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vqa_reports"),
        help="Output directory for reports (default: vqa_reports)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this index (for resume, default: 0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1, currently not used)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already processed samples (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process all samples even if already processed",
    )
    parser.add_argument(
        "--parallel-loading",
        action="store_true",
        default=True,
        help="Load models in parallel (default: True)",
    )
    parser.add_argument(
        "--no-parallel-loading",
        action="store_true",
        help="Disable parallel model loading",
    )
    parser.add_argument(
        "--concatenate",
        action="store_true",
        default=True,
        help="Concatenate all reports into single files (default: True)",
    )
    parser.add_argument(
        "--no-concatenate", action="store_true", help="Skip report concatenation"
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated list of specific sample indices to process (e.g., '42,156,789,1234'). Overrides --start-index and --max-samples.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Validate arguments
    if not args.config.exists():
        console.print(f"[red]Error: Config file not found: {args.config}[/red]")
        return 1

    if not HAS_DATASETS:
        console.print(
            "[red]Error: 'datasets' library is required. Install with: pip install datasets[/red]"
        )
        return 1

    # Handle flags
    skip_existing = args.skip_existing and not args.no_skip_existing
    parallel_loading = args.parallel_loading and not args.no_parallel_loading
    concatenate = args.concatenate and not args.no_concatenate

    # Parse sample indices if provided
    sample_indices = None
    if args.sample_indices:
        try:
            sample_indices = [int(x.strip()) for x in args.sample_indices.split(",")]
            console.print(f"[cyan]Processing specific indices: {sample_indices}[/cyan]")
        except ValueError as e:
            console.print(f"[red]Error parsing --sample-indices: {e}[/red]")
            console.print(
                "[yellow]Expected format: --sample-indices '42,156,789,1234'[/yellow]"
            )
            return 1

    # Display header
    header = f"""
[bold blue]CoRGI VQA Dataset Inference and Report Generation[/bold blue]
[cyan]Dataset:[/cyan] {args.dataset}
[cyan]Split:[/cyan] {args.split}
[cyan]Config:[/cyan] {args.config}
[cyan]Output:[/cyan] {args.output_dir}
[cyan]Start Index:[/cyan] {args.start_index}
[cyan]Max Samples:[/cyan] {args.max_samples or 'All'}
[cyan]Skip Existing:[/cyan] {skip_existing}
[cyan]Parallel Loading:[/cyan] {parallel_loading}
"""
    console.print(header)

    try:
        # Load dataset
        dataset = load_vqa_dataset(args.dataset, args.split)

        # Ensure dataset is concatenated (not DatasetDict) when using sample_indices
        if sample_indices is not None:
            from datasets import DatasetDict

            if isinstance(dataset, DatasetDict):
                from datasets import concatenate_datasets

                splits_list = [dataset[split_name] for split_name in dataset.keys()]
                dataset = concatenate_datasets(splits_list)
                console.print(
                    f"[green]✓[/green] Concatenated {len(splits_list)} splits for sample_indices access"
                )

        # Process dataset
        processed_ids, errors = process_dataset_batch(
            dataset=dataset,
            config_path=args.config,
            output_dir=args.output_dir,
            start_index=args.start_index,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            skip_existing=skip_existing,
            parallel_loading=parallel_loading,
            sample_indices=sample_indices,
        )

        # Concatenate reports
        if concatenate:
            console.print("\n[cyan]Concatenating reports...[/cyan]")
            html_path = concatenate_reports(args.output_dir, "html")
            markdown_path = concatenate_reports(args.output_dir, "markdown")

        # Summary
        console.print("\n[bold green]✓ Processing Complete![/bold green]")
        console.print(f"[cyan]Processed:[/cyan] {len(processed_ids)} samples")
        console.print(f"[cyan]Errors:[/cyan] {len(errors)} samples")
        console.print(f"[cyan]Output Directory:[/cyan] {args.output_dir}")

        if errors:
            console.print("\n[yellow]Errors encountered:[/yellow]")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  - {error['sample_id']}: {error['error'][:100]}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
