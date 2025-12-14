#!/usr/bin/env python
"""
Quality and Performance Check Script

Checks report quality and measures performance metrics.

Usage:
    python test_check_quality.py --output-dir test_output_4samples
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


def check_report_quality(sample_dir: Path, sample_id: str) -> Dict[str, any]:
    """
    Check quality of generated reports.
    
    Args:
        sample_dir: Sample directory path
        sample_id: Sample identifier
        
    Returns:
        Dictionary with quality metrics
    """
    result = {
        "sample_id": sample_id,
        "html_size_mb": 0.0,
        "markdown_size_mb": 0.0,
        "html_readable": False,
        "markdown_readable": False,
        "base64_images_valid": 0,
        "base64_images_invalid": 0,
        "errors": [],
    }
    
    # Check HTML report
    html_reports = list(sample_dir.glob("trace_report_*.html"))
    if html_reports:
        html_path = html_reports[0]
        result["html_size_mb"] = html_path.stat().st_size / (1024 * 1024)
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result["html_readable"] = True
            
            # Validate base64 images
            import base64
            import re
            base64_pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
            matches = re.findall(base64_pattern, content)
            
            for b64_str in matches:
                try:
                    decoded = base64.b64decode(b64_str)
                    if len(decoded) > 0:
                        result["base64_images_valid"] += 1
                    else:
                        result["base64_images_invalid"] += 1
                except:
                    result["base64_images_invalid"] += 1
        except Exception as e:
            result["errors"].append(f"HTML read error: {e}")
    
    # Check Markdown report
    markdown_reports = list(sample_dir.glob("trace_report_*.md"))
    if markdown_reports:
        md_path = markdown_reports[0]
        result["markdown_size_mb"] = md_path.stat().st_size / (1024 * 1024)
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result["markdown_readable"] = True
        except Exception as e:
            result["errors"].append(f"Markdown read error: {e}")
    
    return result


def measure_performance(output_dir: Path, samples_file: Optional[Path] = None) -> Dict[str, any]:
    """
    Measure performance metrics from checkpoint and logs.
    
    Args:
        output_dir: Output directory
        samples_file: Optional path to samples file for reference
        
    Returns:
        Dictionary with performance metrics
    """
    result = {
        "total_samples": 0,
        "processed_samples": 0,
        "failed_samples": 0,
        "average_time_per_sample": 0.0,
        "total_time": 0.0,
        "errors": [],
    }
    
    # Load checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            result["processed_samples"] = len(checkpoint.get("processed_samples", []))
            result["failed_samples"] = len(checkpoint.get("errors", []))
            result["total_samples"] = result["processed_samples"] + result["failed_samples"]
        except Exception as e:
            result["errors"].append(f"Checkpoint read error: {e}")
    
    # Try to get timing from individual sample traces
    samples_dir = output_dir / "samples"
    if samples_dir.exists():
        timing_data = []
        for sample_dir in samples_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            # Try to load trace data
            metadata_dir = sample_dir / "metadata"
            if metadata_dir.exists():
                trace_file = metadata_dir / "pipeline_metadata.json"
                if trace_file.exists():
                    try:
                        with open(trace_file, 'r', encoding='utf-8') as f:
                            trace_data = json.load(f)
                        
                        # Extract timing if available
                        entries = trace_data.get("entries", [])
                        for entry in entries:
                            if "duration_ms" in entry:
                                timing_data.append(entry["duration_ms"])
                    except:
                        pass
        
        if timing_data:
            result["average_time_per_sample"] = sum(timing_data) / len(timing_data) / 1000.0  # Convert to seconds
            result["total_time"] = sum(timing_data) / 1000.0
    
    return result


def display_quality_results(quality_results: List[Dict[str, any]], perf_results: Dict[str, any]) -> None:
    """Display quality and performance results."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Quality and Performance Check[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Quality table
    quality_table = Table(title="Report Quality", show_header=True, header_style="bold magenta")
    quality_table.add_column("Sample", style="cyan", width=15)
    quality_table.add_column("HTML Size (MB)", style="yellow", width=15, justify="right")
    quality_table.add_column("MD Size (MB)", style="yellow", width=15, justify="right")
    quality_table.add_column("HTML Readable", style="green", width=12)
    quality_table.add_column("MD Readable", style="green", width=12)
    quality_table.add_column("Valid Images", style="green", width=12)
    
    for qr in quality_results:
        quality_table.add_row(
            qr["sample_id"],
            f"{qr['html_size_mb']:.2f}",
            f"{qr['markdown_size_mb']:.2f}",
            "[green]✓[/green]" if qr["html_readable"] else "[red]✗[/red]",
            "[green]✓[/green]" if qr["markdown_readable"] else "[red]✗[/red]",
            f"{qr['base64_images_valid']}",
        )
    
    console.print(quality_table)
    
    # Performance summary
    perf_panel = Panel(
        f"[bold]Total Samples:[/bold] {perf_results['total_samples']}\n"
        f"[bold]Processed:[/bold] {perf_results['processed_samples']}\n"
        f"[bold]Failed:[/bold] {perf_results['failed_samples']}\n"
        f"[bold]Average Time/Sample:[/bold] {perf_results['average_time_per_sample']:.2f}s\n"
        f"[bold]Total Time:[/bold] {perf_results['total_time']:.2f}s",
        title="[bold green]Performance Summary[/bold green]",
        border_style="green"
    )
    console.print(perf_panel)
    
    # Quality summary
    total_html_size = sum(qr["html_size_mb"] for qr in quality_results)
    total_md_size = sum(qr["markdown_size_mb"] for qr in quality_results)
    total_valid_images = sum(qr["base64_images_valid"] for qr in quality_results)
    all_html_readable = all(qr["html_readable"] for qr in quality_results)
    all_md_readable = all(qr["markdown_readable"] for qr in quality_results)
    
    quality_summary = Panel(
        f"[bold]Total HTML Size:[/bold] {total_html_size:.2f} MB\n"
        f"[bold]Total Markdown Size:[/bold] {total_md_size:.2f} MB\n"
        f"[bold]Total Valid Images:[/bold] {total_valid_images}\n"
        f"[bold]All HTML Readable:[/bold] {'✓' if all_html_readable else '✗'}\n"
        f"[bold]All Markdown Readable:[/bold] {'✓' if all_md_readable else '✗'}",
        title="[bold green]Quality Summary[/bold green]",
        border_style="green"
    )
    console.print(quality_summary)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check report quality and measure performance"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output_4samples"),
        help="Output directory containing samples (default: test_output_4samples)"
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=Path("test_samples.json"),
        help="Path to test_samples.json (default: test_samples.json)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    console.print(f"[bold blue]Quality and Performance Check[/bold blue]")
    console.print(f"[cyan]Output Directory:[/cyan] {args.output_dir}\n")
    
    # Find sample directories
    samples_dir = args.output_dir / "samples"
    if not samples_dir.exists():
        console.print(f"[red]Error: Samples directory not found: {samples_dir}[/red]")
        return 1
    
    sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        console.print("[red]Error: No sample directories found[/red]")
        return 1
    
    console.print(f"[cyan]Checking {len(sample_dirs)} samples...[/cyan]\n")
    
    # Check quality for each sample
    quality_results = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        quality = check_report_quality(sample_dir, sample_id)
        quality_results.append(quality)
    
    # Measure performance
    perf_results = measure_performance(args.output_dir, args.samples_file if args.samples_file.exists() else None)
    
    # Display results
    display_quality_results(quality_results, perf_results)
    
    # Final assessment
    all_good = (
        all(qr["html_readable"] for qr in quality_results) and
        all(qr["markdown_readable"] for qr in quality_results) and
        all(qr["base64_images_valid"] > 0 for qr in quality_results)
    )
    
    if all_good:
        console.print("\n[bold green]✓ All quality checks passed![/bold green]")
        return 0
    else:
        console.print("\n[yellow]⚠ Some quality issues found[/yellow]")
        return 1


if __name__ == "__main__":
    exit(main())


