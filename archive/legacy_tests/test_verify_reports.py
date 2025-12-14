#!/usr/bin/env python
"""
Report Verification Script

Verifies that generated HTML and Markdown reports are complete and correct.

Usage:
    python test_verify_reports.py --output-dir test_output_4samples
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


def verify_html_report(report_path: Path) -> Dict[str, any]:
    """
    Verify HTML report structure and content.
    
    Args:
        report_path: Path to HTML report file
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "exists": report_path.exists(),
        "readable": False,
        "has_base64_images": False,
        "sections_present": {},
        "image_count": 0,
        "errors": [],
    }
    
    if not result["exists"]:
        result["errors"].append(f"File does not exist: {report_path}")
        return result
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        result["readable"] = True
    except Exception as e:
        result["errors"].append(f"Error reading file: {e}")
        return result
    
    # Check for base64 images
    base64_pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
    base64_matches = re.findall(base64_pattern, content)
    result["has_base64_images"] = len(base64_matches) > 0
    result["image_count"] = len(base64_matches)
    
    # Check for required sections
    required_sections = {
        "header": r'<h1[^>]*>.*?CoRGI.*?</h1>',
        "reasoning": r'[Rr]easoning',
        "grounding": r'[Gg]rounding',
        "captioning": r'[Cc]aptioning',
        "synthesis": r'[Ss]ynthesis',
        "images": r'[Ii]mage',
    }
    
    for section_name, pattern in required_sections.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        result["sections_present"][section_name] = len(matches) > 0
    
    # Validate base64 images
    if result["has_base64_images"]:
        for i, b64_str in enumerate(base64_matches[:5]):  # Check first 5
            try:
                decoded = base64.b64decode(b64_str)
                if len(decoded) == 0:
                    result["errors"].append(f"Base64 image {i} decodes to empty data")
            except Exception as e:
                result["errors"].append(f"Base64 image {i} decode error: {e}")
    
    return result


def verify_markdown_report(report_path: Path) -> Dict[str, any]:
    """
    Verify Markdown report structure and content.
    
    Args:
        report_path: Path to Markdown report file
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "exists": report_path.exists(),
        "readable": False,
        "has_base64_images": False,
        "has_details_tags": False,
        "sections_present": {},
        "image_count": 0,
        "code_blocks": 0,
        "errors": [],
    }
    
    if not result["exists"]:
        result["errors"].append(f"File does not exist: {report_path}")
        return result
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        result["readable"] = True
    except Exception as e:
        result["errors"].append(f"Error reading file: {e}")
        return result
    
    # Check for base64 image data URIs
    base64_pattern = r'!\[.*?\]\(data:image/png;base64,([A-Za-z0-9+/=]+)\)'
    base64_matches = re.findall(base64_pattern, content)
    result["has_base64_images"] = len(base64_matches) > 0
    result["image_count"] = len(base64_matches)
    
    # Check for HTML details tags (collapsible sections)
    details_pattern = r'<details>'
    details_matches = re.findall(details_pattern, content)
    result["has_details_tags"] = len(details_matches) > 0
    
    # Check for code blocks
    code_block_pattern = r'```'
    code_blocks = re.findall(code_block_pattern, content)
    result["code_blocks"] = len(code_blocks) // 2  # Pairs of ```
    
    # Check for required sections
    required_sections = {
        "header": r'^# .*CoRGI',
        "reasoning": r'## .*[Rr]easoning',
        "grounding": r'## .*[Gg]rounding',
        "captioning": r'## .*[Cc]aptioning',
        "synthesis": r'## .*[Ss]ynthesis',
        "images": r'## .*[Ii]mage',
    }
    
    for section_name, pattern in required_sections.items():
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        result["sections_present"][section_name] = len(matches) > 0
    
    # Validate base64 images
    if result["has_base64_images"]:
        for i, b64_str in enumerate(base64_matches[:5]):  # Check first 5
            try:
                decoded = base64.b64decode(b64_str)
                if len(decoded) == 0:
                    result["errors"].append(f"Base64 image {i} decodes to empty data")
            except Exception as e:
                result["errors"].append(f"Base64 image {i} decode error: {e}")
    
    return result


def verify_image_logging(sample_dir: Path) -> Dict[str, any]:
    """
    Verify image logging structure.
    
    Args:
        sample_dir: Sample directory path
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "images_dir_exists": False,
        "original_exists": False,
        "stage_dirs": {},
        "image_count": 0,
        "metadata_files": 0,
        "errors": [],
    }
    
    images_dir = sample_dir / "images"
    result["images_dir_exists"] = images_dir.exists()
    
    if not result["images_dir_exists"]:
        result["errors"].append("images/ directory not found")
        return result
    
    # Check original image
    original_dir = images_dir / "original"
    original_image = original_dir / "input_image.png"
    result["original_exists"] = original_image.exists()
    
    # Check stage directories
    stages = ["reasoning", "grounding", "captioning", "synthesis"]
    for stage in stages:
        stage_dir = images_dir / stage
        exists = stage_dir.exists()
        result["stage_dirs"][stage] = exists
        
        if exists:
            # Count images in stage
            stage_images = list(stage_dir.rglob("*.png"))
            result["image_count"] += len(stage_images)
            
            # Count metadata files
            metadata_files = list(stage_dir.rglob("*.json"))
            result["metadata_files"] += len(metadata_files)
    
    return result


def verify_output_tracing(sample_dir: Path) -> Dict[str, any]:
    """
    Verify output tracing structure.
    
    Args:
        sample_dir: Sample directory path
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "traces_dir_exists": False,
        "trace_files": 0,
        "summary_exists": False,
        "stages_traced": [],
        "errors": [],
    }
    
    traces_dir = sample_dir / "traces"
    result["traces_dir_exists"] = traces_dir.exists()
    
    if not result["traces_dir_exists"]:
        result["errors"].append("traces/ directory not found")
        return result
    
    # Count trace files
    trace_files = list(traces_dir.rglob("*.json"))
    result["trace_files"] = len(trace_files)
    
    # Check for summary
    metadata_dir = sample_dir / "metadata"
    if metadata_dir.exists():
        summary_file = metadata_dir / "pipeline_metadata.json"
        result["summary_exists"] = summary_file.exists()
    
    # Check which stages are traced
    stages = ["reasoning", "grounding", "captioning", "synthesis"]
    for stage in stages:
        stage_traces = [f for f in trace_files if stage in f.name.lower()]
        if stage_traces:
            result["stages_traced"].append(stage)
    
    return result


def verify_sample(sample_dir: Path, sample_id: str) -> Dict[str, any]:
    """
    Verify all reports and traces for a single sample.
    
    Args:
        sample_dir: Sample directory path
        sample_id: Sample identifier
        
    Returns:
        Dictionary with comprehensive verification results
    """
    result = {
        "sample_id": sample_id,
        "sample_dir_exists": sample_dir.exists(),
        "html_report": {},
        "markdown_report": {},
        "image_logging": {},
        "output_tracing": {},
        "overall_status": "unknown",
        "errors": [],
    }
    
    if not result["sample_dir_exists"]:
        result["errors"].append(f"Sample directory not found: {sample_dir}")
        result["overall_status"] = "failed"
        return result
    
    # Find HTML report
    html_reports = list(sample_dir.glob("trace_report_*.html"))
    if html_reports:
        result["html_report"] = verify_html_report(html_reports[0])
    else:
        result["html_report"] = {"exists": False, "errors": ["No HTML report found"]}
    
    # Find Markdown report
    markdown_reports = list(sample_dir.glob("trace_report_*.md"))
    if markdown_reports:
        result["markdown_report"] = verify_markdown_report(markdown_reports[0])
    else:
        result["markdown_report"] = {"exists": False, "errors": ["No Markdown report found"]}
    
    # Verify image logging
    result["image_logging"] = verify_image_logging(sample_dir)
    
    # Verify output tracing
    result["output_tracing"] = verify_output_tracing(sample_dir)
    
    # Collect all errors
    all_errors = []
    if result["html_report"].get("errors"):
        all_errors.extend([f"HTML: {e}" for e in result["html_report"]["errors"]])
    if result["markdown_report"].get("errors"):
        all_errors.extend([f"Markdown: {e}" for e in result["markdown_report"]["errors"]])
    if result["image_logging"].get("errors"):
        all_errors.extend([f"Images: {e}" for e in result["image_logging"]["errors"]])
    if result["output_tracing"].get("errors"):
        all_errors.extend([f"Traces: {e}" for e in result["output_tracing"]["errors"]])
    
    result["errors"] = all_errors
    
    # Determine overall status
    if all_errors:
        result["overall_status"] = "failed"
    elif (result["html_report"].get("exists") and 
          result["markdown_report"].get("exists") and
          result["image_logging"].get("images_dir_exists") and
          result["output_tracing"].get("traces_dir_exists")):
        result["overall_status"] = "passed"
    else:
        result["overall_status"] = "partial"
    
    return result


def display_verification_results(results: List[Dict[str, any]]) -> None:
    """Display verification results in formatted tables."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Report Verification Results[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Summary table
    summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Sample", style="cyan", width=15)
    summary_table.add_column("Status", style="yellow", width=12)
    summary_table.add_column("HTML", style="green", width=8)
    summary_table.add_column("Markdown", style="green", width=10)
    summary_table.add_column("Images", style="green", width=8)
    summary_table.add_column("Traces", style="green", width=8)
    summary_table.add_column("Errors", style="red", width=8)
    
    for result in results:
        sample_id = result["sample_id"]
        status = result["overall_status"]
        status_display = {
            "passed": "[green]✓ Pass[/green]",
            "failed": "[red]✗ Fail[/red]",
            "partial": "[yellow]⚠ Partial[/yellow]",
            "unknown": "[dim]? Unknown[/dim]",
        }.get(status, status)
        
        html_ok = "[green]✓[/green]" if result["html_report"].get("exists") else "[red]✗[/red]"
        md_ok = "[green]✓[/green]" if result["markdown_report"].get("exists") else "[red]✗[/red]"
        img_ok = "[green]✓[/green]" if result["image_logging"].get("images_dir_exists") else "[red]✗[/red]"
        trace_ok = "[green]✓[/green]" if result["output_tracing"].get("traces_dir_exists") else "[red]✗[/red]"
        error_count = len(result.get("errors", []))
        error_display = f"[red]{error_count}[/red]" if error_count > 0 else "[green]0[/green]"
        
        summary_table.add_row(
            sample_id,
            status_display,
            html_ok,
            md_ok,
            img_ok,
            trace_ok,
            error_display,
        )
    
    console.print(summary_table)
    
    # Detailed results for each sample
    console.print("\n[bold cyan]Detailed Results:[/bold cyan]\n")
    for result in results:
        panel_content = f"[bold]Sample:[/bold] {result['sample_id']}\n"
        panel_content += f"[bold]Status:[/bold] {result['overall_status']}\n\n"
        
        # HTML report details
        html = result["html_report"]
        panel_content += f"[bold]HTML Report:[/bold]\n"
        panel_content += f"  Exists: {'✓' if html.get('exists') else '✗'}\n"
        panel_content += f"  Base64 Images: {html.get('image_count', 0)}\n"
        panel_content += f"  Sections: {sum(html.get('sections_present', {}).values())}/{len(html.get('sections_present', {}))}\n"
        
        # Markdown report details
        md = result["markdown_report"]
        panel_content += f"\n[bold]Markdown Report:[/bold]\n"
        panel_content += f"  Exists: {'✓' if md.get('exists') else '✗'}\n"
        panel_content += f"  Base64 Images: {md.get('image_count', 0)}\n"
        panel_content += f"  Details Tags: {'✓' if md.get('has_details_tags') else '✗'}\n"
        panel_content += f"  Code Blocks: {md.get('code_blocks', 0)}\n"
        
        # Image logging details
        img = result["image_logging"]
        panel_content += f"\n[bold]Image Logging:[/bold]\n"
        panel_content += f"  Images Dir: {'✓' if img.get('images_dir_exists') else '✗'}\n"
        panel_content += f"  Original Image: {'✓' if img.get('original_exists') else '✗'}\n"
        panel_content += f"  Total Images: {img.get('image_count', 0)}\n"
        panel_content += f"  Metadata Files: {img.get('metadata_files', 0)}\n"
        
        # Output tracing details
        trace = result["output_tracing"]
        panel_content += f"\n[bold]Output Tracing:[/bold]\n"
        panel_content += f"  Traces Dir: {'✓' if trace.get('traces_dir_exists') else '✗'}\n"
        panel_content += f"  Trace Files: {trace.get('trace_files', 0)}\n"
        panel_content += f"  Summary: {'✓' if trace.get('summary_exists') else '✗'}\n"
        panel_content += f"  Stages Traced: {', '.join(trace.get('stages_traced', []))}\n"
        
        # Errors
        if result.get("errors"):
            panel_content += f"\n[bold red]Errors:[/bold red]\n"
            for error in result["errors"][:5]:  # Show first 5
                panel_content += f"  - {error}\n"
            if len(result["errors"]) > 5:
                panel_content += f"  ... and {len(result['errors']) - 5} more\n"
        
        panel = Panel(
            panel_content,
            title=f"[bold blue]{result['sample_id']}[/bold blue]",
            border_style="blue" if result["overall_status"] == "passed" else "red"
        )
        console.print(panel)


def generate_verification_report(results: List[Dict[str, any]], output_path: Path) -> None:
    """Generate Markdown verification report."""
    lines = [
        "# Report Verification Results",
        "",
        f"**Total Samples:** {len(results)}",
        "",
        "## Summary",
        "",
        "| Sample | Status | HTML | Markdown | Images | Traces | Errors |",
        "|--------|--------|------|----------|--------|--------|--------|",
    ]
    
    for result in results:
        status = result["overall_status"]
        html_ok = "✓" if result["html_report"].get("exists") else "✗"
        md_ok = "✓" if result["markdown_report"].get("exists") else "✗"
        img_ok = "✓" if result["image_logging"].get("images_dir_exists") else "✗"
        trace_ok = "✓" if result["output_tracing"].get("traces_dir_exists") else "✗"
        error_count = len(result.get("errors", []))
        
        lines.append(
            f"| {result['sample_id']} | {status} | {html_ok} | {md_ok} | {img_ok} | {trace_ok} | {error_count} |"
        )
    
    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])
    
    for result in results:
        lines.extend([
            f"### {result['sample_id']}",
            "",
            f"**Status:** {result['overall_status']}",
            "",
            "#### HTML Report",
            f"- Exists: {'✓' if result['html_report'].get('exists') else '✗'}",
            f"- Base64 Images: {result['html_report'].get('image_count', 0)}",
            f"- Sections Present: {sum(result['html_report'].get('sections_present', {}).values())}/{len(result['html_report'].get('sections_present', {}))}",
            "",
            "#### Markdown Report",
            f"- Exists: {'✓' if result['markdown_report'].get('exists') else '✗'}",
            f"- Base64 Images: {result['markdown_report'].get('image_count', 0)}",
            f"- Details Tags: {'✓' if result['markdown_report'].get('has_details_tags') else '✗'}",
            f"- Code Blocks: {result['markdown_report'].get('code_blocks', 0)}",
            "",
            "#### Image Logging",
            f"- Images Dir: {'✓' if result['image_logging'].get('images_dir_exists') else '✗'}",
            f"- Original Image: {'✓' if result['image_logging'].get('original_exists') else '✗'}",
            f"- Total Images: {result['image_logging'].get('image_count', 0)}",
            f"- Metadata Files: {result['image_logging'].get('metadata_files', 0)}",
            "",
            "#### Output Tracing",
            f"- Traces Dir: {'✓' if result['output_tracing'].get('traces_dir_exists') else '✗'}",
            f"- Trace Files: {result['output_tracing'].get('trace_files', 0)}",
            f"- Summary: {'✓' if result['output_tracing'].get('summary_exists') else '✗'}",
            f"- Stages Traced: {', '.join(result['output_tracing'].get('stages_traced', []))}",
            "",
        ])
        
        if result.get("errors"):
            lines.extend([
                "#### Errors",
                "",
            ])
            for error in result["errors"]:
                lines.append(f"- {error}")
            lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    console.print(f"\n[green]✓[/green] Verification report saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Verify generated HTML and Markdown reports"
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
        help="Path to test_samples.json to get sample IDs (default: test_samples.json)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("test_verification_report.md"),
        help="Output path for verification report (default: test_verification_report.md)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    console.print(f"[bold blue]Report Verification Tool[/bold blue]")
    console.print(f"[cyan]Output Directory:[/cyan] {args.output_dir}\n")
    
    # Load sample IDs if available
    sample_ids = None
    if args.samples_file.exists():
        try:
            with open(args.samples_file, 'r', encoding='utf-8') as f:
                selection_data = json.load(f)
            selected_indices = selection_data.get("selected_indices", [])
            sample_ids = [f"sample_{idx}" for idx in selected_indices]
            console.print(f"[cyan]Found {len(sample_ids)} samples from {args.samples_file}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load samples file: {e}[/yellow]")
    
    # Find all sample directories
    samples_dir = args.output_dir / "samples"
    if not samples_dir.exists():
        console.print(f"[red]Error: Samples directory not found: {samples_dir}[/red]")
        return 1
    
    if sample_ids is None:
        # Discover sample directories
        sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
        sample_ids = [d.name for d in sample_dirs]
        console.print(f"[cyan]Discovered {len(sample_ids)} sample directories[/cyan]")
    
    if not sample_ids:
        console.print("[red]Error: No samples found to verify[/red]")
        return 1
    
    # Verify each sample
    console.print(f"\n[cyan]Verifying {len(sample_ids)} samples...[/cyan]\n")
    results = []
    
    for sample_id in sample_ids:
        sample_dir = samples_dir / sample_id
        result = verify_sample(sample_dir, sample_id)
        results.append(result)
    
    # Display results
    display_verification_results(results)
    
    # Generate report
    generate_verification_report(results, args.report)
    
    # Summary
    passed = sum(1 for r in results if r["overall_status"] == "passed")
    failed = sum(1 for r in results if r["overall_status"] == "failed")
    partial = sum(1 for r in results if r["overall_status"] == "partial")
    
    console.print(f"\n[bold cyan]Verification Summary:[/bold cyan]")
    console.print(f"[green]Passed:[/green] {passed}/{len(results)}")
    console.print(f"[yellow]Partial:[/yellow] {partial}/{len(results)}")
    console.print(f"[red]Failed:[/red] {failed}/{len(results)}")
    
    if failed == 0 and partial == 0:
        console.print("\n[bold green]✓ All samples verified successfully![/bold green]")
        return 0
    else:
        console.print("\n[yellow]⚠ Some samples have verification issues[/yellow]")
        return 1


if __name__ == "__main__":
    exit(main())


