#!/usr/bin/env python
"""
Test Inference on 4 Selected Samples

Wrapper script to run inference on 4 pre-selected samples from test_samples.json.

Usage:
    python test_inference_4_samples.py --samples-file test_samples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run inference on 4 pre-selected samples"
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=Path("test_samples.json"),
        help="Path to test_samples.json file (default: test_samples.json)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/florence_qwen.yaml"),
        help="Path to config YAML file (default: configs/florence_qwen.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output_4samples"),
        help="Output directory for reports (default: test_output_4samples)"
    )
    parser.add_argument(
        "--parallel-loading",
        action="store_true",
        default=True,
        help="Enable parallel model loading (default: True)"
    )
    parser.add_argument(
        "--no-parallel-loading",
        action="store_true",
        help="Disable parallel model loading"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # Load selected samples
    if not args.samples_file.exists():
        console.print(f"[red]Error: Samples file not found: {args.samples_file}[/red]")
        console.print("[yellow]Please run test_select_samples.py first to generate test_samples.json[/yellow]")
        return 1
    
    try:
        with open(args.samples_file, 'r', encoding='utf-8') as f:
            selection_data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading samples file: {e}[/red]")
        return 1
    
    selected_indices = selection_data.get("selected_indices", [])
    dataset_name = selection_data.get("dataset", "5CD-AI/Viet-ShareGPT-4o-Text-VQA")
    split = selection_data.get("split", "train")
    
    if not selected_indices:
        console.print("[red]Error: No selected indices found in samples file[/red]")
        return 1
    
    if len(selected_indices) != 4:
        console.print(f"[yellow]Warning: Expected 4 samples, found {len(selected_indices)}[/yellow]")
    
    # Display header
    header = Panel(
        f"[bold blue]Test Inference on 4 Selected Samples[/bold blue]\n"
        f"[cyan]Dataset:[/cyan] {dataset_name}\n"
        f"[cyan]Split:[/cyan] {split}\n"
        f"[cyan]Indices:[/cyan] {selected_indices}\n"
        f"[cyan]Config:[/cyan] {args.config}\n"
        f"[cyan]Output:[/cyan] {args.output_dir}",
        border_style="bold blue",
        padding=(1, 2)
    )
    console.print(header)
    
    # Build command
    sample_indices_str = ','.join(map(str, selected_indices))
    parallel_flag = "" if (args.parallel_loading and not args.no_parallel_loading) else "--no-parallel-loading"
    
    cmd = [
        sys.executable,
        "inference_to_markdown.py",
        "--dataset", dataset_name,
        "--split", split,
        "--config", str(args.config),
        "--output-dir", str(args.output_dir),
        "--sample-indices", sample_indices_str,
        "--skip-existing",
    ]
    
    if args.parallel_loading and not args.no_parallel_loading:
        cmd.append("--parallel-loading")
    else:
        cmd.append("--no-parallel-loading")
    
    console.print(f"\n[cyan]Executing command:[/cyan] {' '.join(cmd)}\n")
    
    # Run inference script
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error running inference: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())


