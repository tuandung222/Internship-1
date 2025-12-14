#!/usr/bin/env python
"""
Random Sample Selection Script

Randomly selects 4 samples from the dataset for testing and saves the selection.

Usage:
    python test_select_samples.py --dataset "5CD-AI/Viet-ShareGPT-4o-Text-VQA"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Try to import datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    exit(1)

# Import extraction function from inference script
from inference_to_markdown import extract_question_from_conversation, process_dataset_sample

logger = logging.getLogger(__name__)
console = Console()


def select_random_samples(dataset: Any, num_samples: int = 4, seed: int = 42) -> list[int]:
    """
    Select random sample indices from dataset.
    
    Args:
        dataset: Hugging Face dataset object
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of selected indices
    """
    random.seed(seed)
    total = len(dataset)
    if num_samples > total:
        console.print(f"[yellow]Warning: Requested {num_samples} samples but dataset only has {total}[/yellow]")
        num_samples = total
    
    selected = sorted(random.sample(range(total), num_samples))
    return selected


def preview_samples(dataset: Any, indices: list[int]) -> list[dict]:
    """
    Preview selected samples.
    
    Args:
        dataset: Hugging Face dataset object
        indices: List of sample indices
        
    Returns:
        List of preview dictionaries
    """
    previews = []
    
    for idx in indices:
        try:
            sample = dataset[idx]
            preview = {
                "index": idx,
                "sample_id": f"sample_{idx}",
            }
            
            # Try to extract question
            question = None
            if "conversation" in sample:
                question = extract_question_from_conversation(sample["conversation"])
            elif "question" in sample:
                question = sample["question"]
            elif "text" in sample:
                question = sample["text"]
            
            preview["question"] = question[:200] if question else "N/A"
            preview["question_length"] = len(question) if question else 0
            
            # Check image
            if "image" in sample:
                img = sample["image"]
                if hasattr(img, 'size'):
                    preview["image_size"] = img.size
                    preview["image_mode"] = img.mode if hasattr(img, 'mode') else "N/A"
                else:
                    preview["image_info"] = str(type(img).__name__)
            
            # Check conversation structure
            if "conversation" in sample:
                conv = sample["conversation"]
                if isinstance(conv, list):
                    preview["conversation_length"] = len(conv)
                    if len(conv) > 0:
                        first_turn = conv[0]
                        if isinstance(first_turn, dict):
                            preview["first_turn_from"] = first_turn.get("from", "N/A")
            
            # Try to process
            try:
                image, question, sample_id = process_dataset_sample(sample, idx)
                preview["processing_success"] = True
                preview["processed_question"] = question[:200]
            except Exception as e:
                preview["processing_success"] = False
                preview["processing_error"] = str(e)
            
            previews.append(preview)
            
        except Exception as e:
            previews.append({
                "index": idx,
                "error": str(e),
                "processing_success": False,
            })
    
    return previews


def display_selection(previews: list[dict], indices: list[int]) -> None:
    """Display selected samples in a formatted table."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Selected Samples for Testing[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=8)
    table.add_column("Question Preview", style="white", width=60)
    table.add_column("Image Size", style="yellow", width=15)
    table.add_column("Status", style="green", width=12)
    
    for preview in previews:
        idx = preview["index"]
        question = preview.get("question", "N/A")[:60] + "..." if len(preview.get("question", "")) > 60 else preview.get("question", "N/A")
        
        image_size = "N/A"
        if "image_size" in preview:
            image_size = f"{preview['image_size'][0]}x{preview['image_size'][1]}"
        elif "image_info" in preview:
            image_size = preview["image_info"]
        
        status = "[green]✓ Ready[/green]" if preview.get("processing_success", False) else "[red]✗ Error[/red]"
        
        table.add_row(
            str(idx),
            question,
            image_size,
            status
        )
    
    console.print(table)
    
    # Show detailed preview for each sample
    console.print("\n[bold cyan]Detailed Preview:[/bold cyan]\n")
    for preview in previews:
        panel_content = f"[bold]Index:[/bold] {preview['index']}\n"
        panel_content += f"[bold]Sample ID:[/bold] {preview.get('sample_id', 'N/A')}\n"
        
        if preview.get("processing_success"):
            panel_content += f"[bold]Question:[/bold] {preview.get('processed_question', 'N/A')}\n"
            if "image_size" in preview:
                panel_content += f"[bold]Image:[/bold] {preview['image_size']} ({preview.get('image_mode', 'N/A')})\n"
            panel_content += f"[green]✓ Processing successful[/green]"
        else:
            panel_content += f"[red]✗ Processing failed: {preview.get('processing_error', 'Unknown error')}[/red]"
        
        panel = Panel(
            panel_content,
            title=f"[bold blue]Sample {preview['index']}[/bold blue]",
            border_style="blue"
        )
        console.print(panel)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Randomly select 4 samples from dataset for testing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="5CD-AI/Viet-ShareGPT-4o-Text-VQA",
        help="Hugging Face dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (default: None, will concatenate all splits)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to select (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_samples.json"),
        help="Output file for selected indices (default: test_samples.json)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    console.print(f"[bold blue]Random Sample Selection Tool[/bold blue]")
    console.print(f"[cyan]Dataset:[/cyan] {args.dataset}")
    console.print(f"[cyan]Split:[/cyan] {args.split}")
    console.print(f"[cyan]Number of samples:[/cyan] {args.num_samples}")
    console.print(f"[cyan]Random seed:[/cyan] {args.seed}\n")
    
    try:
        # Load dataset
        console.print(f"[cyan]Loading dataset...[/cyan]")
        if args.split:
            dataset = load_dataset(args.dataset, split=args.split)
        else:
            # Load all splits and concatenate
            from datasets import concatenate_datasets
            all_splits = load_dataset(args.dataset)
            splits_list = [all_splits[split_name] for split_name in all_splits.keys()]
            dataset = concatenate_datasets(splits_list)
            console.print(f"[green]✓[/green] Concatenated {len(splits_list)} splits: {list(all_splits.keys())}")
        console.print(f"[green]✓[/green] Loaded {len(dataset)} samples\n")
        
        # Select random samples
        console.print(f"[cyan]Selecting {args.num_samples} random samples...[/cyan]")
        selected_indices = select_random_samples(dataset, args.num_samples, args.seed)
        console.print(f"[green]✓[/green] Selected indices: {selected_indices}\n")
        
        # Preview samples
        previews = preview_samples(dataset, selected_indices)
        display_selection(previews, selected_indices)
        
        # Save selection
        selection_data = {
            "dataset": args.dataset,
            "split": args.split,
            "selected_indices": selected_indices,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
            "previews": previews,
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(selection_data, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"\n[green]✓[/green] Selection saved to: {args.output}")
        
        # Check if all samples are processable
        all_success = all(p.get("processing_success", False) for p in previews)
        if all_success:
            console.print("\n[bold green]✓ All selected samples are ready for testing![/bold green]")
            return 0
        else:
            console.print("\n[yellow]⚠ Some selected samples have issues[/yellow]")
            console.print("Consider re-running with a different seed.")
            return 1
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())


