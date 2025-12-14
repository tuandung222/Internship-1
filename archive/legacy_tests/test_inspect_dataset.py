#!/usr/bin/env python
"""
Dataset Inspection Script

Inspects the structure of the VQA dataset to understand format and validate compatibility
with the inference script.

Usage:
    python test_inspect_dataset.py --dataset "5CD-AI/Viet-ShareGPT-4o-Text-VQA"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from PIL import Image

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


def inspect_dataset_structure(dataset: Any, num_samples: int = 10) -> Dict[str, Any]:
    """
    Inspect dataset structure and validate format.
    
    Args:
        dataset: Hugging Face dataset object
        num_samples: Number of samples to inspect
        
    Returns:
        Dictionary with inspection results
    """
    inspection = {
        "total_samples": len(dataset),
        "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
        "samples": [],
        "structure_analysis": {},
        "validation_results": {},
    }
    
    console.print(f"[cyan]Inspecting first {num_samples} samples...[/cyan]")
    
    # Inspect samples
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        sample_info = {
            "index": idx,
            "fields": list(sample.keys()),
            "has_image": "image" in sample,
            "has_conversation": "conversation" in sample,
            "has_question": "question" in sample,
            "has_text": "text" in sample,
            "has_id": "id" in sample,
        }
        
        # Check image format
        if "image" in sample:
            img = sample["image"]
            sample_info["image_type"] = type(img).__name__
            if isinstance(img, Image.Image):
                sample_info["image_size"] = img.size
                sample_info["image_mode"] = img.mode
            else:
                sample_info["image_info"] = str(img)[:100]
        
        # Check conversation structure
        if "conversation" in sample:
            conv = sample["conversation"]
            sample_info["conversation_type"] = type(conv).__name__
            if isinstance(conv, list):
                sample_info["conversation_length"] = len(conv)
                if len(conv) > 0:
                    first_turn = conv[0]
                    sample_info["first_turn_keys"] = list(first_turn.keys()) if isinstance(first_turn, dict) else []
                    sample_info["first_turn_from"] = first_turn.get("from") if isinstance(first_turn, dict) else None
                    sample_info["first_turn_value_preview"] = (
                        str(first_turn.get("value", ""))[:100] 
                        if isinstance(first_turn, dict) and "value" in first_turn 
                        else None
                    )
                    
                    # Test question extraction
                    try:
                        extracted_question = extract_question_from_conversation(conv)
                        sample_info["extracted_question"] = extracted_question[:100] if extracted_question else None
                        sample_info["question_extraction_success"] = extracted_question is not None
                    except Exception as e:
                        sample_info["question_extraction_error"] = str(e)
                        sample_info["question_extraction_success"] = False
            else:
                sample_info["conversation_info"] = str(conv)[:100]
        
        # Check alternative question fields
        if "question" in sample:
            sample_info["question_preview"] = str(sample["question"])[:100]
        if "text" in sample:
            sample_info["text_preview"] = str(sample["text"])[:100]
        if "id" in sample:
            sample_info["id_value"] = sample["id"]
        
        # Try to process sample
        try:
            image, question, sample_id = process_dataset_sample(sample, idx)
            sample_info["processing_success"] = True
            sample_info["processed_question"] = question[:100]
            sample_info["processed_sample_id"] = sample_id
            sample_info["processed_image_size"] = image.size
        except Exception as e:
            sample_info["processing_success"] = False
            sample_info["processing_error"] = str(e)
        
        inspection["samples"].append(sample_info)
    
    # Structure analysis
    all_fields = set()
    for sample_info in inspection["samples"]:
        all_fields.update(sample_info.get("fields", []))
    inspection["structure_analysis"]["all_fields"] = list(all_fields)
    inspection["structure_analysis"]["common_fields"] = [
        field for field in all_fields
        if all(field in s.get("fields", []) for s in inspection["samples"])
    ]
    
    # Validation results
    inspection["validation_results"] = {
        "all_have_image": all(s.get("has_image", False) for s in inspection["samples"]),
        "all_have_conversation": all(s.get("has_conversation", False) for s in inspection["samples"]),
        "question_extraction_success_rate": sum(
            1 for s in inspection["samples"] 
            if s.get("question_extraction_success", False)
        ) / len(inspection["samples"]) if inspection["samples"] else 0,
        "processing_success_rate": sum(
            1 for s in inspection["samples"]
            if s.get("processing_success", False)
        ) / len(inspection["samples"]) if inspection["samples"] else 0,
    }
    
    return inspection


def display_inspection_results(inspection: Dict[str, Any]) -> None:
    """Display inspection results in a formatted table."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Dataset Structure Inspection Results[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    # Summary
    summary_panel = Panel(
        f"[bold]Total Samples:[/bold] {inspection['total_samples']}\n"
        f"[bold]Features:[/bold] {', '.join(inspection['features'])}\n"
        f"[bold]Common Fields:[/bold] {', '.join(inspection['structure_analysis'].get('common_fields', []))}",
        title="[bold green]Summary[/bold green]",
        border_style="green"
    )
    console.print(summary_panel)
    
    # Validation results
    validation = inspection["validation_results"]
    validation_table = Table(title="Validation Results", show_header=True, header_style="bold magenta")
    validation_table.add_column("Check", style="cyan", width=30)
    validation_table.add_column("Status", style="yellow", width=15)
    validation_table.add_column("Details", style="white", width=40)
    
    validation_table.add_row(
        "All samples have 'image' field",
        "[green]✓[/green]" if validation["all_have_image"] else "[red]✗[/red]",
        "Required for processing"
    )
    validation_table.add_row(
        "All samples have 'conversation' field",
        "[green]✓[/green]" if validation["all_have_conversation"] else "[yellow]⚠[/yellow]",
        "May have alternative fields"
    )
    validation_table.add_row(
        "Question extraction success rate",
        f"{validation['question_extraction_success_rate']*100:.1f}%",
        f"{sum(1 for s in inspection['samples'] if s.get('question_extraction_success', False))}/{len(inspection['samples'])} samples"
    )
    validation_table.add_row(
        "Processing success rate",
        f"{validation['processing_success_rate']*100:.1f}%",
        f"{sum(1 for s in inspection['samples'] if s.get('processing_success', False))}/{len(inspection['samples'])} samples"
    )
    
    console.print(validation_table)
    
    # Sample details
    console.print("\n[bold cyan]Sample Details:[/bold cyan]")
    for sample_info in inspection["samples"][:5]:  # Show first 5
        sample_table = Table(title=f"Sample {sample_info['index']}", show_header=True, header_style="bold blue")
        sample_table.add_column("Property", style="cyan", width=25)
        sample_table.add_column("Value", style="white", width=50)
        
        if sample_info.get("has_image"):
            sample_table.add_row("Image", f"✓ {sample_info.get('image_type', 'Unknown')}")
            if "image_size" in sample_info:
                sample_table.add_row("Image Size", f"{sample_info['image_size']}")
        else:
            sample_table.add_row("Image", "[red]✗ Missing[/red]")
        
        if sample_info.get("has_conversation"):
            sample_table.add_row("Conversation", f"✓ Length: {sample_info.get('conversation_length', 0)}")
            if "extracted_question" in sample_info:
                status = "[green]✓[/green]" if sample_info.get("question_extraction_success") else "[red]✗[/red]"
                sample_table.add_row("Extracted Question", f"{status} {sample_info.get('extracted_question', 'N/A')[:80]}")
        else:
            sample_table.add_row("Conversation", "[yellow]⚠ Missing[/yellow]")
        
        if sample_info.get("processing_success"):
            sample_table.add_row("Processing", "[green]✓ Success[/green]")
            sample_table.add_row("Processed Question", sample_info.get("processed_question", "N/A")[:80])
            sample_table.add_row("Sample ID", sample_info.get("processed_sample_id", "N/A"))
        else:
            sample_table.add_row("Processing", f"[red]✗ Failed: {sample_info.get('processing_error', 'Unknown')}[/red]")
        
        console.print(sample_table)
        console.print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Inspect VQA dataset structure and validate compatibility"
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
        help="Dataset split to inspect (default: None, will concatenate all splits)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to inspect (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_inspection.json"),
        help="Output file for inspection results (default: dataset_inspection.json)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    console.print(f"[bold blue]Dataset Inspection Tool[/bold blue]")
    console.print(f"[cyan]Dataset:[/cyan] {args.dataset}")
    console.print(f"[cyan]Split:[/cyan] {args.split}")
    console.print(f"[cyan]Samples to inspect:[/cyan] {args.num_samples}\n")
    
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
        
        # Inspect structure
        inspection = inspect_dataset_structure(dataset, args.num_samples)
        
        # Display results
        display_inspection_results(inspection)
        
        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(inspection, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"\n[green]✓[/green] Inspection results saved to: {args.output}")
        
        # Final summary
        if inspection["validation_results"]["processing_success_rate"] == 1.0:
            console.print("\n[bold green]✓ All samples are compatible![/bold green]")
            return 0
        else:
            console.print("\n[yellow]⚠ Some samples may have compatibility issues[/yellow]")
            console.print("Review the inspection results for details.")
            return 1
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())


