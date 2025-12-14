#!/usr/bin/env python
"""
Error Handling Test Script

Tests error handling scenarios for the inference script.

Usage:
    python test_error_handling.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from inference_to_markdown import (
    extract_question_from_conversation,
    process_dataset_sample,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)
console = Console()


def test_empty_conversation() -> Dict[str, Any]:
    """Test handling of empty conversation."""
    console.print("[cyan]Test 1: Empty conversation[/cyan]")
    
    result = {
        "test_name": "empty_conversation",
        "passed": False,
        "error": None,
    }
    
    try:
        question = extract_question_from_conversation([])
        if question is None:
            result["passed"] = True
            result["message"] = "Correctly returned None for empty conversation"
        else:
            result["error"] = f"Expected None, got: {question}"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_missing_fields() -> Dict[str, Any]:
    """Test handling of missing 'from' or 'value' fields."""
    console.print("[cyan]Test 2: Missing fields in conversation[/cyan]")
    
    result = {
        "test_name": "missing_fields",
        "passed": False,
        "error": None,
    }
    
    test_cases = [
        ([], "empty list"),
        ([{}], "empty dict"),
        ([{"from": "user"}], "missing value"),
        ([{"value": "question"}], "missing from"),
    ]
    
    passed_count = 0
    for conv, desc in test_cases:
        try:
            question = extract_question_from_conversation(conv)
            # Should handle gracefully (return None or extract what's available)
            if question is None or isinstance(question, str):
                passed_count += 1
        except Exception as e:
            result["error"] = f"Error with {desc}: {e}"
            break
    
    if passed_count == len(test_cases):
        result["passed"] = True
        result["message"] = f"Handled all {len(test_cases)} test cases gracefully"
    
    return result


def test_missing_image() -> Dict[str, Any]:
    """Test handling of missing image field."""
    console.print("[cyan]Test 3: Missing image field[/cyan]")
    
    result = {
        "test_name": "missing_image",
        "passed": False,
        "error": None,
    }
    
    try:
        sample = {
            "conversation": [{"from": "user", "value": "What is this?"}],
        }
        image, question, sample_id = process_dataset_sample(sample, 0)
        result["error"] = "Should have raised ValueError for missing image"
    except ValueError as e:
        if "image" in str(e).lower():
            result["passed"] = True
            result["message"] = f"Correctly raised ValueError: {e}"
        else:
            result["error"] = f"Wrong error message: {e}"
    except Exception as e:
        result["error"] = f"Unexpected exception: {e}"
    
    return result


def test_missing_question() -> Dict[str, Any]:
    """Test handling of missing question."""
    console.print("[cyan]Test 4: Missing question[/cyan]")
    
    result = {
        "test_name": "missing_question",
        "passed": False,
        "error": None,
    }
    
    try:
        from PIL import Image
        sample = {
            "image": Image.new('RGB', (100, 100)),
            "conversation": [],
        }
        image, question, sample_id = process_dataset_sample(sample, 0)
        result["error"] = "Should have raised ValueError for missing question"
    except ValueError as e:
        if "question" in str(e).lower() or "no question" in str(e).lower():
            result["passed"] = True
            result["message"] = f"Correctly raised ValueError: {e}"
        else:
            result["error"] = f"Wrong error message: {e}"
    except Exception as e:
        result["error"] = f"Unexpected exception: {e}"
    
    return result


def test_checkpoint_resume() -> Dict[str, Any]:
    """Test checkpoint save/load functionality."""
    console.print("[cyan]Test 5: Checkpoint save/load[/cyan]")
    
    result = {
        "test_name": "checkpoint_resume",
        "passed": False,
        "error": None,
    }
    
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save checkpoint
            checkpoint = {
                "last_processed_index": 5,
                "processed_samples": ["sample_0", "sample_1", "sample_2"],
                "errors": [],
            }
            save_checkpoint(output_dir, checkpoint)
            
            # Load checkpoint
            loaded = load_checkpoint(output_dir)
            
            if (loaded["last_processed_index"] == checkpoint["last_processed_index"] and
                set(loaded["processed_samples"]) == set(checkpoint["processed_samples"])):
                result["passed"] = True
                result["message"] = "Checkpoint save/load works correctly"
            else:
                result["error"] = f"Mismatch: expected {checkpoint}, got {loaded}"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_alternative_question_fields() -> Dict[str, Any]:
    """Test extraction from alternative question fields."""
    console.print("[cyan]Test 6: Alternative question fields[/cyan]")
    
    result = {
        "test_name": "alternative_fields",
        "passed": False,
        "error": None,
    }
    
    try:
        from PIL import Image
        
        test_cases = [
            ({"image": Image.new('RGB', (100, 100)), "question": "What is this?"}, "question field"),
            ({"image": Image.new('RGB', (100, 100)), "text": "What is this?"}, "text field"),
            ({"image": Image.new('RGB', (100, 100)), "conversation": [{"from": "user", "value": "What is this?"}]}, "conversation field"),
        ]
        
        passed_count = 0
        for sample, desc in test_cases:
            try:
                image, question, sample_id = process_dataset_sample(sample, 0)
                if question and question.strip():
                    passed_count += 1
                else:
                    result["error"] = f"Failed to extract question from {desc}"
                    break
            except Exception as e:
                result["error"] = f"Error with {desc}: {e}"
                break
        
        if passed_count == len(test_cases):
            result["passed"] = True
            result["message"] = f"Successfully extracted from all {len(test_cases)} field types"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def display_test_results(results: list[Dict[str, Any]]) -> None:
    """Display test results in a formatted table."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Error Handling Test Results[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", width=30)
    table.add_column("Status", style="yellow", width=12)
    table.add_column("Message", style="white", width=50)
    
    for result in results:
        status = "[green]✓ Pass[/green]" if result["passed"] else "[red]✗ Fail[/red]"
        message = result.get("message", result.get("error", "N/A"))
        table.add_row(result["test_name"], status, message)
    
    console.print(table)
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    console.print(f"\n[bold cyan]Summary:[/bold cyan] {passed}/{total} tests passed")
    
    if passed == total:
        console.print("[bold green]✓ All error handling tests passed![/bold green]")
    else:
        console.print("[yellow]⚠ Some tests failed[/yellow]")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test error handling scenarios"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_error_handling_results.json"),
        help="Output file for test results (default: test_error_handling_results.json)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    console.print("[bold blue]Error Handling Test Suite[/bold blue]\n")
    
    # Run all tests
    tests = [
        test_empty_conversation,
        test_missing_fields,
        test_missing_image,
        test_missing_question,
        test_checkpoint_resume,
        test_alternative_question_fields,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            results.append({
                "test_name": test_func.__name__,
                "passed": False,
                "error": f"Test execution failed: {e}",
            })
    
    # Display results
    display_test_results(results)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    console.print(f"\n[green]✓[/green] Test results saved to: {args.output}")
    
    # Return exit code
    passed = sum(1 for r in results if r["passed"])
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit(main())


