#!/usr/bin/env python3
"""
CoRGI Unified Inference Script

Runs the CoRGI pipeline (V1 or V2) on images without requiring the Gradio UI.
All results (answer, evidence, reasoning steps, visualizations) are saved to an output folder.

Usage:
    # Single image (auto-detect pipeline version from config)
    python inference.py --image path/to/image.jpg --question "What is in the image?" --output results/

    # Explicit pipeline version
    python inference.py --pipeline v2 --image image.jpg --question "..." --output results/

    # Batch processing
    python inference.py --batch images.txt --output results/

    # With specific config
    python inference.py --image image.jpg --question "..." --config configs/qwen_only_v2.yaml --output results/

    # Production mode (less logging)
    CORGI_LOG_LEVEL=WARNING python inference.py --image image.jpg --question "..." --output results/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("corgi.inference")


# =============================================================================
# Pipeline Loading
# =============================================================================


def detect_pipeline_version(config_path: Path) -> str:
    """
    Detect pipeline version from config filename or content.
    
    Args:
        config_path: Path to config file
        
    Returns:
        "v1" or "v2"
    """
    config_name = config_path.name.lower()
    
    # Check filename for version hints
    if "v2" in config_name or "_v2" in config_name:
        return "v2"
    
    # Check config content for use_v2 flag
    try:
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        if config_data.get("pipeline", {}).get("use_v2", False):
            return "v2"
    except Exception:
        pass
    
    return "v1"


def load_pipeline(
    config_path: Path,
    pipeline_version: str,
    warm_up: bool = True,
    sequential_loading: bool = False,
):
    """
    Load the appropriate pipeline based on version with optional warm-up.
    
    Args:
        config_path: Path to config YAML file
        pipeline_version: "v1" or "v2"
        warm_up: Whether to run dummy inference to warm CUDA kernels
        sequential_loading: Load models sequentially (more stable but slower)
        
    Returns:
        Tuple of (pipeline, pipeline_version)
    """
    from corgi.utils.warm_up import warm_up_pipeline, WarmUpConfig, verify_cuda_ready
    
    # Verify CUDA
    if not verify_cuda_ready():
        logger.warning("CUDA not available, continuing with CPU")
    
    # Use warm-up utility for full initialization
    if warm_up:
        warm_up_config = WarmUpConfig(
            run_dummy_inference=True,
            dummy_image_size=(224, 224),
            clear_cache_after=True,
            sequential_loading=sequential_loading,
        )
        
        use_v2 = pipeline_version == "v2"
        pipeline = warm_up_pipeline(
            config_path=config_path,
            warm_up_config=warm_up_config,
            use_v2=use_v2,
        )
    else:
        # Load without warm-up (faster but first inference slower)
        from corgi.core.config import CoRGiConfig
        from corgi.models.factory import VLMClientFactory
        
        logger.info(f"Loading pipeline {pipeline_version.upper()} from: {config_path}")
        
        config = CoRGiConfig.from_yaml(str(config_path))
        client = VLMClientFactory.create_from_config(
            config,
            parallel_loading=not sequential_loading,
        )
        
        if pipeline_version == "v2":
            from corgi.core.pipeline_v2 import CoRGIPipelineV2
            pipeline = CoRGIPipelineV2(vlm_client=client)
        else:
            from corgi.core.pipeline import CoRGIPipeline
            pipeline = CoRGIPipeline(vlm_client=client)
        
        logger.info(f"✓ Pipeline {pipeline_version.upper()} loaded (no warm-up)")
    
    return pipeline, pipeline_version


# =============================================================================
# Result Processing
# =============================================================================


def result_to_dict(result: Any, pipeline_version: str) -> Dict[str, Any]:
    """
    Convert pipeline result to a dictionary.
    
    Works with both PipelineResult (V1) and PipelineResultV2 (V2).
    
    Args:
        result: Pipeline result object
        pipeline_version: "v1" or "v2"
        
    Returns:
        Dictionary with standardized result data
    """
    # Use built-in to_json if available
    if hasattr(result, "to_json"):
        data = result.to_json()
    else:
        data = {}
    
    # Ensure common fields are present
    data.setdefault("question", getattr(result, "question", ""))
    data.setdefault("answer", getattr(result, "answer", ""))
    data.setdefault("explanation", getattr(result, "explanation", None))
    data.setdefault("total_duration_ms", getattr(result, "total_duration_ms", 0))
    
    # Steps
    if "steps" not in data and hasattr(result, "steps"):
        data["steps"] = []
        for step in result.steps:
            step_dict = {
                "index": getattr(step, "index", 0),
                "statement": getattr(step, "statement", ""),
                "needs_vision": getattr(step, "needs_vision", False),
            }
            # V1 fields
            if hasattr(step, "need_ocr"):
                step_dict["need_ocr"] = step.need_ocr
            # V2 fields
            if hasattr(step, "need_object_captioning"):
                step_dict["need_object_captioning"] = step.need_object_captioning
            if hasattr(step, "need_text_ocr"):
                step_dict["need_text_ocr"] = step.need_text_ocr
            if hasattr(step, "has_bbox"):
                step_dict["has_bbox"] = step.has_bbox
            if hasattr(step, "bbox") and step.bbox:
                step_dict["bbox"] = list(step.bbox)
            if hasattr(step, "evidence_type"):
                step_dict["evidence_type"] = step.evidence_type
            data["steps"].append(step_dict)
    
    # Evidence
    if "evidence" not in data and hasattr(result, "evidence"):
        data["evidence"] = []
        for ev in result.evidence:
            ev_dict = {
                "step_index": getattr(ev, "step_index", 0),
                "bbox": list(getattr(ev, "bbox", [])),
                "description": getattr(ev, "description", None),
                "ocr_text": getattr(ev, "ocr_text", None),
                "confidence": getattr(ev, "confidence", None),
            }
            # V2 fields
            if hasattr(ev, "evidence_type"):
                ev_dict["evidence_type"] = ev.evidence_type
            if hasattr(ev, "statement"):
                ev_dict["statement"] = ev.statement
            data["evidence"].append(ev_dict)
    
    # Key evidence
    if "key_evidence" not in data and hasattr(result, "key_evidence"):
        data["key_evidence"] = [
            {
                "bbox": list(getattr(ke, "bbox", [])),
                "description": getattr(ke, "description", ""),
                "reasoning": getattr(ke, "reasoning", ""),
            }
            for ke in result.key_evidence
        ]
    
    # Timings
    if "timings" not in data and hasattr(result, "timings"):
        data["timings"] = [
            {
                "name": getattr(t, "name", ""),
                "duration_ms": getattr(t, "duration_ms", 0),
                "step_index": getattr(t, "step_index", None),
            }
            for t in result.timings
        ]
    
    # V2 stats
    if pipeline_version == "v2":
        data["v2_stats"] = {
            "bbox_from_phase1_count": getattr(result, "bbox_from_phase1_count", 0),
            "object_evidence_count": getattr(result, "object_evidence_count", 0),
            "text_evidence_count": getattr(result, "text_evidence_count", 0),
        }
    
    # Additional fields
    data.setdefault("cot_text", getattr(result, "cot_text", None))
    data.setdefault("paraphrased_question", getattr(result, "paraphrased_question", None))
    
    return data


def evidence_to_bboxes(evidence_list: List[Dict], pipeline_version: str) -> List[Dict]:
    """
    Convert evidence list to bbox format for annotation.
    
    Args:
        evidence_list: List of evidence dicts
        pipeline_version: "v1" or "v2"
        
    Returns:
        List of bbox dicts for annotate_image_with_bboxes
    """
    bboxes = []
    for ev in evidence_list:
        step_index = ev.get("step_index", 0)
        evidence_type = ev.get("evidence_type", "default")
        
        # Build label
        if pipeline_version == "v2" and evidence_type:
            label = f"S{step_index}:{evidence_type[0].upper()}"
        else:
            label = f"S{step_index}"
        
        bboxes.append({
            "bbox": ev.get("bbox", [0, 0, 1, 1]),
            "label": label,
            "step_index": step_index,
            "evidence_type": evidence_type,
        })
    
    return bboxes


# =============================================================================
# Single Image Inference
# =============================================================================


def run_inference(
    image_path: Path,
    question: str,
    pipeline: Any,
    pipeline_version: str,
    output_dir: Path,
    max_steps: int = 6,
    max_regions: int = 5,
    save_crops: bool = True,
    save_visualization: bool = True,
) -> Dict[str, Any]:
    """
    Run inference on a single image and save all results.
    
    Args:
        image_path: Path to input image
        question: Question to ask
        pipeline: Pipeline instance (V1 or V2)
        pipeline_version: "v1" or "v2"
        output_dir: Directory to save results
        max_steps: Maximum reasoning steps
        max_regions: Maximum regions per step
        save_crops: Whether to save evidence crops
        save_visualization: Whether to save annotated image
        
    Returns:
        Result dictionary
    """
    # Import helpers
    from corgi.utils.inference_helpers import (
        setup_output_dir,
        annotate_image_with_bboxes,
        save_evidence_crops,
        save_results_json,
        save_summary_report,
    )
    
    logger.info(f"Processing: {image_path}")
    logger.info(f"Question: {question}")
    logger.info(f"Pipeline: {pipeline_version.upper()}")
    
    # Setup output directories
    paths = setup_output_dir(output_dir)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image.size}")
    
    # Run pipeline
    start_time = time.time()
    
    # V2 uses different default for max_regions
    if pipeline_version == "v2":
        max_regions = min(max_regions, 1)  # V2 typically uses 1 bbox per step
    
    result = pipeline.run(
        image=image,
        question=question,
        max_steps=max_steps,
        max_regions=max_regions,
    )
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s")
    logger.info(f"Answer: {result.answer}")
    
    # Convert result to dict
    result_data = result_to_dict(result, pipeline_version)
    
    # Save original image
    image.save(paths["images"] / "original.jpg", quality=95)
    
    # Save annotated image with evidence bboxes
    if save_visualization and result_data.get("evidence"):
        bboxes = evidence_to_bboxes(result_data["evidence"], pipeline_version)
        color_by = "type" if pipeline_version == "v2" else "step"
        
        annotate_image_with_bboxes(
            image=image,
            bboxes=bboxes,
            output_path=paths["visualizations"] / "annotated.jpg",
            color_by=color_by,
        )
    
    # Save evidence crops
    if save_crops and result_data.get("evidence"):
        save_evidence_crops(
            image=image,
            evidences=result_data["evidence"],
            evidence_dir=paths["evidence"],
        )
    
    # Save JSON results
    save_results_json(
        result_data=result_data,
        output_path=paths["root"] / "results.json",
        pipeline_version=pipeline_version,
    )
    
    # Save summary report
    save_summary_report(
        image_path=image_path,
        question=question,
        result_data=result_data,
        output_path=paths["root"] / "summary.txt",
        pipeline_version=pipeline_version,
    )
    
    logger.info(f"All results saved to: {output_dir}")
    return result_data


# =============================================================================
# Batch Inference
# =============================================================================


def batch_inference(
    batch_file: Path,
    pipeline: Any,
    pipeline_version: str,
    output_root: Path,
    max_steps: int = 6,
    max_regions: int = 5,
) -> None:
    """
    Run inference on a batch of images from a file.
    
    Batch file format: image_path|question (one per line)
    Lines starting with # are treated as comments.
    
    Args:
        batch_file: Path to batch file
        pipeline: Pipeline instance
        pipeline_version: "v1" or "v2"
        output_root: Root directory for all results
        max_steps: Maximum reasoning steps
        max_regions: Maximum regions per step
    """
    # Read batch file
    with open(batch_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
    
    logger.info(f"Processing {len(lines)} images from {batch_file}")
    
    results = []
    for idx, line in enumerate(lines, 1):
        try:
            # Parse line
            parts = line.split("|", 1)
            if len(parts) != 2:
                logger.warning(f"Skipping invalid line {idx}: {line}")
                continue
            
            image_path = Path(parts[0].strip())
            question = parts[1].strip()
            
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                continue
            
            # Create output directory for this image
            output_dir = output_root / f"result_{idx:04d}_{image_path.stem}"
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing {idx}/{len(lines)}: {image_path.name}")
            logger.info(f"{'=' * 80}")
            
            # Run inference
            result_data = run_inference(
                image_path=image_path,
                question=question,
                pipeline=pipeline,
                pipeline_version=pipeline_version,
                output_dir=output_dir,
                max_steps=max_steps,
                max_regions=max_regions,
            )
            
            # Collect summary
            results.append({
                "index": idx,
                "image": str(image_path),
                "question": question,
                "answer": result_data.get("answer", ""),
                "duration_ms": result_data.get("total_duration_ms", 0),
            })
            
            # Add V2 stats if available
            if pipeline_version == "v2" and "v2_stats" in result_data:
                results[-1]["v2_stats"] = result_data["v2_stats"]
        
        except Exception as e:
            logger.error(f"Failed to process line {idx}: {e}", exc_info=True)
            continue
    
    # Save batch summary
    batch_summary = output_root / "batch_summary.json"
    with open(batch_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_version": pipeline_version,
                "total": len(lines),
                "processed": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    
    logger.info(f"\nBatch processing complete: {len(results)}/{len(lines)} successful")
    logger.info(f"Batch summary saved to: {batch_summary}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CoRGI Unified Inference - Run V1 or V2 pipeline and save results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image with auto-detected pipeline version
    python inference.py --image photo.jpg --question "What is this?" --output results/

    # Explicit V2 pipeline
    python inference.py --pipeline v2 --image photo.jpg --question "What is this?"

    # Batch processing
    python inference.py --batch questions.txt --output batch_results/

    # With specific config
    python inference.py --config configs/qwen_only_v2.yaml --image photo.jpg --question "..."
        """,
    )
    
    # Pipeline version
    parser.add_argument(
        "--pipeline",
        choices=["v1", "v2", "auto"],
        default="auto",
        help="Pipeline version: v1, v2, or auto (detect from config). Default: auto",
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to single input image",
    )
    input_group.add_argument(
        "--batch",
        type=Path,
        help="Path to batch file (format: image_path|question per line)",
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (required for single image mode)",
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen_florence2_smolvlm2_v2.yaml"),
        help="Path to pipeline config YAML. Default: configs/qwen_florence2_smolvlm2_v2.yaml (multi-model V2)",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_results"),
        help="Output directory for results. Default: inference_results",
    )
    
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Skip saving individual evidence crops",
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip creating annotated visualizations",
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum reasoning steps. Default: 6",
    )
    
    parser.add_argument(
        "--max-regions",
        type=int,
        default=5,
        help="Maximum regions per step. Default: 5 (V1) or 1 (V2)",
    )
    
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="Skip model warm-up (faster start but first inference will be slower)",
    )
    
    parser.add_argument(
        "--sequential-loading",
        action="store_true",
        help="Load models sequentially instead of parallel (more stable but slower)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.image and not args.question:
        logger.error("--question is required when using --image")
        sys.exit(1)
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Determine pipeline version
    if args.pipeline == "auto":
        pipeline_version = detect_pipeline_version(args.config)
        logger.info(f"Auto-detected pipeline version: {pipeline_version.upper()}")
    else:
        pipeline_version = args.pipeline
    
    # Print header
    logger.info("=" * 80)
    logger.info(f"CoRGI Unified Inference - Pipeline {pipeline_version.upper()}")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    
    # Load pipeline with warm-up
    try:
        pipeline, pipeline_version = load_pipeline(
            config_path=args.config,
            pipeline_version=pipeline_version,
            warm_up=not args.no_warm_up,
            sequential_loading=args.sequential_loading,
        )
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        sys.exit(1)
    
    # Run inference
    try:
        if args.batch:
            # Batch mode
            batch_inference(
                batch_file=args.batch,
                pipeline=pipeline,
                pipeline_version=pipeline_version,
                output_root=args.output,
                max_steps=args.max_steps,
                max_regions=args.max_regions,
            )
        else:
            # Single image mode
            if not args.image.exists():
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)
            
            run_inference(
                image_path=args.image,
                question=args.question,
                pipeline=pipeline,
                pipeline_version=pipeline_version,
                output_dir=args.output,
                max_steps=args.max_steps,
                max_regions=args.max_regions,
                save_crops=not args.no_crops,
                save_visualization=not args.no_visualization,
            )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Inference complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
