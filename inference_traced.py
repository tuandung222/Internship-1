#!/usr/bin/env python3
"""
CoRGI Traced Inference

Enhanced inference with full component tracing for debugging and explainability.

Outputs:
- trace.json: Complete trace data in JSON format
- trace_report.html: Visual HTML report with all component I/O
- images/: Original and annotated images
- crops/: Cropped regions for each evidence
- visualizations/: Step-by-step bbox visualizations
- prompts/: Prompt text files for each component

Usage:
    # Basic usage
    python inference_traced.py --image test.jpg --question "What is this?" --output results/

    # With specific config
    python inference_traced.py --image test.jpg --question "..." \
        --config configs/qwen_florence2_smolvlm2_v2.yaml --output results/

    # Open HTML report after inference
    python inference_traced.py --image test.jpg --question "..." --output results/ --open-report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("corgi.inference_traced")


def run_traced_inference(
    image_path: Path,
    question: str,
    config_path: Path,
    output_dir: Path,
    max_steps: int = 6,
) -> Dict[str, Any]:
    """
    Run inference with full component tracing.
    
    Returns trace data with all component inputs/outputs.
    """
    from corgi.core.config import load_config
    from corgi.core.pipeline_v2 import CoRGIPipelineV2
    from corgi.core.streaming import StreamEventType
    from corgi.models.factory import VLMClientFactory
    from corgi.utils.trace_reporter import TraceReporter
    
    # Load config
    logger.info(f"Loading config: {config_path}")
    config = load_config(str(config_path))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trace reporter
    reporter = TraceReporter(
        output_dir=output_dir,
        save_crops=True,
        save_visualizations=True,
        save_prompts=True,
    )
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Start trace
    reporter.start_trace(
        question=question,
        image=image,
        image_path=str(image_path),
        pipeline_version="v2",
        config_path=str(config_path),
    )
    
    # Create pipeline
    logger.info("Loading models...")
    client = VLMClientFactory.create_from_config(config, parallel_loading=True)
    pipeline = CoRGIPipelineV2(vlm_client=client)
    
    # Run pipeline with streaming to capture all events
    logger.info("Running pipeline...")
    start_time = time.monotonic()
    
    current_step = 0
    steps_data: List[Dict] = []
    evidence_data: List[Dict] = []
    
    for event in pipeline.run_streaming(image, question, max_steps=max_steps):
        event_type = event.type
        
        # === PHASE START ===
        if event_type == StreamEventType.PHASE_START:
            phase = event.phase
            logger.info(f"Starting phase: {phase}")
            
            if phase == "reasoning_grounding":
                reporter.log_component_start(
                    component_name="reasoning_grounding",
                    phase="Phase 1+2: Reasoning + Grounding",
                    input_data={
                        "question": question,
                        "image_size": list(image.size),
                        "max_steps": max_steps,
                    },
                    model_id=config.reasoning.model.model_id,
                )
        
        # === COT TEXT ===
        elif event_type == StreamEventType.COT_TEXT:
            cot_text = event.data.get("cot_text", "")
            logger.info(f"Chain-of-thought generated ({len(cot_text)} chars)")
            
            # Update current component with COT
            if reporter.current_component:
                reporter.current_component.output_data["cot_text"] = cot_text
                reporter.current_component.raw_response = cot_text
        
        # === STEP ===
        elif event_type == StreamEventType.STEP:
            current_step += 1
            step_data = event.data
            steps_data.append(step_data)
            
            logger.info(f"Step {current_step}: {step_data.get('statement', '')[:50]}...")
            
            # End reasoning component after all steps
            if reporter.current_component and reporter.current_component.component_name == "reasoning_grounding":
                reporter.log_component_end(
                    output_data={
                        "steps_count": current_step,
                        "steps": steps_data,
                    }
                )
        
        # === EVIDENCE ===
        elif event_type == StreamEventType.EVIDENCE:
            ev_data = event.data
            evidence_data.append(ev_data)
            
            ev_type = ev_data.get("evidence_type", "unknown")
            bbox = ev_data.get("bbox", [])
            step_idx = event.step_index or 0
            
            # Determine which model was used
            if ev_type == "text":
                model_id = getattr(config.captioning, "ocr", None)
                if model_id and hasattr(model_id, "model"):
                    model_id = model_id.model.model_id
                else:
                    model_id = "Florence-2 OCR"
                component_name = f"ocr_step_{step_idx}"
            else:
                model_id = getattr(config.captioning, "caption", None)
                if model_id and hasattr(model_id, "model"):
                    model_id = model_id.model.model_id
                else:
                    model_id = "SmolVLM2 Caption"
                component_name = f"caption_step_{step_idx}"
            
            # Log component
            reporter.log_component_start(
                component_name=component_name,
                phase=f"Phase 3: Evidence Extraction ({ev_type})",
                input_data={
                    "step_index": step_idx,
                    "bbox": list(bbox) if bbox else None,
                    "evidence_type": ev_type,
                },
                model_id=str(model_id),
                bbox=tuple(bbox) if bbox and len(bbox) == 4 else None,
            )
            
            reporter.log_component_end(
                output_data={
                    "evidence_type": ev_type,
                    "description": ev_data.get("description"),
                    "ocr_text": ev_data.get("ocr_text"),
                },
                bbox=tuple(bbox) if bbox and len(bbox) == 4 else None,
            )
            
            logger.info(f"Evidence ({ev_type}): {ev_data.get('description') or ev_data.get('ocr_text', '')[:50]}...")
        
        # === ANSWER ===
        elif event_type == StreamEventType.ANSWER:
            answer = event.data.get("answer", "")
            explanation = event.data.get("explanation", "")
            
            # Log synthesis component
            reporter.log_component_start(
                component_name="synthesis",
                phase="Phase 4: Answer Synthesis",
                input_data={
                    "steps_count": len(steps_data),
                    "evidence_count": len(evidence_data),
                },
                model_id=config.reasoning.model.model_id,
            )
            
            reporter.log_component_end(
                output_data={
                    "answer": answer,
                    "explanation": explanation,
                },
                raw_response=answer,
            )
            
            logger.info(f"Answer: {answer[:100]}...")
        
        # === PIPELINE END ===
        elif event_type == StreamEventType.PIPELINE_END:
            total_duration = event.data.get("total_duration_ms", 0)
            
            reporter.log_final_result(
                answer=event.data.get("answer", ""),
                explanation=None,
                total_duration_ms=total_duration,
            )
    
    end_time = time.monotonic()
    total_ms = (end_time - start_time) * 1000
    
    if reporter.trace:
        reporter.trace.total_duration_ms = total_ms
    
    # Save trace
    trace_json_path = reporter.save_trace_json()
    logger.info(f"Saved trace JSON: {trace_json_path}")
    
    # Generate HTML report
    html_path = reporter.generate_html_report()
    logger.info(f"Generated HTML report: {html_path}")
    
    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CoRGI Traced Inference Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Duration: {total_ms/1000:.2f}s\n\n")
        f.write(f"Steps: {len(steps_data)}\n")
        f.write(f"Evidence: {len(evidence_data)}\n\n")
        f.write(f"Answer:\n{reporter.trace.final_answer if reporter.trace else 'N/A'}\n")
    
    logger.info(f"Saved summary: {summary_path}")
    
    return reporter.trace.to_dict() if reporter.trace else {}


def main():
    parser = argparse.ArgumentParser(
        description="CoRGI Traced Inference - Full component tracing for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Structure:
    output_dir/
    ├── trace.json           # Complete trace data
    ├── trace_report.html    # Visual HTML report (open this!)
    ├── summary.txt          # Quick summary
    ├── images/
    │   └── original.jpg     # Input image
    ├── crops/
    │   └── *.jpg           # Cropped regions per component
    ├── visualizations/
    │   └── *.jpg           # Images with bboxes drawn
    └── prompts/
        └── *.txt           # Prompt text per component

Examples:
    # Basic usage
    python inference_traced.py --image photo.jpg --question "What is this?"

    # With specific config
    python inference_traced.py --image chart.png --question "What numbers are shown?" \\
        --config configs/qwen_florence2_smolvlm2_v2.yaml

    # Open HTML report automatically
    python inference_traced.py --image photo.jpg --question "..." --open-report
        """,
    )
    
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the image",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen_florence2_smolvlm2_v2.yaml"),
        help="Path to config YAML (default: multi-model V2)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trace"),
        help="Output directory for trace (default: results/trace)",
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum reasoning steps (default: 6)",
    )
    
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open HTML report in browser after inference",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image.exists():
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
    
    if not args.config.exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    # Run traced inference
    print("=" * 70)
    print("CoRGI TRACED INFERENCE")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print()
    
    try:
        trace_data = run_traced_inference(
            image_path=args.image,
            question=args.question,
            config_path=args.config,
            output_dir=args.output,
            max_steps=args.max_steps,
        )
        
        print()
        print("=" * 70)
        print("TRACE COMPLETE!")
        print("=" * 70)
        print(f"Output directory: {args.output}")
        print()
        print("Files generated:")
        print(f"  📊 trace.json         - Complete trace data")
        print(f"  🌐 trace_report.html  - Visual HTML report (recommended)")
        print(f"  📝 summary.txt        - Quick summary")
        print(f"  🖼️  images/            - Original image")
        print(f"  ✂️  crops/             - Cropped evidence regions")
        print(f"  📍 visualizations/    - Step-by-step bbox images")
        print(f"  📄 prompts/           - Component prompts")
        print()
        
        if args.open_report:
            html_path = args.output / "trace_report.html"
            if html_path.exists():
                print(f"Opening report: {html_path}")
                webbrowser.open(f"file://{html_path.absolute()}")
        else:
            print(f"To view the report, open: {args.output / 'trace_report.html'}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
