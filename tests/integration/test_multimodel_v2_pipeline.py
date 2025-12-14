#!/usr/bin/env python3
"""
Comprehensive Integration Test for Multi-Model V2 Pipeline

Tests the full pipeline with:
- Qwen3-VL (Reasoning + Grounding)
- Florence-2 (OCR)
- SmolVLM2 (Captioning)

Test Scenarios:
1. Object Reasoning: Dog/animal image with reasoning questions
2. OCR Localization: Chart/text image with text extraction questions

This script validates:
- Individual component initialization
- Component I/O compatibility
- Full pipeline end-to-end execution
- Output quality and format

Usage:
    python tests/integration/test_multimodel_v2_pipeline.py
    python tests/integration/test_multimodel_v2_pipeline.py --gpu cuda:0
    python tests/integration/test_multimodel_v2_pipeline.py --skip-model-load  # Skip actual model loading
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parents[2]))

from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("test_multimodel_v2")


# =============================================================================
# Test Data
# =============================================================================

TEST_DATA_DIR = Path(__file__).parents[2] / "test_data"

# Test scenarios
TEST_SCENARIOS = [
    {
        "id": "object_reasoning",
        "name": "Object Reasoning (Dog Scene)",
        "image": "dog_cat_scene.jpg",
        "question": "What is the dog doing in this image? Describe its posture and expression.",
        "expected_evidence_type": "object",  # Should use SmolVLM2 captioning
        "description": "Tests object recognition and reasoning with captioning",
    },
    {
        "id": "ocr_chart",
        "name": "OCR Chart Reading",
        "image": "dashboard_chart.jpg",
        "question": "What numbers or text can you see in this chart? Read any visible labels.",
        "expected_evidence_type": "text",  # Should use Florence-2 OCR
        "description": "Tests text/number recognition with OCR",
    },
    {
        "id": "ocr_document",
        "name": "OCR Document Reading",
        "image": "book_page.jpg",
        "question": "What text can you read from this image?",
        "expected_evidence_type": "text",
        "description": "Tests document text extraction with OCR",
    },
    {
        "id": "mixed_scene",
        "name": "Mixed Scene (Objects + Text)",
        "image": "restaurant_menu.jpg",
        "question": "What type of place is this and what items can you see or read?",
        "expected_evidence_type": "mixed",  # May use both
        "description": "Tests mixed object and text recognition",
    },
]


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    test_name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Component Tests
# =============================================================================


def test_config_loading() -> TestResult:
    """Test that the default config can be loaded."""
    test_id = "config_loading"
    start = time.monotonic()
    
    try:
        from corgi.core.config import load_config
        
        config_path = Path("configs/qwen_florence2_smolvlm2_v2.yaml")
        config = load_config(str(config_path))
        
        # Validate config structure
        assert hasattr(config, "reasoning"), "Missing reasoning config"
        assert hasattr(config, "captioning"), "Missing captioning config"
        assert hasattr(config, "grounding"), "Missing grounding config"
        assert hasattr(config, "synthesis"), "Missing synthesis config"
        
        # Check reasoning model
        assert config.reasoning.model.model_type == "qwen_instruct", \
            f"Expected qwen_instruct, got {config.reasoning.model.model_type}"
        
        # Check composite captioning
        assert config.captioning.model.model_type == "composite", \
            f"Expected composite, got {config.captioning.model.model_type}"
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Config Loading",
            passed=True,
            duration_ms=duration,
            details={
                "reasoning_model": config.reasoning.model.model_id,
                "captioning_type": config.captioning.model.model_type,
                "grounding_reuse": getattr(config.grounding, "reuse_reasoning", False),
            }
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Config Loading",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


def test_streaming_api() -> TestResult:
    """Test streaming API components."""
    test_id = "streaming_api"
    start = time.monotonic()
    
    try:
        from corgi.core.streaming import (
            StreamEventType,
            StreamEvent,
            StreamingPipelineExecutor,
        )
        from corgi.core import (
            CoRGIPipelineV2,
            StreamEventType as SE,
        )
        
        # Test event creation
        event = StreamEvent(
            type=StreamEventType.STEP,
            phase="reasoning",
            step_index=0,
            data={"statement": "Test"},
        )
        
        assert event.type == StreamEventType.STEP
        assert event.to_dict()["type"] == "STEP"
        
        # Test all event types exist
        required_types = [
            "PIPELINE_START", "PIPELINE_END",
            "PHASE_START", "PHASE_END",
            "STEP", "EVIDENCE", "ANSWER",
        ]
        for t in required_types:
            assert hasattr(StreamEventType, t), f"Missing event type: {t}"
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Streaming API",
            passed=True,
            duration_ms=duration,
            details={"event_types": len(list(StreamEventType))},
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Streaming API",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


def test_inference_helpers() -> TestResult:
    """Test inference helper utilities."""
    test_id = "inference_helpers"
    start = time.monotonic()
    
    try:
        from corgi.utils.inference_helpers import (
            setup_output_dir,
            annotate_image_with_bboxes,
            save_results_json,
            evidence_to_bbox_list,
        )
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test setup_output_dir
            paths = setup_output_dir(Path(tmpdir))
            assert paths["root"].exists()
            assert paths["images"].exists()
            assert paths["visualizations"].exists()
            
            # Test annotate_image_with_bboxes
            img = Image.new("RGB", (640, 480), color=(100, 150, 200))
            bboxes = [
                {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "Test", "type": "object"},
            ]
            annotated = annotate_image_with_bboxes(img, bboxes)
            assert isinstance(annotated, Image.Image)
            
            # Test save_results_json
            result_data = {"question": "Test", "answer": "Answer"}
            output_path = Path(tmpdir) / "test.json"
            save_results_json(result_data, output_path)
            assert output_path.exists()
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Inference Helpers",
            passed=True,
            duration_ms=duration,
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Inference Helpers",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


def test_type_definitions() -> TestResult:
    """Test V2 type definitions."""
    test_id = "type_definitions"
    start = time.monotonic()
    
    try:
        from corgi.core.types_v2 import ReasoningStepV2, GroundedEvidenceV2
        from corgi.core.types import KeyEvidence, StageTiming
        from corgi.core.pipeline_v2 import PipelineResultV2
        
        # Test ReasoningStepV2
        step = ReasoningStepV2(
            index=1,
            statement="Test statement",
            need_object_captioning=True,
            need_text_ocr=False,
            bbox=[0.1, 0.2, 0.3, 0.4],
        )
        assert step.needs_vision == True
        assert step.has_bbox == True
        
        # Test GroundedEvidenceV2
        evidence = GroundedEvidenceV2(
            step_index=1,
            statement="Test",
            bbox=(0.1, 0.2, 0.3, 0.4),
            evidence_type="object",
            description="A test object",
            confidence=0.95,
        )
        assert evidence.evidence_type == "object"
        
        # Test to_dict methods
        step_dict = step.to_dict()
        assert "statement" in step_dict
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="V2 Type Definitions",
            passed=True,
            duration_ms=duration,
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="V2 Type Definitions",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


def test_image_loading() -> TestResult:
    """Test that all test images can be loaded."""
    test_id = "image_loading"
    start = time.monotonic()
    
    try:
        loaded_images = []
        
        for scenario in TEST_SCENARIOS:
            img_path = TEST_DATA_DIR / scenario["image"]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                loaded_images.append({
                    "name": scenario["image"],
                    "size": img.size,
                })
            else:
                raise FileNotFoundError(f"Missing test image: {img_path}")
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Image Loading",
            passed=True,
            duration_ms=duration,
            details={"images": loaded_images},
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Image Loading",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


# =============================================================================
# Model Loading Tests (Requires GPU)
# =============================================================================


def test_model_factory(gpu_device: str = "cuda:0") -> TestResult:
    """Test VLM client factory."""
    test_id = "model_factory"
    start = time.monotonic()
    
    try:
        from corgi.core.config import load_config
        from corgi.models.factory import VLMClientFactory
        
        # Load config
        config = load_config("configs/qwen_florence2_smolvlm2_v2.yaml")
        
        # Override device for testing
        if hasattr(config.reasoning.model, "device"):
            config.reasoning.model.device = gpu_device
        
        # Test factory can create client
        # Note: This will actually load models
        client = VLMClientFactory.create_from_config(
            config,
            parallel_loading=True,
        )
        
        # Validate client has required methods
        assert hasattr(client, "structured_reasoning_v2"), "Missing structured_reasoning_v2"
        assert hasattr(client, "ocr_region"), "Missing ocr_region"
        assert hasattr(client, "caption_region"), "Missing caption_region"
        assert hasattr(client, "synthesize_answer"), "Missing synthesize_answer"
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Model Factory",
            passed=True,
            duration_ms=duration,
            details={"device": gpu_device},
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Model Factory",
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


def test_pipeline_initialization(gpu_device: str = "cuda:0") -> Tuple[TestResult, Any]:
    """Test pipeline initialization and return pipeline for further tests."""
    test_id = "pipeline_init"
    start = time.monotonic()
    pipeline = None
    
    try:
        from corgi.core.config import load_config
        from corgi.core.pipeline_v2 import CoRGIPipelineV2
        from corgi.models.factory import VLMClientFactory
        
        # Load config
        config = load_config("configs/qwen_florence2_smolvlm2_v2.yaml")
        
        # Create client
        client = VLMClientFactory.create_from_config(
            config,
            parallel_loading=True,
        )
        
        # Create pipeline
        pipeline = CoRGIPipelineV2(vlm_client=client)
        
        # Validate pipeline
        assert hasattr(pipeline, "run"), "Missing run method"
        assert hasattr(pipeline, "run_streaming"), "Missing run_streaming method"
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Pipeline Initialization",
            passed=True,
            duration_ms=duration,
        ), pipeline
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name="Pipeline Initialization",
            passed=False,
            duration_ms=duration,
            error=str(e),
        ), None


def test_scenario(
    pipeline: Any,
    scenario: Dict[str, Any],
) -> TestResult:
    """Test a single scenario with the pipeline."""
    test_id = f"scenario_{scenario['id']}"
    start = time.monotonic()
    
    try:
        from corgi.core.streaming import StreamEventType
        
        # Load image
        img_path = TEST_DATA_DIR / scenario["image"]
        image = Image.open(img_path).convert("RGB")
        question = scenario["question"]
        
        logger.info(f"Testing scenario: {scenario['name']}")
        logger.info(f"Question: {question}")
        
        # Run pipeline with streaming to capture all events
        events_captured = []
        steps_count = 0
        evidence_count = 0
        evidence_types = []
        answer = None
        
        for event in pipeline.run_streaming(image, question, max_steps=4):
            events_captured.append(event.type.name)
            
            if event.type == StreamEventType.STEP:
                steps_count += 1
                logger.info(f"  Step {steps_count}: {event.data.get('statement', '')[:50]}...")
                
            elif event.type == StreamEventType.EVIDENCE:
                evidence_count += 1
                ev_type = event.data.get("evidence_type", "unknown")
                evidence_types.append(ev_type)
                logger.info(f"  Evidence: {ev_type}")
                
            elif event.type == StreamEventType.ANSWER:
                answer = event.data.get("answer", "")
                logger.info(f"  Answer: {answer[:100]}...")
        
        # Validate results
        assert "PIPELINE_START" in events_captured, "Missing PIPELINE_START"
        assert "PIPELINE_END" in events_captured, "Missing PIPELINE_END"
        assert steps_count > 0, "No reasoning steps generated"
        assert answer is not None and len(answer) > 0, "No answer generated"
        
        # Check expected evidence type
        expected_type = scenario.get("expected_evidence_type")
        if expected_type and expected_type != "mixed" and evidence_types:
            assert expected_type in evidence_types, \
                f"Expected {expected_type} evidence, got {evidence_types}"
        
        duration = (time.monotonic() - start) * 1000
        return TestResult(
            test_id=test_id,
            test_name=scenario["name"],
            passed=True,
            duration_ms=duration,
            details={
                "steps": steps_count,
                "evidence": evidence_count,
                "evidence_types": evidence_types,
                "answer_preview": answer[:200] if answer else None,
            },
        )
        
    except Exception as e:
        duration = (time.monotonic() - start) * 1000
        logger.error(f"Scenario failed: {e}")
        logger.error(traceback.format_exc())
        return TestResult(
            test_id=test_id,
            test_name=scenario["name"],
            passed=False,
            duration_ms=duration,
            error=str(e),
        )


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests(
    skip_model_load: bool = False,
    gpu_device: str = "cuda:0",
    output_file: Optional[Path] = None,
) -> bool:
    """Run all tests and return success status."""
    
    print("=" * 70)
    print("MULTI-MODEL V2 PIPELINE INTEGRATION TESTS")
    print(f"Pipeline: Qwen + Florence-2 + SmolVLM2")
    print(f"GPU: {gpu_device}")
    print("=" * 70)
    print()
    
    results: List[TestResult] = []
    
    # === Phase 1: Component Tests (no GPU required) ===
    print("PHASE 1: Component Tests (No GPU Required)")
    print("-" * 50)
    
    component_tests = [
        ("Config Loading", test_config_loading),
        ("Streaming API", test_streaming_api),
        ("Inference Helpers", test_inference_helpers),
        ("V2 Type Definitions", test_type_definitions),
        ("Image Loading", test_image_loading),
    ]
    
    for name, test_fn in component_tests:
        result = test_fn()
        results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {name} ({result.duration_ms:.0f}ms)")
        if not result.passed:
            print(f"       Error: {result.error}")
    
    print()
    
    # === Phase 2: Model Loading Tests (GPU required) ===
    if not skip_model_load:
        print("PHASE 2: Model Loading Tests (GPU Required)")
        print("-" * 50)
        
        # Initialize pipeline (loads all models)
        print("  Loading models (this may take a while)...")
        init_result, pipeline = test_pipeline_initialization(gpu_device)
        results.append(init_result)
        
        status = "✅ PASS" if init_result.passed else "❌ FAIL"
        print(f"  {status} Pipeline Initialization ({init_result.duration_ms:.0f}ms)")
        
        if not init_result.passed:
            print(f"       Error: {init_result.error}")
            print()
            print("Skipping scenario tests due to pipeline initialization failure.")
        else:
            print()
            
            # === Phase 3: Scenario Tests ===
            print("PHASE 3: Scenario Tests (Full Pipeline)")
            print("-" * 50)
            
            for scenario in TEST_SCENARIOS:
                result = test_scenario(pipeline, scenario)
                results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"  {status} {scenario['name']} ({result.duration_ms:.0f}ms)")
                if not result.passed:
                    print(f"       Error: {result.error}")
                elif result.details:
                    print(f"       Steps: {result.details.get('steps', 0)}, "
                          f"Evidence: {result.details.get('evidence', 0)}")
    else:
        print("PHASE 2: Skipped (--skip-model-load)")
        print("PHASE 3: Skipped (--skip-model-load)")
    
    print()
    
    # === Summary ===
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    success = passed == total
    
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if not success:
        print()
        print("FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"  - {r.test_name}: {r.error}")
    
    # Save results to file
    if output_file:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_device": gpu_device,
            "skip_model_load": skip_model_load,
            "passed": passed,
            "total": total,
            "success": success,
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                    "details": r.details,
                }
                for r in results
            ],
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model V2 Pipeline Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="GPU device to use (default: cuda:0)",
    )
    
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip tests that require loading models (quick validation)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    
    args = parser.parse_args()
    
    success = run_all_tests(
        skip_model_load=args.skip_model_load,
        gpu_device=args.gpu,
        output_file=args.output,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
