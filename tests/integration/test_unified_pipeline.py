#!/usr/bin/env python3
"""
Integration Tests for Unified CoRGI Pipeline

Tests the refactored components:
- Unified inference.py (V1 + V2 support)
- app_unified.py entrypoint
- Streaming API
- inference_helpers utilities

Usage:
    pytest tests/integration/test_unified_pipeline.py -v
    python tests/integration/test_unified_pipeline.py  # Direct run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    return img


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return "What objects are in this image?"


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test: Streaming API
# =============================================================================


class TestStreamingAPI:
    """Tests for corgi.core.streaming module."""

    def test_stream_event_type_enum(self):
        """Test StreamEventType has all expected values."""
        from corgi.core.streaming import StreamEventType

        expected_types = [
            "PIPELINE_START",
            "PIPELINE_END",
            "PHASE_START",
            "PHASE_END",
            "COT_TEXT",
            "STEP",
            "BBOX",
            "EVIDENCE",
            "ANSWER",
            "KEY_EVIDENCE",
            "WARNING",
            "ERROR",
        ]

        for et in expected_types:
            assert hasattr(StreamEventType, et), f"Missing event type: {et}"

    def test_stream_event_creation(self):
        """Test StreamEvent dataclass creation."""
        from corgi.core.streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.STEP,
            phase="reasoning",
            step_index=0,
            data={"statement": "Test statement"},
            timestamp=1.5,
            duration_ms=100.0,
        )

        assert event.type == StreamEventType.STEP
        assert event.phase == "reasoning"
        assert event.step_index == 0
        assert event.data["statement"] == "Test statement"
        assert event.timestamp == 1.5
        assert event.duration_ms == 100.0

    def test_stream_event_to_dict(self):
        """Test StreamEvent.to_dict() serialization."""
        from corgi.core.streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.EVIDENCE,
            phase="evidence_extraction",
            step_index=2,
            data={"evidence_type": "object", "description": "A cat"},
        )

        d = event.to_dict()

        assert d["type"] == "EVIDENCE"
        assert d["phase"] == "evidence_extraction"
        assert d["step_index"] == 2
        assert d["data"]["evidence_type"] == "object"
        assert d["data"]["description"] == "A cat"

    def test_streaming_executor_import(self):
        """Test StreamingPipelineExecutor can be imported."""
        from corgi.core.streaming import StreamingPipelineExecutor
        from corgi.core import StreamingPipelineExecutor as Executor

        assert StreamingPipelineExecutor is Executor


# =============================================================================
# Test: Inference Helpers
# =============================================================================


class TestInferenceHelpers:
    """Tests for corgi.utils.inference_helpers module."""

    def test_setup_output_dir(self, temp_output_dir):
        """Test setup_output_dir creates correct structure."""
        from corgi.utils.inference_helpers import setup_output_dir

        paths = setup_output_dir(temp_output_dir)

        assert "root" in paths
        assert "images" in paths
        assert "visualizations" in paths
        assert "evidence" in paths

        assert paths["root"].exists()
        assert paths["images"].exists()
        assert paths["visualizations"].exists()
        assert paths["evidence"].exists()

    def test_annotate_image_with_bboxes(self, sample_image, temp_output_dir):
        """Test bbox annotation on images."""
        from corgi.utils.inference_helpers import annotate_image_with_bboxes

        bboxes = [
            {"bbox": [0.1, 0.1, 0.5, 0.5], "label": "Object 1", "type": "object"},
            {"bbox": [0.6, 0.2, 0.9, 0.8], "label": "Text 1", "type": "text"},
        ]

        output_path = temp_output_dir / "annotated.jpg"
        result = annotate_image_with_bboxes(
            sample_image,
            bboxes,
            output_path=output_path,
        )

        assert isinstance(result, Image.Image)
        assert output_path.exists()

    def test_save_results_json(self, temp_output_dir):
        """Test saving results to JSON."""
        from corgi.utils.inference_helpers import save_results_json

        result_data = {
            "question": "Test question",
            "answer": "Test answer",
            "steps": [{"index": 1, "statement": "Step 1"}],
            "evidence": [{"bbox": [0.1, 0.1, 0.5, 0.5], "type": "object"}],
        }

        output_path = temp_output_dir / "results.json"
        save_results_json(result_data, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["question"] == "Test question"
        assert loaded["answer"] == "Test answer"

    def test_save_summary_report(self, temp_output_dir):
        """Test saving summary report."""
        from corgi.utils.inference_helpers import save_summary_report

        result_data = {
            "question": "What is this?",
            "answer": "A test image",
            "explanation": "This is an explanation",
            "steps": [{"index": 1, "statement": "Analyze image"}],
            "evidence": [],
            "total_duration_ms": 1500.0,
        }

        image_path = temp_output_dir / "test.jpg"
        # Create a dummy image file
        Image.new("RGB", (100, 100)).save(image_path)

        output_path = temp_output_dir / "summary.txt"
        save_summary_report(
            image_path=image_path,
            question="What is this?",
            result_data=result_data,
            output_path=output_path,
        )

        assert output_path.exists()

        content = output_path.read_text()
        assert "What is this?" in content
        assert "A test image" in content


# =============================================================================
# Test: Unified Inference Script
# =============================================================================


class TestUnifiedInference:
    """Tests for unified inference.py script."""

    def test_detect_pipeline_version_v2(self):
        """Test V2 detection from config name."""
        from inference import detect_pipeline_version

        v2_config = Path("configs/qwen_only_v2.yaml")
        version = detect_pipeline_version(v2_config)
        assert version == "v2"

    def test_detect_pipeline_version_v1(self):
        """Test V1 detection from config name."""
        from inference import detect_pipeline_version

        v1_config = Path("configs/legacy/qwen_only.yaml")
        version = detect_pipeline_version(v1_config)
        assert version == "v1"

    def test_result_to_dict_basic(self):
        """Test result conversion to dict."""
        from inference import result_to_dict

        # Create mock result with to_json method
        mock_result = MagicMock()
        mock_result.question = "Test question"
        mock_result.answer = "Test answer"
        mock_result.explanation = "Test explanation"
        mock_result.cot_text = "Chain of thought"
        mock_result.steps = []
        mock_result.evidence = []
        mock_result.key_evidence = []
        mock_result.total_duration_ms = 1000.0
        mock_result.timings = []
        
        # Mock to_json to return None so it falls through to attribute access
        mock_result.to_json = MagicMock(return_value={
            "question": "Test question",
            "answer": "Test answer",
        })

        result_dict = result_to_dict(mock_result, "v2")

        # The function should extract these from the mock
        assert result_dict.get("question") == "Test question"
        assert result_dict.get("answer") == "Test answer"

    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "inference.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )

        assert result.returncode == 0
        assert "--pipeline" in result.stdout
        assert "--config" in result.stdout
        assert "--image" in result.stdout
        assert "--question" in result.stdout


# =============================================================================
# Test: Unified App Entrypoint
# =============================================================================


class TestUnifiedApp:
    """Tests for app_unified.py."""

    def test_resolve_config_v2(self):
        """Test config resolution for V2."""
        from app_unified import resolve_config

        config = resolve_config(None, "v2")
        assert config.exists()
        assert "v2" in config.name.lower() or config.name == "qwen_only_v2.yaml"

    def test_resolve_config_v1(self):
        """Test config resolution for V1."""
        from app_unified import resolve_config

        config = resolve_config(None, "v1")
        assert config.exists()

    def test_resolve_config_explicit(self):
        """Test explicit config path."""
        from app_unified import resolve_config

        explicit = Path("configs/qwen_only_v2.yaml")
        config = resolve_config(explicit, "v2")
        assert config == explicit

    def test_build_standard_app_callable(self):
        """Test build_standard_app is callable."""
        from app_unified import build_standard_app

        assert callable(build_standard_app)

    def test_build_chatbot_app_callable(self):
        """Test build_chatbot_app is callable."""
        from app_unified import build_chatbot_app

        assert callable(build_chatbot_app)

    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "app_unified.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )

        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--pipeline" in result.stdout
        assert "--port" in result.stdout


# =============================================================================
# Test: Deprecation Warnings
# =============================================================================


class TestDeprecationWarnings:
    """Tests for deprecation warnings on old files."""

    def test_inference_v2_deprecation(self):
        """Test inference_v2.py shows deprecation warning."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-W", "default", "-c", "import inference_v2"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )

        assert "DeprecationWarning" in result.stderr

    def test_app_v2_deprecation(self):
        """Test app_v2.py shows deprecation warning."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-W", "default", "-c", "import app_v2"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )

        assert "DeprecationWarning" in result.stderr


# =============================================================================
# Test: Pipeline V2 with Streaming
# =============================================================================


class TestPipelineV2Streaming:
    """Tests for CoRGIPipelineV2 with streaming support."""

    def test_run_streaming_method_exists(self):
        """Test run_streaming method exists on CoRGIPipelineV2."""
        from corgi.core.pipeline_v2 import CoRGIPipelineV2

        assert hasattr(CoRGIPipelineV2, "run_streaming")

    def test_core_module_exports(self):
        """Test corgi.core exports streaming components."""
        from corgi.core import (
            CoRGIPipelineV2,
            PipelineResultV2,
            StreamEvent,
            StreamEventType,
            StreamingPipelineExecutor,
        )

        assert CoRGIPipelineV2 is not None
        assert PipelineResultV2 is not None
        assert StreamEvent is not None
        assert StreamEventType is not None
        assert StreamingPipelineExecutor is not None


# =============================================================================
# Test: Gradio Chatbot V2
# =============================================================================


class TestGradioChatbotV2:
    """Tests for refactored gradio_chatbot_v2.py."""

    def test_import_success(self):
        """Test gradio_chatbot_v2.py can be imported."""
        import gradio_chatbot_v2

        assert hasattr(gradio_chatbot_v2, "demo")
        assert hasattr(gradio_chatbot_v2, "load_pipeline")
        assert hasattr(gradio_chatbot_v2, "stream_pipeline_execution")

    def test_uses_streaming_api(self):
        """Test gradio_chatbot_v2.py uses streaming API."""
        import inspect
        import gradio_chatbot_v2

        source = inspect.getsource(gradio_chatbot_v2.stream_pipeline_execution)

        assert "run_streaming" in source
        assert "StreamEventType" in source

    def test_demo_is_gradio_blocks(self):
        """Test demo is a Gradio Blocks instance."""
        import gradio as gr
        import gradio_chatbot_v2

        assert isinstance(gradio_chatbot_v2.demo, gr.Blocks)


# =============================================================================
# Main: Run tests directly
# =============================================================================


def run_all_tests():
    """Run all tests and print results."""
    import traceback

    test_classes = [
        TestStreamingAPI,
        TestInferenceHelpers,
        TestUnifiedInference,
        TestUnifiedApp,
        TestDeprecationWarnings,
        TestPipelineV2Streaming,
        TestGradioChatbotV2,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("UNIFIED PIPELINE INTEGRATION TESTS")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()

        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                method = getattr(instance, name)

                try:
                    # Setup fixtures if needed based on method signature
                    import inspect
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    
                    if "sample_image" in params and "temp_output_dir" in params:
                        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
                        with tempfile.TemporaryDirectory() as tmpdir:
                            method(img, Path(tmpdir))
                    elif "temp_output_dir" in params:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            method(Path(tmpdir))
                    else:
                        method()

                    print(f"  ✅ {name}")
                    passed += 1

                except Exception as e:
                    print(f"  ❌ {name}: {e}")
                    failed += 1
                    errors.append((test_class.__name__, name, traceback.format_exc()))

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFailed Tests Details:")
        for class_name, test_name, tb in errors:
            print(f"\n{class_name}.{test_name}:")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
