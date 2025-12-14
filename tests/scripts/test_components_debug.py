#!/usr/bin/env python
"""Test each CoRGI component with real Qwen3-VL model to find bugs."""

import os
import sys
from pathlib import Path
from PIL import Image

# Add corgi to path
sys.path.insert(0, str(Path(__file__).parent))

from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig
from corgi.parsers import parse_structured_reasoning, parse_roi_evidence
from corgi.pipeline import CoRGIPipeline


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def test_model_load():
    """Test 1: Check if model loads correctly."""
    print_section("TEST 1: Model Loading")
    try:
        config = QwenGenerationConfig(model_id="Qwen/Qwen3-VL-8B-Thinking")
        client = Qwen3VLClient(config)
        print("✓ Model loaded successfully!")
        print(f"  Model ID: {config.model_id}")
        print(f"  Max tokens: {config.max_new_tokens}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Do sample: {config.do_sample}")
        return client
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_basic_inference(client: Qwen3VLClient):
    """Test 2: Check basic model inference."""
    print_section("TEST 2: Basic Inference")
    
    # Use a simple test image
    test_image_path = "/home/dungvpt/workspace/corgi_implementation/Qwen3-VL/cookbooks/assets/spatial_understanding/spatio_case1.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"✗ Test image not found: {test_image_path}")
        return None
    
    image = Image.open(test_image_path).convert("RGB")
    print(f"✓ Loaded test image: {image.size}")
    
    # Simple question
    prompt = "Describe what you see in this image in one sentence."
    
    try:
        print(f"\nPrompt: {prompt}\n")
        response = client._chat(image=image, prompt=prompt, max_new_tokens=100)
        print(f"Response:\n{response}\n")
        print("✓ Basic inference works!")
        return image
    except Exception as e:
        print(f"✗ Basic inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_structured_reasoning(client: Qwen3VLClient, image: Image.Image):
    """Test 3: Check structured reasoning output."""
    print_section("TEST 3: Structured Reasoning")
    
    question = "How many cakes are visible in this image?"
    max_steps = 3
    
    try:
        print(f"Question: {question}")
        print(f"Max steps: {max_steps}\n")
        
        steps = client.structured_reasoning(image=image, question=question, max_steps=max_steps)
        
        print(f"✓ Got {len(steps)} reasoning steps\n")
        
        if client.reasoning_log:
            print("--- RAW PROMPT ---")
            print(client.reasoning_log.prompt[:500] + "..." if len(client.reasoning_log.prompt) > 500 else client.reasoning_log.prompt)
            print("\n--- RAW RESPONSE ---")
            print(client.reasoning_log.response)
            print("\n--- PARSED STEPS ---")
        
        for step in steps:
            print(f"  Step {step.index}: {step.statement}")
            print(f"    Needs vision: {step.needs_vision}")
            if step.reason:
                print(f"    Reason: {step.reason}")
            print()
        
        return steps
    except Exception as e:
        print(f"✗ Structured reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show what was returned
        if client.reasoning_log:
            print("\n--- PROBLEMATIC RESPONSE ---")
            print(client.reasoning_log.response)
        
        return None


def test_roi_extraction(client: Qwen3VLClient, image: Image.Image):
    """Test 4: Check ROI extraction output."""
    print_section("TEST 4: ROI Extraction")
    
    from corgi.types import ReasoningStep
    
    # Create a test step that needs vision
    test_step = ReasoningStep(
        index=1,
        statement="Count the number of small cakes visible on the table",
        needs_vision=True,
        reason="Visual counting required"
    )
    
    question = "How many cakes are in the image?"
    max_regions = 3
    
    try:
        print(f"Testing ROI extraction for step: {test_step.statement}")
        print(f"Max regions: {max_regions}\n")
        
        evidences = client.extract_step_evidence(
            image=image,
            question=question,
            step=test_step,
            max_regions=max_regions
        )
        
        print(f"✓ Got {len(evidences)} evidence regions\n")
        
        if client.grounding_logs:
            last_log = client.grounding_logs[-1]
            print("--- RAW PROMPT ---")
            print(last_log.prompt[:500] + "..." if len(last_log.prompt) > 500 else last_log.prompt)
            print("\n--- RAW RESPONSE ---")
            print(last_log.response)
            print("\n--- PARSED EVIDENCE ---")
        
        for i, ev in enumerate(evidences, 1):
            print(f"  Evidence {i}:")
            print(f"    BBox: {ev.bbox}")
            print(f"    Description: {ev.description}")
            print(f"    Confidence: {ev.confidence}")
            print()
        
        return evidences
    except Exception as e:
        print(f"✗ ROI extraction failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to show what was returned
        if client.grounding_logs:
            print("\n--- PROBLEMATIC RESPONSE ---")
            print(client.grounding_logs[-1].response)
        
        return None


def test_answer_synthesis(client: Qwen3VLClient, image: Image.Image):
    """Test 5: Check answer synthesis."""
    print_section("TEST 5: Answer Synthesis")
    
    from corgi.types import ReasoningStep, GroundedEvidence
    
    # Mock some steps and evidence
    steps = [
        ReasoningStep(index=1, statement="Identify cakes in the image", needs_vision=True),
        ReasoningStep(index=2, statement="Count the total number", needs_vision=False),
    ]
    
    evidences = [
        GroundedEvidence(step_index=1, bbox=(0.1, 0.1, 0.3, 0.3), description="cake 1"),
        GroundedEvidence(step_index=1, bbox=(0.5, 0.5, 0.7, 0.7), description="cake 2"),
    ]
    
    question = "How many cakes are visible?"
    
    try:
        print(f"Question: {question}")
        print(f"Steps: {len(steps)}, Evidence: {len(evidences)}\n")
        
        answer = client.synthesize_answer(
            image=image,
            question=question,
            steps=steps,
            evidences=evidences
        )
        
        print(f"✓ Got answer\n")
        
        if client.answer_log:
            print("--- RAW PROMPT ---")
            print(client.answer_log.prompt[:500] + "..." if len(client.answer_log.prompt) > 500 else client.answer_log.prompt)
            print("\n--- RAW RESPONSE ---")
            print(client.answer_log.response)
            print("\n--- FINAL ANSWER ---")
            print(answer)
        
        return answer
    except Exception as e:
        print(f"✗ Answer synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        
        if client.answer_log:
            print("\n--- PROBLEMATIC RESPONSE ---")
            print(client.answer_log.response)
        
        return None


def test_full_pipeline(client: Qwen3VLClient, image: Image.Image):
    """Test 6: Run full pipeline."""
    print_section("TEST 6: Full Pipeline")
    
    question = "How many small cakes are on the table?"
    max_steps = 3
    max_regions = 2
    
    try:
        print(f"Question: {question}")
        print(f"Max steps: {max_steps}, Max regions: {max_regions}\n")
        
        pipeline = CoRGIPipeline(vlm_client=client)
        result = pipeline.run(
            image=image,
            question=question,
            max_steps=max_steps,
            max_regions=max_regions
        )
        
        print("✓ Pipeline completed!\n")
        
        print(f"Steps: {len(result.steps)}")
        for step in result.steps:
            print(f"  [{step.index}] {step.statement} (vision: {step.needs_vision})")
        
        print(f"\nEvidence: {len(result.evidence)}")
        for ev in result.evidence:
            print(f"  Step {ev.step_index}: {ev.bbox} - {ev.description}")
        
        print(f"\nAnswer: {result.answer}")
        
        print(f"\nTiming:")
        for timing in result.timings:
            print(f"  {timing.name}: {timing.duration_ms:.2f}ms")
        
        return result
    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("=" * 80)
    print(" CoRGI Component Testing with Real Qwen3-VL Model")
    print("=" * 80)
    
    # Test 1: Load model
    client = test_model_load()
    if client is None:
        print("\n❌ Cannot proceed without model. Exiting.")
        return 1
    
    # Test 2: Basic inference
    image = test_basic_inference(client)
    if image is None:
        print("\n❌ Cannot proceed without working inference. Exiting.")
        return 1
    
    # Test 3: Structured reasoning
    steps = test_structured_reasoning(client, image)
    
    # Test 4: ROI extraction
    evidences = test_roi_extraction(client, image)
    
    # Test 5: Answer synthesis
    answer = test_answer_synthesis(client, image)
    
    # Test 6: Full pipeline
    result = test_full_pipeline(client, image)
    
    print_section("TEST SUMMARY")
    tests = [
        ("Model Loading", client is not None),
        ("Basic Inference", image is not None),
        ("Structured Reasoning", steps is not None and len(steps) > 0),
        ("ROI Extraction", evidences is not None),
        ("Answer Synthesis", answer is not None),
        ("Full Pipeline", result is not None),
    ]
    
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in tests)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

