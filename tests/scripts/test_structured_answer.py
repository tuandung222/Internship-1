#!/usr/bin/env python
"""Test structured answer with key evidence."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

from PIL import Image
from corgi.pipeline import CoRGIPipeline
from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig
from corgi.florence_client import Florence2Client

def test_structured_answer(use_florence=False):
    print("=" * 80)
    print("Testing Structured Answer with Key Evidence")
    print("=" * 80)
    
    # Load test image
    test_img_path = "/home/dungvpt/workspace/corgi_implementation/Qwen3-VL/cookbooks/assets/spatial_understanding/spatio_case1.jpg"
    image = Image.open(test_img_path).convert("RGB")
    print(f"\n✓ Loaded test image: {image.size}")
    
    # Question
    question = "How many small cakes are visible on the table?"
    print(f"Question: {question}\n")
    
    # Create pipeline with 4B-Thinking model
    config = QwenGenerationConfig()  # Uses 4B-Thinking by default now
    print(f"Model: {config.model_id}")
    
    # OPTIMIZATION: Use Florence-2 if requested
    florence_client = None
    if use_florence:
        print("Creating Florence-2 client for ROI extraction...")
        florence_client = Florence2Client(device="cuda:7")
        print("✓ Florence-2 ready")
    
    client = Qwen3VLClient(config, florence_client=florence_client)
    pipeline = CoRGIPipeline(vlm_client=client)
    
    if use_florence:
        print("✓ Pipeline configured with Florence-2 for ROI extraction")
    
    # Run pipeline
    print("\n⏳ Running pipeline...")
    try:
        result = pipeline.run(
            image=image,
            question=question,
            max_steps=3,
            max_regions=3
        )
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        # Reasoning steps
        print(f"\n📝 Reasoning Steps ({len(result.steps)}):")
        for step in result.steps:
            print(f"  [{step.index}] {step.statement}")
            print(f"       Needs vision: {step.needs_vision}")
        
        # Evidence
        print(f"\n🔍 Visual Evidence ({len(result.evidence)}):")
        for ev in result.evidence:
            print(f"  Step {ev.step_index}: {ev.bbox}")
            if ev.description:
                print(f"    → {ev.description}")
        
        # Answer with Key Evidence
        print(f"\n💡 Final Answer:")
        print(f"  {result.answer}")
        
        print(f"\n🎯 Key Evidence with Reasoning ({len(result.key_evidence)}):")
        if result.key_evidence:
            for i, ke in enumerate(result.key_evidence, 1):
                print(f"\n  Evidence {i}:")
                print(f"    BBox: {ke.bbox}")
                print(f"    Description: {ke.description}")
                print(f"    Reasoning: {ke.reasoning}")
        else:
            print("  (No structured key evidence returned)")
        
        # Timing
        print(f"\n⏱️  Performance:")
        for timing in result.timings:
            if timing.name == "total_pipeline":
                print(f"  Total: {timing.duration_ms/1000:.1f}s")
            else:
                print(f"  {timing.name}: {timing.duration_ms/1000:.1f}s")
        
        print("\n" + "=" * 80)
        print("✅ Test completed!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Show logs if available
        if hasattr(client, '_reasoning_log') and client._reasoning_log:
            print("\n--- Reasoning Log ---")
            print(f"Response length: {len(client._reasoning_log.response) if client._reasoning_log.response else 0}")
            if client._reasoning_log.response:
                print(f"First 500 chars: {client._reasoning_log.response[:500]}")
        
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-florence", action="store_true", help="Test with Florence-2 for ROI extraction")
    args = parser.parse_args()
    
    success = test_structured_answer(use_florence=args.use_florence)
    sys.exit(0 if success else 1)

