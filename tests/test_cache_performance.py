"""
Quick test to verify image encoding cache is working.

This test creates a simple Qwen client and makes multiple calls with the same image
to verify that the cache is being used (hit rate should be >0%).
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
from corgi.core.config import ModelConfig
from corgi.models.qwen.qwen_instruct_client import Qwen3VLInstructClient


def create_test_image():
    """Create a simple test image."""
    # Create a 512x512 RGB image with random content
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_image_cache():
    """Test that image encoding cache is working."""
    print("=" * 60)
    print("Image Encoding Cache Verification Test")
    print("=" * 60)

    # Create a minimal config
    config = ModelConfig(
        model_type="qwen_instruct",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda:0",
        enable_compile=False,  # Disable for faster testing
    )

    print(f"\n1. Creating Qwen client...")
    client = Qwen3VLInstructClient(config)

    print(f"2. Creating test image...")
    test_image = create_test_image()

    print(f"3. Making first call (should be cache MISS)...")
    response1 = client._chat(
        image=test_image,
        prompt="What do you see?",
        max_new_tokens=50,
    )
    print(f"   Response: {response1[:50]}...")
    print(f"   Cache stats: {client._cache_hits} hits, {client._cache_misses} misses")

    print(f"\n4. Making second call with SAME image (should be cache HIT)...")
    response2 = client._chat(
        image=test_image,
        prompt="Describe this image.",
        max_new_tokens=50,
    )
    print(f"   Response: {response2[:50]}...")
    print(f"   Cache stats: {client._cache_hits} hits, {client._cache_misses} misses")

    print(f"\n5. Making third call with SAME image (should be cache HIT)...")
    response3 = client._chat(
        image=test_image,
        prompt="What colors are present?",
        max_new_tokens=50,
    )
    print(f"   Response: {response3[:50]}...")
    print(f"   Cache stats: {client._cache_hits} hits, {client._cache_misses} misses")

    # Calculate hit rate
    total_calls = client._cache_hits + client._cache_misses
    hit_rate = (client._cache_hits / total_calls * 100) if total_calls > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Total calls: {total_calls}")
    print(f"Cache hits: {client._cache_hits}")
    print(f"Cache misses: {client._cache_misses}")
    print(f"Hit rate: {hit_rate:.1f}%")

    # Verify cache is working
    if client._cache_hits >= 2:
        print("\n✅ SUCCESS: Image encoding cache is working!")
        print("   Expected: 1 miss (first call), 2 hits (second & third calls)")
        return True
    else:
        print("\n❌ FAILED: Cache not working as expected")
        print("   Expected at least 2 cache hits, got", client._cache_hits)
        return False


if __name__ == "__main__":
    try:
        success = test_image_cache()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
