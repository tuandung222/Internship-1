#!/usr/bin/env python
"""Quick test to verify single GPU usage with 4B model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from corgi.qwen_client import Qwen3VLClient, QwenGenerationConfig

def test_single_gpu():
    print("=" * 80)
    print("Testing Single GPU with Qwen3-VL-2B-Instruct")
    print("=" * 80)
    
    # Create config with 2B model (default)
    config = QwenGenerationConfig()
    print(f"\n✓ Config created: {config.model_id}")
    
    # Load model
    print("\n⏳ Loading model...")
    client = Qwen3VLClient(config)
    print("✓ Model loaded!")
    
    # Check GPU usage and dtype
    import torch
    if torch.cuda.is_available():
        model_param = next(client._model.parameters())
        print(f"\n📊 GPU Status:")
        print(f"  - Device: {model_param.device}")
        print(f"  - Dtype: {model_param.dtype}")
        print(f"  - BFloat16 supported: {torch.cuda.is_bf16_supported()}")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  - Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Simple inference test
    test_img_path = "/home/dungvpt/workspace/corgi_implementation/Qwen3-VL/cookbooks/assets/spatial_understanding/spatio_case1.jpg"
    image = Image.open(test_img_path).convert("RGB")
    
    print(f"\n✓ Loaded test image: {image.size}")
    print("\n⏳ Running inference...")
    
    response = client._chat(
        image=image,
        prompt="Describe this image in one sentence.",
        max_new_tokens=100
    )
    
    print(f"\n✓ Inference complete!")
    print(f"\nResponse: {response[:200]}...")
    
    # Check GPU usage after inference
    if torch.cuda.is_available():
        print(f"\n📊 GPU Status after inference:")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  - Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Check if model is on single GPU and dtype
        devices = set()
        dtypes = set()
        for param in client._model.parameters():
            devices.add(str(param.device))
            dtypes.add(str(param.dtype))
        
        print(f"\n✅ Model devices: {devices}")
        print(f"✅ Model dtypes: {dtypes}")
        
        if len(devices) == 1 and 'cuda:0' in list(devices)[0]:
            print("✅ SUCCESS: Model is on single GPU (cuda:0)!")
        else:
            print(f"⚠️  WARNING: Model is on multiple devices: {devices}")
        
        if torch.cuda.is_bf16_supported() and 'bfloat16' in list(dtypes)[0]:
            print("✅ SUCCESS: Using bfloat16 as hardware supports it!")
        elif not torch.cuda.is_bf16_supported() and 'float16' in list(dtypes)[0]:
            print("✅ SUCCESS: Using float16 (bfloat16 not supported by hardware)")
        else:
            print(f"ℹ️  INFO: Using dtype: {list(dtypes)[0]}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_single_gpu()

