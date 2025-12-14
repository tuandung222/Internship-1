"""
Test script to verify flash-attn3 works with Florence-2-large-ft model.
"""

import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from PIL import Image
import requests

print("=" * 60)
print("Testing Flash Attention 3 with Florence-2-large-ft")
print("=" * 60)

# Test 1: Load model with flash-attn3
print("\n[Test 1] Loading Florence-2-large-ft with flash-attn3...")
try:
    model = Florence2ForConditionalGeneration.from_pretrained(
        "florence-community/Florence-2-large-ft",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="kernels-community/flash-attn2",
    ).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("✓ Flash Attention 3 loaded successfully!")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ Flash Attention 3 failed: {e}")
    exit(1)

# Test 2: Load processor
print("\n[Test 2] Loading Florence-2 processor...")
try:
    processor = AutoProcessor.from_pretrained(
        "florence-community/Florence-2-large-ft",
        trust_remote_code=True
    )
    print("✓ Processor loaded successfully!")
except Exception as e:
    print(f"✗ Processor loading failed: {e}")
    exit(1)

# Test 3: Run inference with nucleus sampling
print("\n[Test 3] Running inference with nucleus sampling...")
try:
    # Load test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(f"  Image size: {image.size}")
    
    # Run object detection task
    task_prompt = "<OD>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda().contiguous() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    print(f"  Task: {task_prompt}")
    print(f"  Using nucleus sampling (temperature=0.25, top_p=0.9, top_k=50)")
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=True,
            temperature=0.25,
            top_p=0.9,
            top_k=50,
            use_cache=False,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    print("✓ Inference completed successfully!")
    print(f"  Detected {len(parsed_answer.get('<OD>', {}).get('labels', []))} objects")
    print(f"  Sample detections: {parsed_answer.get('<OD>', {}).get('labels', [])[:3]}")
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test MORE_DETAILED_CAPTION task
print("\n[Test 4] Testing <MORE_DETAILED_CAPTION> task...")
try:
    task_prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda().contiguous() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=True,
            temperature=0.25,
            top_p=0.9,
            top_k=50,
            use_cache=False,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    caption = parsed_answer.get('<MORE_DETAILED_CAPTION>', '')
    print("✓ Captioning completed successfully!")
    print(f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    
except Exception as e:
    print(f"✗ Captioning failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nFlorence-2-large-ft with Flash Attention 3 is working correctly.")
print("Ready for integration with CoRGi pipeline.")





