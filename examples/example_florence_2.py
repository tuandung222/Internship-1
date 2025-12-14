import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration, BitsAndBytesConfig


model = Florence2ForConditionalGeneration.from_pretrained(
    "florence-community/Florence-2-large-ft",
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("florence-community/Florence-2-large-ft")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)

print(parsed_answer)
