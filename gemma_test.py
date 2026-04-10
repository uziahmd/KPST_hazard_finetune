import json
from pathlib import Path
import torch
import os
from dotenv import load_dotenv  
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Load variables from .env into os.environ
load_dotenv(override=True)

# Verification (Optional: remove this in production)
if os.getenv("HF_TOKEN"):
    print("HF_TOKEN successfully loaded from .env")
else:
    print("Warning: HF_TOKEN not found in .env")

MODEL_ID = "google/gemma-4-26B-A4B-it"
JSONL_PATH = Path("vlm_dataset_both_aug/test_chat.jsonl")

# 1) Load the first sample from the JSONL
with JSONL_PATH.open("r", encoding="utf-8") as f:
    sample = json.loads(next(f))

sample_id = sample.get("sample_id", "unknown_sample")

# 2) Extract only the user message and rebuild it for inference
user_msg = next(m for m in sample["messages"] if m["role"] == "user")

rebuilt_content = []
resolved_video_path = None

for item in user_msg["content"]:
    if item["type"] == "video":
        raw_video_path = Path(item["video"])

        # Resolve relative path like "vlm_dataset_both_aug/..."
        if raw_video_path.is_absolute():
            resolved_video_path = raw_video_path
        else:
            resolved_video_path = (raw_video_path).resolve()

        if not resolved_video_path.exists():
            raise FileNotFoundError(
                f"Video file not found:\n"
                f"  raw path in JSONL: {raw_video_path}\n"
                f"  resolved path:     {resolved_video_path}"
            )

        rebuilt_content.append({
            "type": "video",
            "video": str(resolved_video_path),
        })

    elif item["type"] == "text":
        rebuilt_content.append({
            "type": "text",
            "text": item["text"],
        })

messages = [
    {
        "role": "user",
        "content": rebuilt_content
    }
]

print(f"Loaded sample_id: {sample_id}")
print(f"Resolved video:   {resolved_video_path}")

# 3) Load model and processor
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 4) Build inputs
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)

# Move tensors to model device
inputs = {
    k: (v.to(model.device) if hasattr(v, "to") else v)
    for k, v in inputs.items()
}

# 5) Run generation
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

# 6) Decode only newly generated tokens
input_len = inputs["input_ids"].shape[1]
generated_ids = outputs[:, input_len:]
response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n=== MODEL OUTPUT ===")
print(response)