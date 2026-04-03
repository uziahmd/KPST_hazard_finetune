#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen/Qwen3.5-9B video SFT on local conversational JSONL
16-bit LoRA version for forklift hazard detection.

What this script does:
- loads local train/test JSONL files
- keeps raw local video clip paths
- builds multimodal chat inputs on the fly
- trains only on the assistant JSON response
- uses TRL SFTTrainer + PEFT LoRA
- evaluates on test set
- saves:
    1) LoRA adapter
    2) processor
    3) merged full model as qwen35_9B_forklift1
- runs one test inference example at the end

Expected JSONL structure per row:
{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "vlm_dataset/clips/test.mp4"},
        {"type": "text", "text": "Task prompt ..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{\"hazard_label\":\"...\", ... }"}
      ]
    }
  ],
  "meta": {...}
}

Recommended install:
pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
pip install -U trl peft accelerate datasets torch torchvision pillow torchcodec

Example:
python qwen_sft.py \
  --train_file vlm_dataset/train_chat.jsonl \
  --test_file vlm_dataset/test_chat.jsonl \
  --project_root . \
  --output_dir runs/qwen35_9B_forklift1 \
  --merged_model_dir qwen35_9B_forklift1 \
  --num_frames 12 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype float16
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen/Qwen3.5-9B on local raw-video JSONL with 16-bit LoRA")

    # model / paths
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train_file", type=str, default="vlm_dataset/train_chat.jsonl")
    parser.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="runs/qwen35_9B_forklift1")
    parser.add_argument("--merged_model_dir", type=str, default="qwen35_9B_forklift1")
    parser.add_argument("--adapter_subdir", type=str, default="adapter")

    # video sampling: use exactly one
    parser.add_argument("--num_frames", type=int, default=12, help="Uniform number of sampled frames per video")
    parser.add_argument("--fps", type=float, default=None, help="Use fps-based sampling instead of num_frames")

    # training
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # precision / memory
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])

    # lora
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules",
    )

    # inference
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)

    # misc
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()

    if args.num_frames is not None and args.fps is not None:
        raise ValueError("Use only one of --num_frames or --fps")

    return args


def get_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def is_remote_path(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resolve_local_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Absolute path does not exist: {path}")
        return path

    candidate = os.path.abspath(os.path.join(project_root, path))
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve media path: {path}\n"
        f"Tried relative to project_root={project_root} and current working directory."
    )


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join([t for t in texts if t]).strip()

    raise TypeError(f"Unsupported content type for text extraction: {type(content)}")


def normalize_content_blocks(content: Any, project_root: str) -> List[Dict[str, Any]]:
    """
    Converts user JSONL content blocks into processor-ready multimodal chat blocks.

    Supported input forms:
      {"type":"video","video":"..."}
      {"type":"video","path":"..."}
      {"type":"video","url":"..."}
      {"type":"text","text":"..."}
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise TypeError(f"Expected content to be a list or string, got {type(content)}")

    out = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Each content block must be a dict, got {type(block)}")

        block_type = block.get("type")

        if block_type == "text":
            out.append({"type": "text", "text": block.get("text", "")})
            continue

        if block_type == "video":
            raw_video = (
                block.get("path")
                or block.get("video")
                or block.get("url")
                or (block.get("video_url", {}) if isinstance(block.get("video_url"), dict) else {}).get("url")
            )
            if not raw_video:
                raise ValueError(f"Video block missing path/video/url: {block}")

            if is_remote_path(raw_video):
                out.append({"type": "video", "url": raw_video})
            else:
                out.append({"type": "video", "path": resolve_local_path(raw_video, project_root)})
            continue

        raise ValueError(f"Unsupported content block type: {block_type}")

    return out


def normalize_messages(messages: List[Dict[str, Any]], project_root: str) -> List[Dict[str, Any]]:
    normalized = []
    for msg in messages:
        normalized.append(
            {
                "role": msg["role"],
                "content": normalize_content_blocks(msg["content"], project_root),
            }
        )
    return normalized


def first_video_path(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        for block in msg.get("content", []):
            if block.get("type") == "video":
                return block.get("path") or block.get("url")
    return None


def first_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return extract_text_from_content(msg.get("content"))
    raise ValueError("No user message found")


def last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = extract_text_from_content(msg.get("content"))
            # validate target is JSON
            json.loads(text)
            return text
    raise ValueError("No assistant message found")


def enrich_record(example: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    normalized = normalize_messages(example["messages"], project_root)

    return {
        "messages": normalized,
        "video_path": first_video_path(normalized),
        "prompt_text": first_user_text(normalized),
        "assistant_text": last_assistant_text(normalized),
    }


def load_local_jsonl_dataset(jsonl_path: str, project_root: str):
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.map(lambda ex: enrich_record(ex, project_root))
    return ds


class VideoJSONCollator:
    """
    Builds full multimodal inputs and masks loss so only the assistant JSON is supervised.
    """

    def __init__(
        self,
        processor,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.fps = fps

        if self.num_frames is not None and self.fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive")

    def _videos_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        if self.num_frames is not None:
            kwargs["num_frames"] = self.num_frames
        elif self.fps is not None:
            kwargs["fps"] = self.fps

        return kwargs

    def _template_kwargs(self, padding: bool = True) -> Dict[str, Any]:
        return {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": padding,
            "processor_kwargs": {
                "videos_kwargs": self._videos_kwargs(),
            },
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_conversations = [ex["messages"] for ex in examples]

        prompt_only_conversations = []
        for ex in examples:
            messages = ex["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError("Each example must end with one assistant message")
            prompt_only_conversations.append(messages[:-1])

        full_batch = self.processor.apply_chat_template(
            full_conversations,
            add_generation_prompt=False,
            **self._template_kwargs(padding=True),
        )

        prompt_batch = self.processor.apply_chat_template(
            prompt_only_conversations,
            add_generation_prompt=True,
            **self._template_kwargs(padding=True),
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, plen in enumerate(prompt_lengths):
            labels[row_idx, :plen] = -100

        full_batch["labels"] = labels
        return full_batch


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def load_model_and_processor(args: argparse.Namespace):
    torch_dtype = get_torch_dtype(args.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)

    return model, processor, torch_dtype


def move_batch_to_model_device(batch: Dict[str, Any], model) -> Dict[str, Any]:
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device

    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if torch.is_tensor(v) else v
    return moved


@torch.inference_mode()
def generate_one_example(model, processor, example: Dict[str, Any], args: argparse.Namespace) -> str:
    prompt_messages = example["messages"][:-1]

    videos_kwargs = {}
    if args.num_frames is not None:
        videos_kwargs["num_frames"] = args.num_frames
    elif args.fps is not None:
        videos_kwargs["fps"] = args.fps

    inputs = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"videos_kwargs": videos_kwargs},
    )

    inputs = move_batch_to_model_device(inputs, model)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=False,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]

    text = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_merged_model(
    base_model_name: str,
    adapter_dir: str,
    merged_model_dir: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
    attn_implementation: str = "sdpa",
) -> None:
    """
    Reload base model on CPU, attach adapter, merge, and save.
    This avoids needing extra VRAM for the merge step.
    """
    print("\nReloading base model on CPU for merge...")

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )

    merged = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = merged.merge_and_unload()

    os.makedirs(merged_model_dir, exist_ok=True)
    merged.save_pretrained(
        merged_model_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    print(f"Merged model saved to: {merged_model_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = load_local_jsonl_dataset(args.train_file, args.project_root)
    test_dataset = load_local_jsonl_dataset(args.test_file, args.project_root)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    print("\nLoading model and processor...")
    model, processor, torch_dtype = load_model_and_processor(args)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    collator = VideoJSONCollator(
        processor=processor,
        num_frames=args.num_frames,
        fps=args.fps,
    )

    bf16 = args.torch_dtype == "bfloat16"
    fp16 = args.torch_dtype == "float16"

    train_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=False,
        dataloader_num_workers=args.dataloader_num_workers,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    print("\nStarting training...")
    train_result = trainer.train()
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    print("\nTrain metrics:")
    print(json.dumps(train_metrics, indent=2, ensure_ascii=False))
    save_json(train_metrics, os.path.join(args.output_dir, "train_metrics.json"))

    print("\nRunning evaluation...")
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))
    save_json(eval_metrics, os.path.join(args.output_dir, "eval_metrics.json"))

    adapter_dir = os.path.join(args.output_dir, args.adapter_subdir)
    os.makedirs(adapter_dir, exist_ok=True)

    print(f"\nSaving LoRA adapter to: {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    sample_index = max(0, min(args.sample_index, len(test_dataset) - 1))
    sample = test_dataset[sample_index]

    print("\n=== Inference example on one test clip ===")
    print(f"Sample index: {sample_index}")
    print(f"Video path:   {sample['video_path']}")
    print(f"Prompt text:\n{sample['prompt_text']}\n")
    print(f"Target JSON:\n{sample['assistant_text']}\n")

    generated = generate_one_example(trainer.model, processor, sample, args)

    print("Generated output:")
    print(generated)

    with open(os.path.join(args.output_dir, "sample_generation.txt"), "w", encoding="utf-8") as f:
        f.write(generated + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nSaving merged model...")
    try:
        save_merged_model(
            base_model_name=args.model_name,
            adapter_dir=adapter_dir,
            merged_model_dir=args.merged_model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        processor.save_pretrained(args.merged_model_dir)
    except Exception as e:
        print(
            "\nWARNING: Adapter was saved successfully, but merged model export failed.\n"
            f"Reason: {e}\n"
            "This usually means CPU RAM was insufficient for reloading and merging the full base model."
        )

    print("\nDone.")
    print(f"Adapter saved at: {adapter_dir}")
    print(f"Merged model target: {args.merged_model_dir}")


if __name__ == "__main__":
    main()