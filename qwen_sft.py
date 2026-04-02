#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune Qwen/Qwen3.5-9B on a local raw-video conversational JSONL dataset
using TRL SFTTrainer + QLoRA/LoRA.

Expected JSONL example structure:
{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "vlm_dataset/clips/test.mp4"},
        {"type": "text", "text": "Task prompt here..."}
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
  pip install -U trl peft accelerate bitsandbytes datasets torch torchvision pillow torchcodec

Example run:
  python train_qwen_sft.py \
    --train_file vlm_dataset/train_chat.jsonl \
    --test_file vlm_dataset/test_chat.jsonl \
    --project_root . \
    --output_dir runs/qwen35_9B_forklift1 \
    --merged_model_dir qwen35_9B_forklift1 \
    --num_frames 12 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

try:
    # Preferred explicit class for latest Transformers main.
    from transformers import Qwen3_5ForConditionalGeneration

    MODEL_CLASS = Qwen3_5ForConditionalGeneration
except Exception:
    # Fallback for environments where the explicit class import name differs.
    from transformers import AutoModelForImageTextToText

    MODEL_CLASS = AutoModelForImageTextToText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Qwen3.5-9B on raw video JSONL")

    # Paths
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train_file", type=str, default="vlm_dataset/train_chat.jsonl")
    parser.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="runs/qwen35_9B_forklift1")
    parser.add_argument("--merged_model_dir", type=str, default="qwen35_9B_forklift1")

    # Data / video processing
    parser.add_argument("--num_frames", type=int, default=12, help="Uniformly sampled frames per video.")
    parser.add_argument("--fps", type=float, default=None, help="Use fps-based sampling instead of num_frames.")
    parser.add_argument("--disable_thinking", action="store_true", default=True)
    parser.add_argument("--sample_index", type=int, default=0, help="Test example index for final generation demo.")

    # Training
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Precision / memory
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA.")
    parser.add_argument("--no_use_qlora", dest="use_qlora", action="store_false", help="Disable QLoRA and use 16-bit LoRA.")
    parser.set_defaults(use_qlora=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names.",
    )

    # Inference demo
    parser.add_argument("--gen_max_new_tokens", type=int, default=256)

    # Misc
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    args = parser.parse_args()

    if args.num_frames is not None and args.fps is not None:
        raise ValueError("Use only one of --num_frames or --fps, not both.")

    return args


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def is_remote_path(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resolve_local_path(path: str, project_root: str) -> str:
    if os.path.isabs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Absolute media path does not exist: {path}")
        return path

    candidate = os.path.abspath(os.path.join(project_root, path))
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve local media path: {path}\n"
        f"Tried relative to project_root={project_root} and current working directory."
    )


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join([p for p in parts if p]).strip()

    raise TypeError(f"Unsupported content type for text extraction: {type(content)}")


def normalize_content_blocks(content: Any, project_root: str) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        raise TypeError(f"Expected content to be a list or string, got {type(content)}")

    normalized = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Each content block must be a dict, got {type(block)}")

        block_type = block.get("type")
        if block_type == "text":
            normalized.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "video":
            # User dataset uses {"type":"video","video":"..."}.
            raw_video = (
                block.get("path")
                or block.get("video")
                or block.get("url")
                or (block.get("video_url", {}) if isinstance(block.get("video_url"), dict) else {})
                   .get("url")
            )
            if not raw_video:
                raise ValueError(f"Video block missing path/video/url: {block}")

            if is_remote_path(raw_video):
                normalized.append({"type": "video", "url": raw_video})
            else:
                resolved = resolve_local_path(raw_video, project_root)
                normalized.append({"type": "video", "path": resolved})

        else:
            raise ValueError(f"Unsupported content block type: {block_type}")

    return normalized


def normalize_messages(messages: List[Dict[str, Any]], project_root: str) -> List[Dict[str, Any]]:
    normalized = []
    for msg in messages:
        role = msg["role"]
        content = normalize_content_blocks(msg["content"], project_root)
        normalized.append({"role": role, "content": content})
    return normalized


def first_video_path(messages: List[Dict[str, Any]]) -> Optional[str]:
    for msg in messages:
        for block in msg.get("content", []):
            if block.get("type") == "video":
                return block.get("path") or block.get("url")
    return None


def first_user_prompt_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return text_from_content(msg.get("content"))
    raise ValueError("No user message found.")


def last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = text_from_content(msg.get("content"))
            # Fail fast if targets are not valid JSON strings.
            json.loads(text)
            return text
    raise ValueError("No assistant message found.")


def enrich_record(example: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    normalized = normalize_messages(example["messages"], project_root)
    return {
        "messages": normalized,
        "video_path": first_video_path(normalized),
        "prompt_text": first_user_prompt_text(normalized),
        "assistant_text": last_assistant_text(normalized),
    }


class VideoJSONResponseCollator:
    """
    Keeps the script simple and correct:
    - full conversation -> model inputs + labels base
    - prompt-only conversation -> prompt length
    - mask everything before the assistant response
    """

    def __init__(
        self,
        processor,
        num_frames: Optional[int] = 12,
        fps: Optional[float] = None,
        disable_thinking: bool = True,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.fps = fps
        self.disable_thinking = disable_thinking

    def _common_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.num_frames is not None:
            kwargs["num_frames"] = self.num_frames
        if self.fps is not None:
            kwargs["fps"] = self.fps
        if self.disable_thinking:
            kwargs["enable_thinking"] = False
        return kwargs

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_conversations = [ex["messages"] for ex in examples]
        prompt_only_conversations = []

        for ex in examples:
            messages = ex["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                raise ValueError("Expected each sample to end with an assistant message.")
            prompt_only_conversations.append(messages[:-1])

        full_batch = self.processor.apply_chat_template(
            full_conversations,
            add_generation_prompt=False,
            **self._common_kwargs(),
        )
        prompt_batch = self.processor.apply_chat_template(
            prompt_only_conversations,
            add_generation_prompt=True,
            **self._common_kwargs(),
        )

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, :prompt_len] = -100

        full_batch["labels"] = labels
        return full_batch


def load_local_dataset(jsonl_path: str, project_root: str):
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.map(lambda ex: enrich_record(ex, project_root))
    return ds


def build_quant_config(args: argparse.Namespace, torch_dtype: torch.dtype):
    if not args.use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )


def load_model_and_processor(args: argparse.Namespace):
    torch_dtype = get_torch_dtype(args.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    quant_config = build_quant_config(args, torch_dtype)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = MODEL_CLASS.from_pretrained(args.model_name, **model_kwargs)

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    return model, processor, torch_dtype


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )


def move_batch_to_model_device(batch: Dict[str, Any], model) -> Dict[str, Any]:
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device

    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


@torch.inference_mode()
def generate_one_example(model, processor, example: Dict[str, Any], args: argparse.Namespace) -> str:
    prompt_messages = example["messages"][:-1]

    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames
    if args.fps is not None:
        kwargs["fps"] = args.fps
    if args.disable_thinking:
        kwargs["enable_thinking"] = False

    inputs = processor.apply_chat_template(prompt_messages, **kwargs)
    inputs = move_batch_to_model_device(inputs, model)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

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


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_merged_model(
    base_model_name: str,
    adapter_dir: str,
    merged_dir: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
) -> None:
    print(f"\nReloading base model for merge: {base_model_name}")
    base_model = MODEL_CLASS.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    print(f"Merged model saved to: {merged_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = load_local_dataset(args.train_file, args.project_root)
    test_dataset = load_local_dataset(args.test_file, args.project_root)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    print("\nLoading model and processor...")
    model, processor, torch_dtype = load_model_and_processor(args)
    peft_config = build_peft_config(args)

    collator = VideoJSONResponseCollator(
        processor=processor,
        num_frames=args.num_frames,
        fps=args.fps,
        disable_thinking=args.disable_thinking,
    )

    bf16 = args.torch_dtype == "bfloat16"
    fp16 = args.torch_dtype == "float16"

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=False,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        processing_class=processor,
        peft_config=peft_config,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    print("\nStarting training...")
    train_result = trainer.train()
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    print("\nTrain metrics:")
    print(json.dumps(train_metrics, indent=2))
    save_metrics(train_metrics, os.path.join(args.output_dir, "train_metrics.json"))

    print("\nRunning evaluation on test set...")
    eval_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(json.dumps(eval_metrics, indent=2))
    save_metrics(eval_metrics, os.path.join(args.output_dir, "eval_metrics.json"))

    adapter_dir = os.path.join(args.output_dir, "adapter")
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

    generated = generate_one_example(trainer.model, processor, sample, args)
    print("Generated assistant JSON:")
    print(generated)

    prediction_path = os.path.join(args.output_dir, "sample_generation.txt")
    with open(prediction_path, "w", encoding="utf-8") as f:
        f.write(generated + "\n")

    # Free some memory before merge.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nSaving merged model...")
    try:
        save_merged_model(
            base_model_name=args.model_name,
            adapter_dir=adapter_dir,
            merged_dir=args.merged_model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        processor.save_pretrained(args.merged_model_dir)
    except Exception as exc:
        print(
            "\nWARNING: Adapter was saved successfully, but merged-model export failed.\n"
            f"Reason: {exc}\n"
            "This usually means the machine did not have enough RAM/VRAM for reloading the full base model."
        )

    print("\nDone.")
    print(f"Adapter dir:      {adapter_dir}")
    print(f"Merged model dir: {args.merged_model_dir}")


if __name__ == "__main__":
    main()