#!/usr/bin/env python3
"""
Gemma 4 E4B-it LoRA fine-tuning on a raw-video chat JSONL dataset.

Designed to mirror the user's Qwen 3.5 raw-video training script while
keeping Gemma 4 on its native multimodal path:
- Same JSONL chat format
- Same assistant-only loss masking
- Same raw video paths inside messages
- Same Trainer-style loop
- Gemma 4 video loading via AutoModelForMultimodalLM + AutoProcessor

Expected row format:
{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "vlm_dataset_both_aug/clips/...mp4"},
        {"type": "text", "text": "...prompt..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{\"hazard_label\": ... }"}
      ]
    }
  ],
  "meta": {...}
}

Example run:
python train_gemma4.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file   vlm_dataset_both_aug/val_chat.jsonl \
  --test_file  vlm_dataset_both_aug/test_chat.jsonl \
  --project_root hazard_finetuning \
  --model_name_or_path google/gemma-4-E4B-it \
  --output_dir runs/gemma4_e4b_video_lora \
  --video_load_backend pyav \
  --fps 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --gradient_checkpointing \
  --use_fp16
"""

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv 
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Load variables from .env into os.environ
load_dotenv(override=True)

# Verification (Optional: remove this in production)
if os.getenv("HF_TOKEN"):
    print("HF_TOKEN successfully loaded from .env")
else:
    print("Warning: HF_TOKEN not found in .env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Gemma 4 E4B-it on raw-video chat JSONL.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default="")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--output_dir", type=str, default="runs/gemma4_e4b_video_lora")

    parser.add_argument("--num_frames", type=int, default=12,
                        help="Requested frame count for video processing if the processor version supports it.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Optional target frames-per-second for processor-side video sampling.")
    parser.add_argument(
        "--video_load_backend",
        type=str,
        default="pyav",
        choices=["pyav", "decord", "opencv", "torchvision"],
        help="Video decoder backend passed to processor.apply_chat_template. "
             "Use this to avoid the default torchcodec path when it is unavailable.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Used only for the post-training demo inference.")
    parser.add_argument("--inference_index", type=int, default=0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help='Use "all-linear" to match the Gemma sample more closely, or pass a comma-separated list.',
    )
    parser.add_argument(
        "--modules_to_save",
        type=str,
        default="lm_head,embed_tokens",
        help='Comma-separated PEFT modules_to_save. Use "" to disable.',
    )

    parser.add_argument("--report_to", type=str, default="none")
    return parser.parse_args()


def select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def is_probably_url(path_str: str) -> bool:
    parsed = urlparse(path_str)
    return parsed.scheme in {"http", "https", "s3", "gs"}


def resolve_local_media_path(raw_value: str, project_root: Path) -> Path:
    p = Path(raw_value)
    if p.is_absolute():
        return p

    candidates = [
        project_root / p,
        Path.cwd() / p,
        p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fall back to the project-root-relative location for a deterministic path.
    return (project_root / p).resolve()


def absolutize_multimodal_paths(messages: List[Dict[str, Any]], project_root: Path) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(messages)

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            # Normalize repo-local shorthand into the multimodal HF schema:
            # {"type": "video", "video": "..."} -> {"type": "video", "path": "..."}
            # {"type": "image", "image": "..."} -> {"type": "image", "path": "..."}
            if item.get("type") == "video":
                raw_value = item.get("path", item.get("video"))
                if isinstance(raw_value, str) and raw_value:
                    if is_probably_url(raw_value):
                        item["url"] = raw_value
                        item.pop("path", None)
                    else:
                        item["path"] = str(resolve_local_media_path(raw_value, project_root))
                        item.pop("url", None)
                item.pop("video", None)

            if item.get("type") == "image":
                raw_value = item.get("path", item.get("image"))
                if isinstance(raw_value, str) and raw_value:
                    if is_probably_url(raw_value):
                        item["url"] = raw_value
                        item.pop("path", None)
                    else:
                        item["path"] = str(resolve_local_media_path(raw_value, project_root))
                        item.pop("url", None)
                item.pop("image", None)

    return messages


def extract_assistant_text(message: Dict[str, Any]) -> str:
    if message["role"] != "assistant":
        raise ValueError("Expected an assistant message.")
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts).strip()


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def apply_chat_template_gemma(
    processor,
    messages,
    *,
    add_generation_prompt: bool,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    video_load_backend: Optional[str] = None,
):
    """
    Use Gemma's native multimodal chat template path for raw-video messages.
    We try a few compatible call signatures because processor support can vary a bit by version.
    """
    base_kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )

    attempts = []

    # Most feature-rich attempt
    processor_kwargs = {}
    if num_frames is not None:
        processor_kwargs["num_frames"] = num_frames
    if fps is not None:
        processor_kwargs["fps"] = fps
    if video_load_backend is not None:
        processor_kwargs["video_load_backend"] = video_load_backend

    if processor_kwargs:
        attempts.append({**base_kwargs, "enable_thinking": False, "processor_kwargs": processor_kwargs})
        attempts.append({**base_kwargs, "processor_kwargs": processor_kwargs})

    attempts.append({**base_kwargs, "enable_thinking": False})
    attempts.append(base_kwargs)

    last_err = None
    for kwargs in attempts:
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except TypeError as e:
            last_err = e
        except ValueError as e:
            # keep trying a simpler signature
            last_err = e

    raise last_err


class RawVideoGemmaCollator:
    """
    Assistant-only loss masking, same idea as the Qwen script.

    This collator intentionally assumes per_device_*_batch_size == 1.
    That is the safest setup for raw-video Gemma training on a V100.
    Use gradient accumulation for larger effective batch size.
    """

    def __init__(
        self,
        processor,
        project_root: str,
        num_frames: int,
        fps: Optional[float],
        video_load_backend: Optional[str],
    ):
        self.processor = processor
        self.project_root = Path(project_root)
        self.num_frames = num_frames
        self.fps = fps
        self.video_load_backend = video_load_backend

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                "This raw-video Gemma collator assumes batch size 1. "
                "Please keep per_device_train_batch_size=1 and use gradient_accumulation_steps."
            )

        example = examples[0]
        messages = absolutize_multimodal_paths(example["messages"], self.project_root)

        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Each example must end with an assistant message.")

        full_messages = messages
        prompt_messages = messages[:-1]

        full_batch = apply_chat_template_gemma(
            self.processor,
            full_messages,
            add_generation_prompt=False,
            num_frames=self.num_frames,
            fps=self.fps,
            video_load_backend=self.video_load_backend,
        )
        prompt_batch = apply_chat_template_gemma(
            self.processor,
            prompt_messages,
            add_generation_prompt=True,
            num_frames=self.num_frames,
            fps=self.fps,
            video_load_backend=self.video_load_backend,
        )

        labels = full_batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        prompt_len = int(prompt_batch["attention_mask"].sum().item())
        labels[:, :prompt_len] = -100

        full_batch["labels"] = labels
        return full_batch


def load_datasets(train_file: str, val_file: str, test_file: str):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    data_files: Dict[str, str] = {
        "train": train_file,
        "test": test_file,
    }

    has_val = bool(val_file and os.path.exists(val_file))
    if has_val:
        data_files["val"] = val_file
    elif val_file:
        print(f"[WARN] Validation file not found, disabling eval: {val_file}")

    dataset = load_dataset("json", data_files=data_files)
    return dataset, has_val


def load_model_and_processor(args: argparse.Namespace, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    quantization_config = None
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["dtype"] = dtype

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules: Any = args.lora_target_modules
    if args.lora_target_modules != "all-linear":
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    modules_to_save = None
    if args.modules_to_save.strip():
        modules_to_save = [m.strip() for m in args.modules_to_save.split(",") if m.strip()]

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )

    # Some PEFT versions support ensure_weight_tying, some do not.
    try:
        lora_config = LoraConfig(ensure_weight_tying=True, **lora_kwargs)
    except TypeError:
        lora_config = LoraConfig(**lora_kwargs)

    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model, processor


def make_training_arguments(args: argparse.Namespace, dtype: torch.dtype) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters

    common_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": args.gradient_checkpointing,
        "label_names": ["labels"],
        "report_to": [] if args.report_to == "none" else args.report_to,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    }

    if args.save_strategy == "steps":
        common_kwargs["save_steps"] = args.save_steps

    if "eval_strategy" in sig:
        common_kwargs["eval_strategy"] = args.eval_strategy
    else:
        common_kwargs["evaluation_strategy"] = args.eval_strategy

    if args.eval_strategy == "steps":
        common_kwargs["eval_steps"] = args.eval_steps

    load_best = (
        args.eval_strategy != "no"
        and args.save_strategy != "no"
        and args.eval_strategy == args.save_strategy
    )
    common_kwargs["load_best_model_at_end"] = load_best
    if load_best:
        common_kwargs["metric_for_best_model"] = "eval_loss"
        common_kwargs["greater_is_better"] = False

    return TrainingArguments(**common_kwargs)


def run_one_inference_example(
    model,
    processor,
    example: Dict[str, Any],
    project_root: str,
    num_frames: int,
    fps: Optional[float],
    video_load_backend: Optional[str],
    max_new_tokens: int,
) -> None:
    model.eval()

    messages = absolutize_multimodal_paths(example["messages"], Path(project_root))
    prompt_messages = messages[:-1]
    ground_truth = extract_assistant_text(messages[-1])

    inputs = apply_chat_template_gemma(
        processor,
        prompt_messages,
        add_generation_prompt=True,
        num_frames=num_frames,
        fps=fps,
        video_load_backend=video_load_backend,
    )

    device = next(model.parameters()).device
    inputs = move_batch_to_device(inputs, device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, input_len:]
    prediction = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n=== Validation Example ===")
    print("sample_id   :", example.get("sample_id", "<missing>"))
    print("ground_truth:", ground_truth)
    print("prediction  :", prediction)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = select_dtype(args)

    dataset, has_val = load_datasets(args.train_file, args.val_file, args.test_file)

    print(f"Train samples: {len(dataset['train'])}")
    if has_val:
        print(f"Val samples  : {len(dataset['val'])}")
    print(f"Test samples : {len(dataset['test'])}")

    if args.per_device_train_batch_size != 1 or args.per_device_eval_batch_size != 1:
        raise ValueError(
            "This Gemma raw-video script currently assumes per_device_train_batch_size=1 "
            "and per_device_eval_batch_size=1."
        )

    if not args.load_in_4bit:
        print("[WARN] You are not using --load_in_4bit. On a V100 this may be too heavy for Gemma 4 E4B-it.")

    model, processor = load_model_and_processor(args, dtype)

    collator = RawVideoGemmaCollator(
        processor=processor,
        project_root=args.project_root,
        num_frames=args.num_frames,
        fps=args.fps,
        video_load_backend=args.video_load_backend,
    )

    # Disable eval if val file is absent
    if args.eval_strategy != "no" and not has_val:
        print("[WARN] eval_strategy requested but no valid val_file found. Disabling eval.")
        args.eval_strategy = "no"

    training_args = make_training_arguments(args, dtype)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=(dataset["val"] if has_val and args.eval_strategy != "no" else None),
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    demo_idx = max(0, min(args.inference_index, len(dataset["test"]) - 1))
    run_one_inference_example(
        model=trainer.model,
        processor=processor,
        example=dataset["test"][demo_idx],
        project_root=args.project_root,
        num_frames=args.num_frames,
        fps=args.fps,
        video_load_backend=args.video_load_backend,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
