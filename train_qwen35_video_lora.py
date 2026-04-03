#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen 3.5 9B on a raw-video chat dataset.

This script is the practical replacement for an image-based Unsloth example when
your dataset is already in chat JSONL format with raw video paths.

Important changes from the image example:
1) We load your existing JSONL files directly:
      - vlm_dataset/train_chat.jsonl
      - vlm_dataset/test_chat.jsonl
2) We keep the modality as RAW VIDEO.
3) We preserve each row's existing `messages` conversation structure.
4) We DO NOT silently swap to image-only training.
5) We DO NOT use UnslothVisionDataCollator here, because the current Qwen3.5
   vision path is not a clean native raw-video training path in this stack.
   Instead, we use the official processor + a custom raw-video collator.

Expected row format:
{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "path/to/clip.mp4"},
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

Suggested install base:
  pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"
  pip install -U datasets accelerate peft trl bitsandbytes

Depending on your environment, you may also need a video backend such as torchcodec.
"""

import argparse
import copy
import inspect
import os
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen 3.5 9B on raw-video chat JSONL.")
    parser.add_argument("--train_file", type=str, default="vlm_dataset/train_chat.jsonl")
    parser.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output_dir", type=str, default="runs/qwen35_9b_video_lora")

    parser.add_argument("--num_frames", type=int, default=12, help="Frames sampled per video clip by the processor.")
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length after chat templating.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Used only for the post-training demo inference.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true", help="Recommended for single-GPU LoRA.")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help='Use "all-linear" or a comma-separated list such as q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    )

    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--inference_index", type=int, default=0, help="Test sample index for the post-training demo.")
    return parser.parse_args()


def select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def extract_assistant_text(message: Dict[str, Any]) -> str:
    if message["role"] != "assistant":
        raise ValueError("Expected the last message to be from the assistant.")
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts).strip()


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def apply_chat_template_video_safe(
    processor,
    conversation,
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    return_dict: bool,
    return_tensors: Optional[str],
    enable_thinking: bool,
    num_frames: Optional[int],
    padding: bool = False,
):
    processor_kwargs = {}
    if num_frames is not None:
        # Pass video-specific args only through processor_kwargs.
        # Do not also pass fps here.
        processor_kwargs["num_frames"] = num_frames

    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_dict=return_dict,
        return_tensors=return_tensors,
        enable_thinking=False,
        processor_kwargs=processor_kwargs,
        padding=padding,
        truncation=False,   # critical fix
    )
    if max_length is not None:
        base_kwargs["max_length"] = max_length

    if num_frames is not None:
        try:
            return processor.apply_chat_template(
                conversation,
                processor_kwargs={"num_frames": num_frames, "fps": None},
                **base_kwargs,
            )
        except TypeError:
            return processor.apply_chat_template(
                conversation,
                num_frames=num_frames,
                fps=None,
                **base_kwargs,
            )

    return processor.apply_chat_template(conversation, **base_kwargs)


class RawVideoChatCollator:
    """
    Raw-video collator for chat JSONL.

    Core idea:
    - tokenize the FULL conversation (user video+text + assistant JSON)
    - tokenize the PROMPT ONLY (user video+text, with generation prompt)
    - mask the prompt tokens out in labels
    - keep loss only on the assistant JSON text

    This preserves your exact assistant target string and existing message structure.
    """

    def __init__(self, processor, num_frames: int):
        self.processor = processor
        self.num_frames = num_frames

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_messages_batch: List[List[Dict[str, Any]]] = []
        prompt_messages_batch: List[List[Dict[str, Any]]] = []

        for example in examples:
            messages = copy.deepcopy(example["messages"])
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError("Each example must end with one assistant message.")
            full_messages_batch.append(messages)
            prompt_messages_batch.append(messages[:-1])

        full_batch = apply_chat_template_video_safe(
            self.processor,
            full_messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            num_frames=self.num_frames,
            padding=True,
        )

        prompt_batch = apply_chat_template_video_safe(
            self.processor,
            prompt_messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            num_frames=self.num_frames,
            padding=True,

        )

        labels = full_batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        # Because we use right padding, attention_mask.sum() gives the real prompt length.
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100

        full_batch["labels"] = labels
        return full_batch


def load_datasets(train_file: str, test_file: str):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "test": test_file,
        },
    )
    return dataset


def load_model_and_processor(args: argparse.Namespace, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    processor.video_processor.size = {
        "longest_edge": 2048 * 32 * 32 * 2,
        "shortest_edge": 256 * 32 * 32 * 2,
    }
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model_load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    else:
        model_load_kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules: Any = args.lora_target_modules
    if args.lora_target_modules != "all-linear":
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        and args.save_strategy == args.eval_strategy
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
    num_frames: int,
    max_new_tokens: int,
) -> None:
    model.eval()

    prompt_messages = copy.deepcopy(example["messages"][:-1])
    ground_truth = extract_assistant_text(example["messages"][-1])

    inputs = apply_chat_template_video_safe(
        processor,
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        num_frames=num_frames,
        padding=False,
        truncation=False,
        max_length=None,
    )

    device = next(model.parameters()).device
    inputs = move_batch_to_device(inputs, device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    prediction = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n=== Validation Example ===")
    print("sample_id:", example.get("sample_id", "<missing>"))
    print("ground_truth:", ground_truth)
    print("prediction:", prediction)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = select_dtype(args)
    dataset = load_datasets(args.train_file, args.test_file)

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples:  {len(dataset['test'])}")

    model, processor = load_model_and_processor(args, dtype)
    collator = RawVideoChatCollator(
        processor=processor,
        num_frames=args.num_frames,
    )

    training_args = make_training_arguments(args, dtype)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if args.eval_strategy != "no" else None,
        data_collator=collator,
    )

    trainer.train()

    # Saves the LoRA adapter.
    trainer.save_model(args.output_dir)

    # Saves tokenizer + video/image processor config locally.
    processor.save_pretrained(args.output_dir)

    demo_idx = max(0, min(args.inference_index, len(dataset["test"]) - 1))
    run_one_inference_example(
        model=trainer.model,
        processor=processor,
        example=dataset["test"][demo_idx],
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()