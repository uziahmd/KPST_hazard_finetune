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

RUN

python train_qwen35_video_lora.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file   vlm_dataset_both_aug/val_chat.jsonl \
  --test_file  vlm_dataset_both_aug/test_chat.jsonl \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --output_dir runs/qwen35_9b_both_aug_2 \
  --num_frames 12 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 9 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --gradient_checkpointing \
  --use_fp16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 9 \
  --attn_implementation sdpa \
  --seed 3407 \
  --pause_on_interrupt \
  --resume_from_checkpoint last
"""

import argparse
import copy
import inspect
import os
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

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
from training_pause_resume import (
    PauseResumeManager,
    create_pause_resume_callback,
    resolve_resume_checkpoint,
)

# Load variables from .env into os.environ
load_dotenv(override=True)

# Verification (Optional: remove this in production)
if os.getenv("HF_TOKEN"):
    print("HF_TOKEN successfully loaded from .env")
else:
    print("Warning: HF_TOKEN not found in .env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen 3.5 9B on raw-video chat JSONL.")
    parser.add_argument("--train_file", type=str, default="vlm_dataset/train_chat.jsonl")
    parser.add_argument("--val_file",   type=str, default="vlm_dataset/val_chat.jsonl",
                        help="Validation chat JSONL (produced by data_gen.py). Used for "
                             "eval_strategy=epoch. Pass empty string to disable.")
    parser.add_argument("--test_file",  type=str, default="vlm_dataset/test_chat.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output_dir", type=str, default="runs/qwen35_9b_video_lora")

    parser.add_argument("--num_frames", type=int, default=12, help="Frames sampled per video clip by the processor.")
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length after chat templating.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Used only for the post-training demo inference.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
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
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        help="Use 'eager' on V100 (no Flash Attn). Use 'flash_attention_2' on A100+.")
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
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='Comma-separated list of modules to target with LoRA. '
             'Use "all-linear" to target every linear layer (aggressive on small datasets). '
             'Default targets only the LLM decoder layers, leaving the vision encoder unchanged.',
    )

    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--inference_index", type=int, default=0, help="Test sample index for the post-training demo.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Trainer checkpoint directory to resume from. Use 'last' to resume from the latest complete checkpoint in output_dir.",
    )
    parser.add_argument(
        "--pause_file",
        type=str,
        default="",
        help="Optional control file. Create this file while training is running to request a safe pause at the next optimizer-step boundary.",
    )
    parser.add_argument(
        "--pause_on_interrupt",
        action="store_true",
        help="Treat Ctrl+C as a safe pause request. Training will finish the current optimizer step, save a full checkpoint, and exit cleanly.",
    )
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
    truncation: bool = False,
    max_length: Optional[int] = None,
):
    # processor_kwargs go to processor.__call__() for video/image/tokenizer
    # processing.  enable_thinking is a TEMPLATE parameter (controls the
    # Jinja chat template), so it stays top-level on apply_chat_template.
    processor_kwargs: Dict[str, Any] = {
        "padding": padding,
        "truncation": truncation,
    }
    if num_frames is not None:
        processor_kwargs["num_frames"] = num_frames
        # Qwen3 VL's sample_frames() rejects getting BOTH num_frames and fps.
        # The processor ships with a default fps, so we must clear it.
        processor_kwargs["fps"] = None
    if max_length is not None:
        processor_kwargs["max_length"] = max_length

    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_dict=return_dict,
        return_tensors=return_tensors,
        enable_thinking=enable_thinking,
        processor_kwargs=processor_kwargs,
    )


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


def load_datasets(train_file: str, val_file: str, test_file: str):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    data_files: Dict[str, str] = {
        "train": train_file,
        "test": test_file,
    }

    has_val = val_file and os.path.exists(val_file)
    if val_file and not has_val:
        print(f"[WARN] val_file not found: {val_file} — evaluation during training will be disabled.")
    if has_val:
        data_files["val"] = val_file

    dataset = load_dataset("json", data_files=data_files)
    return dataset, has_val


def load_model_and_processor(args: argparse.Namespace, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # ── Video frame resolution ────────────────────────────────────────────────
    # Source clips: ~1280×720 CCTV @ 15fps.
    # Qwen3 VL vision encoder uses 28×28 px patches, so dimensions must be
    # multiples of 28 for clean tiling. Both keys MUST be present as integers
    # (the processor raises ValueError if either is None).
    #
    # PRIMARY (V100 32GB + 4-bit + 12 frames + gradient checkpointing):
    #   longest_edge  = 560  →  20 patches × 28px
    #   shortest_edge = 308  →  11 patches × 28px  (preserves ~16:9 ratio)
    #
    # OOM FALLBACK — swap in if you hit CUDA OOM (saves ~30% vision VRAM):
    #   longest_edge  = 448  →  16 patches × 28px
    #   shortest_edge = 252  →   9 patches × 28px
    processor.video_processor.size = {
        "longest_edge": 560,    # ← change to 448 if OOM
        "shortest_edge": 308,   # ← change to 252 if OOM
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
        # >0 workers allow video I/O to overlap GPU compute.
        # Keep at 2 for V100 where host RAM is the bottleneck.
        "dataloader_num_workers": 2,
        "gradient_checkpointing": args.gradient_checkpointing,
        "label_names": ["labels"],
        "report_to": [] if args.report_to == "none" else args.report_to,
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    }
    if "ignore_data_skip" in sig:
        common_kwargs["ignore_data_skip"] = False

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
    dataset, has_val = load_datasets(args.train_file, args.val_file, args.test_file)

    print(f"Train samples: {len(dataset['train'])}")
    if has_val:
        print(f"Val samples  : {len(dataset['val'])}")
    print(f"Test samples : {len(dataset['test'])}")

    model, processor = load_model_and_processor(args, dtype)
    collator = RawVideoChatCollator(
        processor=processor,
        num_frames=args.num_frames,
    )

    pause_manager = None
    trainer_callbacks = []
    if args.pause_file or args.pause_on_interrupt or args.resume_from_checkpoint:
        pause_manager = PauseResumeManager(
            output_dir=args.output_dir,
            pause_request_path=args.pause_file or None,
            pause_on_interrupt=args.pause_on_interrupt,
        )
        resume_checkpoint, resume_warnings = resolve_resume_checkpoint(
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            state_path=str(pause_manager.state_path),
        )
        pause_manager.resume_checkpoint = Path(resume_checkpoint).resolve() if resume_checkpoint else None
        for warning in resume_warnings:
            print(warning)
        for line in pause_manager.describe_startup():
            print(line)
        trainer_callbacks.append(create_pause_resume_callback(pause_manager))
    else:
        resume_checkpoint = None

    training_args = make_training_arguments(args, dtype)

    # Use val split for in-training eval when available.
    # Test split is reserved for final post-training evaluation only.
    eval_ds = dataset["val"] if has_val and args.eval_strategy != "no" else None
    if eval_ds is None and args.eval_strategy != "no":
        print("[WARN] eval_strategy is set but no val set is available — disabling eval.")
        args.eval_strategy = "no"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=trainer_callbacks,
    )

    train_kwargs = {}
    if resume_checkpoint:
        train_kwargs["resume_from_checkpoint"] = resume_checkpoint

    if pause_manager is not None:
        with pause_manager.signal_handlers():
            trainer.train(**train_kwargs)
    else:
        trainer.train(**train_kwargs)

    if pause_manager is not None and pause_manager.pause_armed:
        paused_checkpoint = pause_manager.finalize_pause(trainer.state)
        print(
            f"[Pause/Resume] Training paused safely at global step {trainer.state.global_step}. "
            f"Resume with --resume_from_checkpoint last. Checkpoint: {paused_checkpoint}"
        )
        return

    # Saves the LoRA adapter.
    trainer.save_model(args.output_dir)

    # Saves tokenizer + video/image processor config locally.
    processor.save_pretrained(args.output_dir)

    if pause_manager is not None:
        pause_manager.finalize_completion(trainer.state)

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
