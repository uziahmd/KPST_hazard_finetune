#!/usr/bin/env python3
"""
Supervised fine-tuning for a local conversational raw-video dataset using:
- Qwen/Qwen3.5-9B
- Hugging Face Transformers + TRL SFTTrainer
- PEFT LoRA / QLoRA

The script expects JSONL rows similar to:
{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "path/to/clip.mp4"},
        {"type": "text",  "text":  "...task prompt..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{...strict json label...}"}
      ]
    }
  ]
}

Important:
- Requires a recent Transformers build with Qwen3.5 support.
- Requires a video backend supported by Transformers for local video decoding
  (e.g. PyAV, Decord, or torchvision).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
load_dotenv()
token = os.getenv("HF_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT Qwen3.5-9B on a raw-video safety dataset")

    # Paths
    parser.add_argument("--train_file", type=str, default="vlm_dataset/train_chat.jsonl")
    parser.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl")
    parser.add_argument("--video_root", type=str, default="", help="Optional root directory that contains video files.")
    parser.add_argument(
        "--strip_video_prefix",
        type=str,
        default="",
        help="Optional prefix to strip from dataset video paths before joining with --video_root.",
    )
    parser.add_argument("--output_dir", type=str, default="qwen35_9B_forklift1_lora")
    parser.add_argument("--merged_output_dir", type=str, default="qwen35_9B_forklift1")

    # Model / processor
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "none"],
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )

    # Video sampling
    parser.add_argument("--num_frames", type=int, default=12, help="Uniformly sample this many frames per clip.")
    parser.add_argument(
        "--video_fps",
        type=float,
        default=0.0,
        help="Optional FPS sampling target. Leave 0 to ignore and use --num_frames only.",
    )

    # PEFT / quantization
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--no_use_qlora", action="store_false", dest="use_qlora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="all-linear")
    parser.add_argument(
        "--modules_to_save",
        type=str,
        default="lm_head,embed_tokens",
        help="Comma-separated module names to keep trainable/savable in PEFT, when present.",
    )
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Recommended: paged_adamw_8bit for QLoRA, adamw_torch_fused otherwise.",
    )
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False)

    # Debug / subset
    parser.add_argument("--train_max_samples", type=int, default=0)
    parser.add_argument("--test_max_samples", type=int, default=0)

    # Generation-based eval / demo
    parser.add_argument("--generation_eval_max_samples", type=int, default=32)
    parser.add_argument("--demo_index", type=int, default=0)
    parser.add_argument("--gen_max_new_tokens", type=int, default=192)

    # Merge / save full model
    parser.add_argument("--merge_after_training", action="store_true", default=True)
    parser.add_argument("--no_merge_after_training", action="store_false", dest="merge_after_training")
    parser.add_argument(
        "--merge_device_map",
        type=str,
        default="cpu",
        help='Where to load the base model for merging: e.g. "cpu", "auto", or "cuda:0".',
    )
    parser.add_argument("--max_shard_size", type=str, default="5GB")

    return parser.parse_args()


def pick_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32

    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def split_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def canonicalize_json_string(text: str) -> Optional[str]:
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return None


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    chunks: List[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            chunks.append(item.get("text", ""))
    return "\n".join([x for x in chunks if x]).strip()


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def resolve_video_path(raw_path: str, source_jsonl_path: str, args: argparse.Namespace) -> str:
    source_dir = Path(source_jsonl_path).resolve().parent
    raw_path = str(raw_path)
    candidates: List[Path] = []

    p = Path(raw_path)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path(raw_path))
        candidates.append(source_dir / raw_path)

        if args.video_root:
            video_root = Path(args.video_root)
            candidates.append(video_root / raw_path)
            candidates.append(video_root / Path(raw_path).name)

            if args.strip_video_prefix and raw_path.startswith(args.strip_video_prefix):
                stripped = raw_path[len(args.strip_video_prefix):].lstrip("/\\")
                candidates.append(video_root / stripped)

    candidates = dedupe_paths(candidates)

    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    # Fall back to the last candidate so the error message is at least useful.
    return str(candidates[-1].resolve()) if candidates else raw_path


class DatasetFormatter:
    def __init__(self, source_jsonl_path: str, args: argparse.Namespace) -> None:
        self.source_jsonl_path = source_jsonl_path
        self.args = args

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raw_messages = example.get("messages", [])
        if not raw_messages:
            raise ValueError(f"Missing messages in example: {example}")

        processed_messages: List[Dict[str, Any]] = []
        resolved_video_paths: List[str] = []
        user_texts: List[str] = []

        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [content]

            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append({"type": "text", "text": str(item)})
                    continue

                item_type = item.get("type")

                if item_type == "video":
                    raw_video_path = item.get("path") or item.get("video") or item.get("url")
                    if not isinstance(raw_video_path, str):
                        raise ValueError(f"Unsupported video payload in example: {item}")
                    resolved = resolve_video_path(raw_video_path, self.source_jsonl_path, self.args)
                    resolved_video_paths.append(resolved)
                    new_content.append({"type": "video", "path": resolved})
                elif item_type == "video_url":
                    raw_video_url = item.get("video_url", {}).get("url")
                    if isinstance(raw_video_url, str) and Path(raw_video_url).exists():
                        resolved = str(Path(raw_video_url).resolve())
                        resolved_video_paths.append(resolved)
                        new_content.append({"type": "video", "path": resolved})
                    else:
                        # Keep a true URL unchanged.
                        new_content.append(item)
                elif item_type == "text":
                    text = item.get("text", "")
                    new_content.append({"type": "text", "text": text})
                    if role == "user" and text:
                        user_texts.append(text)
                else:
                    # Preserve other multimodal items if present.
                    new_content.append(item)

            processed_messages.append({"role": role, "content": new_content})

        if processed_messages[-1].get("role") != "assistant":
            raise ValueError("Expected the last message to be the supervised assistant target.")

        prompt_messages = processed_messages[:-1]
        assistant_response = extract_text_from_content(processed_messages[-1].get("content", []))
        prompt_text = "\n\n".join([t for t in user_texts if t]).strip()
        video_path = resolved_video_paths[0] if resolved_video_paths else ""
        video_exists = bool(video_path) and Path(video_path).exists()

        return {
            "sample_id": example.get("sample_id", ""),
            "video_path": video_path,
            "video_exists": video_exists,
            "prompt": prompt_text,
            "response": assistant_response,
            "response_canonical": canonicalize_json_string(assistant_response),
            "messages": processed_messages,
            "prompt_messages": prompt_messages,
            "raw_meta": example.get("meta", {}),
        }


def build_dataset(train_file: str, test_file: str, args: argparse.Namespace) -> DatasetDict:
    raw = load_dataset("json", data_files={"train": train_file, "test": test_file})

    train_fmt = DatasetFormatter(train_file, args)
    test_fmt = DatasetFormatter(test_file, args)

    train_ds = raw["train"].map(train_fmt, remove_columns=raw["train"].column_names)
    test_ds = raw["test"].map(test_fmt, remove_columns=raw["test"].column_names)

    if args.train_max_samples > 0:
        train_ds = train_ds.select(range(min(args.train_max_samples, len(train_ds))))
    if args.test_max_samples > 0:
        test_ds = test_ds.select(range(min(args.test_max_samples, len(test_ds))))

    missing_train = [ex for ex in train_ds if not ex["video_exists"]]
    missing_test = [ex for ex in test_ds if not ex["video_exists"]]
    if missing_train or missing_test:
        missing_examples = (missing_train + missing_test)[:5]
        details = "\n".join(
            f"sample_id={ex['sample_id']} video_path={ex['video_path']}" for ex in missing_examples
        )
        raise FileNotFoundError(
            "Some dataset video paths could not be resolved.\n"
            "Check --video_root and --strip_video_prefix.\n"
            f"Examples:\n{details}"
        )

    return DatasetDict({"train": train_ds, "test": test_ds})


def maybe_set_padding(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def resolve_modules_to_save(model: torch.nn.Module, requested: List[str]) -> Optional[List[str]]:
    if not requested:
        return None

    module_names = [name for name, _ in model.named_modules()]
    kept: List[str] = []
    for target in requested:
        if any(name == target or name.endswith(f".{target}") for name in module_names):
            kept.append(target)
    return kept or None


def get_single_device_map() -> Optional[Dict[str, int]]:
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return None


def maybe_apply_chat_template(processor: Any, conversations: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
    """
    Tries batched multimodal chat templating first.
    If that fails, it retries without the optional `enable_thinking` kwarg, and then
    falls back to per-example processing plus manual padding/concatenation.
    """
    def _single_call(conv: Any, kw: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return processor.apply_chat_template(conv, **kw)

    attempt_kwargs = dict(kwargs)
    try:
        return _single_call(conversations, attempt_kwargs)
    except TypeError:
        attempt_kwargs = {k: v for k, v in attempt_kwargs.items() if k != "enable_thinking"}
        try:
            return _single_call(conversations, attempt_kwargs)
        except Exception:
            pass
    except Exception:
        pass

    if not isinstance(conversations, list) or not conversations or not isinstance(conversations[0], list):
        return _single_call(conversations, attempt_kwargs)

    single_kwargs = dict(attempt_kwargs)
    single_kwargs.pop("padding", None)

    encoded_list = [_single_call(conv, single_kwargs) for conv in conversations]

    max_len = max(enc["input_ids"].shape[1] for enc in encoded_list)
    pad_id = processor.tokenizer.pad_token_id
    merged: Dict[str, torch.Tensor] = {}
    all_keys = set().union(*[enc.keys() for enc in encoded_list])

    for key in all_keys:
        values = [enc[key] for enc in encoded_list if key in enc]
        if not values:
            continue

        if key in {"input_ids", "attention_mask"}:
            padded = []
            for tensor in values:
                pad_value = pad_id if key == "input_ids" else 0
                pad_len = max_len - tensor.shape[1]
                if pad_len > 0:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_value)
                padded.append(tensor)
            merged[key] = torch.cat(padded, dim=0)
        else:
            merged[key] = torch.cat(values, dim=0)

    return merged


class VideoConversationCollator:
    """
    Custom collator because the dataset is raw-video conversational JSONL, not the
    image/images schema documented as the native TRL VLM path.

    Loss is computed only on the final assistant response by masking everything up to
    the assistant generation boundary.
    """

    def __init__(
        self,
        processor: Any,
        num_frames: int = 12,
        video_fps: float = 0.0,
        disable_thinking: bool = True,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames
        self.video_fps = video_fps
        self.disable_thinking = disable_thinking
        self.pad_token_id = processor.tokenizer.pad_token_id

    def _template_kwargs(self, add_generation_prompt: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if self.num_frames > 0:
            kwargs["num_frames"] = self.num_frames
            kwargs["do_sample_frames"] = True
        if self.video_fps > 0:
            kwargs["fps"] = self.video_fps
            kwargs["do_sample_frames"] = True
        if self.disable_thinking:
            # Qwen3.5's chat template can consume this kwarg when supported.
            kwargs["enable_thinking"] = False
        return kwargs

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_conversations = [ex["messages"] for ex in examples]
        prompt_only_conversations = [ex["prompt_messages"] for ex in examples]

        full_batch = maybe_apply_chat_template(
            self.processor,
            full_conversations,
            **self._template_kwargs(add_generation_prompt=False),
        )
        prompt_batch = maybe_apply_chat_template(
            self.processor,
            prompt_only_conversations,
            **self._template_kwargs(add_generation_prompt=True),
        )

        labels = full_batch["input_ids"].clone()

        if "attention_mask" in prompt_batch:
            prefix_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        else:
            prefix_lengths = (prompt_batch["input_ids"] != self.pad_token_id).sum(dim=1).tolist()

        for i, prefix_len in enumerate(prefix_lengths):
            labels[i, :prefix_len] = -100

        labels[full_batch["input_ids"] == self.pad_token_id] = -100
        full_batch["labels"] = labels
        return full_batch


def move_batch_to_model_device(batch: Dict[str, Any], model: torch.nn.Module) -> Dict[str, Any]:
    try:
        device = model.device
    except Exception:
        device = next(model.parameters()).device

    moved: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def generate_one(
    model: torch.nn.Module,
    processor: Any,
    example: Dict[str, Any],
    num_frames: int,
    video_fps: float,
    max_new_tokens: int,
) -> str:
    template_kwargs: Dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if num_frames > 0:
        template_kwargs["num_frames"] = num_frames
        template_kwargs["do_sample_frames"] = True
    if video_fps > 0:
        template_kwargs["fps"] = video_fps
        template_kwargs["do_sample_frames"] = True

    inputs = maybe_apply_chat_template(processor, example["prompt_messages"], **template_kwargs)
    inputs = move_batch_to_model_device(inputs, model)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, prompt_len:]
    text = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def evaluate_generation(
    model: torch.nn.Module,
    processor: Any,
    dataset: Dataset,
    num_frames: int,
    video_fps: float,
    max_new_tokens: int,
    max_samples: int,
) -> Dict[str, Any]:
    n = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)
    if n == 0:
        return {}

    strict_json = 0
    exact_match = 0
    outputs: List[Dict[str, Any]] = []

    for idx in range(n):
        ex = dataset[idx]
        pred_text = generate_one(model, processor, ex, num_frames, video_fps, max_new_tokens)
        pred_canonical = canonicalize_json_string(pred_text)
        gold_canonical = ex["response_canonical"]

        if pred_canonical is not None:
            strict_json += 1
        if pred_canonical is not None and gold_canonical is not None and pred_canonical == gold_canonical:
            exact_match += 1

        outputs.append(
            {
                "sample_id": ex["sample_id"],
                "prediction": pred_text,
                "prediction_canonical": pred_canonical,
                "gold": ex["response"],
                "gold_canonical": gold_canonical,
                "strict_json_ok": pred_canonical is not None,
                "exact_match": pred_canonical is not None and gold_canonical is not None and pred_canonical == gold_canonical,
            }
        )

    metrics = {
        "num_samples": n,
        "strict_json_rate": strict_json / n,
        "canonical_exact_match": exact_match / n,
        "samples": outputs[: min(5, len(outputs))],
    }
    return metrics


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model_and_processor(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, torch.dtype]:
    dtype = pick_torch_dtype(args.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    # You are using fixed-frame sampling, so disable fps-based sampling.
    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = None
        
    maybe_set_padding(processor)

    quant_config = None
    device_map = None
    if args.use_qlora:
        compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        device_map = get_single_device_map()

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.attn_implementation != "none":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)

    if args.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    modules_to_save = resolve_modules_to_save(model, split_csv(args.modules_to_save))
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, processor, dtype


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def merge_and_save_full_model(
    adapter_dir: str,
    merged_output_dir: str,
    model_name: str,
    processor: Any,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> None:
    print("\nReloading base model to merge LoRA adapter...")
    free_memory()

    merge_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "none":
        merge_kwargs["attn_implementation"] = args.attn_implementation
    if args.merge_device_map:
        merge_kwargs["device_map"] = args.merge_device_map

    base_model = AutoModelForImageTextToText.from_pretrained(model_name, **merge_kwargs)
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()

    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        merged_output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(merged_output_dir)
    print(f"Merged model saved to: {merged_output_dir}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading dataset...")
    dataset = build_dataset(args.train_file, args.test_file, args)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples : {len(dataset['test'])}")

    print("\nLoading processor + model...")
    model, processor, dtype = build_model_and_processor(args)

    collator = VideoConversationCollator(
        processor=processor,
        num_frames=args.num_frames,
        video_fps=args.video_fps,
        disable_thinking=True,
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to=args.report_to,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        processing_class=processor,
    )

    print("\nStarting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_state()

    print("\nRunning eval loss on test set...")
    eval_metrics = trainer.evaluate(dataset["test"])
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False, default=str))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_json(Path(args.output_dir) / "train_metrics.json", train_result.metrics)
    save_json(Path(args.output_dir) / "eval_metrics.json", eval_metrics)

    print(f"\nSaving adapter + processor to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Generation-based eval on a subset.
    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = True

    if args.generation_eval_max_samples > 0:
        print("\nRunning small generation-based evaluation...")
        gen_metrics = evaluate_generation(
            model=trainer.model,
            processor=processor,
            dataset=dataset["test"],
            num_frames=args.num_frames,
            video_fps=args.video_fps,
            max_new_tokens=args.gen_max_new_tokens,
            max_samples=args.generation_eval_max_samples,
        )
        print(json.dumps(gen_metrics, indent=2, ensure_ascii=False, default=str))
        save_json(Path(args.output_dir) / "generation_eval.json", gen_metrics)

    # Demo inference.
    if len(dataset["test"]) > 0:
        demo_index = max(0, min(args.demo_index, len(dataset["test"]) - 1))
        demo_example = dataset["test"][demo_index]
        print(f"\nDemo inference on test sample index {demo_index} (sample_id={demo_example['sample_id']})")
        pred = generate_one(
            model=trainer.model,
            processor=processor,
            example=demo_example,
            num_frames=args.num_frames,
            video_fps=args.video_fps,
            max_new_tokens=args.gen_max_new_tokens,
        )
        print("\nGenerated JSON:")
        print(pred)
        print("\nGold JSON:")
        print(demo_example["response"])
        save_json(
            Path(args.output_dir) / "demo_prediction.json",
            {
                "sample_id": demo_example["sample_id"],
                "prediction": pred,
                "gold": demo_example["response"],
            },
        )

    if args.merge_after_training:
        merge_and_save_full_model(
            adapter_dir=args.output_dir,
            merged_output_dir=args.merged_output_dir,
            model_name=args.model_name,
            processor=processor,
            dtype=dtype,
            args=args,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()