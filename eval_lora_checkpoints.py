#!/usr/bin/env python3
"""
Evaluate:
1) the original pretrained base model,
2) the final LoRA adapter,
3) every checkpoint-* adapter,

on a test chat file that looks like:

{
  "sample_id": "...",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "video": "relative/or/absolute/path.mp4"},
        {"type": "text", "text": "prompt text ..."}
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

Also supports a legacy conversational format with `from` / `value` and top-level `video`.

Outputs:
- <adapter_dir>/eval_results/base_pretrained.json
- <adapter_dir>/eval_results/final.json
- <adapter_dir>/eval_results/checkpoint-XXXX.json
- <adapter_dir>/eval_results/summary.md
- <adapter_dir>/eval_results/all_metrics.json

Example:
python eval_lora_checkpoints.py \
    --adapter_dir runs/qwen35_9b_v3 \
    --test_file vlm_dataset_v2/test_chat.jsonl \
    --project_root . \
    --use_fp16 \
    --num_frames 12 \
    --max_new_tokens 256
"""

import argparse
import copy
import gc
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


EVAL_FIELDS = [
    "hazard_label",
    "hazard_present",
    "zone_relation",
    "object_state",
    "object_direction",
]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate pretrained base model + LoRA final adapter + checkpoints on test_chat.jsonl."
    )
    p.add_argument("--adapter_dir", type=str, default="runs/qwen35_9b_video_lora",
                   help="Directory containing the final adapter and checkpoint-* subdirs.")
    p.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl",
                   help="JSONL test file.")
    p.add_argument("--project_root", type=str, default=".",
                   help="Project root used to resolve relative video paths.")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-9B",
                   help="Base pretrained model hub id or local path.")
    p.add_argument("--base_eval_name", type=str, default="base_pretrained",
                   help="Name used in outputs for the original pretrained model.")
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--skip_base_eval", action="store_true",
                   help="Skip the original pretrained model evaluation.")
    p.add_argument("--only_final", action="store_true",
                   help="Evaluate only the final adapter, skip checkpoints.")
    p.add_argument("--only_checkpoints", type=str, default=None,
                   help="Comma-separated checkpoint steps to evaluate, e.g. '26,52'.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


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
    processor_kwargs: Dict[str, Any] = {
        "padding": padding,
        "truncation": truncation,
    }
    if num_frames is not None:
        processor_kwargs["num_frames"] = num_frames
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


def resolve_media_path(path_str: str, project_root: str, test_file_dir: str) -> str:
    """Resolve video/image paths without changing already valid absolute paths."""
    if not path_str:
        return path_str

    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    if os.path.exists(path_str):
        return path_str

    candidates = [
        os.path.join(project_root, path_str),
        os.path.join(test_file_dir, path_str),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path_str


def normalize_content_items(
    content: Any,
    example: Dict[str, Any],
    project_root: str,
    test_file_dir: str,
) -> List[Dict[str, Any]]:
    """
    Normalize message content into the shape expected by processor chat templates:
    [
      {"type": "video", "video": "..."},
      {"type": "text", "text": "..."}
    ]
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    normalized: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
            continue

        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue

        item_type = item.get("type")

        if item_type == "text":
            normalized.append({
                "type": "text",
                "text": item.get("text", ""),
            })

        elif item_type == "video":
            video_path = item.get("video", "")
            video_path = resolve_media_path(video_path, project_root, test_file_dir)
            normalized.append({
                "type": "video",
                "video": video_path,
            })

        elif item_type == "image":
            image_path = item.get("image", "")
            image_path = resolve_media_path(image_path, project_root, test_file_dir)
            normalized.append({
                "type": "image",
                "image": image_path,
            })

        else:
            # Preserve unknown item types as text fallback.
            normalized.append({
                "type": "text",
                "text": json.dumps(item, ensure_ascii=False),
            })

    return normalized


def normalize_message(
    msg: Dict[str, Any],
    example: Dict[str, Any],
    project_root: str,
    test_file_dir: str,
) -> Dict[str, Any]:
    """
    Supports:
    1) {"role": "...", "content": [...]}
    2) {"from": "human"/"gpt", "value": "..."} with optional top-level example["video"]
    """
    if "role" in msg:
        return {
            "role": msg["role"],
            "content": normalize_content_items(
                msg.get("content", []),
                example=example,
                project_root=project_root,
                test_file_dir=test_file_dir,
            ),
        }

    if "from" in msg:
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        role = role_map.get(msg["from"], msg["from"])
        value = msg.get("value", "")

        content: List[Dict[str, Any]] = []
        if isinstance(value, str) and "<video>" in value and example.get("video"):
            video_path = resolve_media_path(example["video"], project_root, test_file_dir)
            content.append({"type": "video", "video": video_path})
            text_only = value.replace("<video>", "").strip()
            if text_only:
                content.append({"type": "text", "text": text_only})
        else:
            content = normalize_content_items(
                value,
                example=example,
                project_root=project_root,
                test_file_dir=test_file_dir,
            )

        return {
            "role": role,
            "content": content,
        }

    raise ValueError(f"Unsupported message format: {msg}")


def normalize_messages(
    example: Dict[str, Any],
    project_root: str,
    test_file_dir: str,
) -> List[Dict[str, Any]]:
    raw_messages = example.get("messages")
    if raw_messages is None:
        raw_messages = example.get("conversations")
    if raw_messages is None:
        raise ValueError("Example has neither 'messages' nor 'conversations'.")

    return [
        normalize_message(m, example, project_root, test_file_dir)
        for m in raw_messages
    ]


def extract_text_from_message(message: Dict[str, Any]) -> str:
    content = message.get("content", [])

    if isinstance(content, str):
        return content.strip()

    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts).strip()


def split_prompt_and_target(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Use the LAST assistant message as target and everything before it as input prompt.
    """
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        raise ValueError("No assistant message found; cannot extract ground truth target.")
    if last_assistant_idx == 0:
        raise ValueError("Assistant message is the first message; no prompt remains.")

    return messages[:last_assistant_idx], messages[last_assistant_idx]


def extract_first_json_object(text: str) -> Optional[str]:
    """Find the first balanced JSON object in a string."""
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    json_block = extract_first_json_object(text)
    if json_block is not None:
        try:
            obj = json.loads(json_block)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def canonicalize_prediction_dict(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if obj is None:
        return None
    out = {}
    for k, v in obj.items():
        out[str(k)] = "" if v is None else str(v).strip()
    return out


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_base_model_and_processor(args: argparse.Namespace, dtype: torch.dtype):
    print(f"[INFO] Loading base model: {args.model_name_or_path}")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "size"):
        processor.video_processor.size = {
            "longest_edge": 560,
            "shortest_edge": 308,
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
        model_load_kwargs["device_map"] = "auto"
    else:
        model_load_kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs,
    )

    if quantization_config is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    model.config.use_cache = True
    model.eval()

    try:
        print(f"[INFO] Base model first parameter device: {next(model.parameters()).device}")
    except StopIteration:
        pass

    print(f"[INFO] Base model loaded. dtype={dtype}")
    return model, processor


def load_adapter(base_model, adapter_path: str):
    print(f"[INFO] Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    try:
        print(f"[INFO] Adapter model first parameter device: {next(model.parameters()).device}")
    except StopIteration:
        pass
    return model


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

@torch.inference_mode()
def run_inference_single(
    model,
    processor,
    prompt_messages: List[Dict[str, Any]],
    num_frames: int,
    max_new_tokens: int,
) -> str:
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
    # print(f"[DEBUG] model device: {device}", flush=True)  # Optional: Hide if too noisy
    inputs = move_batch_to_device(inputs, device)
    
    # if "input_ids" in inputs:
    #     print(f"[DEBUG] input_ids device: {inputs['input_ids'].device}", flush=True) # Optional: Hide if too noisy

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    input_ids = inputs["input_ids"]
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]

    pred_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return pred_text.strip()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_classification_stats(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "labels": {},
        }

    labels = sorted(set(y_true) | set(y_pred))
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    per_label = {}
    macro_p = macro_r = macro_f1 = 0.0
    weighted_p = weighted_r = weighted_f1 = 0.0
    total_support = 0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        total_support += support

    k = len(labels)
    macro_p = macro_p / k if k > 0 else 0.0
    macro_r = macro_r / k if k > 0 else 0.0
    macro_f1 = macro_f1 / k if k > 0 else 0.0

    weighted_p = weighted_p / total_support if total_support > 0 else 0.0
    weighted_r = weighted_r / total_support if total_support > 0 else 0.0
    weighted_f1 = weighted_f1 / total_support if total_support > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "labels": per_label,
    }


def compute_hazard_present_binary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0

    for r in results:
        gt = r["ground_truth"]
        pred = r.get("prediction_parsed")

        gt_hp = str(gt.get("hazard_present", "no")).strip()
        pred_hp = "no"
        if pred is not None:
            pred_hp = str(pred.get("hazard_present", "no")).strip()

        if gt_hp == "yes" and pred_hp == "yes":
            tp += 1
        elif gt_hp == "no" and pred_hp == "yes":
            fp += 1
        elif gt_hp == "yes" and pred_hp != "yes":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"error": "no samples"}

    parse_failures = 0
    exact_match = 0
    gt_parse_failures = 0

    bucket_correct: Dict[str, int] = defaultdict(int)
    bucket_total: Dict[str, int] = defaultdict(int)

    field_y_true = {f: [] for f in EVAL_FIELDS}
    field_y_pred = {f: [] for f in EVAL_FIELDS}

    for r in results:
        gt = r["ground_truth"]
        pred = r.get("prediction_parsed")
        bucket = r.get("hard_negative_bucket") or "positive"

        bucket_total[bucket] += 1

        if r.get("ground_truth_parse_failed", False):
            gt_parse_failures += 1
            continue

        if pred is None:
            parse_failures += 1

        all_match = True
        for field in EVAL_FIELDS:
            gt_val = str(gt.get(field, "__MISSING__")).strip()
            pred_val = "__PARSE_FAIL__" if pred is None else str(pred.get(field, "__MISSING__")).strip()

            field_y_true[field].append(gt_val)
            field_y_pred[field].append(pred_val)

            if gt_val != pred_val:
                all_match = False

        if all_match:
            exact_match += 1
            bucket_correct[bucket] += 1

    field_metrics = {
        field: compute_classification_stats(field_y_true[field], field_y_pred[field])
        for field in EVAL_FIELDS
    }

    per_bucket = {}
    for bucket in sorted(bucket_total.keys()):
        c = bucket_correct.get(bucket, 0)
        t = bucket_total[bucket]
        per_bucket[bucket] = {
            "correct": c,
            "total": t,
            "accuracy": round(c / t, 4) if t > 0 else 0.0,
        }

    return {
        "total_samples": total,
        "ground_truth_parse_failures": gt_parse_failures,
        "parse_failures": parse_failures,
        "exact_match_accuracy": round(exact_match / total, 4),
        "per_field_accuracy": {
            field: field_metrics[field]["accuracy"] for field in EVAL_FIELDS
        },
        "per_field_metrics": field_metrics,
        "hazard_present_binary": compute_hazard_present_binary(results),
        "per_bucket_accuracy": per_bucket,
    }


# -----------------------------------------------------------------------------
# Summary writing
# -----------------------------------------------------------------------------

def write_summary_markdown(all_results: List[Tuple[str, Dict[str, Any]]], out_path: str) -> None:
    lines: List[str] = []
    lines.append("# Evaluation Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall
    lines.append("## Overall Comparison\n")
    lines.append(
        "| Model | Exact Match | Parse Fail | hazard_label Acc | hazard_present Acc | zone_relation Acc | object_state Acc | object_direction Acc | HP Precision | HP Recall | HP F1 |"
    )
    lines.append(
        "|------|-------------|------------|------------------|--------------------|-------------------|------------------|----------------------|-------------|-----------|-------|"
    )

    for name, metrics in all_results:
        pfa = metrics.get("per_field_accuracy", {})
        hpb = metrics.get("hazard_present_binary", {})
        lines.append(
            f"| {name} "
            f"| {metrics.get('exact_match_accuracy', 0):.4f} "
            f"| {metrics.get('parse_failures', 0)} "
            f"| {pfa.get('hazard_label', 0):.4f} "
            f"| {pfa.get('hazard_present', 0):.4f} "
            f"| {pfa.get('zone_relation', 0):.4f} "
            f"| {pfa.get('object_state', 0):.4f} "
            f"| {pfa.get('object_direction', 0):.4f} "
            f"| {hpb.get('precision', 0):.4f} "
            f"| {hpb.get('recall', 0):.4f} "
            f"| {hpb.get('f1', 0):.4f} |"
        )
    lines.append("")

    # Per-field macro F1
    lines.append("## Per-Field Macro F1\n")
    lines.append(
        "| Model | hazard_label | hazard_present | zone_relation | object_state | object_direction |"
    )
    lines.append(
        "|------|--------------|----------------|---------------|--------------|------------------|"
    )
    for name, metrics in all_results:
        pfm = metrics.get("per_field_metrics", {})
        lines.append(
            f"| {name} "
            f"| {pfm.get('hazard_label', {}).get('macro_f1', 0):.4f} "
            f"| {pfm.get('hazard_present', {}).get('macro_f1', 0):.4f} "
            f"| {pfm.get('zone_relation', {}).get('macro_f1', 0):.4f} "
            f"| {pfm.get('object_state', {}).get('macro_f1', 0):.4f} "
            f"| {pfm.get('object_direction', {}).get('macro_f1', 0):.4f} |"
        )
    lines.append("")

    # Improvement vs base
    base_metrics = None
    if all_results:
        base_name, first_metrics = all_results[0]
        base_metrics = first_metrics

        lines.append(f"## Improvement vs {base_name}\n")
        lines.append(
            "| Model | Δ Exact Match | Δ hazard_label Acc | Δ hazard_present Acc | Δ zone_relation Acc | Δ object_state Acc | Δ object_direction Acc | Δ HP F1 |"
        )
        lines.append(
            "|------|----------------|--------------------|----------------------|---------------------|--------------------|------------------------|--------|"
        )

        base_pfa = base_metrics.get("per_field_accuracy", {})
        base_hpf1 = base_metrics.get("hazard_present_binary", {}).get("f1", 0)

        for name, metrics in all_results[1:]:
            pfa = metrics.get("per_field_accuracy", {})
            hp_f1 = metrics.get("hazard_present_binary", {}).get("f1", 0)
            lines.append(
                f"| {name} "
                f"| {metrics.get('exact_match_accuracy', 0) - base_metrics.get('exact_match_accuracy', 0):+.4f} "
                f"| {pfa.get('hazard_label', 0) - base_pfa.get('hazard_label', 0):+.4f} "
                f"| {pfa.get('hazard_present', 0) - base_pfa.get('hazard_present', 0):+.4f} "
                f"| {pfa.get('zone_relation', 0) - base_pfa.get('zone_relation', 0):+.4f} "
                f"| {pfa.get('object_state', 0) - base_pfa.get('object_state', 0):+.4f} "
                f"| {pfa.get('object_direction', 0) - base_pfa.get('object_direction', 0):+.4f} "
                f"| {hp_f1 - base_hpf1:+.4f} |"
            )
        lines.append("")

    # Binary confusion matrices
    lines.append("## Confusion Matrices (hazard_present)\n")
    for name, metrics in all_results:
        hpb = metrics.get("hazard_present_binary", {})
        lines.append(f"### {name}\n")
        lines.append("| | Pred YES | Pred NO |")
        lines.append("|---|---------|---------|")
        lines.append(f"| **GT YES** | {hpb.get('tp', 0)} | {hpb.get('fn', 0)} |")
        lines.append(f"| **GT NO** | {hpb.get('fp', 0)} | {hpb.get('tn', 0)} |")
        lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Summary written to: {out_path}")


# -----------------------------------------------------------------------------
# Adapter discovery
# -----------------------------------------------------------------------------

def discover_adapters(adapter_dir: str, only_final: bool, only_checkpoints: Optional[str]) -> List[Tuple[str, str]]:
    adapters: List[Tuple[str, str]] = []

    final_config = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(final_config):
        adapters.append(("final", adapter_dir))
    else:
        print(f"[WARN] No adapter_config.json in {adapter_dir} — skipping 'final'.")

    if only_final:
        return adapters

    checkpoint_dirs = sorted(
        glob.glob(os.path.join(adapter_dir, "checkpoint-*")),
        key=lambda d: int(re.search(r"checkpoint-(\d+)", d).group(1))
        if re.search(r"checkpoint-(\d+)", d) else 0,
    )

    allowed = None
    if only_checkpoints:
        allowed = set(int(x.strip()) for x in only_checkpoints.split(","))

    for ckpt_dir in checkpoint_dirs:
        m = re.search(r"checkpoint-(\d+)", ckpt_dir)
        if not m:
            continue
        step = int(m.group(1))
        if allowed is not None and step not in allowed:
            continue
        ckpt_config = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.isfile(ckpt_config):
            adapters.append((f"checkpoint-{step}", ckpt_dir))
        else:
            print(f"[WARN] No adapter_config.json in {ckpt_dir} — skipping.")

    return adapters


# -----------------------------------------------------------------------------
# Evaluation core
# -----------------------------------------------------------------------------

def evaluate_named_model(
    eval_name: str,
    model,
    processor,
    test_data: List[Dict[str, Any]],
    args: argparse.Namespace,
    eval_dir: str,
) -> Dict[str, Any]:
    sample_results: List[Dict[str, Any]] = []
    t_start = time.time()
    test_file_dir = os.path.dirname(os.path.abspath(args.test_file))
    project_root = os.path.abspath(args.project_root)

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {eval_name}")
    print(f"{'=' * 80}")

    pbar = None
    iterator = enumerate(test_data)
    if HAS_TQDM:
        pbar = tqdm(test_data, total=len(test_data), desc=f"Eval {eval_name}", unit="sample")
        iterator = enumerate(pbar)

    for i, example in iterator:
        sample_id = example.get("sample_id", f"sample_{i}")
        meta = example.get("meta", {})

        gt_parse_failed = False
        pred_text = ""
        pred_parsed = None
        gt_text = ""
        gt_parsed = {}
        inference_time = 0.0

        try:
            normalized_messages = normalize_messages(
                example,
                project_root=project_root,
                test_file_dir=test_file_dir,
            )
            prompt_messages, target_message = split_prompt_and_target(normalized_messages)

            gt_text = extract_text_from_message(target_message)
            gt_parsed = canonicalize_prediction_dict(try_parse_json(gt_text)) or {}
            if not gt_parsed:
                gt_parse_failed = True

            # Track Inference Time Start
            t_inf_start = time.time()
            pred_text = run_inference_single(
                model=model,
                processor=processor,
                prompt_messages=prompt_messages,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
            )
            # Track Inference Time End
            inference_time = time.time() - t_inf_start
            
            pred_parsed = canonicalize_prediction_dict(try_parse_json(pred_text))

        except Exception as e:
            pred_text = ""
            pred_parsed = None
            inference_time = 0.0
            print(f"\n[ERROR] {eval_name} :: {sample_id} :: {type(e).__name__}: {e}")

        field_matches = {}
        for field in EVAL_FIELDS:
            gt_val = gt_parsed.get(field, "__MISSING__")
            pred_val = "__PARSE_FAIL__" if pred_parsed is None else pred_parsed.get(field, "__MISSING__")
            field_matches[field] = (gt_val == pred_val)

        exact_match = all(field_matches.values())

        result = {
            "sample_id": sample_id,
            "ground_truth_text": gt_text,
            "ground_truth": gt_parsed,
            "ground_truth_parse_failed": gt_parse_failed,
            "prediction_raw": pred_text,
            "prediction_parsed": pred_parsed,
            "field_matches": field_matches,
            "exact_match": exact_match,
            "inference_time_sec": round(inference_time, 4), # NEW FIELD ADDED
            "hard_negative_bucket": meta.get("hard_negative_bucket"),
            "label_signature": meta.get("label_signature", ""),
            "meta": meta,
        }
        # Optional: Disable these prints if you find them too noisy for per-sample execution
        print(gt_parsed)
        print("")
        print(pred_parsed)
        print(inference_time)
        sample_results.append(result)

        # Live console line for non-tqdm fallback
        if not HAS_TQDM:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0.0
            gt_hp = gt_parsed.get("hazard_present", "?")
            pred_hp = "PARSE_FAIL" if pred_parsed is None else pred_parsed.get("hazard_present", "?")
            mark = "✓" if exact_match else "✗"
            print(
                f"[{i+1:3d}/{len(test_data)}] {mark} "
                f"GT={gt_hp:>3s} PRED={pred_hp:>10s} "
                f"({rate:.2f} it/s, ETA {eta:.0f}s) "
                f"— {sample_id[:60]}"
            )
        else:
            gt_hp = gt_parsed.get("hazard_present", "?")
            pred_hp = "PARSE_FAIL" if pred_parsed is None else pred_parsed.get("hazard_present", "?")
            pbar.set_postfix({
                "GT": gt_hp,
                "PRED": pred_hp,
                "Exact": int(exact_match),
            })

    elapsed_total = time.time() - t_start
    metrics = compute_metrics(sample_results)
    metrics["adapter_name"] = eval_name
    metrics["inference_time_sec"] = round(elapsed_total, 2)
    metrics["samples_per_sec"] = round(len(test_data) / elapsed_total, 4) if elapsed_total > 0 else 0.0

    out_path = os.path.join(eval_dir, f"{eval_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "per_sample": sample_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[INFO] {eval_name} finished in {elapsed_total:.1f}s")
    print(f"[INFO] Results saved to: {out_path}")
    print(f"[INFO] Exact match: {metrics['exact_match_accuracy']:.4f}")
    print("[INFO] Per-field accuracy:")
    for field in EVAL_FIELDS:
        print(f"  - {field:16s}: {metrics['per_field_accuracy'][field]:.4f}")

    print("[INFO] Per-field macro F1:")
    for field in EVAL_FIELDS:
        print(f"  - {field:16s}: {metrics['per_field_metrics'][field]['macro_f1']:.4f}")

    hpb = metrics["hazard_present_binary"]
    print(f"[INFO] hazard_present Precision: {hpb['precision']:.4f}")
    print(f"[INFO] hazard_present Recall:    {hpb['recall']:.4f}")
    print(f"[INFO] hazard_present F1:        {hpb['f1']:.4f}")
    print(f"[INFO] Parse failures:           {metrics['parse_failures']}")

    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = select_dtype(args)

    if not os.path.isfile(args.test_file):
        print(f"[ERROR] Test file not found: {args.test_file}")
        sys.exit(1)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"[INFO] Loaded {len(test_data)} test samples from {args.test_file}")

    adapters = discover_adapters(args.adapter_dir, args.only_final, args.only_checkpoints)
    adapter_names = [name for name, _ in adapters]
    print(f"[INFO] Adapter evaluations queued: {adapter_names}")

    eval_dir = os.path.join(args.adapter_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    base_model, processor = load_base_model_and_processor(args, dtype)
    all_summaries: List[Tuple[str, Dict[str, Any]]] = []

    # 1) Base pretrained model
    if not args.skip_base_eval:
        base_model.eval()
        base_metrics = evaluate_named_model(
            eval_name=args.base_eval_name,
            model=base_model,
            processor=processor,
            test_data=test_data,
            args=args,
            eval_dir=eval_dir,
        )
        all_summaries.append((args.base_eval_name, base_metrics))

    # 2) Final adapter + checkpoints
    for idx, (adapter_name, adapter_path) in enumerate(adapters, start=1):
        print(f"\n[INFO] Loading adapter {idx}/{len(adapters)}: {adapter_name}")
        model = load_adapter(base_model, adapter_path)

        metrics = evaluate_named_model(
            eval_name=adapter_name,
            model=model,
            processor=processor,
            test_data=test_data,
            args=args,
            eval_dir=eval_dir,
        )
        metrics["adapter_path"] = adapter_path
        all_summaries.append((adapter_name, metrics))

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = os.path.join(eval_dir, "summary.md")
    write_summary_markdown(all_summaries, summary_path)

    combined_path = os.path.join(eval_dir, "all_metrics.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump({name: metrics for name, metrics in all_summaries}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Combined metrics written to: {combined_path}")
    print("\n[DONE] All evaluations complete.")


if __name__ == "__main__":
    main()