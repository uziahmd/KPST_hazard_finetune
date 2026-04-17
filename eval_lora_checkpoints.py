#!/usr/bin/env python3
"""
Evaluate:
1) the original pretrained base model,
2) the final LoRA adapter,
3) every checkpoint-* adapter,

on a test chat file that may contain:
- forklift task samples:
    fields = hazard_label, hazard_present, zone_relation, object_state, object_direction
- robot task samples:
    fields = hazard_label, hazard_present, zone_relation, object_state

Key updates:
- task-adaptive scoring
- exact match scored only on fields relevant to that sample's task
- separate metrics for forklift, robot, and combined
- CLI toggle for mixed-task vs single-task test files via --task_mode

Example: (MIXED)
python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_both_aug_2 \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --project_root . \
  --task_mode both \
  --use_fp16

  (FORKLIFT)
python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_forklift \
  --test_file vlm_dataset_forklift/test_chat.jsonl \
  --project_root . \
  --task_mode forklift \
  --use_fp16


    (ROBOT)
python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_robot \
  --test_file vlm_dataset_robot/test_chat.jsonl \
  --project_root . \
  --task_mode robot \
  --use_fp16
"""

import argparse
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


# -----------------------------------------------------------------------------
# Task config
# -----------------------------------------------------------------------------

ALL_FIELDS = [
    "hazard_label",
    "hazard_present",
    "zone_relation",
    "object_state",
    "object_direction",
]

TASK_FIELDS = {
    "forklift": [
        "hazard_label",
        "hazard_present",
        "zone_relation",
        "object_state",
        "object_direction",
    ],
    "robot": [
        "hazard_label",
        "hazard_present",
        "zone_relation",
        "object_state",
    ],
}

TASK_CHOICES = ["forklift", "robot", "both"]


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
    p.add_argument(
        "--task_mode",
        type=str,
        default="both",
        choices=TASK_CHOICES,
        help=(
            "Task mode for the test file: "
            "'forklift' for forklift-only data, "
            "'robot' for robot-only data, "
            "'both' for mixed-task data."
        ),
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def safe_round(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    if x is None:
        return None
    return round(x, ndigits)


def fmt_metric(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


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


# -----------------------------------------------------------------------------
# Task inference
# -----------------------------------------------------------------------------

def canonicalize_task_name(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    s = str(task).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    if s in {"forklift", "fork", "fork_lift", "forklift_entry_hazard"}:
        return "forklift"

    if s in {
        "robot",
        "machine",
        "machinery",
        "robot_zone_intrusion",
        "human_machine_shared_workspace",
        "robot_hazard",
    }:
        return "robot"

    return None


def infer_task_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "forklift" in t:
        return "forklift"
    if "robot" in t or "machine workspace" in t or "robot zone" in t:
        return "robot"
    return None


def infer_task(
    example: Dict[str, Any],
    prompt_messages: List[Dict[str, Any]],
    gt_parsed: Dict[str, Any],
    task_mode: str,
) -> str:
    if task_mode in {"forklift", "robot"}:
        return task_mode

    meta = example.get("meta", {}) or {}

    # 1) explicit task metadata
    for candidate in [
        meta.get("task"),
        example.get("task"),
        meta.get("task_name"),
        meta.get("eval_task"),
    ]:
        task = canonicalize_task_name(candidate)
        if task is not None:
            return task

    # 2) sample_id conventions
    sample_id = str(example.get("sample_id", "")).lower()
    if sample_id.startswith("fork_") or sample_id.startswith("forklift_") or "_fork_" in sample_id:
        return "forklift"
    if sample_id.startswith("robot_") or "_robot_" in sample_id:
        return "robot"

    # 3) label signature conventions
    label_signature = str(meta.get("label_signature", "")).lower()
    if label_signature.startswith("forklift|") or label_signature.startswith("fork|"):
        return "forklift"
    if label_signature.startswith("robot|"):
        return "robot"

    # 4) ground truth keys
    if "object_direction" in gt_parsed:
        return "forklift"
    robot_core = {"hazard_label", "hazard_present", "zone_relation", "object_state"}
    if robot_core.issubset(set(gt_parsed.keys())):
        return "robot"

    # 5) prompt text
    joined_prompt = "\n".join(
        extract_text_from_message(m) for m in prompt_messages if m.get("role") != "assistant"
    )
    task = infer_task_from_text(joined_prompt)
    if task is not None:
        return task

    # Final fallback:
    # If mixed mode and still unknown, default based on presence/absence of object_direction.
    # If it isn't there, robot is safer.
    return "robot"


def get_fields_for_task(task: str) -> List[str]:
    return TASK_FIELDS.get(task, TASK_FIELDS["robot"])


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
    inputs = move_batch_to_device(inputs, device)

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
            "num_scored_samples": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
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
            "precision": safe_round(precision),
            "recall": safe_round(recall),
            "f1": safe_round(f1),
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        total_support += support

    k = len(labels)
    macro_p = macro_p / k if k > 0 else None
    macro_r = macro_r / k if k > 0 else None
    macro_f1 = macro_f1 / k if k > 0 else None

    weighted_p = weighted_p / total_support if total_support > 0 else None
    weighted_r = weighted_r / total_support if total_support > 0 else None
    weighted_f1 = weighted_f1 / total_support if total_support > 0 else None

    return {
        "num_scored_samples": n,
        "accuracy": safe_round(accuracy),
        "macro_precision": safe_round(macro_p),
        "macro_recall": safe_round(macro_r),
        "macro_f1": safe_round(macro_f1),
        "weighted_precision": safe_round(weighted_p),
        "weighted_recall": safe_round(weighted_r),
        "weighted_f1": safe_round(weighted_f1),
        "labels": per_label,
    }


def compute_hazard_present_binary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(results) == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": None, "recall": None, "f1": None,
        }

    tp = fp = fn = tn = 0

    for r in results:
        gt = r["ground_truth"]
        if r.get("ground_truth_parse_failed", False):
            continue

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
        "precision": safe_round(precision),
        "recall": safe_round(recall),
        "f1": safe_round(f1),
    }


def filter_results_by_task(results: List[Dict[str, Any]], task: Optional[str]) -> List[Dict[str, Any]]:
    if task is None:
        return results
    return [r for r in results if r.get("task") == task]


def compute_metrics_for_subset(results: List[Dict[str, Any]], subset_name: str) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "subset_name": subset_name,
            "total_samples": 0,
            "task_counts": {},
            "ground_truth_parse_failures": 0,
            "parse_failures": 0,
            "exact_match_accuracy": None,
            "relevant_field_accuracy": None,
            "relevant_field_correct": 0,
            "relevant_field_total": 0,
            "per_field_accuracy": {f: None for f in ALL_FIELDS},
            "per_field_metrics": {
                f: compute_classification_stats([], []) for f in ALL_FIELDS
            },
            "hazard_present_binary": compute_hazard_present_binary([]),
            "per_bucket_accuracy": {},
        }

    parse_failures = 0
    exact_match = 0
    gt_parse_failures = 0
    relevant_field_correct = 0
    relevant_field_total = 0

    bucket_correct: Dict[str, int] = defaultdict(int)
    bucket_total: Dict[str, int] = defaultdict(int)

    field_y_true = {f: [] for f in ALL_FIELDS}
    field_y_pred = {f: [] for f in ALL_FIELDS}

    task_counts = Counter(r.get("task", "unknown") for r in results)

    for r in results:
        gt = r["ground_truth"]
        pred = r.get("prediction_parsed")
        bucket = r.get("hard_negative_bucket") or "positive"
        task = r.get("task", "robot")
        task_fields = get_fields_for_task(task)


        bucket_total[bucket] += 1

        if r.get("ground_truth_parse_failed", False):
            gt_parse_failures += 1
            continue

        if pred is None:
            parse_failures += 1

        all_match = True

        for field in task_fields:
            gt_val = str(gt.get(field, "__MISSING__")).strip()
            pred_val = "__PARSE_FAIL__" if pred is None else str(pred.get(field, "__MISSING__")).strip()

            field_y_true[field].append(gt_val)
            field_y_pred[field].append(pred_val)

            relevant_field_total += 1
            if gt_val == pred_val:
                relevant_field_correct += 1
            else:
                all_match = False

        if all_match:
            exact_match += 1
            bucket_correct[bucket] += 1

    field_metrics = {
        field: compute_classification_stats(field_y_true[field], field_y_pred[field])
        for field in ALL_FIELDS
    }

    per_bucket = {}
    for bucket in sorted(bucket_total.keys()):
        c = bucket_correct.get(bucket, 0)
        t = bucket_total[bucket]
        per_bucket[bucket] = {
            "correct": c,
            "total": t,
            "accuracy": safe_round(c / t) if t > 0 else None,
        }

    return {
        "subset_name": subset_name,
        "total_samples": total,
        "task_counts": dict(task_counts),
        "ground_truth_parse_failures": gt_parse_failures,
        "parse_failures": parse_failures,
        "exact_match_accuracy": safe_round(exact_match / total) if total > 0 else None,
        "relevant_field_accuracy": safe_round(relevant_field_correct / relevant_field_total) if relevant_field_total > 0 else None,
        "relevant_field_correct": relevant_field_correct,
        "relevant_field_total": relevant_field_total,
        "per_field_accuracy": {
            field: field_metrics[field]["accuracy"] for field in ALL_FIELDS
        },
        "per_field_metrics": field_metrics,
        "hazard_present_binary": compute_hazard_present_binary(results),
        "per_bucket_accuracy": per_bucket,
    }


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined = compute_metrics_for_subset(results, "combined")
    forklift = compute_metrics_for_subset(filter_results_by_task(results, "forklift"), "forklift")
    robot = compute_metrics_for_subset(filter_results_by_task(results, "robot"), "robot")

    return {
        "combined": combined,
        "by_task": {
            "forklift": forklift,
            "robot": robot,
        },
    }


# -----------------------------------------------------------------------------
# Summary writing
# -----------------------------------------------------------------------------

def write_summary_markdown(all_results: List[Tuple[str, Dict[str, Any]]], out_path: str) -> None:
    lines: List[str] = []
    lines.append("# Evaluation Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # -------------------------------------------------------------------------
    # Overall model comparison
    # -------------------------------------------------------------------------
    lines.append("## Overall Comparison\n")
    lines.append(
        "| Model | Combined Exact | Combined Field Acc | Parse Fail | HP F1 | Forklift Exact | Robot Exact |"
    )
    lines.append(
        "|------|----------------|--------------------|------------|-------|----------------|-------------|"
    )
    for name, metrics in all_results:
        c = metrics["combined"]
        f = metrics["by_task"]["forklift"]
        r = metrics["by_task"]["robot"]

        lines.append(
            f"| {name} "
            f"| {fmt_metric(c.get('exact_match_accuracy'))} "
            f"| {fmt_metric(c.get('relevant_field_accuracy'))} "
            f"| {c.get('parse_failures', 0)} "
            f"| {fmt_metric(c.get('hazard_present_binary', {}).get('f1'))} "
            f"| {fmt_metric(f.get('exact_match_accuracy'))} "
            f"| {fmt_metric(r.get('exact_match_accuracy'))} |"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Combined per-field accuracy
    # -------------------------------------------------------------------------
    lines.append("## Combined Per-Field Accuracy\n")
    lines.append(
        "| Model | hazard_label | hazard_present | zone_relation | object_state | object_direction |"
    )
    lines.append(
        "|------|--------------|----------------|---------------|--------------|------------------|"
    )
    for name, metrics in all_results:
        pfa = metrics["combined"]["per_field_accuracy"]
        lines.append(
            f"| {name} "
            f"| {fmt_metric(pfa.get('hazard_label'))} "
            f"| {fmt_metric(pfa.get('hazard_present'))} "
            f"| {fmt_metric(pfa.get('zone_relation'))} "
            f"| {fmt_metric(pfa.get('object_state'))} "
            f"| {fmt_metric(pfa.get('object_direction'))} |"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Forklift per-field accuracy
    # -------------------------------------------------------------------------
    lines.append("## Forklift Per-Field Accuracy\n")
    lines.append(
        "| Model | hazard_label | hazard_present | zone_relation | object_state | object_direction |"
    )
    lines.append(
        "|------|--------------|----------------|---------------|--------------|------------------|"
    )
    for name, metrics in all_results:
        pfa = metrics["by_task"]["forklift"]["per_field_accuracy"]
        lines.append(
            f"| {name} "
            f"| {fmt_metric(pfa.get('hazard_label'))} "
            f"| {fmt_metric(pfa.get('hazard_present'))} "
            f"| {fmt_metric(pfa.get('zone_relation'))} "
            f"| {fmt_metric(pfa.get('object_state'))} "
            f"| {fmt_metric(pfa.get('object_direction'))} |"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Robot per-field accuracy
    # -------------------------------------------------------------------------
    lines.append("## Robot Per-Field Accuracy\n")
    lines.append(
        "| Model | hazard_label | hazard_present | zone_relation | object_state |"
    )
    lines.append(
        "|------|--------------|----------------|---------------|--------------|"
    )
    for name, metrics in all_results:
        pfa = metrics["by_task"]["robot"]["per_field_accuracy"]
        lines.append(
            f"| {name} "
            f"| {fmt_metric(pfa.get('hazard_label'))} "
            f"| {fmt_metric(pfa.get('hazard_present'))} "
            f"| {fmt_metric(pfa.get('zone_relation'))} "
            f"| {fmt_metric(pfa.get('object_state'))} |"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Combined per-field macro F1
    # -------------------------------------------------------------------------
    lines.append("## Combined Per-Field Macro F1\n")
    lines.append(
        "| Model | hazard_label | hazard_present | zone_relation | object_state | object_direction |"
    )
    lines.append(
        "|------|--------------|----------------|---------------|--------------|------------------|"
    )
    for name, metrics in all_results:
        pfm = metrics["combined"]["per_field_metrics"]
        lines.append(
            f"| {name} "
            f"| {fmt_metric(pfm.get('hazard_label', {}).get('macro_f1'))} "
            f"| {fmt_metric(pfm.get('hazard_present', {}).get('macro_f1'))} "
            f"| {fmt_metric(pfm.get('zone_relation', {}).get('macro_f1'))} "
            f"| {fmt_metric(pfm.get('object_state', {}).get('macro_f1'))} "
            f"| {fmt_metric(pfm.get('object_direction', {}).get('macro_f1'))} |"
        )
    lines.append("")

    # -------------------------------------------------------------------------
    # Binary confusion matrices
    # -------------------------------------------------------------------------
    lines.append("## Confusion Matrices (Combined hazard_present)\n")
    for name, metrics in all_results:
        hpb = metrics["combined"]["hazard_present_binary"]
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

    iterator = enumerate(test_data)
    pbar = None
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
        gt_parsed: Dict[str, Any] = {}
        inference_time = 0.0
        task = None
        task_fields: List[str] = []

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

            task = infer_task(
                example=example,
                prompt_messages=prompt_messages,
                gt_parsed=gt_parsed,
                task_mode=args.task_mode,
            )
            task_fields = get_fields_for_task(task)

            t_inf_start = time.time()
            pred_text = run_inference_single(
                model=model,
                processor=processor,
                prompt_messages=prompt_messages,
                num_frames=args.num_frames,
                max_new_tokens=args.max_new_tokens,
            )
            inference_time = time.time() - t_inf_start

            pred_parsed = canonicalize_prediction_dict(try_parse_json(pred_text))

        except Exception as e:
            pred_text = ""
            pred_parsed = None
            inference_time = 0.0
            if task is None:
                task = args.task_mode if args.task_mode != "both" else "robot"
                task_fields = get_fields_for_task(task)
            print(f"\n[ERROR] {eval_name} :: {sample_id} :: {type(e).__name__}: {e}")

        field_matches = {}
        for field in ALL_FIELDS:
            if field not in task_fields:
                field_matches[field] = None
                continue

            gt_val = gt_parsed.get(field, "__MISSING__")
            pred_val = "__PARSE_FAIL__" if pred_parsed is None else pred_parsed.get(field, "__MISSING__")
            field_matches[field] = (gt_val == pred_val)

        exact_match = all(v is True for v in field_matches.values() if v is not None)

        result = {
            "sample_id": sample_id,
            "task": task,
            "task_fields": task_fields,
            "ground_truth_text": gt_text,
            "ground_truth": gt_parsed,
            "ground_truth_parse_failed": gt_parse_failed,
            "prediction_raw": pred_text,
            "prediction_parsed": pred_parsed,
            "field_matches": field_matches,
            "exact_match": exact_match,
            "inference_time_sec": round(inference_time, 4),
            "hard_negative_bucket": meta.get("hard_negative_bucket"),
            "label_signature": meta.get("label_signature", ""),
            "meta": meta,
        }
        print(gt_text)
        print("")
        print(pred_parsed)
        sample_results.append(result)

        if not HAS_TQDM:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0.0
            gt_hp = gt_parsed.get("hazard_present", "?")
            pred_hp = "PARSE_FAIL" if pred_parsed is None else pred_parsed.get("hazard_present", "?")
            mark = "✓" if exact_match else "✗"
            print(
                f"[{i+1:3d}/{len(test_data)}] {mark} "
                f"TASK={task:>8s} GT={gt_hp:>3s} PRED={pred_hp:>10s} "
                f"({rate:.2f} it/s, ETA {eta:.0f}s) "
                f"— {sample_id[:60]}"
            )
        else:
            gt_hp = gt_parsed.get("hazard_present", "?")
            pred_hp = "PARSE_FAIL" if pred_parsed is None else pred_parsed.get("hazard_present", "?")
            pbar.set_postfix({
                "TASK": task,
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

    combined = metrics["combined"]
    forklift = metrics["by_task"]["forklift"]
    robot = metrics["by_task"]["robot"]

    print(f"[INFO] Combined exact match:       {fmt_metric(combined['exact_match_accuracy'])}")
    print(f"[INFO] Combined field accuracy:    {fmt_metric(combined['relevant_field_accuracy'])}")
    print(f"[INFO] Combined parse failures:    {combined['parse_failures']}")
    print(f"[INFO] Combined hazard_present F1: {fmt_metric(combined['hazard_present_binary']['f1'])}")

    print("[INFO] Combined per-field accuracy:")
    for field in ALL_FIELDS:
        print(f"  - {field:16s}: {fmt_metric(combined['per_field_accuracy'][field])}")

    print(f"[INFO] Forklift exact match:       {fmt_metric(forklift['exact_match_accuracy'])}")
    print(f"[INFO] Robot exact match:          {fmt_metric(robot['exact_match_accuracy'])}")

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
    print(f"[INFO] Task mode: {args.task_mode}")

    adapters = discover_adapters(args.adapter_dir, args.only_final, args.only_checkpoints)
    adapter_names = [name for name, _ in adapters]
    print(f"[INFO] Adapter evaluations queued: {adapter_names}")

    eval_dir = os.path.join(args.adapter_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    base_model, processor = load_base_model_and_processor(args, dtype)
    all_summaries: List[Tuple[str, Dict[str, Any]]] = []

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
        if torch.cuda.is_available():
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