#!/usr/bin/env python3
"""
Evaluate the final LoRA adapter and each training checkpoint on the test set.

Usage (on V100):
    python eval_lora_checkpoints.py \
        --adapter_dir  runs/qwen35_9b_video_lora \
        --test_file    vlm_dataset/test_chat.jsonl \
        --load_in_4bit \
        --use_fp16

The script will:
  1. Discover the final adapter  (adapter_dir itself).
  2. Discover every checkpoint-* subdirectory.
  3. Evaluate the FINAL adapter first.
  4. Evaluate each checkpoint in numerical order (lowest → highest step).
  5. Print a live progress bar and per-sample predictions.
  6. Write per-adapter results to  <adapter_dir>/eval_results/<name>.json
  7. Write a combined summary to   <adapter_dir>/eval_results/summary.md

Metrics computed:
  - Overall accuracy (exact JSON match).
  - Per-field accuracy: hazard_label, hazard_present, zone_relation,
    object_state, object_direction.
  - hazard_present binary: precision, recall, F1 (positive = "yes").
  - Confusion matrix for hazard_present.
  - Per-hard-negative-bucket accuracy.
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

# ── Fields we care about (order matters for the table) ───────────────────────
EVAL_FIELDS = [
    "hazard_label",
    "hazard_present",
    "zone_relation",
    "object_state",
    "object_direction",
]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LoRA checkpoints on the test set.")
    p.add_argument("--adapter_dir", type=str, default="runs/qwen35_9b_video_lora",
                   help="Directory containing the final adapter AND checkpoint-* subdirs.")
    p.add_argument("--test_file", type=str, default="vlm_dataset/test_chat.jsonl")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-9B",
                   help="Base model hub id / local path.")
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens to generate. Set generously to capture full JSON.")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="eager")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--only_final", action="store_true",
                   help="Evaluate only the final adapter, skip checkpoints.")
    p.add_argument("--only_checkpoints", type=str, default=None,
                   help="Comma-separated checkpoint numbers to evaluate, e.g. '26,52'.")
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def select_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_bf16:
        return torch.bfloat16
    if args.use_fp16:
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


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


def extract_assistant_text(message: Dict[str, Any]) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts).strip()


def try_parse_json(text: str) -> Optional[Dict[str, str]]:
    """Best-effort parse of the model output into a dict."""
    text = text.strip()
    # Strip markdown fences if present.
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
    # Try to find a JSON object in the text.
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


# ─── Model Management ───────────────────────────────────────────────────────

def load_base_model_and_processor(args: argparse.Namespace, dtype: torch.dtype):
    """Load the base model once (expensive). Returns model, processor."""
    print(f"[INFO] Loading base model: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
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
    else:
        model_load_kwargs["dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs,
    )
    model.config.use_cache = True
    print(f"[INFO] Base model loaded. dtype={dtype}")
    return model, processor


def load_adapter(base_model, adapter_path: str):
    """Wrap the base model with a LoRA adapter."""
    print(f"[INFO] Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def unload_adapter(model):
    """Remove the LoRA adapter from the base model."""
    if hasattr(model, "unload"):
        model.unload()
    elif hasattr(model, "base_model"):
        pass  # PeftModel.unload() not available in all versions.
    gc.collect()
    torch.cuda.empty_cache()


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference_single(
    model,
    processor,
    example: Dict[str, Any],
    num_frames: int,
    max_new_tokens: int,
) -> str:
    """Run inference on a single test example. Returns raw decoded text."""
    prompt_messages = copy.deepcopy(example["messages"][:-1])

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
    return prediction


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics from a list of per-sample result dicts."""
    total = len(results)
    if total == 0:
        return {"error": "no samples"}

    # ── Per-field accuracy ────────────────────────────────────────────────────
    field_correct = Counter()
    exact_match = 0
    parse_failures = 0

    # ── hazard_present binary ─────────────────────────────────────────────────
    tp = fp = fn = tn = 0

    # ── Per-bucket accuracy ───────────────────────────────────────────────────
    bucket_correct: Dict[str, int] = defaultdict(int)
    bucket_total: Dict[str, int] = defaultdict(int)

    for r in results:
        gt = r["ground_truth"]
        pred_parsed = r.get("prediction_parsed")

        if pred_parsed is None:
            parse_failures += 1
            # Count as all wrong.
            bucket_key = r.get("hard_negative_bucket", "unknown") or "positive"
            bucket_total[bucket_key] += 1
            # hazard_present: treat parse failure as predicting opposite.
            if gt.get("hazard_present") == "yes":
                fn += 1
            else:
                fp += 1
            continue

        bucket_key = r.get("hard_negative_bucket") or "positive"
        bucket_total[bucket_key] += 1

        all_match = True
        for field in EVAL_FIELDS:
            gt_val = gt.get(field, "")
            pred_val = pred_parsed.get(field, "")
            if gt_val == pred_val:
                field_correct[field] += 1
            else:
                all_match = False

        if all_match:
            exact_match += 1
            bucket_correct[bucket_key] += 1

        # Binary confusion matrix for hazard_present.
        gt_hp = gt.get("hazard_present", "no")
        pred_hp = pred_parsed.get("hazard_present", "no")
        if gt_hp == "yes" and pred_hp == "yes":
            tp += 1
        elif gt_hp == "no" and pred_hp == "yes":
            fp += 1
        elif gt_hp == "yes" and pred_hp == "no":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_bucket = {}
    for bucket in sorted(bucket_total.keys()):
        c = bucket_correct.get(bucket, 0)
        t = bucket_total[bucket]
        per_bucket[bucket] = {"correct": c, "total": t, "accuracy": round(c / t, 4) if t > 0 else 0.0}

    return {
        "total_samples": total,
        "parse_failures": parse_failures,
        "exact_match_accuracy": round(exact_match / total, 4),
        "per_field_accuracy": {
            f: round(field_correct[f] / total, 4) for f in EVAL_FIELDS
        },
        "hazard_present_binary": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_bucket_accuracy": per_bucket,
    }


# ─── Summary Writer ─────────────────────────────────────────────────────────

def write_summary_markdown(all_results: List[Tuple[str, Dict]], out_path: str) -> None:
    """Write a combined summary comparing all adapters."""
    lines: List[str] = []
    lines.append("# Evaluation Summary\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Overview table ────────────────────────────────────────────────────────
    lines.append("## Overall Comparison\n")
    header = "| Adapter | Exact Match | hazard_label | hazard_present | zone_relation | object_state | object_direction | HP Precision | HP Recall | HP F1 | Parse Fail |"
    sep =    "|---------|-------------|--------------|----------------|---------------|--------------|------------------|-------------|-----------|-------|------------|"
    lines.append(header)
    lines.append(sep)

    for name, metrics in all_results:
        pf = metrics.get("per_field_accuracy", {})
        hpb = metrics.get("hazard_present_binary", {})
        row = (
            f"| {name} "
            f"| {metrics.get('exact_match_accuracy', 0):.4f} "
            f"| {pf.get('hazard_label', 0):.4f} "
            f"| {pf.get('hazard_present', 0):.4f} "
            f"| {pf.get('zone_relation', 0):.4f} "
            f"| {pf.get('object_state', 0):.4f} "
            f"| {pf.get('object_direction', 0):.4f} "
            f"| {hpb.get('precision', 0):.4f} "
            f"| {hpb.get('recall', 0):.4f} "
            f"| {hpb.get('f1', 0):.4f} "
            f"| {metrics.get('parse_failures', 0)} |"
        )
        lines.append(row)

    lines.append("")

    # ── Confusion matrices ────────────────────────────────────────────────────
    lines.append("## Confusion Matrices (hazard_present)\n")
    for name, metrics in all_results:
        hpb = metrics.get("hazard_present_binary", {})
        lines.append(f"### {name}\n")
        lines.append("| | Pred YES | Pred NO |")
        lines.append("|---|---------|---------|")
        lines.append(f"| **GT YES** | {hpb.get('tp', 0)} | {hpb.get('fn', 0)} |")
        lines.append(f"| **GT NO**  | {hpb.get('fp', 0)} | {hpb.get('tn', 0)} |")
        lines.append("")

    # ── Per-bucket ────────────────────────────────────────────────────────────
    lines.append("## Per-Bucket Exact Match Accuracy\n")

    # Collect all buckets across all adapters.
    all_buckets = set()
    for _, metrics in all_results:
        all_buckets.update(metrics.get("per_bucket_accuracy", {}).keys())
    all_buckets = sorted(all_buckets)

    if all_buckets:
        header = "| Adapter | " + " | ".join(all_buckets) + " |"
        sep = "|---------|" + "|".join(["------" for _ in all_buckets]) + "|"
        lines.append(header)
        lines.append(sep)
        for name, metrics in all_results:
            pb = metrics.get("per_bucket_accuracy", {})
            cells = []
            for b in all_buckets:
                info = pb.get(b, {})
                acc = info.get("accuracy", 0)
                tot = info.get("total", 0)
                cells.append(f"{acc:.4f} ({tot})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[INFO] Summary written to: {out_path}")


# ─── Discovery ───────────────────────────────────────────────────────────────

def discover_adapters(adapter_dir: str, only_final: bool, only_checkpoints: Optional[str]) -> List[Tuple[str, str]]:
    """
    Return a list of (name, path) for adapters to evaluate.
    The final adapter is always first if present.
    """
    adapters: List[Tuple[str, str]] = []

    # Final adapter is the adapter_dir itself.
    final_config = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(final_config):
        adapters.append(("final", adapter_dir))
    else:
        print(f"[WARN] No adapter_config.json in {adapter_dir} — skipping 'final'.")

    if only_final:
        return adapters

    # Discover checkpoints.
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


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = select_dtype(args)

    # ── Load test data ────────────────────────────────────────────────────────
    if not os.path.isfile(args.test_file):
        print(f"[ERROR] Test file not found: {args.test_file}")
        sys.exit(1)

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"[INFO] Loaded {len(test_data)} test samples from {args.test_file}")

    # ── Discover adapters ─────────────────────────────────────────────────────
    adapters = discover_adapters(args.adapter_dir, args.only_final, args.only_checkpoints)
    if not adapters:
        print("[ERROR] No adapters found to evaluate.")
        sys.exit(1)
    print(f"[INFO] Will evaluate {len(adapters)} adapter(s): {[n for n, _ in adapters]}")

    # ── Load base model once ──────────────────────────────────────────────────
    base_model, processor = load_base_model_and_processor(args, dtype)

    # ── Output dir ────────────────────────────────────────────────────────────
    eval_dir = os.path.join(args.adapter_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    all_summaries: List[Tuple[str, Dict]] = []

    for adapter_idx, (adapter_name, adapter_path) in enumerate(adapters):
        print(f"\n{'='*70}")
        print(f"  Evaluating [{adapter_idx+1}/{len(adapters)}]: {adapter_name}")
        print(f"  Path: {adapter_path}")
        print(f"{'='*70}\n")

        # ── Load adapter ──────────────────────────────────────────────────────
        model = load_adapter(base_model, adapter_path)

        sample_results: List[Dict[str, Any]] = []
        t_start = time.time()

        for i, example in enumerate(test_data):
            sample_id = example.get("sample_id", f"sample_{i}")
            gt_text = extract_assistant_text(example["messages"][-1])
            gt_parsed = try_parse_json(gt_text)
            meta = example.get("meta", {})

            # Run inference.
            try:
                pred_text = run_inference_single(
                    model, processor, example,
                    num_frames=args.num_frames,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(test_data)}] {sample_id} — ERROR: {e}")
                pred_text = ""

            pred_parsed = try_parse_json(pred_text)

            # Record.
            result = {
                "sample_id": sample_id,
                "ground_truth": gt_parsed or {},
                "prediction_raw": pred_text,
                "prediction_parsed": pred_parsed,
                "hard_negative_bucket": meta.get("hard_negative_bucket"),
                "label_signature": meta.get("label_signature", ""),
            }
            sample_results.append(result)

            # Live log.
            gt_hp = (gt_parsed or {}).get("hazard_present", "?")
            pred_hp = (pred_parsed or {}).get("hazard_present", "PARSE_FAIL")
            match = "✓" if gt_parsed == pred_parsed else "✗"
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:3d}/{len(test_data)}] {match} "
                f"GT={gt_hp:3s} PRED={pred_hp:10s} "
                f"({rate:.2f} it/s, ETA {eta:.0f}s) "
                f"— {sample_id[:60]}"
            )

        elapsed_total = time.time() - t_start
        print(f"\n  Inference completed in {elapsed_total:.1f}s ({len(test_data)/elapsed_total:.2f} samples/s)")

        # ── Compute metrics ───────────────────────────────────────────────────
        metrics = compute_metrics(sample_results)
        metrics["adapter_name"] = adapter_name
        metrics["adapter_path"] = adapter_path
        metrics["inference_time_sec"] = round(elapsed_total, 2)

        # Save per-adapter JSON.
        result_path = os.path.join(eval_dir, f"{adapter_name}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "per_sample": sample_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to: {result_path}")

        # Print quick summary.
        print(f"\n  === {adapter_name} Summary ===")
        print(f"  Exact match:     {metrics['exact_match_accuracy']:.4f}")
        for field in EVAL_FIELDS:
            print(f"  {field:20s}: {metrics['per_field_accuracy'][field]:.4f}")
        hpb = metrics["hazard_present_binary"]
        print(f"  HP Precision:    {hpb['precision']:.4f}")
        print(f"  HP Recall:       {hpb['recall']:.4f}")
        print(f"  HP F1:           {hpb['f1']:.4f}")
        print(f"  Parse failures:  {metrics['parse_failures']}")

        all_summaries.append((adapter_name, metrics))

        # ── Unload adapter to free VRAM for next one ──────────────────────────
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Write combined summary ────────────────────────────────────────────────
    summary_path = os.path.join(eval_dir, "summary.md")
    write_summary_markdown(all_summaries, summary_path)

    # Also write a machine-readable combined JSON.
    combined_path = os.path.join(eval_dir, "all_metrics.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(
            {name: metrics for name, metrics in all_summaries},
            f, ensure_ascii=False, indent=2,
        )
    print(f"[INFO] Combined metrics written to: {combined_path}")
    print("\n[DONE] All evaluations complete.")


if __name__ == "__main__":
    main()
