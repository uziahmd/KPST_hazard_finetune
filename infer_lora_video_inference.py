#!/usr/bin/env python3
"""
Inference-only entrypoint for the LoRA multimodal hazard video pipeline.

Typical use:
python infer_lora_video_inference.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_both_aug/checkpoint-4796 \
  --video_dir test \
  --output_dir runs/qwen35_9b_both_aug_infer_run \
  --task_mode both \
  --device cuda:0

This script keeps chunk files by default so the overlay script can be run later
with a minimal command.
"""

from __future__ import annotations

import argparse

from infer_lora_video_overlay import default_chunk_workers, default_overlay_workers, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunking + LoRA inference only, and save JSON results for a later overlay pass."
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base multimodal model name or local path.")
    parser.add_argument("--adapter_dir", type=str, required=True, help="LoRA adapter directory.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for chunks, results, and logs.")
    parser.add_argument(
        "--task_mode",
        type=str,
        default="both",
        choices=["robot", "forklift", "both"],
        help="Which task videos to process.",
    )
    parser.add_argument(
        "--robot_prompt_file",
        type=str,
        default="prompts/robot_propmt_v1.txt",
        help="Prompt file for the robot task.",
    )
    parser.add_argument(
        "--fork_prompt_file",
        type=str,
        default="prompts/fork_prompt_v2.txt",
        help="Prompt file for the forklift task.",
    )
    parser.add_argument("--chunk_sec", type=float, default=5.0, help="Chunk length in seconds.")
    parser.add_argument("--num_frames", type=int, default=12, help="Frames sampled per chunk by the processor.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens per chunk.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stochastic generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--save_chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep intermediate chunk mp4 files. Defaults to on so overlay can be run later.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="",
        help="Optional path for the results JSON. Defaults to <output_dir>/results/inference_results.json.",
    )
    parser.add_argument(
        "--chunk_workers",
        type=int,
        default=default_chunk_workers(),
        help="Number of parallel threads for ffmpeg video chunking (one thread per video).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code through to Transformers model and processor loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        argparse.Namespace(
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            task_mode=args.task_mode,
            robot_prompt_file=args.robot_prompt_file,
            fork_prompt_file=args.fork_prompt_file,
            chunk_sec=args.chunk_sec,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            device=args.device,
            save_chunks=args.save_chunks,
            save_overlay=False,
            font_file="",
            chunk_workers=args.chunk_workers,
            overlay_workers=default_overlay_workers(),
            skip_inference=False,
            skip_overlay=True,
            results_json=args.results_json,
            trust_remote_code=args.trust_remote_code,
        )
    )


if __name__ == "__main__":
    main()
