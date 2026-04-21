#!/usr/bin/env python3
"""
Overlay-only entrypoint for the LoRA multimodal hazard video pipeline.

Typical use:
python render_lora_video_overlay.py runs/qwen35_9b_both_aug_infer_run

It will look for:
- <run_dir>/results/inference_results.json
- chunk paths recorded inside that JSON
"""

from __future__ import annotations

import argparse
from pathlib import Path

from infer_lora_video_overlay import default_overlay_workers, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render overlays and concatenate annotated videos from an existing inference results JSON."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="",
        help="Run directory containing results/, overlays/, and usually chunks/.",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="",
        help="Optional explicit results JSON path. Defaults to <run_dir>/results/inference_results.json.",
    )
    parser.add_argument(
        "--overlay_workers",
        type=int,
        default=default_overlay_workers(),
        help="Number of parallel worker processes for overlay rendering.",
    )
    parser.add_argument(
        "--font_file",
        type=str,
        default="",
        help="Optional font file for ffmpeg drawtext. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--save_overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate annotated chunk mp4 files after concatenation.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.results_json:
        results_json = Path(args.results_json).resolve()
        if results_json.parent.name == "results":
            run_dir = results_json.parent.parent.resolve()
        else:
            run_dir = results_json.parent.resolve()
    else:
        raise ValueError("Provide either run_dir or --results_json.")

    if args.results_json:
        results_json = Path(args.results_json).resolve()
    else:
        results_json = run_dir / "results" / "inference_results.json"

    return run_dir, results_json


def main() -> None:
    args = parse_args()
    run_dir, results_json = resolve_paths(args)

    run_pipeline(
        argparse.Namespace(
            base_model="",
            adapter_dir="",
            video_dir=str(run_dir),
            output_dir=str(run_dir),
            task_mode="both",
            robot_prompt_file="prompts/robot_propmt_v1.txt",
            fork_prompt_file="prompts/fork_prompt_v2.txt",
            chunk_sec=5.0,
            num_frames=12,
            video_longest_edge=560,
            video_shortest_edge=308,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            device="auto",
            save_chunks=True,
            save_overlay=args.save_overlay,
            font_file=args.font_file,
            chunk_workers=1,
            overlay_workers=args.overlay_workers,
            skip_inference=True,
            skip_overlay=False,
            results_json=str(results_json),
            trust_remote_code=True,
            attn_implementation="",
        )
    )


if __name__ == "__main__":
    main()
