#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  echo "Cleaning up Python + CUDA memory..."
  python - <<'PY'
import gc

try:
    import torch
except Exception:
    torch = None

gc.collect()

if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("Cleanup done.")
PY
}

echo "=== Run 1: both ==="
python infer_lora_video_overlay.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_both_aug/checkpoint-4796 \
  --video_dir test \
  --output_dir runs/qwen35_9b_both_aug_infer_overlay_run \
  --task_mode both \
  --robot_prompt_file prompts/robot_propmt_v1.txt \
  --fork_prompt_file prompts/fork_prompt_v2.txt \
  --chunk_sec 5 \
  --num_frames 12 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --no-do_sample \
  --device cuda:0 \
  --save_chunks \
  --save_overlay

cleanup

echo "=== Run 2: forklift ==="
python infer_lora_video_overlay.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_v3/checkpoint-960 \
  --video_dir test \
  --output_dir runs/qwen35_9b_v3_infer_overlay_run \
  --task_mode forklift \
  --robot_prompt_file prompts/robot_propmt_v1.txt \
  --fork_prompt_file prompts/fork_prompt_v2.txt \
  --chunk_sec 5 \
  --num_frames 12 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --no-do_sample \
  --device cuda:0 \
  --save_chunks \
  --save_overlay

cleanup

echo "=== Run 3: robot ==="
python infer_lora_video_overlay.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_robot/checkpoint-510 \
  --video_dir test \
  --output_dir runs/qwen35_9b_robot_infer_overlay_run \
  --task_mode robot \
  --robot_prompt_file prompts/robot_propmt_v1.txt \
  --fork_prompt_file prompts/fork_prompt_v2.txt \
  --chunk_sec 5 \
  --num_frames 12 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --no-do_sample \
  --device cuda:0 \
  --save_chunks \
  --save_overlay

cleanup

echo "All runs completed."