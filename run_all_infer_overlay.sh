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


echo "=== Run 1: train ==="
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

cleanup

echo "=== Run 2: infer ==="

python infer_lora_video_inference.py \
  --base_model Qwen/Qwen3.5-9B \
  --adapter_dir runs/qwen35_9b_both_aug_2/checkpoint-4796 \
  --video_dir test \
  --output_dir runs/qwen35_9b_both_aug_infer_overlay_run_2 \
  --task_mode both \
  --robot_prompt_file prompts/robot_propmt_v1.txt \
  --fork_prompt_file prompts/fork_prompt_v2.txt \
  --chunk_sec 5 \
  --num_frames 12 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --no-do_sample \
  --device cuda:0 \
  --save_chunks

cleanup

python render_lora_video_overlay.py runs/qwen35_9b_both_aug_infer_overlay_run_2 --overlay_workers 4 --save_overlay


cleanup

echo "All runs completed."