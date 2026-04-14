#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------
# Config
# -----------------------------
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

# -----------------------------
# Helpers
# -----------------------------
log() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

garbage_clean() {
  log "Running garbage clean"

  python - <<'PY'
import gc
import os
import shutil

print("[cleanup] Python GC collect...")
gc.collect()

try:
    import torch
    if torch.cuda.is_available():
        print("[cleanup] Clearing CUDA cache...")
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"[cleanup] torch.cuda.ipc_collect() skipped: {e}")
    else:
        print("[cleanup] CUDA not available.")
except Exception as e:
    print(f"[cleanup] Torch cleanup skipped: {e}")

for cache_dir in ["__pycache__", ".pytest_cache"]:
    if os.path.isdir(cache_dir):
        print(f"[cleanup] Removing {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

print("[cleanup] Done.")
PY

  sync || true
  sleep 2
}

run_cmd() {
  echo
  echo "[RUN] $*"
  "$@"
}

# -----------------------------
# Conda init
# -----------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda command not found. Run this from a shell where conda is available."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# -----------------------------
# Phase 1: Qwen env
# -----------------------------
log "Activating conda env: hazard_finetune_v100"
conda deactivate >/dev/null 2>&1 || true
conda activate hazard_finetune_v100

log "Training Qwen 3.5 LoRA with explicit target modules"
run_cmd python train_qwen35_video_lora.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file vlm_dataset_both_aug/val_chat.jsonl \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
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
  --seed 3407

garbage_clean

log "Training Qwen 3.5 LoRA with all-linear target modules"
run_cmd python train_qwen35_video_lora.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file vlm_dataset_both_aug/val_chat.jsonl \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --output_dir runs/qwen35_9b_both_aug_all \
  --num_frames 12 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules all-linear \
  --gradient_checkpointing \
  --use_fp16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 20 \
  --attn_implementation eager \
  --seed 3407

garbage_clean

log "Evaluating Qwen checkpoints: runs/qwen35_9b_both_aug"
run_cmd python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_both_aug \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --project_root . \
  --task_mode robot \
  --use_fp16

log "Evaluating Qwen checkpoints: runs/qwen35_9b_both_aug_all"
run_cmd python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_both_aug_all \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --project_root . \
  --task_mode robot \
  --use_fp16

garbage_clean

log "Deactivating hazard_finetune_v100"
conda deactivate

# -----------------------------
# Phase 2: Gemma env
# -----------------------------
log "Activating conda env: gemma4"
conda activate gemma4

log "Training Gemma 4"
run_cmd python train_gemma4.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file vlm_dataset_both_aug/val_chat.jsonl \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --project_root . \
  --model_name_or_path google/gemma-4-E4B-it \
  --output_dir runs/gemma4_e4b_video_lora \
  --video_load_backend torchvision \
  --num_frames 8 \
  --fps 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 20 \
  --save_total_limit 20 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --gradient_checkpointing \
  --use_fp16

garbage_clean

log "Deactivating gemma4"
conda deactivate

log "Pipeline completed successfully"