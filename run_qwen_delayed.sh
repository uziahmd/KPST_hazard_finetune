#!/usr/bin/env bash
set -Eeuo pipefail

# -----------------------------
# Config
# -----------------------------
ROOT_DIR="${ROOT_DIR:-$(pwd)}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-10000}"   # 1 hour default
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
# Startup delay
# -----------------------------
log "Delaying start for ${START_DELAY_SECONDS} seconds"
sleep "$START_DELAY_SECONDS"

# -----------------------------
# Phase 1: Qwen env
# -----------------------------
log "Activating conda env: hazard_finetune_v100"
conda deactivate >/dev/null 2>&1 || true
conda activate hazard_finetune_v100

log "Training Qwen 3.5 LoRA with explicit target modules"
run_cmd python train_qwen35_video_lora.py \
  --train_file vlm_dataset_both_aug/train_chat.jsonl \
  --val_file   vlm_dataset_both_aug/val_chat.jsonl \
  --test_file  vlm_dataset_both_aug/test_chat.jsonl \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --output_dir runs/qwen35_9b_both_aug_all \
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
  --lora_target_modules "all-linear" \
  --gradient_checkpointing \
  --use_fp16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 9 \
  --attn_implementation sdpa \
  --seed 3407 \
  --pause_on_interrupt

garbage_clean

log "Evaluating Qwen checkpoints: runs/qwen35_9b_both_aug_all"
run_cmd python eval_lora_checkpoints.py \
  --adapter_dir runs/qwen35_9b_both_aug_all \
  --test_file vlm_dataset_both_aug/test_chat.jsonl \
  --project_root . \
  --task_mode both \
  --use_fp16

garbage_clean

log "Deactivating hazard_finetune_v100"
conda deactivate