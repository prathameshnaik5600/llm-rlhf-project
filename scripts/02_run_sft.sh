#!/usr/bin/env bash
# scripts/02_run_sft.sh
# LoRA/QLoRA supervised fine-tuning.
set -e

MODEL=${1:-"mistralai/Mistral-7B-v0.1"}
DATASET_PATH=${2:-"data/processed"}
OUTPUT_DIR=${3:-"models/finetuned"}

echo "======================================"
echo " Stage 2: SFT Training (LoRA/QLoRA)"
echo " Model: $MODEL"
echo "======================================"

mkdir -p "$OUTPUT_DIR"

# Optional: login to WandB
# wandb login

python -m src.training.sft_trainer \
  --config configs/sft_config.yaml \
  --model_name "$MODEL" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --use_4bit

echo ""
echo "✅ SFT complete → $OUTPUT_DIR"
