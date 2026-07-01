#!/usr/bin/env bash
# scripts/03_collect_preferences.sh
# Launch Gradio UI for human preference annotation.
set -e

MODEL_PATH=${1:-"models/finetuned"}
PROMPTS_PATH=${2:-"data/processed/rlhf_prompts.jsonl"}
OUTPUT_PATH=${3:-"data/preferences/train_preferences.jsonl"}
PORT=${4:-7860}

echo "======================================"
echo " Stage 3: Human Preference Collection"
echo " Model: $MODEL_PATH"
echo " Output: $OUTPUT_PATH"
echo "======================================"

mkdir -p data/preferences

# If you already have preference data, skip this step.
# Minimum recommended: 1,000 preference pairs for a useful reward model.
# Good target: 5,000–10,000 pairs.

python -m src.data.preference_collector \
  --model_path "$MODEL_PATH" \
  --prompts_path "$PROMPTS_PATH" \
  --output_path "$OUTPUT_PATH" \
  --port "$PORT"

echo ""
echo "✅ Preference collection complete → $OUTPUT_PATH"
