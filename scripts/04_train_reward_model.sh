#!/usr/bin/env bash
# scripts/04_train_reward_model.sh
# Train reward model on human preference pairs.
set -e

CONFIG=${1:-"configs/reward_config.yaml"}
OUTPUT_DIR=${2:-"models/reward"}

echo "======================================"
echo " Stage 4: Reward Model Training"
echo " Config: $CONFIG"
echo "======================================"

# Validate preferences exist
PREF_DIR="data/preferences"
if [ -z "$(ls -A $PREF_DIR 2>/dev/null)" ]; then
  echo "❌ No preference data found in $PREF_DIR"
  echo "   Run scripts/03_collect_preferences.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python -m src.rlhf.reward_model \
  --config "$CONFIG"

echo ""
echo "✅ Reward model trained → $OUTPUT_DIR"
