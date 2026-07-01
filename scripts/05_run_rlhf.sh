#!/usr/bin/env bash
# scripts/05_run_rlhf.sh
# Run PPO RLHF loop.
set -e

CONFIG=${1:-"configs/rlhf_config.yaml"}
OUTPUT_DIR=${2:-"models/rlhf_final"}

echo "======================================"
echo " Stage 5: RLHF PPO Training"
echo " Config: $CONFIG"
echo "======================================"

mkdir -p "$OUTPUT_DIR"

python -m src.rlhf.ppo_trainer \
  --config "$CONFIG"

echo ""
echo "✅ RLHF training complete → $OUTPUT_DIR"
echo ""
echo "Next: evaluate with scripts/06_evaluate.sh"
