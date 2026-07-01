#!/usr/bin/env bash
# scripts/06_evaluate.sh
# Evaluate RLHF model vs SFT baseline.
set -e

DOMAIN=${1:-legal}
MODEL_DIR=${2:-"models/rlhf_final"}
BASELINE_DIR=${3:-"models/finetuned"}
REWARD_DIR=${4:-"models/reward"}
EVAL_SET=${5:-"data/processed/test.jsonl"}

echo "======================================"
echo " Stage 6: Evaluation"
echo " Model:    $MODEL_DIR"
echo " Baseline: $BASELINE_DIR"
echo " Domain:   $DOMAIN"
echo "======================================"

mkdir -p outputs

python -m src.evaluation.evaluator \
  --model_dir "$MODEL_DIR" \
  --eval_set "$EVAL_SET" \
  --domain "$DOMAIN" \
  --reward_model_path "$REWARD_DIR" \
  --compare_model "$BASELINE_DIR" \
  --output_path "outputs/eval_results.json"

echo ""
echo "✅ Evaluation complete → outputs/eval_results.json"
