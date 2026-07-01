#!/usr/bin/env bash
# scripts/01_prepare_data.sh
# Prepares domain-specific instruction dataset.
set -e

DOMAIN=${1:-legal}
MAX_SAMPLES=${2:-50000}
OUTPUT_DIR="data/processed"

echo "======================================"
echo " Stage 1: Data Preparation"
echo " Domain: $DOMAIN | Max samples: $MAX_SAMPLES"
echo "======================================"

mkdir -p data/raw data/processed data/preferences

python -m src.data.dataset_builder \
  --domain "$DOMAIN" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --seed 42

echo ""
echo "✅ Data prepared → $OUTPUT_DIR"
echo "   Files: train.jsonl, validation.jsonl, test.jsonl, rlhf_prompts.jsonl"
