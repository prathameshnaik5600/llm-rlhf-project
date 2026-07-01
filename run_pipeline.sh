#!/usr/bin/env bash
# run_pipeline.sh — Single command to run the full RLHF pipeline
# Usage:
#   ./run_pipeline.sh                         # uses defaults
#   ./run_pipeline.sh legal mistralai/Mistral-7B-v0.1
#   DEMO=1 ./run_pipeline.sh                  # use synthetic demo data (no GPU needed for data)
set -e

DOMAIN=${1:-legal}
MODEL=${2:-"mistralai/Mistral-7B-v0.1"}
DEMO=${DEMO:-0}

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${YELLOW}━━━ $1 ━━━${NC}"; }
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }

echo "=================================================="
echo " Domain-Specific LLM Fine-Tuning + RLHF Pipeline"
echo " Domain: $DOMAIN | Model: $MODEL"
echo "=================================================="

# ── Stage 0: Generate demo data (if DEMO=1 or no data exists) ────────
if [ "$DEMO" = "1" ] || [ ! -f "data/processed/train.jsonl" ]; then
  step "Stage 0: Generating demo dataset"
  python3 scripts/generate_demo_data.py
  # Copy sample preferences for reward model training
  mkdir -p data/preferences
  cp data/preferences/sample_preferences.jsonl data/preferences/train_preferences.jsonl 2>/dev/null || true
  ok "Demo data ready"
else
  step "Stage 1: Preparing real domain data"
  bash scripts/01_prepare_data.sh "$DOMAIN"
  ok "Data prepared"
fi

# ── Stage 2: SFT ─────────────────────────────────────────────────────
step "Stage 2: SFT Training (LoRA/QLoRA)"
bash scripts/02_run_sft.sh "$MODEL" data/processed models/finetuned
ok "SFT complete → models/finetuned"

# ── Stage 3: Preference data ─────────────────────────────────────────
if [ ! -f "data/preferences/train_preferences.jsonl" ]; then
  step "Stage 3: Collecting Human Preferences (launching UI)"
  echo "→ Open http://localhost:7860 to annotate preferences"
  echo "→ Minimum: 200 pairs. Press Ctrl+C when done."
  bash scripts/03_collect_preferences.sh models/finetuned data/processed/rlhf_prompts.jsonl
else
  step "Stage 3: Preference data found — skipping collection"
  ok "Using existing: data/preferences/train_preferences.jsonl"
fi

# ── Stage 4: Reward model ─────────────────────────────────────────────
step "Stage 4: Training Reward Model"
bash scripts/04_train_reward_model.sh configs/reward_config.yaml models/reward
ok "Reward model → models/reward"

# ── Stage 5: RLHF ────────────────────────────────────────────────────
step "Stage 5: RLHF Training (RLOO)"
bash scripts/05_run_rlhf.sh configs/rlhf_config.yaml models/rlhf_final
ok "RLHF complete → models/rlhf_final"

# ── Stage 6: Evaluate ────────────────────────────────────────────────
step "Stage 6: Evaluation"
bash scripts/06_evaluate.sh "$DOMAIN" models/rlhf_final models/finetuned models/reward data/processed/test.jsonl
ok "Evaluation → outputs/eval_results.json"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo -e "${GREEN} Pipeline complete!${NC}"
echo " Models:     models/rlhf_final/"
echo " Eval:       outputs/eval_results.json"
echo " Inference:  python -m src.inference --model_path models/rlhf_final --interactive"
echo "=================================================="
