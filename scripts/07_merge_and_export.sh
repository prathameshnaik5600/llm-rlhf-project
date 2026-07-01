#!/usr/bin/env bash
# scripts/07_merge_and_export.sh
# Merge LoRA adapters into base model weights for deployment.
set -e

BASE_MODEL=${1:-"mistralai/Mistral-7B-v0.1"}
ADAPTER_PATH=${2:-"models/rlhf_final"}
OUTPUT_PATH=${3:-"models/merged_final"}
HUB_REPO=${4:-""}

echo "======================================"
echo " Stage 7: Merge LoRA + Export"
echo " Base:    $BASE_MODEL"
echo " Adapter: $ADAPTER_PATH"
echo " Output:  $OUTPUT_PATH"
echo "======================================"

PUSH_FLAG=""
HUB_FLAG=""
if [ -n "$HUB_REPO" ]; then
  PUSH_FLAG="--push_to_hub"
  HUB_FLAG="--hub_repo_id $HUB_REPO"
fi

python -c "
from src.utils.checkpoint_utils import merge_and_save_lora
merge_and_save_lora(
    base_model_name='$BASE_MODEL',
    lora_adapter_path='$ADAPTER_PATH',
    output_path='$OUTPUT_PATH',
    push_to_hub=$([ -n '$HUB_REPO' ] && echo 'True' || echo 'False'),
    hub_repo_id='$HUB_REPO' if '$HUB_REPO' else None,
)
"

echo ""
echo "✅ Merged model saved → $OUTPUT_PATH"
echo "   Ready for vLLM, llama.cpp, or HuggingFace inference."
