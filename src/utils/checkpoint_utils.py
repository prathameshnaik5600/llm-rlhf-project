"""
src/utils/checkpoint_utils.py

Utilities for saving and loading model checkpoints, merging LoRA adapters,
and preparing models for deployment.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save_lora(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
):
    """
    Merge LoRA adapter weights into the base model and save as a full model.

    Why merge? 
    - Inference is faster without the adapter overhead
    - Easier to deploy (single model, no PEFT dependency)
    - Required for some serving frameworks (vLLM, llama.cpp)

    Note: merged model requires full precision — cannot be stored in 4-bit.
    Recommended: save in bfloat16 (~14 GB for 7B model).
    """
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # Load to CPU for merging (less VRAM)
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter: {lora_adapter_path}")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Merging LoRA weights into base model...")
    merged_model = model_with_lora.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    if push_to_hub and hub_repo_id:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_repo_id)

    logger.info("Merge complete!")
    return merged_model


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    path = Path(checkpoint_dir)
    checkpoints = sorted(
        [d for d in path.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    latest = str(checkpoints[-1])
    logger.info(f"Latest checkpoint: {latest}")
    return latest


def save_training_metadata(
    output_dir: str,
    config: dict,
    metrics: dict,
    stage: str,
):
    """Save training metadata alongside the model."""
    metadata = {
        "stage": stage,
        "config": config,
        "metrics": metrics,
    }
    out_path = Path(output_dir) / "training_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {out_path}")
