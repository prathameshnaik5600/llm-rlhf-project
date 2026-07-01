"""
src/rlhf/reward_model.py

Trains a reward model on (prompt, chosen, rejected) triplets.
TRL 1.5+ compatible — uses RewardTrainer + RewardConfig.

Usage:
    python -m src.rlhf.reward_model --config configs/reward_config.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import RewardConfig, RewardTrainer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preferences(preferences_path: str, max_samples: int = 10_000) -> List[dict]:
    path = Path(preferences_path)
    files = list(path.glob("*.jsonl")) if path.is_dir() else [path]
    examples = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if all(k in obj for k in ("prompt", "chosen", "rejected")):
                        examples.append(obj)
                except json.JSONDecodeError:
                    pass
                if len(examples) >= max_samples:
                    break
    logger.info(f"Loaded {len(examples)} preference examples")
    return examples


def build_preference_dataset(examples: List[dict], tokenizer: AutoTokenizer, max_length: int = 1024) -> Dataset:
    """
    RewardTrainer expects columns: input_ids_chosen, attention_mask_chosen,
    input_ids_rejected, attention_mask_rejected (as lists, not tensors).
    """
    rows = []
    for ex in examples:
        chosen_text = ex["prompt"] + ex["chosen"]
        rejected_text = ex["prompt"] + ex["rejected"]

        enc_c = tokenizer(chosen_text, max_length=max_length, truncation=True, padding="max_length")
        enc_r = tokenizer(rejected_text, max_length=max_length, truncation=True, padding="max_length")

        rows.append({
            "input_ids_chosen": enc_c["input_ids"],
            "attention_mask_chosen": enc_c["attention_mask"],
            "input_ids_rejected": enc_r["input_ids"],
            "attention_mask_rejected": enc_r["attention_mask"],
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reward_model(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg_path = Path(config_path).parent / "model_config.yaml"
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)

    set_seed(42)
    t = cfg["training"]
    rm_cfg = cfg["reward_model"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(rm_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model — sequence classification (reward = scalar logit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        rm_cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=model_cfg["lora"]["r"],
        lora_alpha=model_cfg["lora"]["lora_alpha"],
        lora_dropout=model_cfg["lora"]["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=model_cfg["lora"].get("reward_target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data
    data = cfg["data"]
    examples = load_preferences(data["preferences_path"], max_samples=data.get("max_samples", 10_000))

    n_eval = max(1, int(len(examples) * 0.1))
    train_ds = build_preference_dataset(examples[n_eval:], tokenizer, t["max_length"])
    eval_ds = build_preference_dataset(examples[:n_eval], tokenizer, t["max_length"])

    # Config
    reward_config = RewardConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        max_grad_norm=t["max_grad_norm"],
        bf16=t["bf16"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        logging_steps=t["logging_steps"],
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "reward-model"),
        max_length=t["max_length"],
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    logger.info("Training reward model...")
    trainer.train()

    logger.info(f"Saving to {t['output_dir']}")
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    logger.info("Reward model training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reward_config.yaml")
    args = parser.parse_args()
    train_reward_model(args.config)
