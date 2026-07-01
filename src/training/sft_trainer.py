"""
src/training/sft_trainer.py

Supervised Fine-Tuning (SFT) with LoRA/QLoRA — TRL 1.5+ compatible.

Usage:
    python -m src.training.sft_trainer --config configs/sft_config.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, use_4bit: bool = False) -> AutoModelForCausalLM:
    if use_4bit:
        logger.info("Loading model with 4-bit NF4 quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    return model


def apply_lora(model: AutoModelForCausalLM, lora_cfg: dict) -> AutoModelForCausalLM:
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_and_prepare_dataset(dataset_path: str, text_field: str = "text"):
    try:
        dataset = load_from_disk(str(Path(dataset_path) / "hf_dataset"))
    except Exception:
        from datasets import load_dataset
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(Path(dataset_path) / "train.jsonl"),
                "validation": str(Path(dataset_path) / "validation.jsonl"),
            },
        )
    # Filter unlabelled
    for split in dataset:
        dataset[split] = dataset[split].filter(
            lambda x: not str(x.get(text_field, "")).endswith("[NEEDS_LABELLING]")
        )
    logger.info(f"Dataset — Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_sft(
    model_name: str,
    config_path: str,
    dataset_path: str,
    output_dir: str,
    use_4bit: bool = False,
    resume_from_checkpoint: Optional[str] = None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg_path = Path(config_path).parent / "model_config.yaml"
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)

    set_seed(42)
    t = cfg["training"]

    tokenizer = load_tokenizer(model_name)
    print("DEBUG use_4bit =", use_4bit)
    model = load_model(model_name, use_4bit=use_4bit)
    model = apply_lora(model, model_cfg["lora"])

    dataset = load_and_prepare_dataset(dataset_path, cfg["data"]["text_field"])

    sft_config = SFTConfig(
        output_dir=output_dir,
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
        fp16=t["fp16"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        logging_steps=t["logging_steps"],
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "sft-training"),
        dataset_text_field=cfg["data"]["text_field"],
        max_seq_length=t.get("max_seq_length", 2048),
        packing=t.get("packing", True),
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    logger.info("SFT training started...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("SFT complete!")
    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset_path", default="data/processed")
    parser.add_argument("--output_dir", default="models/finetuned")
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sft(
        model_name=args.model_name,
        config_path=args.config,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
