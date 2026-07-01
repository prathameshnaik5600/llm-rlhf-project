"""
src/rlhf/ppo_trainer.py

RLHF training via RLOO (REINFORCE Leave-One-Out) — TRL 1.5+ replacement for PPO.
RLOOTrainer uses a reward_funcs callable; no separate reward model forward pass needed.

Usage:
    python -m src.rlhf.ppo_trainer --config configs/rlhf_config.yaml
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
import yaml
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import RLOOConfig, RLOOTrainer, create_reference_model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_prompts_dataset(prompts_path: str, max_prompt_length: int = 512) -> Dataset:
    data = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                if prompt:
                    data.append({"prompt": prompt[:max_prompt_length]})
    logger.info(f"Loaded {len(data)} prompts for RLHF")
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# Reward function factory
# ---------------------------------------------------------------------------

def make_reward_fn(reward_model, reward_tokenizer, device: str, max_length: int = 1024):
    """Returns a reward function compatible with RLOOTrainer.reward_funcs."""
    @torch.no_grad()
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = [""] * len(completions)
        texts = [p + c for p, c in zip(prompts, completions)]
        enc = reward_tokenizer(
            texts, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)
        out = reward_model(**enc)
        rewards = out.logits.squeeze(-1).tolist()
        return [rewards] if isinstance(rewards, float) else rewards
    return reward_fn


# ---------------------------------------------------------------------------
# RLOO training
# ---------------------------------------------------------------------------

def run_ppo(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    ppo_cfg = cfg["ppo"]
    train_cfg = cfg["training"]
    model_cfg = cfg["models"]
    data_cfg = cfg["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"RLHF training on: {device}")

    # Tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["policy_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    reward_tokenizer = AutoTokenizer.from_pretrained(model_cfg["reward_model"], trust_remote_code=True)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # Dataset
    dataset = load_prompts_dataset(data_cfg["prompts_path"], data_cfg["max_prompt_length"])

    # Reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["reward_model"],
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        num_labels=1,
    )
    reward_model.eval()

    # Policy model
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["policy_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = create_reference_model(policy_model)

    # RLOO config
    rloo_config = RLOOConfig(
        output_dir=train_cfg["output_dir"],
        learning_rate=ppo_cfg["learning_rate"],
        per_device_train_batch_size=ppo_cfg["batch_size"],
        gradient_accumulation_steps=ppo_cfg["gradient_accumulation_steps"],
        max_completion_length=ppo_cfg["max_new_tokens"],
        num_train_epochs=1,
        save_steps=train_cfg.get("save_freq", 100),
        logging_steps=train_cfg.get("log_freq", 10),
        report_to=train_cfg.get("report_to", "none"),
        run_name=train_cfg.get("run_name", "rlhf-rloo"),
        beta=ppo_cfg.get("init_kl_coef", 0.2),   # KL penalty coefficient
        bf16=True,
        temperature=ppo_cfg.get("temperature", 0.7),
        num_generations=4,  # RLOO leave-one-out samples
    )

    reward_fn = make_reward_fn(reward_model, reward_tokenizer, device)

    trainer = RLOOTrainer(
        model=policy_model,
        reward_funcs=[reward_fn],
        args=rloo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting RLOO RLHF training...")
    trainer.train()

    out_dir = train_cfg["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info(f"RLHF complete → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rlhf_config.yaml")
    args = parser.parse_args()
    run_ppo(args.config)
