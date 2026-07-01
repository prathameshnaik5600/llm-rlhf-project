# Domain-Specific LLM Fine-Tuning with RLHF
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange)
![RLHF](https://img.shields.io/badge/RLHF-PPO-purple)
A fully production-ready pipeline for fine-tuning open-source LLMs (Mistral 7B / LLaMA 3 8B) on niche domains using **LoRA/QLoRA**, followed by a **lightweight RLHF loop** driven by human preference data.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Stage 1 — Data Preparation](#stage-1--data-preparation)
5. [Stage 2 — LoRA/QLoRA Fine-Tuning (SFT)](#stage-2--loraqloRA-fine-tuning-sft)
6. [Stage 3 — Reward Model Training](#stage-3--reward-model-training)
7. [Stage 4 — RLHF Loop (PPO)](#stage-4--rlhf-loop-ppo)
8. [Stage 5 — Evaluation](#stage-5--evaluation)
9. [Configuration Guide](#configuration-guide)
10. [Hardware Requirements](#hardware-requirements)

---

## Architecture Overview

```
Raw Domain Text
      │
      ▼
┌─────────────────┐
│  Data Pipeline  │  ← Cleaning, chunking, instruction formatting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SFT Training   │  ← LoRA/QLoRA fine-tune on base model
│ (Mistral/LLaMA) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Human Preference│  ← Annotators choose preferred responses
│    Collection   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reward Model   │  ← Trained on (prompt, chosen, rejected) pairs
│    Training     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PPO / RLHF    │  ← Policy updated to maximise reward signal
│     Loop        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Aligned  │
│     Model       │
└─────────────────┘
```

---

## Project Structure

```
llm-rlhf-project/
├── configs/
│   ├── sft_config.yaml          # SFT training hyperparameters
│   ├── reward_config.yaml       # Reward model training config
│   ├── rlhf_config.yaml         # PPO / RLHF loop config
│   └── model_config.yaml        # Base model & LoRA settings
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py   # Load & format domain datasets
│   │   ├── instruction_formatter.py  # Prompt templates
│   │   └── preference_collector.py   # Human preference UI/pipeline
│   │
│   ├── training/
│   │   ├── sft_trainer.py       # Supervised fine-tuning (LoRA/QLoRA)
│   │   └── lora_config.py       # LoRA adapter configuration
│   │
│   ├── rlhf/
│   │   ├── reward_model.py      # Reward model architecture & training
│   │   └── ppo_trainer.py       # PPO training loop
│   │
│   ├── evaluation/
│   │   └── evaluator.py         # ROUGE, win-rate, domain benchmarks
│   │
│   └── utils/
│       ├── logging_utils.py     # Structured logging
│       └── checkpoint_utils.py  # Save/load checkpoints
│
├── scripts/
│   ├── 01_prepare_data.sh
│   ├── 02_run_sft.sh
│   ├── 03_collect_preferences.sh
│   ├── 04_train_reward_model.sh
│   └── 05_run_rlhf.sh
│
├── data/
│   ├── raw/                     # Raw domain text
│   ├── processed/               # Formatted instruction pairs
│   └── preferences/             # Human preference JSONL files
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_demo.ipynb
│   └── 03_rlhf_analysis.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo>
cd llm-rlhf-project
pip install -r requirements.txt

# 2. Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# 3. Run the full pipeline
bash scripts/01_prepare_data.sh
bash scripts/02_run_sft.sh
bash scripts/03_collect_preferences.sh
bash scripts/04_train_reward_model.sh
bash scripts/05_run_rlhf.sh
```

---

## Stage 1 — Data Preparation

**Goal:** Convert raw domain text into instruction-following pairs.

```bash
python -m src.data.dataset_builder \
  --domain legal \
  --source_dir data/raw \
  --output_dir data/processed \
  --max_samples 50000
```

**Supported domains:** `legal`, `medical`, `financial`

**Output format (JSONL):**
```json
{
  "instruction": "Summarise the key obligations in this contract clause.",
  "input": "The licensee shall...",
  "output": "The licensee must...",
  "domain": "legal",
  "source": "contracts"
}
```

---

## Stage 2 — LoRA/QLoRA Fine-Tuning (SFT)

```bash
python -m src.training.sft_trainer \
  --config configs/sft_config.yaml \
  --model_name mistralai/Mistral-7B-v0.1 \
  --dataset_path data/processed \
  --output_dir models/finetuned \
  --use_4bit True
```

**LoRA config (default):**
| Parameter | Value |
|-----------|-------|
| `r` (rank) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| Target modules | `q_proj, v_proj, k_proj, o_proj` |
| Quantization | 4-bit NF4 (QLoRA) |

---

## Stage 3 — Reward Model Training

```bash
python -m src.rlhf.reward_model \
  --config configs/reward_config.yaml \
  --preferences_path data/preferences \
  --base_model models/finetuned \
  --output_dir models/reward
```

**Preference data format:**
```json
{
  "prompt": "What does force majeure mean?",
  "chosen": "Force majeure refers to...",
  "rejected": "I'm not sure, but maybe..."
}
```

---

## Stage 4 — RLHF Loop (PPO)

```bash
python -m src.rlhf.ppo_trainer \
  --config configs/rlhf_config.yaml \
  --policy_model models/finetuned \
  --reward_model models/reward \
  --output_dir models/rlhf_final
```

---

## Stage 5 — Evaluation

```bash
python -m src.evaluation.evaluator \
  --model_dir models/rlhf_final \
  --eval_set data/processed/test.jsonl \
  --domain legal
```

**Metrics:** ROUGE-L, BERTScore, domain accuracy, reward win-rate vs SFT baseline.

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|-------|---------|-------------|
| GPU VRAM | 16 GB (QLoRA 4-bit) | 40 GB (A100) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 200 GB |
| CUDA | 11.8+ | 12.1+ |

**Cloud options:** RunPod A100, Lambda Labs, Vast.ai (~$1–2/hr)
