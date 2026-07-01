"""
src/evaluation/evaluator.py

Comprehensive evaluation for domain-specific fine-tuned models.

Metrics:
  - ROUGE-L:       Recall-oriented text overlap
  - BERTScore:     Semantic similarity via contextual embeddings
  - Win Rate:      % of RLHF responses rated higher than SFT by reward model
  - Domain Score:  Task-specific accuracy (legal/medical/financial)
  - Perplexity:    Language model quality measure

Usage:
    python -m src.evaluation.evaluator \
        --model_dir models/rlhf_final \
        --eval_set data/processed/test.jsonl \
        --domain legal \
        --compare_model models/finetuned
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from loguru import logger
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> List[str]:
    """Generate responses for a list of prompts."""
    responses = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i: i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,        # Low temp for eval (deterministic-ish)
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for j, out in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            response = tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
            responses.append(response)

    return responses


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bert_score(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Compute BERTScore P/R/F1."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bertscore_f1": -1.0}


@torch.no_grad()
def compute_reward_win_rate(
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    prompts: List[str],
    model_a_responses: List[str],     # RLHF model
    model_b_responses: List[str],     # SFT baseline
    batch_size: int = 8,
    max_length: int = 1024,
) -> Dict[str, float]:
    """
    Compute win rate of model_a vs model_b according to reward model.

    Win rate = % of prompts where reward(model_a) > reward(model_b)
    """
    device = next(reward_model.parameters()).device
    wins = 0
    ties = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        batch_a = model_a_responses[i: i + batch_size]
        batch_b = model_b_responses[i: i + batch_size]

        texts_a = [p + r for p, r in zip(batch_prompts, batch_a)]
        texts_b = [p + r for p, r in zip(batch_prompts, batch_b)]

        enc_a = reward_tokenizer(
            texts_a, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)
        enc_b = reward_tokenizer(
            texts_b, max_length=max_length, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)

        rewards_a = reward_model(**enc_a).logits.squeeze(-1)
        rewards_b = reward_model(**enc_b).logits.squeeze(-1)

        wins += (rewards_a > rewards_b).sum().item()
        ties += (rewards_a == rewards_b).sum().item()

    n = len(prompts)
    win_rate = wins / n
    tie_rate = ties / n
    return {
        "win_rate": win_rate,
        "loss_rate": 1 - win_rate - tie_rate,
        "tie_rate": tie_rate,
        "total_compared": n,
    }


@torch.no_grad()
def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    stride: int = 512,
    max_length: int = 1024,
) -> float:
    """Compute average perplexity over a list of texts."""
    device = next(model.parameters()).device
    nlls = []

    for text in tqdm(texts[:100], desc="Computing perplexity"):  # Cap at 100 for speed
        encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encodings.input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())

    mean_nll = sum(nlls) / len(nlls)
    perplexity = math.exp(mean_nll)
    return perplexity


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def run_evaluation(
    model_dir: str,
    eval_set_path: str,
    domain: str,
    reward_model_path: Optional[str] = None,
    compare_model_path: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """Run full evaluation suite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running evaluation on {device}")

    # ── Load eval data ─────────────────────────────────────────────────
    logger.info(f"Loading eval set from {eval_set_path}")
    examples = []
    with open(eval_set_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    prompts = [e["prompt"] for e in examples]
    references = [e.get("output", "") for e in examples]
    logger.info(f"Evaluating on {len(prompts)} examples")

    # ── Load model ─────────────────────────────────────────────────────
    logger.info(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_4bit=True,
    )
    model.eval()

    # ── Generate responses ─────────────────────────────────────────────
    logger.info("Generating responses for eval model...")
    responses = generate_responses(model, tokenizer, prompts)

    # ── Compute metrics ────────────────────────────────────────────────
    results = {"domain": domain, "model": model_dir, "n_samples": len(prompts)}

    logger.info("Computing ROUGE scores...")
    rouge = compute_rouge(responses, references)
    results.update(rouge)

    logger.info("Computing BERTScore...")
    bert = compute_bert_score(responses, references)
    results.update(bert)

    logger.info("Computing perplexity...")
    ref_texts = [e.get("text", "") for e in examples if e.get("text")]
    if ref_texts:
        ppl = compute_perplexity(model, tokenizer, ref_texts[:100])
        results["perplexity"] = ppl

    # ── Win rate (vs SFT baseline) ─────────────────────────────────────
    if reward_model_path and compare_model_path:
        logger.info("Computing reward win rate...")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            num_labels=1,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token

        baseline_model = AutoModelForCausalLM.from_pretrained(
            compare_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        baseline_tokenizer = AutoTokenizer.from_pretrained(compare_model_path)
        baseline_responses = generate_responses(baseline_model, baseline_tokenizer, prompts)

        win_rate_results = compute_reward_win_rate(
            reward_model, reward_tokenizer,
            prompts, responses, baseline_responses,
        )
        results.update(win_rate_results)

    # ── Print & save results ───────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:.4f}")
        else:
            logger.info(f"  {k:30s}: {v}")
    logger.info("="*60)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--domain", required=True, choices=["legal", "medical", "financial"])
    parser.add_argument("--reward_model_path", default=None)
    parser.add_argument("--compare_model", default=None, help="SFT baseline for win-rate")
    parser.add_argument("--output_path", default="outputs/eval_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        eval_set_path=args.eval_set,
        domain=args.domain,
        reward_model_path=args.reward_model_path,
        compare_model_path=args.compare_model,
        output_path=args.output_path,
    )
