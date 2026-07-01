"""
src/inference.py

Simple inference script for the fine-tuned / RLHF-aligned model.
Supports single prompt, batch, and interactive chat modes.

Usage:
    # Single prompt
    python -m src.inference --model_path models/rlhf_final --prompt "What is force majeure?"

    # Interactive chat
    python -m src.inference --model_path models/rlhf_final --interactive

    # Batch from file
    python -m src.inference --model_path models/rlhf_final --batch_file prompts.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


SYSTEM_PROMPTS = {
    "legal": "You are an expert legal analyst. Provide accurate, well-reasoned responses grounded in legal principles.",
    "medical": "You are a knowledgeable medical assistant. Provide clear, accurate medical information. Always recommend consulting a qualified healthcare professional.",
    "financial": "You are a senior financial analyst. Provide data-driven, objective financial analysis. Note that this is not personal financial advice.",
    "general": "You are a helpful, accurate, and thoughtful assistant.",
}


class LLMInference:
    def __init__(
        self,
        model_path: str,
        domain: str = "legal",
        load_in_4bit: bool = True,
        device: str = "auto",
    ):
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=load_in_4bit,
        )
        self.model.eval()
        self.system_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["general"])
        logger.info(f"Model ready. Domain: {domain}")

    def build_prompt(self, user_input: str) -> str:
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str:
        full_prompt = self.build_prompt(prompt)
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        if stream:
            return ""  # Already printed via streamer

        prompt_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()
        return response

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def interactive(self):
        print(f"\n{'='*60}")
        print(f" Domain-Specific LLM — Interactive Mode")
        print(f" System: {self.system_prompt[:80]}...")
        print(f" Type 'quit' or Ctrl+C to exit.")
        print(f"{'='*60}\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            print("Assistant: ", end="", flush=True)
            response = self.generate(user_input, stream=True)
            print()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--domain", default="legal", choices=list(SYSTEM_PROMPTS.keys()))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--batch_file", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = LLMInference(
        model_path=args.model_path,
        domain=args.domain,
        load_in_4bit=not args.no_4bit,
    )

    if args.interactive:
        engine.interactive()

    elif args.batch_file:
        prompts = Path(args.batch_file).read_text().strip().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
        responses = engine.batch_generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for p, r in zip(prompts, responses):
            print(f"PROMPT:   {p}")
            print(f"RESPONSE: {r}")
            print("-" * 60)

    elif args.prompt:
        response = engine.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=True,
        )

    else:
        print("Provide --prompt, --batch_file, or --interactive")
        sys.exit(1)
