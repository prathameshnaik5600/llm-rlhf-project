"""
src/data/preference_collector.py

A lightweight Gradio UI for collecting human preference data.
Annotators are shown a prompt and two model responses side-by-side,
and select which response they prefer.

Collected data is saved as JSONL in data/preferences/.

Usage:
    python -m src.data.preference_collector \
        --model_path models/finetuned \
        --prompts_path data/processed/rlhf_prompts.jsonl \
        --output_path data/preferences/train_preferences.jsonl
"""

import argparse
import json
import os
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

try:
    import gradio as gr
except ImportError:
    gr = None  # Gradio is optional; only needed when launching the UI
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# Response generator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Generates two different responses for a given prompt.

    In a real setup, these would typically come from:
    - Response A: The SFT model (temperature 0.7)
    - Response B: The SFT model with different decoding (temperature 1.2)
    Or from two different model checkpoints.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True,
        )
        self.model.eval()

    @torch.no_grad()
    def generate_pair(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> Tuple[str, str]:
        """Generate two different responses for the same prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # Response A — lower temperature (more conservative)
        out_a = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_a = self.tokenizer.decode(
            out_a[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Response B — higher temperature (more creative/varied)
        out_b = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_b = self.tokenizer.decode(
            out_b[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return response_a, response_b


# ---------------------------------------------------------------------------
# Preference data writer
# ---------------------------------------------------------------------------

class PreferenceWriter:
    """Thread-safe JSONL writer for preference data."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.count = 0

    def write(self, prompt: str, chosen: str, rejected: str, annotator: str = "human"):
        record = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "annotator": annotator,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with self._lock:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.count += 1
        return self.count


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", ""))
    random.shuffle(prompts)
    return [p for p in prompts if p]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(
    generator: ResponseGenerator,
    writer: PreferenceWriter,
    prompts: List[str],
):
    if gr is None:
        raise ImportError("gradio is required for the preference UI. Install with: pip install gradio")
    return _build_ui_impl(generator, writer, prompts)


def _build_ui_impl(
    generator: ResponseGenerator,
    writer: PreferenceWriter,
    prompts: List[str],
):
    """Build the preference annotation UI."""

    state = {"index": 0, "current_prompt": "", "response_a": "", "response_b": ""}

    def load_next_prompt():
        idx = state["index"]
        if idx >= len(prompts):
            return (
                "✅ All prompts annotated!",
                "",
                "",
                f"Total annotations saved: {writer.count}",
            )
        prompt = prompts[idx]
        state["current_prompt"] = prompt
        state["index"] += 1

        resp_a, resp_b = generator.generate_pair(prompt)
        state["response_a"] = resp_a
        state["response_b"] = resp_b

        return (
            prompt,
            resp_a,
            resp_b,
            f"Prompt {idx + 1} / {len(prompts)} | Saved: {writer.count}",
        )

    def choose_a():
        writer.write(
            prompt=state["current_prompt"],
            chosen=state["response_a"],
            rejected=state["response_b"],
        )
        return load_next_prompt()

    def choose_b():
        writer.write(
            prompt=state["current_prompt"],
            chosen=state["response_b"],
            rejected=state["response_a"],
        )
        return load_next_prompt()

    def skip():
        state["index"] += 1
        return load_next_prompt()

    with gr.Blocks(
        title="Preference Annotation Tool",
        theme=gr.themes.Soft(),
        css="""
            .response-box { border: 2px solid #e0e0e0; border-radius: 8px; padding: 12px; }
            .chosen-a { border-color: #4CAF50 !important; }
            .chosen-b { border-color: #2196F3 !important; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🏷️ Preference Annotation Tool
            Read the prompt and both responses. Click **Prefer A** or **Prefer B**
            to select the better response. Use **Skip** if both are equally good or bad.
            """
        )

        status_bar = gr.Textbox(
            label="Status",
            value="Click 'Load Prompt' to start",
            interactive=False,
        )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="📋 Prompt",
                lines=4,
                interactive=False,
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Response A")
                response_a_box = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False,
                    elem_classes=["response-box"],
                )
                btn_a = gr.Button("👍 Prefer A", variant="primary")

            with gr.Column():
                gr.Markdown("### Response B")
                response_b_box = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False,
                    elem_classes=["response-box"],
                )
                btn_b = gr.Button("👍 Prefer B", variant="primary")

        with gr.Row():
            load_btn = gr.Button("▶ Load Prompt", variant="secondary")
            skip_btn = gr.Button("⏭ Skip", variant="secondary")

        outputs = [prompt_box, response_a_box, response_b_box, status_bar]

        load_btn.click(fn=load_next_prompt, outputs=outputs)
        btn_a.click(fn=choose_a, outputs=outputs)
        btn_b.click(fn=choose_b, outputs=outputs)
        skip_btn.click(fn=skip, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Collect human preferences via UI")
    parser.add_argument("--model_path", required=True, help="Path to SFT model")
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_path", default="data/preferences/train_preferences.jsonl")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generator = ResponseGenerator(args.model_path)
    writer = PreferenceWriter(args.output_path)
    prompts = load_prompts(args.prompts_path)

    logger.info(f"Loaded {len(prompts)} prompts")
    logger.info(f"Saving preferences to {args.output_path}")

    ui = build_ui(generator, writer, prompts)
    ui.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )
