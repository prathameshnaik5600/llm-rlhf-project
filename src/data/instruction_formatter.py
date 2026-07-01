"""
src/data/instruction_formatter.py

Converts raw domain text samples into structured instruction-following pairs.
Supports Alpaca, ChatML, and Mistral/LLaMA chat formats.
"""

import random
from typing import Optional


# ---------------------------------------------------------------------------
# Task-specific instruction templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "legal": {
        "summarise": [
            "Summarise the key legal obligations described in the following text.",
            "Provide a concise legal summary of the following passage.",
            "What are the main legal points in this text? Summarise them clearly.",
            "Distil the legal significance of the following into plain language.",
        ],
        "qa": [
            "Based on the following legal text, answer the question: {question}",
            "Review the legal passage below and explain what it means in practice.",
            "What legal rights or obligations does the following text establish?",
            "Identify any risks or liabilities described in the following legal text.",
        ],
        "classify": [
            "Classify the type of legal document described in the following text.",
            "What area of law does the following passage relate to?",
            "Identify whether the following text is from a contract, statute, case, or regulation.",
        ],
        "extract": [
            "Extract all defined terms from the following legal text.",
            "List the parties, obligations, and key dates in the following clause.",
            "Identify any penalty clauses or indemnity provisions in this text.",
        ],
    },
    "medical": {
        "qa": [
            "A patient asks: {question}. Provide an accurate, helpful response.",
            "Answer the following medical question clearly and accurately.",
            "Explain the following medical concept in terms a patient can understand.",
            "What does the medical literature say about the following condition or treatment?",
        ],
        "diagnose": [
            "Given the following symptoms, what conditions should be considered?",
            "What differential diagnoses are consistent with the described presentation?",
            "Summarise what the described symptoms suggest clinically.",
        ],
        "explain": [
            "Explain the mechanism of the condition described in the following text.",
            "How does the treatment described in the following passage work?",
            "What are the clinical implications of the findings described below?",
        ],
        "summarise": [
            "Summarise the clinical findings described in the following medical text.",
            "Provide a structured summary of this medical passage for a clinician.",
        ],
    },
    "financial": {
        "analyse": [
            "Analyse the financial situation described in the following text.",
            "What are the key financial risks and opportunities in this passage?",
            "Provide a structured financial analysis of the following.",
        ],
        "summarise": [
            "Summarise the key financial information in the following passage.",
            "What are the main financial takeaways from this text?",
        ],
        "forecast": [
            "Based on the financial data described, what trends or outcomes might be expected?",
            "What does the following financial information imply about future performance?",
        ],
        "explain": [
            "Explain the financial concept or instrument described in the following text.",
            "What does the following financial passage mean in plain terms?",
        ],
    },
}

# Fallback questions for QA tasks without an explicit question
QA_FALLBACK_QUESTIONS = {
    "legal": [
        "What are the key legal implications of this text?",
        "What obligations does this create?",
        "What are the potential legal risks here?",
    ],
    "medical": [
        "What does this indicate clinically?",
        "What should a patient know about this?",
        "What are the treatment implications?",
    ],
    "financial": [
        "What does this mean for investors?",
        "What are the financial risks here?",
        "How should this be interpreted financially?",
    ],
}


class InstructionFormatter:
    """
    Converts raw text samples into instruction-following format.

    Output schema:
    {
        "instruction": str,     # The task instruction
        "input": str,           # The domain text context (may be empty)
        "output": str,          # The target response
        "prompt": str,          # Full formatted prompt (for training)
        "text": str,            # Full prompt + response (for SFT)
        "domain": str,
        "task_type": str,
        "source": str,
    }
    """

    def __init__(
        self,
        domain: str,
        system_prompt: str,
        format_style: str = "chatml",  # chatml | alpaca | mistral
    ):
        self.domain = domain
        self.system_prompt = system_prompt
        self.format_style = format_style
        self.templates = TEMPLATES.get(domain, {})

    def format(self, sample: dict) -> Optional[dict]:
        """Format a single raw sample into instruction-following format."""
        text = sample.get("text", "").strip()
        output = sample.get("output", "").strip()
        task_type = sample.get("task_type", "qa")

        if not text:
            return None

        # Pick an instruction template
        instruction = self._pick_instruction(task_type)

        # If no pre-existing output, we can't train on this directly
        # In practice, you'd generate responses or use a strong teacher model
        # For now, we still format it — useful for RLHF prompts
        if not output:
            output = self._generate_placeholder_output(text, task_type)

        # Build the full formatted strings
        prompt = self._build_prompt(instruction, text)
        full_text = self._build_full_text(instruction, text, output)

        return {
            "instruction": instruction,
            "input": text,
            "output": output,
            "prompt": prompt,
            "text": full_text,
            "domain": sample.get("domain", self.domain),
            "task_type": task_type,
            "source": sample.get("source", "unknown"),
        }

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, instruction: str, context: str) -> str:
        """Build the prompt portion (no response)."""
        if self.format_style == "chatml":
            return (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}\n\n{context}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif self.format_style == "mistral":
            return (
                f"[INST] {self.system_prompt}\n\n"
                f"{instruction}\n\n{context} [/INST]"
            )
        else:  # alpaca
            return (
                f"### System:\n{self.system_prompt}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{context}\n\n"
                f"### Response:\n"
            )

    def _build_full_text(self, instruction: str, context: str, output: str) -> str:
        """Build the full prompt + response for SFT training."""
        if self.format_style == "chatml":
            return (
                f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}\n\n{context}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
        elif self.format_style == "mistral":
            return (
                f"[INST] {self.system_prompt}\n\n"
                f"{instruction}\n\n{context} [/INST] {output}</s>"
            )
        else:  # alpaca
            return (
                f"### System:\n{self.system_prompt}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{context}\n\n"
                f"### Response:\n{output}"
            )

    # ------------------------------------------------------------------
    # Template selection
    # ------------------------------------------------------------------

    def _pick_instruction(self, task_type: str) -> str:
        options = self.templates.get(task_type, [])
        if not options:
            # Fallback generic instruction
            return f"Please respond to the following {self.domain} text:"
        template = random.choice(options)
        # Fill in any {question} placeholder
        if "{question}" in template:
            fallback_qs = QA_FALLBACK_QUESTIONS.get(self.domain, ["What does this mean?"])
            template = template.replace("{question}", random.choice(fallback_qs))
        return template

    def _generate_placeholder_output(self, text: str, task_type: str) -> str:
        """
        Minimal placeholder for samples without a gold output.
        In production, replace this with a teacher model's generation
        (e.g., GPT-4 or a large open-source model).
        """
        # This signals to the data pipeline that this sample needs labelling
        return f"[NEEDS_LABELLING] Task: {task_type}"
