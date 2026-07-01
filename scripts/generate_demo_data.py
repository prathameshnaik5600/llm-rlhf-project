"""
Generate a tiny synthetic demo dataset for smoke-testing the full pipeline
without needing HuggingFace Hub access or a GPU.

Creates:
  data/processed/train.jsonl         (80 samples)
  data/processed/validation.jsonl    (10 samples)
  data/processed/test.jsonl          (10 samples)
  data/processed/rlhf_prompts.jsonl  (80 prompts)
  data/processed/hf_dataset/         (HuggingFace Dataset format)
"""

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

DOMAIN = "legal"
SYSTEM = "You are an expert legal analyst."

PAIRS = [
    ("Summarise the key legal obligations in this clause: The licensee shall not sublicense or resell the software.",
     "The licensee is prohibited from sublicensing or reselling the software. The licence is personal and non-transferable."),
    ("What does 'force majeure' mean in a contract?",
     "Force majeure is a contractual provision that excuses a party from performance obligations when extraordinary events beyond their control occur, such as natural disasters, war, or pandemics."),
    ("Explain the concept of 'indemnification' in plain terms.",
     "Indemnification is an obligation by one party to compensate the other for losses or liabilities arising from a specified event. In practice, the indemnifying party agrees to cover legal costs and damages if a claim arises."),
    ("What is a 'warranty' in a commercial contract?",
     "A warranty is a contractual promise that certain facts are true at the time the contract is made. Unlike a condition, a breach of warranty does not automatically end the contract but entitles the innocent party to damages."),
    ("Identify the key risks in this clause: 'The Company's liability shall not exceed £500.'",
     "The clause imposes a very low liability cap of £500, which may not adequately cover losses in a commercial dispute. Courts may deem this unreasonable under UCTA 1977, especially where negligence causes significant harm."),
    ("What is the difference between an assignment and a novation?",
     "An assignment transfers rights under a contract to a third party while leaving the original party still liable. A novation substitutes a new party entirely, releasing the original party from all obligations with consent from all parties."),
    ("Summarise the obligations of a data controller under GDPR.",
     "Under GDPR, a data controller must: process personal data lawfully and transparently; collect data for specified purposes only; ensure data minimisation; maintain accuracy; limit storage periods; and implement appropriate security measures."),
    ("What constitutes a binding contract?",
     "A binding contract requires: (1) offer — a clear proposal; (2) acceptance — unqualified agreement; (3) consideration — something of value exchanged; (4) intention to create legal relations; and (5) capacity of parties to contract."),
    ("Explain liquidated damages clauses.",
     "A liquidated damages clause pre-agrees the amount payable on breach. It is enforceable where it represents a genuine pre-estimate of loss, not a penalty. Courts will strike down clauses that are punitive rather than compensatory."),
    ("What is 'tortious interference' with a contract?",
     "Tortious interference occurs when a third party intentionally causes one contracting party to breach their agreement with another. The claimant must show the third party knew of the contract and deliberately induced the breach without justification."),
]

def make_record(instruction, output, idx, split):
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    text = prompt + output + "<|im_end|>"
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "prompt": prompt,
        "text": text,
        "domain": DOMAIN,
        "task_type": "qa",
        "source": "demo",
        "id": f"{split}_{idx}",
    }

def main():
    random.seed(42)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    all_records = []
    for epoch in range(10):  # 10 * 10 = 100 records
        for i, (inst, resp) in enumerate(PAIRS):
            # Slight variation to avoid exact duplicates
            all_records.append(make_record(inst, resp, epoch * 10 + i, "all"))

    random.shuffle(all_records)
    train = all_records[:80]
    val   = all_records[80:90]
    test  = all_records[90:100]

    for split_name, data in [("train", train), ("validation", val), ("test", test)]:
        with open(out / f"{split_name}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {split_name}.jsonl — {len(data)} records")

    # RLHF prompts
    with open(out / "rlhf_prompts.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps({"prompt": r["prompt"]}) + "\n")
    print(f"Wrote rlhf_prompts.jsonl — {len(train)} prompts")

    # HuggingFace Dataset format
    hf_ds = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })
    hf_ds.save_to_disk(str(out / "hf_dataset"))
    print(f"Wrote hf_dataset/")
    print("✅ Demo dataset ready.")

if __name__ == "__main__":
    main()
