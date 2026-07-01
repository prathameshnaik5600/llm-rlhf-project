"""
src/data/dataset_builder.py

Loads raw domain text (legal, medical, financial) from HuggingFace Hub
or local files, cleans it, and formats it into instruction-following pairs.

Usage:
    python -m src.data.dataset_builder --domain legal --output_dir data/processed
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Optional

import datasets
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm

from src.data.instruction_formatter import InstructionFormatter


# ---------------------------------------------------------------------------
# Domain dataset configurations
# ---------------------------------------------------------------------------

DOMAIN_CONFIGS = {
    "legal": {
        "hf_datasets": [
            {
                "name": "rceborg/legal-contracts",
                "split": "train",
                "text_col": "text",
                "task_type": "summarise",
            },
            {
                "name": "pile-of-law/pile-of-law",
                "split": "train",
                "text_col": "text",
                "task_type": "qa",
                "subset": "courtlistener_opinions",
                "streaming": True,
            },
        ],
        "local_glob": "data/raw/legal/**/*.txt",
        "task_types": ["summarise", "qa", "classify", "extract"],
        "system_prompt": (
            "You are an expert legal analyst. Provide accurate, "
            "well-reasoned responses grounded in legal principles."
        ),
    },
    "medical": {
        "hf_datasets": [
            {
                "name": "medalpaca/medical_meadow_medqa",
                "split": "train",
                "text_col": "input",
                "output_col": "output",
                "task_type": "qa",
            },
            {
                "name": "lavita/ChatDoctor-HealthCareMagic-100k",
                "split": "train",
                "text_col": "input",
                "output_col": "output",
                "task_type": "qa",
            },
        ],
        "local_glob": "data/raw/medical/**/*.txt",
        "task_types": ["qa", "diagnose", "explain", "summarise"],
        "system_prompt": (
            "You are a knowledgeable medical assistant. Provide clear, "
            "accurate medical information. Always recommend consulting a "
            "qualified healthcare professional for personal medical advice."
        ),
    },
    "financial": {
        "hf_datasets": [
            {
                "name": "winddude/reddit_finance_43_250k",
                "split": "train",
                "text_col": "body",
                "task_type": "qa",
            },
            {
                "name": "FinGPT/fingpt-sentiment-train",
                "split": "train",
                "text_col": "input",
                "output_col": "output",
                "task_type": "analyse",
            },
        ],
        "local_glob": "data/raw/financial/**/*.txt",
        "task_types": ["analyse", "summarise", "forecast", "explain"],
        "system_prompt": (
            "You are a senior financial analyst. Provide data-driven, "
            "objective financial analysis. Always note that this is not "
            "personal financial advice."
        ),
    },
}


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Builds an instruction-tuning dataset for a specific domain.

    Pipeline:
        1. Load raw data from HuggingFace Hub + local files
        2. Clean and deduplicate text
        3. Format into (instruction, input, output) triples
        4. Split into train / validation / test
        5. Save to disk as JSONL + HuggingFace Dataset
    """

    def __init__(
        self,
        domain: str,
        output_dir: str,
        max_samples: int = 50_000,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
        min_text_length: int = 100,
        max_text_length: int = 4096,
    ):
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Domain '{domain}' not supported. Choose from: {list(DOMAIN_CONFIGS.keys())}")

        self.domain = domain
        self.config = DOMAIN_CONFIGS[domain]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.formatter = InstructionFormatter(
            domain=domain,
            system_prompt=self.config["system_prompt"],
        )

        random.seed(seed)
        logger.info(f"DatasetBuilder initialised for domain: {domain}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> DatasetDict:
        """Full pipeline: load → clean → format → split → save."""
        logger.info("Step 1/5 — Loading raw data...")
        raw_samples = self._load_all_sources()

        logger.info(f"Step 2/5 — Cleaning {len(raw_samples)} samples...")
        clean_samples = self._clean(raw_samples)

        logger.info(f"Step 3/5 — Formatting {len(clean_samples)} samples...")
        formatted = self._format(clean_samples)

        logger.info("Step 4/5 — Splitting dataset...")
        dataset_dict = self._split(formatted)

        logger.info("Step 5/5 — Saving...")
        self._save(dataset_dict)

        logger.info(
            f"Done! Train: {len(dataset_dict['train'])} | "
            f"Val: {len(dataset_dict['validation'])} | "
            f"Test: {len(dataset_dict['test'])}"
        )
        return dataset_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all_sources(self) -> list[dict]:
        samples = []

        # HuggingFace datasets
        for cfg in self.config["hf_datasets"]:
            try:
                logger.info(f"  Loading HF dataset: {cfg['name']}")
                ds_samples = self._load_hf_dataset(cfg)
                samples.extend(ds_samples)
                logger.info(f"  → {len(ds_samples)} samples loaded")
            except Exception as e:
                logger.warning(f"  Failed to load {cfg['name']}: {e}")

        # Local files
        local_samples = self._load_local_files()
        if local_samples:
            logger.info(f"  Loaded {len(local_samples)} local samples")
            samples.extend(local_samples)

        # Cap to max_samples
        random.shuffle(samples)
        return samples[: self.max_samples]

    def _load_hf_dataset(self, cfg: dict) -> list[dict]:
        """Load a single HuggingFace dataset config into raw dicts."""
        load_kwargs = {
            "path": cfg["name"],
            "split": cfg.get("split", "train"),
            "trust_remote_code": True,
        }
        if cfg.get("subset"):
            load_kwargs["name"] = cfg["subset"]
        if cfg.get("streaming"):
            load_kwargs["streaming"] = True

        ds = load_dataset(**load_kwargs)

        samples = []
        text_col = cfg.get("text_col", "text")
        output_col = cfg.get("output_col")
        task_type = cfg.get("task_type", "qa")

        iterable = ds if cfg.get("streaming") else ds
        limit = self.max_samples // max(len(self.config["hf_datasets"]), 1)

        for i, row in enumerate(iterable):
            if i >= limit:
                break
            text = row.get(text_col, "")
            if not text or len(text) < self.min_text_length:
                continue
            sample = {
                "text": text[: self.max_text_length],
                "output": row.get(output_col, "") if output_col else "",
                "task_type": task_type,
                "source": cfg["name"],
                "domain": self.domain,
            }
            samples.append(sample)

        return samples

    def _load_local_files(self) -> list[dict]:
        """Load raw .txt files from the local data/raw directory."""
        import glob

        pattern = self.config.get("local_glob", f"data/raw/{self.domain}/**/*.txt")
        files = glob.glob(pattern, recursive=True)
        samples = []

        for fpath in files[:1000]:  # Cap local files
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                if len(text) < self.min_text_length:
                    continue
                samples.append({
                    "text": text[: self.max_text_length],
                    "output": "",
                    "task_type": random.choice(self.config["task_types"]),
                    "source": "local",
                    "domain": self.domain,
                })
            except Exception:
                pass

        return samples

    def _clean(self, samples: list[dict]) -> list[dict]:
        """Remove duplicates, low-quality text, and noise."""
        seen = set()
        clean = []

        for s in tqdm(samples, desc="Cleaning"):
            text = s["text"].strip()

            # Deduplicate on first 200 chars
            fingerprint = text[:200]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            # Quality filters
            if len(text) < self.min_text_length:
                continue
            if text.count("\n") / max(len(text), 1) > 0.3:
                continue  # Too many newlines = likely garbage
            if len(set(text.split())) < 20:
                continue  # Too few unique words

            s["text"] = text
            clean.append(s)

        return clean

    def _format(self, samples: list[dict]) -> list[dict]:
        """Convert raw samples into instruction-following format."""
        formatted = []
        for s in tqdm(samples, desc="Formatting"):
            try:
                record = self.formatter.format(s)
                if record:
                    formatted.append(record)
            except Exception as e:
                logger.debug(f"Format error: {e}")
        return formatted

    def _split(self, samples: list[dict]) -> DatasetDict:
        """Train / validation / test split."""
        random.shuffle(samples)
        n = len(samples)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)

        test = samples[:n_test]
        val = samples[n_test: n_test + n_val]
        train = samples[n_test + n_val:]

        return DatasetDict({
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        })

    def _save(self, dataset_dict: DatasetDict):
        """Save as HuggingFace Dataset + JSONL files."""
        # HF format
        dataset_dict.save_to_disk(str(self.output_dir / "hf_dataset"))

        # JSONL files (for manual inspection / non-HF pipelines)
        for split, ds in dataset_dict.items():
            out_path = self.output_dir / f"{split}.jsonl"
            with open(out_path, "w") as f:
                for row in ds:
                    f.write(json.dumps(row) + "\n")
            logger.info(f"  Saved {split}.jsonl — {len(ds)} rows")

        # RLHF prompts (just the instruction part, no answer)
        rlhf_path = self.output_dir / "rlhf_prompts.jsonl"
        with open(rlhf_path, "w") as f:
            for row in dataset_dict["train"]:
                f.write(json.dumps({"prompt": row["prompt"]}) + "\n")
        logger.info(f"  Saved rlhf_prompts.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Build domain dataset")
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIGS.keys()))
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    builder = DatasetBuilder(
        domain=args.domain,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    builder.build()
