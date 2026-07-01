from setuptools import setup, find_packages

setup(
    name="llm-rlhf",
    version="0.1.0",
    description="Domain-specific LLM fine-tuning with LoRA/QLoRA + RLHF",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.16.0",
        "accelerate>=0.26.0",
        "peft>=0.8.0",
        "trl>=0.7.10",
        "bitsandbytes>=0.42.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "evaluate>=0.4.1",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "loguru>=0.7.2",
        "gradio>=4.15.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
)
