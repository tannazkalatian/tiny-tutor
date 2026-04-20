"""
LoRA fine-tuning of SmolLM2-360M-Instruct on the filtered ELI5 dataset.

Designed to run on a free Colab T4 GPU via Unsloth.
Logs training metrics to Weights & Biases.

Usage:
    python src/train.py \
        --dataset data/filtered.jsonl \
        --output-dir checkpoints/ \
        --wandb-project tiny-tutor
"""

import argparse
import json
from pathlib import Path

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}

TRAIN_CONFIG = {
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "max_seq_length": 512,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "evaluation_strategy": "steps",
}

SYSTEM_PROMPT = (
    "You are a friendly science tutor. Explain STEM concepts simply, "
    "using analogies a 7-year-old would understand."
)


def record_to_messages(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explain '{record['concept']}' like I'm 5."},
        {"role": "assistant", "content": record["explanation"]},
    ]


def load_dataset(path: Path):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="checkpoints/")
    parser.add_argument("--wandb-project", default="tiny-tutor")
    parser.add_argument("--test-size", type=float, default=0.05)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import wandb
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install unsloth trl transformers wandb"
        ) from e

    wandb.init(project=args.wandb_project, config={**LORA_CONFIG, **TRAIN_CONFIG})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=TRAIN_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing=True,
    )

    records = load_dataset(Path(args.dataset))
    split = int(len(records) * (1 - args.test_size))
    train_records, eval_records = records[:split], records[split:]
    print(f"Train: {len(train_records)} | Eval: {len(eval_records)}")

    def format_example(record):
        messages = record_to_messages(record)
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    from datasets import Dataset

    train_dataset = Dataset.from_list([format_example(r) for r in train_records])
    eval_dataset = Dataset.from_list([format_example(r) for r in eval_records])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=TRAIN_CONFIG["max_seq_length"],
        args=TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=TRAIN_CONFIG["learning_rate"],
            per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
            num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
            warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
            lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
            fp16=TRAIN_CONFIG["fp16"],
            logging_steps=TRAIN_CONFIG["logging_steps"],
            save_steps=TRAIN_CONFIG["save_steps"],
            eval_steps=TRAIN_CONFIG["eval_steps"],
            evaluation_strategy=TRAIN_CONFIG["evaluation_strategy"],
            report_to="wandb",
        ),
    )

    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
