#!/usr/bin/env python3
"""
Fine-tune a language model for domain-specific relation extraction using TRL SFT.

Usage:
    python scripts/finetune_relations.py \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --model-name Qwen/Qwen3-1.7B \
        --output-dir models/finetuned-relations

Requirements:
    pip install torch transformers trl datasets peft accelerate

This script:
1. Loads training data in messages format (JSONL with system/user/assistant)
2. Fine-tunes the model using TRL's SFTTrainer with LoRA (PEFT)
3. Saves the merged model + tokenizer for ONNX export
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_messages_jsonl(path: str) -> list[dict]:
    """Load JSONL file in messages format."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def validate_sample(sample: dict) -> bool:
    """Validate a training sample has the expected structure."""
    if "messages" not in sample:
        return False
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [m.get("role") for m in messages]
    return "user" in roles and "assistant" in roles


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for relation extraction with TRL SFT"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training JSONL file (messages format)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation JSONL file (optional)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned-relations",
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 training (requires Ampere+ GPU or Apple Silicon)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate data and config without training",
    )
    args = parser.parse_args()

    # ── 1. Load and validate data ──────────────────────────────────────
    print(f"Loading training data from {args.train_data}...")
    train_samples = load_messages_jsonl(args.train_data)
    print(f"  Loaded {len(train_samples)} training samples")

    invalid = [i for i, s in enumerate(train_samples) if not validate_sample(s)]
    if invalid:
        print(f"  WARNING: {len(invalid)} invalid samples at lines: {invalid[:10]}")
        train_samples = [s for i, s in enumerate(train_samples) if i not in set(invalid)]

    if not train_samples:
        print("ERROR: No valid training samples found")
        sys.exit(1)

    val_samples = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_samples = load_messages_jsonl(args.val_data)
        val_samples = [s for s in val_samples if validate_sample(s)]
        print(f"  Loaded {len(val_samples)} validation samples")

    if args.dry_run:
        print("\n=== Dry Run Summary ===")
        print(f"Model: {args.model_name}")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples) if val_samples else 0}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
        print(f"Output: {args.output_dir}")
        print("Dry run complete. Data is valid.")
        sys.exit(0)

    # ── 2. Import ML libraries (deferred to avoid slow import on dry-run) ──
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install torch transformers trl datasets peft accelerate")
        sys.exit(1)

    # ── 3. Load model and tokenizer ────────────────────────────────────
    print(f"\nLoading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"  Model loaded: {model.config.model_type}, {model.num_parameters():,} params")

    # ── 4. Prepare datasets ────────────────────────────────────────────
    def format_messages(sample):
        """Apply chat template to messages."""
        messages = sample["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback: concatenate messages manually
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|{role}|>\n{content}")
            text = "\n".join(parts)
        return {"text": text}

    train_dataset = Dataset.from_list(train_samples).map(format_messages)
    eval_dataset = None
    if val_samples:
        eval_dataset = Dataset.from_list(val_samples).map(format_messages)

    print(f"  Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"  Eval dataset: {len(eval_dataset)} samples")

    # ── 5. Configure LoRA ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # ── 6. Training arguments ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    # ── 7. Train ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    print("\nStarting training...")
    result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Train runtime: {result.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Samples/sec: {result.metrics.get('train_samples_per_second', 0):.1f}")

    # ── 8. Save model (merge LoRA weights) ─────────────────────────────
    print(f"\nSaving merged model to {args.output_dir}...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    metadata = {
        "base_model": args.model_name,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples) if val_samples else 0,
        "final_loss": result.training_loss,
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {args.output_dir}/")
    print("Next step: Run scripts/export_to_onnx.py to convert to ONNX format")


if __name__ == "__main__":
    main()
