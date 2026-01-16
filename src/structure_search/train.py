#!/usr/bin/env python3
"""
Training script for sequence-to-structure prediction using Llama 3.1 8B.

Uses DeepSpeed ZeRO-3 for multi-GPU training on 8x H100 GPUs.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from .dataset import AA_START, SEP_TOKEN, SS_START, StructurePredictionDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer):
    """Add special tokens for structure prediction."""
    special_tokens = {
        "additional_special_tokens": [AA_START, SS_START, SEP_TOKEN],
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens")
    return num_added


def create_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_flash_attn: bool = True,
):
    """Create model and tokenizer with optional LoRA and quantization."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add special tokens
    add_special_tokens(tokenizer)

    # Quantization config
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    if use_lora:
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def create_dataloaders(
    tokenizer,
    db_path: str,
    batch_size: int = 4,
    max_length: int = 1024,
    num_workers: int = 4,
):
    """Create train and validation dataloaders."""

    train_dataset = StructurePredictionDataset(
        db_path=db_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
    )

    val_dataset = StructurePredictionDataset(
        db_path=db_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train(
    model_name: str = "meta-llama/Llama-3.1-8B",
    db_path: str = "data/foldseek/afdb50/afdb50",
    output_dir: str = "outputs/structure_predictor",
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_length: int = 1024,
    warmup_ratio: float = 0.03,
    use_lora: bool = True,
    use_4bit: bool = False,
    use_flash_attn: bool = True,
    log_interval: int = 10,
    save_interval: int = 1000,
    eval_interval: int = 500,
    max_steps: int = -1,
    resume_from: str | None = None,
):
    """Main training function."""

    # Initialize accelerator (mixed_precision and gradient_accumulation handled by DeepSpeed config)
    accelerator = Accelerator(
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using {accelerator.num_processes} GPUs")

    # Initialize Wandb tracking
    if os.environ.get("WANDB_API_KEY"):
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "structure-prediction"),
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_length": max_length,
                "use_lora": use_lora,
            },
        )

    # Create model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = create_model_and_tokenizer(
        model_name=model_name,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_flash_attn=use_flash_attn,
    )

    # Create dataloaders
    logger.info(f"Loading data from: {db_path}")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        db_path=db_path,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Calculate training steps (use accelerator's gradient_accumulation_steps from config)
    grad_accum = accelerator.gradient_accumulation_steps
    num_training_steps = len(train_loader) * num_epochs // grad_accum
    if max_steps > 0:
        num_training_steps = min(num_training_steps, max_steps)
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Warmup steps: {num_warmup_steps}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Resume from checkpoint
    starting_epoch = 0
    global_step = 0
    if resume_from:
        logger.info(f"Resuming from: {resume_from}")
        accelerator.load_state(resume_from)

    # Training loop
    model.train()
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    running_loss = 0.0
    for epoch in range(starting_epoch, num_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}/{num_training_steps} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )
                    accelerator.log(
                        {"train_loss": avg_loss, "learning_rate": lr, "step": global_step},
                        step=global_step,
                    )
                    running_loss = 0.0

                # Evaluation
                if global_step % eval_interval == 0:
                    model.eval()
                    eval_loss = 0.0
                    eval_steps = 0
                    with torch.no_grad():
                        for eval_batch in val_loader:
                            outputs = model(
                                input_ids=eval_batch["input_ids"],
                                attention_mask=eval_batch["attention_mask"],
                                labels=eval_batch["labels"],
                            )
                            eval_loss += outputs.loss.detach().float()
                            eval_steps += 1
                            if eval_steps >= 50:  # Limit eval steps
                                break

                    avg_eval_loss = eval_loss / eval_steps
                    if accelerator.is_main_process:
                        logger.info(f"Step {global_step} | Eval Loss: {avg_eval_loss:.4f}")
                    accelerator.log({"eval_loss": avg_eval_loss}, step=global_step)
                    model.train()

                # Save checkpoint
                if global_step % save_interval == 0:
                    if accelerator.is_main_process:
                        save_path = Path(output_dir) / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                if max_steps > 0 and global_step >= max_steps:
                    break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        final_path = Path(output_dir) / "final"
        accelerator.save_state(final_path)

        # Save model and tokenizer
        unwrapped_model = accelerator.unwrap_model(model)
        if use_lora:
            unwrapped_model.save_pretrained(final_path / "adapter")
        else:
            unwrapped_model.save_pretrained(final_path / "model")
        tokenizer.save_pretrained(final_path / "tokenizer")
        logger.info(f"Saved final model to {final_path}")

    accelerator.end_training()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train structure prediction model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model name",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/foldseek/afdb50/afdb50",
        help="Path to Foldseek database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/structure_predictor",
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--eval-interval", type=int, default=500, help="Eval interval")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        db_path=args.db_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        use_lora=not args.no_lora,
        use_4bit=args.use_4bit,
        use_flash_attn=not args.no_flash_attn,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        max_steps=args.max_steps,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
