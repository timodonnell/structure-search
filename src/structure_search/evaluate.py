#!/usr/bin/env python3
"""
Evaluation module for comparing structure prediction accuracy against ProstT5.
"""

import re
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer


class ProstT5Baseline:
    """ProstT5 baseline for amino acid to 3Di structure prediction."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ProstT5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(self.device)

        # Use half precision on GPU
        if self.device.type == "cuda":
            self.model.half()
        else:
            self.model.float()

        self.model.eval()

    def preprocess_sequence(self, aa_seq: str) -> str:
        """Preprocess amino acid sequence for ProstT5."""
        # Replace non-standard amino acids with X
        aa_seq = re.sub(r"[UZOB]", "X", aa_seq.upper())
        # Space-separate the sequence
        spaced = " ".join(list(aa_seq))
        # Add AA-to-3Di prefix
        return f"<AA2fold> {spaced}"

    @torch.no_grad()
    def predict(self, aa_sequences: list[str], max_length: int = 512) -> list[str]:
        """Predict 3Di structure tokens from amino acid sequences.

        Args:
            aa_sequences: List of amino acid sequences
            max_length: Maximum output length

        Returns:
            List of predicted 3Di structure strings
        """
        # Preprocess sequences
        processed = [self.preprocess_sequence(s) for s in aa_sequences]

        # Tokenize
        inputs = self.tokenizer.batch_encode_plus(
            processed,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        # Generate predictions
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=3,
            do_sample=False,
        )

        # Decode results
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Remove spaces to get 3Di string
        return ["".join(t.split()) for t in decoded]


def compute_token_accuracy(pred: str, target: str) -> tuple[float, int, int]:
    """Compute per-token accuracy between prediction and target.

    Args:
        pred: Predicted 3Di string
        target: Ground truth 3Di string

    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    # Align sequences by length
    min_len = min(len(pred), len(target))
    if min_len == 0:
        return 0.0, 0, 0

    correct = sum(p == t for p, t in zip(pred[:min_len], target[:min_len]))
    return correct / min_len, correct, min_len


def evaluate_accuracy(
    our_model,
    our_tokenizer,
    prostt5: ProstT5Baseline,
    eval_samples: list[tuple[str, str]],
    device: torch.device,
    max_length: int = 512,
) -> dict:
    """Evaluate our model against ProstT5 baseline.

    Args:
        our_model: Our trained model
        our_tokenizer: Our tokenizer
        prostt5: ProstT5 baseline model
        eval_samples: List of (amino_acid_seq, ground_truth_3di) tuples
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        Dictionary with evaluation metrics
    """
    from .dataset import AA_START, SEP_TOKEN, SS_START

    our_model.eval()

    our_correct = 0
    our_total = 0
    prostt5_correct = 0
    prostt5_total = 0

    # Process in batches
    batch_size = 8
    for i in range(0, len(eval_samples), batch_size):
        batch = eval_samples[i : i + batch_size]
        aa_seqs = [s[0] for s in batch]
        gt_3di = [s[1] for s in batch]

        # Get ProstT5 predictions
        prostt5_preds = prostt5.predict(aa_seqs, max_length=max_length)

        # Get our model predictions
        our_preds = []
        for aa_seq in aa_seqs:
            # Format input: <AA> M K T L ... <SEP> <SS>
            aa_spaced = " ".join(aa_seq)
            prompt = f"{AA_START} {aa_spaced} {SEP_TOKEN} {SS_START}"

            inputs = our_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs = our_model.generate(
                    **inputs,
                    max_new_tokens=len(aa_seq) + 10,
                    do_sample=False,
                    pad_token_id=our_tokenizer.pad_token_id,
                )

            # Decode and extract 3Di portion
            decoded = our_tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract 3Di tokens after <SS>
            if SS_START in decoded:
                ss_part = decoded.split(SS_START)[1]
                # Remove special tokens and spaces
                ss_part = ss_part.replace(our_tokenizer.eos_token, "").strip()
                pred_3di = "".join(ss_part.split())
            else:
                pred_3di = ""

            our_preds.append(pred_3di)

        # Compute accuracies
        for pred, gt in zip(our_preds, gt_3di):
            acc, correct, total = compute_token_accuracy(pred, gt)
            our_correct += correct
            our_total += total

        for pred, gt in zip(prostt5_preds, gt_3di):
            acc, correct, total = compute_token_accuracy(pred, gt)
            prostt5_correct += correct
            prostt5_total += total

    results = {
        "our_accuracy": our_correct / our_total if our_total > 0 else 0.0,
        "our_correct": our_correct,
        "our_total": our_total,
        "prostt5_accuracy": prostt5_correct / prostt5_total if prostt5_total > 0 else 0.0,
        "prostt5_correct": prostt5_correct,
        "prostt5_total": prostt5_total,
        "num_samples": len(eval_samples),
    }

    return results


def run_baseline_comparison(
    checkpoint_path: str,
    db_path: str,
    num_samples: int = 100,
    max_length: int = 512,
):
    """Run a comparison between our model and ProstT5.

    Args:
        checkpoint_path: Path to our model checkpoint
        db_path: Path to Foldseek database
        num_samples: Number of samples to evaluate
        max_length: Maximum sequence length
    """
    import logging
    from pathlib import Path

    from accelerate import Accelerator
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .dataset import StructurePredictionDataset
    from .train import add_special_tokens

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load our model
    logger.info(f"Loading our model from {checkpoint_path}")
    checkpoint_dir = Path(checkpoint_path)

    # Find the base model name from config
    base_model = "meta-llama/Llama-3.1-8B"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    add_special_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter if exists
    adapter_path = checkpoint_dir / "adapter"
    if adapter_path.exists():
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Try loading accelerator state
        logger.info("Loading from accelerator checkpoint")
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        accelerator.load_state(checkpoint_path)

    model.eval()

    # Load ProstT5 baseline
    logger.info("Loading ProstT5 baseline...")
    prostt5 = ProstT5Baseline(device=device)

    # Load evaluation samples
    logger.info(f"Loading evaluation samples from {db_path}")
    from .foldseek_db import PairedFoldseekDB

    db = PairedFoldseekDB(db_path)

    # Get random samples
    import random

    indices = random.sample(range(len(db)), min(num_samples, len(db)))
    eval_samples = []
    for idx in indices:
        aa_seq, ss_seq = db.get_pair(idx)
        # Filter by length
        if len(aa_seq) <= max_length // 3 and len(aa_seq) == len(ss_seq):
            eval_samples.append((aa_seq, ss_seq))
        if len(eval_samples) >= num_samples:
            break

    logger.info(f"Evaluating on {len(eval_samples)} samples...")

    # Run evaluation
    results = evaluate_accuracy(
        our_model=model,
        our_tokenizer=tokenizer,
        prostt5=prostt5,
        eval_samples=eval_samples,
        device=device,
        max_length=max_length,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Number of samples: {results['num_samples']}")
    logger.info(f"Our model accuracy:    {results['our_accuracy']:.4f} ({results['our_correct']}/{results['our_total']})")
    logger.info(f"ProstT5 accuracy:      {results['prostt5_accuracy']:.4f} ({results['prostt5_correct']}/{results['prostt5_total']})")
    logger.info("=" * 60)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model against ProstT5")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--db-path", type=str, default="data/foldseek/afdb50/afdb50", help="Path to Foldseek DB")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")

    args = parser.parse_args()

    run_baseline_comparison(
        checkpoint_path=args.checkpoint,
        db_path=args.db_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
