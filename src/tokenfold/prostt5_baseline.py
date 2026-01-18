#!/usr/bin/env python3
"""
Standalone script to run ProstT5 baseline evaluation and log to wandb.

This script uses the same fixed eval samples as the training loop for
consistent comparison between our model and ProstT5.

Usage:
    # Create a new wandb run
    python -m tokenfold.prostt5_baseline --db-path data/foldseek/afdb50/afdb50

    # Log to an existing wandb run
    python -m tokenfold.prostt5_baseline --db-path data/foldseek/afdb50/afdb50 --wandb-run-id abc123

    # Use eval samples saved during training
    python -m tokenfold.prostt5_baseline --eval-samples-file outputs/structure_predictor_v22/eval_samples.txt
"""

import argparse
import logging
import os
import random
import time

import torch
import wandb

from .evaluate import ProstT5Baseline, compute_token_accuracy, VALID_3DI_CHARS
from .foldseek_db import PairedFoldseekDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_fixed_eval_samples(db_path: str, num_samples: int = 100, seed: int = 42, max_seq_len: int = 300):
    """Get a fixed set of eval samples for consistent evaluation.

    Uses the same logic as train.py to ensure consistent samples.
    """
    rng = random.Random(seed)

    with PairedFoldseekDB(db_path) as db:
        total = len(db)
        # Use same seed as val split to get val indices
        all_indices = list(range(total))
        rng_split = random.Random(42)  # Same seed as dataset split
        rng_split.shuffle(all_indices)
        val_size = int(total * 0.001)  # Same as val_fraction
        val_indices = all_indices[:val_size]

        # Sample from val indices
        rng.shuffle(val_indices)

        samples = []
        for idx in val_indices:
            if len(samples) >= num_samples:
                break
            aa_seq, ss_seq = db.get_pair(idx)
            # Filter by length
            if len(aa_seq) <= max_seq_len and len(aa_seq) == len(ss_seq):
                samples.append((aa_seq, ss_seq))

        return samples


def load_eval_samples_from_file(filepath: str) -> list[tuple[str, str]]:
    """Load eval samples from a file saved during training."""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                samples.append((parts[0], parts[1]))
    return samples


def run_prostt5_baseline(
    eval_samples: list[tuple[str, str]],
    device: str = "cuda:0",
) -> dict:
    """Run ProstT5 baseline on eval samples.

    Args:
        eval_samples: List of (aa_seq, ss_seq) tuples
        device: Device to run on

    Returns:
        Dictionary with metrics
    """
    logger.info(f"Loading ProstT5 model on {device}...")
    load_start = time.time()
    prostt5 = ProstT5Baseline(device=torch.device(device))
    logger.info(f"ProstT5 loaded in {time.time() - load_start:.1f}s")

    correct_total = 0
    tokens_total = 0
    length_matches = 0
    valid_chars = 0

    logger.info(f"Running ProstT5 on {len(eval_samples)} samples...")
    eval_start = time.time()

    for i, (aa_seq, gt_3di) in enumerate(eval_samples):
        # Get ProstT5 prediction
        pred_3di = prostt5.predict([aa_seq])[0]

        # Compute token accuracy
        _, correct, total = compute_token_accuracy(pred_3di, gt_3di)
        correct_total += correct
        tokens_total += total

        # Check length match
        if len(pred_3di) == len(gt_3di):
            length_matches += 1

        # Check valid characters
        if pred_3di and all(c in VALID_3DI_CHARS for c in pred_3di.lower()):
            valid_chars += 1

        # Progress logging
        if (i + 1) % 10 == 0:
            elapsed = time.time() - eval_start
            eta = elapsed / (i + 1) * (len(eval_samples) - i - 1)
            current_acc = correct_total / tokens_total if tokens_total > 0 else 0
            logger.info(
                f"  Progress: {i + 1}/{len(eval_samples)} samples, "
                f"acc={current_acc:.4f} ({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)"
            )

    n = len(eval_samples)
    total_time = time.time() - eval_start

    metrics = {
        "prostt5_token_accuracy": correct_total / tokens_total if tokens_total > 0 else 0.0,
        "prostt5_length_match_rate": length_matches / n if n > 0 else 0.0,
        "prostt5_valid_char_rate": valid_chars / n if n > 0 else 0.0,
        "prostt5_num_samples": n,
        "prostt5_eval_time_seconds": total_time,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run ProstT5 baseline evaluation")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/foldseek/afdb50/afdb50",
        help="Path to Foldseek database",
    )
    parser.add_argument(
        "--eval-samples-file",
        type=str,
        default=None,
        help="Path to eval samples file saved during training (overrides --db-path)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=300,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Existing wandb run ID to log to (creates new run if not specified)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name (default: from WANDB_PROJECT env var)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Don't log to wandb",
    )

    args = parser.parse_args()

    # Load eval samples
    if args.eval_samples_file:
        logger.info(f"Loading eval samples from {args.eval_samples_file}")
        eval_samples = load_eval_samples_from_file(args.eval_samples_file)
        logger.info(f"Loaded {len(eval_samples)} samples from file")
    else:
        logger.info(f"Getting fixed eval samples from {args.db_path}")
        eval_samples = get_fixed_eval_samples(
            db_path=args.db_path,
            num_samples=args.num_samples,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
        )
        logger.info(f"Got {len(eval_samples)} samples")

    # Run ProstT5 baseline
    metrics = run_prostt5_baseline(eval_samples, device=args.device)

    # Print results
    logger.info("=" * 60)
    logger.info("PROSTT5 BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Token Accuracy:     {metrics['prostt5_token_accuracy']:.4f}")
    logger.info(f"Length Match Rate:  {metrics['prostt5_length_match_rate']:.4f}")
    logger.info(f"Valid Char Rate:    {metrics['prostt5_valid_char_rate']:.4f}")
    logger.info(f"Num Samples:        {metrics['prostt5_num_samples']}")
    logger.info(f"Eval Time:          {metrics['prostt5_eval_time_seconds']:.1f}s")
    logger.info("=" * 60)

    # Log to wandb
    if not args.no_wandb:
        wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "structure-prediction")

        if args.wandb_run_id:
            # Resume existing run
            logger.info(f"Logging to existing wandb run: {args.wandb_run_id}")
            wandb.init(
                project=wandb_project,
                id=args.wandb_run_id,
                resume="must",
            )
        else:
            # Create new run
            logger.info(f"Creating new wandb run in project: {wandb_project}")
            wandb.init(
                project=wandb_project,
                name="prostt5-baseline",
                config={
                    "model": "ProstT5",
                    "num_samples": len(eval_samples),
                    "seed": args.seed,
                    "max_seq_len": args.max_seq_len,
                },
            )

        # Log metrics at step 0 (baseline)
        wandb.log(metrics, step=0)
        logger.info("Logged metrics to wandb")
        wandb.finish()


if __name__ == "__main__":
    main()
