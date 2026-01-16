"""
Dataset for sequence-to-structure prediction training.

Formats protein data for causal language modeling:
<|begin_of_text|><AA>SEQUENCE<SEP><3Di>STRUCTURE<|end_of_text|>
"""

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset

from .foldseek_db import PairedFoldseekDB


# Special tokens for sequence-to-structure format
AA_START = "<AA>"
SS_START = "<3Di>"
SEP_TOKEN = "<SEP>"


class StructurePredictionDataset(Dataset):
    """Map-style dataset for structure prediction training."""

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        split: str = "train",
        val_fraction: float = 0.001,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            db_path: Path to Foldseek database (e.g., 'data/foldseek/afdb50/afdb50')
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length for tokenization
            split: 'train' or 'val'
            val_fraction: Fraction of data for validation
            seed: Random seed for train/val split
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load index to get dataset size
        self.paired_db = PairedFoldseekDB(db_path)
        self.total_size = len(self.paired_db)

        # Create train/val split indices
        rng = random.Random(seed)
        all_indices = list(range(self.total_size))
        rng.shuffle(all_indices)

        val_size = int(self.total_size * val_fraction)
        if split == "val":
            self.indices = all_indices[:val_size]
        else:
            self.indices = all_indices[val_size:]

        # Open database handles
        self.paired_db.__enter__()

    def __len__(self) -> int:
        return len(self.indices)

    def __del__(self):
        try:
            self.paired_db.__exit__(None, None, None)
        except Exception:
            pass

    def format_example(self, aa_seq: str, ss_seq: str) -> str:
        """Format a sequence pair for training.

        Space-separates characters to ensure 1:1 token alignment.
        """
        aa_spaced = " ".join(aa_seq)
        ss_spaced = " ".join(ss_seq)
        return f"{AA_START} {aa_spaced} {SEP_TOKEN} {SS_START} {ss_spaced}"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        aa_seq, ss_seq = self.paired_db.get_pair(real_idx)

        # Format the text
        text = self.format_example(aa_seq, ss_seq)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask loss on input sequence (before <SEP>)
        labels = input_ids.clone()

        # Find <SEP> token position to mask input
        sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        if sep_token_id is not None and sep_token_id != self.tokenizer.unk_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                # Mask everything before and including <SEP>
                sep_pos = sep_positions[0].item()
                labels[:sep_pos + 1] = -100
        else:
            # Fallback: find position of <3Di> token pattern
            ss_start_id = self.tokenizer.convert_tokens_to_ids(SS_START)
            if ss_start_id is not None and ss_start_id != self.tokenizer.unk_token_id:
                ss_positions = (input_ids == ss_start_id).nonzero(as_tuple=True)[0]
                if len(ss_positions) > 0:
                    ss_pos = ss_positions[0].item()
                    labels[:ss_pos] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingStructureDataset(IterableDataset):
    """Memory-efficient streaming dataset for large protein databases."""

    def __init__(
        self,
        db_path: str | Path,
        tokenizer: Any,
        max_length: int = 1024,
        world_size: int = 1,
        rank: int = 0,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        """Initialize streaming dataset.

        Args:
            db_path: Path to Foldseek database
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            world_size: Total number of processes
            rank: Current process rank
            shuffle_buffer_size: Size of shuffle buffer
            seed: Random seed
        """
        self.db_path = Path(db_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.world_size = world_size
        self.rank = rank
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def format_example(self, aa_seq: str, ss_seq: str) -> str:
        """Format a sequence pair for training.

        Space-separates characters to ensure 1:1 token alignment.
        """
        aa_spaced = " ".join(aa_seq)
        ss_spaced = " ".join(ss_seq)
        return f"{AA_START} {aa_spaced} {SEP_TOKEN} {SS_START} {ss_spaced}"

    def process_example(self, aa_seq: str, ss_seq: str) -> dict[str, torch.Tensor] | None:
        """Process a single example."""
        # Skip if sequences are too long or mismatched
        if len(aa_seq) != len(ss_seq):
            return None
        if len(aa_seq) * 2 + 20 > self.max_length:  # Rough estimate with special tokens
            return None

        text = self.format_example(aa_seq, ss_seq)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask input sequence (before structure)
        sep_token_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        if sep_token_id is not None and sep_token_id != self.tokenizer.unk_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                labels[:sep_pos + 1] = -100

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self):
        """Iterate over dataset with sharding and shuffling."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Combine rank and worker for unique sharding
        effective_rank = self.rank * num_workers + worker_id
        effective_world_size = self.world_size * num_workers

        rng = random.Random(self.seed + effective_rank)
        buffer = []

        with PairedFoldseekDB(self.db_path) as db:
            for idx, aa_seq, ss_seq in db:
                # Shard by index
                if idx % effective_world_size != effective_rank:
                    continue

                example = self.process_example(aa_seq, ss_seq)
                if example is None:
                    continue

                buffer.append(example)

                if len(buffer) >= self.shuffle_buffer_size:
                    rng.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

            # Yield remaining items
            if buffer:
                rng.shuffle(buffer)
                for item in buffer:
                    yield item
