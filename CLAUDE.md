# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokenFold trains language models (TinyLlama 1.1B) from scratch to predict protein structure from amino acid sequence. Two output modes are supported:

1. **3Di mode**: Predicts Foldseek's 20-letter structural alphabet
2. **Kanzi mode**: Predicts 1000 discrete tokens that decode to 3D C-alpha coordinates

## Key Commands

```bash
# Install dependencies
uv sync

# Train Kanzi model
uv run python -m tokenfold.train_kanzi --no-flash-attn

# Train 3Di model
uv run python -m tokenfold.train
```

## Data Directory

The `data/foldseek/afdb50/` contains AlphaFold Database at 50% sequence identity (~46M structures):
- `afdb50` - Amino acid sequences
- `afdb50_ss` - 3Di structure encodings
- `afdb50_ca` - C-alpha coordinates

## Package Structure

- `src/tokenfold/dataset.py` - Dataset classes for 3Di and Kanzi modes
- `src/tokenfold/foldseek_db.py` - Foldseek database reader
- `src/tokenfold/kanzi_tokenizer.py` - Kanzi encode/decode wrapper
- `src/tokenfold/train.py` - 3Di training script
- `src/tokenfold/train_kanzi.py` - Kanzi training script with RMSD evaluation
