# Structure Search

Train Llama 3.1 8B to predict protein structure from amino acid sequence using the Foldseek 3Di structural alphabet.

## Overview

This project fine-tunes Llama 3.1 8B as a sequence-to-structure translation model. Given an amino acid sequence, the model predicts the corresponding 3Di structural alphabet encoding from Foldseek.

### What is 3Di?

[Foldseek](https://github.com/steineggerlab/foldseek) uses a 20-letter structural alphabet called 3Di that encodes local protein backbone geometry. Each 3Di character represents a discretized description of the local 3D environment around an amino acid. This allows structure comparison to be performed as fast sequence alignment.

### Training Approach

The model is trained as a causal language model on sequence pairs:

```
<AA>MKTLKDLLKEKQNLIK...<SEP><3Di>DDPLVVVLVVLVVVLVV...
```

- Input: Amino acid sequence with `<AA>` prefix
- Output: 3Di structure encoding with `<3Di>` prefix
- Loss is computed only on the structure tokens (the model learns to predict structure given sequence)

## Data

The training data comes from the AlphaFold Database clustered at 50% sequence identity (afdb50):
- ~67 million protein structures
- Amino acid sequences paired with 3Di encodings
- Located in `data/foldseek/afdb50/`

## Setup

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Install flash attention (optional but recommended for H100s)
uv pip install flash-attn --no-build-isolation
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install flash-attn --no-build-isolation
```

## Training

### Quick Start

```bash
# Using the training script (8x GPU with DeepSpeed ZeRO-3)
./scripts/train.sh
```

### Custom Training

```bash
# With accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --num-epochs 3
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `meta-llama/Llama-3.1-8B` | Base model from Hugging Face |
| `--db-path` | `data/foldseek/afdb50/afdb50` | Path to Foldseek database |
| `--output-dir` | `outputs/structure_predictor` | Output directory |
| `--batch-size` | 4 | Per-GPU batch size |
| `--gradient-accumulation-steps` | 8 | Gradient accumulation steps |
| `--learning-rate` | 2e-4 | Learning rate |
| `--max-length` | 1024 | Maximum sequence length |
| `--num-epochs` | 3 | Number of training epochs |
| `--no-lora` | False | Disable LoRA (full fine-tuning) |
| `--use-4bit` | False | Enable 4-bit quantization |
| `--max-steps` | -1 | Max steps (-1 for full training) |

### Hardware Requirements

- **Recommended**: 8x NVIDIA H100 80GB (or equivalent)
- **Minimum**: 4x GPUs with 40GB+ VRAM each
- Uses DeepSpeed ZeRO-3 for memory-efficient training
- LoRA reduces trainable parameters to ~0.5% of the model

### Effective Batch Size

With default settings on 8 GPUs:
- Per-GPU batch size: 4
- Gradient accumulation: 8
- **Effective batch size**: 4 × 8 × 8 = 256

## Project Structure

```
structure-search/
├── configs/
│   ├── accelerate_config.yaml  # Accelerate config for 8 GPUs
│   └── deepspeed_zero3.json    # DeepSpeed ZeRO-3 config
├── data/
│   └── foldseek/               # Symlink to Foldseek databases
│       ├── afdb50/             # AlphaFold DB @ 50% identity
│       └── alphafold_swissprot # SwissProt structures
├── scripts/
│   └── train.sh                # Training launch script
├── src/
│   └── structure_search/
│       ├── __init__.py
│       ├── dataset.py          # Dataset classes
│       ├── foldseek_db.py      # Foldseek DB reader
│       └── train.py            # Training script
├── pyproject.toml
└── README.md
```

## Foldseek Database Format

Foldseek uses MMseqs2 database format:
- Main file: Contains sequences at byte offsets
- `.index`: Tab-separated (id, offset, length)
- `_ss`: 3Di structure encoding (same format)

The `PairedFoldseekDB` class reads both amino acid and structure databases in parallel.

## Model Architecture

- **Base**: Llama 3.1 8B
- **Fine-tuning**: LoRA (r=64, alpha=128)
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision**: bfloat16
- **Attention**: Flash Attention 2

## Wandb Logging

Set your Wandb API key to enable experiment tracking:

```bash
export WANDB_API_KEY=your_key_here
```

## License

MIT
