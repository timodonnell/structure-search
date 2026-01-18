# TokenFold

Train language models from scratch to predict protein structure from amino acid sequence.

## Overview

This project trains language models from scratch as sequence-to-structure translation models. Given an amino acid sequence, the model predicts the corresponding structural encoding.

**Supported architectures:**
- TinyLlama 1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
- Llama 3.2 1B (`meta-llama/Llama-3.2-1B`)

**Output formats:**

1. **3Di mode**: Predicts Foldseek's 20-letter structural alphabet
2. **Kanzi mode**: Predicts 1000 discrete tokens that can be decoded to 3D coordinates

### What is 3Di?

[Foldseek](https://github.com/steineggerlab/foldseek) uses a 20-letter structural alphabet called 3Di that encodes local protein backbone geometry. Each 3Di character represents a discretized description of the local 3D environment around an amino acid. This allows structure comparison to be performed as fast sequence alignment.

### What is Kanzi?

[Kanzi](https://github.com/microsoft/kanzi) is a flow-based autoencoder that encodes protein backbone coordinates into 1000 discrete tokens using Finite Scalar Quantization (FSQ). Unlike 3Di which only captures local geometry, Kanzi tokens can be decoded back to full 3D C-alpha coordinates, enabling direct structure prediction with RMSD evaluation.

### Training Approach

The model is trained as a causal language model on sequence pairs:

**3Di mode:**
```
<AA> M K T L K D L L K ... <SEP> <3Di> D D P L V V V L V ...
```

**Kanzi mode:**
```
<AA> M K T L K D L L K ... <SEP> <KANZI> <K599> <K358> <K127> ...
```

- Input: Amino acid sequence with `<AA>` prefix (space-separated)
- Output: Structure encoding with `<3Di>` or `<KANZI>` prefix
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

### Kanzi Mode (Recommended)

Train from scratch with a minimal vocabulary containing only amino acids and Kanzi tokens (~1028 tokens total):

**Using TinyLlama 1.1B architecture:**
```bash
WANDB_API_KEY="your_key" WANDB_PROJECT="tokenfold" \
uv run python -m tokenfold.train_kanzi \
    --from-scratch \
    --model-name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --no-flash-attn \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --log-interval 50 \
    --eval-interval 500 \
    --rmsd-eval-interval 1000 \
    --rmsd-eval-samples 20
```

**Using Llama 3.2 1B architecture:**
```bash
WANDB_API_KEY="your_key" WANDB_PROJECT="tokenfold" \
uv run python -m tokenfold.train_kanzi \
    --from-scratch \
    --model-name meta-llama/Llama-3.2-1B \
    --no-flash-attn \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --log-interval 50 \
    --eval-interval 500 \
    --rmsd-eval-interval 1000 \
    --rmsd-eval-samples 20
```

> **Note:** Llama 3.2 is a gated model. You need to accept the license on HuggingFace and set `HF_TOKEN` environment variable.

### 3Di Mode

```bash
# With accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m tokenfold.train \
    --model-name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-4 \
    --num-epochs 3
```

### Training Arguments (Kanzi mode)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Base model architecture |
| `--from-scratch` | False | Train from scratch with minimal vocab (~1028 tokens) |
| `--db-path` | `data/foldseek/afdb50/afdb50` | Path to Foldseek database |
| `--output-dir` | `outputs/kanzi_predictor` | Output directory |
| `--batch-size` | 4 | Per-GPU batch size |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--learning-rate` | 1e-4 | Learning rate |
| `--max-length` | 1024 | Maximum sequence length |
| `--max-protein-length` | 400 | Maximum protein length to include |
| `--eval-interval` | 500 | Steps between loss evaluation |
| `--rmsd-eval-interval` | 500 | Steps between RMSD evaluation |
| `--rmsd-eval-samples` | 25 | Number of samples for RMSD eval |
| `--max-steps` | -1 | Max steps (-1 for full training) |

### Hardware Requirements

- **Recommended**: 1x NVIDIA H100 80GB
- Both TinyLlama 1.1B and Llama 3.2 1B fit comfortably on a single GPU
- ~46M training examples from AlphaFold DB

## Project Structure

```
tokenfold/
├── checkpoints/                # Model checkpoints (not in git)
│   └── cleaned_model.pt        # Kanzi model weights
├── configs/
│   ├── accelerate_config.yaml  # Accelerate config
│   └── deepspeed_zero3.json    # DeepSpeed ZeRO-3 config
├── data/
│   └── foldseek/               # Symlink to Foldseek databases
│       └── afdb50/             # AlphaFold DB @ 50% identity
├── src/
│   └── tokenfold/
│       ├── __init__.py
│       ├── dataset.py          # Dataset classes (3Di and Kanzi)
│       ├── foldseek_db.py      # Foldseek DB reader (incl. C-alpha)
│       ├── kanzi_tokenizer.py  # Kanzi encode/decode wrapper
│       ├── train.py            # 3Di training script
│       └── train_kanzi.py      # Kanzi training script
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

- **Base**: TinyLlama 1.1B or Llama 3.2 1B (trained from scratch on protein data)
- **Vocabulary**: Minimal vocab with amino acids + Kanzi tokens (~1028 tokens)
- **Precision**: bfloat16
- **Training**: Full model from random initialization, no LoRA

## Wandb Logging

Set your Wandb API key to enable experiment tracking:

```bash
export WANDB_API_KEY=your_key_here
```

## License

MIT
