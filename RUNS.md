# Training Runs

This file documents training runs for the structure prediction model.

---

## Run 3: Optimized Batch Size (2026-01-16) ✅ CURRENT

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/m6oumnrn**

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v7 \
    --batch-size 48 \
    --gradient-accumulation-steps 1 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 250
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B` |
| Dataset | afdb50 (66.7M proteins) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Batch size (per GPU) | 48 |
| Gradient accumulation | 1 |
| Effective batch size | 384 (48 × 8 GPUs) |
| Learning rate | 2e-4 |
| Max sequence length | 1024 tokens |
| Precision | bfloat16 |

### Notes
- Fixed wandb logging with `accelerator.init_trackers()` and `float()` conversion
- Optimized batch size from 4 to 48 per GPU after OOM testing
- No gradient accumulation needed with larger batch size

---

## Run 2: Fixed Tokenization (2026-01-16)

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/5lv9i3dc**

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v2 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500
```

### Fix Applied
**Space-separated tokenization** to ensure 1:1 alignment between amino acids and 3Di characters.

Before (broken):
```
<AA>MKTLKDLLK  →  tokens: ['MK', 'TL', 'K', 'DLL', 'K']  (merged)
```

After (fixed):
```
<AA> M K T L K D L L K  →  tokens: ['<AA>', 'ĠM', 'ĠK', 'ĠT', 'ĠL', ...]  (1:1)
```

### Configuration
Same as Run 1, but with corrected tokenization.

---

## Run 1: Initial Training (2026-01-16) ❌ CANCELLED - Tokenization Bug

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/nm9f8ymf**

### Command
```bash
export WANDB_API_KEY="<your-key>"
export WANDB_PROJECT="structure-prediction"

accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.1-8B` |
| Dataset | afdb50 (66.7M proteins) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Batch size (per GPU) | 4 |
| Gradient accumulation | 8 |
| Effective batch size | 256 (4 × 8 GPUs × 8 accum) |
| Learning rate | 2e-4 |
| Warmup ratio | 3% |
| Max sequence length | 1024 tokens |
| Precision | bfloat16 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight decay | 0.01 |

### Hardware
- 8× NVIDIA H100 80GB HBM3
- DeepSpeed ZeRO-3 (no offloading)
- Flash Attention 2

### Description

First training run to fine-tune Llama 3.1 8B for sequence-to-structure prediction. The model learns to translate amino acid sequences to Foldseek's 3Di structural alphabet.

**Training format:**
```
<AA>MKTLKDLLKEKQNLIK...<SEP><3Di>DDPLVVVLVVLVVVLVV...
```

Loss is computed only on the structure tokens (after `<SEP>`), so the model learns to predict 3Di given the amino acid sequence.

### Notes
- **CANCELLED** after ~50 steps due to tokenization bug
- BPE tokenizer was merging amino acids (e.g., "MK", "DLL") breaking 1:1 alignment
- Loss was ~30 which is abnormally high
- See Run 2 for the corrected version
