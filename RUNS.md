# Training Runs

This file documents training runs for the structure prediction model.

---

## Run 9: TinyLlama Full Fine-tune (2026-01-17) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/structure-search/runs/9c49x0fd**

### Status
Completed 1000 steps of full fine-tuning on TinyLlama 1.1B.

### Command
```bash
WANDB_API_KEY="<key>" WANDB_PROJECT="structure-search" \
uv run python -m structure_search.train \
    --mode tinyllama-full \
    --output-dir outputs/tinyllama-run2 \
    --max-steps 1000 \
    --no-flash-attn
```

### Results

| Metric | Initial (Step 0) | Final (Step 1000) |
|--------|------------------|-------------------|
| Eval Loss | 2.1456 | 1.3841 |
| Gen Token Accuracy | 4.6% | **23.2%** |
| Length Match Rate | 2% | **18%** |
| Valid Chars Rate | 12% | **100%** |

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | TinyLlama-1.1B-intermediate-step-1431k-3T |
| Mode | Full fine-tuning (no LoRA) |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Max steps | 1000 |
| GPUs | 1x H100 80GB |
| Duration | ~34 minutes |

### Notes
- Model learned to produce 100% valid 3Di characters
- Token accuracy improved from 4.6% → 23.2%
- Saved to `outputs/tinyllama-run2/final/`

---

## Run 8: TinyLlama Full Fine-tune (2026-01-17) ✅ COMPLETED

### Wandb Link
**https://wandb.ai/timodonnell/structure-search/runs/6uwnq1nt**

### Status
Completed 1000 steps of full fine-tuning on TinyLlama 1.1B.

### Results

| Metric | Initial (Step 0) | Final (Step 1000) |
|--------|------------------|-------------------|
| Eval Loss | 2.1457 | 1.3848 |
| Gen Token Accuracy | 4.6% | **22.2%** |
| Length Match Rate | 2% | 4% |
| Valid Chars Rate | 12% | **100%** |

### Configuration
Same as Run 9. Saved to `outputs/tinyllama-run/final/`.

### Notes
- First successful TinyLlama training run
- Duration: ~39 minutes

---

## Run 7: ProstT5 Baseline Only (2026-01-16) ✅ CURRENT

### Wandb Link
**(pending - check wandb for latest run)**

### Status
Training with simplified ProstT5 evaluation (baseline only, no model generation).

### Command
```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m structure_search.train \
    --mode llama-8b-lora \
    --db-path data/foldseek/afdb50/afdb50 \
    --output-dir outputs/structure_predictor_v18 \
    --batch-size 24 \
    --gradient-accumulation-steps 1 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 500 \
    --prostt5-eval-interval 1000 \
    --prostt5-eval-samples 50
```

### Key Fix
**Removed model generation from ProstT5 evaluation**: The previous crash was caused by calling `model.generate()` on only rank 0, which triggered NCCL collective operations that other ranks weren't participating in. Now we only evaluate ProstT5 baseline vs ground truth during training. Full model evaluation will be done separately.

### Configuration
Same as Run 6, with simplified ProstT5 evaluation.

---

## Run 6: Multi-Model Support + ProstT5 Fix (2026-01-16) - CRASHED

### Wandb Link
**(check wandb for run)**

### Status
Crashed at step 1000 during ProstT5 comparison - NCCL timeout due to model.generate() on single rank.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.55 | ~1.52 |
| 1000 | 1.53 | 1.52 |

### Crash Details
- NCCL timeout at step 1000 after eval completed
- Root cause: `model.generate()` triggers NCCL collective ops even when only called on rank 0
- Rank 0 had 320 more NCCL operations enqueued than other ranks
- Fixed in Run 7 by removing model generation from ProstT5 eval

### Features Added
1. **Multi-model support**: New `--mode` argument with presets:
   - `llama-8b-lora`: LoRA fine-tuning on Llama 3.1 8B (default)
   - `tinyllama-full`: Full fine-tuning on TinyLlama 1.1B

2. **ProstT5 comparison** (barrier fix was insufficient)

---

## Run 5: ProstT5 Comparison + Validity Metrics (2026-01-16) - CRASHED

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/yuaqtvnq**

### Status
Crashed at step 1000 during ProstT5 comparison due to NCCL timeout.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.6210 | 1.5777 |
| 1000 | 1.4777 | 1.5272 |

### Crash Details
- NCCL collective timeout at step 1000 when ProstT5 comparison started
- Cause: Only main process ran ProstT5 eval while other processes continued and timed out waiting for collective operations
- Fixed in Run 6 by adding `accelerator.wait_for_everyone()` barrier

---

## Run 4: Stable Training with Eval Fix (2026-01-16) - STOPPED

### Wandb Link
**https://wandb.ai/timodonnell/structure-prediction/runs/kr4u1yod**

### Status
Stopped at step ~1420 to restart with ProstT5 eval enabled.

| Step | Train Loss | Eval Loss |
|------|------------|-----------|
| 500 | 1.4827 | 1.5777 |
| 1000 | 1.4695 | 1.5221 |

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
    --output-dir outputs/structure_predictor_v15 \
    --batch-size 24 \
    --gradient-accumulation-steps 1 \
    --max-length 1024 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --log-interval 10 \
    --save-interval 2000 \
    --eval-interval 500
```

### Key Fixes
1. **Distributed evaluation fix**: Fixed NCCL timeout during evaluation
   - Use fixed iteration count (50 steps) with iterator reset
   - Proper gathering of losses across all GPUs with `accelerator.gather()`
   - Call `accelerator.save_state()` on all processes

2. **Gradient checkpointing**: Reduces memory for larger batch sizes

---

## Run 3: Optimized Batch Size (2026-01-16) - CRASHED

### Notes
- Crashed during evaluation due to NCCL collective timeout
- Fixed in Run 4
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
