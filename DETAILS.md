# Structure Search: Technical Deep Dive

This document provides a detailed walkthrough of the codebase for training Llama 3.1 8B to predict protein structure from amino acid sequences.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Data Format](#data-format)
3. [Foldseek Database Reader](#foldseek-database-reader)
4. [Dataset Implementation](#dataset-implementation)
5. [Training Pipeline](#training-pipeline)
6. [Model Configuration](#model-configuration)
7. [Multi-GPU Training](#multi-gpu-training)

---

## Problem Overview

### The Goal

Given an amino acid sequence like:
```
MKTLKDLLKEKQNLIKEEILERKKLEVLTKKQQKDEIEHQK
```

Predict its 3D structure encoded as a 3Di sequence:
```
DDPLVVVLVVLVVVLVVLVVVDDPPQDPVNVVVSVVVSVVS
```

### What is 3Di?

Foldseek's 3Di is a 20-letter structural alphabet where each character encodes the local 3D geometry around an amino acid residue. Key properties:

- **20 characters**: Like amino acids, but encoding structure not chemistry
- **Position-aligned**: The i-th 3Di character describes the structure at the i-th amino acid
- **Captures local geometry**: Each character encodes backbone angles, distances to neighbors, etc.

This allows structure comparison to be done as fast string matching rather than expensive 3D alignment.

---

## Data Format

### MMseqs2/Foldseek Database Format

Foldseek uses the MMseqs2 database format, which consists of:

1. **Main data file** (e.g., `afdb50`): Binary blob containing all sequences concatenated
2. **Index file** (e.g., `afdb50.index`): Tab-separated mapping of (id, byte_offset, length)
3. **Structure file** (e.g., `afdb50_ss`): Same format but containing 3Di sequences

Example index entry:
```
0       0       78
1       78      187
2       265     104
```

This means:
- Entry 0 starts at byte 0, is 78 bytes long
- Entry 1 starts at byte 78, is 187 bytes long
- etc.

---

## Foldseek Database Reader

### Location: `src/structure_search/foldseek_db.py`

### Core Classes

#### `FoldseekDB` - Single Database Reader

```python
from structure_search.foldseek_db import FoldseekDB

# Open a single database (amino acid sequences)
with FoldseekDB("data/foldseek/afdb50/afdb50") as db:
    print(f"Total entries: {len(db)}")  # 66,725,340

    # Get a single sequence by index
    seq = db.get_sequence(0)
    print(seq)  # "MTFLKPYFPRTHGFVDQATLRDTALMMPEHPEAPNTDPLYTCFCVAPRLI..."

    # Iterate over all sequences
    for entry_id, sequence in db:
        process(sequence)
        if entry_id >= 100:
            break
```

#### `PairedFoldseekDB` - Paired AA + Structure Reader

```python
from structure_search.foldseek_db import PairedFoldseekDB

# Open paired databases (automatically finds both AA and _ss files)
with PairedFoldseekDB("data/foldseek/afdb50/afdb50") as db:
    print(f"Total pairs: {len(db)}")

    # Get a single pair
    aa_seq, ss_seq = db.get_pair(0)
    print(f"AA:  {aa_seq[:50]}...")
    print(f"3Di: {ss_seq[:50]}...")
    print(f"Same length: {len(aa_seq) == len(ss_seq)}")  # True

    # Iterate over all pairs
    for idx, aa_seq, ss_seq in db:
        train_on(aa_seq, ss_seq)
```

### Implementation Details

The reader uses memory-mapped I/O for efficiency:

```python
class FoldseekDB:
    def __enter__(self):
        self._file = open(self.db_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def get_sequence(self, idx: int) -> str:
        _, offset, length = self._index[idx]
        data = self._mmap[offset:offset + length]
        return data.rstrip(b"\x00").decode("utf-8")
```

Benefits:
- **Zero-copy access**: OS handles paging, no need to load entire file
- **Random access**: O(1) access to any sequence by index
- **Memory efficient**: Can handle 67M sequences without loading all into RAM

---

## Dataset Implementation

### Location: `src/structure_search/dataset.py`

### Training Format

Each training example is formatted as:

```
<AA>MKTLKDLLKEKQNLIK...<SEP><3Di>DDPLVVVLVVLVVVLVV...
```

Where:
- `<AA>` marks the start of amino acid sequence
- `<SEP>` separates input from output
- `<3Di>` marks the start of structure sequence

### Special Tokens

```python
AA_START = "<AA>"
SS_START = "<3Di>"
SEP_TOKEN = "<SEP>"
```

These are added to the tokenizer:

```python
tokenizer.add_special_tokens({
    "additional_special_tokens": [AA_START, SS_START, SEP_TOKEN]
})
```

### Label Masking

For causal LM training, we only compute loss on the structure tokens:

```python
def __getitem__(self, idx):
    # ... tokenize text ...

    # Create labels (copy of input_ids)
    labels = input_ids.clone()

    # Find <SEP> position
    sep_positions = (input_ids == sep_token_id).nonzero()
    if len(sep_positions) > 0:
        sep_pos = sep_positions[0].item()
        # Mask everything before and including <SEP>
        labels[:sep_pos + 1] = -100  # -100 = ignore in loss

    # Also mask padding
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

This means:
- Loss is computed only on `<3Di>DDPLVVV...` (the structure prediction)
- The model learns: given sequence, predict structure

### Dataset Classes

#### `StructurePredictionDataset` - Map-style Dataset

```python
from transformers import AutoTokenizer
from structure_search.dataset import StructurePredictionDataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

dataset = StructurePredictionDataset(
    db_path="data/foldseek/afdb50/afdb50",
    tokenizer=tokenizer,
    max_length=1024,
    split="train",      # or "val"
    val_fraction=0.001, # 0.1% for validation
)

print(f"Training examples: {len(dataset)}")  # ~66.6M

# Get a single example
example = dataset[0]
print(example["input_ids"].shape)      # torch.Size([1024])
print(example["attention_mask"].shape) # torch.Size([1024])
print(example["labels"].shape)         # torch.Size([1024])
```

#### `StreamingStructureDataset` - Iterable Dataset

For memory-efficient training with very large datasets:

```python
from structure_search.dataset import StreamingStructureDataset

dataset = StreamingStructureDataset(
    db_path="data/foldseek/afdb50/afdb50",
    tokenizer=tokenizer,
    max_length=1024,
    world_size=8,           # Total GPUs
    rank=0,                 # This GPU's rank
    shuffle_buffer_size=10000,
)

for batch in DataLoader(dataset, batch_size=4):
    train_step(batch)
```

---

## Training Pipeline

### Location: `src/structure_search/train.py`

### Model Creation

```python
def create_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.1-8B",
    use_lora: bool = True,
    use_4bit: bool = False,
    use_flash_attn: bool = True,
):
    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<AA>", "<3Di>", "<SEP>"]
    })

    # Load model with Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,                    # LoRA rank
            lora_alpha=128,          # LoRA alpha
            lora_dropout=0.05,
            target_modules=[         # Which layers to adapt
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer
```

### Training Loop

```python
def train(...):
    # Initialize distributed training
    accelerator = Accelerator(
        log_with="wandb" if os.environ.get("WANDB_API_KEY") else None,
    )

    # Create model, optimizer, scheduler
    model, tokenizer = create_model_and_tokenizer(...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, ...)

    # Prepare for distributed training
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log to Wandb
            accelerator.log({"train_loss": loss, "lr": scheduler.get_last_lr()[0]})
```

---

## Model Configuration

### LoRA Configuration

We use LoRA (Low-Rank Adaptation) to efficiently fine-tune the 8B parameter model:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r` | 64 | Rank of low-rank matrices |
| `lora_alpha` | 128 | Scaling factor (alpha/r = 2) |
| `lora_dropout` | 0.05 | Dropout for regularization |
| `target_modules` | All attention + MLP | Which layers to adapt |

This results in:
- **Trainable parameters**: 167M (2% of 8B)
- **Memory savings**: ~4x less GPU memory than full fine-tuning
- **Training speed**: Similar to full fine-tuning

### Why These Target Modules?

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP (SwiGLU)
]
```

- **Attention layers**: Enable the model to learn new attention patterns for protein sequences
- **MLP layers**: Enable learning of new feature transformations

---

## Multi-GPU Training

### DeepSpeed ZeRO-3 Configuration

Location: `configs/deepspeed_zero3.json`

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 4
}
```

### ZeRO-3 Memory Distribution

ZeRO Stage 3 partitions:
- **Optimizer states** across GPUs
- **Gradients** across GPUs
- **Parameters** across GPUs

This allows training models that don't fit on a single GPU.

### Effective Batch Size Calculation

```
Effective batch size = micro_batch × num_gpus × gradient_accumulation
                     = 4 × 8 × 8
                     = 256
```

### Accelerate Configuration

Location: `configs/accelerate_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: configs/deepspeed_zero3.json
  zero3_init_flag: true
num_processes: 8
num_machines: 1
```

### Launching Training

```bash
# Using accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    -m structure_search.train \
    --model-name meta-llama/Llama-3.1-8B \
    --db-path data/foldseek/afdb50/afdb50 \
    --batch-size 4 \
    --learning-rate 2e-4

# Or using the convenience script
./scripts/train.sh
```

---

## Example: End-to-End Inference

After training, use the model for structure prediction:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "outputs/structure_predictor/final/adapter")

# Load tokenizer with special tokens
tokenizer = AutoTokenizer.from_pretrained("outputs/structure_predictor/final/tokenizer")

# Predict structure for a new sequence
aa_sequence = "MKTLKDLLKEKQNLIKEEILERKKLEVLTK"
prompt = f"<AA>{aa_sequence}<SEP><3Di>"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=len(aa_sequence),  # Structure same length as sequence
    do_sample=False,
)

predicted_structure = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(predicted_structure)
# Output: <AA>MKTLKDLLKEKQNLIKEEILERKKLEVLTK<SEP><3Di>DDPLVVVLVVLVVVLVVLVVVDDPPQDPVN
```

---

## Performance Characteristics

### Training Speed (8x H100 80GB)

| Phase | Time per Step |
|-------|---------------|
| First step (JIT warmup) | ~50s |
| Subsequent steps | ~8-10s |
| Steps per hour | ~400 |

### Memory Usage

| Component | Memory per GPU |
|-----------|----------------|
| Model parameters (sharded) | ~2 GB |
| Activations | ~50 GB |
| **Total** | ~56 GB |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total proteins | 66,725,340 |
| Training set (99.9%) | 66,658,615 |
| Validation set (0.1%) | 66,725 |
| Avg sequence length | ~300 AA |
| Total training steps (1 epoch) | 16,664,654 |
