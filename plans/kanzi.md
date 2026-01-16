# Plan: Predicting Kanzi Tokens Instead of Foldseek 3Di

## Overview

Replace Foldseek's 3Di structural alphabet (20 tokens) with Kanzi discrete tokens (1000 tokens) for structure prediction. Kanzi is a flow-based autoencoder that provides:
- Better reconstruction quality (~0.53 Å RMSD on test proteins)
- Larger vocabulary enabling more nuanced structural representation
- Direct mapping to 3D C-alpha coordinates for validation

## Background

### Current Approach (Foldseek 3Di)
- 20-character alphabet (lowercase letters: `pdbvslathmigqnwyfkce`)
- Represents local structural neighborhoods
- No direct inverse mapping to coordinates
- Training format: `<AA> M K T L ... <SEP> <3Di> d d p l v v ...`

### Kanzi Tokens
- **Model**: 44M parameter flow-based autoencoder (DAE)
- **Vocabulary**: 1000 discrete tokens (via FSQ - Finite Scalar Quantization)
- **Input/Output**: C-alpha coordinates in nanometers
- **Reconstruction RMSD**: ~0.53 Å (tested on crambin, 46 residues)
- **GitHub**: https://github.com/rdilip/kanzi

## Verified Kanzi Functionality

Tested on crambin (PDB 1CRN, 46 residues):

```
Kanzi model: 44,117,745 parameters
Codebook size: 1000 tokens
Reconstruction RMSD: 0.526 Å
Token range: 3 to 995 (uses full vocabulary)
```

Example tokens for crambin:
```
[599, 358, 983, 769, 849, 680, 44, 120, 984, 575, 42, 176, ...]
```

## Data Requirements

### Current Foldseek Database Issue

The afdb50 database downloaded from Foldseek contains:
- `afdb50` - Amino acid sequences (18GB)
- `afdb50_ss` - 3Di structural tokens (18GB)
- `afdb50_ca` - C-alpha coordinates (6.4GB) **BUT no index file**

Without `afdb50_ca.index`, we cannot extract C-alpha coordinates for Kanzi tokenization.

### Solutions for C-alpha Coordinates

**Option A: Download Full AlphaFold/UniProt50 Database (Recommended for production)**
```bash
foldseek databases Alphafold/UniProt50 afdb50_full tmp/
```
- Size: ~950GB extracted
- Includes proper C-alpha coordinate indices

**Option B: Download Individual Structures on Demand**
```bash
# Example: Download AlphaFold structure for UniProt A0A2H3KIQ8
curl -O "https://alphafold.ebi.ac.uk/files/AF-A0A2H3KIQ8-F1-model_v4.cif"
```
- Suitable for validation/testing
- Can batch download for training subsets

**Option C: Generate Coordinates from ProstT5-predicted 3Di (Experimental)**
- Use inverse Foldseek reconstruction
- Lower quality than original coordinates

## Implementation Plan

### Phase 1: Data Pipeline

1. **Create C-alpha extraction module** (`src/structure_search/ca_extraction.py`)
   ```python
   def extract_ca_coords_from_cif(cif_path: str) -> np.ndarray:
       """Extract C-alpha coordinates from mmCIF file."""
       pass

   def download_alphafold_structure(uniprot_id: str, output_dir: str) -> str:
       """Download AlphaFold structure for a UniProt ID."""
       pass
   ```

2. **Create Kanzi tokenization module** (`src/structure_search/kanzi_tokenizer.py`)
   ```python
   class KanziTokenizer:
       def __init__(self, checkpoint_path: str):
           self.model = DAE.from_pretrained(checkpoint_path)

       def encode(self, ca_coords_nm: np.ndarray) -> list[int]:
           """Convert C-alpha coords (nm) to Kanzi tokens."""
           pass

       def decode(self, tokens: list[int]) -> np.ndarray:
           """Convert Kanzi tokens to C-alpha coords (nm)."""
           pass
   ```

3. **Create preprocessed dataset**
   - Download structures for afdb50 proteins
   - Extract C-alpha coordinates
   - Tokenize with Kanzi
   - Store as: `{uniprot_id: {"aa_seq": "MKT...", "kanzi_tokens": [599, 358, ...]}}`

### Phase 2: Training Modifications

1. **Update dataset class** (`src/structure_search/train.py`)
   - Current: Loads AA + 3Di from Foldseek database
   - New: Loads AA + Kanzi tokens from preprocessed dataset

2. **Update tokenization format**
   - Current: `<AA> M K T L ... <SEP> <3Di> d d p l v v ...`
   - New: `<AA> M K T L ... <SEP> <KANZI> 599 358 983 769 ...`

3. **Add Kanzi special tokens**
   - Add `<KANZI>` separator token
   - Token IDs: Map Kanzi 0-999 to new token range (e.g., 32001-33000)

4. **Update model vocabulary**
   ```python
   # Add 1000 Kanzi tokens + separator
   num_new_tokens = 1001  # 1000 Kanzi + <KANZI> separator
   tokenizer.add_tokens([f"<K{i}>" for i in range(1000)] + ["<KANZI>"])
   model.resize_token_embeddings(len(tokenizer))
   ```

### Phase 3: Validation Metrics

1. **Kanzi Token Accuracy**
   - Compare predicted Kanzi tokens vs ground truth
   - Per-position accuracy: `sum(pred[i] == gt[i]) / len(gt)`

2. **Reconstruction RMSD** (Key Metric)
   ```python
   def compute_rmsd_metric(pred_tokens, gt_tokens, kanzi_model):
       pred_coords = kanzi_model.decode(pred_tokens)  # (L, 3) in nm
       gt_coords = kanzi_model.decode(gt_tokens)      # (L, 3) in nm

       # Use Kabsch alignment for RMSD
       rmsd = kabsch_rmsd(pred_coords, gt_coords)
       return rmsd * 10.0  # Convert to Angstroms
   ```

3. **Comparison to ProstT5**
   - Keep existing ProstT5 comparison for AA-to-3Di
   - Add: ProstT5-to-Kanzi conversion for apples-to-apples comparison
   - Note: This requires mapping 3Di back to coordinates (lossy)

4. **Structure Validity Metrics**
   - Inter-CA distance distribution (should peak at ~3.8 Å)
   - Radius of gyration
   - Contact map similarity to ground truth

### Phase 4: Evaluation Script

Create `src/structure_search/evaluate_kanzi.py`:

```python
def evaluate_kanzi_predictions(
    model_path: str,
    test_data: list[dict],  # [{aa_seq, gt_kanzi_tokens}, ...]
    kanzi_checkpoint: str,
):
    """
    Evaluate LLM predictions on Kanzi token task.

    Metrics:
    - Token accuracy (exact match)
    - Reconstruction RMSD (Å)
    - RMSD distribution histogram
    - Comparison to baseline (e.g., ProstT5)
    """
    pass
```

## Training Command (Future)

```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 8 \
    -m structure_search.train \
    --mode llama-8b-lora \
    --representation kanzi \
    --kanzi-checkpoint checkpoints/kanzi_cleaned_model.pt \
    --db-path data/kanzi_afdb50/ \
    --output-dir outputs/structure_predictor_kanzi_v1 \
    --batch-size 24 \
    --learning-rate 2e-4 \
    --num-epochs 1
```

## Resource Estimates

### Storage
- Full afdb50 with coordinates: ~950GB
- Preprocessed Kanzi dataset (tokens only): ~10GB estimated
- Kanzi model checkpoint: 530MB

### Compute (Data Preprocessing)
- Kanzi tokenization: ~100 proteins/second on GPU
- 66.7M proteins × 0.01s = ~185 GPU-hours
- Can parallelize across multiple GPUs

### Training
- Same as current 3Di training
- Slightly larger vocabulary (1000 vs 20) may slightly increase memory

## Advantages Over 3Di

| Aspect | Foldseek 3Di | Kanzi |
|--------|-------------|-------|
| Vocabulary | 20 tokens | 1000 tokens |
| Reconstruction | No direct inverse | 0.5-1.0 Å RMSD |
| Validation | Indirect (TM-score) | Direct RMSD |
| Expressiveness | Coarse | Fine-grained |
| Model | Rule-based | Learned (44M params) |

## Risks and Mitigations

1. **Data Preprocessing Time**
   - Risk: Preprocessing 66.7M proteins is slow
   - Mitigation: Start with subset (1M proteins), parallelize

2. **Longer Sequences**
   - Risk: Kanzi tokens are integers (need multi-token representation)
   - Mitigation: Map to fixed vocabulary tokens `<K0>` to `<K999>`

3. **Kanzi Model Quality**
   - Risk: Kanzi may not generalize to all protein types
   - Mitigation: Validate on diverse test set before training

## Next Steps

1. [ ] Download Kanzi checkpoint to `checkpoints/kanzi_cleaned_model.pt`
2. [ ] Create CA extraction module for AlphaFold structures
3. [ ] Create Kanzi tokenization module
4. [ ] Preprocess pilot dataset (10K proteins)
5. [ ] Modify training script for Kanzi tokens
6. [ ] Add RMSD validation metrics
7. [ ] Run pilot training
8. [ ] Compare to 3Di baseline
