"""
Kanzi tokenizer for protein structure encoding.

Kanzi uses a 1000-token vocabulary to encode C-alpha backbone coordinates.
Each residue maps to one token, enabling direct structure prediction.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from kanzi import DAE, kabsch_rmsd


class KanziTokenizer:
    """Tokenizer for encoding protein structures using Kanzi."""

    VOCAB_SIZE = 1000  # Kanzi uses 1000 discrete tokens
    DEFAULT_CHECKPOINT = "checkpoints/cleaned_model.pt"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        n_decode_steps: int = 50,
    ):
        """Initialize Kanzi tokenizer.

        Args:
            checkpoint_path: Path to Kanzi model checkpoint.
            device: Device to run model on ('cuda' or 'cpu').
            n_decode_steps: Number of ODE steps for decoding (more = better quality).
        """
        if checkpoint_path is None:
            # Try default locations
            for path in [
                self.DEFAULT_CHECKPOINT,
                Path(__file__).parent.parent.parent / "checkpoints" / "cleaned_model.pt",
            ]:
                if Path(path).exists():
                    checkpoint_path = str(path)
                    break

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Kanzi checkpoint not found. Please download from: "
                f"https://drive.google.com/uc?export=download&id=1ZOcqJ9E3aC-m6letqXR3iruNBMzMKAEm"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_decode_steps = n_decode_steps

        # Load model
        self.model = DAE.from_pretrained(checkpoint_path).to(self.device).eval()

    def encode(self, coords: np.ndarray | torch.Tensor) -> list[int]:
        """Encode C-alpha coordinates to Kanzi tokens.

        Args:
            coords: C-alpha coordinates in Angstroms, shape (L, 3).

        Returns:
            List of token indices, length L.
        """
        # Convert to tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()

        # Convert from Angstroms to nanometers (Kanzi uses nm)
        coords_nm = coords / 10.0

        # Add batch dimension
        if coords_nm.dim() == 2:
            coords_nm = coords_nm.unsqueeze(0)

        coords_nm = coords_nm.to(self.device)

        with torch.no_grad():
            _, _, tokens = self.model.encode(coords_nm)

        return tokens[0].cpu().tolist()

    def decode(self, tokens: list[int] | torch.Tensor) -> np.ndarray:
        """Decode Kanzi tokens to C-alpha coordinates.

        Args:
            tokens: List of token indices, length L.

        Returns:
            C-alpha coordinates in Angstroms, shape (L, 3).
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.int32)

        # Add batch dimension
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.to(self.device)

        with torch.no_grad():
            coords_nm = self.model.decode(tokens, n_steps=self.n_decode_steps)

        # Convert from nm to Angstroms
        coords_angstrom = coords_nm[0].cpu().numpy() * 10.0

        return coords_angstrom

    def compute_rmsd(
        self,
        pred_tokens: list[int],
        gt_tokens: list[int],
    ) -> float:
        """Compute RMSD between predicted and ground truth structures.

        Args:
            pred_tokens: Predicted Kanzi tokens.
            gt_tokens: Ground truth Kanzi tokens.

        Returns:
            RMSD in Angstroms after Kabsch alignment.
        """
        # Decode both to coordinates
        pred_coords = self.decode(pred_tokens)
        gt_coords = self.decode(gt_tokens)

        # Compute RMSD (kabsch_rmsd returns nm, convert to Angstroms)
        rmsd_nm = kabsch_rmsd(pred_coords / 10.0, gt_coords / 10.0)
        return float(rmsd_nm * 10.0)

    def compute_rmsd_from_coords(
        self,
        pred_tokens: list[int],
        gt_coords: np.ndarray,
    ) -> float:
        """Compute RMSD between predicted tokens and ground truth coordinates.

        Args:
            pred_tokens: Predicted Kanzi tokens.
            gt_coords: Ground truth C-alpha coordinates in Angstroms, shape (L, 3).

        Returns:
            RMSD in Angstroms after Kabsch alignment.
        """
        pred_coords = self.decode(pred_tokens)

        # Ensure same length
        min_len = min(len(pred_coords), len(gt_coords))
        pred_coords = pred_coords[:min_len]
        gt_coords = gt_coords[:min_len]

        rmsd_nm = kabsch_rmsd(pred_coords / 10.0, gt_coords / 10.0)
        return float(rmsd_nm * 10.0)

    def tokens_to_str(self, tokens: list[int]) -> str:
        """Convert tokens to space-separated string for LLM training.

        Args:
            tokens: List of token indices.

        Returns:
            Space-separated string like "599 358 983 769".
        """
        return " ".join(str(t) for t in tokens)

    def str_to_tokens(self, s: str) -> list[int]:
        """Parse token string back to list.

        Args:
            s: Space-separated token string.

        Returns:
            List of token indices.
        """
        return [int(t) for t in s.split() if t.strip()]
