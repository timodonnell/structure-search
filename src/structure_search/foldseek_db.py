"""
Foldseek/MMseqs2 database reader.

Foldseek uses MMseqs2 database format:
- .index file: tab-separated (id, offset, length)
- main file: contains sequences at those offsets

C-alpha coordinate format:
- 2-byte header (skipped)
- Differential int16 values for x, y, z
- Cumsum to get absolute positions
- Divide by 1000 to get Angstroms
"""

import mmap
from pathlib import Path
from typing import Iterator

import numpy as np


class FoldseekDB:
    """Reader for Foldseek/MMseqs2 database format."""

    def __init__(self, db_path: str | Path):
        """Initialize database reader.

        Args:
            db_path: Path to database (without extension, e.g., 'afdb50/afdb50')
        """
        self.db_path = Path(db_path)
        self.index_path = Path(f"{db_path}.index")

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        self._file = None
        self._mmap = None
        self._index = None

    def _load_index(self) -> list[tuple[int, int, int]]:
        """Load the index file."""
        if self._index is not None:
            return self._index

        self._index = []
        with open(self.index_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    entry_id = int(parts[0])
                    offset = int(parts[1])
                    length = int(parts[2])
                    self._index.append((entry_id, offset, length))
        return self._index

    def __len__(self) -> int:
        """Return number of entries in database."""
        return len(self._load_index())

    def __enter__(self):
        """Open database for reading."""
        self._file = open(self.db_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        """Close database."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def get_sequence(self, idx: int) -> str:
        """Get sequence at index."""
        if self._mmap is None:
            raise RuntimeError("Database not opened. Use 'with' context manager.")

        index = self._load_index()
        if idx < 0 or idx >= len(index):
            raise IndexError(f"Index {idx} out of range [0, {len(index)})")

        _, offset, length = index[idx]
        data = self._mmap[offset:offset + length]
        # Sequences are null-terminated
        return data.rstrip(b"\x00").decode("utf-8")

    def __iter__(self) -> Iterator[tuple[int, str]]:
        """Iterate over all sequences."""
        if self._mmap is None:
            raise RuntimeError("Database not opened. Use 'with' context manager.")

        for idx, (entry_id, offset, length) in enumerate(self._load_index()):
            data = self._mmap[offset:offset + length]
            seq = data.rstrip(b"\x00").decode("utf-8")
            yield entry_id, seq


class FoldseekCaDB:
    """Reader for Foldseek C-alpha coordinate database.

    The C-alpha database stores coordinates in a compact format:
    - 2-byte header
    - Differential int16 encoding for x, y, z
    - Scale factor of 1000 (values are in milliangstroms)
    """

    def __init__(self, db_path: str | Path):
        """Initialize C-alpha database reader.

        Args:
            db_path: Path to database (without extension, e.g., 'afdb50/afdb50_ca')
        """
        self.db_path = Path(db_path)
        self.index_path = Path(f"{db_path}.index")

        if not self.db_path.exists():
            raise FileNotFoundError(f"C-alpha database not found: {self.db_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"C-alpha index not found: {self.index_path}")

        self._file = None
        self._mmap = None
        self._index = None

    def _load_index(self) -> list[tuple[int, int, int]]:
        """Load the index file."""
        if self._index is not None:
            return self._index

        self._index = []
        with open(self.index_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    entry_id = int(parts[0])
                    offset = int(parts[1])
                    length = int(parts[2])
                    self._index.append((entry_id, offset, length))
        return self._index

    def __len__(self) -> int:
        return len(self._load_index())

    def __enter__(self):
        self._file = open(self.db_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, *args):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def get_coords(self, idx: int, n_residues: int | None = None) -> np.ndarray:
        """Get C-alpha coordinates at index.

        Args:
            idx: Entry index.
            n_residues: Expected number of residues (for validation).

        Returns:
            C-alpha coordinates in Angstroms, shape (L, 3).
        """
        if self._mmap is None:
            raise RuntimeError("Database not opened. Use 'with' context manager.")

        index = self._load_index()
        if idx < 0 or idx >= len(index):
            raise IndexError(f"Index {idx} out of range [0, {len(index)})")

        _, offset, length = index[idx]
        data = bytes(self._mmap[offset:offset + length])

        # Skip 2-byte header
        coords_data = data[2:]

        # Parse as int16
        coords_int16 = np.frombuffer(coords_data, dtype="<i2")

        # Determine number of residues from data length
        n_coords = len(coords_int16) // 3
        if n_residues is not None and n_coords != n_residues:
            # Truncate or warn
            n_coords = min(n_coords, n_residues)

        coords_int16 = coords_int16[: n_coords * 3].reshape(n_coords, 3)

        # Differential decoding: cumsum to get absolute positions
        coords = np.cumsum(coords_int16.astype(np.float32), axis=0)

        # Scale from milliangstroms to Angstroms
        coords = coords / 1000.0

        return coords


class PairedFoldseekDB:
    """Paired reader for amino acid and 3Di structure databases."""

    def __init__(self, base_path: str | Path, include_ca: bool = False):
        """Initialize paired database reader.

        Args:
            base_path: Base path to database (e.g., 'afdb50/afdb50')
                      Will read both base_path (AA) and base_path_ss (3Di)
            include_ca: Whether to also read C-alpha coordinates.
        """
        self.aa_db = FoldseekDB(base_path)
        self.ss_db = FoldseekDB(f"{base_path}_ss")
        self.include_ca = include_ca
        self.ca_db = None

        if include_ca:
            ca_path = f"{base_path}_ca"
            if Path(ca_path).exists() and Path(f"{ca_path}.index").exists():
                self.ca_db = FoldseekCaDB(ca_path)
            else:
                raise FileNotFoundError(
                    f"C-alpha database not found at {ca_path}. "
                    f"Download with: foldseek databases Alphafold/UniProt50 <path> tmp/"
                )

    def __len__(self) -> int:
        # Return minimum of all databases to ensure valid indices
        if self.ca_db is not None:
            return min(len(self.aa_db), len(self.ss_db), len(self.ca_db))
        return min(len(self.aa_db), len(self.ss_db))

    def __enter__(self):
        self.aa_db.__enter__()
        self.ss_db.__enter__()
        if self.ca_db:
            self.ca_db.__enter__()
        return self

    def __exit__(self, *args):
        self.aa_db.__exit__(*args)
        self.ss_db.__exit__(*args)
        if self.ca_db:
            self.ca_db.__exit__(*args)

    def get_pair(self, idx: int) -> tuple[str, str]:
        """Get (amino_acid_sequence, 3di_structure) pair at index."""
        aa_seq = self.aa_db.get_sequence(idx)
        ss_seq = self.ss_db.get_sequence(idx)
        return aa_seq, ss_seq

    def get_triplet(self, idx: int) -> tuple[str, str, np.ndarray]:
        """Get (amino_acid_sequence, 3di_structure, ca_coords) triplet at index.

        Returns:
            Tuple of (aa_seq, ss_seq, ca_coords) where ca_coords is shape (L, 3) in Angstroms.
        """
        aa_seq = self.aa_db.get_sequence(idx)
        ss_seq = self.ss_db.get_sequence(idx)

        if self.ca_db is None:
            raise RuntimeError("C-alpha database not loaded. Set include_ca=True.")

        ca_coords = self.ca_db.get_coords(idx, n_residues=len(aa_seq))
        return aa_seq, ss_seq, ca_coords

    def __iter__(self) -> Iterator[tuple[int, str, str]]:
        """Iterate over all (id, aa_seq, ss_seq) tuples."""
        aa_index = self.aa_db._load_index()

        for idx in range(len(aa_index)):
            aa_seq = self.aa_db.get_sequence(idx)
            ss_seq = self.ss_db.get_sequence(idx)
            yield idx, aa_seq, ss_seq
