"""Load common Hessian artifacts with unit-aware normalization."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..errors import InputError


def load_hessian(path: str | Path, natoms: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"Hessian file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.asarray(np.load(path), dtype=float)
        source = "numpy"
    elif suffix == ".json":
        raw = json.loads(path.read_text())
        matrix = np.asarray(raw.get("hessian", raw), dtype=float)
        source = "json"
    elif suffix in {".fchk", ".hess", ".out", ".log"}:
        matrix, source = _load_cclib_or_text(path, natoms)
    else:
        raise InputError(f"unsupported Hessian format {suffix!r}: {path}")
    expected = {(natoms, natoms, 3, 3), (3 * natoms, 3 * natoms)}
    if tuple(matrix.shape) not in expected:
        raise InputError(f"Hessian {path} has shape {matrix.shape}; expected {(natoms, natoms, 3, 3)} or {(3*natoms, 3*natoms)}")
    if not np.all(np.isfinite(matrix)):
        raise InputError(f"Hessian {path} contains non-finite values")
    if matrix.shape == (3 * natoms, 3 * natoms):
        matrix = 0.5 * (matrix + matrix.T)
    else:
        matrix = 0.5 * (matrix + np.swapaxes(np.swapaxes(matrix, 0, 1), 2, 3))
    return matrix, {"source": source, "path": str(path.resolve()), "shape": list(matrix.shape), "units": "hartree/bohr^2"}


def _load_cclib_or_text(path: Path, natoms: int) -> tuple[np.ndarray, str]:
    try:
        import cclib  # type: ignore[import-not-found]

        parsed = cclib.io.ccread(str(path))
        if parsed is not None and hasattr(parsed, "hessian"):
            return np.asarray(parsed.hessian, dtype=float), "cclib"
    except Exception:
        pass
    numbers = [float(value) for value in re.findall(r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", path.read_text(errors="replace"))]
    needed = 9 * natoms * natoms
    if len(numbers) < needed:
        raise InputError(f"could not find {needed} Hessian values in {path}")
    return np.asarray(numbers[-needed:], dtype=float).reshape(natoms, natoms, 3, 3), "numeric-text-tail"
