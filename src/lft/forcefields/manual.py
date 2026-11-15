"""Manual force-field mapping helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import InputError


def load_manual_file(path: str | Path) -> tuple[dict[int, str], dict[str, dict[str, float]]]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"manual force-field file does not exist: {path}")
    raw = json.loads(path.read_text())
    atoms = raw.get("atoms", {})
    types = raw.get("types", raw.get("atom_types", {}))
    if not isinstance(atoms, dict) or not isinstance(types, dict):
        raise InputError("manual force-field JSON requires object fields 'atoms' and 'types'")
    return {int(index): str(value) for index, value in atoms.items()}, {str(name): {key: float(value[key]) for key in ("mass", "sigma", "epsilon") if key in value} for name, value in types.items()}
