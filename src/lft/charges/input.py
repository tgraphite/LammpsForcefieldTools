"""Input charge readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain.structure import Structure
from ..errors import ChargeError, InputError


def read_charge_file(path: str | Path) -> list[float]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"charge file does not exist: {path}")
    if path.suffix.lower() == ".json":
        raw: Any = json.loads(path.read_text())
        if isinstance(raw, dict):
            raw = raw.get("charges")
        if isinstance(raw, dict):
            try:
                raw = [raw[str(index)] if str(index) in raw else raw[index] for index in range(1, len(raw) + 1)]
            except (KeyError, TypeError) as exc:
                raise InputError(f"charge JSON mapping must use contiguous atom indices: {path}") from exc
        if not isinstance(raw, list):
            raise InputError(f"charge JSON must contain a list or a 'charges' list: {path}")
        try:
            return [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise InputError(f"charge JSON contains a non-numeric value: {path}") from exc
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise InputError(f"charge file is empty: {path}")
    values: list[float] = []
    start = 0
    try:
        if len(lines[0].split()) == 1 and int(lines[0]) == int(lines[0]):
            expected = int(lines[0])
            if expected == len(lines) - 1:
                start = 1
    except ValueError:
        pass
    for line in lines[start:]:
        fields = line.replace(",", " ").split()
        try:
            values.append(float(fields[-1]))
        except (IndexError, ValueError) as exc:
            raise InputError(f"charge file has no numeric charge in line {line!r}") from exc
    return values


def apply_input_charges(structure: Structure, values: list[float] | None = None) -> None:
    if values is not None:
        if len(values) != structure.natoms:
            raise ChargeError(f"input charge count {len(values)} does not match atom count {structure.natoms}")
        for atom, charge in zip(structure.atoms, values):
            atom.charge = float(charge)
            atom.charge_source = "input"
    elif not any(atom.charge_source == "input" or abs(atom.charge) > 0.0 for atom in structure.atoms):
        raise ChargeError("input charge mode requires charges in the structure or --charge <file>")
    else:
        for atom in structure.atoms:
            atom.charge_source = "input"
