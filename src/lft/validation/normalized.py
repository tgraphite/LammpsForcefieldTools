"""Validation that is independent of writer whitespace and file layout."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..charges.dispatcher import CHARGE_SUM_TOLERANCE
from ..domain.structure import Structure
from ..errors import ValidationError
from ..topology.enumerate_terms import validate_term_lifts


def validate_structure(structure: Structure) -> dict[str, Any]:
    structure.validate()
    validate_term_lifts(structure)
    if structure.cell is not None:
        for bond in structure.bonds:
            for value, periodic in zip(bond.image_shift, structure.cell.pbc):
                if not periodic and value:
                    raise ValidationError(f"bond {bond.i}-{bond.j} has an image shift in a non-periodic dimension")
    return {
        "natoms": structure.natoms,
        "nbonds": len(structure.bonds),
        "nangles": len(structure.angles),
        "ndihedrals": len(structure.dihedrals),
        "nimpropers": len(structure.impropers),
        "npairs": len(structure.pairs),
        "net_charge": structure.net_charge,
    }


def compare_charge_vectors(left: list[float] | np.ndarray, right: list[float] | np.ndarray, *, atol: float = 2.0e-4) -> dict[str, float | bool]:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.shape != rhs.shape:
        raise ValidationError(f"charge vectors have different shapes: {lhs.shape} and {rhs.shape}")
    difference = np.abs(lhs - rhs)
    return {"rms": float(np.sqrt(np.mean(difference**2))), "max_abs": float(difference.max(initial=0.0)), "pass": bool(np.all(difference <= atol))}


def validate_net_charge(structure: Structure, requested: float, *, tolerance: float = CHARGE_SUM_TOLERANCE) -> None:
    if abs(structure.net_charge - requested) > tolerance:
        raise ValidationError(f"net charge error {abs(structure.net_charge-requested):.6g} e exceeds {tolerance:.6g} e")
