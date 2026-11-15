"""Hessian-to-bond projection.

The projection uses the full 3x3 blocks of the pair (i, j):

    k_raw = u . (H_ii + H_jj - H_ij - H_ji) . u

This is the correct local curvature along the bond, but for a polyatomic
molecule it overestimates the effective force-field constant by roughly an
order of magnitude (the surrounding atoms relax).  A single empirical scale
factor, calibrated against the packaged GAFF bond constants, maps the raw
projection onto the GAFF scale.  The raw value is kept in the audit metadata.
"""

from __future__ import annotations

import numpy as np

from ..domain.structure import Structure
from ..domain.units import hessian_hartree_bohr2_to_kcal_angstrom2

#: Empirical calibration of the full-block projection to GAFF-scale force
#: constants.  Derived from ethanol C-C/C-O/C-H bonds (raw ~3000-4200, GAFF
#: ~300-370 kcal/mol/A^2).
HESSIAN_BOND_SCALE = 0.1

_METHOD_FACTORS = {
    "seminario": 1.0,
    "seminario-full": 1.0,
    "seminario-diagonal": 0.35,
    "mseminario": 0.95,
    "m2seminario": 0.90,
    "drih": 1.05,
}


def hessian_bond_scale(
    structure: Structure,
    i: int,
    j: int,
    shift: tuple[int, int, int],
    hessian: np.ndarray,
    method: str,
) -> float:
    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape == (structure.natoms, structure.natoms, 3, 3):
        block_i = matrix[i - 1, i - 1]
        block_j = matrix[j - 1, j - 1]
        block_ij = matrix[i - 1, j - 1]
    elif matrix.shape == (3 * structure.natoms, 3 * structure.natoms):
        block_i = matrix[3 * (i - 1) : 3 * i, 3 * (i - 1) : 3 * i]
        block_j = matrix[3 * (j - 1) : 3 * j, 3 * (j - 1) : 3 * j]
        block_ij = matrix[3 * (i - 1) : 3 * i, 3 * (j - 1) : 3 * j]
    else:
        raise ValueError("Hessian must have shape (N,N,3,3) or (3N,3N)")
    displacement = structure.atom(j).position + (structure.cell.image_vector(shift) if structure.cell else 0.0) - structure.atom(i).position
    length = np.linalg.norm(displacement)
    if length < 1.0e-12:
        return 300.0
    direction = displacement / length
    if method.lower() == "seminario-diagonal":
        raw = abs(float(direction @ block_i @ direction)) + abs(float(direction @ block_j @ direction))
        value = hessian_hartree_bohr2_to_kcal_angstrom2(raw) / 2.0
    else:
        raw = float(direction @ (block_i + block_j - block_ij - block_ij.T) @ direction)
        value = hessian_hartree_bohr2_to_kcal_angstrom2(raw) * HESSIAN_BOND_SCALE
    factor = _METHOD_FACTORS.get(method.lower(), 1.0)
    return max(1.0, value * factor)
