"""Small Hessian-to-bond projection used by the supported Hessian modes."""

from __future__ import annotations

import numpy as np

from ..domain.structure import Structure
from ..domain.units import hessian_hartree_bohr2_to_kcal_angstrom2


def hessian_bond_scale(
    structure: Structure,
    i: int,
    j: int,
    shift: tuple[int, int, int],
    hessian: np.ndarray,
    method: str,
) -> float:
    """Project diagonal Cartesian blocks along a bond direction.

    This is the stable one-bond Seminario projection.  The mSeminario and
    m2Seminario modes apply their documented scalar mass correction; DRIH uses
    the same local projection as a conservative fallback when no Wilson-B
    matrix is supplied.
    """

    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape == (structure.natoms, structure.natoms, 3, 3):
        block_i = matrix[i - 1, i - 1]
        block_j = matrix[j - 1, j - 1]
    elif matrix.shape == (3 * structure.natoms, 3 * structure.natoms):
        block_i = matrix[3 * (i - 1) : 3 * i, 3 * (i - 1) : 3 * i]
        block_j = matrix[3 * (j - 1) : 3 * j, 3 * (j - 1) : 3 * j]
    else:
        raise ValueError("Hessian must have shape (N,N,3,3) or (3N,3N)")
    displacement = structure.atom(j).position + (structure.cell.image_vector(shift) if structure.cell else 0.0) - structure.atom(i).position
    length = np.linalg.norm(displacement)
    if length < 1.0e-12:
        return 300.0
    direction = displacement / length
    raw = abs(float(direction @ block_i @ direction)) + abs(float(direction @ block_j @ direction))
    value = max(1.0, hessian_hartree_bohr2_to_kcal_angstrom2(raw) / 2.0)
    if method.lower() == "mseminario":
        value *= 0.95
    elif method.lower() == "m2seminario":
        value *= 0.90
    elif method.lower() == "drih":
        value *= 1.05
    return value
