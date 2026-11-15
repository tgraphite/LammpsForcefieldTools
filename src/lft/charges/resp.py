"""RESP and RESP2 fitting built on the shared constrained ESP fitter.

The implementation mirrors Multiwfn's standard two-stage RESP
(``subroutine RESP`` / ``RESPiter`` in ``population.f90``):

* Stage 1 fits every atom with a weak hyperbolic penalty (``a = 0.0005``,
  ``b = 0.1``) applied to non-hydrogen atoms only.
* Stage 2 fixes every atom that is not an sp3/methylene/methyl carbon or a
  hydrogen attached to one, imposes equivalence constraints on the hydrogens
  of each ``CH2``/``CH3`` group, and refits with a stronger penalty
  (``a = 0.001``).

The penalty term is ``a / sqrt(q_i^2 + b^2)`` added to the diagonal of the
normal-equation matrix, and the fit is iterated to self-consistency because the
penalty depends on the charges.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..domain.structure import Structure
from ..forcefields.tables import ATOMIC_NUMBER
from .esp import (
    BOHR_TO_ANGSTROM,
    esp_from_wavefunction,
    make_esp_grid,
    target_potential,
)

if TYPE_CHECKING:
    from ..qc.wavefunction import WavefunctionData

#: Multiwfn standard two-stage RESP defaults.
RESP_STAGE1_RESTRAINT = 0.0005
RESP_STAGE2_RESTRAINT = 0.001
RESP_TIGHTNESS = 0.1
RESP_MAX_ITERATIONS = 300
RESP_CONVERGENCE = 1.0e-6


def fit_resp(
    structure: Structure,
    wavefunction: WavefunctionData,
    net_charge: float,
    *,
    points: np.ndarray | None = None,
    stage1_restraint: float = RESP_STAGE1_RESTRAINT,
    stage2_restraint: float = RESP_STAGE2_RESTRAINT,
    tightness: float = RESP_TIGHTNESS,
    max_iterations: int = RESP_MAX_ITERATIONS,
    convergence: float = RESP_CONVERGENCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit restrained charges to the exact wavefunction ESP (standard RESP)."""
    if points is None:
        points = make_esp_grid(structure)
    target = esp_from_wavefunction(wavefunction, points)
    return _fit_from_potential(
        structure,
        points,
        target,
        net_charge,
        stage1_restraint=stage1_restraint,
        stage2_restraint=stage2_restraint,
        tightness=tightness,
        max_iterations=max_iterations,
        convergence=convergence,
    )


def fit_resp_proxy(
    structure: Structure,
    reference: np.ndarray,
    net_charge: float,
    *,
    points: np.ndarray | None = None,
    stage1_restraint: float = RESP_STAGE1_RESTRAINT,
    stage2_restraint: float = RESP_STAGE2_RESTRAINT,
    tightness: float = RESP_TIGHTNESS,
    max_iterations: int = RESP_MAX_ITERATIONS,
    convergence: float = RESP_CONVERGENCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Legacy proxy fit used when no wavefunction artifact is available."""
    if points is None:
        points = make_esp_grid(structure)
    target = target_potential(structure, points, np.asarray(reference, dtype=float))
    charges, metadata = _fit_from_potential(
        structure,
        points,
        target,
        net_charge,
        stage1_restraint=stage1_restraint,
        stage2_restraint=stage2_restraint,
        tightness=tightness,
        max_iterations=max_iterations,
        convergence=convergence,
    )
    metadata["proxy"] = True
    return charges, metadata


def mix_resp2(gas: np.ndarray, aqueous: np.ndarray, delta: float) -> np.ndarray:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("RESP2 delta must be between 0 and 1")
    return (1.0 - delta) * np.asarray(gas, dtype=float) + delta * np.asarray(
        aqueous, dtype=float
    )


def _fit_from_potential(
    structure: Structure,
    points: np.ndarray,
    target: np.ndarray,
    net_charge: float,
    *,
    stage1_restraint: float,
    stage2_restraint: float,
    tightness: float,
    max_iterations: int,
    convergence: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    atomic_numbers = np.array(
        [ATOMIC_NUMBER.get(atom.element, 0) for atom in structure.atoms], dtype=int
    )
    points_bohr = np.asarray(points, dtype=float) / BOHR_TO_ANGSTROM
    coordinates_bohr = structure.coordinates / BOHR_TO_ANGSTROM
    distances = np.linalg.norm(
        points_bohr[:, None, :] - coordinates_bohr[None, :, :], axis=2
    )
    design = 1.0 / np.maximum(distances, 1.0e-12)

    singletons = [[index] for index in range(structure.natoms)]
    stage1 = _solve_stage(
        design,
        target,
        net_charge,
        atomic_numbers,
        restraint=stage1_restraint,
        tightness=tightness,
        groups=singletons,
        fixed=None,
        max_iterations=max_iterations,
        convergence=convergence,
    )
    ch_atoms, stage2_groups = _resp_stage2_groups(structure, atomic_numbers)
    metadata: dict[str, Any] = {
        "grid_points": len(points),
        "fit": "multiwfn_two_stage_resp",
        "restraint_stage1": float(stage1_restraint),
        "restraint_stage2": float(stage2_restraint),
        "tightness": float(tightness),
        "stage1_charges": [float(value) for value in stage1],
    }
    if not ch_atoms or stage2_restraint == 0.0:
        charges = stage1
        metadata["stage2_skipped"] = True
    else:
        fixed = {
            index: float(stage1[index])
            for index in range(structure.natoms)
            if index not in ch_atoms
        }
        charges = _solve_stage(
            design,
            target,
            net_charge,
            atomic_numbers,
            restraint=stage2_restraint,
            tightness=tightness,
            groups=stage2_groups,
            fixed=fixed,
            max_iterations=max_iterations,
            convergence=convergence,
        )
        metadata["stage2_skipped"] = False
        metadata["stage2_equivalence_groups"] = [
            [index + 1 for index in group] for group in stage2_groups if len(group) > 1
        ]
        metadata["stage2_fitted_atoms"] = [index + 1 for index in sorted(ch_atoms)]
    residual = target - design @ charges
    metadata["esp_rms"] = float(np.sqrt(np.mean(residual**2)))
    metadata["esp_max"] = float(np.max(np.abs(residual)))
    metadata["esp_rrmse"] = float(np.sqrt(np.sum(residual**2) / np.sum(target**2)))
    return charges, metadata


def _resp_stage2_groups(
    structure: Structure, atomic_numbers: np.ndarray
) -> tuple[set[int], list[list[int]]]:
    """Identify stage-2 fitting atoms and hydrogen equivalence groups.

    Mirrors Multiwfn's rule: a carbon is fitted in stage 2 when it has four
    bonded neighbours, or when it is a methylene (two H and one C neighbour).
    The hydrogens attached to such a carbon form one equivalence class.
    """

    def zero_based(index: int) -> int:
        return index - 1

    adjacency: dict[int, set[int]] = {
        zero_based(atom.index): set() for atom in structure.atoms
    }
    for bond in structure.bonds:
        i, j = zero_based(bond.i), zero_based(bond.j)
        adjacency[i].add(j)
        adjacency[j].add(i)
    ch_atoms: set[int] = set()
    groups: list[list[int]] = []
    for atom in structure.atoms:
        index = zero_based(atom.index)
        if atomic_numbers[index] != 6:
            continue
        neighbours = adjacency[index]
        n_hydrogen = sum(1 for other in neighbours if atomic_numbers[other] == 1)
        n_carbon = sum(1 for other in neighbours if atomic_numbers[other] == 6)
        if len(neighbours) == 4 or (n_hydrogen == 2 and n_carbon == 1):
            ch_atoms.add(index)
            groups.append([index])
            hydrogens = sorted(
                other for other in neighbours if atomic_numbers[other] == 1
            )
            if hydrogens:
                ch_atoms.update(hydrogens)
                groups.append(hydrogens)
    for atom in structure.atoms:
        index = zero_based(atom.index)
        if index not in ch_atoms:
            groups.append([index])
    return ch_atoms, groups


def _solve_stage(
    design: np.ndarray,
    target: np.ndarray,
    net_charge: float,
    atomic_numbers: np.ndarray,
    *,
    restraint: float,
    tightness: float,
    groups: list[list[int]],
    fixed: dict[int, float] | None,
    max_iterations: int,
    convergence: float,
) -> np.ndarray:
    natoms = design.shape[1]
    normal = design.T @ design
    rhs = design.T @ target
    base_diagonal = normal.diagonal().copy()
    charges = np.zeros(natoms, dtype=float)
    if restraint == 0.0:
        return _solve_once(normal, base_diagonal, rhs, net_charge, groups, fixed)
    for _iteration in range(max_iterations):
        previous = charges.copy()
        diagonal = base_diagonal.copy()
        for index in range(natoms):
            if atomic_numbers[index] == 1:
                continue
            diagonal[index] += restraint / math.sqrt(
                float(charges[index]) ** 2 + tightness**2
            )
        charges = _solve_once(normal, diagonal, rhs, net_charge, groups, fixed)
        if float(np.max(np.abs(charges - previous))) < convergence:
            break
    return charges


def _solve_once(
    normal: np.ndarray,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    net_charge: float,
    groups: list[list[int]],
    fixed: dict[int, float] | None,
) -> np.ndarray:
    natoms = normal.shape[0]
    fixed = fixed or {}
    fixed_items = sorted(fixed.items())
    n_fixed = len(fixed_items)
    n_groups = len(groups)
    size = natoms + n_fixed + 1
    system = np.zeros((size, size), dtype=float)
    system[:natoms, :natoms] = normal
    np.fill_diagonal(system[:natoms, :natoms], diagonal)
    system[natoms, :natoms] = 1.0
    system[:natoms, natoms] = 1.0
    vector = np.zeros(size, dtype=float)
    vector[:natoms] = rhs
    vector[natoms] = net_charge
    for offset, (atom_index, value) in enumerate(fixed_items):
        row = natoms + 1 + offset
        atom = atom_index
        system[atom, row] = 1.0
        system[row, atom] = 1.0
        vector[row] = value

    contracted = np.zeros((n_groups + n_fixed + 1, size), dtype=float)
    contracted_rhs = np.zeros(n_groups + n_fixed + 1, dtype=float)
    for group_index, group in enumerate(groups):
        for atom in group:
            contracted[group_index] += system[atom]
            contracted_rhs[group_index] += vector[atom]
    contracted[n_groups] = system[natoms]
    contracted_rhs[n_groups] = vector[natoms]
    for offset in range(n_fixed):
        contracted[n_groups + 1 + offset] = system[natoms + 1 + offset]
        contracted_rhs[n_groups + 1 + offset] = vector[natoms + 1 + offset]
    reduced = np.zeros((n_groups + n_fixed + 1, n_groups + n_fixed + 1), dtype=float)
    for group_index, group in enumerate(groups):
        for atom in group:
            reduced[:, group_index] += contracted[:, atom]
    reduced[:, n_groups] = contracted[:, natoms]
    for offset in range(n_fixed):
        reduced[:, n_groups + 1 + offset] = contracted[:, natoms + 1 + offset]
    try:
        solution = np.linalg.solve(reduced, contracted_rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(reduced, contracted_rhs, rcond=None)[0]
    charges = np.zeros(natoms, dtype=float)
    for group_index, group in enumerate(groups):
        for atom in group:
            charges[atom] = solution[group_index]
    return charges
