"""Deterministic electrostatic-potential grid and constrained charge fit.

The fitting grid replicates the Merz-Kollman (MK) point distribution used by
Multiwfn's ``setMKpt`` (``population.f90``): four shells at 1.4/1.6/1.8/2.0 x
the element vdW radius, a surface point density of 6.0 A^-2, and a pruning rule
that drops points lying closer than 1.4 x the vdW radius of any other atom.
Points are generated with the same ``unitspherept`` spiral used by Multiwfn so
that the grid (and therefore the fitted charges) is reproducible.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..domain.structure import Structure
from ..forcefields.tables import ATOMIC_NUMBER
from ._cm5_data import adch_vdw_radius

if TYPE_CHECKING:
    from ..qc.wavefunction import WavefunctionData

#: Angstrom per Bohr.
BOHR_TO_ANGSTROM = 0.529177210903

#: vdW radii (Angstrom) used specifically for MK fitting points.  Values for
#: Z = 1..17 are copied from Multiwfn ``setESPfitvdwr(iradtype=1)``; heavier
#: elements fall back to the Multiwfn ``vdwr`` table via ``adch_vdw_radius``.
MK_VDW_RADII_ANGSTROM: dict[int, float] = {
    1: 1.20,
    2: 1.20,
    3: 1.37,
    4: 1.45,
    5: 1.45,
    6: 1.50,
    7: 1.50,
    8: 1.40,
    9: 1.35,
    10: 1.30,
    11: 1.57,
    12: 1.65,
    13: 1.65,
    14: 1.80,
    15: 1.80,
    16: 1.75,
    17: 1.70,
}

#: MK shell scale factors (multiples of the element vdW radius).
MK_LAYER_SCALES = (1.4, 1.6, 1.8, 2.0)

#: MK surface point density in points per Angstrom^2 (Multiwfn default 6.0).
MK_POINT_DENSITY_ANGSTROM2 = 6.0

#: Transient memory budget for one ``int1e_grids`` block (bytes).  The integral
#: tensor is (block, nao, nao) float64, so this caps the block size.
_INTEGRAL_BUDGET_BYTES = 256 * 1024 * 1024


def _nint(value: float) -> int:
    """Fortran ``nint``: round half away from zero (inputs here are positive)."""

    return math.floor(value + 0.5) if value >= 0.0 else -math.floor(-value + 0.5)


def mk_vdw_radius_angstrom(atomic_number: int) -> float:
    if atomic_number in MK_VDW_RADII_ANGSTROM:
        return MK_VDW_RADII_ANGSTROM[atomic_number]
    return adch_vdw_radius(atomic_number)


def _unit_sphere_points(count: int) -> np.ndarray:
    """Port of Multiwfn ``unitspherept`` (returns fewer points if needed)."""

    if count <= 0:
        return np.zeros((0, 3), dtype=float)
    pi = math.pi
    numequ = int(math.sqrt(count * pi))
    numvert = max(numequ // 2, 1)
    points: list[tuple[float, float, float]] = []
    for ivert in range(numvert + 1):
        angz = ivert / numvert * pi
        scalexy = math.sin(angz)
        z = math.cos(angz)
        numxy = int(numequ * scalexy)
        if numxy == 0:
            numxy = 1
        for ihori in range(1, numxy + 1):
            if len(points) >= count:
                return np.asarray(points, dtype=float)
            angxy = 2.0 * pi * ihori / numxy
            points.append((math.cos(angxy) * scalexy, math.sin(angxy) * scalexy, z))
    return np.asarray(points, dtype=float)


def make_esp_grid(structure: Structure) -> np.ndarray:
    """Build the Multiwfn MK fitting grid (Angstrom) for ``structure``."""

    atomic_numbers = [ATOMIC_NUMBER.get(atom.element, 0) for atom in structure.atoms]
    radii = np.array([mk_vdw_radius_angstrom(z) for z in atomic_numbers], dtype=float)
    coordinates = structure.coordinates
    cut_inner = min(MK_LAYER_SCALES)
    points: list[np.ndarray] = []
    for center, radius in enumerate(radii):
        for scale in MK_LAYER_SCALES:
            shell_radius = float(radius) * scale
            count = _nint(4.0 * math.pi * shell_radius**2 * MK_POINT_DENSITY_ANGSTROM2)
            directions = _unit_sphere_points(count)
            if directions.size == 0:
                continue
            shell = directions * shell_radius + coordinates[center]
            for point in shell:
                conflicts = (
                    np.linalg.norm(coordinates - point, axis=1) < radii * cut_inner
                )
                conflicts[center] = False
                if not bool(conflicts.any()):
                    points.append(point)
    if not points:
        return structure.coordinates + np.array([0.0, 0.0, 2.0])
    return np.asarray(points, dtype=float)


def esp_from_wavefunction(
    wavefunction: WavefunctionData, points_angstrom: np.ndarray
) -> np.ndarray:
    """Exact total ESP (nuclear + electronic) at the given points.

    The electronic part uses the analytic ``(mu nu | 1/r_grids)`` integral so
    the fitted charges match a quantum-chemistry reference rather than a
    point-charge proxy.  Returns atomic units (e/bohr).
    """
    bohr = 1.0 / BOHR_TO_ANGSTROM
    points_bohr = np.asarray(points_angstrom, dtype=float) * bohr
    total_dm = wavefunction.total_dm
    electronic = np.empty(points_bohr.shape[0], dtype=float)
    # ``int1e_grids`` materialises an (npoints, nao, nao) block, so a 4-shell MK
    # grid on a moderate molecule easily reaches tens of thousands of points.
    # Evaluate it in blocks to keep the transient allocation bounded.
    nao = int(wavefunction.mol.nao_nr())
    block = max(1, _INTEGRAL_BUDGET_BYTES // max(nao * nao * 8, 1))
    for start in range(0, points_bohr.shape[0], block):
        chunk = points_bohr[start : start + block]
        integrals = np.asarray(
            wavefunction.mol.intor("int1e_grids", grids=chunk), dtype=float
        )
        electronic[start : start + block] = -np.einsum("nuv,uv->n", integrals, total_dm)
    coords = wavefunction.coordinates_bohr
    distances = np.linalg.norm(points_bohr[:, None, :] - coords[None, :, :], axis=2)
    distances = np.maximum(distances, 1.0e-12)
    nuclear = (wavefunction.nuclear_charges[None, :] / distances).sum(axis=1)
    return nuclear + electronic


def fit_constrained_charges(
    structure: Structure,
    points: np.ndarray,
    target: np.ndarray,
    net_charge: float,
    *,
    restraint: float = 0.0,
) -> np.ndarray:
    """Plain (optionally Tikhonov-restrained) least-squares ESP fit.

    This is the single-stage, unrestrained fit used for the ``mk`` method.  It
    solves the same normal equations as Multiwfn's ``fitESP`` with no
    hyperbolic penalty and no equivalence/charge constraints.
    """

    coordinates_bohr = structure.coordinates / BOHR_TO_ANGSTROM
    points_bohr = np.asarray(points, dtype=float) / BOHR_TO_ANGSTROM
    distances = np.linalg.norm(
        points_bohr[:, None, :] - coordinates_bohr[None, :, :], axis=2
    )
    distances = np.maximum(distances, 0.5)
    design = 1.0 / distances
    normal = design.T @ design
    rhs = design.T @ target
    if restraint > 0.0:
        normal = normal + restraint * np.eye(structure.natoms)
    constraint = np.ones((structure.natoms, 1), dtype=float)
    kkt = np.block([[normal, constraint], [constraint.T, np.zeros((1, 1))]])
    vector = np.concatenate((rhs, [net_charge]))
    try:
        solution = np.linalg.solve(kkt, vector)[: structure.natoms]
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(kkt, vector, rcond=None)[0][: structure.natoms]
    return solution


def target_potential(
    structure: Structure, points: np.ndarray, reference_charges: np.ndarray
) -> np.ndarray:
    """Point-charge ESP (atomic units) used by the legacy proxy RESP path."""

    coordinates_bohr = structure.coordinates / BOHR_TO_ANGSTROM
    points_bohr = np.asarray(points, dtype=float) / BOHR_TO_ANGSTROM
    distances = np.linalg.norm(
        points_bohr[:, None, :] - coordinates_bohr[None, :, :], axis=2
    )
    distances = np.maximum(distances, 0.5)
    return (reference_charges[None, :] / distances).sum(axis=1)
