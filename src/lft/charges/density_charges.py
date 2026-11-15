"""Density-partition charge models (Mulliken, Hirshfeld, CM5, ADCH).

All formulas replicate the reference Multiwfn implementations so the internal
engine agrees with the reference toolchain without invoking an external binary.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..qc.wavefunction import WavefunctionData
from ._atomic_density import atomic_radial_density_vectorized
from ._cm5_data import CM5_ALPHA, adch_vdw_radius, cm5_radius, cm5_tval
from ._grid import atomic_grid_separated

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


def mulliken_charges(wf: WavefunctionData) -> tuple[np.ndarray, dict[str, Any]]:
    """Mulliken gross atom populations and charges from the density matrix.

    ``q_A = Z_A - sum_{mu in A} (P S)_{mu mu}`` with the total density matrix.
    Atom/AO mapping uses ``ao_labels()`` to stay robust across basis orderings.
    """
    density = wf.total_dm
    gross = density @ wf.overlap
    labels = wf.mol.ao_labels()
    ao_atom = np.asarray([int(label.split()[0]) for label in labels], dtype=int)
    charges = np.zeros(wf.natoms, dtype=float)
    populations = np.zeros(wf.natoms, dtype=float)
    for a in range(wf.natoms):
        indices = np.flatnonzero(ao_atom == a)
        populations[a] = np.diag(gross)[indices].sum()
        charges[a] = wf.nuclear_charges[a] - populations[a]
    return charges, {"model": "mulliken", "populations": populations.tolist()}


def _molecular_density(wf: WavefunctionData, points_bohr: np.ndarray) -> np.ndarray:
    from pyscf.dft import numint  # type: ignore[import-not-found]

    ao = wf.mol.eval_gto("GTOval", points_bohr)
    rho = numint.eval_rho(wf.mol, ao, wf.dm_alpha, hermi=1)
    if wf.is_unrestricted:
        rho = rho + numint.eval_rho(wf.mol, ao, wf.dm_beta, hermi=1)
    return rho


def hirshfeld_charges(wf: WavefunctionData) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hirshfeld charges and atomic dipole moments.

    Replicates Multiwfn ``spacecharge`` with ``chgtype=1``: a single-center
    Becke/Lebedev quadrature grid per atom, promolecular density from built-in
    sphericalized free-atom densities.  Coordinates and densities are in Bohr.
    """
    centers = wf.coordinates_bohr
    atomic_numbers = wf.atomic_numbers
    charges = np.zeros(wf.natoms, dtype=float)
    dipoles = np.zeros((wf.natoms, 3), dtype=float)
    for a in range(wf.natoms):
        points, radw, angular_weights = atomic_grid_separated(centers[a])
        flat = points.reshape(-1, 3)
        weights = (radw[:, None] * angular_weights[None, :]).reshape(-1)
        molecular = _molecular_density(wf, flat)
        promolecular = np.zeros_like(molecular)
        for b in range(wf.natoms):
            distance = np.linalg.norm(flat - centers[b], axis=1)
            promolecular += atomic_radial_density_vectorized(atomic_numbers[b], distance)
        self_density = atomic_radial_density_vectorized(atomic_numbers[a], np.linalg.norm(flat - centers[a], axis=1))
        nonzero = promolecular != 0.0
        weighted = np.zeros_like(molecular)
        weighted[nonzero] = self_density[nonzero] / promolecular[nonzero] * molecular[nonzero] * weights[nonzero]
        charges[a] = wf.nuclear_charges[a] - weighted.sum()
        relative = flat - centers[a]
        dipoles[a] = -(weighted[:, None] * relative).sum(axis=0)
    return charges, dipoles, {"model": "hirshfeld", "grid": {"radial": int(len(radw)), "angular": int(len(angular_weights))}}


def cm5_charges(hirshfeld: np.ndarray, wf: WavefunctionData) -> tuple[np.ndarray, dict[str, Any]]:
    """CM5 charges from Hirshfeld charges (Multiwfn ``doCM5``, itype=0).

    ``q_i = q_i^H + sum_{j != i} T(Z_i, Z_j) exp(-alpha (R_ij - r_i - r_j))``
    with ``alpha = 2.474`` and the CRC-average covalent radii.
    """
    coordinates_angstrom = wf.coordinates_bohr * BOHR_TO_ANGSTROM
    atomic_numbers = wf.atomic_numbers
    corrected = np.array(hirshfeld, dtype=float)
    terms: list[float] = []
    for i in range(wf.natoms):
        zi = int(atomic_numbers[i])
        ri = cm5_radius(zi)
        for j in range(wf.natoms):
            if i == j:
                continue
            zj = int(atomic_numbers[j])
            distance = float(np.linalg.norm(coordinates_angstrom[i] - coordinates_angstrom[j]))
            decay = float(np.exp(-CM5_ALPHA * (distance - ri - cm5_radius(zj))))
            value = cm5_tval(zi, zj) * decay
            terms.append(value)
            corrected[i] += value
    return corrected, {"model": "cm5", "alpha": CM5_ALPHA, "correction_sum": float(np.sum(terms))}


def adch_charges(hirshfeld: np.ndarray, dipoles: np.ndarray, wf: WavefunctionData) -> tuple[np.ndarray, dict[str, Any]]:
    """Atomic-dipole-corrected Hirshfeld (ADCH) charges.

    Replicates Multiwfn ``doADC`` with ``chgtype=6``.
    """
    coords = wf.coordinates_bohr
    atomic_numbers = wf.atomic_numbers
    corrected = np.array(hirshfeld, dtype=float)
    for i in range(wf.natoms):
        center = coords[i]
        weights = np.zeros(wf.natoms, dtype=float)
        avg_r = np.zeros(3, dtype=float)
        avg_rr = np.zeros((3, 3), dtype=float)
        for j in range(wf.natoms):
            r = coords[j] - center
            distance = float(np.linalg.norm(r))
            rmax = (adch_vdw_radius(int(atomic_numbers[i])) + adch_vdw_radius(int(atomic_numbers[j]))) * ANGSTROM_TO_BOHR
            tr = distance / (rmax / 2.0) - 1.0
            tr = 1.5 * tr - 0.5 * tr**3
            tr = 1.5 * tr - 0.5 * tr**3
            weights[j] = 0.5 * (1.0 - (1.5 * tr - 0.5 * tr**3))
            if distance > rmax:
                weights[j] = 0.0
            avg_r += weights[j] * r
            avg_rr += weights[j] * np.outer(r, r)
        total = weights.sum()
        avg_r = avg_r / total
        avg_rr = avg_rr / total
        gamma = avg_rr - np.outer(avg_r, avg_r)
        eigvals, eigvecs = np.linalg.eigh(gamma)
        mat = np.zeros((3, 3), dtype=float)
        for k in range(3):
            if abs(eigvals[k]) > 1.0e-5:
                mat[k, k] = 1.0 / eigvals[k]
        avg_r_new = eigvecs.T @ avg_r
        dipole_new = eigvecs.T @ np.asarray(dipoles[i], dtype=float)
        for j in range(wf.natoms):
            r = coords[j] - center
            r_new = eigvecs.T @ r
            transfer = weights[j] / total * float((r_new - avg_r_new).T @ mat @ dipole_new)
            corrected[j] += transfer
    return corrected, {"model": "adch"}


def normalize_populations(charges: np.ndarray, wf: WavefunctionData) -> np.ndarray:
    """Multiwfn ``normalize_atmchg``: rescale populations to the exact nelec."""
    atomic_charges = wf.nuclear_charges
    populations = atomic_charges - np.asarray(charges, dtype=float)
    total_electrons = wf.mol.nelectron
    factor = total_electrons / populations.sum()
    return atomic_charges - factor * populations