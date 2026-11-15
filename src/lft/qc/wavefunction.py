"""In-memory wavefunction data produced by the QC backend and consumed by the
internal population/ESP charge engine.

The object is intentionally not JSON-serializable: ``pipeline._qc_artifact``
only serializes the small summary fields of ``QCResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyscf.gto import Mole  # type: ignore[import-not-found]


@dataclass(slots=True)
class WavefunctionData:
    mol: Any
    dm_alpha: np.ndarray
    dm_beta: np.ndarray
    atomic_numbers: np.ndarray
    coordinates_bohr: np.ndarray
    method: str
    basis: str
    net_charge: int
    spin: int
    population_model: str = "density_matrix"
    mo_coeff: Any | None = None
    mo_occ: Any | None = None

    @property
    def natoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def nao(self) -> int:
        return self.dm_alpha.shape[0]

    @property
    def total_dm(self) -> np.ndarray:
        return self.dm_alpha + self.dm_beta

    @property
    def overlap(self) -> np.ndarray:
        return np.asarray(self.mol.intor("int1e_ovlp"), dtype=float)

    @property
    def nuclear_charges(self) -> np.ndarray:
        return self.atomic_numbers.astype(float)

    @property
    def is_unrestricted(self) -> bool:
        return bool(np.any(self.dm_beta != 0.0))


def build_wavefunction(mol: Any, mean_field: Any, *, method: str, basis: str, spin: int) -> WavefunctionData:
    rdm1 = mean_field.make_rdm1()
    if isinstance(rdm1, tuple):
        dm_alpha, dm_beta = rdm1
    else:
        dm_alpha, dm_beta = rdm1, np.zeros_like(rdm1)
    return WavefunctionData(
        mol=mol,
        dm_alpha=np.asarray(dm_alpha, dtype=float),
        dm_beta=np.asarray(dm_beta, dtype=float),
        atomic_numbers=np.asarray(mol.atom_charges(), dtype=int),
        coordinates_bohr=np.asarray(mol.atom_coords(unit="Bohr"), dtype=float),
        method=method,
        basis=basis,
        net_charge=int(round(mol.charge)),
        spin=int(spin),
        mo_coeff=np.asarray(mean_field.mo_coeff, dtype=float),
        mo_occ=np.asarray(mean_field.mo_occ, dtype=float),
    )