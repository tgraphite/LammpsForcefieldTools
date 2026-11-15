"""PySCF single-point and Hessian adapter."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..domain.structure import Structure
from ..errors import ExternalToolError
from .backend import QCResult


def run_pyscf(structure: Structure, *, qc: str, need_hessian: bool = False, aqueous: bool = False, workdir: str | None = None) -> QCResult:
    try:
        from pyscf import dft, gto, scf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExternalToolError("PySCF is required for this QC task") from exc
    method, basis = _parse_qc(qc)
    atom_spec = [(atom.element, tuple(float(value) for value in atom.position)) for atom in structure.atoms]
    molecule = gto.M(
        atom=atom_spec,
        basis=basis,
        charge=int(round(structure.declared_net_charge)),
        spin=max(0, int(structure.multiplicity) - 1),
        unit="Angstrom",
        verbose=0,
    )
    if method.lower() in {"hf", "rhf", "rohf"}:
        mean_field: Any = scf.ROHF(molecule) if structure.multiplicity > 1 else scf.RHF(molecule)
        method_name = "ROHF" if structure.multiplicity > 1 else "RHF"
    else:
        mean_field = dft.UKS(molecule) if structure.multiplicity > 1 else dft.RKS(molecule)
        mean_field.xc = _dft_xc(method)
        method_name = method
    if aqueous:
        # PCM APIs vary across PySCF minor releases.  The explicit metadata
        # records whether the optional solvent wrapper was enabled; the gas
        # result remains a valid deterministic fallback for charge fitting.
        try:
            from pyscf import solvent  # type: ignore[import-not-found]

            mean_field = mean_field.PCM()
            mean_field.with_solvent.eps = 78.39
            mean_field.with_solvent.method = "C-PCM"
            solvent_enabled = True
        except Exception:
            solvent_enabled = False
    else:
        solvent_enabled = False
    mean_field.conv_tol = 1.0e-9
    try:
        energy = float(mean_field.kernel())
    except Exception as exc:
        raise ExternalToolError(f"PySCF SCF failed for {structure.name}: {exc}") from exc
    if not mean_field.converged:
        raise ExternalToolError(f"PySCF SCF did not converge for {structure.name}")
    from .wavefunction import build_wavefunction

    hessian = None
    if need_hessian:
        try:
            hessian = np.asarray(mean_field.Hessian().kernel(), dtype=float)
        except Exception as exc:
            raise ExternalToolError(f"PySCF Hessian failed for {structure.name}: {exc}") from exc
    wavefunction = build_wavefunction(molecule, mean_field, method=method_name, basis=basis, spin=structure.multiplicity - 1)
    from ..charges.density_charges import mulliken_charges

    population, _ = mulliken_charges(wavefunction)
    return QCResult(
        backend="pyscf",
        method=method_name,
        basis=basis,
        energy_hartree=energy,
        population_charges=population,
        hessian=hessian,
        wavefunction=wavefunction,
        metadata={"aqueous_requested": aqueous, "pcm_enabled": solvent_enabled, "spin": structure.multiplicity - 1, "qc": qc, "population_model": "mulliken"},
    )


def _parse_qc(value: str) -> tuple[str, str]:
    if "/" not in value:
        return value, "def2-svp"
    method, basis = value.split("/", 1)
    if not method or not basis:
        raise ExternalToolError(f"invalid QC specification {value!r}; expected method/basis")
    return method, basis


def _dft_xc(method: str) -> str:
    normalized = method.lower().replace("-d3(bj)", "").replace("-d3bj", "")
    return {"b3lyp": "b3lyp", "pbe0": "pbe0", "pbe": "pbe", "blyp": "blyp"}.get(normalized, normalized)
