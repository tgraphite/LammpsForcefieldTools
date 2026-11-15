"""QC backend protocol and the built-in PySCF implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..domain.structure import Structure
from ..errors import ExternalToolError


@dataclass(slots=True)
class QCResult:
    backend: str
    method: str
    basis: str
    energy_hartree: float | None = None
    population_charges: np.ndarray | None = None
    hessian: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    aqueous_population_charges: np.ndarray | None = None
    wavefunction: Any | None = None
    aqueous_wavefunction: Any | None = None


def run_qc(
    structure: Structure,
    *,
    qc: str,
    backend: str = "pyscf",
    need_hessian: bool = False,
    aqueous: bool = False,
    workdir: str | None = None,
) -> QCResult:
    if backend not in {"pyscf", "auto"}:
        raise ExternalToolError(f"QC backend {backend!r} is not available in this installation", hint="use --qc-backend pyscf or provide an external Hessian/charge artifact")
    try:
        from .pyscf_backend import run_pyscf
    except Exception as exc:
        raise ExternalToolError(f"PySCF backend import failed: {exc}") from exc
    return run_pyscf(structure, qc=qc, need_hessian=need_hessian, aqueous=aqueous, workdir=workdir)
