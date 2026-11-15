"""Validated pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .charges import CHARGE_METHODS
from .errors import InputError

# Structures with more than this many atoms default to AM1-BCC instead of RESP
# when no charge model is supplied explicitly.
AUTO_AM1BCC_ATOM_THRESHOLD = 100


@dataclass(slots=True)
class PipelineConfig:
    structure: Path
    output: str = "lammps"
    parameter_mode: str = "classic"
    force_field: str = "gaff+uff"
    charge: str | None = None
    net_charge: int = 0
    multiplicity: int = 1
    qc: str = "b3lyp-d3bj/def2-svp"
    qc_backend: str = "pyscf"
    resp2_delta: float = 0.5
    outdir: Path | None = None
    hessian_file: Path | None = None
    hessian_method: str = "seminario"
    manual_file: Path | None = None
    no_qc: bool = False
    engine_smoke: bool = False
    emit_lt: bool = False
    debug: bool = False
    force: bool = False

    def validate(self) -> PipelineConfig:
        self.structure = Path(self.structure)
        if self.output not in {"gromacs", "lammps"}:
            raise InputError(f"output must be 'gromacs' or 'lammps', got {self.output!r}")
        if self.parameter_mode not in {"classic", "hessian"}:
            raise InputError(f"parameter mode must be 'classic' or 'hessian', got {self.parameter_mode!r}")
        if self.force_field not in {"gaff", "gaff+uff", "uff", "manual"}:
            raise InputError(f"unsupported force field {self.force_field!r}")
        if self.qc_backend not in {"pyscf", "auto"}:
            raise InputError(f"unsupported QC backend {self.qc_backend!r}")
        if self.charge is not None and not isinstance(self.charge, str):
            raise InputError(f"charge spec must be a method name or file path, got {self.charge!r}")
        if int(self.net_charge) != self.net_charge:
            raise InputError("net charge must be an integer formal charge")
        if self.multiplicity < 1:
            raise InputError("multiplicity must be a positive integer")
        if not 0.0 <= self.resp2_delta <= 1.0:
            raise InputError("resp2_delta must be between 0 and 1")
        if not self.structure.exists():
            raise InputError(f"structure file does not exist: {self.structure}")
        if self.hessian_file is not None:
            self.hessian_file = Path(self.hessian_file)
        if self.hessian_method.lower() not in {"seminario", "mseminario", "m2seminario", "drih"}:
            raise InputError(f"unsupported Hessian parameter method {self.hessian_method!r}")
        if self.manual_file is not None:
            self.manual_file = Path(self.manual_file)
        if self.outdir is not None:
            self.outdir = Path(self.outdir)
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "structure": str(self.structure),
            "output": self.output,
            "parameter_mode": self.parameter_mode,
            "force_field": self.force_field,
            "charge": self.charge,
            "net_charge": self.net_charge,
            "multiplicity": self.multiplicity,
            "qc": self.qc,
            "qc_backend": self.qc_backend,
            "resp2_delta": self.resp2_delta,
            "outdir": str(self.outdir) if self.outdir else None,
            "hessian_file": str(self.hessian_file) if self.hessian_file else None,
            "hessian_method": self.hessian_method,
            "manual_file": str(self.manual_file) if self.manual_file else None,
            "no_qc": self.no_qc,
            "engine_smoke": self.engine_smoke,
            "emit_lt": self.emit_lt,
            "debug": self.debug,
            "force": self.force,
        }


def known_charge_methods() -> tuple[str, ...]:
    """Expose the accepted charge method keywords for help text."""

    return CHARGE_METHODS
