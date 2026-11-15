"""Backend-neutral force-field parameter records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AtomType:
    name: str
    element: str
    mass: float
    sigma: float
    epsilon: float
    source: str
    atomic_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "element": self.element,
            "atomic_number": self.atomic_number,
            "mass": self.mass,
            "sigma_angstrom": self.sigma,
            "epsilon_kcal_mol": self.epsilon,
            "source": self.source,
        }


@dataclass(slots=True)
class BondParameter:
    r0: float
    k: float
    function: str = "harmonic"
    source: str = "classic_guess"

    def to_dict(self) -> dict[str, Any]:
        return {"r0_angstrom": self.r0, "k_kcal_mol_angstrom2": self.k, "function": self.function, "source": self.source}


@dataclass(slots=True)
class AngleParameter:
    theta0: float
    k: float
    function: str = "harmonic"
    source: str = "classic_guess"

    def to_dict(self) -> dict[str, Any]:
        return {"theta0_degree": self.theta0, "k_kcal_mol_rad2": self.k, "function": self.function, "source": self.source}


@dataclass(slots=True)
class DihedralParameter:
    phase: float
    k: float
    periodicity: int
    function: str = "fourier"
    source: str = "classic_guess"
    multiplicity_divisor: float = 1.0
    #: Full multi-term Fourier expansion (k, periodicity, phase).  GAFF
    #: dihedral types routinely need 2-4 terms; the scalar fields above mirror
    #: the first term for backwards compatibility.
    terms: tuple[tuple[float, int, float], ...] = ()

    def fourier_terms(self) -> tuple[tuple[float, int, float], ...]:
        if self.terms:
            return self.terms
        return ((self.k, self.periodicity, self.phase),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_degree": self.phase,
            "k_kcal_mol": self.k,
            "periodicity": self.periodicity,
            "function": self.function,
            "source": self.source,
            "multiplicity_divisor": self.multiplicity_divisor,
            "terms": [
                {"k_kcal_mol": k, "periodicity": n, "phase_degree": phase}
                for k, n, phase in self.fourier_terms()
            ],
        }


@dataclass(slots=True)
class ImproperParameter:
    phase: float
    k: float
    periodicity: int = 2
    function: str = "cvff"
    source: str = "classic_guess"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_degree": self.phase,
            "k_kcal_mol": self.k,
            "periodicity": self.periodicity,
            "function": self.function,
            "source": self.source,
        }


@dataclass(slots=True)
class ParameterSet:
    force_field: str
    parameter_mode: str
    atom_types: dict[str, AtomType] = field(default_factory=dict)
    bond_parameters: list[BondParameter] = field(default_factory=list)
    angle_parameters: list[AngleParameter] = field(default_factory=list)
    dihedral_parameters: list[DihedralParameter] = field(default_factory=list)
    improper_parameters: list[ImproperParameter] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_field": self.force_field,
            "parameter_mode": self.parameter_mode,
            "atom_types": {key: value.to_dict() for key, value in sorted(self.atom_types.items())},
            "bond_parameters": [item.to_dict() for item in self.bond_parameters],
            "angle_parameters": [item.to_dict() for item in self.angle_parameters],
            "dihedral_parameters": [item.to_dict() for item in self.dihedral_parameters],
            "improper_parameters": [item.to_dict() for item in self.improper_parameters],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
