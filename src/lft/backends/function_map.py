"""Explicit GROMACS/LAMMPS functional-form mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FunctionMap:
    family: str
    bond: str
    angle: str
    dihedral: str
    improper: str


# Proper dihedrals are always emitted as a single periodic (fourier) term in
# both backends.  The "harmonic" input family (UFF/conjugated placeholders) is
# exactly equivalent to a fourier term at phase 0 or 180 degrees.
GAFF_FUNCTIONS = FunctionMap("gaff", "harmonic", "harmonic", "fourier", "cvff")
UFF_FUNCTIONS = FunctionMap("uff", "harmonic", "harmonic", "fourier", "cvff")


def gromacs_periodic_energy(phi_degree: float, phase_degree: float, k_kcal_mol: float, periodicity: int) -> float:
    return k_kcal_mol * (1.0 + math.cos(math.radians(periodicity * phi_degree - phase_degree)))


def lammps_fourier_energy(phi_degree: float, phase_degree: float, k_kcal_mol: float, periodicity: int) -> float:
    return gromacs_periodic_energy(phi_degree, phase_degree, k_kcal_mol, periodicity)


def gromacs_harmonic_energy(phi_degree: float, phase_degree: float, k_kcal_mol_rad2: float) -> float:
    return 0.5 * k_kcal_mol_rad2 * math.radians(phi_degree - phase_degree) ** 2


def lammps_harmonic_energy(phi_degree: float, phase_degree: float, k_kcal_mol: float, periodicity: int) -> float:
    # LAMMPS dihedral_style harmonic is periodic.  This function is kept
    # explicit because it is not the same formula as GROMACS functype 2.
    sign = 1.0 if abs((phase_degree % 360.0)) < 90.0 else -1.0
    return k_kcal_mol * (1.0 + sign * math.cos(math.radians(periodicity * phi_degree)))
