"""Named unit conversions shared by both output writers."""

from __future__ import annotations

import math

ANGSTROM_PER_NM = 10.0
KCAL_PER_KJ = 0.2390057361376673
KJ_PER_KCAL = 1.0 / KCAL_PER_KJ
KCAL_MOL_PER_ANGSTROM2_TO_KJ_MOL_NM2 = 418.4
BOHR_PER_ANGSTROM = 1.8897261246257702
ANGSTROM_PER_BOHR = 1.0 / BOHR_PER_ANGSTROM
HARTREE_TO_KCAL_MOL = 627.5094740631
COULOMB_CONSTANT_REAL = 332.06371


def angstrom_to_nm(value: float) -> float:
    return value / ANGSTROM_PER_NM


def nm_to_angstrom(value: float) -> float:
    return value * ANGSTROM_PER_NM


def kcal_to_kj(value: float) -> float:
    return value * KJ_PER_KCAL


def kj_to_kcal(value: float) -> float:
    return value * KCAL_PER_KJ


def bond_kcal_to_gromacs(value: float) -> float:
    # GROMACS bond functype 1 is V = (1/2) k (r-r0)^2; the reference stores the
    # AMBER/GAFF full k in kcal/mol/A^2.  Convert with the extra factor of two.
    return value * 2.0 * KCAL_MOL_PER_ANGSTROM2_TO_KJ_MOL_NM2


def angle_kcal_to_gromacs(value: float) -> float:
    # GROMACS angle functype 1 is V = (1/2) k (theta-theta0)^2; the reference
    # stores the AMBER/GAFF full k in kcal/mol/rad^2.
    return value * 2.0 * KJ_PER_KCAL


def angle_deg_to_rad(value: float) -> float:
    return math.radians(value)


def angle_rad_to_deg(value: float) -> float:
    return math.degrees(value)


def sigma_from_rmin(rmin: float) -> float:
    return 2.0 * rmin / (2.0 ** (1.0 / 6.0))


def rmin_from_sigma(sigma: float) -> float:
    return sigma * (2.0 ** (1.0 / 6.0)) / 2.0


def hessian_hartree_bohr2_to_kcal_angstrom2(value: float) -> float:
    return value * HARTREE_TO_KCAL_MOL * BOHR_PER_ANGSTROM**2
