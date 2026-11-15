"""Multiwfn reference data for the internal Hirshfeld/CM5 engine.

Constants transcribed from the reference-only Fortran source:

- ``covr`` / ``covr_pyy``: covalent radii (Angstrom) used by the CM5 equation,
  from ``define.f90`` of Multiwfn 2026.8.31.
- CM5 ``T`` coefficients from ``getCM5Tval`` in ``population.f90``.
"""

from __future__ import annotations

import numpy as np

#: CM5 bond-order-decay exponent (a.u.^{-1})
CM5_ALPHA = 2.474

#: Covalent radii, CSD / Multiwfn ``covr`` array, index = atomic number (Angstrom).
COVR = np.array(
    [
        0.1, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
        2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26,
        1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
        2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42,
        1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
        2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94,
        1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41,
        1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21,
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
    ],
    dtype=float,
)

#: Covalent radii, Pyykko / Multiwfn ``covr_pyy`` array, index = atomic number (Angstrom).
COVR_PYY = np.array(
    [
        0.1, 0.32, 0.46, 1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64, 0.67,
        1.55, 1.39, 1.26, 1.16, 1.11, 1.03, 0.99, 0.96,
        1.96, 1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.11,
        1.10, 1.12, 1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17,
        2.10, 1.85, 1.63, 1.54, 1.47, 1.38, 1.28, 1.25, 1.25,
        1.20, 1.28, 1.36, 1.42, 1.40, 1.40, 1.36, 1.33, 1.31,
        2.32, 1.96, 1.80, 1.63, 1.76, 1.74, 1.73, 1.72, 1.68, 1.69, 1.68,
        1.67, 1.66, 1.65, 1.64, 1.70, 1.62, 1.52, 1.46, 1.37, 1.31, 1.29, 1.22,
        1.23, 1.24, 1.34, 1.44, 1.44, 1.51, 1.45, 1.47, 1.42, 2.23, 2.01,
        1.86, 1.75, 1.69, 1.70, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65, 1.67,
        1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28, 1.21, 1.22,
        1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
    ],
    dtype=float,
)


def cm5_radius(z: int) -> float:
    """CM5 covalent radius (Angstrom): average of CSD and Pyykko radii."""
    covr = COVR[z] if z < len(COVR) else 1.5
    pyy = COVR_PYY[z] if z < len(COVR_PYY) else 1.5
    return 0.5 * (covr + pyy)


#: CM5 diagonal parameter Dparm (index = atomic number). Transcribed from
#: ``getCM5Tval``. Values default to 0.0.
_DPARM: dict[int, float] = {
    1: 0.0056, 2: -0.1543, 4: 0.0333, 5: -0.1030, 6: -0.0446, 7: -0.1072,
    8: -0.0802, 9: -0.0629, 10: -0.1088, 11: 0.0184, 13: -0.0726, 14: -0.0790,
    15: -0.0756, 16: -0.0565, 17: -0.0444, 18: -0.0767, 19: 0.0130,
    31: -0.0512, 32: -0.0557, 33: -0.0533, 34: -0.0399, 35: -0.0313, 36: -0.0541,
    37: 0.0092, 49: -0.0361, 50: -0.0393, 51: -0.0376, 52: -0.0281, 53: -0.0220,
    54: -0.0381, 55: 0.0065, 81: -0.0255, 82: -0.0277, 83: -0.0265, 84: -0.0198,
    85: -0.0155, 86: -0.0269, 87: 0.0046, 113: -0.0179, 114: -0.0195,
    115: -0.0187, 116: -0.0140, 117: -0.0110, 118: -0.0189,
}

#: CM5 explicit off-diagonal T(Z_i, Z_j) terms beyond the Dparm difference.
_CM5_EXPLICIT: dict[tuple[int, int], float] = {
    (1, 6): 0.0502, (6, 1): -0.0502,
    (1, 7): 0.1747, (7, 1): -0.1747,
    (1, 8): 0.1671, (8, 1): -0.1671,
    (6, 7): 0.0556, (7, 6): -0.0556,
    (6, 8): 0.0234, (8, 6): -0.0234,
    (7, 8): -0.0346, (8, 7): 0.0346,
}


def cm5_tval(zi: int, zj: int) -> float:
    """CM5 T(Z_i, Z_j) coefficient (see Multiwfn ``getCM5Tval``)."""
    pair = _CM5_EXPLICIT.get((zi, zj))
    if pair is not None:
        return pair
    return _DPARM.get(zi, 0.0) - _DPARM.get(zj, 0.0)


#: Modified Bondi vdW radii (Angstrom) used by ADCH weighting, from Multiwfn
#: ``vdwr`` in ``define.f90`` (index = atomic number). Values for Z>=83 are 2.0.
_VDW_VALUES = [
    1.2, 1.4,
    1.82, 1.77, 1.74, 1.7, 1.55, 1.52, 1.47, 1.54,
    2.27, 1.73, 1.73, 2.1, 1.8, 1.8, 1.75, 1.88,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9, 1.85, 2.02,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06, 1.98, 2.16,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    1.72, 1.66, 1.55, 1.96, 2.02,
]
VDW_RADII = np.zeros(119, dtype=float)
VDW_RADII[1 : 1 + len(_VDW_VALUES)] = _VDW_VALUES
VDW_RADII = np.where(VDW_RADII == 0.0, 2.0, VDW_RADII)


def adch_vdw_radius(z: int) -> float:
    """ADCH vdW radius (Angstrom) for atomic number ``z``."""
    if z < len(VDW_RADII) and z > 0:
        return float(VDW_RADII[z])
    return 2.0