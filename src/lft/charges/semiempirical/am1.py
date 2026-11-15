"""AM1 (Austin Model 1) semi-empirical RHF SCF, pure numpy/scipy port.

This module is a faithful port of the AmberTools ``sqm`` AM1 RHF
single-point machinery (Fortran sources in ``AmberTools/src/sqm``).  Only
the sp (s/p valence) NDDO path is implemented, which covers the AM1
elements H, C, N, O, F, Mg, Al, Si, P, S, Cl, Zn, Ge, Br, I and Hg.

The implementation intentionally mirrors the reference code, including
the STO-6G Gaussian expansion tables, the two-centre multipole electron
repulsion integrals (RI), the local-frame rotation that builds the W
integrals, the core-electron attraction, the core-core repulsion with
AM1 Gaussian corrections, the Fock assembly (full symmetric matrices,
equivalent to sqm's packed lower-triangle layout) and the density matrix
extrapolation used by ``qm2_cnvg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

__all__ = ["AM1", "AM1_PARAMS"]

# ---------------------------------------------------------------------------
# Physical constants (mirroring AmberTools src/lib/constants.F90)
# ---------------------------------------------------------------------------
AU_TO_EV = 27.21
HALF_AU_TO_EV = AU_TO_EV * 0.5
FOURTH_AU_TO_EV = AU_TO_EV * 0.25
EIGHTH_AU_TO_EV = AU_TO_EV * 0.125
SXNTH_AU_TO_EV = EIGHTH_AU_TO_EV * 0.5
EV_TO_KCAL = 23.061
KCAL_TO_EV = 1.0 / EV_TO_KCAL
BOHRS_TO_A = 0.529177249
A_TO_BOHRS = 1.0 / BOHRS_TO_A
A2_TO_BOHRS2 = A_TO_BOHRS * A_TO_BOHRS
A3_TO_BOHRS3 = A2_TO_BOHRS2 * A_TO_BOHRS
EXPONENTIAL_CUTOFF = 30.0
OVERLAP_CUTOFF = 100.0 * A2_TO_BOHRS2

MAX_GAUSSIAN_EXPANSION = 6

class _Am1Param(TypedDict):
    s_orb_exp: float
    p_orb_exp: float
    betas: float
    betap: float
    GSS: float
    GSP: float
    GPP: float
    GP2: float
    HSP: float
    alp: float
    USS: float
    UPP: float
    NUM_FN: int
    FN1: list[float]
    FN2: list[float]
    FN3: list[float]


# AM1 parameters, keyed by atomic number (from qm2_parameters.F90)
AM1_PARAMS: dict[int, _Am1Param] = {
    1: {
        "s_orb_exp": 1.188078,
        "p_orb_exp": 0.0,
        "betas": -6.173787,
        "betap": 0.0,
        "GSS": 12.848,
        "GSP": 0.0,
        "GPP": 0.0,
        "GP2": 0.0,
        "HSP": 0.0,
        "alp": 2.882324,
        "USS": -11.396427,
        "UPP": 0.0,
        "NUM_FN": 3,
        "FN1": [0.122796, 0.00509, -0.018336, 0.0],
        "FN2": [5.0, 5.0, 2.0, 0.0],
        "FN3": [1.2, 1.8, 2.1, 0.0],
    },
    6: {
        "s_orb_exp": 1.808665,
        "p_orb_exp": 1.685116,
        "betas": -15.715783,
        "betap": -7.719283,
        "GSS": 12.23,
        "GSP": 11.47,
        "GPP": 11.08,
        "GP2": 9.84,
        "HSP": 2.43,
        "alp": 2.648274,
        "USS": -52.028658,
        "UPP": -39.614239,
        "NUM_FN": 4,
        "FN1": [0.011355, 0.045924, -0.020061, -0.00126],
        "FN2": [5.0, 5.0, 5.0, 5.0],
        "FN3": [1.6, 1.85, 2.05, 2.65],
    },
    7: {
        "s_orb_exp": 2.31541,
        "p_orb_exp": 2.15794,
        "betas": -20.29911,
        "betap": -18.238666,
        "GSS": 13.59,
        "GSP": 12.66,
        "GPP": 12.98,
        "GP2": 11.59,
        "HSP": 3.14,
        "alp": 2.947286,
        "USS": -71.86,
        "UPP": -57.167581,
        "NUM_FN": 3,
        "FN1": [0.025251, 0.028953, -0.005806, 0.0],
        "FN2": [5.0, 5.0, 2.0, 0.0],
        "FN3": [1.5, 2.1, 2.4, 0.0],
    },
    8: {
        "s_orb_exp": 3.108032,
        "p_orb_exp": 2.524039,
        "betas": -29.272773,
        "betap": -29.272773,
        "GSS": 15.42,
        "GSP": 14.48,
        "GPP": 14.52,
        "GP2": 12.98,
        "HSP": 3.94,
        "alp": 4.455371,
        "USS": -97.83,
        "UPP": -78.26238,
        "NUM_FN": 2,
        "FN1": [0.280962, 0.08143, 0.0, 0.0],
        "FN2": [5.0, 7.0, 0.0, 0.0],
        "FN3": [0.847918, 1.445071, 0.0, 0.0],
    },
    9: {
        "s_orb_exp": 3.770082,
        "p_orb_exp": 2.49467,
        "betas": -69.590277,
        "betap": -27.92236,
        "GSS": 16.92,
        "GSP": 17.25,
        "GPP": 16.71,
        "GP2": 14.91,
        "HSP": 4.83,
        "alp": 5.5178,
        "USS": -136.105579,
        "UPP": -104.889885,
        "NUM_FN": 2,
        "FN1": [0.242079, 0.003607, 0.0, 0.0],
        "FN2": [4.8, 4.6, 0.0, 0.0],
        "FN3": [0.93, 1.66, 0.0, 0.0],
    },
    12: {
        "s_orb_exp": 1.2233927,
        "p_orb_exp": 1.02030798,
        "betas": -1.25974355,
        "betap": -0.77836604,
        "GSS": 7.50132277,
        "GSP": 6.34591536,
        "GPP": 4.77534467,
        "GP2": 4.34017279,
        "HSP": 0.48930466,
        "alp": 1.67049799,
        "USS": -14.96959313,
        "UPP": -11.56229248,
        "NUM_FN": 3,
        "FN1": [2.55017735, -0.00565806, -0.00610286, 0.0],
        "FN2": [2.55017735, 2.9605391, 2.61416919, 0.0],
        "FN3": [0.79989601, 1.47499983, 2.4260404, 0.0],
    },
    13: {
        "s_orb_exp": 1.516593,
        "p_orb_exp": 1.306347,
        "betas": -3.866822,
        "betap": -2.317146,
        "GSS": 8.09,
        "GSP": 6.63,
        "GPP": 5.98,
        "GP2": 5.4,
        "HSP": 0.7,
        "alp": 1.976586,
        "USS": -24.353585,
        "UPP": -18.363645,
        "NUM_FN": 1,
        "FN1": [0.09, 0.0, 0.0, 0.0],
        "FN2": [12.392443, 0.0, 0.0, 0.0],
        "FN3": [2.050394, 0.0, 0.0, 0.0],
    },
    14: {
        "s_orb_exp": 1.830697,
        "p_orb_exp": 1.284953,
        "betas": -3.784852,
        "betap": -1.968123,
        "GSS": 9.82,
        "GSP": 8.36,
        "GPP": 7.31,
        "GP2": 6.54,
        "HSP": 1.32,
        "alp": 2.257816,
        "USS": -33.953622,
        "UPP": -28.934749,
        "NUM_FN": 3,
        "FN1": [0.25, 0.061513, 0.020789, 0.0],
        "FN2": [9.0, 5.0, 5.0, 0.0],
        "FN3": [0.911453, 1.995569, 2.99061, 0.0],
    },
    15: {
        "s_orb_exp": 1.98128,
        "p_orb_exp": 1.87515,
        "betas": -6.353764,
        "betap": -6.590709,
        "GSS": 11.560005,
        "GSP": 5.237449,
        "GPP": 7.877589,
        "GP2": 7.307648,
        "HSP": 0.779238,
        "alp": 2.455322,
        "USS": -42.029863,
        "UPP": -34.030709,
        "NUM_FN": 3,
        "FN1": [-0.031827, 0.01847, 0.03329, 0.0],
        "FN2": [6.0, 7.0, 9.0, 0.0],
        "FN3": [1.474323, 1.779354, 3.006576, 0.0],
    },
    16: {
        "s_orb_exp": 2.366515,
        "p_orb_exp": 1.667263,
        "betas": -3.920566,
        "betap": -7.905278,
        "GSS": 11.786329,
        "GSP": 8.663127,
        "GPP": 10.039308,
        "GP2": 7.781688,
        "HSP": 2.532137,
        "alp": 2.461648,
        "USS": -56.694056,
        "UPP": -48.717049,
        "NUM_FN": 3,
        "FN1": [-0.509195, -0.011863, 0.012334, 0.0],
        "FN2": [4.593691, 5.865731, 13.557336, 0.0],
        "FN3": [0.770665, 1.503313, 2.009173, 0.0],
    },
    17: {
        "s_orb_exp": 3.631376,
        "p_orb_exp": 2.076799,
        "betas": -24.59467,
        "betap": -14.637216,
        "GSS": 15.03,
        "GSP": 13.16,
        "GPP": 11.3,
        "GP2": 9.97,
        "HSP": 2.42,
        "alp": 2.919368,
        "USS": -111.613948,
        "UPP": -76.640107,
        "NUM_FN": 2,
        "FN1": [0.094243, 0.027168, 0.0, 0.0],
        "FN2": [4.0, 4.0, 0.0, 0.0],
        "FN3": [1.3, 2.1, 0.0, 0.0],
    },
    30: {
        "s_orb_exp": 1.954299,
        "p_orb_exp": 1.372365,
        "betas": -1.997429,
        "betap": -4.758119,
        "GSS": 11.8,
        "GSP": 11.182018,
        "GPP": 13.3,
        "GP2": 12.93052,
        "HSP": 0.484606,
        "alp": 1.484563,
        "USS": -21.040008,
        "UPP": -17.655574,
        "NUM_FN": 0,
        "FN1": [0.0, 0.0, 0.0, 0.0],
        "FN2": [0.0, 0.0, 0.0, 0.0],
        "FN3": [0.0, 0.0, 0.0, 0.0],
    },
    32: {
        "s_orb_exp": 1.219631,
        "p_orb_exp": 1.982794,
        "betas": -4.356607,
        "betap": -0.991091,
        "GSS": 10.168605,
        "GSP": 8.144473,
        "GPP": 6.671902,
        "GP2": 6.269706,
        "HSP": 0.937093,
        "alp": 2.136405,
        "USS": -34.183889,
        "UPP": -28.640811,
        "NUM_FN": 0,
        "FN1": [0.0, 0.0, 0.0, 0.0],
        "FN2": [0.0, 0.0, 0.0, 0.0],
        "FN3": [0.0, 0.0, 0.0, 0.0],
    },
    35: {
        "s_orb_exp": 3.064133,
        "p_orb_exp": 2.038333,
        "betas": -19.39988,
        "betap": -8.957195,
        "GSS": 15.0364395,
        "GSP": 13.0346824,
        "GPP": 11.2763254,
        "GP2": 9.8544255,
        "HSP": 2.4558683,
        "alp": 2.576546,
        "USS": -104.656063,
        "UPP": -74.930052,
        "NUM_FN": 2,
        "FN1": [0.066685, 0.025568, 0.0, 0.0],
        "FN2": [4.0, 4.0, 0.0, 0.0],
        "FN3": [1.5, 2.3, 0.0, 0.0],
    },
    53: {
        "s_orb_exp": 2.102858,
        "p_orb_exp": 2.161153,
        "betas": -8.443327,
        "betap": -6.323405,
        "GSS": 15.0404486,
        "GSP": 13.056558,
        "GPP": 11.1477837,
        "GP2": 9.9140907,
        "HSP": 2.456382,
        "alp": 2.299424,
        "USS": -103.589663,
        "UPP": -74.429997,
        "NUM_FN": 2,
        "FN1": [0.004361, 0.015706, 0.0, 0.0],
        "FN2": [2.3, 3.0, 0.0, 0.0],
        "FN3": [1.8, 2.24, 0.0, 0.0],
    },
    80: {
        "s_orb_exp": 2.036413,
        "p_orb_exp": 1.955766,
        "betas": -0.908657,
        "betap": -4.909384,
        "GSS": 10.8,
        "GSP": 9.3,
        "GPP": 14.3,
        "GP2": 13.5,
        "HSP": 1.3,
        "alp": 1.484734,
        "USS": -19.941578,
        "UPP": -11.11087,
        "NUM_FN": 0,
        "FN1": [0.0, 0.0, 0.0, 0.0],
        "FN2": [0.0, 0.0, 0.0, 0.0],
        "FN3": [0.0, 0.0, 0.0, 0.0],
    },
}

# Element data: core charge, number of valence orbitals, s/p occupancies,
# principal shell and gas-phase heat of formation (kcal/mol).
_ELEMENT_DATA: dict[int, dict[str, float]] = {
    1: {"core_chg": 1, "natomic_orbs": 1, "ios": 1, "iop": 0, "nsshell": 1, "heat_of_form": 52.102},
    6: {"core_chg": 4, "natomic_orbs": 4, "ios": 2, "iop": 2, "nsshell": 2, "heat_of_form": 170.890},
    7: {"core_chg": 5, "natomic_orbs": 4, "ios": 2, "iop": 3, "nsshell": 2, "heat_of_form": 113.000},
    8: {"core_chg": 6, "natomic_orbs": 4, "ios": 2, "iop": 4, "nsshell": 2, "heat_of_form": 59.559},
    9: {"core_chg": 7, "natomic_orbs": 4, "ios": 2, "iop": 5, "nsshell": 2, "heat_of_form": 18.890},
    12: {"core_chg": 2, "natomic_orbs": 4, "ios": 2, "iop": 0, "nsshell": 3, "heat_of_form": 35.000},
    13: {"core_chg": 3, "natomic_orbs": 4, "ios": 2, "iop": 1, "nsshell": 3, "heat_of_form": 79.490},
    14: {"core_chg": 4, "natomic_orbs": 4, "ios": 2, "iop": 2, "nsshell": 3, "heat_of_form": 108.390},
    15: {"core_chg": 5, "natomic_orbs": 4, "ios": 2, "iop": 3, "nsshell": 3, "heat_of_form": 75.570},
    16: {"core_chg": 6, "natomic_orbs": 4, "ios": 2, "iop": 4, "nsshell": 3, "heat_of_form": 66.400},
    17: {"core_chg": 7, "natomic_orbs": 4, "ios": 2, "iop": 5, "nsshell": 3, "heat_of_form": 28.990},
    30: {"core_chg": 2, "natomic_orbs": 4, "ios": 2, "iop": 0, "nsshell": 4, "heat_of_form": 31.170},
    32: {"core_chg": 4, "natomic_orbs": 4, "ios": 2, "iop": 2, "nsshell": 4, "heat_of_form": 89.500},
    35: {"core_chg": 7, "natomic_orbs": 4, "ios": 2, "iop": 5, "nsshell": 4, "heat_of_form": 26.740},
    53: {"core_chg": 7, "natomic_orbs": 4, "ios": 2, "iop": 5, "nsshell": 5, "heat_of_form": 25.517},
    80: {"core_chg": 2, "natomic_orbs": 4, "ios": 2, "iop": 0, "nsshell": 6, "heat_of_form": 14.690},
}

# Principal quantum number (SPPrincipalQuantumNumber in elementOrbitalIndex.F90)
_SP_QUANTUM_NUMBER: dict[int, int] = {
    1: 1,
    6: 2, 7: 2, 8: 2, 9: 2,
    12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3,
    30: 4, 32: 4, 35: 4,
    53: 5,
    80: 6,
}

_ELEMENT_SYMBOL: dict[str, int] = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36,
    "RB": 37, "SR": 38, "Y": 39, "ZR": 40, "NB": 41, "MO": 42, "TC": 43,
    "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57,
    "CE": 58, "PR": 59, "ND": 60, "PM": 61, "SM": 62, "EU": 63, "GD": 64,
    "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70, "LU": 71,
    "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78,
    "AU": 79, "HG": 80, "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85,
    "RN": 86,
}

# ---------------------------------------------------------------------------
# STO-6G expansion (R.F. Stewart, J. Chem. Phys. 52, 431 (1970))
# from qm2_setup_orb_exp.F90, indexed as (NQN, orbital type) -> 6 values.
# Orbital type 1 = s, 2 = p.
# ---------------------------------------------------------------------------
_STO6G_Z: dict[tuple[int, int], list[float]] = {
    (1, 1): [23.10303149, 4.235915534, 1.185056519, 0.4070988982, 0.1580884151, 0.06510953954],
    (2, 1): [27.68496241, 5.077140627, 1.42678605, 0.2040335729, 0.09260298399, 0.04416183978],
    (2, 2): [5.868285913, 1.530329631, 0.5475665231, 0.2288932733, 0.1046655969, 0.04948220127],
    (3, 1): [3.273031938, 0.9200611311, 0.3593349765, 0.08636686991, 0.04797373812, 0.02724741144],
    (3, 2): [5.077973607, 1.34078694, 0.2248434849, 0.1131741848, 0.06076408893, 0.03315424265],
    (4, 1): [3.232838646, 0.3605788802, 0.1717905487, 0.05277666487, 0.03163400284, 0.01874093091],
    (4, 2): [2.389722618, 0.7960947826, 0.341554138, 0.08847434525, 0.04958248334, 0.02816929784],
    (5, 1): [1.410128298, 0.5077878915, 0.1847926858, 0.1061070594, 0.03669584901, 0.0221355843],
    (5, 2): [3.778623374, 0.3499121109, 0.1683175469, 0.05404070736, 0.03328911801, 0.02063815019],
    (6, 1): [0.5800292686, 0.2718262251, 0.07938523262, 0.04975088254, 0.02983643556, 0.01886067216],
    (6, 2): [0.6696537714, 0.1395089793, 0.0816389496, 0.04586329272, 0.02961305556, 0.01882221321],
}
_STO6G_C: dict[tuple[int, int], list[float]] = {
    (1, 1): [0.00916359628, 0.04936149294, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841],
    (2, 1): [-0.004151277819, -0.02067024148, -0.05150303337, 0.3346271174, 0.5621061301, 0.1712994697],
    (2, 2): [0.007924233646, 0.05144104825, 0.189840006, 0.4049863191, 0.4012362861, 0.1051855189],
    (3, 1): [-0.006775596947, -0.05639325779, -0.1587856086, 0.5534527651, 0.501535102, 0.07223633674],
    (3, 2): [-0.00332992984, -0.0141948834, 0.163939577, 0.4485358256, 0.390881305, 0.07411456232],
    (4, 1): [0.001374817488, -0.08666390043, -0.3130627309, 0.7812787397, 0.4389247988, 0.02487178756],
    (4, 2): [-0.001665913575, -0.01657464971, -0.05958513378, 0.4053115554, 0.5433958189, 0.1204970491],
    (5, 1): [0.002695439582, 0.01850157487, -0.09588628125, -0.520067356, 1.08761949, 0.3103964343],
    (5, 2): [0.0001163246387, -0.02920771322, -0.1381051233, 0.5706134877, 0.476880814, 0.06021665516],
    (6, 1): [0.004554359511, 0.05286443143, -0.7561016358, -0.226980382, 1.332494651, 0.3622518293],
    (6, 2): [0.00278272368, -0.128288778, -0.2266255943, 0.4682259383, 0.6752048848, 0.1091534212],
}

# Standard NDDO pair ordering for s/p orbitals: (s,s),(x,s),(x,x),(y,s),
# (y,x),(y,y),(z,s),(z,x),(z,y),(z,z).
_PAIR_INDEX = np.array(
    [
        [0, 1, 3, 6],
        [1, 2, 4, 7],
        [3, 4, 5, 8],
        [6, 7, 8, 9],
    ],
    dtype=int,
)


def _get_atomic_number(symbol: str) -> int:
    """Return the atomic number for an element symbol."""
    key = symbol.strip().upper()
    if key not in _ELEMENT_SYMBOL:
        raise ValueError(f"unknown element symbol: {symbol!r}")
    return _ELEMENT_SYMBOL[key]


def _fsign(a: float, b: float) -> float:
    """Fortran SIGN(a, b): |a| with the sign of b (positive if b == 0)."""
    return abs(a) if b >= 0 else -abs(a)


# ---------------------------------------------------------------------------
# Slater-Condon / one-centre helpers are not needed for the sp AM1 path.
# ---------------------------------------------------------------------------


@dataclass
class _AtomParams:
    """Per-atom derived AM1 parameters (mirrors qm2_params entries)."""

    z: int
    core_chg: float
    natomic_orbs: int
    nsshell: int
    s_orb_exp: float
    p_orb_exp: float
    betas: float
    betap: float
    gss: float
    gsp: float
    gpp: float
    gp2: float
    hsp: float
    alp: float
    uss: float
    upp: float
    num_fn: int
    fn1: np.ndarray
    fn2: np.ndarray
    fn3: np.ndarray
    onec2elec: np.ndarray  # 0.5*GSS, GSP, 0.5*GPP, 1.25*GP2, 0.5*HSP
    multip: np.ndarray  # DD, QQ, AM, AD, AQ


def _secant_ad(hsp: float, dd: float) -> float:
    """Solve for AD via the secant iteration in qm2_load_params."""
    dd1 = (hsp / (AU_TO_EV * dd * dd)) ** (1.0 / 3.0)
    dd2 = dd1 + 0.04
    for _ in range(5):
        dd_diff = dd2 - dd1
        hsp1 = 0.5 * dd1 - 0.5 / np.sqrt(4.0 * dd * dd + 1.0 / (dd1 * dd1))
        hsp2 = 0.5 * dd2 - 0.5 / np.sqrt(4.0 * dd * dd + 1.0 / (dd2 * dd2))
        if abs(hsp2 - hsp1) < 1.0e-25:
            break
        dd3 = dd1 + dd_diff * (hsp / AU_TO_EV - hsp1) / (hsp2 - hsp1)
        dd1, dd2 = dd2, dd3
    return 0.5 / dd2


def _secant_aq(hpp: float, qq: float) -> float:
    """Solve for AQ via the secant iteration in qm2_load_params."""
    hpp = max(0.1, hpp)
    dd1 = (16.0 * hpp / (AU_TO_EV * 48.0 * qq**4)) ** (1.0 / 5.0)
    dd2 = dd1 + 0.04
    for _ in range(5):
        dd_diff = dd2 - dd1
        hsp1 = (
            0.25 * dd1
            - 0.5 / np.sqrt(4.0 * qq * qq + 1.0 / (dd1 * dd1))
            + 0.25 / np.sqrt(8.0 * qq * qq + 1.0 / (dd1 * dd1))
        )
        hsp2 = (
            0.25 * dd2
            - 0.5 / np.sqrt(4.0 * qq * qq + 1.0 / (dd2 * dd2))
            + 0.25 / np.sqrt(8.0 * qq * qq + 1.0 / (dd2 * dd2))
        )
        if abs(hsp2 - hsp1) < 1.0e-25:
            break
        dd3 = dd1 + dd_diff * (hpp / AU_TO_EV - hsp1) / (hsp2 - hsp1)
        dd1, dd2 = dd2, dd3
    return 0.5 / dd2


def _derive_atom_params(z: int) -> _AtomParams:
    """Compute the derived per-atom parameters used by the integrals."""
    p = AM1_PARAMS[z]
    e = _ELEMENT_DATA[z]
    nsshell = int(e["nsshell"])
    s_orb = p["s_orb_exp"]
    p_orb = p["p_orb_exp"]
    gss = p["GSS"]
    gsp = p["GSP"]
    gpp = p["GPP"]
    gp2 = p["GP2"]
    hsp = p["HSP"]

    # DD, QQ
    if (p_orb != 0.0 or s_orb != 0.0) and p_orb != 0.0:
        dd = ((4.0 * s_orb * p_orb) ** (nsshell + 0.5) * (2.0 * nsshell + 1.0)) / (
            (s_orb + p_orb) ** (2.0 * nsshell + 2.0) * np.sqrt(3.0)
        )
        qq = np.sqrt((4.0 * nsshell**2 + 6.0 * nsshell + 2.0) / 20.0) / p_orb
    else:
        dd = 0.0
        qq = 0.0

    am = (0.5 * AU_TO_EV) / gss if gss != 0.0 else 0.0
    if z == 1:
        ad = am
        aq = am
    else:
        ad = _secant_ad(hsp, dd)
        hpp = 0.5 * (gpp - gp2)
        aq = _secant_aq(hpp, qq)

    onec = np.array([0.5 * gss, gsp, 0.5 * gpp, 1.25 * gp2, 0.5 * hsp], dtype=float)
    multip = np.array([dd, qq, am, ad, aq], dtype=float)
    return _AtomParams(
        z=z,
        core_chg=e["core_chg"],
        natomic_orbs=int(e["natomic_orbs"]),
        nsshell=nsshell,
        s_orb_exp=s_orb,
        p_orb_exp=p_orb,
        betas=p["betas"],
        betap=p["betap"],
        gss=gss,
        gsp=gsp,
        gpp=gpp,
        gp2=gp2,
        hsp=hsp,
        alp=p["alp"],
        uss=p["USS"],
        upp=p["UPP"],
        num_fn=p["NUM_FN"],
        fn1=np.array(p["FN1"], dtype=float),
        fn2=np.array(p["FN2"], dtype=float),
        fn3=np.array(p["FN3"], dtype=float),
        onec2elec=onec,
        multip=multip,
    )


class _OrbitalTables:
    """Pre-computed STO-6G overlap quantities (qm2_setup_orb_exp)."""

    def __init__(self, atoms: list[_AtomParams]) -> None:
        ntypes = len(atoms)
        self.s_orb = np.array([a.s_orb_exp for a in atoms])
        self.p_orb = np.array([a.p_orb_exp for a in atoms])
        self.sp_qn = np.array([_SP_QUANTUM_NUMBER[a.z] for a in atoms])

        ng = MAX_GAUSSIAN_EXPANSION
        cc_s = np.zeros((ng, ntypes))
        zz_s = np.zeros((ng, ntypes))
        cc_p = np.zeros((ng, ntypes))
        zz_p = np.zeros((ng, ntypes))
        for k, a in enumerate(atoms):
            nqn = self.sp_qn[k]
            xi_s = a.s_orb_exp * a.s_orb_exp
            xi_p = a.p_orb_exp * a.p_orb_exp
            cc_s[:, k] = _STO6G_C[(nqn, 1)]
            zz_s[:, k] = np.array(_STO6G_Z[(nqn, 1)]) * xi_s
            if (nqn, 2) in _STO6G_C:
                cc_p[:, k] = _STO6G_C[(nqn, 2)]
                zz_p[:, k] = np.array(_STO6G_Z[(nqn, 2)]) * xi_p

        shape = (ng, ng, ntypes, ntypes)
        self.sxs_over_sas = np.zeros(shape)
        self.ss_eqn = np.zeros(shape)
        self.sxp_over_sap = np.zeros(shape)
        self.sp_ovlp = np.zeros(shape)
        self.pxp_over_pap = np.zeros(shape)
        self.pp_ovlp_inj = np.zeros(shape)
        self.pp_ovlp_ieqj1 = np.zeros(shape)
        self.pp_ovlp_ieqj2 = np.zeros(shape)

        for k in range(ntypes):
            for l in range(ntypes):
                for i in range(ng):
                    for j in range(ng):
                        zs_ik = zz_s[i, k]
                        zs_jl = zz_s[j, l]
                        zp_ik = zz_p[i, k]
                        zp_jl = zz_p[j, l]
                        cc_s_ik = cc_s[i, k]
                        cc_s_jl = cc_s[j, l]
                        cc_p_ik = cc_p[i, k]
                        cc_p_jl = cc_p[j, l]

                        zz_ss = zs_ik * zs_jl
                        one_ss = 1.0 / (zs_ik + zs_jl) if (zs_ik + zs_jl) != 0.0 else 0.0
                        self.sxs_over_sas[i, j, k, l] = zz_ss * one_ss
                        self.ss_eqn[i, j, k, l] = (
                            np.sqrt((2.0 * np.sqrt(zz_ss) * one_ss) ** 3) * cc_s_ik * cc_s_jl
                        )

                        zz_sp = zs_ik * zp_jl
                        one_sp = 1.0 / (zs_ik + zp_jl) if (zs_ik + zp_jl) != 0.0 else 0.0
                        self.sxp_over_sap[i, j, k, l] = zz_sp * one_sp
                        sp_eqn = np.sqrt((2.0 * np.sqrt(zz_sp) * one_sp) ** 3) * cc_s_ik * cc_p_jl
                        self.sp_ovlp[i, j, k, l] = (
                            2.0 * zs_ik * np.sqrt(zp_jl) * one_sp * sp_eqn
                        )

                        zz_pp = zp_ik * zp_jl
                        one_pp = 1.0 / (zp_ik + zp_jl) if (zp_ik + zp_jl) != 0.0 else 0.0
                        self.pxp_over_pap[i, j, k, l] = zz_pp * one_pp
                        pp_eqn = np.sqrt((2.0 * np.sqrt(zz_pp) * one_pp) ** 3) * cc_p_ik * cc_p_jl
                        self.pp_ovlp_inj[i, j, k, l] = (
                            -4.0 * np.sqrt(zz_pp) * one_pp * (zz_pp * one_pp) * pp_eqn
                        )
                        self.pp_ovlp_ieqj1[i, j, k, l] = (
                            -4.0 * pp_eqn * np.sqrt(zz_pp) * one_pp * (zz_pp * one_pp)
                        )
                        self.pp_ovlp_ieqj2[i, j, k, l] = (
                            2.0 * pp_eqn * np.sqrt(zz_pp) * one_pp
                        )


class AM1:
    """AM1 RHF single point calculation for a fixed molecular geometry."""

    def __init__(self, elements: list[str], coords_angstrom: np.ndarray) -> None:
        self.elements = list(elements)
        self.coords = np.asarray(coords_angstrom, dtype=float)
        if self.coords.shape != (len(self.elements), 3):
            raise ValueError("coords must have shape (len(elements), 3)")

        self.atomic_numbers = [_get_atomic_number(e) for e in self.elements]
        for z in self.atomic_numbers:
            if z not in AM1_PARAMS:
                raise ValueError(f"no AM1 parameters for element with Z={z}")

        # Per-type parameters (one per atom here; sqm deduplicates types but
        # the integrals only ever reference an atom's own type index).
        self.atoms = [_derive_atom_params(z) for z in self.atomic_numbers]
        self.natoms = len(self.atoms)
        self.orb_tables = _OrbitalTables(self.atoms)

        # Orbital layout (0-based, unlike the Fortran 1-based orb_loc).
        self.orb_loc: list[tuple[int, int]] = []
        self.natomic_orbs: list[int] = []
        n = 0
        for a in self.atoms:
            self.orb_loc.append((n, n + a.natomic_orbs - 1))
            self.natomic_orbs.append(a.natomic_orbs)
            n += a.natomic_orbs
        self.norbs = n

        self.nelectrons = int(round(sum(a.core_chg for a in self.atoms)))
        self.nclosed = self.nelectrons // 2
        if self.nelectrons % 2 != 0:
            raise ValueError("AM1 RHF requires an even number of electrons")

        # betasas / betasap / betapap type-pair sums.
        nt = self.natoms
        self.betasas = np.zeros((nt, nt))
        self.betasap = np.zeros((nt, nt))
        self.betapap = np.zeros((nt, nt))
        for i in range(nt):
            for j in range(nt):
                self.betasas[i, j] = self.atoms[i].betas + self.atoms[j].betas
                self.betasap[i, j] = self.atoms[i].betas + self.atoms[j].betap
                self.betapap[i, j] = self.atoms[i].betap + self.atoms[j].betap

    # ------------------------------------------------------------------
    # Two-centre multipole electron repulsion integrals (qm2_repp)
    # ------------------------------------------------------------------
    @staticmethod
    def _repp(
        a: _AtomParams, b: _AtomParams, r: float, rr2: float, sqrtaee: float
    ) -> tuple[np.ndarray, np.ndarray]:
        ri = np.zeros(22)
        core = np.zeros((10, 2))
        ri[0] = AU_TO_EV * sqrtaee
        core[0, 0] = -b.core_chg * ri[0]
        core[0, 1] = -a.core_chg * ri[0]

        a_s = a.natomic_orbs == 1
        b_s = b.natomic_orbs == 1

        if (not a_s) and b_s:
            # SP atom - H
            da = a.multip[0]
            qa = a.multip[1] * 2.0
            ade = (a.multip[3] + b.multip[2]) ** 2
            aqe = (a.multip[4] + b.multip[2]) ** 2
            arg = np.empty(6)
            arg[0] = (r + da) ** 2 + ade
            arg[1] = (r - da) ** 2 + ade
            arg[2] = (r + qa) ** 2 + aqe
            arg[3] = (r - qa) ** 2 + aqe
            arg[4] = rr2 + aqe
            arg[5] = arg[4] + qa * qa
            sqr = 1.0 / np.sqrt(arg)
            ri[1] = HALF_AU_TO_EV * sqr[0] - HALF_AU_TO_EV * sqr[1]
            ri[2] = ri[0] + FOURTH_AU_TO_EV * sqr[2] + FOURTH_AU_TO_EV * sqr[3] - HALF_AU_TO_EV * sqr[4]
            ri[3] = ri[0] + HALF_AU_TO_EV * sqr[5] - HALF_AU_TO_EV * sqr[4]
            core[0, 0] = -b.core_chg * ri[0]
            core[1, 0] = b.core_chg * ri[1]
            core[2, 0] = -b.core_chg * ri[2]
            core[3, 0] = -b.core_chg * ri[3]
            core[0, 1] = -a.core_chg * ri[0]

        elif a_s and (not b_s):
            # H - SP atom
            db = b.multip[0]
            qb = b.multip[1] * 2.0
            aed = (a.multip[2] + b.multip[3]) ** 2
            aeq = (a.multip[2] + b.multip[4]) ** 2
            arg = np.empty(6)
            arg[0] = (r - db) ** 2 + aed
            arg[1] = (r + db) ** 2 + aed
            arg[2] = (r - qb) ** 2 + aeq
            arg[3] = (r + qb) ** 2 + aeq
            arg[4] = rr2 + aeq
            arg[5] = arg[4] + qb * qb
            sqr = 1.0 / np.sqrt(arg)
            ri[4] = HALF_AU_TO_EV * sqr[0] - HALF_AU_TO_EV * sqr[1]
            ri[10] = ri[0] + FOURTH_AU_TO_EV * sqr[2] + FOURTH_AU_TO_EV * sqr[3] - HALF_AU_TO_EV * sqr[4]
            ri[11] = ri[0] + HALF_AU_TO_EV * sqr[5] - HALF_AU_TO_EV * sqr[4]
            core[0, 0] = -b.core_chg * ri[0]
            core[0, 1] = -a.core_chg * ri[0]
            core[1, 1] = a.core_chg * ri[4]
            core[2, 1] = -a.core_chg * ri[10]
            core[3, 1] = -a.core_chg * ri[11]

        elif (not a_s) and (not b_s):
            # SP atom - SP atom
            da = a.multip[0]
            db = b.multip[0]
            qa = a.multip[1] * 2.0
            qb = b.multip[1] * 2.0
            ade = (a.multip[3] + b.multip[2]) ** 2
            aqe = (a.multip[4] + b.multip[2]) ** 2
            aed = (a.multip[2] + b.multip[3]) ** 2
            aeq = (a.multip[2] + b.multip[4]) ** 2
            axx = (a.multip[3] + b.multip[3]) ** 2
            adq = (a.multip[3] + b.multip[4]) ** 2
            aqd = (a.multip[4] + b.multip[3]) ** 2
            aqq = (a.multip[4] + b.multip[4]) ** 2

            arg = np.empty(71)
            arg[0] = (r + da) ** 2 + ade
            arg[1] = (r - da) ** 2 + ade
            arg[2] = (r - qa) ** 2 + aqe
            arg[3] = (r + qa) ** 2 + aqe
            arg[4] = rr2 + aqe
            arg[5] = arg[4] + qa * qa
            arg[6] = (r - db) ** 2 + aed
            arg[7] = (r + db) ** 2 + aed
            arg[8] = (r - qb) ** 2 + aeq
            arg[9] = (r + qb) ** 2 + aeq
            arg[10] = rr2 + aeq
            arg[11] = arg[10] + qb * qb
            arg[12] = rr2 + axx + (da - db) ** 2
            arg[13] = rr2 + axx + (da + db) ** 2
            arg[14] = (r + da - db) ** 2 + axx
            arg[15] = (r - da + db) ** 2 + axx
            arg[16] = (r - da - db) ** 2 + axx
            arg[17] = (r + da + db) ** 2 + axx
            arg[18] = (r + da) ** 2 + adq
            arg[19] = arg[18] + qb * qb
            arg[20] = (r - da) ** 2 + adq
            arg[21] = arg[20] + qb * qb
            arg[22] = (r - db) ** 2 + aqd
            arg[23] = arg[22] + qa * qa
            arg[24] = (r + db) ** 2 + aqd
            arg[25] = arg[24] + qa * qa
            arg[26] = (r + da - qb) ** 2 + adq
            arg[27] = (r - da - qb) ** 2 + adq
            arg[28] = (r + da + qb) ** 2 + adq
            arg[29] = (r - da + qb) ** 2 + adq
            arg[30] = (r + qa - db) ** 2 + aqd
            arg[31] = (r + qa + db) ** 2 + aqd
            arg[32] = (r - qa - db) ** 2 + aqd
            arg[33] = (r - qa + db) ** 2 + aqd
            arg[34] = rr2 + aqq
            arg[35] = arg[34] + (qa - qb) ** 2
            arg[36] = arg[34] + (qa + qb) ** 2
            arg[37] = arg[34] + qa * qa
            arg[38] = arg[34] + qb * qb
            arg[39] = arg[37] + qb * qb
            arg[40] = (r - qb) ** 2 + aqq
            arg[41] = arg[40] + qa * qa
            arg[42] = (r + qb) ** 2 + aqq
            arg[43] = arg[42] + qa * qa
            arg[44] = (r + qa) ** 2 + aqq
            arg[45] = arg[44] + qb * qb
            arg[46] = (r - qa) ** 2 + aqq
            arg[47] = arg[46] + qb * qb
            arg[48] = (r + qa - qb) ** 2 + aqq
            arg[49] = (r + qa + qb) ** 2 + aqq
            arg[50] = (r - qa - qb) ** 2 + aqq
            arg[51] = (r - qa + qb) ** 2 + aqq
            qa = a.multip[1]
            qb = b.multip[1]
            xxx = (da - qb) ** 2
            yyy = (r - qb) ** 2
            zzz = (da + qb) ** 2
            www = (r + qb) ** 2
            arg[52] = xxx + yyy + adq
            arg[53] = xxx + www + adq
            arg[54] = zzz + yyy + adq
            arg[55] = zzz + www + adq
            xxx = (qa - db) ** 2
            yyy = (qa + db) ** 2
            zzz = (r + qa) ** 2
            www = (r - qa) ** 2
            arg[56] = zzz + xxx + aqd
            arg[57] = www + xxx + aqd
            arg[58] = zzz + yyy + aqd
            arg[59] = www + yyy + aqd
            xxx = (qa - qb) ** 2
            arg[60] = arg[34] + 2.0 * xxx
            yyy = (qa + qb) ** 2
            arg[61] = arg[34] + 2.0 * yyy
            arg[62] = arg[34] + 2.0 * (qa * qa + qb * qb)
            zzz = (r + qa - qb) ** 2
            arg[63] = zzz + xxx + aqq
            arg[64] = zzz + yyy + aqq
            zzz = (r + qa + qb) ** 2
            arg[65] = zzz + xxx + aqq
            arg[66] = zzz + yyy + aqq
            zzz = (r - qa - qb) ** 2
            arg[67] = zzz + xxx + aqq
            arg[68] = zzz + yyy + aqq
            zzz = (r - qa + qb) ** 2
            arg[69] = zzz + xxx + aqq
            arg[70] = zzz + yyy + aqq
            sqr = 1.0 / np.sqrt(arg)

            dze = -HALF_AU_TO_EV * sqr[0] + HALF_AU_TO_EV * sqr[1]
            qzze = FOURTH_AU_TO_EV * sqr[2] + FOURTH_AU_TO_EV * sqr[3] - HALF_AU_TO_EV * sqr[4]
            qxxe = HALF_AU_TO_EV * sqr[5] - HALF_AU_TO_EV * sqr[4]
            edz = -HALF_AU_TO_EV * sqr[6] + HALF_AU_TO_EV * sqr[7]
            eqzz = FOURTH_AU_TO_EV * sqr[8] + FOURTH_AU_TO_EV * sqr[9] - HALF_AU_TO_EV * sqr[10]
            eqxx = HALF_AU_TO_EV * sqr[11] - HALF_AU_TO_EV * sqr[10]
            dxdx = HALF_AU_TO_EV * sqr[12] - HALF_AU_TO_EV * sqr[13]
            dzdz = (
                FOURTH_AU_TO_EV * sqr[14]
                + FOURTH_AU_TO_EV * sqr[15]
                - FOURTH_AU_TO_EV * sqr[16]
                - FOURTH_AU_TO_EV * sqr[17]
            )
            dzqxx = (
                FOURTH_AU_TO_EV * sqr[18]
                - FOURTH_AU_TO_EV * sqr[19]
                - FOURTH_AU_TO_EV * sqr[20]
                + FOURTH_AU_TO_EV * sqr[21]
            )
            qxxdz = (
                FOURTH_AU_TO_EV * sqr[22]
                - FOURTH_AU_TO_EV * sqr[23]
                - FOURTH_AU_TO_EV * sqr[24]
                + FOURTH_AU_TO_EV * sqr[25]
            )
            dzqzz = (
                -EIGHTH_AU_TO_EV * sqr[26]
                + EIGHTH_AU_TO_EV * sqr[27]
                - EIGHTH_AU_TO_EV * sqr[28]
                + EIGHTH_AU_TO_EV * sqr[29]
                - FOURTH_AU_TO_EV * sqr[20]
                + FOURTH_AU_TO_EV * sqr[18]
            )
            qzzdz = (
                -EIGHTH_AU_TO_EV * sqr[30]
                + EIGHTH_AU_TO_EV * sqr[31]
                - EIGHTH_AU_TO_EV * sqr[32]
                + EIGHTH_AU_TO_EV * sqr[33]
                + FOURTH_AU_TO_EV * sqr[22]
                - FOURTH_AU_TO_EV * sqr[24]
            )
            qxxqxx = (
                EIGHTH_AU_TO_EV * sqr[35]
                + EIGHTH_AU_TO_EV * sqr[36]
                - FOURTH_AU_TO_EV * sqr[37]
                - FOURTH_AU_TO_EV * sqr[38]
                + FOURTH_AU_TO_EV * sqr[34]
            )
            qxxqyy = (
                FOURTH_AU_TO_EV * sqr[39]
                - FOURTH_AU_TO_EV * sqr[37]
                - FOURTH_AU_TO_EV * sqr[38]
                + FOURTH_AU_TO_EV * sqr[34]
            )
            qxxqzz = (
                EIGHTH_AU_TO_EV * sqr[41]
                + EIGHTH_AU_TO_EV * sqr[43]
                - EIGHTH_AU_TO_EV * sqr[40]
                - EIGHTH_AU_TO_EV * sqr[42]
                - FOURTH_AU_TO_EV * sqr[37]
                + FOURTH_AU_TO_EV * sqr[34]
            )
            qzzqxx = (
                EIGHTH_AU_TO_EV * sqr[45]
                + EIGHTH_AU_TO_EV * sqr[47]
                - EIGHTH_AU_TO_EV * sqr[44]
                - EIGHTH_AU_TO_EV * sqr[46]
                - FOURTH_AU_TO_EV * sqr[38]
                + FOURTH_AU_TO_EV * sqr[34]
            )
            qzzqzz = (
                SXNTH_AU_TO_EV * sqr[48]
                + SXNTH_AU_TO_EV * sqr[49]
                + SXNTH_AU_TO_EV * sqr[50]
                + SXNTH_AU_TO_EV * sqr[51]
                - EIGHTH_AU_TO_EV * sqr[46]
                - EIGHTH_AU_TO_EV * sqr[44]
                - EIGHTH_AU_TO_EV * sqr[40]
                - EIGHTH_AU_TO_EV * sqr[42]
                + FOURTH_AU_TO_EV * sqr[34]
            )
            dxqxz = (
                -FOURTH_AU_TO_EV * sqr[52]
                + FOURTH_AU_TO_EV * sqr[53]
                + FOURTH_AU_TO_EV * sqr[54]
                - FOURTH_AU_TO_EV * sqr[55]
            )
            qxzdz = (
                -FOURTH_AU_TO_EV * sqr[56]
                + FOURTH_AU_TO_EV * sqr[57]
                + FOURTH_AU_TO_EV * sqr[58]
                - FOURTH_AU_TO_EV * sqr[59]
            )
            qxzqxz = (
                EIGHTH_AU_TO_EV * sqr[63]
                - EIGHTH_AU_TO_EV * sqr[65]
                - EIGHTH_AU_TO_EV * sqr[67]
                + EIGHTH_AU_TO_EV * sqr[69]
                - EIGHTH_AU_TO_EV * sqr[64]
                + EIGHTH_AU_TO_EV * sqr[66]
                + EIGHTH_AU_TO_EV * sqr[68]
                - EIGHTH_AU_TO_EV * sqr[70]
            )
            ri[1] = -dze
            ri[2] = ri[0] + qzze
            ri[3] = ri[0] + qxxe
            ri[4] = -edz
            ri[5] = dzdz
            ri[6] = dxdx
            ri[7] = -edz - qzzdz
            ri[8] = -edz - qxxdz
            ri[9] = -qxzdz
            ri[10] = ri[0] + eqzz
            ri[11] = ri[0] + eqxx
            ri[12] = -dze - dzqzz
            ri[13] = -dze - dzqxx
            ri[14] = -dxqxz
            ri[15] = ri[0] + eqzz + qzze + qzzqzz
            ri[16] = ri[0] + eqzz + qxxe + qxxqzz
            ri[17] = ri[0] + eqxx + qzze + qzzqxx
            ri[18] = ri[0] + eqxx + qxxe + qxxqxx
            ri[19] = qxzqxz
            ri[20] = ri[0] + eqxx + qxxe + qxxqyy
            ri[21] = 0.5 * (qxxqxx - qxxqyy)

            core[0, 0] = -b.core_chg * ri[0]
            core[1, 0] = b.core_chg * ri[1]
            core[2, 0] = -b.core_chg * ri[2]
            core[3, 0] = -b.core_chg * ri[3]
            core[0, 1] = -a.core_chg * ri[0]
            core[1, 1] = a.core_chg * ri[4]
            core[2, 1] = -a.core_chg * ri[10]
            core[3, 1] = -a.core_chg * ri[11]

        return ri, core

    # ------------------------------------------------------------------
    # Local frame rotation and W integrals (qm2_rotate_qmqm)
    # ------------------------------------------------------------------
    def _rotate_qmqm(
        self, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (W, RI, CORE) for the ordered atom pair (i, j)."""
        a = self.atoms[i]
        b = self.atoms[j]
        xi = self.coords[i]
        xj = self.coords[j]

        d = xi - xj
        rij2 = float(np.dot(d, d))
        rr2 = rij2 * A2_TO_BOHRS2
        one_rij = 1.0 / np.sqrt(rij2)
        rij = rij2 * one_rij
        rr = rij * A_TO_BOHRS
        bdd1ij = (a.multip[2] + b.multip[2]) ** 2
        sqrtaee = 1.0 / np.sqrt(rr2 + bdd1ij)
        x = d * one_rij

        ri, core = self._repp(a, b, rr, rr2, sqrtaee)

        if abs(x[2]) > 0.99999999:
            x[2] = _fsign(1.0, x[2])
            y = np.array([0.0, 1.0, 0.0])
            z = np.array([1.0, 0.0, 0.0])
        else:
            z = np.zeros(3)
            z[2] = np.sqrt(1.0 - x[2] * x[2])
            aa = 1.0 / z[2]
            y = np.array([-aa * x[1] * _fsign(1.0, x[0]), abs(aa * x[0]), 0.0])
            z[0] = -aa * x[0] * x[2]
            z[1] = -aa * x[1] * x[2]

        i_s = a.natomic_orbs == 1
        j_s = b.natomic_orbs == 1

        w = np.zeros(100)
        w[0] = ri[0]
        ki = 1

        # Direction-cosine products used by both p-orbital rotation blocks.
        xx11 = xx21 = xx22 = xx31 = xx32 = xx33 = 0.0
        yy11 = yy21 = yy22 = 0.0
        zz11 = zz21 = zz22 = zz31 = zz32 = zz33 = 0.0
        yyzz11 = yyzz21 = yyzz22 = 0.0
        xy11 = xy21 = xy22 = xy31 = xy32 = 0.0
        xz11 = xz21 = xz22 = xz31 = xz32 = xz33 = 0.0
        yz11 = yz21 = yz22 = yz31 = yz32 = 0.0
        if (not i_s) or (not j_s):
            xx11 = x[0] * x[0]
            xx21 = x[1] * x[0]
            xx22 = x[1] * x[1]
            xx31 = x[2] * x[0]
            xx32 = x[2] * x[1]
            xx33 = x[2] * x[2]
            yy11 = y[0] * y[0]
            yy21 = y[1] * y[0]
            yy22 = y[1] * y[1]
            zz11 = z[0] * z[0]
            zz21 = z[1] * z[0]
            zz22 = z[1] * z[1]
            zz31 = z[2] * z[0]
            zz32 = z[2] * z[1]
            zz33 = z[2] * z[2]
            yyzz11 = yy11 + zz11
            yyzz21 = yy21 + zz21
            yyzz22 = yy22 + zz22
            xy11 = 2.0 * x[0] * y[0]
            xy21 = x[0] * y[1] + x[1] * y[0]
            xy22 = 2.0 * x[1] * y[1]
            xy31 = x[2] * y[0]
            xy32 = x[2] * y[1]
            xz11 = 2.0 * x[0] * z[0]
            xz21 = x[0] * z[1] + x[1] * z[0]
            xz22 = 2.0 * x[1] * z[1]
            xz31 = x[0] * z[2] + x[2] * z[0]
            xz32 = x[1] * z[2] + x[2] * z[1]
            xz33 = 2.0 * x[2] * z[2]
            yz11 = 2.0 * y[0] * z[0]
            yz21 = y[0] * z[1] + y[1] * z[0]
            yz22 = 2.0 * y[1] * z[1]
            yz31 = y[0] * z[2]
            yz32 = y[1] * z[2]

        if not j_s:
            w[1] = ri[4] * x[0]
            w[2] = ri[10] * xx11 + ri[11] * yyzz11
            w[3] = ri[4] * x[1]
            w[4] = ri[10] * xx21 + ri[11] * yyzz21
            w[5] = ri[10] * xx22 + ri[11] * yyzz22
            w[6] = ri[4] * x[2]
            w[7] = ri[10] * xx31 + ri[11] * zz31
            w[8] = ri[10] * xx32 + ri[11] * zz32
            w[9] = ri[10] * xx33 + ri[11] * zz33
            ki = 10

        if not i_s:
            if not j_s:
                w[10] = ri[1] * x[0]
                w[11] = ri[5] * xx11 + ri[6] * yyzz11
                w[12] = x[0] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (y[0] * xy11 + z[0] * xz11)
                w[13] = ri[5] * xx21 + ri[6] * yyzz21
                w[14] = x[0] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (y[0] * xy21 + z[0] * xz21)
                w[15] = x[0] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (y[0] * xy22 + z[0] * xz22)
                w[16] = ri[5] * xx31 + ri[6] * zz31
                w[17] = x[0] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (y[0] * xy31 + z[0] * xz31)
                w[18] = x[0] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (y[0] * xy32 + z[0] * xz32)
                w[19] = x[0] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[0] * xz33)
                w[20] = ri[2] * xx11 + ri[3] * yyzz11
                w[21] = x[0] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (y[0] * xy11 + z[0] * xz11)
                w[22] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx11
                    + ri[17] * xx11 * yyzz11
                    + ri[18] * (yy11 * yy11 + zz11 * zz11)
                    + ri[19] * (xy11 * xy11 + xz11 * xz11)
                    + ri[20] * (yy11 * zz11 + zz11 * yy11)
                    + ri[21] * yz11 * yz11
                )
                w[23] = x[1] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (y[1] * xy11 + z[1] * xz11)
                w[24] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx21
                    + ri[17] * xx11 * yyzz21
                    + ri[18] * (yy11 * yy21 + zz11 * zz21)
                    + ri[19] * (xy11 * xy21 + xz11 * xz21)
                    + ri[20] * (yy11 * zz21 + zz11 * yy21)
                    + ri[21] * yz11 * yz21
                )
                w[25] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx22
                    + ri[17] * xx11 * yyzz22
                    + ri[18] * (yy11 * yy22 + zz11 * zz22)
                    + ri[19] * (xy11 * xy22 + xz11 * xz22)
                    + ri[20] * (yy11 * zz22 + zz11 * yy22)
                    + ri[21] * yz11 * yz22
                )
                w[26] = x[2] * (ri[7] * xx11 + ri[8] * yyzz11) + ri[9] * (z[2] * xz11)
                w[27] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx31
                    + (ri[17] * xx11 + ri[18] * zz11 + ri[20] * yy11) * zz31
                    + ri[19] * (xy11 * xy31 + xz11 * xz31)
                    + ri[21] * yz11 * yz31
                )
                w[28] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx32
                    + (ri[17] * xx11 + ri[18] * zz11 + ri[20] * yy11) * zz32
                    + ri[19] * (xy11 * xy32 + xz11 * xz32)
                    + ri[21] * yz11 * yz32
                )
                w[29] = (
                    (ri[15] * xx11 + ri[16] * yyzz11) * xx33
                    + (ri[17] * xx11 + ri[18] * zz11 + ri[20] * yy11) * zz33
                    + ri[19] * xz11 * xz33
                )
                w[30] = ri[1] * x[1]
                w[31] = ri[5] * xx21 + ri[6] * yyzz21
                w[32] = x[1] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (y[1] * xy11 + z[1] * xz11)
                w[33] = ri[5] * xx22 + ri[6] * yyzz22
                w[34] = x[1] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (y[1] * xy21 + z[1] * xz21)
                w[35] = x[1] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (y[1] * xy22 + z[1] * xz22)
                w[36] = ri[5] * xx32 + ri[6] * zz32
                w[37] = x[1] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (y[1] * xy31 + z[1] * xz31)
                w[38] = x[1] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (y[1] * xy32 + z[1] * xz32)
                w[39] = x[1] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[1] * xz33)
                w[40] = ri[2] * xx21 + ri[3] * yyzz21
                w[41] = x[0] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (y[0] * xy21 + z[0] * xz21)
                w[42] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx11
                    + ri[17] * xx21 * yyzz11
                    + ri[18] * (yy21 * yy11 + zz21 * zz11)
                    + ri[19] * (xy21 * xy11 + xz21 * xz11)
                    + ri[20] * (yy21 * zz11 + zz21 * yy11)
                    + ri[21] * yz21 * yz11
                )
                w[43] = x[1] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (y[1] * xy21 + z[1] * xz21)
                w[44] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx21
                    + ri[17] * xx21 * yyzz21
                    + ri[18] * (yy21 * yy21 + zz21 * zz21)
                    + ri[19] * (xy21 * xy21 + xz21 * xz21)
                    + ri[20] * (yy21 * zz21 + zz21 * yy21)
                    + ri[21] * yz21 * yz21
                )
                w[45] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx22
                    + ri[17] * xx21 * yyzz22
                    + ri[18] * (yy21 * yy22 + zz21 * zz22)
                    + ri[19] * (xy21 * xy22 + xz21 * xz22)
                    + ri[20] * (yy21 * zz22 + zz21 * yy22)
                    + ri[21] * yz21 * yz22
                )
                w[46] = x[2] * (ri[7] * xx21 + ri[8] * yyzz21) + ri[9] * (z[2] * xz21)
                w[47] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx31
                    + (ri[17] * xx21 + ri[18] * zz21 + ri[20] * yy21) * zz31
                    + ri[19] * (xy21 * xy31 + xz21 * xz31)
                    + ri[21] * yz21 * yz31
                )
                w[48] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx32
                    + (ri[17] * xx21 + ri[18] * zz21 + ri[20] * yy21) * zz32
                    + ri[19] * (xy21 * xy32 + xz21 * xz32)
                    + ri[21] * yz21 * yz32
                )
                w[49] = (
                    (ri[15] * xx21 + ri[16] * yyzz21) * xx33
                    + (ri[17] * xx21 + ri[18] * zz21 + ri[20] * yy21) * zz33
                    + ri[19] * xz21 * xz33
                )
                w[50] = ri[2] * xx22 + ri[3] * yyzz22
                w[51] = x[0] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (y[0] * xy22 + z[0] * xz22)
                w[52] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx11
                    + ri[17] * xx22 * yyzz11
                    + ri[18] * (yy22 * yy11 + zz22 * zz11)
                    + ri[19] * (xy22 * xy11 + xz22 * xz11)
                    + ri[20] * (yy22 * zz11 + zz22 * yy11)
                    + ri[21] * yz22 * yz11
                )
                w[53] = x[1] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (y[1] * xy22 + z[1] * xz22)
                w[54] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx21
                    + ri[17] * xx22 * yyzz21
                    + ri[18] * (yy22 * yy21 + zz22 * zz21)
                    + ri[19] * (xy22 * xy21 + xz22 * xz21)
                    + ri[20] * (yy22 * zz21 + zz22 * yy21)
                    + ri[21] * yz22 * yz21
                )
                w[55] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx22
                    + ri[17] * xx22 * yyzz22
                    + ri[18] * (yy22 * yy22 + zz22 * zz22)
                    + ri[19] * (xy22 * xy22 + xz22 * xz22)
                    + ri[20] * (yy22 * zz22 + zz22 * yy22)
                    + ri[21] * yz22 * yz22
                )
                w[56] = x[2] * (ri[7] * xx22 + ri[8] * yyzz22) + ri[9] * (z[2] * xz22)
                w[57] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx31
                    + (ri[17] * xx22 + ri[18] * zz22 + ri[20] * yy22) * zz31
                    + ri[19] * (xy22 * xy31 + xz22 * xz31)
                    + ri[21] * yz22 * yz31
                )
                w[58] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx32
                    + (ri[17] * xx22 + ri[18] * zz22 + ri[20] * yy22) * zz32
                    + ri[19] * (xy22 * xy32 + xz22 * xz32)
                    + ri[21] * yz22 * yz32
                )
                w[59] = (
                    (ri[15] * xx22 + ri[16] * yyzz22) * xx33
                    + (ri[17] * xx22 + ri[18] * zz22 + ri[20] * yy22) * zz33
                    + ri[19] * xz22 * xz33
                )
                w[60] = ri[1] * x[2]
                w[61] = ri[5] * xx31 + ri[6] * zz31
                w[62] = x[2] * (ri[12] * xx11 + ri[13] * yyzz11) + ri[14] * (z[2] * xz11)
                w[63] = ri[5] * xx32 + ri[6] * zz32
                w[64] = x[2] * (ri[12] * xx21 + ri[13] * yyzz21) + ri[14] * (z[2] * xz21)
                w[65] = x[2] * (ri[12] * xx22 + ri[13] * yyzz22) + ri[14] * (z[2] * xz22)
                w[66] = ri[5] * xx33 + ri[6] * zz33
                w[67] = x[2] * (ri[12] * xx31 + ri[13] * zz31) + ri[14] * (z[2] * xz31)
                w[68] = x[2] * (ri[12] * xx32 + ri[13] * zz32) + ri[14] * (z[2] * xz32)
                w[69] = x[2] * (ri[12] * xx33 + ri[13] * zz33) + ri[14] * (z[2] * xz33)
                w[70] = ri[2] * xx31 + ri[3] * zz31
                w[71] = x[0] * (ri[7] * xx31 + ri[8] * zz31) + ri[9] * (y[0] * xy31 + z[0] * xz31)
                w[72] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx11
                    + ri[17] * xx31 * yyzz11
                    + ri[18] * zz31 * zz11
                    + ri[19] * (xy31 * xy11 + xz31 * xz11)
                    + ri[20] * zz31 * yy11
                    + ri[21] * yz31 * yz11
                )
                w[73] = x[1] * (ri[7] * xx31 + ri[8] * zz31) + ri[9] * (y[1] * xy31 + z[1] * xz31)
                w[74] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx21
                    + ri[17] * xx31 * yyzz21
                    + ri[18] * zz31 * zz21
                    + ri[19] * (xy31 * xy21 + xz31 * xz21)
                    + ri[20] * zz31 * yy21
                    + ri[21] * yz31 * yz21
                )
                w[75] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx22
                    + ri[17] * xx31 * yyzz22
                    + ri[18] * zz31 * zz22
                    + ri[19] * (xy31 * xy22 + xz31 * xz22)
                    + ri[20] * zz31 * yy22
                    + ri[21] * yz31 * yz22
                )
                w[76] = x[2] * (ri[7] * xx31 + ri[8] * zz31) + ri[9] * (z[2] * xz31)
                w[77] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx31
                    + (ri[17] * xx31 + ri[18] * zz31) * zz31
                    + ri[19] * (xy31 * xy31 + xz31 * xz31)
                    + ri[21] * yz31 * yz31
                )
                w[78] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx32
                    + (ri[17] * xx31 + ri[18] * zz31) * zz32
                    + ri[19] * (xy31 * xy32 + xz31 * xz32)
                    + ri[21] * yz31 * yz32
                )
                w[79] = (
                    (ri[15] * xx31 + ri[16] * zz31) * xx33
                    + (ri[17] * xx31 + ri[18] * zz31) * zz33
                    + ri[19] * xz31 * xz33
                )
                w[80] = ri[2] * xx32 + ri[3] * zz32
                w[81] = x[0] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (y[0] * xy32 + z[0] * xz32)
                w[82] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx11
                    + ri[17] * xx32 * yyzz11
                    + ri[18] * zz32 * zz11
                    + ri[19] * (xy32 * xy11 + xz32 * xz11)
                    + ri[20] * zz32 * yy11
                    + ri[21] * yz32 * yz11
                )
                w[83] = x[1] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (y[1] * xy32 + z[1] * xz32)
                w[84] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx21
                    + ri[17] * xx32 * yyzz21
                    + ri[18] * zz32 * zz21
                    + ri[19] * (xy32 * xy21 + xz32 * xz21)
                    + ri[20] * zz32 * yy21
                    + ri[21] * yz32 * yz21
                )
                w[85] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx22
                    + ri[17] * xx32 * yyzz22
                    + ri[18] * zz32 * zz22
                    + ri[19] * (xy32 * xy22 + xz32 * xz22)
                    + ri[20] * zz32 * yy22
                    + ri[21] * yz32 * yz22
                )
                w[86] = x[2] * (ri[7] * xx32 + ri[8] * zz32) + ri[9] * (z[2] * xz32)
                w[87] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx31
                    + (ri[17] * xx32 + ri[18] * zz32) * zz31
                    + ri[19] * (xy32 * xy31 + xz32 * xz31)
                    + ri[21] * yz32 * yz31
                )
                w[88] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx32
                    + (ri[17] * xx32 + ri[18] * zz32) * zz32
                    + ri[19] * (xy32 * xy32 + xz32 * xz32)
                    + ri[21] * yz32 * yz32
                )
                w[89] = (
                    (ri[15] * xx32 + ri[16] * zz32) * xx33
                    + (ri[17] * xx32 + ri[18] * zz32) * zz33
                    + ri[19] * xz32 * xz33
                )
                w[90] = ri[2] * xx33 + ri[3] * zz33
                w[91] = x[0] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[0] * xz33)
                w[92] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx11
                    + ri[17] * xx33 * yyzz11
                    + ri[18] * zz33 * zz11
                    + ri[19] * xz33 * xz11
                    + ri[20] * zz33 * yy11
                )
                w[93] = x[1] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[1] * xz33)
                w[94] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx21
                    + ri[17] * xx33 * yyzz21
                    + ri[18] * zz33 * zz21
                    + ri[19] * xz33 * xz21
                    + ri[20] * zz33 * yy21
                )
                w[95] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx22
                    + ri[17] * xx33 * yyzz22
                    + ri[18] * zz33 * zz22
                    + ri[19] * xz33 * xz22
                    + ri[20] * zz33 * yy22
                )
                w[96] = x[2] * (ri[7] * xx33 + ri[8] * zz33) + ri[9] * (z[2] * xz33)
                w[97] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx31
                    + (ri[17] * xx33 + ri[18] * zz33) * zz31
                    + ri[19] * xz33 * xz31
                )
                w[98] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx32
                    + (ri[17] * xx33 + ri[18] * zz33) * zz32
                    + ri[19] * xz33 * xz32
                )
                w[99] = (
                    (ri[15] * xx33 + ri[16] * zz33) * xx33
                    + (ri[17] * xx33 + ri[18] * zz33) * zz33
                    + ri[19] * xz33 * xz33
                )
                ki = 100
            else:
                w[1] = ri[1] * x[0]
                w[2] = ri[2] * xx11 + ri[3] * yyzz11
                w[3] = ri[1] * x[1]
                w[4] = ri[2] * xx21 + ri[3] * yyzz21
                w[5] = ri[2] * xx22 + ri[3] * yyzz22
                w[6] = ri[1] * x[2]
                w[7] = ri[2] * xx31 + ri[3] * zz31
                w[8] = ri[2] * xx32 + ri[3] * zz32
                w[9] = ri[2] * xx33 + ri[3] * zz33
                ki = 10

        return w[:ki], ri, core

    # ------------------------------------------------------------------
    # Core-core repulsion (qm2_core_core_repulsion, AM1 branch)
    # ------------------------------------------------------------------
    def _core_core_repulsion(self, i: int, j: int, rij: float, one_rij: float, ri1: float) -> float:
        a = self.atoms[i]
        b = self.atoms[j]
        iatnum = a.z
        jatnum = b.z
        ijmin = min(iatnum, jatnum)
        ijmax = max(iatnum, jatnum)

        giijj = ri1
        alpi = np.exp(-a.alp * rij)
        alpj = alpi if i == j else np.exp(-b.alp * rij)
        scale = 1.0 + alpi + alpj
        if ijmin == 1 and (ijmax == 7 or ijmax == 8):
            if iatnum == 1:
                scale = scale + (rij - 1.0) * alpj
            else:
                scale = scale + (rij - 1.0) * alpi

        ccij = a.core_chg * b.core_chg
        enuc = ccij * giijj * scale

        # AM1 Gaussian corrections
        anam1 = 0.0
        for k in range(a.num_fn):
            tmp = rij - a.fn3[k]
            tmp = a.fn2[k] * tmp * tmp
            if tmp < EXPONENTIAL_CUTOFF:
                anam1 += a.fn1[k] * np.exp(-tmp)
        if i == j:
            anam1 = anam1 + anam1
        else:
            for k in range(b.num_fn):
                tmp = rij - b.fn3[k]
                tmp = b.fn2[k] * tmp * tmp
                if tmp < EXPONENTIAL_CUTOFF:
                    anam1 += b.fn1[k] * np.exp(-tmp)
        anam1 = ccij * one_rij * anam1
        enuc += anam1
        return enuc

    # ------------------------------------------------------------------
    # Rotation matrix for core-electron attraction (rotation.F90)
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_matrix(i: int, j: int, coords: np.ndarray) -> np.ndarray:
        """Return the 3x3 P matrix (molecular components of local axes)."""
        xij = coords[j] - coords[i]
        x11, x22, x33 = xij
        b = x11 * x11 + x22 * x22
        r = np.sqrt(b + x33 * x33)
        sqb = np.sqrt(b)
        sb = sqb / r
        if sb > 1.0e-7:
            ca = x11 / sqb
            sa = x22 / sqb
            cb = x33 / r
        else:
            sa = 0.0
            sb = 0.0
            if x33 < 0.0:
                ca = -1.0
                cb = -1.0
            elif x33 > 0.0:
                ca = 1.0
                cb = 1.0
            else:
                ca = 0.0
                cb = 0.0
        # Pmat[K, m]: molecular m-component of local axis K (K=0 sigma).
        pmat = np.zeros((3, 3))
        pmat[0, 0] = ca * sb
        pmat[0, 1] = sa * sb
        pmat[0, 2] = cb
        pmat[1, 0] = ca * cb
        pmat[1, 1] = sa * cb
        pmat[1, 2] = -sb
        pmat[2, 0] = -sa
        pmat[2, 1] = ca
        pmat[2, 2] = 0.0
        return pmat

    def _add_core_attraction(
        self, h: np.ndarray, i: int, j: int, core: np.ndarray, pmat: np.ndarray
    ) -> None:
        """Add the intra-atomic core-electron attraction (RotateCore)."""
        for kk, (atom, core_col) in enumerate(((i, 0), (j, 1))):
            a = self.atoms[atom]
            s0 = self.orb_loc[atom][0]
            c1 = core[0, core_col]
            h[s0, s0] += c1
            if a.natomic_orbs == 1:
                continue
            # P-S mixing
            for m in range(3):
                h[s0 + 1 + m, s0] += core[1, core_col] * pmat[0, m]
                h[s0, s0 + 1 + m] += core[1, core_col] * pmat[0, m]
            # P-P mixing: local axes sigma (K=0), pi1 (K=1), pi2 (K=2)
            pp = np.zeros((3, 3))
            for m in range(3):
                for n in range(3):
                    val = core[2, core_col] * pmat[0, m] * pmat[0, n]
                    val += core[3, core_col] * (pmat[1, m] * pmat[1, n] + pmat[2, m] * pmat[2, n])
                    pp[m, n] = val
            for m in range(3):
                for n in range(3):
                    h[s0 + 1 + m, s0 + 1 + n] += pp[m, n]

    # ------------------------------------------------------------------
    # One-electron two-centre matrix (qm2_h1elec)
    # ------------------------------------------------------------------
    def _h1elec(self, i: int, j: int, r2_in_au: float) -> np.ndarray:
        a = self.atoms[i]
        b = self.atoms[j]
        t = self.orb_tables
        sh = np.zeros((4, 4))

        vec = (self.coords[i] - self.coords[j]) * A_TO_BOHRS

        # S-S
        for k in range(6):
            for l in range(6):
                adbr2 = t.sxs_over_sas[k, l, i, j] * r2_in_au
                if adbr2 < EXPONENTIAL_CUTOFF:
                    sh[0, 0] += t.ss_eqn[k, l, i, j] * np.exp(-adbr2)
        sh[0, 0] *= 0.5 * self.betasas[i, j]

        # I:S - J:P
        if b.natomic_orbs != 1:
            for k in range(6):
                for l in range(6):
                    adbr2 = t.sxp_over_sap[k, l, i, j] * r2_in_au
                    if adbr2 < EXPONENTIAL_CUTOFF:
                        s = t.sp_ovlp[k, l, i, j] * np.exp(-adbr2)
                        sh[0, 1] += s * vec[0]
                        sh[0, 2] += s * vec[1]
                        sh[0, 3] += s * vec[2]
            bij = 0.5 * self.betasap[i, j]
            sh[0, 1] *= bij
            sh[0, 2] *= bij
            sh[0, 3] *= bij

        # I:P - J:S
        if a.natomic_orbs != 1:
            for k in range(6):
                for l in range(6):
                    adbr2 = t.sxp_over_sap[l, k, j, i] * r2_in_au
                    if adbr2 < EXPONENTIAL_CUTOFF:
                        s = -t.sp_ovlp[l, k, j, i] * np.exp(-adbr2)
                        sh[1, 0] += s * vec[0]
                        sh[2, 0] += s * vec[1]
                        sh[3, 0] += s * vec[2]
            bij = 0.5 * self.betasap[j, i]
            sh[1, 0] *= bij
            sh[2, 0] *= bij
            sh[3, 0] *= bij

        # I:P - J:P
        if a.natomic_orbs != 1 and b.natomic_orbs != 1:
            bij = 0.5 * self.betapap[i, j]
            for ii in range(3):
                for jj in range(3):
                    tomb = vec[ii] * vec[jj]
                    for k in range(6):
                        for l in range(6):
                            adbr2 = t.pxp_over_pap[k, l, i, j] * r2_in_au
                            if adbr2 < EXPONENTIAL_CUTOFF:
                                if ii == jj:
                                    s = np.exp(-adbr2) * (
                                        t.pp_ovlp_ieqj1[k, l, i, j] * tomb
                                        + t.pp_ovlp_ieqj2[k, l, i, j]
                                    )
                                else:
                                    s = np.exp(-adbr2) * tomb * t.pp_ovlp_inj[k, l, i, j]
                                sh[ii + 1, jj + 1] += s
                    sh[ii + 1, jj + 1] *= bij
        return sh

    # ------------------------------------------------------------------
    # Full one-electron core Hamiltonian and W integrals
    # ------------------------------------------------------------------
    def _build_hcore(self) -> tuple[np.ndarray, list[list[np.ndarray | None]], float]:
        n = self.norbs
        h = np.zeros((n, n))
        enuclr = 0.0
        w_pairs: list[list[np.ndarray | None]] = [
            [None] * self.natoms for _ in range(self.natoms)
        ]

        # Diagonal one-electron integrals.
        for i, a in enumerate(self.atoms):
            s0 = self.orb_loc[i][0]
            h[s0, s0] = a.uss
            if a.natomic_orbs >= 4:
                for m in range(1, 4):
                    h[s0 + m, s0 + m] = a.upp

        for i in range(1, self.natoms):
            for j in range(i):
                ai = self.atoms[i]
                aj = self.atoms[j]
                ia = self.orb_loc[i][0]
                ja = self.orb_loc[j][0]
                d = self.coords[i] - self.coords[j]
                r2 = float(np.dot(d, d))
                rij = np.sqrt(r2)
                r2_in_au = r2 * A2_TO_BOHRS2
                one_rij = 1.0 / rij

                # One-electron 2-centre matrix.
                if r2_in_au < OVERLAP_CUTOFF:
                    sh = self._h1elec(i, j, r2_in_au)
                    ni = ai.natomic_orbs
                    nj = aj.natomic_orbs
                    for i1 in range(ni):
                        for j1 in range(nj):
                            h[ia + i1, ja + j1] += sh[i1, j1]
                            h[ja + j1, ia + i1] += sh[i1, j1]

                # Two-electron integrals and core attraction.
                w, ri, core = self._rotate_qmqm(i, j)
                npair_i = ai.natomic_orbs * (ai.natomic_orbs + 1) // 2
                npair_j = aj.natomic_orbs * (aj.natomic_orbs + 1) // 2
                wmat = np.array(w).reshape(npair_i, npair_j)
                w_pairs[i][j] = wmat
                w_pairs[j][i] = wmat.T

                pmat = self._rotation_matrix(i, j, self.coords)
                self._add_core_attraction(h, i, j, core, pmat)

                enuc = self._core_core_repulsion(i, j, rij, one_rij, ri[0])
                enuclr += enuc

        return h, w_pairs, enuclr

    # ------------------------------------------------------------------
    # Fock matrix assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_pair(a1: int, a2: int) -> int:
        if a1 >= a2:
            return _PAIR_INDEX[a1, a2]
        return _PAIR_INDEX[a2, a1]

    def _fock1(self, f: np.ndarray, p: np.ndarray) -> None:
        """One-centre two-electron contributions (qm2_fock1)."""
        for ii, a in enumerate(self.atoms):
            s0 = self.orb_loc[ii][0]
            gss = a.onec2elec[0]
            pss = p[s0, s0]
            if a.natomic_orbs == 1:
                f[s0, s0] += pss * gss
                continue
            gsp = a.onec2elec[1]
            gpp = a.onec2elec[2]
            gp2 = a.onec2elec[3]
            hsp = a.onec2elec[4]
            gsp_hsp = gsp - hsp
            hsp_gsp = 6.0 * hsp - gsp
            gp2_gpp = gp2 - 0.5 * gpp
            gpp_gp2 = 1.5 * gpp - gp2

            p0, p1, p2 = s0 + 1, s0 + 2, s0 + 3
            ppop = p[p0, p0] + p[p1, p1] + p[p2, p2]
            f[s0, s0] += pss * gss + ppop * gsp_hsp
            for pp in (p0, p1, p2):
                p_pp = p[pp, pp]
                f[pp, pp] += gsp_hsp * pss + p_pp * gpp + (ppop - p_pp) * gp2_gpp
                f[s0, pp] += 0.5 * p[s0, pp] * hsp_gsp
                f[pp, s0] += 0.5 * p[s0, pp] * hsp_gsp
            for pp in (p0, p1, p2):
                for qq in (p0, p1, p2):
                    if qq > pp:
                        val = p[pp, qq] * gpp_gp2
                        f[pp, qq] += val
                        f[qq, pp] += val

    def _fock2(
        self, f: np.ndarray, p: np.ndarray, w_pairs: list[list[np.ndarray | None]]
    ) -> None:
        """Two-centre two-electron contributions (qm2_fock2 / W2Fock_atompair).

        Vectorised: the pair-wise Coulomb and exchange contractions are
        computed with numpy einsum instead of Python loops.
        """
        for i in range(1, self.natoms):
            ai = self.atoms[i]
            ia = self.orb_loc[i][0]
            ni = ai.natomic_orbs
            for j in range(i):
                aj = self.atoms[j]
                ja = self.orb_loc[j][0]
                nj = aj.natomic_orbs
                wmat = w_pairs[i][j]
                assert wmat is not None

                # W4[a1,a2,b1,b2] from the packed pair matrix.
                pidx_i = _PAIR_INDEX[:ni, :ni]
                pidx_j = _PAIR_INDEX[:nj, :nj]
                w4 = wmat[pidx_i[:, :, None, None], pidx_j[None, None, :, :]]

                pi = p[ia : ia + ni, ia : ia + ni]
                pj = p[ja : ja + nj, ja : ja + nj]
                pij = p[ia : ia + ni, ja : ja + nj]

                # Coulomb on atom i's block (triangle sum w/ (2-delta) factor
                # equals a full-grid contraction over P without the factor).
                coul_i = np.einsum("abxy,xy->ab", w4, pj)
                # Coulomb on atom j's block.
                coul_j = np.einsum("abxy,ab->xy", w4, pi)
                # Exchange.
                exch = np.einsum("axby,xy->ab", w4, pij)

                f[ia : ia + ni, ia : ia + ni] += coul_i
                f[ja : ja + nj, ja : ja + nj] += coul_j
                f[ia : ia + ni, ja : ja + nj] -= 0.5 * exch
                f[ja : ja + nj, ia : ia + ni] -= 0.5 * exch.T

    def _build_fock(
        self, p: np.ndarray, h: np.ndarray, w_pairs: list[list[np.ndarray | None]]
    ) -> np.ndarray:
        f = h.copy()
        self._fock1(f, p)
        self._fock2(f, p, w_pairs)
        return f

    @staticmethod
    def _helect(p: np.ndarray, h: np.ndarray, f: np.ndarray) -> float:
        """Electronic energy (qm2_HELECT), in eV."""
        norbs = p.shape[0]
        energy = 0.0
        for i in range(norbs):
            energy += 0.5 * p[i, i] * (h[i, i] + f[i, i])
            for j in range(i):
                energy += p[i, j] * (h[i, j] + f[i, j])
        return energy

    # ------------------------------------------------------------------
    # Density matrix and SCF
    # ------------------------------------------------------------------
    def _initial_density(self) -> np.ndarray:
        p = np.zeros((self.norbs, self.norbs))
        for i, a in enumerate(self.atoms):
            s0 = self.orb_loc[i][0]
            k = 4
            if a.core_chg <= 2 and a.natomic_orbs == 1:
                k = 1
            pdiag = a.core_chg / k
            for m in range(k):
                p[s0 + m, s0 + m] = pdiag
        return p

    def _density_from_eigvecs(self, eigvecs: np.ndarray) -> np.ndarray:
        occ = eigvecs[:, : self.nclosed]
        return 2.0 * occ @ occ.T

    def _cnvg(
        self, pnew: np.ndarray, old_p: np.ndarray, old2_diag: np.ndarray, it: int
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Density matrix extrapolation (qm2_cnvg)."""
        norbs = self.norbs
        damp = 0.05 if it > 4 else 1.0e10
        density_diff = 0.0
        faca = 0.0
        facb = 0.0
        fac = 0.0
        current_den_sum = 0.0

        if (it - 1) % 3 != 0:
            for i in range(norbs):
                den = pnew[i, i]
                old = old_p[i, i]
                old2_diag[i] = old
                density_diff = max(abs(den - old), density_diff)
                old_p[i, i] = den
                current_den_sum += den
        else:
            for i in range(norbs):
                den = pnew[i, i]
                old = old_p[i, i]
                current_den_sum += den
                diff = abs(den - old)
                density_diff = max(diff, density_diff)
                faca += diff * diff
                facb_temp = den - 2.0 * old + old2_diag[i]
                facb += facb_temp * facb_temp
                old2_diag[i] = old
                old_p[i, i] = den
        if facb > 0.0 and faca < 100.0 * facb:
            fac = np.sqrt(faca / facb)

        # Interpolation over the packed lower triangle.
        sum2 = 0.0
        # diagonal i = 0
        old = old_p[0, 0]
        o2 = old2_diag[0]
        if abs(old - o2) > damp:
            old = o2 + np.copysign(damp, old - o2)
        else:
            old = old + fac * (old - o2)
        old = min(2.0, max(old, 0.0))
        sum2 += old
        old_p[0, 0] = old
        pnew[0, 0] = old

        for i in range(1, norbs):
            for j in range(i):
                den = pnew[i, j]
                old = den + fac * (den - old_p[i, j])
                old_p[i, j] = old
                old_p[j, i] = old
                pnew[i, j] = old
                pnew[j, i] = old
            o2 = old2_diag[i]
            old = old_p[i, i]
            if abs(old - o2) > damp:
                old = o2 + np.copysign(damp, old - o2)
            else:
                old = old + fac * (old - o2)
            old = min(2.0, max(old, 0.0))
            old_p[i, i] = old
            pnew[i, i] = old
            sum2 += old

        # Re-normalisation (qm2_cnvg tail).
        sum0 = current_den_sum
        while True:
            sum3 = current_den_sum / sum2 if sum2 > 1.0e-3 else 0.0
            current_den_sum = sum0
            if sum2 <= 1.0e-3 or abs(sum3 - 1.0) <= 1.0e-5:
                break
            sum2 = 0.0
            for i in range(norbs):
                val = max(old_p[i, i] * sum3 + 1.0e-20, 0.0)
                if val > 2.0:
                    val = 2.0
                    current_den_sum -= 2.0
                else:
                    sum2 += val
                old_p[i, i] = val
                pnew[i, i] = val
        return density_diff, pnew, old_p, old2_diag

    def single_point(
        self, max_iter: int = 1000, scfconv: float = 1.0e-8, verbose: bool = False
    ) -> dict[str, object]:
        """Run the AM1 RHF SCF and return energy, charges and density.

        ``scfconv`` is the energy convergence threshold in kcal/mol, matching
        the sqm default (1.0e-8 kcal/mol).
        """
        h, w_pairs, enuclr = self._build_hcore()
        p = self._initial_density()
        old_p = p.copy()
        old2_diag = np.zeros(self.norbs)
        eigvecs: np.ndarray | None = None
        eold = 0.0
        scf_energy = 0.0
        energy_diff = np.inf
        density_diff = np.inf
        density_conv = 0.05 * np.sqrt(scfconv)
        converged = False
        f = h.copy()

        for it in range(1, max_iter + 1):
            if it > 1:
                assert eigvecs is not None
                pnew = self._density_from_eigvecs(eigvecs)
                density_diff, p, old_p, old2_diag = self._cnvg(pnew, old_p, old2_diag, it)

            f = self._build_fock(p, h, w_pairs)
            elec_eng = self._helect(p, h, f)
            scf_energy = elec_eng * EV_TO_KCAL
            energy_diff = scf_energy - eold

            if it > 2:
                if abs(energy_diff) < scfconv and (
                    density_diff < density_conv or density_diff == 0.0
                ):
                    converged = True
            if verbose:
                print(
                    f"AM1 SCF iter {it:4d}  E={scf_energy:18.10f} kcal/mol  "
                    f"dE={energy_diff: .3e}  dP={density_diff: .3e}"
                )
            if converged:
                break

            eold = scf_energy
            f_sym = 0.5 * (f + f.T)
            _, eigvecs = np.linalg.eigh(f_sym)

        # Mulliken charges from the final density matrix.
        charges = np.zeros(self.natoms)
        for i, a in enumerate(self.atoms):
            s0, s1 = self.orb_loc[i]
            charges[i] = a.core_chg - float(np.sum(np.diag(p)[s0 : s1 + 1]))

        total_energy_ev = elec_eng + enuclr
        heat_of_form = sum(_ELEMENT_DATA[a.z]["heat_of_form"] for a in self.atoms)
        return {
            "energy_eV": float(total_energy_ev),
            "electronic_energy_eV": float(elec_eng),
            "nuclear_energy_eV": float(enuclr),
            "heat_of_formation_kcal": float(heat_of_form),
            "total_energy_kcal": float(total_energy_ev * EV_TO_KCAL + heat_of_form),
            "mulliken_charges": charges,
            "density": p,
            "norbs": self.norbs,
            "converged": converged,
            "n_iterations": it,
            "energy_diff_kcal": float(energy_diff),
            "density_diff": float(density_diff),
        }

    def optimize(
        self,
        *,
        grms_tol: float = 0.0005,
        max_evals: int = 4000,
        fd_step: float = 2.0e-3,
        verbose: bool = False,
    ) -> dict[str, object]:
        """Optimize the geometry with a quasi-Newton minimizer.

        ``grms_tol`` is the RMS gradient tolerance in kcal/mol/Angstrom,
        matching the sqm default (0.0005).  Numerical gradients with a fixed
        finite-difference step keep the optimisation deterministic and require
        no analytic-derivative port.  Returns the single_point() dict for the
        optimized geometry.
        """
        from scipy.optimize import minimize  # type: ignore[import-untyped]

        coords = self.coords.copy()
        natoms = self.natoms

        def energy_at(x: np.ndarray) -> float:
            self.coords = x.reshape(-1, 3)
            return float(np.asarray(self.single_point()["energy_eV"], dtype=float))  # type: ignore[call-overload]

        def energy_flat(x: np.ndarray) -> float:
            return energy_at(x)

        def grad_flat(x: np.ndarray) -> np.ndarray:
            g = np.zeros_like(x)
            for k in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[k] += fd_step
                xm[k] -= fd_step
                ep = energy_at(xp)
                em = energy_at(xm)
                g[k] = (ep - em) / (2.0 * fd_step)
            return g

        # gtol is in eV/Angstrom; convert from kcal/mol/Angstrom.
        gtol = grms_tol * KCAL_TO_EV * np.sqrt(natoms * 3.0)
        res = minimize(
            energy_flat,
            coords.ravel(),
            method="BFGS",
            jac=grad_flat,
            options={"gtol": gtol, "maxiter": max_evals, "norm": np.inf},
        )
        self.coords = res.x.reshape(-1, 3)
        result = dict(self.single_point())  # type: ignore[dict-item]
        result["optimized"] = True
        result["geometry_converged"] = bool(res.success)
        result["n_geometry_evals"] = int(res.nfev)
        result["rms_gradient_kcal_ang"] = float(np.linalg.norm(grad_flat(res.x)) / np.sqrt(natoms * 3.0) / KCAL_TO_EV)
        result["optimized_coords"] = self.coords.copy()
        if verbose:
            print(f"AM1 geometry: success={res.success} nfev={res.nfev} energy={res.fun:.6f} eV")
        return result
