"""Tests for the pure-Python AM1 semi-empirical RHF SCF (sqm port)."""

from __future__ import annotations

import numpy as np
import pytest

from lft.charges.semiempirical.am1 import AM1, AM1_PARAMS

# Methyl benzoate (C8 H8 O2), input geometry in Angstrom (not optimized).
METHYL_BENZOATE_ELEMENTS = [
    "C", "C", "C", "C", "C", "C",
    "H", "H", "H", "H", "H",
    "C", "O", "O", "C", "H", "H", "H",
]
METHYL_BENZOATE_COORDS = np.array(
    [
        [-2.9831, -0.2774, 0.0],
        [-2.1235, -1.3707, 0.0],
        [-0.7489, -1.1754, 0.0],
        [-0.2320, 0.1199, 0.0],
        [-1.0958, 1.2143, 0.0],
        [-2.4685, 1.0151, 0.0],
        [-4.0563, -0.4333, 0.0],
        [-2.5257, -2.3773, 0.0],
        [-0.0729, -2.0209, 0.0],
        [-0.6737, 2.2122, 0.0],
        [-3.1391, 1.8668, 0.0],
        [1.2335, 0.3914, 0.0],
        [1.7213, 1.4936, 0.0],
        [1.9654, -0.7318, 0.0],
        [3.3809, -0.5492, 0.0],
        [3.8068, -1.5503, 0.0],
        [3.6956, 0.0004, -0.8887],
        [3.6956, 0.0004, 0.8887],
    ],
    dtype=float,
)

# Reference Mulliken charges from an sqm AM1 run (after geometry optimisation).
REFERENCE_CHARGES = {
    "C1": -0.096, "C2": -0.149, "C3": -0.070, "C4": -0.115, "C5": -0.073,
    "C6": -0.147, "H7": 0.136, "H8": 0.139, "H9": 0.155, "H10": 0.157,
    "H11": 0.140, "C12": 0.345, "O13": -0.355, "O14": -0.278, "C15": -0.062,
    "H16": 0.101, "H17": 0.086, "H18": 0.086,
}


def test_am1_parameters_available() -> None:
    assert set(AM1_PARAMS) >= {1, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 30, 32, 35, 53, 80}
    c = AM1_PARAMS[6]
    assert c["betas"] == pytest.approx(-15.715783)
    assert c["GSS"] == pytest.approx(12.23)
    h = AM1_PARAMS[1]
    assert h["p_orb_exp"] == 0.0
    assert h["NUM_FN"] == 3


def test_methyl_benzoate_scf_converges() -> None:
    calc = AM1(METHYL_BENZOATE_ELEMENTS, METHYL_BENZOATE_COORDS)
    result = calc.single_point()
    assert result["converged"] is True
    assert result["norbs"] == 48
    assert result["n_iterations"] < 100
    # Total (electronic + nuclear) energy is large and negative (in eV).
    assert result["energy_eV"] < -1000.0


def test_methyl_benzoate_charges_match_reference() -> None:
    calc = AM1(METHYL_BENZOATE_ELEMENTS, METHYL_BENZOATE_COORDS)
    result = calc.single_point()
    charges = result["mulliken_charges"]
    ref = np.array([REFERENCE_CHARGES[f"C{i}"] for i in range(1, 7)]
                   + [REFERENCE_CHARGES[f"H{i}"] for i in range(7, 12)]
                   + [REFERENCE_CHARGES["C12"], REFERENCE_CHARGES["O13"],
                      REFERENCE_CHARGES["O14"], REFERENCE_CHARGES["C15"]]
                   + [REFERENCE_CHARGES[f"H{i}"] for i in range(16, 19)])
    # Single point at the un-optimised geometry; charges should be close to the
    # optimised-geometry reference (within ~0.02 e).
    assert np.abs(charges - ref).max() < 0.05
    # Chemically sensible: carbonyl C positive, both oxygens strongly negative.
    assert charges[11] > 0.2
    assert charges[12] < -0.2
    assert charges[13] < -0.2
    # Charge neutrality.
    assert abs(charges.sum()) < 1.0e-6


def test_water_charges() -> None:
    elements = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]])
    calc = AM1(elements, coords)
    result = calc.single_point()
    assert result["converged"] is True
    q = result["mulliken_charges"]
    assert q[0] < -0.2  # oxygen negative
    assert q[1] > 0.0 and q[2] > 0.0  # hydrogens positive
    assert q[1] == pytest.approx(q[2], abs=1.0e-6)  # symmetric molecule


def test_diatomic_along_z_special_frame() -> None:
    # Bond along the z axis exercises the special-case local frame.
    elements = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    calc = AM1(elements, coords)
    result = calc.single_point()
    assert result["converged"] is True
    q = result["mulliken_charges"]
    assert q[0] == pytest.approx(q[1], abs=1.0e-6)
    assert abs(q.sum()) < 1.0e-6


def test_unsupported_element_raises() -> None:
    from lft.charges.semiempirical.am1 import _get_atomic_number

    with pytest.raises(ValueError):
        AM1(["U"], np.zeros((1, 3)))
    with pytest.raises(ValueError):
        _get_atomic_number("Xx")