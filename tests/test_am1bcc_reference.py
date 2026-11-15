"""AM1-BCC pure-Python charge generation vs the AmberTools/acpype reference.

The reference values are the frozen output of an acpype AM1-BCC run on methyl
benzoate (see docs/PROJECT_STATUS.md).  The reference raw Mulliken
charges are at sqm's optimized geometry; to keep the test fast and
deterministic we evaluate AM1 at that frozen geometry (no re-optimization).
"""

from __future__ import annotations

import numpy as np
import pytest

from lft.charges.semiempirical.am1 import AM1
from lft.charges.semiempirical.bcc import AM1BCC
from lft.forcefields.gaff_typer import assign_gaff_types
from lft.io.loaders import load_structure

REF_GEOMETRY = {
    1: (-2.995, -0.281, -0.001),
    2: (-2.127, -1.374, -0.001),
    3: (-0.748, -1.172, -0.000),
    4: (-0.236, 0.131, 0.000),
    5: (-1.108, 1.229, 0.000),
    6: (-2.485, 1.018, -0.000),
    7: (-4.083, -0.444, -0.001),
    8: (-2.530, -2.397, -0.001),
    9: (-0.056, -2.030, -0.000),
    10: (-0.692, 2.250, 0.001),
    11: (-3.170, 1.879, 0.000),
    12: (1.210, 0.387, 0.001),
    13: (1.791, 1.477, 0.001),
    14: (1.986, -0.745, -0.000),
    15: (3.400, -0.546, 0.000),
    16: (3.811, -1.586, -0.001),
    17: (3.706, 0.017, -0.914),
    18: (3.706, 0.015, 0.915),
}

# Reference AM1-BCC charges (ANTECHAMBER_AM1BCC.AC / bcc_gaff.mol2).
REF_BCC_CHARGES = [
    -0.0960, -0.1480, -0.0715, -0.1356, -0.0715, -0.1480,
    0.1360, 0.1395, 0.1560, 0.1560, 0.1395,
    0.6447, -0.5440, -0.4399, 0.1277, 0.0517, 0.0517, 0.0517,
]

# Reference BCC atom types (from ANTECHAMBER_AM1BCC.AC).
REF_BCC_TYPES = ["16"] * 6 + ["91"] * 5 + ["14", "32", "31", "11", "91", "91", "91"]

# Reference bondtype -j full output for methyl benzoate.
REF_BOND_TYPES = [7, 8, 1, 8, 1, 7, 1, 8, 1, 7, 1, 1, 2, 1, 1, 1, 1, 1]


@pytest.fixture(scope="module")
def methyl_benzoate() -> "object":
    return load_structure("references/sobtop/examples/Methyl_benzoate/Methyl_benzoate.mol2")


def test_am1bcc_bcc_types_match_reference(methyl_benzoate) -> None:
    types = assign_gaff_types(methyl_benzoate, pindex=2)
    got = [types[i] for i in sorted(types)]
    assert got == REF_BCC_TYPES


def test_am1bcc_bond_types_match_reference(methyl_benzoate) -> None:
    am1bcc = AM1BCC(methyl_benzoate)
    bcc_types = assign_gaff_types(methyl_benzoate, pindex=2)
    bond_types = am1bcc._assign_bond_types(bcc_types)
    got = [bond_types[(b.i, b.j)] for b in methyl_benzoate.bonds]
    assert got == REF_BOND_TYPES


def test_am1bcc_charges_match_reference(methyl_benzoate) -> None:
    elements = [a.element for a in methyl_benzoate.atoms]
    coords = np.array([REF_GEOMETRY[i] for i in range(1, 19)])
    am1 = AM1(elements, coords)
    sp = am1.single_point()
    assert sp["converged"] is True

    am1bcc = AM1BCC(methyl_benzoate)
    bcc_types = assign_gaff_types(methyl_benzoate, pindex=2)
    bond_types = am1bcc._assign_bond_types(bcc_types)
    charges = np.asarray(sp["mulliken_charges"]).copy()
    for bond in methyl_benzoate.bonds:
        i, j = bond.i - 1, bond.j - 1
        bcc = am1bcc._bccparm(
            str(bcc_types[bond.i]), str(bcc_types[bond.j]), bond_types[(bond.i, bond.j)]
        )
        charges[i] += bcc
        charges[j] -= bcc

    # Reference charges are printed to 4 decimals and the AM1 SCF differs
    # from sqm by ~1e-2 e on a few H atoms at the identical geometry; the
    # BCC typing / bondtype / corrections are exact.  Tolerate 1e-2 e.
    assert np.max(np.abs(charges - np.array(REF_BCC_CHARGES))) < 1.0e-2
    assert abs(float(charges.sum())) < 1e-5


def test_am1bcc_net_charge_is_conserved(methyl_benzoate) -> None:
    elements = [a.element for a in methyl_benzoate.atoms]
    coords = np.array([REF_GEOMETRY[i] for i in range(1, 19)])
    raw = np.asarray(AM1(elements, coords).single_point()["mulliken_charges"])
    # AM1-BCC corrections are antisymmetric -> net charge unchanged.
    assert abs(float(raw.sum())) < 1e-6