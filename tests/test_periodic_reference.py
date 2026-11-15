from __future__ import annotations

from pathlib import Path

import pytest

from lft.io.loaders import load_structure
from lft.topology import compute_winding, enumerate_terms, infer_bonds


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "bond_count", "cross_count", "winding_rank"),
    [
        ("references/sobtop/examples/diamond_3x3x3/diamond_3x3x3.pdb", 432, 99, 3),
        ("references/sobtop/examples/SiO2_3x3x3/SiO2_3x3x3.pdb", 432, 54, 3),
        ("references/sobtop/examples/FeCl2/exact/opted_5x5x1.pdb", 150, 28, 2),
    ],
)
def test_periodic_reference_counts(relative_path: str, bond_count: int, cross_count: int, winding_rank: int) -> None:
    structure = load_structure(ROOT / relative_path)
    infer_bonds(structure)
    enumerate_terms(structure)
    assert len(structure.bonds) == bond_count
    assert sum(bond.image_shift != (0, 0, 0) for bond in structure.bonds) == cross_count
    assert compute_winding(structure)["rank"] == winding_rank
