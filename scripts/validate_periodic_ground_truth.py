#!/usr/bin/env python3
"""Validate the checked-in periodic topology invariants without writing refs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lft.io.loaders import load_structure
from lft.topology import compute_winding, enumerate_terms, infer_bonds


CASES = (
    ("diamond", "references/sobtop/examples/diamond_3x3x3/diamond_3x3x3.pdb", 216, 432, 99, 3),
    ("sio2", "references/sobtop/examples/SiO2_3x3x3/SiO2_3x3x3.pdb", 324, 432, 54, 3),
    ("fecl2", "references/sobtop/examples/FeCl2/exact/opted_5x5x1.pdb", 75, 150, 28, 2),
)


def main() -> int:
    for name, relative, atoms, bonds, cross_edges, rank in CASES:
        structure = load_structure(ROOT / relative)
        infer_bonds(structure)
        enumerate_terms(structure)
        actual = (structure.natoms, len(structure.bonds), sum(bond.image_shift != (0, 0, 0) for bond in structure.bonds), compute_winding(structure)["rank"])
        expected = (atoms, bonds, cross_edges, rank)
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, got {actual}")
        print(f"{name}: atoms={actual[0]} bonds={actual[1]} cross_edges={actual[2]} winding_rank={actual[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
