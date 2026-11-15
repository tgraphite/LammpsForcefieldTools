from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lft.domain.structure import Atom, Cell, Structure
from lft.domain.topology import Bond
from lft.io.loaders import load_structure
from lft.topology import compute_winding, enumerate_terms, infer_bonds


ROOT = Path(__file__).resolve().parents[1]


def _periodic_square_ring() -> Structure:
    cell = Cell.from_parameters(4.0, 4.0, 4.0)
    atoms = [
        Atom(1, "C", "C1", np.array([0.1, 0.1, 0.1]), mass=12.0),
        Atom(2, "C", "C2", np.array([1.1, 0.1, 0.1]), mass=12.0),
        Atom(3, "C", "C3", np.array([1.1, 1.1, 0.1]), mass=12.0),
        Atom(4, "C", "C4", np.array([0.1, 1.1, 0.1]), mass=12.0),
    ]
    bonds = [Bond(1, 2), Bond(2, 3), Bond(3, 4), Bond(4, 1)]
    return Structure("square", atoms, bonds, cell, explicit_bonds=True)


def test_square_ring_pairs_exclude_shorter_path_relationships() -> None:
    structure = _periodic_square_ring()
    enumerate_terms(structure)

    bonded = {(min(b.i, b.j), max(b.i, b.j)) for b in structure.bonds}
    one_three = {(min(a.atoms[0], a.atoms[2]), max(a.atoms[0], a.atoms[2])) for a in structure.angles}
    # The only dihedral endpoint pair in a four-membered ring is also a bond;
    # it must be treated as a 1-2 exclusion, not emitted as a 1-4 pair.
    for pair in structure.pairs:
        key = tuple(pair.atoms)
        assert key not in bonded
        assert key not in one_three


def test_periodic_reference_pairs_are_unique_true_one_four() -> None:
    structure = load_structure(ROOT / "references/sobtop/examples/SiO2_3x3x3/SiO2_3x3x3.pdb")
    infer_bonds(structure)
    enumerate_terms(structure)

    keys = [tuple(pair.atoms) for pair in structure.pairs]
    assert len(keys) == len(set(keys))
    bonded = {(min(b.i, b.j), max(b.i, b.j)) for b in structure.bonds}
    one_three = {(min(a.atoms[0], a.atoms[2]), max(a.atoms[0], a.atoms[2])) for a in structure.angles}
    assert all(key not in bonded and key not in one_three for key in keys)


def _lifted_bond_vectors(structure: Structure) -> list[tuple[int, int, tuple[float, float, float]]]:
    vectors = []
    for bond in structure.bonds:
        vector = structure.atom(bond.j).position - structure.atom(bond.i).position
        if structure.cell is not None:
            vector = vector + structure.cell.image_vector(bond.image_shift)
        vectors.append((bond.i, bond.j, tuple(float(value) for value in np.round(vector, 6))))
    return sorted(vectors)


@pytest.mark.parametrize(
    "relative_path",
    [
        "references/sobtop/examples/diamond_3x3x3/diamond_3x3x3.pdb",
        "references/sobtop/examples/SiO2_3x3x3/SiO2_3x3x3.pdb",
        "references/sobtop/examples/FeCl2/exact/opted_5x5x1.pdb",
    ],
)
def test_cell_face_wrapping_preserves_lifted_topology(relative_path: str) -> None:
    original = load_structure(ROOT / relative_path)
    infer_bonds(original)
    enumerate_terms(original)

    # Move every atom across cell faces by wrapping its fractional coordinates;
    # the inferred bond graph, lifted bond vectors, and winding must be invariant.
    fractional = np.array([original.cell.fractional(atom.position) for atom in original.atoms])
    fractional -= np.floor(fractional)
    atoms = [
        Atom(atom.index, atom.element, atom.name, original.cell.cartesian(fractional[atom.index - 1]), charge=atom.charge, atom_type=atom.atom_type, mass=atom.mass)
        for atom in original.atoms
    ]
    wrapped = Structure(original.name + "_wrapped", atoms, [], original.cell, explicit_bonds=False)
    infer_bonds(wrapped)
    enumerate_terms(wrapped)

    assert sorted((b.i, b.j) for b in wrapped.bonds) == sorted((b.i, b.j) for b in original.bonds)
    assert _lifted_bond_vectors(wrapped) == _lifted_bond_vectors(original)
    assert compute_winding(wrapped)["rank"] == compute_winding(original)["rank"]
