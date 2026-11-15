"""Enumerate angles, dihedrals, impropers and 1-4 pairs with image lifts."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import numpy as np

from ..domain.structure import ImageShift, Structure
from ..domain.topology import Angle, Dihedral, Improper, Pair
from ..errors import ValidationError
from .infer_bonds import assign_image_shifts, validate_image_shifts


def _add(a: ImageShift, b: ImageShift) -> ImageShift:
    return tuple(int(x + y) for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: ImageShift, b: ImageShift) -> ImageShift:
    return tuple(int(x - y) for x, y in zip(a, b))  # type: ignore[return-value]


def _directed_neighbors(structure: Structure) -> dict[int, list[tuple[int, ImageShift]]]:
    neighbors: dict[int, list[tuple[int, ImageShift]]] = defaultdict(list)
    for bond in structure.bonds:
        neighbors[bond.i].append((bond.j, bond.image_shift))
        neighbors[bond.j].append((bond.i, tuple(-value for value in bond.image_shift)))
    for values in neighbors.values():
        values.sort(key=lambda item: (item[0], item[1]))
    return neighbors


def _pair_separation(structure: Structure, pair: Pair) -> float:
    points: list[np.ndarray] = []
    for atom_index, image in zip(pair.atoms, pair.atom_images):
        position = structure.atom(atom_index).position
        if structure.cell is not None:
            position = position + structure.cell.image_vector(image)
        points.append(position)
    return float(np.linalg.norm(points[0] - points[1]))


def enumerate_terms(structure: Structure) -> Structure:
    """Populate all bonded term lists in deterministic canonical order."""

    assign_image_shifts(structure)
    validate_image_shifts(structure)
    neighbors = _directed_neighbors(structure)
    angles: dict[tuple[tuple[int, ...], tuple[ImageShift, ...]], Angle] = {}
    for center in range(1, structure.natoms + 1):
        values = neighbors.get(center, [])
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left, left_shift = values[left_index]
                right, right_shift = values[right_index]
                term = Angle((left, center, right), (left_shift, (0, 0, 0), right_shift))
                reverse = Angle((right, center, left), (right_shift, (0, 0, 0), left_shift))
                selected = min((term, reverse), key=lambda item: item.key)
                angles[selected.key] = selected

    dihedrals: dict[tuple[tuple[int, ...], tuple[ImageShift, ...]], Dihedral] = {}
    for center_bond in structure.bonds:
        b, c = center_bond.i, center_bond.j
        b_to_c = center_bond.image_shift
        for a, b_to_a in neighbors.get(b, []):
            if a == c:
                continue
            for d, c_to_d in neighbors.get(c, []):
                if d in {b, a}:
                    continue
                dihedral_term = Dihedral(
                    (a, b, c, d),
                    (b_to_a, (0, 0, 0), b_to_c, _add(b_to_c, c_to_d)),
                )
                reverse_dihedral = cast(Dihedral, dihedral_term.reversed_about_second())
                selected_dihedral = min((dihedral_term, reverse_dihedral), key=lambda item: item.key)
                dihedrals[selected_dihedral.key] = selected_dihedral

    impropers: dict[tuple[tuple[int, ...], tuple[ImageShift, ...]], Improper] = {}
    for center in range(1, structure.natoms + 1):
        values = neighbors.get(center, [])
        if len(values) < 3 or not _needs_improper(structure, center, values):
            continue
        # A single improper per center is the stable, force-field-neutral
        # representation.  Atom typing can later decide whether it is needed.
        selected_neighbors = values[:3]
        atoms = tuple(item[0] for item in selected_neighbors)
        images = tuple(item[1] for item in selected_neighbors)
        improper_term = Improper((atoms[0], center, atoms[1], atoms[2]), (images[0], (0, 0, 0), images[1], images[2]))
        impropers[improper_term.key] = improper_term

    # Like LAMMPS special_bonds, a 1-4 relationship is a property of the atom
    # pair, not of the periodic path that produced it.  A dihedral endpoint pair
    # whose shortest graph path is 1-2 or 1-3 is excluded by nrexcl=3, not
    # scaled as 1-4, so only true 1-4 pairs (shortest path exactly three bonds)
    # are emitted.  In a winding graph the same pair can otherwise appear via
    # several image shifts; keep one pair per unordered atom pair, choosing the
    # shortest lifted separation.
    one_two = {(min(bond.i, bond.j), max(bond.i, bond.j)) for bond in structure.bonds}
    one_three = {
        (min(angle.atoms[0], angle.atoms[2]), max(angle.atoms[0], angle.atoms[2]))
        for angle in angles.values()
    }
    pairs: dict[tuple[int, int], Pair] = {}
    for dihedral_term in dihedrals.values():
        a, _, _, d = dihedral_term.atoms
        images = (dihedral_term.atom_images[0], dihedral_term.atom_images[3])
        if a == d:
            continue
        key = (min(a, d), max(a, d))
        if key in one_two or key in one_three:
            continue
        if a < d:
            pair = Pair((a, d), images, source="1-4")
        else:
            pair = Pair((d, a), (_sub(images[1], images[1]), _sub(images[0], images[1])), source="1-4")
        existing = pairs.get(key)
        if existing is None or _pair_separation(structure, pair) < _pair_separation(structure, existing):
            pairs[key] = pair

    structure.angles = sorted(angles.values(), key=lambda term: term.key)
    structure.dihedrals = sorted(dihedrals.values(), key=lambda term: term.key)
    structure.impropers = sorted(impropers.values(), key=lambda term: term.key)
    structure.pairs = sorted(pairs.values(), key=lambda term: term.key)
    validate_term_lifts(structure)
    return structure


def _needs_improper(structure: Structure, center: int, values: list[tuple[int, ImageShift]]) -> bool:
    atom = structure.atom(center)
    if atom.element not in {"B", "C", "N", "P", "S"}:
        return False
    if len(values) != 3:
        return False
    if any(
        bond.order >= 1.4
        for bond in structure.bonds
        if bond.i == center or bond.j == center
    ):
        return True
    if atom.element == "N":
        neighbors = {neighbor for neighbor, _image in values}
        return any(
            structure.atom(neighbor).element == "C"
            and any(
                other.order >= 1.4
                and structure.atom(other.j if other.i == neighbor else other.i).element == "O"
                for other in structure.bonds
                if other.i == neighbor or other.j == neighbor
            )
            for neighbor in neighbors
        )
    return False


def validate_term_lifts(structure: Structure) -> None:
    """Verify that every lifted bonded edge agrees with the bond IR."""

    edge_shifts: dict[tuple[int, int], ImageShift] = {}
    for bond in structure.bonds:
        edge_shifts[(bond.i, bond.j)] = bond.image_shift
        edge_shifts[(bond.j, bond.i)] = tuple(-value for value in bond.image_shift)

    def check_path(atoms: tuple[int, ...], images: tuple[ImageShift, ...], *, term_name: str) -> None:
        for left, right, image_left, image_right in zip(atoms, atoms[1:], images, images[1:]):
            expected = edge_shifts.get((left, right))
            if expected is None:
                continue
            actual = _sub(image_right, image_left)
            if actual != expected:
                raise ValidationError(f"{term_name} {atoms} lift disagrees with bond {left}-{right}: {actual} != {expected}")

    for term in structure.angles:
        check_path(term.atoms, term.atom_images, term_name="angle")
    for term in structure.dihedrals:
        check_path(term.atoms, term.atom_images, term_name="dihedral")
    for term in structure.impropers:
        center = term.atoms[1]
        center_image = term.atom_images[1]
        for atom, image in ((term.atoms[0], term.atom_images[0]), (term.atoms[2], term.atom_images[2]), (term.atoms[3], term.atom_images[3])):
            expected = edge_shifts.get((center, atom))
            if expected is not None and _sub(image, center_image) != expected:
                raise ValidationError(f"improper {term.atoms} lift disagrees with bond {center}-{atom}")
