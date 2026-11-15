"""Connectivity inference and periodic image-shift normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..domain.structure import Structure
from ..domain.topology import Bond
from ..errors import ValidationError
from ..forcefields.tables import COVALENT_RADII, IONIC_METALS, MAX_VALENCE


@dataclass(frozen=True, slots=True)
class BondCandidate:
    distance: float
    i: int
    j: int
    shift: tuple[int, int, int]


def assign_image_shifts(structure: Structure) -> None:
    """Fill the canonical minimum-image shift for each existing bond."""

    if structure.cell is None:
        return
    for bond in structure.bonds:
        displacement = structure.atom(bond.j).position - structure.atom(bond.i).position
        bond.image_shift = structure.cell.nearest_shift(displacement)


def infer_bonds(structure: Structure, *, factor: float = 1.20) -> list[Bond]:
    """Infer covalent bonds using radii, distance ordering, and valence limits.

    Explicit connectivity is intentionally not replaced.  The caller should
    invoke this function only for formats without reliable bond records.
    """

    if structure.bonds:
        assign_image_shifts(structure)
        return structure.bonds
    atoms = structure.atoms
    candidates: list[BondCandidate] = []
    for left, atom_i in enumerate(atoms, start=1):
        for right in range(left + 1, len(atoms) + 1):
            atom_j = structure.atom(right)
            if _skip_ionic_pair(atom_i.element, atom_j.element):
                continue
            radius_i = COVALENT_RADII.get(atom_i.element)
            radius_j = COVALENT_RADII.get(atom_j.element)
            if radius_i is None or radius_j is None:
                continue
            displacement = atom_j.position - atom_i.position
            shift = structure.cell.nearest_shift(displacement) if structure.cell else (0, 0, 0)
            vector = displacement + (structure.cell.image_vector(shift) if structure.cell else 0.0)
            distance = math.sqrt(float(vector @ vector))
            cutoff = factor * (radius_i + radius_j)
            if 0.35 <= distance <= cutoff:
                candidates.append(BondCandidate(distance, left, right, shift))
    candidates.sort(key=lambda candidate: (candidate.distance, candidate.i, candidate.j, candidate.shift))
    valence = {atom.index: 0.0 for atom in atoms}
    bonds: list[Bond] = []
    for candidate in candidates:
        order = 1.0
        element_i = structure.atom(candidate.i).element
        element_j = structure.atom(candidate.j).element
        max_i = MAX_VALENCE.get(element_i, 0)
        max_j = MAX_VALENCE.get(element_j, 0)
        # In crystal coordination environments a halide can be shared by
        # several metal centers.  Treat metal-halide contacts as coordination
        # edges rather than enforcing the molecular valence of an isolated ion.
        if element_i in {"F", "Cl", "Br", "I"} and element_j == "Fe":
            max_i = 10**6
        if element_j in {"F", "Cl", "Br", "I"} and element_i == "Fe":
            max_j = 10**6
        if max_i == 0 or max_j == 0:
            continue
        if valence[candidate.i] + order > max_i or valence[candidate.j] + order > max_j:
            continue
        bonds.append(Bond(candidate.i, candidate.j, order, candidate.shift, source="inferred"))
        valence[candidate.i] += order
        valence[candidate.j] += order
    structure.bonds = sorted(bonds, key=lambda bond: bond.key)
    return structure.bonds


def _skip_ionic_pair(element_i: str, element_j: str) -> bool:
    """Do not turn common salt contacts into covalent bonds."""

    if element_i in IONIC_METALS and element_j in IONIC_METALS:
        return True
    ionic_anions = {"F", "Cl", "Br", "I", "O", "S"}
    if (element_i in IONIC_METALS and element_j in ionic_anions) or (element_j in IONIC_METALS and element_i in ionic_anions):
        return True
    return False


def validate_image_shifts(structure: Structure) -> None:
    if structure.cell is None:
        if any(any(bond.image_shift) for bond in structure.bonds):
            raise ValidationError("non-periodic structure contains an image-shifted bond")
        return
    for bond in structure.bonds:
        if any(not periodic and value != 0 for value, periodic in zip(bond.image_shift, structure.cell.pbc)):
            raise ValidationError(f"bond {bond.i}-{bond.j} shifts a non-periodic dimension")
