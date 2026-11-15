"""The normalized molecular structure used by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ..errors import InputError, ValidationError

ANGSTROM_TO_NM = 0.1


ImageShift = tuple[int, int, int]


@dataclass(slots=True)
class Cell:
    """A crystallographic cell with column-wise lattice vectors in Angstrom."""

    matrix: np.ndarray
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise InputError("cell matrix must have shape (3, 3)")
        self.pbc = tuple(bool(x) for x in self.pbc)  # type: ignore[assignment]
        if abs(float(np.linalg.det(self.matrix))) < 1.0e-10:
            raise InputError("cell matrix is singular")

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        pbc: tuple[bool, bool, bool] = (True, True, True),
    ) -> "Cell":
        if min(a, b, c) <= 0.0:
            raise InputError("cell lengths must be positive")
        al, be, ga = np.radians([alpha, beta, gamma])
        sin_gamma = math.sin(float(ga))
        if abs(sin_gamma) < 1.0e-12:
            raise InputError("cell angle gamma cannot be 0 or 180 degrees")
        v1 = np.array([a, 0.0, 0.0])
        v2 = np.array([b * math.cos(float(ga)), b * sin_gamma, 0.0])
        cx = c * math.cos(float(be))
        cy = c * (math.cos(float(al)) - math.cos(float(be)) * math.cos(float(ga))) / sin_gamma
        cz2 = c * c - cx * cx - cy * cy
        if cz2 <= 1.0e-12:
            raise InputError("cell parameters do not describe a 3-D cell")
        v3 = np.array([cx, cy, math.sqrt(cz2)])
        return cls(np.column_stack((v1, v2, v3)), pbc)

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.matrix)))

    @property
    def lengths_angles(self) -> tuple[float, float, float, float, float, float]:
        a, b, c = (float(np.linalg.norm(self.matrix[:, i])) for i in range(3))

        def angle(u: np.ndarray, v: np.ndarray) -> float:
            value = float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))
            return math.degrees(math.acos(max(-1.0, min(1.0, value))))  # type: ignore[arg-type]

        return (
            a,
            b,
            c,
            angle(self.matrix[:, 1], self.matrix[:, 2]),
            angle(self.matrix[:, 0], self.matrix[:, 2]),
            angle(self.matrix[:, 0], self.matrix[:, 1]),
        )

    def fractional(self, cartesian: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return np.linalg.solve(self.matrix, np.asarray(cartesian, dtype=float))

    def cartesian(self, fractional: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self.matrix @ np.asarray(fractional, dtype=float)

    def image_vector(self, shift: ImageShift) -> np.ndarray:
        return self.matrix @ np.asarray(shift, dtype=float)

    def cell_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(self.matrix[:, index].copy() for index in range(3))  # type: ignore[return-value]

    def nearest_shift(self, displacement: np.ndarray) -> ImageShift:
        """Return the integer shift that minimizes ``displacement + H @ shift``."""

        if not any(self.pbc):
            return (0, 0, 0)
        fractional = self.fractional(displacement)
        initial = np.rint(-fractional).astype(int)
        candidates: list[tuple[float, ImageShift]] = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    raw = initial + np.array([da, db, dc])
                    shift: ImageShift = tuple(int(raw[i]) if self.pbc[i] else 0 for i in range(3))  # type: ignore[assignment]
                    vector = displacement + self.image_vector(shift)
                    candidates.append((float(np.dot(vector, vector)), shift))
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    def wrap_fractional(self, fractional: np.ndarray) -> np.ndarray:
        result = np.asarray(fractional, dtype=float).copy()
        for i, periodic in enumerate(self.pbc):
            if periodic:
                result[i] -= math.floor(result[i])
        return result

    def to_dict(self) -> dict[str, Any]:
        lengths = self.lengths_angles
        return {
            "matrix_angstrom": self.matrix.tolist(),
            "lengths_angles": list(lengths),
            "pbc": list(self.pbc),
            "volume_angstrom3": self.volume,
        }


@dataclass(slots=True)
class Atom:
    index: int
    element: str
    name: str
    position: np.ndarray
    residue_id: int = 1
    residue_name: str = "MOL"
    chain: str = ""
    molecule_id: int = 1
    charge: float = 0.0
    atom_type: str | None = None
    mass: float = 0.0
    charge_source: str | None = None
    input_type: str | None = None

    def __post_init__(self) -> None:
        self.element = self.element.capitalize()
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (3,):
            raise InputError(f"atom {self.index} position must have three coordinates")
        if self.mass <= 0.0:
            self.mass = 0.0

    @property
    def stable_id(self) -> str:
        return f"a{self.index}"

    def coord(self) -> np.ndarray:
        return self.position.copy()

    def set_coord(self, value: Any) -> None:
        self.position = np.asarray(value, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "stable_id": self.stable_id,
            "element": self.element,
            "name": self.name,
            "position_angstrom": [float(x) for x in self.position],
            "residue_id": self.residue_id,
            "residue_name": self.residue_name,
            "chain": self.chain,
            "molecule_id": self.molecule_id,
            "charge": float(self.charge),
            "charge_source": self.charge_source,
            "atom_type": self.atom_type,
            "mass": float(self.mass),
            "input_type": self.input_type,
        }


@dataclass(slots=True)
class Structure:
    name: str
    atoms: list[Atom]
    bonds: list[Any] = field(default_factory=list)
    cell: Cell | None = None
    source_path: str | None = None
    source_format: str | None = None
    declared_net_charge: float = 0.0
    multiplicity: int = 1
    explicit_bonds: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    angles: list[Any] = field(default_factory=list)
    dihedrals: list[Any] = field(default_factory=list)
    impropers: list[Any] = field(default_factory=list)
    pairs: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        for expected, atom in enumerate(self.atoms, start=1):
            if atom.index != expected:
                raise InputError(f"atom indices must be contiguous starting at 1; found {atom.index}")
        if self.declared_net_charge == -0.0:
            self.declared_net_charge = 0.0

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    def n(self) -> int:
        """Small compatibility alias used by reference topology scripts."""
        return self.natoms

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([atom.position for atom in self.atoms], dtype=float)

    @property
    def net_charge(self) -> float:
        return float(sum(atom.charge for atom in self.atoms))

    def atom(self, index: int) -> Atom:
        if not 1 <= index <= self.natoms:
            raise IndexError(index)
        return self.atoms[index - 1]

    def bond_order(self, i: int, j: int) -> float:
        for bond in self.bonds:
            if {bond.i, bond.j} == {i, j}:
                return float(bond.order)
        return 0.0

    def add_warning(self, code: str, message: str, **details: Any) -> None:
        warning = {"code": code, "message": message, **details}
        if warning not in self.warnings:
            self.warnings.append(warning)

    def validate(self) -> None:
        if not self.atoms:
            raise ValidationError("structure contains no atoms", hint="provide a non-empty xyz, mol2, pdb, pqr, gro, or cif file")
        if self.multiplicity < 1:
            raise ValidationError("multiplicity must be a positive integer")
        seen: set[tuple[int, int, ImageShift]] = set()
        for bond in self.bonds:
            if bond.i == bond.j:
                raise ValidationError(f"self bond on atom {bond.i}")
            key = (min(bond.i, bond.j), max(bond.i, bond.j), bond.image_shift)
            if key in seen:
                raise ValidationError(f"duplicate bond {key}")
            seen.add(key)
            if not 1 <= bond.i <= self.natoms or not 1 <= bond.j <= self.natoms:
                raise ValidationError(f"bond {key} refers to an atom outside the structure")
            if self.cell is None and any(bond.image_shift):
                raise ValidationError(f"non-periodic bond {key} carries image shift {bond.image_shift}")
            if self.cell is not None:
                for i, periodic in enumerate(self.cell.pbc):
                    if not periodic and bond.image_shift[i] != 0:
                        raise ValidationError(f"bond {key} shifts a non-periodic cell dimension")

    def is_neutral(self, tolerance: float = 1.0e-5) -> bool:
        return abs(self.declared_net_charge) <= tolerance and abs(self.net_charge) <= tolerance

    def to_dict(self) -> dict[str, Any]:
        def term_dict(term: Any) -> dict[str, Any]:
            return term.to_dict()

        return {
            "schema": "lft.ir.v1",
            "name": self.name,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "declared_net_charge": float(self.declared_net_charge),
            "computed_net_charge": self.net_charge,
            "multiplicity": self.multiplicity,
            "cell": self.cell.to_dict() if self.cell else None,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "bonds": [bond.to_dict() for bond in sorted(self.bonds, key=lambda x: x.key)],
            "angles": [term_dict(term) for term in sorted(self.angles, key=lambda x: x.key)],
            "dihedrals": [term_dict(term) for term in sorted(self.dihedrals, key=lambda x: x.key)],
            "impropers": [term_dict(term) for term in sorted(self.impropers, key=lambda x: x.key)],
            "pairs": [term_dict(term) for term in sorted(self.pairs, key=lambda x: x.key)],
        }
