"""Packaged GAFF parameter library with indexed matching.

The table in data/gaff_bonded.json is generated from the Sobtop reference
files (which are derived from AmberTools gaff.dat 1.81) by
scripts/build_parameter_table.py.  Shipping the table inside the wheel is
what makes GAFF bonded parameters available to ordinary installs; the old
references/sobtop lookup only worked in the development checkout.

Matching preserves Amber's "last matching parameter wins" rule: dihedral
groups are ordered by their last occurrence in the source table, and improper
entries are scanned in file order.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..domain.parameters import (
    AngleParameter,
    BondParameter,
    DihedralParameter,
    ImproperParameter,
    ParameterSet,
)
from ..domain.structure import Structure
from ..domain.units import sigma_from_rmin

_DATA_FILE = Path(__file__).with_name("data") / "gaff_bonded.json"


def _matches(pattern: Sequence[str], actual: Sequence[str]) -> bool:
    return len(pattern) == len(actual) and all(
        expected == "X" or expected == observed
        for expected, observed in zip(pattern, actual)
    )


def _set_matches(pattern: Sequence[str], actual: Sequence[str]) -> bool:
    remaining = list(actual)
    for expected in pattern:
        if expected == "X":
            continue
        if expected not in remaining:
            return False
        remaining.remove(expected)
    return True


def _parse_sobtop_directory(root: Path) -> dict[str, Any]:
    """Parse a Sobtop-style directory into the packaged payload schema."""

    bonded = root / "bonded_param.dat"
    sections: dict[str, list[tuple[tuple[str, ...], list[float]]]] = {}
    section: str | None = None
    for raw in bonded.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("$"):
            section = line[1:].strip().upper()
            sections.setdefault(section, [])
            continue
        fields = line.split()
        token = ""
        numbers: list[float] | None = None
        for index, field in enumerate(fields):
            try:
                float(field)
            except ValueError:
                continue
            token = "".join(fields[:index])
            numbers = [float(value) for value in fields[index:]]
            break
        if not token or not numbers:
            continue
        types = tuple(part for part in token.split("-") if part)
        if types:
            sections[section].append((types, numbers))

    bonds: dict[tuple[str, ...], list[float]] = {}
    for types, numbers in sections.get("BOND", []):
        if len(types) == 2 and len(numbers) >= 2:
            bonds[types] = [numbers[0], numbers[1]]
    angles: dict[tuple[str, ...], list[float]] = {}
    for types, numbers in sections.get("ANGLE", []):
        if len(types) == 3 and len(numbers) >= 2:
            angles[types] = [numbers[0], numbers[1]]
    grouped: dict[tuple[str, ...], list[list[float]]] = {}
    last_index: dict[tuple[str, ...], int] = {}
    for index, (types, numbers) in enumerate(sections.get("DIHEDRAL", [])):
        if len(types) != 4 or len(numbers) < 4:
            continue
        grouped.setdefault(types, []).append([numbers[0], numbers[1], numbers[2], int(abs(numbers[3]))])
        last_index[types] = index
    dihedrals = [[list(types), grouped[types]] for types in sorted(grouped, key=lambda t: last_index[t])]
    impropers = [
        [list(types), numbers[0], numbers[1], int(abs(numbers[2]))]
        for types, numbers in sections.get("IMPROPER", [])
        if len(types) == 4 and len(numbers) >= 3
    ]
    lj: dict[str, list[float]] = {}
    lj_path = root / "LJ_param.dat"
    if lj_path.is_file():
        for raw in lj_path.read_text(errors="replace").splitlines():
            fields = raw.split(";", 1)[0].split()
            if len(fields) >= 3:
                try:
                    lj[fields[0]] = [float(fields[1]), float(fields[2])]
                except ValueError:
                    continue
    return {"bonds": [[list(t), *v] for t, v in sorted(bonds.items())],
            "angles": [[list(t), *v] for t, v in sorted(angles.items())],
            "dihedrals": dihedrals, "impropers": impropers, "lj": lj}


class ParameterLibrary:
    """Indexed GAFF parameter table used by the classic parameteriser."""

    def __init__(self, payload: dict[str, Any], *, label: str = "packaged") -> None:
        self.payload = payload
        self.label = label
        self.provenance = dict(payload.get("provenance", {}))
        self.bonds: dict[tuple[str, ...], tuple[float, float]] = {}
        for types, k, r0 in payload.get("bonds", []):
            self.bonds[tuple(types)] = (float(k), float(r0))
        self.angles: dict[tuple[str, ...], tuple[float, float]] = {}
        for types, k, theta0 in payload.get("angles", []):
            self.angles[tuple(types)] = (float(k), float(theta0))
        self._dihedral_groups: list[tuple[tuple[str, ...], tuple[tuple[float, int, float], ...]]] = []
        self._dihedral_exact: dict[tuple[str, ...], tuple[int, tuple[tuple[float, int, float], ...]]] = {}
        self._dihedral_wild: list[tuple[int, tuple[str, ...], tuple[tuple[float, int, float], ...]]] = []
        for index, (types, terms) in enumerate(payload.get("dihedrals", [])):
            pattern = tuple(types)
            parsed = tuple(
                (float(k) / float(divisor), int(periodicity), float(phase))
                for divisor, k, phase, periodicity in terms
            )
            self._dihedral_groups.append((pattern, parsed))
            if "X" in pattern:
                self._dihedral_wild.append((index, pattern, parsed))
            else:
                self._dihedral_exact[pattern] = (index, parsed)
        self.impropers: list[tuple[tuple[str, ...], float, float, int]] = [
            (tuple(types), float(k), float(phase), int(periodicity))
            for types, k, phase, periodicity in payload.get("impropers", [])
        ]
        self.lj: dict[str, tuple[float, float]] = {
            str(name): (float(values[0]), float(values[1]))
            for name, values in payload.get("lj", {}).items()
        }
        self._cache: dict[Any, Any] = {}

    @classmethod
    def from_directory(cls, root: str | Path) -> "ParameterLibrary":
        root = Path(root)
        return cls(_parse_sobtop_directory(root), label=str(root))

    @classmethod
    def discover(cls) -> "ParameterLibrary":
        configured = os.environ.get("LFT_PARAM_DIR")
        if configured:
            root = Path(configured)
            if (root / "bonded_param.dat").is_file():
                return cls.from_directory(root)
        return cls.packaged()

    @classmethod
    @lru_cache(maxsize=1)
    def packaged(cls) -> "ParameterLibrary":
        payload = json.loads(_DATA_FILE.read_text())
        return cls(payload, label=str(_DATA_FILE))

    @property
    def available(self) -> bool:
        return bool(self.bonds or self.angles or self._dihedral_groups or self.impropers)

    def lookup_bond(self, i: str, j: str) -> tuple[float, float] | None:
        key = ("bond", i, j)
        if key not in self._cache:
            self._cache[key] = self.bonds.get((i, j)) or self.bonds.get((j, i))
        return self._cache[key]

    def lookup_angle(self, i: str, j: str, k: str) -> tuple[float, float] | None:
        key = ("angle", i, j, k)
        if key not in self._cache:
            self._cache[key] = self.angles.get((i, j, k)) or self.angles.get((k, j, i))
        return self._cache[key]

    def lookup_dihedral(self, types: tuple[str, str, str, str]) -> tuple[tuple[float, int, float], ...] | None:
        key = ("dihedral",) + types
        if key in self._cache:
            return self._cache[key]
        reversed_types = tuple(reversed(types))
        best_index = -1
        best: tuple[tuple[float, int, float], ...] | None = None
        for pattern in (types, reversed_types):
            entry = self._dihedral_exact.get(pattern)
            if entry is not None and entry[0] > best_index:
                best_index, best = entry
            for index, wildcard, terms in self._dihedral_wild:
                if index <= best_index:
                    continue
                if _matches(wildcard, pattern):
                    best_index, best = index, terms
        self._cache[key] = best
        return best

    def lookup_improper(self, types: tuple[str, str, str, str]) -> tuple[float, float, int] | None:
        # Internal order is (neighbour, centre, neighbour, neighbour); GAFF
        # stores (neighbour, neighbour, centre, neighbour) with X wildcards.
        candidate = (types[0], types[2], types[1], types[3])
        key = ("improper",) + candidate
        if key in self._cache:
            return self._cache[key]
        best: tuple[float, float, int] | None = None
        for pattern, k, phase, periodicity in self.impropers:
            if pattern[2] != candidate[2]:
                continue
            if _set_matches((pattern[0], pattern[1], pattern[3]), (candidate[0], candidate[1], candidate[3])):
                best = (k, phase, periodicity)
        self._cache[key] = best
        return best

    def apply(self, structure: Structure, params: ParameterSet, *, bonded: bool = True) -> None:
        used = {"bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0, "lj": 0}
        for type_name, atom_type in params.atom_types.items():
            entry = self.lj.get(type_name)
            if entry is not None:
                atom_type.sigma = sigma_from_rmin(entry[0])
                atom_type.epsilon = entry[1]
                if "gaff_lj" not in atom_type.source:
                    atom_type.source = f"{atom_type.source}+gaff_lj"
                used["lj"] += 1
        if not bonded:
            params.metadata["parameter_library"] = self._metadata(used, bonded=False)
            return

        def atom_type(index: int) -> str:
            return structure.atom(index).atom_type or ""

        new_bonds: list[BondParameter] = []
        for bond, old in zip(structure.bonds, params.bond_parameters):
            entry = self.lookup_bond(atom_type(bond.i), atom_type(bond.j))
            if entry is None:
                new_bonds.append(old)
            else:
                new_bonds.append(BondParameter(entry[1], entry[0], "harmonic", "gaff:bond"))
                used["bonds"] += 1
        params.bond_parameters = new_bonds

        new_angles: list[AngleParameter] = []
        for term, old in zip(structure.angles, params.angle_parameters):
            entry = self.lookup_angle(*(atom_type(index) for index in term.atoms))  # type: ignore[arg-type]
            if entry is None:
                new_angles.append(old)
            else:
                new_angles.append(AngleParameter(entry[1], entry[0], "harmonic", "gaff:angle"))
                used["angles"] += 1
        params.angle_parameters = new_angles

        new_dihedrals: list[DihedralParameter] = []
        for term, old in zip(structure.dihedrals, params.dihedral_parameters):
            types = tuple(atom_type(index) for index in term.atoms)
            terms = self.lookup_dihedral(types)  # type: ignore[arg-type]
            if terms is None:
                new_dihedrals.append(old)
            else:
                first_k, first_n, first_phase = terms[0]
                new_dihedrals.append(
                    DihedralParameter(first_phase, first_k, first_n, "fourier", "gaff:dihedral", 1.0, terms=terms)
                )
                used["dihedrals"] += 1
        params.dihedral_parameters = new_dihedrals

        new_impropers: list[ImproperParameter] = []
        for term, old in zip(structure.impropers, params.improper_parameters):
            types = tuple(atom_type(index) for index in term.atoms)
            entry = self.lookup_improper(types)  # type: ignore[arg-type]
            if entry is None:
                new_impropers.append(old)
            else:
                k, phase, periodicity = entry
                new_impropers.append(ImproperParameter(phase, k, periodicity, "cvff", "gaff:improper"))
                used["impropers"] += 1
        params.improper_parameters = new_impropers

        params.metadata["parameter_library"] = self._metadata(used, bonded=True)

    def _metadata(self, used: dict[str, int], *, bonded: bool) -> dict[str, Any]:
        return {
            "library": self.label,
            "provenance": self.provenance,
            "matched": used,
            "bonded_applied": bonded,
        }
