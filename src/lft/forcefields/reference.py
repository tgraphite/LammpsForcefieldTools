"""Reader for the repository's Sobtop parameter reference.

The parser is independent of the old executable.  When the reference files
are present they provide reproducible GAFF-like bonded/LJ values; every
unmatched term still follows the explicit generic fallback path and is marked
in the parameter source.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import hashlib
from typing import Callable, Sequence, TypeVar, cast

from ..domain.parameters import AngleParameter, BondParameter, DihedralParameter, ImproperParameter, ParameterSet
from ..domain.structure import Structure
from ..domain.units import sigma_from_rmin

@dataclass(frozen=True, slots=True)
class BondEntry:
    types: tuple[str, str]
    k_half: float
    r0: float


@dataclass(frozen=True, slots=True)
class AngleEntry:
    types: tuple[str, str, str]
    k_half: float
    theta0: float


@dataclass(frozen=True, slots=True)
class FourierEntry:
    types: tuple[str, str, str, str]
    divisor: float
    k: float
    phase: float
    periodicity: int


@dataclass(frozen=True, slots=True)
class ImproperEntry:
    types: tuple[str, str, str, str]
    k: float
    phase: float
    periodicity: int


class ReferenceParameters:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bonds: list[BondEntry] = []
        self.angles: list[AngleEntry] = []
        self.dihedrals: list[FourierEntry] = []
        self.harmonic_dihedrals: list[FourierEntry] = []
        self.impropers: list[ImproperEntry] = []
        self.lj: dict[str, tuple[float, float]] = {}
        self._load()

    @classmethod
    def discover(cls) -> "ReferenceParameters | None":
        configured = os.environ.get("LFT_PARAM_DIR")
        candidates = [Path(configured)] if configured else []
        candidates.extend([Path(__file__).resolve().parents[3] / "references" / "sobtop", Path.cwd() / "references" / "sobtop"])
        for candidate in candidates:
            if (candidate / "bonded_param.dat").exists() and (candidate / "LJ_param.dat").exists():
                return cls(candidate)
        return None

    def apply(self, structure: Structure, params: ParameterSet, *, bonded: bool = True) -> None:
        used = {"bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0, "lj": 0}
        for type_name, atom_type in params.atom_types.items():
            if type_name in self.lj:
                rmin, epsilon = self.lj[type_name]
                atom_type.sigma = sigma_from_rmin(rmin)
                atom_type.epsilon = epsilon
                used["lj"] += 1
        if not bonded:
            params.metadata["reference_parameters"] = self._metadata(used, bonded=False)
            return
        new_bonds: list[BondParameter] = []
        for bond, old in zip(structure.bonds, params.bond_parameters):
            bond_types = (structure.atom(bond.i).atom_type or "", structure.atom(bond.j).atom_type or "")
            bond_entry = self._find_bond(bond_types)
            if bond_entry is None:
                new_bonds.append(old)
            else:
                new_bonds.append(BondParameter(bond_entry.r0, bond_entry.k_half, "harmonic", "reference:bonded_param.dat"))
                used["bonds"] += 1
        params.bond_parameters = new_bonds
        new_angles: list[AngleParameter] = []
        for term, angle_old in zip(structure.angles, params.angle_parameters):
            angle_types = tuple(structure.atom(index).atom_type or "" for index in term.atoms)
            angle_entry = self._find_angle(cast(tuple[str, str, str], angle_types))
            if angle_entry is None:
                new_angles.append(angle_old)
            else:
                new_angles.append(AngleParameter(angle_entry.theta0, angle_entry.k_half, "harmonic", "reference:bonded_param.dat"))
                used["angles"] += 1
        params.angle_parameters = new_angles
        new_dihedrals: list[DihedralParameter] = []
        for term, dihedral_old in zip(structure.dihedrals, params.dihedral_parameters):
            dihedral_types = tuple(structure.atom(index).atom_type or "" for index in term.atoms)
            dihedral_entry = self._find_dihedral(cast(tuple[str, str, str, str], dihedral_types), harmonic=dihedral_old.function == "harmonic")
            if dihedral_entry is None:
                new_dihedrals.append(dihedral_old)
            else:
                function = "harmonic" if dihedral_old.function == "harmonic" else "fourier"
                new_dihedrals.append(DihedralParameter(dihedral_entry.phase, dihedral_entry.k / dihedral_entry.divisor, dihedral_entry.periodicity, function, "reference:bonded_param.dat", dihedral_entry.divisor))
                used["dihedrals"] += 1
        params.dihedral_parameters = new_dihedrals
        new_impropers: list[ImproperParameter] = []
        for term, improper_old in zip(structure.impropers, params.improper_parameters):
            improper_types = tuple(structure.atom(index).atom_type or "" for index in term.atoms)
            improper_entry = self._find_improper(cast(tuple[str, str, str, str], improper_types))
            if improper_entry is None:
                new_impropers.append(improper_old)
            else:
                new_impropers.append(ImproperParameter(improper_entry.phase, improper_entry.k, improper_entry.periodicity, "cvff", "reference:bonded_param.dat"))
                used["impropers"] += 1
        params.improper_parameters = new_impropers
        params.metadata["reference_parameters"] = self._metadata(used, bonded=True)

    def _metadata(self, matched: dict[str, int], *, bonded: bool) -> dict[str, object]:
        files = {}
        for name in ("bonded_param.dat", "LJ_param.dat"):
            path = self.root / name
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[name] = {"path": str(path.resolve()), "sha256": digest}
        return {"root": str(self.root.resolve()), "files": files, "matched": matched, "bonded_applied": bonded}

    def _load(self) -> None:
        bonded = self.root / "bonded_param.dat"
        section: str | None = None
        for raw in bonded.read_text(errors="replace").splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("$"):
                section = line[1:].strip().upper()
                continue
            fields = line.split()
            if not fields:
                continue
            numbers = _numeric_tail(fields)
            type_token = _type_token(fields)
            if section == "BOND" and len(numbers) >= 2:
                types = _types(type_token, 2)
                if types:
                    self.bonds.append(BondEntry(cast(tuple[str, str], types), numbers[0], numbers[1]))
            elif section == "ANGLE" and len(numbers) >= 2:
                types = _types(type_token, 3)
                if types:
                    self.angles.append(AngleEntry(cast(tuple[str, str, str], types), numbers[0], numbers[1]))
            elif section in {"DIHEDRAL", "HARM_DIH"} and len(numbers) >= 4:
                types = _types(type_token, 4)
                if types:
                    entry = FourierEntry(cast(tuple[str, str, str, str], types), numbers[0], numbers[1], numbers[2], int(abs(numbers[3])))
                    (self.dihedrals if section == "DIHEDRAL" else self.harmonic_dihedrals).append(entry)
            elif section == "IMPROPER" and len(numbers) >= 3:
                types = _types(type_token, 4)
                if types:
                    self.impropers.append(ImproperEntry(cast(tuple[str, str, str, str], types), numbers[0], numbers[1], int(abs(numbers[2]))))
        lj_path = self.root / "LJ_param.dat"
        for raw in lj_path.read_text(errors="replace").splitlines():
            line = raw.split(";", 1)[0].strip()
            fields = line.split()
            if len(fields) >= 3:
                try:
                    self.lj[fields[0]] = (float(fields[1]), float(fields[2]))
                except ValueError:
                    continue

    def _find_bond(self, actual: tuple[str, str]) -> BondEntry | None:
        return _last_match(self.bonds, actual, lambda entry: _match(entry.types, actual) or _match(entry.types[::-1], actual))

    def _find_angle(self, actual: tuple[str, str, str]) -> AngleEntry | None:
        return _last_match(self.angles, actual, lambda entry: _match(entry.types, actual) or _match(entry.types[::-1], actual))

    def _find_dihedral(self, actual: tuple[str, str, str, str], *, harmonic: bool) -> FourierEntry | None:
        entries = self.harmonic_dihedrals if harmonic else self.dihedrals
        return _last_match(entries, actual, lambda entry: _match(entry.types, actual) or _match(entry.types[::-1], actual))

    def _find_improper(self, actual: tuple[str, str, str, str]) -> ImproperEntry | None:
        return _last_match(self.impropers, actual, lambda entry: entry.types[2] == actual[2] and _wildcard_set_match(entry.types[:2] + (entry.types[3],), (actual[0], actual[1], actual[3])))


def _numeric_tail(fields: list[str]) -> list[float]:
    for index, value in enumerate(fields):
        try:
            float(value)
        except ValueError:
            continue
        return [float(number) for number in fields[index:]]
    return []


def _type_token(fields: list[str]) -> str:
    for index, value in enumerate(fields):
        try:
            float(value)
            return "".join(fields[:index])
        except ValueError:
            continue
    return ""


def _types(value: str, count: int) -> tuple[str, ...] | None:
    values = tuple(item for item in value.split("-") if item)
    return values if len(values) == count else None


def _match(pattern: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    return len(pattern) == len(actual) and all(expected == "X" or expected == observed for expected, observed in zip(pattern, actual))


def _wildcard_set_match(pattern: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    remaining = list(actual)
    for expected in pattern:
        if expected == "X":
            continue
        if expected not in remaining:
            return False
        remaining.remove(expected)
    return True


T = TypeVar("T")


def _last_match(entries: Sequence[T], actual: tuple[str, ...], predicate: Callable[[T], bool]) -> T | None:
    selected: T | None = None
    for entry in entries:
        if predicate(entry):
            selected = entry
    return selected
