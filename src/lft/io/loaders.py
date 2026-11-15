"""Load common molecular formats into the normalized structure IR."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from ..domain.structure import Atom, Cell, Structure
from ..domain.topology import Bond
from ..errors import InputError, LFTError
from ..forcefields.tables import ELEMENT_MASS


def _element(value: str, fallback: str = "C") -> str:
    token = re.sub(r"[^A-Za-z]", "", value or "")
    if len(token) >= 2 and token[:2].capitalize() in ELEMENT_MASS:
        return token[:2].capitalize()
    if token and token[0].upper() in ELEMENT_MASS:
        return token[0].upper()
    resolved = re.sub(r"[^A-Za-z]", "", fallback or "")
    if len(resolved) >= 2 and resolved[:2].capitalize() in ELEMENT_MASS:
        return resolved[:2].capitalize()
    if resolved and resolved[0].upper() in ELEMENT_MASS:
        return resolved[0].upper()
    return fallback or "C"


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _pdb_residue(tokens: list[str]) -> tuple[str, str, int]:
    """Split the residue/chain/resSeq tokens of a whitespace-parsed PDB line."""

    residue_id = 1
    chain = ""
    if tokens:
        match = re.match(r"[+-]?\d+", tokens[-1])
        if match:
            residue_id = int(match.group(0))
            tokens = tokens[:-1]
    if tokens and len(tokens[-1]) == 1 and tokens[-1].isalpha():
        chain = tokens[-1]
        tokens = tokens[:-1]
    residue_name = "".join(tokens).strip() or "MOL"
    return residue_name, chain, residue_id


def _atom(index: int, element: str, name: str, position: Iterable[float], **kwargs: Any) -> Atom:
    element = _element(element, name)
    return Atom(index, element, name or f"{element}{index}", np.asarray(tuple(position), dtype=float), mass=ELEMENT_MASS.get(element, 0.0), **kwargs)


def _bond_order(value: str) -> float:
    low = value.lower()
    if low in {"ar", "arom", "am"}:
        return 1.5
    if low in {"du", "un", "wk", "weak", "single", "s"}:
        return 1.0
    if low in {"double", "d"}:
        return 2.0
    if low in {"triple", "t"}:
        return 3.0
    try:
        return float(value)
    except ValueError:
        return 1.0


def _mol2_element(atom_type: str, name: str) -> str:
    base = atom_type.split(".", 1)[0]
    if base.capitalize() in ELEMENT_MASS:
        return base.capitalize()
    return _element(name, "C")


def _structure_name(path: Path) -> str:
    return path.stem or "molecule"


SUPPORTED_STRUCTURE_SUFFIXES = (
    ".mol2",
    ".pdb",
    ".ent",
    ".pqr",
    ".gro",
    ".xyz",
    ".cif",
    ".mmcif",
    ".json",
)


def load_structure(path: str | Path) -> Structure:
    path = Path(path)
    if not path.exists():
        raise InputError(f"structure file does not exist: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".mol2":
            structure = load_mol2(path)
        elif suffix in {".pdb", ".ent", ".pqr"}:
            structure = load_pdb_or_pqr(path)
        elif suffix == ".gro":
            structure = load_gro(path)
        elif suffix == ".xyz":
            structure = load_xyz(path)
        elif suffix in {".cif", ".mmcif"}:
            structure = load_cif(path)
        elif suffix == ".json":
            structure = load_structure_json(path)
        else:
            raise InputError(
                f"unsupported structure format {suffix!r}: {path}",
                hint=f"use one of: {', '.join(SUPPORTED_STRUCTURE_SUFFIXES)}",
            )
    except LFTError:
        raise
    except (ValueError, IndexError, KeyError, TypeError, OSError) as exc:
        raise InputError(f"could not parse structure {path}: {exc}") from exc
    structure.source_path = str(path.resolve())
    structure.source_format = suffix.lstrip(".")
    structure.validate()
    return structure


def load_mol2(path: str | Path) -> Structure:
    path = Path(path)
    sections: dict[str, list[str]] = {}
    section: str | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.upper().startswith("@<TRIPOS>"):
            section = line[9:].strip().upper()
            sections.setdefault(section, [])
        elif section and line and not line.startswith("#"):
            sections[section].append(line)
    if "ATOM" not in sections:
        raise InputError(f"mol2 has no ATOM section: {path}")
    declared_charge = 0.0
    molecule = sections.get("MOLECULE", [])
    if len(molecule) >= 5:
        try:
            declared_charge = float(molecule[4].split()[0])
        except (ValueError, IndexError):
            declared_charge = 0.0
    atoms: list[Atom] = []
    for line in sections["ATOM"]:
        fields = line.split()
        if len(fields) < 6:
            raise InputError(f"invalid mol2 atom record: {line!r}")
        index = int(fields[0])
        name = fields[1]
        position = [float(fields[2]), float(fields[3]), float(fields[4])]
        input_type = fields[5]
        charge = 0.0
        if len(fields) >= 9:
            try:
                charge = float(fields[8])
            except ValueError:
                pass
        residue_id = int(fields[6]) if len(fields) >= 7 and fields[6].lstrip("+-").isdigit() else 1
        residue_name = fields[7] if len(fields) >= 8 else "MOL"
        atoms.append(_atom(index, _mol2_element(input_type, name), name, position, residue_id=residue_id, residue_name=residue_name, charge=charge, charge_source="input" if len(fields) >= 9 else None, input_type=input_type))
    atoms.sort(key=lambda atom: atom.index)
    bonds: list[Bond] = []
    for line in sections.get("BOND", []):
        fields = line.split()
        if len(fields) >= 4:
            bonds.append(Bond(int(fields[1]), int(fields[2]), _bond_order(fields[3]), source="explicit"))
    cell = None
    if sections.get("CRYSIN"):
        fields = sections["CRYSIN"][0].replace(",", " ").split()
        if len(fields) >= 6:
            cell = Cell.from_parameters(*tuple(float(value) for value in fields[:6]))  # type: ignore[arg-type]
    charge_type = molecule[3].upper() if len(molecule) >= 4 else ""
    has_charges = charge_type not in {"NO_CHARGES", "NO_CHARGE"}
    if not has_charges:
        for atom in atoms:
            atom.charge = 0.0
            atom.charge_source = None
    return Structure(_structure_name(path), atoms, bonds, cell, declared_net_charge=declared_charge, explicit_bonds=bool(bonds), metadata={"mol2_sections": sorted(sections)})


def load_pdb_or_pqr(path: str | Path) -> Structure:
    path = Path(path)
    is_pqr = path.suffix.lower() == ".pqr"
    atoms: list[Atom] = []
    serial_to_index: dict[int, int] = {}
    conect: list[tuple[int, int]] = []
    cell: Cell | None = None
    lines = path.read_text(errors="replace").splitlines()
    for line in lines:
        if line.startswith("CRYST1"):
            try:
                values = [float(line[start:end]) for start, end in ((6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54))]
                cell = Cell.from_parameters(*tuple(values))  # type: ignore[arg-type]
            except (ValueError, InputError):
                fields = line.split()
                if len(fields) >= 7:
                    cell = Cell.from_parameters(*tuple(float(value) for value in fields[1:7]))  # type: ignore[arg-type]
        elif line.startswith("CONECT"):
            fields = line.split()
            if len(fields) >= 3:
                for value in fields[2:]:
                    conect.append((int(fields[1]), int(value)))
        elif line.startswith(("ATOM", "HETATM")):
            if is_pqr:
                fields = line.split()
                if len(fields) < 9:
                    raise InputError(f"invalid PQR atom record: {line!r}")
                serial = int(fields[1])
                name, residue_name = fields[2], fields[3]
                if fields[4].lstrip("+-").isdigit():
                    chain = ""
                    residue_id = int(fields[4])
                    x, y, z, charge = map(float, fields[5:9])
                    element = fields[9] if len(fields) >= 10 and fields[9].isalpha() else name
                else:
                    chain = fields[4] if len(fields) > 4 else ""
                    residue_id = int(fields[5]) if fields[5].lstrip("+-").isdigit() else 1
                    x, y, z, charge = map(float, fields[6:10])
                    element = fields[10] if len(fields) >= 11 else name
            else:
                # Parse PDB records by whitespace rather than fixed columns so
                # that non-standard alignment (shifted element columns, missing
                # residue names) still yields the correct atom identity.
                fields = line.split()
                if len(fields) < 8:
                    raise InputError(f"invalid PDB atom record: {line!r}")
                serial = int(fields[1])
                name = fields[2] or f"A{len(atoms)+1}"
                rest = fields[3:]
                numeric = [position for position, token in enumerate(rest) if _is_float(token)]
                if len(numeric) >= 5:
                    coord_start = numeric[-5]
                elif len(numeric) >= 3:
                    coord_start = numeric[-3]
                else:
                    raise InputError(f"invalid PDB atom record: {line!r}")
                x, y, z = (float(rest[coord_start + offset]) for offset in range(3))
                residue_name, chain, residue_id = _pdb_residue(rest[:coord_start])
                trailing = fields[-1]
                element = trailing if trailing.isalpha() and len(trailing) <= 2 else name
                charge = 0.0
            index = len(atoms) + 1
            serial_to_index[serial] = index
            atoms.append(_atom(index, element, name, (x, y, z), residue_id=residue_id, residue_name=residue_name, chain=chain, charge=charge, charge_source="input" if is_pqr else None))
    if not atoms:
        raise InputError(f"PDB/PQR has no atoms: {path}")
    bonds: list[Bond] = []
    seen_bonds: set[tuple[int, int]] = set()
    for serial_i, serial_j in conect:
        if serial_i in serial_to_index and serial_j in serial_to_index:
            first = serial_to_index[serial_i]
            second = serial_to_index[serial_j]
            key = (min(first, second), max(first, second))
            if key not in seen_bonds:
                bonds.append(Bond(key[0], key[1], source="explicit"))
                seen_bonds.add(key)
    return Structure(_structure_name(path), atoms, bonds, cell, explicit_bonds=bool(bonds), metadata={"conect_records": len(conect)})


def load_gro(path: str | Path) -> Structure:
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        raise InputError(f"invalid GRO file: {path}")
    try:
        count = int(lines[1].strip())
    except ValueError as exc:
        raise InputError(f"invalid GRO atom count in {path}") from exc
    if len(lines) < count + 3:
        raise InputError(f"GRO file ends before {count} atom records: {path}")
    atoms: list[Atom] = []
    velocity_present = False
    for offset in range(count):
        line = lines[2 + offset]
        try:
            residue_id = int(line[0:5])
            residue_name = line[5:10].strip() or "MOL"
            name = line[10:15].strip() or f"A{offset+1}"
            index = int(line[15:20])
            x, y, z = (float(line[start:end]) * 10.0 for start, end in ((20, 28), (28, 36), (36, 44)))
            velocity_present = len(line) >= 68
        except (ValueError, IndexError):
            raise InputError(f"invalid GRO atom record: {line!r}")
        element = re.match(r"[A-Za-z]+", name)
        atoms.append(_atom(offset + 1, _element(element.group(0) if element else name), name, (x, y, z), residue_id=residue_id, residue_name=residue_name))
    fields = lines[2 + count].split()
    cell = None
    if len(fields) >= 3:
        if len(fields) == 3:
            lengths = tuple(float(value) * 10.0 for value in fields[:3])
            if all(value > 0.0 for value in lengths):
                cell = Cell.from_parameters(*lengths)  # type: ignore[arg-type]
        elif len(fields) >= 9:
            # GRO stores v1x v2y v3z v1y v1z v2x v2z v3x v3y, in nm.
            values = [float(value) * 10.0 for value in fields[:9]]
            v1 = np.array([values[0], values[3], values[4]])
            v2 = np.array([values[5], values[1], values[6]])
            v3 = np.array([values[7], values[8], values[2]])
            if abs(float(np.linalg.det(np.column_stack((v1, v2, v3))))) > 1.0e-10:
                cell = Cell(np.column_stack((v1, v2, v3)))
    return Structure(_structure_name(path), atoms, [], cell, explicit_bonds=False, metadata={"gro_velocity_columns": velocity_present})


def load_xyz(path: str | Path) -> Structure:
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise InputError(f"invalid XYZ file: {path}")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise InputError(f"invalid XYZ atom count in {path}") from exc
    if len(lines) < count + 2:
        raise InputError(f"XYZ file ends before {count} atom records: {path}")
    comment = lines[1]
    cell = _xyz_cell(comment)
    atoms: list[Atom] = []
    for index, line in enumerate(lines[2 : count + 2], start=1):
        fields = line.split()
        if len(fields) < 4:
            raise InputError(f"invalid XYZ atom record: {line!r}")
        charge = float(fields[4]) if len(fields) >= 5 else 0.0
        atoms.append(_atom(index, fields[0], fields[0] + str(index), map(float, fields[1:4]), charge=charge, charge_source="input" if len(fields) >= 5 else None))
    return Structure(_structure_name(path), atoms, [], cell, explicit_bonds=False, metadata={"xyz_comment": comment})


def _xyz_cell(comment: str) -> Cell | None:
    lattice = re.search(r"Lattice\s*=\s*[\"']([^\"']+)[\"']", comment, re.IGNORECASE)
    if lattice:
        values = [float(value) for value in lattice.group(1).split()]
        if len(values) == 9:
            return Cell(np.array(values, dtype=float).reshape(3, 3).T)
    match = re.search(r"cell\s*[:=]\s*([^;]+)", comment, re.IGNORECASE)
    if match:
        values = [float(value) for value in re.split(r"[ ,]+", match.group(1).strip()) if value]
        if len(values) >= 6:
            return Cell.from_parameters(*tuple(values[:6]))  # type: ignore[arg-type]
    return None


def load_cif(path: str | Path) -> Structure:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    values: dict[str, str] = {}
    for key, value in re.findall(r"(?im)^(_cell_[a-z_]+)\s+([^\s#]+)", "\n".join(lines)):
        values[key.lower()] = value.strip("'\"")
    required = ["_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"]
    cell = None
    if all(key in values for key in required):
        cell = Cell.from_parameters(*tuple(float(re.sub(r"\([^)]*\)", "", values[key])) for key in required))  # type: ignore[arg-type]
    loop_start = None
    for i, line in enumerate(lines):
        if line.lower() == "loop_":
            headers: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].startswith("_"):
                headers.append(lines[j].split()[0].lower())
                j += 1
            if any("atom_site" in header for header in headers):
                loop_start = (headers, j)
                break
    if loop_start is None:
        raise InputError(f"CIF has no atom_site loop: {path}")
    headers, data_start = loop_start
    def find_header(*names: str) -> int | None:
        for name in names:
            if name in headers:
                return headers.index(name)
        return None

    label_col = find_header("_atom_site_label")
    element_col = find_header("_atom_site_type_symbol")
    fx, fy, fz = (find_header(name) for name in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"))
    if fx is None or fy is None or fz is None:
        raise InputError(f"CIF atom_site loop lacks fractional coordinates: {path}")
    charge_col = find_header("_atom_site_charge")
    atoms: list[Atom] = []
    for line in lines[data_start:]:
        if line.startswith("_") or line.lower() == "loop_":
            break
        fields = line.split()
        if len(fields) < len(headers):
            continue
        label = fields[label_col] if label_col is not None else f"A{len(atoms)+1}"
        element = fields[element_col] if element_col is not None else label
        fractions = [float(re.sub(r"\([^)]*\)", "", fields[col])) for col in (fx, fy, fz)]
        if cell is None:
            raise InputError(f"CIF fractional coordinates require valid cell parameters: {path}")
        position = cell.cartesian(fractions)
        charge = float(fields[charge_col]) if charge_col is not None else 0.0
        atoms.append(_atom(len(atoms) + 1, element, label, position, charge=charge, charge_source="input" if charge_col is not None else None))
    if not atoms:
        raise InputError(f"CIF atom loop contains no usable atoms: {path}")
    bonds: list[Bond] = []
    label_to_index = {atom.name: atom.index for atom in atoms}
    for i, line in enumerate(lines):
        if line.lower() != "loop_":
            continue
        loop_headers: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j].startswith("_"):
            loop_headers.append(lines[j].split()[0].lower())
            j += 1
        first_col = "_geom_bond_atom_site_label_1"
        second_col = "_geom_bond_atom_site_label_2"
        if first_col not in loop_headers or second_col not in loop_headers:
            continue
        for row in lines[j:]:
            if row.startswith("_") or row.lower() == "loop_":
                break
            fields = row.split()
            if len(fields) < len(loop_headers):
                continue
            label_i = fields[loop_headers.index(first_col)]
            label_j = fields[loop_headers.index(second_col)]
            if label_i in label_to_index and label_j in label_to_index:
                order = 1.0
                if "_geom_bond_type" in loop_headers:
                    order = _bond_order(fields[loop_headers.index("_geom_bond_type")])
                bonds.append(Bond(label_to_index[label_i], label_to_index[label_j], order, source="explicit"))
    return Structure(_structure_name(path), atoms, bonds, cell, explicit_bonds=bool(bonds), metadata={"cif_headers": headers})


def load_structure_json(path: str | Path) -> Structure:
    import json

    raw = json.loads(Path(path).read_text())
    if "atoms" not in raw and isinstance(raw.get("structure"), dict):
        raw = raw["structure"]
    if "atoms" not in raw:
        raise InputError(f"structure JSON has no atoms: {path}")
    atoms = [_atom(int(item.get("index", i + 1)), item["element"], item.get("name", item["element"] + str(i + 1)), item["position_angstrom"], residue_id=int(item.get("residue_id", 1)), residue_name=item.get("residue_name", "MOL"), charge=float(item.get("charge", 0.0)), charge_source="input") for i, item in enumerate(raw["atoms"])]
    cell_data = raw.get("cell")
    cell = None
    if cell_data:
        if "matrix_angstrom" in cell_data:
            cell = Cell(np.asarray(cell_data["matrix_angstrom"], dtype=float), tuple(cell_data.get("pbc", (True, True, True))))
        elif "lengths_angles" in cell_data:
            lengths = tuple(float(value) for value in cell_data["lengths_angles"])
            pbc = tuple(bool(value) for value in cell_data.get("pbc", (True, True, True)))
            cell = Cell.from_parameters(*lengths, pbc=pbc)  # type: ignore[arg-type,misc]
    bonds = [Bond(int(item["i"]), int(item["j"]), float(item.get("order", 1.0)), tuple(item.get("image_shift", (0, 0, 0))), source="explicit") for item in raw.get("bonds", [])]
    return Structure(raw.get("name") or _structure_name(Path(path)), atoms, bonds, cell, declared_net_charge=float(raw.get("declared_net_charge", 0.0)), multiplicity=int(raw.get("multiplicity", 1)), explicit_bonds=bool(bonds), metadata=raw.get("metadata", {}))


# Public aliases kept small and format-oriented for library callers.
load_pdb = load_pdb_or_pqr
load_pqr = load_pdb_or_pqr
