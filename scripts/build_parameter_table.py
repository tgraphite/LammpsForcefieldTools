#!/usr/bin/env python3
"""Build the packaged GAFF parameter table from the Sobtop reference files.

Dev-only tool.  Run from the repository root:

    python scripts/build_parameter_table.py

The output src/lft/forcefields/data/gaff_bonded.json is what ships in the
wheel, so a user install needs no references/ directory.  The source file is
derived from AmberTools' gaff.dat (GAFF 1.81); the repository owner holds a
valid AmberTools license.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references" / "sobtop" / "bonded_param.dat"
LJ_SOURCE = ROOT / "references" / "sobtop" / "LJ_param.dat"
OUT = ROOT / "src" / "lft" / "forcefields" / "data" / "gaff_bonded.json"


def _parse_bonded(path: Path):
    sections: dict[str, list] = {}
    section = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("$"):
            section = line[1:].strip().upper()
            sections.setdefault(section, [])
            continue
        fields = line.split()
        token = ""
        numbers = None
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
    return sections


def main() -> int:
    sections = _parse_bonded(SOURCE)

    bonds: dict[tuple[str, ...], list[float]] = {}
    for types, numbers in sections.get("BOND", []):
        if len(types) == 2 and len(numbers) >= 2:
            bonds[types] = [numbers[0], numbers[1]]

    angles: dict[tuple[str, ...], list[float]] = {}
    for types, numbers in sections.get("ANGLE", []):
        if len(types) == 3 and len(numbers) >= 2:
            angles[types] = [numbers[0], numbers[1]]

    # Group by type tuple and keep the order of the *last* occurrence so that
    # "last matching parameter wins" is preserved exactly.
    grouped: dict[tuple[str, ...], list[list[float]]] = {}
    last_index: dict[tuple[str, ...], int] = {}
    for index, (types, numbers) in enumerate(sections.get("DIHEDRAL", [])):
        if len(types) != 4 or len(numbers) < 4:
            continue
        grouped.setdefault(types, []).append([numbers[0], numbers[1], numbers[2], int(abs(numbers[3]))])
        last_index[types] = index
    dihedrals = [[list(types), grouped[types]] for types in sorted(grouped, key=lambda t: last_index[t])]

    impropers: list[list] = []
    for types, numbers in sections.get("IMPROPER", []):
        if len(types) == 4 and len(numbers) >= 3:
            impropers.append([list(types), numbers[0], numbers[1], int(abs(numbers[2]))])

    lj: dict[str, list[float]] = {}
    for raw in LJ_SOURCE.read_text(errors="replace").splitlines():
        fields = raw.split(";", 1)[0].split()
        if len(fields) >= 3:
            try:
                lj[fields[0]] = [float(fields[1]), float(fields[2])]
            except ValueError:
                continue

    payload = {
        "schema": "lft.gaff_bonded.v1",
        "provenance": {
            "source": "references/sobtop/bonded_param.dat",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "lj_source": "references/sobtop/LJ_param.dat",
            "lj_sha256": hashlib.sha256(LJ_SOURCE.read_bytes()).hexdigest(),
            "gaff_version": "1.81",
            "note": "Derived from AmberTools gaff.dat; AmberTools license applies.",
        },
        "bonds": [[list(types), *values] for types, values in sorted(bonds.items())],
        "angles": [[list(types), *values] for types, values in sorted(angles.items())],
        "dihedrals": dihedrals,
        "impropers": impropers,
        "lj": dict(sorted(lj.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): "
        f"{len(payload['bonds'])} bonds, {len(payload['angles'])} angles, "
        f"{len(dihedrals)} dihedral groups, {len(impropers)} impropers, {len(lj)} LJ types"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
