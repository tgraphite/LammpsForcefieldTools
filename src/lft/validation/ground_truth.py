"""Ground-truth manifest loading and normalized count checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audit import sha256_file
from ..io.serializers import load_json
from ..io.loaders import load_structure
from ..topology import compute_winding, enumerate_terms, infer_bonds


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(path.read_text())
    return load_json(path)


def compare_expected_counts(ir_path: str | Path, expected: dict[str, int]) -> dict[str, tuple[int | None, int]]:
    raw = load_json(ir_path)
    structure = raw.get("structure", raw)
    actual = {"atoms": len(structure.get("atoms", [])), "bonds": len(structure.get("bonds", [])), "angles": len(structure.get("angles", [])), "dihedrals": len(structure.get("dihedrals", [])), "impropers": len(structure.get("impropers", [])), "pairs": len(structure.get("pairs", []))}
    return {key: (actual.get(key), value) for key, value in expected.items()}


def validate_manifest(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Run the structural part of a GT manifest without changing references."""

    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    base = Path(root) if root is not None else manifest_path.parents[3]
    input_path = Path(manifest["input"])
    if not input_path.is_absolute():
        input_path = base / input_path
    actual_hash = sha256_file(input_path)
    expected_hash = manifest.get("input_hash")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"{manifest.get('id', manifest_path)} input hash mismatch: {actual_hash} != {expected_hash}")
    structure = load_structure(input_path)
    infer_bonds(structure)
    enumerate_terms(structure)
    winding = compute_winding(structure)
    actual = {"atoms": structure.natoms, "bonds": len(structure.bonds), "angles": len(structure.angles), "dihedrals": len(structure.dihedrals), "impropers": len(structure.impropers), "cross_edges": sum(bond.image_shift != (0, 0, 0) for bond in structure.bonds), "winding_rank": winding["rank"]}
    expected = manifest.get("expected_counts", {})
    mismatches = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ValueError(f"{manifest.get('id', manifest_path)} expected-count mismatch: {mismatches}")
    return {"id": manifest.get("id", manifest_path.stem), "input": str(input_path), "input_hash": actual_hash, "actual": actual, "winding": winding}
