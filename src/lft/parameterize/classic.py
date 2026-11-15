"""Prepare backend-neutral bonded parameters from topology and geometry.

The classic path has three layers:

1. deterministic starting values from the current geometry;
2. an override from the packaged GAFF parameter library (when available);
3. policy handling for terms that remain generic (missing torsions and
   impropers), plus the optional current_geometry mode that pins equilibrium
   values to the input geometry while keeping the library force constants.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..domain.parameters import (
    AngleParameter,
    BondParameter,
    DihedralParameter,
    ImproperParameter,
    ParameterSet,
)
from ..domain.structure import Structure
from ..errors import ParameterError
from ..forcefields.reference import ReferenceParameters
from .hessian import hessian_bond_scale

_GENERIC_PREFIXES = ("classic:", "generic:", "uff:")
_TORSION_POLICIES = {"error", "allow"}
_IMPROPER_POLICIES = {"omit", "amber", "k10"}


def _is_generic(source: str) -> bool:
    return source.startswith(_GENERIC_PREFIXES)


def parameterize(
    structure: Structure,
    params: ParameterSet,
    *,
    mode: str = "classic",
    hessian: np.ndarray | None = None,
    hessian_method: str = "seminario",
    current_geometry: bool = False,
    missing_torsion: str = "error",
    missing_improper: str = "omit",
) -> ParameterSet:
    if mode not in {"classic", "hessian"}:
        raise ParameterError(f"unsupported parameter mode {mode!r}")
    if missing_torsion not in _TORSION_POLICIES:
        raise ParameterError(f"unsupported missing-torsion policy {missing_torsion!r}")
    if missing_improper not in _IMPROPER_POLICIES:
        raise ParameterError(f"unsupported missing-improper policy {missing_improper!r}")
    params.parameter_mode = mode
    if mode == "hessian" and hessian is None:
        raise ParameterError("hessian parameter mode requires a Hessian artifact or a QC backend that can generate one")
    manual_value = params.metadata.get("manual_parameters", {})
    manual = manual_value if isinstance(manual_value, dict) else {}

    params.bond_parameters.clear()
    params.angle_parameters.clear()
    params.dihedral_parameters.clear()
    params.improper_parameters.clear()

    for bond in structure.bonds:
        distance = _bond_distance(structure, bond.i, bond.j, bond.image_shift)
        manual_entry = _find_manual(
            manual.get("bonds", {}),
            tuple(structure.atom(index).atom_type or "" for index in (bond.i, bond.j)),
            reverse=True,
        )
        if manual_entry is not None:
            force_constant = float(manual_entry["k"])
            distance = float(manual_entry["r0"])
            source = "manual"
        elif mode == "hessian":
            force_constant = hessian_bond_scale(structure, bond.i, bond.j, bond.image_shift, hessian, hessian_method)  # type: ignore[arg-type]
            source = f"hessian:{hessian_method}"
        else:
            force_constant = _generic_bond_k(bond.order)
            source = "uff:bond_placeholder" if _uses_uff(structure, params, (bond.i, bond.j)) else "classic:geometry_guess"
        params.bond_parameters.append(BondParameter(distance, force_constant, "harmonic", source))

    for angle in structure.angles:
        value = _angle_degrees(structure, angle.atom_images, angle.atoms)
        manual_entry = _find_manual(
            manual.get("angles", {}),
            tuple(structure.atom(index).atom_type or "" for index in angle.atoms),
            reverse=True,
        )
        if manual_entry is not None:
            params.angle_parameters.append(
                AngleParameter(float(manual_entry["theta0"]), float(manual_entry["k"]), str(manual_entry.get("function", "harmonic")), "manual")
            )
        else:
            if mode == "hessian":
                source = "generic:angle"
            elif _uses_uff(structure, params, angle.atoms):
                source = "uff:angle_placeholder"
            else:
                source = "classic:geometry_guess"
            params.angle_parameters.append(
                AngleParameter(value, 50.0 if mode == "classic" else 65.0, "harmonic", source)
            )

    for dihedral in structure.dihedrals:
        manual_entry = _find_manual(
            manual.get("dihedrals", {}),
            tuple(structure.atom(index).atom_type or "" for index in dihedral.atoms),
            reverse=True,
        )
        if manual_entry is not None:
            terms = _manual_terms(manual_entry)
            params.dihedral_parameters.append(
                DihedralParameter(terms[0][2], terms[0][0], terms[0][1], str(manual_entry.get("function", "fourier")), "manual", 1.0, terms=terms)
            )
            continue
        if _uses_uff(structure, params, dihedral.atoms):
            params.dihedral_parameters.append(
                DihedralParameter(0.0, 0.5, 2, "fourier", "uff:periodic_placeholder", 1.0, terms=((0.5, 2, 0.0),))
            )
        else:
            params.dihedral_parameters.append(
                DihedralParameter(0.0, 0.15, 3, "fourier", "classic:generic_periodic", 1.0, terms=((0.15, 3, 0.0),))
            )

    for improper in structure.impropers:
        manual_entry = _find_manual(
            manual.get("impropers", {}),
            tuple(structure.atom(index).atom_type or "" for index in improper.atoms),
            reverse=True,
        )
        if manual_entry is not None:
            params.improper_parameters.append(
                ImproperParameter(float(manual_entry.get("phase", 0.0)), float(manual_entry["k"]), int(manual_entry.get("periodicity", 2)), "cvff", "manual")
            )
        else:
            params.improper_parameters.append(ImproperParameter(0.0, 10.0, 2, "cvff", "classic:planarity_guess"))

    _assign_parameter_ids(structure)

    if params.force_field != "manual":
        reference = ReferenceParameters.discover()
        if reference is not None:
            reference.apply(structure, params, bonded=mode == "classic")

    if current_geometry:
        _apply_current_geometry(structure, params)

    omitted_torsions = _resolve_missing_torsions(structure, params, missing_torsion)
    omitted_impropers = _resolve_missing_impropers(structure, params, missing_improper)
    params.metadata["current_geometry"] = bool(current_geometry)
    params.metadata["missing_torsion_policy"] = missing_torsion
    params.metadata["missing_improper_policy"] = missing_improper
    params.metadata["omitted_terms"] = {"dihedrals": omitted_torsions, "impropers": omitted_impropers}

    _assign_parameter_ids(structure)
    params.metadata["parameter_counts"] = {
        "bonds": len(params.bond_parameters),
        "angles": len(params.angle_parameters),
        "dihedrals": len(params.dihedral_parameters),
        "impropers": len(params.improper_parameters),
    }
    params.metadata["parameter_audit"] = _audit(params)

    if params.force_field == "manual":
        if any(value.get("generic") for value in params.metadata["parameter_audit"].values()):
            raise ParameterError("manual force field requires explicit bonded parameters; generic guesses are disabled")

    for warning in _parameter_warnings(params, omitted_torsions, omitted_impropers):
        params.warnings.append(warning)
        structure.add_warning(
            str(warning["code"]),
            str(warning["message"]),
            **{key: value for key, value in warning.items() if key not in {"code", "message"}},
        )
    return params


def _generic_bond_k(order: float) -> float:
    return 300.0 if order < 1.5 else (500.0 if order < 2.5 else 800.0)


def _manual_terms(entry: dict[str, Any]) -> tuple[tuple[float, int, float], ...]:
    raw = entry.get("terms")
    if isinstance(raw, list) and raw:
        terms = []
        for item in raw:
            if isinstance(item, dict):
                terms.append((float(item["k"]), int(item.get("periodicity", 1)), float(item.get("phase", 0.0))))
            else:
                values = list(item)
                terms.append((float(values[0]), int(values[1]), float(values[2])))
        return tuple(terms)
    return ((float(entry["k"]), int(entry.get("periodicity", 1)), float(entry.get("phase", 0.0))),)


def _apply_current_geometry(structure: Structure, params: ParameterSet) -> None:
    for bond, parameter in zip(structure.bonds, params.bond_parameters):
        parameter.r0 = _bond_distance(structure, bond.i, bond.j, bond.image_shift)
    for angle, parameter in zip(structure.angles, params.angle_parameters):
        parameter.theta0 = _angle_degrees(structure, angle.atom_images, angle.atoms)


def _resolve_missing_torsions(structure: Structure, params: ParameterSet, policy: str) -> list[int]:
    if params.force_field == "manual":
        # Manual mode has its own strict check below (generic terms raise).
        return []
    keep_terms = []
    keep_parameters = []
    omitted: list[int] = []
    missing: list[str] = []
    reasons = {"uff": 0, "linear": 0, "allowed": 0}
    for index, (term, parameter) in enumerate(zip(structure.dihedrals, params.dihedral_parameters), start=1):
        if not _is_generic(parameter.source):
            keep_terms.append(term)
            keep_parameters.append(parameter)
            continue
        uff_term = _term_uses_uff(structure, params, term.atoms)
        linear = _is_linear_dihedral(structure, term)
        if policy == "allow" or linear or uff_term:
            omitted.append(index)
            if uff_term:
                reasons["uff"] += 1
            elif linear:
                reasons["linear"] += 1
            else:
                reasons["allowed"] += 1
            continue
        missing.append("-".join(structure.atom(atom).atom_type or "?" for atom in term.atoms))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise ParameterError(
            f"{len(missing)} dihedral(s) have no GAFF parameter: {preview}{suffix}",
            hint="pass --allow-missing-torsions to omit them, or supply a manual parameter file",
        )
    structure.dihedrals = keep_terms
    params.dihedral_parameters = keep_parameters
    params.metadata["omitted_dihedral_reasons"] = reasons
    return omitted


def _resolve_missing_impropers(structure: Structure, params: ParameterSet, policy: str) -> list[int]:
    if params.force_field == "manual" or policy == "k10":
        return []
    keep_terms = []
    keep_parameters = []
    omitted: list[int] = []
    for index, (term, parameter) in enumerate(zip(structure.impropers, params.improper_parameters), start=1):
        if not _is_generic(parameter.source):
            keep_terms.append(term)
            keep_parameters.append(parameter)
            continue
        if policy == "omit":
            omitted.append(index)
            continue
        parameter.k = 1.1
        parameter.phase = 180.0
        parameter.periodicity = 2
        parameter.source = "generic:amber_sp2"
        keep_terms.append(term)
        keep_parameters.append(parameter)
    structure.impropers = keep_terms
    params.improper_parameters = keep_parameters
    return omitted


def _term_uses_uff(structure: Structure, params: ParameterSet, atoms: tuple[int, ...]) -> bool:
    for atom in atoms:
        atom_type = structure.atom(atom).atom_type or ""
        entry = params.atom_types.get(atom_type)
        if entry is not None and entry.source.startswith("uff"):
            return True
    return False


def _is_linear_dihedral(structure: Structure, term: Any) -> bool:
    first = _angle_degrees(structure, (term.atom_images[0], term.atom_images[1], term.atom_images[2]), term.atoms[:3])
    second = _angle_degrees(structure, (term.atom_images[1], term.atom_images[2], term.atom_images[3]), term.atoms[1:])
    return abs(first - 180.0) < 10.0 or abs(second - 180.0) < 10.0


def _assign_parameter_ids(structure: Structure) -> None:
    for index, term in enumerate(structure.angles, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.dihedrals, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.impropers, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.pairs, start=1):
        term.parameter_id = index


def _audit(params: ParameterSet) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for kind, values in (
        ("bonds", params.bond_parameters),
        ("angles", params.angle_parameters),
        ("dihedrals", params.dihedral_parameters),
        ("impropers", params.improper_parameters),
    ):
        total = len(values)
        generic = sum(1 for value in values if _is_generic(value.source))
        audit[kind] = {"total": total, "generic": generic, "matched": total - generic}
    return audit


def _parameter_warnings(params: ParameterSet, omitted_torsions: list[int], omitted_impropers: list[int]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if omitted_torsions:
        reasons = params.metadata.get("omitted_dihedral_reasons", {})
        uff_count = int(reasons.get("uff", 0))
        other_count = len(omitted_torsions) - uff_count
        if uff_count and not other_count:
            code = "PARAMETER_UFF_TORSION_OMITTED"
            message = f"{uff_count} UFF dihedral(s) were omitted (no UFF torsion table is bundled)"
        else:
            code = "PARAMETER_TORSION_OMITTED"
            message = f"{len(omitted_torsions)} dihedral(s) without a bonded parameter were omitted"
        warnings.append({
            "code": code,
            "message": message,
            "term_kind": "dihedrals",
            "term_indices": omitted_torsions,
        })
    if omitted_impropers:
        warnings.append({
            "code": "PARAMETER_IMPROPER_OMITTED",
            "message": f"{len(omitted_impropers)} improper(s) without a GAFF parameter were omitted",
            "term_kind": "impropers",
            "term_indices": omitted_impropers,
        })
    for kind, values in (
        ("bonds", params.bond_parameters),
        ("angles", params.angle_parameters),
        ("dihedrals", params.dihedral_parameters),
        ("impropers", params.improper_parameters),
    ):
        indices = [index for index, value in enumerate(values, start=1) if value.source.startswith("classic:")]
        if indices:
            warnings.append({
                "code": "PARAMETER_GENERIC_GUESS",
                "message": f"{len(indices)} {kind} use deterministic generic parameters",
                "term_kind": kind,
                "term_indices": indices,
            })
        uff_indices = [index for index, value in enumerate(values, start=1) if value.source.startswith("uff:")]
        if uff_indices:
            warnings.append({
                "code": "PARAMETER_UFF_PLACEHOLDER",
                "message": f"{len(uff_indices)} {kind} use UFF placeholder parameters",
                "term_kind": kind,
                "term_indices": uff_indices,
            })
    return warnings


def _uses_uff(structure: Structure, params: ParameterSet, atoms: tuple[int, ...]) -> bool:
    return any(
        params.atom_types[structure.atom(index).atom_type or ""].source.startswith("uff")
        for index in atoms
    )


def _find_manual(mapping: object, actual: tuple[str, ...], *, reverse: bool) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    candidates = ["-".join(actual)]
    if reverse:
        candidates.append("-".join(reversed(actual)))
    for candidate in candidates:
        value = mapping.get(candidate)
        if isinstance(value, dict):
            return value
    return None


def lifted_position(structure: Structure, atom_index: int, image: tuple[int, int, int]) -> np.ndarray:
    position = structure.atom(atom_index).position
    if structure.cell is not None:
        return position + structure.cell.image_vector(image)
    return position.copy()


def _bond_distance(structure: Structure, i: int, j: int, shift: tuple[int, int, int]) -> float:
    vector = lifted_position(structure, j, shift) - lifted_position(structure, i, (0, 0, 0))
    return float(np.linalg.norm(vector))


def _angle_degrees(structure: Structure, images: tuple[tuple[int, int, int], ...], atoms: tuple[int, ...]) -> float:
    points = [lifted_position(structure, atom, image) for atom, image in zip(atoms, images)]
    first = points[0] - points[1]
    second = points[2] - points[1]
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1.0e-12:
        return 109.5
    cosine = float(np.dot(first, second)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _dihedral_degrees(structure: Structure, images: tuple[tuple[int, int, int], ...], atoms: tuple[int, ...]) -> float:
    p0, p1, p2, p3 = [lifted_position(structure, atom, image) for atom, image in zip(atoms, images)]
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_unit = b1 / max(np.linalg.norm(b1), 1.0e-15)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    return math.degrees(math.atan2(float(np.dot(np.cross(b1_unit, v), w)), float(np.dot(v, w))))
