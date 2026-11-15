"""Prepare backend-neutral bonded parameters from topology and geometry."""

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


def parameterize(structure: Structure, params: ParameterSet, *, mode: str = "classic", hessian: np.ndarray | None = None, hessian_method: str = "seminario") -> ParameterSet:
    if mode not in {"classic", "hessian"}:
        raise ParameterError(f"unsupported parameter mode {mode!r}")
    params.parameter_mode = mode
    if mode == "hessian" and hessian is None:
        raise ParameterError("hessian parameter mode requires a Hessian artifact or a QC backend that can generate one")
    hessian_value = hessian
    manual_value = params.metadata.get("manual_parameters", {})
    manual = manual_value if isinstance(manual_value, dict) else {}
    params.bond_parameters.clear()
    params.angle_parameters.clear()
    params.dihedral_parameters.clear()
    params.improper_parameters.clear()
    for bond in structure.bonds:
        distance = _bond_distance(structure, bond.i, bond.j, bond.image_shift)
        manual_value = _find_manual(manual.get("bonds", {}), tuple(structure.atom(index).atom_type or "" for index in (bond.i, bond.j)), reverse=True)
        if manual_value is not None:
            force_constant = float(manual_value["k"])
            distance = float(manual_value["r0"])
            source = "manual"
        elif mode == "hessian":
            force_constant = hessian_bond_scale(structure, bond.i, bond.j, bond.image_shift, hessian_value, hessian_method)  # type: ignore[arg-type]
            source = f"hessian:{hessian_method}"
        else:
            force_constant = 300.0 if bond.order < 1.5 else (500.0 if bond.order < 2.5 else 800.0)
            source = "classic:geometry_guess"
        params.bond_parameters.append(BondParameter(distance, force_constant, "harmonic", source))
    for angle in structure.angles:
        value = _angle_degrees(structure, angle.atom_images, angle.atoms)
        manual_value = _find_manual(manual.get("angles", {}), tuple(structure.atom(index).atom_type or "" for index in angle.atoms), reverse=True)
        if manual_value is not None:
            params.angle_parameters.append(AngleParameter(float(manual_value["theta0"]), float(manual_value["k"]), str(manual_value.get("function", "harmonic")), "manual"))
        else:
            params.angle_parameters.append(AngleParameter(value, 50.0 if mode == "classic" else 65.0, "harmonic", "classic:geometry_guess" if mode == "classic" else f"hessian:{hessian_method}"))
    for dihedral in structure.dihedrals:
        manual_value = _find_manual(manual.get("dihedrals", {}), tuple(structure.atom(index).atom_type or "" for index in dihedral.atoms), reverse=True)
        if manual_value is not None:
            params.dihedral_parameters.append(DihedralParameter(float(manual_value.get("phase", 0.0)), float(manual_value["k"]), int(manual_value.get("periodicity", 1)), str(manual_value.get("function", "fourier")), "manual"))
            continue
        if _uses_uff(structure, params, dihedral.atoms):
            # UFF/conjugated placeholder.  It is kept as the "harmonic" tag only
            # so the reference table lookup uses the conjugated-dihedral set;
            # both writers emit it as an equivalent single periodic term
            # (phase 0/180) with no hybrid style.
            function = "harmonic"
            phase = 0.0
            periodicity = 2
            k = 0.50
        else:
            function = "fourier"
            phase = 0.0
            periodicity = 3
            k = 0.15
        params.dihedral_parameters.append(DihedralParameter(phase, k, periodicity, function, "classic:generic_periodic" if mode == "classic" else f"hessian:{hessian_method}"))
    for _improper in structure.impropers:
        manual_value = _find_manual(manual.get("impropers", {}), tuple(structure.atom(index).atom_type or "" for index in _improper.atoms), reverse=True)
        if manual_value is not None:
            params.improper_parameters.append(ImproperParameter(float(manual_value.get("phase", 0.0)), float(manual_value["k"]), int(manual_value.get("periodicity", 2)), "cvff", "manual"))
        else:
            params.improper_parameters.append(ImproperParameter(0.0, 10.0, 2, "cvff", "classic:planarity_guess" if mode == "classic" else f"hessian:{hessian_method}"))
    for index, term in enumerate(structure.angles, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.dihedrals, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.impropers, start=1):
        term.parameter_id = index
    for index, term in enumerate(structure.pairs, start=1):
        term.parameter_id = index
    params.metadata["parameter_counts"] = {
        "bonds": len(params.bond_parameters),
        "angles": len(params.angle_parameters),
        "dihedrals": len(params.dihedral_parameters),
        "impropers": len(params.improper_parameters),
    }
    if params.force_field != "manual":
        reference = ReferenceParameters.discover()
        if reference is not None:
            reference.apply(structure, params, bonded=mode == "classic")
    generic_terms = {
        "bonds": [index + 1 for index, value in enumerate(params.bond_parameters) if value.source.startswith("classic:")],
        "angles": [index + 1 for index, value in enumerate(params.angle_parameters) if value.source.startswith("classic:")],
        "dihedrals": [index + 1 for index, value in enumerate(params.dihedral_parameters) if value.source.startswith("classic:")],
        "impropers": [index + 1 for index, value in enumerate(params.improper_parameters) if value.source.startswith("classic:")],
    }
    if params.force_field == "manual" and any(generic_terms.values()):
        raise ParameterError("manual force field requires explicit bonded parameters; generic guesses are disabled")
    for kind, indices in generic_terms.items():
        if indices:
            params.warnings.append({"code": "PARAMETER_GENERIC_GUESS", "message": f"{len(indices)} {kind} use deterministic generic parameters", "term_kind": kind, "term_indices": indices})
    return params


def _uses_uff(structure: Structure, params: ParameterSet, atoms: tuple[int, ...]) -> bool:
    return any(
        params.atom_types[structure.atom(index).atom_type or ""].source in {"uff", "uff_fallback"}
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
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))  # type: ignore[arg-type]


def _dihedral_degrees(structure: Structure, images: tuple[tuple[int, int, int], ...], atoms: tuple[int, ...]) -> float:
    p0, p1, p2, p3 = [lifted_position(structure, atom, image) for atom, image in zip(atoms, images)]
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    b1_unit = b1 / max(np.linalg.norm(b1), 1.0e-15)
    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit
    return math.degrees(math.atan2(float(np.dot(np.cross(b1_unit, v), w)), float(np.dot(v, w))))
