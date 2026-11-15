"""Deterministic GAFF/UFF/manual atom typing."""

from __future__ import annotations

from ..domain.parameters import AtomType, ParameterSet
from ..domain.structure import Structure
from ..domain.units import sigma_from_rmin
from ..errors import ParameterError
from .atomtype import (
    run_atomtype,
    run_python_atomtype,
    use_external_atomtype,
)
from .tables import ATOMIC_NUMBER, ELEMENT_MASS, GAFF_ION_LJ, UFF_LJ

GAFF_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}


def assign_force_field(
    structure: Structure,
    force_field: str,
    *,
    manual_types: dict[int, str] | None = None,
    manual_atom_types: dict[str, dict[str, float]] | None = None,
    source_path: str | None = None,
    workdir: str | None = None,
) -> ParameterSet:
    """Assign atom types and create the non-bonded portion of a parameter set."""

    if force_field not in {"gaff", "gaff+uff", "uff", "manual"}:
        raise ParameterError(f"unsupported force field {force_field!r}")
    manual_types = manual_types or {}
    manual_atom_types = manual_atom_types or {}
    params = ParameterSet(force_field=force_field, parameter_mode="classic")
    external_types: dict[int, str] = {}
    if force_field in {"gaff", "gaff+uff"}:
        if use_external_atomtype():
            if workdir is None:
                raise ParameterError(
                    "external atomtype validation requires an explicit workdir"
                )
            external_types, atomtype_metadata = run_atomtype(structure, workdir=workdir)
        else:
            external_types, atomtype_metadata = run_python_atomtype(structure)
        if set(external_types) != {atom.index for atom in structure.atoms}:
            raise ParameterError("GAFF engine returned an incomplete atom type mapping")
        params.metadata["typing_backend"] = atomtype_metadata
    for atom in structure.atoms:
        if force_field == "manual":
            type_name = manual_types.get(atom.index)
            if not type_name:
                raise ParameterError(
                    f"manual force field has no type for atom {atom.index} ({atom.name})",
                    hint="provide --manual-file with an atoms mapping",
                )
            source = "manual"
        elif force_field == "uff":
            type_name = uff_type(atom.element)
            source = "uff"
        else:
            type_name = external_types[atom.index]
            source = "gaff"
            if atom.element not in GAFF_ELEMENTS and force_field == "gaff":
                raise ParameterError(
                    f"GAFF has no supported type for element {atom.element}"
                )
            # GAFF acceptance: lone-pair / dummy types are not genuine GAFF
            # types and must fall back to UFF or raise under strict gaff.
            # Element-symbol fallbacks (e.g. Zn/Fe) are preserved exactly like
            # the historical AmberTools atomtype behaviour.
            if type_name is not None and type_name.lower() in {
                "du",
                "dum",
                "lp",
                "lone",
                "dummy",
                "xx",
                "none",
                "unknown",
            }:
                type_name = None
            if type_name is None:
                if force_field == "gaff":
                    raise ParameterError(
                        f"GAFF has no supported type for atom {atom.index} ({atom.element})",
                        hint="use --ff gaff+uff, --ff uff, or provide a manual type",
                    )
                type_name = uff_type(atom.element)
                source = "uff_fallback"
                warning = {
                    "code": "FF_UFF_FALLBACK",
                    "message": f"atom {atom.index} uses UFF fallback",
                    "atom": atom.index,
                    "element": atom.element,
                }
                params.warnings.append(warning)
                structure.add_warning(
                    str(warning["code"]),
                    str(warning["message"]),
                    atom=atom.index,
                    element=atom.element,
                )
        atom.atom_type = type_name
        atom.mass = atom.mass or ELEMENT_MASS.get(atom.element, 0.0)
        if atom.mass <= 0.0:
            raise ParameterError(
                f"no atomic mass is available for atom {atom.index} ({atom.element})"
            )
        if type_name not in params.atom_types:
            params.atom_types[type_name] = make_atom_type(
                atom.element, type_name, source, manual_atom_types.get(type_name)
            )
        atom.mass = params.atom_types[type_name].mass
    params.metadata["typing"] = {
        str(atom.index): {
            "type": atom.atom_type,
            "source": _type_source(params, atom.atom_type),
        }
        for atom in structure.atoms
    }
    return params


def _type_source(params: ParameterSet, type_name: str | None) -> str | None:
    if type_name is None:
        return None
    return params.atom_types[type_name].source


def uff_type(element: str) -> str:
    if element not in UFF_LJ:
        raise ParameterError(f"UFF table has no parameters for element {element!r}")
    return f"UF_{element}"


def make_atom_type(
    element: str, type_name: str, source: str, override: dict[str, float] | None = None
) -> AtomType:
    values = override or {}
    if source == "manual" and override is None:
        raise ParameterError(
            f"manual atom type {type_name!r} has no explicit parameters"
        )
    if source == "manual" and not {"sigma", "epsilon"}.issubset(values):
        raise ParameterError(
            f"manual atom type {type_name!r} must specify sigma and epsilon"
        )
    ion = GAFF_ION_LJ.get(element) if source == "gaff" else None
    if ion is not None:
        # AMBER ion parameters are already LJ sigma/epsilon (Angstrom, kcal/mol).
        sigma, epsilon = ion
    else:
        rmin, epsilon = UFF_LJ.get(element, (1.75, 0.05))
        sigma = sigma_from_rmin(rmin)
    sigma = float(values.get("sigma", sigma))
    epsilon = float(values.get("epsilon", epsilon))
    if sigma <= 0.0 or epsilon < 0.0:
        raise ParameterError(f"invalid LJ parameters for atom type {type_name!r}")
    mass = float(values.get("mass", ELEMENT_MASS.get(element, 0.0)))
    return AtomType(
        type_name, element, mass, sigma, epsilon, source, ATOMIC_NUMBER.get(element, 0)
    )


def gaff_type(structure: Structure, index: int) -> str | None:
    """Compatibility API backed by the same full-structure GAFF rule engine."""
    atom = structure.atom(index)
    if atom.element not in GAFF_ELEMENTS:
        return None
    types, _ = run_python_atomtype(structure)
    name = types[index]
    return None if name.lower() in {"du", "lp"} else name
