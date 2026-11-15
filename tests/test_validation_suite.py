"""1:1 validation suite for non-periodic small molecules.

Every case has a frozen AmberTools atomtype reference (GAFF types) and a
deterministic parameter table.  Periodic/crystal cases are intentionally not
part of this suite: they have no 1:1 numeric reference yet.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lft.backends.gromacs import write_gromacs
from lft.backends.moltemplate import dihedral_terms, write_lammps
from lft.charges.dispatcher import CHARGE_METHODS
from lft.config import PipelineConfig
from lft.domain.structure import Atom, Structure
from lft.errors import ParameterError
from lft.forcefields.gaff_typer import assign_gaff_types
from lft.forcefields.matching import assign_force_field
from lft.io.loaders import load_structure
from lft.parameterize.classic import parameterize
from lft.pipeline import _check_charge_size
from lft.topology import enumerate_terms, infer_bonds

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "fixtures" / "validation"
REFERENCE = json.loads((SUITE / "reference.json").read_text())["cases"]
CASES = sorted(REFERENCE)


def _prepared(name: str):
    case = REFERENCE[name]
    structure = load_structure(SUITE / case["mol2"])
    structure.declared_net_charge = float(case["net_charge"])
    infer_bonds(structure)
    enumerate_terms(structure)
    params = assign_force_field(structure, "gaff")
    params = parameterize(structure, params, mode="classic")
    return structure, params


@pytest.mark.parametrize("name", CASES)
def test_gaff_types_match_amber_oracle(name: str) -> None:
    case = REFERENCE[name]
    structure = load_structure(SUITE / case["mol2"])
    types = assign_gaff_types(structure)
    got = {str(index): types[index] for index in sorted(types)}
    assert got == case["gaff_types"]


@pytest.mark.parametrize("name", CASES)
def test_every_term_has_a_parameter(name: str) -> None:
    structure, params = _prepared(name)
    assert len(params.bond_parameters) == len(structure.bonds)
    assert len(params.angle_parameters) == len(structure.angles)
    assert len(params.dihedral_parameters) == len(structure.dihedrals)
    assert len(params.improper_parameters) == len(structure.impropers)
    for collection in (params.bond_parameters, params.angle_parameters, params.dihedral_parameters, params.improper_parameters):
        if not collection:
            continue
        for value in collection:
            assert math.isfinite(value.k) and value.k >= 0.0
            if hasattr(value, "fourier_terms"):
                assert all(math.isfinite(term[0]) and term[0] >= 0.0 for term in value.fourier_terms())


@pytest.mark.parametrize("name", CASES)
def test_only_documented_generic_angles(name: str) -> None:
    _, params = _prepared(name)
    generic = [value for value in params.angle_parameters if value.source.startswith("classic:")]
    expected = {"caffeine": 2, "formaldehyde": 1}.get(name, 0)
    assert len(generic) == expected
    if expected:
        assert any(warning["code"] == "PARAMETER_GENERIC_GUESS" for warning in params.warnings)


@pytest.mark.parametrize("name", CASES)
def test_no_generic_bonds_or_dihedrals(name: str) -> None:
    _, params = _prepared(name)
    assert not [value for value in params.bond_parameters if value.source.startswith(("classic:", "uff:"))]
    assert not [value for value in params.dihedral_parameters if value.source.startswith(("classic:", "uff:"))]


@pytest.mark.parametrize("name", CASES)
def test_unmatched_impropers_are_omitted(name: str) -> None:
    structure, params = _prepared(name)
    assert len(params.improper_parameters) == len(structure.impropers)
    assert not [value for value in params.improper_parameters if value.source.startswith(("classic:", "uff:"))]
    omitted = params.metadata["omitted_terms"]["impropers"]
    if omitted:
        assert any(warning["code"] == "PARAMETER_IMPROPER_OMITTED" for warning in params.warnings)


def test_library_values_are_exact() -> None:
    structure, params = _prepared("ethanol")
    bond = next(
        value
        for bond, value in zip(structure.bonds, params.bond_parameters)
        if {structure.atom(bond.i).atom_type, structure.atom(bond.j).atom_type} == {"c3"}
    )
    assert bond.k == pytest.approx(300.9, abs=1e-9)
    assert bond.r0 == pytest.approx(1.5375, abs=1e-9)

    angle = next(
        value
        for term, value in zip(structure.angles, params.angle_parameters)
        if [structure.atom(index).atom_type for index in term.atoms] == ["c3", "c3", "oh"]
        or [structure.atom(index).atom_type for index in term.atoms] == ["oh", "c3", "c3"]
    )
    assert angle.k == pytest.approx(67.5, abs=1e-9)
    assert angle.theta0 == pytest.approx(110.19, abs=1e-9)

    dihedral = next(
        value
        for term, value in zip(structure.dihedrals, params.dihedral_parameters)
        if [structure.atom(index).atom_type for index in term.atoms] in (["hc", "c3", "c3", "oh"], ["oh", "c3", "c3", "hc"])
    )
    assert dihedral.fourier_terms() == ((0.0, 3, 0.0), (0.25, 1, 0.0))


def test_multi_term_dihedrals_are_preserved() -> None:
    _, params = _prepared("glucose")
    multi = [value for value in params.dihedral_parameters if len(value.fourier_terms()) > 1]
    assert len(multi) >= 10
    total_terms = sum(len(value.fourier_terms()) for value in params.dihedral_parameters)
    assert total_terms > len(params.dihedral_parameters)


def _read_lammps_sections(path: Path) -> dict[str, list[list[str]]]:
    lines = path.read_text().splitlines()
    sections: dict[str, list[list[str]]] = {}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped in {"Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"}:
            current = stripped
            sections[current] = []
            continue
        if current is not None:
            if not stripped:
                continue
            if stripped.startswith(("Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Masses", "Pair Coeffs")):
                current = None
                continue
            fields = stripped.split("#")[0].split()
            if fields and fields[0].isdigit():
                sections[current].append(fields)
    return sections


def _read_gromacs_terms(path: Path) -> dict[str, list[list[str]]]:
    lines = path.read_text().splitlines()
    sections: dict[str, list[list[str]]] = {}
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped in {"[ bonds ]", "[ angles ]", "[ dihedrals ]", "[ pairs ]"}:
            current = stripped
            sections[current] = []
            continue
        if current is not None:
            if not stripped or stripped.startswith("[") or stripped.startswith(";"):
                if stripped.startswith("["):
                    current = None
                continue
            sections[current].append(stripped.split(";")[0].split())
    return sections


def test_lammps_and_gromacs_terms_agree(tmp_path) -> None:
    structure, params = _prepared("ethanol")
    write_lammps(structure, params, tmp_path, emit_lt=False, emit_data=True, emit_image_map=False)
    write_gromacs(structure, params, tmp_path, emit_image_map=False)
    lammps = _read_lammps_sections(tmp_path / "ethanol.data")
    gromacs = _read_gromacs_terms(tmp_path / "ethanol.itp")

    # The LAMMPS data path de-duplicates types, so compare the unique
    # parameter sets rather than the per-term line counts.
    bond_lammps = np.array(sorted({(round(float(row[1]), 4), round(float(row[2]), 4)) for row in lammps["Bond Coeffs"]}))
    bond_gromacs = np.array(sorted({(round(float(row[4]) / (2.0 * 418.4), 4), round(float(row[3]) * 10.0, 4)) for row in gromacs["[ bonds ]"]}))
    assert bond_lammps.shape == bond_gromacs.shape
    assert np.allclose(bond_lammps, bond_gromacs, atol=1e-3)

    angle_lammps = np.array(sorted({(round(float(row[1]), 4), round(float(row[2]), 4)) for row in lammps["Angle Coeffs"]}))
    angle_gromacs = np.array(sorted({(round(float(row[5]) / (2.0 * 4.184), 4), round(float(row[4]), 4)) for row in gromacs["[ angles ]"]}))
    assert angle_lammps.shape == angle_gromacs.shape
    assert np.allclose(angle_lammps, angle_gromacs, atol=1e-3)

    dihedral_lammps = set()
    for row in lammps["Dihedral Coeffs"]:
        count = int(row[1])
        for term in range(count):
            dihedral_lammps.add(
                (
                    round(float(row[2 + 3 * term]) * 4.184, 3),
                    int(row[3 + 3 * term]),
                    round(float(row[4 + 3 * term]) % 360.0, 3),
                )
            )
    dihedral_gromacs = {
        (round(float(row[6]), 3), int(row[7]), round(float(row[5]) % 360.0, 3))
        for row in gromacs["[ dihedrals ]"]
        if row[4] == "9"
    }
    assert dihedral_lammps == dihedral_gromacs


@pytest.mark.parametrize(
    "method,atoms,should_raise",
    [
        ("dft-resp", 50, False),
        ("dft-resp", 51, True),
        ("resp", 200, False),
        ("resp", 201, True),
        ("am1-bcc", 500, False),
        ("am1-bcc", 501, True),
        ("mmff94", 5000, False),
    ],
)
def test_charge_size_guards(method: str, atoms: int, should_raise: bool) -> None:
    assert method in CHARGE_METHODS
    if should_raise:
        with pytest.raises(Exception):
            _check_charge_size(method[4:] if method.startswith("dft-") else method, method.startswith("dft-"), atoms)
    else:
        _check_charge_size(method[4:] if method.startswith("dft-") else method, method.startswith("dft-"), atoms)


def test_current_geometry_pins_equilibrium_values() -> None:
    structure = load_structure(SUITE / "molecules" / "ethanol.mol2")
    infer_bonds(structure)
    enumerate_terms(structure)
    params = assign_force_field(structure, "gaff")
    params = parameterize(structure, params, mode="classic", current_geometry=True)
    for bond, value in zip(structure.bonds, params.bond_parameters):
        expected = float(np.linalg.norm(structure.atom(bond.j).position - structure.atom(bond.i).position))
        assert value.r0 == pytest.approx(expected, abs=1e-9)
    assert params.metadata["current_geometry"] is True


def _unmatched_structure():
    from lft.domain.parameters import AtomType, ParameterSet

    structure = load_structure(SUITE / "molecules" / "ethanol.mol2")
    infer_bonds(structure)
    enumerate_terms(structure)
    for atom in structure.atoms:
        atom.atom_type = "zz"
    params = ParameterSet(
        force_field="gaff",
        parameter_mode="classic",
        atom_types={"zz": AtomType("zz", "C", 12.0, 3.4, 0.1, "gaff")},
    )
    return structure, params


def test_missing_torsion_policy() -> None:
    structure, params = _unmatched_structure()
    with pytest.raises(ParameterError):
        parameterize(structure, params, mode="classic")

    structure, params = _unmatched_structure()
    result = parameterize(structure, params, mode="classic", missing_torsion="allow")
    assert result.metadata["omitted_terms"]["dihedrals"]
    assert len(result.dihedral_parameters) == len(structure.dihedrals)


def test_improper_policy_amber() -> None:
    structure = load_structure(SUITE / "molecules" / "thiophene.mol2")
    infer_bonds(structure)
    enumerate_terms(structure)
    params = assign_force_field(structure, "gaff")
    params = parameterize(structure, params, mode="classic", missing_improper="amber")
    assert len(params.improper_parameters) == len(structure.impropers)
    assert all(math.isclose(value.k, 1.1) for value in params.improper_parameters)
    assert all(value.source == "generic:amber_sp2" for value in params.improper_parameters)


def test_manual_parameters_can_override_multi_term() -> None:
    from lft.parameterize.classic import _manual_terms

    assert _manual_terms({"terms": [[0.1, 3, 0.0], [0.2, 2, 180.0]]}) == ((0.1, 3, 0.0), (0.2, 2, 180.0))
