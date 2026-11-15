"""LAMMPS output-mode invariants added in 0.2.0."""

from __future__ import annotations

import numpy as np

from lft.backends.moltemplate import write_lammps
from lft.config import PipelineConfig
from lft.domain.parameters import AtomType, ParameterSet
from lft.domain.structure import Atom, Cell, Structure
from lft.domain.topology import Bond
from lft.pipeline import _ChargeSpec, _preopt_method
from lft.topology import enumerate_terms


def _water_box(net_charge: float = 0.0) -> Structure:
    atoms = [
        Atom(1, "O", "O", np.array([0.0, 0.0, 0.0]), charge=-0.834, atom_type="oh"),
        Atom(2, "H", "H1", np.array([0.957, 0.0, 0.0]), charge=0.417, atom_type="ho"),
        Atom(3, "H", "H2", np.array([-0.24, 0.927, 0.0]), charge=0.417, atom_type="ho"),
    ]
    bonds = [Bond(1, 2), Bond(1, 3)]
    structure = Structure("waterbox", atoms, bonds, Cell.from_parameters(20.0, 20.0, 20.0), explicit_bonds=True)
    enumerate_terms(structure)
    return structure


def _params(structure: Structure) -> ParameterSet:
    from lft.domain.parameters import AngleParameter, BondParameter

    params = ParameterSet(
        "gaff",
        "classic",
        atom_types={
            "oh": AtomType("oh", "O", 16.0, 3.15, 0.15, "gaff"),
            "ho": AtomType("ho", "H", 1.008, 1.0, 0.02, "gaff"),
        },
    )
    params.bond_parameters = [BondParameter(0.957, 553.0) for _ in structure.bonds]
    params.angle_parameters = [AngleParameter(104.52, 100.0)]
    return params


def test_periodic_uses_pppm_after_read_data(tmp_path) -> None:
    structure = _water_box()
    write_lammps(structure, _params(structure), tmp_path, emit_lt=False, emit_data=True, emit_image_map=False)
    text = (tmp_path / "waterbox.in").read_text()
    assert "pair_style lj/cut/coul/long 10.0" in text
    assert "kspace_style pppm 1.0e-4" in text
    assert text.index("read_data") < text.index("kspace_style")


def test_orthogonal_box_has_no_tilt_line(tmp_path) -> None:
    structure = _water_box()
    write_lammps(structure, _params(structure), tmp_path, emit_lt=False, emit_data=True, emit_image_map=False)
    text = (tmp_path / "waterbox.data").read_text()
    assert "xy xz yz" not in text


def test_charged_periodic_falls_back_to_cutoff(tmp_path) -> None:
    structure = _water_box()
    structure.atoms[0].charge = -1.834  # net -1
    write_lammps(structure, _params(structure), tmp_path, emit_lt=False, emit_data=True, emit_image_map=False)
    text = (tmp_path / "waterbox.in").read_text()
    assert "pair_style lj/cut/coul/cut 10.0" in text
    assert "kspace_style" not in text
    assert any(warning["code"] == "PERIODIC_CHARGED_CUTOFF" for warning in structure.warnings)


def test_types_are_deduplicated(tmp_path) -> None:
    from lft.io.loaders import load_structure
    from lft.forcefields.matching import assign_force_field
    from lft.parameterize.classic import parameterize
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    structure = load_structure(root / "tests" / "fixtures" / "validation" / "molecules" / "ethanol.mol2")
    enumerate_terms(structure)
    params = parameterize(structure, assign_force_field(structure, "gaff"), mode="classic")
    write_lammps(structure, params, tmp_path, emit_lt=False, emit_data=True, emit_image_map=False)
    lines = (tmp_path / "ethanol.data").read_text().splitlines()
    start = lines.index("Dihedral Coeffs")
    unique_dihedral_lines = []
    for line in lines[start + 2 :]:
        if not line.strip():
            break
        unique_dihedral_lines.append(line)
    assert len(unique_dihedral_lines) < len(structure.dihedrals)


def test_preopt_mapping() -> None:
    auto = PipelineConfig("x.xyz", preopt="auto")
    assert _preopt_method(auto, _ChargeSpec("mmff94")) == "mmff94"
    assert _preopt_method(auto, _ChargeSpec("resp")) == "uff"
    assert _preopt_method(auto, _ChargeSpec("am1-bcc")) == "uff"
    assert _preopt_method(auto, _ChargeSpec("input")) == "none"
    assert _preopt_method(auto, _ChargeSpec("resp", dft_geometry=True)) == "dft"
    forced = PipelineConfig("x.xyz", preopt="none")
    assert _preopt_method(forced, _ChargeSpec("resp")) == "none"
