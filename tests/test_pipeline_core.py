from __future__ import annotations

import json

import numpy as np
import pytest

from lft.backends.function_map import gromacs_periodic_energy, lammps_fourier_energy
from lft.charges.dispatcher import assign_charges
from lft.config import PipelineConfig
from lft.domain.structure import Atom, Cell, Structure
from lft.domain.topology import Bond
from lft.errors import ChargeError, ParameterError
from lft.forcefields.matching import assign_force_field
from lft.io.loaders import load_structure
from lft.pipeline import run_pipeline
from lft.topology import compute_winding, enumerate_terms, infer_bonds


def _periodic_triangle() -> Structure:
    cell = Cell.from_parameters(2.0, 2.0, 2.0)
    atoms = [
        Atom(1, "C", "C1", np.array([0.1, 0.1, 0.1]), mass=12.0),
        Atom(2, "C", "C2", np.array([0.8, 0.1, 0.1]), mass=12.0),
        Atom(3, "C", "C3", np.array([1.4, 0.1, 0.1]), mass=12.0),
    ]
    bonds = [Bond(1, 2), Bond(2, 3), Bond(1, 3, image_shift=(1, 0, 0))]
    return Structure("triangle", atoms, bonds, cell, explicit_bonds=True)


def test_periodic_image_lift_and_winding() -> None:
    structure = _periodic_triangle()
    enumerate_terms(structure)
    assert compute_winding(structure)["rank"] == 1
    assert any(any(term.atom_images[0]) for term in structure.angles)


def test_xyz_inference_and_loader(tmp_path) -> None:
    path = tmp_path / "water.xyz"
    path.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    structure = load_structure(path)
    infer_bonds(structure)
    assert len(structure.bonds) == 2
    assert {structure.atom(bond.i).element for bond in structure.bonds} == {"O"}


def test_pqr_gro_and_cif_loaders(tmp_path) -> None:
    pqr = tmp_path / "water.pqr"
    pqr.write_text("ATOM      1 O    HOH 1 0.000 0.000 0.000 -0.8 1.5 O\nATOM      2 H1   HOH 1 0.960 0.000 0.000 0.4 1.0 H\nATOM      3 H2   HOH 1 -0.240 0.930 0.000 0.4 1.0 H\n")
    pqr_structure = load_structure(pqr)
    assert pqr_structure.atom(1).charge == pytest.approx(-0.8)
    gro = tmp_path / "water.gro"
    gro.write_text("water\n3\n    1HOH      O    1   0.000   0.000   0.000\n    1HOH     H1    2   0.096   0.000   0.000\n    1HOH     H2    3  -0.024   0.093   0.000\n   2.00000   2.00000   2.00000\n")
    gro_structure = load_structure(gro)
    assert gro_structure.cell is not None
    assert gro_structure.atom(2).position[0] == pytest.approx(0.96)
    cif = tmp_path / "carbon.cif"
    cif.write_text("data_carbon\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\n")
    cif_structure = load_structure(cif)
    assert cif_structure.atom(1).position.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_charge_tolerance_boundaries(tmp_path) -> None:
    structure = Structure("tol", [Atom(1, "H", "H1", np.zeros(3), mass=1.0), Atom(2, "H", "H2", np.ones(3), mass=1.0)])
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps([1.0e-5, 0.0]))
    assign_charges(structure, "input", net_charge=0, charge_file=accepted)
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps([1.0001e-5, 0.0]))
    with pytest.raises(ChargeError):
        assign_charges(structure, "input", net_charge=0, charge_file=rejected)


def test_charged_cm5_is_raw_and_warns(tmp_path) -> None:
    structure = Structure("ion", [Atom(1, "H", "H1", np.zeros(3), mass=1.0), Atom(2, "H", "H2", np.ones(3), mass=1.0)])
    charges = tmp_path / "charges.json"
    charges.write_text(json.dumps([0.4, 0.6]))
    with pytest.warns(RuntimeWarning, match="CHARGE_CM5_CHARGED_FALLBACK"):
        metadata = assign_charges(structure, "cm5", net_charge=1, charge_file=charges)
    assert metadata["effective_method"] == "raw_cm5"
    assert sum(atom.charge for atom in structure.atoms) == pytest.approx(1.0)
    assert structure.warnings[0]["fallback_reason"] == "charged_system"


def test_function_mapping_is_identical() -> None:
    assert gromacs_periodic_energy(37.0, 0.0, 0.15, 3) == pytest.approx(lammps_fourier_energy(37.0, 0.0, 0.15, 3))


def test_gaff_does_not_silently_type_metals() -> None:
    structure = Structure("metal", [Atom(1, "Fe", "Fe", np.zeros(3), mass=55.845)])
    with pytest.raises(ParameterError):
        assign_force_field(structure, "gaff")


def test_manual_type_and_bond_parameters(tmp_path) -> None:
    structure_path = tmp_path / "manual.xyz"
    structure_path.write_text("2\nmanual\nC 0 0 0 0\nH 1.1 0 0 0\n")
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps({
        "atoms": {"1": "Cmanual", "2": "Hmanual"},
        "types": {
            "Cmanual": {"mass": 12.0, "sigma": 3.4, "epsilon": 0.1},
            "Hmanual": {"mass": 1.0, "sigma": 2.5, "epsilon": 0.02},
        },
        "bonds": {"Cmanual-Hmanual": {"r0": 1.1, "k": 300.0}},
    }))
    result = run_pipeline(PipelineConfig(structure_path, force_field="manual", charge="input", manual_file=manual_path, outdir=tmp_path / "out"))
    assert result.parameters.bond_parameters[0].source == "manual"


def test_pipeline_writes_runnable_artifacts(tmp_path) -> None:
    structure_path = tmp_path / "methane_fragment.xyz"
    structure_path.write_text("2\nC-H\nC 0 0 0 0.0\nH 1.09 0 0 0.0\n")
    gromacs = run_pipeline(PipelineConfig(structure_path, output="gromacs", charge="input", outdir=tmp_path / "gromacs"))
    assert all(gromacs.artifacts[key].exists() for key in ("itp", "top", "gro", "mdp"))
    lammps = run_pipeline(PipelineConfig(structure_path, output="lammps", charge="input", outdir=tmp_path / "lammps"))
    assert all(lammps.artifacts[key].exists() for key in ("data", "input"))
    assert "lt" not in lammps.artifacts
    assert "normalized_ir" not in lammps.artifacts


def test_debug_bundle_normalized_json_can_be_loaded_again(tmp_path) -> None:
    structure_path = tmp_path / "roundtrip.xyz"
    structure_path.write_text("2\nroundtrip\nC 0 0 0 0\nH 1.09 0 0 0\n")
    result = run_pipeline(PipelineConfig(structure_path, charge="input", outdir=tmp_path / "roundtrip-out", debug=True))
    assert result.artifacts["debug"].name == "debug.txt"
    restored = load_structure(result.artifacts["normalized_ir"])
    assert restored.natoms == 2
    assert len(restored.bonds) == 1
