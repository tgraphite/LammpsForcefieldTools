from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lft.charges.density_charges import adch_charges, cm5_charges, hirshfeld_charges, normalize_populations
from lft.domain.structure import Atom, Structure
from lft.io.loaders import load_structure
from lft.qc.pyscf_backend import run_pyscf

REFERENCE = json.loads((Path(__file__).parent / "ground_truth" / "charges_multiwfn.json").read_text())["molecules"]


def _structure_from_geometry(geometry: list) -> Structure:
    atoms = [Atom(i + 1, element, f"{element}{i + 1}", np.asarray(position, dtype=float)) for i, (element, position) in enumerate(geometry)]
    structure = Structure(f"mol{len(atoms)}", atoms)
    structure.declared_net_charge = 0
    structure.multiplicity = 1
    return structure


@pytest.mark.parametrize("molecule", ["water"])
def test_charges_match_multiwfn_reference(molecule: str) -> None:
    case = REFERENCE[molecule]
    structure = _structure_from_geometry(case["geometry_angstrom"])
    wf = run_pyscf(structure, qc="b3lyp/sto-3g").wavefunction
    hirshfeld, dipoles, _ = hirshfeld_charges(wf)
    assert np.abs(np.asarray(hirshfeld) - np.asarray(case["multiwfn_hirshfeld"])).max() < 1.0e-5

    cm5, _ = cm5_charges(hirshfeld, wf)
    assert np.abs(normalize_populations(cm5, wf) - np.asarray(case["multiwfn_cm5"])).max() < 1.0e-5

    adch, _ = adch_charges(hirshfeld, dipoles, wf)
    assert np.abs(normalize_populations(adch, wf) - np.asarray(case["multiwfn_adch"])).max() < 1.0e-5


@pytest.mark.parametrize("molecule", ["methyl_benzoate"])
def test_cm5_adch_match_multiwfn_reference(molecule: str) -> None:
    case = REFERENCE[molecule]
    structure = load_structure(case["mol2"])
    structure.declared_net_charge = 0
    structure.multiplicity = 1
    wf = run_pyscf(structure, qc="b3lyp/sto-3g").wavefunction
    hirshfeld, dipoles, _ = hirshfeld_charges(wf)
    cm5, _ = cm5_charges(hirshfeld, wf)
    assert np.abs(normalize_populations(cm5, wf) - np.asarray(case["multiwfn_cm5"])).max() < 1.0e-5
    adch, _ = adch_charges(hirshfeld, dipoles, wf)
    assert np.abs(normalize_populations(adch, wf) - np.asarray(case["multiwfn_adch"])).max() < 1.0e-5