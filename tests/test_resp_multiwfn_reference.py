"""Regression tests for the Multiwfn-compatible MK grid and two-stage RESP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lft.charges.esp import make_esp_grid
from lft.charges.resp import fit_resp
from lft.io.loaders import load_structure
from lft.qc.pyscf_backend import run_pyscf
from lft.topology.infer_bonds import infer_bonds

EXAMPLE = Path(__file__).resolve().parent / "fixtures" / "EC.xyz"


def _ec() -> object:
    structure = load_structure(EXAMPLE)
    structure.declared_net_charge = 0
    structure.multiplicity = 1
    infer_bonds(structure)
    return structure


def test_mk_grid_matches_multiwfn_point_count() -> None:
    # Multiwfn reports "Number of MK fitting points used: 7611" for this
    # ethylene-carbonate geometry (population.f90 setMKpt defaults).
    assert len(make_esp_grid(_ec())) == 7611


def test_resp_equivalence_and_charge_conservation() -> None:
    structure = _ec()
    wavefunction = run_pyscf(structure, qc="b3lyp/sto-3g").wavefunction
    charges, metadata = fit_resp(structure, wavefunction, 0.0)

    assert np.sum(charges) == pytest.approx(0.0, abs=1.0e-8)
    assert metadata["fit"] == "multiwfn_two_stage_resp"
    assert metadata["stage2_skipped"] is False
    assert metadata["stage2_equivalence_groups"] == [[7, 8], [9, 10]]
    assert charges[6] == pytest.approx(charges[7], abs=1.0e-8)
    assert charges[8] == pytest.approx(charges[9], abs=1.0e-8)
    assert metadata["esp_rms"] < metadata["esp_max"]
    assert metadata["esp_rms"] < 0.01
