from __future__ import annotations

import numpy as np
import pytest

from lft.charges.density_charges import adch_charges, cm5_charges, hirshfeld_charges, mulliken_charges, normalize_populations
from lft.charges._cm5_data import cm5_tval
from lft.charges.esp import esp_from_wavefunction, make_esp_grid
from lft.charges.resp import fit_resp
from lft.domain.structure import Atom, Structure
from lft.qc.pyscf_backend import run_pyscf


def _water() -> tuple[Structure, object]:
    atoms = [
        Atom(1, "O", "O", np.array([0.0, 0.0, 0.0])),
        Atom(2, "H", "H1", np.array([0.96, 0.0, 0.0])),
        Atom(3, "H", "H2", np.array([-0.24, 0.93, 0.0])),
    ]
    structure = Structure("water", atoms)
    structure.declared_net_charge = 0
    structure.multiplicity = 1
    return structure, run_pyscf(structure, qc="b3lyp/sto-3g").wavefunction


def test_mulliken_matches_pyscf_reference() -> None:
    _, wf = _water()
    charges, metadata = mulliken_charges(wf)
    assert np.sum(charges) == pytest.approx(0.0, abs=1.0e-12)
    assert charges[0] == pytest.approx(-0.3657, abs=1.0e-3)
    assert charges[1] == pytest.approx(charges[2], abs=1.0e-3)
    assert metadata["model"] == "mulliken"


def test_hirshfeld_sums_and_symmetry() -> None:
    _, wf = _water()
    charges, dipoles, metadata = hirshfeld_charges(wf)
    assert abs(charges.sum()) < 1.0e-3
    assert charges[1] == pytest.approx(charges[2], abs=1.0e-4)
    assert charges[0] < 0.0 < charges[1]
    assert metadata["grid"]["angular"] == 434


def test_hirshfeld_dipoles_reproduce_molecular_dipole() -> None:
    structure, wf = _water()
    charges, dipoles, _ = hirshfeld_charges(wf)
    charge_dipole = (structure.coordinates / 0.529177210903 * charges[:, None]).sum(axis=0)
    total = charge_dipole + dipoles.sum(axis=0)
    electronic = -np.einsum("xuv,uv->x", np.asarray(wf.mol.intor("int1e_r"), dtype=float), wf.total_dm)
    nuclear = (wf.nuclear_charges[:, None] * wf.coordinates_bohr).sum(axis=0)
    reference = electronic + nuclear
    assert np.linalg.norm(total - reference) < 1.0e-2


def test_cm5_coefficients() -> None:
    assert cm5_tval(1, 6) == pytest.approx(0.0502)
    assert cm5_tval(6, 1) == pytest.approx(-0.0502)
    assert cm5_tval(8, 1) == pytest.approx(-0.1671)
    assert cm5_tval(6, 8) == pytest.approx(0.0234)
    assert cm5_tval(1, 8) == pytest.approx(0.1671)


def test_cm5_increases_polarization_and_conserves_charge() -> None:
    _, wf = _water()
    hirshfeld, _, _ = hirshfeld_charges(wf)
    cm5, metadata = cm5_charges(hirshfeld, wf)
    assert metadata["alpha"] == pytest.approx(2.474)
    assert abs(cm5.sum() - hirshfeld.sum()) < 1.0e-9
    assert abs(cm5[0]) > abs(hirshfeld[0])


def test_normalize_populations_exact_sum() -> None:
    _, wf = _water()
    hirshfeld, _, _ = hirshfeld_charges(wf)
    cm5, _ = cm5_charges(hirshfeld, wf)
    normalized = normalize_populations(cm5, wf)
    assert np.sum(normalized) == pytest.approx(0.0, abs=1.0e-12)


def test_adch_conserves_charge() -> None:
    _, wf = _water()
    hirshfeld, dipoles, _ = hirshfeld_charges(wf)
    adch, metadata = adch_charges(hirshfeld, dipoles, wf)
    assert metadata["model"] == "adch"
    assert abs(adch.sum() - hirshfeld.sum()) < 1.0e-9


def test_resp_fits_wavefunction_esp_and_conserves_charge() -> None:
    structure, wf = _water()
    charges, metadata = fit_resp(structure, wf, 0.0)
    assert np.sum(charges) == pytest.approx(0.0, abs=1.0e-10)
    assert metadata["esp_rms"] < 0.01
    points = make_esp_grid(structure)
    target = esp_from_wavefunction(wf, points)
    assert np.all(np.isfinite(target))


def test_pipeline_cm5_via_qc(tmp_path) -> None:
    from lft.config import PipelineConfig
    from lft.pipeline import run_pipeline

    mol2 = tmp_path / "water.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nwater\n3 2 1 0 0\nSMALL\nUSER_CHARGES\n\n"
        "@<TRIPOS>ATOM\n     1 O1   0.0000   0.0000   0.0000 O   1 WAT  0.0000\n"
        "     2 H1   0.9600   0.0000   0.0000 H   1 WAT  0.0000\n"
        "     3 H2  -0.2400   0.9300   0.0000 H   1 WAT  0.0000\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n"
    )
    result = run_pipeline(PipelineConfig(mol2, force_field="gaff+uff", charge="cm5", qc="hf/sto-3g", outdir=tmp_path / "out"))
    assert result.structure.net_charge == pytest.approx(0.0, abs=1.0e-6)
    audit = result.audit
    charges_task = next(task for task in audit.tasks if task.get("name") == "charges")
    assert charges_task["effective_method"] == "cm5"
    assert "hirshfeld" in charges_task