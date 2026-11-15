"""End-to-end CLI behaviour for the simplified default workflow."""

from __future__ import annotations

import numpy as np
import pytest

from lft.cli import main
from lft.config import PipelineConfig
from lft.domain.structure import Atom, Cell, Structure
from lft.errors import ChargeError
from lft.pipeline import _resolve_charge

_CHARGED_XYZ = "3\nwater\nO 0.000 0.000 0.120 -0.8\nH 0.760 0.000 -0.480 0.4\nH -0.760 0.000 -0.480 0.4\n"


def test_cli_default_writes_moltemplate_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "water.xyz").write_text(_CHARGED_XYZ)
    assert main(["-f", "water.xyz"]) == 0
    out = capsys.readouterr().out
    assert "water.lt" in out and "system.lt" in out
    assert (tmp_path / "water.lt").is_file()
    assert (tmp_path / "system.lt").is_file()
    assert not (tmp_path / "water.data").exists()
    # No JSON sidecars on the happy path.
    assert not list(tmp_path.glob("*.json"))


def test_cli_lammpsdata_writes_data_and_input(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "water.xyz").write_text(_CHARGED_XYZ)
    assert main(["-f", "water.xyz", "--lammpsdata"]) == 0
    assert (tmp_path / "water.data").is_file()
    assert (tmp_path / "water.in").is_file()
    assert not (tmp_path / "water.lt").exists()


def test_cli_charge_file_path_and_overwrite_guard(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mol.xyz").write_text("3\nwater\nO 0 0 0.12\nH 0.76 0 -0.48\nH -0.76 0 -0.48\n")
    (tmp_path / "charges.txt").write_text("-0.8\n0.4\n0.4\n")
    assert main(["-f", "mol.xyz", "--charge", "charges.txt", "--lammpsdata"]) == 0
    assert (tmp_path / "mol.data").is_file()
    # Second run refuses to overwrite unless --force is given.
    assert main(["-f", "mol.xyz", "--charge", "charges.txt", "--lammpsdata"]) == 2
    assert main(["-f", "mol.xyz", "--charge", "charges.txt", "--lammpsdata", "--force"]) == 0


def test_cli_rejects_unknown_charge_spec(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mol.xyz").write_text("2\nC-H\nC 0 0 0\nH 1.09 0 0\n")
    assert main(["-f", "mol.xyz", "--charge", "respp"]) == 2
    err = capsys.readouterr().err
    assert "neither a known method nor an existing charge file" in err


def test_charge_auto_selection_uses_structure_charges() -> None:
    structure = Structure("m", [Atom(1, "H", "H1", np.zeros(3), charge=0.1, charge_source="input", mass=1.0)])
    method, charge_file = _resolve_charge(PipelineConfig("x.xyz", charge=None), structure)
    assert (method, charge_file) == ("input", None)


def test_charge_auto_selection_size_threshold() -> None:
    small = Structure("s", [Atom(i, "C", f"C{i}", np.zeros(3), mass=12.0) for i in range(1, 11)])
    method, _ = _resolve_charge(PipelineConfig("x.xyz", charge=None), small)
    assert method == "resp"

    large = Structure("l", [Atom(i, "C", f"C{i}", np.zeros(3), mass=12.0) for i in range(1, 102)])
    method, _ = _resolve_charge(PipelineConfig("x.xyz", charge=None), large)
    assert method == "am1-bcc"
    assert any(w["code"] == "CHARGE_AUTO_LARGE_MOLECULE" for w in large.warnings)

    # An explicit method always overrides the size-based default.
    method, _ = _resolve_charge(PipelineConfig("x.xyz", charge="resp"), large)
    assert method == "resp"


def test_lft_config_supplies_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "lft.config"
    config.write_text("output = gromacs\nff = gaff+uff\n")
    monkeypatch.setenv("LFT_CONFIG", str(config))
    (tmp_path / "water.xyz").write_text(_CHARGED_XYZ)
    assert main(["-f", "water.xyz"]) == 0
    assert (tmp_path / "water.itp").is_file()
    assert not (tmp_path / "water.data").exists()
    # An explicit flag still wins over the config file.
    assert main(["-f", "water.xyz", "-o", "lammps", "--lammpsdata", "--force"]) == 0
    assert (tmp_path / "water.data").is_file()


def test_periodic_requires_explicit_charges() -> None:
    structure = Structure(
        "cell",
        [Atom(1, "H", "H1", np.zeros(3), mass=1.0)],
        cell=Cell.from_parameters(5.0, 5.0, 5.0),
    )
    with pytest.raises(ChargeError):
        _resolve_charge(PipelineConfig("x.cif", charge=None), structure)
