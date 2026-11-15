"""Regression tests for the moltemplate (``.lt``) writer format."""

from __future__ import annotations

from lft.config import PipelineConfig
from lft.pipeline import run_pipeline


def _water_mol2(tmp_path) -> str:
    mol2 = tmp_path / "water.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nwater\n3 2 1 0 0\nSMALL\nUSER_CHARGES\n\n"
        "@<TRIPOS>ATOM\n"
        "     1 O1   0.0000   0.0000   0.0000 O   1 WAT  0.0000\n"
        "     2 H1   0.9600   0.0000   0.0000 H   1 WAT  0.0000\n"
        "     3 H2  -0.2400   0.9300   0.0000 H   1 WAT  0.0000\n"
        "@<TRIPOS>BOND\n1 1 2 1\n2 1 3 1\n"
    )
    return mol2


def test_moltemplate_emits_molecule_and_system(tmp_path) -> None:
    result = run_pipeline(
        PipelineConfig(
            _water_mol2(tmp_path),
            force_field="gaff+uff",
            charge="input",
            emit_lt=True,
            outdir=tmp_path / "out",
        )
    )
    lt = (result.output_dir / "water.lt").read_text()
    system = (result.output_dir / "system.lt").read_text()

    # Molecule template: named block, no system wrapper, named atom types.
    assert "water {" in lt
    assert "LFT_System" not in lt
    assert "$mol:water" in lt
    assert "@atom:oh" in lt
    assert "@atom:ho" in lt
    assert 'write_once("Data Boundary")' not in lt

    # "In Init" carries units/styles/global controls; "In Settings" only holds
    # *_coeff parameters (so system.in.settings stays a pure coefficient file).
    init = lt.split('write_once("In Init")')[1].split("}")[0]
    settings = lt.split('write_once("In Settings")')[1].split("}")[0]
    assert "units real" in init
    assert "pair_style" in init
    assert "bond_style harmonic" in init
    assert "pair_style" not in settings
    assert "units" not in settings
    assert "pair_coeff" in settings
    assert "bond_coeff" in settings

    # Assembly template instantiates the molecule and owns the boundary.
    assert 'import "water.lt"' in system
    assert "mols = new water [1]" in system
    assert 'write_once("Data Boundary")' in system
