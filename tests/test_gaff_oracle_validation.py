"""Failures in the differential harness must never count as matches."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from lft.errors import ExternalToolError
from lft.io.loaders import load_structure
from lft.validation.gaff_oracle import (
    PROJECT_ROOT,
    OracleConfig,
    parse_oracle_ac,
    run_oracle,
)

WATER = PROJECT_ROOT / "tests/fixtures/gaff/historical29/water.mol2"


def _ac_record(index: int, name: str = "H123", atom_type: str = "ho") -> str:
    return f"ATOM{index:7d}  {name:<4s}{'MOL':<4s}{1:5d}{0:12.3f}{0:8.3f}{0:8.3f}{0:10.6f}{atom_type:>10s}\n"


def _fake_oracle(tmp_path: Path, body: str) -> OracleConfig:
    binary = tmp_path / "atomtype"
    binary.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nimport sys\n" + body
    )
    binary.chmod(0o755)
    data = tmp_path / "dat/antechamber"
    data.mkdir(parents=True)
    (data / "CORR_NAME_TYPE.DAT").write_text("H H\n")
    definition = data / "ATOMTYPE_GFF.DEF"
    definition.write_text("ATD DU &\n")
    return OracleConfig(binary, tmp_path, definition)


def test_oracle_ac_handles_adjacent_fixed_width_names(tmp_path: Path) -> None:
    ac = tmp_path / "out.ac"
    ac.write_text(_ac_record(1))
    assert parse_oracle_ac(ac, 1) == {1: "ho"}


@pytest.mark.parametrize(
    "content,natoms",
    [
        ("", 1),
        (_ac_record(1), 2),
        (_ac_record(1) * 2, 1),
        (_ac_record(2), 1),
        ("ATOM 1 ho\n", 1),
    ],
)
def test_oracle_ac_rejects_invalid_mapping(
    content: str, natoms: int, tmp_path: Path
) -> None:
    ac = tmp_path / "out.ac"
    ac.write_text(content)
    with pytest.raises(ExternalToolError):
        parse_oracle_ac(ac, natoms)


def test_oracle_rejects_nonzero_even_with_output(tmp_path: Path) -> None:
    record = _ac_record(1, atom_type="oh") + _ac_record(2) + _ac_record(3)
    config = _fake_oracle(
        tmp_path,
        f"Path(sys.argv[sys.argv.index('-o')+1]).write_text({record!r})\nsys.exit(3)\n",
    )
    with pytest.raises(ExternalToolError, match="exit code 3"):
        run_oracle(load_structure(WATER), config, tmp_path / "cases")


def test_oracle_repeated_run_cannot_reuse_old_output(tmp_path: Path) -> None:
    record = _ac_record(1, atom_type="oh") + _ac_record(2) + _ac_record(3)
    config = _fake_oracle(
        tmp_path, f"Path(sys.argv[sys.argv.index('-o')+1]).write_text({record!r})\n"
    )
    work = tmp_path / "cases"
    assert run_oracle(load_structure(WATER), config, work) == {
        1: "oh",
        2: "ho",
        3: "ho",
    }
    config.executable.write_text(f"#!{sys.executable}\n")
    with pytest.raises(ExternalToolError, match="no output"):
        run_oracle(load_structure(WATER), config, work)
    assert len(list(work.iterdir())) == 2
    assert not (tmp_path / "ATOMTYPE.INF").exists()


def _validator():
    spec = importlib.util.spec_from_file_location(
        "gaff_validator", PROJECT_ROOT / "scripts/validate_gaff_typer_vs_oracle.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_missing_oracle_is_fatal(tmp_path: Path) -> None:
    assert _validator().main(["--oracle", str(tmp_path / "missing")]) == 2


def test_validator_missing_explicit_input_is_failure(tmp_path: Path) -> None:
    config = _fake_oracle(tmp_path, "")
    out = tmp_path / "results"
    assert (
        _validator().main(
            [
                "--oracle",
                str(config.executable),
                "--amberhome",
                str(config.amberhome),
                "--only",
                str(tmp_path / "missing.mol2"),
                "--outdir",
                str(out),
            ]
        )
        == 1
    )
    report = json.loads(next(out.glob("run-*/report.json")).read_text())
    assert report["summary"] == {"passed": 0, "failed": 1, "skipped": 0}


def test_validator_pass_fail_skip_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _fake_oracle(tmp_path, "")
    module = _validator()
    out = tmp_path / "results"
    xyz = tmp_path / "bondless.xyz"
    xyz.write_text("1\nC\nC 0 0 0\n")
    monkeypatch.setattr(
        module, "find_corpus", lambda _: [WATER, xyz, tmp_path / "missing.mol2"]
    )
    # Restrict default frozen inputs to a temporary empty directory.
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "run_oracle", lambda *args: {1: "DU", 2: "DU", 3: "DU"})
    assert (
        module.main(
            [
                "--discover",
                "--oracle",
                str(config.executable),
                "--amberhome",
                str(config.amberhome),
                "--outdir",
                str(out),
            ]
        )
        == 1
    )
    report = json.loads(next(out.glob("run-*/report.json")).read_text())
    assert report["summary"] == {"passed": 1, "failed": 1, "skipped": 1}
