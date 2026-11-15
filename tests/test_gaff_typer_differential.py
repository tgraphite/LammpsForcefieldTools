"""Binary-free frozen regressions plus optional live AmberTools26 comparisons."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lft.domain.structure import Atom, Structure
from lft.domain.topology import Bond
from lft.errors import ExternalToolError, ParameterError
from lft.forcefields import gaff_typer
from lft.forcefields.atomtype import run_python_atomtype
from lft.forcefields.gaff import gaff_type
from lft.forcefields.matching import assign_force_field
from lft.io.loaders import load_structure
from lft.validation.gaff_oracle import OracleConfig, run_oracle

FIXTURES = Path(__file__).parent / "fixtures/gaff"
REFERENCE = json.loads((FIXTURES / "reference.json").read_text())
CASES = sorted(REFERENCE["cases"])


@pytest.fixture
def oracle() -> OracleConfig:
    config = OracleConfig.from_environment()
    try:
        config.validate()
    except ExternalToolError as exc:
        pytest.skip(f"optional AmberTools26 oracle unavailable: {exc}")
    return config


@pytest.mark.parametrize("case", CASES)
def test_gaff_typer_frozen_reference(case: str) -> None:
    path = FIXTURES / case
    expected = REFERENCE["cases"][case]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["input_sha256"]
    structure = load_structure(path)
    assert gaff_typer.assign_gaff_types(structure) == {
        int(i): value for i, value in expected["types"].items()
    }


@pytest.mark.parametrize("case", CASES)
def test_gaff_typer_matches_live_oracle(
    case: str, oracle: OracleConfig, tmp_path: Path
) -> None:
    structure = load_structure(FIXTURES / case)
    assert gaff_typer.assign_gaff_types(structure) == run_oracle(
        structure, oracle, tmp_path
    )


def test_gaff_typer_input_unchanged_and_no_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    structure = load_structure(FIXTURES / "boundary/indole_h.mol2")
    before = deepcopy(structure.to_dict())
    results = []
    for folder in (tmp_path / "first", tmp_path / "second"):
        folder.mkdir()
        monkeypatch.chdir(folder)
        for _ in range(2):
            results.append(run_python_atomtype(structure)[0])
        assert list(folder.iterdir()) == []
        assert structure.to_dict() == before
    assert all(result == results[0] for result in results)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"source_path": "nonexistent.mol2"},
        {"workdir": "unused"},
        {"source_path": "missing", "workdir": "unused"},
    ],
)
def test_gaff_api_always_uses_rule_engine(
    arguments: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LFT_ATOMTYPE_EXTERNAL", raising=False)
    monkeypatch.chdir(tmp_path)
    structure = load_structure(FIXTURES / "boundary/butadiene_h.mol2")
    expected = gaff_typer.assign_gaff_types(structure)
    assert any(name in {"ce", "cf"} for name in expected.values())
    for atom in structure.atoms:
        assert gaff_type(structure, atom.index) == expected[atom.index]
    params = assign_force_field(structure, "gaff", **arguments)
    assert params.metadata["typing_backend"]["backend"] == "pure-python"
    assert {a.index: a.atom_type for a in structure.atoms} == expected
    assert list(tmp_path.iterdir()) == []


def _environment_case(closed: bool) -> Structure:
    return Structure(
        "environment",
        [
            Atom(1, "O", "O", np.zeros(3)),
            Atom(2, "C", "C", np.zeros(3)),
            Atom(3, "C", "C", np.zeros(3)),
        ],
        [Bond(1, 2), Bond(1, 3)] + ([Bond(2, 3)] if closed else []),
    )


def test_gaff_concurrent_environment_matching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    definition = tmp_path / "environment.DEF"
    definition.write_text(
        "ATD yes * 8 2 * * * (C<a>,C<b>) a:b:sb\nATD no * 8 &\nATD c3 * 6 &\n"
    )
    barrier = threading.Barrier(2)
    original = gaff_typer._dccheck

    def synchronized(*args: object) -> int:
        barrier.wait(timeout=10)
        return original(*args)

    monkeypatch.setattr(gaff_typer, "_dccheck", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda closed: gaff_typer.assign_gaff_types(
                    _environment_case(closed), definition
                ),
                [True, False],
            )
        )
    assert [r[1] for r in results] == ["yes", "no"]


@pytest.mark.parametrize("constraint", ["any", "sb", "SB", "ab"])
def test_gaff_environment_constraint_source_reference(
    constraint: str, tmp_path: Path
) -> None:
    definition = tmp_path / "environment.DEF"
    definition.write_text(
        f"ATD yes * 8 2 * * * (C<a>,C<b>) a:b:{constraint}\nATD no * 8 &\nATD c3 * 6 &\n"
    )
    assert gaff_typer.assign_gaff_types(_environment_case(False), definition)[1] == "no"
    assert gaff_typer.assign_gaff_types(_environment_case(True), definition)[1] == (
        "yes" if constraint in {"any", "sb"} else "no"
    )


def test_gaff_rule_priority_and_no_match_preserves_input_type(tmp_path: Path) -> None:
    definition = tmp_path / "priority.DEF"
    definition.write_text("ATD first * 8 &\nATD second * 8 &\n")
    assert gaff_typer.assign_gaff_types(_environment_case(False), definition) == {
        1: "first",
        2: "C",
        3: "C",
    }


@pytest.mark.parametrize("name", ["1H", "h1", "Q", "Cafoo"])
def test_gaff_invalid_name_has_no_element_fallback(name: str) -> None:
    structure = _environment_case(False)
    structure.atoms[0].name = name
    if name == "Cafoo":
        # Case-sensitive Ca means calcium even when Structure.element is O.
        assert gaff_typer.assign_gaff_types(structure)[1] == "Ca"
    else:
        with pytest.raises(ParameterError, match="atomic symbol"):
            gaff_typer.assign_gaff_types(structure)


def test_gaff_missing_definition_is_not_a_heuristic_fallback(tmp_path: Path) -> None:
    with pytest.raises(ParameterError, match="definition"):
        gaff_typer.assign_gaff_types(_environment_case(False), tmp_path / "missing.DEF")


def test_gaff_external_request_fails_when_oracle_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LFT_ATOMTYPE_EXTERNAL", "1")
    monkeypatch.setattr("lft.forcefields.atomtype.discover_atomtype", lambda: None)
    with pytest.raises(ExternalToolError):
        assign_force_field(_environment_case(False), "gaff", workdir=str(tmp_path))
