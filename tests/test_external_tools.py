from __future__ import annotations

import numpy as np

from lft.domain.structure import Atom, Structure
from lft.domain.topology import Bond
from lft.forcefields.atomtype import _parse_ac
from lft.forcefields.matching import assign_force_field


def test_atomtype_ac_parser(tmp_path) -> None:
    path = tmp_path / "typed.ac"
    path.write_text("ATOM 1 C1 MOL 0 0 0 0.0 ca\nATOM 2 H2 MOL 0 0 0 0.0 ha\n")
    assert _parse_ac(path) == {1: "ca", 2: "ha"}


def test_atomtype_backend_is_recorded(monkeypatch, tmp_path) -> None:
    atoms = [
        Atom(1, "C", "C1", np.zeros(3)),
        Atom(2, "H", "H1", np.array([1.09, 0.0, 0.0])),
        Atom(3, "H", "H2", np.array([-0.55, 0.94, 0.0])),
        Atom(4, "H", "H3", np.array([-0.55, -0.47, 0.82])),
        Atom(5, "H", "H4", np.array([-0.55, -0.47, -0.82])),
    ]
    bonds = [Bond(1, 2), Bond(1, 3), Bond(1, 4), Bond(1, 5)]
    structure = Structure("methane", atoms, bonds, explicit_bonds=True)
    params = assign_force_field(
        structure, "gaff", source_path="input.mol2", workdir=str(tmp_path)
    )
    assert params.atom_types["c3"].source == "gaff"
    assert params.metadata["typing_backend"]["backend"] == "pure-python"
    assert params.metadata["typing_backend"]["external_binary"] is None


def test_atomtype_external_backend_only_when_enabled(monkeypatch, tmp_path) -> None:
    structure = Structure("typed", [Atom(1, "C", "C1", np.zeros(3))])
    monkeypatch.setenv("LFT_ATOMTYPE_EXTERNAL", "1")
    monkeypatch.setattr(
        "lft.forcefields.matching.run_atomtype",
        lambda _structure, workdir: (
            {1: "c3"},
            {"backend": "atomtype", "executable": "/opt/sobtop/atomtype"},
        ),
    )
    params = assign_force_field(
        structure, "gaff", source_path="input.mol2", workdir=str(tmp_path)
    )
    assert params.metadata["typing_backend"]["backend"] == "atomtype"
