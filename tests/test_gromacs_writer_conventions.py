from __future__ import annotations

import numpy as np

from lft.backends.gromacs import write_gromacs
from lft.domain.parameters import AtomType, DihedralParameter, ParameterSet
from lft.domain.structure import Atom, Structure
from lft.domain.topology import Bond, Dihedral


def _chain_with_harmonic_dihedral(phase: float) -> tuple[Structure, ParameterSet]:
    atoms = [
        Atom(index, "C", f"C{index}", np.array([float(index), 0.0, 0.0]), mass=12.0, atom_type="c")
        for index in range(1, 5)
    ]
    bonds = [Bond(1, 2), Bond(2, 3), Bond(3, 4)]
    dihedral = Dihedral((1, 2, 3, 4), ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    structure = Structure("dih", atoms, bonds, None, explicit_bonds=True)
    structure.dihedrals = [dihedral]
    params = ParameterSet("uff", "classic", atom_types={"c": AtomType("c", "C", 12.0, 3.4, 0.1, "uff")})
    params.dihedral_parameters = [DihedralParameter(phase, 0.5, 2, "harmonic", "test")]
    return structure, params


def _dihedral_line(itp_text: str) -> str:
    lines = itp_text.splitlines()
    start = lines.index("[ dihedrals ]")
    return next(line for line in lines[start + 1 :] if line.strip() and not line.strip().startswith(";"))


def test_gromacs_harmonic_dihedral_uses_proper_periodic_functype(tmp_path) -> None:
    structure, params = _chain_with_harmonic_dihedral(0.0)
    write_gromacs(structure, params, tmp_path)
    line = _dihedral_line((tmp_path / "dih.itp").read_text())

    fields = line.split()
    assert fields[4] == "9", "GROMACS functype 2 is a harmonic improper; proper harmonic must use functype 9"
    assert fields[5] == "0.000000"
    assert fields[7] == "2"
    assert "function=harmonic" in line


def test_gromacs_harmonic_dihedral_phase_180_stays_proper(tmp_path) -> None:
    structure, params = _chain_with_harmonic_dihedral(180.0)
    write_gromacs(structure, params, tmp_path)
    line = _dihedral_line((tmp_path / "dih.itp").read_text())

    fields = line.split()
    assert fields[4] == "9"
    assert fields[5] == "180.000000"
