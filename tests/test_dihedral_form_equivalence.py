"""Cross-engine proper-dihedral function-form and unit equivalence.

Both backends emit a single periodic term:
    E = K [1 + cos(n phi - phase)]
LAMMPS uses ``dihedral_style fourier`` with K in kcal/mol; GROMACS uses
functype 9 with K in kJ/mol.  The historical UFF/conjugated placeholder was a
LAMMPS ``harmonic`` term with d = +/-1; it is exactly this function at phase
0/180 degrees, so no hybrid style and no unit ambiguity remain.  These tests
parse the emitted files and compare the functions near their equilibrium
(minimum) points rather than only at a single arbitrary angle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lft.backends.gromacs import write_gromacs
from lft.backends.moltemplate import write_lammps
from lft.domain.parameters import AtomType, DihedralParameter, ParameterSet
from lft.domain.structure import Atom, Structure
from lft.domain.topology import Bond, Dihedral
from lft.domain.units import KCAL_PER_KJ


def _chain(function: str, phase: float, k: float, periodicity: int) -> tuple[Structure, ParameterSet]:
    atoms = [
        Atom(index, "C", f"C{index}", np.array([float(index) * 1.4, 0.0, 0.0]), mass=12.0, atom_type="c")
        for index in range(1, 5)
    ]
    bonds = [Bond(1, 2), Bond(2, 3), Bond(3, 4)]
    dihedral = Dihedral((1, 2, 3, 4), ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)))
    structure = Structure("dih", atoms, bonds, None, explicit_bonds=True)
    structure.dihedrals = [dihedral]
    params = ParameterSet("gaff", "classic", atom_types={"c": AtomType("c", "C", 12.0, 3.4, 0.1, "gaff")})
    params.dihedral_parameters = [DihedralParameter(phase, k, periodicity, function, "test")]
    return structure, params


def _lammps_fourier(path) -> tuple[float, int, float]:
    lines = (path / "dih.data").read_text().splitlines()
    start = lines.index("Dihedral Coeffs")
    fields = lines[start + 2].split()
    assert fields[1] == "1", "fourier term count must be 1"
    return float(fields[2]), int(fields[3]), float(fields[4])


def _gromacs_functype9(path) -> tuple[float, int, float, str]:
    lines = (path / "dih.itp").read_text().splitlines()
    start = lines.index("[ dihedrals ]")
    line = next(line for line in lines[start + 1 :] if line.strip() and not line.strip().startswith(";"))
    fields = line.split()
    return float(fields[6]), int(fields[7]), float(fields[5]), fields[4]


def _lammps_energy(k_kcal: float, n: int, phase_deg: float, phi_deg: float) -> float:
    return k_kcal * (1.0 + math.cos(math.radians(n * phi_deg - phase_deg)))


def _gromacs_energy(k_kj: float, n: int, phase_deg: float, phi_deg: float) -> float:
    return k_kj * (1.0 + math.cos(math.radians(n * phi_deg - phase_deg)))


@pytest.mark.parametrize(
    "function,phase,k,periodicity",
    [
        ("fourier", 0.0, 0.15, 3),
        ("fourier", 180.0, 0.42, 2),
        ("harmonic", 0.0, 0.50, 2),
        ("harmonic", 180.0, 0.50, 2),
    ],
)
def test_periodic_dihedral_matches_near_equilibrium(tmp_path, function, phase, k, periodicity) -> None:
    structure, params = _chain(function, phase, k, periodicity)
    write_lammps(structure, params, tmp_path, emit_image_map=False)
    write_gromacs(structure, params, tmp_path, emit_image_map=False)

    k_lammps, n_lammps, phase_lammps = _lammps_fourier(tmp_path)
    k_gromacs, n_gromacs, phase_gromacs, functype = _gromacs_functype9(tmp_path)

    assert functype == "9"
    assert n_lammps == n_gromacs == periodicity
    assert phase_lammps == pytest.approx(phase_gromacs)
    # Units: internal kcal/mol -> GROMACS kJ/mol.
    assert k_gromacs == pytest.approx(k_lammps * 4.184, rel=1.0e-12)

    # Equilibrium of 1 + cos(n phi - phase) is at n*phi - phase = 180 degrees.
    equilibrium = (phase_lammps + 180.0) / periodicity
    for offset in (-15.0, -5.0, -1.0, 0.0, 1.0, 5.0, 15.0):
        phi = equilibrium + offset
        assert k_gromacs * (1.0 + math.cos(math.radians(n_gromacs * phi - phase_gromacs))) == pytest.approx(
            k_lammps * (1.0 + math.cos(math.radians(n_lammps * phi - phase_lammps))) * 4.184,
            rel=1.0e-12,
            abs=1.0e-12,
        )


def test_harmonic_placeholder_has_no_hybrid_style(tmp_path) -> None:
    structure, params = _chain("harmonic", 0.0, 0.5, 2)
    fourier = DihedralParameter(0.0, 0.15, 3, "fourier", "test")
    params.dihedral_parameters.append(fourier)
    structure.dihedrals.append(Dihedral((1, 2, 3, 4), ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))))
    write_lammps(structure, params, tmp_path, emit_image_map=False)

    settings = (tmp_path / "dih.in").read_text()
    assert "dihedral_style fourier" in settings
    assert "hybrid" not in settings
    coeffs = (tmp_path / "dih.data").read_text()
    # Every proper dihedral line starts with "type 1 K n phase".
    start = coeffs.splitlines().index("Dihedral Coeffs")
    terms = [line.split() for line in coeffs.splitlines()[start + 2 :] if line.strip()][:2]
    assert all(term[1] == "1" for term in terms)


def test_kcal_to_kj_conversion_is_consistent(tmp_path) -> None:
    structure, params = _chain("fourier", 0.0, 1.0, 1)
    write_gromacs(structure, params, tmp_path, emit_image_map=False)
    k_gromacs, _, _, _ = _gromacs_functype9(tmp_path)
    assert k_gromacs == pytest.approx(KCAL_PER_KJ ** -1, rel=1.0e-12)
