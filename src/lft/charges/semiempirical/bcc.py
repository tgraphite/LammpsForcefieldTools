"""Pure-Python AM1-BCC charge generation.

Reproduces the AmberTools ``antechamber -c bcc`` pipeline without any external
binary:

  1. AM1 RHF SCF (``lft.charges.semiempirical.am1``), geometry-optimized.
  2. Mulliken charges from the converged density matrix.
  3. BCC atom typing (``ATOMTYPE_BCC.DEF`` via the pure-Python typer).
  4. Bond-type assignment from the explicit bond orders.
  5. Antisymmetric bond-charge corrections from ``BCCPARM.DAT``.

The corrections exactly follow ``antechamber/am1bcc.c`` (``chk_bcc_parm`` /
``bccparm`` / ``charge``).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from ...domain.structure import Structure
from ...forcefields.gaff_typer import assign_gaff_types

#: Largest molecule (atom count) that is geometry-optimized by the AM1 step of
#: the AM1-BCC workflow when ``LFT_AM1BCC_OPTIMIZE=auto``.  Larger molecules use
#: a single point at the input geometry and emit an audit warning.
AM1_OPTIMIZE_MAX_ATOMS = 200


@dataclass
class _BccParam:
    ati: str
    atj: str
    bt: int
    bcc: float


@dataclass
class _BccCorr:
    at: str
    ref: str


def _load_bcc_parm(path: str | Path) -> tuple[list[_BccParam], list[_BccCorr]]:
    parm: list[_BccParam] = []
    corr: list[_BccCorr] = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("CORR"):
            toks = line.split()
            if len(toks) >= 3:
                corr.append(_BccCorr(toks[1], toks[2]))
        else:
            toks = line.split()
            if len(toks) >= 5:
                try:
                    parm.append(_BccParam(toks[1], toks[2], int(toks[3]), float(toks[4])))
                except ValueError:
                    continue
    return parm, corr


class AM1BCC:
    """Compute AM1-BCC charges for a structure entirely in Python."""

    def __init__(self, structure: Structure) -> None:
        self.structure = structure
        def_file = resources.files("lft.charges.data").joinpath("BCCPARM.DAT")
        self._parm, self._corr = _load_bcc_parm(Path(str(def_file)))

    # ------------------------------------------------------------------
    # bond type: the BCC corrections need the bondtype -j full assignment
    # (aromatic bonds disambiguated to 7/8 and carbonyl/other promoted to
    # double as required by valence).
    # ------------------------------------------------------------------
    @staticmethod
    def _bond_type(order: float) -> int:
        if 1.3 <= float(order) <= 1.7:
            return 7
        return int(round(float(order)))

    def _assign_bond_types(self, bcc_types: dict[int, str]) -> dict[tuple[int, int], int]:
        """Run the pure-Python ``bondtype -j full`` equivalent."""
        from .bondtype import assign_bond_types

        elements = [atom.element for atom in self.structure.atoms]
        type_names = [bcc_types.get(i, "") for i in range(1, self.structure.natoms + 1)]
        bonds_in = [
            (bond.i, bond.j, self._bond_type(bond.order))
            for bond in self.structure.bonds
        ]
        new_types, _ = assign_bond_types(type_names, elements, bonds_in)
        result: dict[tuple[int, int], int] = {}
        for bond, bt in zip(self.structure.bonds, new_types):
            result[(bond.i, bond.j)] = bt
        return result

    def _chk_bcc_parm(self, ati: str, atj: str, bt: int) -> float:
        for p in self._parm:
            if p.ati == ati and p.atj == atj and p.bt == bt:
                return p.bcc
            if p.ati == atj and p.atj == ati and p.bt == bt:
                return -p.bcc
        return -999.9

    def _bccparm(self, ati: str, atj: str, bt: int) -> float:
        bcc = self._chk_bcc_parm(ati, atj, bt)
        if bcc > -999.0:
            return bcc
        icorr_ati = None
        icorr_atj = None
        for c in self._corr:
            if c.at == ati:
                icorr_ati = c.ref
            if c.at == atj:
                icorr_atj = c.ref
        if icorr_ati is not None:
            bcc = self._chk_bcc_parm(icorr_ati, atj, bt)
            if bcc > -999.0:
                return bcc
        if icorr_atj is not None:
            bcc = self._chk_bcc_parm(ati, icorr_atj, bt)
            if bcc > -999.0:
                return bcc
        if icorr_ati is not None and icorr_atj is not None:
            bcc = self._chk_bcc_parm(icorr_ati, icorr_atj, bt)
            if bcc > -999.0:
                return bcc
        return 0.0

    def assign(
        self,
        *,
        optimize: bool | None = None,
        geometry_grms_tol: float = 0.0005,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return (charges, metadata) for the AM1-BCC workflow.

        Geometry optimization is disabled by default: the input geometry is
        taken as acceptable and charges are evaluated with a single point.
        It can be opted into with ``optimize=True`` or
        ``LFT_AM1BCC_OPTIMIZE=yes``; ``auto`` restores the size-based rule
        (``AM1_OPTIMIZE_MAX_ATOMS``).
        """
        import os

        from .am1 import AM1

        mode = os.environ.get("LFT_AM1BCC_OPTIMIZE", "no")
        if optimize is None:
            if mode == "yes":
                optimize = True
            elif mode == "auto":
                optimize = self.structure.natoms <= AM1_OPTIMIZE_MAX_ATOMS
            else:  # "no" (default)
                optimize = False

        elements = [atom.element for atom in self.structure.atoms]
        coords = np.array([atom.position for atom in self.structure.atoms], dtype=float)

        am1 = AM1(elements, coords)
        if optimize:
            sp = am1.optimize(grms_tol=geometry_grms_tol)
        else:
            sp = am1.single_point()
        raw_mulliken = np.asarray(sp["mulliken_charges"], dtype=float)

        bcc_types = assign_gaff_types(self.structure, pindex=2)
        type_map = {i: bcc_types.get(i, "") for i in range(1, len(elements) + 1)}

        charges = raw_mulliken.copy()
        bond_types = self._assign_bond_types(bcc_types)
        corrections: list[dict[str, object]] = []
        for bond in self.structure.bonds:
            i, j = bond.i - 1, bond.j - 1
            bt = bond_types[(bond.i, bond.j)]
            ati = type_map.get(i + 1, "")
            atj = type_map.get(j + 1, "")
            bcc = self._bccparm(ati, atj, bt)
            charges[i] += bcc
            charges[j] -= bcc
            corrections.append({
                "bond": (i + 1, j + 1),
                "ati": ati,
                "atj": atj,
                "bt": bt,
                "correction": bcc,
            })

        net = float(charges.sum())
        metadata: dict[str, object] = {
            "backend": "pure-python-am1bcc",
            "scf_energy_eV": sp["energy_eV"],
            "mulliken_charges": raw_mulliken.tolist(),
            "bcc_types": {str(k): v for k, v in type_map.items()},
            "bcc_bond_types": {f"{i}-{j}": bt for (i, j), bt in bond_types.items()},
            "bcc_corrections": corrections,
            "net_charge": net,
            "geometry_optimized": bool(sp.get("optimized", False)),
            "geometry_converged": bool(sp.get("geometry_converged", False)),
            "geometry_skipped": bool(not optimize),
            "n_scf_iterations": sp.get("n_iterations"),
            "energy_converged": bool(sp.get("converged", False)),
        }
        if not optimize:
            metadata["geometry_note"] = (
                "geometry optimization disabled; AM1-BCC evaluated as a single point "
                "at the user-supplied geometry"
            )
        return charges, metadata


def assign_am1bcc(
    structure: Structure,
    *,
    optimize: bool | None = None,
    geometry_grms_tol: float = 0.0005,
) -> tuple[np.ndarray, dict[str, object]]:
    """Convenience wrapper returning (charges, metadata)."""
    return AM1BCC(structure).assign(optimize=optimize, geometry_grms_tol=geometry_grms_tol)