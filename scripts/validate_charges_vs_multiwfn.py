#!/usr/bin/env python3
"""Validate the internal density-charge engine against the reference Multiwfn.

The reference is Multiwfn compiled (noGUI) from
``thirdparty/Multiwfn_2026.8.31_src_Linux`` (gfortran; see
``COMPILATION_METHOD.txt`` and ``_archive/qc/README.md``).  The production
program never calls Multiwfn; this harness is validation-only tooling that
compares the internal Mulliken/Hirshfeld/CM5/ADCH output with the reference
toolchain on the same wavefunction.

Usage:
    .venv/bin/python scripts/validate_charges_vs_multiwfn.py [--molecule MOL2]...
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lft.charges.density_charges import (
    adch_charges,
    cm5_charges,
    hirshfeld_charges,
    normalize_populations,
)
from lft.charges.resp import fit_resp
from lft.io.loaders import load_structure
from lft.qc.pyscf_backend import run_pyscf
from lft.topology.infer_bonds import infer_bonds

REFERENCE = ROOT / "thirdparty" / "Multiwfn_2026.8.31_src_Linux" / "Multiwfn_noGUI"
TOLERANCE = 1.0e-5
# RESP is an iterative fit to the ESP, so it additionally accumulates the
# (small) difference between pySCF's analytic ESP and the ESP Multiwfn
# reconstructs from the round-tripped Molden coefficients.
RESP_TOLERANCE = 1.0e-4


def _discover_reference() -> str:
    configured = shutil.which("Multiwfn_noGUI")
    candidates = [configured] if configured else []
    candidates.append(str(REFERENCE))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "reference Multiwfn binary not found; build it from thirdparty/Multiwfn_2026.8.31_src_Linux"
    )


def _run_reference(
    executable: str, molden: Path, workdir: Path, suboption: int
) -> list[float]:
    script = "7\n%d\n1\ny\n0\nq\n" % suboption
    result = subprocess.run(
        [executable, str(molden)],
        cwd=workdir,
        input=script,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Final atomic charges:\n(.*?)\n\n", result.stdout, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"could not parse reference output for suboption {suboption}"
        )
    values = [
        float(line.rsplit(":", 1)[1])
        for line in match.group(1).strip().splitlines()
        if ":" in line
    ]
    return values


def _run_reference_resp(executable: str, molden: Path, workdir: Path) -> list[float]:
    """Run Multiwfn's standard two-stage RESP and parse the fitted charges."""

    script = "7\n18\n1\nn\n0\n0\nq\n"
    result = subprocess.run(
        [executable, str(molden)],
        cwd=workdir,
        input=script,
        capture_output=True,
        text=True,
    )
    match = re.search(
        r"Center\s+Charge\n(.*?)\n\s*Sum of charges", result.stdout, re.DOTALL
    )
    if not match:
        raise RuntimeError("could not parse reference RESP output")
    return [float(line.split()[-1]) for line in match.group(1).strip().splitlines()]


def _compare(
    name: str, ours: np.ndarray, reference: list[float], tolerance: float = TOLERANCE
) -> None:
    max_error = float(np.abs(np.asarray(ours) - np.asarray(reference)).max())
    status = "PASS" if max_error <= tolerance else "FAIL"
    print(f"[{status}] {name}: max|diff| = {max_error:.3e} (tolerance {tolerance:.0e})")
    if status == "FAIL":
        raise SystemExit(
            f"{name} deviates from the reference toolchain by {max_error:.3e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--molecule",
        action="append",
        default=[],
        help="mol2 path (repeatable); defaults to the pinned set",
    )
    args = parser.parse_args()
    executable = _discover_reference()
    workdir = ROOT / "_scratch" / "charges_multiwfn_validation"
    workdir.mkdir(parents=True, exist_ok=True)

    molecules = args.molecule or [
        "references/sobtop/examples/Methyl_benzoate/Methyl_benzoate.mol2",
        "references/sobtop/examples/SF6/SF6.mol2",
        "references/sobtop/examples/exercise/7-helicene.mol2",
    ]
    for mol2 in molecules:
        structure = load_structure(ROOT / mol2)
        structure.declared_net_charge = 0
        structure.multiplicity = 1
        infer_bonds(structure)
        wf = run_pyscf(structure, qc="b3lyp/sto-3g").wavefunction
        from pyscf.tools import molden  # type: ignore[import-not-found]

        molden_path = workdir / (structure.name + ".molden")
        molden.from_mo(wf.mol, str(molden_path), wf.mo_coeff, occ=wf.mo_occ)

        hirshfeld, dipoles, _ = hirshfeld_charges(wf)
        cm5, _ = cm5_charges(hirshfeld, wf)
        adch, _ = adch_charges(hirshfeld, dipoles, wf)
        resp, _ = fit_resp(structure, wf, 0.0)
        print(f"\n== {structure.name} ({structure.natoms} atoms) ==")
        _compare(
            "hirshfeld",
            normalize_populations(hirshfeld, wf),
            _run_reference(executable, molden_path, workdir, 1),
        )
        _compare(
            "cm5",
            normalize_populations(cm5, wf),
            _run_reference(executable, molden_path, workdir, 16),
        )
        _compare(
            "adch",
            normalize_populations(adch, wf),
            _run_reference(executable, molden_path, workdir, 11),
        )
        _compare(
            "resp",
            resp,
            _run_reference_resp(executable, molden_path, workdir),
            RESP_TOLERANCE,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
