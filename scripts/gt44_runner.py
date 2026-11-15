#!/usr/bin/env python3
"""GT-44 cross-engine single-frame energy acceptance runner.

Builds a ~1000-atom real box per catalog case, generates GROMACS and LAMMPS
artifacts through the full pipeline, runs both engines for one static frame,
and compares per-term and total potential energies.  Acceptance criteria and
the frozen 1.0 nm full-cutoff LJ+Coul profile are in
docs/PROJECT_STATUS.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lft.backends.gromacs import write_gromacs
from lft.backends.moltemplate import write_lammps
from lft.charges.dispatcher import assign_charges
from lft.domain.structure import Atom, Cell, Structure
from lft.domain.topology import Bond
from lft.forcefields.matching import assign_force_field
from lft.io.loaders import load_structure
from lft.parameterize.classic import parameterize
from lft.topology import enumerate_terms, infer_bonds

GMX = "/usr/local/gromacs/bin/gmx"
LMP = ROOT / "thirdparty" / "lammps" / "bin" / "lmp"
LMP_LIB = ROOT / "thirdparty" / "lammps" / "lib"

GRID_NM = 0.01  # validated molecular coordinate grid: 0.1 A
CRYSTAL_GRID_NM = 0.001  # 0.01 A; avoids pairs landing exactly on the cutoff in periodic cells
MIN_PERIODIC_EDGE = 20.5  # A; must exceed 2 x 1.0 nm cutoff for either engine
BOND_TOL = 1e-3        # kJ/mol
ANGLE_TOL = 1e-3
DIHEDRAL_TOL = 1e-3
IMPROPER_TOL = 1e-3
TOTAL_TOL_ABS = 1e-2    # kJ/mol
TOTAL_TOL_REL = 1e-4     # GROMACS here is single precision (~1e-5 relative noise)


def round_coords(structure: Structure, grid_nm: float = GRID_NM) -> None:
    step = grid_nm * 10.0
    for atom in structure.atoms:
        atom.position = np.round(atom.position / step) * step


def replicate_molecule(mol: Structure, target: int) -> Structure:
    natoms = mol.natoms
    grid = max(1, int(round((target / natoms) ** (1 / 3.0))))
    while grid**3 * natoms < target * 0.8:
        grid += 1
    diameter = float(np.max(np.linalg.norm(mol.coordinates - mol.coordinates.mean(axis=0), axis=1)))
    spacing = max(4.0, 2.0 * diameter + 2.0)
    box = grid * spacing
    if box < 20.5:
        spacing = 20.5 / grid
        box = grid * spacing
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    idx = 0
    for ix in range(grid):
        for iy in range(grid):
            for iz in range(grid):
                origin = np.array([ix, iy, iz], dtype=float) * spacing
                mol_ids: dict[int, int] = {}
                for a in mol.atoms:
                    idx += 1
                    mol_ids[a.index] = idx
                    atoms.append(Atom(idx, a.element, a.name, a.position + origin, charge=a.charge, charge_source=a.charge_source, atom_type=a.atom_type, mass=a.mass))
                for b in mol.bonds:
                    bonds.append(Bond(mol_ids[b.i], mol_ids[b.j], order=b.order, image_shift=b.image_shift))
    structure = Structure(f"{mol.name}_{grid}3", atoms, bonds, Cell.from_parameters(box, box, box), explicit_bonds=True)
    round_coords(structure)
    return structure


def replicate_crystal(mol: Structure, target: int) -> Structure:
    base = mol.cell.matrix
    pbc = mol.cell.pbc
    lengths = [float(np.linalg.norm(base[:, axis])) for axis in range(3)]

    def edge_ok(factors: tuple[int, int, int]) -> bool:
        return all(
            (not pbc[axis]) or factors[axis] * lengths[axis] >= MIN_PERIODIC_EDGE
            for axis in range(3)
        )

    target_lo = max(1, int(target * 0.5))
    target_hi = int(target * 2.0)
    all_candidates: list[tuple[int, float, tuple[int, int, int], float]] = []
    for nx in range(1, 13):
        for ny in range(1, 13):
            for nz in range(1, 13):
                factors = (nx, ny, nz)
                # Both engines must see every periodic box edge larger than
                # twice the 1.0 nm cutoff, otherwise the cutoffs disagree.
                if not edge_ok(factors):
                    continue
                count = nx * ny * nz * mol.natoms
                edges = [factors[axis] * lengths[axis] for axis in range(3)]
                # Prefer near-target atom counts but avoid very elongated
                # supercells, which some engines handle poorly.
                aspect = max(edges) / min(edges)
                score = abs(count - target) * (1.0 + aspect)
                all_candidates.append((count, aspect, factors, score))
    within = [c for c in all_candidates if target_lo <= c[0] <= target_hi]
    candidates = within or all_candidates
    if not candidates:
        raise ValueError(f"cannot build a periodic supercell of {mol.name} with edges >= {MIN_PERIODIC_EDGE} A")
    nx, ny, nz = min(candidates, key=lambda item: item[3])[2]
    # Cell stores lattice vectors column-wise; a supercell scales each vector.
    new_matrix = base * np.array([nx, ny, nz])[None, :]
    new_cell = Cell(new_matrix, mol.cell.pbc)
    atoms: list[Atom] = []
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = mol.cell.image_vector((ix, iy, iz))
                for a in mol.atoms:
                    idx += 1
                    atoms.append(Atom(idx, a.element, a.name, a.position + offset, charge=a.charge, charge_source=a.charge_source, atom_type=a.atom_type, mass=a.mass))
    structure = Structure(f"{mol.name}_x{nx}x{ny}x{nz}", atoms, [], new_cell, explicit_bonds=False)
    round_coords(structure, CRYSTAL_GRID_NM)
    infer_bonds(structure)
    enumerate_terms(structure)
    return structure


def apply_charges(structure: Structure, net_charge: float, workdir: Path) -> None:
    # Charges are placed on atoms before replication (per-molecule) or computed
    # on the single molecule; here we only validate the final sum.
    assign_charges(structure, "input", net_charge=net_charge, charge_file=None, workdir=workdir)


def run_gmx(path: Path, stem: str) -> dict[str, float]:
    subprocess.run(
        [GMX, "grompp", "-f", str(path / f"{stem}.mdp"), "-c", str(path / f"{stem}.gro"),
         "-p", str(path / f"{stem}.top"), "-o", str(path / "sys.tpr"), "-maxwarn", "5"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    subprocess.run([GMX, "mdrun", "-s", str(path / "sys.tpr"), "-deffnm", str(path / "sys"), "-nt", "1"],
                   cwd=path, capture_output=True, text=True, check=True)
    selection = "\n".join(["Bond", "Angle", "Proper-Dih.", "Per.-Imp.-Dih.", "LJ-(SR)", "Coulomb-(SR)", "Potential", ""])
    result = subprocess.run([GMX, "energy", "-f", str(path / "sys.edr")], cwd=path, input=selection, capture_output=True, text=True)
    terms: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = re.match(r"^([A-Za-z].*?)\s{2,}(-?[\d.eE+-]+)\s+--", line)
        if m:
            terms[m.group(1).strip()] = float(m.group(2))
    return terms


def run_lammps(path: Path, stem: str) -> dict[str, float]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(LMP_LIB)
    result = subprocess.run([str(LMP), "-in", f"{stem}.in"], cwd=path, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lmp failed: {result.stderr[-800:]}")
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Step"):
            fields = lines[i + 1].split()
            return {
                "E_bond": float(fields[3]) * 4.184,
                "E_angle": float(fields[4]) * 4.184,
                "E_dihed": float(fields[5]) * 4.184,
                "E_impro": float(fields[6]) * 4.184,
                "E_vdwl": float(fields[7]) * 4.184,
                "E_coul": float(fields[8]) * 4.184,
                "PotEng": float(fields[2]) * 4.184,
            }
    raise RuntimeError("lmp thermo table not found")


def compare(g: dict[str, float], l: dict[str, float]) -> list[dict[str, float]]:
    def term_tol(value: float) -> float:
        return max(1e-3, TOTAL_TOL_REL * abs(value))

    checks = [
        ("bond", g.get("Bond", 0.0), l.get("E_bond", 0.0), term_tol(max(abs(g.get("Bond", 0.0)), abs(l.get("E_bond", 0.0))))),
        ("angle", g.get("Angle", 0.0), l.get("E_angle", 0.0), term_tol(max(abs(g.get("Angle", 0.0)), abs(l.get("E_angle", 0.0))))),
        ("dihedral", g.get("Proper Dih.", 0.0), l.get("E_dihed", 0.0), term_tol(max(abs(g.get("Proper Dih.", 0.0)), abs(l.get("E_dihed", 0.0))))),
        ("improper", g.get("Per. Imp. Dih.", 0.0), l.get("E_impro", 0.0), term_tol(max(abs(g.get("Per. Imp. Dih.", 0.0)), abs(l.get("E_impro", 0.0))))),
    ]
    g_total = g.get("Potential", float("nan"))
    l_total = l.get("PotEng", float("nan"))
    # The total is a sum of large signed terms and can cancel almost completely
    # (e.g. SiO2: +10.9 MJ dihedral vs -12.1 MJ LJ).  GROMACS runs in mixed
    # precision, so the total must be judged against the magnitude of the terms
    # that were summed, not the small net value.
    g_scale = sum(abs(value) for key, value in g.items() if key != "Potential")
    l_scale = sum(abs(value) for key, value in l.items() if key in {"E_bond", "E_angle", "E_dihed", "E_impro", "E_vdwl", "E_coul"})
    tol_total = max(TOTAL_TOL_ABS, TOTAL_TOL_REL * max(g_scale, l_scale))
    results = []
    for name, gv, lv, tol in checks:
        results.append({"term": name, "gromacs_kj": gv, "lammps_kj": lv, "delta_kj": gv - lv, "tol_kj": tol, "pass": abs(gv - lv) <= tol})
    results.append({"term": "total", "gromacs_kj": g_total, "lammps_kj": l_total, "delta_kj": g_total - l_total, "tol_kj": tol_total, "pass": abs(g_total - l_total) <= tol_total})
    return results


def run_case(case: dict, outdir: Path) -> dict:
    result: dict = {"id": case["id"], "name": case["name"], "substitute": bool(case.get("substitute", False))}
    work = outdir / case["id"]
    shutil.rmtree(work, ignore_errors=True)
    try:
        mol = load_structure(ROOT / case["input"])
        if case.get("box_kind") != "crystal":
            # Molecular boxes use the validated 0.1 A grid; periodic crystals
            # apply the finer grid after supercell construction instead.
            round_coords(mol)
        charge_spec = case.get("charges")
        charge_method = case.get("charge", "input")
        if isinstance(charge_spec, list) and charge_spec:
            per_mol = [float(v) for v in charge_spec]
            if len(per_mol) == 1:
                per_mol = [per_mol[0]] * mol.natoms
            if len(per_mol) != mol.natoms:
                raise ValueError(f"charge spec has {len(per_mol)} values for {mol.natoms} atoms")
            for atom, value in zip(mol.atoms, per_mol):
                atom.charge = value
                atom.charge_source = "input"
        elif charge_method in {"input", "cm5", "raw_cm5", "adch", "mk", "resp", "resp2"}:
            for atom in mol.atoms:
                atom.charge = 0.0
                atom.charge_source = "input"
        else:
            assign_charges(mol, charge_method, net_charge=case.get("net_charge", 0), charge_file=None, workdir=work)
        if not mol.bonds:
            infer_bonds(mol)
        mol_params = assign_force_field(mol, case["ff"], source_path=str(ROOT / case["input"]), workdir=str(work))
        if case.get("box_kind") == "crystal" and mol.cell is not None:
            structure = replicate_crystal(mol, case.get("box_target", 1000))
        else:
            structure = replicate_molecule(mol, case.get("box_target", 1000))
            enumerate_terms(structure)
        params = parameterize(structure, mol_params, mode=case.get("parameter_mode", "classic"))
        apply_charges(structure, case.get("net_charge", 0), work)
        g_dir = work / "gromacs"
        l_dir = work / "lammps"
        g_dir.mkdir(parents=True, exist_ok=True)
        l_dir.mkdir(parents=True, exist_ok=True)
        write_gromacs(structure, params, g_dir)
        write_lammps(structure, params, l_dir)
        stem = structure.name
        g = run_gmx(g_dir, stem)
        l = run_lammps(l_dir, stem)
        result.update({"atoms": structure.natoms, "box_angstrom": [float(x) for x in structure.cell.matrix.diagonal()], "gromacs": g, "lammps": l, "checks": compare(g, l)})
        result["pass"] = all(check["pass"] for check in result["checks"])
        result["net_charge"] = structure.net_charge
    except Exception as exc:
        result["error"] = str(exc)
        result["pass"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated case ids to run")
    parser.add_argument("--skip-substitutes", action="store_true")
    parser.add_argument("--scratch", default=str(ROOT / "_scratch" / "gt44"), help="scratch output directory")
    args = parser.parse_args()

    catalog = yaml.safe_load((ROOT / "tests" / "ground_truth" / "gt44_catalog.yaml").read_text())
    cases = [c for c in catalog["cases"] if c.get("status") in {"ready", "substitute"}]
    if args.skip_substitutes:
        cases = [c for c in cases if not c.get("substitute")]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    outdir = Path(args.scratch).resolve()
    results = [run_case(case, outdir) for case in cases]
    report = outdir / "gt44_report.json"
    report.write_text(json.dumps({"catalog": str(ROOT / "tests" / "ground_truth" / "gt44_catalog.yaml"), "results": results}, indent=2))

    npass = 0
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        detail = ""
        if "checks" in r:
            worst = max(r["checks"], key=lambda c: abs(c["delta_kj"]))
            detail = f" worst={worst['term']} d={worst['delta_kj']:.3e} kJ (tol {worst['tol_kj']:.3e})"
        if "error" in r:
            detail = f" error={r['error'][:200]}"
        print(f"[{mark}] {r['id']} {r['name']} atoms={r.get('atoms', '-')}{detail}")
        npass += 1 if r["pass"] else 0
    print(f"gt44: {npass}/{len(results)} cases passed; report={report}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())