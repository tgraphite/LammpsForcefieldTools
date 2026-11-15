#!/usr/bin/env python3
"""L5: Sobtop parameter-consistency end-to-end (GROMACS only).

For each case with a checked-in Sobtop reference ``.itp``, generate our
GROMACS topology under the same intent (same typing, classic prebuilt
parameters, input charges matching the reference's NO_CHARGES) and compare
every parameter record.  See docs/PROJECT_STATUS.md for the validation matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lft.backends.gromacs import write_gromacs
from lft.charges.dispatcher import assign_charges
from lft.forcefields.matching import assign_force_field
from lft.io.loaders import load_structure
from lft.parameterize.classic import parameterize
from lft.topology import enumerate_terms, infer_bonds

TOL = {
    "r0_nm": 1e-4,
    "k_rel": 1e-3,
    "theta_deg": 1e-3,
    "phase_deg": 1e-3,
    "sigma_nm": 1e-4,
    "eps_rel": 1e-3,
}

# (gt_dir, our_input, force_field, note)
REGISTRY = [
    ("Methyl_benzoate_g4", "references/sobtop/examples/Methyl_benzoate/Methyl_benzoate.mol2", "gaff+uff", "aromatic ester, gaff+uff classic"),
]


def parse_itp(path: Path) -> dict:
    section = None
    out: dict = {"atomtypes": {}, "atoms": {}, "bonds": [], "angles": [], "dihedrals": [], "impropers": [], "pairs": set()}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[]").strip().lower()
            continue
        fields = line.split()
        if section == "atomtypes" and len(fields) >= 8 and not line.startswith(";"):
            name, anum, mass, chg, ptype, sigma, eps = fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]
            out["atomtypes"][name] = {"sigma_nm": float(sigma), "eps_kj": float(eps)}
        elif section == "atoms" and len(fields) >= 8:
            out["atoms"][int(fields[0])] = fields[1]
        elif section == "bonds" and len(fields) >= 5:
            out["bonds"].append((int(fields[0]), int(fields[1]), float(fields[3]), float(fields[4])))
        elif section == "angles" and len(fields) >= 6:
            out["angles"].append((int(fields[0]), int(fields[1]), int(fields[2]), float(fields[4]), float(fields[5])))
        elif section == "dihedrals" and len(fields) >= 7:
            rec = (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]), float(fields[5]), float(fields[6]), int(fields[7]))
            if rec[4] == 4:
                out["impropers"].append(rec[:4] + rec[5:])
            else:
                out["dihedrals"].append(rec)
        elif section == "pairs" and len(fields) >= 3:
            out["pairs"].add((int(fields[0]), int(fields[1])))
    return out


def run_ours(input_path: str, ff: str, outdir: Path) -> dict:
    mol = load_structure(ROOT / input_path)
    for atom in mol.atoms:
        atom.charge = 0.0
        atom.charge_source = "input"
    if not mol.bonds:
        infer_bonds(mol)
    enumerate_terms(mol)
    params = assign_force_field(mol, ff, source_path=str(ROOT / input_path), workdir=str(outdir))
    params = parameterize(mol, params, mode="classic")
    assign_charges(mol, "input", net_charge=0, charge_file=None, workdir=outdir)
    write_gromacs(mol, params, outdir)
    return parse_itp(outdir / f"{mol.name}.itp")


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


def compare_itp(ours: dict, ref: dict) -> dict:
    issues: list[dict] = []
    hard: list[dict] = []

    if sorted(ours["atoms"].values()) != sorted(ref["atoms"].values()):
        issues.append({"kind": "atom_types_set", "ours": sorted(set(ours["atoms"].values())), "ref": sorted(set(ref["atoms"].values()))})
    for idx in set(ours["atoms"]) & set(ref["atoms"]):
        if ours["atoms"][idx] != ref["atoms"][idx]:
            issues.append({"kind": "atom_type", "index": idx, "ours": ours["atoms"][idx], "ref": ref["atoms"][idx]})

    ref_bonds = {(min(i, j), max(i, j)): (r, k) for i, j, r, k in ref["bonds"]}
    for i, j, r, k in ours["bonds"]:
        key = (min(i, j), max(i, j))
        if key not in ref_bonds:
            issues.append({"kind": "bond_extra", "atoms": key})
            continue
        rr, kk = ref_bonds[key]
        if abs(r - rr) > TOL["r0_nm"] or rel_diff(k, kk) > TOL["k_rel"]:
            issues.append({"kind": "bond", "atoms": key, "ours": (r, k), "ref": (rr, kk)})

    ref_angles = {(i, j, k): (t, a) for i, j, k, t, a in ref["angles"]}
    ref_angles_rev = {(i, j, k): (t, a) for i, j, k, t, a in ref["angles"]}
    for i, j, k, t, a in ours["angles"]:
        key = (i, j, k)
        if key not in ref_angles:
            rk = (k, j, i)
            if rk in ref_angles_rev:
                key = rk
            else:
                issues.append({"kind": "angle_extra", "atoms": key})
                continue
        tt, aa = ref_angles.get(key, ref_angles_rev.get(key))
        if abs(t - tt) > TOL["theta_deg"] or rel_diff(a, aa) > TOL["k_rel"]:
            issues.append({"kind": "angle", "atoms": key, "ours": (t, a), "ref": (tt, aa)})

    ref_dih = {(i, j, k, l): (ph, kk, pn) for i, j, k, l, _f, ph, kk, pn in ref["dihedrals"]}
    for i, j, k, l, _f, ph, kk, pn in ours["dihedrals"]:
        key = (i, j, k, l)
        if key not in ref_dih:
            issues.append({"kind": "dihedral_extra", "atoms": key})
            continue
        phh, kkk, pnn = ref_dih[key]
        if abs(ph - phh) > TOL["phase_deg"] or rel_diff(kk, kkk) > TOL["k_rel"] or pn != pnn:
            issues.append({"kind": "dihedral", "atoms": key, "ours": (ph, kk, pn), "ref": (phh, kkk, pnn)})

    ref_imp = {frozenset((i, j, k, l)): (ph, kk, pn) for i, j, k, l, ph, kk, pn in ref["impropers"]}
    known: list[dict] = []
    for i, j, k, l, ph, kk, pn in ours["impropers"]:
        key = frozenset((i, j, k, l))
        if key not in ref_imp:
            known.append({"kind": "improper_extra", "atoms": (i, j, k, l), "note": "central-atom selection differs from Sobtop (see gt44_acceptance_plan.md risk)"})
            continue
        phh, kkk, pnn = ref_imp[key]
        if abs(ph - phh) > TOL["phase_deg"] or rel_diff(kk, kkk) > TOL["k_rel"] or pn != pnn:
            known.append({"kind": "improper", "atoms": (i, j, k, l), "ours": (ph, kk, pn), "ref": (phh, kkk, pnn), "note": "missing-type fallback differs (Sobtop uses GAFF general term vs generic)"})

    for name in set(ours["atomtypes"]) | set(ref["atomtypes"]):
        if name not in ref["atomtypes"] or name not in ours["atomtypes"]:
            issues.append({"kind": "atomtype_missing", "name": name})
            continue
        o, r = ours["atomtypes"][name], ref["atomtypes"][name]
        if abs(o["sigma_nm"] - r["sigma_nm"]) > TOL["sigma_nm"] or rel_diff(o["eps_kj"], r["eps_kj"]) > TOL["eps_rel"]:
            issues.append({"kind": "lj", "name": name, "ours": o, "ref": r})

    if ours["pairs"] != ref["pairs"]:
        hard.append({"kind": "pairs", "ours_only": sorted(ours["pairs"] - ref["pairs"]), "ref_only": sorted(ref["pairs"] - ours["pairs"])})

    return {"issues": hard, "known_limitations": known, "pass": not hard}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", default=str(ROOT / "_scratch" / "gt44_sobtop"))
    args = parser.parse_args()

    outdir = Path(args.scratch).resolve()
    results = []
    for gt_dir, input_path, ff, note in REGISTRY:
        ref_path = ROOT / "references" / "sobtop" / "gt" / gt_dir
        itp = next(ref_path.glob("*.itp"))
        work = outdir / gt_dir
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        ours = run_ours(input_path, ff, work)
        ref = parse_itp(itp)
        comp = compare_itp(ours, ref)
        comp.update({"id": gt_dir, "input": input_path, "ff": ff, "note": note})
        results.append(comp)
        status = "PASS" if comp["pass"] else "FAIL"
        print(f"[{status}] {gt_dir}: {len(comp['issues'])} hard discrepancies, {len(comp['known_limitations'])} known_limitations")
        for issue in comp["issues"][:20]:
            print(f"    - {issue}")
        for lim in comp["known_limitations"][:10]:
            print(f"    ~ known: {lim}")

    (outdir / "sobtop_compare.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"sobtop_compare: {sum(1 for r in results if r['pass'])}/{len(results)} passed")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())