#!/usr/bin/env python3
"""Build the non-periodic small-molecule validation suite.

Dev-only tool.  It generates 3D MOL2 inputs from SMILES with OpenBabel and
freezes the AmberTools atomtype oracle output as the 1:1 GAFF typing
reference.  The resulting fixtures are what the test suite consumes, so CI
does not need the oracle binary.

    python scripts/build_validation_suite.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "validation"
sys.path.insert(0, str(ROOT / "src"))

from lft.io.loaders import load_structure  # noqa: E402
from lft.validation.gaff_oracle import OracleConfig, run_oracle  # noqa: E402

MOLECULES = [
    ("water", "O", 0),
    ("methane", "C", 0),
    ("ethanol", "CCO", 0),
    ("acetone", "CC(=O)C", 0),
    ("benzene", "c1ccccc1", 0),
    ("pyridine", "c1ccncc1", 0),
    ("acetamide", "CC(N)=O", 0),
    ("acetate", "CC(=O)[O-]", -1),
    ("methylammonium", "C[NH3+]", 1),
    ("phenol", "c1ccc(O)cc1", 0),
    ("imidazole", "c1cnc[nH]1", 0),
    ("naphthalene", "c1ccc2ccccc2c1", 0),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 0),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 0),
    ("cyclohexane", "C1CCCCC1", 0),
    ("dmso", "CS(C)=O", 0),
    ("trifluoroethanol", "FC(F)(F)CO", 0),
    ("diethyl_ether", "CCOCC", 0),
    ("aniline", "Nc1ccccc1", 0),
    ("thiophene", "c1ccsc1", 0),
    ("acetonitrile", "CC#N", 0),
    ("formaldehyde", "C=O", 0),
]


def build_mol2(smiles: str, net_charge: int, path: Path) -> None:
    from openbabel import openbabel as ob

    conversion = ob.OBConversion()
    conversion.SetInAndOutFormats("smi", "mol2")
    molecule = ob.OBMol()
    if not conversion.ReadString(molecule, smiles):
        raise SystemExit(f"OpenBabel could not parse SMILES {smiles!r}")
    molecule.AddHydrogens()
    molecule.SetTotalCharge(net_charge)
    ob.OBBuilder().Build(molecule)
    if not conversion.WriteFile(molecule, str(path)):
        raise SystemExit(f"OpenBabel could not write {path}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    molecules = OUT / "molecules"
    molecules.mkdir(exist_ok=True)
    config = OracleConfig.from_environment()
    cases = {}
    for name, smiles, net_charge in MOLECULES:
        path = molecules / f"{name}.mol2"
        build_mol2(smiles, net_charge, path)
        structure = load_structure(str(path))
        structure.declared_net_charge = float(net_charge)
        types = run_oracle(structure, config, OUT / "oracle_work")
        cases[name] = {
            "smiles": smiles,
            "net_charge": net_charge,
            "mol2": f"molecules/{name}.mol2",
            "natoms": structure.natoms,
            "mol2_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "gaff_types": {str(index): value for index, value in sorted(types.items())},
        }
        print(f"{name}: {structure.natoms} atoms, {len(types)} types", flush=True)
    payload = {
        "schema": "lft.validation.small_molecules.v1",
        "provenance": {
            "generator": "scripts/build_validation_suite.py",
            "oracle": "AmberTools26 atomtype",
            "oracle_sha256": hashlib.sha256(config.executable.read_bytes()).hexdigest(),
            "definition": str(config.definition),
            "note": "Non-periodic molecules only; every case has a 1:1 GAFF type reference.",
        },
        "cases": cases,
    }
    (OUT / "reference.json").write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT / 'reference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
