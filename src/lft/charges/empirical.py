"""Empirical charge adapters."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from ..domain.structure import Structure
from ..errors import ChargeError, ExternalToolError


def assign_mmff94(structure: Structure, *, source_path: str | None = None, workdir: str | Path | None = None) -> dict[str, object]:
    """Use OpenBabel's MMFF94 charge model through its Python API or CLI."""

    try:
        charges = _mmff94_python(structure, source_path)
        _set(structure, charges, "mmff94")
        return {"backend": "openbabel-python", "version": _openbabel_version()}
    except Exception as python_error:
        executable = os.environ.get("LFT_OBABEL") or shutil.which("obabel")
        if executable and source_path:
            try:
                charges = _mmff94_cli(executable, source_path, workdir)
                _set(structure, charges, "mmff94")
                return {"backend": executable, "version": _openbabel_version(), "python_error": str(python_error)}
            except Exception as cli_error:
                raise ExternalToolError(f"OpenBabel MMFF94 failed: {cli_error}") from cli_error
        raise ExternalToolError(f"OpenBabel MMFF94 is unavailable: {python_error}", hint="install openbabel-wheel or choose input charges") from python_error


def assign_am1bcc(structure: Structure, *, source_path: str | None, net_charge: float, workdir: str | Path) -> dict[str, object]:
    """AM1-BCC charges.

    Production path is pure Python (``lft.charges.semiempirical.bcc``).  The
    external antechamber/acpype chain is only used when
    ``LFT_AM1BCC_EXTERNAL=1`` is set (validation/comparison mode).
    """
    if os.environ.get("LFT_AM1BCC_EXTERNAL", "0") == "1":
        if shutil.which("antechamber"):
            return _assign_am1bcc_antechamber(structure, net_charge=net_charge, workdir=workdir)
        if source_path is None:
            raise ChargeError("AM1-BCC external backend requires a structure source path")
        try:
            from .acpype import run_acpype_am1bcc

            charges, metadata = run_acpype_am1bcc(structure, source_path=source_path, net_charge=net_charge, workdir=workdir)
            _set(structure, list(charges), "am1-bcc")
            return metadata
        except ExternalToolError as exc:
            raise ExternalToolError("AM1-BCC external backend unavailable", hint="install AmberTools/acpype") from exc

    from .semiempirical.bcc import AM1BCC

    py_charges, metadata = AM1BCC(structure).assign()
    _set(structure, [float(c) for c in py_charges], "am1-bcc")
    return metadata


def _assign_am1bcc_antechamber(structure: Structure, *, net_charge: float, workdir: str | Path) -> dict[str, object]:
    executable = shutil.which("antechamber")
    if not executable:
        raise ExternalToolError("AM1-BCC requested but antechamber is not available", hint="install AmberTools or provide an input charge file")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = workdir / "am1bcc_input.mol2"
    output_path = workdir / "am1bcc_output.mol2"
    _write_minimal_mol2(structure, input_path)
    command = [executable, "-i", str(input_path), "-fi", "mol2", "-o", str(output_path), "-fo", "mol2", "-c", "bcc", "-nc", str(int(net_charge)), "-at", "gaff", "-pf", "y"]
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise ExternalToolError(f"antechamber AM1-BCC failed with exit code {result.returncode}: {(result.stderr or result.stdout)[-800:]}")
    charges = _read_mol2_charges(output_path, structure.natoms)
    _set(structure, charges, "am1-bcc")
    return {"backend": executable, "command": command, "stdout": result.stdout, "stderr": result.stderr}


def _mmff94_python(structure: Structure, source_path: str | None) -> list[float]:
    from openbabel import openbabel as ob  # type: ignore[import-not-found]

    if source_path:
        conversion = ob.OBConversion()
        if not conversion.SetInFormat(Path(source_path).suffix.lstrip(".")):
            raise ValueError(f"OpenBabel cannot read {source_path}")
        molecule = ob.OBMol()
        if not conversion.ReadFile(molecule, source_path):
            raise ValueError(f"OpenBabel cannot parse {source_path}")
    else:
        molecule = ob.OBMol()
        editable = ob.OBMolAtomIter(molecule)
        del editable
        raise ValueError("OpenBabel Python MMFF94 needs a source structure path")
    model = ob.OBChargeModel.FindType("mmff94")
    if model is None or not model.ComputeCharges(molecule):
        raise ValueError("OpenBabel MMFF94 charge model could not compute charges")
    values = [float(atom.GetPartialCharge()) for atom in ob.OBMolAtomIter(molecule)]
    if len(values) != structure.natoms:
        raise ValueError("OpenBabel changed atom count/order")
    return values


def _mmff94_cli(executable: str, source_path: str, workdir: str | Path | None) -> list[float]:
    command = [executable, source_path, "-omolreport", "--partialcharge", "MMFF94"]
    result = subprocess.run(command, cwd=workdir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(result.stderr[-500:])
    values: dict[int, float] = {}
    pattern = re.compile(r"^ATOM:\s*(\d+).*?CHARGE:\s*(-?[0-9.]+)")
    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match:
            values[int(match.group(1))] = float(match.group(2))
    if not values:
        raise ValueError("OpenBabel molreport contained no charges")
    return [values[index] for index in range(1, max(values) + 1)]


def _openbabel_version() -> str:
    try:
        from openbabel import openbabel as ob  # type: ignore[import-not-found]

        return str(ob.OBReleaseVersion())
    except Exception:
        return "unknown"


def _set(structure: Structure, values: list[float], source: str) -> None:
    if len(values) != structure.natoms:
        raise ChargeError(f"{source} returned {len(values)} charges for {structure.natoms} atoms")
    for atom, value in zip(structure.atoms, values):
        atom.charge = float(value)
        atom.charge_source = source


def _read_mol2_charges(path: Path, count: int) -> list[float]:
    values: dict[int, float] = {}
    in_atoms = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.upper() == "@<TRIPOS>ATOM":
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms = False
        if in_atoms and line:
            fields = line.split()
            if len(fields) >= 9:
                values[int(fields[0])] = float(fields[8])
    if len(values) != count:
        raise ExternalToolError(f"AM1-BCC output has {len(values)} atom charges; expected {count}")
    return [values[index] for index in range(1, count + 1)]


def _write_minimal_mol2(structure: Structure, path: Path) -> None:
    lines = ["@<TRIPOS>MOLECULE", structure.name, f"{structure.natoms} {len(structure.bonds)} 1 0 0", "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for atom in structure.atoms:
        lines.append(f"{atom.index:7d} {atom.name:<8s} {atom.position[0]:12.6f} {atom.position[1]:12.6f} {atom.position[2]:12.6f} {atom.element:<6s} {atom.residue_id:4d} {atom.residue_name:<8s} {atom.charge: .8f}")
    lines.append("@<TRIPOS>BOND")
    for index, bond in enumerate(structure.bonds, start=1):
        lines.append(f"{index:7d} {bond.i:4d} {bond.j:4d} {int(round(bond.order))}")
    path.write_text("\n".join(lines) + "\n")
