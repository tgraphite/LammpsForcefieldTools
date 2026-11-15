"""Controlled adapter for AmberTools ``atomtype``.

Production typing is pure Python (``lft.forcefields.gaff_typer``).  The
external ``atomtype`` binary is only used when ``LFT_ATOMTYPE_EXTERNAL=1`` is
set (validation/comparison mode); it is never required.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from ..domain.structure import Structure
from ..errors import ExternalToolError
from ..runners.external import run_command
from .gaff_typer import assign_gaff_types


def use_external_atomtype() -> bool:
    """Whether to use the external atomtype binary (validation mode)."""
    return os.environ.get("LFT_ATOMTYPE_EXTERNAL", "0") == "1"


def run_python_atomtype(
    structure: Structure, *, definition: str | Path | None = None
) -> tuple[dict[int, str], dict[str, object]]:
    """Run pure-Python GAFF typing and return (types, metadata)."""
    from importlib import resources

    if definition is None:
        definition = Path(
            str(resources.files("lft.forcefields.data").joinpath("ATOMTYPE_GFF.DEF"))
        )
    atom_types = assign_gaff_types(
        structure, definition=definition, pindex=0, post_adjustment=True
    )
    metadata: dict[str, object] = {
        "backend": "pure-python",
        "definition": str(Path(definition)),
        "definition_sha256": hashlib.sha256(Path(definition).read_bytes()).hexdigest(),
        "engine": "lft.forcefields.gaff_typer",
        "external_binary": None,
    }
    return atom_types, metadata


def discover_atomtype() -> Path | None:
    """Find the Linux atomtype executable without requiring it on PATH.

    Only consulted when ``LFT_ATOMTYPE_EXTERNAL=1`` (validation mode).
    """
    configured = os.environ.get("LFT_ATOMTYPE")
    candidates = [Path(configured)] if configured else []
    candidates.append(
        Path(__file__).resolve().parents[3] / "references/sobtop/atomtype"
    )
    on_path = shutil.which("atomtype")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate.resolve()
    return None


def run_atomtype(
    structure: Structure, *, workdir: str | Path
) -> tuple[dict[int, str], dict[str, object]]:
    """Run external GAFF typing and parse its ``.ac`` result."""

    executable = discover_atomtype()
    if executable is None:
        raise ExternalToolError(
            "GAFF atomtype executable was not found",
            hint="set LFT_ATOMTYPE or install AmberTools/Sobtop atomtype",
        )
    base = Path(workdir).resolve() / "atomtype"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix="case-", dir=base))
    input_path = work / "input.mol2"
    output_path = work / "output.ac"
    _write_mol2(structure, input_path)

    definition = _definition_file(executable)
    command = [
        str(executable),
        "-i",
        str(input_path),
        "-f",
        "mol2",
        "-o",
        str(output_path),
    ]
    if definition is not None:
        command.extend(["-d", str(definition), "-a", "1"])
    else:
        command.extend(["-p", "gaff"])
    # Temporary files belong to this invocation, never the installation.
    if definition is not None:
        shutil.copyfile(definition, work / "ATOMTYPE_GFF.DEF")
    result = run_command(command, cwd=work, timeout=120)
    if not output_path.exists():
        raise ExternalToolError(
            f"atomtype completed without producing output: {output_path}"
        )
    atom_types = _parse_ac(output_path)
    if set(atom_types) != {atom.index for atom in structure.atoms}:
        raise ExternalToolError(
            f"atomtype returned {len(atom_types)} atom types for {structure.natoms} atoms"
        )
    metadata: dict[str, object] = {
        "backend": "atomtype",
        "executable": str(executable),
        "definition": str(definition) if definition is not None else None,
        "command": result.command,
        "input": str(input_path),
        "output": str(output_path),
        "cwd": str(work),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return atom_types, metadata


def _definition_file(executable: Path) -> Path | None:
    configured = os.environ.get("LFT_ATOMTYPE_DEF")
    candidates = [Path(configured)] if configured else []
    candidates.append(executable.parent / "ATOMTYPE_GFF.DEF")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _parse_ac(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        fields = raw.split()
        if len(fields) >= 3 and fields[0].upper() == "ATOM":
            try:
                index = int(fields[1])
            except ValueError:
                continue
            atom_type = fields[-1].strip()
            if atom_type:
                values[index] = atom_type
    return values


def _write_mol2(structure: Structure, path: Path) -> None:
    charge_section = (
        "USER_CHARGES"
        if any(atom.charge_source for atom in structure.atoms)
        else "NO_CHARGES"
    )
    lines = [
        "@<TRIPOS>MOLECULE",
        structure.name,
        f"{structure.natoms} {len(structure.bonds)} 1 0 0",
        "SMALL",
        charge_section,
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in structure.atoms:
        lines.append(
            f"{atom.index:7d} {atom.name:<8s} {atom.position[0]:12.6f} "
            f"{atom.position[1]:12.6f} {atom.position[2]:12.6f} {atom.element:<6s} "
            f"{atom.residue_id:4d} {atom.residue_name:<8s} {atom.charge: .8f}"
        )
    lines.append("@<TRIPOS>BOND")
    for index, bond in enumerate(structure.bonds, start=1):
        lines.append(f"{index:7d} {bond.i:4d} {bond.j:4d} {_bond_type(bond.order)}")
    path.write_text("\n".join(lines) + "\n")


def _bond_type(order: float) -> str:
    # Preserve aromatic ~1.5 as "ar"; write true bond orders numerically so a
    # carbonyl C=O (order 2.0) is not mis-typed as an aromatic bond.
    if 1.3 <= float(order) <= 1.7:
        return "ar"
    return str(round(float(order)))
