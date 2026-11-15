"""Controlled adapter for the acpype (AmberTools) AM1-BCC charge path.

The pip-distributed ``acpype`` bundles an AmberTools runtime.  Its scripts
only work when the bundled library directory is on ``LD_LIBRARY_PATH``; this
module locates that directory and records every command and artifact in the
audit trail, mirroring the rest of the external-runner layer.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

from ..domain.structure import Structure
from ..errors import ChargeError, ExternalToolError
from ..runners.external import run_command


def run_acpype_am1bcc(
    structure: Structure,
    *,
    source_path: str,
    net_charge: float,
    workdir: str | Path,
) -> tuple[list[float], dict[str, object]]:
    """Run ``acpype -c bcc -a gaff`` and read the resulting mol2 charges."""
    executable = _discover_acpype()
    if executable is None:
        raise ExternalToolError(
            "acpype executable was not found",
            hint="install the acpype package into the active environment",
        )
    work = Path(workdir) / "acpype"
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "input.mol2"
    _write_mol2(structure, input_path)
    basename = "acpype_out"
    command = [
        str(executable),
        "-i", str(input_path),
        "-a", "gaff",
        "-n", str(int(net_charge)),
        "-c", "bcc",
        "-o", "gmx",
        "-b", basename,
        "-d",
    ]
    amber_root = _amber_root(executable)
    result = run_command(command, cwd=work, env=_environment(amber_root))
    result_dir = work / f"{basename}.acpype"
    mol2_path = result_dir / f"{basename}_bcc_gaff.mol2"
    if not mol2_path.exists():
        mol2_path = work / f"{basename}_bcc_gaff.mol2"
    if not mol2_path.exists():
        mol2_path = work / f"{basename}_{int(net_charge)}_gaff.mol2"
    if not mol2_path.exists():
        candidates = sorted((result_dir if result_dir.is_dir() else work).glob(f"{basename}*bcc*.mol2"))
        if candidates:
            mol2_path = candidates[0]
        else:
            raise ExternalToolError(
                f"acpype completed without a bcc mol2 artifact in {work}",
                hint="check acpype stdout/stderr in the work directory",
            )
    charges = _read_mol2_charges(mol2_path, structure.natoms)
    metadata: dict[str, object] = {
        "backend": "acpype",
        "executable": str(executable),
        "amber_root": str(amber_root) if amber_root is not None else None,
        "command": result.command,
        "input": str(input_path),
        "output": str(mol2_path),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return list(charges), metadata


def _discover_acpype() -> Path | None:
    configured = os.environ.get("LFT_ACPYPE")
    candidates = [Path(configured)] if configured else []
    on_path = shutil.which("acpype")
    if on_path:
        candidates.append(Path(on_path))
    if sys.prefix and (Path(sys.prefix) / "bin" / "acpype").is_file():
        candidates.append(Path(sys.prefix) / "bin" / "acpype")
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate.resolve()
    return None


def _amber_root(executable: Path) -> Path | None:
    """Return the bundled ``amber_linux`` directory if one exists nearby."""
    for parent in (executable.parent.parent / "site-packages" / "acpype", Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "acpype"):
        candidate = parent / "amber_linux"
        if candidate.is_dir():
            return candidate
    return None


def _environment(amber_root: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if amber_root is not None:
        env["AMBERHOME"] = str(amber_root)
        env["PATH"] = str(amber_root / "bin") + os.pathsep + os.environ.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(amber_root / "lib") + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    return env


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
        raise ChargeError(f"acpype returned {len(values)} atom charges for {count} atoms")
    return [values[index] for index in range(1, count + 1)]


def _write_mol2(structure: Structure, path: Path) -> None:
    lines = ["@<TRIPOS>MOLECULE", structure.name, f"{structure.natoms} {len(structure.bonds)} 1 0 0", "SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"]
    for atom in structure.atoms:
        lines.append(
            f"{atom.index:7d} {atom.name:<8s} {atom.position[0]:12.6f} "
            f"{atom.position[1]:12.6f} {atom.position[2]:12.6f} {atom.element:<6s} "
            f"{atom.residue_id:4d} {atom.residue_name:<8s} {atom.charge: .8f}"
        )
    lines.append("@<TRIPOS>BOND")
    for index, bond in enumerate(structure.bonds, start=1):
        order = "Ar" if 1.3 <= bond.order <= 1.7 else str(int(round(bond.order)))
        lines.append(f"{index:7d} {bond.i:4d} {bond.j:4d} {order}")
    path.write_text("\n".join(lines) + "\n")