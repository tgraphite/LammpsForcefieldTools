"""Project-local LAMMPS wrapper."""

from __future__ import annotations

import os
from pathlib import Path

from .external import CommandResult, run_command


def lammps_executable(project_root: str | Path | None = None) -> str:
    import shutil

    configured = os.environ.get("LFT_LMP")
    if configured:
        return configured
    if project_root is not None:
        candidate = Path(project_root) / "thirdparty" / "lammps" / "bin" / "lmp"
        if candidate.exists():
            return str(candidate)
    return shutil.which("lmp") or "lmp"


def run_lammps(*, workdir: str | Path, input_file: str, project_root: str | Path | None = None) -> CommandResult:
    executable = lammps_executable(project_root)
    library = Path(project_root) / "thirdparty" / "lammps" / "lib" if project_root else None
    env = {"LD_LIBRARY_PATH": str(library)} if library and library.exists() else None
    return run_command([executable, "-in", input_file], cwd=workdir, env=env)
