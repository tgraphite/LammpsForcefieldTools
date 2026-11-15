"""GROMACS wrapper."""

from __future__ import annotations

import os
from pathlib import Path

from .external import CommandResult, run_command


def gmx_executable() -> str:
    import shutil

    return os.environ.get("LFT_GMX") or shutil.which("gmx") or "/usr/local/gromacs/bin/gmx"


def grompp(*, workdir: str | Path, mdp: str, gro: str, top: str, output: str = "topol.tpr") -> CommandResult:
    return run_command([gmx_executable(), "grompp", "-f", mdp, "-c", gro, "-p", top, "-o", output], cwd=workdir)


def mdrun(*, workdir: str | Path, tpr: str, deffnm: str = "md") -> CommandResult:
    return run_command([gmx_executable(), "mdrun", "-s", tpr, "-deffnm", deffnm], cwd=workdir)
