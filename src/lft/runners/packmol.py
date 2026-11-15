"""Packmol wrapper."""

from __future__ import annotations

from pathlib import Path
import shutil

from .external import CommandResult, run_command
from ..errors import ExternalToolError


def run_packmol(*, input_file: str | Path, workdir: str | Path) -> CommandResult:
    executable = shutil.which("packmol") or "/opt/packmol/packmol"
    if not Path(executable).exists():
        raise ExternalToolError("Packmol is not available")
    return run_command([executable], cwd=workdir, stdin=Path(input_file).read_text())
