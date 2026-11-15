"""moltemplate wrapper."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..errors import ExternalToolError
from .external import CommandResult, run_command


def run_moltemplate(*, lt_file: str | Path, workdir: str | Path) -> CommandResult:
    executable = os.environ.get("LFT_MOLTEMPLATE") or shutil.which("moltemplate.sh") or "/opt/moltemplate/moltemplate/scripts/moltemplate.sh"
    if not Path(executable).exists():
        raise ExternalToolError("moltemplate.sh is not available")
    return run_command([executable, str(Path(lt_file).resolve())], cwd=workdir)
