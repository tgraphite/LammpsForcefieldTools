"""Safe subprocess execution with captured provenance."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..errors import ExternalToolError


@dataclass(slots=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    environment: dict[str, str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    command: Iterable[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    allowed_environment: Iterable[str] = (),
    stdin: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    argv = [str(value) for value in command]
    if not argv:
        raise ExternalToolError("cannot run an empty external command")
    executable = shutil.which(argv[0]) or (argv[0] if Path(argv[0]).exists() else None)
    if executable is None:
        raise ExternalToolError(f"external executable is not available: {argv[0]}")
    workdir = Path(cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    allowed = set(allowed_environment)
    environment = {
        key: process_env[key] for key in sorted(allowed) if key in process_env
    }
    try:
        result = subprocess.run(
            argv,
            cwd=workdir,
            env=process_env,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ExternalToolError(
            f"external command could not complete: {argv[0]}: {exc}"
        ) from exc
    captured = CommandResult(
        argv,
        str(workdir.resolve()),
        result.returncode,
        result.stdout,
        result.stderr,
        environment,
    )
    if not captured.ok:
        raise ExternalToolError(
            f"command failed with exit code {captured.returncode}: {' '.join(argv)}\n{captured.stdout[-1000:]}\n{captured.stderr[-1000:]}"
        )
    return captured
