"""Run manifest, provenance, hashes, and warning audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import platform
import sys

from .io.serializers import dump_json


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class Audit:
    run_id: str
    config: dict[str, Any]
    tasks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, config: dict[str, Any], input_hash: str) -> "Audit":
        run_id = hashlib.sha256((input_hash + stable_hash(config)).encode()).hexdigest()[:16]
        return cls(run_id, config, metadata={"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0], "platform": platform.platform()})

    def task(self, name: str, **details: Any) -> None:
        self.tasks.append({"name": name, **details})

    def add_artifact(self, name: str, path: str | Path, *, kind: str | None = None) -> None:
        path = Path(path)
        self.artifacts[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "kind": kind}

    def add_warning(self, warning: dict[str, Any]) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lft.audit.v1",
            "run_id": self.run_id,
            "config": self.config,
            "tasks": self.tasks,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def write(self, path: str | Path) -> None:
        dump_json(self.to_dict(), path)
