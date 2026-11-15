"""Deterministic JSON serialization for normalized artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
