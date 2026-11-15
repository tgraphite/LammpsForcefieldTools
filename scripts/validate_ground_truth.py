#!/usr/bin/env python3
"""Validate every checked-in structural ground-truth manifest."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lft.validation.ground_truth import validate_manifest


def main() -> int:
    manifests = sorted((ROOT / "tests" / "ground_truth").glob("*/manifest.y*ml"))
    for manifest in manifests:
        result = validate_manifest(manifest, root=ROOT)
        actual = result["actual"]
        print(f"{result['id']}: atoms={actual['atoms']} bonds={actual['bonds']} cross_edges={actual['cross_edges']} winding_rank={actual['winding_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
