#!/usr/bin/env python3
"""Extract Multiwfn's built-in sphericalized atomic densities into package data.

Reads the reference-only Fortran source under ``thirdparty/`` and writes a
compressed numpy asset used by the internal Hirshfeld/CM5/ADCH engine so the
production code has no runtime dependency on the external Multiwfn binary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thirdparty" / "Multiwfn_2026.8.31_src_Linux" / "atmraddens.f90"
OUT = ROOT / "src" / "lft" / "charges" / "data" / "multiwfn_atmraddens.npz"

POSARR_RE = re.compile(r"posarr\(\s*(\d+)\)\s*=\s*([0-9.ED+-]+)")
ELEM_START_RE = re.compile(r"(?:else )?if \(iele==\s*(\d+)\) then")
NPT_RE = re.compile(r"npt=\s*(\d+)")
RHO_RE = re.compile(r"rhoarr\(\s*(\d+)\)\s*=\s*([0-9.ED+-]+)")


def parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def main() -> int:
    if not SOURCE.exists():
        print(f"missing source: {SOURCE}", file=sys.stderr)
        return 1
    text = SOURCE.read_text()

    positions = np.zeros(200, dtype=np.float64)
    for index, value in POSARR_RE.findall(text.split("subroutine genatmradpos")[1].split("subroutine genatmraddens")[0]):
        positions[int(index) - 1] = parse_float(value)

    densities: dict[int, tuple[int, np.ndarray]] = {}
    start = text.index("if (iele== 1) then")
    end = text.index("\nend if\n", start)
    lines = text[start:end]
    current: int | None = None
    npt = 0
    values: list[float] = []
    for raw in lines.splitlines():
        line = raw.strip()
        start = ELEM_START_RE.match(line)
        if start:
            if current is not None:
                densities[current] = (npt, np.array(values, dtype=np.float64))
            current = int(start.group(1))
            npt = 0
            values = []
            continue
        npt_match = NPT_RE.match(line)
        if npt_match and current is not None:
            npt = int(npt_match.group(1))
            continue
        rho = RHO_RE.match(line)
        if rho and current is not None:
            values.append(parse_float(rho.group(2)))
    if current is not None:
        densities[current] = (npt, np.array(values, dtype=np.float64))

    max_z = max(densities)
    packed = np.zeros((max_z + 1, 200), dtype=np.float64)
    npts = np.zeros(max_z + 1, dtype=np.int64)
    for z in range(1, max_z + 1):
        npts[z] = densities[z][0]
        packed[z, : npts[z]] = densities[z][1]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, radial_positions_bohr=positions, densities=packed, npts=npts)
    print(f"wrote {OUT}")
    print(f"elements 1..{max_z}, radial points {len(positions)}, max_npt={npts.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())