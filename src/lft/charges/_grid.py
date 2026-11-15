"""Deterministic quadrature grid replicating Multiwfn ``gen1cintgrid``.

Radial part is the second-kind Gauss-Chebyshev / Becke transform with
``radpot=75`` shells and ``radcut=10`` Bohr (the Multiwfn defaults in
``settings.ini`` and ``define.f90``); angular part is the 434-point Lebedev
grid (order 41) taken from pySCF, whose points/weights are the standard
Lebedev-Laikov values used by Multiwfn's ``Lebedevgen``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RADIAL_POINTS = 75
ANGULAR_POINTS = 434
RADCUT_BOHR = 10.0


def radial_shells() -> tuple[np.ndarray, np.ndarray]:
    """Return (radr_bohr, radw) for shells inside ``RADCUT_BOHR``.

    Replicates Multiwfn ``gen1cintgrid``: ``radr=(1+x)/(1-x)``,
    ``radw=8 pi^2/(N+1) (1+x)^2.5/(1-x)^3.5``, keeping only ``radr<radcut``.
    """
    index = np.arange(1, RADIAL_POINTS + 1, dtype=float)
    radx = np.cos(index * np.pi / (RADIAL_POINTS + 1))
    radr = (1.0 + radx) / (1.0 - radx)
    radw = 8.0 * np.pi**2 / (RADIAL_POINTS + 1) * (1.0 + radx) ** 2.5 / (1.0 - radx) ** 3.5
    mask = radr < RADCUT_BOHR
    return radr[mask], radw[mask]


def _lebedev() -> np.ndarray:
    from pyscf.dft import LebedevGrid  # type: ignore[import-not-found]

    return np.asarray(LebedevGrid.MakeAngularGrid(ANGULAR_POINTS), dtype=float)


def atomic_grid(center_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-center quadrature grid around ``center_bohr``.

    Returns ``(points_bohr (n,3), weights (n,))``.  ``radw`` already carries the
    ``4 pi r^2`` radial factor, so the Lebedev angular weights are normalized to
    sum to one (Multiwfn convention).
    """
    radr, radw = radial_shells()
    angular = _lebedev()
    directions = angular[:, :3]
    angular_weights = angular[:, 3]
    points = center_bohr[None, None, :] + radr[:, None, None] * directions[None, :, :]
    weights = radw[:, None] * angular_weights[None, :]
    return points.reshape(-1, 3), weights.reshape(-1)


def _grid_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radr, radw = radial_shells()
    angular = _lebedev()
    return radr, radw, angular[:, :3], angular[:, 3]


def atomic_grid_separated(center_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-shell grid data: (points (nrad,nang,3), radial weights, angular weights)."""
    radr, radw, directions, angular_weights = _grid_cache()
    points = center_bohr[None, None, :] + radr[:, None, None] * directions[None, :, :]
    return points, radw, angular_weights