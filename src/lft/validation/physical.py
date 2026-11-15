"""Energy formula spot checks used by E2E validation."""

from __future__ import annotations

import math


def bond_energy(distance: float, equilibrium: float, force_constant: float) -> float:
    return 0.5 * force_constant * (distance - equilibrium) ** 2


def angle_energy(angle_degree: float, equilibrium_degree: float, force_constant: float) -> float:
    return 0.5 * force_constant * math.radians(angle_degree - equilibrium_degree) ** 2


def lj_energy(distance: float, sigma: float, epsilon: float) -> float:
    ratio = sigma / distance
    ratio6 = ratio**6
    return 4.0 * epsilon * (ratio6**2 - ratio6)
