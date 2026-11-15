"""Simple backend-independent differential reports."""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_arrays(left: Any, right: Any) -> dict[str, float | bool]:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.shape != rhs.shape:
        return {"pass": False, "rms": float("inf"), "max_abs": float("inf")}
    delta = np.abs(lhs - rhs)
    return {"pass": bool(np.allclose(lhs, rhs, rtol=1.0e-6, atol=1.0e-5)), "rms": float(np.sqrt(np.mean(delta**2))), "max_abs": float(delta.max(initial=0.0))}
