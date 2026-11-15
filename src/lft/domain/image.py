"""Public image-shift helpers."""

from __future__ import annotations

import numpy as np

from .structure import Cell, ImageShift


def minimum_image_shift(cell: Cell, displacement: np.ndarray) -> ImageShift:
    return cell.nearest_shift(np.asarray(displacement, dtype=float))


def lifted_displacement(cell: Cell | None, displacement: np.ndarray, shift: ImageShift) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=float)
    return displacement + (cell.image_vector(shift) if cell is not None else 0.0)


def minimum_image_displacement(cell: Cell | None, origin: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ImageShift]:
    displacement = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    shift = cell.nearest_shift(displacement) if cell is not None else (0, 0, 0)
    return lifted_displacement(cell, displacement, shift), shift


__all__ = ["Cell", "ImageShift", "lifted_displacement", "minimum_image_displacement", "minimum_image_shift"]
