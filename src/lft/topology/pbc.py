"""Integer cycle winding invariants for periodic molecular graphs."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..domain.structure import ImageShift, Structure


def compute_winding(structure: Structure) -> dict[str, Any]:
    """Return cycle residuals and their integer rank on the periodic torus."""

    images: dict[int, ImageShift] = {}
    residuals: list[ImageShift] = []
    neighbors: dict[int, list[tuple[int, ImageShift]]] = {i: [] for i in range(1, structure.natoms + 1)}
    for bond in structure.bonds:
        neighbors[bond.i].append((bond.j, bond.image_shift))
        neighbors[bond.j].append((bond.i, tuple(-x for x in bond.image_shift)))
    for root in range(1, structure.natoms + 1):
        if root in images:
            continue
        images[root] = (0, 0, 0)
        queue: deque[int] = deque([root])
        tree_edges: set[tuple[int, int]] = set()
        while queue:
            current = queue.popleft()
            for neighbor, shift in neighbors[current]:
                edge = (min(current, neighbor), max(current, neighbor))
                if neighbor not in images:
                    images[neighbor] = _add(images[current], shift)
                    tree_edges.add(edge)
                    queue.append(neighbor)
                elif edge not in tree_edges:
                    residual = _sub(shift, _sub(images[neighbor], images[current]))
                    if residual != (0, 0, 0):
                        # Each undirected non-tree edge can be encountered
                        # twice; retain one deterministic orientation.
                        canonical = residual if current < neighbor else _neg(residual)
                        if canonical not in residuals:
                            residuals.append(canonical)
    residuals.sort()
    rank = _integer_rank(residuals)
    basis: list[ImageShift] = []
    for residual in residuals:
        if _integer_rank(basis + [residual]) > len(basis):
            basis.append(residual)
        if len(basis) == rank:
            break
    return {
        "rank": rank,
        "residuals": [list(value) for value in residuals],
        "basis": [list(value) for value in basis],
        "atom_images": {str(index): list(image) for index, image in sorted(images.items())},
    }


def winding_rank(structure: Structure) -> int:
    return int(compute_winding(structure)["rank"])


def _add(a: ImageShift, b: ImageShift) -> ImageShift:
    return tuple(int(x + y) for x, y in zip(a, b))  # type: ignore[return-value]


def _sub(a: ImageShift, b: ImageShift) -> ImageShift:
    return tuple(int(x - y) for x, y in zip(a, b))  # type: ignore[return-value]


def _neg(a: ImageShift) -> ImageShift:
    return tuple(-x for x in a)  # type: ignore[return-value]


def _integer_rank(vectors: list[ImageShift]) -> int:
    if not vectors:
        return 0
    matrix = np.asarray(vectors, dtype=float)
    return int(np.linalg.matrix_rank(matrix, tol=1.0e-10))
