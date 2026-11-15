"""Idempotency: repeated runs on identical input produce identical results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lft.charges.semiempirical.am1 import AM1
from lft.forcefields.gaff_typer import assign_gaff_types
from lft.io.loaders import load_structure


def _water() -> "object":
    return load_structure("references/sobtop/gt/probes/water.mol2")


def test_gaff_typer_is_idempotent() -> None:
    s = _water()
    first = assign_gaff_types(s)
    second = assign_gaff_types(s)
    assert first == second
    assert sorted(first) == [1, 2, 3]


def test_am1_single_point_is_deterministic() -> None:
    s = _water()
    coords = np.array([a.position for a in s.atoms])
    elements = [a.element for a in s.atoms]
    r1 = AM1(elements, coords).single_point()
    r2 = AM1(elements, coords).single_point()
    assert r1["energy_eV"] == r2["energy_eV"]
    assert np.array_equal(r1["mulliken_charges"], r2["mulliken_charges"])
    assert r1["n_iterations"] == r2["n_iterations"]


def test_stable_artifact_reproducibility() -> None:
    # Same input -> same charges and typing across two independent passes.
    from lft.charges.semiempirical.bcc import AM1BCC

    s = _water()
    charges1, meta1 = AM1BCC(s).assign(optimize=False)
    charges2, meta2 = AM1BCC(s).assign(optimize=False)
    assert np.allclose(charges1, charges2, atol=0.0)
    assert meta1["bcc_corrections"] == meta2["bcc_corrections"]
    assert meta1["mulliken_charges"] == meta2["mulliken_charges"]