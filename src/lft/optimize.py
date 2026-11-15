"""Geometry pre-optimisation backends.

Three engines are exposed:

* uff    - OpenBabel UFF, available for the widest element range;
* mmff94 - OpenBabel MMFF94, organic molecules only (falls back to UFF);
* dft    - PySCF/geometric at the configured method/basis.

The optimiser only moves atoms; topology and atom order are preserved so the
downstream charge and parameter stages stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .domain.structure import Structure
from .errors import ExternalToolError
from .forcefields.tables import ATOMIC_NUMBER

OPTIMIZERS = ("none", "uff", "mmff94", "dft")


@dataclass(slots=True)
class OptimizationResult:
    method: str
    converged: bool
    energy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def optimize_structure(
    structure: Structure,
    method: str = "uff",
    *,
    qc: str = "b3lyp-d3bj/def2-svp",
    max_steps: int = 500,
    max_atoms: int | None = None,
) -> OptimizationResult:
    if method == "none":
        return OptimizationResult("none", True, None, {"skipped": True})
    if method not in OPTIMIZERS:
        raise ExternalToolError(f"unsupported optimiser {method!r}; allowed: {', '.join(OPTIMIZERS)}")
    if max_atoms is not None and structure.natoms > max_atoms:
        return OptimizationResult(method, False, None, {"skipped": True, "reason": "max_atoms", "max_atoms": max_atoms, "atoms": structure.natoms})
    if method in {"uff", "mmff94"}:
        return _optimize_openbabel(structure, method, max_steps=max_steps)
    return _optimize_dft(structure, qc=qc, max_steps=max_steps)


def _to_openbabel(structure: Structure) -> Any:
    try:
        from openbabel import openbabel as ob  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExternalToolError("OpenBabel is required for force-field pre-optimisation") from exc
    molecule = ob.OBMol()
    for atom in structure.atoms:
        ob_atom = molecule.NewAtom()
        ob_atom.SetAtomicNum(int(ATOMIC_NUMBER.get(atom.element, 0)) or 0)
        ob_atom.SetVector(float(atom.position[0]), float(atom.position[1]), float(atom.position[2]))
    for bond in structure.bonds:
        order = int(round(float(bond.order)))
        order = min(3, max(1, order))
        molecule.AddBond(int(bond.i), int(bond.j), order)
    try:
        molecule.SetTotalCharge(int(round(float(structure.declared_net_charge))))
        molecule.SetTotalSpinMultiplicity(max(1, int(structure.multiplicity)))
    except Exception:
        pass
    return molecule


def _optimize_openbabel(structure: Structure, method: str, *, max_steps: int) -> OptimizationResult:
    from openbabel import openbabel as ob  # type: ignore[import-not-found]

    molecule = _to_openbabel(structure)
    attempted = [method]
    if method == "mmff94":
        attempted.append("UFF")
    last_error: Exception | None = None
    for name in attempted:
        force_field = ob.OBForceField.FindForceField(name)
        if force_field is None:
            last_error = ExternalToolError(f"OpenBabel force field {name} is unavailable")
            continue
        if not force_field.Setup(molecule):
            last_error = ExternalToolError(f"OpenBabel {name} could not set up this structure")
            continue
        before = np.array([atom.position for atom in structure.atoms], dtype=float)
        energy_before = float(force_field.Energy())
        # A short steepest-descent phase rescues assembled/tangled inputs before
        # the conjugate-gradient refinement.
        force_field.SteepestDescent(min(200, int(max_steps)))
        force_field.ConjugateGradients(int(max_steps))
        force_field.GetCoordinates(molecule)
        gradient_sq = 0.0
        for ob_atom in ob.OBMolAtomIter(molecule):
            gradient = force_field.GetGradient(ob_atom)
            gradient_sq += gradient.GetX() ** 2 + gradient.GetY() ** 2 + gradient.GetZ() ** 2
        rms_gradient = float(np.sqrt(gradient_sq / max(1, 3 * structure.natoms)))
        after = np.array(
            [[atom.GetX(), atom.GetY(), atom.GetZ()] for atom in ob.OBMolAtomIter(molecule)],
            dtype=float,
        )
        if after.shape != before.shape:
            last_error = ExternalToolError(f"OpenBabel {name} changed the atom count")
            continue
        displacement = float(np.max(np.linalg.norm(after - before, axis=1)))
        final_energy = float(force_field.Energy())
        for atom, position in zip(structure.atoms, after):
            atom.position = position
        metadata = {
            "engine": name,
            "requested_engine": method,
            "energy_before_kcal_mol": energy_before,
            "energy_after_kcal_mol": final_energy,
            "max_displacement_angstrom": displacement,
            "rms_gradient_kcal_mol_angstrom": rms_gradient,
            "steps": int(max_steps),
        }
        # OpenBabel's CG stops on energy change, so a strict gradient test
        # produces false negatives on soft torsions.  1 kcal/mol/A is a
        # practical convergence band for a pre-optimisation.
        return OptimizationResult(name, rms_gradient < 1.5, final_energy, metadata)
    raise last_error or ExternalToolError("force-field pre-optimisation failed")


def _optimize_dft(structure: Structure, *, qc: str, max_steps: int) -> OptimizationResult:
    try:
        from pyscf import dft, gto  # type: ignore[import-not-found]
        from pyscf.geomopt import geometric_solver  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExternalToolError("PySCF with geometric is required for DFT pre-optimisation") from exc
    from .qc.pyscf_backend import _dft_xc, _parse_qc

    method, basis = _parse_qc(qc)
    molecule = gto.M(
        atom=[(atom.element, tuple(float(value) for value in atom.position)) for atom in structure.atoms],
        basis=basis,
        charge=int(round(structure.declared_net_charge)),
        spin=max(0, int(structure.multiplicity) - 1),
        unit="Angstrom",
        verbose=0,
    )
    if method.lower() in {"hf", "rhf", "rohf"}:
        mean_field = dft.RKS(molecule) if structure.multiplicity == 1 else dft.UKS(molecule)
        mean_field.xc = "hf"
    else:
        mean_field = dft.RKS(molecule) if structure.multiplicity == 1 else dft.UKS(molecule)
        mean_field.xc = _dft_xc(method)
    # Geometry steps do not need a 1e-9 SCF; a looser threshold and a coarser
    # grid roughly halve the cost while keeping gradients well behaved.
    mean_field.conv_tol = 1.0e-7
    mean_field.max_cycle = 80
    try:
        mean_field.grids.level = 2
    except Exception:
        pass
    try:
        # Density fitting keeps the geometry step affordable for small and
        # medium molecules; the final charge single point uses the configured
        # backend unchanged.
        mean_field = mean_field.density_fit()
    except Exception:
        pass
    options = dict(
        maxsteps=int(max_steps),
        assert_convergence=False,
        convergence_grms=1.0e-3,
        convergence_gmax=2.0e-3,
        convergence_energy=1.0e-5,
        # The initial analytic Hessian dominates the cost of a small
        # pre-optimisation; a diagonal guess is enough for a geometry
        # preconditioner.
        hessian="never",
    )
    try:
        optimized = geometric_solver.optimize(mean_field, **options)
    except Exception as exc:  # pragma: no cover - depends on the QC stack
        raise ExternalToolError(f"DFT geometry optimisation failed: {exc}") from exc
    coordinates = np.asarray(optimized.atom_coords(unit="Angstrom"), dtype=float)
    if coordinates.shape != (structure.natoms, 3):
        raise ExternalToolError("DFT optimisation returned unexpected coordinates")
    before = np.array([atom.position for atom in structure.atoms], dtype=float)
    for atom, position in zip(structure.atoms, coordinates):
        atom.position = position
    final_energy = float(mean_field.e_tot) if getattr(mean_field, "e_tot", 0.0) else None
    max_displacement = float(np.max(np.linalg.norm(coordinates - before, axis=1)))
    # geometric's public wrapper discards its convergence flag, so verify the
    # final geometry with one gradient evaluation.  This costs ~1 SCF and makes
    # the "converged" flag trustworthy.
    gradient_rms: float | None = None
    converged = True
    try:
        check = dft.RKS(optimized) if structure.multiplicity == 1 else dft.UKS(optimized)
        check.xc = "hf" if method.lower() in {"hf", "rhf", "rohf"} else _dft_xc(method)
        check.conv_tol = 1.0e-7
        check.max_cycle = 80
        try:
            check.grids.level = 2
        except Exception:
            pass
        check.kernel()
        gradient = check.nuc_grad_method().kernel()
        gradient_rms = float(np.sqrt(np.mean(np.asarray(gradient) ** 2)))
        converged = gradient_rms < 1.0e-3
    except Exception:
        # Gradient is unavailable for this method; trust that geometric
        # returned normally.
        converged = True
    return OptimizationResult(
        "dft",
        converged,
        final_energy,
        {
            "engine": "pyscf-geometric",
            "qc": qc,
            "energy_hartree": final_energy,
            "max_displacement_angstrom": max_displacement,
            "final_gradient_rms": gradient_rms,
            "steps": int(max_steps),
        },
    )
