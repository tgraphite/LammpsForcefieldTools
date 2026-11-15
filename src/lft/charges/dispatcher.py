"""Public charge-method dispatcher."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.structure import Structure
from ..errors import ChargeError
from .density_charges import (
    adch_charges,
    cm5_charges,
    hirshfeld_charges,
    mulliken_charges,
    normalize_populations,
)
from .empirical import assign_am1bcc, assign_mmff94
from .input import apply_input_charges, read_charge_file
from .resp import fit_resp, fit_resp_proxy, mix_resp2

BASE_CHARGE_METHODS = ("input", "mmff94", "cm5", "raw_cm5", "resp", "resp2", "mk", "adch", "am1-bcc")
#: "dft-" prefix means the geometry is optimised at the DFT level before the
#: charge model is evaluated.  The charge model itself is the same.
DFT_CHARGE_METHODS = tuple(f"dft-{name}" for name in ("resp", "resp2", "mk", "cm5", "raw_cm5", "adch"))
CHARGE_METHODS = BASE_CHARGE_METHODS + DFT_CHARGE_METHODS
CHARGE_SUM_TOLERANCE = 1.0e-5
REFERENCE_CHARGE_TOLERANCE = 1.0e-8


def charge_base(method: str) -> str:
    """Return the charge model behind a dft- prefixed method name."""

    return method[4:] if method.startswith("dft-") else method


def is_dft_charge(method: str) -> bool:
    return method.startswith("dft-")


def assign_charges(
    structure: Structure,
    method: str,
    *,
    net_charge: float,
    charge_file: str | Path | None = None,
    resp2_delta: float = 0.5,
    workdir: str | Path = ".",
    qc_result: Any | None = None,
) -> dict[str, Any]:
    method = charge_base(method)
    if method not in BASE_CHARGE_METHODS:
        raise ChargeError(f"unsupported charge method {method!r}; allowed: {', '.join(CHARGE_METHODS)}")
    structure.declared_net_charge = float(net_charge)
    provided = read_charge_file(charge_file) if charge_file else None
    metadata: dict[str, Any] = {"requested_method": method, "requested_net_charge": float(net_charge), "charge_sum_tolerance": CHARGE_SUM_TOLERANCE}
    if method == "input":
        apply_input_charges(structure, provided)
    elif method == "mmff94":
        if provided is not None:
            raise ChargeError("mmff94 does not accept a precomputed charge artifact; use --charge input for explicit charges")
        metadata.update(assign_mmff94(structure, source_path=structure.source_path, workdir=workdir))
    elif method == "am1-bcc":
        if provided is not None:
            raise ChargeError("am1-bcc does not accept a precomputed charge artifact; use --charge input for explicit charges")
        metadata.update(assign_am1bcc(structure, source_path=structure.source_path, net_charge=net_charge, workdir=workdir))
    else:
        wavefunction = getattr(qc_result, "wavefunction", None) if qc_result is not None else None
        if wavefunction is None:
            _assign_from_artifact(structure, method, provided, net_charge, metadata)
        elif method in {"raw_cm5", "cm5", "adch"}:
            _assign_from_wavefunction(structure, method, wavefunction, net_charge, metadata)
        elif method == "mk":
            charges, fit_metadata = fit_resp(structure, wavefunction, net_charge, stage1_restraint=0.0, stage2_restraint=0.0)
            _set(structure, charges, "mk")
            metadata["effective_method"] = "mk"
            metadata["fit"] = fit_metadata
        else:
            gas, gas_meta = fit_resp(structure, wavefunction, net_charge)
            metadata["gas_fit"] = gas_meta
            if method == "resp":
                _set(structure, gas, "resp")
            else:
                aqueous_wavefunction = getattr(qc_result, "aqueous_wavefunction", None)
                if aqueous_wavefunction is not None:
                    aqueous, aqueous_meta = fit_resp(structure, aqueous_wavefunction, net_charge)
                else:
                    aqueous, aqueous_meta = fit_resp(structure, wavefunction, net_charge)
                metadata["aqueous_fit"] = aqueous_meta
                mixed = mix_resp2(gas, aqueous, resp2_delta)
                _set(structure, mixed, "resp2")
                metadata["resp2_delta"] = resp2_delta
                metadata["effective_method"] = "resp2"
    _validate_sum(structure, net_charge, method)
    metadata["computed_net_charge"] = structure.net_charge
    metadata["neutral"] = abs(net_charge) <= CHARGE_SUM_TOLERANCE and abs(structure.net_charge) <= CHARGE_SUM_TOLERANCE
    metadata["atom_charges"] = [float(atom.charge) for atom in structure.atoms]
    metadata.setdefault("raw_atom_charges", list(metadata["atom_charges"]))
    return metadata


def _assign_from_wavefunction(structure: Structure, method: str, wavefunction: Any, net_charge: float, metadata: dict[str, Any]) -> None:
    hirshfeld, dipoles, hirshfeld_meta = hirshfeld_charges(wavefunction)
    metadata["hirshfeld"] = hirshfeld_meta
    metadata["raw_hirshfeld"] = [float(value) for value in hirshfeld]
    if method in {"raw_cm5", "cm5"}:
        corrected, cm5_meta = cm5_charges(hirshfeld, wavefunction)
        metadata["cm5"] = cm5_meta
        effective = "raw_cm5"
        charges = normalize_populations(corrected, wavefunction)
        if method == "cm5":
            if abs(net_charge) <= CHARGE_SUM_TOLERANCE:
                charges = 1.2 * charges
                effective = "cm5"
            else:
                warning = {
                    "code": "CHARGE_CM5_CHARGED_FALLBACK",
                    "message": "charged system requested cm5; using raw_cm5 without 1.2 scaling",
                    "requested_method": "cm5",
                    "effective_method": "raw_cm5",
                    "fallback_reason": "charged_system",
                    "requested_net_charge": float(net_charge),
                }
                structure.add_warning(str(warning["code"]), str(warning["message"]), requested_method="cm5", effective_method="raw_cm5", fallback_reason="charged_system", requested_net_charge=float(net_charge))
                warnings.warn("CHARGE_CM5_CHARGED_FALLBACK: cm5 -> raw_cm5 for charged system", RuntimeWarning, stacklevel=2)
        _set(structure, charges, effective)
        metadata["effective_method"] = effective
        metadata["raw_atom_charges"] = [float(value) for value in corrected]
    else:  # adch
        corrected, adch_meta = adch_charges(hirshfeld, dipoles, wavefunction)
        metadata["adch"] = adch_meta
        charges = normalize_populations(corrected, wavefunction)
        _set(structure, charges, "adch")
        metadata["effective_method"] = "adch"
        metadata["raw_atom_charges"] = [float(value) for value in corrected]


def _assign_from_artifact(structure: Structure, method: str, provided: Any, net_charge: float, metadata: dict[str, Any]) -> None:
    """Legacy path: a precomputed population/charge artifact without a wavefunction."""
    raw = np.asarray(provided, dtype=float) if provided is not None else None
    if raw is None:
        raise ChargeError(f"{method} requires a QC wavefunction (run without --no-qc) or --charge <file>")
    if raw.size != structure.natoms:
        raise ChargeError(f"charge artifact has {raw.size} values for {structure.natoms} atoms")
    if method in {"raw_cm5", "cm5", "adch"}:
        charges = raw.copy()
        effective = method if method == "raw_cm5" else "raw_cm5"
        if method == "cm5":
            if abs(net_charge) <= CHARGE_SUM_TOLERANCE:
                charges = 1.2 * charges
                effective = "cm5"
            else:
                warning = {
                    "code": "CHARGE_CM5_CHARGED_FALLBACK",
                    "message": "charged system requested cm5; using raw_cm5 without 1.2 scaling",
                    "requested_method": "cm5",
                    "effective_method": "raw_cm5",
                    "fallback_reason": "charged_system",
                    "requested_net_charge": float(net_charge),
                }
                structure.add_warning(str(warning["code"]), str(warning["message"]), requested_method="cm5", effective_method="raw_cm5", fallback_reason="charged_system", requested_net_charge=float(net_charge))
                warnings.warn("CHARGE_CM5_CHARGED_FALLBACK: cm5 -> raw_cm5 for charged system", RuntimeWarning, stacklevel=2)
        if method == "adch":
            raise ChargeError("adch requires a QC wavefunction; --charge <file> cannot provide atomic dipole data")
        _set(structure, charges, effective)
        metadata["effective_method"] = effective
        metadata["artifact_backend"] = "charge-file"
        metadata["raw_atom_charges"] = [float(value) for value in raw]
    elif method == "mk":
        charges, fit_metadata = fit_resp_proxy(structure, raw, net_charge, stage1_restraint=0.0, stage2_restraint=0.0)
        _set(structure, charges, "mk")
        metadata["effective_method"] = "mk"
        metadata["fit"] = fit_metadata
    else:
        gas, gas_meta = fit_resp_proxy(structure, raw, net_charge)
        metadata["gas_fit"] = gas_meta
        if method == "resp":
            _set(structure, gas, "resp")
        else:
            aqueous, aqueous_meta = fit_resp_proxy(structure, raw, net_charge)
            metadata["aqueous_fit"] = aqueous_meta
            mixed = mix_resp2(gas, aqueous, 0.5)
            _set(structure, mixed, "resp2")
            metadata["resp2_delta"] = 0.5
            metadata["effective_method"] = "resp2"


def _set(structure: Structure, values: np.ndarray, source: str) -> None:
    if len(values) != structure.natoms:
        raise ChargeError("charge model returned the wrong number of charges")
    for atom, value in zip(structure.atoms, values):
        atom.charge = float(value)
        atom.charge_source = source


def _validate_sum(structure: Structure, net_charge: float, method: str) -> None:
    error = abs(structure.net_charge - net_charge)
    if error > CHARGE_SUM_TOLERANCE:
        raise ChargeError(f"{method} charges sum to {structure.net_charge:.12g} e, requested {net_charge:.12g} e; error {error:.3g} e exceeds {CHARGE_SUM_TOLERANCE:.1e} e")