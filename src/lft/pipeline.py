"""End-to-end lft task graph.

By default the pipeline writes only runnable artifacts into the output
directory.  A hidden ``<name>.lft/`` disclosure directory is created on demand
for information that a reader genuinely needs (periodic image lifts, warnings,
or the ``--debug`` diagnostic bundle).  Audit data is still collected in
memory so the CLI can report warnings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import Audit, sha256_file
from .backends import write_gromacs, write_lammps
from .charges import CHARGE_METHODS, assign_charges
from .config import AUTO_AM1BCC_ATOM_THRESHOLD, PipelineConfig
from .domain.parameters import ParameterSet
from .domain.structure import Structure
from .errors import ChargeError, InputError, ParameterError
from .forcefields.matching import assign_force_field
from .io.loaders import load_structure
from .io.serializers import dump_json
from .parameterize import parameterize
from .qc import load_hessian, run_qc
from .runners.gmx import grompp
from .runners.lammps import run_lammps
from .topology import compute_winding, enumerate_terms, infer_bonds
from .topology.infer_bonds import validate_image_shifts

_QC_METHODS = {"raw_cm5", "cm5", "adch", "mk", "resp", "resp2"}


@dataclass(slots=True)
class RunResult:
    run_id: str
    output_dir: Path
    structure: Structure
    parameters: ParameterSet
    audit: Audit
    artifacts: dict[str, Path]


def run_pipeline(config: PipelineConfig) -> RunResult:
    config.validate()
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = sha256_file(config.structure)
    audit = Audit.create(config.to_dict(), input_hash)
    audit.task("load", input=str(config.structure), input_sha256=input_hash)
    for label, artifact_path in (("hessian_file", config.hessian_file), ("manual_file", config.manual_file)):
        if artifact_path is not None and artifact_path.exists():
            audit.add_artifact(f"input_{label}", artifact_path, kind="input_artifact")

    structure = load_structure(config.structure)
    structure.declared_net_charge = float(config.net_charge)
    structure.multiplicity = config.multiplicity
    if not structure.bonds:
        infer_bonds(structure)
        audit.task("infer_bonds", count=len(structure.bonds), source="distance_radii_valence")
    else:
        infer_bonds(structure)
        audit.task("normalize_bonds", count=len(structure.bonds), source="explicit")
    validate_image_shifts(structure)
    enumerate_terms(structure)
    audit.task("enumerate_terms", angles=len(structure.angles), dihedrals=len(structure.dihedrals), impropers=len(structure.impropers), pairs=len(structure.pairs))

    workdir = output_dir / f"{structure.name}.lft" / "work"
    manual_types, manual_atom_types, manual_parameters = _load_manual(config.manual_file)
    params = assign_force_field(
        structure,
        config.force_field,
        manual_types=manual_types,
        manual_atom_types=manual_atom_types,
        source_path=str(config.structure),
        workdir=str(workdir),
    )
    params.metadata["manual_parameters"] = manual_parameters
    audit.task("type_atoms", force_field=config.force_field, atom_types=params.metadata.get("typing", {}))
    typing_backend = params.metadata.get("typing_backend")
    if isinstance(typing_backend, dict):
        audit.task("atomtype", **typing_backend)
        for name in ("input", "output", "definition"):
            value = typing_backend.get(name)
            if isinstance(value, str) and Path(value).is_file():
                audit.add_artifact(f"atomtype_{name}", Path(value), kind="atomtype")

    charge_method, charge_file = _resolve_charge(config, structure)
    if charge_method != config.charge:
        audit.task("charge_selection", requested=config.charge, method=charge_method, file=str(charge_file) if charge_file else None)

    qc_result = None
    needs_qc = charge_method in _QC_METHODS and charge_file is None
    if config.parameter_mode == "hessian" and config.hessian_file is None:
        needs_qc = True
    if needs_qc:
        if config.no_qc:
            raise ParameterError(f"{charge_method} or hessian mode requires QC; --no-qc is incompatible without an artifact")
        workdir.mkdir(parents=True, exist_ok=True)
        qc_result = run_qc(structure, qc=config.qc, backend=config.qc_backend, need_hessian=config.parameter_mode == "hessian", workdir=str(workdir))
        audit.task("qc", backend=qc_result.backend, method=qc_result.method, basis=qc_result.basis, energy_hartree=qc_result.energy_hartree, metadata=qc_result.metadata)
        for label in ("input", "output"):
            value = qc_result.metadata.get(label)
            if isinstance(value, str) and Path(value).is_file():
                audit.add_artifact(f"qc_{label}", Path(value), kind="qc")
        if charge_method == "resp2" and charge_file is None:
            aqueous_qc = run_qc(structure, qc=config.qc, backend=config.qc_backend, need_hessian=False, aqueous=True, workdir=str(workdir))
            qc_result.aqueous_population_charges = aqueous_qc.population_charges
            qc_result.aqueous_wavefunction = aqueous_qc.wavefunction
            audit.task("qc_aqueous", backend=aqueous_qc.backend, method=aqueous_qc.method, basis=aqueous_qc.basis, energy_hartree=aqueous_qc.energy_hartree, metadata=aqueous_qc.metadata)
    hessian = None
    hessian_metadata: dict[str, Any] | None = None
    if config.hessian_file is not None:
        hessian, hessian_metadata = load_hessian(config.hessian_file, structure.natoms)
        audit.task("load_hessian", **hessian_metadata)
    elif qc_result is not None and qc_result.hessian is not None:
        hessian = qc_result.hessian
        hessian_metadata = {"source": "qc", "shape": list(hessian.shape), "units": "hartree/bohr^2"}
    charge_metadata = assign_charges(
        structure,
        charge_method,
        net_charge=config.net_charge,
        charge_file=charge_file,
        resp2_delta=config.resp2_delta,
        workdir=workdir,
        qc_result=qc_result,
    )
    audit.task("charges", **charge_metadata)
    params.metadata["charge"] = charge_metadata
    for label in ("input", "output"):
        value = charge_metadata.get(label)
        if isinstance(value, str) and Path(value).is_file():
            audit.add_artifact(f"charges_{label}", Path(value), kind="charges")
    for warning in structure.warnings + params.warnings:
        audit.add_warning(warning)
    params = parameterize(structure, params, mode=config.parameter_mode, hessian=hessian, hessian_method=config.hessian_method)
    audit.task("parameterize", mode=config.parameter_mode, method=config.hessian_method, parameter_counts=params.metadata.get("parameter_counts", {}), hessian=hessian_metadata)
    structure.validate()
    if config.output == "gromacs" and structure.cell is None:
        structure.add_warning("GROMACS_COMPATIBILITY_BOX", "non-periodic source received a finite compatibility box because modern GROMACS requires Verlet with a box")
    if config.output == "gromacs" and structure.cell is not None and structure.cell.pbc != (True, True, True):
        structure.add_warning("GROMACS_PARTIAL_PBC_COMPATIBILITY", "GROMACS writer uses a full compatibility box for a partial-PBC source; source pbc mask remains in the IR", pbc=list(structure.cell.pbc))
    winding_rank = compute_winding(structure)["rank"]
    if config.output == "lammps" and winding_rank:
        structure.add_warning("LAMMPS_PERIODIC_LIFT_SIDECAR", "periodic graph winding cannot be represented by one globally consistent atom-image assignment; image_map is authoritative for bonded lifts", winding_rank=winding_rank)
    for warning in structure.warnings + params.warnings:
        audit.add_warning(warning)

    _check_overwrite(config, output_dir)
    disclose_dir = output_dir / f"{structure.name}.lft"
    emit_image_map = bool(winding_rank) or config.debug
    image_dir = disclose_dir if emit_image_map else None
    if config.output == "gromacs":
        artifacts = write_gromacs(structure, params, output_dir, disclose_dir=image_dir, emit_image_map=emit_image_map)
    else:
        artifacts = write_lammps(structure, params, output_dir, emit_lt=config.emit_lt, disclose_dir=image_dir, emit_image_map=emit_image_map)
    audit.task("write_backend", backend=config.output, artifacts=sorted(artifacts))
    for name, path in artifacts.items():
        audit.add_artifact(name, path, kind=f"{config.output}:{name}")
    if config.engine_smoke:
        smoke = _run_engine_smoke(config, artifacts, output_dir, audit)
        artifacts.update(smoke)

    artifacts.update(_write_disclosure(structure, params, audit, output_dir, disclose_dir, config))
    return RunResult(audit.run_id, output_dir, structure, params, audit, artifacts)


def _resolve_charge(config: PipelineConfig, structure: Structure) -> tuple[str, Path | None]:
    """Decide the charge model from the CLI spec, structure charges, and size."""

    spec = config.charge
    if spec is not None and spec.lower() in CHARGE_METHODS:
        spec = spec.lower()
    if spec is not None and spec not in CHARGE_METHODS:
        path = Path(spec).expanduser()
        if path.is_file():
            return "input", path
        raise InputError(
            f"--charge value {spec!r} is neither a known method nor an existing charge file",
            hint="pass a method such as resp, am1-bcc, or input, or a path to a text file with one charge per atom",
        )
    periodic = structure.cell is not None
    if spec is None:
        if _has_structure_charges(structure):
            return "input", None
        if periodic:
            raise ChargeError(
                "periodic structures require explicit charges",
                hint="pass --charge charges.txt (one charge per atom) or a structure that already carries charges",
            )
        if structure.natoms > AUTO_AM1BCC_ATOM_THRESHOLD:
            structure.add_warning(
                "CHARGE_AUTO_LARGE_MOLECULE",
                f"{structure.natoms} atoms exceed {AUTO_AM1BCC_ATOM_THRESHOLD}; using AM1-BCC instead of RESP (pass --charge to override)",
                atom_count=structure.natoms,
                effective_method="am1-bcc",
            )
            return "am1-bcc", None
        return "resp", None
    if periodic and spec != "input":
        raise ChargeError("periodic structures require --charge input with explicit charges; ESP/population fitting is not applied to a unit cell")
    return spec, None


def _has_structure_charges(structure: Structure) -> bool:
    return any(atom.charge_source == "input" or abs(atom.charge) > 1.0e-12 for atom in structure.atoms)


def _write_disclosure(
    structure: Structure,
    params: ParameterSet,
    audit: Audit,
    output_dir: Path,
    disclose_dir: Path,
    config: PipelineConfig,
) -> dict[str, Path]:
    """Write only the hidden sidecars a reader actually needs."""

    artifacts: dict[str, Path] = {}
    if audit.warnings:
        disclose_dir.mkdir(parents=True, exist_ok=True)
        notes = disclose_dir / "notes.txt"
        notes.write_text(_notes_text(structure, params, audit))
        artifacts["notes"] = notes
    if config.debug:
        disclose_dir.mkdir(parents=True, exist_ok=True)
        debug_path = disclose_dir / "debug.txt"
        debug_path.write_text(_debug_text(structure, params, audit, config))
        artifacts["debug"] = debug_path
        ir_path = disclose_dir / "normalized_ir.json"
        dump_json({"structure": structure.to_dict(), "parameters": params.to_dict()}, ir_path)
        artifacts["normalized_ir"] = ir_path
    return artifacts


def _notes_text(structure: Structure, params: ParameterSet, audit: Audit) -> str:
    lines = [
        f"lft run {audit.run_id}",
        f"structure: {structure.name}",
        f"force field: {params.force_field}",
        "",
        "warnings:",
    ]
    lines.extend(f"  [{warning.get('code', 'LFT_WARNING')}] {warning.get('message', '')}" for warning in audit.warnings)
    lines.append("")
    return "\n".join(lines)


def _debug_text(structure: Structure, params: ParameterSet, audit: Audit, config: PipelineConfig) -> str:
    lines = [
        f"lft debug bundle (run_id={audit.run_id})",
        f"created_utc: {audit.metadata.get('created_utc')}",
        f"structure: {structure.name} ({config.structure})",
        f"input_sha256: {audit.artifacts.get('load', {}).get('sha256', sha256_file(config.structure))}",
        f"output: {config.output}",
        f"force_field: {params.force_field}",
        f"parameter_mode: {params.parameter_mode}",
        f"charge: {config.charge or 'auto'}",
        f"net_charge: {structure.declared_net_charge}",
        f"atoms: {structure.natoms}",
        f"bonds/angles/dihedrals/impropers: {len(structure.bonds)}/{len(structure.angles)}/{len(structure.dihedrals)}/{len(structure.impropers)}",
        "",
        "stages:",
    ]
    for task in audit.tasks:
        name = task.get("name", "?")
        details = ", ".join(f"{key}={value}" for key, value in task.items() if key != "name")
        lines.append(f"  {name}: {details}")
    lines.extend(["", "warnings:"])
    lines.extend(f"  [{warning.get('code', 'LFT_WARNING')}] {warning.get('message', '')}" for warning in audit.warnings)
    lines.extend(["", "artifacts:"])
    for name, record in sorted(audit.artifacts.items()):
        lines.append(f"  {name}: {record.get('path')} sha256={record.get('sha256')} bytes={record.get('bytes')}")
    lines.append("")
    return "\n".join(lines)


def _check_overwrite(config: PipelineConfig, output_dir: Path) -> None:
    if config.force:
        return
    stem = Path(config.structure).stem
    if config.output == "gromacs":
        candidates = [output_dir / f"{stem}{suffix}" for suffix in (".itp", ".top", ".gro", ".mdp")]
    else:
        candidates = [output_dir / f"{stem}{suffix}" for suffix in (".data", ".in")]
        if config.emit_lt:
            candidates.append(output_dir / f"{stem}.lt")
            candidates.append(output_dir / "system.lt")
    existing = [path.name for path in candidates if path.exists()]
    if existing:
        raise InputError(f"output already exists: {', '.join(existing)}", hint="pass --force to overwrite, or use --outdir")


def _run_engine_smoke(config: PipelineConfig, artifacts: dict[str, Path], output_dir: Path, audit: Audit) -> dict[str, Path]:
    result_paths: dict[str, Path] = {}
    if config.output == "gromacs":
        result = grompp(workdir=output_dir, mdp=artifacts["mdp"].name, gro=artifacts["gro"].name, top=artifacts["top"].name, output="lft_smoke.tpr")
        stdout = output_dir / "grompp.stdout"
        stderr = output_dir / "grompp.stderr"
        stdout.write_text(result.stdout)
        stderr.write_text(result.stderr)
        result_paths.update({"grompp_stdout": stdout, "grompp_stderr": stderr, "grompp_tpr": output_dir / "lft_smoke.tpr"})
        audit.task("engine_smoke", engine="gromacs", command=result.command, returncode=result.returncode)
    else:
        project_root = Path(__file__).resolve().parents[2]
        result = run_lammps(workdir=output_dir, input_file=artifacts["input"].name, project_root=project_root)
        stdout = output_dir / "lammps.stdout"
        stderr = output_dir / "lammps.stderr"
        stdout.write_text(result.stdout)
        stderr.write_text(result.stderr)
        result_paths.update({"lammps_stdout": stdout, "lammps_stderr": stderr})
        audit.task("engine_smoke", engine="lammps", command=result.command, returncode=result.returncode)
    for name, path in result_paths.items():
        if path.exists():
            audit.add_artifact(name, path, kind="engine_smoke")
    return result_paths


def _output_dir(config: PipelineConfig) -> Path:
    if config.outdir is not None:
        return Path(config.outdir).resolve()
    return Path.cwd()


def _load_manual(path: Path | None) -> tuple[dict[int, str], dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    if path is None:
        return {}, {}, {}
    if not path.exists():
        raise InputError(f"manual force-field file does not exist: {path}")
    raw = json.loads(path.read_text())
    atom_mapping = raw.get("atoms", raw.get("atom_types_by_index", {}))
    if not isinstance(atom_mapping, dict):
        raise InputError("manual force-field file 'atoms' must be an object mapping atom index to type")
    types = {int(index): str(value) for index, value in atom_mapping.items()}
    atom_type_values = raw.get("types", raw.get("atom_types", {}))
    if not isinstance(atom_type_values, dict):
        raise InputError("manual force-field file 'types' must be an object")
    overrides: dict[str, dict[str, float]] = {}
    for name, value in atom_type_values.items():
        if not isinstance(value, dict):
            raise InputError(f"manual atom type {name!r} must be an object")
        overrides[str(name)] = {key: float(value[key]) for key in ("mass", "sigma", "epsilon") if key in value}
    bonded_keys = ("bonds", "angles", "dihedrals", "impropers")
    bonded: dict[str, dict[str, Any]] = {}
    for key in bonded_keys:
        value = raw.get(key, {})
        if not isinstance(value, dict):
            raise InputError(f"manual force-field file {key!r} must be an object")
        bonded[key] = {str(term): dict(values) for term, values in value.items() if isinstance(values, dict)}
    return types, overrides, bonded
