"""Command-line entry point for the complete lft pipeline.

The happy path is deliberately small: point lft at a structure and it writes a
runnable force-field/topology.  Small molecules default to moltemplate output;
pass --lammpsdata for a classic LAMMPS data/input pair.  Advanced flags are
accepted but suppressed from --help.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import (
    AUTO_AM1BCC_ATOM_THRESHOLD,
    AUTO_MMFF94_ATOM_THRESHOLD,
    PipelineConfig,
    known_charge_methods,
)
from .errors import LFTError
from .pipeline import run_pipeline
from .settings import apply_environment, load_settings

_VISIBLE_FF = ("gaff", "gaff+uff", "uff", "manual")

_IMPORTANT_WARNINGS = {
    "CHARGE_AUTO_LARGE_MOLECULE",
    "FF_UFF_FALLBACK",
    "CHARGE_CM5_CHARGED_FALLBACK",
    "LAMMPS_PERIODIC_LIFT_SIDECAR",
    "PREOPT_NOT_CONVERGED",
    "PREOPT_SKIPPED_SIZE",
    "PARAMETER_TORSION_OMITTED",
    "PARAMETER_UFF_TORSION_OMITTED",
    "PARAMETER_IMPROPER_OMITTED",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lft",
        description="Generate ready-to-run LAMMPS or GROMACS force field and topology files.",
        epilog=(
            "examples:\n"
            "  lft -f molecule.mol2                 # molecule.lt + system.lt (moltemplate)\n"
            "  lft -f molecule.mol2 --lammpsdata    # molecule.data + molecule.in\n"
            "  lft -f molecule.mol2 -o gromacs      # .itp/.top/.gro/.mdp\n"
            "  lft -f crystal.pdb --charge charges.txt --ff uff --lammpsdata\n"
            f"\ncharges default to RESP (<= {AUTO_AM1BCC_ATOM_THRESHOLD} atoms), AM1-BCC "
            f"(<= {AUTO_MMFF94_ATOM_THRESHOLD}) or MMFF94 above that; dft- prefixed "
            "methods optimise the geometry with DFT first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--structure", required=False, help="input mol2/pdb/pqr/gro/xyz/cif/JSON structure")
    parser.add_argument("-o", "--output", choices=("lammps", "gromacs"), default=None, help="output backend (default: lammps)")
    parser.add_argument("--ff", choices=_VISIBLE_FF, default=None, help="force-field typing policy (default: gaff+uff)")
    parser.add_argument("--charge", default=None, metavar="METHOD|FILE", help=f"charge model or charge file; methods: {', '.join(known_charge_methods())}")
    parser.add_argument("--net-charge", dest="net_charge", type=int, default=0, help="declared molecular net charge")
    parser.add_argument("--mult", "--multiplicity", dest="multiplicity", type=int, default=1, help="spin multiplicity")
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: current directory)")
    parser.add_argument("--lammpsdata", action="store_true", help="write <name>.data + <name>.in instead of moltemplate files")
    parser.add_argument("--preopt", choices=("auto", "none", "uff", "mmff94", "dft"), default="auto", help="geometry pre-optimisation engine (default: auto, UFF for computed charges)")
    parser.add_argument("--current-geometry", action="store_true", help="pin bond/angle equilibrium values to the input geometry (conformer-specific; an INFO message is emitted)")
    parser.add_argument("--allow-missing-torsions", action="store_true", help="omit dihedrals that have no GAFF parameter instead of failing")
    parser.add_argument("--generic-improper", choices=("omit", "amber", "k10"), default="omit", help="fallback for impropers without GAFF parameters (default: omit)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    parser.add_argument("--debug", action="store_true", help="write a plain-text diagnostic bundle into <name>.lft/")
    parser.add_argument("--check-update", action="store_true", help="check PyPI and configured mirrors for a newer release and exit")
    parser.add_argument("--no-update-check", action="store_true", help="skip the automatic update notice for this run")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    advanced = parser.add_argument_group("advanced (hidden defaults are usually correct)")
    advanced.add_argument("--parameter-mode", choices=("classic", "hessian"), default="classic", help=argparse.SUPPRESS)
    advanced.add_argument("--qc", default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--qc-backend", choices=("pyscf", "auto"), default="pyscf", help=argparse.SUPPRESS)
    advanced.add_argument("--hessian-file", type=Path, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--hessian-method", choices=("seminario", "mseminario", "m2seminario", "drih"), default="seminario", help=argparse.SUPPRESS)
    advanced.add_argument("--manual-file", type=Path, default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--no-qc", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("--engine-smoke", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("--moltemplate", dest="emit_lt", action="store_true", default=None, help=argparse.SUPPRESS)
    advanced.add_argument("--no-lt", dest="no_lt", action="store_true", help=argparse.SUPPRESS)
    advanced.add_argument("--preopt-max-atoms", type=int, default=2000, help=argparse.SUPPRESS)
    advanced.add_argument("--dft-opt-max-atoms", type=int, default=50, help=argparse.SUPPRESS)
    advanced.add_argument("--dft-opt-max-steps", type=int, default=15, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.check_update:
        from .update import force_check

        return force_check(settings)
    if not args.structure:
        parser.error("-f/--structure is required unless --check-update is used")
    apply_environment(settings)
    if not args.no_update_check:
        from .update import maybe_notify

        maybe_notify(settings)

    output = args.output or settings.get("output") or "lammps"
    force_field = args.ff or settings.get("ff") or "gaff+uff"
    charge = args.charge if args.charge is not None else settings.get("charge") or None
    qc = args.qc or settings.get("qc") or "b3lyp-d3bj/def2-svp"
    emit_lt = not args.lammpsdata
    if args.emit_lt:
        emit_lt = True
    if args.no_lt:
        emit_lt = False
    if not emit_lt and not args.lammpsdata and output == "lammps":
        print("error: [INPUT_ERROR] --no-lt requires --lammpsdata for LAMMPS output", file=sys.stderr)
        return 2

    try:
        config = PipelineConfig(
            structure=args.structure,
            output=output,
            parameter_mode=args.parameter_mode,
            force_field=force_field,
            charge=charge,
            net_charge=args.net_charge,
            multiplicity=args.multiplicity,
            qc=qc,
            qc_backend=args.qc_backend,
            outdir=args.outdir,
            hessian_file=args.hessian_file,
            hessian_method=args.hessian_method,
            manual_file=args.manual_file,
            no_qc=args.no_qc,
            engine_smoke=args.engine_smoke,
            emit_lt=emit_lt,
            lammps_data=args.lammpsdata,
            preopt=args.preopt,
            preopt_max_atoms=args.preopt_max_atoms,
            dft_opt_max_atoms=args.dft_opt_max_atoms,
            dft_opt_max_steps=args.dft_opt_max_steps,
            current_geometry=args.current_geometry,
            missing_torsion="allow" if args.allow_missing_torsions else "error",
            missing_improper=args.generic_improper,
            debug=args.debug,
            force=args.force,
        )
        result = run_pipeline(config)
    except LFTError as exc:
        print(f"error: {exc.formatted()}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: [PIPELINE_ERROR] {exc}", file=sys.stderr)
        return 2
    _report_warnings(result)
    internal = {"image_map", "notes", "debug", "normalized_ir"}
    primary = [path.name for key, path in sorted(result.artifacts.items()) if key not in internal and path.exists()]
    print(f"lft: wrote {', '.join(primary)} in {result.output_dir}")
    return 0


def _report_warnings(result: object) -> None:
    audit = getattr(result, "audit", None)
    warnings = list(getattr(audit, "warnings", [])) if audit is not None else []
    if not warnings:
        return
    info = [warning for warning in warnings if str(warning.get("code", "")).startswith("INFO_")]
    for warning in info:
        print(f"info[{warning.get('code')}]: {warning.get('message', '')}", file=sys.stderr)
    for warning in warnings:
        if warning.get("code") in _IMPORTANT_WARNINGS:
            print(f"warning[{warning.get('code')}]: {warning.get('message', '')}", file=sys.stderr)
    quiet = [
        warning
        for warning in warnings
        if warning.get("code") not in _IMPORTANT_WARNINGS and not str(warning.get("code", "")).startswith("INFO_")
    ]
    if quiet:
        codes = sorted({str(warning.get("code", "LFT_WARNING")) for warning in quiet})
        artifacts = getattr(result, "artifacts", {})
        notes = artifacts.get("notes") if hasattr(artifacts, "get") else None
        pointer = f"; details in {notes}" if notes else ""
        print(f"note: {len(quiet)} informational item(s) [{', '.join(codes)}]{pointer}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
