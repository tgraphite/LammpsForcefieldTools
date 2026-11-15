"""Command-line entry point for the complete lft pipeline.

The happy path is deliberately small: point ``lft`` at a structure and it
writes runnable force-field/topology files into the current directory.  The
advanced flags are still accepted (and are used by the validation scripts) but
are suppressed from ``--help`` so new users are not asked to make decisions
they do not need to make.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import AUTO_AM1BCC_ATOM_THRESHOLD, PipelineConfig, known_charge_methods
from .errors import LFTError
from .pipeline import run_pipeline
from .settings import apply_environment, load_settings

_VISIBLE_FF = ("gaff", "gaff+uff", "uff", "manual")

# Warnings that a new user should still see on the console.  Everything else
# is aggregated into a single pointer so repeated/trivial notes do not bury the
# one-line result.
_IMPORTANT_WARNINGS = {
    "CHARGE_AUTO_LARGE_MOLECULE",
    "FF_UFF_FALLBACK",
    "CHARGE_CM5_CHARGED_FALLBACK",
    "LAMMPS_PERIODIC_LIFT_SIDECAR",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lft",
        description="Generate ready-to-run LAMMPS or GROMACS force field and topology files.",
        epilog=(
            "examples:\n"
            "  lft -f molecule.mol2\n"
            "  lft -f molecule.xyz --output gromacs\n"
            "  lft -f crystal.pdb --charge charges.txt --ff uff\n"
            f"\ncharges default to RESP (<= {AUTO_AM1BCC_ATOM_THRESHOLD} atoms) or AM1-BCC above that; "
            "charges already present in the structure are used automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--structure", required=True, help="input mol2/pdb/pqr/gro/xyz/cif/JSON structure")
    parser.add_argument("-o", "--output", choices=("lammps", "gromacs"), default=None, help="output backend (default: lammps)")
    parser.add_argument("--ff", choices=_VISIBLE_FF, default=None, help="force-field typing policy (default: gaff+uff)")
    parser.add_argument("--charge", default=None, metavar="METHOD|FILE", help=f"charge model or charge file; methods: {', '.join(known_charge_methods())}")
    parser.add_argument("--net-charge", dest="net_charge", type=int, default=0, help="declared molecular net charge")
    parser.add_argument("--mult", "--multiplicity", dest="multiplicity", type=int, default=1, help="spin multiplicity")
    parser.add_argument("--outdir", type=Path, default=None, help="output directory (default: current directory)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    parser.add_argument("--debug", action="store_true", help="write a plain-text diagnostic bundle into <name>.lft/")
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
    advanced.add_argument("--moltemplate", dest="emit_lt", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    apply_environment(settings)

    output = args.output or settings.get("output") or "lammps"
    force_field = args.ff or settings.get("ff") or "gaff+uff"
    charge = args.charge if args.charge is not None else settings.get("charge") or None
    qc = args.qc or settings.get("qc") or "b3lyp-d3bj/def2-svp"

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
            emit_lt=args.emit_lt,
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
    for warning in warnings:
        if warning.get("code") in _IMPORTANT_WARNINGS:
            print(f"warning[{warning.get('code')}]: {warning.get('message', '')}", file=sys.stderr)
    quiet = [warning for warning in warnings if warning.get("code") not in _IMPORTANT_WARNINGS]
    if quiet:
        codes = sorted({str(warning.get("code", "LFT_WARNING")) for warning in quiet})
        artifacts = getattr(result, "artifacts", {})
        notes = artifacts.get("notes") if hasattr(artifacts, "get") else None
        pointer = f"; details in {notes}" if notes else ""
        print(f"note: {len(quiet)} informational item(s) [{', '.join(codes)}]{pointer}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
