#!/usr/bin/env python3
"""Validate GAFF against AmberTools26; an oracle is REQUIRED for acceptance.

Default: frozen historical and boundary corpus. --discover adds repository
examples (bondless inputs are reported as skips). --only accepts explicit paths.
Each invocation and each oracle case uses a fresh directory under --outdir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lft.errors import ExternalToolError, LFTError
from lft.forcefields.gaff_typer import assign_gaff_types
from lft.io.loaders import load_structure
from lft.validation.gaff_oracle import (
    PROJECT_ROOT,
    OracleConfig,
    run_oracle,
)


def find_corpus(root: Path) -> list[Path]:
    return sorted(
        p
        for folder in ("references/sobtop/gt", "examples")
        for p in (root / folder).rglob("*")
        if p.suffix.lower() in {".mol2", ".pdb", ".pqr", ".gro", ".xyz"}
        and not any(part.startswith("_") for part in p.relative_to(root).parts)
    )


def main(argv: list[str] | None = None) -> int:
    defaults = OracleConfig.from_environment()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", type=Path)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--oracle", type=Path, default=defaults.executable)
    ap.add_argument("--amberhome", type=Path, default=defaults.amberhome)
    ap.add_argument("--def", dest="definition", type=Path)
    ap.add_argument(
        "--outdir", type=Path, default=PROJECT_ROOT / "_scratch/gaff_typer_diff"
    )
    args = ap.parse_args(argv)
    config = replace(
        defaults,
        executable=args.oracle.resolve(),
        amberhome=args.amberhome.resolve(),
        definition=(
            args.definition
            or (
                defaults.definition
                if args.amberhome == defaults.amberhome
                else args.amberhome / "dat/antechamber/ATOMTYPE_GFF.DEF"
            )
        ).resolve(),
    )
    try:
        config.validate()
    except ExternalToolError as exc:
        print(f"FAIL oracle preflight: {exc}", file=sys.stderr)
        return 2
    fixture = PROJECT_ROOT / "tests/fixtures/gaff"
    frozen = (
        json.loads((fixture / "corpus.json").read_text())
        if not args.only and not args.discover
        else {}
    )
    files = [fixture / name for name in sorted(frozen)]
    if args.discover:
        files = find_corpus(PROJECT_ROOT)
    if args.only:
        files = [(PROJECT_ROOT / p).resolve() for p in args.only]
    files = list(dict.fromkeys(files))
    args.outdir.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=args.outdir.resolve()))
    records: list[dict[str, object]] = []
    for path in files:
        record: dict[str, object] = {"path": str(path), "status": "failed"}
        try:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                frozen
                and record["sha256"] != frozen[str(path.relative_to(fixture))]["sha256"]
            ):
                raise ValueError("frozen input checksum mismatch")
            structure = load_structure(path)
            record["natoms"] = structure.natoms
            if not structure.bonds and args.discover and not args.only:
                record.update(
                    status="skipped",
                    reason="no bonds; no topology inference in this validator",
                )
            else:
                ref = run_oracle(structure, config, run)
                mine = assign_gaff_types(structure, definition=config.definition)
                if set(mine) != set(ref):
                    raise ValueError(
                        f"Python atom ids/count mismatch: {len(mine)} vs {len(ref)}"
                    )
                diffs = [
                    {"atom": i, "python": mine[i], "oracle": ref[i]}
                    for i in ref
                    if mine[i] != ref[i]
                ]
                record.update(status="failed" if diffs else "passed", differences=diffs)
        except (LFTError, OSError, ValueError, IndexError, RuntimeError) as exc:
            record["reason"] = str(exc)
        records.append(record)
        print(
            f"{str(record['status']).upper():7s} {path.name}: {record.get('reason', record.get('differences') or 'matched')}"
        )
    counts = Counter(str(r["status"]) for r in records)
    summary = {key: counts[key] for key in ("passed", "failed", "skipped")}
    evidence = {
        "summary": summary,
        "cases": records,
        "oracle": {
            "executable": str(config.executable),
            "sha256": hashlib.sha256(config.executable.read_bytes()).hexdigest(),
            "definition": str(config.definition),
            "definition_sha256": hashlib.sha256(
                config.definition.read_bytes()
            ).hexdigest(),
        },
    }
    (run / "report.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"{summary['passed']} passed; {summary['failed']} failed; {summary['skipped']} skipped; {len(records)} total"
    )
    print(f"Report: {run / 'report.json'}")
    return 1 if counts["failed"] or not counts["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
