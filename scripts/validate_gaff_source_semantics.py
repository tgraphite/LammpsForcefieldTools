#!/usr/bin/env python3
"""Probe unmodified C functions, checking intermediate semantics as well as types."""

from __future__ import annotations

import json
import random
import string
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lft.errors import ExternalToolError
from lft.forcefields import gaff_typer as gt
from lft.io.loaders import load_structure
from lft.runners.external import run_command
from lft.validation.gaff_oracle import PROJECT_ROOT, RESEARCH_ROOT


def main() -> int:
    base = PROJECT_ROOT / "_scratch/gaff_source_semantics"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix="run-", dir=base))
    manifest = json.loads((RESEARCH_ROOT / "build_manifest.json").read_text())
    source = RESEARCH_ROOT / "ambertools26_src/AmberTools/src/antechamber"
    exe = work / "probe"
    command = [
        manifest["compiler"],
        "-O2",
        "-I",
        str(source),
        "-o",
        str(exe),
        str(RESEARCH_ROOT / "semantic_probe.c"),
        str(source / "eprintf.c"),
        "-lm",
    ]
    run_command(command, cwd=work)
    failures = []
    counts: dict[str, int] = {}

    def check(label, expected, actual):
        group = label.split(":")[0]
        counts[group] = counts.get(group, 0) + 1
        if expected != actual:
            failures.append({"case": label, "C": expected, "python": actual})

    def probe(mode, *args, stdin=None):
        result = run_command(
            [str(exe), mode, *map(str, args)], cwd=work, stdin=stdin, timeout=30
        )
        return [
            line[6:].split()
            for line in result.stdout.splitlines()
            if line.startswith("PROBE ")
        ]

    # atomicnum dispatch only depends on the first two bytes; cover letters,
    # digits, punctuation, lone symbols, isotopes and dummy/lone-pair names.
    for first in string.ascii_letters + "1*_":
        for second in string.ascii_letters + "1_*" + "\0":
            name = first + (second if second != "\0" else "")
            try:
                value = probe("name", name)[0]
                expected = [int(value[0]), value[1]]
            except ExternalToolError:
                expected = None
            try:
                actual = list(gt._atomicnum_from_name(name))
            except ValueError:
                actual = None
            check("atomicnum:" + name, expected, actual)
    for btype in (1, 2, 3, 7, 8, 9, 10):
        for word in (
            "any",
            "ANY",
            "Any",
            "aNy",
            "sb",
            "SB",
            "db",
            "DB",
            "tb",
            "TB",
            "ab",
            "AB",
        ):
            for endpoint in (1, 2):
                expected = int(probe("ces", btype, word, endpoint)[0][0])
                check(
                    f"ces:{btype}:{word}:{endpoint}",
                    expected,
                    gt._cesbondcheck([gt._Bond(0, 1, btype)], word, 0, endpoint),
                )

    def graph_input(atoms, bonds):
        return (
            "\n".join(
                [f"{len(atoms)} {len(bonds)}"]
                + [f"{a.atomicnum} {a.ambername}" for a in atoms]
                + [f"{b.bondi} {b.bondj} {b.type}" for b in bonds]
            )
            + "\n"
        )

    for path in sorted((PROJECT_ROOT / "tests/fixtures/gaff").glob("*/*.mol2")):
        atoms, bonds = gt._build_graph(load_structure(path))
        expected = [
            [int(v) for v in row]
            for row in probe("ring", stdin=graph_input(atoms, bonds))
        ]
        arom = [gt._Arom() for _ in atoms]
        gt._ringdetect(atoms, bonds, arom, bcc_ring=False)
        actual = [
            [
                ar.nr,
                ar.ar1,
                ar.ar2,
                ar.ar3,
                ar.ar4,
                ar.ar5,
                a.ewd,
                a.saturate,
                *ar.rg[:11],
            ]
            for a, ar in zip(atoms, arom)
        ]
        check("ring:" + str(path.relative_to(PROJECT_ROOT)), expected, actual)
    rng = random.Random(260913)
    for mode in ("at", "cp"):
        for case in range(100):
            atoms = [
                gt._Atom(
                    "C",
                    "C",
                    6,
                    0,
                    ambername=rng.choice(("cc", "ce", "cg", "nc", "ne", "pc", "pe"))
                    if mode == "at"
                    else "cp",
                )
                for _ in range(12)
            ]
            bonds = [
                gt._Bond(i, i + 1, rng.choice((1, 2, 3, 7, 8, 10)))
                for i in range(11)
                if rng.random() < 0.8
            ]
            rng.shuffle(bonds)
            expected = [row[0] for row in probe(mode, stdin=graph_input(atoms, bonds))]
            copied = deepcopy(atoms)
            (gt._atadjust if mode == "at" else gt._cpadjust)(copied, bonds)
            check(f"post_{mode}:{case}", expected, [a.ambername for a in copied])
    report = {
        "counts": counts,
        "failed": len(failures),
        "failures": failures,
        "build_command": command,
    }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": counts, "failed": len(failures)}))
    print(f"Report: {work / 'report.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
