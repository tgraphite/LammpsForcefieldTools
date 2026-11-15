#!/usr/bin/env python3
"""Restore and build the AmberTools26 atomtype oracle in a persistent directory.

Usage: .venv/bin/python scripts/build_gaff_oracle.py --archive /path/ambertools26.tar.bz2
Only antechamber source, rule data and licensing files are restored. Existing
files must match the archive; this command never modifies references/thirdparty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lft.runners.external import run_command
from lft.validation.gaff_oracle import RESEARCH_ROOT


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", type=Path, default=RESEARCH_ROOT)
    parser.add_argument("--cc", default="gcc")
    args = parser.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    prefixes = (
        "ambertools26_src/AmberTools/src/antechamber/",
        "ambertools26_src/dat/antechamber/",
    )
    licenses = {
        "ambertools26_src/AmberTools/LICENSE",
        "ambertools26_src/AmberTools/GNU_LGPL_v3",
    }
    hashes: dict[str, str] = {}
    with tarfile.open(args.archive, "r|bz2") as archive:
        for member in archive:
            if not member.name.startswith(prefixes) and member.name not in licenses:
                continue
            if not member.isfile():
                continue
            path = dest / member.name
            if not path.resolve().is_relative_to(dest):
                raise ValueError(f"unsafe archive path: {member.name}")
            stream = archive.extractfile(member)
            assert stream is not None
            data = stream.read()
            digest = hashlib.sha256(data).hexdigest()
            if path.exists() and sha256(path) != digest:
                raise ValueError(f"existing source differs from archive: {path}")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            hashes[member.name] = digest
    source = dest / "ambertools26_src/AmberTools/src/antechamber"
    build = dest / "build"
    build.mkdir(exist_ok=True)
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise ValueError(f"compiler not found: {args.cc}")
    version = run_command([compiler, "--version"], cwd=build).stdout
    command = [
        compiler,
        "-O2",
        "-o",
        str(build / "atomtype"),
        str(source / "atomtype.c"),
        str(source / "eprintf.c"),
        "-lm",
    ]
    result = run_command(command, cwd=build)
    (build / "build.stdout").write_text(result.stdout)
    (build / "build.stderr").write_text(result.stderr)
    manifest = {
        "archive": str(args.archive.resolve()),
        "archive_sha256": sha256(args.archive),
        "restored_files_sha256": hashes,
        "compiler": compiler,
        "compiler_version": version,
        "command": command,
        "cwd": str(build),
        "binary_sha256": sha256(build / "atomtype"),
        "scope": "Unmodified AmberTools26 atomtype only; not a full AmberTools installation",
    }
    (dest / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Built {build / 'atomtype'}; provenance: {dest / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
