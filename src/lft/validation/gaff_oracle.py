"""Strict, isolated AmberTools26 oracle execution for GAFF validation only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from lft.domain.structure import Structure
from lft.errors import ExternalToolError
from lft.forcefields.atomtype import _write_mol2
from lft.runners.external import run_command

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = PROJECT_ROOT / "research" / "ambertools26"


@dataclass(frozen=True)
class OracleConfig:
    executable: Path
    amberhome: Path
    definition: Path

    @classmethod
    def from_environment(cls) -> OracleConfig:
        home = Path(os.environ.get("AMBERHOME", RESEARCH_ROOT / "ambertools26_src"))
        return cls(
            Path(
                os.environ.get("LFT_ORACLE_ATOMTYPE", RESEARCH_ROOT / "build/atomtype")
            ).resolve(),
            home.resolve(),
            Path(
                os.environ.get(
                    "LFT_ORACLE_DEF", home / "dat/antechamber/ATOMTYPE_GFF.DEF"
                )
            ).resolve(),
        )

    def validate(self) -> None:
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ExternalToolError(f"atomtype oracle unavailable: {self.executable}")
        for path in (
            self.definition,
            self.amberhome / "dat/antechamber/CORR_NAME_TYPE.DAT",
        ):
            if not path.is_file():
                raise ExternalToolError(f"atomtype oracle data unavailable: {path}")


def parse_oracle_ac(path: Path, natoms: int) -> dict[int, str]:
    if not path.is_file():
        raise ExternalToolError(f"atomtype produced no output: {path}")
    types: dict[int, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "ATOM":
            continue
        # wac uses adjacent fixed-width name/residue fields, so whitespace
        # splitting fails for a four-character atom name (e.g. O123MOL).
        try:
            index = int(line[4:11])
            int(line[21:26])
            for start, end in ((26, 38), (38, 46), (46, 54), (54, 64)):
                float(line[start:end])
            atom_type = line[64:].strip()
            if not atom_type or len(atom_type.split()) != 1:
                raise ValueError("missing/invalid atom type")
        except ValueError as exc:
            raise ExternalToolError(f"malformed atomtype ATOM record: {line}") from exc
        if index in types:
            raise ExternalToolError(f"duplicate atomtype atom id: {index}")
        types[index] = atom_type
    if set(types) != set(range(1, natoms + 1)) or natoms == 0:
        raise ExternalToolError(
            f"atomtype returned invalid atom ids/count: {len(types)} for {natoms} atoms"
        )
    return types


def run_oracle(
    structure: Structure, config: OracleConfig, workdir: Path
) -> dict[int, str]:
    """Create a fresh case directory; a failed run can never reuse old output."""
    config.validate()
    workdir.mkdir(parents=True, exist_ok=True)
    case = Path(mkdtemp(prefix="case-", dir=workdir.resolve()))
    mol2, output = case / "input.mol2", case / "output.ac"
    _write_mol2(structure, mol2)
    result = run_command(
        [
            str(config.executable),
            "-i",
            str(mol2),
            "-f",
            "mol2",
            "-o",
            str(output),
            "-d",
            str(config.definition),
            "-a",
            "1",
        ],
        cwd=case,
        env={"AMBERHOME": str(config.amberhome)},
        allowed_environment=("AMBERHOME",),
        timeout=120,
    )
    (case / "stdout.txt").write_text(result.stdout)
    (case / "stderr.txt").write_text(result.stderr)
    return parse_oracle_ac(output, structure.natoms)
