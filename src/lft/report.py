"""Human-readable run summary generated from audited IR."""

from __future__ import annotations

from pathlib import Path

from .audit import Audit
from .domain.parameters import ParameterSet
from .domain.structure import Structure


def write_report(structure: Structure, params: ParameterSet, audit: Audit, path: str | Path) -> Path:
    path = Path(path)
    lines = [
        f"# lft run {audit.run_id}",
        "",
        f"- Structure: `{structure.name}`",
        f"- Atoms: {structure.natoms}",
        f"- Bonds/angles/dihedrals/impropers: {len(structure.bonds)}/{len(structure.angles)}/{len(structure.dihedrals)}/{len(structure.impropers)}",
        f"- Declared net charge: `{structure.declared_net_charge:.8f} e`",
        f"- Computed net charge: `{structure.net_charge:.12f} e`",
        f"- Force field: `{params.force_field}`",
        f"- Parameter mode: `{params.parameter_mode}`",
        "",
        "## Warnings",
        "",
    ]
    if audit.warnings:
        lines.extend(f"- `{warning.get('code', 'LFT_WARNING')}`: {warning.get('message', '')}" for warning in audit.warnings)
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
