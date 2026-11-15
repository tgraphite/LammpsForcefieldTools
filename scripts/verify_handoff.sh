#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VENV_PYTHON:-$ROOT/.venv/bin/python}"

if [ ! -x "$PY" ]; then
    echo "error: $PY does not exist; run scripts/bootstrap_env.sh first" >&2
    exit 1
fi

mkdir -p "$ROOT/_scratch"

echo "[verify] pytest"
"$PY" -m pytest -q "$ROOT/tests"

echo "[verify] periodic ground truth"
"$PY" "$ROOT/scripts/validate_periodic_ground_truth.py"

echo "[verify] manifest ground truth"
"$PY" "$ROOT/scripts/validate_ground_truth.py"

echo "[verify] GAFF typer vs AmberTools26 atomtype oracle (required)"
"$PY" "$ROOT/scripts/validate_gaff_typer_vs_oracle.py"

echo "[verify] GAFF C source semantics (required)"
"$PY" "$ROOT/scripts/validate_gaff_source_semantics.py"

echo "[verify] L5 Sobtop parameter consistency"
"$PY" "$ROOT/scripts/gt44_sobtop_compare.py"

if [ -x "$ROOT/thirdparty/Multiwfn_2026.8.31_src_Linux/Multiwfn_noGUI" ]; then
    echo "[verify] charge engine vs reference Multiwfn (optional oracle)"
    "$PY" "$ROOT/scripts/validate_charges_vs_multiwfn.py" --molecule references/sobtop/examples/SF6/SF6.mol2
fi

echo "[verify] L4 cross-engine single-frame energy (GT-44 ready cases)"
"$PY" "$ROOT/scripts/gt44_runner.py"

echo "[verify] pySCF smoke"
"$PY" "$ROOT/scripts/pyscf_smoke.py" | tee "$ROOT/_scratch/handoff_pyscf.out"

echo "[verify] lft CLI"
"$PY" -m lft --help >/dev/null

echo "[verify] LAMMPS runtime"
LMP="$ROOT/thirdparty/lammps/bin/lmp"
LMP_LIB="$ROOT/thirdparty/lammps/lib"
if [ ! -x "$LMP" ]; then
    echo "error: $LMP missing" >&2
    exit 1
fi
LD_LIBRARY_PATH="$LMP_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$LMP" -h >/dev/null

echo "[verify] all checks passed"
