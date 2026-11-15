#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/usr/bin/python3}"
VENV="$ROOT/.venv"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: python not found: $PYTHON" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] < (3, 11):
    raise SystemExit(f"Python >= 3.11 required, got {sys.version}")
PY

echo "[bootstrap] creating $VENV with $PYTHON"
"$PYTHON" -m venv --clear "$VENV"

PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip

if [ "${OFFLINE:-0}" = "1" ] && [ -d "$ROOT/vendor/wheels" ]; then
    echo "[bootstrap] installing from vendor/wheels (offline)"
    "$PIP" install --no-index --find-links "$ROOT/vendor/wheels" -r "$ROOT/requirements.lock"
else
    "$PIP" install -r "$ROOT/requirements.lock"
fi

"$PIP" install --no-deps -e "$ROOT"
"$PIP" check

echo "[bootstrap] done. Use:"
echo "  $VENV/bin/python -m pytest -q"
echo "  $VENV/bin/python scripts/pyscf_smoke.py"
echo "  $VENV/bin/python -m lft --help"
