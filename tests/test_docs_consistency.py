"""Static consistency checks between the CLI/codebase and the documentation.

These tests are intentionally import/introspection based: they fail when a new
CLI option, charge method, configuration key, supported format, or module is
added without updating the user manual or the module status document.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import lft
from lft.charges import CHARGE_METHODS
from lft.cli import build_parser
from lft.io.loaders import SUPPORTED_STRUCTURE_SUFFIXES
from lft.settings import _BOOL_ENV_KEYS, _ENV_KEYS

ROOT = Path(__file__).resolve().parents[1]
MANUAL = (ROOT / "docs" / "MANUAL.md").read_text(encoding="utf-8")
STATUS = (ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

def _backticked(token: str) -> str:
    return f"`{token}`"


def test_manual_documents_every_cli_option() -> None:
    parser = build_parser()
    documented = {token for action in parser._actions for token in action.option_strings if _backticked(token) in MANUAL}
    missing = sorted(token for action in parser._actions for token in action.option_strings if token not in documented)
    assert not missing, f"CLI options missing from docs/MANUAL.md: {missing}"


def test_manual_documents_every_charge_method() -> None:
    missing = [method for method in CHARGE_METHODS if _backticked(method) not in MANUAL]
    assert not missing, f"charge methods missing from docs/MANUAL.md: {missing}"


def test_manual_documents_every_supported_structure_suffix() -> None:
    missing = [suffix for suffix in SUPPORTED_STRUCTURE_SUFFIXES if _backticked(suffix) not in MANUAL]
    assert not missing, f"structure suffixes missing from docs/MANUAL.md: {missing}"


def test_manual_documents_configuration_keys_and_environment_variables() -> None:
    keys = {"output", "ff", "charge", "qc", *(_ENV_KEYS), *(_BOOL_ENV_KEYS)}
    missing_keys = sorted(key for key in keys if _backticked(key) not in MANUAL)
    assert not missing_keys, f"lft.config keys missing from docs/MANUAL.md: {missing_keys}"

    env_names = {"LFT_CONFIG", *_ENV_KEYS.values(), *_BOOL_ENV_KEYS.values()}
    missing_env = sorted(name for name in env_names if _backticked(name) not in MANUAL)
    assert not missing_env, f"environment variables missing from docs/MANUAL.md: {missing_env}"


def test_project_status_covers_every_module() -> None:
    package_root = ROOT / "src" / "lft"
    modules = []
    for path in sorted(package_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        modules.append(str(path.relative_to(ROOT)))
    missing = [module for module in modules if _backticked(module) not in STATUS]
    assert not missing, f"modules missing from docs/PROJECT_STATUS.md: {missing}"


def test_readme_covers_install_modules_and_common_parameters() -> None:
    for heading in ("## 安装", "## 模块", "## 通用参数", "## 文档", "## 许可证"):
        assert heading in README, f"README is missing section {heading!r}"
    assert "lammps-forcefield-tools" in README
    assert "lft -f water.mol2" in README
    missing = [
        option
        for option in ("--structure", "--output", "--ff", "--charge", "--net-charge", "--mult", "--outdir", "--force", "--debug")
        if _backticked(option) not in README
    ]
    assert not missing, f"common CLI options missing from README.md: {missing}"


def test_package_version_is_consistent() -> None:
    assert PYPROJECT["project"]["name"] == "lammps-forcefield-tools"
    assert PYPROJECT["project"]["version"] == lft.__version__
    assert f"[{lft.__version__}]" in CHANGELOG
    assert lft.__version__ in README or "PyPI" in README


def test_cli_help_renders_and_names_the_entry_point() -> None:
    help_text = build_parser().format_help()
    assert help_text.startswith("usage: lft")
    assert "lft -f molecule.mol2" in help_text


def test_docs_do_not_reference_the_old_project_name() -> None:
    for name, text in (("README.md", README), ("docs/MANUAL.md", MANUAL), ("docs/PROJECT_STATUS.md", STATUS)):
        assert "lammps_ff_topo" not in text, f"{name} still references the old working name"
        assert "lammps-ff-topo" not in text, f"{name} still references the old distribution name"
