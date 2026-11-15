"""Global lft configuration loaded from ``lft.config``.

The intent is that a user installs lft once and edits a single ``lft.config``
file in the software directory.  There is deliberately no per-project config
lookup.  Values are plain ``key = value`` lines; ``#`` and ``;`` start
comments.  Command-line flags always take precedence over this file.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_NAME = "lft.config"
_SOFTWARE_ROOT = Path(__file__).resolve().parents[2]

_ENV_KEYS = {
    "gmx": "LFT_GMX",
    "lmp": "LFT_LMP",
    "moltemplate": "LFT_MOLTEMPLATE",
    "obabel": "LFT_OBABEL",
    "atomtype": "LFT_ATOMTYPE",
    "orca": "LFT_APP_ORCA",
    "param_dir": "LFT_PARAM_DIR",
}

_BOOL_ENV_KEYS = {
    "external_atomtype": "LFT_ATOMTYPE_EXTERNAL",
    "am1bcc_external": "LFT_AM1BCC_EXTERNAL",
}


def config_path() -> Path | None:
    """Return the active ``lft.config`` path, honouring ``$LFT_CONFIG``."""

    override = os.environ.get("LFT_CONFIG")
    if override:
        return Path(override)
    for candidate in (_SOFTWARE_ROOT / CONFIG_NAME, Path(__file__).resolve().parent / CONFIG_NAME):
        if candidate.is_file():
            return candidate
    return None


def load_settings() -> dict[str, str]:
    """Parse the global config file into a flat string mapping."""

    path = config_path()
    if path is None or not path.is_file():
        return {}
    settings: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        if key:
            settings[key.lower()] = value
    return settings


def apply_environment(settings: dict[str, str]) -> None:
    """Promote tool-path settings to environment variables for runners.

    Explicit environment variables win, so an operator can always override the
    config file from the shell without editing it.
    """

    for key, env_name in _ENV_KEYS.items():
        value = settings.get(key)
        if value and not os.environ.get(env_name):
            os.environ[env_name] = value
    for key, env_name in _BOOL_ENV_KEYS.items():
        value = settings.get(key)
        if value and not os.environ.get(env_name):
            os.environ[env_name] = "1" if value.strip().lower() in {"1", "yes", "true", "on"} else "0"
