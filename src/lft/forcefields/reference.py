"""Compatibility layer for the packaged GAFF parameter library.

Historically this module parsed references/sobtop from the development
checkout, which meant installed wheels silently fell back to generic
parameters.  The implementation now lives in lft.forcefields.library and the
table ships inside the package.
"""

from __future__ import annotations

from .library import ParameterLibrary as ReferenceParameters

__all__ = ["ReferenceParameters"]
