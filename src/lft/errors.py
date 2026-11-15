"""Structured errors used by the command line and pipeline layers."""

from __future__ import annotations


class LFTError(Exception):
    """An expected user-facing pipeline error."""

    code = "LFT_ERROR"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(message)

    def formatted(self) -> str:
        text = f"[{self.code}] {self.message}"
        if self.hint:
            text += f" Hint: {self.hint}"
        return text


class InputError(LFTError):
    code = "INPUT_ERROR"


class ValidationError(LFTError):
    code = "VALIDATION_ERROR"


class UnsupportedError(LFTError):
    code = "UNSUPPORTED"


class ExternalToolError(LFTError):
    code = "EXTERNAL_TOOL_ERROR"


class ChargeError(LFTError):
    code = "CHARGE_ERROR"


class ParameterError(LFTError):
    code = "PARAMETER_ERROR"
