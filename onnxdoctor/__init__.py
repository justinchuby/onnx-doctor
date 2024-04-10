"""Diagnose your ONNX model."""
from __future__ import annotations

__all__ = [
    "diagnose",
    "DiagnosticsProvider",
    "DiagnosticsMessageIterator",
    "DiagnosticsMessage",
]

from ._checker import diagnose
from ._diagnostics import DiagnosticsMessageIterator, DiagnosticsProvider
from ._message import DiagnosticsMessage
