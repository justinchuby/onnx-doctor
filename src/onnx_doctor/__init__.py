"""Diagnose your ONNX model."""

from __future__ import annotations

__all__ = [
    "DiagnosticsMessage",
    "DiagnosticsMessageIterator",
    "DiagnosticsProvider",
    "Rule",
    "RuleRegistry",
    "diagnose",
]

from ._checker import diagnose
from ._diagnostics import DiagnosticsMessageIterator, DiagnosticsProvider
from ._message import DiagnosticsMessage
from ._rule import Rule
from ._rule_registry import RuleRegistry
