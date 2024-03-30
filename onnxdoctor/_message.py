"""Diagnostics messages."""
from __future__ import annotations

from typing import Literal
from onnxrewriter.experimental import intermediate_representation as ir


PossibleTargetTypes = Literal[
    "model",
    "node",
    "attribute",
    "tensor",
    "graph",
    "function",
]

PossibleTargets = ir.Model | ir.Node | ir.Attr | ir.Tensor | ir.Graph | ir.Function

PossibleSeverities = Literal["error", "warning", "info", "debug"]


class DiagnosticsMessage:
    """Diagnostics message."""

    def __init__(
        self,
        target_type: PossibleTargetTypes,
        target: PossibleTargets,
        message: str,
        severity: PossibleSeverities,
    ):
        self.target_type = target_type
        self.target = target
        self.message = message
        self.severity = severity

    def __str__(self):
        return f"""\
{self.severity}: {self.target_type} : {self.message}
{self.target!r}"""
