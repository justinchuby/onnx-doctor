"""Diagnostics messages."""
from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass
class DiagnosticsMessage:
    """Diagnostics message."""

    target_type: PossibleTargetTypes
    target: PossibleTargets
    message: str
    severity: PossibleSeverities
