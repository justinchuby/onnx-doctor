"""Diagnostics messages."""

from __future__ import annotations

import dataclasses

from typing import Literal
from onnxscript import ir


PossibleTargetTypes = Literal[
    "model",
    "node",
    "attribute",
    "tensor",
    "graph",
    "function",
]

PossibleTargets = (
    ir.ModelProtocol
    | ir.NodeProtocol
    | ir.AttributeProtocol
    | ir.TensorProtocol
    | ir.GraphProtocol
    | ir.FunctionProtocol
    | ir.ReferenceAttributeProtocol
)

PossibleSeverities = Literal[
    "error", "warning", "info", "recommendation", "debug", "failure"
]


@dataclasses.dataclass
class DiagnosticsMessage:
    """Diagnostics message."""

    target_type: PossibleTargetTypes
    target: PossibleTargets
    message: str
    severity: PossibleSeverities
    producer: str
    error_code: str
