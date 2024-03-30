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
    # TODO: Mark as required
    producer: str = ""
    error_code: str = ""
