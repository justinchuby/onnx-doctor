"""Diagnostics messages."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal, Union

import onnx_ir as ir

from onnx_doctor._rule import Rule

PossibleTargetTypes = Literal[
    "model",
    "node",
    "attribute",
    "tensor",
    "graph",
    "function",
]

PossibleTargets = Union[
    ir.Model,
    ir.Graph,
    ir.Node,
    ir.Attr,
    ir.Tensor,
    ir.Value,
    ir.Function,
]

PossibleSeverities = Literal[
    "error", "warning", "info", "recommendation", "debug", "failure"
]

# A fix is a callable that takes no arguments and mutates the IR in place.
Fix = Callable[[], None]


@dataclasses.dataclass
class DiagnosticsMessage:
    """Diagnostics message."""

    target_type: PossibleTargetTypes
    target: PossibleTargets
    message: str
    severity: PossibleSeverities
    producer: str
    error_code: str
    rule: Rule | None = None
    suggestion: str | None = None
    location: str | None = None
    fix: Fix | None = None
