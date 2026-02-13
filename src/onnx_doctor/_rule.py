"""Rule definitions for onnx_doctor diagnostics."""

from __future__ import annotations

import dataclasses
from typing import Literal

RuleCategory = Literal["spec", "ir", "protobuf"]
RuleSeverity = Literal["error", "warning", "info"]
TargetType = Literal[
    "model", "graph", "node", "value", "tensor", "function", "attribute"
]


@dataclasses.dataclass(frozen=True)
class Rule:
    """A lint rule definition.

    Attributes:
        code: Unique rule code, e.g. "ONNX001".
        name: Human-readable kebab-case name, e.g. "empty-graph-name".
        message: Default message template for this rule.
        default_severity: Default severity level.
        category: Whether this rule applies to spec, ir, or protobuf.
        target_type: The IR object type this rule targets.
        explanation: Detailed explanation with examples (markdown).
        suggestion: Default suggestion text for fixing the issue.
        fixable: Whether this rule supports autofix (future).
    """

    code: str
    name: str
    message: str
    default_severity: RuleSeverity
    category: RuleCategory
    target_type: TargetType
    explanation: str = ""
    suggestion: str = ""
    fixable: bool = False
