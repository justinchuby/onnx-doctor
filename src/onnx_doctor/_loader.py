"""Load rules from YAML files and build the global rule registry."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from onnx_doctor._rule import Rule, RuleCategory, RuleSeverity, TargetType
from onnx_doctor._rule_registry import RuleRegistry

_SPEC_YAML = (
    pathlib.Path(__file__).parent / "diagnostics_providers" / "onnx_spec" / "spec.yaml"
)
_PROTOBUF_YAML = (
    pathlib.Path(__file__).parent
    / "diagnostics_providers"
    / "onnx_spec"
    / "protobuf.yaml"
)
_SIMPLIFICATION_YAML = (
    pathlib.Path(__file__).parent
    / "diagnostics_providers"
    / "simplification"
    / "simplification.yaml"
)


def load_rules_from_yaml(path: pathlib.Path) -> list[Rule]:
    """Load rules from a YAML file.

    The YAML is expected to have a top-level 'rules' key, with sub-keys
    for each target type (graph, model, node, value, tensor, function, attribute).
    Each sub-key contains a list of rule definitions.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules: list[Rule] = []
    rules_data: dict[str, list[dict[str, Any]]] = data.get("rules", {})
    for target_type_str, rule_list in rules_data.items():
        if not isinstance(rule_list, list):
            continue
        rules.extend(_parse_rule(rule_data, target_type_str) for rule_data in rule_list)
    return rules


def _parse_rule(data: dict[str, Any], target_type: str) -> Rule:
    """Parse a single rule from YAML data."""
    return Rule(
        code=data["code"],
        name=data["name"],
        message=data.get("message", ""),
        default_severity=_parse_severity(data.get("severity", "error")),
        category=_parse_category(data.get("category", "spec")),
        target_type=_parse_target_type(target_type),
        explanation=data.get("explanation", ""),
        suggestion=data.get("suggestion", ""),
        fixable=data.get("fixable", False),
        default_enabled=data.get("default_enabled", True),
    )


def _parse_severity(value: str) -> RuleSeverity:
    value = value.lower()
    if value in ("error", "warning", "info"):
        return value  # type: ignore[return-value]
    raise ValueError(f"Invalid severity: {value}")


def _parse_category(value: str) -> RuleCategory:
    if value in ("spec", "ir", "protobuf"):
        return value  # type: ignore[return-value]
    raise ValueError(f"Invalid category: {value}")


def _parse_target_type(value: str) -> TargetType:
    if value in {"model", "graph", "node", "value", "function"}:
        return value  # type: ignore[return-value]
    raise ValueError(f"Invalid target type: {value}")


def build_default_registry() -> RuleRegistry:
    """Build the default rule registry with all built-in rules."""
    registry = RuleRegistry()
    for rule in load_rules_from_yaml(_SPEC_YAML):
        registry.register(rule)
    for rule in load_rules_from_yaml(_PROTOBUF_YAML):
        registry.register(rule)
    for rule in load_rules_from_yaml(_SIMPLIFICATION_YAML):
        registry.register(rule)

    # Register rules defined in Python (ORT provider)
    from onnx_doctor.diagnostics_providers.onnxruntime_compatibility import (  # noqa: PLC0415
        ORT001,
        ORT002,
        ORT003,
        ORT004,
        ORT005,
    )

    for rule in (ORT001, ORT002, ORT003, ORT004, ORT005):
        registry.register(rule)

    return registry


# Global default registry, lazily populated on first access.
_default_registry: RuleRegistry | None = None


def get_default_registry() -> RuleRegistry:
    """Get the default rule registry (singleton)."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry
