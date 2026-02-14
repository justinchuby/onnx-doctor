"""Simplification provider — detects unused code in ONNX models."""

from __future__ import annotations

import onnx_ir as ir

import onnx_doctor
from onnx_doctor._loader import get_default_registry
from onnx_doctor._rule import Rule


def _rule(code: str) -> Rule:
    """Look up a rule by code from the default registry."""
    rule = get_default_registry().get_by_code(code)
    assert rule is not None, f"Rule {code} not found in registry"
    return rule


def _emit(
    rule: Rule,
    target_type: onnx_doctor._message.PossibleTargetTypes,
    target: object,
    message: str | None = None,
    suggestion: str | None = None,
    location: str | None = None,
    fix: onnx_doctor._message.Fix | None = None,
) -> onnx_doctor.DiagnosticsMessage:
    """Create a DiagnosticsMessage from a rule."""
    return onnx_doctor.DiagnosticsMessage(
        target_type=target_type,
        target=target,
        message=message or rule.message,
        severity=rule.default_severity,
        producer="SimplificationProvider",
        error_code=rule.code,
        rule=rule,
        suggestion=suggestion or rule.suggestion or None,
        location=location,
        fix=fix,
    )


class SimplificationProvider(onnx_doctor.DiagnosticsProvider):
    """Provider that detects unused code (dead functions, nodes, opsets)."""

    PRODUCER = "SimplificationProvider"

    def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        """Analyze the model for simplification opportunities."""
        # Check for unused functions, opsets at model level
        yield from self._check_model(model)

        # Check for unused nodes in the main graph
        yield from self._check_graph(model.graph, model)

        # Check subgraphs recursively
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_graph(attr.value, model)
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_graph(subgraph, model)

    def _check_model(
        self,
        model: ir.Model,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check model-level simplification opportunities."""
        # SIM001: unused-functions
        if model.functions:
            used_ids: set[tuple[str, str]] = set()
            for node in ir.traversal.RecursiveGraphIterator(model.graph):
                domain = node.domain if node.domain else ""
                used_ids.add((domain, node.op_type))
            # Also check references from within functions
            for func in model.functions.values():
                for node in func:
                    domain = node.domain if node.domain else ""
                    used_ids.add((domain, node.op_type))

            unused_funcs = [
                f
                for f in model.functions.values()
                if (f.domain or "", f.name or "") not in used_ids
            ]
            if unused_funcs:
                names = ", ".join(f"'{f.domain or ''}:{f.name}'" for f in unused_funcs)
                yield _emit(
                    _rule("SIM001"),
                    "model",
                    model,
                    message=f"Model has {len(unused_funcs)} unused function(s): {names}.",
                    fix=lambda: _apply_remove_unused_functions(model),
                )

        # SIM002: unused-opset-imports
        used_domains: set[str] = {""}  # Default domain always used
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            used_domains.add(node.domain if node.domain else "")
        for func in model.functions.values():
            used_domains.add(func.domain or "")
            for node in func:
                used_domains.add(node.domain if node.domain else "")

        unused_opsets = [
            domain for domain in model.opset_imports if domain not in used_domains
        ]
        if unused_opsets:
            names = ", ".join(f"'{d}'" for d in unused_opsets)
            yield _emit(
                _rule("SIM002"),
                "model",
                model,
                message=f"Model has {len(unused_opsets)} unused opset import(s): {names}.",
                fix=lambda: _apply_remove_unused_opsets(model),
            )

    def _check_graph(
        self,
        graph: ir.Graph,
        model: ir.Model,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a graph for unused nodes."""
        # SIM003: unused-nodes
        graph_outputs = frozenset(graph.outputs)
        for node in reversed(list(graph)):
            removable = True
            for output in node.outputs:
                if output in graph_outputs or output.uses():
                    removable = False
                    break
            if removable:
                if node.name:
                    node_label = f'"{node.name}" ({node.op_type})'
                else:
                    node_label = f"({node.op_type})"
                yield _emit(
                    _rule("SIM003"),
                    "node",
                    node,
                    message=f"Node {node_label} outputs are not consumed.",
                    fix=lambda: _apply_remove_unused_nodes(model),
                )


def _apply_remove_unused_functions(model: ir.Model) -> None:
    from onnx_ir.passes.common import RemoveUnusedFunctionsPass  # noqa: PLC0415

    RemoveUnusedFunctionsPass()(model)


def _apply_remove_unused_opsets(model: ir.Model) -> None:
    from onnx_ir.passes.common import RemoveUnusedOpsetsPass  # noqa: PLC0415

    RemoveUnusedOpsetsPass()(model)


def _apply_remove_unused_nodes(model: ir.Model) -> None:
    from onnx_ir.passes.common import RemoveUnusedNodesPass  # noqa: PLC0415

    RemoveUnusedNodesPass()(model)
