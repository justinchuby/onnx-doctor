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
    target_type: onnx_doctor.DiagnosticsMessage.target_type,
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

    def __init__(self) -> None:
        self._model: ir.Model | None = None

    def check_model(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        self._model = model

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
                for key, f in model.functions.items()
                if (f.domain or "", f.name or "") not in used_ids
            ]
            if unused_funcs:
                names = ", ".join(
                    f"'{f.domain or ''}:{f.name}'" for f in unused_funcs
                )
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
            domain
            for domain in model.opset_imports
            if domain not in used_domains
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

    def check_graph(self, graph: ir.Graph) -> onnx_doctor.DiagnosticsMessageIterator:
        model = self._model

        # SIM003: unused-nodes
        graph_outputs = frozenset(graph.outputs)
        unused_count = 0
        for node in reversed(list(graph)):
            removable = True
            for output in node.outputs:
                if output in graph_outputs or output.uses():
                    removable = False
                    break
            if removable:
                unused_count += 1
        if unused_count > 0:
            fix = None
            if model is not None:
                fix = lambda: _apply_remove_unused_nodes(model)
            yield _emit(
                _rule("SIM003"),
                "graph",
                graph,
                message=f"Graph has {unused_count} unused node(s) whose outputs are not consumed.",
                fix=fix,
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
