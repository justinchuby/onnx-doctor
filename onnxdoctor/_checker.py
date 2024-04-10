from __future__ import annotations

import typing
from typing import Iterable, Sequence

from onnxscript import ir

from . import _diagnostics, _message


def diagnose(  # noqa: PLR0911
    ir_object: _message.PossibleTargets,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> Sequence[_message.DiagnosticsMessage]:
    if isinstance(ir_object, ir.ModelProtocol):
        return list(diagnose_model(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.GraphProtocol):
        return list(diagnose_graph(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.FunctionProtocol):
        return list(diagnose_function(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.NodeProtocol):
        return list(diagnose_node(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.TensorProtocol):
        return list(diagnose_tensor(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.ValueProtocol):
        return list(diagnose_value(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.AttributeProtocol):
        return list(diagnose_attribute(ir_object, diagnostics_providers))
    if isinstance(ir_object, ir.ReferenceAttributeProtocol):
        return list(diagnose_attribute(ir_object, diagnostics_providers))
    raise TypeError(f"Unknown IR object: {ir_object}")


def diagnose_model(
    model: ir.ModelProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_model(model)
    for func in model.functions.values():
        yield from diagnose_function(func, diagnostics_providers)
    yield from diagnose_graph(model.graph, diagnostics_providers)


def diagnose_graph(
    graph: ir.GraphProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_graph(graph)
    for value in graph.inputs:
        yield from diagnose_value(value, diagnostics_providers)
    for node in graph.nodes:
        yield from diagnose_node(node, diagnostics_providers)
    for initializer in graph.initializers.values():
        yield from diagnose_tensor(initializer, diagnostics_providers)


def diagnose_function(
    function: ir.FunctionProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_function(function)
    for value in function.inputs:
        yield from diagnose_value(value, diagnostics_providers)
    for node in function.nodes:
        yield from diagnose_node(node, diagnostics_providers)
    # for value in function.outputs:
    #     yield from diagnose_value(value, diagnostics_providers)


def diagnose_node(
    node: ir.NodeProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_node(node)
    for value in node.inputs:
        yield from diagnose_value(value, diagnostics_providers)
    for value in node.outputs:
        yield from diagnose_value(value, diagnostics_providers)
    for attr in node.attributes.values():
        yield from diagnose_attribute(attr, diagnostics_providers)


def diagnose_tensor(
    tensor: ir.TensorProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_tensor(tensor)


def diagnose_value(
    value: ir.ValueProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_value(value)


def diagnose_attribute(
    attribute: ir.AttributeProtocol | ir.ReferenceAttributeProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from diagnostics_provider.check_attribute(attribute)
    if attribute.type == ir.AttributeType.TENSOR:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        yield from diagnose_tensor(attribute.value, diagnostics_providers)
    elif attribute.type == ir.AttributeType.TENSORS:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        for tensor in attribute.value:
            yield from diagnose_tensor(tensor, diagnostics_providers)
    elif attribute.type == ir.AttributeType.GRAPH:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        yield from diagnose_graph(attribute.value, diagnostics_providers)
    elif attribute.type == ir.AttributeType.GRAPHS:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        for graph in attribute.value:
            yield from diagnose_graph(graph, diagnostics_providers)
