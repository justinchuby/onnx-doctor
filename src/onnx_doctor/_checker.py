from __future__ import annotations

import typing
from collections.abc import Iterable, Iterator, Sequence

import onnx_ir as ir

from . import _diagnostics, _message


def _set_location(
    msgs: _diagnostics.DiagnosticsMessageIterator,
    location: str,
) -> Iterator[_message.DiagnosticsMessage]:
    """Set location on messages that don't already have one."""
    for msg in msgs:
        if not msg.location:
            msg.location = location
        yield msg


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
        yield from _set_location(diagnostics_provider.check_model(model), "model")
    for func in model.functions.values():
        func_name = func.name or "unnamed"
        func_domain = func.domain or ""
        func_id = f"{func_domain}:{func_name}" if func_domain else func_name
        yield from diagnose_function(
            func, diagnostics_providers, _location=f"function({func_id})"
        )
    yield from diagnose_graph(model.graph, diagnostics_providers, _location="graph")


def diagnose_graph(
    graph: ir.GraphProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "graph",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(diagnostics_provider.check_graph(graph), _location)
    for value in graph.inputs:
        val_name = value.name or "?"
        yield from diagnose_value(
            value,
            diagnostics_providers,
            _location=f"{_location}:input({val_name})",
        )
    for i, node in enumerate(graph):
        node_label = node.name or node.op_type
        node_loc = f"{_location}:node/{i}({node_label})"
        yield from diagnose_node(
            node,
            diagnostics_providers,
            _location=node_loc,
        )
    for initializer in graph.initializers.values():
        yield from diagnose_tensor(
            initializer,
            diagnostics_providers,
            _location=_location,
        )


def diagnose_function(
    function: ir.FunctionProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "function",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(
            diagnostics_provider.check_function(function),
            _location,
        )
    for value in function.inputs:
        val_name = value.name or "?"
        yield from diagnose_value(
            value,
            diagnostics_providers,
            _location=f"{_location}:input({val_name})",
        )
    for i, node in enumerate(function):
        node_label = node.name or node.op_type
        node_loc = f"{_location}:node/{i}({node_label})"
        yield from diagnose_node(
            node,
            diagnostics_providers,
            _location=node_loc,
        )


def diagnose_node(
    node: ir.NodeProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "node",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(diagnostics_provider.check_node(node), _location)
    for value in node.outputs:
        val_name = value.name or "?"
        yield from diagnose_value(
            value,
            diagnostics_providers,
            _location=_location,
        )
    for attr in node.attributes.values():
        yield from diagnose_attribute(
            attr,
            diagnostics_providers,
            _location=_location,
        )


def diagnose_tensor(
    tensor: ir.TensorProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "tensor",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(diagnostics_provider.check_tensor(tensor), _location)


def diagnose_value(
    value: ir.ValueProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "value",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(diagnostics_provider.check_value(value), _location)


def diagnose_attribute(
    attribute: ir.AttributeProtocol | ir.ReferenceAttributeProtocol,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
    _location: str = "attribute",
) -> _diagnostics.DiagnosticsMessageIterator:
    for diagnostics_provider in diagnostics_providers:
        yield from _set_location(
            diagnostics_provider.check_attribute(attribute),
            _location,
        )
    if attribute.type == ir.AttributeType.TENSOR:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        yield from diagnose_tensor(
            attribute.value,
            diagnostics_providers,
            _location=_location,
        )
    elif attribute.type == ir.AttributeType.TENSORS:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        for tensor in attribute.value:
            yield from diagnose_tensor(
                tensor,
                diagnostics_providers,
                _location=_location,
            )
    elif attribute.type == ir.AttributeType.GRAPH:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        subgraph_loc = f"{_location}:{attribute.name}"
        yield from diagnose_graph(
            attribute.value,
            diagnostics_providers,
            _location=subgraph_loc,
        )
    elif attribute.type == ir.AttributeType.GRAPHS:
        attribute = typing.cast(ir.AttributeProtocol, attribute)
        for i, graph in enumerate(attribute.value):
            subgraph_loc = f"{_location}:{attribute.name}[{i}]"
            yield from diagnose_graph(
                graph,
                diagnostics_providers,
                _location=subgraph_loc,
            )
