from __future__ import annotations

from collections.abc import Iterable, Sequence

import onnx_ir as ir

from . import _diagnostics, _message


def _node_location(prefix: str, index: int, node: ir.Node) -> str:
    """Build a location string for a node.

    Format: `prefix:node[index](op_type, "name")` or `prefix:node[index](op_type)` if unnamed.
    """
    if node.name:
        return f'{prefix}:node[{index}]({node.op_type}, "{node.name}")'
    return f"{prefix}:node[{index}]({node.op_type})"


def _build_location_map(model: ir.Model) -> dict[int, str]:
    """Build a mapping from object id to location path.

    Walks the model structure once to create paths for all IR objects.
    Uses object id() as key since IR objects may not be hashable.
    """
    locations: dict[int, str] = {}

    # Model
    locations[id(model)] = "model"

    # Functions
    for func in model.functions.values():
        func_name = func.name or "unnamed"
        func_domain = func.domain or ""
        func_id = f"{func_domain}:{func_name}" if func_domain else func_name
        func_loc = f"function({func_id})"
        locations[id(func)] = func_loc

        for inp in func.inputs:
            locations[id(inp)] = f"{func_loc}:input({inp.name or '?'})"

        for i, node in enumerate(func):
            node_loc = _node_location(prefix=func_loc, index=i, node=node)
            locations[id(node)] = node_loc
            for out in node.outputs:
                locations[id(out)] = f"{node_loc}:output({out.name or '?'})"

    # Main graph
    _build_graph_locations(model.graph, "graph", locations)

    return locations


def _build_graph_locations(
    graph: ir.Graph,
    prefix: str,
    locations: dict[int, str],
) -> None:
    """Recursively build location paths for a graph and its contents."""
    locations[id(graph)] = prefix

    for inp in graph.inputs:
        locations[id(inp)] = f"{prefix}:input({inp.name or '?'})"

    for name, init in graph.initializers.items():
        locations[id(init)] = f"{prefix}:initializer({name})"

    for i, node in enumerate(graph):
        node_loc = _node_location(prefix=prefix, index=i, node=node)
        locations[id(node)] = node_loc

        for out in node.outputs:
            locations[id(out)] = f"{node_loc}:output({out.name or '?'})"

        for attr in node.attributes.values():
            # Recurse into subgraphs
            if attr.type == ir.AttributeType.GRAPH and attr.value is not None:
                attr_loc = f"{node_loc}:{attr.name}"
                _build_graph_locations(attr.value, attr_loc, locations)
            elif attr.type == ir.AttributeType.GRAPHS and attr.value is not None:
                for j, subgraph in enumerate(attr.value):
                    attr_loc = f"{node_loc}:{attr.name}[{j}]"
                    _build_graph_locations(subgraph, attr_loc, locations)


def _infer_location(
    msg: _message.DiagnosticsMessage,
    locations: dict[int, str],
) -> None:
    """Set location on a message if not already set, using the location map."""
    if msg.location:
        return  # Already has location

    target_id = id(msg.target)
    msg.location = locations.get(target_id, msg.target_type)


def diagnose(
    model: ir.Model,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> Sequence[_message.DiagnosticsMessage]:
    """Run all diagnostics providers on a model.

    Each provider is responsible for walking the model structure as needed.
    After collecting messages, location is inferred for messages that don't
    have one set, based on the target object.

    Args:
        model: The ONNX IR model to diagnose.
        diagnostics_providers: Providers to run.

    Returns:
        A sequence of diagnostics messages from all providers.
    """
    # Build location map for inference
    locations = _build_location_map(model)

    # Collect messages from all providers
    messages: list[_message.DiagnosticsMessage] = []
    for provider in diagnostics_providers:
        for msg in provider.diagnose(model):
            _infer_location(msg, locations)
            messages.append(msg)

    return messages
