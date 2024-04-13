"""Capture snapshots of the current state of the IR."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from onnxscript import ir


@dataclasses.dataclass
class ModelSnapshot:
    ir_version: int
    producer_name: str | None
    producer_version: str | None
    domain: str | None
    model_version: int | None
    doc_string: str | None
    opset_imports: dict[str, int]
    metadata_props: dict[str, str]
    graph: int
    functions: list[int]


@dataclasses.dataclass
class GraphSnapshot:
    id: int
    nodes: list[int]
    inputs: list[int]
    outputs: list[int]
    initializers: list[int]
    name: str | None


@dataclasses.dataclass
class NodeSnapshot:
    id: int
    domain: str
    op_type: str
    overload: str
    inputs: list[int]
    outputs: list[int]
    attributes: list[int | None]


@dataclasses.dataclass
class ValueSnapshot:
    id: int
    name: str | None
    type: str | None
    shape: str


@dataclasses.dataclass
class AttributeSnapshot:
    id: int
    name: str
    type: str
    value: str | int | float | Sequence[int] | Sequence[float] | Sequence[str]


@dataclasses.dataclass
class ReferenceAttributeSnapshot:
    id: int
    name: str
    type: str
    ref_attr_name: str


@dataclasses.dataclass
class FunctionSnapshot:
    id: int
    domain: str
    name: str
    overload: str
    inputs: list[int]
    attributes: list[int]
    outputs: list[int]
    nodes: list[int]
    opset_imports: dict[str, int]


class TensorSnapshot(ValueSnapshot):
    id: int
    name: str | None
    dtype: str
    shape: str
    value: Any


@dataclasses.dataclass
class Snapshot:
    root: int
    root_type: str
    model: ModelSnapshot | None = None
    graphs: dict[int, GraphSnapshot] = dataclasses.field(default_factory=dict)
    nodes: dict[int, NodeSnapshot] = dataclasses.field(default_factory=dict)
    values: dict[int, ValueSnapshot] = dataclasses.field(default_factory=dict)
    attributes: dict[int, AttributeSnapshot] = dataclasses.field(default_factory=dict)
    reference_attributes: dict[int, ReferenceAttributeSnapshot] = dataclasses.field(
        default_factory=dict
    )
    functions: dict[int, FunctionSnapshot] = dataclasses.field(default_factory=dict)
    tensors: dict[int, TensorSnapshot] = dataclasses.field(default_factory=dict)


def capture(
    obj: ir.Model | ir.Graph | ir.Node | ir.Value | ir.Attr | ir.RefAttr | ir.Function,
):
    """Capture a snapshot of the current state of the IR."""
    snapshot = Snapshot(id(obj), type(obj).__name__)
    return snapshot


def _capture_model(snapshot: Snapshot, model: ir.ModelProtocol):
    snapshot.model = ModelSnapshot(
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        opset_imports=dict(model.opset_imports.items()),
        metadata_props=model.metadata_props,
        graph=id(model.graph),
        functions=[id(func) for func in model.functions.values()],
    )
    _capture_graph(snapshot, model.graph)
    for func in model.functions.values():
        _capture_function(snapshot, func)


def _capture_graph(snapshot: Snapshot, graph: ir.GraphProtocol | ir.GraphView):
    snapshot.graphs[id(graph)] = GraphSnapshot(
        id=id(graph),
        nodes=[id(node) for node in graph.nodes],
        inputs=[id(input_) for input_ in graph.inputs],
        outputs=[id(output) for output in graph.outputs],
        initializers=[id(init) for init in graph.initializers],
        name=graph.name,
    )
    for node in graph.nodes:
        for input_ in node.inputs:
            _capture_value(snapshot, input_)
        for output in node.outputs:
            _capture_value(snapshot, output)
        _capture_node(snapshot, node)
    for initializer in graph.initializers.values():
        _capture_tensor(snapshot, initializer)


def _capture_function(snapshot: Snapshot, func: ir.FunctionProtocol):
    snapshot.functions[id(func)] = FunctionSnapshot(
        id=id(func),
        domain=func.domain,
        name=func.name,
        overload=func.overload,
        inputs=[id(input_) for input_ in func.inputs],
        outputs=[id(output) for output in func.outputs],
        nodes=[id(node) for node in func.nodes],
        attributes=[id(attr) for attr in func.attributes],
        opset_imports=dict(func.opset_imports.items()),
    )


def _capture_node(snapshot: Snapshot, node: ir.NodeProtocol):
    snapshot.nodes[id(node)] = NodeSnapshot(
        id=id(node),
        domain=node.domain,
        op_type=node.op_type,
        overload=node.overload,
        inputs=[id(input_) for input_ in node.inputs],
        outputs=[id(output) for output in node.outputs],
        attributes=[id(attr) if attr is not None else None for attr in node.attributes],
    )
    for input_ in node.inputs:
        _capture_value(snapshot, input_)
    for output in node.outputs:
        _capture_value(snapshot, output)
    for attr in node.attributes.values():
        if isinstance(attr, ir.ReferenceAttributeProtocol):
            _capture_reference_attribute(snapshot, attr)
        else:
            _capture_attribute(snapshot, attr)


def _capture_value(snapshot: Snapshot, value: ir.ValueProtocol):
    snapshot.values[id(value)] = ValueSnapshot(
        id=id(value),
        name=value.name,
        type=str(value.type),
        shape=str(value.shape),
    )


def _capture_attribute(snapshot: Snapshot, attr: ir.AttributeProtocol):
    if attr.type == ir.AttributeType.GRAPH:
        value = id(attr.value)
        _capture_graph(snapshot, attr.value)
    elif attr.type == ir.AttributeType.TENSOR:
        value = id(attr.value)
        _capture_tensor(snapshot, attr.value)
    elif attr.type == ir.AttributeType.GRAPHS:
        value = [id(graph) for graph in attr.value]
        for graph in attr.value:
            _capture_graph(snapshot, graph)
    elif attr.type == ir.AttributeType.TENSORS:
        value = [id(tensor) for tensor in attr.value]
        for tensor in attr.value:
            _capture_tensor(snapshot, tensor)
    else:
        value = attr.value

    snapshot.attributes[id(attr)] = AttributeSnapshot(
        id=id(attr),
        name=attr.name,
        type=attr.type.name,
        value=value,
    )


def _capture_reference_attribute(
    snapshot: Snapshot, ref_attr: ir.ReferenceAttributeProtocol
):
    snapshot.reference_attributes[id(ref_attr)] = ReferenceAttributeSnapshot(
        id=id(ref_attr),
        name=ref_attr.name,
        type=ref_attr.type.name,
        ref_attr_name=ref_attr.ref_attr_name,
    )


def _capture_tensor(snapshot: Snapshot, tensor: ir.TensorProtocol):
    size_limit = 100
    if tensor.size <= size_limit:
        value = tensor.numpy().tolist()
    else:
        value = None
    snapshot.tensors[id(tensor)] = TensorSnapshot(
        id=id(tensor),
        name=tensor.name,
        dtype=tensor.dtype.name,
        shape=str(tensor.shape),
        value=value,
    )
