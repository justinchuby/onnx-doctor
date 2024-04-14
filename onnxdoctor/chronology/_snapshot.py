"""Capture snapshots of the current state of the IR."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from onnxscript import ir
import onnx


METADATA_KEY_ID = "pkg.onnxdoctor.chronology.object_id"


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


@dataclasses.dataclass
class TensorSnapshot:
    id: int
    class_name: str
    name: str | None
    dtype: str
    shape: str
    value: list[Any] | None


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
    obj: ir.ModelProtocol
    | ir.GraphProtocol
    | ir.GraphView
    | ir.NodeProtocol
    | ir.FunctionProtocol,
    assign_id: bool = False,
) -> Snapshot:
    """Capture a snapshot of the current state of the IR.

    Args:
        obj: An object in the IR.
        assign_id: Whether to record the object ID to the ``metadata_props`` of the object.
            This is useful for persisting the ID across sessions. The IDs will be
            preserved in the protobuf objects.
    """
    snapshot = Snapshot(id(obj), type(obj).__name__)
    if isinstance(obj, ir.ModelProtocol):
        _capture_model(snapshot, obj, assign_id=assign_id)
    elif isinstance(obj, (ir.GraphProtocol, ir.GraphView)):
        _capture_graph(snapshot, obj, assign_id=assign_id)
    elif isinstance(obj, ir.NodeProtocol):
        _capture_node(snapshot, obj, assign_id=assign_id)
    elif isinstance(obj, ir.FunctionProtocol):
        _capture_function(snapshot, obj, assign_id=assign_id)
    return snapshot


def capture_proto(
    proto: onnx.ModelProto | onnx.GraphProto | onnx.NodeProto | onnx.FunctionProto,
) -> Snapshot:
    if isinstance(proto, onnx.ModelProto):
        model = ir.serde.deserialize_model(proto)
        snapshot = Snapshot(id(model), type(model).__name__)
        _capture_model(snapshot, model, assign_id=True)
    elif isinstance(proto, onnx.GraphProto):
        graph = ir.serde.deserialize_graph(proto)
        snapshot = Snapshot(id(graph), type(graph).__name__)
        _capture_graph(snapshot, graph, assign_id=True)
    elif isinstance(proto, onnx.NodeProto):
        node = ir.serde.deserialize_node(proto)
        snapshot = Snapshot(id(node), type(node).__name__)
        _capture_node(snapshot, node, assign_id=True)
    elif isinstance(proto, onnx.FunctionProto):
        func = ir.serde.deserialize_function(proto)
        snapshot = Snapshot(id(func), type(func).__name__)
        _capture_function(snapshot, func, assign_id=True)

    return snapshot


def _get_or_create_id(obj: Any, assign_id: bool) -> int:
    if obj.metadata_props.get(METADATA_KEY_ID) is not None:
        return int(obj.metadata_props[METADATA_KEY_ID])
    object_id = id(obj)
    if assign_id:
        obj.metadata_props[METADATA_KEY_ID] = str(object_id)
    return object_id


def _capture_model(
    snapshot: Snapshot, model: ir.ModelProtocol, *, assign_id: bool = False
):
    snapshot.model = ModelSnapshot(
        ir_version=model.ir_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string,
        opset_imports=dict(model.opset_imports.items()),
        metadata_props=dict(model.metadata_props.items()),
        graph=_get_or_create_id(model.graph, assign_id),
        functions=[
            _get_or_create_id(func, assign_id) for func in model.functions.values()
        ],
    )
    _capture_graph(snapshot, model.graph, assign_id=assign_id)
    for func in model.functions.values():
        _capture_function(snapshot, func, assign_id=assign_id)


def _capture_graph(
    snapshot: Snapshot,
    graph: ir.GraphProtocol | ir.GraphView,
    *,
    assign_id: bool = False,
):
    graph_id = _get_or_create_id(graph, assign_id)
    snapshot.graphs[graph_id] = GraphSnapshot(
        id=graph_id,
        nodes=[_get_or_create_id(node, assign_id) for node in graph.nodes],
        inputs=[_get_or_create_id(input_, assign_id) for input_ in graph.inputs],
        outputs=[_get_or_create_id(output, assign_id) for output in graph.outputs],
        # TODO: It is easy to mistakenly capture the name of the initializers if we forget to call value()
        initializers=[
            _get_or_create_id(init, assign_id) for init in graph.initializers.values()
        ],
        name=graph.name,
    )
    for input_ in graph.inputs:
        _capture_value(snapshot, input_, assign_id=assign_id)
    for node in graph.nodes:
        _capture_node(snapshot, node, assign_id=assign_id)
    for initializer in graph.initializers.values():
        _capture_tensor(snapshot, initializer, assign_id=assign_id)


def _capture_function(
    snapshot: Snapshot, func: ir.FunctionProtocol, *, assign_id: bool = False
):
    function_id = _get_or_create_id(func, assign_id)
    snapshot.functions[function_id] = FunctionSnapshot(
        id=function_id,
        domain=func.domain,
        name=func.name,
        overload=func.overload,
        inputs=[_get_or_create_id(input_, assign_id) for input_ in func.inputs],
        outputs=[_get_or_create_id(output, assign_id) for output in func.outputs],
        nodes=[_get_or_create_id(node, assign_id) for node in func.nodes],
        attributes=[_get_or_create_id(attr, assign_id) for attr in func.attributes],
        opset_imports=dict(func.opset_imports.items()),
    )


def _capture_node(
    snapshot: Snapshot, node: ir.NodeProtocol, *, assign_id: bool = False
):
    node_id = _get_or_create_id(node, assign_id)
    snapshot.nodes[node_id] = NodeSnapshot(
        id=node_id,
        domain=node.domain,
        op_type=node.op_type,
        overload=node.overload,
        inputs=[_get_or_create_id(input_, assign_id) for input_ in node.inputs],
        outputs=[_get_or_create_id(output, assign_id) for output in node.outputs],
        attributes=[
            _get_or_create_id(attr, assign_id) if attr is not None else None
            for attr in node.attributes
        ],
    )
    for input_ in node.inputs:
        if input_ is not None:
            _capture_value(snapshot, input_, assign_id=assign_id)
    for output in node.outputs:
        _capture_value(snapshot, output, assign_id=assign_id)
    for attr in node.attributes.values():
        if isinstance(attr, ir.ReferenceAttributeProtocol):
            _capture_reference_attribute(snapshot, attr, assign_id=assign_id)
        else:
            _capture_attribute(snapshot, attr, assign_id=assign_id)


def _capture_value(
    snapshot: Snapshot, value: ir.ValueProtocol, *, assign_id: bool = False
):
    value_id = _get_or_create_id(value, assign_id)
    snapshot.values[value_id] = ValueSnapshot(
        id=value_id,
        name=value.name,
        type=str(value.type),
        shape=str(value.shape),
    )


def _capture_attribute(
    snapshot: Snapshot, attr: ir.AttributeProtocol, *, assign_id: bool = False
):
    if attr.type == ir.AttributeType.GRAPH:
        value = _get_or_create_id(attr.value, assign_id)
        _capture_graph(snapshot, attr.value)
    elif attr.type == ir.AttributeType.TENSOR:
        value = _get_or_create_id(attr.value, assign_id)
        _capture_tensor(snapshot, attr.value)
    elif attr.type == ir.AttributeType.GRAPHS:
        value = [_get_or_create_id(graph, assign_id) for graph in attr.value]
        for graph in attr.value:
            _capture_graph(snapshot, graph, assign_id=assign_id)
    elif attr.type == ir.AttributeType.TENSORS:
        value = [_get_or_create_id(tensor, assign_id) for tensor in attr.value]
        for tensor in attr.value:
            _capture_tensor(snapshot, tensor, assign_id=assign_id)
    else:
        value = attr.value

    attr_id = _get_or_create_id(attr, assign_id)
    snapshot.attributes[attr_id] = AttributeSnapshot(
        id=attr_id,
        name=attr.name,
        type=attr.type.name,
        value=value,
    )


def _capture_reference_attribute(
    snapshot: Snapshot,
    ref_attr: ir.ReferenceAttributeProtocol,
    *,
    assign_id: bool = False,
):
    ref_attr_id = _get_or_create_id(ref_attr, assign_id)
    snapshot.reference_attributes[ref_attr_id] = ReferenceAttributeSnapshot(
        id=ref_attr_id,
        name=ref_attr.name,
        type=ref_attr.type.name,
        ref_attr_name=ref_attr.ref_attr_name,
    )


def _capture_tensor(
    snapshot: Snapshot, tensor: ir.TensorProtocol, *, assign_id: bool = False
):
    tensor_id = _get_or_create_id(tensor, assign_id)
    size_limit = 100
    if tensor.size <= size_limit:
        value = tensor.numpy().tolist()
    else:
        value = None
    snapshot.tensors[tensor_id] = TensorSnapshot(
        id=tensor_id,
        class_name=type(tensor).__name__,
        name=tensor.name,
        dtype=tensor.dtype.name,
        shape=str(tensor.shape),
        value=value,
    )
