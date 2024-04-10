from __future__ import annotations

from typing import Iterable, Iterator, TypeAlias

from onnxscript import ir

from . import _message

DiagnosticsMessageIterator: TypeAlias = (
    Iterable[_message.DiagnosticsMessage] | Iterator[_message.DiagnosticsMessage]
)


class DiagnosticsProvider:
    def check_model(self, model: ir.ModelProtocol) -> DiagnosticsMessageIterator:
        del model
        return
        yield

    def check_graph(self, graph: ir.GraphProtocol) -> DiagnosticsMessageIterator:
        del graph
        return
        yield

    def check_function(
        self, function: ir.FunctionProtocol
    ) -> DiagnosticsMessageIterator:
        del function
        return
        yield

    def check_node(self, node: ir.NodeProtocol) -> DiagnosticsMessageIterator:
        del node
        return
        yield

    def check_value(self, value: ir.ValueProtocol) -> DiagnosticsMessageIterator:
        del value
        return
        yield

    def check_attribute(
        self, attr: ir.AttributeProtocol | ir.ReferenceAttributeProtocol
    ) -> DiagnosticsMessageIterator:
        del attr
        return
        yield

    def check_tensor(self, tensor: ir.TensorProtocol) -> DiagnosticsMessageIterator:
        del tensor
        return
        yield
