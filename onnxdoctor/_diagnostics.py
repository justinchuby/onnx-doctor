import abc
from typing import Iterable, Iterator, TypeAlias
from onnxrewriter.experimental import intermediate_representation as ir
from . import _message


DiagnosticsMessageIterator: TypeAlias = (
    Iterable[_message.DiagnosticsMessage] | Iterator[_message.DiagnosticsMessage]
)


class DiagnosticsProvider(abc.ABC):
    def check_model(self, model: ir.ModelProtocol) -> DiagnosticsMessageIterator:
        del model
        return []

    def check_graph(self, graph: ir.GraphProtocol) -> DiagnosticsMessageIterator:
        del graph
        return []

    def check_function(self, function: ir.FunctionProtocol) -> DiagnosticsMessageIterator:
        del function
        return []

    def check_node(self, node: ir.NodeProtocol) -> DiagnosticsMessageIterator:
        del node
        return []

    def check_value(self, value: ir.ValueProtocol) -> DiagnosticsMessageIterator:
        del value
        return []

    def check_attribute(self, attr: ir.AttributeProtocol | ir.ReferenceAttributeProtocol) -> DiagnosticsMessageIterator:
        del attr
        return []

    def check_tensor(self, tensor: ir.TensorProtocol) -> DiagnosticsMessageIterator:
        del tensor
        return []
