import abc
from typing import Iterable, Iterator, TypeAlias
from onnxrewriter.experimental import intermediate_representation as ir
from . import _message


DiagnosticsMessages: TypeAlias = (
    Iterable[_message.DiagnosticsMessage] | Iterator[_message.DiagnosticsMessage]
)


class DiagnosticsProvider(abc.ABC):
    def check_model(self, model: ir.Model) -> DiagnosticsMessages:
        del model
        return []

    def check_graph(self, graph: ir.GraphProtocol) -> DiagnosticsMessages:
        del graph
        return []

    def check_node(self, node: ir.NodeProtocol) -> DiagnosticsMessages:
        del node
        return []

    def check_tensor(self, tensor: ir.TensorProtocol) -> DiagnosticsMessages:
        del tensor
        return []

    def check_value(self, value: ir.ValueProtocol) -> DiagnosticsMessages:
        del value
        return []
