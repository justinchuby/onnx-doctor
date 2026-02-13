from __future__ import annotations

import typing
from collections.abc import Iterable, Iterator
from typing import Union

import onnx_ir as ir

from . import _message

if typing.TYPE_CHECKING:
    from typing_extensions import TypeAlias

DiagnosticsMessageIterator: TypeAlias = Union[
    Iterable[_message.DiagnosticsMessage], Iterator[_message.DiagnosticsMessage]
]


class DiagnosticsProvider:
    def check_model(self, model: ir.Model) -> DiagnosticsMessageIterator:
        del model
        return
        yield

    def check_graph(self, graph: ir.Graph) -> DiagnosticsMessageIterator:
        del graph
        return
        yield

    def check_function(self, function: ir.Function) -> DiagnosticsMessageIterator:
        del function
        return
        yield

    def check_node(self, node: ir.Node) -> DiagnosticsMessageIterator:
        del node
        return
        yield

    def check_value(self, value: ir.Value) -> DiagnosticsMessageIterator:
        del value
        return
        yield

    def check_attribute(self, attr: ir.Attr) -> DiagnosticsMessageIterator:
        del attr
        return
        yield

    def check_tensor(self, tensor: ir.Tensor) -> DiagnosticsMessageIterator:
        del tensor
        return
        yield
