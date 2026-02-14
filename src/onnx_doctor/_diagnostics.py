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
    """Base class for diagnostics providers.

    Providers analyze an ONNX model and yield diagnostics messages.
    Each provider is responsible for walking the model structure as needed.
    """

    def diagnose(self, model: ir.Model) -> DiagnosticsMessageIterator:
        """Analyze the model and yield diagnostics messages.

        Providers should override this method to implement their checks.
        The provider is responsible for walking the model graph/functions
        as needed for its analysis.

        Args:
            model: The ONNX IR model to analyze.

        Yields:
            DiagnosticsMessage objects for any issues found.
        """
        del model
        return
        yield
