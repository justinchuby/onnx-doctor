from __future__ import annotations

from collections.abc import Iterable, Sequence

import onnx_ir as ir

from . import _diagnostics, _message


def diagnose(
    model: ir.Model,
    diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider],
) -> Sequence[_message.DiagnosticsMessage]:
    """Run all diagnostics providers on a model.

    Each provider is responsible for walking the model structure as needed.

    Args:
        model: The ONNX IR model to diagnose.
        diagnostics_providers: Providers to run.

    Returns:
        A sequence of diagnostics messages from all providers.
    """
    messages: list[_message.DiagnosticsMessage] = []
    for provider in diagnostics_providers:
        messages.extend(provider.diagnose(model))
    return messages
