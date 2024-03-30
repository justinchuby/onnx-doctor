from typing import Iterable
from . import _diagnostics


def diagnose(ir_object, diagnostics_providers: Iterable[_diagnostics.DiagnosticsProvider]):
    # TODO: Walk the object with a visitor
    ...
