from __future__ import annotations

import numpy as np
import onnx_ir as ir

import onnx_doctor
from onnx_doctor._rule import Rule

SP001 = Rule(
    code="SP001",
    name="tensor-sparsity",
    message="Tensor has significant sparsity.",
    default_severity="info",
    category="spec",
    target_type="value",
    default_enabled=False,
    explanation="Reports the sparsity percentage of tensor data. High sparsity may indicate opportunities for optimization.",
    suggestion="Consider using sparse tensor representations for highly sparse tensors.",
)


class SparsityAnalyzer(onnx_doctor.DiagnosticsProvider):
    """Analyzes tensor sparsity in ONNX models."""

    PRODUCER = "SparsityAnalyzer"

    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold

    def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        """Analyze sparsity of all initializer tensors in the model."""
        # Check initializers in main graph
        yield from self._check_graph_initializers(model.graph)

        # Check initializers in subgraphs
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_graph_initializers(attr.value)
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_graph_initializers(subgraph)

    def _check_graph_initializers(
        self,
        graph: ir.Graph,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check sparsity of initializers in a graph."""
        for initializer in graph.initializers.values():
            if initializer.const_value is None:
                continue
            tensor = initializer.const_value
            try:
                array = tensor.numpy()
                sparsity = (
                    np.count_nonzero(np.abs(array) <= self.threshold) / array.size
                )
                yield onnx_doctor.DiagnosticsMessage(
                    target_type="value",
                    target=initializer,
                    message=f"Tensor '{tensor.name}' sparsity is {sparsity * 100:.2f}%",
                    severity="info",
                    producer=self.PRODUCER,
                    error_code=SP001.code,
                    rule=SP001,
                )
            except Exception:
                # Skip tensors that can't be converted to numpy
                pass
