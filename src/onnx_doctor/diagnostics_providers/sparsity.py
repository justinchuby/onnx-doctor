from __future__ import annotations

import numpy as np
import onnx_ir as ir

import onnxdoctor
from onnxdoctor._rule import Rule

SP001 = Rule(
    code="SP001",
    name="tensor-sparsity",
    message="Tensor has significant sparsity.",
    default_severity="info",
    category="spec",
    target_type="tensor",
    explanation="Reports the sparsity percentage of tensor data. High sparsity may indicate opportunities for optimization.",
    suggestion="Consider using sparse tensor representations for highly sparse tensors.",
)


class SparsityAnalyzer(onnxdoctor.DiagnosticsProvider):
    PRODUCER = "SparsityAnalyzer"

    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold

    def check_tensor(
        self, tensor: ir.TensorProtocol
    ) -> onnxdoctor.DiagnosticsMessageIterator:
        array = tensor.numpy()
        sparsity = np.count_nonzero(np.abs(array) <= self.threshold) / array.size
        if tensor is not None:
            yield onnxdoctor.DiagnosticsMessage(
                target_type="tensor",
                target=tensor,
                message=f"Sparsity is {sparsity * 100:.2f}%",
                severity="info",
                producer=self.PRODUCER,
                error_code=SP001.code,
                rule=SP001,
            )
