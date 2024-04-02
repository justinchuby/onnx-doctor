import onnxdoctor
from onnxscript import ir
import numpy as np


class SparsityAnalyzer(onnxdoctor.DiagnosticsProvider):
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
            )
