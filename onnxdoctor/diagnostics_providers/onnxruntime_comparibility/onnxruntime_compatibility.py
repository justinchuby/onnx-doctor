"""Analyze model compatibility with ONNX Runtime."""

import dataclasses
from . import _support_table

import onnxdoctor
from onnxrewriter.experimental import intermediate_representation as ir


@dataclasses.dataclass
class OnnxRuntimeOpSchema:
    domain: str
    name: str
    input_types: list[str]  # Name -> TypeStr
    outputs_types: list[str]  # Name -> TypeStr
    # TODO: Handle variadic inputs and outputs
    type_constraints: dict[str, list[str]] = dataclasses.field(
        default_factory=dict
    )  # Type -> Constraints
    version_range: tuple[int, int] | None = None
    execution_provider: str = ""


class OnnxRuntimeCompatibilityLinter(onnxdoctor.DiagnosticsProvider):
    def __init__(self, execution_provider: str = "CPUExecutionProvider"):
        self.execution_provider = execution_provider
        self.ir_version = None
        self.opset_version = None

    def check_model(self, model: ir.ModelProtocol) -> onnxdoctor.DiagnosticsMessageIterator:
        self.ir_version = model.ir_version
        self.opset_version = model.opset_version

    def check_node(self, node: ir.NodeProtocol) -> onnxdoctor.DiagnosticsMessageIterator:
        array = tensor.numpy()
        sparsity = np.count_nonzero(array < self.threshold) / array.size
        if tensor is not None:
            yield onnxdoctor.DiagnosticsMessage(
                target_type="tensor",
                target=tensor,
                message="Sparsity is {:.2f}%".format(sparsity * 100),
                severity="info",
            )
