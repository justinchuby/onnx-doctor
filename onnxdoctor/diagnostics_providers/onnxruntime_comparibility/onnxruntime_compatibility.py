"""Analyze model compatibility with ONNX Runtime."""

import collections
import dataclasses
from . import _support_table

import onnxdoctor
from onnxrewriter.experimental import intermediate_representation as ir
from onnxrewriter.experimental.intermediate_representation import _core

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


def _get_op_support_table(op_schemas: list[dict]) -> dict[tuple[str, str], list[OnnxRuntimeOpSchema]]:
    op_support_table = collections.defaultdict(list)
    for elem in op_schemas:
        schema = OnnxRuntimeOpSchema(**elem)
        op_support_table[(schema.domain, schema.name)].append(schema)
    return op_support_table


def _version_in_range(version: int, version_range: tuple[int, int]) -> bool:
    return version_range[0] <= version <= version_range[1]


def _to_type_str(type_: ir.TypeProtocol) -> str:
    if isinstance(type_, _core.TensorType):
        return f"tensor({type_.dtype.name.lower()})"
    raise NotImplementedError(f"Type {type_} is not supported.")

class OnnxRuntimeCompatibilityLinter(onnxdoctor.DiagnosticsProvider):
    def __init__(self, execution_provider: str = "CPUExecutionProvider"):
        self.execution_provider = execution_provider
        self.ir_version = None
        self.support_table = _get_op_support_table(_support_table.TABLE)
        self.opset_imports = {}

    def check_model(self, model: ir.ModelProtocol) -> onnxdoctor.DiagnosticsMessageIterator:
        self.ir_version = model.ir_version
        self.opset_imports = model.opset_imports
        return []

    def check_node(self, node: ir.NodeProtocol) -> onnxdoctor.DiagnosticsMessageIterator:
        op_id = (node.domain, node.op_type)
        if (schemas := self.support_table.get(op_id)) is None:
            yield onnxdoctor.DiagnosticsMessage(
                target_type="node",
                target=node,
                message=f"ONNX Runtime does not support operator {node.domain}::{node.op_type} with {self.execution_provider}",
                # TODO: Allow customizing severity
                severity="error",
                error_code="operator-unsupported",
            )
            return
        opset_version = self.opset_imports[node.domain]
        found_schema = None
        for schema in schemas:
            if _version_in_range(opset_version, schema.version_range):
                found_schema = schema
                break
        if found_schema is None:
            yield onnxdoctor.DiagnosticsMessage(
                target_type="node",
                target=node,
                message=(
                    f"ONNX Runtime does not support operator {node.domain}::{node.op_type} with {self.execution_provider} in opset {opset_version}. "
                    f"All supported versions: {', '.join(f'{schema.version_range[0]}-{schema.version_range[1]}' for schema in schemas)}."
                ),
                severity="error",
                error_code="operator-version-unsupported",
            )
            return

        # Check types
        bounded_types: dict[str, ir.TypeProtocol] = {}
        bounded_index = {}
        for i, (input_, type_str) in enumerate(zip(node.inputs, found_schema.input_types)):
            if input_.type is None:
                continue
            if type_str not in bounded_types:
                bounded_types[type_str] = input_.type
                bounded_index[type_str] = i
            elif bounded_types[type_str] != input_.type:
                yield onnxdoctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"ONNX Runtime expects input {input_.name} of operator {node.domain}::{node.op_type} to have type {type_str}={bounded_types[type_str]} (bounded by index {bounded_index[type_str]}), but found {input_.type}."
                    ),
                    severity="error",
                    error_code="node-type-inconsistent",
                )
        for i, (output, type_str) in enumerate(zip(node.outputs, found_schema.outputs_types)):
            if output.type is None:
                continue
            if type_str not in bounded_types:
                bounded_types[type_str] = output.type
                # TODO: Differentiate between input and output indices
                bounded_index[type_str] = i
            elif bounded_types[type_str] != output.type:
                yield onnxdoctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"ONNX Runtime expects output {output.name} of operator {node.domain}::{node.op_type} to have type {type_str}={bounded_types[type_str]} (bounded by index {bounded_index[type_str]}), but found {output.type}."
                    ),
                    severity="error",
                    error_code="node-type-inconsistent",
                )
        for type_str, type_ in bounded_types.items():
            supported_types = found_schema.type_constraints.get(type_str)
            assert supported_types is not None, f"Bug: Type {type_str} is not defined in the schema {found_schema}"
            if _to_type_str(type_) not in supported_types:
                yield onnxdoctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"Operator {node.domain}::{node.op_type}-{opset_version} binds type string {type_str}={type_} which is not supported by ONNX Runtime. Supported types: {', '.join(supported_types)}."
                    ),
                    severity="error",
                    error_code="type-unsupported",
                )
