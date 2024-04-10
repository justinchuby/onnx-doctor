"""Analyze model compatibility with ONNX Runtime."""

from __future__ import annotations

import collections
import dataclasses

from onnxscript import ir

import onnxdoctor

from . import _support_table

_SPECIAL_OPS_TO_SKIP = {
    ("", "Constant"),
    ("", "CastLike"),
}


@dataclasses.dataclass
class OnnxRuntimeOpSchema:
    domain: str
    name: str
    input_types: list[str]  # Name -> TypeStr
    outputs_types: list[str]  # Name -> TypeStr
    # TODO: Handle variadic inputs and outputs
    version_range: tuple[int, int]
    type_constraints: dict[str, list[str]] = dataclasses.field(
        default_factory=dict
    )  # Type -> Constraints
    execution_provider: str = ""


def _get_op_support_table(
    op_schemas: list[dict],
) -> dict[tuple[str, str], list[OnnxRuntimeOpSchema]]:
    op_support_table = collections.defaultdict(list)
    for elem in op_schemas:
        schema = OnnxRuntimeOpSchema(**elem)
        op_support_table[(schema.domain, schema.name)].append(schema)
    return op_support_table


def _version_in_range(version: int, version_range: tuple[int, int]) -> bool:
    return version_range[0] <= version <= version_range[1]


def _to_onnx_string_type_format(type_: ir.TypeProtocol) -> str:
    if isinstance(type_, ir.TensorType):
        return f"tensor({type_.dtype.name.lower()})"
    if isinstance(type_, ir.SequenceType):
        return f"seq({_to_onnx_string_type_format(type_.elem_type)})"
    if isinstance(type_, ir.OptionalType):
        return f"optional({_to_onnx_string_type_format(type_.elem_type)})"
    raise NotImplementedError(f"Type {type(type_)} is not supported.")


class OnnxRuntimeCompatibilityLinter(onnxdoctor.DiagnosticsProvider):
    PRODUCER = "OnnxRuntimeCompatibilityLinter"

    def __init__(self, execution_provider: str = "CPUExecutionProvider"):
        self.execution_provider = execution_provider
        self.ir_version = None
        self.support_table = _get_op_support_table(_support_table.TABLE)
        self.opset_imports = {}
        self.imported_functions = set()

    def check_model(
        self, model: ir.ModelProtocol
    ) -> onnxdoctor.DiagnosticsMessageIterator:
        self.ir_version = model.ir_version
        self.opset_imports = model.opset_imports
        self.imported_functions = set(model.functions)
        return
        yield

    def check_node(  # noqa: PLR0912
        self, node: ir.NodeProtocol
    ) -> onnxdoctor.DiagnosticsMessageIterator:
        function_op_id = (node.domain, node.op_type, node.overload)
        if function_op_id in self.imported_functions:
            # The op is a function op and the function is defined
            # TODO: Handle opset version
            return
        op_id = (node.domain, node.op_type)
        if op_id in _SPECIAL_OPS_TO_SKIP:
            return
        if (schemas := self.support_table.get(op_id)) is None:
            yield onnxdoctor.DiagnosticsMessage(
                target_type="node",
                target=node,
                message=f"Operator {node.domain}::{node.op_type} not supported by {self.execution_provider} in ONNX Runtime.",
                # TODO: Allow customizing severity
                severity="error",
                error_code="operator-unsupported",
                producer=self.PRODUCER,
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
                    f"Operator {node.domain}::{node.op_type} in opset {opset_version} not supported by {self.execution_provider} in ONNX Runtime. "
                    f"All supported versions: {', '.join(f'{schema.version_range[0]}-{schema.version_range[1]}' for schema in schemas)}."
                ),
                severity="error",
                error_code="operator-version-unsupported",
                producer=self.PRODUCER,
            )
            return

        # Check types
        bounded_types: dict[str, ir.TypeProtocol] = {}
        bounded_index = {}
        for i, (input_, type_str) in enumerate(
            zip(node.inputs, found_schema.input_types)
        ):
            if input_ is None:
                continue
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
                    producer=self.PRODUCER,
                )
        for i, (output, type_str) in enumerate(
            zip(node.outputs, found_schema.outputs_types)
        ):
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
                    producer=self.PRODUCER,
                )
        for type_str, type_ in bounded_types.items():
            # type_str can be a type constraint name like T, or a type string like tensor(float)
            # 1/3. Handle the tensor(float) case fist
            onnx_type = _to_onnx_string_type_format(type_)
            if onnx_type == type_str:
                continue
            # 2/3. Handle the B case
            if type_str == "B" and onnx_type == "tensor(bool)":
                # Special case: B means boolean and is sometimes not specified
                continue
            # 3/3. Handle the <T> case
            supported_types = found_schema.type_constraints.get(type_str)
            if supported_types is None:
                yield onnxdoctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"Bug: Type {type_str} is not defined in the schema {found_schema}"
                    ),
                    severity="failure",
                    error_code="typestr-not-exist-in-schema",
                    producer=self.PRODUCER,
                )
                continue
            if (
                onnx_type == "tensor(float16)"
                and self.execution_provider == "CPUExecutionProvider"
            ):
                # Special case: ONNX Runtime supports float16 on CPU by inserting a Cast node
                continue
            if onnx_type not in supported_types:
                yield onnxdoctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"Operator {node.domain}::{node.op_type}-{opset_version} binds type string "
                        f"{type_str}={type_} which is not supported by ONNX Runtime's "
                        f"{self.execution_provider}. Supported types: {', '.join(supported_types)}."
                    ),
                    severity="error",
                    error_code="type-unsupported",
                    producer=self.PRODUCER,
                )
