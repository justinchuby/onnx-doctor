"""Analyze model compatibility with ONNX Runtime."""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib

import onnx_ir as ir

import onnx_doctor
from onnx_doctor._rule import Rule

_SPECIAL_OPS_TO_SKIP = {
    ("", "Constant"),
    ("", "CastLike"),
}

ORT001 = Rule(
    code="ORT001",
    name="operator-unsupported",
    message="Operator is not supported by the execution provider in ONNX Runtime.",
    default_severity="error",
    category="spec",
    target_type="node",
    explanation="The operator used by this node is not supported by the specified ONNX Runtime execution provider.",
    suggestion="Use a different operator, or switch to an execution provider that supports it.",
)

ORT002 = Rule(
    code="ORT002",
    name="operator-version-unsupported",
    message="Operator at this opset version is not supported by the execution provider.",
    default_severity="error",
    category="spec",
    target_type="node",
    explanation="The operator exists but not at the opset version used by the model.",
    suggestion="Change the opset version to one supported by the execution provider.",
)

ORT003 = Rule(
    code="ORT003",
    name="node-type-inconsistent",
    message="Type constraint mismatch for operator inputs/outputs.",
    default_severity="error",
    category="spec",
    target_type="node",
    explanation="ONNX Runtime expects inputs/outputs sharing a type constraint to have the same type.",
    suggestion="Ensure all inputs/outputs bound to the same type constraint have matching types.",
)

ORT004 = Rule(
    code="ORT004",
    name="type-unsupported",
    message="Data type is not supported by the execution provider for this operator.",
    default_severity="error",
    category="spec",
    target_type="node",
    explanation="The data type used by this operator is not supported by the specified execution provider.",
    suggestion="Cast inputs to a supported type before this operator.",
)

ORT005 = Rule(
    code="ORT005",
    name="typestr-not-in-schema",
    message="Type string is not defined in the operator schema.",
    default_severity="error",
    category="spec",
    target_type="node",
    explanation="Internal error: a type constraint string referenced by the operator is not found in its schema.",
)


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


class OnnxRuntimeCompatibilityLinter(onnx_doctor.DiagnosticsProvider):
    PRODUCER = "OnnxRuntimeCompatibilityLinter"

    def __init__(self, execution_provider: str = "CPUExecutionProvider"):
        self.execution_provider = execution_provider
        self.ir_version = None
        with open(
            pathlib.Path(__file__).parent / "ort_supported_schemas.json",
            encoding="utf-8",
        ) as f:
            support_table = json.load(f)
        self.support_table = _get_op_support_table(support_table)
        self.opset_imports = {}
        self.imported_functions = set()

    def check_model(
        self, model: ir.ModelProtocol
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        self.ir_version = model.ir_version
        self.opset_imports = model.opset_imports
        self.imported_functions = set(model.functions)
        return
        yield

    def check_node(  # noqa: PLR0912
        self, node: ir.NodeProtocol
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        function_op_id = (node.domain, node.op_type, node.overload)
        if function_op_id in self.imported_functions:
            # The op is a function op and the function is defined
            # TODO: Handle opset version
            return
        op_id = (node.domain, node.op_type)
        if op_id in _SPECIAL_OPS_TO_SKIP:
            return
        if (schemas := self.support_table.get(op_id)) is None:
            yield onnx_doctor.DiagnosticsMessage(
                target_type="node",
                target=node,
                message=f"Operator {node.domain}::{node.op_type} not supported by {self.execution_provider} in ONNX Runtime.",
                severity="error",
                error_code=ORT001.code,
                producer=self.PRODUCER,
                rule=ORT001,
            )
            return
        opset_version = self.opset_imports[node.domain]
        found_schema = None
        for schema in schemas:
            if _version_in_range(opset_version, schema.version_range):
                found_schema = schema
                break
        if found_schema is None:
            yield onnx_doctor.DiagnosticsMessage(
                target_type="node",
                target=node,
                message=(
                    f"Operator {node.domain}::{node.op_type} in opset {opset_version} not supported by {self.execution_provider} in ONNX Runtime. "
                    f"All supported versions: {', '.join(f'{schema.version_range[0]}-{schema.version_range[1]}' for schema in schemas)}."
                ),
                severity="error",
                error_code=ORT002.code,
                producer=self.PRODUCER,
                rule=ORT002,
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
                yield onnx_doctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"ONNX Runtime expects input {input_.name} of operator {node.domain}::{node.op_type} to have type {type_str}={bounded_types[type_str]} (bounded by index {bounded_index[type_str]}), but found {input_.type}."
                    ),
                    severity="error",
                    error_code=ORT003.code,
                    rule=ORT003,
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
                yield onnx_doctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"ONNX Runtime expects output {output.name} of operator {node.domain}::{node.op_type} to have type {type_str}={bounded_types[type_str]} (bounded by index {bounded_index[type_str]}), but found {output.type}."
                    ),
                    severity="error",
                    error_code=ORT003.code,
                    rule=ORT003,
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
                yield onnx_doctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"Bug: Type {type_str} is not defined in the schema {found_schema}"
                    ),
                    severity="failure",
                    error_code=ORT005.code,
                    rule=ORT005,
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
                yield onnx_doctor.DiagnosticsMessage(
                    target_type="node",
                    target=node,
                    message=(
                        f"Operator {node.domain}::{node.op_type}-{opset_version} binds type string "
                        f"{type_str}={type_} which is not supported by ONNX Runtime's "
                        f"{self.execution_provider}. Supported types: {', '.join(supported_types)}."
                    ),
                    severity="error",
                    error_code=ORT004.code,
                    rule=ORT004,
                    producer=self.PRODUCER,
                )
