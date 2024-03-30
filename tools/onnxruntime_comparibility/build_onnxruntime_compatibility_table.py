# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import dataclasses
import json
from typing import Tuple

import onnxruntime.capi.onnxruntime_pybind11_state as ort_api

# NOTE: get_all_opkernel_def() segfaults on MacOS

OpId = Tuple[str, str, int]


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


def get_supported_schemas() -> list[OnnxRuntimeOpSchema]:
    op_schemas = ort_api.get_all_operator_schema()
    op_information: dict[OpId, OnnxRuntimeOpSchema] = {}
    for schema in op_schemas:
        op_information[(schema.domain, schema.name, schema.since_version)] = (
            OnnxRuntimeOpSchema(
                domain=schema.domain,
                name=schema.name,
                input_types=[input_.typeStr for input_ in schema.inputs],
                outputs_types=[output.typeStr for output in schema.outputs],
            )
        )

    # Example usage of get_all_opkernel_def
    # >>> ort_api.get_all_opkernel_def()[0].domain
    # ''
    # >>> ort_api.get_all_opkernel_def()[0].op_name
    # 'Abs'
    # >>> ort_api.get_all_opkernel_def()[0].type_constraints
    # {'T': ['tensor(float)']}
    # >>> ort_api.get_all_opkernel_def()[0].version_range
    # (6, 12)
    # >>> ort_api.get_all_opkernel_def()[0].provider
    # 'CPUExecutionProvider'

    # (domain, name, version_range, provider) -> OnnxRuntimeOpSchema
    provider_support_information: dict[
        tuple[str, str, tuple[int, int], str], OnnxRuntimeOpSchema
    ] = {}
    for kernel_def in ort_api.get_all_opkernel_def():
        domain = kernel_def.domain
        name = kernel_def.op_name
        provider = kernel_def.provider
        version_range = kernel_def.version_range
        since_version = version_range[0]
        schema = op_information.get((domain, name, since_version))
        assert schema is not None, f"Missing schema for {domain}.{name}-{since_version}"
        schema_key = (domain, name, version_range, provider)
        if schema_key in provider_support_information:
            known_type_constraints = provider_support_information[
                schema_key
            ].type_constraints
            for constraint_name, types in known_type_constraints.items():
                # Add the new constraints into the known types
                types.extend(kernel_def.type_constraints[constraint_name])
        else:
            type_constraints = {}
            for name, types in kernel_def.type_constraints.items():
                type_constraints[name] = list(types)
            provider_support_information[schema_key] = dataclasses.replace(
                schema,
                version_range=version_range,
                execution_provider=provider,
                type_constraints=type_constraints,
            )

    return sorted(
        provider_support_information.values(),
        key=lambda schema: (
            schema.execution_provider,
            schema.domain,
            schema.name,
            schema.version_range,
        ),
    )


def main():
    supported_schemas = get_supported_schemas()
    schemas = [dataclasses.asdict(schema) for schema in supported_schemas]
    with open("ort_supported_schemas.json", "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)


if __name__ == "__main__":
    main()
