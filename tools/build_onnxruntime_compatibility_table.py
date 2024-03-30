# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import dataclasses
import os
import pathlib
from collections import defaultdict

import onnxruntime.capi.onnxruntime_pybind11_state as ort_api

# NOTE: get_all_opkernel_def() segfaults on MacOS


def format_version_range(version_range: tuple[int, int]):
    start, end = version_range



def format_type_constraints(tc):
    tcstr = ""
    firsttcitem = True
    for tcitem in tc:
        if firsttcitem:
            firsttcitem = False
        else:
            tcstr += ", "
        tcstr += tcitem
    return tcstr


def format_param_strings(params):
    firstparam = True
    s = ""
    if params:
        for param in sorted(params):
            if firstparam:
                firstparam = False
            else:
                s += "<br><br>or<br><br>"
            s += param
    return s


def expand_providers(provider_filter: [str]):
    providers = set()
    if provider_filter:
        for provider in provider_filter:
            p = provider.lower()
            if not p.endswith("executionprovider"):
                p += "executionprovider"
            providers.add(p)

    return providers


@dataclasses.dataclass(frozen=True)
class OnnxRuntimeOpSchema:
    domain: str
    name: str
    input_types: list[str]  # Name -> TypeStr
    outputs_types: list[str]  # Name -> TypeStr
    # TODO: Handle variadic inputs and outputs
    type_constraints: dict[str, list[str]] = dataclasses.field(default_factory=dict)  # Type -> Constraints
    version_range: tuple[int, int] | None = None
    execution_provider: str = ""


#
#     >>> ort_api.get_all_opkernel_def()[0].domain
# ''
# >>> ort_api.get_all_opkernel_def()[0].op_name
# 'Abs'
# >>> ort_api.get_all_opkernel_def()[0].type_constraints
# {'T': ['tensor(float)']}
# >>> ort_api.get_all_opkernel_def()[0].version_range
# (6, 12)
# >>> ort_api.get_all_opkernel_def()[0].provider
# 'CPUExecutionProvider'

OpId = tuple[str, str, int]
def build_schema_map():
    op_schemas = ort_api.get_all_operator_schema()
    op_information: dict[OpId, OnnxRuntimeOpSchema] = {}
    provider_support_information = []
    for schema in op_schemas:
        op_information[(schema.domain, schema.name, schema.since_version)] = OnnxRuntimeOpSchema(
            domain=schema.domain,
            name=schema.name,
            input_types=[input.typeStr for input in schema.inputs],
            outputs_types=[output.typeStr for output in schema.outputs],
        )

    for kernel_def in ort_api.get_all_opkernel_def():
        domain = kernel_def.domain
        name = kernel_def.op_name
        provider = kernel_def.provider
        version_range = kernel_def.version_range
        since_version = version_range[0]
        schema = op_information.get((domain, name, since_version))
        assert schema is not None, f"Missing schema for {domain}.{name}-{since_version}"
        new_schema = dataclasses.replace(schema, version_range=version_range, execution_provider=provider, type_constraints=kernel_def.type_constraints)
        provider_support_information.append(new_schema)

    return provider_support_information
