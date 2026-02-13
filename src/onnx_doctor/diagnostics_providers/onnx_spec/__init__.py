"""ONNX spec linter."""

from __future__ import annotations

import os
import pathlib

import onnx
import onnx_ir as ir

import onnx_doctor
from onnx_doctor._loader import get_default_registry
from onnx_doctor._rule import Rule

# Maximum supported IR version by this checker.
_MAX_IR_VERSION = 14


def _rule(code: str) -> Rule:
    """Look up a rule by code from the default registry."""
    rule = get_default_registry().get_by_code(code)
    assert rule is not None, f"Rule {code} not found in registry"
    return rule


def _emit(
    rule: Rule,
    target_type: onnx_doctor._message.PossibleTargetTypes,
    target: object,
    message: str | None = None,
    suggestion: str | None = None,
    location: str | None = None,
    fix: onnx_doctor._message.Fix | None = None,
) -> onnx_doctor.DiagnosticsMessage:
    """Create a DiagnosticsMessage from a rule."""
    return onnx_doctor.DiagnosticsMessage(
        target_type=target_type,
        target=target,
        message=message or rule.message,
        severity=rule.default_severity,
        producer="OnnxSpecProvider",
        error_code=rule.code,
        rule=rule,
        suggestion=suggestion or rule.suggestion or None,
        location=location,
        fix=fix,
    )


class OnnxSpecProvider(onnx_doctor.DiagnosticsProvider):
    """Provider that checks models against the ONNX specification."""

    PRODUCER = "OnnxSpecProvider"

    def __init__(self) -> None:
        self._ir_version: int | None = None
        self._opset_imports: dict[str, int] = {}
        self._model_dir: pathlib.Path | None = None
        self._root_graph: ir.Graph | None = None
        self._model: ir.Model | None = None

    def check_model(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        self._ir_version = model.ir_version
        self._opset_imports = dict(model.opset_imports)
        self._root_graph = model.graph
        self._model = model

        # ONNX012: invalid-ir-version
        if model.ir_version is None or model.ir_version < 1:
            yield _emit(
                _rule("ONNX012"),
                "model",
                model,
                message=f"Model ir_version is {model.ir_version!r}, which is invalid.",
            )

        # ONNX013: ir-version-too-new
        if model.ir_version is not None and model.ir_version > _MAX_IR_VERSION:
            yield _emit(
                _rule("ONNX013"),
                "model",
                model,
                message=f"Model ir_version {model.ir_version} is newer than the checker supports (max {_MAX_IR_VERSION}).",
            )

        # ONNX014: duplicate-metadata-keys
        if model.metadata_props is not None:
            seen_keys: set[str] = set()
            for key in model.metadata_props:
                if key in seen_keys:
                    yield _emit(
                        _rule("ONNX014"),
                        "model",
                        model,
                        message=f"Duplicate metadata key: '{key}'.",
                    )
                seen_keys.add(key)

        # ONNX015: missing-default-opset
        if (
            model.ir_version is not None
            and model.ir_version >= 3
            and "" not in model.opset_imports
        ):
            yield _emit(_rule("ONNX015"), "model", model)

        # ONNX016: unexpected-opset-import
        if (
            model.ir_version is not None
            and model.ir_version < 3
            and len(model.opset_imports) > 0
        ):
            yield _emit(_rule("ONNX016"), "model", model)

    def check_graph(self, graph: ir.Graph) -> onnx_doctor.DiagnosticsMessageIterator:
        # ONNX001: empty-graph-name
        if not graph.name:
            yield _emit(
                _rule("ONNX001"),
                "graph",
                graph,
                fix=lambda: setattr(graph, "name", "main_graph"),
            )

        # ONNX036/ONNX037: only apply to the root graph, not subgraphs
        is_root_graph = self._root_graph is not None and graph is self._root_graph

        # ONNX036: graph-io-missing-type / ONNX037: graph-io-missing-shape
        if is_root_graph:
            for value in graph.inputs:
                if value.type is None:
                    yield _emit(
                        _rule("ONNX036"),
                        "graph",
                        graph,
                        message=f"Graph input '{value.name}' is missing type information.",
                    )
                elif value.shape is None:
                    yield _emit(
                        _rule("ONNX037"),
                        "graph",
                        graph,
                        message=f"Graph input '{value.name}' is missing shape information.",
                    )
            for value in graph.outputs:
                if value.type is None:
                    yield _emit(
                        _rule("ONNX036"),
                        "graph",
                        graph,
                        message=f"Graph output '{value.name}' is missing type information.",
                    )
                elif value.shape is None:
                    yield _emit(
                        _rule("ONNX037"),
                        "graph",
                        graph,
                        message=f"Graph output '{value.name}' is missing shape information.",
                    )

        # ONNX101: duplicate-graph-io
        seen: set[ir.Value] = set()
        for value in graph.inputs:
            if value in seen:
                yield _emit(
                    _rule("ONNX101"),
                    "graph",
                    graph,
                    message=f"Duplicate Value object in graph inputs: '{value.name}'.",
                )
            seen.add(value)
        seen.clear()
        for value in graph.outputs:
            if value in seen:
                yield _emit(
                    _rule("ONNX101"),
                    "graph",
                    graph,
                    message=f"Duplicate Value object in graph outputs: '{value.name}'.",
                )
            seen.add(value)

        # ONNX003: empty-initializer-name
        for tensor in graph.initializers.values():
            if not tensor.name:
                yield _emit(
                    _rule("ONNX003"),
                    "graph",
                    graph,
                    fix=lambda: _apply_name_fix(self._model),
                )

        # Collect known values for topological order check
        known_values: set[ir.Value] = set()
        for inp in graph.inputs:
            known_values.add(inp)
        for init in graph.initializers.values():
            known_values.add(init)

        # ONNX004: unsorted-graph-nodes + ONNX005: unknown-node-input
        is_sorted = True
        for node in graph:
            for inp in node.inputs:
                if inp is None:
                    continue
                if inp not in known_values:
                    producer = inp.producer()
                    if producer is not None and producer.graph is graph:
                        # Producer is in this graph but hasn't been seen yet — unsorted
                        is_sorted = False
                    elif (
                        producer is None
                        and not inp.is_graph_input()
                        and not inp.is_initializer()
                    ):
                        # Truly unknown value — not from any scope
                        is_sorted = False
                        # ONNX005: unknown-node-input
                        yield _emit(
                            _rule("ONNX005"),
                            "graph",
                            graph,
                            message=f"Node '{node.op_type}' has input '{inp.name}' not produced by any node or graph input.",
                        )
                    # Otherwise: outer-scope reference (parent graph input/initializer/node) — not a sort issue
            for out in node.outputs:
                known_values.add(out)

        if not is_sorted:
            yield _emit(
                _rule("ONNX004"),
                "graph",
                graph,
                fix=graph.sort,
            )

        # ONNX006: experimental-op
        for node in graph:
            try:
                domain = node.domain if node.domain else ""
                opset_version = self._opset_imports.get(domain)
                if opset_version is not None:
                    schema = onnx.defs.get_schema(node.op_type, opset_version, domain)
                    if (
                        schema.support_level
                        == onnx.defs.OpSchema.SupportType.EXPERIMENTAL
                    ):
                        yield _emit(
                            _rule("ONNX006"),
                            "graph",
                            graph,
                            message=f"Node '{node.op_type}' uses experimental operator '{domain}::{node.op_type}'.",
                        )
            except Exception:  # noqa: PERF203
                pass

        # ONNX007: duplicate-value-name + ONNX010: graph-ssa-violation
        seen_names: dict[str, int] = {}
        for inp in graph.inputs:
            if inp.name:
                seen_names[inp.name] = seen_names.get(inp.name, 0) + 1
        for node in graph:
            for out in node.outputs:
                if out.name:
                    seen_names[out.name] = seen_names.get(out.name, 0) + 1

        for name, count in seen_names.items():
            if count > 1:
                yield _emit(
                    _rule("ONNX007"),
                    "graph",
                    graph,
                    message=f"Value name '{name}' appears {count} times in the graph.",
                )
                yield _emit(
                    _rule("ONNX010"),
                    "graph",
                    graph,
                    message=f"Value name '{name}' is assigned {count} times, violating SSA form.",
                )

        # ONNX009: graph-output-not-produced
        for out in graph.outputs:
            producer = out.producer()
            if producer is None and not out.is_graph_input() and not out.is_initializer():
                yield _emit(
                    _rule("ONNX009"),
                    "graph",
                    graph,
                    message=f"Graph output '{out.name}' is not produced by any node in the graph.",
                )
            elif producer is not None and producer.graph is not graph:
                yield _emit(
                    _rule("ONNX009"),
                    "graph",
                    graph,
                    message=f"Graph output '{out.name}' is produced in a different graph.",
                    suggestion="Apply `--fix` to insert Identity nodes with `OutputFixPass`.",
                    fix=lambda: _apply_output_fix(self._model),
                )

        # ONNX011: initializer-name-conflict
        # Check for name conflicts between initializers and subgraph inputs.
        # This is checked within subgraphs of nodes in this graph.
        init_names = set(graph.initializers.keys())
        for node in graph:
            for attr in node.attributes.values():
                if hasattr(attr, "type") and attr.type == ir.AttributeType.GRAPH:
                    subgraph = attr.value
                    for sub_input in subgraph.inputs:
                        if sub_input.name and sub_input.name in init_names:
                            yield _emit(
                                _rule("ONNX011"),
                                "graph",
                                graph,
                                message=f"Initializer '{sub_input.name}' conflicts with subgraph input name.",
                            )

    def check_node(self, node: ir.Node) -> onnx_doctor.DiagnosticsMessageIterator:
        domain = node.domain if node.domain else ""

        # ONNX017: missing-opset-for-domain
        if domain not in self._opset_imports:
            yield _emit(
                _rule("ONNX017"),
                "node",
                node,
                message=f"No opset imported for domain '{domain}' used by node '{node.op_type}'.",
            )
            return

        opset_version = self._opset_imports[domain]

        # ONNX018: deprecated-op + ONNX019: unregistered-op
        # Only check schema for official ONNX domains
        official_domains = {"", "ai.onnx", "ai.onnx.ml"}
        try:
            schema = onnx.defs.get_schema(node.op_type, opset_version, domain)
            if schema.deprecated:
                yield _emit(
                    _rule("ONNX018"),
                    "node",
                    node,
                    message=f"Operator '{domain}::{node.op_type}' (opset {opset_version}) is deprecated.",
                )
        except onnx.defs.SchemaError:
            if domain in official_domains:
                yield _emit(
                    _rule("ONNX019"),
                    "node",
                    node,
                    message=f"No schema found for '{domain}::{node.op_type}' at opset version {opset_version}.",
                )

    def check_value(self, value: ir.Value) -> onnx_doctor.DiagnosticsMessageIterator:
        # ONNX102: empty-value-name
        if not value.name:
            yield _emit(
                _rule("ONNX102"),
                "node",
                value,
                fix=lambda: _apply_name_fix(self._model),
            )

        # ONNX020: missing-value-type
        if value.type is None:
            yield _emit(
                _rule("ONNX020"),
                "node",
                value,
                message=f"Value '{value.name}' has no type annotation.",
            )
        elif isinstance(value.type, ir.TensorType):
            # ONNX021: undefined-value-dtype
            if value.type.dtype == ir.DataType.UNDEFINED:
                yield _emit(
                    _rule("ONNX021"),
                    "node",
                    value,
                    message=f"Value '{value.name}' has tensor type with UNDEFINED dtype.",
                )

    def check_tensor(self, tensor: ir.Tensor) -> onnx_doctor.DiagnosticsMessageIterator:
        # ONNX022: undefined-tensor-dtype
        if tensor.dtype == ir.DataType.UNDEFINED:
            yield _emit(
                _rule("ONNX022"),
                "tensor",
                tensor,
                message=f"Tensor '{tensor.name}' has UNDEFINED dtype.",
            )

        # External tensor checks (ONNX023-ONNX027)
        if isinstance(tensor, ir.ExternalTensor):
            location = tensor.location

            # ONNX024: external-tensor-empty-location
            if not location:
                yield _emit(
                    _rule("ONNX024"),
                    "tensor",
                    tensor,
                    message=f"External tensor '{tensor.name}' has empty location.",
                )
                return

            # ONNX023: external-tensor-absolute-path
            if os.path.isabs(location):
                yield _emit(
                    _rule("ONNX023"),
                    "tensor",
                    tensor,
                    message=f"External tensor '{tensor.name}' has absolute path: '{location}'.",
                )

            # ONNX025: external-tensor-outside-model-dir
            if ".." in pathlib.PurePosixPath(location).parts:
                yield _emit(
                    _rule("ONNX025"),
                    "tensor",
                    tensor,
                    message=f"External tensor '{tensor.name}' path '{location}' escapes model directory.",
                )

            # ONNX026/ONNX027: Check file accessibility if base_dir is set
            if tensor.base_dir:
                full_path = pathlib.Path(tensor.base_dir) / location
                if not full_path.exists():
                    yield _emit(
                        _rule("ONNX026"),
                        "tensor",
                        tensor,
                        message=f"External tensor '{tensor.name}' file not found: '{full_path}'.",
                    )
                elif not full_path.is_file():
                    yield _emit(
                        _rule("ONNX027"),
                        "tensor",
                        tensor,
                        message=f"External tensor '{tensor.name}' path is not a regular file: '{full_path}'.",
                    )

    def check_function(
        self, function: ir.Function
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        # ONNX028: function-empty-name
        if not function.name:
            yield _emit(_rule("ONNX028"), "function", function)

        # ONNX029: function-missing-domain
        if (
            self._ir_version is not None
            and self._ir_version >= 8
            and not function.domain
        ):
            yield _emit(_rule("ONNX029"), "function", function)

        # ONNX030: function-duplicate-inputs
        input_names = [inp.name for inp in function.inputs if inp.name]
        if len(input_names) != len(set(input_names)):
            seen: set[str] = set()
            for name in input_names:
                if name in seen:
                    yield _emit(
                        _rule("ONNX030"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate input name: '{name}'.",
                    )
                seen.add(name)

        # ONNX031: function-duplicate-outputs
        output_names = [out.name for out in function.outputs if out.name]
        if len(output_names) != len(set(output_names)):
            seen_out: set[str] = set()
            for name in output_names:
                if name in seen_out:
                    yield _emit(
                        _rule("ONNX031"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate output name: '{name}'.",
                    )
                seen_out.add(name)

        # ONNX032: function-duplicate-attributes
        attr_names: list[str] = [
            a.name for a in function.attributes if hasattr(a, "name")
        ]
        if len(attr_names) != len(set(attr_names)):
            seen_attr: set[str] = set()
            for name in attr_names:
                if name in seen_attr:
                    yield _emit(
                        _rule("ONNX032"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate attribute: '{name}'.",
                    )
                seen_attr.add(name)

        # ONNX033: unsorted-function-nodes + ONNX034: function-ssa-violation
        known: set[ir.Value] = set()
        for inp in function.inputs:
            known.add(inp)

        assigned_names: dict[str, int] = {}
        for inp in function.inputs:
            if inp.name:
                assigned_names[inp.name] = assigned_names.get(inp.name, 0) + 1

        unsorted = False
        for node in function:
            for inp in node.inputs:
                if inp is not None and inp not in known:
                    unsorted = True
            for out in node.outputs:
                known.add(out)
                if out.name:
                    assigned_names[out.name] = assigned_names.get(out.name, 0) + 1

        if unsorted:
            yield _emit(
                _rule("ONNX033"),
                "function",
                function,
                fix=function.sort,
            )

        for name, count in assigned_names.items():
            if count > 1:
                yield _emit(
                    _rule("ONNX034"),
                    "function",
                    function,
                    message=f"Value name '{name}' is assigned {count} times in function '{function.name}'.",
                )

        # ONNX035: function-opset-mismatch
        if self._opset_imports:
            for domain, version in function.opset_imports.items():
                if (
                    domain in self._opset_imports
                    and self._opset_imports[domain] != version
                ):
                    yield _emit(
                        _rule("ONNX035"),
                        "function",
                        function,
                        message=(
                            f"Function '{function.name}' imports opset {domain}:{version}, "
                            f"but model imports {domain}:{self._opset_imports[domain]}."
                        ),
                    )


def _apply_name_fix(model: ir.Model) -> None:
    """Apply NameFixPass to auto-name all values and nodes."""
    from onnx_ir.passes.common import NameFixPass  # noqa: PLC0415

    NameFixPass()(model)


def _apply_output_fix(model: ir.Model) -> None:
    """Apply OutputFixPass to insert Identity nodes for invalid outputs."""
    from onnx_ir.passes.common import OutputFixPass  # noqa: PLC0415

    OutputFixPass()(model)
