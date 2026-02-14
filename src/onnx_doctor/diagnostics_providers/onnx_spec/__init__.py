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

    def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        """Analyze the model for ONNX spec compliance issues."""
        # Store model-level state
        ir_version = model.ir_version
        opset_imports = dict(model.opset_imports)

        # Model-level checks
        yield from self._check_model(model, ir_version, opset_imports)

        # Check all functions
        for func in model.functions.values():
            yield from self._check_function(func, ir_version, opset_imports)
            yield from self._check_function_contents(func, model, opset_imports)

        # Check main graph (root graph)
        yield from self._check_graph(model.graph, model, opset_imports, is_root=True)

        # Whole-model shadowing analysis
        yield from self._analyze_shadowing(model)

    def _check_function_contents(
        self,
        func: ir.Function,
        model: ir.Model,
        opset_imports: dict[str, int],
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check values, nodes, and attributes inside a function.

        Mirrors the graph traversal to ensure function contents get the same
        checks as graph contents (ONNX018/ONNX103 for values, ONNX020-ONNX025
        for tensors, and recursive subgraph checks).
        """
        # Check function inputs
        for value in func.inputs:
            yield from self._check_value(value, model)

        # Check function outputs
        for value in func.outputs:
            yield from self._check_value(value, model)

        # Check nodes and their outputs/attributes
        for node in func:
            yield from self._check_node(node, opset_imports)
            for out in node.outputs:
                yield from self._check_value(out, model)
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.TENSOR:
                    yield from self._check_tensor(attr.value, node)
                elif attr.type == ir.AttributeType.TENSORS:
                    for tensor in attr.value:
                        yield from self._check_tensor(tensor, node)
                elif attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_graph(
                        attr.value, model, opset_imports, in_function=True
                    )
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_graph(
                            subgraph, model, opset_imports, in_function=True
                        )

    def _check_model(
        self,
        model: ir.Model,
        ir_version: int | None,
        opset_imports: dict[str, int],
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check model-level properties."""
        # ONNX010: invalid-ir-version
        if ir_version is None or ir_version < 1:
            yield _emit(
                _rule("ONNX010"),
                "model",
                model,
                message=f"Model ir_version is {ir_version!r}, which is invalid.",
            )

        # ONNX011: ir-version-too-new
        if ir_version is not None and ir_version > _MAX_IR_VERSION:
            yield _emit(
                _rule("ONNX011"),
                "model",
                model,
                message=f"Model ir_version {ir_version} is newer than the checker supports (max {_MAX_IR_VERSION}).",
            )

        # ONNX012: duplicate-metadata-keys
        if model.metadata_props is not None:
            seen_keys: set[str] = set()
            for key in model.metadata_props:
                if key in seen_keys:
                    yield _emit(
                        _rule("ONNX012"),
                        "model",
                        model,
                        message=f"Duplicate metadata key: '{key}'.",
                    )
                seen_keys.add(key)

        # ONNX013: missing-default-opset
        if ir_version is not None and ir_version >= 3 and "" not in opset_imports:
            yield _emit(_rule("ONNX013"), "model", model)

        # ONNX014: unexpected-opset-import
        if ir_version is not None and ir_version < 3 and len(opset_imports) > 0:
            yield _emit(_rule("ONNX014"), "model", model)

    def _check_graph(
        self,
        graph: ir.Graph,
        model: ir.Model,
        opset_imports: dict[str, int],
        is_root: bool = False,
        in_function: bool = False,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a graph and its contents recursively."""
        # ONNX001: empty-graph-name (root graph only)
        if is_root and not graph.name:
            yield _emit(
                _rule("ONNX001"),
                "graph",
                graph,
                fix=lambda: setattr(graph, "name", "main_graph"),
            )

        # ONNX034/ONNX035: input type/shape, ONNX036/ONNX037: output type/shape (root only)
        if is_root:
            for value in graph.inputs:
                if value.type is None:
                    yield _emit(
                        _rule("ONNX034"),
                        "graph",
                        graph,
                        message=f"Graph input '{value.name}' is missing type information.",
                    )
                elif isinstance(value.type, ir.TensorType) and value.shape is None:
                    # Only check shape for TensorType (SequenceType etc. don't have shapes)
                    yield _emit(
                        _rule("ONNX035"),
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
                elif isinstance(value.type, ir.TensorType) and value.shape is None:
                    # Only check shape for TensorType (SequenceType etc. don't have shapes)
                    yield _emit(
                        _rule("ONNX037"),
                        "graph",
                        graph,
                        message=f"Graph output '{value.name}' is missing shape information.",
                    )

        # ONNX101: duplicate-graph-input
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

        # ONNX102: duplicate-graph-output
        seen.clear()
        for value in graph.outputs:
            if value in seen:
                yield _emit(
                    _rule("ONNX102"),
                    "graph",
                    graph,
                    message=f"Duplicate Value object in graph outputs: '{value.name}'.",
                    fix=lambda: _apply_output_fix(model),
                )
            seen.add(value)

        # ONNX002: empty-initializer-name
        for tensor in graph.initializers.values():
            if not tensor.name:
                yield _emit(
                    _rule("ONNX002"),
                    "graph",
                    graph,
                    fix=lambda: _apply_name_fix(model),
                )

        # Collect known values for topological order check
        known_values: set[ir.Value] = set()
        for inp in graph.inputs:
            known_values.add(inp)
        for init in graph.initializers.values():
            known_values.add(init)

        # ONNX003: unsorted-graph-nodes + ONNX004: unknown-node-input
        is_sorted = True
        for node in graph:
            for inp in node.inputs:
                if inp is None:
                    continue
                if inp not in known_values:
                    producer = inp.producer()
                    if producer is not None and producer.graph is graph:
                        is_sorted = False
                    elif (
                        producer is None
                        and not inp.is_graph_input()
                        and not inp.is_initializer()
                    ):
                        is_sorted = False
                        yield _emit(
                            _rule("ONNX004"),
                            "graph",
                            graph,
                            message=f"Node '{node.op_type}' has input '{inp.name}' not produced by any node or graph input.",
                        )
            for out in node.outputs:
                known_values.add(out)

        if not is_sorted:
            yield _emit(
                _rule("ONNX003"),
                "graph",
                graph,
                fix=graph.sort,
            )

        # ONNX005: experimental-op
        for node in graph:
            try:
                domain = node.domain if node.domain else ""
                opset_version = opset_imports.get(domain)
                if opset_version is not None:
                    schema = onnx.defs.get_schema(node.op_type, opset_version, domain)
                    if (
                        schema.support_level
                        == onnx.defs.OpSchema.SupportType.EXPERIMENTAL
                    ):
                        yield _emit(
                            _rule("ONNX005"),
                            "graph",
                            graph,
                            message=f"Node '{node.op_type}' uses experimental operator '{domain}::{node.op_type}'.",
                        )
            except Exception:  # noqa: PERF203
                pass

        # ONNX006: duplicate-value-name + ONNX008: graph-ssa-violation
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
                    _rule("ONNX006"),
                    "graph",
                    graph,
                    message=f"Value name '{name}' appears {count} times in the graph.",
                )
                yield _emit(
                    _rule("ONNX008"),
                    "graph",
                    graph,
                    message=f"Value name '{name}' is assigned {count} times, violating SSA form.",
                )

        # ONNX007: graph-output-not-produced
        for out in graph.outputs:
            producer = out.producer()
            if (
                producer is None
                and not out.is_graph_input()
                and not out.is_initializer()
            ):
                yield _emit(
                    _rule("ONNX007"),
                    "graph",
                    graph,
                    message=f"Graph output '{out.name}' is not produced by any node in the graph.",
                )
            elif producer is not None and producer.graph is not graph:
                yield _emit(
                    _rule("ONNX007"),
                    "graph",
                    graph,
                    message=f"Graph output '{out.name}' is produced in a different graph.",
                    suggestion="Apply `--fix` to insert Identity nodes with `OutputFixPass`.",
                    fix=lambda: _apply_output_fix(model),
                )

        # ONNX009: initializer-name-conflict
        init_names = set(graph.initializers.keys())
        for node in graph:
            for attr in node.attributes.values():
                if hasattr(attr, "type") and attr.type == ir.AttributeType.GRAPH:
                    subgraph = attr.value
                    for sub_input in subgraph.inputs:
                        if sub_input.name and sub_input.name in init_names:
                            yield _emit(
                                _rule("ONNX009"),
                                "graph",
                                graph,
                                message=f"Initializer '{sub_input.name}' conflicts with subgraph input name.",
                            )

        # Check nodes, values, tensors, and subgraphs
        for value in graph.inputs:
            yield from self._check_value(value, model)

        for node in graph:
            yield from self._check_node(node, opset_imports)
            for out in node.outputs:
                yield from self._check_value(out, model)
            for attr in node.attributes.values():
                # ONNX039: ref-attr-outside-function
                if attr.is_ref() and not in_function:
                    yield _emit(
                        _rule("ONNX039"),
                        "node",
                        node,
                        message=f"Attribute '{attr.name}' is a reference attribute, which is only valid inside functions.",
                    )
                if attr.type == ir.AttributeType.TENSOR:
                    yield from self._check_tensor(attr.value, node)
                elif attr.type == ir.AttributeType.TENSORS:
                    for tensor in attr.value:
                        yield from self._check_tensor(tensor, node)
                elif attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_graph(
                        attr.value, model, opset_imports, in_function=in_function
                    )
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_graph(
                            subgraph, model, opset_imports, in_function=in_function
                        )

        for initializer in graph.initializers.values():
            # ONNX104: initializer-missing-const-value
            if initializer.const_value is None:
                yield _emit(
                    _rule("ONNX104"),
                    "graph",
                    graph,
                    message=f"Initializer '{initializer.name}' has no const_value set.",
                )
            else:
                yield from self._check_tensor(initializer.const_value, initializer)

    def _check_node(
        self,
        node: ir.Node,
        opset_imports: dict[str, int],
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a single node."""
        domain = node.domain if node.domain else ""

        # ONNX015: missing-opset-for-domain
        if domain not in opset_imports:
            yield _emit(
                _rule("ONNX015"),
                "node",
                node,
                message=f"No opset imported for domain '{domain}' used by node '{node.op_type}'.",
            )
            return

        opset_version = opset_imports[domain]

        # ONNX016: deprecated-op + ONNX017: unregistered-op
        official_domains = {"", "ai.onnx", "ai.onnx.ml"}
        try:
            schema = onnx.defs.get_schema(node.op_type, opset_version, domain)
            if schema.deprecated:
                yield _emit(
                    _rule("ONNX016"),
                    "node",
                    node,
                    message=f"Operator '{domain}::{node.op_type}' (opset {opset_version}) is deprecated.",
                )
        except onnx.defs.SchemaError:
            if domain in official_domains:
                yield _emit(
                    _rule("ONNX017"),
                    "node",
                    node,
                    message=f"No schema found for '{domain}::{node.op_type}' at opset version {opset_version}.",
                )

    def _check_value(
        self,
        value: ir.Value,
        model: ir.Model,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a single value."""
        # ONNX103: empty-value-name
        # Only check if value has consumers OR is a graph input/output/initializer
        needs_name = (
            value.uses()
            or value.is_graph_input()
            or value.is_graph_output()
            or value.is_initializer()
        )
        if not value.name and needs_name:
            yield _emit(
                _rule("ONNX103"),
                "value",
                value,
                fix=lambda: _apply_name_fix(model),
            )

        # ONNX018: missing-value-type
        if value.type is None:
            yield _emit(
                _rule("ONNX018"),
                "node",
                value,
                message=f"Value '{value.name}' has no type annotation.",
            )
        elif isinstance(value.type, ir.TensorType):
            # ONNX019: undefined-value-dtype
            if value.type.dtype == ir.DataType.UNDEFINED:
                yield _emit(
                    _rule("ONNX019"),
                    "node",
                    value,
                    message=f"Value '{value.name}' has tensor type with UNDEFINED dtype.",
                )

    def _check_tensor(
        self,
        tensor: ir.Tensor,
        container: ir.Node | ir.Value,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a single tensor.

        Args:
            tensor: The tensor to check.
            container: The containing Node (for attribute tensors) or Value (for initializers).
        """
        # Determine target_type based on container
        target_type: onnx_doctor._message.PossibleTargetTypes = (
            "node" if isinstance(container, ir.Node) else "value"
        )

        # ONNX020: undefined-tensor-dtype
        if tensor.dtype == ir.DataType.UNDEFINED:
            yield _emit(
                _rule("ONNX020"),
                target_type,
                container,
                message=f"Tensor '{tensor.name}' has UNDEFINED dtype.",
            )

        # External tensor checks (ONNX021-ONNX025)
        if isinstance(tensor, ir.ExternalTensor):
            location = tensor.location

            # ONNX022: external-tensor-empty-location
            if not location:
                yield _emit(
                    _rule("ONNX022"),
                    target_type,
                    container,
                    message=f"External tensor '{tensor.name}' has empty location.",
                )
                return

            # ONNX021: external-tensor-absolute-path
            if os.path.isabs(location):
                yield _emit(
                    _rule("ONNX021"),
                    target_type,
                    container,
                    message=f"External tensor '{tensor.name}' has absolute path: '{location}'.",
                )

            # ONNX023: external-tensor-outside-model-dir
            if ".." in pathlib.PurePosixPath(location).parts:
                yield _emit(
                    _rule("ONNX023"),
                    target_type,
                    container,
                    message=f"External tensor '{tensor.name}' path '{location}' escapes model directory.",
                )

            # ONNX024/ONNX025: Check file accessibility if base_dir is set
            if tensor.base_dir:
                full_path = pathlib.Path(tensor.base_dir) / location
                if not full_path.exists():
                    yield _emit(
                        _rule("ONNX024"),
                        target_type,
                        container,
                        message=f"External tensor '{tensor.name}' file not found: '{full_path}'.",
                    )
                elif not full_path.is_file():
                    yield _emit(
                        _rule("ONNX025"),
                        target_type,
                        container,
                        message=f"External tensor '{tensor.name}' path is not a regular file: '{full_path}'.",
                    )

    def _check_function(
        self,
        function: ir.Function,
        ir_version: int | None,
        opset_imports: dict[str, int],
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check a single function."""
        # ONNX026: function-empty-name
        if not function.name:
            yield _emit(_rule("ONNX026"), "function", function)

        # ONNX027: function-missing-domain
        if ir_version is not None and ir_version >= 8 and not function.domain:
            yield _emit(_rule("ONNX027"), "function", function)

        # ONNX028: function-duplicate-inputs
        input_names = [inp.name for inp in function.inputs if inp.name]
        if len(input_names) != len(set(input_names)):
            seen: set[str] = set()
            for name in input_names:
                if name in seen:
                    yield _emit(
                        _rule("ONNX028"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate input name: '{name}'.",
                    )
                seen.add(name)

        # ONNX029: function-duplicate-outputs
        output_names = [out.name for out in function.outputs if out.name]
        if len(output_names) != len(set(output_names)):
            seen_out: set[str] = set()
            for name in output_names:
                if name in seen_out:
                    yield _emit(
                        _rule("ONNX029"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate output name: '{name}'.",
                    )
                seen_out.add(name)

        # ONNX030: function-duplicate-attributes
        attr_names: list[str] = [
            a.name for a in function.attributes if hasattr(a, "name")
        ]
        if len(attr_names) != len(set(attr_names)):
            seen_attr: set[str] = set()
            for name in attr_names:
                if name in seen_attr:
                    yield _emit(
                        _rule("ONNX030"),
                        "function",
                        function,
                        message=f"Function '{function.name}' has duplicate attribute: '{name}'.",
                    )
                seen_attr.add(name)

        # ONNX031: unsorted-function-nodes + ONNX032: function-ssa-violation
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
                _rule("ONNX031"),
                "function",
                function,
                fix=function.sort,
            )

        for name, count in assigned_names.items():
            if count > 1:
                yield _emit(
                    _rule("ONNX032"),
                    "function",
                    function,
                    message=f"Value name '{name}' is assigned {count} times in function '{function.name}'.",
                )

        # ONNX033: function-opset-mismatch
        if opset_imports:
            for domain, version in function.opset_imports.items():
                if domain in opset_imports and opset_imports[domain] != version:
                    yield _emit(
                        _rule("ONNX033"),
                        "function",
                        function,
                        message=(
                            f"Function '{function.name}' imports opset {domain}:{version}, "
                            f"but model imports {domain}:{opset_imports[domain]}."
                        ),
                    )

    def _analyze_shadowing(
        self,
        model: ir.Model,
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Check for variable shadowing across scopes."""
        # ONNX038: subgraph-variable-shadowing
        yield from self._check_shadowing(model.graph, frozenset())
        for func in model.functions.values():
            func_names = _collect_scope_names_function(func)
            for node in func:
                for attr in node.attributes.values():
                    if attr.type == ir.AttributeType.GRAPH:
                        yield from self._check_shadowing(attr.value, func_names)
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for subgraph in attr.value:
                            yield from self._check_shadowing(subgraph, func_names)

    def _check_shadowing(
        self,
        graph: ir.Graph,
        outer_names: frozenset[str],
    ) -> onnx_doctor.DiagnosticsMessageIterator:
        """Recursively check that subgraph names don't shadow outer scope names."""
        local_names = _collect_scope_names_graph(graph)
        shadowed = local_names & outer_names
        for name in sorted(shadowed):
            yield _emit(
                _rule("ONNX038"),
                "graph",
                graph,
                message=f"Subgraph value name '{name}' shadows a name from an outer scope.",
            )
        visible = outer_names | local_names
        for node in graph:
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_shadowing(attr.value, visible)
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_shadowing(subgraph, visible)


def _collect_scope_names_graph(graph: ir.Graph) -> frozenset[str]:
    """Collect all value names defined in a graph scope."""
    names: set[str] = set()
    for v in graph.inputs:
        if v.name:
            names.add(v.name)
    for name in graph.initializers:
        if name:
            names.add(name)
    for node in graph:
        for v in node.outputs:
            if v.name:
                names.add(v.name)
    return frozenset(names)


def _collect_scope_names_function(func: ir.Function) -> frozenset[str]:
    """Collect all value names defined in a function scope."""
    names: set[str] = set()
    for v in func.inputs:
        if v.name:
            names.add(v.name)
    for node in func:
        for v in node.outputs:
            if v.name:
                names.add(v.name)
    return frozenset(names)


def _apply_name_fix(model: ir.Model) -> None:
    """Apply NameFixPass to auto-name all values and nodes."""
    from onnx_ir.passes.common import NameFixPass  # noqa: PLC0415

    NameFixPass()(model)


def _apply_output_fix(model: ir.Model) -> None:
    """Apply OutputFixPass to insert Identity nodes for invalid outputs."""
    from onnx_ir.passes.common import OutputFixPass  # noqa: PLC0415

    OutputFixPass()(model)
