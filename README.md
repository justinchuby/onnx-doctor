# ONNX Doctor

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-doctor.svg)](https://pypi.org/project/onnx-doctor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-doctor.svg)](https://pypi.org/project/onnx-doctor)
[![Documentation](https://readthedocs.org/projects/onnx-doctor/badge/?version=latest)](https://onnx-doctor.readthedocs.io)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An extensible linter for ONNX models — like [ruff](https://github.com/astral-sh/ruff), but for ONNX. Catch spec violations, compatibility issues, and common pitfalls with clear messages and actionable suggestions.

## Installation

```bash
pip install onnx-doctor
```

## Quick Start

### CLI

Check a model for issues:

```bash
onnx-doctor check model.onnx
```

Example output:

```
model.onnx:graph: ONNX001 Graph name is empty.
  suggestion: Set the name of the graph, e.g. `graph.name = 'main_graph'`.

model.onnx:graph:node/5(MyCustomOp): ONNX019 No schema found for '::MyCustomOp' at opset version 21.
  suggestion: Verify the operator name, domain, and imported opset version.

model.onnx:graph:node/5(MyCustomOp): ONNX020 Value 'custom_out' has no type annotation.
  suggestion: Run shape inference on the model, e.g. `onnx.shape_inference.infer_shapes(model)`.

Found 2 errors, 1 warning.
```

The location path shows where the issue is in the graph hierarchy. For nodes,
it includes the node index and name (e.g. `node/5(MyCustomOp)`). For values,
it shows the producer node. For subgraphs, the full path is shown:

```
model.onnx:graph:node/3(If_0):then_branch:node/1(Add): ONNX019 ...
```

### Programmatic API

```python
import onnx_ir as ir
import onnx_doctor
from onnx_doctor.diagnostics_providers import OnnxSpecProvider

model = ir.load("model.onnx")
messages = onnx_doctor.diagnose(model, [OnnxSpecProvider()])

for msg in messages:
    print(f"[{msg.severity}] {msg.error_code}: {msg.message}")
```

## CLI Reference

```
onnx-doctor <command> [options]
```

### `check` — Lint an ONNX model

```bash
onnx-doctor check model.onnx [options]
```

| Option | Description |
|--------|-------------|
| `--select CODE [...]` | Only report rules matching these codes or prefixes (e.g. `ONNX001`, `ONNX`). |
| `--ignore CODE [...]` | Ignore rules matching these codes or prefixes. |
| `--output-format {text,json,github}` | Output format. Default: `text`. |
| `--severity {error,warning,info}` | Minimum severity to report. |
| `--fix` | Apply available fixes and save the model. |
| `-o, --output PATH` | Output path for the fixed model (default: overwrite input). Only used with `--fix`. |
| `--diff` | Show a unified diff of what `--fix` would change, without writing. |
| `--ort` | Enable ONNX Runtime compatibility checks (ORT rules). |
| `--ort-provider NAME` | Execution provider for ORT checks (default: `CPUExecutionProvider`). |

**Exit codes:** `0` = no errors (warnings may be present), `1` = errors found.

Examples:

```bash
# Ignore ir-version warnings
onnx-doctor check model.onnx --ignore ONNX013

# Only show errors
onnx-doctor check model.onnx --severity error

# Apply auto-fixes in place
onnx-doctor check model.onnx --fix

# Apply auto-fixes to a new file
onnx-doctor check model.onnx --fix -o fixed_model.onnx

# Preview what --fix would change
onnx-doctor check model.onnx --diff

# Enable ORT compatibility checks
onnx-doctor check model.onnx --ort

# JSON output for CI
onnx-doctor check model.onnx --output-format json

# GitHub Actions annotations
onnx-doctor check model.onnx --output-format github
```

JSON output example:

```json
[
  {
    "file": "model.onnx",
    "code": "ONNX001",
    "severity": "error",
    "message": "Graph name is empty.",
    "target_type": "graph",
    "location": "graph",
    "rule_name": "empty-graph-name",
    "suggestion": "Set the name of the graph, e.g. `graph.name = 'main_graph'`."
  }
]
```

### `explain` — Show rule details

```bash
onnx-doctor explain ONNX001
```

```
ONNX001: empty-graph-name

  Message: Graph name is empty.
  Severity: error
  Category: spec
  Target: graph

  Suggestion: Set the name of the graph, e.g. `graph.name = 'main_graph'`.

  ## Details
  The 'name' field of a graph must not be empty per the ONNX spec.
```

You can also look up rules by name:

```bash
onnx-doctor explain empty-graph-name
```

### `list-rules` — Show all available rules

```bash
onnx-doctor list-rules
```

## Rules

ONNX Doctor ships with **61 built-in rules** across five providers:

| Prefix | Provider | Description |
|--------|----------|-------------|
| `ONNX` | ONNX Spec | 40 rules for ONNX spec compliance (graph, model, node, value, tensor, function) |
| `PB` | Protobuf | 13 rules for protobuf-specific issues |
| `SIM` | Simplification | 3 rules for removing unused elements (functions, opsets, nodes) |
| `ORT` | ORT Compatibility | 5 rules for ONNX Runtime compatibility checks (opt-in via `--ort`) |
| `SP` | Sparsity | Tensor sparsity analysis (example provider, not enabled by default) |

## Writing Custom Providers

Create your own rules by subclassing `DiagnosticsProvider`:

```python
import onnx_ir as ir
import onnx_doctor

class MyProvider(onnx_doctor.DiagnosticsProvider):
    def check_graph(self, graph: ir.Graph):
        node_count = sum(1 for _ in graph.all_nodes())
        if node_count > 1000:
            yield onnx_doctor.DiagnosticsMessage(
                target_type="graph",
                target=graph,
                message=f"Graph has {node_count} nodes — consider optimizing.",
                severity="warning",
                producer="MyProvider",
                error_code="CUSTOM001",
            )

model = ir.load("model.onnx")
messages = onnx_doctor.diagnose(model, [MyProvider()])
```

## License

[MIT](LICENSE)
