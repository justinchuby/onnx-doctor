# CLI Reference

## Usage

```bash
onnx-doctor <command> [options]
```

## `check` — Lint an ONNX model

```bash
onnx-doctor check model.onnx [options]
```

| Option | Description |
|--------|-------------|
| `--select CODE [...]` | Only report rules matching these codes or prefixes (e.g. `ONNX001`, `ONNX`). |
| `--ignore CODE [...]` | Ignore rules matching these codes or prefixes. |
| `--output-format {text,json,github}` | Output format. Default: `text`. |
| `--severity {error,warning,info}` | Minimum severity to report. |
| `--fix` | Apply available auto-fixes and save the model. |
| `-o, --output PATH` | Output path for the fixed model (default: overwrite input). Only used with `--fix`. |

**Exit codes:** `0` = no errors (warnings may be present), `1` = errors found.

### Examples

```bash
# Ignore ir-version warnings
onnx-doctor check model.onnx --ignore ONNX013

# Only show errors
onnx-doctor check model.onnx --severity error

# Apply auto-fixes in place
onnx-doctor check model.onnx --fix

# Apply auto-fixes to a new file
onnx-doctor check model.onnx --fix -o fixed_model.onnx

# JSON output for CI
onnx-doctor check model.onnx --output-format json

# GitHub Actions annotations
onnx-doctor check model.onnx --output-format github
```

### JSON output

```json
[
  {
    "file": "model.onnx",
    "code": "ONNX001",
    "severity": "error",
    "message": "Graph name is empty.",
    "target_type": "graph",
    "rule_name": "empty-graph-name",
    "suggestion": "Set the name of the graph, e.g. `graph.name = 'main_graph'`."
  }
]
```

## `explain` — Show rule details

```bash
onnx-doctor explain ONNX001
```

You can also look up rules by name:

```bash
onnx-doctor explain empty-graph-name
```

## `list-rules` — Show all available rules

```bash
onnx-doctor list-rules
```
