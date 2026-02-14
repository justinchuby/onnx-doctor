---
name: onnx-doctor
description: Development conventions for the onnx-doctor ONNX model linter. Use this when adding rules, writing providers, or modifying the linter codebase.
---

# ONNX Doctor — Development Skills

## Package Structure

- **Package**: `onnx_doctor` at `src/onnx_doctor/`
- **Core modules**: `_rule.py`, `_rule_registry.py`, `_loader.py`, `_checker.py`, `_cli.py`, `_formatter.py`, `_message.py`, `_diagnostics.py`
- **Providers**: `diagnostics_providers/` — each provider is a subpackage or module
- **Tests**: `src/onnx_doctor/tests/`
- **Docs**: `docs/` (Sphinx + MyST markdown, furo theme)

## Rule Numbering Convention

| Prefix | Code Range | Category | Description |
|--------|-----------|----------|-------------|
| `ONNX` | 001–099 | `spec` | ONNX spec compliance rules |
| `ONNX` | 101–199 | `ir` | IR-specific rules (issues unique to `onnx_ir`) |
| `PB` | 001+ | `protobuf` | Protobuf-specific rules |
| `SIM` | 001+ | `spec` | Simplification / dead code elimination rules |
| `ORT` | 001+ | `spec` | ONNX Runtime compatibility |
| `SP` | 001+ | `spec` | Sparsity analysis |

- Spec rules that apply to both protobuf and IR use `ONNX001`–`ONNX099`.
- IR-only rules (e.g., duplicate `Value` object identity) use `ONNX101`+.
- Protobuf-only rules (impossible in IR by construction) use `PB` prefix.
- Simplification rules (unused functions/nodes/opsets) use `SIM` prefix.

## Adding a New Rule

1. **Define in YAML** (`spec.yaml` or provider-specific YAML):

   ```yaml
   - code: ONNX036
     name: kebab-case-name
     category: spec
     severity: error
     message: Short description of the issue.
     suggestion: How to fix it.
     explanation: |
       ## Details
       Extended markdown explanation.
   ```

2. **Implement check in the provider** (e.g., `onnx_spec/__init__.py`):

   ```python
   # ONNX036: kebab-case-name
   if condition:
       yield _emit(_rule("ONNX036"), "node", node, message=f"...")
   ```

3. **Add a test** in `tests/test_onnx_spec_provider.py`:

   ```python
   def test_kebab_case_name(self):
       model = _make_model(...)
       messages = _diagnose(model)
       self.assertIn("ONNX036", _codes(messages))
   ```

## Build & Test

```bash
pip install -e .                              # Editable install
pip install -r requirements-dev.txt           # Dev dependencies
python -m pytest src/onnx_doctor/tests/       # Run tests
ruff check src/                               # Lint
ruff format src/                              # Format
onnx-doctor check model.onnx                  # CLI
```

## Code Style

- **Every `.py` file** must start with `from __future__ import annotations`.
- Google-style docstrings. Target Python 3.9.
- Ruff enforced (see `pyproject.toml` for full config).
- Private modules prefixed with `_` (e.g., `_rule.py`, `_checker.py`).

## Key Dependencies

- **`onnx_ir`**: The linter operates on IR objects (`ir.Model`, `ir.Graph`, etc.), **not** protobuf directly.
- **`onnx`**: Used for op schema lookups (`onnx.defs.get_schema`).
- **`pyyaml`**: Rule definitions loaded from YAML files.
- **`rich`**: CLI output formatting.

## Architecture Notes

- `_checker.py` provides the `diagnose(model, providers)` entry point that calls each provider's `diagnose(model)` method.
- Each provider is responsible for walking the model structure as needed (using `ir.traversal.RecursiveGraphIterator` or manual iteration).
- **Location inference**: The driver builds a location map by walking the model once. For messages without a `location` set, it infers the location from the `target` object (e.g., `graph:node/3(MatMul)`).
- `_loader.py` has a lazy singleton `get_default_registry()` that loads all YAML rule files on first access.
- Providers yield `DiagnosticsMessage` objects with `target` and `target_type` fields for context.

## Autofix Architecture

- `Fix = Callable[[], None]` — a no-arg callable that mutates the IR in place. Stored on `DiagnosticsMessage.fix`.
- Rules marked `fixable: true` in YAML should attach a `fix` callable via the `_emit()` helper.
- CLI `--fix` applies all fixes, saves the model, then re-diagnoses to show remaining issues.
- CLI `--diff` shows a unified diff of what `--fix` would change, without writing.
- Fix deduplication: `_apply_fixes()` deduplicates by callable identity (`id(fix)`) to avoid running the same pass multiple times.

### Available IR Passes for Fixes

From `onnx_ir.passes.common` (all take `model: ir.Model`, return `PassResult`):

| Pass | Used by | Description |
|------|---------|-------------|
| `NameFixPass` | ONNX003, ONNX103 | Auto-names all unnamed values and nodes |
| `OutputFixPass` | ONNX009 | Inserts Identity nodes for invalid output configurations |
| `RemoveUnusedFunctionsPass` | SIM001 | Removes unreferenced functions |
| `RemoveUnusedNodesPass` | SIM003 | Removes dead nodes and unused initializers |
| `RemoveUnusedOpsetsPass` | SIM002 | Removes unused opset imports |

### Adding a Fixable Rule

1. Mark `fixable: true` in YAML.
2. In the provider, pass `fix=` to `_emit()`:

   ```python
   yield _emit(
       _rule("ONNX004"), "graph", graph,
       fix=graph.sort,
   )
   ```

3. For model-level passes, capture the model in a closure:

   ```python
   yield _emit(
       _rule("ONNX003"), "graph", graph,
       fix=lambda: _apply_name_fix(model),
   )
   ```

## Provider Structure

Providers implement a single `diagnose(model: ir.Model)` method that yields `DiagnosticsMessage` objects. Each provider is responsible for its own traversal strategy.

| Provider | Module | Rules | Notes |
|----------|--------|-------|-------|
| `OnnxSpecProvider` | `diagnostics_providers/onnx_spec/` | ONNX001–ONNX103 (YAML) | Default, always enabled |
| (protobuf rules) | `diagnostics_providers/onnx_spec/` | PB001–PB013 (YAML) | Registered but no Python checker yet |
| `SimplificationProvider` | `diagnostics_providers/simplification/` | SIM001–SIM003 (YAML) | Default, always enabled |
| `OnnxRuntimeCompatibilityLinter` | `diagnostics_providers/onnxruntime_compatibility/` | ORT001–ORT005 (Python) | Opt-in via `--ort` flag |
| `SparsityAnalyzer` | `diagnostics_providers/sparsity.py` | SP001 (Python) | Example provider, not registered |

### Example Provider Implementation

```python
class MyProvider(onnx_doctor.DiagnosticsProvider):
    def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        # Model-level checks
        if not model.graph.name:
            yield _emit(_rule("MYRULE001"), "graph", model.graph)

        # Walk all nodes (including subgraphs)
        for node in ir.traversal.RecursiveGraphIterator(model.graph):
            if some_condition(node):
                yield _emit(_rule("MYRULE002"), "node", node)

        # Check functions
        for func in model.functions.values():
            yield from self._check_function(func)
```
