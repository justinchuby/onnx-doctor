# Copilot Instructions for onnx-doctor

## Build, Test, and Lint

```bash
pip install -e .                        # Editable install
pip install -r requirements-dev.txt     # Dev dependencies (pytest, sphinx, etc.)
python -m pytest src/onnx_doctor/tests/ # Run all tests
python -m pytest src/onnx_doctor/tests/test_onnx_spec_provider.py           # Single test file
python -m pytest src/onnx_doctor/tests/test_onnx_spec_provider.py -k test_empty_graph_name  # Single test
ruff check src/                         # Lint
ruff format src/                        # Format
```

## Architecture

onnx-doctor is an extensible linter for ONNX models. It operates on `onnx_ir` IR objects (`ir.Model`, `ir.Graph`, `ir.Node`, `ir.Value`, etc.) — never on protobuf directly.

**Core flow:**

1. `_cli.py` parses args, loads the model via `ir.load()`, and calls `diagnose()`.
2. `_checker.py::diagnose(model, providers)` builds a location map by walking the model once, then calls each provider's `diagnose(model)` method and infers locations for returned messages.
3. Each **provider** (`DiagnosticsProvider` subclass) implements `diagnose(model: ir.Model)` and yields `DiagnosticsMessage` objects. Providers are responsible for their own traversal strategy.
4. `_formatter.py` renders messages to text/JSON/GitHub annotations.

**Rule system:**

- Rules are defined in YAML files (e.g., `diagnostics_providers/onnx_spec/spec.yaml`) and loaded by `_loader.py` into a `RuleRegistry` singleton.
- Each `Rule` dataclass has: `code`, `name` (kebab-case), `message`, `default_severity`, `category`, `target_type`, and optionally `explanation`, `suggestion`, `fixable`.
- Providers look up rules from the registry and yield messages via an `_emit()` helper.

**Autofix:**

- `Fix = Callable[[], None]` — a no-arg callable that mutates the IR in place, stored on `DiagnosticsMessage.fix`.
- Rules marked `fixable: true` in YAML attach a fix via `_emit()`. CLI `--fix` applies all fixes, `--diff` previews them.
- Fix deduplication uses callable identity (`id(fix)`) to avoid running the same pass twice.

**Providers:**

| Provider | Location | Prefixes | Notes |
|----------|----------|----------|-------|
| `OnnxSpecProvider` | `diagnostics_providers/onnx_spec/` | ONNX, PB | Default, always enabled |
| `SimplificationProvider` | `diagnostics_providers/simplification/` | SIM | Default, always enabled |
| `OnnxRuntimeCompatibilityLinter` | `diagnostics_providers/onnxruntime_compatibility/` | ORT | Opt-in via `--ort` |
| `SparsityAnalyzer` | `diagnostics_providers/sparsity.py` | SP | Example, not registered |

## Key Conventions

### Code style

- Every `.py` file must start with `from __future__ import annotations` (enforced by ruff isort `required-imports`).
- Google-style docstrings. Target Python 3.9.
- Private modules are prefixed with `_` (e.g., `_checker.py`, `_rule.py`).
- No relative imports (ruff `TID252`).

### Rule numbering

| Prefix | Range | Scope |
|--------|-------|-------|
| `ONNX` | 001–099 | Spec rules (both protobuf and IR) |
| `ONNX` | 101–199 | IR-only rules |
| `PB` | 001+ | Protobuf-only rules |
| `SIM` | 001+ | Simplification (dead code) |
| `ORT` | 001+ | ONNX Runtime compatibility |
| `SP` | 001+ | Sparsity analysis |

### Adding a new rule

1. Define in YAML (provider's `.yaml` file) with `code`, `name` (kebab-case), `category`, `severity`, `message`, `suggestion`.
2. Implement the check in the provider, yielding via `_emit(_rule("CODE"), target_type, target, message=...)`.
3. Add a test in `src/onnx_doctor/tests/`.

### IR API notes

- `onnx_ir` `Value` objects are hashable — use them directly in sets/dicts (no need for `id()`).
- Use `ir.traversal.RecursiveGraphIterator(graph)` to walk all nodes including subgraphs.
- Available IR passes for fixes: `NameFixPass`, `OutputFixPass`, `RemoveUnusedFunctionsPass`, `RemoveUnusedNodesPass`, `RemoveUnusedOpsetsPass` (all from `onnx_ir.passes.common`).
