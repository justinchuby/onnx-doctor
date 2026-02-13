# Getting Started

**ONNX Doctor** is an extensible linter for [ONNX](https://onnx.ai) models — like [ruff](https://github.com/astral-sh/ruff), but for ONNX. It catches spec violations, compatibility issues, and common pitfalls with clear messages and actionable suggestions.

- Provider-based architecture with pluggable rule sets
- Auto-fix support for common issues
- Rich CLI output with JSON and GitHub Actions formats

📦 [GitHub Repository](https://github.com/justinchuby/onnx-doctor)

## Installation

```bash
pip install onnx-doctor
```

## Quick Start

Check a model from the command line:

```bash
onnx-doctor check model.onnx
```

Or use the programmatic API:

```python
import onnx_ir as ir
import onnx_doctor
from onnx_doctor.diagnostics_providers import OnnxSpecProvider

model = ir.load("model.onnx")
messages = onnx_doctor.diagnose(model, [OnnxSpecProvider()])

for msg in messages:
    print(f"[{msg.severity}] {msg.error_code}: {msg.message}")
```
