"""Output formatters for diagnostics messages."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.text import Text

from onnx_doctor._message import DiagnosticsMessage


class TextFormatter:
    """Ruff-like concise text output with color."""

    def __init__(self, file_path: str = "<model>") -> None:
        self._file_path = file_path
        self._console = Console(stderr=True)

    def format(self, messages: Sequence[DiagnosticsMessage]) -> None:
        """Print all messages to stderr."""
        for i, msg in enumerate(messages):
            self._format_one(msg)
            # Add blank line after messages with suggestions (except the last)
            suggestion = msg.suggestion or (msg.rule.suggestion if msg.rule else None)
            if suggestion and i < len(messages) - 1:
                self._console.print()
        self._print_summary(messages)

    def _format_one(self, msg: DiagnosticsMessage) -> None:
        location = self._build_location(msg)

        # Color the severity
        severity_colors = {
            "error": "bold red",
            "warning": "bold yellow",
            "info": "bold blue",
            "recommendation": "bold cyan",
            "debug": "dim",
            "failure": "bold red",
        }
        color = severity_colors.get(msg.severity, "white")

        line = Text()
        line.append(location, style="bold")
        line.append(" ")
        line.append(msg.error_code, style=color)
        line.append(" ")
        line.append(msg.message)

        self._console.print(line)

        # Print suggestion if available
        suggestion = msg.suggestion or (msg.rule.suggestion if msg.rule else None)
        if suggestion:
            hint = Text()
            hint.append("  suggestion: ", style="dim green")
            hint.append(suggestion, style="green")
            self._console.print(hint)

    def _build_location(self, msg: DiagnosticsMessage) -> str:
        loc = msg.location or msg.target_type
        return f"{self._file_path}:{loc}:"

    def _print_summary(self, messages: Sequence[DiagnosticsMessage]) -> None:
        errors = sum(1 for m in messages if m.severity == "error")
        warnings = sum(1 for m in messages if m.severity == "warning")
        infos = sum(1 for m in messages if m.severity not in ("error", "warning"))

        parts = []
        if errors:
            parts.append(f"{errors} error{'s' if errors != 1 else ''}")
        if warnings:
            parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
        if infos:
            parts.append(f"{infos} info")

        if parts:
            summary = Text()
            summary.append("\nFound ", style="bold")
            summary.append(", ".join(parts), style="bold")
            summary.append(".", style="bold")
            self._console.print(summary)
        else:
            self._console.print(Text("\nAll checks passed.", style="bold green"))


class JsonFormatter:
    """Machine-readable JSON output."""

    def __init__(self, file_path: str = "<model>") -> None:
        self._file_path = file_path

    def format(self, messages: Sequence[DiagnosticsMessage]) -> None:
        """Print JSON to stdout."""
        results = []
        for msg in messages:
            result = {
                "file": self._file_path,
                "code": msg.error_code,
                "severity": msg.severity,
                "message": msg.message,
                "target_type": msg.target_type,
            }
            if msg.location:
                result["location"] = msg.location
            if msg.rule:
                result["rule_name"] = msg.rule.name
            suggestion = msg.suggestion or (msg.rule.suggestion if msg.rule else None)
            if suggestion:
                result["suggestion"] = suggestion
            results.append(result)

        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")


class GithubFormatter:
    """GitHub Actions annotation format."""

    def __init__(self, file_path: str = "<model>") -> None:
        self._file_path = file_path

    def format(self, messages: Sequence[DiagnosticsMessage]) -> None:
        """Print GitHub Actions annotations to stdout."""
        for msg in messages:
            level = "error" if msg.severity == "error" else "warning"
            location = msg.location or msg.target_type
            suggestion = msg.suggestion or (msg.rule.suggestion if msg.rule else "")
            text = f"{msg.error_code}: {msg.message}"
            if suggestion:
                text += f" Suggestion: {suggestion}"
            print(f"::{level} file={self._file_path},title={msg.error_code}::{text}")
