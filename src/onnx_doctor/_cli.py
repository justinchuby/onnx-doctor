"""CLI entry point for onnx-doctor."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import onnx_ir as ir

import onnx_doctor
from onnx_doctor._formatter import GithubFormatter, JsonFormatter, TextFormatter
from onnx_doctor._loader import get_default_registry
from onnx_doctor.diagnostics_providers.onnx_spec import OnnxSpecProvider


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onnx-doctor",
        description="Diagnose and lint ONNX models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # check command
    check_parser = subparsers.add_parser(
        "check", help="Check an ONNX model for issues."
    )
    check_parser.add_argument("model", type=str, help="Path to the .onnx model file.")
    check_parser.add_argument(
        "--select",
        type=str,
        nargs="*",
        default=None,
        help="Only report rules matching these codes or prefixes (e.g. ONNX001 ONNX).",
    )
    check_parser.add_argument(
        "--ignore",
        type=str,
        nargs="*",
        default=None,
        help="Ignore rules matching these codes or prefixes.",
    )
    check_parser.add_argument(
        "--output-format",
        choices=["text", "json", "github"],
        default="text",
        help="Output format (default: text).",
    )
    check_parser.add_argument(
        "--severity",
        choices=["error", "warning", "info"],
        default=None,
        help="Minimum severity to report.",
    )

    # explain command
    explain_parser = subparsers.add_parser(
        "explain", help="Show detailed explanation for a rule."
    )
    explain_parser.add_argument(
        "code", type=str, help="Rule code or name (e.g. ONNX001 or empty-graph-name)."
    )

    # list-rules command
    subparsers.add_parser("list-rules", help="List all available rules.")

    return parser


def _severity_rank(severity: str) -> int:
    return {"error": 0, "warning": 1, "info": 2}.get(severity, 3)


def _filter_messages(
    messages: Sequence[onnx_doctor.DiagnosticsMessage],
    select: list[str] | None,
    ignore: list[str] | None,
    min_severity: str | None,
) -> list[onnx_doctor.DiagnosticsMessage]:
    """Filter messages by select/ignore codes and minimum severity.

    Rules with ``default_enabled=False`` are excluded unless explicitly
    listed in *select*.
    """
    filtered = list(messages)

    # Exclude rules that are off by default, unless explicitly selected
    if select is not None:
        # When --select is used, show only selected rules (enabled or not)
        filtered = [
            m
            for m in filtered
            if any(m.error_code == s or m.error_code.startswith(s) for s in select)
        ]
    else:
        # No --select: exclude default-disabled rules
        filtered = [m for m in filtered if m.rule is None or m.rule.default_enabled]

    if ignore is not None:
        filtered = [
            m
            for m in filtered
            if not any(m.error_code == i or m.error_code.startswith(i) for i in ignore)
        ]

    if min_severity is not None:
        max_rank = _severity_rank(min_severity)
        filtered = [m for m in filtered if _severity_rank(m.severity) <= max_rank]

    return filtered


def _get_providers() -> list[onnx_doctor.DiagnosticsProvider]:
    """Get the default set of providers."""
    return [OnnxSpecProvider()]


def _cmd_check(args: argparse.Namespace) -> int:
    """Run the check command."""
    model_path = args.model

    # Load model
    try:
        model = ir.load(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}", file=sys.stderr)
        return 1

    # Run diagnostics
    providers = _get_providers()
    messages = onnx_doctor.diagnose(model, providers)

    # Filter
    filtered = _filter_messages(
        messages,
        select=args.select,
        ignore=args.ignore,
        min_severity=args.severity,
    )

    # Format output
    if args.output_format == "json":
        formatter = JsonFormatter(file_path=model_path)
    elif args.output_format == "github":
        formatter = GithubFormatter(file_path=model_path)
    else:
        formatter = TextFormatter(file_path=model_path)

    formatter.format(filtered)

    # Exit code: 1 if any errors
    has_errors = any(m.severity == "error" for m in filtered)
    return 1 if has_errors else 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """Show detailed explanation for a rule."""
    registry = get_default_registry()
    rule = registry.get(args.code)

    if rule is None:
        print(f"Unknown rule: '{args.code}'", file=sys.stderr)
        print(
            "Use 'onnx-doctor list-rules' to see all available rules.", file=sys.stderr
        )
        return 1

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    console = Console()

    # Header
    header = Text()
    header.append(f"{rule.code}", style="bold cyan")
    header.append(f": {rule.name}", style="bold")
    console.print(header)
    console.print()

    # Message
    console.print(Text(f"  Message: {rule.message}"))
    console.print(Text(f"  Severity: {rule.default_severity}"))
    console.print(Text(f"  Category: {rule.category}"))
    console.print(Text(f"  Target: {rule.target_type}"))

    if rule.suggestion:
        console.print()
        console.print(Text(f"  Suggestion: {rule.suggestion}", style="green"))

    if rule.explanation:
        console.print()
        console.print(Markdown(rule.explanation))

    return 0


def _cmd_list_rules(_args: argparse.Namespace) -> int:
    """List all available rules."""
    registry = get_default_registry()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Rules")
    table.add_column("Code", style="bold cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Severity")
    table.add_column("Target")
    table.add_column("Message")

    for rule in registry.rules():
        severity_style = {
            "error": "red",
            "warning": "yellow",
            "info": "blue",
        }.get(rule.default_severity, "white")

        table.add_row(
            rule.code,
            rule.name,
            f"[{severity_style}]{rule.default_severity}[/{severity_style}]",
            rule.target_type,
            rule.message,
        )

    console.print(table)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _cmd_check(args)
    elif args.command == "explain":
        return _cmd_explain(args)
    elif args.command == "list-rules":
        return _cmd_list_rules(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
