"""Generate RST rule reference pages from the rule registry."""

from __future__ import annotations

import os
import sys

# Add the src directory to sys.path so we can import onnx_doctor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from onnx_doctor._loader import get_default_registry  # noqa: E402


def generate_rule_page(rule, output_dir: str) -> None:
    """Generate an RST page for a single rule."""
    path = os.path.join(output_dir, f"{rule.code}.rst")
    lines = [
        f"{rule.code}: {rule.name}",
        "=" * len(f"{rule.code}: {rule.name}"),
        "",
        f"**Code:** ``{rule.code}``",
        "",
        f"**Name:** ``{rule.name}``",
        "",
        f"**Severity:** {rule.default_severity}",
        "",
        f"**Category:** {rule.category}",
        "",
        f"**Target:** {rule.target_type}",
        "",
        "Message",
        "-------",
        "",
        rule.message,
        "",
    ]

    if rule.suggestion:
        lines.extend([
            "Suggestion",
            "----------",
            "",
            rule.suggestion,
            "",
        ])

    if rule.explanation:
        lines.extend([
            "Details",
            "-------",
            "",
            rule.explanation,
            "",
        ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_index(rules, output_dir: str) -> None:
    """Generate the rules index page."""
    # Group by prefix
    by_prefix: dict[str, list] = {}
    for rule in rules:
        # Extract prefix (all non-digit chars at start)
        prefix = ""
        for c in rule.code:
            if c.isdigit():
                break
            prefix += c
        by_prefix.setdefault(prefix, []).append(rule)

    lines = [
        "Rule Reference",
        "==============",
        "",
        f"ONNX Doctor includes **{len(rules)} rules** across multiple providers.",
        "",
    ]

    for prefix, prefix_rules in sorted(by_prefix.items()):
        lines.extend([
            f"{prefix} Rules",
            "-" * len(f"{prefix} Rules"),
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 15 30 12 12 31",
            "",
            "   * - Code",
            "     - Name",
            "     - Severity",
            "     - Target",
            "     - Message",
        ])

        for rule in prefix_rules:
            lines.extend([
                f"   * - :doc:`{rule.code}`",
                f"     - ``{rule.name}``",
                f"     - {rule.default_severity}",
                f"     - {rule.target_type}",
                f"     - {rule.message}",
            ])

        lines.append("")

    # Add toctree
    lines.extend([
        ".. toctree::",
        "   :hidden:",
        "   :maxdepth: 1",
        "",
    ])
    for rule in rules:
        lines.append(f"   {rule.code}")
    lines.append("")

    path = os.path.join(output_dir, "index.rst")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), "rules")
    os.makedirs(output_dir, exist_ok=True)

    registry = get_default_registry()
    rules = registry.rules()

    print(f"Generating docs for {len(rules)} rules...")
    for rule in rules:
        generate_rule_page(rule, output_dir)
    generate_index(rules, output_dir)
    print(f"Generated {len(rules)} rule pages in {output_dir}/")


if __name__ == "__main__":
    main()
