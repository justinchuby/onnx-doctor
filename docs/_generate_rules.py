"""Generate markdown rule reference pages from the rule registry."""

from __future__ import annotations

import os
import sys

# Add the src directory to sys.path so we can import onnx_doctor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from onnx_doctor._loader import get_default_registry


def generate_rule_page(rule, output_dir: str) -> None:
    """Generate a markdown page for a single rule."""
    path = os.path.join(output_dir, f"{rule.code}.md")
    lines = [
        f"# {rule.code}: {rule.name}",
        "",
        f"**Code:** `{rule.code}`",
        "",
        f"**Name:** `{rule.name}`",
        "",
        f"**Severity:** {rule.default_severity}",
        "",
        f"**Category:** {rule.category}",
        "",
        f"**Target:** {rule.target_type}",
        "",
        f"**Fixable:** {'🔧 Yes' if rule.fixable else 'No'}",
        "",
    ]

    if not rule.default_enabled:
        lines.extend(
            [
                "**Enabled by default:** No (use `--select "
                + rule.code
                + "` to enable)",
                "",
            ]
        )

    lines.extend(
        [
            "## Message",
            "",
            rule.message,
            "",
        ]
    )

    if rule.suggestion:
        lines.extend(
            [
                "## Suggestion",
                "",
                rule.suggestion,
                "",
            ]
        )

    if rule.explanation:
        lines.extend(
            [
                "## Details",
                "",
                rule.explanation.strip(),
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_index(rules, output_dir: str) -> None:
    """Generate the rules index page in markdown."""
    # Group by prefix
    by_prefix: dict[str, list] = {}
    for rule in rules:
        prefix = ""
        for c in rule.code:
            if c.isdigit():
                break
            prefix += c
        by_prefix.setdefault(prefix, []).append(rule)

    fixable_count = sum(1 for r in rules if r.fixable)

    lines = [
        "# Rule Reference",
        "",
        f"ONNX Doctor includes **{len(rules)} rules** across multiple providers"
        f" ({fixable_count} with auto-fix 🔧).",
        "",
    ]

    for prefix, prefix_rules in sorted(by_prefix.items()):
        lines.extend(
            [
                f"## {prefix} Rules",
                "",
                "| Code | Name | Severity | Fix | Target | Message |",
                "|------|------|----------|-----|--------|---------|",
            ]
        )

        lines.extend(
            f"| [{rule.code}]({rule.code}.md) | `{rule.name}` | {rule.default_severity} | {'🔧' if rule.fixable else ''} | {rule.target_type} | {rule.message} |"
            for rule in prefix_rules
        )

        lines.append("")

    # Add toctree (MyST syntax)
    lines.extend(
        [
            "```{toctree}",
            ":hidden:",
            ":maxdepth: 1",
            "",
        ]
    )
    lines.extend(rule.code for rule in rules)
    lines.extend(["```", ""])

    path = os.path.join(output_dir, "index.md")
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
