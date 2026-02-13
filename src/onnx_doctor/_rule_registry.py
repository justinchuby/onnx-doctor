"""Rule registry for managing and looking up lint rules."""

from __future__ import annotations

from onnxdoctor._rule import Rule


class RuleRegistry:
    """Registry for looking up rules by code or name."""

    def __init__(self) -> None:
        self._by_code: dict[str, Rule] = {}
        self._by_name: dict[str, Rule] = {}

    def register(self, rule: Rule) -> None:
        """Register a rule."""
        if rule.code in self._by_code:
            raise ValueError(f"Duplicate rule code: {rule.code}")
        if rule.name in self._by_name:
            raise ValueError(f"Duplicate rule name: {rule.name}")
        self._by_code[rule.code] = rule
        self._by_name[rule.name] = rule

    def get_by_code(self, code: str) -> Rule | None:
        """Look up a rule by its code (e.g. 'ONNX001')."""
        return self._by_code.get(code)

    def get_by_name(self, name: str) -> Rule | None:
        """Look up a rule by its name (e.g. 'empty-graph-name')."""
        return self._by_name.get(name)

    def get(self, code_or_name: str) -> Rule | None:
        """Look up a rule by code or name."""
        return self._by_code.get(code_or_name) or self._by_name.get(code_or_name)

    def rules(self) -> list[Rule]:
        """Return all registered rules, ordered by code."""
        return sorted(self._by_code.values(), key=lambda r: r.code)

    def rules_by_prefix(self, prefix: str) -> list[Rule]:
        """Return all rules whose code starts with the given prefix."""
        return [r for r in self.rules() if r.code.startswith(prefix)]

    def __len__(self) -> int:
        return len(self._by_code)

    def __contains__(self, code_or_name: str) -> bool:
        return code_or_name in self._by_code or code_or_name in self._by_name
