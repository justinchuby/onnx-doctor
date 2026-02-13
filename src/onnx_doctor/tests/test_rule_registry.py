"""Tests for the rule registry and rule loading."""

from __future__ import annotations

import unittest

from onnx_doctor._loader import _SPEC_YAML, get_default_registry, load_rules_from_yaml
from onnx_doctor._rule import Rule
from onnx_doctor._rule_registry import RuleRegistry


class RuleRegistryTest(unittest.TestCase):
    def test_register_and_lookup_by_code(self):
        registry = RuleRegistry()
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        self.assertEqual(registry.get_by_code("TEST001"), rule)

    def test_register_and_lookup_by_name(self):
        registry = RuleRegistry()
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        self.assertEqual(registry.get_by_name("test-rule"), rule)

    def test_get_by_code_or_name(self):
        registry = RuleRegistry()
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        self.assertEqual(registry.get("TEST001"), rule)
        self.assertEqual(registry.get("test-rule"), rule)
        self.assertIsNone(registry.get("nonexistent"))

    def test_duplicate_code_raises(self):
        registry = RuleRegistry()
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        with self.assertRaises(ValueError):
            registry.register(rule)

    def test_rules_returns_sorted(self):
        registry = RuleRegistry()
        r2 = Rule(
            code="B002",
            name="b",
            message="B.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        r1 = Rule(
            code="A001",
            name="a",
            message="A.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(r2)
        registry.register(r1)
        self.assertEqual(registry.rules(), [r1, r2])

    def test_rules_by_prefix(self):
        registry = RuleRegistry()
        r1 = Rule(
            code="ONNX001",
            name="a",
            message="A.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        r2 = Rule(
            code="ORT001",
            name="b",
            message="B.",
            default_severity="error",
            category="spec",
            target_type="node",
        )
        registry.register(r1)
        registry.register(r2)
        self.assertEqual(registry.rules_by_prefix("ONNX"), [r1])
        self.assertEqual(registry.rules_by_prefix("ORT"), [r2])

    def test_contains(self):
        registry = RuleRegistry()
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        self.assertIn("TEST001", registry)
        self.assertIn("test-rule", registry)
        self.assertNotIn("nonexistent", registry)

    def test_len(self):
        registry = RuleRegistry()
        self.assertEqual(len(registry), 0)
        rule = Rule(
            code="TEST001",
            name="test-rule",
            message="Test.",
            default_severity="error",
            category="spec",
            target_type="graph",
        )
        registry.register(rule)
        self.assertEqual(len(registry), 1)


class DefaultRegistryTest(unittest.TestCase):
    def test_loads_all_rules(self):
        registry = get_default_registry()
        # 37 ONNX rules + 13 PB rules + 3 SIM rules + 5 ORT rules = 58
        self.assertEqual(len(registry), 58)

    def test_all_rules_have_valid_prefix(self):
        registry = get_default_registry()
        valid_prefixes = ("ONNX", "PB", "SIM", "ORT")
        for rule in registry.rules():
            self.assertTrue(
                any(rule.code.startswith(p) for p in valid_prefixes),
                f"Rule {rule.code} doesn't start with a valid prefix {valid_prefixes}",
            )

    def test_all_rules_have_required_fields(self):
        registry = get_default_registry()
        for rule in registry.rules():
            self.assertTrue(rule.code, "Rule missing code")
            self.assertTrue(rule.name, f"Rule {rule.code} missing name")
            self.assertTrue(rule.message, f"Rule {rule.code} missing message")
            self.assertIn(rule.default_severity, ("error", "warning", "info"))
            self.assertIn(rule.category, ("spec", "ir", "protobuf"))
            self.assertIn(
                rule.target_type,
                ("model", "graph", "node", "value", "tensor", "function", "attribute"),
            )

    def test_no_duplicate_codes(self):
        rules = load_rules_from_yaml(_SPEC_YAML)
        codes = [r.code for r in rules]
        self.assertEqual(len(codes), len(set(codes)))

    def test_no_duplicate_names(self):
        rules = load_rules_from_yaml(_SPEC_YAML)
        names = [r.name for r in rules]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
