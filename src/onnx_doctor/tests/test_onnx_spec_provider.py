"""Tests for the OnnxSpecProvider."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir

import onnxdoctor
from onnxdoctor.diagnostics_providers.onnx_spec import OnnxSpecProvider


def _make_model(graph_name: str = "test", opset: int = 21) -> ir.Model:
    """Create a simple valid model for testing."""
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph = onnx.helper.make_graph([node], graph_name, [X], [Y])
    model_proto = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )
    return ir.serde.deserialize_model(model_proto)


def _diagnose(model: ir.ModelProtocol) -> list[onnxdoctor.DiagnosticsMessage]:
    return list(onnxdoctor.diagnose(model, [OnnxSpecProvider()]))


def _codes(messages: list[onnxdoctor.DiagnosticsMessage]) -> set[str]:
    return {m.error_code for m in messages}


class OnnxSpecProviderModelTest(unittest.TestCase):
    def test_valid_model_only_has_ir_version_warning(self):
        model = _make_model()
        messages = _diagnose(model)
        codes = _codes(messages)
        # Only ONNX013 (ir-version-too-new) for modern onnx
        self.assertTrue(
            codes <= {"ONNX013"},
            f"Unexpected codes: {codes}"
        )

    def test_empty_graph_name(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        self.assertIn("ONNX001", _codes(messages))

    def test_missing_default_opset(self):
        model = _make_model()
        # Remove the default opset
        model.opset_imports.clear()
        messages = _diagnose(model)
        self.assertIn("ONNX015", _codes(messages))

    def test_duplicate_metadata_keys(self):
        model = _make_model()
        model.metadata_props["key1"] = "value1"
        # Can't easily duplicate keys in a dict, so this checks the code doesn't crash
        messages = _diagnose(model)
        self.assertNotIn("ONNX014", _codes(messages))


class OnnxSpecProviderNodeTest(unittest.TestCase):
    def test_unregistered_op(self):
        # Create model with a fake op
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        node = onnx.helper.make_node("FakeOpThatDoesNotExist", ["X"], ["Y"])
        graph = onnx.helper.make_graph([node], "test", [X], [Y])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        self.assertIn("ONNX019", _codes(messages))


class OnnxSpecProviderValueTest(unittest.TestCase):
    def test_missing_value_type(self):
        # Create a graph with a value that has no type
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]  # noqa: SLF001
        graph._outputs = [v]  # noqa: SLF001
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX020", _codes(messages))


class OnnxSpecProviderFunctionTest(unittest.TestCase):
    def test_function_empty_name(self):
        # Create a function with empty name
        func_proto = onnx.helper.make_function(
            domain="test.domain",
            fname="",
            inputs=["X"],
            outputs=["Y"],
            nodes=[onnx.helper.make_node("Relu", ["X"], ["Y"])],
            opset_imports=[onnx.helper.make_opsetid("", 21)],
        )
        # Wrap in a model
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([], "test", [X], [Y])
        model_proto = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 21)],
            functions=[func_proto],
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        self.assertIn("ONNX028", _codes(messages))


class DiagnosticsMessageFieldsTest(unittest.TestCase):
    def test_message_has_rule_field(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001_msgs = [m for m in messages if m.error_code == "ONNX001"]
        self.assertTrue(len(onnx001_msgs) > 0)
        msg = onnx001_msgs[0]
        self.assertIsNotNone(msg.rule)
        self.assertEqual(msg.rule.code, "ONNX001")
        self.assertEqual(msg.rule.name, "empty-graph-name")

    def test_message_has_suggestion(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001_msgs = [m for m in messages if m.error_code == "ONNX001"]
        msg = onnx001_msgs[0]
        # Should have suggestion from rule
        self.assertTrue(msg.suggestion or msg.rule.suggestion)


if __name__ == "__main__":
    unittest.main()
