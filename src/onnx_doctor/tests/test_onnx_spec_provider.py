"""Tests for the OnnxSpecProvider."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir

import onnx_doctor
from onnx_doctor.diagnostics_providers.onnx_spec import OnnxSpecProvider


def _make_model(graph_name: str = "test", opset: int = 21) -> ir.Model:
    """Create a simple valid model for testing."""
    x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])
    y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph = onnx.helper.make_graph([node], graph_name, [x_info], [y_info])
    model_proto = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )
    return ir.serde.deserialize_model(model_proto)


def _diagnose(model: ir.Model) -> list[onnx_doctor.DiagnosticsMessage]:
    return list(onnx_doctor.diagnose(model, [OnnxSpecProvider()]))


def _codes(messages: list[onnx_doctor.DiagnosticsMessage]) -> set[str]:
    return {m.error_code for m in messages}


class OnnxSpecProviderModelTest(unittest.TestCase):
    def test_valid_model_has_no_issues(self):
        model = _make_model()
        messages = _diagnose(model)
        codes = _codes(messages)
        # No issues for modern onnx models
        self.assertEqual(codes, set(), f"Unexpected codes: {codes}")

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
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        node = onnx.helper.make_node("FakeOpThatDoesNotExist", ["X"], ["Y"])
        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        self.assertIn("ONNX019", _codes(messages))


class OnnxSpecProviderValueTest(unittest.TestCase):
    def test_missing_value_type_emitted(self):
        # ONNX020 is still emitted by the provider (filtering is CLI-level)
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX020", _codes(messages))

    def test_graph_io_missing_type(self):
        # ONNX036: graph inputs/outputs must have type info
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX036", _codes(messages))


class LocationTrackingTest(unittest.TestCase):
    def test_graph_message_has_graph_location(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001 = [m for m in messages if m.error_code == "ONNX001"]
        self.assertTrue(len(onnx001) > 0)
        self.assertEqual(onnx001[0].location, "graph")

    def test_node_message_has_node_location(self):
        # Create model with a fake op to trigger ONNX019
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        node = onnx.helper.make_node("FakeOp", ["X"], ["Y"], name="fake_node")
        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        onnx019 = [m for m in messages if m.error_code == "ONNX019"]
        self.assertTrue(len(onnx019) > 0)
        self.assertIn("node/0(fake_node)", onnx019[0].location)

    def test_value_message_has_producer_location(self):
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        onnx020 = [m for m in messages if m.error_code == "ONNX020"]
        self.assertTrue(len(onnx020) > 0)
        self.assertIn("input(untyped)", onnx020[0].location)


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
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([], "test", [x_info], [y_info])
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


class FixTest(unittest.TestCase):
    def test_onnx001_fix_sets_graph_name(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001 = [m for m in messages if m.error_code == "ONNX001"]
        self.assertTrue(len(onnx001) > 0)
        msg = onnx001[0]
        self.assertIsNotNone(msg.fix)
        # Apply the fix
        msg.fix()
        self.assertEqual(model.graph.name, "main_graph")
        # Re-diagnose should not have ONNX001
        messages2 = _diagnose(model)
        self.assertNotIn("ONNX001", _codes(messages2))

    def test_onnx004_fix_sorts_graph(self):
        # Create a model with unsorted nodes
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        z_info = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1])
        node1 = onnx.helper.make_node("Relu", ["Y"], ["Z"])
        node2 = onnx.helper.make_node("Relu", ["X"], ["Y"])
        graph = onnx.helper.make_graph([node1, node2], "test", [x_info], [z_info])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        onnx004 = [m for m in messages if m.error_code == "ONNX004"]
        self.assertTrue(len(onnx004) > 0)
        msg = onnx004[0]
        self.assertIsNotNone(msg.fix)
        # Apply the fix
        msg.fix()
        # Re-diagnose should not have ONNX004
        messages2 = _diagnose(model)
        self.assertNotIn("ONNX004", _codes(messages2))

    def test_fixable_messages_have_fix_callable(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        for msg in messages:
            if msg.rule and msg.rule.fixable:
                self.assertIsNotNone(msg.fix, f"{msg.error_code} is fixable but has no fix")


if __name__ == "__main__":
    unittest.main()
