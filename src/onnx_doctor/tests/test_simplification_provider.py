"""Tests for the SimplificationProvider."""

from __future__ import annotations

import unittest

import onnx
import onnx_ir as ir

import onnx_doctor
from onnx_doctor.diagnostics_providers.simplification import SimplificationProvider


def _make_model(opset: int = 21) -> ir.Model:
    """Create a simple valid model for testing."""
    x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])
    y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])
    node = onnx.helper.make_node("Relu", ["X"], ["Y"])
    graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
    model_proto = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )
    return ir.serde.deserialize_model(model_proto)


def _diagnose(model: ir.Model) -> list[onnx_doctor.DiagnosticsMessage]:
    return list(onnx_doctor.diagnose(model, [SimplificationProvider()]))


def _codes(messages: list[onnx_doctor.DiagnosticsMessage]) -> set[str]:
    return {m.error_code for m in messages}


class SimplificationProviderTest(unittest.TestCase):
    def test_valid_model_no_sim_issues(self):
        model = _make_model()
        messages = _diagnose(model)
        self.assertEqual(_codes(messages), set())

    def test_sim003_unused_nodes(self):
        """A model with a dead node should trigger SIM003."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3]
        )
        y_info = onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [1, 3]
        )
        relu_node = onnx.helper.make_node("Relu", ["X"], ["Y"])
        # Dead node — output "Z" is not a graph output
        dead_node = onnx.helper.make_node("Neg", ["X"], ["Z"])
        graph = onnx.helper.make_graph(
            [relu_node, dead_node], "test", [x_info], [y_info]
        )
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        self.assertIn("SIM003", _codes(messages))

    def test_sim003_fix_removes_unused_nodes(self):
        """Applying the SIM003 fix should remove unused nodes."""
        x_info = onnx.helper.make_tensor_value_info(
            "X", onnx.TensorProto.FLOAT, [1, 3]
        )
        y_info = onnx.helper.make_tensor_value_info(
            "Y", onnx.TensorProto.FLOAT, [1, 3]
        )
        relu_node = onnx.helper.make_node("Relu", ["X"], ["Y"])
        dead_node = onnx.helper.make_node("Neg", ["X"], ["Z"])
        graph = onnx.helper.make_graph(
            [relu_node, dead_node], "test", [x_info], [y_info]
        )
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        sim003 = [m for m in messages if m.error_code == "SIM003"]
        self.assertTrue(len(sim003) > 0)
        self.assertIsNotNone(sim003[0].fix)
        sim003[0].fix()
        # Re-diagnose should have no SIM003
        messages2 = _diagnose(model)
        self.assertNotIn("SIM003", _codes(messages2))

    def test_sim002_unused_opset(self):
        """A model with unused opset import triggers SIM002."""
        model = _make_model()
        # Add an unused opset
        model.opset_imports["ai.onnx.fake"] = 1
        messages = _diagnose(model)
        self.assertIn("SIM002", _codes(messages))

    def test_sim002_fix_removes_unused_opset(self):
        model = _make_model()
        model.opset_imports["ai.onnx.fake"] = 1
        messages = _diagnose(model)
        sim002 = [m for m in messages if m.error_code == "SIM002"]
        self.assertTrue(len(sim002) > 0)
        self.assertIsNotNone(sim002[0].fix)
        sim002[0].fix()
        self.assertNotIn("ai.onnx.fake", model.opset_imports)


if __name__ == "__main__":
    unittest.main()
