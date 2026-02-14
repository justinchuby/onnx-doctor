"""Tests for the OnnxSpecProvider."""

from __future__ import annotations

import unittest

import numpy as np
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
        self.assertIn("ONNX013", _codes(messages))

    def test_duplicate_metadata_keys(self):
        model = _make_model()
        model.metadata_props["key1"] = "value1"
        # Can't easily duplicate keys in a dict, so this checks the code doesn't crash
        messages = _diagnose(model)
        self.assertNotIn("ONNX012", _codes(messages))

    def test_no_false_onnx003_for_subgraph_with_outer_scope_input(self):
        """Subgraph referencing a parent graph input should not trigger ONNX003."""
        # Parent graph: X -> Relu -> relu_out -> Loop(body uses X from outer scope)
        x = ir.Value(
            name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1])
        )
        relu = ir.Node("", "Relu", inputs=[x], name="relu")
        relu_out = relu.outputs[0]
        relu_out.name = "relu_out"

        # Subgraph body: uses x (parent graph input) as an input to Add
        sub_in = ir.Value(name="sub_in")
        sub_add = ir.Node("", "Add", inputs=[sub_in, x], name="sub_add")
        sub_out = sub_add.outputs[0]
        sub_out.name = "sub_out"

        subgraph = ir.Graph(
            inputs=[sub_in],
            outputs=[sub_out],
            nodes=[sub_add],
            name="body",
            opset_imports={"": 21},
        )

        body_attr = ir.Attr("body", ir.AttributeType.GRAPH, subgraph)
        loop_node = ir.Node("", "Loop", [relu_out], [body_attr], name="loop")
        loop_out = loop_node.outputs[0]
        loop_out.name = "Y"
        loop_out.type = ir.TensorType(ir.DataType.FLOAT)
        loop_out.shape = ir.Shape([1])

        graph = ir.Graph(
            inputs=[x],
            outputs=[loop_out],
            nodes=[relu, loop_node],
            name="main",
            opset_imports={"": 21},
        )
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21

        messages = _diagnose(model)
        codes = _codes(messages)
        self.assertNotIn(
            "ONNX003", codes, "Outer-scope input should not cause false ONNX003"
        )
        self.assertNotIn(
            "ONNX004", codes, "Outer-scope input should not cause false ONNX004"
        )


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
        self.assertIn("ONNX017", _codes(messages))


class OnnxSpecProviderValueTest(unittest.TestCase):
    def test_missing_value_type_emitted(self):
        # ONNX018 is still emitted by the provider (filtering is CLI-level)
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX018", _codes(messages))

    def test_graph_io_missing_type(self):
        # ONNX034: graph inputs/outputs must have type info
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX034", _codes(messages))

    def test_initializer_missing_const_value(self):
        # ONNX104: initializer must have const_value
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="init_without_value")
        # Add to initializers without setting const_value
        graph.initializers["init_without_value"] = v
        graph.inputs.append(v)
        graph.outputs.append(v)
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        self.assertIn("ONNX104", _codes(messages))


class LocationTrackingTest(unittest.TestCase):
    def test_graph_message_has_correct_target(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001 = [m for m in messages if m.error_code == "ONNX001"]
        self.assertTrue(len(onnx001) > 0)
        self.assertEqual(onnx001[0].target_type, "graph")
        self.assertIs(onnx001[0].target, model.graph)

    def test_graph_message_has_inferred_location(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        onnx001 = [m for m in messages if m.error_code == "ONNX001"]
        self.assertTrue(len(onnx001) > 0)
        self.assertEqual(onnx001[0].location, "graph")

    def test_node_message_has_correct_target(self):
        # Create model with a fake op to trigger ONNX017
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        node = onnx.helper.make_node("FakeOp", ["X"], ["Y"], name="fake_node")
        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        onnx017 = [m for m in messages if m.error_code == "ONNX017"]
        self.assertTrue(len(onnx017) > 0)
        self.assertEqual(onnx017[0].target_type, "node")

    def test_node_message_has_inferred_location(self):
        # Create model with a fake op to trigger ONNX017
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        node = onnx.helper.make_node("FakeOp", ["X"], ["Y"], name="fake_node")
        graph = onnx.helper.make_graph([node], "test", [x_info], [y_info])
        model_proto = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 21)]
        )
        model = ir.serde.deserialize_model(model_proto)
        messages = _diagnose(model)
        onnx017 = [m for m in messages if m.error_code == "ONNX017"]
        self.assertTrue(len(onnx017) > 0)
        # Format: node[index](op_type, "name")
        self.assertIn('node[0](FakeOp, "fake_node")', onnx017[0].location)

    def test_value_message_has_correct_target(self):
        graph = ir.Graph([], [], nodes=[], name="test")
        v = ir.Value(name="untyped")
        graph._inputs = [v]
        graph._outputs = [v]
        model = ir.Model(graph, ir_version=10)
        model.opset_imports[""] = 21
        messages = _diagnose(model)
        onnx018 = [m for m in messages if m.error_code == "ONNX018"]
        self.assertTrue(len(onnx018) > 0)
        # target_type is "node" for values (per the existing code)
        self.assertEqual(onnx018[0].target_type, "node")
        self.assertIs(onnx018[0].target, v)


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
        self.assertIn("ONNX026", _codes(messages))

    def test_ref_attr_in_main_graph_raises_onnx039(self):
        """Reference attributes should only appear in functions, not the main graph."""
        model = _make_model()
        # Add a reference attribute to the node in the main graph
        node = list(model.graph)[0]
        node.attributes["test_ref"] = ir.RefAttr(
            "test_ref", "external_attr", ir.AttributeType.FLOAT
        )
        messages = _diagnose(model)
        self.assertIn("ONNX039", _codes(messages))

    def test_ref_attr_in_function_is_valid(self):
        """Reference attributes inside functions should not raise ONNX039."""
        # Create a function with a node that has a ref attr
        func_proto = onnx.helper.make_function(
            domain="test.domain",
            fname="MyFunc",
            inputs=["X"],
            outputs=["Y"],
            nodes=[onnx.helper.make_node("Relu", ["X"], ["Y"])],
            opset_imports=[onnx.helper.make_opsetid("", 21)],
        )
        x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([], "test", [x_info], [y_info])
        model_proto = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 21)],
            functions=[func_proto],
        )
        model = ir.serde.deserialize_model(model_proto)
        # Add ref attr to the function node (valid usage)
        func = list(model.functions.values())[0]
        node = list(func)[0]
        node.attributes["test_ref"] = ir.RefAttr(
            "test_ref", "external_attr", ir.AttributeType.FLOAT
        )
        messages = _diagnose(model)
        # ONNX039 should NOT be raised
        self.assertNotIn("ONNX039", _codes(messages))


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

    def test_onnx003_fix_sorts_graph(self):
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
        onnx003 = [m for m in messages if m.error_code == "ONNX003"]
        self.assertTrue(len(onnx003) > 0)
        msg = onnx003[0]
        self.assertIsNotNone(msg.fix)
        # Apply the fix
        msg.fix()
        # Re-diagnose should not have ONNX003
        messages2 = _diagnose(model)
        self.assertNotIn("ONNX003", _codes(messages2))

    def test_fixable_messages_have_fix_callable(self):
        model = _make_model(graph_name="")
        messages = _diagnose(model)
        for msg in messages:
            if msg.rule and msg.rule.fixable:
                self.assertIsNotNone(
                    msg.fix, f"{msg.error_code} is fixable but has no fix"
                )

    def test_no_false_onnx007_for_initializer_as_graph_output(self):
        """Initializers used as subgraph outputs should not trigger ONNX007."""
        # Build a model with an If node whose then_branch has an initializer as output
        x = ir.Value(name="X")
        cond = ir.Value(name="cond")

        # Subgraph: initializer 'init_val' is a graph output (no producer node)
        init_val = ir.Value(
            name="init_val",
            const_value=ir.Tensor(np.array([1.0], dtype=np.float32), name="init_val"),
        )
        sub_graph = ir.Graph(
            inputs=[],
            outputs=[init_val],
            nodes=[],
            initializers=[init_val],
            name="then_branch",
            opset_imports={"": 21},
        )
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[ir.Attr("then_branch", ir.AttributeType.GRAPH, sub_graph)],
            num_outputs=1,
        )
        graph = ir.Graph(
            inputs=[x, cond],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main",
            opset_imports={"": 21},
        )
        model = ir.Model(graph, ir_version=9)
        messages = _diagnose(model)
        onnx007_codes = [m for m in messages if m.error_code == "ONNX007"]
        self.assertEqual(onnx007_codes, [], f"Unexpected ONNX007: {onnx007_codes}")


class SubgraphShadowingTest(unittest.TestCase):
    def test_subgraph_shadowing_detected(self):
        """Subgraph that redefines a name from the outer graph triggers ONNX038."""
        x = ir.Value(
            name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1])
        )
        cond = ir.Value(
            name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        relu = ir.Node("", "Relu", inputs=[x], name="relu")
        relu_out = relu.outputs[0]
        relu_out.name = "relu_out"

        # Subgraph reuses name "X" — this shadows the outer graph input
        sub_x = ir.Value(name="X")
        sub_relu = ir.Node("", "Relu", inputs=[sub_x], name="sub_relu")
        sub_out = sub_relu.outputs[0]
        sub_out.name = "sub_out"

        subgraph = ir.Graph(
            inputs=[sub_x],
            outputs=[sub_out],
            nodes=[sub_relu],
            name="then_branch",
            opset_imports={"": 21},
        )

        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[ir.Attr("then_branch", ir.AttributeType.GRAPH, subgraph)],
            name="if_node",
        )
        if_out = if_node.outputs[0]
        if_out.name = "Y"
        if_out.type = ir.TensorType(ir.DataType.FLOAT)
        if_out.shape = ir.Shape([1])

        graph = ir.Graph(
            inputs=[x, cond],
            outputs=[if_out],
            nodes=[relu, if_node],
            name="main",
            opset_imports={"": 21},
        )
        model = ir.Model(graph, ir_version=9)
        messages = _diagnose(model)
        self.assertIn("ONNX038", _codes(messages))

    def test_no_shadowing_no_onnx038(self):
        """Subgraph with unique names should not trigger ONNX038."""
        x = ir.Value(
            name="X", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1])
        )
        cond = ir.Value(
            name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        relu = ir.Node("", "Relu", inputs=[x], name="relu")
        relu_out = relu.outputs[0]
        relu_out.name = "relu_out"

        # Subgraph uses distinct name "Z" — no shadowing
        sub_z = ir.Value(name="Z")
        sub_relu = ir.Node("", "Relu", inputs=[sub_z], name="sub_relu")
        sub_out = sub_relu.outputs[0]
        sub_out.name = "sub_out"

        subgraph = ir.Graph(
            inputs=[sub_z],
            outputs=[sub_out],
            nodes=[sub_relu],
            name="then_branch",
            opset_imports={"": 21},
        )

        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[ir.Attr("then_branch", ir.AttributeType.GRAPH, subgraph)],
            name="if_node",
        )
        if_out = if_node.outputs[0]
        if_out.name = "Y"
        if_out.type = ir.TensorType(ir.DataType.FLOAT)
        if_out.shape = ir.Shape([1])

        graph = ir.Graph(
            inputs=[x, cond],
            outputs=[if_out],
            nodes=[relu, if_node],
            name="main",
            opset_imports={"": 21},
        )
        model = ir.Model(graph, ir_version=9)
        messages = _diagnose(model)
        self.assertNotIn("ONNX038", _codes(messages))


if __name__ == "__main__":
    unittest.main()
