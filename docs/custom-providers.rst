Writing Custom Providers
========================

You can create custom diagnostics providers by subclassing ``DiagnosticsProvider``.

Example
-------

.. code-block:: python

   import onnx_ir as ir
   import onnxdoctor
   from onnxdoctor import Rule

   MY_RULE = Rule(
       code="CUSTOM001",
       name="large-model",
       message="Model has more than 1000 nodes.",
       default_severity="warning",
       category="spec",
       target_type="graph",
       suggestion="Consider optimizing or splitting the model.",
   )

   class MyProvider(onnxdoctor.DiagnosticsProvider):
       def check_graph(self, graph: ir.GraphProtocol):
           node_count = sum(1 for _ in graph)
           if node_count > 1000:
               yield onnxdoctor.DiagnosticsMessage(
                   target_type="graph",
                   target=graph,
                   message=f"Graph has {node_count} nodes.",
                   severity=MY_RULE.default_severity,
                   producer="MyProvider",
                   error_code=MY_RULE.code,
                   rule=MY_RULE,
               )

   # Use it
   model = ir.load("model.onnx")
   messages = onnxdoctor.diagnose(model, [MyProvider()])

Available Check Methods
-----------------------

Override any of these methods in your provider:

- ``check_model(model)`` — Called once per model.
- ``check_graph(graph)`` — Called for each graph.
- ``check_function(function)`` — Called for each function.
- ``check_node(node)`` — Called for each node.
- ``check_value(value)`` — Called for each value.
- ``check_tensor(tensor)`` — Called for each tensor.
- ``check_attribute(attribute)`` — Called for each attribute.
