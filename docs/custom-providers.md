# Writing Custom Providers

Create custom diagnostics providers by subclassing `DiagnosticsProvider`.

## Example

```python
import onnx_ir as ir
import onnx_doctor
from onnx_doctor import Rule

MY_RULE = Rule(
    code="CUSTOM001",
    name="large-model",
    message="Model has more than 1000 nodes.",
    default_severity="warning",
    category="spec",
    target_type="graph",
    suggestion="Consider optimizing or splitting the model.",
)

class MyProvider(onnx_doctor.DiagnosticsProvider):
    def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
        """Analyze the model and yield diagnostics messages."""
        # Check each graph in the model
        yield from self._check_graph(model.graph)

        # Optionally check functions
        for func in model.functions.values():
            for node in func:
                # ... check nodes in functions
                pass

    def _check_graph(self, graph: ir.Graph) -> onnx_doctor.DiagnosticsMessageIterator:
        if len(graph) > 1000:
            yield onnx_doctor.DiagnosticsMessage(
                target_type="graph",
                target=graph,
                message=f"Graph has {len(graph)} nodes.",
                severity=MY_RULE.default_severity,
                producer="MyProvider",
                error_code=MY_RULE.code,
                rule=MY_RULE,
            )

        # Recursively check subgraphs in node attributes
        for node in graph:
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    yield from self._check_graph(attr.value)
                elif attr.type == ir.AttributeType.GRAPHS:
                    for subgraph in attr.value:
                        yield from self._check_graph(subgraph)

# Use it
model = ir.load("model.onnx")
messages = onnx_doctor.diagnose(model, [MyProvider()])
```

## The `diagnose` Method

Each provider implements a single method:

```python
def diagnose(self, model: ir.Model) -> onnx_doctor.DiagnosticsMessageIterator:
    ...
```

The provider is responsible for walking the model structure as needed. This gives
providers full control over their traversal strategy. Common patterns:

- Use `ir.traversal.RecursiveGraphIterator(graph)` to iterate all nodes including subgraphs, or use `graph.all_nodes()` or `func.all_nodes()`.
- Manually iterate `model.graph`, `model.functions`, node attributes, etc.

## Location Inference

You don't need to set `location` on messages. The driver automatically infers
a human-readable location path from the `target` object:

- Node: `graph:node[0](Relu, "my_node")`
- Value: `graph:input[0](X)` or `graph:node[0](Relu):output[0](Y)`
- Graph: `graph` or `graph:node[0](If):then_branch`
- Function: `function(domain:name)`

## Valid Target Types

The `target` field must be one of:

- `ir.Model`
- `ir.Graph`
- `ir.Node`
- `ir.Value`
- `ir.Function`
