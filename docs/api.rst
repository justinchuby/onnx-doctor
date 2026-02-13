Programmatic API
================

Core Function
-------------

.. code-block:: python

   import onnxdoctor

   messages = onnxdoctor.diagnose(model, providers)

The ``diagnose`` function walks the IR tree and dispatches to each provider's
check methods (``check_model``, ``check_graph``, ``check_node``, etc.).

DiagnosticsMessage
------------------

Each message contains:

- ``error_code``: The rule code (e.g. ``ONNX001``).
- ``severity``: ``error``, ``warning``, or ``info``.
- ``message``: Human-readable description.
- ``target_type``: What kind of IR object (``model``, ``graph``, ``node``, etc.).
- ``target``: The actual IR object reference.
- ``rule``: The ``Rule`` object (if available), with ``explanation``, ``suggestion``, etc.
- ``suggestion``: Per-instance fix suggestion.
- ``location``: Human-readable location string.

Rule
----

.. code-block:: python

   from onnxdoctor import Rule

   rule = Rule(
       code="CUSTOM001",
       name="my-custom-rule",
       message="Something is wrong.",
       default_severity="warning",
       category="spec",
       target_type="node",
       explanation="Detailed explanation...",
       suggestion="How to fix it.",
   )

RuleRegistry
------------

.. code-block:: python

   from onnxdoctor import RuleRegistry
   from onnxdoctor._loader import get_default_registry

   # Get all built-in rules
   registry = get_default_registry()
   rule = registry.get("ONNX001")
   rule = registry.get("empty-graph-name")  # Also works by name
