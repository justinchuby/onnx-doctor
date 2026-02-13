CLI Reference
=============

Usage
-----

.. code-block:: bash

   onnx-doctor <command> [options]

Commands
--------

check
^^^^^

Check an ONNX model for issues.

.. code-block:: bash

   onnx-doctor check model.onnx [options]

Options:

- ``--select CODE [CODE ...]`` — Only report rules matching these codes or prefixes.
- ``--ignore CODE [CODE ...]`` — Ignore rules matching these codes or prefixes.
- ``--output-format {text,json,github}`` — Output format (default: ``text``).
- ``--severity {error,warning,info}`` — Minimum severity to report.

Examples:

.. code-block:: bash

   # Check a model, ignoring ir-version warnings
   onnx-doctor check model.onnx --ignore ONNX013

   # Output as JSON for CI integration
   onnx-doctor check model.onnx --output-format json

   # Only show errors
   onnx-doctor check model.onnx --severity error

   # GitHub Actions annotations
   onnx-doctor check model.onnx --output-format github

explain
^^^^^^^

Show detailed explanation for a rule.

.. code-block:: bash

   onnx-doctor explain ONNX001

list-rules
^^^^^^^^^^

List all available rules.

.. code-block:: bash

   onnx-doctor list-rules

Exit Codes
----------

- ``0`` — No errors found (warnings may be present).
- ``1`` — Errors found, or the model could not be loaded.
