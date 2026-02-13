Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install onnx-doctor

Quick Start
-----------

Check a model from the command line:

.. code-block:: bash

   onnx-doctor check model.onnx

Or use the programmatic API:

.. code-block:: python

   import onnx_ir as ir
   import onnxdoctor
   from onnxdoctor.diagnostics_providers import OnnxSpecProvider

   model = ir.load("model.onnx")
   messages = onnxdoctor.diagnose(model, [OnnxSpecProvider()])

   for msg in messages:
       print(f"[{msg.severity}] {msg.error_code}: {msg.message}")
