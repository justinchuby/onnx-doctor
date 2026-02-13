"""Sphinx configuration for ONNX Doctor documentation."""

project = "ONNX Doctor"
copyright = "2024, Justin Chu"
author = "Justin Chu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "ONNX Doctor"
