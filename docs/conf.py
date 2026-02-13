"""Configuration file for the Sphinx documentation builder.

To build the documentation: python -m sphinx docs docs/_build/html
"""

import sys

# -- Project information -----------------------------------------------------

project = "ONNX Doctor"
copyright = "2024, Justin Chu"
author = "Justin Chu"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

master_doc = "index"
language = "en"
exclude_patterns = ["_build"]
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "ONNX Doctor"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "onnx": ("https://onnx.ai/onnx/", None),
}
