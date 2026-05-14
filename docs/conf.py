from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pi-Stack Optimizer"
author = "Arunima Ghosh, Susmita Barik, Roshan J Singh, Sandeep K. Reddy"
release = ""

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_title = "pi-Stack Optimizer Documentation"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosectionlabel_prefix_document = True
