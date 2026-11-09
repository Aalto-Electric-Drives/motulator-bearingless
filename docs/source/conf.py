# ruff: noqa

# Configuration file for the Sphinx documentation builder. The configuration follows
# that of motulator, so that the documentation has the same style.
#
# For a full list of options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ------------------------------------------------------------------------

import os
import sys
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../.."))

os.environ["BUILDING_DOCS"] = "1"

# -- Project information ---------------------------------------------------------------

project = "motulator-bearingless"
copyright = "2026, Aalto Electric Drives"
author = "Aalto Electric Drives"

# The full version, including alpha/beta/rc tags
release = package_version("motulator-bearingless")

# -- General configuration -------------------------------------------------------------

autodoc_mock_imports = ["numpy", "matplotlib", "scipy", "cycler"]

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "myst_parser",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

myst_enable_extensions = [
    "dollarmath",  # For $...$ and $$...$$ math
    "amsmath",  # For numbered equations
    "deflist",  # For definition lists
    "fieldlist",  # For field lists
    "html_admonition",  # For note/warning boxes
    "colon_fence",  # For ::: fenced blocks
    "attrs_inline",  # For inline attributes
    "substitution",  # For substitutions
]

# MathJax macros, the same as in motulator for the common symbols
mathjax3_config = {
    "loader": {"load": ["[tex]/upgreek"]},
    "tex": {
        "packages": {"[+]": ["upgreek"]},
        "macros": {
            # Operators and functions
            "jj": "\\mathrm{j}",
            "e": "\\mathrm{e}",
            "D": "\\mathrm{d}",
            "T": "\\intercal",
            # Space vectors
            "us": "\\mathbf{u}_\\mathrm{s}",
            "uss": "\\mathbf{u}_\\mathrm{s}^\\mathrm{s}",
            "usref": "\\mathbf{u}_\\mathrm{s}^\\mathrm{ref}",
            "is": "\\mathbf{i}_\\mathrm{s}",
            "iss": "\\mathbf{i}_\\mathrm{s}^\\mathrm{s}",
            "psis": "\\boldsymbol{\\uppsi}_\\mathrm{s}",
            "psiss": "\\boldsymbol{\\uppsi}_\\mathrm{s}^\\mathrm{s}",
            # Space vectors of the two planes, the plane index as the argument
            "ish": ["\\mathbf{i}_{\\mathrm{s}#1}", 1],
            "ush": ["\\mathbf{u}_{\\mathrm{s}#1}", 1],
            "psish": ["\\boldsymbol{\\uppsi}_{\\mathrm{s}#1}", 1],
            # Radial displacement, velocity, and force
            "qM": "\\mathbf{q}_\\mathrm{M}",
            "vM": "\\mathbf{v}_\\mathrm{M}",
            "fM": "\\mathbf{f}_\\mathrm{M}",
            # Resistances and inductances
            "Rs": "R_\\mathrm{s}",
            "Ls": "\\mathbf{L}_\\mathrm{s}",
            # Angular frequencies and mechanical quantities
            "omegam": "\\omega_\\mathrm{m}",
            "omegaM": "\\omega_\\mathrm{M}",
            "thetam": "\\vartheta_\\mathrm{m}",
            "tauM": "\\tau_\\mathrm{M}",
            "np": "n_\\mathrm{p}",
            # Other common symbols
            "Ts": "T_\\mathrm{s}",
            "udc": "u_\\mathrm{dc}",
            "thetaM": "\\vartheta_\\mathrm{M}",
            # PI controller gains and bandwidths
            "kt": "k_\\mathrm{t}",
            "kp": "k_\\mathrm{p}",
            "ki": "k_\\mathrm{i}",
            "alphac": "\\alpha_\\mathrm{c}",
            "alphai": "\\alpha_\\mathrm{i}",
        },
    },
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Links to the documentation of motulator, e.g., {class}`motulator.drive.model.SynchronousMachine`
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "motulator": ("https://aalto-electric-drives.github.io/motulator/", None),
    "motulator_multiphase": (
        "https://aalto-electric-drives.github.io/motulator-multiphase/",
        None,
    ),
}

extensions.append("autoapi.extension")
autoapi_type = "python"
autoapi_dirs = ["../../motulator_bearingless"]
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_template_dir = "../_templates/autoapi"
autoapi_python_class_content = "class"
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_member_order = "alphabetical"

# The examples are sorted by the number of code lines (the default)
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["examples"],
}

exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -----------------------------------------------------------

html_theme = "sphinx_book_theme"
html_theme_options = {"navigation_with_keys": False}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
