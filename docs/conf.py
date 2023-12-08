# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sctk"
copyright = "2023, Teichmann lab"
author = "nh3, Sebastian Lobentanzer, Krzysztof Polanski"
release = "0.2.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# TOC only in sidebar
master_doc = "contents"
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ],
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",  # not for output but to remove warnings
    "sphinxext.opengraph",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "myst_parser",  # markdown support
    "sphinx_rtd_theme",
    "sphinx_design",
]
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "sctk"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": True,
}
