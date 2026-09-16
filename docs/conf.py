# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Mirrors the sibling `gert` repo's own docs/conf.py (gert/docs/conf.py) for
# consistency -- gert is the RT/retrieval library, geocarb_simulator is the
# downstream project built on top of it (see gert/docs/PACKAGING.md's own
# "two layers" framing).

import os
import sys

# Make the project root importable so autodoc can find geocarb_gert.*/scripts.*
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../scripts'))

# ── Project information ────────────────────────────────────────────────────────
project   = 'geocarb_simulator'
copyright = '2026, Sean Crowell'
author    = 'Sean Crowell'
release   = '0.1'

# ── General configuration ──────────────────────────────────────────────────────
extensions = [
    'sphinx.ext.autodoc',       # core: pull docstrings from source
    'sphinx.ext.autosummary',   # summary tables at the top of each module page
    'sphinx.ext.napoleon',      # NumPy- and Google-style docstring support
    'sphinx.ext.viewcode',      # add [source] links next to each item
    'sphinx.ext.intersphinx',   # cross-link to NumPy, SciPy, Python docs
    'myst_parser',              # render the existing hand-written Markdown
                                 # docs (PROJECT_STATUS.md, ALGORITHM_ROADMAP.md,
                                 # README.md) as real pages, not just source
                                 # files nobody browsing the site would see.
]

# napoleon settings — NumPy-style docstrings (matches this codebase's own
# convention throughout, same as gert's)
napoleon_numpy_docstring          = True
napoleon_google_docstring         = False
napoleon_include_init_with_doc    = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = False
napoleon_use_ivar                 = True    # avoids duplicate-object warnings for dataclass fields
napoleon_attr_annotations         = True

# 2026-09-16: `gert` itself is a SEPARATE, still-private GitHub repo
# (erselab/gert) -- a public CI workflow can't install it without a
# credentialed step. `autodoc_mock_imports` (mirroring gert's own mocking
# of its unbuildable-in-CI C extensions, mie/gert_rt) is the right tool
# for MOST of this -- Sphinx's own Mock() gracefully absorbs arbitrary
# attribute/method access (e.g. geocarb_gert/instrument.py's eager,
# module-level `GEOCARB_BANDS = _GeoCarbInstrumentConfig.from_yaml().bands`
# chain calls real-looking methods like `ILS.from_fwhm_nm(...)` while
# building that constant -- a Mock() tolerates this fine; a hand-written
# stub class with no such method does not, tried and reverted).
#
# The ONE real exception: geocarb_gert/levels.py computes
# `SIGMA = GERT_P_LEVELS / GERT_P_SFC_STD` eagerly at import time --
# feeding a Mock() into np.asarray() raises a confusing `ValueError:
# invalid __array_struct__`, not a clean, catchable error, since Mock()
# doesn't properly implement numpy's array protocol. Fixed by pre-caching
# a REAL (if fake) `gert.levels` in sys.modules, immediately below,
# BEFORE Sphinx's own mock machinery ever gets a chance to intercept that
# one submodule -- Python's import system checks sys.modules first, so
# this wins over the mock for gert.levels specifically while every OTHER
# gert.* submodule still gets the blunt, tolerant Mock().
autodoc_mock_imports = ['gert', 'xrtm']

sys.path.insert(0, os.path.abspath('_stubs'))
import gert.levels  # noqa: E402  (pre-cache the real stub -- see comment above)

# autodoc behaviour
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,   # hide members that have no docstring
    'private-members':  False,
    'show-inheritance': True,
    'member-order':     'bysource',
}
autodoc_typehints        = 'description'   # render type hints in the description
autodoc_typehints_format = 'short'         # e.g. ndarray instead of numpy.ndarray

# autosummary: generate stub .rst files automatically
autosummary_generate = True

# intersphinx: link to external package docs. No `gert` entry yet -- its own
# docs aren't hosted anywhere; add once they are.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/', None),
}

# myst-parser: let Markdown files be included directly as pages, matching a
# plain .rst toctree entry.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md':  'markdown',
}
myst_heading_anchors = 3

templates_path   = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ── HTML output ────────────────────────────────────────────────────────────────
html_theme = 'furo'

html_theme_options = {
    'sidebar_hide_name': False,
    'light_css_variables': {
        'color-brand-primary':    '#0d6efd',
        'color-brand-content':    '#0d6efd',
    },
}

html_static_path = ['_static']
html_title       = 'geocarb_simulator'
