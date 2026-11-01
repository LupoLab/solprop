# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sphinx configuration for the solprop manual.

Markdown throughout (myst-parser), NumPy-style docstrings rendered by numpydoc, a
recursive autosummary API reference, executed examples via sphinx-gallery, and the
developer's guide under ``docs/dev/``. Warnings are errors and missing references are
warnings (``-W -n``), so every page must build clean.
"""

import faulthandler
import importlib
import importlib.util
import os
import re
import sys
import warnings
from importlib.metadata import version as _version
from pathlib import Path
from typing import Any

from sphinx.errors import ConfigError
from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.utils import get_md5sum

# Read the Docs discards the log of a build it stops at its time limit. There
# `docs/build-on-read-the-docs.sh` sets SOLPROP_DOCS_WATCHDOG to a delay in seconds
# after which every thread's stack is written to the log, before the build is stopped.
if _watchdog := os.environ.get("SOLPROP_DOCS_WATCHDOG"):
    faulthandler.dump_traceback_later(float(_watchdog), exit=False)

# Gallery examples may select the native backend automatically. Load its OpenMP runtime
# before autodoc or a tutorial imports pyFFTW, as the test configuration does.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

# The mlx backend's optional dependency is macOS-only (SEP 33): elsewhere autodoc reads
# its modules against a mock of MLX.
autodoc_mock_imports = [] if importlib.util.find_spec("mlx") is not None else ["mlx"]

# Initialise the reference backend through its package entry point before
# autosummary visits its implementation modules in an arbitrary order. Importing
# populations first encounters the package's existing circular initialisation.
importlib.import_module("solprop.backends.numpy_ref")

project = "solprop"
author = "John C. Travers and contributors"
copyright = "2026, John C. Travers and contributors"
release = _version("solprop")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = [
    "_build",
    "_inventories",
    "Thumbs.db",
    ".DS_Store",
    # Consumed by sphinx-gallery, not pages of their own.
    "introduction/tutorials/README.rst",
    "introduction/lessons/index.rst",  # lessons sit directly in the introduction
    "tutorials/README.rst",
    "tutorials/*/README.rst",
    "examples/README.rst",
    "examples/*/README.rst",
]

# --- myst
# ------------------------------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "attrs_inline",
]
myst_heading_anchors = 3

# --- API reference
# ---------------------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
# Source links belong to the defining module. Following re-exported aliases makes
# viewcode reuse one alias prefix with another page's anchors, breaking backlinks.
# The complete API still documents aliases; their home modules retain source links.
viewcode_follow_imported_members = False
autodoc_typehints = "description"
# numpydoc renders the parameters itself, as one `:Parameters:` field that autodoc
# does not recognise, so with the default target ("all") autodoc appends a second,
# duplicate set of `:type:` fields built from the annotations. For jaxtyping shapes
# (ADR-018) those stringify to `Inexact[ndarray, 'n_deg *value']`, whose every token
# the Python domain then tries to resolve as a class -- unresolvable, and `nitpicky`
# turns each into a `-W` failure. "documented" types only what the docstring already
# names, which is exactly what numpydoc gives us.
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = False

# --- cross-references
# ------------------------------------------------------------------
# Each inventory is fetched from the project's site first and, failing that, read from
# the vendored copy under docs/_inventories/, so the build works offline and survives a
# flaky mirror (docs.scipy.org timed out during a review). Refresh the copies
# occasionally with `curl -o docs/_inventories/<name>.inv <url>objects.inv`.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", (None, "_inventories/python.inv")),
    "numpy": ("https://numpy.org/doc/stable/", (None, "_inventories/numpy.inv")),
    "scipy": ("https://docs.scipy.org/doc/scipy/", (None, "_inventories/scipy.inv")),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        (None, "_inventories/matplotlib.inv"),
    ),
    "h5py": ("https://docs.h5py.org/en/stable/", (None, "_inventories/h5py.inv")),
    "numba": (
        "https://numba.readthedocs.io/en/stable/",
        (None, "_inventories/numba.inv"),
    ),
}
intersphinx_timeout = 10
nitpicky = True
nitpick_ignore = [
    ("py:class", "logging.Logger"),
    ("py:mod", "pyfluids"),  # no published inventory
    # The private base of the two tabulated hollow modes. `show-inheritance`
    # emits it under "Bases:" for both subclasses; documenting it to satisfy the
    # reference would publish an underscore name in the API reference and its
    # search index, which the module's __all__ deliberately excludes.
    ("py:class", "solprop.modes.tabulated._TableMode"),
    # The private wall-pair base RectMode and SlabMode share, for
    # the same reason.
    ("py:class", "solprop.modes.rectangular._WallPairFill"),
    # The private mixin that gives a class its printed block. It is a base
    # of every class that prints specially, so `show-inheritance` names it; it is not
    # part of the API reference for the same reason as above.
    ("py:class", "solprop._display.Displayed"),
    # The private record of a backend's per-stage and per-step helpers (SEP 29),
    # which the shared stage and Adams functions take as a keyword and StepDriver
    # carries as a class attribute; private for the same reason.
    ("py:class", "solprop._helpers.Helpers"),
    ("py:class", "Helpers"),
    # MLX publishes no inventory Sphinx can resolve (the mlx backend, SEP 33).
    ("py:class", "mlx.core.array"),
]
# Array annotations are written with jaxtyping on numpy arrays (ADR-018). Neither
# jaxtyping nor numpy's private typing aliases publish an inventory Sphinx can resolve,
# and a jaxtyping shape specification is a string that Sphinx tries to read as a class.
# The style is mandated by AGENTS.md, so these are ignored rather than avoided.
nitpick_ignore_regex = [
    ("py:class", r"^jaxtyping\..*"),
    ("py:class", r"^numpy\._typing\..*"),
    ("py:class", r"^ndarray$"),
    ("py:class", r"^'.*'$"),
]

# --- gallery
# ---------------------------------------------------------------------------
# Three galleries, each with committed output: the introductory lessons
# (`introduction/tutorials/` → `introduction/lessons/`), which run developed physical
# problems; the focused tutorials (`tutorials/` → `auto_tutorials/`); and the worked
# examples (`examples/` → `auto_examples/`), full-size runs of minutes to half an hour
# each. A normal build executes none of them: their generated pages, figures and
# sphinx-gallery checksums are committed, and sphinx-gallery reuses a script's stored
# output unchanged whenever the source's checksum matches the stored one. The build
# that publishes the manual therefore runs no propagation (Read the Docs' builders
# stalled on the lessons and tutorials from 2026-10-29).
#
# `SOLPROP_RUN_EXAMPLES` executes them. "stale" (`poe docs-examples`) runs every
# script, in any gallery, whose source no longer matches its stored output; a script
# name (`plot_03_...`, with or without `.py`, without its directory) re-runs that one
# even if it is current; commit what either writes. "tutorials" (CI's docs job) runs
# every lesson and tutorial again, current or not, so that each change still executes
# them, and writes nothing that is committed.
#
# A normal build, and a "tutorials" run, refuse a stale script rather than showing it:
# an edited script beside the output of its previous version would be a page whose
# code and figures disagree, with nothing on it to say so, and rendering it without
# output would overwrite the committed page in the working tree. The refusal names the
# command that refreshes it (`_check_committed_galleries` below).
_DOCS = Path(__file__).parent
sys.path.insert(0, str(_DOCS))  # gallery_session, named below
_GALLERIES = (
    (_DOCS / "introduction" / "tutorials", _DOCS / "introduction" / "lessons"),
    (_DOCS / "tutorials", _DOCS / "auto_tutorials"),
    (_DOCS / "examples", _DOCS / "auto_examples"),
)
_RUN_EXAMPLES = os.environ.get("SOLPROP_RUN_EXAMPLES", "").strip()

# sphinx-gallery matches these against native paths, so a separator is either slash.
if not _RUN_EXAMPLES:
    _filename_pattern = r"$^"  # matches no script: every page is the committed one
    _run_stale = False
elif _RUN_EXAMPLES == "stale":
    _filename_pattern = r"[\\/](tutorials|examples)[\\/](.+[\\/])?plot_"
    _run_stale = False
elif _RUN_EXAMPLES == "tutorials":
    _filename_pattern = r"[\\/]tutorials[\\/](.+[\\/])?plot_"
    _run_stale = True  # every lesson and tutorial, current or not
else:
    _name = Path(_RUN_EXAMPLES.removesuffix(".py")).name
    if not any(any(source.rglob(f"{_name}.py")) for source, _ in _GALLERIES):
        msg = f"SOLPROP_RUN_EXAMPLES={_RUN_EXAMPLES!r}: no gallery script {_name}.py"
        raise FileNotFoundError(msg)
    _filename_pattern = (
        rf"[\\/](tutorials|examples)[\\/](.+[\\/])?{re.escape(_name)}\.py$"
    )
    _run_stale = True  # re-run it even though its checksum is current

# A warning that a gallery script raises is committed with its output, so it names its
# category and message, not the path of the file that raised it on the machine that ran
# the script.


def _format_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    line: str | None = None,
) -> str:
    """Format a warning by its category and message alone (see above)."""
    del filename, lineno, line
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _format_warning

sphinx_gallery_conf = {
    "examples_dirs": [str(source.relative_to(_DOCS)) for source, _ in _GALLERIES],
    "gallery_dirs": [str(output.relative_to(_DOCS)) for _, output in _GALLERIES],
    # Each gallery is split into subsections, in this order; within a subsection the
    # scripts run in the order of their file names.
    "subsection_order": ExplicitOrder(
        [
            "tutorials/basics",
            "tutorials/capillaries",
            "tutorials/waveguides",
            "tutorials/free_space",
            "tutorials/tools",
            "examples/processes",
            "examples/supercontinuum",
            "examples/waveguides",
            "examples/performance",
        ]
    ),
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": _filename_pattern,
    "run_stale_examples": _run_stale,
    "ignore_pattern": r"__init__\.py",
    "download_all_examples": False,
    "remove_config_comments": True,
    "reference_url": {"solprop": None},
    "reset_modules": ("matplotlib", "seaborn", "gallery_session.leave_session"),
    "show_signature": False,
}


def _stale_gallery_scripts() -> list[str]:
    """Return the gallery scripts whose committed output is not of their source."""
    stale = []
    for source_dir, output_dir in _GALLERIES:
        for source in sorted(source_dir.rglob("plot_*.py")):
            output = output_dir / source.parent.relative_to(source_dir)
            stored = output / f"{source.name}.md5"
            page = output / f"{source.stem}.rst"
            if (
                not stored.is_file()
                or not page.is_file()
                or stored.read_text().strip() != get_md5sum(source, mode="t")
            ):
                stale.append(str(source.relative_to(_DOCS)))
    return stale


def _check_committed_galleries(app: object) -> None:
    """Refuse a build that would show a stale gallery script's committed output."""
    del app
    if _RUN_EXAMPLES not in ("", "tutorials"):
        return
    stale = _stale_gallery_scripts()
    if stale:
        names = ", ".join(stale)
        msg = (
            f"gallery script(s) {names} changed since their committed output was "
            "generated, and this docs build executes no gallery script to replace it. "
            "Refresh with `uv run poe docs-examples` (every stale script) or "
            "`uv run poe docs-examples <name>` (one), then commit what it writes under "
            "docs/introduction/lessons/, docs/auto_tutorials/ or docs/auto_examples/. "
            "See AGENTS.md, Documentation."
        )
        raise ConfigError(msg)


_GENERATED_PAGES = (
    "introduction/lessons/",
    "auto_tutorials/",
    "auto_examples/",
    "api/generated/",
)


def _no_edit_link_on_generated_pages(
    _app: Any, pagename: str, _template: str, context: dict[str, Any], _doctree: Any
) -> None:
    # A gallery page comes from a tutorials/ or examples/ script and an API page from
    # docstrings: neither has a source file under docs/ for "Edit on GitHub" to open.
    if pagename.startswith(_GENERATED_PAGES):
        context["theme_use_edit_page_button"] = False


def setup(app: Any) -> None:
    # Before sphinx-gallery's own builder-inited handler (priority 500), so a stale
    # script fails the build before any page is written.
    app.connect("builder-inited", _check_committed_galleries, priority=100)
    app.connect("html-page-context", _no_edit_link_on_generated_pages)


# --- plot directive
# --------------------------------------------------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 120)]

# --- theme
# -----------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "solprop"
html_theme_options = {
    "navigation_with_keys": False,
    "show_prev_next": True,
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LupoLab/solprop",
            "icon": "fa-brands fa-github",
        }
    ],
    "use_edit_page_button": True,
}
# The "Edit on GitHub" link of each page.
html_context = {
    "github_user": "LupoLab",
    "github_repo": "solprop",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path = ["_static"]
# The structure figures span the window on their page alone.
html_css_files = ["structure.css"]
