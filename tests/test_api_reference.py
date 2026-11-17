# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The API reference documents every public name, and only public names.

The reference is one recursive autosummary whose module template decides what each
page documents. A name it leaves out raises no Sphinx warning, so a hole in the
reference is invisible to the docs build. This file renders the template for every
page, with the context the build gives it, reads the autodoc directives it writes, and
compares the objects they document with every public module's ``__all__``.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from sphinx.ext.autosummary.generate import AutosummaryRenderer

DOCS = Path(__file__).resolve().parents[1] / "docs"
spec = importlib.util.spec_from_file_location("api_surface", DOCS / "api_surface.py")
assert spec is not None and spec.loader is not None
api_surface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_surface
spec.loader.exec_module(api_surface)

DIRECTIVE = re.compile(r"^\.\. (auto\w+):: (\S+)$")


@pytest.fixture(scope="module")
def pages():
    """Return each page's module and the reStructuredText the template writes for it."""
    session_loaded = "solprop.sol" in sys.modules
    modules = list(api_surface.public_modules())
    if not session_loaded:
        # Importing the session namespace installs its logging and warning filter;
        # the suite tests the library's defaults (conftest.py says why).
        sys.modules["solprop.sol"].end_session()
    renderer = AutosummaryRenderer(
        SimpleNamespace(  # pyright: ignore[reportArgumentType] - the renderer reads these
            srcdir=DOCS,
            config=SimpleNamespace(templates_path=["_templates"]),
            translator=None,
        )
    )
    members = api_surface.package_members()
    rendered = []
    for module in modules:
        submodules = sorted(
            leaf
            for parent, _, leaf in (other.__name__.rpartition(".") for other in modules)
            if parent == module.__name__
        )
        context = {
            "fullname": module.__name__,
            "modules": submodules,
            "package_members": members,
            "attributes": [],
            "functions": [],
            "classes": [],
            "exceptions": [],
        }
        rendered.append((module, renderer.render("autosummary/module.rst", context)))
    return rendered


def _documented(module: ModuleType, text: str) -> list[tuple[str, object]]:
    """The names and objects one rendered page documents."""
    lines = text.splitlines()
    documented = []
    for index, line in enumerate(lines):
        match = DIRECTIVE.match(line.strip())
        if match is None:
            continue
        directive, target = match.groups()
        if directive == "automodule":
            options = []
            for option in lines[index + 1 :]:
                if not option.strip().startswith(":"):
                    break
                options.append(option.strip())
            if ":no-members:" not in options:
                documented += [(name, getattr(module, name)) for name in module.__all__]
        elif directive != "autosummary":
            home, name = target.rsplit(".", 1)
            documented.append((name, getattr(sys.modules[home], name)))
    return documented


def test_the_reference_documents_exactly_the_public_names(pages):
    public = {
        api_surface.object_key(name, getattr(module, name)): f"{module.__name__}.{name}"
        for module, _ in pages
        for name in getattr(module, "__all__", ())
        # A module's page is its own documentation; `__version__` is metadata.
        if not isinstance(getattr(module, name), ModuleType)
        and not name.startswith("__")
    }
    documented = {
        api_surface.object_key(name, value): name
        for module, text in pages
        for name, value in _documented(module, text)
    }
    assert len(public) > 1000  # the walk found the package's surface
    assert len(documented) > 1000
    missing = sorted(public[key] for key in public.keys() - documented.keys())
    assert missing == [], "public names the API reference does not document"
    extra = sorted(documented[key] for key in documented.keys() - public.keys())
    assert extra == [], "names the API reference documents that no __all__ exports"


def test_a_package_page_documents_the_names_only_it_can(pages):
    """The package pages that own names write them; the rest stay module lists."""
    text = {module.__name__: text for module, text in pages}
    assert (
        ".. autofunction:: solprop.calibration.calibrate" in text["solprop.calibration"]
    )
    assert (
        ".. autoclass:: solprop.driver.executors.Executor"
        in text["solprop.driver.executors"]
    )
    assert (
        ".. autofunction:: solprop.processing.example_output"
        in text["solprop.processing"]
    )
    assert ".. autoclass::" not in text["solprop.modes"]
