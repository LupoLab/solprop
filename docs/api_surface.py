# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The public names that only a package page of the API reference can document.

The reference's module template (``_templates/autosummary/module.rst``) documents a
leaf module's ``__all__`` in full and gives a package page its module list. A package
page leaves out the names its ``__all__`` re-exports from a public submodule, whose own
page documents them. A name whose home is the package's ``__init__`` or a private
module (``solprop.calibration.calibrate``, ``solprop.processing.example_output``) has
no such page, so the template documents it on the package page, from the table
:func:`package_members` gives ``conf.py``. ``tests/test_api_reference.py`` holds the
rendered template to every public module's ``__all__``.
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType

__all__ = [
    "NAMESPACES",
    "UNDOCUMENTED_TREES",
    "object_key",
    "package_members",
    "public_modules",
]

#: Packages whose modules get no page of their own: the front end's service changes
#: without notice (the template says why).
UNDOCUMENTED_TREES = ("solprop.service",)

#: Leaf modules that gather names documented on their home pages: their page documents
#: only the functions they define themselves.
NAMESPACES = ("solprop.sol",)


def public_modules(root: str = "solprop") -> Iterator[ModuleType]:
    """Yield the modules the recursive autosummary gives a page.

    Parameters
    ----------
    root : str, optional
        The package the reference starts from.

    Yields
    ------
    module
        ``root``, then every submodule whose name has no leading underscore, depth
        first, as autosummary walks them; nothing below :data:`UNDOCUMENTED_TREES`.
    """
    module = importlib.import_module(root)
    yield module
    if hasattr(module, "__path__") and root not in UNDOCUMENTED_TREES:
        for _, name, _ in pkgutil.iter_modules(module.__path__):
            if not name.startswith("_"):
                yield from public_modules(f"{root}.{name}")


def object_key(name: str, value: object) -> object:
    """Return what identifies a public name's object across the pages that list it.

    A function or class is the same object under every name it is exported by; any
    other value (a number, a string) is matched by its name as well, since equal small
    values may be one object.

    Parameters
    ----------
    name : str
        The name a module exports it under.
    value : object
        The object.

    Returns
    -------
    object
        A hashable key.
    """
    if hasattr(value, "__qualname__") and not isinstance(value, ModuleType):
        return id(value)
    return (name, id(value))


def _leaf_page_keys(module: ModuleType) -> set[object]:
    """Keys of the objects a leaf module's page documents: its ``__all__``."""
    keys = set()
    for name in getattr(module, "__all__", ()):
        value = getattr(module, name)
        home = getattr(value, "__module__", None)
        if module.__name__ in NAMESPACES and home != module.__name__:
            continue
        keys.add(object_key(name, value))
    return keys


def _directive(value: object) -> str:
    """Name the autodoc directive that documents an object."""
    if inspect.isclass(value):
        return "autoexception" if issubclass(value, BaseException) else "autoclass"
    return "autofunction" if callable(value) else "autodata"


def package_members(root: str = "solprop") -> dict[str, list[tuple[str, str]]]:
    """Return, for each package page, the names only that page can document.

    These are the names of a package's ``__all__`` that no leaf module's page
    documents: those defined in the package's ``__init__`` or in a private module. A
    name two packages export is documented once, on the deeper page. Dunder names
    (``__version__``) are package metadata and are left out.

    Parameters
    ----------
    root : str, optional
        The package the reference starts from.

    Returns
    -------
    dict of str to list of (str, str)
        Package name to the ``(directive, name)`` pairs its page documents, sorted by
        name, with ``directive`` the autodoc directive (``"autoclass"``,
        ``"autofunction"``, ...); a package with none is absent.
    """
    modules = list(public_modules(root))
    documented = set().union(
        *(
            _leaf_page_keys(module)
            for module in modules
            if not hasattr(module, "__path__")
        )
    )
    packages = [
        module
        for module in modules
        if hasattr(module, "__path__") and module.__name__ not in UNDOCUMENTED_TREES
    ]
    members: dict[str, list[tuple[str, str]]] = {}
    for package in sorted(packages, key=lambda module: -module.__name__.count(".")):
        for name in sorted(getattr(package, "__all__", ())):
            value = getattr(package, name)
            key = object_key(name, value)
            if (
                name.startswith("__")
                or isinstance(value, ModuleType)
                or key in documented
            ):
                continue
            documented.add(key)
            members.setdefault(package.__name__, []).append((_directive(value), name))
    return members
