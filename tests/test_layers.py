# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The import layers of ``docs/dev/architecture.md`` section 3, checked on every import.

Each module of ``src/solprop`` has the layer the layers figure draws it in
(``docs/dev/figures/build.py``: ``MODULES`` and ``PACKAGE_WIDE``); a private module the
figure leaves out takes its package's layer, and a private module at the root is
package-wide. An import -- at module level, inside a function or under
``TYPE_CHECKING`` -- may reach its own layer or a lower one, except that processing and
plotting (X) reach translation (T) only where the page names the edge, and never a
backend (B). Every other edge is a deliberate exception, named in :data:`ALLOWED` with
the reason the architecture page gives, and the test fails both on a new upward edge
and on an allowance nothing uses any more. A function-level import is no way round a
layer: that is exactly how an upward edge hides.
"""

import ast
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "solprop"

spec = importlib.util.spec_from_file_location(
    "structure_figures_layers", ROOT / "docs" / "dev" / "figures" / "build.py"
)
assert spec is not None and spec.loader is not None
build = importlib.util.module_from_spec(spec)
# NamedTuple resolves its deferred annotations through sys.modules, so register first.
sys.modules[spec.name] = build
spec.loader.exec_module(build)

WIDE = "W"
"""The package-wide band: importable from every layer, importing only its own band."""

ORDER = (WIDE, *build.LAYER_LETTERS)
"""Bands from the bottom up: an import may reach its own band or one before it."""

GUARDED = {"X": {"T", "B"}}
"""Lower bands a band reaches only through an allowance (section 3: X → T, never B)."""

ROOT_MODULES = {"__main__.py": "D"}
"""Private root modules that belong to a layer rather than to the package as a whole."""

_MEASURE = "the plane measure uses the run's own k_z to the bit (SEP 34)"
_RECORD = "a file's measure or reference is rebuilt by the code that built it (SEP 31)"
_SCANS = (
    "a scan evaluates the reductions processing.scans declares, and returns its reader"
)
_READERS = "an observed plane is read with the same processing readers as a file"
_STOP = "a front-end run stops through the session's stop token"

ALLOWED: dict[tuple[str, str], str] = {
    ("solprop._helpers", "solprop.stepper.multistep"): "a record loads what it names",
    ("solprop._helpers", "solprop.translate.chebyshev"): "a record loads what it names",
    ("solprop.spatial", "solprop.translate.flux"): _MEASURE,
    ("solprop.spatial", "solprop.translate.spatial"): _MEASURE,
    ("solprop.backends.registry", "solprop.calibration.first_use"): (
        "native selection asks whether this machine has a calibration profile"
    ),
    ("solprop.driver._scan_run", "solprop.processing.scans"): _SCANS,
    ("solprop.driver.executors.queue", "solprop.processing.scans"): _SCANS,
    ("solprop.driver.scan", "solprop.processing.scans"): _SCANS,
    ("solprop.recipe.core", "solprop.processing.scans"): _SCANS,
    ("solprop.driver.plane_view", "solprop.processing.fields"): _READERS,
    ("solprop.driver.plane_view", "solprop.processing.planes"): _READERS,
    ("solprop.processing.modal", "solprop.translate.general"): _RECORD,
    ("solprop.processing.modal", "solprop.translate.modal_tables"): _RECORD,
    ("solprop.processing.modal", "solprop.translate.chebyshev"): (
        "a reference table's z span is clamped with the slack the run's tables use"
    ),
    ("solprop.processing.planes", "solprop.translate.coupled"): _RECORD,
    ("solprop.processing.planes", "solprop.translate.flux"): _RECORD,
    ("solprop.processing.planes", "solprop.translate.modal_tables"): _RECORD,
    ("solprop.processing.planes", "solprop.translate.spatial"): _RECORD,
    ("solprop.processing.beams", "solprop.translate.flux"): _RECORD,
    ("solprop.processing._examples", "solprop.translate.general"): (
        "the doctests' general modal run is built as a user builds one"
    ),
    ("solprop.processing._examples", "solprop.translate.modal_tables"): (
        "the doctests' general modal run is built as a user builds one"
    ),
    ("solprop.service.record", "solprop.backends.session"): _STOP,
    ("solprop.service.run", "solprop.backends.session"): _STOP,
    ("solprop.service.sheet", "solprop.backends"): (
        "a run sheet shows the backends available and their thread budget"
    ),
}
"""Deliberate upward edges, ``(importer, imported)`` module names, with their reason."""


def _layer_map() -> tuple[dict[str, str], dict[str, str]]:
    """Return the band of each drawn module and the default band of each package."""
    modules: dict[str, str] = {}
    packages: dict[str, str] = {}
    for letter, boxes in build.MODULES.items():
        for box in boxes:
            for path in box.files():
                modules[path] = letter
            if box.path.endswith("/"):
                # A package drawn in two bands gives its unlisted modules the higher.
                previous = packages.get(box.path, letter)
                packages[box.path] = max(previous, letter, key=ORDER.index)
    for box in build.PACKAGE_WIDE:
        modules[box.path] = WIDE
    return modules, packages


MODULE_LAYERS, PACKAGE_LAYERS = _layer_map()


def _relative(name: str) -> str | None:
    """Return the source path of a solprop module name, or None outside the package."""
    if name != "solprop" and not name.startswith("solprop."):
        return None
    parts = name.split(".")[1:]
    stem = "/".join(parts)
    if not parts:
        return "__init__.py"
    if (PACKAGE / f"{stem}.py").is_file():
        return f"{stem}.py"
    if (PACKAGE / stem / "__init__.py").is_file():
        return f"{stem}/__init__.py"
    return None


def layer_of(path: str) -> str:
    """Return the band of a module given by its path under ``src/solprop``."""
    if path in MODULE_LAYERS:
        return MODULE_LAYERS[path]
    if path in ROOT_MODULES:
        return ROOT_MODULES[path]
    for parent in PurePosixPath(path).parents:
        if f"{parent}/" in PACKAGE_LAYERS:
            return PACKAGE_LAYERS[f"{parent}/"]
    if path.startswith("_") or path == "__init__.py":
        return WIDE
    msg = f"{path} has no layer: draw it in docs/dev/figures/build.py"
    raise AssertionError(msg)


def _module_name(path: Path) -> str:
    """Return the dotted name of a source file under ``src/solprop``."""
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path) -> Iterator[str]:
    """Yield every solprop module a source file imports, wherever the import sits."""
    name = _module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            # `from package import module` names a module; anything else, the base.
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                yield candidate if _relative(candidate) else base


def _violations() -> dict[tuple[str, str], str]:
    """Return every upward or guarded edge in the package, with the bands it crosses."""
    found: dict[tuple[str, str], str] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.relative_to(PACKAGE).as_posix()
        if source == "__init__.py":
            continue  # the root's lazy public imports reach every layer by design
        importer = layer_of(source)
        for imported in _imports(path):
            target = _relative(imported)
            if target is None:
                continue
            reached = layer_of(target)
            if ORDER.index(reached) > ORDER.index(importer) or reached in GUARDED.get(
                importer, ()
            ):
                key = (_module_name(path), imported.removesuffix(".__init__"))
                found[key] = f"{importer} -> {reached}"
    return found


def test_every_module_has_a_layer():
    for path in sorted(PACKAGE.rglob("*.py")):
        layer_of(path.relative_to(PACKAGE).as_posix())


def test_no_import_climbs_a_layer_unless_allowed():
    found = _violations()
    unexpected = {edge: bands for edge, bands in found.items() if edge not in ALLOWED}
    assert unexpected == {}, "upward imports; fix them or add them to ALLOWED"


def test_every_allowance_is_still_used():
    stale = set(ALLOWED) - set(_violations())
    assert stale == set(), "these allowances are no longer needed: remove them"


@pytest.mark.parametrize(
    "module", ["solprop.stats", "solprop.stepper", "solprop.translate"]
)
def test_the_run_specification_loads_no_output_handler(module):
    """The record types below translation do not drag in the handlers above it."""
    code = (
        f"import sys, {module}; "
        "loaded = [m for m in ('h5py', 'solprop.output.hdf5', 'solprop.driver') "
        "if m in sys.modules]; print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
