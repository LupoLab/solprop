# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The example recipes are the gallery's calls (SEP 41 section 7, recipe equivalence).

Each example recipe names the gallery script it mirrors in a comment line of a fixed
form, ``# mirrors: docs/<path>.py``. The script is run until its first propagation,
which a stand-in for ``Simulation.run`` stops before any step, and the simulation it
built must have the configuration hash of the recipe's build: the same physics, seed
included. The recipe's Python export, run in a fresh interpreter, must build that
simulation too, with the same execution settings, which the hash does not cover.

A gallery script that calls a route a recipe describes and has no recipe is listed in
:data:`NO_RECIPE` with the reason, and the list is checked against the gallery so that
it stays true when a script is added or changed.

The default lane builds one example of each route that has one (:data:`DEFAULT_LANE`);
no gallery script on the waveguide route has a recipe yet. The slow lane builds them
all. The comparisons are of digests, so they are exact; no tolerance enters,
and the examples carry the scripts' seeds. Warnings are
not what these tests are about: several scripts warn by design (a gas index asked for
beyond its fitted range, a coherent wall near its quasi-static limit) or silence their
own, and the builds here ignore them all.
"""

import ast
import json
import logging
import re
import runpy
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.random.bit_generator
import pytest

from solprop.driver.simulation import Simulation
from solprop.recipe import Recipe

ROOT: Final = Path(__file__).resolve().parents[2]

GALLERY: Final = ("docs/introduction/tutorials", "docs/tutorials", "docs/examples")
"""The gallery directories, whose scripts the examples mirror."""

ROUTE_CALLS: Final = frozenset(
    {
        "prop_gnlse",
        "prop_capillary",
        "prop_waveguide",
        "from_gnlse",
        "from_capillary",
        "from_waveguide",
    }
)
"""The calls that put a script on a route a recipe of format 1 describes."""

MIRRORS: Final = re.compile(r"^# mirrors: (?P<script>docs/\S+\.py)$", re.MULTILINE)
"""The comment line by which an example recipe names the gallery script it mirrors."""

ENTROPY: Final = 0x5EED_0F_A11_5C415
"""What an unseeded draw takes here, so that two unseeded builds can be compared.

Two scripts draw their input noise without a seed, as their readers would. Their
recipes leave the seed out too, and each build would draw a different one; with the
operating system's entropy replaced by this constant, both draw the same.
"""

_PROVIDER: Final = (
    "the guide is a ModalGuide built from a provider at knots taken from the grid, on "
    "the provider's own node rule: a provider has no table form"
)
_PHASE_MATCHING: Final = (
    "the pressure is a numerical phase-matching root (brentq on the HE11 mismatch): "
    "no pin target expresses it, and a literal would freeze a float the platform's "
    "mathematics library can move in its last digit"
)

NO_RECIPE: Final[dict[str, str]] = {
    "docs/examples/processes/plot_08_duv_fwm_dispersive_kerr.py": _PHASE_MATCHING,
    "docs/examples/waveguides/plot_04_graded_index_multimode.py": (
        f"{_PROVIDER}; the core's permittivity is a class of the script's own, and the "
        "launch an incident beam"
    ),
    "docs/examples/waveguides/plot_05_moving_solid_guide.py": (
        f"{_PROVIDER}, on a radius path; the GNLSE call takes fibre=SpecifiedMode"
    ),
    "docs/tutorials/basics/plot_05_pressure_energy_scan.py": (
        "each scan point is built with Simulation.with_mixture and with_input"
    ),
    "docs/tutorials/capillaries/plot_11_dispersive_kerr.py": _PHASE_MATCHING,
    "docs/tutorials/free_space/plot_08_radial_capillary_coupling.py": (
        "the capillary is launched with Simulation.with_input from a coupled "
        "free-space plane, and the script's first runs are radial free space"
    ),
    "docs/tutorials/waveguides/plot_02_tabulated_modes.py": (
        "the capillary takes TabulatedHollowMode objects and the GNLSE fibre an "
        "IndexTable, neither registered"
    ),
    "docs/tutorials/waveguides/plot_03_stepindex_rectangular.py": (
        "the GNLSE call takes fibre=StepIndexMode, not registered, and the "
        "rectangular guide's cladding is aluminium(), a material with no table form"
    ),
    "docs/tutorials/waveguides/plot_05_general_modal.py": (
        f"{_PROVIDER}; the GNLSE calls take a mode channel as fibre, not registered"
    ),
    "docs/tutorials/waveguides/plot_06_graded_index.py": (
        f"{_PROVIDER}, and the run is launched with Simulation.with_input"
    ),
    "docs/tutorials/waveguides/plot_07_split_block.py": (
        f"{_PROVIDER}, and declares a ChannelBasis"
    ),
    "docs/tutorials/waveguides/plot_08_planar.py": (
        "the slab modes are launched with Simulation.with_input, at shares from "
        "gaussian_coupling"
    ),
}
"""Each gallery script on a route a recipe describes without a recipe, and why."""


DEFAULT_LANE: Final = frozenset({"overview", "soliton"})
"""The examples the default lane builds, one per route: each in about a second.

Building every example costs a translation per build, three per example, about two
minutes of processor time in all and far beyond the medium tier's budget
(``docs/dev/testing.md`` section 2), so the rest are the slow lane's, as the guide's
snippets are.
"""


def gallery_scripts() -> Iterator[str]:
    """Yield every script of the gallery, relative to the repository, with ``/``."""
    for directory in GALLERY:
        for path in sorted((ROOT / directory).rglob("*.py")):
            yield path.relative_to(ROOT).as_posix()


def calls_a_route(script: str) -> bool:
    """Return whether a script calls a route a recipe describes, in its code."""
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            if name in ROUTE_CALLS:
                return True
    return False


def mirrored(name: str) -> str:
    """Return the gallery script an example recipe names."""
    text = Recipe.example(name).to_text()
    found = MIRRORS.findall(text)
    assert len(found) == 1, f"{name} names {len(found)} scripts"
    return found[0]


EXAMPLES: Final = Recipe.examples()


def _tiered(name: str, *marks: pytest.MarkDecorator) -> object:
    """Return an example as a parameter, in the tier its cost puts it in."""
    tier = pytest.mark.medium if name in DEFAULT_LANE else pytest.mark.slow
    return pytest.param(name, marks=(tier, *marks), id=name)


class _Captured(Exception):  # noqa: N818 - a signal, not an error
    """Raised in place of a run, carrying the simulation that was about to run."""

    def __init__(self, simulation: Simulation) -> None:
        super().__init__("the run was stopped before its first step")
        self.simulation = simulation


def _stop(self: Simulation, *args: object, **kwargs: object) -> object:
    """Stand in for ``Simulation.run``: stop the script with the simulation."""
    raise _Captured(self)


@contextmanager
def _package_logging() -> Iterator[None]:
    """Restore the package logger after a script that configures it."""
    logger = logging.getLogger("solprop")
    level, handlers = logger.level, list(logger.handlers)
    try:
        yield
    finally:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(level)


def first_simulation(script: str) -> Simulation:
    """Run a gallery script up to its first run and return what it would run."""
    path = ROOT / script
    with _package_logging(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            runpy.run_path(str(path), run_name="__main__")
        except _Captured as captured:
            return captured.simulation
        finally:
            plt.close("all")
    pytest.fail(f"{script} finished without a run")


def built(recipe: Recipe) -> Simulation:
    """Build a recipe, ignoring the warnings its build gives by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return recipe.build()


def record(simulation: Simulation) -> list[object]:
    """Return the configuration hash and the execution settings, as JSON holds them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        digest = simulation.metadata().configuration_hash()
    settings = simulation.settings
    return [
        digest,
        settings.fields,
        list(settings.field_planes),
        settings.resume_history,
    ]


@pytest.fixture
def gallery_run(monkeypatch, tmp_path):
    """Make a gallery script stop at its first run, write nowhere, and draw fixed."""
    mpl.use("Agg")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setattr(Simulation, "run", _stop)
    monkeypatch.setattr(numpy.random.bit_generator, "randbits", lambda _: ENTROPY)


def test_every_example_names_one_gallery_script_that_exists():
    for name in EXAMPLES:
        assert (ROOT / mirrored(name)).is_file(), name


def test_the_gallery_s_route_calls_are_the_examples_and_the_listed_exceptions():
    on_route = {script for script in gallery_scripts() if calls_a_route(script)}
    examples = {mirrored(name) for name in EXAMPLES}
    assert examples.isdisjoint(NO_RECIPE)
    assert on_route == examples | set(NO_RECIPE)


def test_two_examples_do_not_mirror_one_script():
    scripts = [mirrored(name) for name in EXAMPLES]
    assert len(scripts) == len(set(scripts))


@pytest.mark.parametrize("name", [_tiered(name) for name in EXAMPLES])
@pytest.mark.usefixtures("gallery_run")
def test_an_example_builds_its_script_s_simulation(name):
    captured = first_simulation(mirrored(name))
    assert record(built(Recipe.example(name))) == record(captured)


_PROBE: Final = textwrap.dedent(
    """\
    import json
    import sys
    import warnings

    import numpy.random.bit_generator

    from solprop.driver.simulation import Simulation

    numpy.random.bit_generator.randbits = lambda _: {entropy}
    warnings.simplefilter("ignore")


    def run(self, *args, **kwargs):
        settings = self.settings
        print(json.dumps([
            self.metadata().configuration_hash(),
            settings.fields,
            list(settings.field_planes),
            settings.resume_history,
        ]))
        sys.exit(0)


    Simulation.run = run
    """
)
"""Prepended to an exported script: print what the run would be, instead of running."""


@pytest.mark.parametrize(
    "name",
    [_tiered(name) for name in EXAMPLES],
)
def test_an_example_s_python_export_builds_the_same_run(name, tmp_path, monkeypatch):
    monkeypatch.setattr(numpy.random.bit_generator, "randbits", lambda _: ENTROPY)
    recipe = Recipe.example(name)
    path = tmp_path / f"{name}.py"
    path.write_text(_PROBE.format(entropy=ENTROPY) + recipe.to_python())
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == record(built(recipe))
