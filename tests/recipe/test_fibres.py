# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes for solid-core fibres: a step-index mode as the GNLSE fibre."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.modes import StepIndexMode
from solprop.physdata import glass
from solprop.physdata.glasses import GlassIndex
from solprop.recipe import Recipe, RecipeError, RecipeIssue, parameters

# The padded band reaches past 2.3 um, beyond where a glass is transparent or fitted;
# what is tested here is the recipe's form, and the plot_03 tutorial keeps the warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

FIBRE = (
    'fibre = { type = "StepIndexMode", radius = "2 um", profile_wavelength = "1 um" }'
)

GNLSE = textwrap.dedent(
    """\
    [recipe]
    format = 1
    route = "gnlse"

    [build]
    gamma = "none"
    length = "1 mm"
    {fibre}
    n2 = 2.6e-20
    wavelength = "1 um"
    wavelength_limits = ["800 nm", "1.3 um"]
    trange = "200 fs"
    fwhm = "30 fs"
    energy = "10 pJ"
    shape = "sech"
    n_save = 2
    noise = "none"
    backend = "numpy_ref"
    """
)


def recipe_of(fibre: str = FIBRE) -> Recipe:
    return Recipe.loads(GNLSE.replace("{fibre}", fibre))


def errors_of(recipe: Recipe) -> list[RecipeIssue]:
    return [issue for issue in recipe.check() if issue.severity == "error"]


@pytest.mark.medium
def test_a_step_index_fibre_is_the_same_run_as_the_script():
    built = recipe_of().build()
    script = Simulation.from_gnlse(
        None,
        1e-3,
        fibre=StepIndexMode(2e-6, profile_wavelength=1e-6),
        n2=2.6e-20,
        wavelength=1e-6,
        wavelength_limits=(800e-9, 1.3e-6),
        trange=200e-15,
        fwhm=30e-15,
        energy=10e-12,
        shape="sech",
        n_save=2,
        noise=None,
        backend="numpy_ref",
    )
    assert (
        built.metadata().configuration_hash() == script.metadata().configuration_hash()
    )


@pytest.mark.medium
def test_the_recipe_checks_clean():
    assert not errors_of(recipe_of())


@pytest.mark.slow
def test_the_preview_lists_the_fibre_s_mode():
    preview = recipe_of().modes()
    assert len(preview.modes) == 1
    # A guided HE11 lies between the two indices: silica's at 1 um, about 1.45, and air.
    index = preview.modes[0].neff_minus_one + 1.0
    assert np.all(index > 1.0) and np.all(index < 1.5)
    assert preview.modes[0].aeff > 0


@pytest.mark.parametrize(
    ("core", "cladding"),
    [
        ("1.47", "1.45"),
        ('{ type = "glass", name = "BK7" }', "1.0"),
        ('{ type = "glass", name = "sapphire", axis = "o" }', "1.0"),
    ],
    ids=["indices", "glass_core", "crystal_core"],
)
def test_the_materials_are_an_index_or_a_glass(core, cladding):
    fibre = (
        'fibre = { type = "StepIndexMode", radius = "2 um", '
        f'profile_wavelength = "1 um", core = {core}, cladding = {cladding} }}'
    )
    built = recipe_of(fibre).keywords()["fibre"]
    assert isinstance(built, StepIndexMode)
    omega = 2 * np.pi * 299792458.0 / 1e-6
    index = float(np.real(built.neff(np.array([omega])))[0])
    assert 1.0 < index < 1.8


def test_a_glass_table_builds_the_named_glass():
    fibre = (
        'fibre = { type = "StepIndexMode", radius = "2 um", '
        'profile_wavelength = "1 um", core = { type = "glass", name = "BK7" } }'
    )
    built = recipe_of(fibre).keywords()["fibre"]
    assert isinstance(built, StepIndexMode)
    assert isinstance(built.core, GlassIndex)
    assert built.core.source == glass("BK7").source


def test_the_mode_family_and_loss_are_keywords():
    fibre = (
        'fibre = { type = "StepIndexMode", radius = "4 um", '
        'profile_wavelength = "1 um", core = 1.47, cladding = 1.45, kind = "TE", '
        'n = 0, m = 1, loss = "0.5 m^-1" }'
    )
    built = recipe_of(fibre).keywords()["fibre"]
    assert isinstance(built, StepIndexMode)
    assert (built.kind, built.n, built.m, built.loss) == ("TE", 0, 1, 0.5)


@pytest.mark.medium
def test_the_python_export_of_a_step_index_recipe_builds_the_same_run(tmp_path):
    fibre = (
        'fibre = { type = "StepIndexMode", radius = "2 um", '
        'profile_wavelength = "1 um", core = { type = "glass", name = "BK7" } }'
    )
    recipe = recipe_of(fibre)
    script = recipe.to_python()
    assert "glass(name='BK7')" in script
    probe = textwrap.dedent(
        """\
        from solprop.driver.simulation import Simulation

        def run(self, *args, **kwargs):
            print(self.metadata().configuration_hash())
            raise SystemExit(0)

        Simulation.run = run
        """
    )
    path = tmp_path / "export.py"
    path.write_text(probe + script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    expected = recipe.build().metadata().configuration_hash()
    assert result.stdout.strip() == expected


def test_gamma_none_is_the_keyword_a_script_passes_as_none():
    assert parameters("gnlse")["gamma"].none
    assert recipe_of().keywords()["gamma"] is None
    without = GNLSE.replace('gamma = "none"\n', "").replace("{fibre}", FIBRE)
    with pytest.raises(RecipeError, match="gamma: a required keyword is missing"):
        Recipe.loads(without).build()


def test_gamma_none_without_n2_is_refused_by_the_route():
    recipe = Recipe.loads(GNLSE.replace("{fibre}", FIBRE).replace("n2 = 2.6e-20\n", ""))
    assert [issue.rule for issue in errors_of(recipe)] == ["gamma-or-n2"]


def test_an_unregistered_fibre_is_refused_by_name():
    recipe = recipe_of('fibre = { type = "IndexTable" }')
    with pytest.raises(RecipeError, match="IndexTable"):
        recipe.keywords()


def test_a_callable_loss_stays_python_only():
    forms = parameters("gnlse")["fibre"].forms
    assert [form.recipe for form in forms] == [True, False]
