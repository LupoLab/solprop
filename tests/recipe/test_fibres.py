# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes for solid-core fibres: a step-index GNLSE fibre and a graded-index guide."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    ModalGuide,
    PowerLawPermittivity,
    StepIndexMode,
)
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
    with pytest.raises(RecipeError, match="Python export") as caught:
        recipe.keywords()
    assert caught.value.parameter == "fibre"


def test_a_callable_loss_stays_python_only():
    forms = parameters("gnlse")["fibre"].forms
    assert [form.recipe for form in forms] == [True, False]


GRIN_GUIDE = textwrap.dedent(
    """\
    provider = {{ type = "ModalGuide.on_grid", knots = {knots}, n_r = 2, provider = {{ type = "GradedIndexProvider", permittivity = {{ type = "PowerLawPermittivity", centre = {{ type = "IndexMaterial", index = 1.45 }}, contrast = {contrast} }}, exterior = {{ type = "IndexMaterial", index = 1.4427 }}, radius = "25 um", family = "LP", count = 2, elements = 16, extent = 2.0 }} }}
    """  # noqa: E501 - a recipe's inline tables are one line each
)

GRIN = textwrap.dedent(
    """\
    [recipe]
    format = 1
    route = "waveguide"

    [build]
    modes = []
    length = "1 mm"
    {grid}
    {guide}
    wavelength = "1.03 um"
    fwhm = "40 fs"
    energy = "1 fJ"
    n_save = 2
    noise = "none"
    absorber = false
    boundary_check = "none"
    backend = "numpy_ref"
    """
)

GRID = (
    'grid = { type = "FieldGrid.from_limits", length = "1 mm", '
    'reference_wavelength = "1.03 um", wavelength_limits = ["1 um", "1.06 um"], '
    'trange = "100 fs" }'
)


def grin_of(*, grid: str = GRID, knots: int = 13, contrast: float = 0.005) -> Recipe:
    guide = GRIN_GUIDE.format(knots=knots, contrast=contrast)
    return Recipe.loads(GRIN.format(grid=grid, guide=guide))


def _grin_script() -> Simulation:
    grid = FieldGrid.from_limits(1e-3, 1.03e-6, (1e-6, 1.06e-6), 100e-15)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.4427),
        radius=25e-6,
        family="LP",
        count=2,
        elements=16,
        extent=2.0,
    )
    return Simulation.from_waveguide(
        (),
        1e-3,
        provider=ModalGuide.on_grid(grid, provider, knots=13, n_r=2),
        grid=grid,
        wavelength=1.03e-6,
        fwhm=40e-15,
        energy=1e-15,
        n_save=2,
        noise=None,
        absorber=False,
        boundary_check=None,
        backend="numpy_ref",
    )


@pytest.mark.medium
def test_a_graded_index_guide_is_the_same_run_as_the_script():
    built = grin_of().build()
    assert (
        built.metadata().configuration_hash()
        == _grin_script().metadata().configuration_hash()
    )


def test_on_grid_samples_the_active_band_beyond_both_ends():
    grid = FieldGrid.from_limits(1e-3, 1.03e-6, (1e-6, 1.06e-6), 100e-15)
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.4427),
        radius=25e-6,
        family="LP",
        count=2,
        elements=16,
        extent=2.0,
    )
    guide = ModalGuide.on_grid(grid, provider, knots=13, n_r=2)
    active = grid.omega[grid.active_mask]
    assert guide.sampling is not None
    sampled = guide.sampling.omega
    assert sampled.size == 13
    assert sampled[0] == pytest.approx(0.99 * active[0], rel=1e-15)
    assert sampled[-1] == pytest.approx(1.01 * active[-1], rel=1e-15)


def test_a_guide_on_the_grid_needs_the_recipe_to_give_the_grid():
    recipe = grin_of(grid='wavelength_limits = ["1 um", "1.06 um"]\ntrange = "100 fs"')
    with pytest.raises(RecipeError, match="must give grid") as caught:
        recipe.keywords()
    assert caught.value.parameter == "provider"
    with pytest.raises(RecipeError, match="must give grid"):
        recipe.build()
    assert [issue.message for issue in errors_of(recipe)] == [
        "this guide is built on the route's grid, so the recipe must give grid, a "
        "FieldGrid.from_limits table"
    ]


def test_checking_a_guide_does_not_solve_it_on_the_stand_in_grid(monkeypatch):
    calls = []
    real = ModalGuide.from_provider

    def spy(provider, omega, **keywords):
        calls.append(np.asarray(omega))
        return real(provider, omega, **keywords)

    monkeypatch.setattr(ModalGuide, "from_provider", spy)
    recipe = grin_of()
    assert not errors_of(recipe)
    # The one solve is the build's, on the run's band: 1 um to 1.06 um padded to about
    # 2.1 um at the long end, and never the stand-in's 400 nm.
    assert len(calls) == 1
    assert 2 * np.pi * 299792458.0 / calls[0][-1] > 900e-9


def test_a_bad_keyword_of_a_guide_is_reported_by_check_without_a_solve():
    issues = errors_of(grin_of(contrast=0.75))
    assert len(issues) == 1
    assert "contrast" in issues[0].message


def test_too_few_knots_is_the_route_s_refusal():
    assert errors_of(grin_of(knots=4))


@pytest.mark.medium
def test_the_python_export_of_a_graded_index_recipe_builds_the_same_run(tmp_path):
    recipe = grin_of()
    script = recipe.to_python()
    assert "ModalGuide.on_grid(grid, " in script
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
    assert result.stdout.strip() == recipe.build().metadata().configuration_hash()
