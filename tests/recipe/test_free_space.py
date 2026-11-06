# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes on the object routes: radial, free and planar (SEP 41, phase 4).

Their keywords are objects, each a table of a registered type: the time grid, the
transverse grid, the medium, the responses, the beam and the settings, with the
absorber under the settings. The gallery's example recipes are ``test_examples``'s; the
radial calls here are the user guide's (``docs/guide/free-space/beams.md``) with the
beam's power written as a literal. Each recipe must build the
call's configuration hash, so those comparisons are of digests and exact.
"""

import json
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from recipe.test_examples import _PROBE, ENTROPY, record
from solprop.boundaries import SpatialAbsorber, absorber
from solprop.capabilities import route
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import ModeCollection, PlanarGrid, SlabMode
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import gamma3
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.recipe import Recipe, RecipeError, parameters, provenance
from solprop.recipe.metadata import OBJECT_ROUTES, SETTINGS_KEYWORDS
from solprop.recipe.preview import GUIDED
from solprop.recipe.registry import REGISTRY, BuildContext, build_object
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    planar_mode_energy,
    spatial_energy,
    spatial_fluence,
    spatial_power,
)
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

KERR = (
    '{ type = "KerrResponse", chi3 = { type = "gamma3", gas = "Ar", '
    'wavelength = "800 nm" }, thg = false }'
)

RADIAL = textwrap.dedent(
    f"""\
    [recipe]
    format = 1
    route = "radial"

    [build]
    grid = {{ type = "EnvelopeGrid.from_limits", length = "1 mm", reference_wavelength = "800 nm", wavelength_limits = ["600 nm", "1.2 um"], trange = "200 fs", thg = false }}
    space = {{ type = "RadialGrid", radius = "400 um", n_r = 32 }}
    medium = {{ type = "SpatialMedium", material = "Ar", pressure = "1 bar" }}
    responses = [
        {KERR},
        {{ type = "EnvelopePlasmaResponse", gas = "Ar", rate = {{ type = "OptmatRate", source = "Ar", wavelength = "800 nm" }} }},
    ]
    seed = 20261102

    [build.initial]
    type = "SpatialPulse"
    temporal = {{ type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", peak_power = "5 GW" }}
    waist = "100 um"

    [build.settings]
    type = "SimulationSettings"
    controller = {{ type = "PIController", rtol = 1e-6, atol = 1e-12 }}
    save_planes = ["0.5 mm", "1 mm"]
    dz_init = "0.1 mm"
    dz_max = "0.5 mm"
    stats = {{ type = "StatsSpec", names = ["energy", "beam_radius", "electron_density"] }}
    absorber = {{ type = "absorber" }}
    ionisation_field_range = [1e8, 1e11]
    backend = "numpy_ref"
    """  # noqa: E501 - a recipe's inline tables are one line each
)


def _radial_call() -> Simulation:
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15, thg=False)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.5e-3, 1e-3],
        dz_init=1e-4,
        dz_max=5e-4,
        stats=StatsSpec(names=("energy", "beam_radius", "electron_density")),
        absorber=absorber(grid),
        ionisation_field_range=(1e8, 1e11),
        backend="numpy_ref",
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(400e-6, 32),
        SpatialMedium("Ar", pressure=1.0),
        [
            KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False),
            EnvelopePlasmaResponse("Ar", OptmatRate("Ar", 800e-9)),
        ],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, peak_power=5e9), 100e-6
        ),
        settings=settings,
        seed=20261102,
    )


FIELD = textwrap.dedent(
    """\
    [recipe]
    format = 1
    route = "radial"

    [build]
    grid = { type = "FieldGrid.from_limits", length = "0.2 mm", reference_wavelength = "800 nm", wavelength_limits = ["400 nm", "2 um"], trange = "150 fs" }
    space = { type = "RadialGrid", radius = "200 um", n_r = 16, collar_fraction = 0.25 }
    medium = { type = "SpatialMedium", material = "Ar", pressure = "2 bar", temperature = "300 K" }
    responses = [
        { type = "PlasmaResponse", gas = "Ar", rate = { type = "OptmatRate", source = "Ar", wavelength = "800 nm" }, integration = "trapezoid" },
    ]
    initial = { type = "SpatialPulse", temporal = { type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1 uJ" }, waist = "50 um", propz = "-1 mm" }
    noise = "none"
    materials = { type = "MaterialSelection" }
    boundary_check = "none"
    spatial_absorber = { type = "SpatialAbsorber", real_space = false, safety_fraction = 0.2 }

    [build.settings]
    type = "SimulationSettings"
    save_planes = ["0.2 mm"]
    dz_init = "0.05 mm"
    dz_max = "0.1 mm"
    ionisation_field_range = [1e8, 1e11]
    """  # noqa: E501 - a recipe's inline tables are one line each
)


def _field_call() -> Simulation:
    grid = FieldGrid.from_limits(0.2e-3, 800e-9, (400e-9, 2e-6), 150e-15)
    return Simulation.from_radial(
        grid,
        RadialGrid(200e-6, 16, collar_fraction=0.25),
        SpatialMedium("Ar", pressure=2.0, temperature=300.0),
        [PlasmaResponse("Ar", OptmatRate("Ar", 800e-9), integration="trapezoid")],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6),
            50e-6,
            propz=-1e-3,
        ),
        settings=SimulationSettings(
            save_planes=[0.2e-3],
            dz_init=0.05e-3,
            dz_max=0.1e-3,
            ionisation_field_range=(1e8, 1e11),
        ),
        noise=None,
        materials=MaterialSelection(),
        boundary_check=None,
        spatial_absorber=SpatialAbsorber(real_space=False, safety_fraction=0.2),
    )


LINE = textwrap.dedent(
    """\
    [recipe]
    format = 1
    route = "free"

    [build]
    grid = { type = "EnvelopeGrid.from_limits", length = "1 mm", reference_wavelength = "800 nm", wavelength_limits = ["600 nm", "1.2 um"], trange = "200 fs", thg = false }
    space = { type = "LineGrid", radius_x = "400 um", n_x = 32 }
    medium = { type = "SpatialMedium", material = "Ar", pressure = "1 bar" }
    responses = []
    initial = { type = "SpatialPulse", temporal = { type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = 1e-3 }, waist = "100 um" }
    spatial_absorber = false

    [build.settings]
    type = "SimulationSettings"
    controller = { type = "PIController", rtol = 1e-6, atol = 1e-12 }
    save_planes = ["0.5 mm", "1 mm"]
    dz_init = "0.5 mm"
    dz_max = "0.5 mm"
    schedule = { type = "FixedSchedule.uniform", z_start = 0, z_end = "1 mm", dz = "0.5 mm" }
    stats = { type = "default_spec", kind = "free" }
    backend = "numpy_ref"
    """  # noqa: E501 - a recipe's inline tables are one line each
)


def test_a_missing_positional_keyword_is_refused_before_the_route():
    text = LINE.replace(
        'space = { type = "LineGrid", radius_x = "400 um", n_x = 32 }\n', ""
    )
    with pytest.raises(
        RecipeError, match=r"^space: a required keyword is missing$"
    ) as caught:
        Recipe.loads(text).build()
    assert caught.value.parameter == "space"


def _line_call() -> Simulation:
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15, thg=False)
    from solprop.stats import default_spec  # noqa: PLC0415 - one call only

    return Simulation.from_free(
        grid,
        LineGrid(400e-6, 32),
        SpatialMedium("Ar", pressure=1.0),
        [],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-3), 100e-6
        ),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-12),
            save_planes=[0.5e-3, 1e-3],
            dz_init=0.5e-3,
            dz_max=0.5e-3,
            schedule=FixedSchedule.uniform(0.0, 1e-3, 0.5e-3),
            stats=default_spec("free"),
            backend="numpy_ref",
        ),
        spatial_absorber=False,
    )


AIMED = textwrap.dedent(
    """\
    [recipe]
    format = 1
    route = "free"

    [build]
    grid = { type = "EnvelopeGrid.from_limits", length = "1 mm", reference_wavelength = "800 nm", wavelength_limits = ["600 nm", "1.2 um"], trange = "200 fs", thg = false }
    space = { type = "FreeGrid", radius_x = "400 um", n_x = 32, radius_y = "400 um", n_y = 32 }
    medium = { type = "SpatialMedium", material = 1.45 }
    responses = []
    noise = "input"
    seed = "123456789012345678901234567890"
    boundary_check = { type = "BoundaryCheck", budget = 1e-2 }

    [build.initial]
    type = "SpatialPulse"
    temporal = { type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1 nJ" }
    waist = "40 um"
    displacement = ["50 um", 0]
    tilt = ["2 mrad", 0]
    jones = [{ re = 1.0, im = 0.0 }]

    [build.settings]
    type = "SimulationSettings"
    save_planes = ["1 mm"]
    dz_init = "0.1 mm"
    dz_max = "0.5 mm"
    absorber_scheme = "nonlinear"
    """  # noqa: E501 - a recipe's inline tables are one line each
)


def _aimed_call() -> Simulation:
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15, thg=False)
    return Simulation.from_free(
        grid,
        FreeGrid(400e-6, 32, 400e-6, 32),
        SpatialMedium(1.45),
        [],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9),
            40e-6,
            displacement=(50e-6, 0.0),
            tilt=(2e-3, 0.0),
            jones=(1 + 0j,),
        ),
        settings=SimulationSettings(
            save_planes=[1e-3], dz_init=1e-4, dz_max=5e-4, absorber_scheme="nonlinear"
        ),
        noise="input",
        seed=123456789012345678901234567890,
        boundary_check=BoundaryCheck(budget=1e-2),
    )


PLANAR = textwrap.dedent(
    f"""\
    [recipe]
    format = 1
    route = "planar"

    [build]
    grid = {{ type = "EnvelopeGrid.from_limits", length = "1 cm", reference_wavelength = "800 nm", wavelength_limits = ["650 nm", "1000 nm"], trange = "400 fs" }}
    space = {{ type = "PlanarGrid", modes = {{ type = "ModeCollection.slab", half_gap = "63.5 um", orders = [1, 3], gas = "Ar", pressure = "1 bar" }}, radius_y = "1 mm", n_y = 16, n_x = 24 }}
    responses = [{KERR}]
    initial = {{ type = "PlanarPulse", temporal = {{ type = "GaussianPulse", wavelength = "800 nm", fwhm = "100 fs", energy = "0.1 mJ" }}, waist = "0.3 mm", waist_x = "45 um" }}

    [build.settings]
    type = "SimulationSettings"
    controller = {{ type = "PIController", rtol = 1e-3, atol = 1e-9 }}
    save_planes = ["1 cm"]
    dz_max = "5 mm"
    dz_init = "1 mm"
    backend = "numpy_ref"
    """  # noqa: E501 - a recipe's inline tables are one line each
)


def _planar_call() -> Simulation:
    grid = EnvelopeGrid.from_limits(1e-2, 800e-9, (650e-9, 1000e-9), 400e-15)
    modes = ModeCollection.slab(63.5e-6, (1, 3), "Ar", 1.0)
    return Simulation.from_planar(
        grid,
        PlanarGrid(modes, 1e-3, 16, n_x=24),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        PlanarPulse(
            GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-4),
            0.3e-3,
            waist_x=45e-6,
        ),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-3, atol=1e-9),
            save_planes=[1e-2],
            dz_max=5e-3,
            dz_init=1e-3,
            backend="numpy_ref",
        ),
    )


CASES = {
    "radial": (RADIAL, _radial_call),
    "field": (FIELD, _field_call),
    "line": (LINE, _line_call),
    "aimed": (AIMED, _aimed_call),
    "planar": (PLANAR, _planar_call),
}


def _hash(simulation: Simulation) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return simulation.metadata().configuration_hash()


# ------------------------------------------------------------------- building


@pytest.mark.parametrize("name", list(CASES))
def test_a_recipe_builds_the_call_it_stands_for(name):
    text, call = CASES[name]
    recipe = Recipe.loads(text)
    assert _hash(recipe.build()) == _hash(call())


@pytest.mark.parametrize("name", ["radial", "line", "planar"])
def test_check_is_clean(name):
    assert Recipe.loads(CASES[name][0]).check() == []


def test_the_object_routes_list_their_objects_and_leave_the_absorber_to_settings():
    for name in OBJECT_ROUTES:
        table = parameters(name)
        assert {"grid", "space", "responses", "initial", "settings"} <= set(table)
        assert set(SETTINGS_KEYWORDS).isdisjoint(table)
        assert route(name).keywords == tuple(table)
    assert "medium" not in parameters("planar")
    assert "noise" not in parameters("planar")
    assert route("radial").choices["noise"] == ("default", "input", "modified", "none")
    resolver = parameters("radial")["noise"].resolver
    assert resolver is not None
    assert resolver({}).value == "input"
    assert "transverse column" in resolver({}).rule
    single = {"settings": {"type": "SimulationSettings", "precision": "single"}}
    assert resolver(single).value is None


@pytest.mark.parametrize("key", SETTINGS_KEYWORDS)
def test_the_absorber_is_written_under_settings_not_at_the_top(key):
    text = RADIAL.replace("seed = 20261102", f'seed = 20261102\n{key} = "x"')
    with pytest.raises(RecipeError, match=f"settings.{key}") as caught:
        Recipe.loads(text)
    assert caught.value.parameter == key


def test_the_execution_settings_are_refused_in_the_settings_table():
    text = RADIAL.replace(
        'backend = "numpy_ref"', 'backend = "numpy_ref"\nfields = "none"'
    )
    with pytest.raises(RecipeError, match=r"\[execution\]") as caught:
        Recipe.loads(text).build()
    assert caught.value.parameter == "settings.fields"


def test_what_only_python_holds_points_to_the_export():
    recipe = Recipe.loads(RADIAL)
    for value in ([1.0, 2.0], {"file": "launch.npy", "sha256": "0" * 64}):
        document = recipe.to_dict()
        document["build"]["initial"] = value
        with pytest.raises(RecipeError, match="Python export") as caught:
            Recipe.from_dict(document).keywords()
        assert caught.value.parameter == "initial"
    document = recipe.to_dict()
    document["build"]["frame"] = {"type": "Frame"}
    with pytest.raises(RecipeError, match="Python export"):
        Recipe.from_dict(document).keywords()


def test_pins_have_no_target_on_an_object_route():
    document = Recipe.loads(RADIAL).to_dict()
    document["pins"] = {"initial.temporal.energy": {"target": "soliton_order"}}
    with pytest.raises(RecipeError, match="no guide or launch mode") as caught:
        Recipe.from_dict(document).build()
    assert caught.value.parameter == "initial.temporal.energy"


def test_a_slab_mode_names_its_fill_on_the_planar_route_alone():
    table = {"type": "SlabMode", "half_gap": "63.5 um", "gas": "Ar", "pressure": 1.0}
    mode = build_object(table, BuildContext(mode_fill=True), "space.modes[0]")
    assert isinstance(mode, SlabMode)
    assert mode.pressure == 1.0
    with pytest.raises(RecipeError, match="'gas' is the route's") as caught:
        build_object(table, BuildContext(), "modes[0]")
    assert caught.value.parameter == "modes[0]"
    # A waveguide recipe whose slab mode names a fill is refused at that mode.
    text = (
        '[recipe]\nformat = 1\nroute = "waveguide"\n\n[build]\n'
        'modes = [{ type = "SlabMode", half_gap = "63.5 um", gas = "Ar" }]\n'
        'length = "1 cm"\ngas = "Ar"\nwavelength = "800 nm"\nfwhm = "100 fs"\n'
        'energy = 0.5\nwavelength_limits = ["650 nm", "1000 nm"]\ntrange = "1 ps"\n'
    )
    with pytest.raises(RecipeError, match="'gas' is the route's") as caught:
        Recipe.loads(text).build()
    assert caught.value.parameter == "modes[0]"


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"type": "RadialGrid", "radius": "1 mm", "n_r": 8}, RadialGrid),
        (
            {
                "type": "FreeGrid",
                "radius_x": "1 mm",
                "n_x": 8,
                "radius_y": "1 mm",
                "n_y": 8,
            },
            FreeGrid,
        ),
        ({"type": "LineGrid", "radius_x": "1 mm", "n_x": 8}, LineGrid),
        (
            {"type": "SpatialMedium", "material": "Ar", "pressure": "1 bar"},
            SpatialMedium,
        ),
        (
            {
                "type": "OptmatRate",
                "source": "Ar",
                "wavelength": "800 nm",
                "maximum": "1e16 Hz",
            },
            OptmatRate,
        ),
        ({"type": "PIController", "rtol": 1e-6, "atol": 0.0}, PIController),
        (
            {"type": "SpatialAbsorber", "kappa": "1 rad/um", "discard_above": 2e6},
            SpatialAbsorber,
        ),
        ({"type": "default_spec", "kind": "radial"}, StatsSpec),
    ],
)
def test_each_free_space_type_builds_from_a_table(table, expected):
    assert isinstance(build_object(table, BuildContext(), "value"), expected)


def test_the_registered_settings_and_controller_leave_out_what_is_not_a_recipe_s():
    settings = REGISTRY["SimulationSettings"]
    assert {"fields", "field_planes", "resume_history"} <= set(settings.omitted)
    assert all(name.startswith("_") or name in settings.omitted for name in ("_route",))
    assert "weights" in REGISTRY["PIController"].omitted
    assert "susceptibility" in REGISTRY["SpatialMedium"].omitted
    for name in ("SlabMode", "ModeCollection.slab"):
        assert REGISTRY[name].fill == ("gas", "pressure", "temperature"), name


# ------------------------------------------------------------------- export


@pytest.mark.child_runs_auto
@pytest.mark.medium
@pytest.mark.parametrize("name", ["radial", "field", "line", "aimed", "planar"])
def test_the_python_export_builds_the_same_run(name, tmp_path, monkeypatch):
    import numpy.random.bit_generator  # noqa: PLC0415 - the draw is fixed here only

    monkeypatch.setattr(numpy.random.bit_generator, "randbits", lambda _: ENTROPY)
    recipe = Recipe.loads(CASES[name][0])
    script = recipe.to_python()
    path = tmp_path / f"{name}.py"
    path.write_text(_PROBE.format(entropy=ENTROPY) + script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = record(recipe.build())
    assert json.loads(result.stdout.splitlines()[-1]) == expected


def test_the_export_writes_the_grid_once_and_tuples_as_tuples():
    script = Recipe.loads(AIMED).to_python()
    assert script.count("EnvelopeGrid.from_limits(") == 1
    assert "displacement=(5e-05, 0.0)" in script
    assert "jones=((1+0j),)" in script
    assert "prop_free(" in script
    radial = Recipe.loads(RADIAL).to_python()
    assert "absorber=absorber(grid)" in radial


# ------------------------------------------------------------------- the record


def test_the_resolved_recipe_rebuilds_the_run_with_its_drawn_seed():
    document = Recipe.loads(RADIAL).to_dict()
    del document["build"]["seed"]
    recipe = Recipe.from_dict(document)
    simulation = recipe.build()
    simulation.translate()
    resolved = provenance.resolved_document(recipe, simulation)
    assert "seed" in resolved["build"]
    assert _hash(Recipe.from_dict(resolved).build()) == _hash(simulation)


@pytest.mark.medium
def test_a_run_records_its_recipe(tmp_path):
    recipe = Recipe.loads(LINE)
    path = tmp_path / "line.h5"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recipe.run(path, progress="none")
    # The two fixed steps are coarse for the edge's reads, which warns; the recipe is
    # kept all the same.
    assert not [one for one in caught if "recipe" in str(one.message)]
    assert Recipe.from_output(path) == recipe


def test_a_scan_reads_a_beam_s_energy_in_its_unit():
    document = Recipe.loads(LINE).to_dict()
    document["scan"] = {"name": "energy", "initial.temporal.energy": ["1 mJ", "2 mJ"]}
    scan = Recipe.from_dict(document).scan()
    assert scan.scan.variables["initial.temporal.energy"].tolist() == [1e-3, 2e-3]


# ------------------------------------------------------------------- previews


@pytest.mark.medium
@pytest.mark.parametrize("name", ["radial", "line"])
def test_the_input_preview_is_the_run_s_noise_free_entrance(name, tmp_path):
    document = Recipe.loads(CASES[name][0]).to_dict()
    document["build"]["noise"] = "none"
    document["build"].pop("seed", None)
    recipe = Recipe.from_dict(document)
    preview = Recipe.loads(CASES[name][0]).preview_input()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = recipe.build()
        output = simulation.run(progress="none")
    entrance = output.fields("fft")[0]
    space, medium = simulation.spatial_grid, simulation.medium
    assert space is not None
    assert medium is not None
    flux = SpatialFlux.resolve(simulation.grid, space, medium)
    assert preview.fluence is not None
    # The same stored field through the same functions: equal to the last bit.
    assert np.array_equal(preview.fluence, spatial_fluence(entrance, flux))
    assert np.array_equal(preview.power, spatial_power(entrance, flux))
    assert preview.energy == spatial_energy(entrance, flux)
    assert preview.peak_power == float(np.max(preview.power))
    assert preview.time.shape == preview.power.shape


def test_the_input_preview_gives_the_beam_on_its_axes():
    radial = Recipe.loads(RADIAL).preview_input()
    assert radial.fluence is not None
    assert radial.fluence.shape == radial.axes[0].shape == (32,)
    # The fluence peaks on the axis and falls outwards.
    assert int(np.argmax(radial.fluence)) == 0
    aimed = Recipe.loads(AIMED).preview_input()
    assert aimed.fluence is not None
    y, x = aimed.axes
    assert aimed.fluence.shape == (y.size, x.size)
    # Displaced 50 um towards +x, centred in y.
    row, column = np.unravel_index(np.argmax(aimed.fluence), aimed.fluence.shape)
    assert x[column] == pytest.approx(50e-6, abs=float(x[1] - x[0]))
    assert y[row] == pytest.approx(0.0, abs=float(y[1] - y[0]))
    # The noise is left out, so a preview draws nothing.
    assert aimed.energy == pytest.approx(1e-9, rel=1e-12)
    line = Recipe.loads(LINE).preview_input()
    assert line.fluence is not None
    assert line.fluence.shape == line.axes[0].shape == (32,)


def test_the_planar_preview_carries_the_coupled_energy_on_its_nodes():
    recipe = Recipe.loads(PLANAR)
    preview = recipe.preview_input()
    simulation = recipe.build()
    problem = simulation.translate()
    space, medium = simulation.spatial_grid, simulation.medium
    assert isinstance(space, PlanarGrid)
    assert medium is not None
    flux = SpatialFlux.resolve(simulation.grid, space, medium)
    modal = planar_mode_energy(np.asarray(problem.initial.field0)[0], flux)
    # The modes' energies sum to the beam's, to the rounding of the two sums.
    assert preview.energy == pytest.approx(float(np.sum(modal)), rel=1e-12)
    assert 0.0 < preview.energy < 1e-4
    assert preview.fluence is not None
    y, x = preview.axes
    assert preview.fluence.shape == (y.size, x.size)
    assert y.size == 16
    assert np.array_equal(x, space.x)


def test_the_planar_mode_preview_is_the_grid_s_slab_channels():
    assert "planar" in GUIDED
    recipe = Recipe.loads(PLANAR)
    preview = recipe.modes()
    assert [mode.label for mode in preview.modes] == ["TE1", "TE3"]
    space = recipe.build().spatial_grid
    assert isinstance(space, PlanarGrid)
    channels = space.modes
    from solprop.modes.base import beta2  # noqa: PLC0415 - the oracle only

    assert np.array_equal(preview.modes[0].beta2, beta2(channels[0], preview.omega))
    with pytest.raises(RecipeError, match="builds no mode set"):
        Recipe.loads(RADIAL).modes()
