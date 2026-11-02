# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launching each pulse into chosen modes (SEP 41, S5).

The shares are energy fractions on a basis whose flux metric is the identity and a
coordinate split on one that declares a change of basis; a list launches each pulse
into its own modes; labels resolve as ``channel=`` resolves them; the launch is
recorded as ``configuration["modal_launch"]``, compared on resume, and absent from a
run that launches as before.
"""

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop import fourier
from solprop.backends import available
from solprop.coupling import CouplingModel
from solprop.driver import Simulation
from solprop.driver._launch import _resolve_launch, launch_pulse
from solprop.exceptions import DriverError, OutputError
from solprop.modes import MarcatiliMode
from solprop.modes.collection import ModeCollection
from solprop.output.compatibility import neutral_value
from solprop.pulses import GaussianPulse, PulseSum, SpatialPulse
from solprop.stats import energy, physical_energy
from solprop.translate.coupled import metric_at

# Tight time windows for speed: the boundary monitor's warning, and the gas models'
# range warning over a capillary's wide window, are not what this module tests.
pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

CAPILLARY = {
    "wavelength": 800e-9,
    "wavelength_limits": (300e-9, 3e-6),
    "trange": 100e-15,
    "fwhm": 10e-15,
    "energy": 5e-6,
    "modes": 3,
    "plasma": False,
    "n_r": 16,
    "n_save": 2,
    "noise": None,
    "backend": "numpy_ref",
}
LENGTH = 1e-4  # m: two planes a tenth of a millimetre apart, a handful of steps

# Rounding budget of a per-mode energy against its share: one square root, one
# multiplication by a unit phase and a sum of |a|^2 over the grid's ~200 bins, each a
# few ulps, against an energy of order one in its own units.
SHARE_RTOL = 1e-13

PULSE = {"wavelength": 835e-9, "fwhm": 20e-15, "energy": 1e-12, "shape": "gauss"}


def capillary(**overrides):
    """Return a short three-mode helium capillary, launched as `overrides` say."""
    return Simulation.from_capillary(125e-6, LENGTH, "He", 1.0, **CAPILLARY | overrides)


def record_of(simulation) -> dict[str, Any]:
    """Return a simulation's launch record."""
    return cast("dict[str, Any]", simulation.configuration["modal_launch"])


def mode_energies(simulation):
    """Return the energy of each mode of a simulation's input, J."""
    return energy(simulation.grid, simulation._field0).sum(axis=1)


@pytest.fixture(scope="module")
def grid():
    """The field grid of the general modal guides."""
    return general_grid()


@pytest.fixture(scope="module")
def pair_guide(grid):
    """The two parities of a strand's HE11, in the provider's own coordinates."""
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        grid=grid,
    )


@pytest.fixture(scope="module")
def circular_guide(grid):
    """The same pair in a declared circular basis, whose flux metric is dense."""
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        circular=(0, 1),
        grid=grid,
    )


def general(guide, grid, **overrides):
    """Return a general modal simulation on a guide."""
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=guide,
        grid=grid,
        noise=None,
        save_planes=[GENERAL_Z_END / 10],
        backend="numpy_ref",
        **PULSE | overrides,
    )


# ------------------------------------------------------------------- normalisation


def test_on_a_marcatili_basis_the_mode_energies_are_the_shares():
    simulation = capillary(
        launch={"HE1,1": 9.0, "HE1,2": {"share": 1.0, "phase": 1.57}}
    )
    energies = mode_energies(simulation)
    # The shares are normalised to one: 9 and 1 are 0.9 and 0.1 of the pulse.
    assert_allclose(energies / energies.sum(), [0.9, 0.1, 0.0], rtol=SHARE_RTOL)
    alone = capillary()
    assert_allclose(energies.sum(), mode_energies(alone).sum(), rtol=SHARE_RTOL)
    assert record_of(simulation) == {
        "split": "energy",
        "pulses": [
            {
                "labels": ["HE1,1:even", "HE1,2:even"],
                "shares": [0.9, 0.1],
                "phases": [0.0, 1.57],
            }
        ],
    }


def test_a_phase_is_an_optics_convention_rotation_of_the_mode_s_spectrum():
    simulation = capillary(launch={"HE1,1": 0.5, "HE1,2": {"share": 0.5, "phase": 1.0}})
    optics = fourier.to_optics(simulation._field0[:, 0])
    lit = np.abs(optics[0]) > 1e-3 * np.abs(optics[0]).max()
    # Equal shares and a phase of one radian: the second mode is the first times
    # exp(+i), in the convention a file stores; one multiplication of rounding.
    assert_allclose(optics[1, lit] / optics[0, lit], np.exp(1j), rtol=1e-14)


def test_labels_resolve_by_display_label_its_unambiguous_stem_and_index():
    by_label = capillary(launch={"HE1,1:even": 0.7, "HE1,3:even": 0.3})
    by_stem = capillary(launch={"HE1,1": 0.7, "HE1,3": 0.3})
    by_index = capillary(launch={0: 0.7, 2: 0.3})
    assert record_of(by_label) == record_of(by_stem)
    assert record_of(by_label) == record_of(by_index)
    assert_allclose(by_stem._field0, by_label._field0, rtol=0, atol=0)
    assert_allclose(by_index._field0, by_label._field0, rtol=0, atol=0)


def test_a_launch_into_the_first_mode_alone_is_the_default_launch():
    default = capillary()
    first = capillary(launch={"HE1,1": 1.0})
    # One multiplication by sqrt(1) and by the unit phase, each exact.
    assert_allclose(first._field0, default._field0, rtol=1e-15, atol=0)
    assert "modal_launch" not in default.configuration
    assert "modal_launch" not in default.metadata().configuration


# ---------------------------------------------------------------------- per pulse


def two_colours():
    """Return a pump at 800 nm and a seed at 1.2 um, of different energies."""
    return [
        GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=4e-6),
        GaussianPulse(wavelength=1.2e-6, fwhm=20e-15, energy=1e-6, t0=20e-15),
    ]


def test_a_list_launches_each_pulse_into_its_own_modes():
    pulses = two_colours()
    simulation = capillary(
        fwhm=None,
        energy=None,
        pulses=pulses,
        launch=[{"HE1,1": 1.0}, {"HE1,2": 0.5, "HE1,3": 0.5}],
    )
    grid = simulation.grid
    pump, seed = (float(energy(grid, pulse.build(grid)).sum()) for pulse in pulses)
    assert_allclose(
        mode_energies(simulation), [pump, seed / 2, seed / 2], rtol=SHARE_RTOL
    )
    record = record_of(simulation)
    assert [entry["labels"] for entry in record["pulses"]] == [
        ["HE1,1:even"],
        ["HE1,2:even", "HE1,3:even"],
    ]


def test_one_mapping_launches_every_pulse_the_same_way():
    pulses = two_colours()
    simulation = capillary(fwhm=None, energy=None, pulses=pulses, launch={"HE1,2": 1})
    summed = PulseSum(tuple(pulses)).build(simulation.grid)
    # The members are placed one by one and added, as PulseSum adds them.
    assert_allclose(simulation._field0[1], summed[0], rtol=1e-15, atol=1e-30)
    assert not np.any(simulation._field0[[0, 2]])
    record = record_of(simulation)
    assert len(record["pulses"]) == 2
    assert record["pulses"][0] == record["pulses"][1]


def test_a_list_holds_one_mapping_per_pulse():
    with pytest.raises(DriverError, match="one mapping for every pulse") as refused:
        capillary(
            fwhm=None,
            energy=None,
            pulses=two_colours(),
            launch=[{"HE1,1": 1.0}],
        )
    assert refused.value.parameter == ("launch", "pulses")
    assert refused.value.rule == "launch-per-pulse"


# ----------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("launch", "parameter", "match"),
    [
        ({"HE1,4": 1.0}, 'launch["HE1,4"]', "is not a mode of this set"),
        ({5: 1.0}, "launch[5]", "outside this set's 3 mode"),
        ({-1: 1.0}, "launch[-1]", "outside this set's 3 mode"),
        ({True: 1.0}, "launch[True]", "display label or its index"),
        ({"HE1,1": -0.1}, 'launch["HE1,1"]', "at least zero"),
        ({"HE1,1": float("nan")}, 'launch["HE1,1"]', "at least zero"),
        ({"HE1,1": True}, 'launch["HE1,1"]', "at least zero"),
        ({"HE1,1": {"phase": 1.0}}, 'launch["HE1,1"]', "takes 'share'"),
        ({"HE1,1": {"share": 1.0, "chirp": 1.0}}, 'launch["HE1,1"]', "takes 'share'"),
        ({"HE1,1": {"share": 1.0, "phase": np.inf}}, 'launch["HE1,1"]', "a phase"),
        ({"HE1,1": 0.0}, "launch", "sum to zero"),
        ({}, "launch", "maps mode labels to shares"),
        ({"HE1,1": 0.5, 0: 0.5}, "launch", "names HE1,1:even twice"),
    ],
)
def test_a_launch_that_names_no_mode_or_no_share_is_refused(launch, parameter, match):
    with pytest.raises(DriverError, match=match) as refused:
        capillary(launch=launch)
    assert refused.value.parameter == parameter
    # A value refusal, not a combination: it carries no rule.
    assert refused.value.rule is None


@pytest.mark.parametrize(
    ("launch", "parameter", "match"),
    [
        ("HE1,1", "launch", "a mapping from mode label to share"),
        ([1.0], "launch[0]", "maps mode labels to shares"),
    ],
)
def test_a_launch_of_the_wrong_shape_is_refused_without_the_runtime_hook(
    launch, parameter, match
):
    # The session's runtime type hook refuses these at the signature; a script runs
    # without it, and the driver must refuse them by name all the same.
    with pytest.raises(DriverError, match=match) as refused:
        _resolve_launch(launch, ["HE1,1:even"], 1)
    assert refused.value.parameter == parameter


class TwoModePulse:
    """A pulse that builds into two modes, as an array launch of one's own might."""

    def build(self, grid):
        one = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
        return np.concatenate([one, one])


def test_a_pulse_that_builds_into_several_modes_is_refused():
    grid = capillary().grid
    labels = [mode.channel_label for mode in capillary().modes]
    with pytest.raises(DriverError, match="builds into 2 modes") as refused:
        launch_pulse(TwoModePulse(), {"HE1,1": 1.0}, grid, labels)
    assert refused.value.parameter == "launch"


def test_a_label_that_leaves_a_parity_open_is_refused_naming_both():
    modes = ModeCollection.with_polarisations(
        [MarcatiliMode(50e-6, "Ar", 1.0, m=m) for m in (1, 2)]
    ).modes
    keywords = {
        "wavelength": 800e-9,
        "wavelength_limits": (450e-9, 1200e-9),
        "trange": 100e-15,
        "fwhm": 30e-15,
        "energy": 1e-8,
        "envelope": True,
        "plasma": False,
        "raman": False,
        "n_save": 2,
        "backend": "numpy_ref",
        "noise": None,
    }
    with pytest.raises(DriverError, match="leaves a parity open") as refused:
        Simulation.from_waveguide(
            modes, 1e-3, "Ar", 1.0, launch={"HE1,1": 1.0}, **keywords
        )
    assert "['HE1,1:even', 'HE1,1:odd']" in str(refused.value)
    simulation = Simulation.from_waveguide(
        modes, 1e-3, "Ar", 1.0, launch={"HE1,1:odd": 0.5, 2: 0.5}, **keywords
    )
    assert record_of(simulation)["pulses"][0]["labels"] == [
        "HE1,1:odd",
        "HE1,2:even",
    ]
    energies = mode_energies(simulation)
    assert_allclose(energies / energies.sum(), [0, 0.5, 0.5, 0], rtol=SHARE_RTOL)


def test_the_general_route_refuses_a_launch_beside_channel_or_an_incident_beam(
    pair_guide, grid
):
    with pytest.raises(DriverError, match="generalises channel") as refused:
        general(pair_guide, grid, channel=1, launch={"HE1,1:odd": 1.0})
    assert refused.value.parameter == ("launch", "channel")
    assert refused.value.rule == "launch-or-channel"
    beam = SpatialPulse(
        GaussianPulse(wavelength=835e-9, fwhm=20e-15, energy=1e-9), 1e-6
    )
    beside: dict[str, Any]
    for beside in (
        {"incident": beam},
        {"coupling": CouplingModel("vector_paraxial", model_error=0.2)},
    ):
        [name] = beside
        with pytest.raises(DriverError, match="two ways to say") as refused:
            Simulation.from_waveguide(
                (),
                GENERAL_Z_END,
                provider=pair_guide,
                grid=grid,
                wavelength=835e-9,
                launch={"HE1,1:even": 1.0},
                **beside,
            )
        assert refused.value.parameter == ("launch", name)
        assert refused.value.rule == "launch-or-incident"


# ----------------------------------------------------------------- general modal


def test_in_a_guide_s_own_coordinates_the_shares_are_energy_fractions(pair_guide, grid):
    simulation = general(pair_guide, grid, launch={"HE1,1:even": 3, "HE1,1:odd": 1})
    energies = mode_energies(simulation)
    assert_allclose(energies / energies.sum(), [0.75, 0.25], rtol=SHARE_RTOL)
    assert_allclose(energies.sum(), PULSE["energy"], rtol=1e-6)
    assert record_of(simulation)["split"] == "energy"
    # The guide's channels are what channel= names, so one channel at a share of one
    # is the channel launch.
    one = general(pair_guide, grid, launch={1: 1.0})
    assert_allclose(
        one._field0, general(pair_guide, grid, channel=1)._field0, rtol=1e-15
    )


def test_on_a_declared_basis_the_flux_metric_energy_is_the_pulse_s(
    circular_guide, grid
):
    simulation = general(
        circular_guide,
        grid,
        launch={"HE1,1:left": 0.5, "HE1,1:right": {"share": 0.5, "phase": 1.0}},
    )
    record = record_of(simulation)
    assert record["split"] == "coordinate"
    assert record["pulses"][0]["labels"] == ["HE1,1:left", "HE1,1:right"]
    assert simulation.general is not None
    metric = metric_at(simulation.general.metric_tables, 0.0)
    assert metric is not None
    built = GaussianPulse(
        wavelength=PULSE["wavelength"], fwhm=PULSE["fwhm"], energy=PULSE["energy"]
    ).build(simulation.grid)
    # The coefficients are scaled to the pulse's energy under the metric: a ratio of
    # two sums over ~50 bins, a square root and one multiplication of rounding.
    assert_allclose(
        float(physical_energy(simulation.grid, simulation._field0, metric=metric)),
        float(energy(simulation.grid, built).sum()),
        rtol=1e-13,
    )
    # The coordinate split is the shares, to the metric's departure from the
    # identity, which the guide's own gate holds below a part in a thousand.
    energies = mode_energies(simulation)
    assert_allclose(energies / energies.sum(), [0.5, 0.5], rtol=1e-3)


# --------------------------------------------------------------- the record


def test_the_record_is_compared_and_absent_reads_as_no_launch():
    found, value = neutral_value(("configuration", "modal_launch"))
    assert found and value is None
    default = capillary().metadata()
    one = capillary(launch={"HE1,1": 0.9, "HE1,2": 0.1}).metadata()
    other = capillary(launch={"HE1,1": 0.5, "HE1,2": 0.5}).metadata()
    path = ("configuration", "modal_launch")
    assert one.compare(capillary(launch={"HE1,1": 9, "HE1,2": 1}).metadata()).continues
    for stored, requested in ((one, other), (default, one), (one, default)):
        comparison = stored.compare(requested)
        assert not comparison.continues
        assert any(change.path[:2] == path for change in comparison.refused), (
            comparison.summary()
        )
        assert not comparison.layout
    assert path in default.compare(one).neutral


def test_a_file_written_with_one_launch_refuses_another(tmp_path):
    path = tmp_path / "run.h5"
    launch = {"HE1,1": 0.9, "HE1,2": 0.1}
    capillary(launch=launch).run(path, progress="none")
    with pytest.raises(OutputError, match="configuration"):
        capillary(launch={"HE1,1": 0.5, "HE1,2": 0.5}).run(
            path, progress="none", resume=True
        )
    with pytest.raises(OutputError, match="configuration"):
        capillary().run(path, progress="none", resume=True)
    # The same launch, written another way, continues: the file is complete.
    assert (
        capillary(launch={0: 9, 1: 1}).run(path, progress="none", resume=True).n_saved
        == CAPILLARY["n_save"]
    )


def test_a_replaced_input_drops_the_launch_record():
    simulation = capillary(launch={"HE1,1": 0.9, "HE1,2": 0.1})
    replaced = simulation.with_input(simulation._field0 * 0.5)
    assert "modal_launch" not in replaced.configuration


# ------------------------------------------------------------------- the backends


@pytest.mark.medium
@pytest.mark.parametrize(
    "backend",
    [
        "numpy_ref",
        "numba_cpu",
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                "native" not in available(), reason="solprop-native is not installed"
            ),
        ),
    ],
)
def test_a_launched_run_saves_its_launch_at_the_entrance(backend):
    launch = {"HE1,1": 0.6, "HE1,2": {"share": 0.3, "phase": 0.5}, "HE1,3": 0.1}
    simulation = capillary(backend=backend, launch=launch)
    output = simulation.run(progress="none")
    assert output.n_saved == CAPILLARY["n_save"]
    entrance = energy(simulation.grid, np.asarray(output["field"][0])).sum(axis=1)
    assert_allclose(entrance / entrance.sum(), [0.6, 0.3, 0.1], rtol=SHARE_RTOL)
    assert output.metadata is not None
    recorded = output.metadata.configuration["modal_launch"]
    assert recorded == record_of(simulation)
