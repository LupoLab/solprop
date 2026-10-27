# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`mode_average=True` runs a hollow guide's one mode mode-averaged.

The switch is the one-node transform of the multimode pipeline, the route a
`Simulation` with no transverse quadrature already took: the capillary's run under it
is bitwise the run with the quadrature removed by hand. What it adds is the capillary's
own statistics, plasma and species included, and refusals for the keywords that shape
a transverse rule it does not have.
"""

from dataclasses import replace

import numpy as np
import pytest

from solprop.driver import Simulation, prop_capillary, prop_waveguide
from solprop.exceptions import DriverError
from solprop.modes import MarcatiliMode, TubeLattice
from solprop.profiles import Linear
from solprop.stats import default_spec

RADIUS = 100e-6
"""Core radius, m."""

LENGTH = 5e-3
"""Capillary length, m: a few adaptive steps."""

# The helium index is extrapolated beyond its fitted range at the grid's ends; that
# is the material model, not the switch under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


def _keywords(**overrides):
    """Return a short, ionising helium capillary's keywords."""
    keywords = {
        "lambda0": 800e-9,
        "wavelength_limits": (150e-9, 4e-6),
        "trange": 200e-15,
        "fwhm": 30e-15,
        "energy": 200e-6,
        "n_save": 2,
        "backend": "numpy_ref",
        "progress": "none",
    }
    return keywords | overrides


@pytest.mark.medium
def test_the_switch_is_the_run_without_the_quadrature():
    """The keyword's run is bitwise the hand-built averaged run, with more stats."""
    keywords = _keywords()
    progress = keywords.pop("progress")
    by_hand = Simulation.from_capillary(
        RADIUS, LENGTH, "He", 2.0, **{"noise": None, **keywords}
    )
    by_hand.settings = replace(
        by_hand.settings,
        quadrature=None,
        components=None,
        stats=default_spec("mode_averaged"),
    )
    expected = by_hand.run(progress=progress)
    output = prop_capillary(
        RADIUS, LENGTH, "He", 2.0, **{"noise": None, **_keywords()}, mode_average=True
    )
    assert output.metadata is not None
    assert output.metadata.simulation_kind == "mode_averaged"
    assert output.metadata.quadrature is None
    np.testing.assert_array_equal(
        np.asarray(output["field"]), np.asarray(expected["field"])
    )
    names = set(output.stat_names)
    assert {
        "electron_density",
        "peak_ionisation_rate",
        "peak_intensity",
        "species_electron_density",
        "density",
        "pressure",
        "core_radius",
    } <= names
    assert "quadrature_error" not in names
    assert "peak_power_total" not in names


def test_the_default_stays_resolved():
    """Without the switch, one mode is resolved on the radial rule."""
    keywords = _keywords()
    keywords.pop("progress")
    simulation = Simulation.from_capillary(
        RADIUS, LENGTH, "He", 2.0, **{"noise": None, **keywords}
    )
    assert simulation.settings.quadrature is not None
    assert not simulation.translate().flags.mode_averaged


@pytest.mark.medium
@pytest.mark.parametrize(
    ("radius", "gas", "extra"),
    [
        (Linear(RADIUS, 0.8 * RADIUS, LENGTH), "He", {}),
        (RADIUS, "He", {"envelope": True}),
        (
            15e-6,
            "Ar",
            {
                "lattice": TubeLattice(0.7e-6, 5e-6, 6),
                "energy": 1e-6,
                "wavelength_limits": (300e-9, 3e-6),
            },
        ),
    ],
    ids=["taper", "envelope", "antiresonant"],
)
def test_the_switch_runs_the_capillary_variants(radius, gas, extra):
    """A tapered bore, an envelope grid and a tube lattice run mode-averaged."""
    output = prop_capillary(
        radius,
        LENGTH,
        gas,
        2.0,
        **{"noise": None, **_keywords(**extra)},
        mode_average=True,
    )
    assert output.metadata is not None
    assert output.metadata.simulation_kind == "mode_averaged"


@pytest.mark.medium
def test_the_switch_runs_a_supplied_gas_mode():
    """`prop_waveguide` takes the switch for its supplied mode."""
    output = prop_waveguide(
        [MarcatiliMode(RADIUS)],
        LENGTH,
        "He",
        2.0,
        **{"noise": None, **_keywords()},
        mode_average=True,
    )
    assert output.metadata is not None
    assert output.metadata.simulation_kind == "mode_averaged"


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"modes": 2}, "needs exactly one mode"),
        ({"n_r": 8}, "n_r given beside mode_average=True"),
        ({"n_theta": 4}, "n_theta given beside mode_average=True"),
        ({"radial_rule": "gauss"}, "radial_rule given beside mode_average=True"),
    ],
)
def test_the_switch_refuses_what_needs_a_rule(given, match):
    """More than one mode, or a keyword that shapes the rule, is refused by name."""
    keywords = _keywords()
    keywords.pop("progress")
    with pytest.raises(DriverError, match=match):
        Simulation.from_capillary(
            RADIUS,
            LENGTH,
            "He",
            2.0,
            **{"noise": None, **keywords},
            **given,
            mode_average=True,
        )
