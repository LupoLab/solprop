# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the supplied-mode interface and distinct rectangular partner rows."""

import warnings
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from march_bounds import FIXED_ORDER, STATISTIC_FACTOR, assert_matched, simulation_bound
from solprop.driver import Simulation, prop_waveguide
from solprop.exceptions import (
    DriverError,
    MaterialRangeWarning,
    ModeError,
    TranslationError,
)
from solprop.modes import IndexTable, MarcatiliMode, RectMode, StepIndexMode
from solprop.modes.quadrature import cartesian_nodes
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.physdata import Mixture, aluminium
from solprop.profiles import Linear
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

# The aluminium-walled guides here pass the first-order wall-pair model's 0.1 on
# lambda |q| / (2 pi a) -- 0.11 at 800 nm on the 20 um side, 0.18 at 2.3 um in the
# grid's long-wavelength collar on the 100 um side -- and say so once per mode
# (GrazingIncidenceWarning); these tests check the route and its records, and
# the warning itself is qualified in tests/modes/test_rectangular.py.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GrazingIncidenceWarning"
)


def settings(**changes):
    return {
        "lambda0": 800e-9,
        "wavelength_limits": (450e-9, 1200e-9),
        "trange": 100e-15,
        "fwhm": 30e-15,
        "energy": 1e-8,
        "envelope": True,
        "plasma": False,
        "raman": False,
        "n_r": 12,
        "n_theta": 12,
        "n_save": 2,
        "backend": "numpy_ref",
        **changes,
    }


def declared(modes, **changes):
    return Simulation.from_waveguide(modes, 0.001, "Ar", 5.0, **settings(**changes))


@pytest.mark.medium
def test_waveguide_rectangular_backends_records_and_mixture_update():
    mode = RectMode(50e-6, 10e-6, cladding=aluminium())
    simulation = declared([mode])
    assert simulation.settings.components is None
    assert simulation.settings.quadrature is not None
    assert simulation.settings.stats is not None
    assert simulation.settings.quadrature.kind == "cartesian"
    assert "core_radius" not in simulation.profiles
    assert "core_radius" not in simulation.settings.stats.names
    # A Cartesian product rule carries no Kronrod extension, so the quadrature
    # alarm has no error estimate and the statistic would be NaN at every plane.
    assert not simulation.settings.quadrature.kronrod
    assert "quadrature_error" not in simulation.settings.stats.names
    record = simulation.configuration["waveguide"]
    assert isinstance(record, dict)
    assert record["modes"][0].kind.endswith("RectMode")
    outputs = []
    for backend in ("numpy_ref", "numba_cpu"):
        # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
        with nullcontext():
            output = prop_waveguide(
                [mode], 0.001, "Ar", 5.0, **settings(backend=backend), progress="none"
            )
        outputs.append(output["field"])
        assert output.metadata is not None
        assert output.metadata.modes[0].kind.endswith("RectMode")
        for name in simulation.settings.stats.names:
            assert np.all(np.isfinite(output[f"stats/{name}"]))
    # Same operation order with FFT and reduction rounding accumulated over steps.
    assert_allclose(
        outputs[1], outputs[0], rtol=2e-11, atol=2e-11 * np.max(np.abs(outputs[0]))
    )
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        rebound = simulation.with_mixture(Mixture.fill({"Ar": 1.0}, 6.0))
    rebound_record = rebound.configuration["waveguide"]
    assert isinstance(rebound_record, dict)
    assert rebound_record["modes"][0].kind.endswith("RectMode")


@pytest.mark.medium
def test_waveguide_polarisation_partners_keep_independent_operators():
    mode = RectMode(50e-6, 10e-6, cladding=aluminium())
    simulation = declared(
        [mode, mode.orthogonal_polarisation], stats=StatsSpec(names=("energy",))
    )
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        problem = simulation.translate()
    assert problem.transform.n_modes == 2 and problem.transform.n_pol == 2
    assert problem.linear.phi_coefficients is not None
    assert not np.array_equal(
        problem.linear.phi_coefficients[..., 0, :, :],
        problem.linear.phi_coefficients[..., 1, :, :],
    )
    # Both rows must be excited to qualify the propagation of each partner.
    initial = simulation._field0 / np.sqrt(2)
    initial = np.stack([initial[0], initial[0]])
    vector = Simulation(
        simulation.grid,
        simulation.modes,
        simulation.responses,
        initial,
        settings=simulation.settings,
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    )
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always", MaterialRangeWarning)
        result = vector.run(progress="none")
    assert all(isinstance(item.message, MaterialRangeWarning) for item in diagnostics)
    assert np.any(np.abs(result["field"][-1, 0] - result["field"][-1, 1]) > 0)
    duplicate = Simulation(
        simulation.grid,
        [simulation.modes[0]],
        simulation.responses,
        np.repeat(initial[:1], 2, axis=1),
        settings=replace(simulation.settings, components=("x",)),
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    )
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with pytest.raises(TranslationError, match="distinct effective indices"):
        duplicate.translate()


@pytest.mark.medium
def test_waveguide_capillary_construction_and_run_agree():
    # The two entries build the same launch, exactly, and the same problem: on a
    # prescribed schedule at a fixed order the two runs are one march, within its
    # rounding budget (SEP 25).
    mode = MarcatiliMode(125e-6, cladding=1.45)
    options = settings(
        n_theta=None,
        stats=StatsSpec(names=("energy",)),
        schedule=FixedSchedule.uniform(0.0, 0.001, 1e-4, save_planes=[]),
        integrator=FIXED_ORDER,
    )
    capillary = Simulation.from_capillary(
        125e-6, 0.001, "Ar", 5.0, modes=[mode], **options
    )
    waveguide = Simulation.from_waveguide([mode], 0.001, "Ar", 5.0, **options)
    # Ten steps of 0.1 mm are past the in-stage collar's explicit limit; split off,
    # it is integrated exactly.
    for one in (capillary, waveguide):
        one.settings = replace(one.settings, absorber_scheme="split_step")
    assert_array_equal(capillary._field0, waveguide._field0)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        first = capillary.run(progress="none")
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        second = waveguide.run(progress="none")
    bound = simulation_bound(waveguide)
    assert_matched(second["field"], first["field"], bound, what="field")
    assert_matched(
        second["stats/energy"],
        first["stats/energy"],
        STATISTIC_FACTOR * bound,
        what="energy",
    )
    # The one thing an explicit stats= hides: the default specifications, which
    # the two entries build by different helpers, and the profile paths.
    default = settings(n_theta=None)
    capillary = Simulation.from_capillary(
        125e-6, 0.001, "Ar", 5.0, modes=[mode], **default
    )
    waveguide = Simulation.from_waveguide([mode], 0.001, "Ar", 5.0, **default)
    assert capillary.settings.stats is not None
    assert waveguide.settings.stats is not None
    # Field by field, because StatsSpec carries an array and == is ambiguous.
    assert capillary.settings.stats.names == waveguide.settings.stats.names
    assert "quadrature_error" in capillary.settings.stats.names
    assert_array_equal(
        capillary.settings.stats.energy_windows,
        waveguide.settings.stats.energy_windows,
    )
    assert capillary.settings.stats.periodic == waveguide.settings.stats.periodic
    assert capillary.settings.stats.period == waveguide.settings.stats.period
    assert set(capillary.profiles) == set(waveguide.profiles)
    for name, path in capillary.profiles.items():
        positions = np.asarray([0.0, 0.0005, 0.001])
        assert_array_equal(path(positions), waveguide.profiles[name](positions))


def test_waveguide_refuses_empty_non_gas_and_nonorthogonal_sets():
    with pytest.raises(DriverError, match="a waveguide mode set needs at least one"):
        declared([])
    # The capillary entry keeps its own wording and its own exception class.
    with pytest.raises(DriverError, match="a capillary mode set needs at least one"):
        Simulation.from_capillary(125e-6, 0.001, "Ar", 5.0, modes=[], **settings())
    with pytest.raises(DriverError, match=r"prop_gnlse\(fibre="):
        declared([StepIndexMode(1.25e-6, profile_wavelength=835e-9)])
    # The hint is the step-index route only; anything else gets the plain refusal.
    table = IndexTable(np.linspace(1e15, 4e15, 4), np.full(4, 1.45 + 1e-9j))
    supplied = SpecifiedMode.from_index_table(table, aeff=1e-10)
    with pytest.raises(DriverError, match="cannot replace its material") as refusal:
        Simulation.from_capillary(
            125e-6, 0.001, "Ar", 5.0, modes=[supplied], **settings()
        )
    assert "prop_gnlse" not in str(refusal.value)
    mode = RectMode(50e-6, 10e-6)
    # A repeated channel is refused by its label, before any overlap is formed.
    with pytest.raises(ModeError, match="duplicate physical channel"):
        declared([mode, mode])
    # The orthonormality gate itself needs two distinct channels that are not
    # orthogonal: a capillary mode and a partner turned by 0.3 rad overlap by
    # cos(0.3). A rectangular guide has no such pair -- its two polarisations are
    # orthogonal and its (n, m) rows are distinct modes.
    turned = MarcatiliMode(125e-6, "Ar", 5.0)
    with pytest.raises(DriverError, match="not an orthonormal set"):
        declared([turned, turned.rotated(0.3)])


def test_capillary_refuses_a_declaration_a_supplied_set_would_ignore():
    """A supplied set is the geometry; a declared argument it ignores is named.

    The radius is positional and so always declared: a different bore, and a taper
    beside a constant-bore set, are refused by the argument's name. `model`, `loss`
    and `cladding` are refused only away from their defaults and away from the
    modes' own value, so a consistent call -- the same number, the same profile, or
    the modes' own setting repeated -- stays admitted.
    """

    def build(radius, modes, **changes):
        return Simulation.from_capillary(
            radius, 0.001, "Ar", 5.0, modes=modes, **settings(n_theta=None, **changes)
        )

    bore = [MarcatiliMode(125e-6, m=index) for index in (1, 2)]
    with pytest.raises(DriverError, match="radius disagrees with supplied mode 0"):
        build(100e-6, bore)
    taper = Linear(125e-6, 100e-6, 0.001)
    with pytest.raises(DriverError, match=r"radius disagrees .*\(HE1,1:even\)"):
        build(taper, bore)
    with pytest.raises(DriverError, match="radius disagrees with supplied mode 0"):
        build(125e-6, [MarcatiliMode(taper)])
    for name, value in (
        ("model", "reduced"),
        ("loss", False),
        ("loss", 0.5),
        ("cladding", 1.45),
        ("loss", lambda omega: omega),
    ):
        with pytest.raises(DriverError, match=f"from_capillary's {name}="):
            build(125e-6, bore, **{name: value})
    # Consistent declarations: the same number, an equal profile built again, the
    # modes' own settings repeated, and the defaults beside modes that differ.
    build(125e-6, bore)
    build(Linear(125e-6, 100e-6, 0.001), [MarcatiliMode(taper)])
    own = [
        MarcatiliMode(125e-6, m=index, model="reduced", loss=False, cladding=1.45)
        for index in (1, 2)
    ]
    build(125e-6, own, model="reduced", loss=False, cladding=1.45)
    build(125e-6, own)
    # A set with no core radius at all ignores the radius whatever it says.
    with pytest.raises(DriverError, match="declares no core radius"):
        build(50e-6, [RectMode(50e-6, 10e-6)])


def test_waveguide_cartesian_defaults_reach_the_aliasing_bound():
    for (n, m), order, expected in (((1, 1), 1, 12), ((3, 2), 3, 24)):
        mode = RectMode(50e-6, 10e-6, n=n, m=m, cladding=aluminium())
        assert mode.azimuthal_order == order
        simulation = Simulation.from_waveguide(
            [mode], 0.001, "Ar", 5.0, **settings(n_r=None, n_theta=None)
        )
        rule = simulation.settings.quadrature
        assert rule is not None and rule.kind == "cartesian"
        # n_r and n_theta are the x and y counts here; both defaults must reach
        # the measured quartic bound.
        assert expected == cartesian_nodes(order)
        assert min(rule.n_r, rule.n_theta) >= expected


def test_waveguide_mixture_and_geometry_paths_are_rebound():
    path = Linear(50e-6, 45e-6, 0.001)
    mode = RectMode(path, path, cladding=aluminium())
    mixture = Mixture.fill({"He": 0.25, "Ar": 0.75}, 5.0)
    simulation = Simulation.from_waveguide([mode], 0.001, mixture, **settings())
    assert "half_width_x" in simulation.profiles
    assert "half_width_y" in simulation.profiles
    assert isinstance(simulation.modes[0], RectMode)
    assert simulation.modes[0].mixture is mixture
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        problem = simulation.translate()
    assert problem.transform.n_modes == 1
    with pytest.raises(DriverError, match="requires supplied"):
        Simulation._from_hollow(None, 0.001, "Ar", 5.0, waveguide=True, **settings())
