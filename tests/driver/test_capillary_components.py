# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`prop_capillary` resolves the components a supplied mode set needs.

Synthesising E_x alone for every set would lose the E_y of a supplied TE_0m, TM_0m, EH
or rotated set: TE_01's Kerr overlap would be 3/8 of the vector one and the plasma
would see |E_x| alone. A supplied set resolves its components as `prop_waveguide`
does; the capillary's own HE_1m set keeps ``("x",)``. It does so beside a coherent
wall too.
"""

import functools
import logging
import math
from dataclasses import replace

import numpy as np
import pytest

from driver.test_coherent_capillary import with_flow
from flow_parity import build, run
from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.driver.simulation import Simulation
from solprop.modes import CoherentWall, MarcatiliMode
from solprop.stepper.schedule import FixedSchedule

RADIUS = 125e-6
"""Core radius, m."""

LENGTH = 1e-3
"""Capillary length, m: ten fixed steps of 0.1 mm."""

STEPS = 10

# The helium index is extrapolated beyond its fitted range at the grid's ends; that
# is the material model, not the components under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


def _keywords(*, plasma=True, **overrides):
    """Return the cheap capillary's keywords on the ten-step schedule."""
    keywords = {
        "wavelength": 800e-9,
        "wavelength_limits": (200e-9, 2e-6),
        "trange": 100e-15,
        "fwhm": 10e-15,
        "energy": 20e-6,
        "n_r": 8,
        "n_save": 2,
        "plasma": plasma,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        # No order choice on the prescribed schedule: two runs are one march.
        "integrator": FIXED_ORDER,
        "backend": "numpy_ref",
    }
    if plasma:
        # A TE_0m or TM_0m field vanishes on the axis, where the ionisation table's
        # range is otherwise read from.
        keywords["ionisation_field_range"] = (1e8, 1e12)
    return keywords | overrides


def _weak_collar(simulation):
    """Refer the collar to a metre, weak enough for ten explicit 0.1 mm steps.

    The default refers it to the millimetre run, and an explicit method would need
    four times the steps (the in-stage temporal collar is an explicit term).
    """
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=1.0)
    )
    return simulation


def _family(kind, order, count=2):
    """Return the first radial orders of one family in the helium-filled bore."""
    return [
        MarcatiliMode(RADIUS, "He", 1.0, kind=kind, n=order, m=m)
        for m in range(1, count + 1)
    ]


def _capillary(modes, **overrides):
    """Build the capillary entry's run of `modes`, a count or a supplied set."""
    return _weak_collar(
        Simulation.from_capillary(
            RADIUS,
            LENGTH,
            "He",
            1.0,
            **{"noise": None, **_keywords(modes=modes, **overrides)},
        )
    )


@functools.cache
def _field(entry, kind, backend):
    """Ten steps of a two-channel TE_0m or TM_0m set through one entry point."""
    modes = _family(kind, 0)
    if entry == "capillary":
        simulation = _capillary(modes, backend=backend)
    else:
        simulation = _weak_collar(
            Simulation.from_waveguide(
                modes,
                LENGTH,
                "He",
                1.0,
                **{"noise": None, **_keywords(backend=backend)},
            )
        )
    return np.asarray(simulation.run()["field"])


@pytest.mark.medium
@pytest.mark.parametrize("kind", ["TE", "TM"])
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_supplied_te_or_tm_set_through_prop_capillary_is_the_waveguide_run(
    kind, backend
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    problem = _capillary(_family(kind, 0), backend=backend).translate()
    assert problem.transform.n_pol == 2
    field = _field("capillary", kind, backend)
    reference = _field("waveguide", kind, "numpy_ref")
    if backend == "numpy_ref":
        # The two entries translate the same set into the same problem, so the two
        # runs are one march, within its rounding budget.
        assert_matched(field, reference, simulation_bound(_capillary(_family(kind, 0))))
        return
    # The same ten fixed steps: what differs is the order of the floating-point
    # sums, a few ulps per step compounded over ten.
    scale = float(np.linalg.norm(reference[-1]))
    assert np.linalg.norm(field[-1] - reference[-1]) / scale < 1e-10


def _kerr_overlap(problem):
    """Return ``sum P |E|^2 E`` of the first mode: its Kerr self-overlap."""
    transform = problem.transform
    n_pol = transform.n_pol
    synthesis = np.asarray(transform.synthesis)[:, 0].reshape(-1, n_pol)
    projection = np.asarray(transform.projection)[0].reshape(-1, n_pol)
    intensity = np.sum(synthesis**2, axis=1, keepdims=True)
    return float(np.sum(projection * intensity * synthesis))


def test_the_te01_kerr_overlap_is_eight_thirds_of_the_x_only_one():
    # TE_01's field is f(r) theta-hat, so E_x = -f sin(theta): the x-only overlap
    # integrates sin^4 over the angle, 3 pi / 4, where the vector one integrates 1,
    # 2 pi. The radial integral and every normalisation are shared, and the
    # trapezoid rule of 16 angles is exact for sin^4, so the ratio is 8/3 to rounding.
    vector = _capillary(_family("TE", 0, count=1), plasma=False).translate()
    # The behaviour before the fix: the same set, with E_x alone.
    old = _capillary(_family("TE", 0, count=1), plasma=False)
    old.settings = replace(old.settings, components=("x",))
    scalar = old.translate()
    assert (vector.transform.n_pol, scalar.transform.n_pol) == (2, 1)
    ratio = _kerr_overlap(vector) / _kerr_overlap(scalar)
    assert ratio == pytest.approx(8.0 / 3.0, rel=1e-12)
    assert 2.0 * math.pi / (3.0 * math.pi / 4.0) == pytest.approx(8.0 / 3.0)


@pytest.mark.medium
def test_a_y_polarised_he11_set_is_the_x_polarised_run_turned():
    # Before the fix a quarter-turned HE_11 was synthesised along x, where its field
    # is zero, and propagated without a nonlinearity. Its only component is now y.
    turned = [MarcatiliMode(RADIUS, "He", 1.0, phi=math.pi / 2)]
    simulation = _capillary(turned, plasma=False)
    assert simulation.translate().transform.n_pol == 1
    field = np.asarray(simulation.run()["field"])
    reference = np.asarray(_capillary(1, plasma=False).run()["field"])
    # The quarter turn maps the angular rule's nodes onto each other, but only to the
    # rounding of the node angles and of sin and cos there (measured: bitwise).
    scale = float(np.linalg.norm(reference[-1]))
    assert np.linalg.norm(field[-1] - reference[-1]) / scale < 1e-12


@pytest.mark.medium
def test_the_capillarys_own_set_keeps_x_and_an_unrotated_supplied_set_matches_it():
    own = _capillary(2, plasma=False)
    supplied = _capillary(_family("HE", 1), plasma=False)
    assert own.settings.components == ("x",)
    assert supplied.settings.components is None
    assert supplied.translate().transform.n_pol == 1
    # The same problem either way: one march, within its rounding budget.
    assert_matched(
        np.asarray(supplied.run()["field"]),
        np.asarray(own.run()["field"]),
        simulation_bound(own),
    )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
def test_a_wall_beside_a_two_component_set_resolves_both_components(caplog):
    # The backends admit a coupled flow beside two synthesised components: the flow
    # contracts the channel axis
    # of the one-component state and never reads the synthesis. So a wall keeps the
    # set's inference, and nothing is logged.
    with caplog.at_level(logging.WARNING, logger="solprop.driver.simulation"):
        simulation = _capillary(
            _family("TE", 0), wall=CoherentWall(impedance="te_tm"), plasma=False
        )
        problem = simulation.translate()
    assert simulation.settings.components is None
    assert (problem.transform.n_pol, problem.shape[2]) == (2, 1)
    assert problem.linear.coupled_flow is not None
    walled = _capillary(_family("HE", 1), wall=CoherentWall(), plasma=False)
    assert walled.settings.components is None
    assert walled.translate().transform.n_pol == 1
    # The resume identity records what the synthesis resolves.
    walled_te, walled_he = simulation.metadata().modal, walled.metadata().modal
    assert walled_te is not None and walled_he is not None
    assert walled_te["n_components"] == 2
    assert walled_he["n_components"] == 1
    assert not caplog.records


@pytest.mark.medium
@pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GrazingIncidenceWarning",
    "ignore:the temporal absorber attenuates",
)
@pytest.mark.parametrize("kind", ["te", "tm"])
def test_a_negligible_wall_beside_a_two_component_set_is_the_uncoupled_run(kind):
    # The TE/TM wall beside a TE_0m or TM_0m pair, a launch strong enough that the
    # Kerr half (now with E_y) is exercised (``tests/flow_parity.py``).
    simulation = build(kind)
    problem = simulation.translate()
    assert problem.transform.n_pol == 2
    # The wall changes the linear operator alone: the synthesis and projection are
    # the wall-free run's, bit for bit.
    bare = build(kind, wall=None).translate()
    np.testing.assert_array_equal(problem.transform.synthesis, bare.transform.synthesis)
    np.testing.assert_array_equal(
        problem.transform.projection, bare.transform.projection
    )
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    generator = flow.generator[0, 0]
    grid = simulation.grid
    uncoupled = replace(problem, linear=replace(problem.linear, coupled_flow=None))
    # A wall made zero: exactly the run without a flow, an explicitly exact no-op
    # (SEP 25 keeps those structural). The Schur factors of a zero block are exact,
    # and so is multiplying by an exact identity.
    zeroed = with_flow(problem, np.zeros_like(generator))
    np.testing.assert_array_equal(
        run(NumpyRefBackend, zeroed, grid), run(NumpyRefBackend, uncoupled, grid)
    )
    # The wall's coupling made zero: the diagonal flow is the same diagonal inside
    # Phi, to the rounding of one exponential and one dot product per stage, a few
    # hundred roundings over ten steps (the scalar wall's bound, 1e-12).
    index = np.arange(flow.n_modes)
    diagonal = np.zeros_like(generator)
    diagonal[index, index] = generator[index, index]
    assert np.max(abs(generator - diagonal)) > 0.0
    constant = problem.linear.phi_constant
    assert constant is not None
    folded = constant.copy()
    folded[0, :, 0, :] += diagonal[index, index]
    inside = replace(
        problem,
        linear=replace(uncoupled.linear, phi_constant=folded),
    )
    coupled = run(NumpyRefBackend, with_flow(problem, diagonal), grid)
    reference = run(NumpyRefBackend, inside, grid)
    scale = float(np.max(abs(coupled)))
    np.testing.assert_allclose(coupled, reference, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.medium
@pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GrazingIncidenceWarning",
    "ignore:the temporal absorber attenuates",
)
@pytest.mark.parametrize("kind", ["te", "tm"])
def test_beside_the_wall_the_kerr_increment_is_eight_thirds_of_the_x_only_one(kind):
    # TE_01 and TM_01 have |E| independent of the angle, so resolving E_y beside the
    # wall raises the Kerr self-overlap by 2 pi / (3 pi / 4) = 8/3, as without a wall
    # (the test above). The ten-step nonlinear increment -- the run minus the same
    # launch scaled down a thousandfold and back up -- is first order in the Kerr
    # term, so its ratio is 8/3 up to the second-order part, relative O(Phi) with the
    # launch's Phi of order 1e-4 (``tests/flow_parity.py``): bound 1e-3.
    def increment(simulation):
        problem = simulation.translate()
        weak = replace(
            problem,
            initial=replace(problem.initial, field0=problem.initial.field0 * 1e-3),
        )
        grid = simulation.grid
        full = run(NumpyRefBackend, problem, grid)
        return float(np.linalg.norm(full - 1e3 * run(NumpyRefBackend, weak, grid)))

    vector = build(kind)
    scalar = build(kind)
    scalar.settings = replace(scalar.settings, components=("x",))
    assert vector.translate().linear.coupled_flow is not None
    ratio = increment(vector) / increment(scalar)
    assert ratio == pytest.approx(8.0 / 3.0, rel=1e-3)
