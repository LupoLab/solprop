# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The taper coupling of a multimode capillary, on by default.

A multimode capillary whose radius varies along z carries the moving-basis connection
``<psi_m|d_z psi_n>`` between its ``HE_1m`` channels by default, lossless or not, in the
frozen-residual coupled flow the varying coherent wall uses with the wall term zero;
``taper_coupling=False`` is the explicit opt-out to the adiabatic route. This module
checks which sets carry it, the refusals, the record, the adiabatic limit against the
first-order estimate, flux conservation, the opt-out's route and the backends that run
it. Every run is ten steps or a few tens of steps over at most a millimetre and a
quarter.
"""

import itertools
import math
import warnings
from typing import cast

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from driver.test_coherent_capillary import (
    GRID,
    LENGTH,
    STEPS,
    WEAK_COLLAR,
    taper,
    varying,
)
from driver.test_resume import run_until
from driver.test_taper_note import _study as _taper_study
from modal_provider_cases import general_grid, split_guide
from solprop.backends.registry import available
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import (
    AdiabaticTaperWarning,
    DriverError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    ModeError,
    OutputError,
    TranslationError,
)
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall, MarcatiliMode
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import (
    taper_connection,
    taper_coupling_block,
    wall_connection,
)
from solprop.output.memory import MemoryOutput
from solprop.physdata.mixtures import Mixture
from solprop.profiles import FromCallable, Linear
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.problem import TAPER_FLOW_CERTIFICATE, _walk_arrays, assemble

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
    pytest.mark.filterwarnings("ignore:.*explicit residual is the limiting"),
]
"""The qualification cases' deliberate warnings (``test_coherent_capillary.py``), and
the advisory step, which a fixed schedule chosen for a test may exceed on purpose."""


def lossless(**overrides):
    """Return the ten-step three-channel qualification taper, lossless, no wall."""
    return varying(loss=False, cladding=1.45, **overrides)


# ------------------------------------------------------------------ which sets


def test_the_coupling_applies_to_a_multimode_capillary_taper_alone():
    """Two or more contiguous plain HE_1m channels on a varying radius, nothing else.

    A pressure gradient at a constant bore is not a taper: the ``J_0(u_m r/a)`` field
    depends on the radius alone, and the effective index it changes is each channel's
    own diagonal (``MarcatiliMode.radius_is_constant``).
    """
    moving = taper()
    block = [MarcatiliMode(moving, "He", 1.0, m=m, loss=False) for m in (1, 2, 3)]
    assert taper_coupling_block(block) == tuple(block)
    assert taper_coupling_block(block[:1]) is None
    assert taper_coupling_block([block[0], block[2]]) is None
    assert taper_coupling_block(block[::-1]) is None
    assert (
        taper_coupling_block([MarcatiliMode(50e-6, "He", 1.0, m=m) for m in (1, 2)])
        is None
    )
    gradient = varying(radius=50e-6, pressure=(1.0, 0.5), modes=2)
    assert taper_coupling_block(gradient.modes) is None
    assert gradient.translate().linear.coupled_flow is None
    mixed = [
        MarcatiliMode(moving, "He", 1.0, m=1),
        MarcatiliMode(moving, "He", 1.0, kind="TE", n=0, m=1),
    ]
    assert taper_coupling_block(mixed) is None
    lattice = TubeLattice(180e-9, 9e-6, 6)
    vincetti = [VincettiMode(moving, "He", 1.0, m=m, lattice=lattice) for m in (1, 2)]
    assert taper_coupling_block(vincetti) is None


def test_the_connection_is_the_wall_s_and_admits_any_loss_and_model():
    """Bitwise the scalar wall's connection, without the wall's two conditions."""
    moving = taper()
    walled = [MarcatiliMode(moving, "He", 1.0, m=m) for m in (1, 2, 3)]
    for z in (0.0, 1e-4, 1.3e-4):
        assert np.array_equal(taper_connection(walled, z), wall_connection(walled, z))
    for modes in (
        [MarcatiliMode(moving, "He", 1.0, m=m, loss=False) for m in (1, 2)],
        [MarcatiliMode(moving, "He", 1.0, m=m, model="reduced") for m in (1, 2)],
        [MarcatiliMode(moving, "He", 1.0, m=m, loss=0.5) for m in (1, 2)],
    ):
        link = taper_connection(modes, 1e-4)
        assert link[0, 1] == -link[1, 0] != 0.0
        with pytest.raises(ModeError):
            wall_connection(modes, 1e-4)


def test_a_profile_without_an_analytic_slope_is_refused_naming_the_opt_out():
    """The connection is ``a'/a``: a sampled slope is refused, and the opt-out named."""
    radius = FromCallable(lambda z: 50e-6 - 0.05 * np.asarray(z), (0.0, LENGTH))
    assert not radius.derivative_is_analytic
    with pytest.raises(ModeError, match="taper_coupling=False"):
        lossless(radius=radius).translate()
    lossless(radius=radius, taper_coupling=False).translate()


# ------------------------------------------------------------------- the route


def test_a_lossless_multimode_taper_carries_the_coupling_by_default():
    """The default builds the taper's flow, says so in the record, and does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", AdiabaticTaperWarning)
        simulation = lossless()
        problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.certificate_id == TAPER_FLOW_CERTIFICATE
    assert set(flow.section_methods) == {"frozen_residual"}
    assert flow.connection is not None
    assert flow.wall_vectors is None
    # The lowest channel's rate is the carrier's: its diagonal is exactly zero.
    assert flow.generator is not None
    assert not np.any(flow.generator[:, :, 0, 0])
    modal = simulation.metadata().modal
    assert modal is not None
    assert modal["basis"] == "capillary HE_1m"
    record = modal["flow"]
    assert isinstance(record, dict)
    assert record["model"] == "taper coupling"
    assert record["connection"] is True
    assert record["certificate_id"] == TAPER_FLOW_CERTIFICATE
    assert "transverse_ratio" not in record
    assert "taper coupling" in str(simulation.metadata())


def test_a_lossy_multimode_taper_carries_the_coupling_by_default_too():
    """Lossless or not: the Marcatili loss rides each channel's own diagonal."""
    problem = varying().translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.certificate_id == TAPER_FLOW_CERTIFICATE
    assert flow.generator is not None
    # An upper channel attenuates faster than the lowest, which the carrier holds:
    # the real part of its mean diagonal rate is negative on the active band.
    active = GRID.active_mask
    assert np.all(np.real(flow.generator[:, 0, 2, 2, active]) < 0.0)


def test_the_opt_out_is_the_adiabatic_route_without_a_warning():
    """``taper_coupling=False`` builds no flow, carries no identity and does not warn.

    Its problem is the adiabatic diagonal route's, each channel on its own row of the
    cumulative operator, which is what the route was before the default changed (the
    stored regressions of the adiabatic runs are unmoved by it).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", AdiabaticTaperWarning)
        simulation = lossless(taper_coupling=False)
        problem = simulation.translate()
    assert problem.linear.coupled_flow is None
    assert simulation.metadata().modal is None
    direct = assemble(
        simulation.grid,
        simulation.modes,
        list(simulation.responses),
        simulation._field0,
        controller=cast("PIController", simulation.settings.controller),
        save_planes=simulation.settings.save_planes,
        dz_max=simulation.settings.dz_max,
        dz_init=simulation.settings.dz_init,
        schedule=simulation.settings.schedule,
        absorber=simulation.settings.absorber,
        quadrature=simulation.settings.quadrature,
        components=simulation.settings.components,
        stats_spec=simulation.settings.stats,
        taper_coupling=False,
    )
    assert direct.linear.coupled_flow is None
    mine, theirs = (
        [
            (name, array)
            for name, array in _walk_arrays(one)
            if name.startswith("linear")
        ]
        for one in (direct, problem)
    )
    assert [name for name, _ in mine] == [name for name, _ in theirs]
    assert all(
        np.array_equal(a, b) for (_, a), (_, b) in zip(mine, theirs, strict=True)
    )


def test_a_wall_and_the_opt_out_are_refused_together():
    """The wall's basis is attached to the radius: it cannot decline its connection."""
    with pytest.raises(DriverError, match="taper_coupling=False"):
        varying(wall=CoherentWall(), taper_coupling=False)
    simulation = varying(wall=CoherentWall())
    with pytest.raises(TranslationError, match="taper_coupling=False"):
        assemble(
            simulation.grid,
            simulation.modes,
            list(simulation.responses),
            simulation._field0,
            controller=cast("PIController", simulation.settings.controller),
            save_planes=simulation.settings.save_planes,
            dz_max=simulation.settings.dz_max,
            dz_init=simulation.settings.dz_init,
            wall=CoherentWall(),
            taper_coupling=False,
        )


def test_an_envelope_grid_beside_the_coupling_is_refused_naming_the_opt_out():
    """No coupled flow is tabulated on an envelope grid's FFT-ordered axis."""
    with pytest.raises(TranslationError, match="taper_coupling=False"):
        lossless(
            grid=None,
            wavelength_limits=(400e-9, 1.6e-6),
            trange=300e-15,
            envelope=True,
            absorber=None,
        ).translate()


def test_the_modified_noise_beside_the_coupling_is_refused_naming_the_opt_out():
    """A kept-apart draw through a coupled flow is not measured; the opt-out carries it.

    The input draw is part of the entrance state and rides the coupling
    (``tests/driver/test_coupled_input_noise.py``).
    """
    with pytest.raises(TranslationError, match="taper_coupling=False"):
        lossless(noise="modified", seed=1).translate()
    assert (
        lossless(noise="modified", seed=1, taper_coupling=False).translate() is not None
    )
    assert lossless(noise="input", seed=1).translate() is not None


def test_a_supplied_set_couples_through_from_waveguide_and_a_provider_refuses_it():
    """``from_waveguide`` takes the default for a supplied capillary set, or opts out.

    Beside a provider the opt-out is refused: a moving general guide carries its own
    connection.
    """
    modes = [MarcatiliMode(taper(), "He", 1.0, m=m, loss=False) for m in (1, 2)]
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "n_save": 2,
        "backend": "numpy_ref",
        "plasma": False,
        "raman": False,
    }
    coupled = Simulation.from_waveguide(
        modes, LENGTH, "He", 1.0, **{"noise": None, **arguments}
    )
    assert coupled.translate().linear.coupled_flow is not None
    adiabatic = Simulation.from_waveguide(
        modes,
        LENGTH,
        "He",
        1.0,
        taper_coupling=False,
        **{"noise": None, **arguments},
    )
    assert adiabatic.translate().linear.coupled_flow is None
    with pytest.raises(DriverError, match="taper_coupling=False"):
        Simulation.from_waveguide(
            (),
            LENGTH,
            provider=split_guide(general_grid()),
            wavelength=835e-9,
            taper_coupling=False,
            noise=None,
        )


def test_with_input_and_with_mixture_keep_the_declared_route():
    """A copy keeps the coupling, the opt-out and a wall as declared."""
    for simulation, flow in (
        (lossless(), True),
        (lossless(taper_coupling=False), False),
        (varying(wall=CoherentWall()), True),
    ):
        copies = (
            simulation.with_input(simulation._field0.copy()),
            simulation.with_mixture(
                cast("Mixture", simulation.configuration["medium"])
            ),
        )
        for copy in copies:
            assert copy.taper_coupling == simulation.taper_coupling
            assert copy.wall == simulation.wall
            assert (copy.translate().linear.coupled_flow is not None) is flow


# ---------------------------------------------------------------- the physics


def carrier_bin(grid):
    """Return the scored bin nearest 800 nm."""
    target = 2 * math.pi * C / 800e-9
    scored = np.flatnonzero(grid.inband_mask)
    return int(scored[np.argmin(abs(grid.omega[scored] - target))])


def slow_taper(*, taper_coupling=True, steps=STEPS):
    """Return the slow lossless taper of the adiabatic-limit test, linear, no fill."""
    length = 1e-3
    grid = FieldGrid.from_limits(length, 800e-9, (500e-9, 1.6e-6), 200e-15)
    return Simulation.from_capillary(
        Linear(30e-6, 29e-6, length),
        length,
        "He",
        0.0,
        wavelength=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0e3),
        fwhm=30e-15,
        energy=1e-9,
        modes=3,
        n_save=2,
        backend="numpy_ref",
        loss=False,
        cladding=1.45,
        plasma=False,
        raman=False,
        schedule=FixedSchedule(np.full(steps, length / steps)),
        dz_max=length / steps,
        dz_init=length / steps,
        taper_coupling=taper_coupling,
        noise=None,
    )


def first_order(simulation, index):
    """Return the first-order conversion estimate of a lossless taper at one bin.

    In the true-power amplitude ``b`` of a lossless guide, ``b' = (i B - C) b`` with
    ``B`` the channels' propagation constants and ``C`` the antisymmetric connection;
    in the interaction picture the coupling is ``C`` dressed by unit phases, so
    Dyson's series from a launch in ``HE_11`` gives, per channel ``m``,

    * the first-order amplitude ``c_m = -int C_m1 exp(-i phi_m1) dz`` with
      ``phi_m1 = int (beta_m - beta_1)``;
    * its integration-by-parts bound (the first-order form of Love's adiabaticity
      criterion ``|C_m1| << |beta_m - beta_1|``):
      ``|c_m| <= |C/dbeta|(0) + |C/dbeta|(L) + int |d(C/dbeta)/dz| dz``;
    * and everything beyond first order bounded by ``exp(I) - 1 - I`` with
      ``I = int ||C||_2 dz`` -- which, since ``C_11 = 0``, bounds ``HE_11``'s whole
      departure from the adiabatic route.

    Returns
    -------
    tuple
        ``(c, bound, remainder)``: the first-order amplitudes and their bounds per
        channel (entry 0 unused), and the Dyson remainder.
    """
    modes = simulation.modes
    omega = np.array([float(simulation.grid.omega[index])])
    length = float(simulation.settings.save_planes[-1])
    z = np.linspace(0.0, length, 4001)
    beta = (
        np.array(
            [
                [float(np.real(mode.neff(omega, point)[0])) for point in z]
                for mode in modes
            ]
        )
        * omega[0]
        / C
    )
    links = np.array([taper_connection(modes, float(point)) for point in z])
    count = len(modes)
    first = np.zeros(count, dtype=complex)
    bound = np.zeros(count)
    for m in range(1, count):
        phase = cumulative_trapezoid(beta[m] - beta[0], z, initial=0)
        first[m] = -trapezoid(links[:, m, 0] * np.exp(-1j * phase), z)
        ratio = links[:, m, 0] / (beta[m] - beta[0])
        bound[m] = abs(ratio[0]) + abs(ratio[-1]) + float(np.sum(abs(np.diff(ratio))))
    norms = np.linalg.norm(links, ord=2, axis=(1, 2))
    total = trapezoid(norms, z)
    return first, bound, math.expm1(total) - total


@pytest.mark.medium
def test_a_slow_lossless_taper_converts_within_its_first_order_estimate():
    """The adiabatic limit: a slow taper converts as first-order theory says, little.

    Bounds, stated before the run (:func:`first_order`). A 30 to 29 um linear taper
    over a millimetre, three channels, ten fixed steps, no fill (a linear run), read
    at the scored bin nearest 800 nm, amplitudes relative to the launch:

    * every converted channel's amplitude is within the Dyson remainder ``R`` of the
      first-order amplitude, and below its integration-by-parts bound plus ``R``;
    * ``HE_11`` departs from the **opt-out's** ``HE_11`` -- the adiabatic route --
      by at most ``R``, phase included: the coupled flow's diagonal is the adiabatic
      route's propagation constants, not an approximation of them;
    * ``delta``, added to each, covers what the estimate does not model: the weak
      collar (below 1e-6 over the millimetre), the Chebyshev representations of the
      two routes' tables (1e-10) and the explicit residual's truncation over ten
      steps of ``||C|| h`` about 4e-3 (``(||C|| h)^6 / 720`` a step), so 2e-6.
    """
    coupled = cast("MemoryOutput", slow_taper().run(progress="none"))
    adiabatic = cast(
        "MemoryOutput", slow_taper(taper_coupling=False).run(progress="none")
    )
    simulation = slow_taper()
    index = carrier_bin(simulation.grid)
    first, bound, remainder = first_order(simulation, index)
    delta = 2e-6
    entering, leaving = coupled.planes()[0].field, coupled.planes()[-1].field
    alone = adiabatic.planes()[-1].field
    assert entering is not None and leaving is not None and alone is not None
    launch = abs(entering[0, 0, index])
    final = leaving[:, 0, index] / launch
    straight = alone[:, 0, index] / launch
    assert 0.0 < remainder < 1e-2
    for m in (1, 2):
        assert abs(abs(final[m]) - abs(first[m])) <= remainder + delta
        assert abs(final[m]) <= bound[m] + remainder + delta
    # Not vacuous: the taper converts measurably, and the estimate resolves it.
    assert abs(final[1]) ** 2 > 1e-4
    assert remainder + delta < 0.2 * abs(first[1])
    assert abs(final[0] - straight[0]) <= remainder + delta
    assert np.all(straight[1:] == 0.0)


@pytest.mark.medium
def test_the_lossless_coupled_flow_conserves_power_to_its_step_error():
    """Flux is conserved to the discretisation, converging at the method's order.

    The connection is antisymmetric in the true-power amplitude and a lossless
    channel's diagonal is a unit phase, so the exact flow conserves
    ``sum_m |b_m|^2``; what departs from it is the explicit residual's Runge-Kutta
    truncation, fifth order locally. **Stated first**: on the non-adiabatic case of
    the taper note at a quarter of its radius (35 % out of ``HE_11``, four channels,
    12, 24 and 48 fixed steps), the power departure falls by at least ``2^3.5`` at
    each halving (fourth order globally, less half an order for the pre-asymptotic
    coarse level), and is below ``1e-6`` at 48 steps. The order is the Runge-Kutta
    march's, so the march is named; the next test runs the default.
    """
    study = _taper_study()
    base = study.CASES["scaled0.25-1cm-kinked-lossless"]
    departures = [
        abs(study.production(base, 4, per, integrator=RungeKutta())["power"] - 1.0)
        for per in (3, 6, 12)
    ]
    assert departures[-1] < 1e-6
    for coarse, fine in itertools.pairwise(departures):
        assert fine > 0.0
        assert math.log2(coarse / fine) >= 3.5


@pytest.mark.medium
def test_the_lossless_coupled_flow_conserves_power_to_the_default_s_tolerance():
    """The same taper under the default integrator, its steps the controller's.

    The Kerr source carries the explicit residual's compensation into the Adams
    samples (ADR-042, ADR-044), so the controller scores the residual's error and
    the power departure follows the tolerance: measured 7.8e-10 at ``rtol=1e-8`` and
    6.1e-12 at ``1e-10``. **Stated first**: below a tenth of the tolerance at each,
    and falling with it.
    """
    study = _taper_study()
    base = study.CASES["scaled0.25-1cm-kinked-lossless"]
    departures = []
    for rtol in (1e-8, 1e-10):
        result = study.production(
            base, 4, 3, schedule=None, rtol=rtol, integrator=Multistep()
        )
        departures.append(abs(result["power"] - 1.0))
        assert departures[-1] < 0.1 * rtol, (rtol, departures[-1])
    assert departures[1] < departures[0]


# ------------------------------------------------------ backends and the record


@pytest.mark.medium
@pytest.mark.skipif("native" not in available(), reason="native is not installed")
def test_auto_runs_a_lossless_multimode_taper_on_native():
    """The default call runs the taper coupling on ``native``, as ``numpy_ref`` does.

    The coupled flow's carrier is the lowest channel's parameter expansion, which the
    native backend executes (the opt-out, where its per-channel expansion does not
    resolve, takes the z table, which the native CPU build runs too:
    ``test_native_z_table.py``). The final plane is ``numpy_ref``'s to the flow parity
    order (a millionth of the launch is generous for ten steps of a nanojoule).
    """
    reference = cast("MemoryOutput", lossless().run(progress="none"))
    automatic = cast("MemoryOutput", lossless(backend="auto").run(progress="none"))
    assert automatic.metadata is not None
    assert automatic.metadata.backend is not None
    assert automatic.metadata.backend.name == "native"
    expected = reference.planes()[-1].field
    start, actual = reference.planes()[0].field, automatic.planes()[-1].field
    assert expected is not None and start is not None and actual is not None
    scale = float(np.max(np.abs(start)))
    departure = float(np.max(np.abs(actual - expected)))
    assert departure <= 1e-12 * scale


@pytest.mark.medium
def test_a_coupled_file_is_not_resumed_by_the_adiabatic_route(tmp_path):
    """The record keys the coupling: a resume across the opt-out is refused by name."""
    path = tmp_path / "coupled.h5"
    planes = [0.0, LENGTH / 2, LENGTH]
    schedule = FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS, save_planes=planes)
    run_until(lossless(n_save=3, schedule=schedule), path, after=1)
    with pytest.raises(OutputError, match="modal"):
        lossless(n_save=3, schedule=schedule, taper_coupling=False).run(
            path, resume=True, progress="none"
        )


def test_a_lossy_block_over_a_long_step_is_refused_naming_the_opt_out():
    """The flow's conditioning guard, re-worded for a route with no sections or limit.

    A lossy block spreads by the channels' attenuation difference over a step, and at
    ``dz_max = 0.1`` m on this 40 um taper that exceeds the 1e4 the flow admits;
    ``1e-2`` m is admitted.
    """
    with pytest.raises(TranslationError, match="taper_coupling=False"):
        varying(dz_max=0.1, schedule=None).translate()
    assert varying(dz_max=1e-2, schedule=None).translate().linear.coupled_flow


def test_a_second_field_component_beside_the_coupling_is_refused():
    """The flow contracts the channel axis alone: two state components are refused."""
    simulation = lossless()
    field = np.concatenate([simulation._field0, simulation._field0], axis=1)
    with pytest.raises(TranslationError, match="n_pol = 2"):
        assemble(
            simulation.grid,
            simulation.modes,
            list(simulation.responses),
            field,
            controller=cast("PIController", simulation.settings.controller),
            save_planes=simulation.settings.save_planes,
            dz_max=simulation.settings.dz_max,
            dz_init=simulation.settings.dz_init,
            quadrature=simulation.settings.quadrature,
            components=("x", "y"),
        )
