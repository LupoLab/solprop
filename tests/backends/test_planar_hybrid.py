# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform on the oracle: slab modes in x times an FFT in y.

Every run is at most ten fixed steps of an argon-filled gap (2a = 127 um, Akturk et al.
2009) on ``numpy_ref``, the oracle the compiled backends are measured against. Three
kinds of evidence:

- a field **uniform in y** (one ``k_y``, the ``k_y = 0`` column) is the slab mode set
  propagated through ``prop_waveguide``'s multimode route, for every response
  family and both absorber schemes, to the rounding of the operator's square root;
- a **linear** Gaussian beam in y diffracts as the (1+1)-D Gaussian beam and decays
  and disperses as each slab mode does, against the closed forms;
- the **Kerr** phase at the beam centre is the slab mode's closed-form B integral,
  locally in y.
"""

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from backends.test_numba_band_bounds import rate_error_bound
from bounds import fft_bound, gamma, step_evaluations
from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.spatial import (
    slab_columns,
    slab_projection,
    slab_rows,
    slab_synthesis,
)
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.fourier import inverse
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import (
    ModeCollection,
    PlanarGrid,
    SlabMode,
    gaussian_coupling,
)
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3, n2
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.tools import pulse_peak_power

pytestmark = [
    # The FFT padding reaches beyond the argon fit; the notice is not the subject.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    # The inert collar (the slab modes' pattern) cannot stop TE5 walking off over a
    # metre; these runs are ten steps of 10 cm, and the walk is not the subject.
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

EPS = np.finfo(float).eps
LENGTH = 0.1
HALF_GAP = 63.5e-6
ENVELOPE = EnvelopeGrid.from_limits(
    LENGTH, 800e-9, (650e-9, 1000e-9), 1.2e-12, **LEGACY_GRID
)

FAMILIES = {
    # name: (grid, gas, energy J/m, fwhm s, plasma, raman, n_r, n_y, steps, scheme)
    "kerr": (ENVELOPE, "Ar", 0.5, 100e-15, False, False, None, 4, 10, "nonlinear"),
    "kerr-split": (
        ENVELOPE,
        "Ar",
        0.5,
        100e-15,
        False,
        False,
        None,
        4,
        10,
        "split_step",
    ),
    "raman": (ENVELOPE, "N2", 0.5, 100e-15, False, True, None, 4, 10, "nonlinear"),
    # Forty steps: optmat's argon rate (SEP 3, Phase 4a) is 2.5 to 9 times the former
    # PPT's over this launch's fields, and fixed steps must keep the plasma's loss
    # inside the explicit history's stability. Twenty sufficed for the Runge-Kutta
    # march; the default Adams method, whose interval at order five is forty times
    # smaller, diverged there (to 4e192, finite: a replay cannot reject) and is stable
    # from forty, eighty and a hundred and sixty agreeing to 1e-4.
    "field-plasma": (
        FieldGrid.from_limits(
            LENGTH, 800e-9, (400e-9, 2000e-9), 600e-15, **LEGACY_GRID
        ),
        "Ar",
        2.0,
        50e-15,
        True,
        False,
        None,
        4,
        80,
        "nonlinear",
    ),
    # The envelope plasma's cycle moments dominate its cost, so the rule and the y
    # axis are the smallest that still exercise every path.
    "envelope-plasma": (
        ENVELOPE,
        "Ar",
        0.5,
        50e-15,
        True,
        False,
        12,
        2,
        2,
        "nonlinear",
    ),
}
"""Each family's launch, rule, y axis and schedule."""

ORDERS = {
    # The envelope plasma's smaller rule resolves TE1's quartic products alone.
    "envelope-plasma": (1,),
}
"""The slab orders of a family that does not carry TE1, TE3 and TE5."""


def modal_and_planar(family):
    """Translate one launch through the modal route and uniform in y on the planar."""
    grid, gas, energy, fwhm, plasma, raman, n_r, n_y, steps, scheme = FAMILIES[family]
    orders = ORDERS.get(family, (1, 3, 5))
    modes = ModeCollection.slab(HALF_GAP, orders, gas, 1.0).modes
    simulation = Simulation.from_waveguide(
        list(modes),
        LENGTH,
        gas,
        1.0,
        lambda0=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0e3),
        fwhm=fwhm,
        energy=energy,
        envelope=isinstance(grid, EnvelopeGrid),
        plasma=plasma,
        raman=raman,
        n_save=2,
        n_r=n_r,
        dz_max=LENGTH / steps,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
        backend="numpy_ref",
    )
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber_scheme=scheme
    )
    pulse = simulation._field0[0]
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    modal = simulation.with_input(
        np.stack([math.sqrt(share) * pulse for share in shares])
    ).translate()
    space = PlanarGrid(modes, 1e-3, n_y, window_factor=0.0, n_x=n_r)
    # A field uniform in y is its k_y = 0 column alone, n_y times the row value.
    field0 = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    field0[: space.n_p] = modal.initial.field0[0] * n_y
    settings = dataclasses.replace(
        simulation.settings, quadrature=None, components=None, stats=None
    )
    planar = Simulation.from_planar(
        grid, space, list(simulation.responses), field0, settings=settings
    ).translate()
    return grid, modal, planar, space, steps


def run(problem, grid, names=()):
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        field = backend.observe().require_field().copy()
        values = backend.stats(StatsSpec(names=names)) if names else {}
    finally:
        backend.close()
    return field, values


def field_peak_rate_allowance(problem, reference, field_bound, n_y):
    """Propagate the spectral field bound through synthesis and the monotone rate.

    The scalar fixture's real inverse DFT has active-bin infinity norm
    ``2 n_active / n_to``. Multiply by the synthesis row sums to bound the
    physical nodal error. Add both reconstructions' temporal/transverse FFT and
    dot-product rounding against their absolute operands. The nodal peak is
    Lipschitz in this absolute error; monotonicity then encloses its rate by
    evaluating the two field endpoints. No planar result enters the allowance.
    """
    grids, transform = problem.grids, problem.transform
    synthesis = abs(transform.synthesis)
    assert problem.shape[2] == transform.n_pol == 1
    assert not grids.active_mask[0] and not grids.active_mask[-1]
    spectral_error = field_bound * float(np.max(abs(reference)))
    error = (
        spectral_error
        * (2 * np.count_nonzero(grids.active_mask) / grids.n_to)
        * float(np.max(np.sum(synthesis, axis=1)))
    )
    absolute_modes = 2 * np.sum(abs(reference[0, :, 0]), axis=-1) / grids.n_to
    error += (
        2
        * (fft_bound(grids.n_to) + fft_bound(n_y) + gamma(problem.shape[1]))
        * float(np.max(synthesis @ absolute_modes))
    )
    modal_time = passes.to_time(
        reference[0], grids.crop_index, grids.n_to, kind="field"
    )
    peak = float(
        np.max(abs(passes.synthesise(modal_time, transform.synthesis, 1.0, 1)))
    )
    table = problem.responses.plasma_channels[0].rate
    # Check each quadratic log-cubic derivative at its endpoints and any interior
    # vertex: monotone knot values alone would not exclude spline overshoot.
    coefficients, width = table.coefficients, table.d_log_field
    quadratic, linear, constant = (
        3 * coefficients[0],
        2 * coefficients[1],
        coefficients[2],
    )
    minimum = np.minimum(constant, (quadratic * width + linear) * width + constant)
    vertex = np.zeros_like(quadratic)
    np.divide(-linear, 2 * quadratic, out=vertex, where=quadratic != 0)
    inside = (quadratic != 0) & (vertex > 0) & (vertex < width)
    minimum[inside] = np.minimum(
        minimum[inside],
        (quadratic[inside] * vertex[inside] + linear[inside]) * vertex[inside]
        + constant[inside],
    )
    assert np.all(minimum > 0)
    assert table.field_min < peak - error < peak + error < table.field_max
    endpoints = np.asarray([peak - error, peak, peak + error])
    rates = table(endpoints)
    rounding = rate_error_bound(endpoints[None, None, :], table, rates[None, :])
    return float(max(rates[2] - rates[1], rates[1] - rates[0]) + 2 * np.max(rounding))


@pytest.mark.medium
@pytest.mark.parametrize("family", list(FAMILIES))
def test_a_field_uniform_in_y_is_the_slab_mode_set(family):
    """The k_y = 0 column is the modal route's state to the operator's rounding.

    The two routes evaluate the same equations for a uniform field: the same slab
    rule and synthesis (the same translated table, asserted exactly), the same
    responses node by node, the same
    per-mode prefactor ``-i omega / (2 eps0 c Re n_eff)``, and a y transform of
    constant rows. What differs is the operator: the modal route forms
    ``omega Re n_eff / c`` and the hybrid ``Re sqrt(k^2 n_co^2 - kappa^2)``, a few
    rounding units of beta apart, so the phases part by at most ``8 eps beta_max L``
    over the run (a factor two over four units); the stages' own rounding adds
    ``64 eps`` per stage evaluation. Measured 1.2e-10 to 2.1e-10.
    """
    grid, modal, planar, space, steps = modal_and_planar(family)
    assert_array_equal(planar.transform.synthesis, modal.transform.synthesis)
    names = (
        (
            "electron_density",
            "peak_ionisation_rate",
            "species_electron_density",
            "species_peak_ionisation_rate",
        )
        if modal.flags.has_plasma
        else ()
    )
    reference, modal_stats = run(modal, grid, names)
    active = grid.omega[grid.active_mask]
    beta = (
        float(np.max(np.abs(active)))
        / C
        * float(np.max(np.real(space.modes[0].neff(active))))
    )
    bound = 8 * EPS * beta * LENGTH + steps * step_evaluations(modal) * 64 * EPS
    leak = steps * step_evaluations(modal) * 64 * EPS * space.n_y
    rate_allowance = (
        # The inverse y transform adds the permitted nonzero-k_y columns with
        # weight 1 / n_y; include their existing leakage allowance as well.
        field_peak_rate_allowance(
            modal,
            reference,
            bound + (space.n_y - 1) * leak / space.n_y,
            space.n_y,
        )
        if family == "field-plasma"
        else None
    )
    actual, planar_stats = run(planar, grid, names)
    scale = float(np.max(np.abs(reference)))
    uniform = actual[0, : space.n_p] / space.n_y
    assert np.max(np.abs(uniform - reference[0])) <= bound * scale
    # Nothing leaves k_y = 0: the rows stay equal to the rounding of the y transform
    # and of the responses node by node, 64 eps n_y per stage evaluation. Measured
    # exactly zero: equal rows give equal sources, and a DFT of length 2 or 4 has
    # exact twiddles; the bound is for a length whose twiddles round.
    assert np.max(np.abs(actual[0, space.n_p :])) <= leak * scale
    # The pulse did change, so the comparison is not of two copies of the input.
    assert np.max(np.abs(reference[0] - modal.initial.field0[0])) > 0.1 * scale
    for name in names:
        if rate_allowance is not None and name.endswith("peak_ionisation_rate"):
            # Approved SEP 27 correction: the peak rate inherits the accepted
            # field error through synthesis and the rate, rather than an old
            # empirical tolerance. Keep the density and envelope checks below.
            assert rate_allowance > 0
            assert_allclose(
                planar_stats[name], modal_stats[name], rtol=0, atol=rate_allowance
            )
        else:
            assert_allclose(
                planar_stats[name],
                modal_stats[name],
                rtol=1e-7 if name.endswith("peak_ionisation_rate") else 1e-8,
            )
        assert float(np.ravel(planar_stats[name])[0]) > 0


def gaussian_beam(space, waist, modes_amplitudes, grid, energy=1e-3):
    """Build a Gaussian in y carrying given slab amplitudes, in the internal form."""
    profile = np.exp(-((space.y / waist) ** 2))
    temporal = GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=energy).build(
        grid
    )[0, 0]
    columns = np.fft.fft(profile)[:, None] * np.asarray(modes_amplitudes)[None, :]
    return columns.reshape(-1)[:, None, None] * temporal[None, None, :]


def closed_form_loss(mode, omega):
    """Return the first-order TE field attenuation and its truncation bound.

    ``alpha_p = (1/a)(p lambda / 4a)^2 Re q``, ``q = 1/sqrt(eps_cl - eps_co)``, to
    within ``3 (lambda |q| / (2 pi a) + (p lambda / 4a)^2)``.
    """
    wavelength = 2 * math.pi * C / omega
    glass = complex(mode.cladding_index.index(np.array([omega]))[0]) ** 2
    core = float(np.real(SpatialMedium("Ar", 1.0).index(np.array([omega]))[0])) ** 2
    contrast = 1 / np.sqrt(glass - core)
    ratio = (mode.order * wavelength / (4 * HALF_GAP)) ** 2
    truncation = 3 * (wavelength * abs(contrast) / (2 * math.pi * HALF_GAP) + ratio)
    return ratio / HALF_GAP * contrast.real, truncation


@pytest.mark.medium
def test_a_linear_beam_diffracts_in_y_and_decays_as_each_slab_mode():
    """The carrier bin against the (1+1)-D Gaussian beam times each mode's beta.

    TE1 and TE3 carry a Gaussian of 1/e field radius ``w0 = 100 um`` in y through
    ``2 z_R`` of argon, ``z_R = k w0^2 / 2``, ``k = Re beta_p(omega_0)``: the width
    ``w0 sqrt(1 + (z/z_R)^2)``, the amplitude ``(1 + (z/z_R)^2)^(-1/4)`` times the
    mode's own field decay ``exp(-Im beta_p z)``, and the on-axis phase
    ``(Re beta_p - beta_0) z - arctan(z/z_R)/2`` (the 1-D Gouy phase; a phase
    ``exp(-i phi)`` in the internal convention). The exact operator departs from the
    paraxial beam at ``O(1/(k w0)^2)``; the bound is the line free space's
    ``4/(k w0)^2``, 6.5e-6, on the amplitude and the width relative, and on the phase in
    radians. The decay itself is against the slab mode's first-order closed form
    ``alpha_p = (1/a)(p lambda / 4a)^2 Re q_TE`` within its truncation
    ``3 (lambda |q| / (2 pi a) + (p lambda / 4a)^2)``.
    """
    grid = ENVELOPE
    modes = [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3)]
    space = PlanarGrid(modes, 1.2e-3, 128, window_factor=0.0)
    omega0 = grid.omega[0]
    beta = np.array([complex(mode.neff(np.array([omega0]))[0]) for mode in modes])
    beta *= omega0 / C
    waist = 100e-6
    rayleigh = float(beta[0].real) * waist**2 / 2
    length = 2 * rayleigh
    field0 = gaussian_beam(space, waist, [1.0, 0.5], grid)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[length],
        dz_max=length / 10,
        dz_init=length / 10,
        schedule=FixedSchedule(np.full(10, length / 10)),
        backend="numpy_ref",
        absorber=bare_absorber(grid),
    )
    problem = Simulation.from_planar(
        grid, space, [], field0, settings=settings
    ).translate()
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        final = backend.observe().require_field()[0, :, 0, 0]
    finally:
        backend.close()
    start = problem.initial.field0[0, :, 0, 0]
    rows0 = np.fft.ifft(start.reshape(space.n_y, 2), axis=0)
    rows = np.fft.ifft(final.reshape(space.n_y, 2), axis=0)
    ratio = length / rayleigh
    k_w0 = float(beta[0].real) * waist
    bound = 4 / k_w0**2
    separable = problem.linear.separable
    assert separable is not None
    frame = separable.frame[0]
    centre = space.n_y // 2
    assert space.y[centre] == 0
    for channel, mode in enumerate(modes):
        k = float(beta[channel].real)
        local_ratio = length / (k * waist**2 / 2)
        profile = np.abs(rows[:, channel]) / np.abs(rows0[centre, channel])
        decay = math.exp(-float(beta[channel].imag) * length)
        # The amplitude on axis and the 1/e width, from the sampled profile's moment.
        amplitude = (1 + local_ratio**2) ** -0.25 * decay
        assert abs(profile[centre] / amplitude - 1) < bound
        # |E|^2 = exp(-2 y^2 / w^2) has <y^2> = w^2 / 4.
        weights = profile**2
        width = 2 * math.sqrt(float(np.sum(weights * space.y**2) / np.sum(weights)))
        assert abs(width / (waist * math.sqrt(1 + local_ratio**2)) - 1) < bound
        phase = float(np.angle(rows[centre, channel] / rows0[centre, channel]))
        expected = -((k - frame) * length - 0.5 * math.atan(local_ratio))
        assert abs(np.angle(np.exp(1j * (phase - expected)))) < bound
        # The mode's own decay against the first-order closed form.
        closed, truncation = closed_form_loss(mode, omega0)
        assert abs(float(beta[channel].imag) / closed - 1) < truncation
    assert ratio == pytest.approx(2.0)


@pytest.mark.medium
def test_the_kerr_phase_at_the_beam_centre_is_the_closed_form_b_integral():
    """B = n2 (omega/c) P'(0) L_eff / (4a/3) at y = 0, the centre of a wide beam.

    One TE1 channel carrying 0.5 J/m at the centre of a Gaussian of 1/e radius 5 mm
    in y, whose Rayleigh range (98 m) is far beyond the 10 cm run, so the Kerr term
    is the slab mode's SPM locally in y. The phase is taken against the same run without
    the Kerr term, which removes the dispersion and the 1-D Gouy phase in y. The
    closed form leaves out the index factor ``n0**2 / n_eff`` of the prefactor
    (+3.1e-4 in argon; the state is launched with ``|a|^2 = P'``, not the true-power
    ``|b|^2``) and the coupling of the Kerr lens with diffraction in y, of relative
    order ``L / z_R``: 6e-3 at a 2 mm radius, where the measured departure was
    9.4e-4; at 5 mm, 3.4e-4. The bound is the slab mode's own 2e-3.
    """
    grid = ENVELOPE
    mode = SlabMode(HALF_GAP, "Ar", 1.0)
    space = PlanarGrid([mode], 20e-3, 64, window_factor=0.0)
    field0 = gaussian_beam(space, 5e-3, [1.0], grid, energy=0.5)
    responses = Simulation.from_waveguide(
        [mode],
        LENGTH,
        "Ar",
        1.0,
        lambda0=800e-9,
        grid=grid,
        fwhm=100e-15,
        energy=0.5,
        envelope=True,
        plasma=False,
        raman=False,
    ).responses
    assert [type(one) for one in responses] == [KerrResponse]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[LENGTH],
        dz_max=LENGTH / 10,
        dz_init=LENGTH / 10,
        schedule=FixedSchedule(np.full(10, LENGTH / 10)),
        absorber=absorber(grid, boundary_length=1.0e3),
        backend="numpy_ref",
    )
    phases = []
    for nonlinear in (list(responses), []):
        problem = Simulation.from_planar(
            grid, space, nonlinear, field0, settings=settings
        ).translate()
        backend = NumpyRefBackend(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            final = backend.observe().require_field()[0, :, 0, :]
        finally:
            backend.close()
        centre = np.fft.ifft(final.reshape(space.n_y, -1), axis=0)[space.n_y // 2]
        envelope = inverse(centre, kind="envelope")
        phases.append(envelope)
    peak = int(np.argmax(np.abs(phases[1])))
    phase = -float(np.angle(phases[0][peak] / phases[1][peak]))
    omega = np.asarray([2 * np.pi * C / 800e-9])
    alpha = float(np.imag(mode.neff(omega))[0] * omega[0] / C)
    effective = (1 - math.exp(-2 * alpha * LENGTH)) / (2 * alpha)
    power = pulse_peak_power(0.5, 100e-15, "gauss")
    # The run's Kerr coefficient is optmat's at the carrier since SEP 3's Phase 4a,
    # 23 % above the registry's: the closed form takes the same one (B 0.358 -> 0.441).
    closed = (
        n2("Ar", pressure=1.0, wavelength=800e-9, selection=MaterialSelection())
        * omega[0]
        / C
        * power
        / mode.aeff()
        * effective
    )
    assert 0.4 < closed < 0.5
    assert_allclose(phase, closed, rtol=2e-3)


def test_the_planar_launch_carries_the_coupled_share_of_the_incident_energy():
    """PlanarPulse: the incident beam's energy, the modes' share of it, in J.

    The energy statistic at the entrance is the pulse's energy times the sum of the
    x shares (Arnold 2010's 98.8 % into TE1 at 0.735a), to the paraxial difference
    between each mode's and the incident beam's admittance.
    """
    grid = ENVELOPE
    modes = [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3, 5)]
    space = PlanarGrid(modes, 2e-3, 128)
    spec = StatsSpec(names=("energy", "beam_radius", "fwhm_r"))
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        500e-6,
        waist_x=0.735 * HALF_GAP,
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-3,
        schedule=FixedSchedule(np.full(1, 1e-3)),
        backend="numpy_ref",
        stats=spec,
        absorber=bare_absorber(grid),
    )
    simulation = Simulation.from_planar(grid, space, [], pulse, settings=settings)
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(spec)
    finally:
        backend.close()
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    assert shares[0] == pytest.approx(0.988, abs=5e-4)
    # The vacuum overlap shares carry the medium's measure to its paraxial
    # envelope: each mode's energy is weighed by its own Re n_eff, the incident
    # beam's by the k_x-averaged bulk admittance (SEP 34 section 4.3), which
    # differ by about (kappa_p / k)**2 / 2 ~ 5e-6 for this argon gap.
    assert_allclose(float(values["energy"][0]), 1e-3 * float(np.sum(shares)), rtol=1e-5)
    # The fluence integrated across the gap is Gaussian in y of 1/e^2 half-width w.
    moment, radius = np.ravel(values["beam_radius"])
    assert_allclose([moment, radius], [500e-6, 500e-6], rtol=2e-3)
    assert_allclose(
        float(np.ravel(values["fwhm_r"])[0]),
        500e-6 * math.sqrt(2 * math.log(2)),
        rtol=2e-3,
    )


@pytest.mark.medium
def test_a_modulation_in_y_grows_at_the_modulational_instability_rate():
    """A seeded modulation in y grows as the linearised (1+1)-D NLS says, in TE1.

    At the peak of a long pulse the TE1 amplitude obeys, locally in time,
    ``da/dz = i/(2k) d2a/dy2 + i gamma P' |a|^2 a / P'`` (the uniform slab's SPM,
    ``gamma P`` its phase rate). A modulation ``a = sqrt(P)(1 + delta cos(Omega y))``
    with real ``delta = u0`` at the entrance has, to first order in ``u0``,
    ``u'' = G^2 u``, ``v' = (2 gamma P - q) u`` with ``q = Omega^2 / (2k)`` and
    ``G = sqrt(q (2 gamma P - q))``, so its size grows by
    ``sqrt(cosh^2(G L) + ((2 gamma P - q)/G)^2 sinh^2(G L))``. ``Omega`` is the fastest
    growing ``sqrt(2 gamma P k)`` of the nominal ``gamma P = 6 /m``; the test measures
    ``gamma P`` itself from the background's Kerr phase, so it checks the growth, not a
    nonlinear coefficient. The mode is lossless, so the background keeps its power. What
    the closed form leaves out: the integration's own error, held to ``rtol=1e-8`` by
    the adaptive default integrator, the exact operator's ``(Omega/k)^2`` = 1.5e-6
    and the pulse's time structure over ``L/L_D`` ~ 1e-5 (1.5e-4 measured at
    ``G L = 2.5`` with a 1 ps pulse, 6e-5 with a 2 ps one, so the pulse's time
    structure dominates); the bound is 1e-3.
    """
    length = 0.5
    grid = EnvelopeGrid.from_limits(length, 800e-9, (760e-9, 845e-9), 4e-12)
    mode = SlabMode(HALF_GAP, "Ar", 1.0, loss=False)
    k = float(grid.omega0 * np.real(mode.neff(np.array([grid.omega0]))[0]) / C)
    omega = math.sqrt(2 * 6.0 * k)
    space = PlanarGrid([mode], 2 * (2 * math.pi / omega), 16, window_factor=0.0)
    assert space.k_y[4] == pytest.approx(omega, rel=1e-12)
    power = 6.0 / (1e-23 * k / mode.aeff())
    temporal = GaussianPulse(wavelength=800e-9, fwhm=1e-12, peak_power=power).build(
        grid
    )[0, 0]
    seed = 1e-4
    profile = 1 + seed * np.cos(space.k_y[4] * (space.y - space.y[0]))
    field0 = (np.fft.fft(profile)[:, None] * temporal[None, :])[:, None, :]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[length],
        dz_max=length / 10,
        dz_init=length * 1e-4,
        auto_init=True,
        backend="numpy_ref",
    )
    finals = []
    for responses in ([KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)], []):
        problem = Simulation.from_planar(
            grid, space, responses, field0, settings=settings
        ).translate()
        backend = NumpyRefBackend(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(length)
            finals.append(backend.observe().require_field()[0, :, 0, :].copy())
        finally:
            backend.close()
    kerr_run, linear_run = finals
    entrance = inverse(field0[0, 0], kind="envelope")
    peak = int(np.argmax(np.abs(entrance)))
    background = inverse(kerr_run[0], kind="envelope")[peak]
    phase_rate = (
        -float(np.angle(background / inverse(linear_run[0], kind="envelope")[peak]))
        / length
    )
    assert phase_rate == pytest.approx(6.0, rel=0.2)
    start = abs(inverse(field0[4, 0], kind="envelope")[peak] / entrance[peak])
    end = abs(inverse(kerr_run[4], kind="envelope")[peak] / background)
    q = float(space.k_y[4]) ** 2 / (2 * k)
    rate = math.sqrt(q * (2 * phase_rate - q))
    closed = math.sqrt(
        math.cosh(rate * length) ** 2
        + ((2 * phase_rate - q) / rate) ** 2 * math.sinh(rate * length) ** 2
    )
    assert rate * length > 2
    assert abs(end / start / closed - 1) < 1e-3


def test_the_slab_synthesis_and_projection_invert_on_the_columns():
    """projection(synthesis(A)) = pad_scale A on an orthonormal set, per y row too."""
    modes = [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3)]
    space = PlanarGrid(modes, 1e-3, 8)
    problem = Simulation.from_planar(
        ENVELOPE,
        space,
        [],
        np.zeros((space.n_columns, 1, ENVELOPE.n_omega), complex),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-12),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH,
            backend="numpy_ref",
        ),
    ).translate()
    columns = np.random.default_rng(49).standard_normal((space.n_columns, 1, 5)) + 0j
    physical = slab_synthesis(problem.transform, columns)
    assert physical.shape == (space.n_nodes, 1, 5)
    back = slab_projection(problem.transform, physical)
    # The Kronrod rule integrates the modes' products to 1e-12 (the slab Gram check).
    assert_allclose(back, ENVELOPE.pad_scale * columns, atol=1e-12)
    assert_allclose(
        slab_columns(problem.transform, slab_rows(problem.transform, columns)),
        columns,
        atol=1e-15,
    )
