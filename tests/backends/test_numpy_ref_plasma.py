# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plasma passes of the reference backend, against the response that defines them.

:meth:`solprop.nonlinear.responses.PlasmaResponse.evaluate` is the oracle
(``docs/dev/backends.md`` section 2), and the three analytic cases that pin *it* --
the constant field and rate with its closed form, the Drude limit of a pre-ionised gas,
and the energy identity -- are reproduced here through
:func:`~solprop.backends.numpy_ref.passes.ionisation_rate`,
:func:`~solprop.backends.numpy_ref.passes.plasma_scan` and
:func:`~solprop.backends.numpy_ref.passes.plasma_current` at a quadrature node. They are
imported from ``tests/plasma_cases.py`` rather than restated, so there is one statement
of each closed form.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from bounds import fft_bound, scan_bound
from plasma_cases import (
    CARRIER,
    CHARGE_MASS_RATIO,
    DENSITY,
    constant_field_reference,
    constant_rate,
    gaussian_pulse_and_derivative,
    power_rate,
    pulse,
    zero_rate,
)
from rate_oracles import adk_oracle
from solprop import fourier
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.constants import CHARGE_MASS_RATIO as COMPILED_CHARGE_MASS_RATIO
from solprop.constants import E_CHARGE, EPS0, M_E, C
from solprop.exceptions import (
    BackendError,
    MaterialRangeWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas, ionisation_potential
from solprop.profiles import Constant, PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import delta_phi, profile_table
from solprop.translate.problem import assemble
from solprop.translate.responses import (
    PlasmaCoefficients,
    ResponseStage,
    prefactor_at,
)

FIELD_RANGE = (1e9, 2e11)  # V/m, the span the compiled rate table covers

# The passes reproduce the oracle's arithmetic -- the same three trapezoid integrals,
# the same guarded division -- but need not keep its order (SEP 25): they agree to the
# complete plasma chain's rounding, `scan_bound(n)` of the largest polarisation, and
# the ionised fraction, a running sum of positive terms, to 4 n eps. A formula that
# has drifted moves either by orders of magnitude more.


def chain_close(got, want):
    """Assert agreement to the plasma chain's cap, absolute in the largest value."""
    samples = np.shape(want)[-1]
    scale = float(np.max(np.abs(want)))
    assert scale > 0
    assert np.max(np.abs(got - want)) <= scan_bound(samples) * scale


def coefficients_for(gas, preionisation=0.0, *, span=1.0):
    """Build the compiled group from a gas and a pre-ionised fraction.

    ``rate`` is a real table because the field is typed as one, but the scan and the
    current never read it: they take the evaluated rate as an argument, which is what
    lets the analytic cases drive them with a rate no table could represent.
    """
    return PlasmaCoefficients(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        ionisation_potential=ionisation_potential(gas),
        preionisation=profile_table(Constant(preionisation), 0.0, span),
        rate=build_rate_table(adk_oracle(gas), FIELD_RANGE),
    )


def through_the_passes(field, rate_values, coefficients, density, dt, preionisation):
    """Return the plasma polarisation of one node, through the three passes."""
    column = np.asarray(field)[np.newaxis]
    fraction, free_current = passes.plasma_scan(
        column,
        np.asarray(rate_values)[np.newaxis],
        coefficients,
        preionisation=preionisation,
        dt=dt,
    )
    polarisation = passes.plasma_current(
        column,
        np.asarray(rate_values)[np.newaxis],
        fraction,
        free_current,
        coefficients,
        density=density,
        dt=dt,
    )
    return fraction[0], polarisation[0]


# ------------------------------------------------------- the constant-field closed form


@pytest.mark.parametrize("preionisation", [0.0, 0.3])
def test_the_passes_reproduce_the_constant_field_closed_form(preionisation):
    """Every integral is elementary here, so this is an oracle and not a second copy."""
    amplitude, rate, duration = 1e9, 1e14, 1e-14
    t = np.linspace(0.0, duration, 2**12 + 1)
    dt = float(t[1] - t[0])
    field = np.full((1, t.size), amplitude)
    values = constant_rate(rate)(np.abs(field[0]))
    fraction, polarisation = through_the_passes(
        field, values, coefficients_for(Gas.Ar), 1.0, dt, preionisation
    )
    expected_fraction, expected = constant_field_reference(
        t, amplitude, rate, ionisation_potential(Gas.Ar), preionisation
    )
    # The closed form is exact and the passes integrate; the gap is the trapezoid's own
    # second-order error, measured 9.4e-8 at 1025 samples and falling by four at every
    # doubling, so 2e-7 at 4097 samples has a wide margin.
    worst = np.max(np.abs(polarisation[0] - expected)) / np.max(np.abs(expected))
    assert 0.0 < worst < 2e-7
    # The fraction is a closed form too, and the trapezoid accumulates W0 dt 4097 times
    # where it multiplies once: Higham's n * eps, about 1e-12.
    assert np.allclose(fraction, expected_fraction, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("preionisation", [0.0, 0.3])
def test_the_passes_agree_with_the_response_bit_for_bit(preionisation):
    """The passes are the oracle's own arithmetic, to the plasma chain's rounding."""
    _, dt, field = pulse()
    column = field[np.newaxis]
    # The trapezoid, on a hand-made axis no grid designs taps for; the band rule's
    # passes and response are compared in `tests/nonlinear/test_plasma_band.py`.
    response = PlasmaResponse(
        Gas.Ar, power_rate(), preionisation=preionisation, integration="trapezoid"
    )
    expected = response.evaluate(column, DENSITY, dt=dt, kind="field")
    values = power_rate()(np.abs(field))
    fraction, polarisation = through_the_passes(
        column, values, coefficients_for(Gas.Ar), DENSITY, dt, preionisation
    )
    chain_close(polarisation, expected)
    np.testing.assert_allclose(
        fraction,
        response.ionised_fraction(column, dt=dt, kind="field"),
        rtol=4 * field.shape[-1] * np.finfo(float).eps,
        atol=0,
    )


# ------------------------------------------------------------------- the Drude limit


def test_a_pre_ionised_plasma_has_the_drude_susceptibility_through_the_passes():
    """The one case that pins ``e^2/(eps0 m_e)``, and it needs no ionisation model."""
    fraction, amplitude = 0.3, 1e9
    t = np.linspace(0.0, 40e-15, 200001)
    dt = float(t[1] - t[0])
    # Cosine, so the double integral has no secular ramp, only a constant offset.
    field = (amplitude * np.cos(CARRIER * t))[np.newaxis]
    values = zero_rate(field[0])
    _, polarisation = through_the_passes(
        field, values, coefficients_for(Gas.Ar), DENSITY, dt, fraction
    )
    offset = DENSITY * fraction * CHARGE_MASS_RATIO * amplitude / CARRIER**2
    expected = offset * (1.0 - np.cos(CARRIER * t))
    # The trapezoid error on the double integral: measured 3.7e-8 at 200001 samples.
    worst = np.max(np.abs(polarisation[0] - expected)) / np.max(np.abs(expected))
    assert 0.0 < worst < 1e-7
    plasma_frequency_squared = DENSITY * fraction * E_CHARGE**2 / (EPS0 * M_E)
    susceptibility = -np.max(np.abs(polarisation[0] - offset)) / (EPS0 * amplitude)
    assert susceptibility == pytest.approx(
        -plasma_frequency_squared / CARRIER**2, rel=1e-14
    )
    # The sign that lowers the refractive index.
    assert (polarisation[0] - offset)[0] * field[0, 0] < 0.0


# ------------------------------------------------------------------ energy bookkeeping


def energy_deviation(n_samples):
    """Relative deviation of the ionisation-loss energy from Ip times the electrons."""
    _, dt, field, derivative = gaussian_pulse_and_derivative(n_samples)
    column = field[np.newaxis]
    values = power_rate()(np.abs(field))

    def work_per_atom(gas):
        _, polarisation = through_the_passes(
            column, values, coefficients_for(gas), DENSITY, dt, 0.0
        )
        return -float(np.trapezoid(derivative * polarisation[0], dx=dt)) / DENSITY

    # One rate shared by two gases, so the fraction and hence the free-electron work are
    # identical and cancel in the difference; what is left is the loss term alone.
    difference = work_per_atom(Gas.He) - work_per_atom(Gas.Ar)
    fraction, _ = through_the_passes(
        column, values, coefficients_for(Gas.Ar), DENSITY, dt, 0.0
    )
    electrons = float(fraction[-1] - fraction[0])
    potentials = ionisation_potential(Gas.He) - ionisation_potential(Gas.Ar)
    return abs(difference / (potentials * electrons) - 1.0)


def test_the_field_pays_one_ionisation_potential_for_every_electron():
    """``E . J_loss = I_p df/dt``, which is what fixes the form of the loss current."""
    # The work is taken by parts with the same trapezoid on both sides, so the identity
    # converges at second order: measured 9.9e-5 at 8193 samples.
    assert 0.0 < energy_deviation(2**13 + 1) < 2e-4


def test_the_energy_identity_converges_at_second_order():
    coarse, middle, fine = (energy_deviation(2**k + 1) for k in (13, 14, 15))
    assert 3.6 < coarse / middle < 4.4
    assert 3.6 < middle / fine < 4.4


# ------------------------------------------------------------------ the rate lookup


def test_the_rate_pass_is_the_table_on_the_magnitude_of_the_field():
    table = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    _, _, field = pulse()
    scalar = passes.ionisation_rate(field[np.newaxis, np.newaxis], table)
    # Structural: the pass is the table, called on the magnitude, so exactly it.
    assert np.array_equal(scalar[0], table(np.abs(field)))
    # Two components go through hypot, not a sum of squares: the tunnelling exponent
    # turns one unit in the last place of |E| into 1e-13 of the rate.
    angle = 0.7
    vector = np.stack([math.cos(angle) * field, math.sin(angle) * field])
    rotated = passes.ionisation_rate(vector[np.newaxis], table)
    assert np.array_equal(
        rotated[0], table(np.hypot(*(vector[index] for index in (0, 1))))
    )


def test_the_compiled_constant_is_the_one_the_analytic_cases_use():
    assert COMPILED_CHARGE_MASS_RATIO == CHARGE_MASS_RATIO


# --------------------------------------------------------- one step of a real problem


@pytest.fixture(scope="module")
def field_grid():
    """The gradient capillary's grid, from ``tests/backends/conftest.py``."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


@pytest.fixture(scope="module")
def ionising_capillary(field_grid):
    """The gradient capillary, filled with argon and ionising.

    A two-point pressure gradient, so the operator varies along z, the prefactor is a
    table and the fill's stage boundary at 0.2 m is a hard kink. The rate is a real PPT
    table -- optmat's named PPT form for argon (``"ppt_ilkov"``), which replaced
    solprop's own ``PPT`` in SEP 3, Phase 4b -- tabulated once over the range the pulse
    reaches (up to the form's maximum field, where the table holds it).
    """
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    kerr = KerrResponse(chi3=2.2e-52, thg=True)
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), FIELD_RANGE)

    def build(preionisation=0.0):
        plasma = PlasmaResponse(Gas.Ar, table, preionisation=preionisation)
        start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(
            field_grid
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaterialRangeWarning)
            problem = assemble(
                field_grid,
                mode,
                [kerr, plasma],
                start,
                controller=PIController(rtol=1e-8, atol=1e-14),
                save_planes=[0.1, 0.2],
                dz_max=0.01,
                dz_init=1e-4,
            )
        return problem, kerr, plasma

    return build


def reference_nonlinear(problem, kerr, plasma, state, z):
    """Compose the two responses in plain NumPy, the way the scaffold composes Kerr.

    The oracle for one stage: pad, inverse transform, convert to volts per metre with
    the transform's own synthesis, form the physical population and current,
    and evaluate Kerr. Project the collared Kerr and both current products,
    transform, divide the current spectra by frequency, crop and normalise.
    The public plasma evaluate() oracle still returns ungated polarisation.
    """
    grids, transform = problem.grids, problem.transform
    responses = problem.responses
    fine = np.zeros(
        (transform.n_modes, fourier.n_omega_for(grids.n_to, "field")), dtype=complex
    )
    fine[..., grids.crop_index] = state[0, :, 0, :]
    modal = fourier.inverse(fine, kind="field")
    nodes = transform.synthesis @ modal
    plane_plasma = dataclasses.replace(
        plasma, preionisation=float(responses.plasma_coefficients.preionisation(z))
    )
    density = float(responses.density(z))
    rule = responses.plasma_integral
    # Independent assembly from the physical response's rate and population. The
    # physical evaluate() oracle still integrates J, but propagation holds J (SEP 27).
    current = np.empty_like(nodes)
    for index, field in enumerate(nodes):
        values = field[np.newaxis, :]
        fraction = plane_plasma.ionised_fraction(
            values, dt=grids.dt_fine, kind="field", integral=rule
        )
        rate = plane_plasma.rate(np.abs(field))
        loss = np.zeros_like(field)
        np.divide(
            plane_plasma.potential * rate * (1.0 - fraction),
            field,
            out=loss,
            where=(rate > 0.0) & (field != 0.0),
        )
        current[index] = density * (
            rule.integrate(fraction * CHARGE_MASS_RATIO * field, grids.dt_fine) + loss
        )
    polarisation = kerr.evaluate(nodes, density, kind="field")
    projected = transform.projection @ (polarisation * grids.t_window_fine)
    spectrum = fourier.forward(projected, kind="field")[..., grids.crop_index]
    active = grids.active_mask
    q = 1j * grids.omega[active]
    for window, divisor in (
        (grids.t_window_fine, q),
        (grids.t_window_fine_derivative, -(q**2)),
    ):
        stream = fourier.forward(
            transform.projection @ (window * current), kind="field"
        )
        spectrum[..., active] += stream[..., grids.crop_index[active]] / divisor
    return prefactor_at(responses, z) * grids.omega_window * spectrum[:, np.newaxis, :]


@pytest.mark.parametrize("preionisation", [0.0, 0.25])
def test_one_stage_matches_the_responses_composed_by_hand(
    ionising_capillary, preionisation
):
    """The whole chain against the two oracles, with nothing of the backend in it."""
    problem, kerr, plasma = ionising_capillary(preionisation)
    backend = NumpyRefBackend(problem)
    backend.set_initial(problem.initial.field0, None)
    state = problem.initial.field0
    ones = np.ones_like(state)
    got = backend._nonlinear(state, backend._stage_scalars(0.0, 0.05), ones)
    want = reference_nonlinear(problem, kerr, plasma, state, 0.05)
    # Identical formulas, held to the plasma chain's rounding and the transforms'.
    samples = problem.grids.n_to
    scale = float(np.max(np.abs(want)))
    assert scale > 0
    bound = scan_bound(samples) + 2 * fft_bound(samples)
    assert np.max(np.abs(got - want)) <= bound * scale


def test_the_pre_ionisation_is_read_at_the_plane(ionising_capillary):
    """The oracle for a plane is ``dataclasses.replace`` with its value."""
    problem, _, plasma = ionising_capillary(0.4)
    table = problem.responses.plasma_coefficients.preionisation
    for z in (0.0, 0.05, 0.2):
        assert float(table(z)) == pytest.approx(0.4, rel=1e-14)
        replaced = dataclasses.replace(plasma, preionisation=float(table(z)))
        assert replaced.preionisation == pytest.approx(0.4, rel=1e-14)


def test_a_plasma_run_steps_and_ionises(backend_class, ionising_capillary):
    """A propagation that actually frees electrons, end to end."""
    problem, _, _ = ionising_capillary()
    backend = backend_class(problem)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(0.1)
    plane = backend.observe()
    assert summary.met_tolerance
    assert np.all(np.isfinite(plane.field))
    # The plasma blue-shifts the pulse, so the spectrum must not be the linear one.
    linear = np.exp(delta_phi(problem.linear, 0.0, 0.1)) * problem.initial.field0
    moved = np.max(np.abs(plane.field - linear)) / np.max(np.abs(linear))
    assert moved > 1e-3


# ------------------------------------------------------------------------- refusals


def test_an_envelope_grid_refuses_the_plasma_by_name(field_grid):
    """The model is written for a resolved carrier and has no envelope form."""
    grid = EnvelopeGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(TranslationError, match="PlasmaResponse is defined on"):
            assemble(
                grid,
                mode,
                [PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar))],
                start,
                controller=PIController(rtol=1e-8, atol=1e-14),
                save_planes=[0.1],
                dz_max=0.01,
                dz_init=1e-4,
                ionisation_field_range=FIELD_RANGE,
            )


# --------------------------------------------------------------- backend statistics


@pytest.fixture
def ionised_backend(field_grid, ionising_capillary, backend_class):
    """A capillary propagated far enough to leave a plasma behind."""
    problem, _, _ = ionising_capillary()
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    backend = backend_class(problem, grid=field_grid, modes=[mode])
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.1)
        yield problem, backend
    finally:
        backend.close()


def test_the_backend_reports_the_plasma_it_made(ionised_backend):
    """The statistics a saved plane does not carry are the backend's own."""
    problem, backend = ionised_backend
    spec = StatsSpec(
        names=("energy", "electron_density", "peak_ionisation_rate", "peak_intensity")
    )
    values = backend.stats(spec)
    assert sorted(values) == sorted(spec.names)
    for name in spec.names:
        assert values[name].shape[0] == 1
    # The catalogue's shapes, with the leading batch axis the protocol promises.
    for name in ("electron_density", "peak_ionisation_rate", "peak_intensity"):
        assert values[name].shape == (1,)
    assert values["energy"].shape == (1, 1, 1)
    # A 200 uJ pulse in two bar of argon ionises a measurable fraction of it, and the
    # density it leaves cannot exceed the gas it came from.
    density = float(problem.responses.density(0.1))
    assert 0.0 < float(values["electron_density"][0]) < density
    assert float(values["peak_ionisation_rate"][0]) > 0.0
    assert float(values["peak_intensity"][0]) > 0.0


def test_the_electron_density_is_the_fraction_the_pulse_leaves(ionised_backend):
    """``density(z)`` times the fraction at the end of the window, at the busiest node.

    Not the largest fraction reached in passing, and not an on-axis field
    re-synthesised for it.
    """
    problem, backend = ionised_backend
    plasma = problem.responses.plasma_coefficients
    plane = backend.observe()
    transform = problem.transform
    taper = (
        1.0 if transform.taper_scale is None else float(transform.taper_scale(plane.z))
    )
    modal = passes.to_time(
        plane.field[0],
        problem.grids.crop_index,
        problem.grids.n_to,
        kind=problem.grid_kind,
    )
    nodes = passes.synthesise(modal, transform.synthesis, taper, transform.n_pol)
    rate = passes.ionisation_rate(nodes, plasma.rate)
    fraction, _ = passes.plasma_scan(
        nodes,
        rate,
        plasma,
        preionisation=float(plasma.preionisation(0.1)),
        dt=problem.grids.dt_fine,
    )
    densest = int(np.argmax(fraction[:, -1]))
    values = backend.stats(
        StatsSpec(names=("electron_density", "peak_ionisation_rate"))
    )
    expected = float(problem.responses.density(0.1)) * float(fraction[densest, -1])
    # The backend and this independently re-synthesised reference cross FFT, GEMM and
    # scalar/vector libm paths. Use the existing 1e-10 propagated-statistic contract.
    assert float(values["electron_density"][0]) == pytest.approx(expected, rel=1e-10)
    assert float(values["peak_ionisation_rate"][0]) == pytest.approx(
        float(np.max(rate[densest])), rel=1e-10
    )


def test_the_peak_intensity_of_a_mode_averaged_run_is_taken_on_its_mode_s_axis(
    ionised_backend,
):
    """One definition for every run: the on-axis intensity of the mode profile.

    The mode-averaged form ``P/A_eff`` is the average intensity, which is the on-axis
    value only for a top-hat profile.
    """
    problem, backend = ionised_backend
    plane = backend.observe()
    mode = backend._modes[0]
    values = backend.stats(StatsSpec(names=("peak_intensity", "peak_power_total")))
    profile = mode.field(np.zeros((1, 2)), plane.z)[0] / math.sqrt(
        mode.normalisation(plane.z)
    )
    expected = (
        0.5
        * EPS0
        * C
        * float(np.sum(profile**2))
        * float(values["peak_power_total"][0])
    )
    # The profile at one point and the peak power: a two-factor product, so round-off.
    assert float(values["peak_intensity"][0]) == pytest.approx(expected, rel=1e-14)
    # For HE_11 the on-axis intensity is int J0^2 / int J0^4 = 1.768 times the average
    # over the effective area, which is what the mode-averaged form reports.
    average = float(values["peak_power_total"][0]) / float(
        problem.transform.aeff(plane.z)
    )
    assert 1.75 < float(values["peak_intensity"][0]) / average < 1.79


def test_the_quadrature_error_is_reported_and_is_not_a_number_without_a_rule(
    ionised_backend,
):
    """A one-node transform has no embedded pair: the estimate is nan, not zero."""
    _, backend = ionised_backend
    values = backend.stats(StatsSpec(names=("quadrature_error",)))
    assert np.isnan(values["quadrature_error"][0])


def test_a_plasma_statistic_without_a_plasma_is_refused(backend_class, field_grid):
    """The rate has to come from somewhere, and a Kerr-only problem has none."""
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            field_grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=1e-4,
        )
    backend = backend_class(problem, grid=field_grid)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="declares none"):
        backend.stats(StatsSpec(names=("electron_density",)))


# ------------------------------------------------- guards on an inconsistent order


@pytest.mark.parametrize(
    ("field", "named"),
    [
        ("kerr_coefficients", "Kerr stage"),
        ("plasma_channels", "plasma stage"),
        ("ionisation_table", "ionisation stage"),
    ],
)
def test_a_stage_without_its_coefficients_is_refused(ionising_capillary, field, named):
    """`response_order` and the coefficients come from one compiler and can only
    disagree in a hand-built or stale problem; the pass says which is missing rather
    than reading a ``None``."""
    problem, _, _ = ionising_capillary()
    responses = dataclasses.replace(problem.responses)
    object.__setattr__(responses, field, None if field == "kerr_coefficients" else ())
    with pytest.raises(BackendError, match=named):
        passes.nonlinear_polarisation(
            np.zeros((1, 1, problem.grids.n_to)),
            responses,
            problem.grids.t_window_fine,
            kind="field",
            density=1.0,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
        )


def test_a_plasma_stage_out_of_sequence_is_refused(ionising_capillary):
    """The current cannot run before the scan that feeds it."""
    problem, _, _ = ionising_capillary()
    responses = dataclasses.replace(problem.responses)
    object.__setattr__(
        responses,
        "response_order",
        (ResponseStage.PLASMA_CURRENT, ResponseStage.ACCUMULATE),
    )
    with pytest.raises(BackendError, match="out of sequence"):
        passes.nonlinear_polarisation(
            np.zeros((1, 1, problem.grids.n_to)),
            responses,
            problem.grids.t_window_fine,
            kind="field",
            density=1.0,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
        )


def test_the_plasma_statistics_are_the_same_between_planes_as_at_one(
    field_grid, ionising_capillary, backend_class
):
    """The channel's electron density is the plane's, on the run that ionises.

    The expensive half of the periodic subset: the density and the peak rate need the
    field at the quadrature nodes and the run's rate table, so they are the backend's
    own and the row goes through the same ``stats`` a plane does: the same call on
    the same accepted state, so the same numbers to the plasma chain's rounding.
    """
    names = ("energy", "electron_density", "peak_ionisation_rate", "peak_intensity")
    problem, _, _ = ionising_capillary()
    problem = dataclasses.replace(
        problem, stats_spec=StatsSpec(names=names, periodic=names, period=1)
    )
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    backend = backend_class(problem, grid=field_grid, modes=[mode])
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.1)
        observed = backend.observe()
        at_plane = backend.stats(problem.stats_spec)
    finally:
        backend.close()
    rows = observed.step_stats
    assert rows.n_rows == int(np.count_nonzero(observed.step_history.accepted))
    for name in names:
        np.testing.assert_allclose(
            rows.values[name][-1],
            at_plane[name],
            rtol=scan_bound(field_grid.n_to),
            err_msg=name,
        )
    # The channel is a trace, not one number repeated. It is not asserted to rise:
    # this fixture's fill is a *falling* two-point gradient, 2.0 bar to 0.2 bar, so
    # the gas density and the plasma it leaves behind both fall along z.
    density = rows.values["electron_density"][:, 0]
    assert np.all(density > 0.0)
    assert float(np.ptp(density)) > 0.0


def test_the_on_axis_matrix_is_built_once_and_rescaled(
    field_grid, ionising_capillary, monkeypatch, backend_class
):
    """``peak_intensity`` between planes must not rebuild the mode matrix per row.

    What kept it out of the channel was a mode-matrix build on every call;
    the matrix is built once at the entrance and rescaled with the taper, so a run of
    many rows builds it exactly once, and the row at the plane is the plane's own.
    """
    import solprop.backends.driver as driver_module  # noqa: PLC0415 - patched here

    names = ("peak_intensity",)
    problem, _, _ = ionising_capillary()
    problem = dataclasses.replace(
        problem, stats_spec=StatsSpec(names=names, periodic=names, period=1)
    )
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    builds = []
    real = driver_module.mode_matrix
    monkeypatch.setattr(
        driver_module,
        "mode_matrix",
        lambda *args, **kwargs: builds.append(args) or real(*args, **kwargs),
    )
    backend = backend_class(problem, grid=field_grid, modes=[mode])
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.1)
        observed = backend.observe()
        at_plane = backend.stats(problem.stats_spec)
    finally:
        backend.close()
    rows = observed.step_stats
    assert rows.n_rows > 1
    assert len(builds) == 1
    np.testing.assert_allclose(
        rows.values["peak_intensity"][-1],
        at_plane["peak_intensity"],
        rtol=scan_bound(field_grid.n_to),
    )
    assert float(np.ptp(rows.values["peak_intensity"][:, 0])) > 0.0


def test_the_on_axis_matrix_follows_a_taper(field_grid, backend_class):
    """On a tapered capillary the cached matrix is rescaled at every position.

    The entrance matrix times the transform's taper factor is the matrix built at z, to
    the tolerance the factor is tabulated at, and the row at a plane is the plane's own
    number through the same cache.
    """
    from solprop.modes.quadrature import mode_matrix, on_axis_rule  # noqa: PLC0415
    from solprop.profiles import Linear  # noqa: PLC0415 - one test builds a taper

    mode = MarcatiliMode(
        Linear(125e-6, 100e-6, field_grid.length), "Ar", 1.0, cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-7).build(field_grid)
    names = ("peak_intensity",)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            field_grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=1e-4,
        )
    problem = dataclasses.replace(
        problem, stats_spec=StatsSpec(names=names, periodic=names, period=1)
    )
    assert problem.transform.taper_scale is not None
    backend = backend_class(problem, grid=field_grid, modes=[mode])
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(0.1)
        observed = backend.observe()
        at_plane = backend.stats(problem.stats_spec)
        rescaled = backend._on_axis_synthesis([mode])
    finally:
        backend.close()
    rule = on_axis_rule(mode.dimlimits(0.1)[0])
    built = mode_matrix([mode], rule, 0.1, components=("x", "y")).synthesis
    np.testing.assert_allclose(rescaled, built, rtol=1e-8, atol=0.0)
    rows = observed.step_stats
    assert rows.n_rows > 1
    np.testing.assert_allclose(
        rows.values["peak_intensity"][-1],
        at_plane["peak_intensity"],
        rtol=scan_bound(field_grid.n_to),
    )
