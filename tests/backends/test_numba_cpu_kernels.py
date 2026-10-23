# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The fused kernels against the reference passes they were fused from.

Every kernel here replaces two or more of the seven numbered passes of
``architecture.md`` section 8, and every test compares it with those passes run
separately on the same inputs. That is the only comparison worth making: a kernel that
agrees with itself proves nothing, and the whole claim of this backend is that fusing
the passes does not change the arithmetic.

Each comparison is a rounding bound fixed from the operations the two sides perform
(SEP 25): a kernel may contract, reorder and vectorise, so even a traversal that
performs the reference's operations in its order is held to their rounding, not to
their bits -- ``np.abs(z) ** 2`` for a modulus, and numpy's complex product, which
uses a fused multiply-add, already differ. What stays exact is structural: a hard
mask's stored zeros, zero padding, a copy, and arithmetic on binary-exact samples that
has no rounding to differ by.
"""

import math

import numpy as np
import pytest
from scipy.signal import hilbert

from bounds import EPS, assert_within, gamma, higham_gamma
from former_raman import FORMER_H2, former_response
from rate_oracles import adk_oracle
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu import fft, kernels, spatial
from solprop.backends.numpy_ref import passes
from solprop.constants import EPS0
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.gases import Gas
from solprop.profiles import Constant
from solprop.stepper.controller import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.responses import compile_responses
from solprop.translate.spatial import SeparableOperator, separable_values
from spatial_records import record_metric

FIELD_RANGE = (1e9, 2e11)
DENSITY = 2.5e25
PREIONISATION = 0.0625
DT = 1e-16
# The fused column under test is the trapezoid's, at a synthetic sample spacing no grid
# designs taps for; the band rule's kernels are `test_numba_band_kernels.py`'s.
N_TO = 64
N_OMEGA = N_TO // 2 + 1


# --- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def field_grid():
    """A small field grid; only its kind and its response compilation are used here."""
    return FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 30e-15)


@pytest.fixture(scope="module")
def envelope_grid():
    """A small envelope grid, oversampled enough for the harmonic to be legal."""
    return EnvelopeGrid.from_limits(0.1, 800e-9, (400e-9, 2e-6), 200e-15, thg=True)


@pytest.fixture(scope="module")
def mode():
    """A dispersion-specified mode, which every compilation here needs and none uses."""
    return SpecifiedMode.from_taylor([0.0], omega_ref=2.4e15, aeff=1e-10)


def compiled(grid, mode, responses, **options):
    """Compile a response set the way the translation layer does for a run."""
    return compile_responses(
        responses,
        grid,
        mode,
        density=Constant(DENSITY),
        z_max=0.1,
        prefactor_index=1.0,
        shock=False,
        **options,
    )


def four(values):
    """Give a ``(n_columns, n)`` buffer the batch and polarisation axes a pass wants."""
    return np.asarray(values)[np.newaxis, :, np.newaxis]


def three(values):
    """Give a ``(n_columns, n)`` buffer the polarisation axis a pass wants."""
    return np.asarray(values)[:, np.newaxis]


def node_field(n_nodes, n_pol, *, complex_samples, seed=3) -> np.ndarray:
    """Return a chunk's field with the amplitudes an ionising run actually reaches."""
    rng = np.random.default_rng(seed)
    envelope = np.exp(-((np.arange(N_TO) - N_TO / 2) ** 2) / 200.0)
    carrier = np.cos(0.7 * np.arange(N_TO))
    real = 3e10 * envelope * carrier * (1.0 + 0.1 * rng.normal(size=N_TO))
    rows = np.stack(
        [real * (1.0 - 0.3 * component) for component in range(n_nodes * n_pol)]
    ).astype(np.float64)
    if complex_samples:
        return np.ascontiguousarray(hilbert(rows, axis=-1))
    return np.ascontiguousarray(rows)


# --- the small kernels ---------------------------------------------------------


def test_the_propagator_pair_kernel_is_both_exponentials():
    """One real ``exp`` and one ``sincos`` per element, for both signs."""
    rng = np.random.default_rng(5)
    exponent = rng.normal(size=(3, 8)) * 0.4 + 1j * rng.normal(size=(3, 8)) * 3.0
    forward = np.empty_like(exponent)
    inverse = np.empty_like(exponent)
    kernels.exponentiate(exponent, forward, inverse)
    # Two ulp: the same numbers by a different route, so the agreement is round-off in
    # the last place and not identity (test_bodies measures the one-ulp libm gap).
    assert forward == pytest.approx(np.exp(exponent), rel=2.3e-16)
    assert inverse == pytest.approx(np.exp(-exponent), rel=2.3e-16)


def test_the_hard_mask_kernel_is_the_reference_pass():
    """Exact: a multiply by one or a store of zero, whichever the mask says.

    A hard mask is a structural contract (SEP 25): neither operation rounds.
    """
    rng = np.random.default_rng(6)
    state = rng.normal(size=(2, N_OMEGA)) + 1j * rng.normal(size=(2, N_OMEGA))
    mask = np.arange(N_OMEGA) % 3 != 0
    out = np.empty_like(state)
    kernels.apply_hard_mask(state, mask, out)
    wanted = passes.apply_hard_mask(four(state), mask)
    assert np.array_equal(out, wanted[0, :, 0])
    assert np.count_nonzero(out) > 0


def test_the_accept_kernel_is_the_reference_accept_and_norm():
    """The state to one complex product's worth of round-off, and the norm with it.

    Not bitwise: numpy's complex multiply and numba's contracted code each choose
    where to fuse a multiply-add, so the two products round differently. One complex
    product rounds each component by at most about one ulp, which with the add sets
    the 2 eps bound relative to the largest value.
    """
    rng = np.random.default_rng(7)
    shape = (3, N_OMEGA)

    def draw():
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    state, propagator, solution, error = draw(), draw(), draw(), draw()
    mask = np.arange(N_OMEGA) < N_OMEGA - 4
    dz, controller = 1e-4, PIController(rtol=1e-8, atol=1e-20)
    advanced = np.empty_like(state)
    partials = np.empty((3, shape[0]))
    kernels.accept_and_partials(
        state, propagator, dz, solution, error, mask, advanced, partials
    )
    wanted = passes.accept_step(four(state), four(propagator), dz, four(solution))
    largest = np.max(np.abs(advanced - wanted[0, :, 0])) / np.max(np.abs(wanted))
    assert largest < 2 * np.finfo(float).eps
    assert np.max(np.abs(advanced)) > 0.0
    # In the coverage lane the difference is exactly zero, because with the compiler
    # disabled the same numpy scalar multiply runs on both sides. The gap is between
    # numba's code generation and numpy's, not between the two formulas.

    old = math.sqrt(float(np.sum(partials[0])))
    new = math.sqrt(float(np.sum(partials[1])))
    residual = math.sqrt(float(np.sum(partials[2])))
    scale = controller.scale(old, new)
    got = residual / float(scale)
    reference = passes.error_norm(
        four(propagator * (dz * error)), four(state), four(advanced), mask, controller
    )
    # Both are sums of the same squares; the kernel's are grouped by column and the
    # reference's by numpy's pairwise order, so 3 eps on the reduction and no more.
    assert got == pytest.approx(reference, rel=3 * np.finfo(float).eps)
    assert got > 0.0


@pytest.mark.parametrize("threaded", [False, True])
def test_the_accept_pass_is_the_reference_on_a_threaded_state(threaded):
    """Serial and column-threaded accept passes: the state and the partials, bounded.

    The state is one complex product per bin from the reference's, two eps as in the
    test above. Each partial is a sum of ``n_omega`` nonnegative squares, each square
    two roundings: any order of the kernel's and of numpy's pairwise sum is within
    ``gamma(n_omega + 2)`` of the exact sum relative to itself, so the two within twice
    that (SEP 25 section 4.1). The threaded variant is what a state of
    ``ACCEPT_ELEMENTS`` in ``ACCEPT_COLUMNS`` columns takes on more than one thread.
    """
    n_columns = kernels.ACCEPT_COLUMNS
    n_omega = kernels.ACCEPT_ELEMENTS // n_columns
    accept, weighted = kernels.accept_kernels(2 if threaded else 1, n_columns, n_omega)
    # Compiled only: under NUMBA_DISABLE_JIT both variants are the one plain function.
    if hasattr(kernels.accept_and_partials, "py_func"):
        assert (accept is kernels.accept_and_partials_parallel) is threaded
        assert (weighted is kernels.weighted_residual_parallel) is threaded
    rng = np.random.default_rng(11)

    def draw():
        return rng.normal(size=(n_columns, n_omega)) + 1j * rng.normal(
            size=(n_columns, n_omega)
        )

    state, propagator, solution, error = draw(), draw(), draw(), draw()
    mask = np.arange(n_omega) % 7 != 0
    dz = 2e-4
    advanced = np.empty_like(state)
    partials = np.empty((3, n_columns))
    accept(state, propagator, dz, solution, error, mask, advanced, partials)
    wanted = propagator * (state + dz * solution)
    assert_within(advanced, wanted, 2 * EPS, what="advanced state")
    bound = 2 * higham_gamma(n_omega + 2)
    for row, values in enumerate((state, wanted, propagator * (dz * error))):
        expected = np.sum(np.abs(values[:, mask]) ** 2, axis=1)
        assert_within(partials[row], expected, bound, what=f"partial {row}")
    scale = np.linspace(0.5, 2.0, n_omega)
    residual = np.empty(n_columns)
    weighted(propagator, dz, error, mask, scale, residual)
    expected = np.sum(
        np.abs((propagator * (dz * error))[:, mask] / scale[mask]) ** 2, 1
    )
    # One more rounding per bin, the division.
    assert_within(residual, expected, 2 * higham_gamma(n_omega + 4), what="weighted")


def test_a_small_state_takes_the_serial_accept_pass():
    """Too few columns, too few elements or one thread: the serial pass."""
    serial = (kernels.accept_and_partials, kernels.weighted_residual)
    columns, elements = kernels.ACCEPT_COLUMNS, kernels.ACCEPT_ELEMENTS
    assert kernels.accept_kernels(6, columns - 1, elements) == serial
    assert kernels.accept_kernels(6, columns, elements // columns - 1) == serial
    assert kernels.accept_kernels(1, columns, elements) == serial


def test_the_absorber_and_raman_scaling_kernels_are_plain_multiplies():
    """Both are in-place, and both carry a factor the transform does not."""
    samples = np.arange(12.0).reshape(2, 6)
    factor = np.linspace(0.5, 1.0, 6)
    wanted = samples * factor
    kernels.scale_samples(samples, factor)
    # One real product per sample on either side: one rounding each.
    assert_within(samples, wanted, EPS, what="scaled samples")

    spectrum = np.arange(8, dtype=complex).reshape(2, 4) + 1j
    kernel = np.linspace(1.0, 2.0, 4) + 0.5j
    expected = spectrum * kernel * 0.25
    kernels.scale_spectrum(spectrum, kernel, 0.25)
    # One complex product and a binary scaling: the accept kernel's two eps.
    assert_within(spectrum, expected, 2 * EPS, what="scaled spectrum")


def test_the_peak_reduction_takes_the_sum_inside_the_maximum():
    """The largest total at one instant, not the sum of separately timed maxima."""
    columns = np.array([[1.0 + 0j, 0.0, 3.0], [0.0, 2.0 + 0j, 1.0]])
    partials = np.empty(1)
    kernels.peak_reduce(columns, partials)
    assert partials[0] == 10.0
    assert partials[0] == float(np.max(np.sum(np.abs(columns) ** 2, axis=0)))


# --- the fused stage passes ------------------------------------------------------


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("noise", [False, True])
def test_the_first_kernel_is_the_stage_input_and_the_pad(kind, noise):
    """Passes 1 and 2's scatter: the same lab-frame state on the same fine grid.

    The envelope grid is the case the contiguous fast path cannot take: its crop index
    puts the negative frequencies at the far end of the fine spectrum, so the bins are
    scattered rather than copied as a leading block.
    """
    from solprop import fourier  # noqa: PLC0415 - only this test needs the crop index

    n_columns, n_t = 3, 16
    n_omega = fourier.n_omega_for(n_t, kind)
    crop = fourier.crop_index(n_t, N_TO, kind=kind)
    contiguous = bool(np.array_equal(crop, np.arange(n_omega)))
    assert contiguous == (kind == "field")

    rng = np.random.default_rng(11)

    def draw(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    state = draw((n_columns, n_omega))
    stages = draw((2, n_columns, n_omega))
    weights = np.array([0.25, -0.5])
    propagator = draw((n_columns, n_omega))
    drawn = draw((n_columns, n_omega)) if noise else np.zeros((n_columns, n_omega))
    dz, scale = 3e-4, 1.0 / N_TO

    fine = np.empty((n_columns, N_TO if kind == "envelope" else N_TO // 2 + 1), complex)
    kernel = kernels.make_combine_and_pad(contiguous=contiguous, noise=noise)
    kernel(state, stages, weights, 2, dz, propagator, drawn, crop, fine, scale)

    combination = np.zeros_like(four(state))
    for index in range(2):
        combination = passes.accumulate(
            combination, four(stages[index]), weights[index]
        )
    lab = passes.stage_input(
        four(state), combination, dz, four(propagator), four(drawn) if noise else None
    )
    wanted = np.zeros_like(fine)
    wanted[:, crop] = lab[0, :, 0] * scale
    # One complex product apiece, so the FMA gap of the accept kernel again.
    largest = np.max(np.abs(fine - wanted)) / np.max(np.abs(wanted))
    assert largest < 2 * np.finfo(float).eps
    assert np.count_nonzero(fine) > 0


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_the_last_kernel_is_the_stage_output_and_both_accumulations(kind):
    """Passes 6 and 7: crop, prefactor, window, interaction frame, two sums."""
    from solprop import fourier  # noqa: PLC0415 - only this test needs the crop index

    n_columns, n_t = 2, 16
    n_omega = fourier.n_omega_for(n_t, kind)
    crop = fourier.crop_index(n_t, N_TO, kind=kind)
    contiguous = bool(np.array_equal(crop, np.arange(n_omega)))
    n_fine = N_TO if kind == "envelope" else N_TO // 2 + 1

    rng = np.random.default_rng(12)

    def draw(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    spectrum = draw((n_columns, n_fine))
    prefactor = draw((n_columns, n_omega))
    window = rng.uniform(0.5, 1.0, n_omega)
    inverse = draw((n_columns, n_omega))
    weight, error_weight = 0.375, -0.125

    stage = np.empty((n_columns, n_omega), complex)
    solution = np.zeros_like(stage)
    error = np.zeros_like(stage)
    damping = draw((n_columns, n_omega))
    ignored = np.full_like(damping, np.nan)
    kernel = kernels.make_finish_stage(contiguous=contiguous)
    kernel(
        spectrum,
        crop,
        prefactor,
        window,
        inverse,
        ignored,
        weight,
        error_weight,
        stage,
        solution,
        error,
    )

    wanted = passes.stage_output(
        three(spectrum[:, crop]), three(prefactor), window, four(inverse)
    )[0, :, 0]
    largest = np.max(np.abs(stage - wanted)) / np.max(np.abs(wanted))
    assert largest < 3 * np.finfo(float).eps
    # One real scaling of each component on either side: one rounding.
    assert_within(solution, weight * stage, EPS, what="solution accumulation")
    # Compiled without the absorber, the kernel never reads its buffer: a whole array
    # of NaN above would otherwise have poisoned the result.
    assert np.all(np.isfinite(stage))

    # And with it: the term is added to the normalised polarisation, before the frame.
    damped = np.zeros_like(stage)
    kernels.make_finish_stage(contiguous=contiguous, absorber=True)(
        spectrum,
        crop,
        prefactor,
        window,
        inverse,
        damping,
        weight,
        error_weight,
        damped,
        np.zeros_like(stage),
        np.zeros_like(stage),
    )
    with_term = passes.stage_output(
        three(spectrum[:, crop]),
        three(prefactor),
        window,
        four(inverse),
        three(damping),
    )[0, :, 0]
    assert np.max(np.abs(damped - with_term)) / np.max(np.abs(with_term)) < (
        3 * np.finfo(float).eps
    )
    assert_within(error, error_weight * stage, EPS, what="error accumulation")


# --- the fused response column ---------------------------------------------------


def run_column(  # noqa: PLR0913 - buffers are a kernel's parameters
    spec, compiled, nodes, window, *, drive=None, carrier=None, stash=None, current=None
):
    """Run the fused column over one chunk and return its polarisation."""
    kerr = compiled.kerr_coefficients
    plasma = compiled.plasma_coefficients
    n_rows, n_to = nodes.shape
    out = np.zeros((n_rows, n_to), complex if spec.family == "envelope" else float)
    kernel = kernels.make_response_column(spec)
    kernel(
        nodes,
        out,
        np.zeros((n_rows // spec.n_pol, 2 * n_to)) if drive is None else drive,
        window,
        RateArrays.of(plasma.rate) if plasma is not None else EMPTY_RATE,
        np.ones(n_to, complex) if carrier is None else carrier,
        (kerr.coefficient * DENSITY) if kerr is not None else 0.0,
        kerr.self_factor if kerr is not None else 0.0,
        kerr.conjugate_factor if kerr is not None else 0.0,
        kerr.harmonic_factor if kerr is not None else 0.0,
        plasma.charge_mass_ratio if plasma is not None else 0.0,
        plasma.ionisation_potential if plasma is not None else 0.0,
        DENSITY,
        PREIONISATION,
        DT,
        np.zeros((n_rows // spec.n_pol, 2)) if stash is None else stash,
        None,
        np.empty((n_rows, n_to)) if current is None else current,
    )
    return out


EMPTY_RATE = RateArrays(
    coefficients=np.zeros((4, 1)),
    log_field_min=0.0,
    d_log_field=1.0,
    field_min=np.inf,
    field_max=np.inf,
    rate_max=0.0,
)


def reference_column(compiled, field, window, kind, **options):
    """Run the same chunk through the reference's fused pass."""
    return passes.nonlinear_sources(
        field,
        compiled,
        window,
        kind=kind,
        density=DENSITY,
        preionisation=PREIONISATION,
        dt=DT,
        **options,
    ).polarisation


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("plasma", [False, True])
@pytest.mark.parametrize("thg", [True, False])
def test_the_field_column_is_the_reference_response_pass(
    field_grid, mode, n_pol, plasma, thg
):
    """A field grid's whole nonlinear term, fused, against the seven separate passes.

    Both field families and both component counts, with and without the plasma. The
    ``field_analytic`` family is handed the analytic signal, which is what its inverse
    plan produces; the reference is handed the real field and forms the analytic signal
    itself, once per chunk, inside the Kerr pass.
    """
    rate = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    responses: list[object] = [KerrResponse(chi3=3.5e-30, thg=thg)]
    if plasma:
        responses.append(
            PlasmaResponse(
                Gas.Ar, rate, preionisation=PREIONISATION, integration="trapezoid"
            )
        )
    built = compiled(field_grid, mode, responses, n_pol=n_pol)
    spec = kernels.column_spec(built, "field", n_pol, parallel=False)
    assert spec.family == ("field_real" if thg else "field_analytic")
    assert spec.plasma is plasma

    n_nodes = 2
    real = node_field(n_nodes, n_pol, complex_samples=False)
    nodes = real if thg else np.ascontiguousarray(hilbert(real, axis=-1))
    window = np.linspace(0.5, 1.0, N_TO)

    current = np.full(real.shape, np.nan)
    got = run_column(spec, built, nodes, window, current=current)
    if plasma:
        expected_current = passes.nonlinear_sources(
            real.reshape(n_nodes, n_pol, N_TO),
            built,
            window,
            kind="field",
            density=DENSITY,
            preionisation=PREIONISATION,
            dt=DT,
        ).current
        assert expected_current is not None
        # Two prefix scans, guarded loss and the rate's log/exp interpolation.
        allowance = gamma(64 * N_TO) * float(np.max(abs(expected_current)))
        assert allowance > 0
        np.testing.assert_allclose(
            current.reshape(expected_current.shape),
            expected_current,
            rtol=0,
            atol=allowance,
        )
    else:
        assert np.isnan(current).all()
    wanted = reference_column(
        built, real.reshape(n_nodes, n_pol, N_TO), window, "field"
    )
    scale = float(np.max(np.abs(wanted)))
    assert scale > 0.0
    largest = float(np.max(np.abs(got.reshape(n_nodes, n_pol, N_TO) - wanted))) / scale
    # The plasma column performs the same additions as the reference, and the
    # Kerr term costs what test_bodies measured: numpy's complex product and its
    # np.abs(z) ** 2 modulus, which no scalar kernel reproduces. 8 eps covers both
    # families at both component counts; measured below 2 eps on every one.
    assert largest < 8 * np.finfo(float).eps


@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("thg", [True, False])
def test_the_envelope_column_is_the_reference_response_pass(
    envelope_grid, mode, n_pol, thg
):
    """An envelope grid's Kerr term, with and without the radiated harmonic."""
    built = compiled(
        envelope_grid, mode, [KerrResponse(chi3=3.5e-30, thg=thg)], n_pol=n_pol
    )
    spec = kernels.column_spec(built, "envelope", n_pol, parallel=False)
    assert spec.family == "envelope"
    assert spec.harmonic is thg
    assert not spec.plasma

    n_nodes = 2
    nodes = node_field(n_nodes, n_pol, complex_samples=True)
    window = np.linspace(0.5, 1.0, N_TO)
    carrier = np.exp(2j * np.linspace(0.0, 7.0, N_TO))

    got = run_column(spec, built, nodes, window, carrier=carrier)
    wanted = reference_column(
        built,
        nodes.reshape(n_nodes, n_pol, N_TO),
        window,
        "envelope",
        carrier=carrier if thg else None,
    )
    scale = float(np.max(np.abs(wanted)))
    assert scale > 0.0
    largest = float(np.max(np.abs(got.reshape(n_nodes, n_pol, N_TO) - wanted))) / scale
    assert largest < 8 * np.finfo(float).eps


# --- the Raman convolution -------------------------------------------------------


def raman_kernel_spectrum(built, dt, z=0.0):
    """The damped kernel's spectrum on the doubled grid, as the backend forms it."""
    coefficients = built.raman_coefficients
    rates = tuple(float(table(z)) for table in coefficients.decay_rates)
    return passes.raman_spectrum(
        coefficients.undamped, rates, coefficients.fraction, dt
    )


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_the_raman_chain_is_the_reference_convolution(
    field_grid, envelope_grid, mode, kind
):
    """Drive, transform pair, kernel multiply and the delayed term, against the pass.

    The whole Raman path, which is where the fusion has the most to prove: the drive is
    written from the response column's own registers on the doubled grid, one transform
    pair serves a whole node chunk, and the delayed polarisation is added in the same
    traversal that applies the temporal collar.
    """
    grid = field_grid if kind == "field" else envelope_grid
    built = compiled(
        grid, mode, [KerrResponse(chi3=3.5e-30, thg=True), former_response(FORMER_H2)]
    )
    assert built.raman_coefficients is not None
    spec = kernels.column_spec(built, kind, 1, parallel=False)
    assert spec.raman

    n_nodes = 2
    nodes = node_field(n_nodes, 1, complex_samples=kind == "envelope")
    window = np.linspace(0.5, 1.0, N_TO)
    spectrum = raman_kernel_spectrum(built, DT)
    assert spectrum.shape == (grid.n_to + 1,)
    # The kernel spectrum belongs to the grid the response was compiled on; this test
    # convolves a shorter chunk, so it takes the leading bins of a doubled grid of its
    # own length. What is compared is the chain, not the kernel.
    spectrum = np.ascontiguousarray(spectrum[: N_TO + 1])

    plans = fft.build_plans(
        "field_real" if kind == "field" else "envelope",
        1,
        N_TO,
        N_TO,
        flag="estimate",
        chunk_columns=n_nodes,
    )
    out = np.zeros((n_nodes, N_TO), complex if kind == "envelope" else float)
    run_column(spec, built, nodes, window, drive=plans.drive)
    # The column's own drive, before the transform, is the reference's raman_drive.
    from solprop.nonlinear.raman import raman_drive  # noqa: PLC0415 - one test wants it

    for node in range(n_nodes):
        wanted_drive = raman_drive(
            nodes[node : node + 1], kind=kind, thg=built.raman_coefficients.thg
        )
        # A square, or a sum of two halved squares: at most three roundings.
        assert_within(plans.drive[node, :N_TO], wanted_drive, gamma(3), what="drive")
    # The doubled grid's padding is stored zeros, exactly.
    assert np.array_equal(plans.drive[:, N_TO:], np.zeros((n_nodes, N_TO)))

    plans.raman_to_frequency.execute()
    kernels.scale_spectrum(plans.drive_spectrum, spectrum, 1.0 / (2 * N_TO))
    plans.raman_to_time.execute()
    finish = kernels.make_raman_finish(spec)
    finish(out, nodes, plans.convolved[:, :N_TO], window, EPS0 * DENSITY)

    wanted = (
        passes.raman_polarisation(
            nodes.reshape(n_nodes, 1, N_TO),
            spectrum,
            kind=kind,
            thg=built.raman_coefficients.thg,
            density=DENSITY,
        )
        * window
    )
    scale = float(np.max(np.abs(wanted)))
    assert scale > 0.0
    largest = float(np.max(np.abs(out.reshape(n_nodes, 1, N_TO) - wanted))) / scale
    # FFTW against pocketfft over a doubled grid of 2 n_to: log2(2 n_to) eps by Higham,
    # which is 7 eps here.
    assert largest < math.log2(2 * N_TO) * np.finfo(float).eps


def test_the_statistics_column_is_the_reference_scan(field_grid, mode):
    """The plane's plasma statistics, fused, against the rate and the scan passes.

    The reference re-synthesises every node at once and then runs the multipass scan,
    which is two thirds of a whole CtC step on an x86-64 server. Both
    numbers come from the exposure, so one traversal gives them.
    """
    rate = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    built = compiled(
        field_grid,
        mode,
        [
            KerrResponse(chi3=3.5e-30, thg=True),
            PlasmaResponse(
                Gas.Ar, rate, preionisation=PREIONISATION, integration="trapezoid"
            ),
        ],
    )
    spec = kernels.column_spec(built, "field", 1, parallel=False)
    n_nodes = 3
    nodes = node_field(n_nodes, 1, complex_samples=False)

    partials = np.empty((2, n_nodes))
    kernels.make_stats_column(spec)(
        nodes, RateArrays.of(rate), PREIONISATION, DT, partials
    )

    column = nodes.reshape(n_nodes, 1, N_TO)
    rates = passes.ionisation_rate(column, rate)
    coefficients = built.plasma_coefficients
    assert coefficients is not None
    fraction, _ = passes.plasma_scan(
        column, rates, coefficients, preionisation=PREIONISATION, dt=DT
    )
    # Scalar and vector libm paths can differ by one ulp in the rate on x86. The 4 n eps
    # bound covers propagation through the scan and final exponential; taking the
    # maximum adds no arithmetic.
    bound = 4 * rates.shape[-1] * np.finfo(float).eps
    np.testing.assert_allclose(partials[0], fraction[:, -1], rtol=bound, atol=0.0)
    np.testing.assert_array_max_ulp(partials[1], np.max(rates, axis=-1), maxulp=1)
    assert float(np.max(partials[1])) > 0.0


def test_the_compiled_cache_tells_two_specialisations_apart():
    """A closure constant is part of numba's cache key, and this is why it matters.

    Every kernel here is specialised by closure constant rather than by argument, and
    the serial ones are compiled with ``cache=True``. If the on-disk cache were keyed on
    the function's source location alone, the second specialisation would silently get
    the first one's machine code -- a wrong answer with no error anywhere. It is not,
    and this asserts it rather than trusting it.
    """
    crop = np.array([0, 1, 6, 7], dtype=np.intp)
    state = np.ones((1, 4), dtype=complex)
    stages = np.zeros((1, 1, 4), dtype=complex)
    weights = np.zeros(1)
    propagator = np.ones((1, 4), dtype=complex)
    drawn = np.full((1, 4), 5.0 + 0.0j)

    results = {}
    for noise in (False, True):
        fine = np.full((1, 8), np.nan, dtype=complex)
        kernels.make_combine_and_pad(contiguous=False, noise=noise)(
            state, stages, weights, 0, 1.0, propagator, drawn, crop, fine, 1.0
        )
        results[noise] = fine.copy()
    assert np.array_equal(results[False][0, crop], np.ones(4, dtype=complex))
    assert np.array_equal(results[True][0, crop], np.full(4, 6.0 + 0.0j))
    # The scattered form clears the whole buffer; the contiguous one clears only the
    # tail, which on this crop index leaves the gap it does not own.
    fine = np.full((1, 8), np.nan, dtype=complex)
    kernels.make_combine_and_pad(contiguous=True, noise=False)(
        state, stages, weights, 0, 1.0, propagator, drawn, crop, fine, 1.0
    )
    assert np.isnan(fine[0, 2].real)
    assert np.array_equal(results[False][0, 2], 0.0 + 0.0j)


def test_the_weighted_residual_divides_before_the_norm_is_taken():
    """A weighted controller's scale is per bin, and the division is inside the norm.

    :meth:`solprop.stepper.controller.PIController.scale` returns one number when the
    controller has no band weights and one number per bin when it has. The weighted form
    redistributes the tolerance across the band, so a single summed residual divided by
    a scalar is not the same quantity -- which is exactly the defect this kernel exists
    to fix.
    """
    rng = np.random.default_rng(21)
    shape = (3, 16)
    propagator = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    error = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    mask = np.arange(shape[1]) % 4 != 0
    scale = np.linspace(0.5, 2.0, shape[1])
    dz = 3e-4

    partials = np.empty(shape[0])
    kernels.weighted_residual(propagator, dz, error, mask, scale, partials)
    got = math.sqrt(float(np.sum(partials)))

    residual = (propagator * (dz * error))[:, mask] / scale[mask]
    wanted = float(np.linalg.norm(residual))
    assert got == pytest.approx(wanted, rel=4 * np.finfo(float).eps)
    assert got > 0.0
    # A scalar scale is a different number, which is the whole point: dividing the
    # summed residual by the mean of the weights does not reproduce it.
    flat = float(np.linalg.norm((propagator * (dz * error))[:, mask])) / float(
        np.mean(scale)
    )
    assert abs(flat - wanted) / wanted > 1e-3


def ionising_column(field_grid, mode):
    """Return the compiled response set and rate table of an ionising field case."""
    rate = build_rate_table(adk_oracle(Gas.Ar), FIELD_RANGE)
    built = compiled(
        field_grid,
        mode,
        [
            KerrResponse(chi3=3.5e-30, thg=True),
            PlasmaResponse(
                Gas.Ar, rate, preionisation=PREIONISATION, integration="trapezoid"
            ),
        ],
    )
    return built, rate


def test_the_column_stashes_what_the_statistics_channel_reads(field_grid, mode):
    """The between-planes plasma pair, left behind by the pass that already had it.

    The specialised column writes each node's final ionised fraction and largest rate
    as it goes; the statistics column finds the same two by scanning the same nodes
    again. Both run the same compiled bodies over the same samples, so they agree to
    the scan's rounding -- which is what lets the channel read the cheap one
    (``docs/dev/backends.md`` section 3). The bound is the plasma ladder's ``4 n_to
    eps`` for the running sums and the final exponential, which also covers the rate's
    libm paths (:func:`test_the_statistics_column_is_the_reference_scan`).
    """
    built, rate = ionising_column(field_grid, mode)
    spec = kernels.column_spec(built, "field", 1, parallel=False, stash=True)
    assert spec.stash
    n_nodes = 3
    nodes = node_field(n_nodes, 1, complex_samples=False)
    stash = np.zeros((n_nodes, 2))
    run_column(spec, built, nodes, np.ones(N_TO), stash=stash)

    scanned = np.empty((2, n_nodes))
    plain = kernels.column_spec(built, "field", 1, parallel=False)
    kernels.make_stats_column(plain)(
        nodes, RateArrays.of(rate), PREIONISATION, DT, scanned
    )
    bound = 4 * N_TO * EPS
    assert_within(stash[:, 0], scanned[0], bound, what="fraction")
    assert_within(stash[:, 1], scanned[1], bound, what="largest rate")
    # And the values are the physics, not zeros the test would not have noticed.
    assert np.all(stash[:, 0] > PREIONISATION)
    assert np.all(stash[:, 1] > 0.0)


def test_a_column_without_the_stash_never_writes_to_it(field_grid, mode):
    """The flag is a closure constant, so a run that did not ask pays nothing."""
    built, _ = ionising_column(field_grid, mode)
    spec = kernels.column_spec(built, "field", 1, parallel=False)
    assert not spec.stash
    nodes = node_field(3, 1, complex_samples=False)
    untouched = np.zeros((3, 2))
    run_column(spec, built, nodes, np.ones(N_TO), stash=untouched)
    assert not untouched.any()


@pytest.mark.parametrize("n_pol", [1, 2])
def test_separable_spatial_kernels_match_the_frozen_host_operator(n_pol, request):
    """Exercise loss, gain, evanescence and inactive bins without coefficient tables."""
    coefficients = SeparableOperator(
        k_squared=np.array([1, 4 + 1j, 4 - 1j, 0, 9], complex),
        kperp_sq=np.array([0.0, 2.0, 4.0]),
        k_squared_low=np.zeros(5, complex),
        kperp_sq_low=np.zeros(3),
        frame=np.arange(5, dtype=float) / 8,
        spectral_rate=np.arange(5, dtype=float) / 16,
        prefactor_numerator=np.full(5, -2j),
        prefactor_clamped=np.full(5, -0.5j),
        active=np.array([True, True, True, True, False]),
        evanescent_rate=0.5,
        n_pol=n_pol,
        z_span=(0.0, 1.0),
        # A transverse collar's rate on two of the columns (kernel schema 31).
        kperp_rate=np.array([0.0, 0.25, 1.5]),
    )
    data = (
        coefficients.k_squared,
        coefficients.kperp_sq,
        coefficients.k_squared_low,
        coefficients.kperp_sq_low,
        coefficients.frame,
        coefficients.spectral_rate,
        coefficients.prefactor_numerator,
        coefficients.prefactor_clamped,
        coefficients.active,
        coefficients.evanescent_rate,
        n_pol,
        coefficients.kperp_rate,
    )
    operator, prefactor = separable_values(coefficients)
    shape = (3 * n_pol, 5)
    rate = operator.reshape(shape)
    factor = prefactor.reshape(shape)
    # Principal-square-root arithmetic and the complete pointwise chain use
    # fewer than 128 scalar operations; all exponents in this fixture are O(1).
    bound = 128 * np.finfo(float).eps

    def check(name, actual, expected, *, absolute=False):
        np.testing.assert_allclose(
            actual, expected, rtol=bound, atol=bound if absolute else 0
        )
        denominator = np.abs(expected) + (1 if absolute else 0)
        measured = np.max(np.abs(actual - expected) / denominator)
        record_metric(
            request,
            f"Cartesian compiled kernel {name} (scaled)",
            n_pol,
            measured,
            bound,
            "numba_cpu",
        )

    forward, inverse = np.empty(shape, complex), np.empty(shape, complex)
    spatial.write_propagator(data, 0.25, forward, inverse)
    check("propagator", forward, np.exp(0.25 * rate))
    check("inverse pair", inverse * forward, 1)
    exponent = np.empty(shape, complex)
    spatial.write_exponent(data, 0.25, exponent)
    # The propagator's own exponent before exponentiation: the host's to rounding.
    np.testing.assert_allclose(exponent, 0.25 * rate, rtol=bound)
    output = np.empty(shape, complex)
    spatial.write_phase(data, 0.25, np.real(output))
    np.testing.assert_allclose(
        np.real(output), 0.25 * np.imag(rate), rtol=0, atol=bound
    )
    record_metric(
        request,
        "Cartesian compiled kernel phase (rad)",
        n_pol,
        np.max(np.abs(np.real(output) - 0.25 * np.imag(rate))),
        bound,
        "numba_cpu",
    )
    noise = np.random.default_rng(233).normal(size=shape).astype(complex) * (1 + 1j)
    expected = noise * np.exp(0.25j * np.imag(rate))
    spatial.rotate_noise(noise, np.real(output), output)
    check("noise rotation", output, expected, absolute=True)
    spectrum = noise.copy()
    window = np.linspace(0, 1, 5)
    spatial.finish_material(data, spectrum, window, inverse, output)
    expected = factor * spectrum * window * inverse
    check("material prefactor", output, expected, absolute=True)
    solution, error = np.ones(shape, complex), np.zeros(shape, complex)
    spatial.accumulate_stage(output, (0.125, -0.25), solution, error)
    check("solution accumulation", solution, 1 + 0.125 * expected, absolute=True)
    check("error accumulation", error, -0.25 * expected, absolute=True)


@pytest.mark.parametrize("analytic", [False, True])
def test_cartesian_temporal_pack_and_crop_clear_remainder_columns(analytic):
    source = np.arange(10).reshape(2, 5).astype(complex) * (1 + 0.5j)
    crop = np.array([0, 1, 3, 5, 7])
    fine = np.full((3, 8), complex(np.nan))
    spatial.pack_columns(source, crop, fine, 0.25, analytic)
    recovered = np.empty_like(source)
    spatial.crop_columns(fine, crop, recovered)
    # The analytic rule doubles the fine bins strictly between zero and the fine
    # Nyquist bin (4 of 8 here): crop indices 1 and 3, not 5 and 7.
    weights = np.array([1, 2, 2, 1, 1]) if analytic else np.ones(5)
    # Copying, zero-fill and binary scalings of these exactly represented
    # samples have no rounding error; this checks the unused final chunk row.
    np.testing.assert_array_equal(recovered, source * weights / 4)
    np.testing.assert_array_equal(fine[2], 0)
    np.testing.assert_array_equal(fine[:, [2, 4, 6]], 0)


@pytest.mark.parametrize("n_pol", [1, 2])
def test_cartesian_absorber_columns_apply_the_physical_power_rate(n_pol):
    space_rate = np.array([0.5, 1.0, 2.0, 3.0])
    time_rate = np.array([0.0, 0.5, 1.0])
    field = np.arange(2 * n_pol * 3).reshape(2 * n_pol, 3).astype(float)
    output = np.empty_like(field)
    spatial.absorber_columns(field, (space_rate, time_rate), 1, n_pol, output)
    rate = np.repeat(space_rate[1:3], n_pol)[:, None] + time_rate
    # A power rate alpha is a field rate -alpha/2. These binary-exact
    # samples keep the comparison exact and distinguish the factor of two.
    np.testing.assert_array_equal(output, -rate * field / 2)


def test_add_temporal_absorber_adds_half_the_rate_times_the_samples():
    samples = np.arange(6, dtype=float).reshape(2, 3) * (1 + 1j)
    rate = np.array([0.0, 0.5, 2.0])
    output = np.ones((2, 3), dtype=complex)
    kernels.add_temporal_absorber(samples, rate, output)
    # Binary-exact samples and rates: a power rate alpha is the field rate -alpha/2.
    np.testing.assert_array_equal(output, 1 - 0.5 * rate * samples)


def test_write_temporal_absorber_overwrites_with_the_real_part_of_a_view():
    """The analytic family passes its signal's real part, a strided view."""
    analytic = np.arange(6, dtype=float).reshape(2, 3) * (1 - 2j)
    rate = np.array([0.0, 0.5, 2.0])
    output = np.full((2, 3), np.nan)
    kernels.write_temporal_absorber(analytic.real, rate, output)
    # Binary-exact samples and rates, grouped as the reference groups them.
    np.testing.assert_array_equal(output, -0.5 * rate * analytic.real)
