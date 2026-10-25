# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Raman convolution of the reference backend, against the response that defines it.

:meth:`solprop.nonlinear.raman.RamanResponse.evaluate` is the oracle
(``docs/dev/backends.md`` section 2), and
:func:`~solprop.backends.numpy_ref.passes.raman_polarisation` is its arithmetic, so
given the oracle's own kernel the pass reproduces it to one convolution's rounding on
both grids, for a glass and for a molecular gas, with and without the third harmonic.

What a *problem* runs is not quite that kernel: the translation layer tapers the
response's tail with the grid's own temporal collar, where the oracle cuts it with a
step (``docs/guide/media/raman.md``), and damps each term at the rate the contract's own
table
gives at the plane. So the one-stage test composes its reference from the models, the
taper and those rates, and the closed-form tests -- the small-signal gain and the
soliton self-frequency shift -- are stated against quantities the run itself is entitled
to: the imaginary part of its own kernel spectrum, and the first moment of the silica
response.
How far the taper moves a run from the untapered oracle is measured rather than assumed,
in ``test_the_taper_is_the_only_difference_from_the_oracles_kernel``.
"""

import dataclasses
import warnings

import numpy as np
import pytest

from bounds import assert_within, fft_bound
from former_raman import FORMER_H2, FORMER_N2, former_response
from solprop import fourier
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.constants import AMAGAT, EPS0
from solprop.driver import prop_gnlse
from solprop.exceptions import BackendError, MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import transverse_quadrature
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.raman import (
    MolecularOptions,
    raman_drive,
    response_for,
    vibrational,
)
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stepper import PIController
from solprop.translate.linop import delta_phi
from solprop.translate.problem import assemble
from solprop.translate.raman import (
    build_raman_kernels,
    kernel_spectrum,
    kernel_taper,
)
from solprop.translate.responses import prefactor_at

# The pass is the oracle's own arithmetic -- the same padded real transform pair, the
# same product, the same final multiplication -- but need not keep its order (SEP 25):
# the two agree to one round trip on the doubled grid, `fft_bound(2 n_to)` with its
# pointwise work, batched over a chunk of nodes as well as one column at a time.


def convolution_bound(grid):
    """One padded transform round trip and its products, relative (SEP 25)."""
    return fft_bound(2 * grid.n_to)


DENSITY = 1.0 * AMAGAT  # m^-3, about one bar at 273 K


def materials():
    """Return the three responses the pass is checked against, by name."""
    return {
        "N2": former_response(FORMER_N2),
        "H2": former_response(FORMER_H2),
        "SiO2": response_for("SiO2", amplitude=1e-22),
    }


@pytest.fixture(scope="module")
def raman_field_grid():
    """A field grid wide enough for a gas fill, 2048 fine samples."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 200e-15)


@pytest.fixture(scope="module")
def raman_envelope_grid():
    """An envelope grid on the same reference wavelength."""
    return EnvelopeGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 200e-15)


@pytest.fixture(scope="module")
def grids(raman_field_grid, raman_envelope_grid):
    """The two grids, by kind."""
    return {"field": raman_field_grid, "envelope": raman_envelope_grid}


def sample_field(grid, seed):
    """Return one node's field on the fine grid: real on a field grid, complex on an
    envelope one."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(grid.n_to) * 1e9
    if grid.kind == "envelope":
        values = values + 1j * rng.standard_normal(grid.n_to) * 1e9
    return values[np.newaxis]


# ----------------------------------------------------------- the pass against the oracl


@pytest.mark.parametrize("material", ["N2", "H2", "SiO2"])
@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("thg", [True, False])
def test_the_pass_reproduces_the_response(grids, material, kind, thg):
    """Given the oracle's own kernel, the pass *is* the oracle, to its rounding."""
    grid = grids[kind]
    base = materials()[material]
    response = dataclasses.replace(base, thg=thg)
    field = sample_field(grid, seed=11)
    expected = response.evaluate(field, DENSITY, kind=kind, dt=grid.dt_fine)
    # The oracle's kernel, untapered, so that only the pass is under test; the taper the
    # contract's rows carry is the subject of test_the_taper_is_the_only_difference.
    kernel = np.zeros(2 * grid.n_to)
    kernel[: grid.n_to] = response.impulse(np.arange(grid.n_to) * grid.dt_fine, DENSITY)
    got = passes.raman_polarisation(
        field[np.newaxis],
        kernel_spectrum(kernel, grid.dt_fine),
        kind=kind,
        thg=thg,
        density=DENSITY,
    )
    assert_within(got[0], expected, convolution_bound(grid))


def test_a_chunk_of_nodes_is_the_nodes_one_at_a_time(raman_envelope_grid):
    """The batched real transform pair is per-column arithmetic, node for node."""
    grid = raman_envelope_grid
    response = former_response(FORMER_N2)
    spectrum = build_raman_kernels(response, grid).spectrum(DENSITY)
    chunk = np.stack([sample_field(grid, seed) for seed in (1, 2, 3)])
    together = passes.raman_polarisation(
        chunk, spectrum, kind="envelope", thg=True, density=DENSITY
    )
    apart = np.concatenate(
        [
            passes.raman_polarisation(
                chunk[node : node + 1],
                spectrum,
                kind="envelope",
                thg=True,
                density=DENSITY,
            )
            for node in range(chunk.shape[0])
        ]
    )
    assert_within(together, apart, convolution_bound(grid))


def test_a_zero_density_gives_no_polarisation(raman_envelope_grid):
    """The density multiplies the polarisation, whatever the collapsed kernel says."""
    grid = raman_envelope_grid
    response = former_response(FORMER_H2)
    spectrum = build_raman_kernels(response, grid).spectrum(0.0)
    field = sample_field(grid, seed=5)[np.newaxis]
    got = passes.raman_polarisation(
        field, spectrum, kind="envelope", thg=True, density=0.0
    )
    assert np.all(got == 0.0)


# ------------------------------------------------------------ the damped kernel spectru


@pytest.mark.parametrize("material", ["N2", "H2", "SiO2"])
@pytest.mark.parametrize("density", [DENSITY, 2.0 * DENSITY, 0.0])
def test_the_damped_spectrum_is_the_translation_layers_own(
    raman_field_grid, material, density
):
    """The backend forms from the contract's rows what the physics side memoises, and
    the two agree to one transform and the rows' sum (``fft_bound(2 n_to)``)."""
    grid = raman_field_grid
    response = materials()[material]
    kernels = build_raman_kernels(response, grid)
    rates = [
        min(model.dephasing.decay_rate(density), 708.0 / grid.dt_fine)
        for model in response.models
    ]
    got = passes.raman_spectrum(kernels.undamped, rates, response.scale, grid.dt_fine)
    want = kernels.spectrum(density)
    if not np.any(want):
        # A molecular gas at zero density has no response at all: exactly zero.
        np.testing.assert_array_equal(got, want)
    else:
        assert_within(got, want, convolution_bound(grid))


def test_the_spectrum_needs_one_rate_for_every_row(raman_field_grid):
    kernels = build_raman_kernels(former_response(FORMER_N2), raman_field_grid)
    with pytest.raises(BackendError, match="2 row"):
        passes.raman_spectrum(kernels.undamped, [0.0], 1.0, raman_field_grid.dt_fine)


def test_a_vector_field_is_refused_by_the_pass():
    """``raman_drive`` takes the first component and says nothing about the rest, so a
    two-component chunk would be convolved without half of itself."""
    with pytest.raises(BackendError, match="tensor"):
        passes.raman_polarisation(
            np.zeros((1, 2, 16)),
            np.zeros(17, dtype=complex),
            kind="field",
            thg=True,
            density=1.0,
        )


def test_a_mismatched_convolution_is_refused():
    """The doubled grid is built from the field's own length; a spectrum of another
    length would convolve against the wrong axis rather than raise."""
    with pytest.raises(BackendError, match="kernel spectrum has shape"):
        passes.raman_polarisation(
            np.zeros((1, 1, 16)),
            np.zeros(5, dtype=complex),
            kind="field",
            thg=True,
            density=1.0,
        )


def test_a_declared_raman_stage_without_a_spectrum_is_refused(raman_capillary):
    """The order and the spectrum come from one caller and can only disagree in a
    hand-built problem; the pass says which is missing rather than reading a
    ``None``."""
    problem, _, _ = raman_capillary()
    with pytest.raises(BackendError, match="no kernel spectrum"):
        passes.nonlinear_polarisation(
            np.zeros((1, 1, problem.grids.n_to)),
            problem.responses,
            problem.grids.t_window_fine,
            kind="field",
            density=1.0,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
        )


# ----------------------------------------------------- one stage of a hydrogen capillar


@pytest.fixture(scope="module")
def raman_capillary(raman_field_grid):
    """A hydrogen-filled gradient capillary with a Kerr and a Raman response.

    The geometry of the plasma ladder: a two-point pressure gradient, so
    the operator varies along z, the prefactor is a table, the density the Raman kernel
    is damped at moves with it, and the fill's stage boundary at 0.2 m is a hard kink.
    """
    mode = MarcatiliMode(
        125e-6, "H2", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    kerr = KerrResponse(chi3=1.0e-51, thg=True)
    raman = former_response(FORMER_H2, thg=True)

    def build(responses=None, *, modal=False, chunk=None):
        start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=20e-6).build(
            raman_field_grid
        )
        quadrature = (
            transverse_quadrature([mode], n_r=4, radial_rule="gauss_kronrod")
            if modal
            else None
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaterialRangeWarning)
            problem = assemble(
                raman_field_grid,
                [mode] if modal else mode,
                [kerr, raman] if responses is None else responses,
                start,
                controller=PIController(rtol=1e-8, atol=1e-14),
                save_planes=[0.1, 0.2],
                dz_max=0.01,
                dz_init=1e-4,
                quadrature=quadrature,
            )
        if chunk is not None:
            problem = dataclasses.replace(
                problem,
                transform=dataclasses.replace(problem.transform, node_chunk=chunk),
            )
        return problem, kerr, raman

    return build


@pytest.mark.parametrize("override", [None, 1, 2])
def test_raman_pool_uses_its_doubled_operand_and_shared_override(
    thread_budget_lifted, raman_capillary, override, monkeypatch, record_property
):
    """Raman owns its size choice while explicit FFTW requests still govern both."""
    problem, _, _ = raman_capillary()
    monkeypatch.setattr(
        numba_fft, "_thread_budget", lambda: numba_fft.ThreadBudget(4, "affinity")
    )
    monkeypatch.setattr(numba_fft, "FFTW_THREAD_ELEMENTS", 2 * problem.grids.n_to)
    monkeypatch.delenv("SOLPROP_FFTW_THREADS", raising=False)
    if override == 1:
        monkeypatch.setenv("SOLPROP_FFTW_THREADS", "1")
    eps = np.finfo(float).eps
    reduction = problem.transform.n_nodes * problem.transform.n_pol * problem.shape[2]
    gamma = reduction * eps / (1 - reduction * eps)
    # The seven-stage initial-trial envelope, derived before either run.
    bound = 7 * (
        16 * eps * np.ceil(np.log2(problem.grids.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    reference = NumpyRefBackend(problem)
    try:
        reference.set_initial(problem.initial.field0, None)
        expected = reference._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        reference.close()
    backend = NumbaCpuBackend(
        problem,
        threads=1,
        fftw_threads=2 if override == 2 else None,
        plan_flag="estimate",
    )
    try:
        metadata = backend.metadata()
        assert metadata["fftw_threads"] == (override or 1)
        assert metadata["raman_fftw_threads"] == (override or 4)
        assert backend._plans.drive.shape == (1, 2 * problem.grids.n_to)
        backend.set_initial(problem.initial.field0, None)
        actual = backend._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        backend.close()
    deviation = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
    assert deviation <= bound
    record_property("raman_initial_trial", {"deviation": deviation, "bound": bound})


@pytest.mark.parametrize("chunk", [3, 4, 5])
@pytest.mark.parametrize("override", [None, 1, 2])
def test_raman_tail_trial_matches_reference_without_replanning(
    thread_budget_lifted, raman_capillary, chunk, override, monkeypatch, record_property
):
    """Actual remainder plans preserve the trial, padding and explicit pool requests."""
    problem, _, _ = raman_capillary(modal=True, chunk=chunk)
    assert problem.transform.n_nodes == 5
    tail = problem.transform.n_nodes % chunk
    monkeypatch.setattr(
        numba_fft, "_thread_budget", lambda: numba_fft.ThreadBudget(4, "affinity")
    )
    monkeypatch.setattr(numba_fft, "FFTW_THREAD_ELEMENTS", 4 * problem.grids.n_to)
    monkeypatch.delenv("SOLPROP_FFTW_THREADS", raising=False)
    if override == 1:
        monkeypatch.setenv("SOLPROP_FFTW_THREADS", "1")
    eps = np.finfo(float).eps
    reduction = problem.transform.n_nodes * problem.transform.n_pol * problem.shape[2]
    gamma = reduction * eps / (1 - reduction * eps)
    # The same seven-stage envelope at this test's dimensions.
    bound = 7 * (
        16 * eps * np.ceil(np.log2(problem.grids.n_to))
        + 2 * gamma
        + 64 * eps / (1 - 64 * eps)
    )
    reference = NumpyRefBackend(problem)
    try:
        reference.set_initial(problem.initial.field0, None)
        expected = reference._trial_step(0.0, problem.stepper.dz_init)[1].copy()
    finally:
        reference.close()
    backend = NumbaCpuBackend(
        problem,
        threads=1,
        fftw_threads=2 if override == 2 else None,
        plan_flag="estimate",
    )
    try:
        metadata = backend.metadata()
        assert metadata["raman_tail_nodes"] == tail
        assert metadata["raman_fftw_threads"] == (override or 4)
        expected_tail_threads = (override or (4 if tail >= 2 else 1)) if tail else None
        assert metadata["raman_tail_fftw_threads"] == expected_tail_threads
        assert (backend._plans.raman_tail_to_time is not None) == bool(tail)

        def refuse_replanning(*args, **kwargs):
            pytest.fail("a live trial attempted to construct an FFTW plan")

        monkeypatch.setattr(numba_fft, "_plan", refuse_replanning)
        fields = []
        for _ in range(2):
            backend.set_initial(problem.initial.field0, None)
            fields.append(backend._trial_step(0.0, problem.stepper.dz_init)[1].copy())
            assert np.array_equal(
                backend._plans.drive[:, problem.grids.n_to :],
                np.zeros((chunk, problem.grids.n_to)),
            )
        # The same trial twice: within the trial's own bound against the reference.
        repeat = float(
            np.max(np.abs(fields[1] - fields[0])) / np.max(np.abs(fields[0]))
        )
        assert repeat <= bound
        deviation = float(
            np.max(np.abs(fields[0] - expected)) / np.max(np.abs(expected))
        )
        assert deviation <= bound
        ledger = backend.memory_ledger()
        assert ledger["node_raman_drive"] == chunk * 2 * problem.grids.n_to * 8
        assert ledger["node_raman_convolved"] == chunk * 2 * problem.grids.n_to * 8
    finally:
        backend.close()
    record_property(
        "raman_tail_initial_trial", {"deviation": deviation, "bound": bound}
    )


def reference_nonlinear(problem, grid, kerr, raman, state, z):
    """Compose the two responses in plain NumPy, with nothing of the backend in it.

    The Kerr term is :meth:`~solprop.nonlinear.responses.KerrResponse.evaluate`; the
    Raman term is the formula of
    :meth:`~solprop.nonlinear.raman.RamanResponse.evaluate` written out over the
    **tapered** kernel the contract carries -- each model damped by
    :func:`solprop.translate.raman.kernel_taper` and by the decay rate the contract's
    own table gives at this plane, as the pre-ionised fraction is read at the plane in
    the plasma ladder. The rate is read there and not recomputed from the density,
    because the two differ by the table's interpolation error, which
    ``tests/translate/test_compiled_responses.py`` bounds separately.
    """
    grids, transform = problem.grids, problem.transform
    responses = problem.responses
    fine = np.zeros(
        (transform.n_modes, fourier.n_omega_for(grids.n_to, "field")), dtype=complex
    )
    fine[..., grids.crop_index] = state[0, :, 0, :]
    modal = fourier.inverse(fine, kind="field")
    nodes = transform.synthesis @ modal
    density = float(responses.density(z))
    delays = np.arange(grids.n_to) * grids.dt_fine
    taper = kernel_taper(grid)
    kernel = np.zeros(2 * grids.n_to)
    leaf = responses.raman_coefficients
    assert leaf is not None
    for model, table in zip(raman.models, leaf.decay_rates, strict=True):
        kernel[: grids.n_to] += (model.undamped(delays) * taper) * np.exp(
            -float(table(z)) * delays
        )
    spectrum = kernel_spectrum(raman.scale * kernel, grids.dt_fine)
    drive = np.zeros(2 * grids.n_to)
    drive[: grids.n_to] = raman_drive(nodes, kind="field", thg=raman.thg)
    convolved = np.fft.irfft(spectrum * np.fft.rfft(drive), n=2 * grids.n_to)[
        : grids.n_to
    ]
    delayed = (EPS0 * density) * nodes * convolved[np.newaxis, :]
    polarisation = kerr.evaluate(nodes, density, kind="field") + delayed
    windowed = polarisation * grids.t_window_fine
    spectrum_out = fourier.forward(transform.projection @ windowed, kind="field")
    cropped = spectrum_out[..., grids.crop_index][:, np.newaxis, :]
    return prefactor_at(responses, z) * grids.omega_window * cropped


@pytest.mark.parametrize("z", [0.0, 0.05, 0.15])
def test_one_stage_matches_the_responses_composed_by_hand(
    raman_capillary, raman_field_grid, z
):
    """The whole chain against the two oracles, at three densities of the gradient."""
    problem, kerr, raman = raman_capillary()
    backend = NumpyRefBackend(problem)
    backend.set_initial(problem.initial.field0, None)
    state = problem.initial.field0
    got = backend._nonlinear(state, backend._stage_scalars(0.0, z), np.ones_like(state))
    want = reference_nonlinear(problem, raman_field_grid, kerr, raman, state, z)
    # The Raman half is one convolution's rounding -- its kernel spectrum and its
    # convolution each have their own test above -- and the Kerr coefficient's grouping:
    # ``(eps0 chi3) rho`` compiled against ``eps0 chi3 rho`` evaluated, which differ by
    # one unit in the last place at a density that rounds badly. Measured 0.0 at
    # z = 0 and 0.15 and 3.4e-16 at z = 0.05, against the 4e-15 asserted; a Kerr-only
    # problem on this capillary shows the same 5.8e-16 at the same plane.
    assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 4e-15


def test_the_taper_is_the_only_difference_from_the_oracles_kernel(
    raman_field_grid, raman_envelope_grid
):
    """How much the grid's collar changes the answer, measured rather than assumed.

    The translation layer cuts the response with the grid's own temporal collar where
    :meth:`~solprop.nonlinear.raman.RamanResponse.evaluate` cuts it with a step
    (``docs/guide/media/raman.md``). For a glass, whose response is over in a
    picosecond, the
    two agree to round-off on any window that holds it; for a molecular gas, whose
    rotational coherence outlives every realistic window, they do not, and that is the
    departure the manual records.
    """
    measured = {}
    glass_grid = EnvelopeGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 5e-12)
    for grid, name, response in (
        (glass_grid, "SiO2", response_for("SiO2", amplitude=1e-22)),
        (raman_field_grid, "H2", former_response(FORMER_H2)),
    ):
        field = sample_field(grid, seed=3)
        oracle = response.evaluate(field, DENSITY, kind=grid.kind, dt=grid.dt_fine)
        tapered = passes.raman_polarisation(
            field[np.newaxis],
            build_raman_kernels(response, grid).spectrum(DENSITY),
            kind=grid.kind,
            thg=response.thg,
            density=DENSITY,
        )[0]
        measured[name] = float(
            np.max(np.abs(tapered - oracle)) / np.max(np.abs(oracle))
        )
    # Measured: silica 0.0 -- within one convolution's rounding, asserted -- on a 9.1 ps
    # envelope window, because its
    # response has decayed to nothing long before the taper starts (1.4e-6 on a 2.3 ps
    # window, 1.1e-2 on 228 fs); hydrogen 0.12 on a 228 fs field window, because its
    # rotational coherence lives for nanoseconds and the taper is what cuts it. Neither
    # is a tolerance on the physics: they bound how far a run may sit from the untapered
    # oracle, and the loose one is the point: a response that outlives the window is
    # cut by the taper by design.
    assert measured["SiO2"] <= convolution_bound(glass_grid)
    assert 1e-2 < measured["H2"] < 0.5


# ------------------------------------------------------------------------- the memo


@pytest.mark.parametrize(
    ("material", "expected"), [("H2", 3), ("SiO2", 1)], ids=["gas", "glass"]
)
def test_the_kernel_is_transformed_once_per_change_of_the_decay_rates(
    backend_class, raman_capillary, material, expected
):
    """One transform per distinct set of rates, and none for a plane that repeats.

    A glass's rates are zero at every density, so its kernel is transformed once for the
    whole run; a gas on a pressure gradient presents new rates at every plane. The memo
    is keyed on the rates and not on the density, which is what makes the two differ.
    """
    responses = [
        KerrResponse(chi3=1.0e-51, thg=True),
        former_response(FORMER_H2, thg=True)
        if material == "H2"
        else response_for("SiO2", amplitude=1e-22, thg=True),
    ]
    problem, _, _ = raman_capillary(responses)
    backend = backend_class(problem)
    for z in (0.0, 0.05, 0.05, 0.15, 0.15):
        rates = backend._stage_scalars(0.0, z).raman_rates
        backend._raman_spectrum(rates)
    transforms = backend._raman_transforms
    backend.close()
    assert transforms == expected


def test_a_problem_without_a_raman_kernel_has_no_spectrum(
    backend_class, raman_capillary
):
    problem, _, _ = raman_capillary([KerrResponse(chi3=1.0e-51, thg=True)])
    backend = backend_class(problem)
    assert backend._raman_spectrum(backend._stage_scalars(0.0, 0.0).raman_rates) is None
    backend.close()


def test_the_ledger_names_the_doubled_buffers(backend_class, raman_capillary):
    """The drive and its spectrum live on the doubled grid and are sized by the chunk
    (``architecture.md`` section 13); the kernel spectrum is one per stage."""
    with_raman, _, _ = raman_capillary()
    without, _, _ = raman_capillary([KerrResponse(chi3=1.0e-51, thg=True)])
    n_to = with_raman.grids.n_to
    chunk = with_raman.transform.node_chunk * with_raman.transform.n_pol
    owner = backend_class(with_raman)
    ledger = owner.memory_ledger()
    owner.close()
    owner = backend_class(without)
    bare = owner.memory_ledger()
    owner.close()
    assert ledger["node_raman_drive"] == chunk * 2 * n_to * 8
    assert ledger["node_raman_spectrum"] == chunk * (n_to + 1) * 16
    assert ledger["raman_kernel_spectrum"] == (n_to + 1) * 16
    for name in ("node_raman_drive", "node_raman_spectrum", "raman_kernel_spectrum"):
        assert bare[name] == 0


# ------------------------------------------------------------------- P0 with Raman on


def test_a_zero_weight_raman_response_is_the_exact_linear_flow(
    backend_class, linear_problem, narrow_grid
):
    """P0 of the ladder, with every Raman pass running and contributing nothing.

    ``docs/dev/stepper.md`` section 11: with the nonlinear term switched off the
    interaction picture is the whole scheme, so the propagated spectrum is
    ``exp(Phi(L)) E0`` however many steps were taken. A silica response of zero
    amplitude exercises the drive, the transform pair and the convolution and adds
    exactly nothing.
    """
    length = 0.05
    zero = response_for("SiO2", amplitude=0.0)
    problem = linear_problem(narrow_grid, save_planes=[length], responses=[zero])
    assert problem.flags.has_raman
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(length)
    final = backend.observe().require_field()
    backend.close()
    exact = (
        np.exp(delta_phi(problem.linear, 0.0, length))
        * problem.initial.field0
        * problem.grids.active_mask
    )
    # The same bound as the Kerr P0 rung: one propagator factor per step, so the
    # round-off is n_steps * eps (Higham). Measured 1.8e-16 over 43 steps.
    assert summary.n_steps > 8
    assert np.max(np.abs(final - exact)) / np.max(np.abs(exact)) < 1e-14


# ----------------------------------------------------------------- the small-signal gai

GAIN_BETA2 = 2.0e-26  # s^2/m, normal: it holds the anti-Stokes far off phase-match
GAIN_CHI3 = 1.0e-49  # m^5/V^2 per m^-3
GAIN_AEFF = 1.0e-9  # m^2
GAIN_PUMP = 1.0e6  # W
GAIN_SEED = 1.0e-3  # the Stokes seed, as a fraction of the pump amplitude


class GainSetup:
    """A continuous pump and a weak Stokes seed on the vibrational line of nitrogen.

    Everything the two gain tests share: the grid, the responses, the bin the seed sits
    in and the imaginary part of the run's own kernel spectrum there.

    Attributes
    ----------
    grid : solprop.grid.EnvelopeGrid
        The grid.
    kerr, raman : object
        The two responses; the Raman one is the vibration alone, so that its spectrum
        near the line is a single Lorentzian with no rotational tail under it.
    field : numpy.ndarray
        The initial state, two bins: the pump at the reference frequency and the seed
        one Raman shift below it.
    stokes, anti : int
        The bins of the seed and of its anti-Stokes mirror.
    shift : float
        The Raman shift the seed sits at, rad/s, exactly a bin of both the coarse grid
        and the doubled kernel grid.
    im_h : float
        ``Im H(Omega)`` of the run's own kernel there, negative where there is gain.
    """

    def __init__(self, trange):
        self.grid = EnvelopeGrid.from_limits(0.1, 835e-9, (650e-9, 1.2e-6), trange)
        self.kerr = KerrResponse(chi3=GAIN_CHI3, thg=False)
        self.raman = former_response(
            FORMER_N2, MolecularOptions(rotation=False), thg=False
        )
        spectrum = build_raman_kernels(self.raman, self.grid).spectrum(DENSITY)
        # A coarse bin m of the fine grid is bin 2m of the doubled kernel grid, because
        # the two axes have the same total length: n_to dt_fine == n_t dt.
        coarse = np.arange(1, self.grid.n_t // 2)
        best = int(coarse[np.argmin(np.imag(spectrum[2 * coarse]))])
        self.shift = 2.0 * np.pi * best / (self.grid.n_t * self.grid.dt)
        self.im_h = float(np.imag(spectrum[2 * best]))
        self.stokes = int(
            np.argmin(np.abs(self.grid.omega - (self.grid.omega0 - self.shift)))
        )
        self.anti = int(
            np.argmin(np.abs(self.grid.omega - (self.grid.omega0 + self.shift)))
        )
        self.field = np.zeros((1, 1, self.grid.n_omega), dtype=complex)
        self.field[0, 0, 0] = self.grid.n_t * np.sqrt(GAIN_PUMP)
        self.field[0, 0, self.stokes] = self.grid.n_t * GAIN_SEED * np.sqrt(GAIN_PUMP)

    def problem(self, responses, *, beta2=GAIN_BETA2, length=0.1, rtol=1e-9):
        """Assemble the propagation problem with the given responses."""
        mode = SpecifiedMode.from_taylor(
            [0.0, 0.0, beta2], omega_ref=self.grid.omega0, aeff=GAIN_AEFF
        )
        return assemble(
            self.grid,
            mode,
            responses,
            self.field,
            controller=PIController(rtol=rtol, atol=1e-22),
            save_planes=[length],
            dz_max=length / 4,
            dz_init=length / 100,
            density=DENSITY,
            shock=False,
            prefactor_index=1.0,
        )

    def predicted(self, kappa):
        """Return the amplitude growth rate the closed form gives, in 1/m.

        The Stokes component of the drive is ``A_p* a_s / 2`` and the convolution
        multiplies it by ``H(Omega)*``, so the Raman polarisation on the seed is
        ``eps0 rho |A_p|^2 a_s (H(0) + H(Omega)*) / 2`` and the Kerr polarisation on the
        pump is ``(3/4) eps0 chi3 rho |A_p|^2 A_p``. Both go through the same
        normalisation prefactor, so their ratio is free of it and of every unit
        conversion, and only the imaginary part grows an amplitude
        (``docs/guide/media/raman.md``, "The sign that means gain"):

        ``g = kappa (2/3) (-Im H(Omega)) / chi3``,

        with ``kappa`` the pump's own Kerr phase rate in the same configuration.
        """
        return kappa * (2.0 / 3.0) * (-self.im_h) / GAIN_CHI3


def stage_derivative(problem, grid, field):
    """Return ``dA/dz`` at the entrance, through the named passes and nothing else."""
    grids, transform, responses = problem.grids, problem.transform, problem.responses
    modal = passes.to_time(field, grids.crop_index, grids.n_to, kind=grid.kind)
    nodes = passes.synthesise(modal, transform.synthesis, 1.0, transform.n_pol)
    spectrum = None
    leaf = responses.raman_coefficients
    if leaf is not None:
        spectrum = passes.raman_spectrum(
            leaf.undamped,
            [float(table(0.0)) for table in leaf.decay_rates],
            leaf.fraction,
            grids.dt_fine,
        )
    polarisation = passes.nonlinear_polarisation(
        nodes,
        responses,
        grids.t_window_fine,
        kind=grid.kind,
        density=float(responses.density(0.0)),
        preionisation=0.0,
        dt=grids.dt_fine,
        raman_spectrum=spectrum,
    )
    projected = passes.project(polarisation, transform.projection, 1.0, transform.n_pol)
    coarse = passes.to_frequency(projected, grids.crop_index, kind=grid.kind)
    unit = np.ones((1, 1, 1, grid.n_omega), dtype=complex)
    return passes.stage_output(
        coarse, prefactor_at(responses, 0.0), grids.omega_window, unit
    )[0, 0, 0]


@pytest.mark.parametrize(("trange", "bound"), [(3.5e-11, 2.5e-2), (7.0e-11, 1.0e-2)])
def test_the_small_signal_gain_matches_minus_the_imaginary_kernel(trange, bound):
    """The Stokes seed grows at the rate ``-Im H(Omega)`` sets, and the Kerr alone does
    not grow it at all.

    Read off one stage derivative, so that nothing of the step control, the pump's own
    reshaping or the anti-Stokes is in the number. The residual is the drive's missing
    history: the drive is zero-padded, so the convolution has not reached its steady
    value in the first ``tau2`` of the window, and the deficit falls with the window --
    measured 1.6 % on a 70 ps window and 0.41 % on 140 ps, against the 2.5 % and 1 %
    asserted. The line is nitrogen's 69.85 THz vibration, whose 6 ps coherence is what
    sets that scale.
    """
    setup = GainSetup(trange)
    on = stage_derivative(
        setup.problem([setup.kerr, setup.raman]), setup.grid, setup.field
    )
    off = stage_derivative(setup.problem([setup.kerr]), setup.grid, setup.field)
    kappa = -float(np.imag(off[0] / setup.field[0, 0, 0]))
    growth = float(np.real(on[setup.stokes] / setup.field[0, 0, setup.stokes]))
    bare = float(np.real(off[setup.stokes] / setup.field[0, 0, setup.stokes]))
    assert setup.im_h < 0.0
    assert kappa > 0.0
    # The Kerr response is real, so it turns the phase and never the amplitude; what
    # is left is the transform chain's own round-off, measured 1.0e-15 of the Raman
    # rate on the wider window.
    assert abs(bare) < 1e-13 * growth
    assert growth == pytest.approx(setup.predicted(kappa), rel=bound)


def test_the_kernel_spectrum_is_the_lorentzian_of_the_line():
    """The gain the run convolves with is the line's own closed form.

    A single damped oscillator ``K sin(Omega_v t) e^{-t/tau2}`` has
    ``H(Omega) = K Omega_v / ((1/tau2 + i Omega)^2 + Omega_v^2)``, whose imaginary part
    near resonance is the Lorentzian ``-K a / (2 (a^2 + delta^2))`` with ``a = 1/tau2``
    and ``delta`` the detuning; the resonant approximation costs ``a/Omega_v``, which is
    4e-13 for this line. Checking the kernel against it is what stops the gain test
    above from being a statement about the pass alone.
    """
    setup = GainSetup(7.0e-11)
    mode = FORMER_N2.vibrational
    assert mode is not None
    line = vibrational(mode)
    rate = line.dephasing.decay_rate(DENSITY)
    detuning = setup.shift - line.frequency
    lorentzian = -line.amplitude * rate / (2.0 * (rate**2 + detuning**2))
    # Measured 1.00000: the 140 ps window holds the 6 ps coherence to 1e-10 of its peak,
    # so neither the truncation nor the collar's taper is visible at this tolerance.
    assert setup.im_h == pytest.approx(lorentzian, rel=1e-4)
    assert abs(detuning) < rate


@pytest.mark.medium
def test_a_short_propagation_grows_the_stokes_at_the_predicted_rate(backend_class):
    """The same closed form, through the step loop rather than one derivative.

    Twenty millimetres is short enough that the pump is undepleted (the Stokes carries
    1e-6 of its power) and the anti-Stokes the four-wave mixing seeds stays 0.18 % of
    the Stokes, held there by the 3.9e3 1/m phase mismatch of the normal dispersion.
    """
    setup = GainSetup(3.5e-11)
    length = 0.02

    def run(responses):
        problem = setup.problem(responses, length=length, rtol=1e-10)
        backend = backend_class(problem, grid=setup.grid)
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        final = backend.observe().require_field()[0, 0, 0].copy()
        backend.close()
        return final

    on = run([setup.kerr, setup.raman])
    off = run([setup.kerr])
    kappa = -float(np.angle(off[0] / setup.field[0, 0, 0])) / length
    growth = float(np.log(abs(on[setup.stokes]) / abs(off[setup.stokes]))) / length
    assert abs(on[setup.anti]) / abs(on[setup.stokes]) < 5e-3
    # Measured 1.0026 of the prediction; the 3 % asserted covers the derivative test's
    # own 1.6 % on this window and the second-order propagation effects on top of it.
    assert growth == pytest.approx(setup.predicted(kappa), rel=3e-2)


# ------------------------------------------ the soliton self-frequency shift and photon

SSFS_BETA2 = -11.83e-27  # s^2/m, anomalous: the fibre of the soliton ladder
SSFS_GAMMA = 0.11  # 1/(W m)
SSFS_FRACTION = 0.18


def first_moment_of_silica():
    """Return ``integral t h dt / integral h dt`` of the silica shape, in s.

    The quantity Gordon's rate is written in terms of: ``T_R = f_R`` times this, because
    the full response is ``(1 - f_R) delta(t) + f_R h(t)`` and a delta has no first
    moment. It is also minus the slope of ``Im H`` at zero frequency, which is why the
    same number sets both the frequency shift and the low-frequency Raman gain.
    """
    dt = 2e-17
    delays = np.arange(1 << 22) * dt
    shape = response_for("SiO2").impulse(delays, 1.0)
    return float(np.trapezoid(delays * shape, dx=dt) / np.trapezoid(shape, dx=dt))


def test_the_soliton_self_frequency_shift_matches_gordons_rate(
    selected_backend_name,
):
    """A fundamental soliton red-shifts at ``-8 |beta2| T_R / (15 T0^4)``.

    Gordon's rate, from first-order perturbation theory with the delayed response
    linearised about zero frequency, so it holds for a soliton long enough that the
    Raman gain is linear across its bandwidth. Measured against it here at 800 fs, where
    the pulse has covered 1.1 dispersion lengths and the shift is 1.8e-4 of its own
    bandwidth: the centroid is what resolves it, not the spectrum's shape.
    """
    tau0, length = 800e-15, 60.0
    peak = abs(SSFS_BETA2) / (SSFS_GAMMA * tau0**2)
    handler = prop_gnlse(
        SSFS_GAMMA,
        length,
        [0.0, 0.0, SSFS_BETA2],
        lambda0=835e-9,
        wavelength_limits=(700e-9, 1.05e-6),
        trange=16e-12,
        width=tau0,
        peak_power=peak,
        pulseshape="sech",
        shock=False,
        raman=True,
        raman_fraction=SSFS_FRACTION,
        n_save=3,
        rtol=1e-9,
        atol=1e-22,
        backend=selected_backend_name,
    )
    z = handler["z"]
    centroid = handler["stats/omega0"][:, 0, 0]
    rate = float((centroid[-1] - centroid[0]) / (z[-1] - z[0]))
    moment = SSFS_FRACTION * first_moment_of_silica()
    gordon = -8.0 * abs(SSFS_BETA2) * moment / (15.0 * tau0**4)
    # T_R = 2.464 fs for this line set, and the measured rate is 1.0243 of Gordon's.
    # The residual is the T_R approximation's own finite-bandwidth error, which falls
    # with the pulse: measured 19 %, 8.5 % and 2.4 % at 200, 400 and 800 fs. The 3 %
    # asserted is that trend at the width used, not a numerical tolerance.
    assert rate < 0.0
    assert rate == pytest.approx(gordon, rel=3e-2)


def test_a_run_without_raman_does_not_shift_its_centroid(selected_backend_name):
    """The same fibre with Raman off: a soliton keeps its centre frequency."""
    tau0, length = 800e-15, 60.0
    peak = abs(SSFS_BETA2) / (SSFS_GAMMA * tau0**2)
    handler = prop_gnlse(
        SSFS_GAMMA,
        length,
        [0.0, 0.0, SSFS_BETA2],
        lambda0=835e-9,
        wavelength_limits=(700e-9, 1.05e-6),
        trange=16e-12,
        width=tau0,
        peak_power=peak,
        pulseshape="sech",
        shock=False,
        raman=False,
        n_save=3,
        rtol=1e-9,
        atol=1e-22,
        backend=selected_backend_name,
    )
    centroid = handler["stats/omega0"][:, 0, 0]
    # Self-phase modulation broadens the spectrum symmetrically about the carrier, so
    # the centroid does not move: measured 0.5 rad/s of drift, one unit in the last
    # place of a 2.26e15 rad/s centroid, against the 2.3e8 rad/s the same fibre shifts
    # by with Raman on.
    assert abs(centroid[-1] - centroid[0]) < 1e-14 * abs(centroid[0])


@pytest.fixture(scope="module")
def photon_grid():
    """A wide envelope grid, so a third-order soliton stays in band as it broadens."""
    return EnvelopeGrid.from_limits(0.01, 835e-9, (400e-9, 3e-6), 4e-12)


def photon_number_and_energy(handler, grid):
    """Return the photon number and the energy at every saved plane, each over its own
    value at the entrance."""
    power = np.abs(handler["field"][:, 0, 0, :]) ** 2
    inband = grid.omega > 0.0
    photons = (power[:, inband] / grid.omega[inband]).sum(axis=-1)
    energy = power.sum(axis=-1)
    return photons / photons[0], energy / energy[0]


@pytest.mark.parametrize("raman", [True, False], ids=["raman", "kerr"])
def test_the_shock_term_conserves_photon_number_and_raman_costs_energy(
    photon_grid, raman, selected_backend_name
):
    """With the ``omega`` prefactor in place the equation conserves photon number.

    The Raman response moves photons down in frequency and each one carries less energy
    away than it brought, so the pulse energy falls while the photon number
    ``sum |A(omega)|^2 / omega`` does not. That is the invariant the shock term buys:
    with the prefactor frozen at the carrier the equation conserves energy instead
    (``docs/guide/waveguides/gnlse.md``), which is the wrong invariant for a lossless
    fibre with a
    delayed response.
    """
    order = 3
    peak = order**2 * abs(SSFS_BETA2) / (SSFS_GAMMA * 100e-15**2)
    handler = prop_gnlse(
        SSFS_GAMMA,
        0.01,
        [0.0, 0.0, SSFS_BETA2],
        lambda0=835e-9,
        grid=photon_grid,
        width=100e-15,
        peak_power=peak,
        pulseshape="sech",
        shock=True,
        raman=raman,
        raman_fraction=SSFS_FRACTION,
        n_save=5,
        rtol=1e-12,
        atol=1e-26,
        backend=selected_backend_name,
    )
    photons, energy = photon_number_and_energy(handler, photon_grid)
    # Measured with Raman on: the photon number is conserved exactly -- 0.0, not merely
    # to the controller's tolerance -- while the energy falls by 7.2e-6 over a
    # hundredth of a soliton period. With Raman off both hold to 5.6e-16, which is the
    # round-off of the sums themselves.
    assert abs(photons[-1] - 1.0) < 1e-13
    if raman:
        assert energy[-1] - 1.0 < -1e-6
    else:
        assert abs(energy[-1] - 1.0) < 1e-13


# ------------------------------------------------------------------------- the refusals


def test_a_vector_state_with_a_raman_kernel_is_refused_by_name(
    backend_class, soliton_problem, narrow_grid, substitute
):
    """The vector form needs a tensor kernel that no backend can consume.

    The translation layer refuses it first, so the flags have to be declared by hand
    here -- which is exactly the case a backend's own refusal exists for.
    """
    base = soliton_problem(narrow_grid, 1, save_planes=[0.01])
    two = dataclasses.replace(base.transform, _n_pol=2, kind="modal")
    problem = substitute(
        base,
        transform=two,
        flags=dataclasses.replace(base.flags, has_raman=True),
        validate=False,
    )
    with pytest.raises(BackendError, match="tensor-capable"):
        backend_class(problem)


def test_a_raman_run_steps_and_shifts_a_capillary_pulse(backend_class, raman_capillary):
    """A propagation with the convolution in the loop, end to end."""
    problem, _, _ = raman_capillary()
    backend = backend_class(problem, grid=None)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(0.1)
    plane = backend.observe()
    transforms = backend._raman_transforms
    backend.close()
    assert summary.met_tolerance
    assert summary.n_steps > 4
    assert np.all(np.isfinite(plane.field))
    # The kernel is transformed once per plane the stepper visits, never once per
    # column: a stage evaluates it once and shares it between the chunks.
    assert 0 < transforms <= summary.n_steps * 7


def test_the_drive_form_comes_from_the_leaf_and_not_from_the_kerr_flag(
    raman_capillary,
):
    """The answer is carried by the leaf, and it is the one the compiler enforces."""
    problem, _, _ = raman_capillary()
    responses = problem.responses
    leaf = responses.raman_coefficients
    kerr = responses.kerr_coefficients
    assert leaf is not None
    assert kerr is not None
    assert passes._raman_thg(responses) is leaf.thg
    # On a field grid the compiler refuses a Raman response whose thg disagrees with the
    # Kerr response's, which is what made the old inference sound; the two still agree.
    assert leaf.thg == (not kerr.analytic_signal)


def test_the_raman_stage_refuses_a_problem_with_no_leaf(raman_capillary):
    problem, _, _ = raman_capillary()
    stripped = dataclasses.replace(problem.responses, raman_terms=())
    with pytest.raises(BackendError, match=r"no responses\.raman_coefficients"):
        passes._raman_thg(stripped)
