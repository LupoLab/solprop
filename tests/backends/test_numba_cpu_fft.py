# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plans, the wisdom and the four thread pools.

The plans are asserted against :mod:`numpy.fft` rather than against themselves, because
the whole point of replacing pocketfft with FFTW is that the two compute the same
transform; and the analytic-signal plan of the ``field_analytic`` family is asserted
against :func:`scipy.signal.hilbert`, because that family's saving rests on the claim
that the two are the same thing and the reference pays for the second one.
"""

import os

import numpy as np
import pytest
from numba import get_num_threads
from numba.core import config as numba_config
from scipy.signal import hilbert

from solprop import _process
from solprop.backends.numba_cpu import fft
from solprop.exceptions import BackendError

N_TO = 64
N_T = 32
N_COLUMNS = 3


@pytest.fixture
def spectrum():
    """A one-sided spectrum whose inverse is a real signal with structure."""
    rng = np.random.default_rng(4)
    values = rng.normal(size=(N_COLUMNS, N_TO // 2 + 1)) + 1j * rng.normal(
        size=(N_COLUMNS, N_TO // 2 + 1)
    )
    # A real signal's spectrum has a real bin at zero and at Nyquist.
    values[:, 0] = values[:, 0].real
    values[:, -1] = values[:, -1].real
    return values


# --- the pools -------------------------------------------------------------------


def test_a_small_problem_runs_serially_and_a_large_one_does_not():
    """The regime comes from the element count of section 3.4, not the machine."""
    small = fft.choose_pools(1000, 1000, 1000)
    assert small.regime == "small"
    assert (small.numba, small.fftw, small.blas) == (1, 1, 1)
    big = 10 * fft.REGIME_ELEMENTS
    large = fft.choose_pools(big, big, big)
    assert large.regime == "large"
    assert large.numba > 1 or (os.cpu_count() or 1) == 1
    # The transforms and the GEMMs get a *small* measured pool, not the machine: four
    # and two, from a pool sweep on an Apple-silicon laptop and an x86-64 server.
    assert large.fftw == min(fft.FFTW_MAX_THREADS, os.cpu_count() or 1)
    assert large.blas == min(fft.BLAS_MAX_THREADS, os.cpu_count() or 1)


def test_a_field_volume_transform_takes_every_allowed_thread():
    """A free-space evaluation's large transforms are not held to the four-thread pool;
    a modal transform of the same size is."""
    below = fft.choose_pools(1, 1, fft.FFTW_WIDE_ELEMENTS - 1, spatial=True)
    wide = fft.choose_pools(1, 1, fft.FFTW_WIDE_ELEMENTS, spatial=True)
    modal = fft.choose_pools(1, 1, fft.FFTW_WIDE_ELEMENTS)
    capped = min(fft.FFTW_MAX_THREADS, fft._thread_budget().threads)
    assert below.fftw == capped
    assert wide.fftw == fft._thread_budget().threads
    assert modal.fftw == capped


def test_a_large_problem_with_small_operands_still_keeps_its_pools_to_one():
    """The GEMM and the transform are sized separately from the whole problem.

    A problem well into the large regime whose operands are nonetheless below their own
    thresholds keeps both those pools at one, because below them a second thread is a
    straight loss on both machines the sweep was run on.
    """
    pools = fft.choose_pools(10 * fft.REGIME_ELEMENTS, 33 * 1024, 4 * 1024)
    assert pools.regime == "large"
    assert pools.blas == 1
    assert pools.fftw == 1
    assert pools.numba > 1 or (os.cpu_count() or 1) == 1


def test_the_ctc_and_m2_cases_get_the_pools_the_sweep_measured():
    """The two cases the thresholds were set from.

    The CtC full case is the one threading is for: its modal pair is 393 216 elements,
    which a 1e6 threshold would leave serial, and serial the run took 5710 s at 132 % of
    one core. It gets four FFTW threads, where the sweep measured 2.64x on an
    Apple-silicon laptop and 2.33x on an x86-64 server against one, and the case itself
    went 5710 s to 2551 s.

    The M2 case is the other half of the rule and pulls the other way: its modal pair is
    small enough that threading it costs, so the threshold has to exclude it. A rule set
    from the CtC case alone would have made the M2 case 15 % slower.
    """
    ctc = fft.choose_pools(19_267_584, 41 * 65536, 6 * 65536)
    assert ctc.fftw == min(4, os.cpu_count() or 1)
    assert ctc.blas == min(2, os.cpu_count() or 1)
    # The M2 modal pair is 32 768 elements, which is where the sweep breaks even and
    # where a real run loses: threaded, the M2 case end to end goes 11.5-12.9 s to
    # 13.4-14.7 s. So it sits *below* the threshold and keeps one thread.
    m2 = fft.choose_pools(1_081_344, 33 * 8192, 4 * 8192)
    assert fft.FFTW_THREAD_ELEMENTS > 4 * 8192
    assert m2.fftw == 1
    # And a soliton-sized problem is left alone entirely.
    small = fft.choose_pools(4096, 4096, 4096)
    assert (small.regime, small.numba, small.fftw, small.blas) == ("small", 1, 1, 1)


def test_the_overrides_win_and_are_capped_by_the_machine():
    """A run may ask for its own pools, but not for more threads than it may run."""
    pools = fft.choose_pools(
        1000, 1000, 1000, threads=2, fftw_threads=3, blas_threads=1
    )
    assert (pools.numba, pools.fftw, pools.blas) == (
        min(2, os.cpu_count() or 1),
        min(3, os.cpu_count() or 1),
        1,
    )
    huge = fft.choose_pools(1000, 1000, 1000, threads=10_000)
    assert huge.numba <= (os.cpu_count() or 1)


def test_the_kernel_pool_is_capped_by_numbas_own_pool(monkeypatch):
    """``NUMBA_NUM_THREADS`` below the CPUs caps the kernels, and only the kernels.

    A Slurm node granting 64 CPUs to a worker that exported ``NUMBA_NUM_THREADS=8``
    asked Numba for 64 threads in the large regime, and ``set_num_threads`` refused.
    """
    monkeypatch.setattr(fft, "_thread_budget", lambda: fft.ThreadBudget(64, "affinity"))
    monkeypatch.setattr(numba_config, "NUMBA_NUM_THREADS", 8)
    big = 10 * fft.REGIME_ELEMENTS
    large = fft.choose_pools(big, big, fft.FFTW_WIDE_ELEMENTS, spatial=True)
    assert (large.regime, large.numba, large.fftw) == ("large", 8, 64)
    explicit = fft.choose_pools(
        1000, 1000, 1000, threads=32, fftw_threads=32, blas_threads=32
    )
    assert (explicit.numba, explicit.fftw, explicit.blas) == (8, 32, 32)
    assert fft.choose_pools(1000, 1000, 1000, threads=4).numba == 4


def test_solprop_threads_caps_every_pool(monkeypatch):
    """``SOLPROP_THREADS`` holds every pool, chosen or explicit, within the budget."""
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(64)))
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.setattr(numba_config, "NUMBA_NUM_THREADS", 64)
    monkeypatch.setenv("SOLPROP_THREADS", "3")
    assert fft._thread_budget().threads == 3
    big = 10 * fft.REGIME_ELEMENTS
    chosen = fft.choose_pools(big, big, fft.FFTW_WIDE_ELEMENTS, spatial=True)
    assert (chosen.numba, chosen.fftw) == (3, 3)
    assert chosen.as_metadata()["thread_budget_source"] == "SOLPROP_THREADS"
    explicit = fft.choose_pools(
        1000, 1000, 1000, threads=8, fftw_threads=8, blas_threads=8
    )
    assert (explicit.numba, explicit.fftw, explicit.blas) == (3, 3, 3)
    monkeypatch.setenv("SOLPROP_THREADS", "128")
    assert (fft._thread_budget().threads, fft._thread_budget().source) == (
        64,
        "affinity",
    )


def test_oversubscribing_lets_explicit_counts_pass_the_cpus(monkeypatch):
    """``SOLPROP_OVERSUBSCRIBE=1`` honours an explicit count above the CPUs."""
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(2)))
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.setattr(numba_config, "NUMBA_NUM_THREADS", 8)
    monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    monkeypatch.setenv("SOLPROP_OVERSUBSCRIBE", "1")
    explicit = fft.choose_pools(
        1000, 1000, 1000, threads=16, fftw_threads=4, blas_threads=4
    )
    assert (explicit.numba, explicit.fftw, explicit.blas) == (8, 4, 4)
    big = 10 * fft.REGIME_ELEMENTS
    assert fft.choose_pools(big, big, big).numba == 2


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_solprop_threads_must_be_a_positive_count(monkeypatch, value):
    monkeypatch.setenv("SOLPROP_THREADS", value)
    with pytest.raises(BackendError, match="SOLPROP_THREADS"):
        fft.choose_pools(1000, 1000, 1000)


@pytest.mark.parametrize(
    "offset, expected", [(-1, "small"), (0, "small"), (1, "large")]
)
def test_the_modal_cutoff_changes_only_the_kernel_regime(offset, expected, monkeypatch):
    """The cutoff changes node dispatch without changing either library pool."""
    monkeypatch.setattr(fft, "_thread_budget", lambda: fft.ThreadBudget(18, "affinity"))
    monkeypatch.setattr(numba_config, "NUMBA_NUM_THREADS", 18)
    size = fft.MODAL_REGIME_ELEMENTS + offset
    previous = fft.choose_pools(size, 33 * 8192, 2 * 8192)
    candidate = fft.choose_pools(
        size, 33 * 8192, 2 * 8192, regime_elements=fft.MODAL_REGIME_ELEMENTS
    )
    assert previous.regime == "small"
    assert candidate.regime == expected
    assert candidate.numba == (18 if expected == "large" else 1)
    assert (candidate.fftw, candidate.blas) == (previous.fftw, previous.blas)


@pytest.mark.parametrize("forced", ["small", "large"])
def test_the_modal_cutoff_preserves_explicit_regime_and_thread_overrides(forced):
    """An explicit serial regime remains serial above the modal cutoff."""
    pools = fft.choose_pools(
        2 * 33 * 8192,
        33 * 8192,
        2 * 8192,
        regime=forced,
        threads=1,
        regime_elements=fft.MODAL_REGIME_ELEMENTS,
    )
    assert pools.regime == forced
    assert pools.numba == 1


@pytest.mark.parametrize("cutoff", [0, -1])
def test_a_nonpositive_regime_cutoff_is_refused(cutoff):
    """An invalid cutoff is refused even beside an explicit regime override."""
    with pytest.raises(BackendError, match="regime_elements must be a positive"):
        fft.choose_pools(1000, 1000, 1000, regime="small", regime_elements=cutoff)


@pytest.mark.parametrize("name", ["threads", "fftw_threads", "blas_threads"])
def test_a_pool_of_no_threads_is_refused(name):
    """Zero threads is a mistake, not a request to pick for the caller."""
    # Named rather than unpacked: `choose_pools` also takes a `regime` keyword, and a
    # dict splat leaves a type checker unable to say which parameter the zero lands on.
    pools: dict[str, int | None] = dict.fromkeys(
        ("threads", "fftw_threads", "blas_threads")
    )
    pools[name] = 0
    with pytest.raises(BackendError, match=name):
        fft.choose_pools(
            1000,
            1000,
            1000,
            threads=pools["threads"],
            fftw_threads=pools["fftw_threads"],
            blas_threads=pools["blas_threads"],
        )


class FakeLimiter:
    """A threadpoolctl limit that records whether it was entered and released."""

    def __init__(self):
        self.entered = False
        self.released = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exception):
        self.released = True
        return False


class FakeSelection:
    """The result of selecting the BLAS libraries out of a controller's walk."""

    def __init__(self, controllers, limiter):
        self.lib_controllers = controllers
        self._limiter = limiter
        self.limits = None

    def limit(self, *, limits):
        """Record the limit asked for and hand back the fake limiter."""
        self.limits = limits
        return self._limiter


class FakeController:
    """A stand-in for threadpoolctl's library walk, with a chosen answer."""

    def __init__(self, controllers, limiter=None):
        self.selection = FakeSelection(controllers, limiter or FakeLimiter())

    def select(self, **_):
        """Return the selection this controller was built with."""
        return self.selection


class FakeLibrary:
    """One library the walk found."""

    def __init__(self, internal_api):
        self.internal_api = internal_api


def test_a_controllable_blas_is_limited_and_released():
    """Where a library exposes its pool, the backend takes it and gives it back.

    Which branch a machine takes is not a choice: numpy links Accelerate on macOS, which
    has no runtime thread count at all, and OpenBLAS on Linux x86-64, which has. The
    walk is injected so that both are exercised wherever the suite runs; that
    threadpoolctl finds OpenBLAS is threadpoolctl's own tested behaviour, not this
    package's.
    """
    limiter = FakeLimiter()
    controller = FakeController(
        [FakeLibrary("openblas"), FakeLibrary("openblas")], limiter
    )
    pools = fft.choose_pools(1000, 1000, 1000, threads=1, blas_threads=2)
    reached, live = fft.apply_pools(pools, controller=controller)
    assert reached.blas_mechanism == "openblas"
    assert reached.blas == 2
    assert controller.selection.limits == 2
    assert limiter.entered and not limiter.released
    fft.release_pools(live)
    assert limiter.released


def test_a_blas_that_cannot_be_set_is_recorded_as_such():
    """An empty walk is not an error and is not silently reported as a limit."""
    reached, live = fft.apply_pools(
        fft.choose_pools(1000, 1000, 1000, threads=1), controller=FakeController([])
    )
    assert reached.blas_mechanism == "uncontrolled"
    assert live is None
    fft.release_pools(live)


def test_the_pools_are_applied_and_given_back():
    """What was reached is recorded, including a BLAS that cannot be set."""
    before = get_num_threads()
    try:
        reached, limiter = fft.apply_pools(
            fft.choose_pools(1000, 1000, 1000, threads=1)
        )
        assert reached.numba == 1
        assert get_num_threads() == 1
        # Accelerate exposes no runtime knob, OpenBLAS and MKL do; both are honest
        # answers and the metadata says which one this machine gave.
        assert reached.blas_mechanism in {"uncontrolled", "openblas", "mkl", "blis"}
        assert reached.as_metadata()["blas_mechanism"] == reached.blas_mechanism
        fft.release_pools(limiter)
        fft.release_pools(None)
    finally:
        fft.apply_pools(fft.choose_pools(1000, 1000, 1000, threads=before))


# --- the plans -------------------------------------------------------------------


def test_the_field_plans_are_the_reference_transform_pair(spectrum):
    """``field_real``: a complex-to-real inverse and a real-to-complex forward."""
    plans = fft.build_plans("field_real", N_COLUMNS, N_TO, N_T, flag="estimate")
    assert plans.modal_time.dtype == np.float64
    plans.modal_spectrum[:] = spectrum
    plans.to_time.execute()
    # FFTW normalises nothing; the 1/N of the inverse lives in the contract's scales.
    assert plans.modal_time == pytest.approx(N_TO * np.fft.irfft(spectrum, n=N_TO))
    plans.projected[:] = plans.modal_time
    plans.to_frequency.execute()
    assert plans.projected_spectrum == pytest.approx(np.fft.rfft(plans.projected))


def test_the_envelope_plans_are_complex_both_ways(spectrum):
    """``envelope``: the samples are complex throughout, so both plans are c2c."""
    full = np.zeros((N_COLUMNS, N_TO), dtype=complex)
    full[:, : N_TO // 2 + 1] = spectrum
    plans = fft.build_plans("envelope", N_COLUMNS, N_TO, N_T, flag="estimate")
    assert plans.modal_time.dtype == np.complex128
    assert plans.projected.dtype == np.complex128
    plans.modal_spectrum[:] = full
    plans.to_time.execute()
    assert plans.modal_time == pytest.approx(N_TO * np.fft.ifft(full))
    plans.projected[:] = plans.modal_time
    plans.to_frequency.execute()
    assert plans.projected_spectrum == pytest.approx(np.fft.fft(plans.projected))


def test_the_analytic_plan_is_the_hilbert_transform_the_reference_pays_for(spectrum):
    """``field_analytic``: one complex inverse gives what a c2r plus a Hilbert gives.

    The saving this family exists for. The reference forms the real field with an
    inverse transform and then calls :func:`solprop.fourier.analytic_signal`, which is
    a whole complex pair, once per node chunk; the analytic signal is the complex
    inverse of the *same* one-sided spectrum with its positive bins doubled, so the
    forward pass costs one transform rather than three and the difference is round-off.
    """
    plans = fft.build_plans("field_analytic", N_COLUMNS, N_TO, N_T, flag="estimate")
    assert plans.modal_spectrum.shape == (N_COLUMNS, N_TO)
    assert plans.modal_time.dtype == np.complex128
    # The polarisation is real on a field grid whatever the column works on, so the
    # forward plan is the same real-to-complex one the other field family uses.
    assert plans.projected.dtype == np.float64
    doubled = np.zeros((N_COLUMNS, N_TO), dtype=complex)
    doubled[:, 0] = spectrum[:, 0]
    doubled[:, 1 : N_TO // 2] = 2.0 * spectrum[:, 1 : N_TO // 2]
    doubled[:, N_TO // 2] = spectrum[:, N_TO // 2]
    plans.modal_spectrum[:] = doubled
    plans.to_time.execute()
    wanted = hilbert(np.fft.irfft(spectrum, n=N_TO), axis=-1)
    assert plans.modal_time / N_TO == pytest.approx(wanted, rel=0, abs=1e-13)


def test_the_raman_pair_lives_on_the_doubled_grid():
    """One pair for a whole node chunk, and the padding half stays zero."""
    plans = fft.build_plans(
        "field_real", N_COLUMNS, N_TO, N_T, flag="estimate", chunk_columns=2
    )
    assert plans.drive.shape == (2, 2 * N_TO)
    assert plans.drive_spectrum.shape == (2, N_TO + 1)
    assert np.array_equal(plans.drive, np.zeros_like(plans.drive))
    rng = np.random.default_rng(9)
    plans.drive[:, :N_TO] = rng.normal(size=(2, N_TO))
    plans.raman_to_frequency.execute()
    assert plans.drive_spectrum == pytest.approx(np.fft.rfft(plans.drive))
    # A real-to-complex plan does not destroy its input, which is what lets the
    # padding half be written once at construction and never again.
    assert np.array_equal(plans.drive[:, N_TO:], np.zeros((2, N_TO)))


@pytest.mark.parametrize("raman_threads", [None, 2])
def test_raman_plans_own_their_threads_and_keep_raw_normalisation(
    raman_threads, monkeypatch, record_property
):
    """Changing the Raman pool leaves every other plan and raw inverse scale alone."""
    constructor_threads = []
    original = fft._plan

    def tracked_plan(source, target, *, forward, flag, threads):
        constructor_threads.append(threads)
        return original(source, target, forward=forward, flag=flag, threads=threads)

    monkeypatch.setattr(fft, "_plan", tracked_plan)
    plans = fft.build_plans(
        "field_real",
        N_COLUMNS,
        N_TO,
        N_T,
        flag="estimate",
        threads=1,
        raman_threads=raman_threads,
        chunk_columns=2,
        absorber=True,
        embedded=True,
    )
    assert constructor_threads == [1] * 5 + [raman_threads or 1] * 2
    assert plans.drive.shape == (2, 2 * N_TO)
    plans.drive[:] = 0.0
    plans.drive[:, :N_TO] = np.random.default_rng(91).normal(size=(2, N_TO))
    expected = plans.drive.copy()
    # A forward/inverse radix transform pair has O(log2 N) rounding depth. The
    # existing 16-epsilon FFT allowance bounds the two paths before comparison.
    bound = 16 * np.finfo(float).eps * np.ceil(np.log2(2 * N_TO))
    plans.raman_to_frequency.execute()
    plans.raman_to_time.execute()
    deviation = float(np.max(np.abs(plans.convolved / (2 * N_TO) - expected)))
    deviation /= float(np.max(np.abs(expected)))
    assert deviation <= bound
    # Structural: the out-of-place pair reads the drive and never writes it.
    assert np.array_equal(plans.drive, expected)
    record_property(
        "raman_raw_inverse", {"deviation": deviation, "bound": float(bound)}
    )


@pytest.mark.parametrize("threads", [0, -1])
def test_nonpositive_raman_plan_threads_are_refused(threads):
    """An invalid explicit Raman count is rejected even when no pair is needed."""
    with pytest.raises(BackendError, match="raman_threads must be a positive"):
        fft.build_plans("field_real", N_COLUMNS, N_TO, N_T, raman_threads=threads)


@pytest.mark.parametrize("tail", [1, 3])
@pytest.mark.parametrize("tail_threads", [None, 2])
def test_raman_tail_plans_share_buffers_and_preserve_inactive_rows(
    tail, tail_threads, monkeypatch, record_property
):
    """Full and partial batches reuse owned storage with raw inverse scaling."""
    constructor_threads = []
    original = fft._plan

    def tracked_plan(source, target, *, forward, flag, threads):
        constructor_threads.append(threads)
        return original(source, target, forward=forward, flag=flag, threads=threads)

    monkeypatch.setattr(fft, "_plan", tracked_plan)
    plans = fft.build_plans(
        "field_real",
        N_COLUMNS,
        N_TO,
        N_T,
        flag="estimate",
        chunk_columns=4,
        tail_columns=tail,
        raman_tail_threads=tail_threads,
    )
    assert constructor_threads == [1] * 4 + [tail_threads or 1] * 2
    assert np.shares_memory(plans.raman_tail_to_frequency.input_array, plans.drive)
    assert np.shares_memory(
        plans.raman_tail_to_frequency.output_array, plans.drive_spectrum
    )
    assert np.shares_memory(plans.raman_tail_to_time.output_array, plans.convolved)
    assert plans.raman_tail_to_frequency.input_array.shape == (tail, 2 * N_TO)
    ledger = plans.ledger()
    assert (
        ledger["node_raman_drive"] == ledger["node_raman_convolved"] == 4 * 2 * N_TO * 8
    )
    assert ledger["node_raman_spectrum"] == 4 * (N_TO + 1) * 16
    assert np.array_equal(plans.drive, np.zeros_like(plans.drive))
    rng = np.random.default_rng(382)
    # The existing FFT pair allowance, derived before any transform comparison.
    bound = 16 * np.finfo(float).eps * np.ceil(np.log2(2 * N_TO))
    deviations = []
    for count in (4, tail, 4, tail):
        plans.drive[:count, :N_TO] = rng.normal(size=(count, N_TO))
        expected = plans.drive[:count].copy()
        inactive = plans.convolved[count:].copy()
        if count == 4:
            plans.raman_to_frequency.execute()
            plans.raman_to_time.execute()
        else:
            plans.raman_tail_to_frequency.execute()
            plans.raman_tail_to_time.execute()
        deviation = float(
            np.max(np.abs(plans.convolved[:count] / (2 * N_TO) - expected))
        )
        deviation /= float(np.max(np.abs(expected)))
        assert deviation <= bound
        assert np.array_equal(plans.convolved[count:], inactive)
        assert np.array_equal(plans.drive[:, N_TO:], np.zeros((4, N_TO)))
        deviations.append(deviation)
    record_property(
        "raman_tail_roundtrip",
        {"deviation": max(deviations), "bound": float(bound)},
    )


def test_raman_padding_is_established_after_destructive_planning(monkeypatch):
    """A planning implementation may scribble over every shared prefix view."""
    original = fft._plan

    def destructive_plan(source, target, *, forward, flag, threads):
        plan = original(source, target, forward=forward, flag=flag, threads=threads)
        source[:] = 19.0
        return plan

    monkeypatch.setattr(fft, "_plan", destructive_plan)
    plans = fft.build_plans(
        "field_real",
        N_COLUMNS,
        N_TO,
        N_T,
        flag="estimate",
        chunk_columns=4,
        tail_columns=1,
    )
    assert np.array_equal(plans.drive, np.zeros_like(plans.drive))


@pytest.mark.parametrize("chunk,tail", [(0, 1), (4, -1), (4, 4), (4, 5)])
def test_invalid_raman_tail_width_is_refused(chunk, tail):
    """A prefix pair must be a proper nonempty remainder of an owned full batch."""
    with pytest.raises(BackendError, match="tail_columns"):
        fft.build_plans(
            "field_real",
            N_COLUMNS,
            N_TO,
            N_T,
            chunk_columns=chunk,
            tail_columns=tail,
        )


@pytest.mark.parametrize("threads", [0, -1])
def test_nonpositive_raman_tail_threads_are_refused(threads):
    """Reject an invalid explicit tail pool even when no tail pair is needed."""
    with pytest.raises(BackendError, match="raman_tail_threads must be a positive"):
        fft.build_plans("field_real", N_COLUMNS, N_TO, N_T, raman_tail_threads=threads)


def test_the_absorber_pair_is_planned_only_when_it_is_wanted():
    """A problem without a temporal absorber allocates nothing for one."""
    without = fft.build_plans("field_real", N_COLUMNS, N_TO, N_T, flag="estimate")
    assert without.absorber_to_time is None
    # Named even when empty, which is the reference's convention: a budget test reads
    # the same rows off either backend rather than branching on what it found.
    assert without.ledger()["absorber_time"] == 0
    with_one = fft.build_plans(
        "field_real", N_COLUMNS, N_TO, N_T, flag="estimate", absorber=True
    )
    assert with_one.coarse_time.shape == (N_COLUMNS, N_T)
    assert with_one.coarse_spectrum.shape == (N_COLUMNS, N_T // 2 + 1)
    with_one.coarse_time[:] = np.arange(N_COLUMNS * N_T).reshape(N_COLUMNS, N_T)
    expected = np.fft.rfft(with_one.coarse_time)
    with_one.absorber_to_frequency.execute()
    assert with_one.coarse_spectrum == pytest.approx(expected)


@pytest.mark.parametrize(
    ("family", "dtype", "n_omega_fine"),
    [
        ("field_real", np.float64, N_TO // 2 + 1),
        ("field_analytic", np.float64, N_TO // 2 + 1),
        ("envelope", np.complex128, N_TO),
    ],
)
def test_the_in_stage_absorber_has_its_own_planned_forward_transform(
    family, dtype, n_omega_fine
):
    """The term is real on a field grid, even an analytic one, and has its own output.

    Its spectrum cannot share the polarisation's, which the same stage's finish reads
    after the term is formed.
    """
    without = fft.build_plans(family, N_COLUMNS, N_TO, N_T, flag="estimate")
    assert without.in_stage_to_frequency is None
    assert without.ledger()["in_stage_absorber_spectrum"] == 0
    plans = fft.build_plans(
        family, N_COLUMNS, N_TO, N_T, flag="estimate", in_stage_absorber=True
    )
    assert plans.in_stage_time.dtype == dtype
    assert plans.in_stage_spectrum.shape == (N_COLUMNS, n_omega_fine)
    assert plans.in_stage_spectrum is not plans.projected_spectrum
    rows = np.arange(N_COLUMNS * N_TO, dtype=float).reshape(N_COLUMNS, N_TO)
    plans.in_stage_time[:] = rows * (1 - 1j) if dtype == np.complex128 else rows
    expected = (
        np.fft.fft(plans.in_stage_time)
        if dtype == np.complex128
        else np.fft.rfft(plans.in_stage_time)
    )
    plans.in_stage_to_frequency.execute()
    # Two transforms of the same integers, each within a few ulps of the largest bin.
    np.testing.assert_allclose(
        plans.in_stage_spectrum,
        expected,
        rtol=0,
        atol=64 * np.finfo(float).eps * float(np.max(np.abs(expected))),
    )


def test_an_envelope_absorber_works_on_the_full_coarse_grid():
    """Its buffers are complex and full length, not one-sided."""
    plans = fft.build_plans(
        "envelope", N_COLUMNS, N_TO, N_T, flag="estimate", absorber=True
    )
    assert plans.coarse_spectrum.shape == (N_COLUMNS, N_T)
    assert plans.coarse_time.dtype == np.complex128
    plans.coarse_spectrum[:] = 1.0 + 0.0j
    plans.absorber_to_time.execute()
    assert plans.coarse_time == pytest.approx(N_T * np.fft.ifft(plans.coarse_spectrum))


def test_the_ledger_names_every_buffer_the_plans_hold():
    """The plan work areas are in the ledger, as the memory budget test requires."""
    plans = fft.build_plans(
        "field_real",
        N_COLUMNS,
        N_TO,
        N_T,
        flag="estimate",
        chunk_columns=2,
        absorber=True,
    )
    ledger = plans.ledger()
    assert set(ledger) == {
        "fine_spectrum",
        "modal_time",
        "projected",
        "projected_spectrum",
        "absorber_spectrum",
        "absorber_time",
        "in_stage_absorber_time",
        "in_stage_absorber_spectrum",
        "projected_coarse",
        "projected_coarse_spectrum",
        "node_raman_drive",
        "node_raman_spectrum",
        "node_raman_convolved",
        "fft_plan_heap_growth",
    }
    assert isinstance(ledger["fft_plan_heap_growth"], int)
    assert ledger["fft_plan_heap_growth"] >= 0
    assert ledger["projected_coarse"] == 0
    assert ledger["modal_time"] == N_COLUMNS * N_TO * 8
    assert ledger["node_raman_drive"] == 2 * 2 * N_TO * 8
    assert plans.planning_seconds >= 0.0


def test_an_unknown_planning_flag_is_refused():
    """The three efforts are named; a typo is not silently the default."""
    with pytest.raises(BackendError, match="planning flag"):
        fft.build_plans("field_real", 1, N_TO, N_T, flag="fastest")


# --- wisdom ----------------------------------------------------------------------


def test_wisdom_round_trips_through_a_named_file(tmp_path):
    """Explicit in and explicit out, and a digest that changes with the content."""
    path = tmp_path / "nested" / "wisdom.bin"
    fft.build_plans("field_real", 1, N_TO, N_T, flag="estimate")
    assert fft.export_wisdom(path)
    assert path.is_file()
    digest = fft.wisdom_digest()
    assert len(digest) == 16
    assert fft.import_wisdom(path)


def test_wisdom_that_is_not_named_or_not_there_is_not_an_error(tmp_path, monkeypatch):
    """Planning starts from nothing rather than refusing to start."""
    monkeypatch.delenv("SOLPROP_FFTW_WISDOM", raising=False)
    assert not fft.import_wisdom()
    assert not fft.export_wisdom()
    assert not fft.import_wisdom(tmp_path / "absent.bin")
    monkeypatch.setenv("SOLPROP_FFTW_WISDOM", str(tmp_path / "from-the-environment"))
    assert fft.export_wisdom()
    assert fft.import_wisdom()


def test_a_file_that_is_not_wisdom_is_reported_and_ignored(tmp_path, caplog):
    """A truncated or foreign file must not be fed to FFTW as if it were wisdom."""
    path = tmp_path / "rubbish.bin"
    path.write_bytes(b"not wisdom at all")
    with caplog.at_level("WARNING", logger="solprop.backends.numba_cpu.fft"):
        assert not fft.import_wisdom(path)
    assert "not FFTW wisdom" in caplog.text


@pytest.mark.parametrize("forced", ["small", "large"])
def test_the_regime_can_be_forced_either_way(forced):
    """A measurement may need a small problem stepped as a large one, or the reverse.

    The regime decides whether the node loop is a ``prange`` and whether the pools are
    allowed their threads, so a problem below :data:`fft.REGIME_ELEMENTS` cannot be
    measured against thread count at all without this -- its wall time would be flat in
    the thread count and the flatness would mean nothing. An override for a measurement
    and not a tuning knob: a run that does not ask for one still gets the threshold's
    answer.
    """
    small, large = 1000, 10 * fft.REGIME_ELEMENTS
    assert fft.choose_pools(small, large, large, regime=forced).regime == forced
    assert fft.choose_pools(large, large, large, regime=forced).regime == forced
    assert fft.choose_pools(small, large, large).regime == "small"
    assert fft.choose_pools(large, large, large).regime == "large"
    # Forced large, a small problem is allowed the pools its own size would refuse.
    forced_large = fft.choose_pools(small, large, large, regime="large", threads=4)
    assert forced_large.numba == min(4, os.cpu_count() or 1)
    assert forced_large.fftw > 1 or (os.cpu_count() or 1) == 1


def test_an_unknown_regime_is_refused():
    """Two names, and a typo is not silently the threshold's answer."""
    with pytest.raises(BackendError, match="regime must be"):
        fft.choose_pools(1000, 1000, 1000, regime="big")
