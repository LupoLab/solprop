# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reduced policies' arithmetic probes on the native backend (SEP 36 section 6.3).

Each runs on whatever build is installed: a CPU build here, a CUDA or HIP build in the
device lane (``test_native_reduced_gpu.py`` collects them there). The GEMM probe's
families are measured at construction; a reduced math mode on the vendor handle (TF32,
xf32), selected through the test hook, must be refused. The binary32 transforms are
characterised against the established bound at binary32 for each kind and length the
routes use.
"""

import math
from collections.abc import Iterator
from contextlib import ExitStack

import numpy as np
import pytest

from bounds import EPS_32, fft_bound
from precision_cases import capillary, envelope_plasma, gnlse, plasma_beam, spatial_kerr
from solprop.backends.registry import native_precisions, select
from solprop.driver import Simulation
from solprop.exceptions import BackendError

pytestmark = pytest.mark.skipif(
    not {"single", "mixed"} <= set(native_precisions()),
    reason="no native build runs the reduced precisions",
)

GEMM_PROBE_BOUND = 2.0**-17


def probe_error(simulation: Simulation) -> float:
    """Return a native state's construction-time GEMM probe error."""
    problem = simulation.translate()
    with ExitStack() as cleanup:
        backend = select("native", problem, grid=simulation.grid)
        cleanup.callback(backend.close)
        error = backend.metadata()["gemm_probe_error"]
    assert isinstance(error, float)
    return error


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: capillary(p, modes=4), id="multimode"),
        pytest.param(lambda p: spatial_kerr("radial", p), id="radial"),
        pytest.param(plasma_beam, id="radial-filament"),
    ],
)
def test_the_gemm_probe_measures_binary32_products(build, precision):
    """Each family's product is binary32's: nonzero rounding, within 2^-17."""
    error = probe_error(build(precision))
    assert 0.0 < error <= GEMM_PROBE_BOUND


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(gnlse, id="fibre"),
        pytest.param(lambda p: spatial_kerr("cartesian", p), id="cartesian"),
    ],
)
def test_routes_without_products_probe_nothing(build):
    """A single node's folded transform and free space issue no library product."""
    assert probe_error(build("single")) == 0.0


@pytest.fixture
def reduced_math_mode() -> Iterator[None]:
    """Select TF32 (CUDA) or xf32 (HIP) on the vendor handle for one test."""
    import solprop_native._core as core  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    if not core.blas_math_mode_settable():
        pytest.skip("this build issues no products on a vendor handle")
    core.set_blas_math_mode_selection("reduced")
    try:
        yield
    finally:
        core.set_blas_math_mode_selection("pedantic")


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_a_reduced_math_mode_is_refused(reduced_math_mode, precision):
    """TF32 (or xf32) on the handle: the probe sees it and the run is refused."""
    simulation = capillary(precision, modes=4)
    with pytest.raises(BackendError, match=r"beyond 2\^-17.*NVIDIA_TF32_OVERRIDE=0"):
        select("native", simulation.translate(), grid=simulation.grid)


def route_lengths() -> list[tuple[bool, int]]:
    """The binary32 transforms' kinds and lengths the routes' test cases use.

    A field grid's transforms are real-to-complex, an envelope grid's complex; each
    at the fine and the coarse length, and the Cartesian grid's transverse axes.
    """
    lengths: set[tuple[bool, int]] = {(False, 16)}
    for build in (
        gnlse,
        capillary,
        lambda p: capillary(p, envelope=True, plasma=False),
        envelope_plasma,
        lambda p: spatial_kerr("cartesian", p),
    ):
        grid = build("single").grid
        real = type(grid).__name__ == "FieldGrid"
        for n in {grid.n_t, grid.n_to}:
            lengths.add((real, n))
    return sorted(lengths)


@pytest.mark.parametrize(
    ("real", "n"),
    [
        *[(kind, 2**k) for kind in (False, True) for k in (8, 10, 12, 14, 16)],
        *route_lengths(),
    ],
)
def test_binary32_transforms_are_within_their_bound(real, n):
    """Forward and round trip within ``fft_bound`` at binary32 (SEP 33 7.1)."""
    import solprop_native._core as core  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    rng = np.random.default_rng(n)
    rows = 2
    samples = rng.standard_normal((rows, n))
    if not real:
        samples = samples + 1j * rng.standard_normal((rows, n))
    narrowed = (
        samples.astype(np.float32).astype(np.complex128)
        if real
        else samples.astype(np.complex64).astype(np.complex128)
    )
    forward, back = core.binary32_round_trip(np.ascontiguousarray(narrowed), real)
    exact = (
        np.fft.rfft(narrowed.real, axis=-1) if real else np.fft.fft(narrowed, axis=-1)
    )
    forward_error = float(np.linalg.norm(forward - exact) / np.linalg.norm(exact))
    trip_error = float(np.linalg.norm(back / n - narrowed) / np.linalg.norm(narrowed))
    bound = fft_bound(n, EPS_32)
    assert forward_error <= bound
    assert trip_error <= bound
    # And binary32's own: well above binary64's rounding.
    assert forward_error > 2.0**-40 * math.log2(n)
