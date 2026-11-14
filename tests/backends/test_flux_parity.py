# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Backend parity of the axial-flux statistics (SEP 34 sections 4.2 and 5.1).

The same state, loaded at the same plane, is measured by ``numpy_ref`` and, when it is
installed, the native CPU backend: energy, the beam,
aperture and region spectra, the aperture's on-axis spectrum, ``absorbed_power``,
the on-axis spectrum and intensity, ``peak_intensity`` and the fluence widths. The
free-space media are lossy, so the admittance is complex and the local statistics'
cross terms ``Im(eta)`` are far above rounding. The planar walls' loss is not:
making them lossless moves every planar statistic by at most 0.52 of its rounding
budget (measured), and the walls' whole term, ``kappa_sq``, moves them by 1.3e6 to
1.7e6 budgets. The radial argon gradient reads the plane measure's material table at
a position past the entrance.
Every comparison is a matched discrete computation, held to a rounding budget fixed
here before the comparison (SEP 25). The cases and budgets are
``backends/flux_cases.py``'s, and the comparisons the catalogue's rows
(``backends/geometry_rows.py``); what stays here is a width's refusal and native's
ledger.
"""

import importlib.util

import numpy as np
import pytest

from backends.flux_cases import free_case, radial_case
from sessions.runs import launch_statistics
from solprop.backends import select
from solprop.backends.spatial import fluence_budget, signed_fluence_widths
from solprop.exceptions import BackendError
from solprop.grid import FreeGrid, LineGrid
from solprop.profiles import Linear
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.translate.routes import flux_of

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

BACKENDS = ["numpy_ref"] + (
    ["native"] if importlib.util.find_spec("solprop_native") is not None else []
)


def test_a_width_refuses_a_fluence_negative_beyond_its_normwise_budget():
    """Rounding in a beam's wings is read as zero; a real negative lobe is refused."""
    _, problem, _ = free_case(LineGrid(4e-4, 24))
    geometry = problem.transform.geometry
    assert geometry is not None
    x = geometry.axes[0]
    fluence = np.exp(-((x / 6e-5) ** 2))
    magnitude = np.abs(fluence)
    budget = fluence_budget(magnitude, 1000)
    # Synthesis rounding is normwise: a wing value below zero by far more than its
    # own size, but within the budget of the peak, is rounding.
    wing = fluence.copy()
    wing[0] = -0.5 * budget[0]
    assert np.isfinite(signed_fluence_widths(geometry, wing, budget, 1.0)[0])
    lobe = fluence.copy()
    lobe[0] = -1e-6
    with pytest.raises(BackendError, match="nonnegative fluence profile"):
        signed_fluence_widths(geometry, lobe, budget, 1.0)


@pytest.mark.skipif("native" not in BACKENDS, reason="solprop-native is not installed")
def test_the_native_closure_buffers_are_in_the_ledger():
    """The closure's transverse volume, the radial plane measure and its table."""
    grid, problem, run_request = free_case(FreeGrid(3e-4, 24, 3e-4, 16))
    # One state's bytes beside the transverse scratch, sized for the problem's own
    # local statistics.
    closure = 16 * int(np.prod(problem.shape))
    owner = select("native", problem, run_request, grid=grid)
    try:
        ledger = owner.memory_ledger()
        assert ledger["flux_closure_transverse"] == closure
        # A separable run's plane measure is its operator's own views.
        assert ledger["flux_coefficients"] == ledger["flux_table"] == 0
        # Its fluence takes no time transform, so it holds no magnitude per sample.
        assert ledger["statistics_flux_magnitude"] == 0
    finally:
        owner.close()
    # Without a local statistic nothing is held until a call first needs it.
    bare = run_request.for_run(stats=StatsSpec(names=("energy",)))
    owner = select("native", problem, bare, grid=grid)
    try:
        assert owner.memory_ledger()["flux_closure_transverse"] == 0
        launch_statistics(
            owner,
            problem,
            field=problem.initial.field0,
            stats=StatsSpec(names=("peak_intensity",)),
        )
        assert owner.memory_ledger()["flux_closure_transverse"] == closure
    finally:
        owner.close()
    grid, radial, radial_request = radial_case(
        SpatialMedium("Ar", pressure=Linear(0.5, 3.0, 1e-3))
    )
    flux = flux_of(radial.route)
    assert flux is not None and flux.material is not None
    owner = select("native", radial, radial_request, grid=grid)
    try:
        ledger = owner.memory_ledger()
        # The entrance squares (two complex rows of n_omega, two real of n_modes)
        # and the material table's coefficients and breakpoints.
        n_omega, n_modes = radial.shape[2], radial.shape[0]
        assert ledger["flux_coefficients"] == 32 * n_omega + 16 * n_modes
        assert ledger["flux_table"] == (
            flux.material.coefficients.size * 16 + flux.material.breakpoints.size * 8
        )
        assert ledger["flux_closure_transverse"] == 0
        assert ledger["statistics_flux_magnitude"] > 0
    finally:
        owner.close()
