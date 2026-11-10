# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Two channels' final fractions and peak rates on native, from one field.

Against the independent passes, ``passes.plasma_channel_fractions`` on the reference's
own node fields: each of the two exposure scans accumulates ``n_to`` terms, and the rate
interpolation adds log and exp error, so ``64 n_to eps`` per channel, absolute, and the
same relative for the peak rates, which have no cumulative sum. This was 14 times
looser at the launch of ``test_mixture_ladder.py``, and 56 times looser on the spatial
route. Both families (the real and the analytic column), one and two components, and
serial and threaded.

**Weaker than the original.** The fields are a launch of eight radial nodes, peak
``3.6e10`` V/m, as the state cannot carry an arbitrary time-domain field, so its two
sides synthesise their nodes separately: a rounding the original's shared array did not
have, and within the same bound by far (a transform's relative rounding, doubled by the
quadratic rate, against ``64 n_to``). The current is not here: native holds it only at
the stage, in ``native/tests/test_band_column_budget.py``'s
``test_two_channels_are_within_their_budget``, and the original's per-sample
``64 n_to 2 eps`` of it is not reachable. The packed tables and the polarisation buffer
of the original were Numba's own and have no heir.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.forced_threads import assert_threaded
from backends.matched import native_thread_fields
from backends.native_mixture_launch_probe import (
    NAMES,
    launch_case,
    launch_statistics,
)
from bounds import EPS
from solprop.backends.numpy_ref import NumpyRefBackend, passes

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

PROBE = Path(__file__).parent / "native_mixture_launch_probe.py"


def oracle(kerr, n_pol):
    """Return each channel's final fractions per node, peak rate, and their bound."""
    grid, problem = launch_case(kerr, n_pol)
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        bundle = reference._stage_scalars(0.0, 0.0)
        fields = np.real(
            np.concatenate(list(reference._plasma_nodes(reference._require_state()[0])))
        )
    finally:
        reference.close()
    rates, fractions = passes.plasma_channel_fractions(
        fields,
        problem.responses.plasma_channels,
        bundle.plasma_preionisation,
        grid.dt_fine,
        problem.responses.plasma_integral,
    )
    bound = 64 * grid.n_to * EPS
    return fractions[:, :, -1], np.max(rates, axis=(1, 2)), bound


CASES = [(kerr, n_pol) for kerr in (False, True) for n_pol in (1, 2)]


@pytest.mark.parametrize(("kerr", "n_pol"), CASES)
def test_two_channel_fractions_and_rates_are_the_passes(kerr, n_pol, record_property):
    fraction, rate, bound = oracle(kerr, n_pol)
    # Both channels really ionise on axis, beyond their pre-ionisation.
    assert np.all(fraction[:, 0] > np.array([0.03, 0.1]) + 1e-3)
    values = launch_statistics(native.NativeBackend, kerr, n_pol)
    measured = float(np.max(abs(values["species_ionisation"] - fraction)))
    record_property("measured", measured)
    record_property("bound", bound)
    assert_allclose(values["species_ionisation"], fraction, rtol=0, atol=bound)
    assert_allclose(values["species_peak_ionisation_rate"], rate, rtol=bound, atol=0)


@pytest.mark.medium
def test_two_channel_fractions_and_rates_on_several_threads(
    tmp_path, thread_budget_lifted
):
    """A threaded run's values against the same passes, in a fresh process each."""
    names = [
        f"{'kerr' if kerr else 'none'}:{n_pol}:{statistic}"
        for kerr, n_pol in CASES
        for statistic in NAMES
    ]
    threaded = native_thread_fields(PROBE, 4, names, tmp_path)
    assert_threaded(threaded["sizing"], 4, "kernels.stage")
    for kerr, n_pol in CASES:
        fraction, rate, bound = oracle(kerr, n_pol)
        tag = f"{'kerr' if kerr else 'none'}:{n_pol}"
        assert_allclose(threaded[f"{tag}:{NAMES[0]}"], fraction, rtol=0, atol=bound)
        assert_allclose(threaded[f"{tag}:{NAMES[1]}"], rate, rtol=bound, atol=0)
