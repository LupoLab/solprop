# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input shot noise beside every coupled linear flow (SEP 37, section 6).

The input draw is part of the entrance state, so a coupled flow propagates it like any
field and there is no transport question to answer: the modified model, whose draw is
kept apart and would need one, keeps its refusals. The draw is one photon per channel
and bin only in a channel basis orthonormal in true power, which every coupled-flow
route has at its entrance: the capillary's ``HE_1m`` set by its closed-form
normalisation and the Bessel orthogonality of its profiles, and a general guide's basis
by :func:`solprop.translate.general.check_noise_basis`.

Each route is checked three ways on ``numpy_ref``, on a fixed schedule so that the
three runs compared take the same steps and their differences are physics, not the
controller: the entrance carries one photon per channel and bin; the noisy run
completes; and its exit energy is the noiseless run's plus the seed's own transmitted
energy plus the interference term between the two exit fields. That last term is not
zero for one realisation -- it vanishes only on the ensemble -- so it is measured from
the same runs rather than assumed. The seed alone is the same draw beside a pulse
:data:`SEED_ONLY` times weaker, whose interference with the seed is then below 1e-9 of
the seed's energy.

Native carrying the draw to the reference's last plane, guarded by the draw itself, is
a row of the case catalogue per route (``tests/backends/flow_rows.py``,
``input noise ...``).
"""

import numpy as np
import pytest

from input_noise_cases import PULSE_ENERGY, ROUTES
from solprop.constants import HBAR
from solprop.noise import photons_per_bin
from solprop.output.store import Output
from solprop.processing.axes import grid_of

pytestmark = [
    pytest.mark.medium,
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]


SEED_ONLY = 1e-20
"""Energy ratio of the near-empty launch that carries the seed alone, dimensionless."""


SUPERPOSITION = {
    "capillary (coherent wall)": 1e-5,
    "capillary (taper coupling)": 1e-5,
    "general guide (moving)": 1e-3,
    "general guide (split block)": 1e-3,
}
"""Largest relative departure from the exit-energy identity, per route.

Relative to the seed's transmitted energy, declared before the runs. The flows are
linear, so the only departure is the pulse's cross-phase on the seed, ``2 gamma P L``:
about 1e-8 in a helium capillary at a picojoule over 0.2 mm, and about 4e-6 in the
silica-cored guides (a Kerr core, a few watts of peak power). Measured 1.1e-6 on both
capillary routes, 5.8e-5 on the moving guide and 2.0e-5 on the split block; each gate
is above its measurement by an order of magnitude or more, and far below the 1 the
identity would miss by if the seed were not carried.
"""


def _energy(output: Output, plane: int) -> tuple[float, np.ndarray]:
    """Return a plane's energy, J, and its stored raw spectrum, every bin."""
    grid = grid_of(output)
    stored = np.asarray(output.fields("fft")[plane])
    photons = photons_per_bin(grid, stored)
    return float(np.sum(photons * HBAR * grid.omega)), stored


@pytest.mark.parametrize("route", list(ROUTES))
def test_input_noise_rides_a_coupled_flow_as_part_of_the_entrance_state(route):
    """One photon per channel and bin at the entrance; the exit adds up.

    The energy identity is ``E(psi + N) = E(psi) + E(N) + 2 Re <psi, N>`` at the exit,
    with each term from its own run on the same steps; its gate is
    :data:`SUPERPOSITION`. A declared rotated basis (the split block) redistributes the
    per-channel photon numbers of a draw made in the reference basis and keeps their
    sum, so there the total over channels is checked, and it is the channel count.
    """
    build = ROUTES[route]
    clean = build(None, PULSE_ENERGY, "numpy_ref").run(progress="none")
    noisy = build("input", PULSE_ENERGY, "numpy_ref").run(progress="none")
    alone = build("input", PULSE_ENERGY * SEED_ONLY, "numpy_ref").run(progress="none")
    grid = grid_of(noisy)
    metadata = noisy.metadata
    assert metadata is not None
    assert metadata.noise is not None
    assert metadata.noise.model == "input"
    assert metadata.modal is not None
    assert metadata.modal["noise_reference_basis"] == "orthonormal true power"
    assert clean.metadata is not None
    assert clean.metadata.modal is not None
    assert clean.metadata.modal["noise_reference_basis"] is None

    # The entrance: the saved plane at z = 0 less the noiseless one is the draw.
    _, entrance_noisy = _energy(noisy, 0)
    _, entrance_clean = _energy(clean, 0)
    photons = photons_per_bin(grid, entrance_noisy - entrance_clean)[..., 0, :]
    active = grid.active_mask & (grid.omega > 0.0)
    # The subtraction of two saved launches of 1e-12 J, whose seed is 3e-4 of the
    # pulse's amplitude per bin, keeps about 12 of the 16 digits: 1e-9 is ample.
    if route == "general guide (split block)":
        total = np.sum(photons, axis=0)[active]
        np.testing.assert_allclose(total, photons.shape[0], rtol=1e-9, atol=0.0)
    else:
        np.testing.assert_allclose(photons[:, active], 1.0, rtol=1e-9, atol=0.0)
    assert np.all(photons[:, ~active] == 0.0)

    # The exit.
    energy_clean, exit_clean = _energy(clean, -1)
    energy_noisy, _ = _energy(noisy, -1)
    energy_seed, exit_seed = _energy(alone, -1)
    assert np.isfinite(energy_noisy)
    norm = (
        photons_per_bin(grid, np.ones(grid.n_omega, dtype=complex)) * HBAR * grid.omega
    )
    interference = 2.0 * float(np.sum(np.real(np.conj(exit_clean) * exit_seed) * norm))
    seed_in = float(np.sum(HBAR * grid.omega[active])) * photons.shape[0]
    # The flows are passive: the seed can only lose energy on its way.
    assert 0.0 < energy_seed <= seed_in * (1.0 + 1e-12)
    departure = energy_noisy - energy_clean - energy_seed - interference
    assert abs(departure) <= SUPERPOSITION[route] * energy_seed
