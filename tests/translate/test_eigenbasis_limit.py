# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The eigenbasis gate sits where the two derived formation bounds meet."""

import pytest

from flow_parity import eigenbasis_formation_bound, reference_formation_bound
from solprop.translate.coupled import eigenbasis_limit


@pytest.mark.parametrize("n_modes", [1, 2, 3, 4, 8, 16, 64])
def test_the_eigenbasis_limit_is_where_the_two_formation_bounds_meet(n_modes):
    """``eigenbasis_limit`` is the ``kappa`` equating the two routes' bounds at zero.

    At a vanishing step the eigenbasis route's bound is ``kappa (n + 11) u`` and the
    reference's scaling and squaring's ``16 u + 2 sqrt(2) n (n + 2) u``; the gate
    sits where they meet, so a gated bin is never formed less accurately, in bound.
    """
    limit = eigenbasis_limit(n_modes)
    eigen = eigenbasis_formation_bound(n_modes, 0.0, limit, 0.0, 0.0)
    assert eigen == pytest.approx(reference_formation_bound(n_modes, 0.0, 0), rel=1e-14)
    assert limit >= 2.0
