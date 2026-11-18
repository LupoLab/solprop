# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reducing what a run records to what the front end shows."""

import numpy as np

from solprop.service.reduce import stats_row


def test_stats_row_totals_numbers_and_skips_the_rest():
    row = stats_row(
        {
            "energy": np.array([[1.0, 2.0]]),
            "zdw": float("nan"),
            "label": object(),
            "count": 3,
        }
    )
    # Sums of a few small integers in binary64: exact.
    assert row == {"energy": 3.0, "count": 3.0}
