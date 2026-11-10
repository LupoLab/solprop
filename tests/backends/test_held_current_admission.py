# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How the node-chunk plan admits a held plasma current, on no backend (SEP 43).

The production rules of ``solprop.backends.node_chunk``, which the translation applied
before SEP 47 moved the chunk to backend preparation.
"""

import dataclasses

import pytest

from backends.test_band_plasma_parity import case
from solprop.backends.node_chunk import _balanced_chunk, _held_current_chunk
from solprop.exceptions import BackendError
from solprop.translate.plasma_integration import design_running_integral

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


def test_node_admission_counts_all_held_histories_together():
    grid, problem, _ = case("mode_averaged")
    transform, responses = problem.transform, problem.responses
    with pytest.raises(BackendError, match="held plasma current history"):
        _held_current_chunk(1, transform, responses, 1, grid.n_to, 1)
    absent = dataclasses.replace(
        responses,
        plasma_channels=(),
        plasma_integral=design_running_integral(grid, "trapezoid"),
    )
    assert _held_current_chunk(1, transform, absent, 1, grid.n_to, 1) == 1


def test_planar_admission_reserves_a_whole_transverse_row():
    grid, problem, _ = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    # This cannot even hold the row's twenty baseline complex buffers, before
    # adding the held-current streams and integration histories.
    budget = 20 * 16 * grid.n_to * n_x - 1
    with pytest.raises(BackendError, match="full transverse row"):
        _held_current_chunk(n_x, transform, problem.responses, 1, grid.n_to, budget)
    admitted = _held_current_chunk(
        n_x + 1, transform, problem.responses, 1, grid.n_to, 1 << 30
    )
    assert admitted == n_x


def test_planar_chunks_stay_whole_rows_when_balanced():
    _, problem, _ = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    rows = transform.n_nodes // n_x
    assert rows == 4
    # Three of four rows a pass would leave a one-row tail: two rows a pass.
    assert _balanced_chunk(3 * n_x, transform) == 2 * n_x
    # A chunk shorter than a row is not a whole number of rows and is kept.
    assert _balanced_chunk(5, transform) == 5


def test_budget_chunks_spread_the_nodes_evenly_over_their_passes():
    # A budget that admits all but one node must not leave a one-node tail pass:
    # nine nodes under a limit of eight take two passes of five and four.
    _, problem, _ = case("multimode")
    transform = problem.transform
    assert transform.geometry is None
    n_nodes = transform.n_nodes
    assert _balanced_chunk(n_nodes, transform) == n_nodes
    assert _balanced_chunk(n_nodes - 1, transform) == -(-n_nodes // 2)
    assert _balanced_chunk(1, transform) == 1
