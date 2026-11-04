# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How the translation admits and chunks a held plasma current, on no backend (SEP 43).

The production rules of ``translate/problem.py``, which lived with the retired
``test_numba_held_plasma.py`` and owe nothing to it.
"""

import dataclasses

import pytest

from backends.test_band_plasma_parity import case
from solprop.exceptions import TranslationError
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.problem import _balanced_chunk, _held_current_chunk

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


def test_node_admission_counts_all_held_histories_together():
    grid, problem = case("mode_averaged")
    transform, responses = problem.transform, problem.responses
    with pytest.raises(TranslationError, match="held plasma current history"):
        _held_current_chunk(transform, responses, 1, grid.n_to, 1)
    absent = dataclasses.replace(
        responses,
        plasma_channels=(),
        plasma_integral=design_running_integral(grid, "trapezoid"),
    )
    assert _held_current_chunk(transform, absent, 1, grid.n_to, 1) is transform


def test_planar_admission_reserves_a_whole_transverse_row():
    grid, problem = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    # This cannot even hold the row's twenty baseline complex buffers, before
    # adding the held-current streams and integration histories.
    budget = 20 * 16 * grid.n_to * n_x - 1
    with pytest.raises(TranslationError, match="full transverse row"):
        _held_current_chunk(transform, problem.responses, 1, grid.n_to, budget)
    transform = dataclasses.replace(transform, node_chunk=n_x + 1)
    admitted = _held_current_chunk(transform, problem.responses, 1, grid.n_to, 1 << 30)
    assert admitted.node_chunk == n_x


def test_planar_chunks_stay_whole_rows_when_balanced():
    _, problem = case("planar")
    transform = problem.transform
    assert transform.geometry is not None
    n_x = transform.geometry.shape[0]
    rows = transform.n_nodes // n_x
    assert rows == 4
    # Three of four rows a pass would leave a one-row tail: two rows a pass.
    three = dataclasses.replace(transform, node_chunk=3 * n_x)
    assert _balanced_chunk(three).node_chunk == 2 * n_x
    # A chunk shorter than a row is not a whole number of rows and is kept.
    partial = dataclasses.replace(transform, node_chunk=5)
    assert _balanced_chunk(partial) is partial


def test_budget_chunks_spread_the_nodes_evenly_over_their_passes():
    # A budget that admits all but one node must not leave a one-node tail pass:
    # nine nodes under a limit of eight take two passes of five and four.
    _, problem = case("multimode")
    transform = problem.transform
    assert transform.geometry is None
    n_nodes = transform.n_nodes
    assert _balanced_chunk(transform) is transform
    tight = dataclasses.replace(transform, node_chunk=n_nodes - 1)
    assert _balanced_chunk(tight).node_chunk == -(-n_nodes // 2)
    single = dataclasses.replace(transform, node_chunk=1)
    assert _balanced_chunk(single) is single
