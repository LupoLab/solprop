# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An unresolved angular source refuses in the statistics path too (SEP 43).

``numba_cpu``'s ``test_host_refuses_unresolved_angular_source_and_recovers`` held both
passes: the source and the populations refused, wrote nothing, and a zero field then
recovered. ``test_envelope_plasma.py`` holds the source; this is the other.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from solprop.exceptions import ConvergenceError
from solprop.stats import StatsSpec
from test_envelope_plasma import envelope_case, envelope_options, terms

import solprop_native.backend as adapter
from solprop_native import NativeBackend
from solprop_native.backend import (
    _native_problem,  # pyright: ignore[reportPrivateUsage]
)

SPEC = StatsSpec(names=("electron_density", "peak_ionisation_rate"))


@pytest.mark.parametrize(
    "geometry",
    ["mode_averaged", "free"],
)
def test_the_statistics_refuse_and_a_quiet_field_recovers(
    monkeypatch: pytest.MonkeyPatch, geometry: str
) -> None:
    problem, grid = envelope_case(geometry, 1, mixture=False)
    payload = _native_problem(
        problem, terms(problem), node_chunk=envelope_options(geometry).chunk
    )
    nodes = cast(npt.NDArray[np.float64], payload["envelope_rule_nodes"]).copy()
    # As the source's test: alternating incompatible rules must fail refinement.
    for level in (1, 3):
        order = 64 << level
        nodes[:, level, :order] = 0.5 * (nodes[:, level, :order] + 1)
    payload["envelope_rule_nodes"] = nodes

    def corrupt_payload(*args: object, **keywords: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(adapter, "_native_problem", corrupt_payload)
    owner = NativeBackend(problem, grid=grid, options=envelope_options(geometry))
    try:
        owner.set_initial(
            np.asarray(10.0 * problem.initial.field0, dtype=np.complex128), None
        )
        with pytest.raises(ConvergenceError, match="angular refinement"):
            owner.stats(SPEC)
        # A quiet field below threshold: no refinement, and no failure left behind.
        owner.set_initial(np.zeros_like(problem.initial.field0), None)
        quiet = owner.stats(SPEC)
        assert np.all(np.asarray(quiet["peak_ionisation_rate"]) == 0.0)
    finally:
        owner.close()
