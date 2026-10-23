# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Budget all Cartesian column buffers before backend construction."""

import dataclasses

import numpy as np
import pytest

from backends import test_free_ladders
from solprop.stats import StatsSpec
from solprop.translate.free import free_transform
from solprop.translate.problem import _held_current_chunk
from spatial_records import record_metric

free_response = test_free_ladders.free_case


@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_cartesian_column_buffers_stay_within_the_shared_budget(
    free_response, spatial_backend_class, scheme, request
):
    family, grid, space, problem = free_response
    # At most 19 complex-equivalent time columns are simultaneously retained:
    # response/FFT 8, doubled Raman 3, GPU plasma 3.5, split-step 2, statistics 2.5.
    # One further column covers real-FFT endpoint bins. Use a small budget to
    # exercise the same translation division without allocating a full-size run.
    budget = 7 * 20 * 16 * problem.shape[2] * grid.n_to
    transform = free_transform(
        space, grid, n_pol=problem.shape[2], z_max=grid.length, node_budget=budget
    )
    transform = _held_current_chunk(
        transform, problem.responses, problem.shape[2], grid.n_to, budget
    )
    geometry = transform.geometry
    assert geometry is not None
    transform = dataclasses.replace(
        transform,
        geometry=dataclasses.replace(geometry, absorber_rate=np.ones(space.n_columns)),
    )
    coarse_rate, fine_rate = np.ones(grid.n_t), np.ones(grid.n_to)
    coarse_rate.setflags(write=False)
    fine_rate.setflags(write=False)
    problem = dataclasses.replace(
        problem,
        transform=transform,
        responses=dataclasses.replace(
            problem.responses,
            absorber_scheme=scheme,
            temporal_absorber=coarse_rate,
            temporal_absorber_fine=fine_rate,
        ),
    )
    backend = spatial_backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.stats(StatsSpec(names=("energy", "beam_radius", "peak_intensity")))
        ledger = backend.memory_ledger()
        names = (
            "fine_spectrum",
            "modal_time",
            "node_field",
            "node_polarisation",
            "projected",
            "projected_spectrum",
            "chunk_projection",
            "absorber_spectrum",
            "absorber_time",
            "spatial_absorber_time",
            "split_absorber_spectrum",
            "split_absorber_time",
            "node_raman_drive",
            "node_raman_spectrum",
            "node_raman_convolved",
            "plasma_device_scan",
            "node_ionisation_rate",
            "node_fraction",
            "node_free_current",
            "plasma_work",
            "statistics_spectrum",
            "statistics_time",
            "statistics_power",
            "stats_nodes",
        )
        measured = sum(ledger.get(name, 0) for name in names)
        # Include the new persistent histories, products and transform buffers;
        # admission must fit them into the same node budget as the other sources.
        measured += sum(
            value
            for name, value in ledger.items()
            if name.startswith("held_current")
            and not name.endswith("fft_plan_heap_growth")
        )
        assert 0 < measured <= budget, (measured, budget, ledger)
        assert ledger["transverse_scratch"] == 16 * int(np.prod(problem.shape))
        record_metric(
            request,
            "Cartesian retained column buffers (bytes)",
            f"{test_free_ladders.recorded_family(family)}, {scheme}",
            measured,
            budget,
            spatial_backend_class.name,
        )
    finally:
        backend.close()
