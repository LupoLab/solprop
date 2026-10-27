# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Full, frozen and rescaled-area profiles of one short nonlinear run, compared.

Full profiles against the frozen and rescaled-area approximations at matched
accuracy, comparing the three models **without assuming an approximation wins**. The
static-profile source comparison in ``tests/backends/test_general_modal.py`` has the
frozen arm only, and on the source rather than on a propagation.

One HE11 channel of the high-contrast strand -- a 1.5 um silica-like core in air,
where the transverse profile moves most with frequency -- is propagated for a
millimetre with a 20 fs pulse in three models that share the channel's propagation
constant and differ only in the nonlinear one:

* **full**: the general modal route, which carries the channel's vector profile at
  every coarse bin;
* **frozen**: the same route with the synthesis and projection maps replaced by
  their values at the bin nearest the reference frequency, over the maps' own
  support -- the static-profile model, with the route's source frequency factor,
  which is the shock term, left as it is;
* **rescaled area**: the GNLSE, through ``prop_gnlse``, with the channel's own
  dispersion and its **scalar** effective area at the reference frequency
  (:meth:`solprop.nonlinear.gnlse.SpecifiedMode.from_mode` with
  :func:`solprop.translate.modal_tables.channel_area`), ``n2`` from the core's
  ``chi3`` at its index, and the shock term -- what a user who has only an area
  would run.

The observable is the nonlinear phase at the peak of each model's own
dispersion-only run, which removes the frame, the grid and the dispersion from each.
"""

import dataclasses
from typing import cast

import numpy as np
import pytest

from modal_provider_cases import GENERAL_CHI3, general_guide
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import C
from solprop.driver import prop_gnlse
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.normalisation import modal_kerr_coefficient
from solprop.output.memory import MemoryOutput
from solprop.physdata import n2_from_chi3
from solprop.pulses import GaussianPulse
from solprop.stats import analytic_field
from solprop.stepper import PIController
from solprop.translate.general import general_modal_route
from solprop.translate.modal_tables import channel_area, sample_modal_table
from solprop.translate.problem import assemble

LENGTH = 1e-3
"""Propagated length, m: a millimetre."""

WINDOW = (760e-9, 920e-9)
"""Declared band, m: it holds the 20 fs pulse's spectrum to well below its peak."""

TRANGE = 200e-15
FWHM = 20e-15
ENERGY = 4e-11
"""Pulse energy, J: about 0.1 rad of nonlinear phase at the peak over the run.

Perturbative, so that the phase at the peak is first order in the coefficient and a
relative difference in the coefficient is the same relative difference in it.
"""

RTOL = 1e-8
"""Controller tolerance of every run, the same for all three models."""


@pytest.fixture(scope="module")
def strand():
    """The strand's guide on the union of both grids' active bands, and its route."""
    field_grid = FieldGrid.from_limits(LENGTH, 835e-9, WINDOW, TRANGE)
    envelope_grid = EnvelopeGrid.from_limits(LENGTH, 835e-9, WINDOW, TRANGE)
    band = (
        min(
            float(field_grid.omega[field_grid.active_mask].min()),
            float(envelope_grid.omega[envelope_grid.active_mask].min()),
        ),
        max(
            float(field_grid.omega[field_grid.active_mask].max()),
            float(envelope_grid.omega[envelope_grid.active_mask].max()),
        ),
    )
    guide = general_guide(grid=field_grid, band=band, knots=13)
    sampling = guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        guide.field_provider, sampling.omega, np.array([0.0]), sampling.nodes
    )
    route = general_modal_route(
        guide,
        field_grid,
        z_max=LENGTH,
        omega_reference=float(field_grid.omega0),
        kerr_form="analytic",
    )
    return {
        "field_grid": field_grid,
        "envelope_grid": envelope_grid,
        "guide": guide,
        "table": table,
        "nodes": sampling.nodes,
        "route": route,
    }


def general_runs(strand):
    """Return the linear, full and frozen general runs' exit fields and step counts."""
    grid = strand["field_grid"]
    launch = GaussianPulse(wavelength=835e-9, fwhm=FWHM, energy=ENERGY).build(grid)
    problem = assemble(
        grid,
        list(strand["guide"].mode_collection),
        [],
        launch,
        controller=PIController(rtol=RTOL, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH / 10,
        dz_init=LENGTH / 10,
        general=strand["route"],
    )
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None and tables.projection is not None
    centre = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    support = np.any(
        np.abs(tables.synthesis) > 0.0, axis=tuple(range(tables.synthesis.ndim - 1))
    )
    frozen_tables = dataclasses.replace(
        tables,
        synthesis=np.where(
            support, tables.synthesis[..., centre : centre + 1], 0.0
        ).astype(tables.synthesis.dtype),
        projection=np.where(
            support, tables.projection[..., centre : centre + 1], 0.0
        ).astype(tables.projection.dtype),
    )
    problems = {
        "linear": dataclasses.replace(
            problem,
            responses=dataclasses.replace(problem.responses, regional_responses=None),
        ),
        "full": problem,
        "frozen": dataclasses.replace(
            problem,
            transform=dataclasses.replace(
                problem.transform, modal_tables=frozen_tables
            ),
        ),
    }
    runs = {}
    for name, one in problems.items():
        owner = NumpyRefBackend(one, grid=grid)
        try:
            owner.set_initial(one.initial.field0, None)
            summary = owner.propagate_to(LENGTH)
            runs[name] = (
                np.array(owner.observe().require_field()[0, 0, 0]),
                summary.n_steps,
            )
        finally:
            owner.close()
    return runs, launch, support, tables.synthesis, centre


def gnlse_runs(strand, area, n2):
    """Return the linear and nonlinear rescaled-area GNLSE runs' exit fields."""
    channel = strand["guide"].mode_collection.modes[0]
    common = {
        "grid": strand["envelope_grid"],
        "fibre": SpecifiedMode.from_mode(channel, aeff=area),
        "shock": True,
        "lambda0": 835e-9,
        "fwhm": FWHM,
        "energy": ENERGY,
        "pulseshape": "gauss",
        "rtol": RTOL,
        "atol": 1e-14,
        "dz_max": LENGTH / 10,
        "dz_init": LENGTH / 10,
        "noise": None,
        "progress": "none",
        "backend": "numpy_ref",
    }
    linear = cast("MemoryOutput", prop_gnlse(0.0, LENGTH, **{"noise": None, **common}))
    nonlinear = cast(
        "MemoryOutput", prop_gnlse(None, LENGTH, n2=n2, **{"noise": None, **common})
    )
    return {
        name: (
            np.array(cast("np.ndarray", run.planes()[-1].field)[0, 0]),
            run.planes()[-1].step_history.n_steps,
        )
        for name, run in (("linear", linear), ("nonlinear", nonlinear))
    }


def peak_phase(grid, linear, nonlinear):
    """Return the linear run's peak power and the nonlinear phase at that peak."""
    straight = analytic_field(grid, linear)
    peak = int(np.argmax(np.abs(straight) ** 2))
    turned = analytic_field(grid, nonlinear)
    return float(np.abs(straight[peak]) ** 2), float(
        np.angle(turned[peak] / straight[peak])
    )


@pytest.mark.slow
def test_profile_dispersion(strand, record_property):
    """Full, frozen and rescaled-area models of one short run, at matched accuracy.

    **The bounds, written before the runs**, are on each approximation's departure
    from the full model's phase, relative to that phase:

    * **frozen**: the cubic carries three factors of the synthesis map and one of
      the projection, so a profile that varies by ``v`` relative across the bins
      the launch occupies moves the source, and so the perturbative phase, by at
      most ``4 v`` (the argument of the static-profile source test, applied to the
      propagated phase at first order);
    * **rescaled area**: the frozen model's bound, plus the departure of the
      scalar-area coefficient ``n2 omega_0 / (c A_eff)`` from the channel's vector
      one at the reference frequency -- the only other thing the GNLSE changes --
      plus the two grids' different sampling of the peak power, measured on the
      dispersion-only runs.

    Each bound carries the **numerical floor** of the comparison: the controller's
    ``rtol`` times the accepted steps of each of the two runs compared. Both
    departures must also exceed that floor, so that what is reported is a
    difference between models and not between runs. Nothing is asserted about
    which approximation is closer: neither is assumed to win.
    """
    field_grid, envelope_grid = strand["field_grid"], strand["envelope_grid"]
    table, nodes = strand["table"], strand["nodes"]
    omega0 = float(field_grid.omega0)
    sample = table.evaluate(omega0, 0.0)
    vector_gamma = modal_kerr_coefficient(
        sample.electric[:, :, 0],
        nodes.weights,
        np.where(nodes.region_index == 0, GENERAL_CHI3, 0.0),
        omega0,
        metric=float(np.real(sample.flux_metric[0, 0])),
    )
    area = channel_area(sample, nodes, 0)
    n2 = n2_from_chi3(GENERAL_CHI3, 1.45)
    area_gamma = n2 * omega0 / (C * area)
    coefficient = abs(area_gamma / vector_gamma - 1.0)
    record_property("profile_vector_gamma", vector_gamma)
    record_property("profile_area_gamma", area_gamma)

    runs, launch, support, synthesis, centre = general_runs(strand)
    density = np.abs(launch[0, 0]) ** 2
    occupied = support & (density > 1e-6 * density.max())
    reference = synthesis[..., centre]
    variation = max(
        float(np.max(np.abs(synthesis[..., column] - reference)))
        / float(np.max(np.abs(reference)))
        for column in np.flatnonzero(occupied)
    )
    record_property("profile_variation", variation)
    gnlse = gnlse_runs(strand, area, n2)

    power, full = peak_phase(field_grid, runs["linear"][0], runs["full"][0])
    _, frozen = peak_phase(field_grid, runs["linear"][0], runs["frozen"][0])
    envelope_power, rescaled = peak_phase(
        envelope_grid, gnlse["linear"][0], gnlse["nonlinear"][0]
    )
    sampling = abs(power - envelope_power) / power
    record_property("profile_peak_power_sampling", sampling)
    floor_frozen = RTOL * (runs["full"][1] + runs["frozen"][1])
    floor_rescaled = RTOL * (runs["full"][1] + gnlse["nonlinear"][1])
    frozen_departure = abs(frozen / full - 1.0)
    rescaled_departure = abs(rescaled / full - 1.0)
    for name, value in {
        "phase_full": full,
        "phase_frozen": frozen,
        "phase_rescaled": rescaled,
        "frozen_departure": frozen_departure,
        "rescaled_departure": rescaled_departure,
        "frozen_bound": 4 * variation + floor_frozen,
        "rescaled_bound": coefficient + 4 * variation + sampling + floor_rescaled,
        "floor_frozen": floor_frozen,
        "floor_rescaled": floor_rescaled,
    }.items():
        record_property(f"profile_{name}", value)
    # Recorded, not asserted: how much of the GNLSE's departure is its coefficient.
    record_property(
        "profile_rescaled_less_coefficient", rescaled / full - area_gamma / vector_gamma
    )
    # The run is the perturbative one the bounds are written for.
    assert 0.02 < abs(full) < 0.3
    assert floor_frozen < frozen_departure <= 4 * variation + floor_frozen
    assert (
        floor_rescaled
        < rescaled_departure
        <= coefficient + 4 * variation + sampling + floor_rescaled
    )
