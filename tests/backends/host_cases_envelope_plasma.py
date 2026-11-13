# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A two-population envelope mixture on each host geometry, and its thread probe.

The cases of ``test_envelope_plasma_geometries.py``. Run as a script by that module's
thread test, in a subprocess, because the Kokkos runtime reads its thread count once per
process (``backends/matched.py``): the first argument is the ``.npz`` file to write and
the others name what to save, in order, of ``field``, ``steps`` and ``met_tolerance``
for the free-space two-component case's two fixed Runge-Kutta steps.
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from mixture_cases import AnalyticRate, mixture_case
from solprop.backends.session import ExecutionOptions
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table


@dataclass(frozen=True)
class InstantaneousRate(AnalyticRate):
    """Declare the synthetic quadratic magnitude rate's physical convention."""

    convention: str = "instantaneous-isotropic"


def build_envelope_mixture():
    """Return the two-component mixture and its two envelope plasma responses."""
    mixture, _, _, _, _ = mixture_case(kerr=False)
    responses = tuple(
        EnvelopePlasmaResponse(
            component.species,
            build_rate_table(
                InstantaneousRate(scale=(index + 1) * 1e13),
                (1e7, 1e11),
                rtol=1e-7,
            ),
            preionisation=component.preionisation,
            species=component.label,
        )
        for index, component in enumerate(mixture.components)
    )
    return mixture, responses


def geometry_case(envelope_mixture, geometry, n_pol):
    mixture, responses = envelope_mixture
    length = 8e-6
    grid = EnvelopeGrid.from_limits(length, 800e-9, (500e-9, 1.2e-6), 80e-15)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7)
    jones = (1,) if n_pol == 1 else (1, 0.35j)
    kwargs = {}
    if geometry in ("radial", "free"):
        space = (
            RadialGrid(6e-4, 8, collar_fraction=0.0)
            if geometry == "radial"
            else FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0.0)
        )
        medium = SpatialMedium(mixture)
        initial = SpatialPulse(pulse, 80e-6, jones=jones).build(
            grid, space, medium=medium
        )
        mode = None
        kwargs = {"spatial_grid": space, "medium": medium}
    else:
        mode = MarcatiliMode(30e-6, mixture=mixture, cladding=1.45, loss=False)
        initial = pulse.build(grid) * np.asarray(jones)[None, :, None]
    problem, run_request = assemble(
        grid,
        mode,
        responses,
        initial,
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=[length],
        dz_max=length / 2,
        dz_init=length / 2,
        schedule=FixedSchedule.uniform(0.0, length, length / 2),
        **kwargs,
    )
    if geometry != "modal":
        assert problem.transform.n_nodes % 3 != 0
    return problem, run_request, grid


PARTIAL = ExecutionOptions(node_chunk=3)
"""Three-node chunks, which leave a partial chunk on both spatial geometries."""


def geometry_options(geometry):
    """Return the execution options a geometry's case runs with: partial chunks."""
    return None if geometry == "modal" else PARTIAL


# The doctest collection imports every module under tests/, and the arguments it
# was given are not array names: only a subprocess run saves the arrays.

if __name__ == "__main__":
    from backends.forced_threads import record_sizing
    from sessions.runs import run

    record_sizing(sys.argv[1])
    solprop_native = importlib.import_module("solprop_native")
    problem, run_request, grid = geometry_case(build_envelope_mixture(), "free", 2)
    backend = solprop_native.NativeBackend(
        problem, run_request, grid=grid, options=geometry_options("free")
    )
    try:
        result = run(backend, problem)
        # The run's one save plane is its end: its summary spans the run.
        summary = result.observations[-1].summary
        arrays: dict[str, Any] = {
            "field": result.fields[-1],
            "steps": np.array(summary.n_steps),
            "met_tolerance": np.array(summary.met_tolerance),
        }
        np.savez(
            sys.argv[1],
            runtime_threads=np.array(solprop_native.build_info().runtime_threads),
            **{f"case{index}": arrays[key] for index, key in enumerate(sys.argv[2:])},
        )
    finally:
        backend.close()
