# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response: the fused path's cases and their bound.

The capillary, mode-averaged and radial problems, the measurements and the rounding
bound of a compiled backend's ladder (``test_native_dispersive_kerr.py``) against
``numpy_ref``, which applies the multipliers at every node and is the oracle.

**The bounds are derived from summation lengths.** One evaluation of the constant
cubic is two transforms, ``16 eps ceil(log2 n_to)`` (``test_polarised_runs.py``), and
two matrix reductions over the nodes, ``gamma(n_nodes)`` each. The response adds
transforms and reductions per channel: the symmetric evaluator one inverse transform
of the filtered field and one forward transform of its product, one synthesis and one
projection per rule; the polynomial two inverse and three forward transforms, two
syntheses and three projections. The bounds still count the output inverse transform
the response had before its output joined the stage's spectrum (one per channel), so
they are a transform looser than they need be. The reference takes as many again at
every node, so each count is doubled. Each test states what it was measured against.
"""

import dataclasses
import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from bounds import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import MaterialSelection
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, Multistep, PIController
from solprop.translate.problem import KernelProblem, assemble

EPS = np.finfo(float).eps
LENGTH = 2e-3
STEPS = 10
ARGON = OptmatKerr("Ar").dispersive_model()
DENSITY = 4.9e25
"""Two bar of argon at room temperature, m^-3, for the routes built by hand."""

POLICIES = {"symmetric": "symmetric-kerr", "polynomial": "gas-polynomial"}
GRID_CASES: dict[str, dict[str, Any]] = {
    "field full": {"envelope": False, "thg": True},
    "field conserving": {"envelope": False, "thg": False},
    "envelope conserving": {"envelope": True, "thg": False},
    "envelope full": {"envelope": True, "thg": True},
}


# --- problems -------------------------------------------------------------------


def capillary(
    evaluator: str,
    *,
    gas: str | Mapping[str, float] = "Ar",
    stepped: bool = False,
    **overrides: object,
) -> Simulation:
    """Return a short three-mode argon capillary under a dispersive policy.

    Its quadrature has 33 nodes with an embedded rule. A stepped one takes the
    split-step absorber, whose collar is stable at a tenth of the guide, and ten fixed
    steps. Nitrogen's band starts at 180 nm, inside its index fit.
    """
    arguments: dict[str, Any] = {
        "wavelength": 800e-9,
        "wavelength_limits": (180e-9 if gas == "N2" else 150e-9, 4e-6),
        "trange": 150e-15,
        "fwhm": 10e-15,
        "energy": 30e-6,
        "modes": 3,
        "materials": MaterialSelection(electronic=POLICIES.get(evaluator, evaluator)),
        "plasma": False,
        "raman": False,
        "backend": "numpy_ref",
        "n_save": 2,
    }
    if stepped:
        arguments["absorber_scheme"] = "split_step"
        arguments["schedule"] = FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS)
    arguments |= overrides
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            75e-6, LENGTH, gas, 2.0, **{"noise": None, **arguments}
        )


def grid_for(envelope: bool) -> FieldGrid | EnvelopeGrid:
    """Return the hand-built routes' grid, oversampled for an envelope's harmonic."""
    if envelope:
        return EnvelopeGrid.from_limits(
            LENGTH, 800e-9, (150e-9, 4e-6), 150e-15, thg=True
        )
    return FieldGrid.from_limits(LENGTH, 800e-9, (150e-9, 4e-6), 150e-15)


def averaged(
    evaluator: str, *, envelope: bool, thg: bool, vector: bool = False
) -> tuple[KernelProblem, FieldGrid | EnvelopeGrid]:
    """Return a mode-averaged argon problem, one node, or a two-component quadrature.

    The two-component field is the polarised route of ``test_polarised_runs.py``: one
    mode, a 24-point radial rule, an elliptical launch.
    """
    grid = grid_for(envelope)
    mode = MarcatiliMode(75e-6, cladding=1.45, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=2e-6).build(grid)
    start = pulse
    quadrature = None
    if vector:
        angle, phase = math.pi / 5, math.pi / 3
        start = np.concatenate(
            [math.cos(angle) * pulse, math.sin(angle) * np.exp(1j * phase) * pulse],
            axis=1,
        )
        quadrature = transverse_quadrature([mode], n_r=24)
    problem = assemble(
        grid,
        [mode] if vector else mode,
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg)],
        start,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / STEPS,
        quadrature=quadrature,
        density=DENSITY,
        absorber_scheme="split_step",
        integrator=Multistep(),
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
    )
    return problem, grid


def radial(
    evaluator: str, *, envelope: bool, thg: bool
) -> tuple[KernelProblem, FieldGrid | EnvelopeGrid]:
    """Return a radial free-space problem: sixteen Hankel nodes, no embedded rule.

    On an envelope grid this is the backend's real radial GEMM.
    """
    if envelope:
        grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (300e-9, 2e-6), 100e-15, thg=True)
    else:
        grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)
    space = RadialGrid(4e-4, 16, collar_fraction=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    )
    start = pulse.build(grid, space, medium=SpatialMedium()) * grid.active_mask
    problem = assemble(
        grid,
        None,
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        absorber_scheme="split_step",
        integrator=Multistep(),
        schedule=FixedSchedule.uniform(0.0, 1e-4, 1e-5),
        density=DENSITY,
    )
    return problem, grid


# --- the measurements ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One source evaluation's projected polarisation spectrum on both rules."""

    projected: np.ndarray
    coarse: np.ndarray | None


def reference_evaluation(problem: KernelProblem, grid: Any) -> Evaluation:
    """Evaluate the reference's source once at the launch: its projected spectra.

    The reference keeps both rules' projections of its last evaluation in time; their
    unnormalised transforms are what a compiled backend's stage holds in frequency,
    where the response joins every other's.
    """
    backend = NumpyRefBackend(problem, grid=grid)
    forward = np.fft.rfft if problem.grid_kind == "field" else np.fft.fft
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0, 0.0)
        coarse = backend._last_coarse
        return Evaluation(
            forward(np.asarray(backend._last_projected), axis=-1),
            None if coarse is None else forward(np.asarray(coarse), axis=-1),
        )
    finally:
        backend.close()


def run(
    backend_class: type, problem: KernelProblem, grid: Any, **options: object
) -> tuple[np.ndarray, Any]:
    """Propagate to the last save plane and return the field and the summary."""
    backend = backend_class(problem, grid=grid, **options)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field().copy(), summary
    finally:
        backend.close()


def disagreement(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Return the relative L2 difference of two arrays of the same size."""
    ours, theirs = np.ravel(ours), np.ravel(theirs)
    scale = float(np.linalg.norm(theirs))
    assert scale > 0
    return float(np.linalg.norm(ours - theirs) / scale)


def evaluation_bound(problem: KernelProblem) -> float:
    """Bound one evaluation's rounding: transforms and node reductions, both backends.

    The constant cubic's two transforms at ``8 eps ceil(log2 n_to)`` each and its two
    reductions over the nodes, plus the response's per channel (module docstring),
    doubled for the reference's node-by-node arithmetic.
    """
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    symmetric = leaf.evaluator == "symmetric"
    transforms = 2 + (3 if symmetric else 6) * leaf.n_channels
    reductions = 2 + (2 if symmetric else 5) * leaf.n_channels
    per_transform = 8 * EPS * math.ceil(math.log2(problem.grids.n_to))
    return 2 * (
        transforms * per_transform + reductions * gamma(problem.transform.n_nodes)
    )
