# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The planar plates: the hybrid planar kind on vacuum-filled plates, every family.

Vacuum-filled plates of constant index (so no material fit is read and the responses
carry their own density, as the Cartesian ladders'), three slab orders with the beam
off centre in x so that odd and even orders are both driven, and a node chunk of four y
rows over 11, so that the last chunk is partial; every grid, response family and
absorber scheme the reference executes (:data:`FAMILIES`), and the rounding budget the
backends are held to (:func:`amplitude_bound`). The case catalogue's planar rows
(``planar_rows.py``) compare native with the reference on them;
``test_planar_native.py`` holds native's ledger and threads on them.
"""

import dataclasses
import math
from dataclasses import dataclass
from functools import partial

import numpy as np

from backends.spatial_cases import CHI3, LENGTH, QuadraticRate
from bounds import gamma, ladder_bound
from mixture_cases import AnalyticRate
from sessions.runs import run
from solprop.backends.session import ExecutionOptions
from solprop.boundaries import SpatialAbsorber, absorber
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table

EPS = np.finfo(float).eps
HALF_GAP = 63.5e-6
ORDERS = (1, 2, 3)
N_Y = 11
ROWS_PER_CHUNK = 4
COLLAR_N_Y = 22
"""The collar families' rows: enough that the declared band lies below the ramp."""
COLLAR = SpatialAbsorber(kappa=14e3, attenuation=300.0)
"""The collar families' y collar (SEP 40), in a room four times the 0.4 mm aperture.

A declared band, kappa w = 22.4 above the floor 22.3, below the safety ramp's start
at 1.56e4 rad/m; three hundred nepers at its edge make a collar of about 9e3 1/m,
close to one neper over the run, so that it moves the answer by more than the
rounding compared."""

FAMILIES = (
    "envelope_kerr",
    "field_analytic_kerr",
    "field_real_kerr",
    "field_plasma",
    "envelope_raman",
    "envelope_plasma",
    "collar_in_stage",
    "collar_split_step",
)
"""Every grid family, response family and absorber scheme the reference executes."""

PLASMA_NAMES = ("electron_density", "peak_ionisation_rate")
SPATIAL_NAMES = (
    "energy",
    "peak_intensity",
    "on_axis_intensity",
    "on_axis_spectrum",
    "beam_radius",
    "fwhm_r",
)


@dataclass(frozen=True)
class InstantaneousRate(AnalyticRate):
    """Declare the synthetic quadratic rate's envelope convention."""

    convention: str = "instantaneous-isotropic"


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


def responses_for(family):
    """Return a family's responses and the density that scales them."""
    if family == "field_plasma":
        rate = build_rate_table(QuadraticRate(), (1e3, 1e10))
        return [PlasmaResponse("Ar", rate)], 1e22
    if family == "envelope_plasma":
        rate = build_rate_table(InstantaneousRate(), (1e7, 1e11), rtol=1e-7)
        return [EnvelopePlasmaResponse("Ar", rate)], 1e24
    if family == "envelope_raman":
        oscillator = DampedOscillator(1e-8, 2e14, undamped_dephasing())
        return [RamanResponse((oscillator,), thg=False)], 1.0
    # A field grid's Kerr term with its third harmonic is the real-field family; the
    # others are the analytic signal and the envelope.
    return [KerrResponse(CHI3, thg=family == "field_real_kerr")], 1.0


def planar_case(family):
    """Build one family's slab problem, grid and planar grid."""
    field = family.startswith("field") or family == "collar_split_step"
    constructor = (
        FieldGrid.from_limits if field else partial(EnvelopeGrid.from_limits, thg=False)
    )
    # The historical 150 fs field-plasma window has only 30 taper samples and
    # fails the held-current certificate. Keep the other families' grids fixed.
    duration = 200e-15 if family == "field_plasma" else 150e-15
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), duration)
    modes = [SlabMode(HALF_GAP, order=p, cladding=1.45) for p in ORDERS]
    collar = family.startswith("collar")
    space = (
        PlanarGrid(modes, 4e-4, COLLAR_N_Y, collar_fraction=4.0)
        if collar
        else PlanarGrid(modes, 4e-4, N_Y)
    )
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        waist_x=0.735 * HALF_GAP,
        offset_x=10e-6,
    )
    responses, density = responses_for(
        "field_analytic_kerr" if family == "collar_split_step" else family
    )
    problem, run_request = assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        # A reference length short enough that the temporal absorber moves the
        # answer by more than the rounding compared, as the y collar does.
        absorber=absorber(grid, boundary_length=10 * LENGTH) if collar else None,
        spatial_absorber=COLLAR if collar else None,
        absorber_scheme="split_step" if family == "collar_split_step" else "nonlinear",
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / 8,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
        density=density,
    )
    return grid, space, problem, run_request


def chunked(space):
    """Return the options of a case's backends: :data:`ROWS_PER_CHUNK` rows a chunk."""
    return ExecutionOptions(node_chunk=ROWS_PER_CHUNK * space.x.size)


def amplitude_bound(grid, space, family, steps, *, length=LENGTH):
    """Bound the backends' difference relative to the largest reference amplitude.

    Per stage evaluation: the free ladders' ``32 eps`` per FFT level
    (``tests/backends/test_free_ladders.py``) over the temporal and y transforms,
    and the same allowance per term of the two x contractions of the slab rule, of
    lengths ``n_p`` and ``n_x`` (Higham's ``gamma`` of each dot product times the
    ladders' conditioning allowance of 32); a plasma scan's ``gamma(12 n_to)``.
    Seven stages a step. Over a propagation the operator adds the square root's
    rounding of each column's ``k_z``, a few units of ``beta_max`` (D1's ``8 eps
    beta_max L`` for the uniform-in-y comparison of the same formula).
    """
    depth = (
        math.ceil(math.log2(grid.n_to))
        + math.ceil(math.log2(space.n_y))
        + space.n_p
        + int(space.x.size)
    )
    per_call = 32 * EPS * depth
    if family.endswith("plasma"):
        per_call += gamma(12 * grid.n_to)
    evaluations = max(1, 7 * steps)
    bound = ladder_bound("native", evaluations * per_call, grid.n_to, evaluations)
    if steps:
        beta_max = float(np.max(np.abs(grid.omega[grid.active_mask]))) / C
        bound += 8 * EPS * beta_max * length
    return bound


def _landed(problem, run_request, grid, implementation, names=(), options=None):
    """Run a problem to its end through the backend's session, with `names` observed."""
    request = run_request.for_run(stats=StatsSpec(names=names))
    backend = implementation(problem, request, grid=grid, options=options)
    try:
        return run(backend, problem, request)
    finally:
        backend.close()


def propagate(problem, run_request, grid, implementation, names=(), options=None):
    """Run a problem to its end and return the field and requested statistics."""
    landed = _landed(problem, run_request, grid, implementation, names, options)
    return landed.fields[-1], (landed.all_planes()[-1].stats if names else {})


def with_steps(run_request, steps):
    """Replace a request's fixed schedule with ``steps`` uniform steps."""
    schedule = run_request.stepper.schedule
    assert schedule is not None
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=type(schedule).uniform(0, LENGTH, LENGTH / steps),
        )
    )
