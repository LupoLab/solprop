# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform on native CPU, against the oracle.

Native executes schema 15's ``"slab"`` kind in every combination ``numpy_ref`` does: the
y FFT around the slab rule's synthesis and projection in x, each column's operator with
its channel's ``kappa_sq``, the y collar in both absorber schemes, every response family
and the spatial and plasma statistics. Each case is compared at one nonlinear
evaluation, one step and ten fixed steps, under a bound derived before the run
(:func:`amplitude_bound`), and a ``prop_planar`` run is compared end to end through its
HDF5 file.

The cases are vacuum-filled plates of constant index (so no material fit is read and
the responses carry their own density, as the Cartesian ladders'), three slab orders
with the beam off centre in x so that odd and even orders are both driven, and a node
chunk of four y rows over 11, so that the last chunk is partial.
"""

import dataclasses
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.matched import native_thread_fields
from backends.test_radial_nonlinear import CHI3, LENGTH, QuadraticRate, derivative
from bounds import gamma, ladder_bound
from mixture_cases import AnalyticRate
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import SimulationSettings, prop_planar
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output import open_file
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    # The prop_planar case is argon-filled; the FFT padding reaches beyond its fit.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

EPS = np.finfo(float).eps
HALF_GAP = 63.5e-6
ORDERS = (1, 2, 3)
N_Y = 11
ROWS_PER_CHUNK = 4
STEPS = (0, 1, 10)
"""Zero is one nonlinear evaluation, one a single step, ten the qualification run."""

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
PROBE = Path(__file__).with_name("native_planar_thread_probe.py")


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
    space = PlanarGrid(modes, 4e-4, N_Y, window_factor=0.3 if collar else 0.0)
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        waist_x=0.735 * HALF_GAP,
        offset_x=10e-6,
    )
    responses, density = responses_for(
        "field_analytic_kerr" if family == "collar_split_step" else family
    )
    problem = assemble(
        grid,
        None,
        responses,
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        # A reference length short enough that the y collar and the temporal
        # absorber move the answer by more than the rounding compared.
        absorber=absorber(grid, boundary_length=10 * LENGTH) if collar else None,
        absorber_scheme="split_step" if family == "collar_split_step" else "nonlinear",
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / 8,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
        density=density,
    )
    problem = dataclasses.replace(
        problem,
        transform=dataclasses.replace(
            problem.transform, node_chunk=ROWS_PER_CHUNK * space.x.size
        ),
    )
    return grid, space, problem


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


def propagate(problem, grid, implementation, names=()):
    """Run a problem to its end and return the field and requested statistics."""
    backend = implementation(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        field = backend.observe().require_field().copy()
        values = backend.stats(StatsSpec(names=names)) if names else {}
    finally:
        backend.close()
    return field, values


def with_steps(problem, steps):
    """Replace a problem's fixed schedule with ``steps`` uniform steps."""
    schedule = problem.stepper.schedule
    assert schedule is not None
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper,
            schedule=type(schedule).uniform(0, LENGTH, LENGTH / steps),
        ),
    )


@pytest.fixture(params=FAMILIES, scope="module")
def case(request):
    return request.param, *planar_case(request.param)


def test_the_cases_exercise_the_slab_route(case):
    """The problems are the ``"slab"`` kind, with a partial final chunk."""
    family, _, space, problem = case
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "slab"
    assert problem.linear.separable is not None
    assert problem.linear.separable.kappa_sq is not None
    assert problem.transform.synthesis.shape == (space.x.size, len(ORDERS))
    assert N_Y % ROWS_PER_CHUNK != 0
    if family.startswith("collar"):
        assert np.any(geometry.absorber_rate > 0)


@pytest.mark.medium
@pytest.mark.parametrize("steps", STEPS)
def test_the_planar_kind_matches_the_reference(case, steps):
    family, grid, space, problem = case
    bound = amplitude_bound(grid, space, family, steps)
    assert bound > 0
    if steps == 0:
        reference = derivative(NumpyRefBackend, problem, grid)
        actual = derivative(native_class(), problem, grid)
    else:
        problem = with_steps(problem, steps)
        reference, _ = propagate(problem, grid, NumpyRefBackend)
        actual, _ = propagate(problem, grid, native_class())
    scale = float(np.max(np.abs(reference)))
    assert scale > 0
    assert_allclose(actual, reference, rtol=0, atol=bound * scale)


@pytest.mark.medium
@pytest.mark.parametrize("steps", [0, 10])
def test_the_planar_statistics_match_the_reference(case, steps):
    """Energy, intensities, the axis spectrum, widths in y and the plasma records.

    Twice the amplitude bound for squared fields, plus the column and temporal
    reductions (the Cartesian statistics' bound), and the node reductions.
    """
    family, grid, space, problem = case
    bound = 2 * amplitude_bound(grid, space, family, steps)
    bound += gamma(128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to)
    names = SPATIAL_NAMES + (PLASMA_NAMES if family.endswith("plasma") else ())
    if steps:
        problem = with_steps(problem, steps)
    rows = []
    for implementation in (NumpyRefBackend, native_class()):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            if steps:
                backend.propagate_to(LENGTH)
            rows.append(backend.stats(StatsSpec(names=names)))
        finally:
            backend.close()
    reference, actual = rows
    for name in names:
        scale = float(np.max(np.abs(reference[name])))
        assert scale > 0, name
        assert_allclose(actual[name], reference[name], rtol=0, atol=bound * scale)


def test_the_ledger_counts_the_planar_buffers():
    """The y scratch, the slab rule and the wall terms, at their real byte cost."""
    grid, space, problem = planar_case("envelope_kerr")
    backend = native_class()(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    n_x, n_p = space.x.size, space.n_p
    assert ledger["transverse_scratch"] == grid.n_omega * n_p * N_Y * 16
    assert ledger["transform_matrices"] == 2 * n_x * n_p * 8
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    # The free-space vectors, the per-column collar rate, the transverse collar's
    # rate (kernel schema 31, zeros here) and the wall terms.
    expected = (
        4 * grid.n_omega * 16
        + 2 * grid.n_omega * 8
        + 4 * space.n_columns * 8
        + separable.kappa_sq.nbytes
    )
    assert ledger["spatial_coefficients"] == expected
    # One chunk of y rows in time, not every column, and no modal tables.
    assert ledger["modal_time"] == ROWS_PER_CHUNK * n_p * grid.n_to * 16
    assert ledger["fixed_phi_terms"] == ledger["fixed_prefactor_terms"] == 0


@pytest.mark.slow
def test_the_planar_kind_agrees_across_thread_counts(tmp_path):
    """One thread against four, ten Runge-Kutta steps, each in a fresh process.

    The same method on the same fixed schedule, so the two differ by the order of
    their threaded transforms and contractions alone: the ten-step amplitude bound
    (:func:`amplitude_bound`), as against the reference.
    """
    names = ["envelope_kerr", "field_plasma", "collar_split_step"]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    for name in names:
        grid, space, problem = planar_case(name)
        assert problem.stepper.integrator is None
        bound = amplitude_bound(grid, space, name, 10)
        scale = float(np.max(np.abs(one[name])))
        assert scale > 0
        assert_allclose(four[name], one[name], rtol=0, atol=bound * scale)


@pytest.mark.medium
def test_prop_planar_runs_on_native_as_on_the_reference(tmp_path):
    """Ten Kerr steps of an argon-filled gap through ``prop_planar`` into HDF5.

    The fields and every default free-space statistic of the two files agree under
    the ladder bound over the run's 10 cm (the operator's term dominates it).
    """
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 600e-15)
    modes = [SlabMode(HALF_GAP, "Ar", 1.0, order=p) for p in (1, 3, 5)]
    space = PlanarGrid(modes, 1.5e-3, 32)
    launch = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * HALF_GAP,
    )
    records = {}
    for backend in ("numpy_ref", "native"):
        path = tmp_path / f"{backend}.h5"
        settings = SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-12),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=0.01,
            schedule=FixedSchedule(np.full(10, 0.01)),
            backend=backend,
            stats=default_spec("free"),
        )
        prop_planar(
            grid,
            space,
            [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
            launch,
            settings=settings,
            output=path,
            progress="none",
        )
        with open_file(path) as output:
            names = list(default_spec("free").names)
            records[backend] = (
                np.asarray(output.fields("fft")),
                {name: np.asarray(output[f"stats/{name}"]) for name in names},
            )
    (reference, reference_stats), (actual, actual_stats) = (
        records["numpy_ref"],
        records["native"],
    )
    bound = amplitude_bound(grid, space, "envelope_kerr", 10, length=0.1)
    scale = float(np.max(np.abs(reference)))
    assert_allclose(actual, reference, rtol=0, atol=bound * scale)
    statistics_bound = 2 * bound + gamma(
        128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to
    )
    for name, expected in reference_stats.items():
        values = actual_stats[name]
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(np.isfinite(values), finite, err_msg=name)
        if np.any(finite):
            scale = float(np.max(np.abs(expected[finite])))
            assert_allclose(
                values[finite],
                expected[finite],
                rtol=0,
                atol=statistics_bound * scale,
                err_msg=name,
            )
