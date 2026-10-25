# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math
import re
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from rate_oracles import adk_oracle
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import MaterialRangeWarning, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.modes.quadrature import mode_matrix
from solprop.noise import modified_noise
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.profiles import Constant, FromCallable, PressureGradient
from solprop.pulses import GaussianPulse, SechPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, FixedSchedule, PIController, default_dz_min
from solprop.translate.chebyshev import PiecewiseChebyshev, build
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import LinearOperator, constant_phi, profile_table
from solprop.translate.masks import Grids
from solprop.translate.modal import Transform
from solprop.translate.problem import (
    DEFAULT_COMPONENTS,
    SCHEMA_VERSION,
    Flags,
    Initial,
    KernelProblem,
    StepperSettings,
    _walk_arrays,
    assemble,
)
from solprop.translate.responses import CompiledResponses, prefactor_at

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "dev" / "kernel-contract.md"

# The six array groups the contract's own walking rule names; everything else a field
# holds -- Flags, StatsSpec, a Tableau, a RateTable -- is a leaf whose fields are not
# rows of the document. The flags have their own row and their own test below.
GROUPS = (
    Grids,
    LinearOperator,
    Transform,
    CompiledResponses,
    Initial,
    StepperSettings,
)

# The GNLSE's beta is a fitted dispersion rather than a physical effective index, so the
# prefactor takes 1.0 for it and freezes the shock term off (ADR-026).
GNLSE = {"density": 1.0, "shock": False, "prefactor_index": 1.0}

# GNLSE soliton parameters of tests/driver/test_scaffold.py, on a shorter window so the
# whole module stays well under a second.
BETA2 = -11.83e-27
GAMMA = 0.11
TAU0 = 100e-15
WAVELENGTH = 835e-9


@pytest.fixture(scope="module")
def envelope_grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)


@pytest.fixture(scope="module")
def gnlse(envelope_grid):
    return problem_from(
        [0.0, 0.0, BETA2], omega_ref=envelope_grid.omega0, gamma=GAMMA, shock=False
    )


@pytest.fixture(scope="module")
def soliton(envelope_grid, gnlse):
    """The N = 1 soliton problem of the scaffold, assembled for a kernel."""
    peak = abs(BETA2) / (GAMMA * TAU0**2)
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(
        envelope_grid
    )
    return assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.5, 1.0],
        dz_max=0.05,
        dz_init=1e-4,
        **GNLSE,
    )


@pytest.fixture(scope="module")
def field_grid():
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


@pytest.fixture(scope="module")
def capillary():
    """A Marcatili HE11 capillary with a two-point pressure gradient."""
    return MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )


@pytest.fixture(scope="module")
def gradient(field_grid, capillary):
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            field_grid,
            capillary,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1, 0.2],
            dz_max=0.01,
            dz_init=1e-4,
            absorber=absorber(field_grid),
        )


# ------------------------------------------------------------------- the contract test


def contract_field_names(path=CONTRACT):
    """Every field name in the first column of every table of the contract document."""
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cell = line.split("|")[1].strip()
        if not cell or set(cell) <= set("-: "):
            continue
        parts = [part.strip() for part in cell.split(",")]
        if all(re.fullmatch(r"`[a-z_][a-z0-9_]*`", part) for part in parts):
            names.update(part.strip("`") for part in parts)
    return names


def dataclass_field_names(problem):
    """Every contract field of the dataclass tree, by the contract's walking rule."""
    names = set()

    def visit(node):
        for entry in dataclasses.fields(node):
            if entry.name.startswith("_"):
                continue
            names.add(entry.name)
            value = getattr(node, entry.name)
            if isinstance(value, GROUPS):
                visit(value)

    visit(problem)
    return names


def test_the_contract_document_and_the_dataclass_list_the_same_fields(soliton):
    documented = contract_field_names()
    implemented = dataclass_field_names(soliton)
    assert documented - implemented == set(), "documented but not implemented"
    assert implemented - documented == set(), "implemented but not documented"


def test_the_contract_document_lists_every_flag(soliton):
    """The flags row of section 1 names exactly the fields of the Flags dataclass."""
    rows = [
        line
        for line in CONTRACT.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `flags`")
    ]
    assert len(rows) == 1
    documented = set(re.findall(r"`([a-z_]+)`", rows[0].split("|")[3]))
    assert documented == {entry.name for entry in dataclasses.fields(Flags)}


def test_the_document_is_the_version_the_module_writes():
    """A bump of one without the other is exactly what the contract test is for."""
    assert SCHEMA_VERSION == 36
    assert soliton_schema_version_row().startswith("incremented on every change")
    assert f"current version is **{SCHEMA_VERSION}**" in CONTRACT.read_text(
        encoding="utf-8"
    )


def soliton_schema_version_row():
    """The meaning column of the schema_version row, as the document states it."""
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `schema_version`"):
            return line.split("|")[3].strip()
    raise AssertionError("the contract document has no schema_version row")


# ------------------------------------------------------------------------- assembly


def test_the_soliton_problem_assembles(soliton, envelope_grid):
    assert soliton.schema_version == SCHEMA_VERSION
    assert soliton.grid_kind == "envelope"
    assert soliton.shape == (1, 1, 1, envelope_grid.n_omega)
    assert soliton.flags == Flags(
        single_mode=True,
        scalar_pol=True,
        single_batch=True,
        has_plasma=False,
        has_raman=False,
        has_spectral_responses=False,
        has_noise=False,
        has_temporal_absorber=False,
        constant_linop=True,
        mode_averaged=True,
    )
    # A dispersion-specified fibre does not vary along z, so the table is the exact
    # one-interval fast path rather than an adaptive build.
    assert soliton.linear.n_intervals == 1
    assert soliton.linear.phi_degrees.tolist() == [1]
    assert soliton.stepper.z_end == 1.0
    assert soliton.transform.node_chunk == 1


def test_the_gradient_capillary_assembles(gradient, field_grid, capillary):
    assert gradient.grid_kind == "field"
    assert gradient.shape == (1, 1, 1, field_grid.n_omega)
    assert not gradient.flags.constant_linop
    assert gradient.flags.has_temporal_absorber
    assert gradient.flags.mode_averaged
    # The fill's stage boundary is a hard kink and nothing else is.
    assert gradient.linear.hard_kinks.tolist() == [0.2]
    # The table covers the run and its margin, and the profiles cover the same range.
    assert gradient.linear.z_span == pytest.approx((0.0, 0.21))
    assert gradient.responses.density.z_span == gradient.linear.z_span
    assert gradient.transform.aeff.z_span == gradient.linear.z_span
    # The density the Kerr coefficient is multiplied by is the mode's own path.
    z = np.array([0.0, 0.1, 0.2])
    expected = capillary.density(z)
    got = np.array([float(gradient.responses.density(float(one))) for one in z])
    # The density table is built to the Chebyshev tolerance, which bounds the uniform
    # error relative to the value scale; 1e-12 leaves an order over it.
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_every_array_that_crosses_the_seam_is_frozen(gradient):
    arrays = _walk_arrays(gradient)
    assert arrays, "the walk found no arrays at all"
    for name, values in arrays:
        assert not values.flags.writeable, name
        assert values.flags.c_contiguous, name
        assert values.dtype in (
            np.float64,
            np.complex128,
            np.bool_,
            np.intp,
            np.int32,
        ), name


def test_the_polarisation_taper_is_off_unless_asked_for(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    request = {
        "controller": PIController(rtol=1e-8, atol=1e-14),
        "save_planes": [1.0],
        "dz_max": 0.05,
        "dz_init": 1e-4,
        **GNLSE,
    }
    plain = assemble(envelope_grid, gnlse.mode, [gnlse.kerr], start, **request)
    assert np.all(plain.grids.omega_window == 1.0)
    tapered = assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        polarisation_taper=True,
        **request,
    )
    assert np.array_equal(tapered.grids.omega_window, envelope_grid.omega_window)


def test_a_batched_initial_state_keeps_its_batch_axis(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    batched = np.stack([start, 2.0 * start])
    problem = assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        batched,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        batch_params={"energy": [1.0, 4.0]},
        **GNLSE,
    )
    assert problem.shape[0] == 2
    assert not problem.flags.single_batch
    assert problem.initial.batch_params["energy"].tolist() == [1.0, 4.0]
    with pytest.raises(TypeError):
        # A mapping proxy has no __setitem__, which is the point: the record a backend
        # is handed cannot be edited behind its back.
        problem.initial.batch_params["energy"] = np.zeros(2)  # pyright: ignore[reportIndexIssue]


def test_a_noise_realisation_is_carried_in_the_canonical_layout(envelope_grid, gnlse):
    draw = modified_noise(envelope_grid, np.random.default_rng(3))
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    problem = assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        noise=draw,
        **GNLSE,
    )
    assert problem.flags.has_noise
    assert problem.initial.noise_model == "modified"
    assert problem.initial.noise0 is not None
    assert problem.initial.noise0.shape == problem.shape
    assert np.array_equal(problem.initial.noise0[0], draw.spectrum)


def test_the_defaults_follow_the_stepper_design(soliton):
    assert soliton.stepper.tableau is DOPRI5
    assert soliton.stepper.dz_min == default_dz_min(1.0)
    assert soliton.stepper.stretch == 1.1
    assert soliton.stepper.propagator_mode == "auto"
    assert soliton.stepper.schedule is None


def test_a_fixed_schedule_that_lands_on_the_planes_is_accepted(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    schedule = FixedSchedule.uniform(0.0, 1.0, 0.05, save_planes=[0.5, 1.0])
    problem = assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.5, 1.0],
        dz_max=0.05,
        dz_init=1e-4,
        schedule=schedule,
        **GNLSE,
    )
    assert problem.stepper.schedule is schedule


# --------------------------------------------------------------- validate rejects


def frozen(values):
    """Return a read-only, C-contiguous copy, as every array crossing the seam is."""
    copy = np.ascontiguousarray(values)
    copy.setflags(write=False)
    return copy


def corrupt(problem, group, name, value):
    """Overwrite one field of one group of an assembled problem, bypassing frozen.

    A backend must not trust a problem it is handed, and every group refuses a bad
    value at construction, so the only way to reach `validate`'s own guards is to
    corrupt a problem the way a stale cache or a hand-built object would.
    """
    copy = dataclasses.replace(problem)
    member = dataclasses.replace(getattr(copy, group))
    object.__setattr__(member, name, value)
    object.__setattr__(copy, group, member)
    return copy


def test_validate_rejects_a_wrong_schema_version(soliton):
    stale = dataclasses.replace(soliton)
    object.__setattr__(stale, "schema_version", SCHEMA_VERSION + 1)
    with pytest.raises(TranslationError, match="schema_version"):
        stale.validate()


def test_validate_rejects_a_not_finite_array(soliton):
    spoiled = np.array(soliton.grids.omega, copy=True)
    spoiled[3] = np.nan
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="not finite"):
        corrupt(soliton, "grids", "omega", spoiled).validate()


def test_validate_admits_infinite_energy_window_edges_but_not_nan(soliton):
    # A half-open window may run to either end of the axis, as the manual shows.
    windows = np.array([[-np.inf, 2.4e15], [2.4e15, np.inf]])
    spec = StatsSpec(names=("energy_window",), energy_windows=windows)
    corrupt(soliton, "stats_spec", "energy_windows", spec.energy_windows).validate()
    spoiled = np.array([[np.nan, 2.4e15]])
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="not finite"):
        corrupt(soliton, "stats_spec", "energy_windows", spoiled).validate()


def test_validate_rejects_a_wrong_dtype(soliton):
    spoiled = np.asarray(soliton.grids.omega, dtype=np.float32)
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="dtype"):
        corrupt(soliton, "grids", "omega", spoiled).validate()


def test_validate_rejects_a_non_contiguous_array(soliton):
    spoiled = np.repeat(np.asarray(soliton.grids.omega), 2)[::2]
    spoiled.setflags(write=False)
    assert not spoiled.flags.c_contiguous
    with pytest.raises(TranslationError, match="C-contiguous"):
        corrupt(soliton, "grids", "omega", spoiled).validate()


def test_validate_rejects_a_writeable_array(soliton):
    with pytest.raises(TranslationError, match="writeable"):
        corrupt(
            soliton, "grids", "omega", np.array(soliton.grids.omega, copy=True)
        ).validate()


def test_validate_rejects_a_non_boolean_mask(soliton):
    spoiled = np.asarray(soliton.grids.active_mask, dtype=np.float64)
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="must be Boolean"):
        corrupt(soliton, "grids", "active_mask", spoiled).validate()


@pytest.mark.parametrize(
    ("group", "name", "shape", "message"),
    [
        ("initial", "field0", (1, 1, 1, 7), "field0 has shape"),
        ("grids", "omega", (7,), "bins, expected"),
    ],
)
def test_validate_rejects_an_inconsistent_shape(soliton, group, name, shape, message):
    dtype = getattr(getattr(soliton, group), name).dtype
    spoiled = np.zeros(shape, dtype=dtype)
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match=message):
        corrupt(soliton, group, name, spoiled).validate()


def test_validate_rejects_a_phi_table_of_the_wrong_value_shape(soliton):
    """Whichever representation carries the operator, its value is the state."""
    spoiled = np.zeros((1, 2, 1, 1, 1, 3), dtype=np.complex128)
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="linear carries values"):
        corrupt(soliton, "linear", "phi_coefficients", spoiled).validate()


def test_validate_rejects_a_transform_with_the_wrong_mode_count(soliton):
    spoiled = np.zeros((1, 2))
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="transform carries"):
        corrupt(soliton, "transform", "synthesis", spoiled).validate()


def test_validate_rejects_a_temporal_rate_of_the_wrong_length(gradient):
    spoiled = np.zeros(7)
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="temporal_absorber has"):
        corrupt(gradient, "responses", "temporal_absorber", spoiled).validate()


def test_validate_rejects_breakpoints_that_do_not_increase(soliton):
    spoiled = np.array([0.0, 2.0, 1.0])
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="strictly increasing"):
        corrupt(soliton, "linear", "phi_breakpoints", spoiled).validate()


def test_validate_rejects_a_table_that_does_not_cover_the_run(soliton):
    spoiled = np.array([0.0, 0.5])
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="does not contain the run"):
        corrupt(soliton, "linear", "phi_breakpoints", spoiled).validate()


def test_validate_rejects_a_kink_that_is_not_an_interval_edge(gradient):
    spoiled = np.array([0.123])
    spoiled.setflags(write=False)
    with pytest.raises(TranslationError, match="is not an interval edge"):
        corrupt(gradient, "linear", "hard_kinks", spoiled).validate()


def test_validate_rejects_a_profile_table_over_a_different_range(soliton, gradient):
    with pytest.raises(TranslationError, match="every table along z must span"):
        corrupt(soliton, "responses", "density", gradient.responses.density).validate()


@pytest.mark.parametrize(
    "density",
    [
        # Negative at its end, by far more than rounding: not a declared vacuum.
        lambda u: 5e25 * (1.0 - u) - 5e19,
        # Exactly zero at both ends, which tabulate to exact zeros, and negative
        # between them.
        lambda u: 5e25 * u * (u - 0.5) * (1.0 - u),
    ],
    ids=["below zero at its end", "below zero between vacuum ends"],
)
def test_validate_still_rejects_a_negative_density(gradient, density):
    start, end = gradient.responses.density.z_span
    path = FromCallable(lambda z: density(z / 0.2), (0.0, 0.2))
    table = profile_table(path, start, end)
    with pytest.raises(TranslationError, match=r"total density must lie in \[0"):
        corrupt(gradient, "responses", "density", table).validate()


def test_validate_walks_the_raman_rate_tables_too(raman_soliton, gradient):
    # The decay-rate tables of a Raman kernel are tables along z like the density, and
    # validate checks their span with the rest; a stale one from another run is refused
    # by name, through the leaf that holds it.
    stale = dataclasses.replace(
        raman_soliton.responses.raman_coefficients,
        decay_rates=(gradient.responses.density,),
    )
    with pytest.raises(
        TranslationError,
        match=r"raman_terms\[0\]\.decay_rates\[0\] covers",
    ):
        corrupt(raman_soliton, "responses", "raman_terms", (stale,)).validate()


@pytest.fixture(scope="module")
def raman_soliton(envelope_grid, gnlse):
    """The soliton problem with a fused-silica Raman response beside its Kerr term."""
    peak = abs(BETA2) / (GAMMA * TAU0**2)
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(
        envelope_grid
    )
    return assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr, response_for("SiO2", amplitude=1e-22, fraction=0.18)],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.5, 1.0],
        dz_max=0.05,
        dz_init=1e-4,
        **GNLSE,
    )


def test_validate_accepts_the_compiled_raman_rows(raman_soliton):
    assert raman_soliton.flags.has_raman
    assert raman_soliton.validate() is None


def test_validate_rejects_a_kernel_that_is_not_on_the_doubled_grid(raman_soliton):
    # The convolution is built from n_to; a kernel of any other length would be
    # convolved against the wrong axis rather than raise.
    leaf = raman_soliton.responses.raman_coefficients
    stale = dataclasses.replace(leaf, undamped=np.zeros((1, raman_soliton.grids.n_to)))
    with pytest.raises(TranslationError, match="on the doubled fine grid"):
        corrupt(raman_soliton, "responses", "raman_terms", (stale,)).validate()


def test_the_leaf_refuses_a_kernel_with_no_table_for_a_row(raman_soliton):
    leaf = raman_soliton.responses.raman_coefficients
    with pytest.raises(TranslationError, match="one rate per row"):
        dataclasses.replace(leaf, undamped=np.zeros((2, 2 * raman_soliton.grids.n_to)))


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_the_leaf_refuses_a_fraction_outside_its_range(raman_soliton, fraction):
    leaf = raman_soliton.responses.raman_coefficients
    with pytest.raises(TranslationError, match="fraction must lie in"):
        dataclasses.replace(leaf, fraction=fraction)


def test_validate_rejects_a_raman_kernel_on_a_vector_problem(raman_soliton):
    # The vector form needs a tensor kernel no backend can consume, and the
    # compiler refuses it, so only a hand-built problem reaches this guard.
    transform = dataclasses.replace(raman_soliton.transform)
    object.__setattr__(transform, "_n_pol", 2)
    stale = dataclasses.replace(raman_soliton)
    object.__setattr__(stale, "transform", transform)
    with pytest.raises(TranslationError, match="tensor"):
        stale.validate()


def test_validate_rejects_a_schedule_that_misses_a_save_plane(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="does not land on the save plane"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.37, 1.0],
            dz_max=0.05,
            dz_init=1e-4,
            schedule=FixedSchedule.uniform(0.0, 1.0, 0.05, save_planes=[1.0]),
            **GNLSE,
        )


def test_validate_rejects_a_schedule_that_misses_a_hard_kink(field_grid, capillary):
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with (
        warnings.catch_warnings(),
        pytest.raises(TranslationError, match="does not land on the hard kink"),
    ):
        warnings.simplefilter("ignore", MaterialRangeWarning)
        assemble(
            field_grid,
            capillary,
            [],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.25],
            dz_max=0.01,
            dz_init=1e-4,
            # 0.25 in near-equal steps of at most 0.03 puts no edge within 5 mm of
            # the fill's boundary at 0.2 m, which a fixed schedule must land on.
            schedule=FixedSchedule.uniform(0.0, 0.25, 0.03, save_planes=[0.25]),
        )


def test_validate_rejects_a_prefactor_of_the_wrong_value_shape(soliton):
    spoiled = dataclasses.replace(
        soliton.responses.prefactor,
        coefficients=soliton.responses.prefactor.coefficients[..., :7],
    )
    with pytest.raises(TranslationError, match="prefactor carries values"):
        corrupt(soliton, "responses", "prefactor", spoiled).validate()


def test_validate_rejects_a_prefactor_table_over_a_different_range(soliton):
    # The same values over half the run: the shape is right and the span is not.
    edges = np.array([0.0, 0.5])
    edges.setflags(write=False)
    short = dataclasses.replace(
        cast("PiecewiseChebyshev", soliton.responses.prefactor), breakpoints=edges
    )
    with pytest.raises(TranslationError, match=r"responses\.prefactor covers"):
        corrupt(soliton, "responses", "prefactor", short).validate()


def test_validate_rejects_a_constant_prefactor_beside_a_varying_prefactor(gradient):
    """A gradient's prefactor varies, so no array may stand in for it."""
    constant = np.ascontiguousarray(prefactor_at(gradient.responses, 0.0))
    constant.setflags(write=False)
    with pytest.raises(TranslationError, match="fast path is only for a prefactor"):
        corrupt(gradient, "responses", "prefactor_constant", constant).validate()


def test_validate_rejects_a_constant_prefactor_that_disagrees_with_the_rest(soliton):
    wrong = np.ascontiguousarray(2.0 * prefactor_at(soliton.responses, 0.0))
    wrong.setflags(write=False)
    with pytest.raises(TranslationError, match="does not agree with the prefactor"):
        corrupt(soliton, "responses", "prefactor_constant", wrong).validate()


def test_validate_allows_a_fixed_fibre_to_go_without_the_fast_path(soliton):
    # The fast path is a claim about the prefactor table, not about the operator: a
    # synthetic z-varying Phi on a fixed fibre, or a fixed fibre read through its table,
    # both validate.
    corrupt(soliton, "responses", "prefactor_constant", None).validate()


# ------------------------------------------------------------------------ immutability


def test_the_problem_cannot_be_mutated(soliton):
    with pytest.raises(dataclasses.FrozenInstanceError):
        soliton.schema_version = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        soliton.grids.dt = 1.0
    with pytest.raises(ValueError, match="read-only"):
        soliton.grids.omega[0] = 0.0


# --------------------------------------------------------------- assemble rejects


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"save_planes": [0.0]}, "must contain a positive position"),
        ({"save_planes": []}, "must contain a positive position"),
        ({"dz_max": 1e-9, "dz_init": 1e-3}, "dz_min <= dz_init <= dz_max"),
        ({"stretch": 0.5}, "stretch must be at least 1"),
    ],
)
def test_assemble_rejects_a_bad_request(envelope_grid, gnlse, overrides, message):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    request = {
        "controller": PIController(rtol=1e-8, atol=1e-14),
        "save_planes": [1.0],
        "dz_max": 0.05,
        "dz_init": 1e-4,
        **GNLSE,
    } | overrides
    with pytest.raises(TranslationError, match=message):
        assemble(envelope_grid, gnlse.mode, [gnlse.kerr], start, **request)


def test_assemble_rejects_a_state_of_the_wrong_rank(envelope_grid, gnlse):
    with pytest.raises(TranslationError, match="field0 must have shape"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            np.zeros(envelope_grid.n_omega, dtype=complex),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            **GNLSE,
        )


def test_assemble_rejects_a_state_of_the_wrong_width(envelope_grid, gnlse):
    with pytest.raises(TranslationError, match="spectral bins, expected"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            np.zeros((1, 1, 7), dtype=complex),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            **GNLSE,
        )


def test_assemble_rejects_a_mode_set_the_state_does_not_match(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="modes but 2 were given"):
        assemble(
            envelope_grid,
            [gnlse.mode, gnlse.mode],
            [gnlse.kerr],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            **GNLSE,
        )


def test_assemble_rejects_a_mode_set_without_a_quadrature(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="without a transverse quadrature"):
        assemble(
            envelope_grid,
            [gnlse.mode, gnlse.mode],
            [gnlse.kerr],
            np.stack([start[0], start[0]]),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            **GNLSE,
        )


def test_a_batch_parameter_must_have_one_value_per_member(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="one per batch member"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            batch_params={"energy": [1.0, 2.0]},
            **GNLSE,
        )


# --------------------------------------------------------------------------- the groups


def test_the_initial_group_refuses_a_noise_field_of_the_wrong_shape():
    with pytest.raises(TranslationError, match="noise0 has shape"):
        Initial(
            np.zeros((1, 1, 1, 8), complex),
            np.zeros((1, 1, 1, 4), complex),
            "input",
            {},
        )


def test_the_initial_group_refuses_a_model_without_a_realisation():
    with pytest.raises(TranslationError, match="noise_model"):
        Initial(np.zeros((1, 1, 1, 8), complex), None, "input", {})


def test_the_stepper_group_refuses_planes_that_do_not_increase():
    with pytest.raises(TranslationError, match="non-empty and increasing"):
        StepperSettings(
            DOPRI5,
            PIController(rtol=1e-8, atol=1e-14),
            1e-9,
            0.1,
            1e-3,
            1.1,
            None,
            [1.0, 0.5],
            "tabulate",
        )


def test_the_stats_request_is_the_catalogue_object(envelope_grid, gnlse, soliton):
    # Section 8 is solprop.stats.StatsSpec itself, validated against the catalogue at
    # construction; a problem carries the caller's spec unchanged and an empty one
    # otherwise.
    assert soliton.stats_spec.names == ()
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=5)
    problem = assemble(
        envelope_grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        stats_spec=spec,
        **GNLSE,
    )
    assert problem.stats_spec is spec


def test_a_kernel_problem_can_be_built_by_hand_from_its_groups(soliton):
    """The contract is a dataclass, so a backend test can substitute one group."""
    rebuilt = KernelProblem(
        schema_version=soliton.schema_version,
        grid_kind=soliton.grid_kind,
        shape=soliton.shape,
        flags=soliton.flags,
        grids=soliton.grids,
        linear=dataclasses.replace(
            constant_phi(
                np.zeros(soliton.shape[3], dtype=complex), soliton.stepper.z_end
            ),
            power_weight_representation=soliton.linear.power_weight_representation,
            power_weight=soliton.linear.power_weight,
            power_weight_constant=soliton.linear.power_weight_constant,
        ),
        transform=soliton.transform,
        responses=soliton.responses,
        initial=soliton.initial,
        stepper=soliton.stepper,
        stats_spec=soliton.stats_spec,
    )
    rebuilt.validate()
    assert rebuilt.linear.phi_representation == "constant"
    assert math.isclose(rebuilt.linear.z_span[1], 1.05)


def test_the_initial_group_checks_a_batch_parameter_it_is_handed_directly():
    with pytest.raises(TranslationError, match="one per batch member"):
        Initial(
            np.zeros((2, 1, 1, 8), complex),
            None,
            "none",
            {"energy": np.array([1.0, 2.0, 3.0])},
        )


def test_a_save_plane_at_or_below_zero_is_refused(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="save_planes must be positive"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[-0.5, 1.0],
            dz_max=0.05,
            dz_init=1e-4,
            **GNLSE,
        )


def test_a_multimode_problem_assembles_with_a_quadrature(field_grid):
    """A multimode problem assembles with a quadrature; the contract carries the set."""
    modes = [MarcatiliMode(125e-6, "He", 1.0, m=m, cladding=1.45) for m in (1, 2, 3)]
    rule = transverse_quadrature(modes, n_r=10, radial_rule="gauss_kronrod")
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            field_grid,
            modes,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            np.repeat(start, 3, axis=0),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.2],
            dz_max=0.01,
            dz_init=1e-4,
            quadrature=rule,
        )
    assert problem.shape == (1, 3, 1, field_grid.n_omega)
    assert not problem.flags.single_mode
    assert not problem.flags.mode_averaged
    assert problem.transform.n_nodes == rule.n_nodes
    assert problem.responses.prefactor_value_shape == (3, 1, field_grid.n_omega)
    assert problem.responses.prefactor_constant is not None
    # The synthesis must actually carry the mode fields: `components` defaults to the
    # component an HE_1m mode points along, and the y component of every one of them is
    # identically zero (ADR-031).
    assert problem.transform.n_pol == 1
    assert np.max(np.abs(problem.transform.synthesis)) > 0.0


def test_the_default_components_are_the_ones_the_modes_point_along(field_grid):
    """An HE_1m set has no y component at all, so ("y",) would synthesise nothing."""
    modes = [MarcatiliMode(125e-6, "He", 1.0, m=m, cladding=1.45) for m in (1, 2)]
    rule = transverse_quadrature(modes, n_r=10, radial_rule="gauss_kronrod")
    matrices = mode_matrix(modes, rule, 0.0, components=DEFAULT_COMPONENTS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = _multimode_problem(field_grid, modes, rule)
    assert DEFAULT_COMPONENTS == ("x",)
    assert (
        np.max(np.abs(mode_matrix(modes, rule, 0.0, components=("y",)).synthesis))
        == 0.0
    )
    # The stored synthesis is the mode matrix times the oversampling pad scale, which
    # is exact in binary for a power-of-two ratio, so the comparison is exact.
    assert np.array_equal(
        problem.transform.synthesis, matrices.synthesis * field_grid.pad_scale
    )


def test_both_components_resolved_gives_a_two_component_transform(field_grid):
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    modes = [mode, mode.rotated(np.pi / 2)]
    rule = transverse_quadrature(modes, n_r=10, radial_rule="gauss_kronrod")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = _multimode_problem(field_grid, modes, rule, components=("x", "y"))
    assert problem.transform.n_pol == 2
    assert problem.transform.n_nodes == rule.n_nodes
    assert problem.transform.synthesis.shape == (2 * rule.n_nodes, 2)


@pytest.mark.parametrize(
    "angles, n_components",
    [
        ([np.pi / 2], 1),
        ([0.0, np.pi / 2], 2),
        ([np.pi / 4], 2),
    ],
)
def test_omitted_components_resolve_the_nonzero_union(field_grid, angles, n_components):
    modes = [MarcatiliMode(125e-6, phi=phi, cladding=1.45) for phi in angles]
    rule = transverse_quadrature(modes, n_r=10)
    problem = _multimode_problem(field_grid, modes, rule)
    assert problem.transform.n_pol == n_components
    assert np.max(np.abs(problem.transform.synthesis)) > 0


@pytest.mark.parametrize("phi, component", [(0.0, "y"), (np.pi / 2, "x")])
def test_identically_zero_components_are_refused(field_grid, phi, component):
    modes = [MarcatiliMode(125e-6, phi=phi, cladding=1.45)]
    rule = transverse_quadrature(modes, n_r=10)
    with pytest.raises(TranslationError, match=r"identically zero.*phi="):
        _multimode_problem(field_grid, modes, rule, components=(component,))


def test_inferred_vector_rows_require_an_explicit_product_choice(field_grid):
    modes = [MarcatiliMode(125e-6, phi=np.pi / 4, cladding=1.45)]
    rule = transverse_quadrature(modes, n_r=10)
    with pytest.raises(TranslationError, match="carry the transverse components twice"):
        _multimode_problem(field_grid, modes, rule, n_pol=2)
    assert (
        _multimode_problem(field_grid, modes, rule, components=("y",), n_pol=2).shape[2]
        == 2
    )


def test_components_without_a_quadrature_are_refused(envelope_grid, gnlse):
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(
        envelope_grid
    )
    with pytest.raises(TranslationError, match="resolves no components"):
        assemble(
            envelope_grid,
            gnlse.mode,
            [gnlse.kerr],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1.0],
            dz_max=0.05,
            dz_init=1e-4,
            components=("x", "y"),
            **GNLSE,
        )


@pytest.mark.parametrize("bad", [(), ("z",), ("x", "r")])
def test_components_outside_the_transverse_pair_are_refused(field_grid, bad):
    modes = [MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)]
    rule = transverse_quadrature(modes, n_r=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(TranslationError, match="non-empty subset"):
            _multimode_problem(field_grid, modes, rule, components=bad)


def test_a_state_and_a_transform_cannot_both_carry_the_components(field_grid):
    """The components live in the transform's rows or in the state, never in both."""
    modes = [MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)]
    rule = transverse_quadrature(modes, n_r=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(TranslationError, match="with_polarisations"):
            _multimode_problem(field_grid, modes, rule, components=("x", "y"), n_pol=2)


def _multimode_problem(grid, modes, rule, *, components=None, n_pol=1):
    """Assemble a multimode problem on the given rule, for the tests above."""
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    field0 = np.repeat(np.repeat(start, len(modes), axis=0), n_pol, axis=1)
    return assemble(
        grid,
        modes,
        [KerrResponse(chi3=2.2e-52, thg=True)],
        field0,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.2],
        dz_max=0.01,
        dz_init=1e-4,
        quadrature=rule,
        components=components,
    )


# ------------------------------------------------------------------- the plasma group

PLASMA_FIELD_RANGE = (1e9, 1e11)  # V/m, the span the rate table covers


@pytest.fixture
def plasma_problem(field_grid):
    """A gradient capillary carrying a compiled plasma response."""
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            field_grid,
            mode,
            [
                KerrResponse(chi3=2.2e-52, thg=True),
                PlasmaResponse(Gas.Ar, adk_oracle(Gas.Ar)),
            ],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=1e-4,
            ionisation_field_range=PLASMA_FIELD_RANGE,
        )


def test_a_compiled_plasma_validates_and_flags_itself(plasma_problem):
    assert plasma_problem.flags.has_plasma
    plasma_problem.validate()


def test_validate_rejects_a_plasma_on_an_envelope_grid(plasma_problem):
    moved = dataclasses.replace(plasma_problem)
    object.__setattr__(moved, "grid_kind", "envelope")
    with pytest.raises(TranslationError, match="requires a 'field' grid"):
        moved.validate()


def test_validate_rejects_two_rate_tables(plasma_problem):
    # Frozen, because the array checks run first and a fresh table is writeable.
    built = build_rate_table(adk_oracle(Gas.Ne), PLASMA_FIELD_RANGE)
    frozen = {}
    for name in ("log_rate", "coefficients"):
        values = np.ascontiguousarray(getattr(built, name))
        values.setflags(write=False)
        frozen[name] = values
    other = dataclasses.replace(built, **frozen)
    spoiled = corrupt(plasma_problem, "responses", "ionisation_table", (other,))
    with pytest.raises(TranslationError, match="is not the rate table"):
        spoiled.validate()


def test_validate_checks_the_pre_ionisation_table_spans_the_run(plasma_problem):
    plasma = plasma_problem.responses.plasma_coefficients
    assert plasma is not None
    short = dataclasses.replace(plasma)
    object.__setattr__(
        short, "preionisation", profile_table(Constant(0.0), 0.0, 0.5 * 0.105)
    )
    spoiled = corrupt(plasma_problem, "responses", "plasma_channels", (short,))
    with pytest.raises(TranslationError, match="preionisation"):
        spoiled.validate()


def test_the_plasma_group_s_arrays_are_frozen(plasma_problem):
    named = dict(_walk_arrays(plasma_problem))
    plasma = [name for name in named if "plasma_channels" in name]
    assert plasma
    for name in plasma:
        assert not named[name].flags.writeable, name


def test_validate_rejects_a_transform_and_a_state_that_both_carry_the_components(
    field_grid,
):
    """The `assemble` guard again, for a problem built by hand around it.

    A mode-averaged vector run is the legitimate half of the rule -- the transform is
    one scalar factor and the state carries the two components -- so bumping the
    transform's own count is the smallest inconsistency `validate` has to catch.
    """
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            field_grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            np.repeat(start, 2, axis=1),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.2],
            dz_max=0.01,
            dz_init=1e-4,
        )
    assert problem.shape[2] == 2
    doubled = corrupt(problem, "transform", "_n_pol", 2)
    with pytest.raises(TranslationError, match="with_polarisations"):
        doubled.validate()


def test_validate_rejects_a_carrier_without_a_harmonic_term(soliton):
    """The carrier is present exactly when the term that needs it is compiled."""
    stray = np.ones(soliton.grids.n_to, dtype=np.complex128)
    stray.setflags(write=False)
    with pytest.raises(TranslationError, match="carrier_2w0 must be present"):
        corrupt(soliton, "grids", "carrier_2w0", stray).validate()


def test_validate_rejects_a_harmonic_term_without_its_carrier():
    """A third-harmonic envelope problem carries the carrier, and needs it."""
    grid = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12, thg=True)
    gnlse = problem_from(
        [0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False
    )
    start = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=1.0).build(grid)
    problem = assemble(
        grid,
        gnlse.mode,
        [dataclasses.replace(gnlse.kerr, thg=True)],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        **GNLSE,
    )
    assert problem.grids.carrier_2w0 is not None
    with pytest.raises(TranslationError, match="carrier_2w0 must be present"):
        corrupt(problem, "grids", "carrier_2w0", None).validate()


def test_validate_rejects_a_constant_prefactor_beside_a_varying_expansion(gradient):
    """The same claim as for a table, made of the representation a gradient gets."""
    assert gradient.responses.prefactor_representation == "expansion"
    constant = np.ascontiguousarray(prefactor_at(gradient.responses, 0.0))
    constant.setflags(write=False)
    with pytest.raises(TranslationError, match="prefactor expansion"):
        corrupt(gradient, "responses", "prefactor_constant", constant).validate()


def test_validate_rejects_a_constant_prefactor_beside_a_varying_table(gradient):
    """The same claim again, of the representation a fallback problem would get.

    Nothing this package builds today takes the z table for a capillary's prefactor, so
    the table is built here from the values the expansion gives and substituted, the way
    a stale cache or a hand-built problem would present one.
    """
    span = gradient.responses.prefactor_span
    built = build(
        lambda z: np.stack([prefactor_at(gradient.responses, float(one)) for one in z]),
        *span,
        tol=1e-10,
    )
    varying = PiecewiseChebyshev(
        frozen(built.coefficients),
        frozen(built.breakpoints),
        frozen(built.degrees),
        frozen(built.hard_kinks),
    )
    # All four at once: the group refuses a kind without its member, so the swap has
    # to be one consistent step and only `validate` is left to complain. The kind is
    # ``"constant"`` because that is what claims the fast path; a ``"table"`` kind
    # cannot carry a constant at all, and the group says so before `validate` runs.
    responses = dataclasses.replace(
        gradient.responses,
        prefactor_representation="constant",
        prefactor=varying,
        prefactor_expansion=None,
        prefactor_constant=frozen(varying(span[0])),
    )
    spoiled = dataclasses.replace(gradient)
    object.__setattr__(spoiled, "responses", responses)
    with pytest.raises(TranslationError, match="prefactor table"):
        spoiled.validate()


@pytest.mark.parametrize("failure", ["absent", "shape", "positive", "constant"])
def test_modal_power_weight_refuses_an_inconsistent_row(soliton, failure):
    linear = soliton.linear
    table = linear.power_weight
    assert table is not None
    if failure == "absent":
        changed = dataclasses.replace(linear, power_weight=None)
        message = "power_weight representation"
    elif failure == "shape":
        changed = dataclasses.replace(
            linear,
            power_weight=dataclasses.replace(
                table, coefficients=table.coefficients[..., :2]
            ),
        )
        message = "power_weight shape"
    elif failure == "positive":
        negative = -table.coefficients
        negative.setflags(write=False)
        changed = dataclasses.replace(
            linear,
            power_weight=dataclasses.replace(table, coefficients=negative),
        )
        message = "power_weight must be positive"
    else:
        assert linear.power_weight_constant is not None
        changed = dataclasses.replace(
            linear, power_weight_constant=2 * linear.power_weight_constant
        )
        message = "power_weight_constant must match"
    with pytest.raises(TranslationError, match=message):
        dataclasses.replace(soliton, linear=changed).validate()


@pytest.fixture(scope="module")
def radial_problem():
    """A short radial run in uniform argon, translated with its plane measure."""
    grid = EnvelopeGrid.from_limits(1e-3, WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=1e-9), 60e-6
    )
    simulation = Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 16),
        SpatialMedium("Ar", pressure=1.0),
        [],
        pulse,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-12),
            save_planes=[1e-3],
            dz_max=2.5e-4,
            dz_init=2.5e-4,
            backend="numpy_ref",
        ),
    )
    return simulation.translate()


def test_validate_pairs_the_plane_measure_with_spatial_geometry(
    soliton, radial_problem
):
    """A spatial problem needs its plane measure on its band, and others carry none."""
    flux = radial_problem.flux
    assert flux is not None
    radial_problem.validate()
    with pytest.raises(TranslationError, match="requires its plane measure"):
        dataclasses.replace(radial_problem, flux=None).validate()
    with pytest.raises(TranslationError, match="requires its plane measure"):
        dataclasses.replace(soliton, flux=flux).validate()
    narrowed = np.array(flux.base.active, copy=True)
    narrowed[np.flatnonzero(narrowed)[0]] = False
    narrowed.setflags(write=False)
    stale = dataclasses.replace(
        flux, base=dataclasses.replace(flux.base, active=narrowed)
    )
    with pytest.raises(TranslationError, match="active mask must match"):
        dataclasses.replace(radial_problem, flux=stale).validate()
