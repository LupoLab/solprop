# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coherent wall route end to end.

The leaf, its admission at three doors, and what ten steps of it do. Every bound is
stated in the docstring of the test that takes it, before the measurement.

The route is `prop_capillary(..., wall=CoherentWall())`: the natural home, because the
capillary route already builds the contiguous `HE_1m` set the condensation needs, its
gas fill, its quadrature and its responses, while the general modal route has no
capillary provider at all (`ModalGuide.from_provider` takes a `ModalProvider`, and the
two that exist are the step-index and graded-index solvers). Declaring the wall there
also makes the uncoupled route the matched control it has to be compared against.
"""

import itertools
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal, cast

import mpmath
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.special import jn_zeros, jv

import solprop.translate.problem as problem_module
from bounds import assert_within, fft_bound, gamma, step_evaluations
from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER
from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.admission import refuse_coupled_flow
from solprop.backends.numpy_ref.backend import (
    _MAX_FLOW_SQUARINGS,
    _coupled_flow_maps,
    _triangular_exponential,
)
from solprop.backends.numpy_ref.backend import _cf4_action as _backend_cf4_action
from solprop.boundaries import absorber
from solprop.constants import EPS0, C
from solprop.driver import Simulation, prop_capillary, prop_gnlse
from solprop.driver.simulation import Pressure
from solprop.exceptions import (
    AdiabaticTaperWarning,
    BackendError,
    DriverError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    ModeError,
    OutputError,
    SolpropWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.linear import DEFAULT_LOSS_POLICY
from solprop.modes import (
    CoherentWall,
    MarcatiliMode,
    vector_wall_operator,
    wall_operator,
    worst_wall_limits,
)
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import wall_connection
from solprop.modes.protocol import Mode
from solprop.output.capability import capability_of
from solprop.output.hdf5 import open_file
from solprop.output.memory import MemoryOutput
from solprop.physdata.glasses import fused_silica
from solprop.physdata.sellmeier import gamma as gas_law
from solprop.profiles import FromCallable, Linear, PiecewiseLinear, ZProfile
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, FixedSchedule, RungeKutta
from solprop.translate.coupled import (
    CF4_GROWTH_LIMIT,
    SECTION_TOL,
    _cf4_action,
    advisory_step,
    cf4_defect,
    cf4_growth_bound,
    constant_schur_flow,
    flow_weights,
    frozen_generator,
    frozen_residual_flow,
    orthonormal_metric,
    residual_truncation,
    schur_blocks,
    section_expansion,
    wall_identity,
)
from solprop.translate.linop import delta_phi
from solprop.translate.problem import (
    _Assembly,
    _check_wall_route,
    _check_wall_total,
    _check_wall_total_varying,
    _normalisation_rate,
)
from solprop.translate.stage import scalars
from state_records import check_state, record_bound

RADIUS = 50e-6
"""Core radius of the qualification case, m.

Small enough that a millimetre is a measurable loss -- the highest retained channel
attenuates by several per cent over it -- and wide enough that the validity ratios stay
inside the default limit over the whole band.
"""

LENGTH = 2e-4
"""Length of the qualification case, m: a fifth of a millimetre."""

STEPS = 10
"""Steps of the fixed schedule, over a fifth of a millimetre."""

GRID = FieldGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15, **LEGACY_GRID)
"""One grid for every case here, so that a launch can be built against it.

The former geometry: the records here were made on it, and nothing here is about
the temporal window.
"""

WEAK_COLLAR = absorber(GRID, boundary_length=1.0e3)
"""A collar five million times weaker than the default one.

The default collar is an absorber, and an absorber in the **nonlinear** scheme is an
explicit term whose stability caps the step at about 5 micrometres on this grid
(see docs/dev/decisions.md); it also eats a continuous wave, which is what a single
excited bin of a field grid is, so an absolute statement about the flow could not be
made beside it. This one is present -- the masks and the band limit still are what they
are -- and its rate over a fifth of a millimetre is below 2e-7, far under every bound
taken here. It says so, and the module filters that warning: a collar too weak to stop a
packet walking off the window is exactly what is wanted where nothing walks. The
collar itself is qualified in ``tests/test_boundaries.py``."""

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""Two warnings this qualification case raises on purpose.

The band reaches past helium's and silica's fitted ranges, which is not this module's
subject: the block is built from whatever index the fill and the cladding report. And
a fifth of a millimetre of a 50 micrometre bore is a **marginal** case by design --
that is what makes the wall loss measurable over it -- so the quasi-static ratio sits
within a factor ten of its limit and says so. The warning itself is exercised in
``tests/modes/test_wall_operator.py``."""


def kept(field: np.ndarray | None) -> np.ndarray:
    """Return a saved plane's field: every plane of these runs keeps its own."""
    assert field is not None
    return field


def _steps_cf4(wall) -> bool:
    """Return whether a wall declares CF4 sections, which only the RK march steps."""
    return wall is not None and getattr(wall, "method", None) == "cf4"


def capillary(
    *,
    wall=None,
    modes: int | Sequence[Mode] = 3,
    pressure: Pressure = 1.0,
    radius: float | ZProfile = RADIUS,
    **overrides,
):
    """Return a small three-channel capillary simulation, with or without a wall.

    A fixed schedule, so that two runs whose operators differ in the last bits still
    land on the same positions: an adaptive controller would otherwise take different
    steps and agree only to its own tolerance, which is orders above the rounding a
    flow identity is about. Its length is set by the temporal collar, which is
    explicit in the nonlinear scheme and stable here only below 5.2 micrometres
    (see docs/dev/decisions.md); :data:`STEPS` steps of a fifth of a millimetre clear
    it.
    """
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": modes,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    if _steps_cf4(wall) and "integrator" not in arguments:
        # CF4 sections are stepped by the Runge-Kutta march alone.
        arguments["integrator"] = RungeKutta()
    return Simulation.from_capillary(
        radius,
        LENGTH,
        "He",
        pressure,
        wall=wall,
        **{"noise": None, **arguments},
    )


def run_capillary(
    *, wall=None, modes: int | Sequence[Mode] = 3, **overrides
) -> MemoryOutput:
    """Run the same case through `prop_capillary` and return the handler.

    The *entry point*, not the object: `prop_capillary` is what a script calls, and
    it builds the record, checks it against the problem and writes the planes, none
    of which `Simulation.translate` reaches.
    """
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": modes,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    if _steps_cf4(wall) and "integrator" not in arguments:
        # CF4 sections are stepped by the Runge-Kutta march alone.
        arguments["integrator"] = RungeKutta()
    radius = arguments.pop("radius_override", RADIUS)
    return cast(
        "MemoryOutput",
        prop_capillary(
            radius,
            LENGTH,
            "He",
            1.0,
            wall=wall,
            **{"noise": None, **arguments},
        ),
    )


def energy_loss(handler):
    """Return the fraction of the launched energy a run's last plane has lost."""
    planes = handler.planes()
    values = [float(np.sum(plane.stats["energy"])) for plane in planes]
    return 1.0 - values[-1] / values[0]


def propagated(problem, grid, backend=NumpyRefBackend):
    """Run one problem to its end plane on the fixed schedule and return the state."""
    driver = backend(problem, grid=grid)
    driver.set_initial(problem.initial.field0, None)
    driver.propagate_to(LENGTH)
    final = driver.observe().require_field().copy()
    driver.close()
    return final


def with_flow(problem, generator, *, dz_max=None, edges=None):
    """Return the same problem carrying a coupled flow built on `generator`."""
    existing = problem.linear.coupled_flow
    flow = constant_schur_flow(
        problem.grids.omega,
        problem.linear.phi_breakpoints[[0, -1]] if edges is None else edges,
        generator,
        dz_max=problem.stepper.dz_max if dz_max is None else dz_max,
        condition_limit=1e6,
        gauge_id="capillary HE_1m" if existing is None else existing.gauge_id,
        certificate_id="constant-schur"
        if existing is None
        else existing.certificate_id,
    )
    return replace(problem, linear=replace(problem.linear, coupled_flow=flow))


# ------------------------------------------------------------------ the leaf


def test_the_leaf_declares_one_constant_schur_section_with_a_zero_residual():
    """A constant section has no explicit residual at all: the row is exactly zero.

    Bound, stated first: the generator does not vary inside the section, so freezing
    it is not an approximation and `R_s(z) = 0` identically (at constant geometry both
    taper and wall-variation residuals are exactly zero). The representation error is
    the Schur reconstruction residual and the unitary defect, amplified by the departure
    from normality, which for a block of this size is below 1e-12 dimensionless.
    """
    problem = capillary(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.section_methods == ("constant_schur",)
    assert flow.representation == "dense"
    assert flow.connection is None
    assert float(np.max(flow.residual_norm_bound)) == 0.0
    assert float(np.max(flow.generator_error)) < 1e-12
    assert float(np.min(flow.inverse_condition_bound)) >= 1.0


def test_the_wall_pair_reconstructs_the_generators_off_diagonal():
    """The dimensionless vector times the 1/m weight is the off-diagonal, to rounding.

    The units, checked: vectors in 1/sqrt(m) and weights in m would give a
    dimensionless product beside a generator in 1/m. The constructor refuses a pair
    that does not multiply out, so the wrong convention cannot be stored.
    """
    problem = capillary(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.wall_vectors is not None and flow.wall_weights is not None
    assert flow.generator is not None
    vector = flow.wall_vectors[0, :, 0]
    rank_one = flow.wall_weights[0, 0] * np.outer(vector, vector)[:, :, None]
    generator = flow.generator[0, 0]
    off = ~np.eye(flow.n_modes, dtype=bool)
    scale = float(np.max(abs(generator)))
    assert np.max(abs(rank_one[off] - generator[off])) < 1e-12 * scale


def test_a_wall_pair_in_the_wrong_units_is_refused_by_the_constructor():
    """A weight in m rather than 1/m fails the reconstruction and is named."""
    omega = np.array([2.3e15, 2.4e15])
    modes = [MarcatiliMode(125e-6, "He", 1.0, m=index + 1) for index in (0, 1)]
    block = wall_operator(modes, omega)
    with pytest.raises(TranslationError, match="does not reconstruct"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            np.conj(block.generator),
            dz_max=1e-5,
            condition_limit=100.0,
            wall_vectors=block.trace_vector,
            wall_weights=np.conj(block.coupling) * 1e-6,
            gauge_id="lab",
            certificate_id="schur",
        )
    with pytest.raises(TranslationError, match="supplied together"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            np.conj(block.generator),
            dz_max=1e-5,
            condition_limit=100.0,
            wall_vectors=block.trace_vector,
            gauge_id="lab",
            certificate_id="schur",
        )
    with pytest.raises(TranslationError, match="must have shapes"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            np.conj(block.generator),
            dz_max=1e-5,
            condition_limit=100.0,
            wall_vectors=block.trace_vector[:1],
            wall_weights=np.conj(block.coupling),
            gauge_id="lab",
            certificate_id="schur",
        )


def test_the_constructor_refuses_a_conditioning_bound_it_cannot_meet():
    """A step long enough to attenuate past the caller's limit is refused by name.

    Bound, stated first: the declared row is `exp(dz_max (mu_max - mu_min))` with
    `mu` the extreme eigenvalues of the Hermitian part, so a `dz_max` above
    `ln(limit)/spread` must be refused and one below it accepted. The spread of this
    block is about 2.9e3 1/m, so a limit of 10 turns over at about 8e-4 m.
    """
    omega = np.array([2.3e15, 2.4e15])
    modes = [MarcatiliMode(RADIUS, "He", 1.0, m=index + 1) for index in range(3)]
    block = wall_operator(modes, omega)
    generator = np.conj(block.generator)
    hermitian = 0.5 * (
        np.moveaxis(generator, -1, 0)
        + np.conj(np.swapaxes(np.moveaxis(generator, -1, 0), -1, -2))
    )
    spread = float(np.max(np.ptp(np.linalg.eigvalsh(hermitian), axis=-1)))
    turnover = math.log(10.0) / spread
    built = constant_schur_flow(
        omega,
        np.array([0.0, 1e-3]),
        generator,
        dz_max=0.5 * turnover,
        condition_limit=10.0,
        gauge_id="lab",
        certificate_id="schur",
    )
    assert float(np.max(built.inverse_condition_bound)) < 10.0
    with pytest.raises(TranslationError, match="inverse conditioning reaches"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            generator,
            dz_max=2.0 * turnover,
            condition_limit=10.0,
            gauge_id="lab",
            certificate_id="schur",
        )


def test_the_constructor_checks_its_own_arguments():
    """A bad generator shape, dz_max or limit is named before anything is decomposed."""
    omega = np.array([2.3e15, 2.4e15])
    block = np.zeros((2, 2, 2), dtype=complex)
    with pytest.raises(TranslationError, match="must have shape"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            block[:, :, :1],
            dz_max=1e-5,
            condition_limit=10.0,
            gauge_id="lab",
            certificate_id="schur",
        )
    with pytest.raises(TranslationError, match="positive dz_max"):
        constant_schur_flow(
            omega,
            np.array([0.0, 1e-3]),
            block,
            dz_max=0.0,
            condition_limit=10.0,
            gauge_id="lab",
            certificate_id="schur",
        )


def test_a_zero_block_is_not_decomposed_and_a_triangular_one_is():
    """Schur factors of an empty bin are the identity and zero, exactly."""
    block = np.zeros((2, 2, 3), dtype=complex)
    block[0, 1, 1] = 3.0
    vectors, triangular = schur_blocks(block)
    assert np.array_equal(vectors[0], np.eye(2))
    assert not np.any(triangular[0])
    reconstructed = vectors[1] @ triangular[1] @ np.conj(vectors[1].T)
    assert_allclose(reconstructed, block[:, :, 1], atol=1e-15)


# ------------------------------------------------------------------ the route


def test_the_route_keeps_the_carrier_mode_independent_and_the_weight_per_channel():
    """`Phi` is a multiple of the identity and the power weight is the real set's.

    The split the whole route rests on. `exp(A) exp(B)` is `exp(A + B)` only when the
    two commute, so the cumulative diagonal beside a coupled block must be a multiple
    of the identity -- it is built over N copies of the first channel -- while the
    true-power weight, which is a row of its own and not part of the flow, must still
    be each channel's own `Re n_eff` (amplitudes are true-power amplitudes).
    """
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    constant = problem.linear.phi_constant
    assert constant is not None
    assert problem.linear.phi_representation == "constant"
    assert np.array_equal(constant, np.broadcast_to(constant[:, :1], constant.shape))
    weights = problem.linear.power_weight_constant
    assert weights is not None
    band = problem.grids.active_mask
    assert not np.array_equal(weights[0, 0, band], weights[1, 0, band])
    for index, mode in enumerate(simulation.modes):
        expected = np.real(mode.neff(problem.grids.omega[band]))
        assert_allclose(weights[index, 0, band], expected, rtol=1e-14)


def test_the_route_records_its_flow_on_the_modal_identity():
    """`modal.flow` carries the method, the ratios and the certificate, as scalars.

    A key entry of the modal identity: the capability stays `modal`, because a
    coupled flow through a basis does not change the basis, and nothing already
    written reads differently.
    """
    metadata = capillary(wall=CoherentWall()).metadata()
    assert metadata.modal is not None
    record = cast("Mapping[str, object]", metadata.modal["flow"])
    assert record["model"] == "mean impedance coherent wall"
    assert record["section_methods"] == ["constant_schur"]
    assert record["residual_norm_bound"] == 0.0
    assert record["radius"] == RADIUS
    assert 0.0 < float(cast("float", record["quasi_static"])) < 0.25
    assert metadata.schema_version == 2
    assert capability_of(metadata) == "modal"
    assert "mean impedance coherent wall" in str(metadata)
    # Nothing that is not a scalar reaches the resume key.
    assert all(
        isinstance(value, (int, float, str)) or value is None
        for value in record.values()
        if not isinstance(value, list)
    )


def test_a_run_without_a_wall_declares_no_flow_and_no_identity():
    """The uncoupled route is untouched: no leaf, no modal record, same capability."""
    simulation = capillary()
    problem = simulation.translate()
    assert problem.linear.coupled_flow is None
    assert simulation.metadata().modal is None
    assert problem.linear.phi_constant is not None
    constant = problem.linear.phi_constant
    assert not np.array_equal(
        constant, np.broadcast_to(constant[:, :1], constant.shape)
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "modes": [
                    MarcatiliMode(RADIUS, "He", 1.0, m=1),
                    MarcatiliMode(RADIUS, "He", 1.0, m=3),
                ]
            },
            "HE_11 to HE_12",
        ),
        ({"noise": "modified", "seed": 3}, "refuses the 'modified' noise model"),
    ],
)
def test_the_route_refuses_what_the_block_is_not_derived_for(overrides, expected):
    """Every incompatible run is named at the translation layer, not at the backend."""
    with pytest.raises((ModeError, TranslationError), match=expected):
        capillary(wall=CoherentWall(), **overrides).translate()


@pytest.mark.medium
def test_prop_capillary_runs_a_wall_end_to_end_and_records_its_flow():
    """The entry point runs, the record carries `modal.flow`, the loss is derived.

    Bound, stated first. At `N = 1` the block's field-loss rate departs from the
    uncoupled Marcatili rate by the relative `delta(w) = -(4 t^2 + u_1^2/(2(ka)^2))`
    of item 4(a), with `t = v_n/(k_0 a)` -- a loss the block makes *smaller*, because
    the discarded boundary trace and the paraxial reduction both take from it. The
    launched spectrum is known, so the energy that predicts is
    `<P(w) e^{-2 a z}(e^{2 a |delta| z} - 1)>/<P>`, and the two runs must differ by
    that to better than a per cent; the total loss must be within
    `max(2 a |delta| z)` of the uncoupled run's whatever the weighting, and the
    uncoupled run must be the more lossy of the two.

    This is the route a script takes -- `prop_capillary(..., wall=...).run()` --
    rather than a hand-driven backend, because the record and the problem are checked
    against each other only there (`driver/run.py`,
    `_refuse_a_record_that_misnames_the_basis`).
    """
    walled = run_capillary(wall=CoherentWall(), modes=1)
    plain = run_capillary(modes=1)
    written, control = walled.metadata, plain.metadata
    assert written is not None and control is not None
    record = written.modal
    assert record is not None
    assert record["basis"] == "capillary HE_1m"
    flow = cast("Mapping[str, object]", record["flow"])
    assert flow["model"] == "mean impedance coherent wall"
    assert flow["section_methods"] == ["constant_schur"]
    assert capability_of(written) == "modal"
    assert control.modal is None

    mode = MarcatiliMode(RADIUS, "He", 1.0, m=1)
    band = GRID.omega[GRID.active_mask & (GRID.omega > 0)]
    terms = mode.sampled_terms(band)
    radius, density = mode._scalars(0.0)
    vacuum = band / C
    quasi_static = np.asarray(terms.cladding_term) / (vacuum * radius)
    wavenumber = np.sqrt(1.0 + np.asarray(terms.core_susceptibility) * density) * vacuum
    deviation = abs(
        -(4.0 * quasi_static**2 + mode.u_nm**2 / (2.0 * (wavenumber * radius) ** 2))
    )
    rate = np.imag(mode.neff(band)) * band / C
    spectrum = abs(
        kept(walled.planes()[0].field)[0, 0, GRID.active_mask & (GRID.omega > 0)]
    )
    weights = spectrum**2
    predicted = float(
        np.sum(
            weights
            * np.exp(-2.0 * rate * LENGTH)
            * np.expm1(2.0 * rate * deviation * LENGTH)
        )
        / np.sum(weights)
    )
    measured = energy_loss(plain) - energy_loss(walled)
    assert energy_loss(walled) > 0.0
    assert 0.0 < measured < 1.01 * float(np.max(2.0 * rate * deviation * LENGTH))
    assert measured == pytest.approx(predicted, rel=0.01)


@pytest.mark.parametrize(
    "loss", [False, 0.0, 0.5, lambda omega: np.full_like(np.asarray(omega), 100.0)]
)
def test_a_loss_setting_beside_a_wall_is_refused_rather_than_discarded(loss):
    """The wall *is* the loss model, so a `LossSpec` beside it is named.

    Measured before the refusal existed: the largest internal field-loss rate of a
    50 micrometre helium bore at `N = 1` was 17.730 /m with the wall for `loss=True`,
    `False`, `0.0`, `0.5` and a 100 /m callable **alike**, against 17.751, 0.030,
    8.891 and 50.030 /m on the uncoupled route -- the whole setting vanished, because
    the assembly subtracts the entire `HE_11` wavenumber and re-adds the block's own
    diagonal. Both doors are exercised: the keyword the caller typed, and the mode
    set the operator is built from.
    """
    with pytest.raises(DriverError, match="is the loss model"):
        capillary(wall=CoherentWall(), loss=loss)
    with pytest.raises(ModeError, match="is the loss model"):
        wall_operator(
            [
                MarcatiliMode(RADIUS, "He", 1.0, m=index + 1, loss=loss)
                for index in (0,)
            ],
            np.array([2.356e15]),
        )


def test_the_reduced_model_beside_a_wall_is_refused_rather_than_overridden():
    """The condensation inverts the full model's eigenvalue, and says so."""
    with pytest.raises(DriverError, match="one capillary model, the full one"):
        capillary(wall=CoherentWall(), model="reduced")
    with pytest.raises(ModeError, match="one capillary model, the full one"):
        wall_operator(
            [MarcatiliMode(RADIUS, "He", 1.0, m=1, model="reduced")],
            np.array([2.356e15]),
        )


def test_a_tube_lattice_is_refused_rather_than_given_the_bare_capillary_wall():
    """A Vincetti set is a `MarcatiliMode` subclass, and would take the bare wall.

    Measured before the refusal existed, on a 15 micrometre argon-filled lattice of
    six 10.38 micrometre tubes at 10 bar: the flow was built, the generator was **bit
    identical** to the plain Marcatili one, and the `HE_11` field loss it declared was
    3.21e3 /m against the guide's own 1.01e-3 /m -- six orders, in the wrong
    direction. The structured cladding's loss is its tubes' resonances, not a Robin
    condition at one radius.
    """
    lattice = TubeLattice(180e-9, 10.38e-6, 6)
    with pytest.raises(DriverError, match="uniform dielectric wall"):
        Simulation.from_capillary(
            15e-6,
            LENGTH,
            "Ar",
            10.0,
            wavelength=800e-9,
            grid=GRID,
            absorber=WEAK_COLLAR,
            fwhm=30e-15,
            energy=1e-9,
            modes=2,
            n_save=2,
            backend="numpy_ref",
            lattice=lattice,
            wall=CoherentWall(),
            plasma=False,
            raman=False,
            noise=None,
        )
    block = [
        VincettiMode(15e-6, "Ar", 10.0, lattice=lattice, m=index + 1)
        for index in range(2)
    ]
    with pytest.raises(ModeError, match="uniform dielectric wall"):
        wall_operator(block, np.array([2.356e15]))


def test_channels_that_do_not_declare_one_guide_are_refused_by_name():
    """The block reads the lowest channel alone, so the others must agree with it.

    Measured before the refusal existed: two channels declaring `cladding=1.45+0j`
    and `2.6+0.1j` built a flow from the first one's impedance and said nothing. The
    declared settings are compared first, and then what the operator actually
    samples, because two channels can name one glass in two ways and mean it.
    """
    omega = np.array([2.356e15, 2.4e15, 2.5e15])
    with pytest.raises(ModeError, match="core radius"):
        wall_operator(
            [
                MarcatiliMode(RADIUS, "He", 1.0, m=1),
                MarcatiliMode(1.5 * RADIUS, "He", 1.0, m=2),
            ],
            omega,
        )
    with pytest.raises(ModeError, match="density"):
        wall_operator(
            [
                MarcatiliMode(RADIUS, "He", 1.0, m=1),
                MarcatiliMode(RADIUS, "He", 2.0, m=2),
            ],
            omega,
        )
    with pytest.raises(ModeError, match="samples a different cladding"):
        wall_operator(
            [
                MarcatiliMode(RADIUS, "He", 1.0, m=1, cladding=1.45 + 0j),
                MarcatiliMode(RADIUS, "He", 1.0, m=2, cladding=2.6 + 0.1j),
            ],
            omega,
        )
    with pytest.raises(ModeError, match="samples a different fill"):
        wall_operator(
            [
                MarcatiliMode(
                    RADIUS,
                    "He",
                    1.0,
                    m=1,
                    susceptibility=gas_law("He", model="ermolov_2015/sellmeier_He"),
                ),
                MarcatiliMode(
                    RADIUS,
                    "He",
                    1.0,
                    m=2,
                    susceptibility=gas_law("He", model="borzsonyi_2008/sellmeier_He"),
                ),
            ],
            omega,
        )
    # The default glass, asked for by name, is the same guide and is not refused.
    wall_operator(
        [
            MarcatiliMode(RADIUS, "He", 1.0, m=1),
            MarcatiliMode(RADIUS, "He", 1.0, m=2, cladding=fused_silica()),
        ],
        omega,
    )


def test_a_varying_section_takes_the_frozen_residual_route():
    """A pressure gradient builds the frozen generator and its residual.

    Bound, stated first: the residual of a *constant* guide is exactly zero, and the
    only thing that makes this one non-zero is the declared gradient, so every
    section must declare `"frozen_residual"`, must carry more than one Chebyshev
    degree along z, and must declare a strictly positive residual bound. A pressure
    gradient moves no wall, so there is **no connection**: that is the taper's.
    """
    problem = capillary(wall=CoherentWall(), pressure=(1.0, 0.1)).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert set(flow.section_methods) == {"frozen_residual"}
    assert flow.n_deg > 1
    assert flow.connection is None
    # Out-of-band bins carry no block at all, so their residual is exactly zero;
    # every bin the wall formula was evaluated at carries one.
    assert (
        float(np.min(flow.residual_norm_bound[:, flow.omega > 0.0].max(axis=0))) >= 0.0
    )
    assert float(np.max(flow.residual_norm_bound)) > 0.0
    assert problem.linear.phi_representation != "constant"
    # And it runs: a gradient is the varying route with no connection at all, so the
    # residual is the frozen variation alone and the backend holds no connection term.
    final = propagated(problem, GRID)
    assert float(np.max(abs(final))) > 0.0
    assert np.all(np.isfinite(final))


def test_a_vector_state_or_a_noise_draw_refuses_a_wall_beside_it():
    """A wall block is scalar and refuses a polarisation axis and a noise draw.

    The capillary route refuses a `polarisation=` keyword long before translation,
    so these two doors are exercised on the check itself. The routes that *can* be
    reached with a wall beside them are the test below.
    """
    build = _Assembly(
        quadrature=None,
        components=None,
        spatial_grid=None,
        medium=None,
        density=None,
        prefactor_index=None,
        shock=True,
        responses=(),
        n_pol=1,
        z_max=LENGTH,
        absorber=None,
        loss=DEFAULT_LOSS_POLICY,
        margin=0.05,
        tol=1e-10,
        node_budget=1,
        has_noise=False,
        dz_max=1e-5,
        reuse=None,
    )
    with pytest.raises(TranslationError, match="one field component") as refused:
        _check_wall_route(replace(build, n_pol=2))
    # The TE/TM wall is named as the wall's impedances, not the state's axis.
    assert "not the state's polarisation axis" in str(refused.value)
    with pytest.raises(TranslationError, match="refuses the 'modified' noise"):
        _check_wall_route(replace(build, has_noise=True, noise_model="modified"))
    # The input draw is part of the entrance state, which the flow propagates.
    _check_wall_route(replace(build, has_noise=True, noise_model="input"))


def test_a_wall_declared_on_another_route_is_refused_by_name():
    """A general modal or spatial run carrying a wall is named, not ignored.

    `assemble` reads `wall` on its **modal** branch alone, so the refusal has to be
    the first thing it does: without it a run of either other route translates with
    `linear.coupled_flow` absent, no `modal.flow` key on its record and nothing said
    at all. Both are reached here the way a caller reaches them -- through their own
    constructors and the attribute `from_capillary` sets -- rather than through a
    hand-built assembly, because that is the path the leak was on.
    """
    spatial = Simulation.from_radial(
        capillary().grid,
        RadialGrid(0.6e-3, 8),
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 60e-6),
        settings=replace(
            capillary().settings,
            stats=StatsSpec(names=("energy",)),
            quadrature=None,
            prefactor_index=None,
        ),
        noise=None,
    )
    spatial.wall = CoherentWall()
    with pytest.raises(TranslationError, match="a spatial geometry"):
        spatial.translate()
    grid = general_grid()
    general = Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=general_guide(grid=grid),
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        backend="numpy_ref",
        save_planes=[GENERAL_Z_END / 10],
        dz_max=GENERAL_Z_END / 20,
        dz_init=GENERAL_Z_END / 20,
        noise=None,
    )
    general.wall = CoherentWall()
    with pytest.raises(TranslationError, match="the general modal route"):
        general.translate()


# ------------------------------------------------------------------ the backend


def test_the_reference_refuses_every_coupled_flow_it_does_not_execute():
    """Each method, representation, degree, connection and residual by its own name."""
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    grid = simulation.grid

    def spoiled(**changes):
        return replace(
            problem,
            linear=replace(problem.linear, coupled_flow=replace(flow, **changes)),
        )

    # Refused by the Adams path's check or the Runge-Kutta march's, by name either way.
    with pytest.raises(BackendError, match=r"'diagonal'"):
        NumpyRefBackend(spoiled(section_methods=("diagonal",)), grid=grid)
    with pytest.raises(BackendError, match="executes no taper connection"):
        NumpyRefBackend(
            spoiled(connection=np.zeros((1, 1, 3, 3, grid.n_omega), dtype=complex)),
            grid=grid,
        )
    with pytest.raises(BackendError, match="no explicit"):
        NumpyRefBackend(
            spoiled(residual_norm_bound=np.ones((1, grid.n_omega))), grid=grid
        )
    assert flow.generator is not None
    doubled = np.concatenate([flow.generator, flow.generator], axis=1)
    with pytest.raises(BackendError, match="Chebyshev degrees along z"):
        NumpyRefBackend(spoiled(generator=doubled), grid=grid)
    factored = replace(
        problem,
        linear=replace(
            problem.linear,
            coupled_flow=replace(
                flow,
                generator=None,
                fixed_terms=np.zeros((1, 3, 3, grid.n_omega), dtype=complex),
                term_weights=np.zeros((1, 1, 1), dtype=complex),
                wall_vectors=None,
                wall_weights=None,
            ),
        ),
    )
    with pytest.raises(BackendError, match="dense generator"):
        NumpyRefBackend(factored, grid=grid)


def test_the_reference_refuses_a_problem_the_flow_cannot_sit_beside():
    """A mode-dependent Phi, a metric or a noise draw beside a flow is named."""
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    grid = simulation.grid
    constant = problem.linear.phi_constant
    assert constant is not None
    varied = constant.copy()
    varied[0, 1] *= 1.0000001
    with pytest.raises(BackendError, match="mode-independent Phi"):
        NumpyRefBackend(
            replace(problem, linear=replace(problem.linear, phi_constant=varied)),
            grid=grid,
        )
    metric = orthonormal_metric(
        problem.grids.omega,
        problem.linear.phi_breakpoints[[0, -1]],
        defect=0.0,
        gauge_id="lab",
        certificate_id="sampled-gram",
    )
    with pytest.raises(BackendError, match="no explicit metric"):
        NumpyRefBackend(
            replace(problem, linear=replace(problem.linear, modal_metric=metric)),
            grid=grid,
        )
    # The block is scalar and its action contracts the channel axis alone, so a
    # second polarisation would be mixed by it as though it described both. The
    # refusal is taken on the check itself, because a state whose polarisation axis
    # disagrees with the rest of the problem is refused by `KernelProblem.validate`
    # before any backend sees it -- which is exactly why this one is defence in
    # depth rather than the only door.
    batch, channels, _, bins = problem.shape
    with pytest.raises(BackendError, match="only for a scalar state") as refused:
        refuse_coupled_flow(replace(problem, shape=(batch, channels, 2, bins)))
    assert "not the state's polarisation axis" in str(refused.value)
    # It names the state's two components, the product route, and not the
    # synthesis, which may resolve two.
    assert "a state with two polarisation components" in str(refused.value)


def test_the_reference_refuses_a_step_longer_than_the_certificate_was_declared_over():
    """The declared conditioning row is checked against the action, not trusted.

    Bound, stated first: the row is `exp(dz_max (mu_max - mu_min))`, so declaring it
    over a hundredth of the step the run actually takes must be refused by the whole
    step's own measured `norm(U) norm(U^-1)`. This is what makes the certificate a
    claim the executor checks rather than a number it believes.

    The certificate bounds the inverse action the Runge-Kutta march carries the source
    back by; the exponential Adams method uses forward flows only (ADR-044), so this is
    the march's check.
    """
    simulation = capillary(wall=CoherentWall(), integrator=RungeKutta())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    understated = with_flow(
        problem, flow.generator[0, 0], dz_max=problem.stepper.dz_max / 100.0
    )
    with pytest.raises(BackendError, match="measured a coupled flow conditioning"):
        propagated(understated, simulation.grid)


def test_the_reference_refuses_a_flow_beside_a_varying_phi():
    """A cumulative Phi that is not the exact constant representation is named.

    A pressure gradient takes the expansion representation, which is not a constant
    the block can be subtracted from exactly; the route refuses that at the
    translation layer, and this is the backend's own door on the same statement.
    """
    simulation = capillary(pressure=(1.0, 0.9))
    problem = simulation.translate()
    assert problem.linear.phi_representation != "constant"
    edges = problem.linear.phi_breakpoints[[0, -1]]
    spoiled = with_flow(
        problem,
        np.zeros((3, 3, simulation.grid.n_omega), dtype=complex),
        dz_max=problem.stepper.dz_max,
        edges=edges,
    )
    with pytest.raises(BackendError, match="exact constant representation"):
        NumpyRefBackend(spoiled, grid=simulation.grid)


def test_the_reference_refuses_a_noise_draw_beside_a_flow():
    """A draw the route cannot even build is named at the backend all the same."""
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    grid = simulation.grid
    noisy = replace(
        problem,
        initial=replace(
            problem.initial,
            noise0=np.zeros_like(problem.initial.field0),
            noise_model="modified",
        ),
    )
    with pytest.raises(
        BackendError, match="no 'modified' noise model beside a coupled flow"
    ):
        NumpyRefBackend(noisy, grid=grid)


def test_a_flow_action_that_needs_squaring_is_the_exponential_all_the_same():
    """Scaling and squaring reproduces `scipy.linalg.expm` to 1e-13 relative.

    Bound, stated first: the series is summed to degree 16 after scaling the row-sum
    norm below a half, whose truncated tail is 2.1e-20, so the only departure from an
    independently computed exponential is the rounding of the squarings -- below 1e-13
    of the result's own norm for the five squarings this block needs. Every case the
    route actually builds has a norm below the scale and takes none, so this exercises
    the branch deliberately.
    """
    generator = np.random.default_rng(1)
    block = np.triu(
        generator.standard_normal((3, 3)) + 1j * generator.standard_normal((3, 3))
    )
    scaled = block * (12.0 / float(np.max(np.sum(abs(block), axis=-1))))
    taken = _triangular_exponential(scaled[None])[0]
    reference = np.asarray(expm(scaled), dtype=complex)
    assert np.max(abs(taken - reference)) < 1e-13 * float(np.max(abs(reference)))


def test_an_exponent_too_large_to_square_is_refused():
    """A flow action needing more squarings than are admitted is named, not tried."""
    huge = np.full((1, 2, 2), 2.0 ** (_MAX_FLOW_SQUARINGS + 2), dtype=complex)
    with pytest.raises(BackendError, match="squarings"):
        _triangular_exponential(huge)


def test_a_total_generator_that_grows_is_refused_by_name():
    """The third door on the sign: carrier plus block must not gain anywhere.

    Unreachable with the conjugation as written, which is the point -- it is the
    guard that catches `-i B` where `-i conj(B)` was meant -- so it is exercised on a
    generator whose diagonal has been flipped.
    """
    problem = capillary(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    generator = flow.generator[0, 0].copy()
    index = np.arange(flow.n_modes)
    generator[index, index] = -np.conj(generator[index, index])
    with pytest.raises(TranslationError, match="total internal generator grows"):
        _check_wall_total(generator, problem.linear, flow.n_modes)


# ------------------------------------------------------------------ propagation


@pytest.mark.medium
def test_a_zero_generator_propagates_bitwise_like_no_flow_at_all():
    """A coupled flow whose generator is zero changes not one bit of ten steps.

    Bound, stated first: **bitwise**, an explicitly exact no-op, which SEP 25 keeps a
    structural contract. The Schur factors of a zero block are the
    identity and zero exactly, the Taylor series of zero sums to the identity
    exactly, and multiplying a finite state by an exact identity adds only signed
    zeros, which is exact. Anything else would mean the action is applied somewhere
    the flow is not, or the scalar propagator has been reassociated.
    """
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    zeroed = with_flow(problem, np.zeros_like(flow.generator[0, 0]))
    without = replace(problem, linear=replace(problem.linear, coupled_flow=None))
    assert np.array_equal(
        propagated(zeroed, simulation.grid), propagated(without, simulation.grid)
    )


@pytest.mark.medium
def test_a_diagonal_generator_matches_the_same_diagonal_inside_phi():
    """With the wall coupling off the block is diagonal and agrees with `exp(dPhi)`.

    Bound, stated first: **1e-12 relative** on the final state. The two routes do the
    same arithmetic in a different order -- one exponentiates the sum, the other
    multiplies two exponentials and contracts a diagonal matrix -- so they differ by the
    rounding of one exponential and one length-N dot product per stage, which over ten
    fixed steps and six stages is a few hundred roundings. The schedule is fixed for
    exactly this reason: two adaptive runs of operators that differ in the last bits
    agree only to the controller's tolerance. This is the *wall off* half; what it
    checks is the flow, not the physics, which is the same operator both ways.
    """
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    generator = flow.generator[0, 0]
    index = np.arange(flow.n_modes)
    diagonal = np.zeros_like(generator)
    diagonal[index, index] = generator[index, index]
    constant = problem.linear.phi_constant
    assert constant is not None
    folded = constant.copy()
    folded[0, :, 0, :] += diagonal[index, index]
    inside = replace(
        problem,
        linear=replace(problem.linear, phi_constant=folded, coupled_flow=None),
    )
    coupled = propagated(with_flow(problem, diagonal), simulation.grid)
    scale = float(np.max(abs(coupled)))
    assert_allclose(coupled, propagated(inside, simulation.grid), atol=1e-12 * scale)


def linear_launch(problem, grid, channel=0):
    """Return the problem relaunched with one bin of one channel, weakly excited.

    1e-9 sqrt(W) in one bin: the Kerr phase over this length is then below 1e-12 rad,
    so what the ten steps measure is the linear flow and nothing else. A single bin
    of a field grid is exactly periodic on the window, so no leakage can hide an
    error either.
    """
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    state = np.zeros_like(problem.initial.field0)
    state[0, channel, 0, index] = 1e-9
    return index, replace(
        problem, initial=replace(problem.initial, field0=state, noise0=None)
    )


@pytest.mark.medium
def test_ten_steps_apply_the_block_the_translation_layer_declared():
    """The propagated decay of one channel is `exp(Re(Phi + G_0) L)`, to 1e-9.

    Bound, stated first: **1e-9 relative**. The launch is weak enough to be linear,
    the schedule is fixed, and a diagonal-dominant flow of a single excited bin is a
    pure exponential, so the only departure is the rounding of ten products of a
    Schur action by a scalar propagator. What this adds to the operator tests is
    that the step loop applies the operator it was given, in the right convention,
    with the carrier and the block multiplied rather than added.
    """
    simulation = capillary(wall=CoherentWall(), modes=1)
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    index, launched = linear_launch(problem, simulation.grid)
    final = propagated(launched, simulation.grid)
    carrier = complex(delta_phi(problem.linear, 0.0, LENGTH)[0, 0, 0, index])
    assert flow.generator is not None
    rate = complex(flow.generator[0, 0, 0, 0, index])
    expected = 1e-9 * np.exp(carrier + rate * LENGTH)
    assert abs(final[0, 0, 0, index]) == pytest.approx(abs(expected), rel=1e-9)
    assert np.angle(final[0, 0, 0, index]) == pytest.approx(
        float(np.angle(expected)), rel=1e-9
    )


@pytest.mark.medium
def test_the_propagated_single_channel_loss_is_marcatili_to_its_derived_bound():
    """Ten steps of the block against ten steps of the uncoupled route, N = 1.

    Bound, stated first and derived: the block's HE_11 field-loss rate departs from
    this package's Marcatili one by `-(4 t^2 + u_1^2/(2 (ka)^2))` relative, which is
    -7.5e-5 for this geometry at the reference frequency. Over ten steps of 20
    micrometres the two moduli therefore differ by `exp(-that * alpha * L)`, about
    1.7e-8, and this asserts the **measured** log-ratio reproduces that prediction to
    1 %. The control is the same simulation, the same grid, the same fixed schedule
    and the same absorber, with the wall left undeclared.
    """
    with_wall = capillary(wall=CoherentWall(), modes=1)
    without = capillary(modes=1)
    index, walled = linear_launch(with_wall.translate(), with_wall.grid)
    _, plain = linear_launch(without.translate(), without.grid)
    ratio = abs(propagated(walled, with_wall.grid)[0, 0, 0, index]) / abs(
        propagated(plain, without.grid)[0, 0, 0, index]
    )

    mode = MarcatiliMode(RADIUS, "He", 1.0, m=1)
    omega = np.array([float(with_wall.grid.omega[index])])
    block = wall_operator([mode], omega, validity_limit=1.0)
    marcatili = float(omega[0] / C * np.imag(mode.neff(omega)[0]))
    inverse_eta = 1.0 / abs(RADIUS * complex(block.impedance[0]))
    carrier = abs(complex(block.wavenumber[0])) * RADIUS
    predicted = -(4.0 * inverse_eta**2 + mode.u_nm**2 / (2.0 * carrier**2))
    assert abs(predicted) > 1e-5
    assert math.log(ratio) == pytest.approx(-predicted * marcatili * LENGTH, rel=1e-2)


@pytest.mark.medium
def test_each_channel_keeps_its_own_rate_in_the_weakly_coupled_limit():
    """Launched into one channel alone, that channel follows its diagonal rate.

    Bound, stated first: the off-diagonal moves a fraction of order
    `|g h_m h_n/(D_mm - D_nn)|^2` into the others, which is below 1e-4 here, so the
    launched channel's own modulus follows its diagonal to **1e-4 relative** -- the
    leaked power is what it is missing. Each diagonal is that channel's Marcatili
    loss to the derived bound, which the operator tests assert; here it is the
    propagated statement, over ten steps of a three-channel block.
    """
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    index, launched = linear_launch(problem, simulation.grid)
    final = propagated(launched, simulation.grid)
    carrier = complex(delta_phi(problem.linear, 0.0, LENGTH)[0, 0, 0, index])
    assert flow.generator is not None
    rate = complex(flow.generator[0, 0, 0, 0, index])
    expected = abs(1e-9 * np.exp(carrier + rate * LENGTH))
    assert abs(final[0, 0, 0, index]) == pytest.approx(expected, rel=1e-4)
    leaked = (
        float(np.sum(abs(final[0, 1:, 0, index]) ** 2))
        / abs(final[0, 0, 0, index]) ** 2
    )
    assert 0.0 < leaked < 1e-4


@pytest.mark.medium
def test_the_wall_moves_power_between_channels_and_a_diagonal_loss_cannot():
    """The point of the block: a coherent wall couples, a diagonal loss cannot.

    Bound, stated first: launched into `HE_11` alone, the uncoupled flow leaves below
    1e-30 of the launched power in the other channels -- a diagonal exponential
    cannot fill them at all, and what is there is the nonlinear overlap -- while the
    wall block puts more than 1e-10 there.
    """
    simulation = capillary(wall=CoherentWall())
    problem = simulation.translate()
    index, launched = linear_launch(problem, simulation.grid)
    walled = propagated(launched, simulation.grid)
    plain = propagated(
        replace(launched, linear=replace(launched.linear, coupled_flow=None)),
        simulation.grid,
    )
    # Not exactly zero: the capillary's Kerr response couples the channels through
    # their transverse overlap, and at this amplitude that leaves 1e-38 of the
    # launched power there. The wall puts thirty orders of magnitude more.
    leaked = float(np.sum(abs(plain[0, 1:, 0, index]) ** 2)) / 1e-18
    moved = float(np.sum(abs(walled[0, 1:, 0, index]) ** 2)) / 1e-18
    assert leaked < 1e-30
    assert moved > 1e-10


@pytest.mark.slow
def test_the_transmission_of_one_physical_launch_converges_with_the_channel_count():
    """N-convergence of the same Gaussian and the same aperture, projection apart.

    Bound, stated first: the transmitted fraction of the **projected** launch is a
    physical observable of the same launch at every N, so the change from N to 2N
    must shrink; the gate is that `|T_8 - T_4| < |T_4 - T_2|`. The **projection**
    error -- how much of the aperture-truncated Gaussian the retained block does not
    represent -- is reported separately and is not part of that gate, because it is a
    property of the basis and not of the propagation.
    """
    waist = 0.64 * RADIUS
    transmissions = {}
    deficits = {}
    for count in (2, 4, 8):
        zeros = jn_zeros(0, count)
        overlaps = np.array(
            [
                quad(
                    lambda x, zero=zero: (
                        x * np.exp(-((x * RADIUS / waist) ** 2)) * jv(0, zero * x)
                    ),
                    0.0,
                    1.0,
                )[0]
                * math.sqrt(2.0)
                / abs(jv(1, zero))
                for zero in zeros
            ]
        )
        total = quad(lambda x: x * np.exp(-2.0 * (x * RADIUS / waist) ** 2), 0.0, 1.0)[
            0
        ]
        deficits[count] = 1.0 - float(np.sum(overlaps**2)) / total
        simulation = capillary(wall=CoherentWall(), modes=count)
        problem = simulation.translate()
        index = int(np.argmin(abs(simulation.grid.omega - simulation.grid.omega0)))
        state = np.zeros_like(problem.initial.field0)
        # The **same physical launch** at every N: the raw projection coefficients
        # of one Gaussian, not renormalised inside the retained block, so that what
        # the block does not represent stays missing rather than being scaled up.
        state[0, :, 0, index] = 1e-9 * overlaps
        launched = replace(
            problem, initial=replace(problem.initial, field0=state, noise0=None)
        )
        final = propagated(launched, simulation.grid)
        carrier = float(np.real(delta_phi(problem.linear, 0.0, LENGTH)[0, 0, 0, index]))
        transmissions[count] = float(np.sum(abs(final[0, :, 0, index]) ** 2)) / (
            1e-18 * total * math.exp(2.0 * carrier)
        )
    assert deficits[2] > deficits[4] > deficits[8] > 0.0
    first = abs(transmissions[4] - transmissions[2])
    second = abs(transmissions[8] - transmissions[4])
    assert second < first


# ------------------------------------------------------------------ the file


@pytest.mark.medium
def test_the_flow_record_survives_a_file_and_a_resume_checks_it(tmp_path):
    """`modal.flow` round trips through HDF5, and the resume key carries it.

    The record is written, read back and compared entry for entry: it holds scalars
    and strings alone, so a round trip through the file's JSON must be exact. The
    capability read back is still `modal` -- a coupled flow through a basis does not
    change the basis -- so every ordinary accessor stays admitted, which is the whole
    reason the wall's identity declares `capillary HE_1m` rather than
    `general_modal`.

    Two resumes are refused by name because the flow is part of the key: the same
    run without its wall, and the same wall in a different bore.
    """
    path = tmp_path / "wall.h5"
    capillary(wall=CoherentWall(), modes=1).run(path)
    stored = open_file(path)
    try:
        read_back = stored.metadata
        assert read_back is not None and read_back.modal is not None
        assert capability_of(read_back) == "modal"
        fresh = run_capillary(wall=CoherentWall(), modes=1).metadata
        assert fresh is not None and fresh.modal is not None
        assert read_back.modal["flow"] == fresh.modal["flow"]
    finally:
        stored.close()
    with pytest.raises(OutputError, match="different run"):
        capillary(modes=1).run(path, resume=True)
    with pytest.raises(OutputError, match="different run"):
        Simulation.from_capillary(
            1.5 * RADIUS,
            LENGTH,
            "He",
            1.0,
            wall=CoherentWall(),
            wavelength=800e-9,
            grid=GRID,
            absorber=WEAK_COLLAR,
            fwhm=30e-15,
            energy=1e-9,
            modes=1,
            n_save=2,
            backend="numpy_ref",
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
            plasma=False,
            raman=False,
            noise=None,
        ).run(path, resume=True)


# ------------------------------------------------------- the legacy entry points

LEGACY_LENGTH = 2e-4
"""Length of the legacy-identity case, m: ten fixed steps of 20 micrometres."""

LEGACY_SCHEDULE = FixedSchedule(np.full(STEPS, LEGACY_LENGTH / STEPS))
"""The schedule both legacy runs replay, so that a record is a record of one march."""

# The legacy runs' final planes were byte digests, recorded on one platform and
# re-recorded at each deliberate change (the Multistep default, SEP 3's optmat helium
# and silica wall, the rationalised refractivity root, the 293.15 K default
# temperature); the git history keeps them. They are now stored final planes of the
# fixed-order march, compared within its rounding budget (tests/state_records.py).
# Nothing here is a *reference*: the records are this package's own answer, not an
# independent one. The physics of these two entry points is qualified in
# ``tests/driver/test_prop_capillary.py`` and ``tests/driver/test_prop_gnlse.py``.


def legacy_capillary_keywords(backend):
    """Return the keywords of the ten-step two-mode capillary run of the pre-wall route.

    At the Adams method's fixed top order, so that no order choice is left for
    rounding to move: a run is the recorded run's discrete march.
    """
    return {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-8,
        "modes": 2,
        "n_save": 2,
        "backend": backend,
        "dz_max": LEGACY_LENGTH / STEPS,
        "dz_init": LEGACY_LENGTH / STEPS,
        "schedule": LEGACY_SCHEDULE,
        "integrator": FIXED_ORDER,
        "plasma": False,
        "raman": False,
    }


def legacy_gnlse_keywords(backend):
    """Return the keywords of the ten-step GNLSE run of the soliton case.

    On the grid it was recorded on and without a collar, as the run was before
    the default collar: the record is of that march.
    """
    grid = EnvelopeGrid.from_limits(
        LEGACY_LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return {
        "wavelength": 835e-9,
        "grid": grid,
        "absorber": bare_absorber(grid),
        "width": 100e-15,
        "peak_power": 9.8,
        "shape": "sech",
        "n_save": 2,
        "backend": backend,
        "dz_max": LEGACY_LENGTH / STEPS,
        "dz_init": LEGACY_LENGTH / STEPS,
        "schedule": LEGACY_SCHEDULE,
        "integrator": FIXED_ORDER,
    }


GNLSE_FIBRE = (0.11, LEGACY_LENGTH, [0.0, 0.0, -1.183e-26])
"""The legacy GNLSE run's ``gamma`` (1/(W m)), length (m) and Taylor betas."""


def legacy_capillary(backend):
    """Return a ten-step two-mode capillary run of the pre-wall route."""
    return prop_capillary(
        125e-6,
        LEGACY_LENGTH,
        "He",
        1.0,
        **{"noise": None, **legacy_capillary_keywords(backend)},
    )


def legacy_gnlse(backend):
    """Return a ten-step GNLSE run of the soliton case."""
    return prop_gnlse(*GNLSE_FIBRE, **{"noise": None, **legacy_gnlse_keywords(backend)})


def legacy_simulation(entry):
    """Return the simulation a legacy entry point runs, for its rounding budget."""
    if entry == "prop_capillary":
        return Simulation.from_capillary(
            125e-6,
            LEGACY_LENGTH,
            "He",
            1.0,
            **{"noise": None, **legacy_capillary_keywords("numpy_ref")},
        )
    return Simulation.from_gnlse(
        *GNLSE_FIBRE, **{"noise": None, **legacy_gnlse_keywords("numpy_ref")}
    )


@pytest.mark.medium
@pytest.mark.parametrize("entry", ["prop_capillary", "prop_gnlse"])
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_the_legacy_entry_points_are_what_they_were(
    entry, backend, ndarrays_regression
):
    """Ten steps of each legacy route are the run they were, on every backend here.

    A wall is an **opt-in** leaf, so a run that declares none must be the run it was
    before the leaf existed. The final plane matches its record within the matched
    march's rounding budget (``tests/state_records.py``), which a change to the
    route's operator above rounding exceeds.

    The fixed schedule and the fixed order are what make the statement meaningful.
    Two adaptive runs of operators differing in their last bits place their steps
    differently and agree only to the controller's tolerance, which is orders above
    the rounding this is about.

    Kokkos, whose local and distribution builds (vendor BLAS, CPU target) move last
    bits, has no record of its own: two runs agree within the same budget, and the
    reference's field within :func:`legacy_parity_bound`.
    """
    if backend not in available():
        pytest.skip(f"the {backend} backend is not installed")
    build = legacy_capillary if entry == "prop_capillary" else legacy_gnlse
    handler = cast("MemoryOutput", build(backend))
    field = kept(handler.planes()[-1].field)
    bound = record_bound(legacy_simulation(entry), field)
    if backend != "native":
        check_state(ndarrays_regression, field, bound)
        return
    again = cast("MemoryOutput", build(backend)).planes()[-1].field
    assert_within(again, field, bound, what="a repeated Kokkos run")
    reference = kept(cast("MemoryOutput", build("numpy_ref")).planes()[-1].field)
    scale = float(np.max(np.abs(reference)))
    assert (
        float(np.max(np.abs(field - reference)))
        <= legacy_parity_bound(field.shape[-1], legacy_simulation(entry)) * scale
    )


def legacy_parity_bound(samples: int, simulation) -> float:
    """Relative bound on a legacy run's final field between two backends.

    Every evaluation of each of the ten steps -- DOPRI5's seven stages, or the Adams
    march's one, which reaches the step through its predictor's and corrector's
    weights, counted by their absolute sum (``bounds.step_evaluations``) --
    re-evaluates the nonlinear term
    from a field the two backends formed in different orders: one FFT round trip each
    way (``fft_bound``) and the modal synthesis and projection, whose sums run over at
    most the transform's nodes (``gamma`` of the time samples, an over-count of any
    node set here). Errors of the stages add; the linear step is a unit-modulus phase
    and the case's nonlinear phase is far below one radian, so nothing amplifies them.
    Derived before the local-build comparison was run.
    """
    per_stage = 2 * fft_bound(samples) + 2 * gamma(samples)
    evaluations = max(DOPRI5.n_stages, step_evaluations(simulation.translate()))
    return STEPS * evaluations * per_stage


# ------------------------------------------------------------- the varying section

SMOOTH_STAGES = (0.0, 6e-5, 1.4e-4, 2e-4)
"""Where a taper of this qualification case starts and stops varying, m.

The stages land on multiples of `LENGTH / STEPS`, so the fixed schedule can contain
every hard kink of the radius, which `KernelProblem.validate` requires of any schedule
(`_check_schedule`).
"""

NARROW = 40e-6
"""The taper's narrow end, m: a fifth off :data:`RADIUS` over a fifth of a millimetre.

A 50 to 40 micrometre contraction has the shape of a 125 to 62.5 micrometre taper --
straight, transition, straight -- at the radius and length a short test allows.
"""


def taper(stages=SMOOTH_STAGES, ends=(RADIUS, NARROW)):
    """Return the straight-transition-straight radius profile of the varying cases."""
    return PiecewiseLinear.from_stages(stages, [ends[0], ends[0], ends[1], ends[1]])


def varying(
    *,
    wall=None,
    radius=None,
    pressure: Pressure = 1.0,
    modes: int | Sequence[Mode] = 3,
    **overrides,
):
    """Return the varying-guide qualification case, with or without a wall."""
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": modes,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    if _steps_cf4(wall) and "integrator" not in arguments:
        # CF4 sections are stepped by the Runge-Kutta march alone.
        arguments["integrator"] = RungeKutta()
    return Simulation.from_capillary(
        taper() if radius is None else radius,
        LENGTH,
        "He",
        pressure,
        wall=wall,
        **{"noise": None, **arguments},
    )


def test_a_moving_multimode_set_without_the_connection_warns_by_name():
    """The diagonal route omits the taper's connection and says so, once.

    Only where the omission is real: more than one channel **and** a mode shape that
    moves. A single channel's connection is a gauge phase, a capillary's pressure
    gradient leaves `J_0(u r/a)` where it was, and the coherent wall carries the
    connection -- none of those warns, and neither does the contiguous HE_1m block,
    which carries the taper coupling by default, or its declared opt-out.
    What warns is a moving set the coupling is not derived for: HE_11 beside HE_13. A
    with-input copy shares the translated problem and does not warn a second time.
    """
    moving = varying(modes=[MarcatiliMode(taper(), "He", 1.0, m=m) for m in (1, 3)])
    with pytest.warns(AdiabaticTaperWarning, match="wall=CoherentWall") as caught:
        moving.translate()
    assert len([w for w in caught if w.category is AdiabaticTaperWarning]) == 1
    assert "2 channels" in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AdiabaticTaperWarning)
        moving.translate()
        moving.with_input(moving._field0.copy()).translate()
        varying(modes=1).translate()
        capillary(pressure=(1.0, 0.1)).translate()
        capillary().translate()
        varying(wall=CoherentWall(), modes=2).translate()
        # The contiguous block carries the taper coupling by default, and
        # a declared opt-out is not warned about: it asked for the adiabatic route.
        assert varying(modes=2).translate().linear.coupled_flow is not None
        varying(modes=2, taper_coupling=False).translate()


def test_the_varying_leaf_declares_a_frozen_section_per_interval_with_a_connection():
    """A taper builds a frozen-residual flow whose sections are the table's own.

    Bound, stated first. The generator is frozen at each section's midpoint, so the
    leaf's own `frozen_generator` must reproduce `wall_operator` evaluated there: the
    two go through the same conjugation and the same carrier subtraction, and the only
    difference is the Chebyshev interpolation along z, whose relative tail the leaf
    declares in `generator_error` and which is below 1e-11 on this case. The residual
    row is an **enclosure** over each whole section -- `sum_k ||r_k||_2` with
    `|T_k| <= 1` -- so it must bound the residual sampled anywhere inside it, which is
    checked at seventeen positions per section.
    """
    problem = varying(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert set(flow.section_methods) == {"frozen_residual"}
    assert flow.connection is not None
    assert_allclose(flow.z_edges, problem.linear.phi_breakpoints, rtol=0, atol=0)
    assert float(np.max(flow.generator_error)) < 1e-11
    block = [MarcatiliMode(taper(), "He", 1.0, m=index + 1) for index in range(3)]
    for index in range(len(flow.section_methods)):
        lo, hi = float(flow.z_edges[index]), float(flow.z_edges[index + 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = wall_operator(
                block,
                GRID.omega,
                z=0.5 * (lo + hi),
                active=GRID.active_mask,
                validity_limit=math.inf,
            )
        frozen = frozen_generator(flow, index)
        scale = float(np.max(abs(frozen)))
        # The stored diagonal carries the carrier shift conj(i(k - beta_11)) as well,
        # and that shift is the same for every channel, so what is compared there is
        # the difference between two channels, which it cancels out of exactly. It
        # also carries each channel's normalisation beyond the lowest one's,
        # -0.5 d ln(w_n/w_1)/dz, taken here by a fourth-order difference
        # of the implemented log weight, independent of the analytic slope the route
        # uses: its rounding, 3e-16/1e-6 m, is far under the atol below.
        mine = np.diagonal(np.conj(frozen), axis1=0, axis2=1)
        theirs = np.diagonal(direct.generator, axis1=0, axis2=1) + np.transpose(
            _normalisation_difference(block, 0.5 * (lo + hi))
        )
        assert_allclose(
            mine[:, 1:] - mine[:, :1],
            theirs[:, 1:] - theirs[:, :1],
            rtol=1e-9,
            atol=1e-11 * scale,
        )
        off = ~np.eye(3, dtype=bool)
        assert_allclose(
            np.conj(frozen)[off], direct.generator[off], rtol=1e-9, atol=1e-11 * scale
        )
        # The right edge belongs to the *next* section, whose own bound covers it.
        for position in np.linspace(lo, hi, 18)[:-1]:
            weights = flow_weights(flow, float(position))
            terms = flow.generator - flow.connection
            residual = (
                np.tensordot(weights, terms.reshape(-1, 3, 3, GRID.n_omega), axes=1)
                - frozen
            )
            norms = np.linalg.norm(
                np.moveaxis(residual, (0, 1), (-2, -1)), ord=2, axis=(-2, -1)
            )
            assert np.all(norms <= flow.residual_norm_bound[index] * (1.0 + 1e-9))


def _normalisation_difference(block, z, step=1e-6):
    """Return `-0.5 d ln(w_n/w_1)/dz` on the grid by a difference of `ln Re n_eff`.

    Zero off the active band, as the stored generator is. Fourth order in `step`;
    `log1p` of the real refractivity keeps the digits a logarithm of `1 + x` formed
    in floating point would lose.
    """
    band = GRID.omega[GRID.active_mask]

    def logarithm(position):
        return np.stack(
            [np.log1p(np.real(mode.neff_minus_one(band, position))) for mode in block]
        )

    slope = (
        8.0 * (logarithm(z + step / 2) - logarithm(z - step / 2))
        - (logarithm(z + step) - logarithm(z - step))
    ) / (6.0 * step)
    rate = np.zeros((len(block), GRID.n_omega))
    rate[:, GRID.active_mask] = -0.5 * (slope - slope[:1])
    return rate


def test_section_refinement_reduces_the_residual_at_the_rate_freezing_predicts():
    """Refining the sections shrinks the residual; what it does not shrink is named.

    Bounds, stated first. Across a section of width `l` the frozen generator is the
    midpoint value, so `G(z) - G(z_s)` is `G'(z_s)(z - z_s)` to leading order and its
    enclosure falls as `l`: halving every section must roughly halve it, which is a
    rate of one. Measured on a pressure gradient, which carries **no connection** and
    whose residual is therefore the frozen variation alone: 3.728e-2, 2.269e-2,
    1.322e-2, 7.357e-3, 3.939e-3 and 2.050e-3 1/m over five, ten, twenty, forty,
    eighty and a hundred and sixty sections, for rates of 0.716, 0.780, 0.845, 0.901
    and 0.942.

    The rate approaches one **from below**, and the reason is where the maximum sits
    rather than how the widths are cut: `_section_edges` halves every width exactly,
    so the widths are not the asymmetry. What is asymmetric is the profile. The bound
    reported is the largest over all sections, and it is attained on whichever section
    covers the steepest part of `G`; as the sections narrow, that one localises
    further onto the steepest point, where `|G'|` is larger than the section mean it
    replaced. The enclosure therefore falls by a little less than the width does at
    every halving, and the deficit shrinks as the localisation completes -- 0.716 to
    0.942 over five halvings. Predeclared gate: every rate above 0.6, and the last
    above the first.

    **Two things refinement does not reduce, as expected.** The taper
    connection `C(z)` is wholly explicit -- it is not frozen and nothing subtracts it
    -- so a taper's residual tends to `||C||` and falls only from 9465 to 8246 1/m
    over four times the sections. And the inverse amplification
    `exp(dz_max (mu_max - mu_min))` is a property of the **frozen block and the step
    cap**, not of the section width, so it is unchanged to the last bit: what bounds
    it is `dz_max`, which is the knob for it.
    """
    bounds = []
    for sections in (1, 2, 4, 8, 16, 32):
        flow = (
            varying(
                wall=CoherentWall(sections=sections), radius=RADIUS, pressure=(1.0, 0.1)
            )
            .translate()
            .linear.coupled_flow
        )
        assert flow is not None
        bounds.append(float(np.max(flow.residual_norm_bound)))
    rates = [
        math.log2(bounds[index] / bounds[index + 1]) for index in range(len(bounds) - 1)
    ]
    assert min(rates) > 0.6, (bounds, rates)
    # Converging to one from below, not sitting at a constant deficit.
    assert rates[-1] > rates[0], (bounds, rates)
    tapered = []
    conditionings = []
    for sections in (1, 4):
        flow = (
            varying(wall=CoherentWall(sections=sections))
            .translate()
            .linear.coupled_flow
        )
        assert flow is not None
        tapered.append(float(np.max(flow.residual_norm_bound)))
        conditionings.append(float(np.max(flow.inverse_condition_bound)))
    assert tapered[1] > 0.8 * tapered[0]
    assert conditionings[1] == conditionings[0]


def test_the_frozen_split_is_a_rearrangement_and_not_an_approximation():
    """Where the generator is frozen changes the answer only by the step's own error.

    The claim: with R evaluated exactly, freezing is a rearrangement of the original
    equation, not a second-order approximation. Two runs that freeze
    at different places -- one section per interval against four -- therefore solve
    the *same* equation with different explicit and implicit shares, and must agree to
    the embedded pair's own local error over the step, not to the size of the
    residual they moved between the two.

    Bound, stated first. The two declare residual enclosures of 9465 and 8246 1/m,
    and the blocks they freeze differ by the section's own variation; if freezing were
    an approximation the two runs would differ by that difference over the length,
    which is 1.6e-1 of the state. The fifth-order pair's local defect over one step is
    `(h K)^6/3600 + e^{hK}(h K)^7/5040`, which is `residual_truncation(h K)`, and ten
    steps of it -- 1.49e-7 -- is the gate. Measured **1.85e-10** of the state's own
    size, three orders inside it.
    """
    # The gate is DOPRI5's local defect: the Runge-Kutta march's own property.
    coarse = propagated(
        varying(wall=CoherentWall(), integrator=RungeKutta()).translate(), GRID
    )
    fine = propagated(
        varying(wall=CoherentWall(sections=4), integrator=RungeKutta()).translate(),
        GRID,
    )
    scale = float(np.max(abs(coarse)))
    declared = varying(wall=CoherentWall()).translate().linear.coupled_flow
    assert declared is not None
    residual = float(np.max(declared.residual_norm_bound))
    gate = STEPS * residual_truncation(residual * LENGTH / STEPS)
    departure = float(np.max(abs(fine - coarse))) / scale
    assert departure < gate, (departure, gate)


def test_the_accumulated_error_is_verified_by_independent_refinement():
    """Halving the step reduces the departure by 32: the run converges at order five.

    The accumulated error of the explicit residual must be verified by independent
    refinement rather than trusted to the estimate. The refinement here is of the
    **schedule**, not of the sections, so the equation is fixed and only its integration
    changes; the embedded DOPRI5 pair is fifth order, so halving the step must reduce
    the departure from the finest run by 32. Predeclared gate: a measured order above
    4.0 across two halvings. Measured 4.87 and 4.95, against a 160-step reference; the
    run itself is 3.4e-9 of the state at ten steps, so the residual is being integrated
    and not merely held.

    Every step count is a multiple of :data:`STEPS`, because the taper's corners are
    and a fixed schedule must land on every hard kink.
    """
    fine = _refined(STEPS * 16)
    scale = float(np.max(abs(fine)))
    departures = [
        float(np.max(abs(_refined(STEPS * count) - fine))) for count in (1, 2, 4)
    ]
    assert departures[0] / scale < 1e-8
    orders = [
        math.log2(departures[index] / departures[index + 1]) for index in range(2)
    ]
    assert min(orders) > 4.0, (departures, orders)


def _refined(steps):
    """Return the final plane of the varying wall case on a schedule of `steps`.

    Stepped by the Runge-Kutta march, whose fifth order the refinement measures.
    """
    return propagated(
        varying(
            wall=CoherentWall(),
            integrator=RungeKutta(),
            schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
            dz_max=LENGTH / steps,
            dz_init=LENGTH / steps,
        ).translate(),
        GRID,
    )


def _true_power_reference(block, index):
    """Return `b(L)` of one bin, solved in the true-power amplitude by DOP853.

    Independent of the route's split: `b = W^(1/2) a` with `W = diag(Re n_eff,n)`
    carries **each channel's own** normalisation by construction, and its equation
    `b' = (W^(1/2) (i k I + G) W^(-1/2) - C) b` has no normalisation term and no
    derivative of the index in it at all, and its connection is antisymmetric in `b`
    itself, which is what makes a lossless taper conserve power. `G` and `k` are
    `wall_operator`'s at every position the solver asks for, `C` is `wall_connection`'s,
    and the real part of `k` -- a phase common to every channel, 1600 rad over the
    length -- is left out, which changes no ratio and no modulus. Launched as
    `a(0) = 1e-9` in `HE_11`, solved at rtol 1e-13 across each of the taper's straight
    and transition stages.
    """
    omega = GRID.omega[index : index + 1]
    count = len(block)

    def weights(z):
        return np.array([float(np.real(mode.neff(omega, z))[0]) for mode in block])

    def derivative(z, state):
        amplitude = state[:count] + 1j * state[count:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            operator = wall_operator(block, omega, z=z, validity_limit=math.inf)
        wavenumber = complex(operator.wavenumber[0])
        generator = operator.generator[:, :, 0] + 1j * (
            wavenumber - wavenumber.real
        ) * np.eye(count)
        root = np.sqrt(weights(z))
        change = (
            root[:, None] * generator / root[None, :] - wall_connection(block, z)
        ) @ amplitude
        return np.concatenate([np.real(change), np.imag(change)])

    launch = np.zeros(count, dtype=complex)
    launch[0] = 1e-9 * math.sqrt(weights(0.0)[0])
    state = np.concatenate([np.real(launch), np.imag(launch)])
    for lo, hi in itertools.pairwise(SMOOTH_STAGES):
        state = solve_ivp(
            derivative, (lo, hi), state, method="DOP853", rtol=1e-13, atol=1e-30
        ).y[:, -1]
    return state[:count] + 1j * state[count:]


def _routed_true_power(block, index, steps=None, rtol=1e-10):
    """Return the route's `b(L)` of one bin, launched as the reference is.

    On a schedule of `steps` stepped by the Runge-Kutta march, whose order the
    refinement gates; with ``steps=None`` adaptively, by the default integrator, at
    `rtol`.
    """
    stepping = (
        {"rtol": rtol, "atol": 0.0, "schedule": None, "dz_init": LENGTH * 1e-4}
        if steps is None
        else {
            "integrator": RungeKutta(),
            "schedule": FixedSchedule(np.full(steps, LENGTH / steps)),
            "dz_max": LENGTH / steps,
            "dz_init": LENGTH / steps,
        }
    )
    problem = varying(wall=CoherentWall(), **stepping).translate()
    state = np.zeros_like(problem.initial.field0)
    state[0, 0, 0, index] = 1e-9
    launched = replace(problem, initial=replace(problem.initial, field0=state))
    internal = propagated(launched, GRID)[0, :, 0, index]
    omega = GRID.omega[index : index + 1]
    weight = np.array([float(np.real(mode.neff(omega, LENGTH))[0]) for mode in block])
    # The internal amplitude is the FFT convention's: conjugated, it is the optics
    # amplitude the reference solves for.
    return np.conj(internal) * np.sqrt(weight)


@pytest.mark.medium
@pytest.mark.xdist_group("coherent-channel-power-reference")
def test_every_channel_carries_its_own_normalisation_against_a_true_power_solve(
    monkeypatch,
):
    """No per-channel part is omitted: the upper channels match a true-power solve.

    Bounds, stated first. The quantity is each upper channel's amplitude **relative
    to `HE_11`'s**, `b_n/b_1`, which the omitted per-channel part
    `-0.5 ln[(w_n(L)/w_n(0)) (w_1(0)/w_1(L))]` moves and which every channel-common
    factor -- the carrier's phase, the co-moving frame, the temporal and spectral
    collars -- leaves alone. With the lowest channel's term for every channel that
    part reaches 2.2e-5 and 6.3e-5 at 800 nm and more towards the long edge, and the
    converted amplitude, born inside the transition, sees a share of it: that route
    departs from the reference by **more than 1e-6** at both bins (measured 3.3e-5 at
    800 nm and 1.3e-4 at the lowest in-band bin, 1.59 um). With each channel's own
    term the departure is the schedule's alone: below 1e-7 at ten steps (the flow's
    certified 3.45e-9 of the state, over an upper channel carrying a fifth of
    `HE_11`'s amplitude) and falling by more than 16 on halving the step, i.e. at order
    above four. Measured 1.7e-8 and 2.0e-8 at ten steps, 32-fold per halving. Those
    ten-step runs are the Runge-Kutta march's, whose order the gate measures; the
    default integrator, stepped adaptively at ``rtol=1e-10``, meets the same reference
    to below 1e-9 (measured 5.5e-11 and 6.4e-11, its departure following its tolerance).

    The band edge is the error norm's, not the active band's: in the spectral collar the
    explicit residual, and with it the connection, is not applied and the frozen flow
    alone carries the state
    (:func:`test_the_collar_is_carried_by_the_frozen_flow_alone`).

    `|b_1|` itself is checked too, after the spectral collar's `exp(-r L/2)`, to the
    temporal collar's measured 2.2e-7 at every bin (bound 5e-7).
    """
    block = [MarcatiliMode(taper(), "He", 1.0, m=index + 1) for index in range(3)]
    edge = int(np.flatnonzero(GRID.inband_mask)[0])
    centre = int(np.argmin(abs(GRID.omega - GRID.omega0)))
    for index in (centre, edge):
        reference = _true_power_reference(block, index)
        expected = reference[1:] / reference[0]
        departures = []
        for steps in (STEPS, 2 * STEPS):
            routed = _routed_true_power(block, index, steps)
            departures.append(float(np.max(abs(routed[1:] / routed[0] / expected - 1))))
        assert departures[0] < 1e-7, (index, departures)
        assert departures[0] / departures[1] > 16.0, (index, departures)
        default = _routed_true_power(block, index, rtol=1e-10)
        assert float(np.max(abs(default[1:] / default[0] / expected - 1))) < 1e-9
        collar = math.exp(-0.5 * float(WEAK_COLLAR.spectral_rate[index]) * LENGTH)
        assert abs(abs(routed[0]) / (collar * abs(reference[0])) - 1.0) < 5e-7
        # The lowest channel's normalisation for every channel.
        with monkeypatch.context() as patch:
            patch.setattr(
                problem_module,
                "_channel_normalisation_at",
                lambda modes, grid, z: np.zeros((len(modes), grid.n_omega)),
            )
            lowest = _routed_true_power(block, index, STEPS)
        assert float(np.max(abs(lowest[1:] / lowest[0] / expected - 1))) > 1e-6


def test_the_stored_connection_conserves_true_power():
    """The connection is antisymmetric in `b = W^(1/2) a`, not in the kernel's `a`.

    Coupled local-mode theory writes the taper's coupling in power-normalised modes,
    so a lossless taper conserves `sum |b_n|^2 = sum w_n |a_n|^2` exactly, and
    `wall_connection` is that antisymmetric matrix. The kernel propagates `a`, so the
    stored term is `W^(-1/2) C W^(1/2)`. Bounds:
    carried back into `b` the stored term is antisymmetric to rounding, 1e-12 of its
    own size, at every active bin; the unweighted `C` stored directly in `a` is not,
    by the weights' spread, gated above 1e-5 of its size, which is the power a
    lossless taper would lose with it (1.5e-5 of the total at 800 nm, 2.2e-4 at the
    band edge, on the qualification taper).
    """
    block = tuple(MarcatiliMode(taper(), "He", 1.0, m=index + 1) for index in range(3))
    z = 0.5 * LENGTH
    stored = problem_module._wall_connection_at(block, GRID, z)
    band = GRID.omega[GRID.active_mask]
    root = np.sqrt(np.stack([np.real(mode.neff(band, z)) for mode in block]))
    active = stored[:, :, GRID.active_mask]
    carried = root[:, None, :] * active / root[None, :, :]
    scale = float(np.max(abs(carried)))
    assert scale > 0.0
    assert float(np.max(abs(carried + np.swapaxes(carried, 0, 1)))) < 1e-12 * scale
    plain = wall_connection(block, z)[:, :, None] * np.ones(band.size)
    departure = root[:, None, :] * plain / root[None, :, :]
    assert float(np.max(abs(departure + np.swapaxes(departure, 0, 1)))) > 1e-5 * scale
    assert not np.any(stored[:, :, ~GRID.active_mask])


def test_the_section_count_is_checked_at_the_door():
    """`CoherentWall.sections` is a positive integer and nothing else.

    `2.0` is in the list because it used to pass: the check was `int(x) != x`, which
    a whole-valued float satisfies, so the declaration translated and ran and only
    failed when something printed it -- `__repr__` formats the count with `"d"`.
    That is a refusal by the wrong layer at the wrong time, and the check is
    `operator.index` now. It is asserted here rather than left to the test session's
    runtime type check, which production never installs.
    """
    for bad in (0, -1, True):
        with pytest.raises(ModeError, match="sections"):
            CoherentWall(sections=bad)
    for bad in (2.0, "many"):
        # Built without `__init__`, because the test session's runtime type check
        # would stop these at the annotation and production installs no such hook:
        # what is under test is the dataclass's own door, on its own.
        wall = object.__new__(CoherentWall)
        object.__setattr__(wall, "condition_limit", 1.0e4)
        object.__setattr__(wall, "validity_limit", 0.25)
        object.__setattr__(wall, "sections", bad)
        with pytest.raises(ModeError, match="sections"):
            wall.__post_init__()
    assert CoherentWall(sections=3).sections == 3
    assert repr(CoherentWall(sections=3)).endswith("sections=3)")


def test_a_fixed_schedule_that_straddles_a_taper_corner_is_refused_by_name():
    """A schedule edge near a kink but not on it costs the run its order, so it is out.

    Bound, stated first, and it is a gap between two tolerances rather than a
    physical one. `_check_schedule` accepts an edge within `1e-9` of the run of a
    hard kink; `flow_edge_slack` matches a stage to a section within `1e-10` of the
    flow's span, which is 2e-14 m here. An edge in between -- 1e-13 m short of the
    6e-5 m corner -- was accepted, anchored its step in the section closing at the
    corner, and then ran past it: the residual that step's last stage saw jumped
    from 0 to 6.2e3 1/m, and the error against a 40-step reference went from 4.8e-9
    to 5.6e-3, first order instead of fifth, with nothing said.

    The two windows are one quantity now, and the gap between them is named. The
    exact schedule is the control: it is the same run, and it translates.
    """
    edges = np.linspace(0.0, LENGTH, STEPS + 1)
    exact = varying(wall=CoherentWall(), schedule=FixedSchedule(np.diff(edges)))
    assert exact.translate().linear.coupled_flow is not None
    straddled = edges.copy()
    # A tenth of a picometre short of the 6e-5 m corner: inside `_check_schedule`'s
    # own 1e-9 m window and five times outside the flow's 2e-14 m one.
    straddled[3] -= 1e-13
    with pytest.raises(TranslationError, match=r"does not land on it.*coupled_flow"):
        varying(
            wall=CoherentWall(), schedule=FixedSchedule(np.diff(straddled))
        ).translate()


def test_a_varying_wall_refuses_every_declaration_it_would_otherwise_override():
    """Every door on the constant route still stands on the varying one.

    The rule: a declared option is honoured or refused by name. The doors themselves are
    shared -- `_wall_block_modes`, `_check_wall_channels`, `_check_wall_route` -- so
    what this checks is that a **taper** reaches all of them rather than slipping past
    on the new branch, which is the way a new route usually loses an admission.
    """
    with pytest.raises(DriverError, match="loss=False"):
        varying(wall=CoherentWall(), loss=False).translate()
    with pytest.raises(DriverError, match="one capillary model"):
        varying(wall=CoherentWall(), model="reduced").translate()
    with pytest.raises(DriverError, match="tube lattice"):
        varying(
            wall=CoherentWall(), lattice=TubeLattice(180e-9, 10.38e-6, 6)
        ).translate()
    sparse = [MarcatiliMode(taper(), "He", 1.0, m=index) for index in (1, 3)]
    with pytest.raises(ModeError, match="contiguous Marcatili set"):
        varying(wall=CoherentWall(), modes=sparse).translate()
    # Two tapers are one declared radius disagreeing with a supplied mode, which the
    # entry point refuses first; the wall's own door still names a channel
    # that declares another angle on the same guide.
    shared = taper()
    with pytest.raises(DriverError, match="radius disagrees with supplied mode 1"):
        varying(
            wall=CoherentWall(),
            radius=shared,
            modes=[
                MarcatiliMode(shared, "He", 1.0, m=1),
                MarcatiliMode(taper(ends=(RADIUS, 45e-6)), "He", 1.0, m=2),
            ],
        ).translate()
    with pytest.raises(ModeError, match="one guide"):
        varying(
            wall=CoherentWall(),
            radius=shared,
            modes=[
                MarcatiliMode(shared, "He", 1.0, m=1),
                MarcatiliMode(shared, "He", 1.0, m=2, phi=0.3),
            ],
        ).translate()
    with pytest.raises(TranslationError, match="refuses the 'modified' noise"):
        varying(wall=CoherentWall(), noise="modified", seed=1).translate()
    with pytest.raises(DriverError, match="polarisation"):
        varying(wall=CoherentWall(), polarisation="linear").translate()


def test_a_frozen_residual_flow_is_refused_beside_a_carrier_it_cannot_share(
    monkeypatch,
):
    """A z table beside a varying flow is refused: the two weights share one bundle.

    The constant-Schur flow requires the exact constant `Phi`; the
    varying one requires only that `Phi` be a **weighted sum of fixed terms**, which
    is what the stage bundle is, and a mode set whose parameters do not resolve falls
    back to the z table, which has no such form. The refusal names it rather than
    letting the flow's weights and the carrier's come from two different places.

    Nothing this package builds takes the table beside a capillary whose parameters
    are its radius and its density, so the predicate is forced rather than provoked
    -- the check is a statement about the representation, not about a run.
    """
    problem = varying(wall=CoherentWall()).translate()
    monkeypatch.setattr(type(problem), "device_ready", property(lambda self: False))
    with pytest.raises(BackendError, match="took the z table"):
        refuse_coupled_flow(problem)
    constant = capillary(wall=CoherentWall()).translate()
    with pytest.raises(BackendError, match="exact constant representation"):
        refuse_coupled_flow(
            replace(
                problem,
                linear=replace(
                    problem.linear, coupled_flow=constant.linear.coupled_flow
                ),
            )
        )


def test_the_stage_bundle_carries_the_frozen_section_and_the_residual_weights():
    """Schema 13's two rows are the step's section and the stage's own weights.

    The frozen section is the **step anchor's**, so every stage of one step names the
    same block; the weights are the stage position's, are as long as the flattened
    `(n_intervals, n_deg)` coefficient block, and are zero outside the one section the
    stage is evaluated on -- so no expansion is ever extrapolated. A problem with no
    coupled flow declares `-1` and `None`, which is what makes a schema-13 problem
    without one the schema-12 problem bit for bit.
    """
    problem = varying(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    width = len(flow.section_methods) * flow.n_deg
    bundle = scalars(problem, 8e-5, 9e-5)
    assert bundle.flow_section == 1
    assert bundle.flow_weights is not None
    assert bundle.flow_weights.shape == (width,)
    block = bundle.flow_weights.reshape(len(flow.section_methods), flow.n_deg)
    assert np.count_nonzero(block[0]) == 0
    assert np.any(block[1])
    for stage in (8e-5, 9e-5, 1.0e-4):
        assert scalars(problem, 8e-5, stage).flow_section == 1
    # The adiabatic taper, the declared opt-out, is the comparison.
    plain = varying(taper_coupling=False).translate()
    assert plain.linear.coupled_flow is None
    bare = scalars(plain, 8e-5, 9e-5)
    assert bare.flow_section == -1
    assert bare.flow_weights is None


def test_a_declared_step_above_the_advisory_limit_warns_by_name():
    """The `validate` diagnostic names the term, the section, K and the limit.

    The `stable_step` warning is advisory only: it changes no tolerance and no schedule,
    it says plainly that a growth certificate is still required, and it reports the
    local frozen truncation defect `x^6/3600 + e^x x^7/5040` of the step it objects to.
    The limit is `0.9/(K ||U|| ||U^-1||)` from the two certificate rows the leaf already
    carries, taken over the bins the error norm scores, which are the only ones the
    residual is applied on: on the qualification taper it is 1.05e-4 m, where the collar
    alone would have said 9.42e-5 m. So the ten-step schedule at 2e-5 m is inside it, a
    1e-4 m cap is too, and a 2e-4 m cap is not.
    """
    problem = varying(wall=CoherentWall()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    scored = problem.grids.inband_mask
    limit = float(np.min(advisory_step(flow, scored)))
    everywhere = advisory_step(flow)
    assert 1e-4 < limit < 2e-4
    assert float(np.min(everywhere)) < 1e-4
    assert not scored[np.unravel_index(int(np.argmin(everywhere)), everywhere.shape)[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", SolpropWarning)
        warnings.filterwarnings("ignore", "the temporal absorber attenuates")
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        varying(
            wall=CoherentWall(),
            dz_max=1e-4,
            dz_init=1e-4,
            schedule=None,
        ).translate()
    with (
        warnings.catch_warnings(),
        pytest.warns(SolpropWarning, match="advisory step"),
    ):
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        varying(
            wall=CoherentWall(),
            dz_max=2e-4,
            dz_init=2e-4,
            schedule=None,
        ).translate()


def test_prop_capillary_runs_a_varying_wall_end_to_end_and_records_its_flow():
    """The entry point runs a taper, and the record says which flow it took.

    The one end-to-end test each flow route carries: ten steps
    through `prop_capillary(..., wall=...)`, not a hand-driven backend, because the
    record and the problem are checked against each other only there.

    Bound, stated first. A taper from 50 to 40 micrometres loses more than the
    straight 50 micrometre bore and less than a straight 40 micrometre one, because
    the field-loss rate goes as `u^2 v_n/(k_0 a)^3 /a` and the taper's radius lies
    between the two everywhere; the run's own energy loss must therefore be bracketed
    by the two constant-bore runs, and it is: 4.409e-4 < 6.415e-4 < 8.576e-4.

    The **walled** taper loses more than either, and that is the point: the
    connection converts the launch into the higher orders, whose loss goes as
    `u_m^2`. Measured 1.888e-3 (1.890e-3 with the lowest channel's normalisation for
    every channel), with 3.83e-2 of the final energy in `HE_12` and
    3.30e-2 in `HE_13` against 1.00 in `HE_11` -- where the uncoupled taper leaves
    5e-18 and a walled *straight* bore 1e-6. The record names four
    `"frozen_residual"` sections, carries a connection, and reports the **narrow**
    end's validity ratios, which are the worst.
    """
    walled = cast(
        "MemoryOutput",
        prop_capillary(
            taper(),
            LENGTH,
            "He",
            1.0,
            wall=CoherentWall(),
            wavelength=800e-9,
            grid=GRID,
            absorber=WEAK_COLLAR,
            fwhm=30e-15,
            energy=1e-9,
            modes=3,
            n_save=2,
            backend="numpy_ref",
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
            plasma=False,
            raman=False,
            noise=None,
        ),
    )
    written = walled.metadata
    assert written is not None and written.modal is not None
    assert capability_of(written) == "modal"
    record = cast("Mapping[str, object]", written.modal["flow"])
    assert record["section_methods"] == ["frozen_residual"] * 4
    assert record["n_sections"] == 4
    assert record["connection"] is True
    assert record["radius"] == NARROW
    assert float(cast("float", record["advisory_step"])) > LENGTH / STEPS
    # The adiabatic taper, the declared opt-out, is the control.
    plain = cast(
        "MemoryOutput",
        prop_capillary(
            taper(),
            LENGTH,
            "He",
            1.0,
            wavelength=800e-9,
            grid=GRID,
            absorber=WEAK_COLLAR,
            fwhm=30e-15,
            energy=1e-9,
            modes=3,
            n_save=2,
            backend="numpy_ref",
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
            plasma=False,
            raman=False,
            taper_coupling=False,
            noise=None,
        ),
    )
    wide = energy_loss(run_capillary(wall=CoherentWall(), modes=3))
    narrow = energy_loss(
        run_capillary(wall=CoherentWall(), modes=3, radius_override=NARROW)
    )
    # The uncoupled taper is the control, and it is bracketed by the two bores it
    # runs between: 4.409e-4 < 6.415e-4 < 8.576e-4.
    assert wide < energy_loss(plain) < narrow
    # The walled taper loses more than either, because the connection converts the
    # launch into the higher orders, whose loss goes as u_m^2.
    assert energy_loss(walled) > narrow
    converted = _channel_fractions(walled)
    assert converted[0] == pytest.approx(0.9287, abs=5e-3)
    assert float(np.sum(converted[1:])) > 0.06
    # The uncoupled route cannot convert at all: its taper leaves 1e-18 there, which
    # is the difference a moving basis makes.
    assert float(np.sum(_channel_fractions(plain)[1:])) < 1e-15


def _channel_fractions(handler):
    """Return the last plane's energy per channel, as a fraction of its own total."""
    energies = np.asarray(handler.planes()[-1].stats["energy"]).ravel()
    return energies / float(np.sum(energies))


def smooth_taper(ends=(RADIUS, NARROW), stages=SMOOTH_STAGES):
    """Return the cubic smooth-step taper, with its slope.

    ``a(z) = a0 + (a1 - a0) (3 x^2 - 2 x^3)`` across the transition, which is `C^1` at
    both ends -- the slope vanishes there rather than jumping -- and whose peak slope
    is **1.5 times** the linear one.
    """
    lo, hi = float(stages[1]), float(stages[2])
    width = hi - lo
    drop = ends[1] - ends[0]

    def value(z):
        x = np.clip((np.asarray(z) - lo) / width, 0.0, 1.0)
        return ends[0] + drop * x * x * (3.0 - 2.0 * x)

    def slope(z):
        x = np.clip((np.asarray(z) - lo) / width, 0.0, 1.0)
        return drop * 6.0 * x * (1.0 - x) / width

    return FromCallable(
        value, (0.0, float(stages[-1])), kinks=np.array([lo, hi]), slope=slope
    )


def test_a_smooth_and_a_kinked_taper_both_certify_and_the_smooth_one_is_stiffer():
    """Both taper shapes build a certified flow, and the smooth one is worse.

    Bound, stated first: the cubic smooth step's peak slope is **1.5 times** the
    linear ramp's over the same transition, so its peak connection -- and with it the
    residual enclosure the flow declares -- is larger by about that factor (a 125 to
    62.5 micrometre taper gives `rho(C)` of 4267 against 3995). Measured here: 1.42
    times, the ratio of the enclosures rather than of the peaks, which the straight
    sections dilute.
    Predeclared gate: the smooth one is the larger, by between 1.1 and 1.6.

    "Smooth" is about the *slope*: both shapes still kink where the transition starts
    and stops -- the second derivative jumps there -- and both declare those two
    positions, which is what puts a section edge on each of them.
    """
    kinked = varying(wall=CoherentWall()).translate().linear.coupled_flow
    smooth = (
        varying(wall=CoherentWall(), radius=smooth_taper())
        .translate()
        .linear.coupled_flow
    )
    assert kinked is not None and smooth is not None
    for flow in (kinked, smooth):
        assert set(flow.section_methods) == {"frozen_residual"}
        assert flow.connection is not None
        assert float(np.max(flow.generator_error)) < 1e-9
        assert float(np.max(flow.inverse_condition_bound)) < 1.1
    ratio = float(np.max(smooth.residual_norm_bound)) / float(
        np.max(kinked.residual_norm_bound)
    )
    assert 1.1 < ratio < 1.6, ratio


def test_the_record_and_the_admission_warning_name_one_geometry():
    """What the record calls "worst" is what the run reaches, not what it passes.

    Bound, stated first, and it is an agreement rather than a number. The admission
    samples the Lobatto nodes of every section, which include the section ends; the
    record was scanned over the **midpoints** alone. On a straight 50 to 30
    micrometre taper those differ by a whole section half-width: the record said
    `radius = 4e-5` and `quasi_static = 0.113666` for a run that reaches `3e-5` and
    `0.151554`, a third optimistic in every ratio, and disagreed with the warning the
    same construction raises. The scan is over the edges and the midpoints now, so
    the two name one geometry -- which is what a resume key and a printed record are
    for.
    """
    simulation = varying(wall=CoherentWall(), radius=Linear(RADIUS, 30e-6, LENGTH))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulation.translate()
    named = [
        str(entry.message)
        for entry in caught
        if issubclass(entry.category, GrazingIncidenceWarning)
    ]
    identity = simulation._modal_identity()
    assert identity is not None
    record = cast("Mapping[str, object]", identity["flow"])
    assert record["radius"] == 30e-6
    quasi_static = float(cast("float", record["quasi_static"]))
    assert any(f"{quasi_static:.4g}" in message for message in named), (
        named,
        quasi_static,
    )


def test_the_flow_record_of_a_varying_guide_names_the_profile_it_was_built_from(
    tmp_path,
):
    """`modal.flow` round trips, and a resume across a changed profile is refused.

    The record holds scalars and strings alone, so the round trip through the file's
    JSON is exact. What makes it a *profile* key is that the sections, their count and
    the worst section's radius are in it: a run resumed against the same file with a
    taper that ends somewhere else has a different `radius` and a different
    `residual_norm_bound`, and is refused by name rather than continued.
    """
    path = tmp_path / "taper.h5"
    varying(wall=CoherentWall()).run(path)
    stored = open_file(path)
    try:
        read_back = stored.metadata
        assert read_back is not None and read_back.modal is not None
        assert capability_of(read_back) == "modal"
        record = cast("Mapping[str, object]", read_back.modal["flow"])
        assert record["n_sections"] == 4
        assert record["radius"] == NARROW
    finally:
        stored.close()
    with pytest.raises(OutputError, match="different run"):
        varying(wall=CoherentWall(), radius=taper(ends=(RADIUS, 45e-6))).run(
            path, resume=True
        )
    with pytest.raises(OutputError, match="different run"):
        varying(wall=CoherentWall(), radius=smooth_taper()).run(path, resume=True)


# This case's final plane was a byte digest; it is now a stored final plane of the
# fixed-order march, compared within its rounding budget (tests/state_records.py). The
# former digests and their re-recordings are in the git history.


@pytest.mark.medium
def test_the_varying_wall_route_is_pinned_to_its_recorded_answer(ndarrays_regression):
    """Ten steps of the qualification taper on `numpy_ref`, as recorded."""
    handler = cast(
        "MemoryOutput",
        prop_capillary(
            taper(),
            LENGTH,
            "He",
            1.0,
            wall=CoherentWall(),
            wavelength=800e-9,
            grid=GRID,
            absorber=WEAK_COLLAR,
            fwhm=30e-15,
            energy=1e-9,
            modes=3,
            n_save=2,
            backend="numpy_ref",
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
            integrator=FIXED_ORDER,
            plasma=False,
            raman=False,
            noise=None,
        ),
    )
    field = handler.planes()[-1].field
    simulation = varying(wall=CoherentWall(), integrator=FIXED_ORDER)
    check_state(ndarrays_regression, field, record_bound(simulation, field))


def test_the_varying_flow_constructor_checks_its_own_arguments():
    """Every field `frozen_residual_flow` is handed is checked before it is used."""
    omega = np.linspace(1e15, 2e15, 3)
    edges = np.array([0.0, 5e-4, 1e-3])
    block = np.zeros((2, 1, 2, 2, 3), dtype=complex)
    block[:, 0, 0, 0] = block[:, 0, 1, 1] = -1.0

    def build(generator, connection=None, dz_max=1e-4, representation_error=0.0):
        """Call the constructor with one argument replaced."""
        return frozen_residual_flow(
            omega,
            edges,
            generator,
            connection,
            dz_max=dz_max,
            condition_limit=10.0,
            representation_error=representation_error,
            gauge_id="lab",
            certificate_id="frozen-residual",
        )

    with pytest.raises(TranslationError, match="n_intervals, n_deg"):
        build(block[:1])
    with pytest.raises(TranslationError, match="connection must have"):
        build(block, block[:, :, :, :, :2])
    with pytest.raises(TranslationError, match="positive dz_max"):
        build(block, dz_max=0.0)
    with pytest.raises(TranslationError, match="non-negative"):
        build(block, representation_error=-1.0)
    growing = block.copy()
    growing[:, 0, 0, 0] = 1.0e6
    with pytest.raises(TranslationError, match="inverse conditioning reaches"):
        build(growing)
    flow = build(block)
    with pytest.raises(TranslationError, match="no dense generator to freeze"):
        frozen_generator(
            replace(
                flow,
                generator=None,
                fixed_terms=np.zeros((1, 2, 2, 3), dtype=complex),
                term_weights=np.zeros((2, 1, 1), dtype=complex),
            ),
            0,
        )
    with pytest.raises(TranslationError, match="non-negative h K"):
        residual_truncation(-1.0)
    with pytest.raises(ModeError, match="at least one position"):
        worst_wall_limits(block_modes(2), GRID.omega, positions=[])


def block_modes(count):
    """Return a contiguous `HE_1m` set of `count` channels on the taper."""
    return [MarcatiliMode(taper(), "He", 1.0, m=index + 1) for index in range(count)]


def test_a_section_expansion_that_cannot_resolve_is_refused():
    """A tail above `SECTION_TOL` at the highest degree is refused, never accepted.

    The stored diagonal `k - beta_11` is formed from the two refractivities, so the
    samples' own round-off floors the tail at about 4e-14 relative, and a tail
    above the target is structure the interpolant is not describing: noise of 1e-10
    relative is refused as surely as noise of 1e-3. A function the interpolant resolves
    is accepted at the target.
    """
    edges = np.array([0.0, 1.0])

    def noisy(scale):
        rng = np.random.default_rng(0)

        def sample(positions):
            values = np.cos(np.asarray(positions))[:, None]
            return values + scale * rng.standard_normal(values.shape)

        return sample

    for scale in (1e-10, 1e-3):
        with pytest.raises(TranslationError, match="does not resolve"):
            section_expansion(noisy(scale), edges)
    coefficients, tail = section_expansion(noisy(0.0), edges)
    assert coefficients.shape[0] == 1
    assert tail <= SECTION_TOL


def test_the_wall_diagonal_is_the_refractivity_difference():
    """`k - beta_11` is formed from `n_core - 1` and `n_eff - 1`, to rounding.

    Bound, stated first: against the same difference evaluated in 40-digit arithmetic
    from the same double-precision susceptibility and refractivity, the stored diagonal
    agrees to a few ulp of itself, 1e-13 relative; formed as the difference of the two
    wavenumbers it carried their absolute rounding, about 5e-9 1/m, up to 5e-10 of
    itself on this bore.
    """
    mode = MarcatiliMode(RADIUS, "He", 1.0)
    band = GRID.omega[GRID.active_mask]
    generator = problem_module._wall_shifted_generator(
        np.zeros((1, 1, GRID.n_omega), dtype=complex), mode, GRID
    )
    # The diagonal is stored as conj(i s), so s = -i conj(stored).
    stored = -1j * np.conj(generator[0, 0, GRID.active_mask])
    core = mode.core_index_minus_one(band)
    effective = mode.neff_minus_one(band)
    chi = mode.sampled_terms(band).core_susceptibility * mode._scalars(0.0)[1]
    with mpmath.workdps(40):
        exact = np.array(
            [
                complex((mpmath.mpc(c) - mpmath.mpc(n)) * mpmath.mpf(w) / mpmath.mpf(C))
                for c, n, w in zip(core, effective, band, strict=True)
            ]
        )
        root = np.array([complex(mpmath.sqrt(1 + mpmath.mpc(one)) - 1) for one in chi])
    assert np.max(abs(stored - exact) / abs(exact)) < 1e-13
    # n_core - 1 itself: a few ulp of chi / 2, where sqrt(1 + chi) - 1 in double
    # precision would carry the rounding of unity, 2e-16 / 1e-4 relative.
    assert np.max(abs(core - root) / abs(root)) < 1e-14


def test_a_varying_route_whose_total_generator_grows_is_refused_by_name():
    """The varying counterpart of the third door on the sign of the operator.

    Unreachable with the conjugation as written, which is the point, so it is
    exercised on a coefficient block whose frozen diagonal has been flipped. The
    allowance the guard carries is each channel's own normalisation term,
    which is positive wherever that channel's index falls, with nothing wrong: on this
    taper it reaches 13.6 1/m, the upper channels' across the transition at the band
    edge. A flipped conjugation exceeds it by 192 to 373 1/m on every section, so the
    allowance does not blunt the check. The allowance itself is each channel's exact
    mean rate over each section, and the rows differ by channel.
    """
    simulation = varying(wall=CoherentWall())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    spoiled = np.array(flow.generator)
    index = np.arange(flow.n_modes)
    spoiled[:, :, index, index] = -np.conj(spoiled[:, :, index, index])
    with pytest.raises(TranslationError, match="grows on section"):
        _check_wall_total_varying(
            spoiled, problem.linear, flow.z_edges, tuple(simulation.modes), GRID
        )
    band = GRID.omega[GRID.active_mask]
    lo, hi = float(flow.z_edges[1]), float(flow.z_edges[2])
    allowance = _normalisation_rate(tuple(simulation.modes), GRID, lo, hi)
    for channel, mode in enumerate(simulation.modes):
        ratio = np.real(mode.neff(band, hi)) / np.real(mode.neff(band, lo))
        assert_allclose(
            allowance[channel, GRID.active_mask],
            abs(0.5 * np.log(ratio)) / (hi - lo),
            rtol=1e-12,
        )
    assert np.all(allowance[:, ~GRID.active_mask] == 0.0)
    assert float(np.max(allowance[2])) > 2.0 * float(np.max(allowance[0]))


# ------------------------------------------------------------------ the vector block


VECTOR_WALL = CoherentWall(impedance="te_tm")
"""The declaration under test here: the coupled azimuthal block."""


def vector_modes(
    counts=(2, 2), *, radius: float | ZProfile = RADIUS, orders=(1, -1), **kwargs
):
    """Return one azimuthal order of the qualification capillary."""
    return [
        MarcatiliMode(radius, "He", 1.0, n=order, m=index + 1, **kwargs)
        for order, count in zip(orders, counts, strict=True)
        for index in range(count)
    ]


def test_the_vector_route_declares_one_constant_section_without_a_wall_pair():
    """The leaf is the scalar wall's constant-Schur one, with no pair and its own gauge.

    The rank-one pair the leaf can carry beside the dense generator reconstructs a
    rank-**one** off-diagonal; the vector wall term is `V W V^T` with one column per
    family, so half of it would be worse than none and the constructor is given none.
    Everything else is the scalar wall's: one section, an exactly zero residual, and the
    three certificate rows the backend checks.
    """
    problem = capillary(wall=VECTOR_WALL, modes=vector_modes()).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.section_methods == ("constant_schur",)
    assert flow.representation == "dense"
    assert flow.wall_vectors is None and flow.wall_weights is None
    assert flow.connection is None
    assert float(np.max(flow.residual_norm_bound)) == 0.0
    assert flow.gauge_id == "capillary azimuthal order nu=1"
    assert flow.certificate_id == "constant-schur"
    assert flow.n_modes == 4
    assert float(np.max(flow.generator_error)) < 1e-12


def test_the_vector_route_puts_the_block_the_operator_built_across_the_seam():
    """The stored generator is `conj(G + i(k - beta_11) I)`, as the scalar route's.

    Bound, stated first: **bitwise**. The split is unchanged by the boundary
    response: the cumulative `Phi` carries `N` copies of the lowest channel and the
    flow carries everything the block adds to it, so what crosses the seam is the
    conjugate of the whole optics-convention equation with the reference channel
    subtracted. A `-G` instead of `conj(G)` would turn a passive channel into a
    growing one, which `_check_wall_total` refuses at this door. The shift
    `k - beta_11` is formed from the refractivities, and agrees with the
    difference of the block's own wavenumber and `beta_11` to that difference's
    rounding, `|k|` times a few ulp: 1e-8 1/m.
    """
    modes = vector_modes()
    problem = capillary(wall=VECTOR_WALL, modes=modes).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    block = vector_wall_operator(modes, GRID.omega, active=GRID.active_mask)
    band = GRID.omega[GRID.active_mask]
    refractivity = modes[0].core_index_minus_one(band) - modes[0].neff_minus_one(band)
    expected = np.conj(block.generator)
    shift = np.zeros(GRID.n_omega, dtype=np.complex128)
    shift[GRID.active_mask] = refractivity * band / C
    expected[np.arange(4), np.arange(4)] += np.conj(1j * shift)
    assert flow.generator is not None
    assert np.array_equal(flow.generator[0, 0], expected)
    wavenumbers = block.wavenumber[GRID.active_mask] - modes[0].neff(band) * band / C
    assert float(np.max(abs(wavenumbers - shift[GRID.active_mask]))) < 1e-8
    assert np.all(flow.generator[0, 0][:, :, ~GRID.active_mask] == 0.0)


def test_a_varying_guide_beside_a_vector_wall_takes_the_frozen_residual_route():
    """A taper or a pressure gradient with `impedance="te_tm"` is frozen per section.

    The vector block takes the frozen-residual route with its own generator, its own
    gauge and its own connection. Bounds, stated first. The stored connection at a
    position is `W^(-1/2) C W^(1/2)` of the vector `C` (the true-power amplitude), whose
    cross-family blocks are **exactly** zero before the weights and so after them, and
    whose within-family blocks are antisymmetric in `b = W^(1/2) a` to rounding, 1e-12
    of their size. A pressure gradient leaves every Bessel profile where it was and
    carries no connection at all.
    """
    modes = vector_modes(radius=taper())
    problem = varying(wall=VECTOR_WALL, modes=modes).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert set(flow.section_methods) == {"frozen_residual"}
    assert flow.gauge_id == "capillary azimuthal order nu=1"
    assert flow.certificate_id == "frozen-residual"
    assert flow.connection is not None and flow.wall_vectors is None
    z = 0.5 * LENGTH
    stored = problem_module._wall_connection_at(tuple(modes), GRID, z, "te_tm")
    assert np.all(stored[:2, 2:] == 0.0) and np.all(stored[2:, :2] == 0.0)
    band = GRID.omega[GRID.active_mask]
    root = np.sqrt(np.stack([np.real(mode.neff(band, z)) for mode in modes]))
    carried = root[:, None, :] * stored[:, :, GRID.active_mask] / root[None, :, :]
    scale = float(np.max(abs(carried)))
    assert scale > 0.0
    assert float(np.max(abs(carried + np.swapaxes(carried, 0, 1)))) < 1e-12 * scale
    gradient = capillary(
        wall=VECTOR_WALL, modes=vector_modes(), pressure=(1.0, 0.1)
    ).translate()
    pressure_flow = gradient.linear.coupled_flow
    assert pressure_flow is not None
    assert set(pressure_flow.section_methods) == {"frozen_residual"}
    assert pressure_flow.connection is None


def test_one_hybrid_family_on_a_taper_is_the_scalar_varying_route():
    """`HE_1m` alone under the TE/TM wall is the scalar wall's taper, to rounding.

    Bounds, stated first. The connection is the same matrix **bit for bit** (the
    family's Bessel order is zero), and the block agrees with the scalar one to the
    1e-15 of two contractions (``tests/modes/test_wall_operator.py``:
    `test_one_hybrid_family_alone_is_the_scalar_block`), so ten steps of the taper agree
    to that rounding carried through the flow: gate 1e-13 of the state's own size. The
    record says which wall was declared.
    """
    modes = [MarcatiliMode(taper(), "He", 1.0, m=index + 1) for index in range(3)]
    vector = varying(wall=VECTOR_WALL, modes=modes).translate()
    scalar = varying(wall=CoherentWall(), modes=modes).translate()
    vector_flow, scalar_flow = vector.linear.coupled_flow, scalar.linear.coupled_flow
    assert vector_flow is not None and scalar_flow is not None
    assert vector_flow.connection is not None and scalar_flow.connection is not None
    assert np.array_equal(vector_flow.connection, scalar_flow.connection)
    assert vector_flow.gauge_id == "capillary azimuthal order nu=1"
    one, two = propagated(vector, GRID), propagated(scalar, GRID)
    assert float(np.max(abs(one - two))) < 1e-13 * float(np.max(abs(two)))


def test_a_vector_wall_refuses_every_declaration_it_would_otherwise_override():
    """Every door on the scalar route is reached by a vector block too.

    Not assumed: the vector route is a branch of its own inside `_wall_assembly`, so
    each refusal is taken again through it. `loss=`, `model=` and `lattice=` are named
    at the keyword, the mode set's own settings at the block, and `n_pol`, a noise
    draw and a route that cannot carry a wall in the translation layer.
    """
    modes = vector_modes()
    for overrides, error, message in (
        ({"loss": False}, DriverError, "is the loss model"),
        ({"model": "reduced"}, DriverError, "one capillary model"),
        ({"lattice": TubeLattice(180e-9, 9e-6, 6)}, DriverError, "tube lattice"),
        (
            {"noise": "modified", "seed": 1},
            TranslationError,
            "refuses the 'modified' noise model",
        ),
    ):
        with pytest.raises(error, match=message):
            run_capillary(wall=VECTOR_WALL, modes=modes, **overrides)
    mixed = [
        MarcatiliMode(RADIUS, "He", 1.0, n=1, m=1),
        MarcatiliMode(RADIUS, "He", 1.0, n=-1, m=1, phi=0.4),
    ]
    with pytest.raises(ModeError, match="one polarisation angle"):
        capillary(wall=VECTOR_WALL, modes=mixed).translate()
    with pytest.raises(ModeError, match="at most two radial families"):
        capillary(
            wall=VECTOR_WALL,
            modes=[
                MarcatiliMode(RADIUS, "He", 1.0, n=order, m=1) for order in (1, -1, 2)
            ],
        ).translate()
    # `n_pol = 2` is refused for either wall impedance by the one door
    # `test_a_vector_state_or_a_noise_draw_refuses_a_wall_beside_it` exercises: the
    # capillary route has no public way to carry two components at all.


@pytest.mark.medium
@pytest.mark.parametrize("name", ["numba_cpu", "native"])
def test_a_varying_vector_wall_runs_on_the_compiled_backends_to_rounding(name):
    """Every backend executes the TE/TM wall on a taper, to the flow parity bound.

    `numba_cpu` and ``native`` execute every constant-Schur and frozen-residual flow the
    reference admits, the TE/TM block on a varying guide included. The leaf is a
    dense frozen-residual flow like the scalar taper's, so the compiled backends'
    parity bounds cover it: at most 1.9e-12 (`numba_cpu`) and 5.5e-12 (Kokkos) of the
    launch peak after ten steps. Gated here at 1e-11 of the final plane's peak.
    """
    if name not in available():
        pytest.skip(f"{name} is not installed")
    arguments = {
        "wall": VECTOR_WALL,
        "modes": vector_modes(radius=taper()),
        "radius_override": taper(),
    }
    reference = np.asarray(run_capillary(**arguments).planes()[-1].field)
    compiled = np.asarray(run_capillary(backend=name, **arguments).planes()[-1].field)
    peak = float(np.max(abs(reference)))
    assert peak > 0.0
    assert float(np.max(abs(compiled - reference))) < 1e-11 * peak


@pytest.mark.medium
def test_prop_capillary_runs_a_vector_wall_end_to_end_and_records_its_flow():
    """Ten steps through the entry point: the record, the loss and the conversion.

    Bounds, stated first and each derived before the run.

    1. The **energy lost** must be the block's own diagonal over the length, to 1 %:
       the wall's absorption is `2 Re(-G_mm)` per channel and the launch is in
       `HE_11`, so the prediction is the energy-weighted rate of that channel.
    2. The **conversion** into the partner family must be non-zero and small: the
       coupling is `delta/zbar = 0.357` of the diagonal scale, and the beat between
       the two families is `(u_B^2 - u_A^2)/(2 k a^2)`, about 8e4 rad/m here, so the
       admixture is bounded by `(2 |G_AB| / Delta beta)^2` -- a perturbative bound,
       and the measurement must sit under it.
    3. The **record** declares the vector basis, the vector model and the block's own
       gauge, and the ordinary modal capability is unchanged.
    """
    modes = vector_modes()
    handler = run_capillary(wall=VECTOR_WALL, modes=modes)
    record = handler.metadata
    assert record is not None and record.modal is not None
    assert record.modal["basis"] == "capillary azimuthal order"
    flow_record = cast("Mapping[str, object]", record.modal["flow"])
    assert flow_record["model"] == "te_tm impedance coherent wall"
    assert flow_record["gauge_id"] == "capillary azimuthal order nu=1"
    assert record.modal["n_channels"] == 4
    assert capability_of(record) == "modal"
    channels = cast("Sequence[Mapping[str, object]]", record.modal["channels"])
    assert [channel["display_label"] for channel in channels] == [
        "HE1,1:even",
        "HE1,2:even",
        "EH1,1:even",
        "EH1,2:even",
    ]
    planes = handler.planes()
    launched = np.sum(abs(kept(planes[0].field)[:, 0]) ** 2, axis=-1)
    arrived = np.sum(abs(kept(planes[-1].field)[:, 0]) ** 2, axis=-1)
    block = vector_wall_operator(modes, GRID.omega, active=GRID.active_mask)
    active = GRID.active_mask & (GRID.omega > 0)
    weights = abs(kept(planes[0].field)[0, 0, active]) ** 2
    rate = -2.0 * np.real(block.generator[0, 0, active])
    predicted = float(
        np.sum(weights * (1.0 - np.exp(-rate * LENGTH))) / np.sum(weights)
    )
    lost = float(1.0 - arrived.sum() / launched.sum())
    assert abs(lost / predicted - 1.0) < 0.01, (lost, predicted)
    carrier = float(np.argmax(weights))
    bin_index = int(np.flatnonzero(active)[int(carrier)])
    coupling = abs(complex(block.generator[0, 2, bin_index]))
    beat = abs(
        float(
            np.imag(block.generator[2, 2, bin_index] - block.generator[0, 0, bin_index])
        )
    )
    bound = (2.0 * coupling / beat) ** 2
    admixture = float(arrived[2] / arrived.sum())
    assert 0.0 < admixture < bound, (admixture, bound)


@pytest.mark.medium
def test_ten_steps_of_mixed_family_propagation_match_an_independent_solve():
    """The propagated four-channel state is `exp(L (Phi I + G))` applied to the launch.

    Bound, stated first: **1e-9 relative** on every channel ratio and on every phase,
    and 3e-7 on the absolute modulus, which is the weak collar's own attenuation of a
    continuous wave over this length (this module's :data:`WEAK_COLLAR`, below 2e-7)
    and is identical on every channel, so it divides out of a ratio.
    The launch is one weakly excited bin of `HE_11`, so the Kerr phase over the length
    is below 1e-12 rad and the ten steps integrate the linear flow alone; the flow is
    constant, so the exact solution of the coupled system is one matrix exponential of
    the **internal** generator, taken here with `scipy.linalg.expm` and multiplied by
    the scalar carrier the cumulative `Phi` applies. What this adds to the operator
    tests is that the step loop propagates the *mixed-family* block it was given --
    including the conversion, which no diagonal flow can produce -- in the right
    convention, with the carrier and the block multiplied rather than added.

    It is also an independent reference at the propagation level: `expm` is not the
    backend's Schur action, and the conversion it predicts into `EH_11` is 1.68e-4 of
    the surviving amplitude -- 2.8e-8 in power, which is the end-to-end run's own number
    -- five orders above the ratio bound.
    """
    simulation = capillary(wall=VECTOR_WALL, modes=vector_modes())
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    index, launched = linear_launch(problem, simulation.grid)
    final = propagated(launched, simulation.grid)
    carrier = complex(delta_phi(problem.linear, 0.0, LENGTH)[0, 0, 0, index])
    block = np.asarray(flow.generator[0, 0, :, :, index], dtype=complex)
    start = np.zeros(4, dtype=complex)
    start[0] = 1e-9
    expected = np.exp(carrier) * (expm(LENGTH * block) @ start)
    measured = final[0, :, 0, index]
    # The channel **ratios** are what the collar cannot touch: it is a temporal mask,
    # identical on every channel, and a single excited bin of a field grid is the
    # continuous wave it eats -- 2.1e-7 of the modulus over this length, which is the
    # bound the absolute comparison below takes and the reason it is not 1e-9.
    assert_allclose(measured / measured[0], expected / expected[0], rtol=1e-9)
    assert_allclose(np.angle(measured), np.angle(expected), rtol=1e-9)
    assert_allclose(abs(measured), abs(expected), rtol=3e-7)
    # The conversion is what makes this more than the scalar statement, and it is the
    # same 2.8e-8 in power that the end-to-end run measures.
    assert abs(expected[2]) / abs(expected[0]) == pytest.approx(1.677e-4, rel=1e-3)


@pytest.mark.medium
def test_the_vector_wall_converts_between_families_and_no_scalar_block_can():
    """The physical content of the vector wall, measured against its two controls.

    Bounds, stated first. Against the **uncoupled** route, the partner family must
    hold more power than the capillary's own Kerr overlap leaves there, which is
    thirty orders down; against the **scalar** block the comparison cannot be made at
    all, because a set spanning two azimuthal orders is refused by it -- that refusal
    is the statement that two copies of an impedance-averaged block are not this
    operator. And the conversion must scale as the square of the anisotropy: halving
    `delta` by moving the cladding towards the core index must quarter it, to 10 %,
    which is what makes the transfer the wall's anisotropy rather than a leak.
    """
    modes = vector_modes()
    walled = kept(run_capillary(wall=VECTOR_WALL, modes=modes).planes()[-1].field)[:, 0]
    bare = kept(run_capillary(modes=modes).planes()[-1].field)[:, 0]
    walled_share = np.sum(abs(walled) ** 2, axis=-1)
    bare_share = np.sum(abs(bare) ** 2, axis=-1)
    assert walled_share[2] / walled_share[0] > 1e-9
    assert bare_share[2] / bare_share[0] < 1e-30
    with pytest.raises(ModeError, match="contiguous Marcatili set"):
        capillary(wall=CoherentWall(), modes=modes).translate()
    ratios = []
    for cladding in (1.45 + 0j, 1.2099 + 0j):
        pair = [
            MarcatiliMode(RADIUS, "He", 1.0, n=order, m=1, cladding=cladding)
            for order in (1, -1)
        ]
        block = vector_wall_operator(pair, GRID.omega, active=GRID.active_mask)
        impedance = block.impedance[:, GRID.active_mask]
        anisotropy = float(np.max(abs(0.5 * (impedance[0] - impedance[1]))))
        field = kept(run_capillary(wall=VECTOR_WALL, modes=pair).planes()[-1].field)[
            :, 0
        ]
        share = np.sum(abs(field) ** 2, axis=-1)
        ratios.append((anisotropy, float(share[1] / share[0])))
    (first, strong), (second, weak) = ratios
    assert abs((strong / weak) / (first / second) ** 2 - 1.0) < 0.02, ratios


# This case's final plane was a byte digest; it is now a stored final plane of the
# fixed-order march, compared within its rounding budget (tests/state_records.py). The
# former digests and their re-recordings are in the git history.


def _vector_taper_run(**overrides) -> MemoryOutput:
    """Run the vector wall on the qualification taper through `prop_capillary`."""
    arguments = {
        "wall": VECTOR_WALL,
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": vector_modes(radius=taper()),
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    if _steps_cf4(arguments["wall"]) and "integrator" not in arguments:
        # CF4 sections are stepped by the Runge-Kutta march alone.
        arguments["integrator"] = RungeKutta()
    return cast(
        "MemoryOutput",
        prop_capillary(taper(), LENGTH, "He", 1.0, **{"noise": None, **arguments}),
    )


@pytest.mark.medium
def test_prop_capillary_runs_a_vector_wall_on_a_taper_end_to_end_and_is_pinned(
    ndarrays_regression,
):
    """Ten steps of a taper under the TE/TM wall through the entry point.

    Bounds, stated first and each derived before the run.

    1. The **record** names four `"frozen_residual"` sections, a connection, the
       vector model and gauge, and the **narrow** end's ratios, which are the worst.
    2. The **energy lost** exceeds the narrow constant bore's under the same wall
       (8.62e-4): the connection converts the launch into `HE_12`, whose loss goes as
       `u_2^2`, as on the scalar taper. Measured 1.142e-3.
    3. The **`HE_1m` family** follows the scalar two-channel taper: the
       partner family perturbs it only through the anisotropy, at second order, i.e.
       by about the power it moves across, so `HE_12`'s share of the `HE_1m` power
       agrees with the scalar run's to within ten times the `EH_1m` share (measured
       3.7e-7 of the total).
    4. The plane matches its record within the fixed-order march's rounding budget
       (``tests/state_records.py``).
    """
    handler = _vector_taper_run(integrator=FIXED_ORDER)
    written = handler.metadata
    assert written is not None and written.modal is not None
    assert written.modal["basis"] == "capillary azimuthal order"
    record = cast("Mapping[str, object]", written.modal["flow"])
    assert record["model"] == "te_tm impedance coherent wall"
    assert record["section_methods"] == ["frozen_residual"] * 4
    assert record["connection"] is True
    assert record["gauge_id"] == "capillary azimuthal order nu=1"
    assert record["radius"] == NARROW
    narrow = energy_loss(
        run_capillary(
            wall=VECTOR_WALL,
            modes=vector_modes(radius=NARROW),
            radius_override=NARROW,
        )
    )
    lost = energy_loss(handler)
    assert lost > narrow, (lost, narrow)
    fractions = _channel_fractions(handler)
    scalar = _channel_fractions(
        cast(
            "MemoryOutput",
            prop_capillary(
                taper(),
                LENGTH,
                "He",
                1.0,
                wall=CoherentWall(),
                wavelength=800e-9,
                grid=GRID,
                absorber=WEAK_COLLAR,
                fwhm=30e-15,
                energy=1e-9,
                modes=2,
                n_save=2,
                backend="numpy_ref",
                dz_max=LENGTH / STEPS,
                dz_init=LENGTH / STEPS,
                schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
                integrator=FIXED_ORDER,
                plasma=False,
                raman=False,
                noise=None,
            ),
        )
    )
    partner = float(np.sum(fractions[2:]))
    assert 0.0 < partner < 1e-5
    within = fractions[1] / float(np.sum(fractions[:2]))
    assert abs(within - scalar[1]) < 10.0 * partner, (within, scalar[1], partner)
    field = handler.planes()[-1].field
    simulation = varying(
        wall=VECTOR_WALL, modes=vector_modes(radius=taper()), integrator=FIXED_ORDER
    )
    check_state(ndarrays_regression, field, record_bound(simulation, field))


def _stage_interpolant(function, lo, hi, degree=20):
    """Return a Chebyshev interpolant of a matrix-valued `function` over `[lo, hi]`.

    The reference solve's generator is analytic in the radius, whose nearest
    singularity is near `a = 0`, nine half-widths from the transition's centre, so
    twenty nodes resolve it to rounding -- which is asserted, at three positions
    between the nodes, to 1e-12 of its size, before it is used: the reference then
    costs twenty wall blocks per stage rather than one per solver stage.
    """
    nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)
    samples = np.array([function(0.5 * (lo + hi) + 0.5 * (hi - lo) * x) for x in nodes])
    shape = samples.shape[1:]
    coefficients = np.polynomial.chebyshev.chebfit(
        nodes, samples.reshape(degree, -1), degree - 1
    )

    def evaluate(z):
        x = (2.0 * z - lo - hi) / (hi - lo)
        return np.polynomial.chebyshev.chebval(x, coefficients).reshape(shape)

    scale = float(np.max(abs(samples)))
    for fraction in (0.13, 0.5, 0.91):
        z = lo + fraction * (hi - lo)
        assert float(np.max(abs(evaluate(z) - function(z)))) < 1e-12 * scale
    return evaluate


def _implemented_link(modes, profile, z, relative=1e-5):
    """Return `<psi_m|d_z psi_n>` by quadrature of the implemented fields at `z`.

    The independent measurement (the modes suite's `implemented_connection`, restated
    here because a test module does not import another): each field normalised by its
    own closed-form power, differentiated along the profile by a fourth-order central
    difference that moves the radius by `relative` of itself, and paired by
    Gauss-Legendre in `r` and the periodic trapezoid in `theta`. Nothing of the
    connection's formula is in it.
    """
    radius = float(profile(np.array([z]))[0])
    step = relative * radius / abs(float(profile.derivative(np.array([z]))[0]))
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = 0.5 * radius * (nodes + 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    points = np.stack([rr.ravel(), tt.ravel()], axis=-1)
    measure = (0.5 * radius * weights[:, None] * rr * (2.0 * np.pi / 64)).ravel()

    def normed(mode, position):
        scale = math.sqrt(2.0 * mode.normalisation(position) / (EPS0 * C))
        return mode.field(points, position) / scale

    slopes = [
        (
            8.0 * (normed(mode, z + step / 2) - normed(mode, z - step / 2))
            - (normed(mode, z + step) - normed(mode, z - step))
        )
        / (6.0 * step)
        for mode in modes
    ]
    here = [normed(mode, z) for mode in modes]
    return np.array(
        [
            [float(np.sum(measure * np.sum(one * two, axis=-1))) for two in slopes]
            for one in here
        ]
    )


@pytest.mark.medium
@pytest.mark.xdist_group("coherent-vector-power-reference")
def test_a_vector_taper_matches_a_true_power_solve_with_a_measured_connection():
    """The route against a reference that does not contain the connection's formula.

    The reference is the scalar taper's DOP853 solve in the true-power amplitude,
    `b' = (W^(1/2) (i k I + G) W^(-1/2) - C) b`, with `G` and `k` from
    `vector_wall_operator` at every position the solver asks for -- and `C` **not**
    from `wall_connection` but **measured**: the quadrature of the implemented fields'
    z derivative at the transition's midpoint (:func:`_implemented_link`), carried to
    every other position by `a'/a`, because every field is a function of `r/a` alone.
    That scaling is itself measured at a second position (gate 1e-9; measured 1.9e-11).
    The measured matrix has no cross-family entries beyond rounding, which is the
    physics of the block-diagonal connection; here it is the reference's, not the
    route's.

    Bounds, stated first, as for the scalar taper. The quantity is each channel's
    `b_n/b_1`, which every channel-common factor leaves alone. At ten steps the route
    departs from the reference by the schedule's error alone, gated at 1e-7 per ratio
    (measured 1.2e-8 on `EH_11`, the smallest ratio, 2.5e-4 of `HE_11`), falling by more
    than 16 per halving (measured about 30), to a floor at forty steps set by the
    measured connection's own 2e-11 accuracy: gate 1e-10 at 800 nm. The reference's
    generator is a Chebyshev interpolant of the wall blocks per stage, checked against
    direct blocks to 1e-12 before use (:func:`_stage_interpolant`). Those fixed
    schedules are the Runge-Kutta march's, whose order the gates measure; the default
    integrator, stepped adaptively at ``rtol=1e-10``, meets the same reference to below
    3e-9 (measured 3.7e-10 at 800 nm and 2.3e-10 at the band edge).
    """
    profile = taper()
    block = vector_modes(radius=profile)
    middle, second = 1e-4, 0.8e-4

    def rate(z):
        return float(profile.derivative(np.array([z]))[0]) / float(
            profile(np.array([z]))[0]
        )

    shape = _implemented_link(block, profile, middle) / rate(middle)
    check = _implemented_link(block, profile, second) / rate(second)
    assert float(np.max(abs(check - shape))) < 1e-9 * float(np.max(abs(shape)))
    count = len(block)

    def reference(index):
        omega = GRID.omega[index : index + 1]

        def weights(z):
            return np.array([float(np.real(mode.neff(omega, z))[0]) for mode in block])

        def carried(z):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                operator = vector_wall_operator(
                    block, omega, z=z, validity_limit=math.inf
                )
            wavenumber = complex(operator.wavenumber[0])
            generator = operator.generator[:, :, 0] + 1j * (
                wavenumber - wavenumber.real
            ) * np.eye(count)
            root = np.sqrt(weights(z))
            return root[:, None] * generator / root[None, :]

        state = np.zeros(2 * count)
        state[0] = 1e-9 * math.sqrt(weights(0.0)[0])
        for lo, hi in itertools.pairwise(SMOOTH_STAGES):
            generator = _stage_interpolant(carried, lo, hi)

            def derivative(z, state, generator=generator):
                amplitude = state[:count] + 1j * state[count:]
                change = (generator(z) - shape * rate(z)) @ amplitude
                return np.concatenate([np.real(change), np.imag(change)])

            state = solve_ivp(
                derivative, (lo, hi), state, method="DOP853", rtol=1e-13, atol=1e-30
            ).y[:, -1]
        return state[:count] + 1j * state[count:]

    def routed(index, steps=None):
        # On a schedule, the Runge-Kutta march whose order the gates measure; with no
        # step count, the default integrator, adaptively.
        stepping = (
            {"rtol": 1e-10, "atol": 0.0, "schedule": None, "dz_init": LENGTH * 1e-4}
            if steps is None
            else {
                "integrator": RungeKutta(),
                "schedule": FixedSchedule(np.full(steps, LENGTH / steps)),
                "dz_max": LENGTH / steps,
                "dz_init": LENGTH / steps,
            }
        )
        problem = varying(wall=VECTOR_WALL, modes=block, **stepping).translate()
        state = np.zeros_like(problem.initial.field0)
        state[0, 0, 0, index] = 1e-9
        launched = replace(problem, initial=replace(problem.initial, field0=state))
        internal = propagated(launched, GRID)[0, :, 0, index]
        omega = GRID.omega[index : index + 1]
        weight = np.array(
            [float(np.real(mode.neff(omega, LENGTH))[0]) for mode in block]
        )
        return np.conj(internal) * np.sqrt(weight)

    # The error norm's band edge: the collar carries the frozen flow alone.
    edge = int(np.flatnonzero(GRID.inband_mask)[0])
    centre = int(np.argmin(abs(GRID.omega - GRID.omega0)))
    for index in (centre, edge):
        expected = reference(index)
        ratios = expected[1:] / expected[0]
        departures = []
        for steps in (
            (STEPS, 2 * STEPS) if index == edge else (STEPS, 2 * STEPS, 4 * STEPS)
        ):
            answer = routed(index, steps)
            departures.append(float(np.max(abs(answer[1:] / answer[0] / ratios - 1))))
        assert departures[0] < 1e-7, (index, departures)
        assert departures[0] / departures[1] > 16.0, (index, departures)
        if index == centre:
            assert departures[2] < 1e-10, departures
        default = routed(index)
        assert float(np.max(abs(default[1:] / default[0] / ratios - 1))) < 3e-9


@pytest.mark.medium
def test_on_a_taper_the_families_exchange_power_only_through_the_wall_anisotropy():
    """The mixed-family conversion's scaling: the connection adds none across families.

    Bounds, stated first. The connection is block diagonal, so on a taper
    as on a straight bore the only path from the `HE_1m` family into the `EH_1m` one is
    the wall's anisotropy `delta`, and the power it moves goes as `delta^2`: moving
    the cladding towards the core index to shrink `delta` by 0.649 must shrink the
    `EH_1m`-to-`HE_1m` power ratio by 0.649^2 -- to 2 %, the gate on the straight
    bore; measured 0.15 %. A cross-family connection of the size of the
    within-family one would carry `(a'/a)`-sized conversion that does not depend on
    `delta` at all, and would break the scaling at the first figure. And the order
    `nu = 0` has no path at all: `TM_0m` and `TE_0m` see the wall's two impedances on
    the diagonal and are connected by nothing, so a `TM_01` launch leaves the `TE`
    family at the rounding of zero (gate 1e-30 of the power; measured 6e-52) while
    the connection moves 11 % of it into `TM_02`.
    """
    shares = []
    for cladding in (1.45 + 0j, 1.2099 + 0j):
        modes = vector_modes(radius=taper(), cladding=cladding)
        block = vector_wall_operator(
            modes, GRID.omega, z=LENGTH, active=GRID.active_mask
        )
        impedance = block.impedance[:, GRID.active_mask]
        anisotropy = float(np.max(abs(0.5 * (impedance[0] - impedance[1]))))
        field = kept(
            run_capillary(wall=VECTOR_WALL, modes=modes, radius_override=taper())
            .planes()[-1]
            .field
        )[:, 0]
        power = np.sum(abs(field) ** 2, axis=-1)
        shares.append(
            (anisotropy, float((power[2] + power[3]) / (power[0] + power[1])))
        )
    (first, strong), (second, weak) = shares
    assert abs((strong / weak) / (first / second) ** 2 - 1.0) < 0.02, shares
    transverse = [
        MarcatiliMode(taper(), "He", 1.0, kind=kind, n=0, m=index + 1)
        for kind in ("TM", "TE")
        for index in range(2)
    ]
    field = kept(
        run_capillary(wall=VECTOR_WALL, modes=transverse, radius_override=taper())
        .planes()[-1]
        .field
    )[:, 0]
    power = np.sum(abs(field) ** 2, axis=-1) / float(np.sum(abs(field) ** 2))
    assert float(power[2] + power[3]) < 1e-30, power
    assert float(power[1]) > 0.05, power


def _vector_refined(
    steps, method: Literal["frozen_residual", "cf4"] = "frozen_residual"
):
    """Return the final plane of the vector taper on a schedule of `steps`.

    Stepped by the Runge-Kutta march: the refinement measures DOPRI5's fifth order on
    the frozen split, and CF4 sections are stepped by that march alone.
    """
    return propagated(
        varying(
            wall=CoherentWall(impedance="te_tm", method=method),
            integrator=RungeKutta(),
            modes=vector_modes(radius=taper()),
            schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
            dz_max=LENGTH / steps,
            dz_init=LENGTH / steps,
        ).translate(),
        GRID,
    )


@pytest.mark.medium
def test_the_vector_taper_converges_at_each_methods_own_order():
    """Independent refinement: fifth order frozen, fourth order CF4, one reference.

    Bounds, stated first. The embedded DOPRI5 pair integrates the frozen split's
    explicit residual at fifth order, so halving the step divides the departure from a
    160-step frozen run by 32 -- gate an order above 4.0 on the first halving, and a
    ten-step departure below 1e-8 of the state (measured 6.2e-11, then 4.82). CF4
    integrates the same stored expansion whole and is at most fourth order: gate an
    order between 3.8 and 4.2 on both halvings (measured 3.996 and 3.999; 3.993 and
    3.998 on the scalar taper), which is what "CF4 composes with the vector block
    trivially" means -- it reads the expansion and the connection and nothing of the
    block that built them.
    """
    reference = _vector_refined(STEPS * 16)
    scale = float(np.max(abs(reference)))
    frozen = [
        float(np.max(abs(_vector_refined(STEPS * count) - reference))) / scale
        for count in (1, 2)
    ]
    assert frozen[0] < 1e-8, frozen
    assert math.log2(frozen[0] / frozen[1]) > 4.0, frozen
    cf4 = [
        float(np.max(abs(_vector_refined(STEPS * count, "cf4") - reference))) / scale
        for count in (1, 2, 4)
    ]
    orders = [math.log2(cf4[index] / cf4[index + 1]) for index in range(2)]
    assert all(3.8 < order < 4.2 for order in orders), (cf4, orders)


def test_the_vector_flow_record_survives_a_file_and_a_resume_checks_it(tmp_path):
    """The record round-trips, and a scalar run cannot resume a vector one."""
    path = tmp_path / "vector-wall.h5"
    modes = vector_modes(counts=(1, 1))
    capillary(wall=VECTOR_WALL, modes=modes).run(path)
    stored = open_file(path)
    try:
        read_back = stored.metadata
        assert read_back is not None and read_back.modal is not None
        assert capability_of(read_back) == "modal"
        fresh = run_capillary(wall=VECTOR_WALL, modes=modes).metadata
        assert fresh is not None and fresh.modal is not None
        assert read_back.modal["flow"] == fresh.modal["flow"]
        assert read_back.modal["basis"] == "capillary azimuthal order"
    finally:
        stored.close()
    with pytest.raises(OutputError, match="different run"):
        capillary(wall=CoherentWall(), modes=1).run(path, resume=True)


# ------------------------------------------------------------ the CF4 varying flow

CF4_WALL = CoherentWall(method="cf4")
"""The commutator-free fourth-order alternative on the qualification taper."""


def test_a_cf4_wall_declares_cf4_sections_on_the_frozen_split_s_expansion():
    """The same expansion, a different certificate: no residual, the pair bound.

    `method="cf4"` changes what the sections are certified as and nothing about what
    they carry: the generator and the connection are the frozen-residual leaf's to the
    last bit, because both come from one `section_expansion` of one profile. What
    differs is the rows. Bound, stated first: the pair conditioning is
    `exp((2/sqrt(3)) dz_max (m_hi - m_lo))` over the Weyl enclosure of each step's
    reach, at least the frozen block's own `exp(dz_max (mu_max - mu_min))` and here
    1.0190 against 1.0068 to 1.0132; the growth bound is that to the power
    `(2 - sqrt(3))/4`,
    1.00127 at worst -- the qualification taper is deep inside the asymptotic regime
    the growth-bound guard admits.
    """
    flow = varying(wall=CF4_WALL).translate().linear.coupled_flow
    frozen = varying(wall=CoherentWall()).translate().linear.coupled_flow
    assert flow is not None and frozen is not None
    assert flow.section_methods == ("cf4",) * 4
    assert flow.certificate_id == "cf4"
    assert flow.generator is not None and frozen.generator is not None
    assert np.array_equal(flow.generator, frozen.generator)
    assert flow.connection is not None and frozen.connection is not None
    assert np.array_equal(flow.connection, frozen.connection)
    assert np.all(flow.residual_norm_bound == 0.0)
    assert np.all(np.isinf(advisory_step(flow)))
    assert np.all(flow.inverse_condition_bound >= frozen.inverse_condition_bound)
    assert float(np.max(flow.inverse_condition_bound)) < 1.02
    assert 1.0 < float(np.max(cf4_growth_bound(flow))) < 1.0013


def test_the_executor_and_the_estimator_are_two_implementations_of_one_step():
    """`numpy_ref`'s scaled Taylor series and SciPy's exponential agree on every step.

    The translation layer's estimator forms a CF4 step with `scipy.linalg.expm`; the
    backend forms it with its own scaled Taylor series, from the stage bundle's
    evaluation form rather than the leaf's arrays. Bound, stated first: the two
    agree to a few ulp of the action's norm (measured 2.2e-16 at worst over four
    steps, one per section), the backend's inverse is the inverse of its forward to
    rounding (7.8e-16), and the four factor norms it measures stay under the declared
    pair bound -- 1.0132 against 1.0153 on a straight section, which is the
    `1/(1 + 4|a|) = sqrt(3)/2` of the bound's logarithm a constant generator reaches
    exactly (measured 0.8660241), and 1.0073 against 1.0190 on the transition.
    """
    problem = varying(wall=CF4_WALL).translate()
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    assert flow.connection is not None
    held = _coupled_flow_maps(problem)
    assert held is not None
    terms = (flow.generator - flow.connection).reshape(-1, *flow.generator.shape[2:])
    for start, length, section in (
        (0.0, 2e-5, 0),
        (6e-5, 2e-5, 1),
        (1.0e-4, 1e-5, 1),
        (1.5e-4, 2e-5, 2),
    ):
        reference = _cf4_action(flow, terms, start, length, section)[held.active]
        forward, inverse, growth = _backend_cf4_action(held, start, length, section)
        departure = np.linalg.norm(forward - reference, 2, axis=(-2, -1))
        assert np.all(
            departure
            <= 8 * np.finfo(float).eps * np.linalg.norm(reference, 2, axis=(-2, -1))
        )
        identity = np.linalg.norm(forward @ inverse - np.eye(3), 2, axis=(-2, -1))
        assert float(np.max(identity)) < 8 * np.finfo(float).eps
        declared = flow.inverse_condition_bound[section][held.active]
        assert np.all(growth <= declared)
    straight = _backend_cf4_action(held, 1.5e-4, 2e-5, 2)[2]
    bound = flow.inverse_condition_bound[2][held.active]
    ratio = np.log(straight) / np.log(bound)
    # To 1.3e-6 (measured 1.28e-6). This is not the samples' round-off: it was the
    # same 1.28e-6 before and after the wall's diagonal stopped carrying the rounding
    # of a difference of two wavenumbers, and the straight section's
    # residual enclosure is exactly zero. Its source is not established here.
    assert float(np.max(ratio)) == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-5)


def _taper_generator(flow):
    """Return ``z -> L(z)`` on the active bins, frequency first, from the leaf."""
    assert flow.generator is not None and flow.connection is not None
    terms = (flow.generator - flow.connection).reshape(-1, *flow.generator.shape[2:])
    active = np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))

    def generator(z):
        combined = np.tensordot(flow_weights(flow, z), terms, axes=1)
        return np.moveaxis(combined, -1, 0)[active]

    return generator, terms, active


def test_the_step_doubling_defect_tracks_the_true_error_on_the_lossy_taper():
    """The step-doubling estimator against an independent solve of the flow alone.

    The taper's generator is lossy -- the Hermitian part of `L` spans -373 to +21 1/m
    relative to the carrier -- and nonnormal: Henrici's departure
    `||L L^H - L^H L||_F / ||L||_F^2` reaches 4.7e-2 at the transition's middle. The
    reference is `solve_ivp`'s DOP853 on `u' = L(z) u` at rtol 1e-13, which shares
    nothing with the CF4 scheme but the stored expansion. Predeclared: at the
    lowest, the middle and the highest active bin, one CF4 step from the start of the
    transition has a true relative error within 1 % of the step-doubling estimate
    `(16/15) ||V(h) - V(h/2)^2|| / ||V(h/2)^2||`, at 20, 10 and 5 micrometres, and
    both fall at the step's fifth power. Measured ratios 0.99987 to 0.99999, true
    errors 6.19e-8, 1.79e-9 and 5.39e-11 at the lowest bin.
    """
    flow = varying(wall=CF4_WALL).translate().linear.coupled_flow
    assert flow is not None
    generator, terms, active = _taper_generator(flow)
    count = int(np.count_nonzero(active))
    truths = []
    for length in (2e-5, 1e-5, 5e-6):
        coarse = _cf4_action(flow, terms, 6e-5, length, 1)[active]
        fine = (
            _cf4_action(flow, terms, 6e-5 + 0.5 * length, 0.5 * length, 1)[active]
            @ _cf4_action(flow, terms, 6e-5, 0.5 * length, 1)[active]
        )
        for index in (0, count // 2, count - 1):
            solution = solve_ivp(
                lambda z, state, index=index: (
                    generator(z)[index] @ state.reshape(3, 3)
                ).ravel(),
                (6e-5, 6e-5 + length),
                np.eye(3, dtype=complex).ravel(),
                method="DOP853",
                rtol=1e-13,
                atol=1e-16,
            )
            exact = solution.y[:, -1].reshape(3, 3)
            truth = np.linalg.norm(coarse[index] - exact, 2) / np.linalg.norm(exact, 2)
            estimate = (
                16.0
                / 15.0
                * np.linalg.norm(coarse[index] - fine[index], 2)
                / np.linalg.norm(fine[index], 2)
            )
            assert abs(estimate / truth - 1.0) < 1e-2, (length, index)
            if index == 0:
                truths.append(float(truth))
    orders = np.log2(np.array(truths[:-1]) / np.array(truths[1:]))
    assert np.all(abs(orders - 5.0) < 0.3), orders


@pytest.mark.medium
def test_a_cf4_run_converges_at_fourth_order_against_the_frozen_split():
    """Independent refinement: CF4's schedule against the frozen split's finest run.

    The combined method is **at most fourth order**: a fourth-order flow inside a
    fifth-order Runge-Kutta source. The reference is the other method -- the frozen
    split on 160 steps, fifth order and already 3.4e-9 at ten -- so that the refinement
    is not the scheme checked against itself. Predeclared: the observed orders over 10,
    20 and 40 steps within 4 +- 0.3. Measured 3.99 and 4.00, from 4.71e-8 of the state
    at ten steps; the frozen split's own error there is 3.45e-9, so on this taper and
    this step CF4 is the less accurate method by a factor 14.

    Bound, stated first, on the ten-step error: the sum of the ten steps' estimated
    local defects, `10 * max cf4_defect(h) * max ||U||||U^-1||`, is 1.76e-6 -- an
    asymptotic estimate, not a proof -- and the measured error sits under it.
    """
    reference = _refined(STEPS * 16)
    scale = float(np.max(abs(reference)))
    departures = []
    for count in (1, 2, 4):
        steps = STEPS * count
        final = propagated(
            varying(
                wall=CF4_WALL,
                schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
                dz_max=LENGTH / steps,
                dz_init=LENGTH / steps,
            ).translate(),
            GRID,
        )
        departures.append(float(np.max(abs(final - reference))) / scale)
    orders = np.log2(np.array(departures[:-1]) / np.array(departures[1:]))
    assert np.all(abs(orders - 4.0) < 0.3), (departures, orders)
    flow = varying(wall=CF4_WALL).translate().linear.coupled_flow
    assert flow is not None
    estimate = (
        STEPS
        * float(np.max(cf4_defect(flow, LENGTH / STEPS)))
        * float(np.max(flow.inverse_condition_bound))
    )
    assert departures[0] < estimate
    frozen = float(np.max(abs(_refined(STEPS) - reference))) / scale
    assert frozen < departures[0]


def test_the_growth_guard_falls_back_by_section_only_where_declared():
    """A limit between the sections' bounds rejects some; the declaration decides what.

    The transition's two sections carry growth bounds 1.00125 and 1.00127 at
    `dz_max` = 20 um, the
    straight tail 1.00102; a limit of 1.0011 lies between. Without a fallback the
    translation refuses by name; with `fallback="frozen_residual"` the rejected
    sections take the frozen split, the warning names them, the leaf mixes the two
    methods, and those sections carry exactly the frozen rows.
    """
    with pytest.raises(TranslationError, match=r"reject 2 of its 4 section"):
        varying(wall=CoherentWall(method="cf4", growth_limit=1.0011)).translate()
    with pytest.warns(SolpropWarning, match=r"section 0 has internal growth bound"):
        problem = varying(
            wall=CoherentWall(
                method="cf4", growth_limit=1.0011, fallback="frozen_residual"
            )
        ).translate()
    flow = problem.linear.coupled_flow
    frozen = varying(wall=CoherentWall()).translate().linear.coupled_flow
    assert flow is not None and frozen is not None
    assert flow.section_methods == ("frozen_residual", "frozen_residual", "cf4", "cf4")
    for name in ("generator_error", "residual_norm_bound", "inverse_condition_bound"):
        assert np.array_equal(getattr(flow, name)[:2], getattr(frozen, name)[:2])
    assert np.all(flow.residual_norm_bound[2:] == 0.0)


@pytest.mark.medium
def test_a_mixed_flow_runs_each_section_by_its_own_method():
    """Ten steps of the mixed leaf land between the two pure runs' own errors.

    Each step is integrated by the method of the section its anchor lies in -- a
    frozen block with the explicit residual, or CF4 with none -- so the mixed run
    solves the same equation and must agree with the frozen split's 160-step reference
    to within the larger of the two pure runs' own ten-step errors, 4.71e-8 (CF4).
    Measured 3.45e-9, the frozen split's own: the two sections rejected are the
    entrance and the transition, and the straight tail CF4 keeps is a constant
    generator, which it integrates exactly.
    """
    reference = _refined(STEPS * 16)
    scale = float(np.max(abs(reference)))
    with pytest.warns(SolpropWarning, match="fall back"):
        problem = varying(
            wall=CoherentWall(
                method="cf4", growth_limit=1.0011, fallback="frozen_residual"
            )
        ).translate()
    mixed = float(np.max(abs(propagated(problem, GRID) - reference))) / scale
    assert mixed < 4.8e-8


def test_the_reference_measures_the_cf4_factors_against_the_declared_bound():
    """A pair bound declared too small is caught by the factors the backend forms.

    The growth-bound guard at the executor: with the conditioning row of every CF4
    section set to one -- a bound no dissipative generator meets -- the first stage's
    measured four-factor product, 1.0073 on the transition, is refused by name.
    """
    simulation = varying(wall=CF4_WALL)
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    understated = replace(
        problem,
        linear=replace(
            problem.linear,
            coupled_flow=replace(
                flow, inverse_condition_bound=np.ones_like(flow.inverse_condition_bound)
            ),
        ),
    )
    with pytest.raises(BackendError, match="CF4 factor-norm product"):
        propagated(understated, simulation.grid)


def test_a_cf4_step_above_the_estimated_defect_warns_by_name():
    """The step-doubling diagnostic: a declared step whose defect exceeds the tolerance.

    The embedded pair does not see a CF4 flow's error, so `validate` compares the
    step-doubling estimate at the largest declared step with the controller's
    relative tolerance and warns, advisory like the advisory step. At `dz_max` =
    100 um the transition's estimate is 1.02e-4 -- over its whole 80 um, because a
    step ends at the next hard kink -- a hundred times the default tolerance of
    1e-6; at the ten-step schedule's 20 um it is 1.72e-7, under it, and nothing is
    said.
    """
    with (
        warnings.catch_warnings(),
        pytest.warns(SolpropWarning, match="estimated local defect"),
    ):
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        varying(wall=CF4_WALL, dz_max=1e-4, dz_init=1e-4, schedule=None).translate()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        varying(wall=CF4_WALL).translate()
    assert not [one for one in caught if "estimated local defect" in str(one.message)]


def test_cf4_is_refused_wherever_it_would_be_ignored():
    """Every door a CF4 declaration passes, each refusing what it cannot honour.

    The declaration: an unknown method, and a growth limit or a fallback without CF4 --
    or out of range with it. The translation layer: CF4 on a guide that does not vary,
    whose constant-Schur flow is already exact, for either wall impedance. CF4 beside
    the vector block is **admitted**: it integrates the stored expansion and reads
    nothing of the block (`test_the_vector_taper_converges_at_each_methods_own_order`).
    """
    assert CoherentWall(method="cf4", impedance="te_tm").method == "cf4"
    with pytest.raises(ModeError, match="read only with method='cf4'"):
        CoherentWall(growth_limit=3.0)
    with pytest.raises(ModeError, match="read only with method='cf4'"):
        CoherentWall(fallback="frozen_residual")
    for bad in (0.5, math.inf):
        with pytest.raises(ModeError, match="at least one"):
            CoherentWall(method="cf4", growth_limit=bad)
    for field, bad, match in (
        ("method", "midpoint", "'frozen_residual' or 'cf4'"),
        ("fallback", "midpoint", "which integrates it"),
    ):
        # Built without `__init__`: the test session's runtime type check stops a
        # value outside the `Literal` at the annotation, and production has no hook.
        wall = object.__new__(CoherentWall)
        for name, value in (
            ("condition_limit", 1.0e4),
            ("validity_limit", 0.25),
            ("sections", 1),
            ("impedance", "mean"),
            ("method", "cf4"),
            ("growth_limit", None),
            ("fallback", None),
        ):
            object.__setattr__(wall, name, value)
        object.__setattr__(wall, field, bad)
        with pytest.raises(ModeError, match=match):
            wall.__post_init__()
    with pytest.raises(TranslationError, match="its one constant section"):
        capillary(wall=CF4_WALL).translate()
    with pytest.raises(TranslationError, match="its one constant section"):
        capillary(
            wall=CoherentWall(method="cf4", impedance="te_tm"),
            modes=vector_modes(counts=(1, 1)),
        ).translate()
    assert repr(CoherentWall(method="cf4", growth_limit=3.0)).endswith(
        "method=cf4, growth_limit=3, fallback=None)"
    )
    assert "growth limit 3, fallback None" in str(
        CoherentWall(method="cf4", growth_limit=3.0)
    )
    assert "TE/TM impedances, paraxial, frozen per section" in str(VECTOR_WALL)
    assert "TE/TM impedances, paraxial, commutator-free per section" in str(
        CoherentWall(method="cf4", impedance="te_tm")
    )


def test_a_cf4_wall_refuses_every_declaration_the_frozen_one_refuses():
    """The varying route's doors stand for CF4 too: noise, polarisation, batch, kinks.

    The rule: a CF4 taper reaches every door the frozen one does rather
    than slipping past on its new branch. The batch axis is refused by the reference
    backend's own `max_batch = 1` before a coupled flow is looked at, and a fixed
    schedule straddling the taper's corner by `flow_edge_slack`.
    """
    with pytest.raises(TranslationError, match="refuses the 'modified' noise"):
        varying(wall=CF4_WALL, noise="modified", seed=1).translate()
    with pytest.raises(DriverError, match="polarisation"):
        varying(wall=CF4_WALL, polarisation="linear").translate()
    with pytest.raises(DriverError, match="loss=False"):
        varying(wall=CF4_WALL, loss=False).translate()
    edges = np.linspace(0.0, LENGTH, STEPS + 1)
    edges[3] -= 1e-13
    with pytest.raises(TranslationError, match=r"does not land on it.*coupled_flow"):
        varying(wall=CF4_WALL, schedule=FixedSchedule(np.diff(edges))).translate()
    simulation = varying(wall=CF4_WALL)
    problem = simulation.translate()
    batch, channels, n_pol, bins = problem.shape
    widened = replace(problem, shape=(2, channels, n_pol, bins))
    with pytest.raises(BackendError, match="max_batch = 1"):
        NumpyRefBackend(widened, grid=simulation.grid)
    with pytest.raises(BackendError, match="only for a scalar state") as refused:
        refuse_coupled_flow(replace(problem, shape=(batch, channels, 2, bins)))
    assert "not the state's polarisation axis" in str(refused.value)
    # It names the state's two components, the product route, and not the
    # synthesis, which may resolve two.
    assert "a state with two polarisation components" in str(refused.value)
    # `numba_cpu` and `native` execute the frozen flow and refuse a CF4 one at the entry
    # point: CF4 is not yet implemented on the compiled backends.
    for name in ("numba_cpu", "native", "auto"):
        with pytest.raises(DriverError, match=r"method='cf4'"):
            varying(wall=CF4_WALL, backend=name).translate()
    with pytest.raises(BackendError, match=r"does not execute a 'cf4' section"):
        NumbaCpuBackend(problem, grid=simulation.grid)


@pytest.mark.medium
def test_prop_capillary_runs_a_cf4_wall_end_to_end_and_records_its_guards():
    """The entry point runs the CF4 taper, and the record carries both guards.

    Ten steps through `prop_capillary(taper, ..., wall=CoherentWall(method="cf4"))`.
    Bound, stated first: the run solves the equation the frozen split does, so
    its final plane agrees with that run's to the two methods' own ten-step errors,
    4.71e-8 and 3.45e-9 of the state (measured 4.58e-8), and every channel's energy
    fraction with it; the record names four `"cf4"` sections, no advisory limit, the
    growth bound 1.00127 against the default limit 2 and the step-doubling defect
    1.72e-7 at the declared step, named as the asymptotic estimator it is.
    """
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": 3,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
        # CF4 sections are stepped by the Runge-Kutta march alone, and the frozen
        # run is compared with them on the same steps.
        "integrator": RungeKutta(),
    }
    walled = cast(
        "MemoryOutput",
        prop_capillary(
            taper(),
            LENGTH,
            "He",
            1.0,
            wall=CF4_WALL,
            **{"noise": None, **arguments},
        ),
    )
    frozen = cast(
        "MemoryOutput",
        prop_capillary(
            taper(),
            LENGTH,
            "He",
            1.0,
            wall=CoherentWall(),
            **{"noise": None, **arguments},
        ),
    )
    written = walled.metadata
    assert written is not None and written.modal is not None
    record = cast("Mapping[str, object]", written.modal["flow"])
    assert record["section_methods"] == ["cf4"] * 4
    assert record["certificate_id"] == "cf4"
    assert math.isinf(float(cast("float", record["advisory_step"])))
    guards = cast("Mapping[str, object]", record["cf4"])
    assert guards["growth_limit"] == CF4_GROWTH_LIMIT
    assert guards["fallback"] is None
    assert guards["defect_estimator"] == "step doubling, asymptotic"
    assert 1.0 < float(cast("float", guards["growth_bound"])) < 1.0013
    assert 1e-7 < float(cast("float", guards["defect"])) < 1e-6
    field = kept(walled.planes()[-1].field)
    control = kept(frozen.planes()[-1].field)
    scale = float(np.max(abs(control)))
    assert float(np.max(abs(field - control))) / scale < 4.71e-8 + 3.45e-9
    assert_allclose(_channel_fractions(walled), _channel_fractions(frozen), atol=1e-7)
    control_metadata = frozen.metadata
    assert control_metadata is not None and control_metadata.modal is not None
    assert "cf4" not in cast("Mapping[str, object]", control_metadata.modal["flow"])


def test_the_cf4_record_needs_its_declaration():
    """A CF4 flow recorded without its growth limit or its step is refused by name."""
    simulation = varying(wall=CF4_WALL)
    flow = simulation.translate().linear.coupled_flow
    assert flow is not None
    limits = worst_wall_limits(
        cast("Sequence[MarcatiliMode]", simulation.modes),
        GRID.omega,
        positions=[0.0],
        active=GRID.active_mask,
    )
    with pytest.raises(TranslationError, match="pass growth_limit and dz_max"):
        wall_identity(
            flow, limits, channels=[mode.channel_label for mode in simulation.modes]
        )


def _six_channel_taper(wall, dz):
    """Return the qualification taper with six channels on a uniform schedule."""
    return propagated(
        varying(
            wall=wall,
            modes=6,
            schedule=FixedSchedule.uniform(
                0.0,
                LENGTH,
                dz,
                save_planes=[0.0, LENGTH],
                hard_kinks=list(SMOOTH_STAGES[1:3]),
            ),
            dz_max=dz,
            dz_init=dz,
        ).translate(),
        GRID,
    )


@pytest.mark.slow
# Both advisory diagnostics fire at this step on purpose: it is beyond the frozen
# split's advisory limit and CF4's estimated defect, which is the regime measured.
@pytest.mark.filterwarnings("ignore:the coherent wall's")
def test_cf4_wins_at_a_step_beyond_the_frozen_split_s_advisory_limit_and_not_below():
    """Where CF4 is better: at equal step, only once the frozen residual is stiff.

    The frozen split's explicit residual has an advisory step `0.9/(K cond)`, 2.2e-5 m
    on the six-channel taper (K = 3.3e4 1/m). Below it the fifth-order split is the more
    accurate at the same step (the three-channel qualification taper at 20 um: 3.45e-9
    against CF4's 4.71e-8, in the medium tier); at three times it, CF4 is, because it
    has no explicit linear term to be stiff. Predeclared: at four steps of at most
    6.7e-5 m, CF4's error against the frozen split's 320-step reference is below the
    frozen split's own; measured 4.75e-6 against 1.78e-5. **At equal cost the frozen
    split still wins** -- CF4 costs 3.3 times as much per step (measured) and the frozen
    split gains 32 per halving -- which is why CF4 is the certified alternative and not
    the default.
    """
    reference = _six_channel_taper(CoherentWall(sections=4), LENGTH / 320)
    scale = float(np.max(abs(reference)))
    frozen = float(
        np.max(abs(_six_channel_taper(CoherentWall(), LENGTH / 3) - reference))
    )
    cf4 = float(np.max(abs(_six_channel_taper(CF4_WALL, LENGTH / 3) - reference)))
    assert cf4 < frozen, (cf4 / scale, frozen / scale)


def _taper_note_bore(count: int) -> Simulation:
    """Return the taper study's narrow end as a straight bore: 62.5 um at 800 nm.

    The study's grid (``tests/oracles/modal_design/taper_production.py``): requested
    limits 0.625 to 1.25 times the carrier wavelength, so the field grid's active
    band reaches down to half the lowest requested frequency, 2 um.
    """
    wavelength = 800e-9
    length = 1e-3
    grid = FieldGrid.from_limits(
        length, wavelength, (0.625 * wavelength, 1.25 * wavelength), 200e-15
    )
    return Simulation.from_capillary(
        62.5e-6,
        length,
        "He",
        0.0,
        wavelength=wavelength,
        grid=grid,
        absorber=absorber(grid, boundary_length=1e9),
        fwhm=30e-15,
        energy=1e-9,
        modes=count,
        wall=CoherentWall(),
        cladding=1.45,
        plasma=False,
        raman=False,
        backend="numpy_ref",
        dz_max=1e-4,
        dz_init=1e-4,
        noise=None,
    )


def test_the_validity_ratios_are_admitted_and_recorded_over_the_scored_band():
    """N = 20 at 62.5 um and 800 nm is admitted on its scored band.

    Bounds, stated first. The recorded ratios are the scored band's to the record's
    six significant digits (they are quantised measurements). The active band's
    ratios are those of bins at up to twice the longest requested wavelength, and each
    ratio is proportional to the wavelength, so they exceed the scored band's by a
    factor of two to within one bin at each band's edge -- above the default limit
    0.25 here, which the active band would breach (0.3116 on the study's grid). A block
    over the limit on the scored band itself, N = 40, is still refused by name.
    """
    with pytest.warns(GrazingIncidenceWarning, match=r"transverse_ratio reaches 0\.15"):
        simulation = _taper_note_bore(20)
        modal = simulation.metadata().modal
    assert modal is not None
    record = cast("Mapping[str, float]", modal["flow"])
    grid = simulation.grid
    positions = [0.0, 1e-3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        scored = worst_wall_limits(
            simulation.modes,
            grid.omega,
            positions=positions,
            active=grid.active_mask,
            scored=grid.inband_mask,
            validity_limit=math.inf,
        )
        whole = worst_wall_limits(
            simulation.modes,
            grid.omega,
            positions=positions,
            active=grid.active_mask,
            validity_limit=math.inf,
        )
    # Evacuated and with a constant cladding index, each ratio is exactly 1/omega
    # times a constant, so their quotient is that of the two bands' lowest bins: the
    # scored band starts within one bin above omega_min and the active one within
    # one bin above omega_min / 2.
    spacing = float(grid.omega[1] - grid.omega[0])
    lowest = float(grid.omega_min)
    for name in ("transverse_ratio", "vacuum_ratio", "quasi_static"):
        assert record[name] == pytest.approx(getattr(scored, name), rel=1e-5)
        quotient = getattr(whole, name) / getattr(scored, name)
        assert lowest / (0.5 * lowest + spacing) <= quotient
        assert quotient <= 2.0 * (1.0 + spacing / lowest)
    assert scored.worst < 0.25 < whole.worst
    assert scored.band == (
        float(grid.omega[grid.inband_mask][0]),
        float(grid.omega[grid.inband_mask][-1]),
    )
    with pytest.raises(ModeError, match=r"transverse_ratio is 0\.31.* above the 0\.25"):
        _taper_note_bore(40).translate()
