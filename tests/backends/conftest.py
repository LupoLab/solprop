# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures for the backends: the problems, the operators and the backend itself.

Every test here that reaches a backend only through the seam of
:class:`solprop.backends.protocol.Backend` takes ``backend_class`` and is run once per
backend. What stays on ``numpy_ref`` is what could not be anything else: the tests that
reach into its private methods, the ones that call
:mod:`solprop.backends.numpy_ref.passes` directly, and the handful that compare two
paths **within** one backend. No comparison of two computations is bitwise (SEP 25):
cross-backend and within-backend agreement alike are held to rounding budgets.

The GNLSE soliton problems are the ones ``tests/driver/test_scaffold.py`` propagates,
so the two integrators can be compared directly. The synthetic problems substitute one
group of an assembled :class:`~solprop.translate.problem.KernelProblem` -- normally the
linear operator, sometimes the density path -- which is how a closed-form ``Phi`` or a
prescribed ``gamma(z)`` reaches a backend without the physics layer having to be able
to express it.
"""

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from selected_backend import selected_backend

try:
    # Load Kokkos before pyFFTW's wheel-private OpenMP runtime on macOS. The backend
    # still remains optional: a source tree without the plugin collects the base pair.
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]
except ImportError:
    NativeBackend = None

from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SechPulse
from solprop.stepper import PIController
from solprop.translate.chebyshev import PiecewiseChebyshev, build
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import LinearOperator, delta_phi
from solprop.translate.masks import DEFAULT_ABSORBER_SCHEME, AbsorberScheme
from solprop.translate.problem import Flags, KernelProblem, assemble
from solprop.translate.responses import prefactor_at

# The GNLSE soliton of tests/driver/test_scaffold.py and docs/tutorials, so that the
# reference backend and the fixed-step scaffold propagate the very same problem.
BETA2 = -11.83e-27  # s^2/m
GAMMA = 0.11  # 1/(W m)
TAU0 = 100e-15  # s, the natural sech width
WAVELENGTH = 835e-9  # m
DISPERSION_LENGTH = TAU0**2 / abs(BETA2)
SOLITON_PERIOD = math.pi / 2 * DISPERSION_LENGTH
# The N-soliton returns to its shape after one period with the global phase +pi/4 in the
# optics convention; the internal FFT convention is its conjugate.
RECURRENCE_PHASE = -math.pi / 4

# The GNLSE's beta is a fitted dispersion rather than a physical effective index, so the
# prefactor takes 1.0 for it and freezes the shock term off (ADR-026).
GNLSE_SETTINGS = {"density": 1.0, "shock": False, "prefactor_index": 1.0}


BACKENDS: dict[str, type[object]] = {
    "numpy_ref": NumpyRefBackend,
    "numba_cpu": NumbaCpuBackend,
}
if NativeBackend is not None:
    BACKENDS["native"] = NativeBackend
"""Every backend a portable test is run on, by the name the registry knows."""

EQUIVALENCE_BACKENDS: dict[str, type[object]] = {"numba_cpu": NumbaCpuBackend}
if NativeBackend is not None:
    EQUIVALENCE_BACKENDS["native"] = NativeBackend

# Every backend in the table propagates a spatial geometry today; the two selections
# stay separate because spatial execution is a capability a new backend need not have.
SPATIAL_BACKENDS = dict(BACKENDS)
SPATIAL_EQUIVALENCE_BACKENDS = {
    name: implementation
    for name, implementation in EQUIVALENCE_BACKENDS.items()
    if name in SPATIAL_BACKENDS
}

NATIVE_DEFERRED_TESTS = {
    "test_recomputing_the_propagators_costs_less_memory": (
        "Kokkos's propagator scratch has one fixed residency policy"
    ),
}


@pytest.fixture(params=sorted(BACKENDS))
def backend_name(request):
    """The registry name of the backend under test, once per backend."""
    return request.param


@pytest.fixture
def backend_class(backend_name, request):
    """The backend class itself, for a test that constructs one directly.

    Constructing the class rather than going through
    :func:`solprop.backends.registry.select` keeps the keyword arguments the tests
    already pass -- ``modes``, ``profiles`` -- and the registry's own dispatch has its
    own test.
    """
    original_name = getattr(request.node, "originalname", request.node.name)
    if backend_name == "native" and original_name in NATIVE_DEFERRED_TESTS:
        pytest.skip(NATIVE_DEFERRED_TESTS[original_name])
    if backend_name not in BACKENDS:
        # A test that names its backends itself -- to put them in different tiers --
        # names the optional ones too, and this tree may not have them installed.
        pytest.skip(f"the {backend_name} backend is not installed")
    return BACKENDS[backend_name]


@pytest.fixture(params=sorted(EQUIVALENCE_BACKENDS))
def equivalence_backend_name(request):
    """Name one compiled backend checked against the NumPy oracle."""
    return request.param


@pytest.fixture
def equivalence_backend_class(equivalence_backend_name):
    """Return the compiled backend class for an equivalence-ladder rung."""
    return EQUIVALENCE_BACKENDS[equivalence_backend_name]


@pytest.fixture(params=sorted(SPATIAL_BACKENDS))
def spatial_backend_class(request):
    """Return each implementation supporting spatial propagation."""
    return SPATIAL_BACKENDS[request.param]


@pytest.fixture(params=sorted(SPATIAL_EQUIVALENCE_BACKENDS))
def spatial_equivalence_backend_class(request):
    """Return each compiled spatial implementation checked against NumPy."""
    return SPATIAL_EQUIVALENCE_BACKENDS[request.param]


@pytest.fixture(scope="session")
def selected_backend_name():
    """Return the backend selected for direct ladders by ``SOLPROP_TEST_BACKEND``."""
    name = selected_backend()
    if name not in BACKENDS:
        usable = ", ".join(sorted(BACKENDS))
        pytest.fail(f"SOLPROP_TEST_BACKEND={name!r} is unavailable; use {usable}")
    return name


@pytest.fixture(scope="session")
def selected_backend_class(selected_backend_name):
    """Return the class selected for direct ladders by ``SOLPROP_TEST_BACKEND``."""
    return BACKENDS[selected_backend_name]


@pytest.fixture(scope="session")
def narrow_grid():
    """A short-window envelope grid, 1024 bins, for the tests that only need a field."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12)


@pytest.fixture(scope="session")
def soliton_grid():
    """The scaffold's envelope grid, 4096 bins, wide enough for the N = 6 soliton."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 4e-12)


@pytest.fixture(scope="session")
def wide_grid():
    """A band that holds the compressed N = 6 pulse: 350 nm to 3 um, 8192 bins."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (350e-9, 3e-6), 4e-12)


@pytest.fixture(scope="session")
def field_grid():
    """A field grid for the parity and gradient-capillary tests."""
    return FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)


@pytest.fixture(scope="session")
def gnlse(narrow_grid):
    """The dispersion-specified fibre of the soliton problems."""
    return problem_from(
        [0.0, 0.0, BETA2], omega_ref=narrow_grid.omega0, gamma=GAMMA, shock=False
    )


def soliton_field(grid, order):
    """Return the N-soliton input of the given order, in the canonical layout."""
    peak = order**2 * abs(BETA2) / (GAMMA * TAU0**2)
    return SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)


@pytest.fixture(scope="session")
def soliton_problem(gnlse):
    """Return ``(grid, order, **overrides) -> KernelProblem`` for an N-soliton run."""

    def build_problem(grid, order, *, save_planes, **overrides):
        settings = {
            "controller": PIController(rtol=1e-10, atol=1e-18),
            "dz_max": float(save_planes[-1]),
            "dz_init": 1e-4,
        } | overrides
        return assemble(
            grid,
            gnlse.mode,
            [gnlse.kerr],
            soliton_field(grid, order),
            save_planes=save_planes,
            **GNLSE_SETTINGS,
            **settings,
        )

    return build_problem


@pytest.fixture(scope="session")
def linear_problem(gnlse):
    """Return ``(grid, **overrides) -> KernelProblem`` with the Kerr term switched off.

    Two ways of switching it off are wanted: ``chi3 = 0`` keeps every pass in the step
    and multiplies by nothing, and an empty response list removes the nonlinear
    evaluation altogether. Both must give the exact linear flow.
    """

    def build_problem(grid, *, save_planes, responses=(), **overrides):
        settings = {
            "controller": PIController(rtol=1e-8, atol=1e-18),
            "dz_max": float(save_planes[-1]),
            "dz_init": 1e-4,
        } | overrides
        return assemble(
            grid,
            gnlse.mode,
            list(responses),
            soliton_field(grid, 1),
            save_planes=save_planes,
            **GNLSE_SETTINGS,
            **settings,
        )

    return build_problem


@pytest.fixture(scope="session")
def zero_kerr():
    """A Kerr response that evaluates every pass and contributes nothing."""
    return KerrResponse(chi3=0.0, thg=False)


@pytest.fixture(scope="session")
def substitute():
    """Return ``(base, **groups) -> KernelProblem`` replacing whole contract groups.

    The flags are re-derived from the substituted content by
    :meth:`solprop.translate.problem.Flags.from_content`, so they cannot disagree with
    what they describe, and the result is validated exactly as ``assemble`` validates
    its own. Pass ``flags=`` to declare something the compilers cannot yet build -- a
    plasma or Raman term -- which is precisely what a backend has to refuse.
    """

    def replace_groups(base, *, flags=None, validate=True, **groups):
        parts = {
            "grids": base.grids,
            "linear": base.linear,
            "transform": base.transform,
            "responses": base.responses,
            "initial": base.initial,
            "stepper": base.stepper,
        } | groups
        derived = flags or Flags.from_content(
            base.shape,
            parts["linear"],
            parts["transform"],
            parts["responses"],
            parts["initial"],
        )
        problem = KernelProblem(
            schema_version=base.schema_version,
            grid_kind=base.grid_kind,
            shape=base.shape,
            flags=derived,
            stats_spec=base.stats_spec,
            **parts,
        )
        if validate:
            problem.validate()
        return problem

    return replace_groups


def _frozen_table(table: PiecewiseChebyshev) -> PiecewiseChebyshev:
    """Return the table with read-only arrays, as the seam requires of every array."""
    parts = []
    for values in (
        table.coefficients,
        table.breakpoints,
        table.degrees,
        table.hard_kinks,
    ):
        frozen = np.ascontiguousarray(values)
        frozen.setflags(write=False)
        parts.append(frozen)
    return PiecewiseChebyshev(*parts)


@pytest.fixture(scope="session")
def synthetic_expansion():
    """Return ``(base, terms, weights_of, **options) -> LinearOperator`` from a sum.

    ``Phi(omega, z) = sum_j terms[j](omega) * weights_of(z)[..., j]`` -- a closed-form
    operator written as the separable sum that :class:`ParameterExpansion` is, with
    ``terms`` the fixed arrays a backend holds and ``weights_of`` the scalar paths the
    translation layer evaluates per stage. Tabulating a ``Phi`` that is known in closed
    form keeps the table's own error out of a test of the stepper.

    It is an expansion and not a z table because the z table is the oracle's general
    representation alone: a compiled backend refuses it (`device_ready`), and these
    tests are run on every backend. The parameter axis is synthetic -- one nominal path
    of degree ``n_terms - 1``, which is the only shape a tensor product of one
    parameter takes -- so the group's own invariants hold while the weights are
    whatever the case needs.
    """

    def build_operator(
        base: KernelProblem,
        terms: Sequence[np.ndarray],
        weights_of: Callable[[np.ndarray], np.ndarray],
        *,
        kinks: Sequence[float] = (),
        tol: float = 1e-13,
    ) -> LinearOperator:
        z0, z1 = base.linear.z_span
        weights = _frozen_table(
            build(weights_of, z0, z1, tol=tol, mandatory_breakpoints=kinks)
        )
        stacked = np.stack(
            [np.asarray(one, dtype=complex).reshape(1, 1, 1, -1) for one in terms]
        )
        expansion = ParameterExpansion(
            parameters=("synthetic",),
            ranges=np.array([[0.0, 1.0]]),
            degrees=np.array([len(terms) - 1], dtype=np.intp),
            terms=stacked,
            weights=weights,
        )
        return dataclasses.replace(
            base.linear,
            phi_representation="expansion",
            phi_breakpoints=weights.breakpoints,
            phi_coefficients=None,
            phi_degrees=weights.degrees,
            hard_kinks=weights.hard_kinks,
            phi_constant=None,
            phi_expansion=expansion,
            beta0=base.linear.beta0,
            beta1=base.linear.beta1,
        )

    return build_operator


@pytest.fixture(scope="session")
def expanded_prefactor():
    """Return ``responses -> CompiledResponses`` that drops the prefactor fast path.

    A synthetic z-varying ``Phi`` on a fibre that does not vary may keep its constant
    prefactor (``validate`` checks the constant against the prefactor's own table, not
    against the operator); dropping it here makes these problems read the
    prefactor from an expansion, which is the path a real gradient takes anyway. One
    term whose weight is one: the numbers are the constant's, the representation is not.
    """

    def drop(responses):
        constant = responses.prefactor_constant
        span = responses.prefactor_span
        weights = _frozen_table(
            build(lambda z: np.ones((np.asarray(z).size, 1)), *span, tol=1e-13)
        )
        expansion = ParameterExpansion(
            parameters=("synthetic",),
            ranges=np.array([[0.0, 1.0]]),
            degrees=np.array([0], dtype=np.intp),
            terms=np.asarray(constant, dtype=complex)[np.newaxis],
            weights=weights,
        )
        return dataclasses.replace(
            responses,
            prefactor_representation="expansion",
            prefactor=None,
            prefactor_expansion=expansion,
            prefactor_constant=None,
        )

    return drop


@pytest.fixture(scope="session")
def batched_problem(gnlse, narrow_grid):
    """A two-member batch, which the reference backend refuses (``max_batch = 1``)."""
    start = soliton_field(narrow_grid, 1)
    return assemble(
        narrow_grid,
        gnlse.mode,
        [gnlse.kerr],
        np.concatenate([start[np.newaxis], start[np.newaxis]]),
        controller=PIController(rtol=1e-8, atol=1e-18),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=1e-4,
        batch_params={"member": np.array([0.0, 1.0])},
        **GNLSE_SETTINGS,
    )


@pytest.fixture(scope="session")
def thg_grid():
    """An envelope grid oversampled six times, which resolves the third harmonic."""
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12, thg=True)


@pytest.fixture(scope="session")
def thg_problem(thg_grid):
    """Return ``(*, thg) -> KernelProblem`` on the oversampled envelope grid."""
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=thg_grid.omega0, gamma=GAMMA, shock=False
    )

    def build_problem(*, thg):
        return assemble(
            thg_grid,
            fibre.mode,
            [KerrResponse(chi3=fibre.kerr.chi3, thg=thg)],
            soliton_field(thg_grid, 1),
            controller=PIController(rtol=1e-9, atol=1e-18),
            save_planes=[1e-3],
            dz_max=1e-3,
            dz_init=1e-5,
            **GNLSE_SETTINGS,
        )

    return build_problem


@pytest.fixture(scope="session")
def capillary_problem(field_grid):
    """Return ``(*, scheme) -> KernelProblem`` for the gradient capillary.

    The fixture of ``tests/translate/test_problem.py``: a Marcatili HE11 capillary with
    a two-point pressure gradient, so the operator varies along z, the prefactor is a
    table and the fill's stage boundary at 0.2 m is a hard kink. Helium at 2 bar is
    outside the published range of the index model, which is what the warning filter is
    for.
    """
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)

    def build_problem(*, scheme: AbsorberScheme = DEFAULT_ABSORBER_SCHEME):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaterialRangeWarning)
            return assemble(
                field_grid,
                mode,
                [KerrResponse(chi3=2.2e-52, thg=True)],
                start,
                controller=PIController(rtol=1e-8, atol=1e-14),
                save_planes=[0.1, 0.2],
                dz_max=0.01,
                dz_init=1e-4,
                absorber=absorber(field_grid),
                absorber_scheme=scheme,
            )

    return build_problem


@pytest.fixture(scope="session")
def tabulated_problem(capillary_problem, substitute):
    """The gradient capillary with both z-varying quantities on the z table.

    Nothing this package builds today takes the z table for a capillary -- the
    Marcatili mode declares its parameter paths, so the expansion serves it -- so the
    tables are built here from the values the expansions give, the way a mode set that
    declares nothing would present them. It is the one problem a compiled backend
    refuses and the oracle runs.
    """
    problem = capillary_problem()
    phi = build(
        lambda z: np.stack([delta_phi(problem.linear, 0.0, float(one)) for one in z]),
        *problem.linear.z_span,
        tol=1e-10,
    )
    prefactor = _frozen_table(
        build(
            lambda z: np.stack(
                [prefactor_at(problem.responses, float(one)) for one in z]
            ),
            *problem.responses.prefactor_span,
            tol=1e-10,
        )
    )
    return substitute(
        problem,
        linear=dataclasses.replace(
            problem.linear,
            phi_expansion=None,
            phi_representation="table",
            phi_breakpoints=phi.breakpoints,
            phi_coefficients=phi.coefficients,
            phi_degrees=phi.degrees,
            hard_kinks=phi.hard_kinks,
            phi_constant=None,
            beta0=problem.linear.beta0,
            beta1=problem.linear.beta1,
        ),
        responses=dataclasses.replace(
            problem.responses,
            prefactor_representation="table",
            prefactor=prefactor,
            prefactor_expansion=None,
            prefactor_constant=None,
        ),
    )


@dataclasses.dataclass(frozen=True)
class SolitonScales:
    """The three numbers the soliton tests read off the fibre and the pulse."""

    dispersion_length: float
    period: float
    recurrence_phase: float


@pytest.fixture(scope="session")
def scales():
    """The dispersion length, the soliton period and the recurrence phase."""
    return SolitonScales(DISPERSION_LENGTH, SOLITON_PERIOD, RECURRENCE_PHASE)


@pytest.fixture(scope="session")
def gnlse_settings():
    """The three assembly settings that make a fitted dispersion a plain GNLSE."""
    return dict(GNLSE_SETTINGS)
