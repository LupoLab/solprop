# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The host scalars of one stage, and the device form they are the other half of.

A stage of the interaction-picture step needs three z-dependent quantities that are
*arrays* -- the propagator's exponent ``dPhi``, the response prefactor and the noise's
unitary phase -- and a handful that are *scalars*: the number density, the
pre-ionisation, the mode set's taper, the Raman decay rates and, since schema 13, the
frozen section and the residual weights of a coupled flow, and since schema 14
the weights of a moving guide's maps. Until this module the
backends fetched all seven for themselves, each through the function that hides the
representation (:func:`solprop.translate.linop.delta_phi`,
:func:`solprop.translate.linop.unitary_phase`,
:func:`solprop.translate.responses.prefactor_at`) or straight off a
:class:`~solprop.translate.chebyshev.PiecewiseChebyshev`. That works on a host and
cannot work on a device: a Clenshaw pass over a Python object is not something a
compiled step loop can call, and the z table it walks is an array of the state's size
times the degree -- 1.55 GB at the comb-to-continuum size.

The seam's evaluation form is therefore split in two:

* **terms**, the fixed arrays, uploaded once per run (:func:`terms`). A constant
  operator is one term; an expansion is ``n_terms`` of them.
* **scalars**, a small frozen record of host floats per stage (:func:`scalars`). The
  weights that multiply the terms, and every scalar path value the responses read.

A backend evaluates ``dPhi`` as one linear combination of the terms it holds over the
bundle's floats (:func:`combine`) and knows nothing else about the representation. The
z table has no such split: :attr:`solprop.translate.problem.KernelProblem.device_ready`
is the predicate, the host backends evaluate the table, the native backend's CPU build
evaluates it natively into term slots it then weighs by +-1 (SEP 29 section 6.6), and
:func:`require_device_ready` is the refusal an accelerator build raises.

The split costs nothing in accuracy because it is the arithmetic that was already
there: :meth:`~solprop.translate.expansion.ParameterExpansion.delta` *is*
``tensordot(weights(z2) - weights(z1), terms)``, and the constant path *is*
``D (z2 - z1)``, so a backend that goes through this module reproduces
:func:`~solprop.translate.linop.delta_phi` to the last bit.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop._helpers import NUMBA, Helpers
from solprop.exceptions import BackendError, TranslationError
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.coupled import flow_section, flow_weights
from solprop.translate.linop import LinearOperator
from solprop.translate.modal_path import transform_weights
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import CompiledResponses
from solprop.translate.spatial import SeparableOperator

__all__ = [
    "StageScalars",
    "StageTerms",
    "StepAnchor",
    "anchor",
    "combine",
    "require_device_ready",
    "scalars",
    "scalars_at",
    "terms",
]


@dataclass(frozen=True, eq=False)
class StageTerms:
    """The fixed arrays a backend holds for the whole run.

    Each shape annotation carries its own axis name because the two members are
    independent builds with their own term counts; nothing here ties them together.

    Attributes
    ----------
    phi : numpy.ndarray or None
        The terms of ``Phi``, complex128, shape ``(n_phi_terms, n_batch_or_1, n_modes,
        n_pol, n_omega)``. One term for a constant operator, ``n_terms`` for an
        expansion, and ``None`` for the z table, which has no such form.
    prefactor : numpy.ndarray or None
        The terms of the prefactor, complex128, shape ``(n_pre_terms, n_modes, n_pol,
        n_omega)``, on the same three cases.
    separable : solprop.translate.spatial.SeparableOperator or None
        Shared constant spatial coefficients. When present, ``phi`` and ``prefactor``
        are absent; a backend evaluates the operator and prefactor per element.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi
    >>> from solprop.translate.stage import StageTerms
    >>> table = constant_phi(np.array([-1j, -2j]), 1.0)
    >>> StageTerms(phi=table.phi_constant[np.newaxis], prefactor=None).phi.shape
    (1, 1, 1, 1, 2)
    """

    phi: Complex[np.ndarray, "n_phi_terms n_batch n_modes n_pol n_omega"] | None
    prefactor: Complex[np.ndarray, "n_pre_terms n_modes_p n_pol_p n_omega_p"] | None
    separable: SeparableOperator | None = None

    @property
    def device_ready(self) -> bool:
        """Return whether both quantities are a weighted sum of fixed arrays.

        Returns
        -------
        bool
            True when neither member is ``None``.
        """
        return self.separable is not None or (
            self.phi is not None and self.prefactor is not None
        )


@dataclass(frozen=True, eq=False)
class StepAnchor:
    """The weights of ``Phi`` at the start of a step, hoisted for all of its stages.

    ``Phi`` is cumulative and the stage needs its increment, so every stage of a step
    subtracts the weights at the same anchor. Evaluating them once rather than once per
    stage is worth a sixth of the whole per-step scalar cost on the RDW gradient, where
    a Clenshaw pass over the scalar tables is the largest part of what the host does.

    Attributes
    ----------
    z : float
        The step's anchor, m.
    phi : numpy.ndarray or None
        Weights of ``Phi(z)``, float64, shape ``(n_phi_terms,)``; ``None`` when ``Phi``
        is a z table and has no such form.

    Examples
    --------
    >>> from solprop.translate.stage import StepAnchor
    >>> StepAnchor(0.5, None).z
    0.5
    """

    z: float
    phi: Float[np.ndarray, " n_phi_terms"] | None


def anchor(
    problem: KernelProblem, z0: float, *, helpers: Helpers = NUMBA
) -> StepAnchor:
    """Return the weights a step's stages all subtract, evaluated once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated contract.
    z0 : float
        The step's anchor, m, inside the tabulated range.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the tables' Clenshaw pass: the backend's that runs the
        step (``StepDriver.helpers``, SEP 29 section 6.6), or the shared Numba one by
        default.

    Returns
    -------
    StepAnchor
        The weights of ``Phi(z0)``, or ``None`` for the z table.

    Raises
    ------
    solprop.exceptions.TranslationError
        If the position lies outside the tabulated range.
    """
    linear = problem.linear
    if linear.phi_constant is not None or linear.separable is not None:
        # Phi(z) = D z, so the weight is the position itself.
        return StepAnchor(z0, np.array([z0]))
    expansion = linear.phi_expansion
    return StepAnchor(
        z0, None if expansion is None else expansion.weights(z0, helpers=helpers)
    )


@dataclass(frozen=True, eq=False)
class StageScalars:
    """Everything about one stage that is a host scalar.

    One stage runs from the step's anchor `z0` to the node position `z1`; the weights
    for ``Phi`` are the difference between the two, because ``Phi`` is cumulative and
    the step needs its increment, while everything else is a value at `z1`.

    Attributes
    ----------
    z0 : float
        The step's anchor, m.
    z1 : float
        The stage's position, m, ``z0 + c_i h``.
    phi_delta : numpy.ndarray or None
        Weights of ``Phi(z1) - Phi(z0)``, float64, shape ``(n_phi_terms,)``; ``None``
        when ``Phi`` is a z table.
    phi_at : numpy.ndarray or None
        Weights of ``Phi(z1)`` itself, float64, shape ``(n_phi_terms,)``, which the
        noise's unitary phase needs; ``None`` on the same condition.
    prefactor : numpy.ndarray or None
        Weights of the prefactor at `z1`, float64, shape ``(n_pre_terms,)``; ``None``
        when the prefactor is a z table.
    density : float
        Number density at `z1`, m^-3, or dimensionless 1 for an absolute ``chi3``.
    taper_scale : float
        The mode set's scalar rescaling at `z1`, dimensionless; 1.0 with no taper.
    raman_rates : tuple of float
        The decay rate of each Raman term at `z1`, s^-1, in the order of `undamped`.
        Empty with no Raman response. Hashable, because it is what a backend memoises
        its transformed kernel spectrum on.
    kerr_density : float
        The Kerr leaf's own scalar scale at `z1`. With one component it is that
        component's number density in m^-3 (bitwise the `density` above) and the
        leaf's coefficient carries ``eps0 * gamma3``; with several it is the composite
        ``sum(rho_s * gamma3_s)`` in m^2/V^2 and the coefficient is ``eps0`` alone, so
        the product is the same quantity in both cases and only the split differs
        (the single-gas bits are kept by construction).
    plasma_densities : tuple of float
        Channel number densities at z1 in m^-3, in compiled channel order.
    plasma_preionisation : tuple of float
        Channel initial ionised fractions at z1, dimensionless in [0, 1].
    raman_densities : tuple of float
        Raman component amplitude densities at z1 in m^-3. Multiple components add
        these amplitudes to the transformed-kernel cache key beside raman_rates.
    flow_section : int
        Which section of ``linear.coupled_flow`` the step's **anchor** `z0` lies in, so
        that every stage of one step names the same frozen block; ``-1`` for a problem
        carrying no coupled flow. Schema 13.
    flow_weights : numpy.ndarray or None
        Weights of the coupled flow's ``(n_intervals, n_deg)`` coefficient block at
        `z1`, float64, shape ``(n_intervals * n_deg,)``, flattened in that order;
        ``None`` without a coupled flow. Zero outside the section the stage is
        evaluated on, which is the step's own whenever `z1` lies in it -- a stage
        landing exactly on a section edge reads the closing side of that kink and not
        the opening one -- and `z1`'s own otherwise, so no expansion is ever
        extrapolated. The generator and the connection share
        the axes and are combined over the same weights
        (:func:`solprop.translate.coupled.flow_weights`). Schema 13.
    transform_weights : numpy.ndarray or None
        Weights of ``transform.modal_tables``' moving maps -- the ``(n_intervals,
        n_deg)`` coefficient axes of ``synthesis`` and ``projection``, flattened in
        that order -- at `z1`, float64, shape ``(n_intervals * n_deg,)``: a backend
        evaluates ``S(z1)`` and ``Q(z1)`` as that combination, zero outside the
        section `z1` lies in (:func:`solprop.translate.modal_path.transform_weights`).
        ``None`` for a problem whose maps do not move -- no general modal tables, or
        maps of one coefficient -- so such a problem's bundle carries nothing new.
        The transform's sections are its own and need not be the coupled flow's.
        Schema 14.
    dispersive_kerr_densities : tuple of float
        Each channel of ``responses.dispersive_kerr`` its number density at `z1`, in
        m^-3, in the leaf's channel order; empty without the leaf. Schema 24.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> from solprop.translate.stage import scalars
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[1.0], dz_max=0.1, dz_init=1e-3,
    ...                    shock=False, prefactor_index=1.0, density=1.0)
    >>> bundle = scalars(problem, 0.25, 0.75)
    >>> bundle.phi_delta.tolist(), bundle.prefactor.tolist()
    ([0.5], [1.0])
    """

    z0: float
    z1: float
    phi_delta: Float[np.ndarray, " n_phi_terms"] | None
    phi_at: Float[np.ndarray, " n_phi_at"] | None
    prefactor: Float[np.ndarray, " n_pre_terms"] | None
    density: float
    taper_scale: float
    raman_rates: tuple[float, ...]
    kerr_density: float
    plasma_densities: tuple[float, ...]
    plasma_preionisation: tuple[float, ...]
    raman_densities: tuple[float, ...]
    flow_section: int = -1
    flow_weights: Float[np.ndarray, " n_flow_terms"] | None = None
    transform_weights: Float[np.ndarray, " n_transform_terms"] | None = None
    dispersive_kerr_densities: tuple[float, ...] = ()

    @property
    def preionisation(self) -> float:
        """Return the initial fraction for the one-channel specialised path.

        Returns
        -------
        float
            The sole channel's fraction, or zero when there is no sole channel.
        """
        return (
            self.plasma_preionisation[0] if len(self.plasma_preionisation) == 1 else 0.0
        )


def combine(
    weights: Float[np.ndarray, " n_weights"],
    fixed: Inexact[np.ndarray, " n_terms *value"],
) -> Inexact[np.ndarray, " *value"]:
    """Return the weighted sum of the terms, the one arithmetic a backend does.

    A single term is multiplied rather than contracted, so that a constant operator's
    ``D (z2 - z1)`` stays exact to the last bit -- the property ``numpy_ref``'s P0 rung
    rests on -- and is not routed through a BLAS call for one product. A single term
    weighted by exactly one is the **stored array itself**, as
    :func:`solprop.translate.responses.prefactor_at` returns it on the same fast path:
    the alternative is a full array multiply and an allocation per stage for a factor
    of one, which on the comb-to-continuum prefactor is 1.6 MB eight times a step.
    The array is read-only, as everything crossing the seam is.

    Parameters
    ----------
    weights : numpy.ndarray
        Host scalars, float64, shape ``(n_terms,)``.
    fixed : numpy.ndarray
        The terms, shape ``(n_terms, *value_shape)``.

    Returns
    -------
    numpy.ndarray
        The combination, of the terms' value shape.

    Raises
    ------
    BackendError
        If the counts disagree.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.stage import combine
    >>> combine(np.array([0.5]), np.array([[2.0, 4.0]])).tolist()
    [1.0, 2.0]
    >>> fixed = np.array([[2.0, 4.0]])
    >>> bool(np.shares_memory(combine(np.array([1.0]), fixed), fixed))
    True
    """
    if weights.shape[0] != fixed.shape[0]:
        msg = (
            f"{weights.shape[0]} weights cannot combine {fixed.shape[0]} terms; the "
            "bundle and the terms come from the same representation"
        )
        raise BackendError(msg)
    if fixed.shape[0] == 1:
        return fixed[0] if weights[0] == 1.0 else weights[0] * fixed[0]
    return np.tensordot(weights, fixed, axes=1)


def terms(problem: KernelProblem) -> StageTerms:
    """Return the fixed arrays of a problem's two array-valued z quantities.

    Built once per run and uploaded once by a device backend. The constant fast path is
    the one-term case rather than a case of its own, so a backend has one code path
    (``docs/dev/kernel-contract.md`` sections 3 and 5).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated contract.

    Returns
    -------
    StageTerms
        The terms of ``Phi`` and of the prefactor, each ``None`` where the z table
        carries the quantity instead.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> from solprop.translate.stage import terms
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[1.0], dz_max=0.1, dz_init=1e-3,
    ...                    shock=False, prefactor_index=1.0, density=1.0)
    >>> held = terms(problem)
    >>> held.phi.shape[0], held.prefactor.shape[0], held.device_ready
    (1, 1, True)
    """
    return StageTerms(
        phi=_phi_terms(problem.linear),
        prefactor=_prefactor_terms(problem.responses),
        separable=problem.linear.separable,
    )


def _phi_terms(
    linear: LinearOperator,
) -> Complex[np.ndarray, "n_phi_terms n_batch n_modes n_pol n_omega"] | None:
    """Return the terms of ``Phi``, or ``None`` for the z table.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        Section 3 of the contract.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_phi_terms, n_batch_or_1, n_modes, n_pol, n_omega)``.
    """
    if linear.phi_constant is not None:
        return linear.phi_constant[np.newaxis]
    if linear.phi_expansion is not None:
        return linear.phi_expansion.terms
    return None


def _prefactor_terms(
    responses: CompiledResponses,
) -> Complex[np.ndarray, "n_pre_terms n_modes n_pol n_omega"] | None:
    """Return the terms of the prefactor, or ``None`` for the z table.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the contract.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_pre_terms, n_modes, n_pol, n_omega)``.
    """
    if responses.prefactor_constant is not None:
        return responses.prefactor_constant[np.newaxis]
    if responses.prefactor_expansion is not None:
        return responses.prefactor_expansion.terms
    return None


def scalars(
    problem: KernelProblem,
    z0: float,
    z1: float,
    *,
    at: StepAnchor | None = None,
    helpers: Helpers = NUMBA,
) -> StageScalars:
    """Return the host scalars of one stage, from the step's anchor to the node.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated contract.
    z0 : float
        The step's anchor, m, inside the tabulated range.
    z1 : float
        The stage's position, m, inside the tabulated range.
    at : StepAnchor or None, optional
        The anchor's weights from :func:`anchor`, hoisted by a caller that takes every
        stage of one step. ``None`` evaluates them here, which is one Clenshaw pass per
        stage that the step did not need. Must be the anchor of `z0`.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the tables' Clenshaw pass: the backend's that runs the
        stage (``StepDriver.helpers``, SEP 29 section 6.6), or the shared Numba one by
        default.

    Returns
    -------
    StageScalars
        The weights and the scalar path values, all host floats.

    Raises
    ------
    solprop.exceptions.TranslationError
        If a position lies outside the tabulated range.
    solprop.exceptions.BackendError
        If `at` is the anchor of a different position.
    """
    linear, responses = problem.linear, problem.responses
    if at is not None and at.z != z0:
        msg = (
            f"the hoisted anchor is at z = {at.z!r} m and this stage runs from "
            f"{z0!r} m; the two are the same position or the increment is wrong"
        )
        raise BackendError(msg)
    phi_delta, phi_at = _phi_weights(linear, z0, z1, at, helpers)
    taper = problem.transform.taper_scale
    kerr = responses.kerr_coefficients
    flow = linear.coupled_flow

    def read(table: PiecewiseChebyshev) -> float:
        """Return a scalar table's value at the stage, by the backend's pass.

        Parameters
        ----------
        table : solprop.translate.chebyshev.PiecewiseChebyshev
            A scalar table.

        Returns
        -------
        float
            Its value at `z1`.
        """
        return float(table(z1, helpers=helpers))

    density = read(responses.density)
    return StageScalars(
        z0=z0,
        z1=z1,
        phi_delta=phi_delta,
        phi_at=phi_at,
        prefactor=_prefactor_weights(responses, z1, helpers),
        density=density,
        taper_scale=1.0 if taper is None else read(taper),
        raman_rates=tuple(
            read(table) for term in responses.raman_terms for table in term.decay_rates
        ),
        kerr_density=0.0
        if kerr is None
        else density
        if kerr.density is None
        else read(kerr.density),
        plasma_densities=tuple(
            read(channel.density) for channel in responses.population_channels
        ),
        plasma_preionisation=tuple(
            read(channel.preionisation) for channel in responses.population_channels
        ),
        raman_densities=tuple(
            density if term.density is None else read(term.density)
            for term in responses.raman_terms
        ),
        flow_section=-1 if flow is None else flow_section(flow, z0),
        flow_weights=None
        if flow is None
        else flow_weights(flow, z1, flow_section(flow, z0)),
        transform_weights=transform_weights(problem.transform.modal_tables, z1),
        dispersive_kerr_densities=()
        if responses.dispersive_kerr is None
        else tuple(read(table) for table in responses.dispersive_kerr.densities),
    )


def scalars_at(
    problem: KernelProblem,
    z0: float,
    positions: Sequence[float],
    *,
    at: StepAnchor | None = None,
    helpers: Helpers = NUMBA,
) -> tuple[StageScalars, ...]:
    """Return the host scalars of several stages of one step, each table read once.

    A trial knows every position it will evaluate before its first stage -- the step's
    end and ``z0 + c_i h`` for each node -- and :func:`scalars` reads every scalar
    table once per position, a Python call and a compiled Clenshaw pass apiece. Here
    each table is read **once** over all the positions
    (:meth:`~solprop.translate.chebyshev.PiecewiseChebyshev.at_positions`), which keeps
    every operation of the per-position pass, so bundle ``i`` equals
    ``scalars(problem, z0, positions[i], at=at)`` field for field and bit for bit,
    with the same interval choice at a breakpoint and the same refusal outside the
    range. The increment of ``Phi`` is formed as there: ``z1 - z0`` for a constant
    operator, the difference of the integrated weights for an expansion.

    The coupled flow's and the moving guide's weights are a basis row per position
    rather than a table, and are formed one position at a time.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated contract.
    z0 : float
        The step's anchor, m, inside the tabulated range.
    positions : sequence of float
        The stages' positions, m, inside the tabulated range, in the order the bundles
        are returned.
    at : StepAnchor or None, optional
        The anchor's weights from :func:`anchor`; must be the anchor of `z0`.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the tables' Clenshaw pass: the backend's that runs the
        stages (``StepDriver.helpers``, SEP 29 section 6.6), or the shared Numba one by
        default, for callers outside a backend.

    Returns
    -------
    tuple of StageScalars
        One bundle per position, in order.

    Raises
    ------
    solprop.exceptions.TranslationError
        If a position lies outside the tabulated range.
    solprop.exceptions.BackendError
        If `at` is the anchor of a different position.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> from solprop.translate.stage import scalars_at
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11,
    ...                      shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-14),
    ...                    save_planes=[1.0], dz_max=0.1, dz_init=1e-3,
    ...                    shock=False, prefactor_index=1.0, density=1.0)
    >>> [bundle.phi_delta.tolist() for bundle in scalars_at(problem, 0.25, [0.5, 0.75])]
    [[0.25], [0.5]]
    """
    linear, responses = problem.linear, problem.responses
    if at is not None and at.z != z0:
        msg = (
            f"the hoisted anchor is at z = {at.z!r} m and these stages run from "
            f"{z0!r} m; the two are the same position or the increment is wrong"
        )
        raise BackendError(msg)
    where = list(positions)
    count = len(where)
    if count == 0:
        return ()
    zs = np.array(where, dtype=np.float64)
    read: dict[int, list[float]] = {}

    def column(table: PiecewiseChebyshev) -> list[float]:
        """Return a scalar table's values at every position, reading it once.

        Parameters
        ----------
        table : solprop.translate.chebyshev.PiecewiseChebyshev
            A scalar table.

        Returns
        -------
        list of float
            Its values at the step's positions, in order.
        """
        held = read.get(id(table))
        if held is None:
            held = read[id(table)] = table.at_positions(zs, helpers=helpers).tolist()
        return held

    phi = _phi_rows(linear, z0, where, zs, at, helpers)
    prefactor = _prefactor_rows(responses, where, zs, helpers)
    density = column(responses.density)
    taper = problem.transform.taper_scale
    tapers = [1.0] * count if taper is None else column(taper)
    rates = [
        column(table) for term in responses.raman_terms for table in term.decay_rates
    ]
    kerr = responses.kerr_coefficients
    kerrs = (
        [0.0] * count
        if kerr is None
        else density
        if kerr.density is None
        else column(kerr.density)
    )
    channels = responses.population_channels
    plasma = [column(channel.density) for channel in channels]
    preionised = [column(channel.preionisation) for channel in channels]
    raman = [
        density if term.density is None else column(term.density)
        for term in responses.raman_terms
    ]
    dispersive = (
        []
        if responses.dispersive_kerr is None
        else [column(table) for table in responses.dispersive_kerr.densities]
    )
    flow = linear.coupled_flow
    section = -1 if flow is None else flow_section(flow, z0)
    modal = problem.transform.modal_tables
    return tuple(
        StageScalars(
            z0=z0,
            z1=z1,
            phi_delta=phi[i][0],
            phi_at=phi[i][1],
            prefactor=prefactor[i],
            density=density[i],
            taper_scale=tapers[i],
            raman_rates=tuple(values[i] for values in rates),
            kerr_density=kerrs[i],
            plasma_densities=tuple(values[i] for values in plasma),
            plasma_preionisation=tuple(values[i] for values in preionised),
            raman_densities=tuple(values[i] for values in raman),
            flow_section=section,
            flow_weights=None if flow is None else flow_weights(flow, z1, section),
            transform_weights=transform_weights(modal, z1),
            dispersive_kerr_densities=tuple(values[i] for values in dispersive),
        )
        for i, z1 in enumerate(where)
    )


def _phi_rows(
    linear: LinearOperator,
    z0: float,
    where: list[float],
    zs: Float[np.ndarray, " n_points"],
    at: StepAnchor | None,
    helpers: Helpers,
) -> list[
    tuple[
        Float[np.ndarray, " n_phi_terms"] | None, Float[np.ndarray, " n_phi_at"] | None
    ]
]:
    """Return :func:`_phi_weights` at each of several positions, reading ``Phi`` once.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        Section 3 of the contract.
    z0 : float
        The step's anchor, m.
    where : list of float
        The positions, m.
    zs : numpy.ndarray
        The same positions as float64, shape ``(n_points,)``.
    at : StepAnchor or None
        The anchor's weights, hoisted, or ``None`` to evaluate them here.
    helpers : solprop._helpers.Helpers
        The implementation of the Clenshaw pass.

    Returns
    -------
    list of tuple
        Per position, ``(weights of Phi(z1) - Phi(z0), weights of Phi(z1))``.
    """
    if linear.phi_constant is not None or linear.separable is not None:
        # The increment is the distance itself, formed per position exactly as
        # `_phi_weights` forms it: no accumulated phase is ever subtracted.
        return [(np.array([z1 - z0]), np.array([z1])) for z1 in where]
    expansion = linear.phi_expansion
    if expansion is None:
        return [(None, None)] * len(where)
    here = expansion.weights.at_positions(zs, helpers=helpers)
    anchored = (
        expansion.weights(z0, helpers=helpers)
        if at is None or at.phi is None
        else at.phi
    )
    delta = here - anchored
    return [(delta[i], here[i]) for i in range(len(where))]


def _prefactor_rows(
    responses: CompiledResponses,
    where: list[float],
    zs: Float[np.ndarray, " n_points"],
    helpers: Helpers,
) -> list[Float[np.ndarray, " n_pre_terms"] | None]:
    """Return :func:`_prefactor_weights` at each of several positions.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the contract.
    where : list of float
        The positions, m.
    zs : numpy.ndarray
        The same positions as float64, shape ``(n_points,)``.
    helpers : solprop._helpers.Helpers
        The implementation of the Clenshaw pass.

    Returns
    -------
    list
        Per position, the weights, or ``None`` for the z table.
    """
    if responses.prefactor_constant is not None or responses.separable is not None:
        return [_prefactor_weights(responses, z1, helpers) for z1 in where]
    expansion = responses.prefactor_expansion
    if expansion is None:
        return [None] * len(where)
    rows = expansion.weights.at_positions(zs, helpers=helpers)
    return [rows[i] for i in range(len(where))]


def _phi_weights(
    linear: LinearOperator,
    z0: float,
    z1: float,
    at: StepAnchor | None,
    helpers: Helpers,
) -> tuple[
    Float[np.ndarray, " n_phi_terms"] | None, Float[np.ndarray, " n_phi_at"] | None
]:
    """Return the increment's weights and the position's, or two ``None``.

    An expansion of ``Phi`` is cumulative and its integrals vanish at the start of the
    tabulated range, so the value at `z1` -- which is what the noise's unitary phase
    needs -- is the weights at `z1` against the same terms.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        Section 3 of the contract.
    z0, z1 : float
        The step's anchor and the stage's position, m.
    at : StepAnchor or None
        The anchor's weights, hoisted, or ``None`` to evaluate them here.
    helpers : solprop._helpers.Helpers
        The implementation of the Clenshaw pass.

    Returns
    -------
    tuple
        ``(weights of Phi(z1) - Phi(z0), weights of Phi(z1))``, each shape
        ``(n_phi_terms,)``, or ``(None, None)`` for the z table.
    """
    if linear.phi_constant is not None or linear.separable is not None:
        # Phi(z) = D z for a constant operator, so the increment's weight is the
        # distance and the position's is the position: `delta_phi` to the last bit.
        return np.array([z1 - z0]), np.array([z1])
    expansion = linear.phi_expansion
    if expansion is None:
        return None, None
    here = expansion.weights(z1, helpers=helpers)
    anchored = (
        expansion.weights(z0, helpers=helpers)
        if at is None or at.phi is None
        else at.phi
    )
    return here - anchored, here


def _prefactor_weights(
    responses: CompiledResponses, z: float, helpers: Helpers
) -> Float[np.ndarray, " n_pre_terms"] | None:
    """Return the prefactor's weights at a position, or ``None`` for the z table.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        Section 5 of the contract.
    z : float
        Position, m.
    helpers : solprop._helpers.Helpers
        The implementation of the Clenshaw pass.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_pre_terms,)``. Exactly ``[1.0]`` for a constant, so that combining
        returns the stored array's own values.
    """
    if responses.prefactor_constant is not None or responses.separable is not None:
        low, high = responses.prefactor_span
        if not low <= z <= high:
            msg = f"z = {z!r} m lies outside the table, which covers [{low}, {high}] m"
            raise TranslationError(msg)
        return np.array([1.0])
    expansion = responses.prefactor_expansion
    return None if expansion is None else expansion.weights(z, helpers=helpers)


def require_device_ready(problem: KernelProblem, name: str) -> None:
    """Refuse a problem whose z quantities an accelerator would hold as a z table.

    A backend whose step loop runs on an accelerator calls this at construction: the
    native backend's device builds. The z table is a Chebyshev table of the state's own
    size times its degree and intervals, which the device would have to hold resident;
    the translation layer falls back to it for a mode set that declares no scalar
    parameter paths, that disagrees about their values, or whose operator kinks in a
    parameter (``docs/dev/kernel-contract.md`` section 3). Every CPU backend runs it:
    ``numpy_ref`` and ``numba_cpu`` evaluate it on the host, and the native backend's
    CPU build evaluates it natively (SEP 29 section 6.6).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated contract.
    name : str
        The backend's registry name, for the message.

    Raises
    ------
    solprop.exceptions.BackendError
        If either quantity is a z table, naming which, its size and the backends that
        run it.

    Examples
    --------
    >>> from solprop.translate.stage import require_device_ready
    >>> require_device_ready.__name__
    'require_device_ready'
    """
    if problem.device_ready:
        return
    tabulated = [
        quantity
        for quantity, kind in (
            ("the linear operator Phi", problem.linear.phi_representation),
            ("the response prefactor", problem.responses.prefactor_representation),
        )
        if kind == "table"
    ]
    size = 0
    if problem.linear.phi_coefficients is not None:
        size += problem.linear.phi_coefficients.nbytes
    if problem.responses.prefactor is not None:
        size += problem.responses.prefactor.coefficients.nbytes
    # The fix leads each message: "auto" picks the native backend whenever it is
    # installed and never falls back, so this refusal is what a default call on a
    # device build meets, and its reader needs the builds and backends that run it.
    fix = (
        f"The CPU backends run it: a CPU build of {name}, which evaluates the table "
        'natively, backend="numba_cpu" or backend="numpy_ref".'
    )
    held = (
        f"{' and '.join(tabulated)} took the z table, whose coefficients "
        f"({size / 1e6:.3g} MB: the state's size times the degree times the "
        "intervals) a device step loop would hold resident"
    )
    if problem.responses.n_species > 1:
        labels = ", ".join(problem.responses.labels)
        raise BackendError(
            f"{name} cannot run the independent profiles of mixture [{labels}] on an "
            f"accelerator: {held}. {fix} Why the table: independent per-species "
            "density paths exceed the two-parameter tensor expansion (a total-degree "
            "expansion costs too much memory and build time), and a fill at one "
            "total pressure takes the table only for the reasons a single gas does."
        )
    msg = (
        f"{name} cannot run this problem on an accelerator: {held}. {fix} Why the "
        "table: it is the general representation, taken when the mode set declares "
        "no scalar parameter paths, when its modes disagree about them, when a mode's "
        "declared parameters are not the whole of its z dependence (it reports "
        "parameters_own_z = False, as a solprop.nonlinear.gnlse.SpecifiedMode with a "
        "moving dispersion does), when the operator kinks in a parameter rather than "
        "being analytic in it, or when the expansion in the parameters does not "
        "reach its tolerance; a mode that declares them implements "
        "solprop.modes.protocol.ParameterisedMode."
    )
    raise BackendError(msg)
