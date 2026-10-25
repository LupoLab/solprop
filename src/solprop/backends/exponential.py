# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The phi functions and exponential Adams weights, in NumPy (SEP 16 sections 4.2-4.3).

Exponential Adams integrates the polynomial through untransported source samples
exactly against the linear flow. For a constant operator ``D``, a step ``h`` and the
Lagrange basis ``ell_j(theta) = sum_k c_jk theta**k`` on the normalised nodes,

    w_j(x) = int_0^1 exp((1 - theta) x) ell_j(theta) dtheta
           = sum_k c_jk k! phi_{k+1}(x),

with ``x = h D`` per element and ``phi_r(x) = sum_m x**m / (m + r)!``. At a fraction
``xi`` of the step the same integral to ``xi`` is
``sum_k c_jk k! xi**(k+1) phi_{k+1}(xi x)``. At ``x = 0`` these are the ordinary
Adams weights (:func:`solprop.stepper.multistep.integrated_weights`).

The phi functions are evaluated by two algorithms, each where it is stable:

- **near zero** (``|x| <= 8``): scale to ``y = x / 2**s`` with ``|y| <= 1/2``, sum the
  Taylor series of the highest function to a fixed number of terms, recur down with
  ``phi_k(y) = 1/k! + y phi_{k+1}(y)``, which contracts an error by ``|y|`` and never
  divides by ``y``, and undo the scaling by the doubling identity
  ``phi_k(2y) = [exp(y) phi_k(y) + sum_{j=1}^k phi_j(y) / (k - j)!] / 2**k``, obtained
  by splitting the defining integral at one half [1]_ [2]_;
- **away from zero** (``|x| > 8``): recur up from ``phi_0 = exp(x)``,
  ``phi_k(x) = (phi_{k-1}(x) - 1/(k-1)!) / x``, whose division by ``|x| > 8`` contracts
  the error of every earlier function.

Both are checked against augmented matrix exponentials and independent quadrature in
``tests/backends/test_exponential_weights.py``, on both sides of the join. This module
is the reference the compiled backends' moment kernels are checked against.

References
----------
.. [1] S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff
       systems", J. Comput. Phys. 176, 430 (2002). doi:10.1006/jcph.2002.6995
.. [2] M. Hochbruck and A. Ostermann, "Exponential integrators", Acta Numerica 19,
       209 (2010), section 2. doi:10.1017/S0962492910000048
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop._helpers import NUMBA, Helpers
from solprop.exceptions import BackendError, StepperError
from solprop.stepper.multistep import (
    MAX_ADAMS_ORDER,
    CarrierFrame,
    Multistep,
    lagrange_coefficients,
)
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import LinearOperator
from solprop.translate.problem import KernelProblem

__all__ = [
    "COUPLED_SECTIONS",
    "EXPONENTIAL_REPRESENTATIONS",
    "MAX_PHI_ORDER",
    "PHI_JOIN",
    "CarrierChord",
    "CarrierReference",
    "carrier_chord",
    "carrier_reference",
    "exponential_weights",
    "phi_functions",
    "refuse_carrier_frame",
    "refuse_formulation",
]

MAX_PHI_ORDER = MAX_ADAMS_ORDER + 2
"""Highest phi function evaluated: the corrector's ``MAX_ADAMS_ORDER + 1`` nodes need
``phi_1`` to ``phi_{MAX_ADAMS_ORDER + 1}``, and one more serves a neighbouring order."""

PHI_JOIN = 3.0
"""``|x|`` above which the phi functions are recurred up from ``exp(x)``.

Below it the upward recurrence loses digits to the subtraction of ``1/(k-1)!``: at
order eight, 291 ulp of the natural scale at ``|x| = 2`` and 32 at 3, against
augmented exponentials. Above it the scaled series would need more than three
doublings. Three keeps both sides within the witnesses' 256 ulp with a margin and
bounds the doublings, the most expensive part of the near branch (SEP 16 phase 3;
eight, the SEP 15 study's join, cost up to four)."""

COUPLED_SECTIONS = frozenset({"constant_schur", "frozen_residual"})
"""The coupled flow's section methods the exponential formulation steps (SEP 16 Q3)."""

EXPONENTIAL_REPRESENTATIONS = ("constant", "separable", "expansion", "table")
"""The linear-operator representations the exponential formulation steps (SEP 16).

A constant operator's reference flow over a step is its exact flow and its residual
is the source alone, ``R_n = F``. A varying one (``expansion``, ``table``) takes the
compensated reference of SEP 16 section 4.2 (phase 4): the chord over the step and the
transported compensation ``(D(z_j) - B_n) u_j``, with ``D`` the derivative of the
representation itself."""

_SERIES_TERMS = 24
"""Taylor terms of the highest phi function at ``|y| <= 1/2``.

The tail after ``M`` terms is below ``|y|**M / (M + r)!`` times a geometric factor of at
most two: ``2**-24 / 24!`` is 1e-31, far under one ulp of ``phi_r(y) >= 1/(2 r!)``."""

_RECIPROCAL_FACTORIAL = tuple(1.0 / math.factorial(k) for k in range(MAX_PHI_ORDER + 2))


def phi_functions(
    x: Inexact[np.ndarray, "*shape"] | complex, order: int
) -> Complex[np.ndarray, " n_phi *shape"]:
    """Return ``phi_0(x)`` to ``phi_order(x)`` for each element of `x`.

    ``phi_0(x) = exp(x)`` and ``phi_r(x) = int_0^1 exp((1 - theta) x)
    theta**(r-1) / (r-1)! dtheta``.

    Parameters
    ----------
    x : numpy.ndarray or complex
        The arguments, dimensionless (a step times the linear operator). Their real
        parts must not exceed 700, where ``exp`` overflows.
    order : int
        The highest function, in ``[1, MAX_PHI_ORDER]``.

    Returns
    -------
    numpy.ndarray
        ``phi_r(x)`` along a new leading axis of length ``order + 1``.

    Raises
    ------
    StepperError
        If an argument is not finite or overflows, or the order is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.exponential import phi_functions
    >>> values = phi_functions(np.array([0.0, 1.0]), 2)
    >>> np.round(values.real, 12).tolist()
    [[1.0, 2.718281828459], [1.0, 1.718281828459], [0.5, 0.718281828459]]
    """
    if not 1 <= order <= MAX_PHI_ORDER:
        msg = f"phi order must lie in [1, {MAX_PHI_ORDER}], got {order!r}"
        raise StepperError(msg)
    values = np.asarray(x, dtype=np.complex128)
    if not np.all(np.isfinite(values)) or np.any(values.real > 700.0):
        msg = "phi arguments must be finite with real parts at most 700"
        raise StepperError(msg)
    flat = values.reshape(-1)
    out = np.empty((order + 1, flat.size), dtype=np.complex128)
    far = np.abs(flat) > PHI_JOIN
    if np.any(far):
        out[:, far] = _phi_far(flat[far], order)
    if not np.all(far):
        out[:, ~far] = _phi_near(flat[~far], order)
    return out.reshape((order + 1, *values.shape))


def _phi_far(
    x: Complex[np.ndarray, " n_elements"], order: int
) -> Complex[np.ndarray, "n_phi n_elements"]:
    """Recur up from ``exp(x)``: stable where every ``|x|`` exceeds the join.

    Parameters
    ----------
    x : numpy.ndarray
        The arguments, each beyond :data:`PHI_JOIN` in magnitude.
    order : int
        The highest function.

    Returns
    -------
    numpy.ndarray
        ``phi_0`` to ``phi_order``, shape ``(order + 1, n_elements)``.
    """
    out = np.empty((order + 1, x.size), dtype=np.complex128)
    out[0] = np.exp(x)
    for k in range(1, order + 1):
        out[k] = (out[k - 1] - _RECIPROCAL_FACTORIAL[k - 1]) / x
    return out


def _phi_near(
    x: Complex[np.ndarray, " n_elements"], order: int
) -> Complex[np.ndarray, "n_phi n_elements"]:
    """Scale, sum the highest series, recur down and double back, near zero.

    Parameters
    ----------
    x : numpy.ndarray
        The arguments, each within :data:`PHI_JOIN` in magnitude.
    order : int
        The highest function.

    Returns
    -------
    numpy.ndarray
        ``phi_0`` to ``phi_order``, shape ``(order + 1, n_elements)``.
    """
    magnitude = np.abs(x)
    # The number of halvings bringing |x| to at most 1/2; zero at and below it.
    squarings = np.where(
        magnitude > 0.5, np.ceil(np.log2(np.maximum(magnitude, 0.5) / 0.5)), 0.0
    ).astype(np.int64)
    y = x / np.exp2(squarings)
    out = np.empty((order + 1, x.size), dtype=np.complex128)
    term = np.full(x.size, _RECIPROCAL_FACTORIAL[order], dtype=np.complex128)
    total = term.copy()
    for m in range(1, _SERIES_TERMS):
        term = term * y / (m + order)
        total = total + term
    out[order] = total
    for k in range(order - 1, 0, -1):
        out[k] = _RECIPROCAL_FACTORIAL[k] + y * out[k + 1]
    out[0] = np.exp(y)
    for level in range(int(squarings.max(initial=0))):
        active = squarings > level
        if not np.any(active):  # pragma: no cover - squarings.max bounds the loop
            break
        old = out[:, active]
        doubled = np.empty_like(old)
        doubled[0] = old[0] * old[0]
        for k in range(1, order + 1):
            total = old[0] * old[k]
            for j in range(1, k + 1):
                total = total + old[j] * _RECIPROCAL_FACTORIAL[k - j]
            doubled[k] = total / 2.0**k
        out[:, active] = doubled
    return out


def exponential_weights(
    nodes: Float[np.ndarray, " n_nodes"] | Sequence[float],
    x: Inexact[np.ndarray, "*shape"] | complex,
    upper: float = 1.0,
    *,
    helpers: Helpers = NUMBA,
) -> Complex[np.ndarray, " n_nodes *shape"]:
    """Return the exponential Adams weights of distinct nodes at each argument.

    ``w_j = int_0^upper exp((upper - theta) x) ell_j(theta) dtheta``, so that
    ``exp(upper x) u_n + h sum_j w_j F_j`` advances ``u' = D u + F`` by ``upper h``
    when ``F`` is the polynomial interpolating the samples ``F_j`` at the nodes and
    ``x = h D``.

    Parameters
    ----------
    nodes : numpy.ndarray or list of float
        Distinct normalised positions ``(z_j - z_n) / h``.
    x : numpy.ndarray or complex
        ``h D`` per element, dimensionless.
    upper : float, optional
        The fraction of the step integrated, in ``[0, 1]``.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Lagrange basis (SEP 29 section 6.6); the shared
        Numba one by default.

    Returns
    -------
    numpy.ndarray
        One weight per node along a new leading axis, each of `x`'s shape.

    Raises
    ------
    StepperError
        If the nodes are not distinct and finite, `upper` lies outside ``[0, 1]`` or an
        argument overflows.

    Examples
    --------
    At ``x = 0`` the weights are the ordinary Adams-Moulton ones:

    >>> import numpy as np
    >>> from solprop.backends.exponential import exponential_weights
    >>> (24 * exponential_weights([1, 0, -1, -2], 0.0)).real.round(12).tolist()
    [9.0, 19.0, -5.0, 1.0]
    """
    if not 0.0 <= upper <= 1.0:
        msg = f"the integrated fraction must lie in [0, 1], got {upper!r}"
        raise StepperError(msg)
    coefficients = lagrange_coefficients(nodes, helpers=helpers)
    count = coefficients.shape[0]
    arguments = np.asarray(x, dtype=np.complex128)
    phi = phi_functions(upper * arguments, count)
    # The moment of theta**k is k! upper**(k + 1) phi_{k+1}(upper x).
    scaled = np.array(
        [math.factorial(k) * upper ** (k + 1) for k in range(count)], dtype=np.float64
    )
    moments = phi[1 : count + 1] * scaled.reshape((count, *([1] * arguments.ndim)))
    return np.tensordot(coefficients, moments, axes=(1, 0))


def refuse_formulation(problem: KernelProblem, backend: str) -> None:
    """Refuse the exponential formulation where it is not yet delivered (SEP 16).

    The exponential formulation steps constant, separable and varying diagonal
    operators (phases 2--4) and the constant-Schur and frozen-residual sections of a
    coupled flow (phase 5); a CF4 section is excluded (Q3) and refused by name before
    any source is evaluated.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    backend : str
        The backend's name, for the message.

    Raises
    ------
    BackendError
        Naming what is unsupported.
    """
    method = problem.stepper.integrator
    if not isinstance(method, Multistep) or method.formulation != "exponential":
        return
    # Every representation (EXPONENTIAL_REPRESENTATIONS) is stepped since phase 4.
    flow = problem.linear.coupled_flow
    if flow is None:
        return
    methods = set(flow.section_methods)
    if "cf4" in methods:
        # SEP 16 Q3: a CF4 section has no explicit residual and exponentiates the
        # generator inside each stage; it is excluded from delivery.
        msg = (
            f"{backend}: the {method.name} integrator does not step a 'cf4' section "
            "of a coupled flow, which only the Runge-Kutta march does; pass "
            "integrator=RungeKutta()"
        )
        raise BackendError(msg)
    if not methods <= COUPLED_SECTIONS:
        unsupported = ", ".join(sorted(repr(one) for one in methods - COUPLED_SECTIONS))
        msg = (
            f"{backend}: the {method.name} integrator does not step a coupled flow's "
            f"{unsupported} sections; pass integrator=RungeKutta(), the Runge-Kutta "
            "march"
        )
        raise BackendError(msg)


@dataclass(frozen=True)
class CarrierChord:
    """The reference element's cumulative operator alone (SEP 35 sections 6.1, 6.4).

    The operator's own representation restricted to column 0, polarisation 0 and the
    reference bin, every batch member kept: its chord over a step is the reference
    element's exponent, to rounding the one the backend forms, at one scalar per
    member. The step cap reads it for every proposal, and a backend that forms its
    exponent elsewhere (the native engine, on a device too) takes its shift from it
    with nothing read back.

    Attributes
    ----------
    constant : numpy.ndarray or None
        ``D`` at the element, 1/m, shape ``(n_batch_c,)``, for a constant operator.
    expansion : solprop.translate.expansion.ParameterExpansion or None
        The expansion with its terms at the element, for an expansion.
    table : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The z table at the element, for a table.
    """

    constant: Complex[np.ndarray, " n_batch_c"] | None = None
    expansion: ParameterExpansion | None = None
    table: PiecewiseChebyshev | None = None

    def exponent(self, z1: float, z2: float) -> Complex[np.ndarray, " n_batch_c"]:
        """Return ``Phi(z2) - Phi(z1)`` at the element, per batch member.

        Parameters
        ----------
        z1, z2 : float
            Positions in m, inside the operator's span.

        Returns
        -------
        numpy.ndarray
            Dimensionless, shape ``(n_batch_c,)``.
        """
        if self.constant is not None:
            return self.constant * (z2 - z1)
        if self.expansion is not None:
            return np.asarray(self.expansion.delta(z1, z2), dtype=np.complex128)
        assert self.table is not None
        return np.asarray(self.table(z2) - self.table(z1), dtype=np.complex128)


def carrier_chord(linear: LinearOperator, index: int) -> CarrierChord | None:
    """Return the reference element's operator alone, or ``None`` for a separable one.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        The cumulative operator.
    index : int
        The reference bin.

    Returns
    -------
    CarrierChord or None
        ``None`` for the separable (spatial) representation, which the carrier frame
        refuses.
    """

    def element(values: np.ndarray) -> np.ndarray:
        """Return ``values[..., b, 0, 0, index]`` over its leading axes and batch.

        Parameters
        ----------
        values : numpy.ndarray
            A representation's array, its last four axes the state's layout.

        Returns
        -------
        numpy.ndarray
            Column 0, polarisation 0 and the reference bin, every leading axis kept.
        """
        column = min(index, values.shape[-1] - 1)
        return np.ascontiguousarray(values[..., 0, 0, column])

    if linear.phi_constant is not None:
        return CarrierChord(constant=element(linear.phi_constant))
    if linear.phi_expansion is not None:
        expansion = linear.phi_expansion
        return CarrierChord(
            expansion=dataclasses.replace(expansion, terms=element(expansion.terms))
        )
    if linear.phi_coefficients is not None:
        table = linear.polynomial
        return CarrierChord(
            table=dataclasses.replace(table, coefficients=element(table.coefficients))
        )
    return None


@dataclass(frozen=True)
class CarrierReference:
    """Where a run's carrier frame takes its shift, fixed for the run (SEP 35 6.1).

    The shift of batch member ``b`` is ``sigma_b = i Im x[b, 0, 0, bin]``, ``x`` the
    trial's own exponent ``Phi(z + h) - Phi(z)``: column 0 is the frame's mode
    (``build_phi`` fixes the frame on the first mode; a coupled route's carrier is the
    same in every channel), polarisation 0. The change of variable is exact for any
    shift, so the choice of bin only changes how smooth the interpolated source is.

    Attributes
    ----------
    policy : {'scalar'}
        The policy asked for (:data:`solprop.stepper.multistep.CarrierFrame`).
    bin : int or None
        The active bin nearest the reference frequency, ties to the lower index;
        ``None`` where the policy is inert (``sigma = 0`` for the run).
    omega : float or None
        That bin's angular frequency, rad/s; ``None`` with `bin`.
    reason : str
        How the reference was chosen, or why the policy is inert, for the run's log.
    chord : CarrierChord or None, optional
        The reference element's operator alone (:func:`carrier_chord`), for the step
        cap and a backend that forms its shift on the host; ``None`` when inert.
    """

    policy: CarrierFrame
    bin: int | None
    omega: float | None
    reason: str
    chord: CarrierChord | None = None

    @property
    def active(self) -> bool:
        """Return whether the shift can be nonzero.

        Returns
        -------
        bool
            ``False`` when the run takes ``sigma = 0``, the present arithmetic exactly.
        """
        return self.bin is not None

    def shift(
        self, exponent: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch 1 1 1"]:
        """Return each batch member's shift for a trial's exponent.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x = Phi(z + h) - Phi(z)``, in the canonical layout.

        Returns
        -------
        numpy.ndarray
            Purely imaginary, shape ``(n_batch, 1, 1, 1)``, broadcasting against the
            state.

        Raises
        ------
        BackendError
            If the policy is inert: such a run takes the unshifted arithmetic.
        """
        if self.bin is None:
            msg = f"the carrier frame is inert for this run: {self.reason}"
            raise BackendError(msg)
        # The real part (loss) is never shifted (SEP 35 section 10).
        phase = np.imag(exponent[:, 0, 0, self.bin]).reshape((-1, 1, 1, 1))
        return 1j * phase

    def phase(
        self, exponent: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> float:
        """Return the largest carrier phase over the batch, for the step cap.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``x`` over a proposed step.

        Returns
        -------
        float
            ``max_b |Im x[b, 0, 0, bin]|``, rad; 0 when the policy is inert.
        """
        if self.bin is None:
            return 0.0
        return float(np.max(np.abs(np.imag(exponent[:, 0, 0, self.bin]))))

    def chord_phase(self, z1: float, z2: float) -> float:
        """Return the largest carrier phase over the batch from the chord alone.

        Parameters
        ----------
        z1, z2 : float
            The proposed step's ends, m.

        Returns
        -------
        float
            ``max_b |Im (Phi(z2) - Phi(z1))|`` at the reference element, rad; 0 when
            the policy is inert.

        Raises
        ------
        BackendError
            If the reference has no chord (a separable operator).
        """
        if self.bin is None:
            return 0.0
        if self.chord is None:
            msg = "the carrier frame's reference has no chord on this operator"
            raise BackendError(msg)
        return float(np.max(np.abs(np.imag(self.chord.exponent(z1, z2)))))


def carrier_reference(
    problem: KernelProblem, method: Multistep
) -> CarrierReference | None:
    """Return where a run's carrier frame takes its shift, or ``None`` when off.

    The reference rule of SEP 35 section 6.1. An envelope grid without the third
    harmonic subtracts ``beta_0``, so its shift would be the table's rounding: the
    policy is inert there and the run is the present arithmetic exactly. An envelope
    grid with the third harmonic keeps the carrier phase whole, its operator is the
    field grid's, and it is treated as one. The reference frequency is the grids'
    ``omega0``; a custom frame with another reference changes only how much of the
    rotation the shift removes, never the step's exactness.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    method : solprop.stepper.multistep.Multistep
        The method.

    Returns
    -------
    CarrierReference or None
        ``None`` for ``carrier_frame="off"`` or the interaction formulation.
    """
    policy = method.carrier_frame
    if policy == "off" or method.formulation != "exponential":
        return None
    grids = problem.grids
    if problem.grid_kind == "envelope" and grids.carrier_2w0 is None:
        return CarrierReference(
            policy,
            None,
            None,
            "an envelope grid without the third harmonic removes beta_0: sigma = 0",
        )
    nearest = _nearest_bin(grids.omega, grids.omega0)
    if not bool(grids.active_mask[nearest]):
        return CarrierReference(
            policy,
            None,
            None,
            f"the bin nearest the reference frequency ({nearest}) is inactive: "
            "sigma = 0",
        )
    omega = float(grids.omega[nearest])
    return CarrierReference(
        policy,
        nearest,
        omega,
        f"reference bin {nearest} at {omega:.6e} rad/s",
        carrier_chord(problem.linear, nearest),
    )


def _nearest_bin(omega: Float[np.ndarray, " n_omega"], omega0: float) -> int:
    """Return the bin nearest `omega0`, ties going to the lower index.

    Parameters
    ----------
    omega : numpy.ndarray
        The frequency axis, rad/s, in the grid's own order.
    omega0 : float
        The reference frequency, rad/s.

    Returns
    -------
    int
        The bin.
    """
    # argmin returns the first of equal distances: the lower index of the axis's order.
    return int(np.argmin(np.abs(omega - omega0)))


def _carrier_refusal(
    policy: CarrierFrame, backend: str, delivered: frozenset[str], geometry: str | None
) -> str | None:
    """Return why `backend` cannot step a carrier frame, or ``None`` when it can.

    Parameters
    ----------
    policy : {'off', 'scalar'}
        The policy asked for.
    backend : str
        The backend's name, for the message.
    delivered : frozenset of str
        The policies the backend steps.
    geometry : str or None
        The spatial geometry's kind; ``None`` on a guided route.

    Returns
    -------
    str or None
        The refusal.
    """
    if policy == "off":
        return None
    if policy not in delivered:
        return (
            f"{backend}: Multistep(carrier_frame={policy!r}) is not delivered on this "
            "backend; pass carrier_frame='off'"
        )
    if geometry is not None:
        return (
            f"{backend}: Multistep(carrier_frame={policy!r}) is not yet qualified on "
            f"a free-space route ({geometry!r} geometry); pass carrier_frame='off'"
        )
    return None


def refuse_carrier_frame(
    problem: KernelProblem,
    backend: str,
    *,
    delivered: frozenset[str] = frozenset(),
) -> None:
    """Refuse a carrier frame where it is not yet delivered (SEP 35 section 5).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
    backend : str
        The backend's name, for the message.
    delivered : frozenset of str, optional
        The policies this backend steps; none by default.

    Raises
    ------
    BackendError
        Naming the policy and what does not deliver it.
    """
    method = problem.stepper.integrator
    if not isinstance(method, Multistep):
        return
    geometry = problem.transform.geometry
    refusal = _carrier_refusal(
        method.carrier_frame,
        backend,
        delivered,
        None if geometry is None else geometry.kind,
    )
    if refusal is not None:
        raise BackendError(refusal)
