# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The cumulative linear operator in a form a kernel can evaluate.

``D(omega, z)`` is diagonal in ``omega``, so it commutes with itself along z and
``exp(Phi(z2) - Phi(z1))`` is the *exact* linear flow over any step.
For a physical modal index, ``Phi = integral of D - log(w(z)/w(0))/2`` also
conserves true power along a gradient. This module turns
:func:`solprop.linear.dispersion_operator` into
something a kernel can evaluate, plus the hard kinks steps must land on and the frame
quantities as diagnostics. Section 3 of ``docs/dev/kernel-contract.md`` is the class
:class:`LinearOperator`, field for field, and the derivation is in
``docs/guide/numerics/linear-operator.md``.

There are three representations, and :func:`delta_phi` and :func:`unitary_phase` take
any of them, which is what lets the choice change without a backend noticing:

* a **z-constant** operator is the exact ``D (z2 - z1)``, to the last bit;
* a mode set whose whole z dependence runs through scalar paths it declares
  (:class:`solprop.modes.protocol.ParameterisedMode`) takes an **expansion in those
  parameters** with scalar integral tables along z
  (:mod:`solprop.translate.expansion`): ``D`` is analytic in the density and the core
  radius, while it is ``rho(z)`` that carries the cusps and the kinks, so the expensive
  array-valued object is resolved where it is easy and the hard structure stays in
  tables of scalars;
* anything else takes the **z table**, an adaptive piecewise Chebyshev polynomial in z
  with array-valued coefficients, integrated exactly in the Chebyshev basis. It is the
  general representation and the fallback for a mode set the expansion cannot serve.

Three things this module is careful about:

* The z-dependent operator is integrated, not applied by a rectangle rule, which would
  be globally first order in the step and invisible to the error estimate. Either
  representation is exact instead, and :func:`verify_phi` proves it against
  ``scipy.integrate.quad_vec``.
* Both cover slightly beyond the fibre end (a five per cent margin), so the last step
  and the noise phase are defined there.
* The tolerance is the measured round-off floor of ``D``, taken in the acceptance test's
  own norm so that the two are comparable; measured any other way the search is paid
  twice.
"""

import functools
import itertools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float, Int
from numpy.polynomial import chebyshev as chebyshev_basis
from scipy.integrate import quad_vec

from solprop._helpers import Helpers
from solprop.exceptions import TranslationError
from solprop.grid import TimeGrid
from solprop.linear import (
    DEFAULT_LOSS_POLICY,
    Frame,
    LossPolicy,
    dispersion_operator,
    frame_quantities,
    log_power_weight,
    power_weight,
)
from solprop.modes.protocol import Mode, ParameterisedMode
from solprop.profiles import ZProfile
from solprop.translate.chebyshev import (
    DEFAULT_MAX_DEGREE,
    DEFAULT_MAX_INTERVALS,
    DEFAULT_TOL,
    ENVELOPE_FLOOR,
    PiecewiseChebyshev,
    build,
    build_from_paths,
    clenshaw,
    lobatto_points,
    resolve_axis,
)
from solprop.translate.coupled import CoupledFlowTables, ModalMetricTables
from solprop.translate.expansion import (
    ParameterExpansion,
    append_endpoint_expansion,
    build_expansion,
    parameter_roundoff_floor,
    reweight_expansion,
)
from solprop.translate.spatial import SeparableOperator, separable_values

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_VERIFY_POINTS",
    "MAX_TOL",
    "ROUNDOFF_SAFETY",
    "LinearOperator",
    "PhiRepresentation",
    "RepresentationChoice",
    "build_phi",
    "constant_phi",
    "declared_paths",
    "delta_phi",
    "parameter_paths",
    "parameters_own_z",
    "phi_rate",
    "power_weight_at",
    "power_weight_factors",
    "power_weight_row",
    "profile_table",
    "rate_weights",
    "tabulate",
    "unitary_phase",
    "verify_phi",
    "with_power_weight",
]

logger = logging.getLogger(__name__)

DEFAULT_MARGIN = 0.05
"""Fraction of ``z_max`` the table covers beyond the fibre end.

The last step may land exactly on ``z_max`` and the noise phase is asked for there, so
evaluation at the end must not fall off the domain; five per cent costs one extra
interval.
"""

DEFAULT_VERIFY_POINTS = 8
"""Position pairs :func:`build_phi` checks against ``quad_vec`` at debug level."""

ROUNDOFF_SAFETY = 100.0
"""Margin between the measured round-off floor of ``D`` and the tolerance asked for.

The frame subtracts ``beta_0 + beta_1 (omega - origin)`` from ``beta``, and on a wide
field grid that is a cancellation of about eight digits: ``beta`` reaches 8e7 1/m at the
top of the active band while the residual is a few thousand, so ``D`` carries an
absolute round-off of order 1e-7 whatever the frame quantities cost. A hundred is the
smallest round decade that leaves margin above the floor; see
``docs/guide/numerics/linear-operator.md``.

The factor was originally what stood between a floor measured against the *global*
envelope and a tail measured *per element*, and it was not enough: with the spectral
collar active the global envelope is the collar's own rate, the floor is read two orders
too small, and the interval budget is exhausted before the tolerance is relaxed and the
whole search paid again. The floor is now measured in the
acceptance test's own norm, so the two are comparable and the factor is a margin rather
than a correction. Measured on a capillary with its absorber (1.5 m, 90 nm to 4 um,
n_omega 2049, four modes): the floor goes from 1.5e-13 to 2.3e-7 at 1 mm and the first
build from 416 s to 1.1 s, at half the memory and an agreement with ``quad_vec`` of
5.5e-11 against 2.5e-12.
"""

MAX_TOL = 1e-7
"""Loosest tolerance the budget fallback will relax to before giving up and logging."""

PhiRepresentation = Literal["constant", "expansion", "table", "separable"]
"""Which representation of ``Phi`` the contract carries.

``"constant"`` is an operator that does not vary along z, held as ``D`` itself;
``"expansion"`` is the Chebyshev expansion in the mode set's scalar paths of
:mod:`solprop.translate.expansion`, with scalar integral tables along z; ``"table"`` is
the piecewise Chebyshev table of z itself, which is the general representation.

The first two are the **device form**: each evaluates ``Phi`` as a weighted sum of a
fixed set of arrays over host scalars the translation layer computes per stage -- one
term with the distance as its weight for a constant, ``n_terms`` for an expansion -- so
a backend holds the arrays and never the representation
(:mod:`solprop.translate.stage`). The z table is a Clenshaw pass over an array of the
state's size times the degree and is evaluated on the host by NumPy and Numba;
:attr:`solprop.translate.problem.KernelProblem.device_ready` is the predicate, and a
device-resident backend refuses what fails it.
"""

RepresentationChoice = Literal["auto", "expansion", "table"]
"""What a caller may ask :func:`build_phi` for.

``"auto"``, the default, takes the expansion when the mode set allows it and falls back
to the z table otherwise. The other two force the choice, which is what an equivalence
test between the two representations needs; ``"expansion"`` raises rather than falling
back.
"""

_AGREEMENT_PROBES = 17
_FLOOR_PROBES = 3
_FLOOR_STEP_FRACTION = 1e-9

_OperatorAt = Callable[[float], Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]
_OperatorOf = Callable[
    [Mapping[str, float]], Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
]


@dataclass(frozen=True, eq=False)
class LinearOperator:
    """The cumulative linear operator ``Phi(omega, z)`` and its landing points.

    Section 3 of the kernel contract. Every array is a read-only, C-contiguous copy of
    the stated dtype. Each shape annotation carries its own axis name because
    ``__post_init__`` is what enforces their agreement, and that check must stay
    reachable in production where no runtime type checker runs.

    Attributes
    ----------
    phi_representation : {'constant', 'expansion', 'table'}
        Which representation was built. ``"constant"`` carries `phi_constant` and the
        exact single interval beside it; ``"expansion"`` carries `phi_expansion` and no
        `phi_coefficients`; ``"table"`` carries `phi_coefficients` alone. The evaluation
        functions take all three, so nothing downstream branches on this to *evaluate*;
        what does branch on it is whether the problem is in the device form
        (:attr:`solprop.translate.problem.KernelProblem.device_ready`).
    phi_breakpoints : numpy.ndarray
        Interval edges in z, m, shape ``(n_intervals + 1,)``, strictly increasing. For
        an expansion these are the edges of its scalar tables, so the local step cap and
        the schedule read the same field either way.
    phi_coefficients : numpy.ndarray or None
        Chebyshev coefficients of ``Phi``, complex128, shape ``(n_intervals, n_deg,
        n_batch_or_1, n_modes, n_pol, n_omega)``, present when `phi_representation` is
        ``"table"`` and also for ``"constant"``, whose one interval of degree 1 is
        exact; absent for ``"expansion"``. The batch axis is 1 when the members share a
        fibre. Units are dimensionless: ``Phi`` is an exponent.
    phi_degrees : numpy.ndarray
        Degree used on each interval, ``numpy.intp``, shape ``(n_intervals,)``.
        Diagnostic: the zero padding makes it unnecessary for evaluation.
    hard_kinks : numpy.ndarray
        Positions in z, m, shape ``(n_kinks,)``, where ``D`` itself jumps or kinks.
        Steps land on them; the far more numerous split edges are not tracked
        here and nothing lands on those.
    phi_constant : numpy.ndarray or None
        ``D`` itself, complex128, shape ``(n_batch_or_1, n_modes, n_pol, n_omega)``,
        present exactly when `phi_representation` is ``"constant"``, so that
        :func:`delta_phi` returns ``D (z2 - z1)`` to the last bit rather than the
        difference of two Clenshaw sums.
    phi_expansion : solprop.translate.expansion.ParameterExpansion or None
        ``Phi`` as an expansion in the mode set's scalar paths with scalar integral
        tables along z, present exactly when `phi_representation` is ``"expansion"``.
        Its breakpoints, degrees and hard kinks are the fields above, which
        ``__post_init__`` checks.
    beta0 : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The frame's ``beta_0(z)`` in 1/m as a scalar table; diagnostic, already inside
        ``Phi``. ``None`` when the operator was supplied directly rather than built from
        a mode and a frame.
    beta1 : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The frame's ``beta_1(z)`` in s/m as a scalar table; diagnostic, as above.
    separable : solprop.translate.spatial.SeparableOperator or None
        Constant spatial coefficients without full operator arrays, present exactly
        for representation ``"separable"``. Shared with the response prefactor.

    power_weight_representation : {'constant', 'expansion', 'table'} or None
        Modal power-weight representation; absent for spatial geometry.
    power_weight : solprop.translate.chebyshev.PiecewiseChebyshev or None
        Dimensionless real index table, value shape ``(n_modes, n_pol, n_omega)``.
        Present for table and constant representations.
    power_weight_expansion : solprop.translate.expansion.ParameterExpansion or None
        Dimensionless real index expansion with plain basis-product weights.
    power_weight_constant : numpy.ndarray or None
        Positive dimensionless weights, shape ``(n_modes, n_pol, n_omega)``,
        present exactly for the constant representation.

    coupled_flow : solprop.translate.coupled.CoupledFlowTables or None
        Kernel schema 12: the generator of a guide whose channels couple -- a
        coherent wall, a taper's connection, a split block, a moving guide;
        executed by all three backends, a CF4 section by ``numpy_ref`` alone.
        ``None`` on every route that declares no coupling.
    modal_metric : solprop.translate.coupled.ModalMetricTables or None
        Kernel schema 12: the explicit physical flux metric, a documented leaf here
        and not a nested record, so that the controller, the flow and the output
        layer share one W by identity. ``None`` for the legacy orthogonal routes,
        whose power weight is the row above.

    Raises
    ------
    TranslationError
        If the interval counts disagree, a breakpoint is not finite and increasing, a
        hard kink is not an interval edge, or the member present is not the one
        `phi_representation` names.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, delta_phi
    >>> table = constant_phi(np.full(4, -1j), 2.0)
    >>> table.phi_representation, table.phi_degrees.tolist()
    ('constant', [1])
    >>> bool(np.array_equal(delta_phi(table, 0.0, 0.5), np.full((1, 1, 1, 4), -0.5j)))
    True
    """

    phi_representation: PhiRepresentation
    phi_breakpoints: Float[np.ndarray, " n_edges"]
    phi_coefficients: (
        Complex[np.ndarray, "n_intervals n_deg n_batch n_modes n_pol n_omega"] | None
    )
    phi_degrees: Int[np.ndarray, " n_degrees"]
    hard_kinks: Float[np.ndarray, " n_kinks"]
    phi_constant: Complex[np.ndarray, "n_batch_c n_modes_c n_pol_c n_omega_c"] | None
    phi_expansion: ParameterExpansion | None = field(default=None)
    beta0: PiecewiseChebyshev | None = field(default=None)
    beta1: PiecewiseChebyshev | None = field(default=None)
    separable: SeparableOperator | None = field(default=None)
    power_weight_representation: Literal["constant", "expansion", "table"] | None = None
    power_weight: PiecewiseChebyshev | None = None
    power_weight_expansion: ParameterExpansion | None = None
    power_weight_constant: Float[np.ndarray, "n_modes_w n_pol_w n_omega_w"] | None = (
        None
    )
    coupled_flow: CoupledFlowTables | None = None
    modal_metric: ModalMetricTables | None = None

    def __post_init__(self) -> None:
        """Adopt read-only copies and check what the shape annotations cannot.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        for name, dtype in (
            ("phi_breakpoints", np.float64),
            ("phi_degrees", np.intp),
            ("hard_kinks", np.float64),
        ):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in ("phi_coefficients", "phi_constant"):
            optional = getattr(self, name)
            if optional is None:
                continue
            frozen = np.ascontiguousarray(optional, dtype=np.complex128)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if self.power_weight_constant is not None:
            object.__setattr__(
                self,
                "power_weight_constant",
                _readonly(
                    np.ascontiguousarray(self.power_weight_constant, dtype=np.float64)
                ),
            )
        self._check_representation()
        n_intervals = self.phi_degrees.size
        if self.phi_breakpoints.size != n_intervals + 1:
            msg = (
                f"phi_breakpoints has {self.phi_breakpoints.size} edges for "
                f"{n_intervals} intervals, expected {n_intervals + 1}"
            )
            raise TranslationError(msg)
        if self.phi_coefficients is not None and (
            self.phi_coefficients.shape[0] != n_intervals
        ):
            msg = (
                f"phi_degrees has {n_intervals} entries for "
                f"{self.phi_coefficients.shape[0]} intervals"
            )
            raise TranslationError(msg)
        if self.phi_constant is not None and (
            self.phi_constant.shape != self.value_shape
        ):
            msg = (
                f"phi_constant has shape {self.phi_constant.shape}, expected "
                f"{self.value_shape} to match the operator"
            )
            raise TranslationError(msg)
        # PiecewiseChebyshev validates the breakpoints, the dtype and that every hard
        # kink is an interior edge; building it here means that never has to be repeated
        # and its Clenshaw evaluation is reused rather than written a second time. An
        # expansion's scalar tables are already a validated PiecewiseChebyshev, and
        # `_check_representation` has checked that these arrays are theirs.
        if self.phi_coefficients is not None:
            _ = self.polynomial

    def _check_representation(self) -> None:
        """Check that the stored representation is the one the kind names.

        A kind that is neither is caught by the two presence checks together and named
        by their message, so it needs no membership test of its own -- which could not
        be reached anyway from a constructor whose annotation is a literal.

        Raises
        ------
        TranslationError
            If the wrong member is present for the kind named, or an expansion's scalar
            tables disagree with the breakpoints, degrees and kinks beside them.
        """
        expansion = self.phi_expansion
        if (self.separable is not None) != (self.phi_representation == "separable"):
            msg = "separable must be present exactly for a separable linear operator"
            raise TranslationError(msg)
        if self.separable is not None and (
            not np.array_equal(self.phi_breakpoints, self.separable.z_span)
            or not np.array_equal(self.phi_degrees, [1])
            or self.hard_kinks.size != 0
        ):
            msg = (
                "a separable operator has one linear interval over its z_span "
                "and no kinks"
            )
            raise TranslationError(msg)
        if (expansion is not None) != (self.phi_representation == "expansion"):
            msg = (
                "phi_expansion must be present exactly when phi_representation is "
                f"'expansion', got {self.phi_representation!r} with "
                f"phi_expansion={'an expansion' if expansion is not None else None}"
            )
            raise TranslationError(msg)
        constant = self.phi_constant is not None
        if constant != (self.phi_representation == "constant"):
            msg = (
                "phi_constant must be present exactly when phi_representation is "
                f"'constant', got {self.phi_representation!r} with "
                f"phi_constant={'an array' if constant else None}"
            )
            raise TranslationError(msg)
        # A constant carries its exact one interval of degree 1 as well as the array, so
        # that the breakpoints, the local step cap and the schedule read the same fields
        # for every kind; only an expansion replaces the coefficients outright.
        wants_coefficients = self.phi_representation not in ("expansion", "separable")
        if (self.phi_coefficients is not None) != wants_coefficients:
            msg = (
                "phi_coefficients must be present for a 'table' and for a 'constant' "
                "and absent for expansion/separable, got "
                f"{self.phi_representation!r} with "
                "phi_coefficients="
                f"{'an array' if self.phi_coefficients is not None else None}"
            )
            raise TranslationError(msg)
        if expansion is None:
            return
        weights = expansion.weights
        for name, mine, theirs in (
            ("phi_breakpoints", self.phi_breakpoints, weights.breakpoints),
            ("phi_degrees", self.phi_degrees, weights.degrees),
            ("hard_kinks", self.hard_kinks, weights.hard_kinks),
        ):
            if not np.array_equal(mine, theirs):
                msg = (
                    f"{name} must be the expansion's own {name.split('_')[-1]}: the z "
                    "structure of Phi is that of its scalar tables and nothing else"
                )
                raise TranslationError(msg)

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the shape of ``Phi`` at one position.

        Returns
        -------
        tuple of int
            ``(n_batch_or_1, n_modes, n_pol, n_omega)``, whichever representation
            carries it.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> constant_phi(np.full(4, -1j), 2.0).value_shape
        (1, 1, 1, 4)
        """
        if self.separable is not None:
            return self.separable.value_shape
        if self.phi_expansion is not None:
            return self.phi_expansion.value_shape
        # `_check_representation` has already made the narrowing true; a cast rather
        # than an assertion, which `python -O` strips.
        return tuple(cast("np.ndarray", self.phi_coefficients).shape[2:])

    @functools.cached_property
    def polynomial(self) -> PiecewiseChebyshev:
        """Return the z table as a piecewise Chebyshev polynomial.

        A view over the stored arrays, not a copy, so evaluation costs one Clenshaw pass
        and no memory. Only a ``"table"`` representation has one; an expansion's
        equivalent is ``phi_expansion.weights``, whose value is ``n_terms`` scalars.

        Returns
        -------
        solprop.translate.chebyshev.PiecewiseChebyshev
            ``Phi`` over ``[0, z_max (1 + margin)]``.

        Raises
        ------
        TranslationError
            If the representation is an expansion.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> table = constant_phi(np.full(4, -1j), 2.0)
        >>> table.polynomial.z_span
        (0.0, 2.1)
        """
        if self.phi_coefficients is None:
            msg = (
                "this operator is a parameter expansion and has no z table; evaluate "
                "it through delta_phi and unitary_phase, as a backend does"
            )
            raise TranslationError(msg)
        return PiecewiseChebyshev(
            self.phi_coefficients,
            self.phi_breakpoints,
            self.phi_degrees,
            self.hard_kinks,
        )

    @functools.cached_property
    def rate(self) -> PiecewiseChebyshev:
        """Return the derivative of the cumulative operator's scalar or z table.

        ``D = dPhi/dz`` from the representation that defines the flow, so that a
        residual ``(D(z) - B) u`` integrates to zero against the chord ``B`` exactly
        as the table does (SEP 16 section 4.2): an expansion's scalar integral tables
        differentiate to its resolved integrands, a z table to its resolved ``D``.

        Returns
        -------
        solprop.translate.chebyshev.PiecewiseChebyshev
            The derivative of ``phi_expansion.weights`` (value shape
            ``(n_terms,)``) for an expansion, of :attr:`polynomial` otherwise.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> table = constant_phi(np.full(2, -1j), 2.0)
        >>> np.round(table.rate(0.5), 12).ravel().tolist()
        [-1j, -1j]
        """
        if self.phi_expansion is not None:
            return self.phi_expansion.weights.derivative()
        return self.polynomial.derivative()

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the range of z the table covers.

        Returns
        -------
        tuple of float
            ``(first breakpoint, last breakpoint)`` in m.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> constant_phi(np.full(4, -1j), 2.0).z_span
        (0.0, 2.1)
        """
        return float(self.phi_breakpoints[0]), float(self.phi_breakpoints[-1])

    @property
    def n_intervals(self) -> int:
        """Return the number of Chebyshev intervals.

        Returns
        -------
        int
            The length of ``phi_degrees``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> constant_phi(np.full(4, -1j), 2.0).n_intervals
        1
        """
        return int(self.phi_degrees.size)

    @property
    def nbytes(self) -> int:
        """Return the memory the representation occupies.

        Returns
        -------
        int
            Bytes: the coefficient array for a z table, the expansion's terms and
            scalar tables together for an expansion. This is the number
            ``docs/dev/kernel-contract.md`` section 3 quotes.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.linop import constant_phi
        >>> constant_phi(np.full(4, -1j), 2.0).nbytes
        128
        """
        if self.phi_expansion is not None:
            return self.phi_expansion.nbytes
        if self.separable is not None:
            return sum(
                value.nbytes
                for item in fields(self.separable)
                if isinstance(value := getattr(self.separable, item.name), np.ndarray)
            )
        return int(cast("np.ndarray", self.phi_coefficients).nbytes)


def _value_shape(
    operator: Complex[np.ndarray, "*value n_omega"], n_pol: int
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Bring an operator to the canonical ``(n_batch, n_modes, n_pol, n_omega)`` shape.

    Parameters
    ----------
    operator : numpy.ndarray
        ``D`` as ``(n_omega,)`` or already in the canonical layout.
    n_pol : int
        Polarisation components to broadcast a one-dimensional operator over.

    Returns
    -------
    numpy.ndarray
        The operator with four axes.

    Raises
    ------
    TranslationError
        If the array has neither one nor four axes, or `n_pol` is not 1 or 2.
    """
    values = np.asarray(operator)
    if values.ndim == 4:
        return values
    if values.ndim != 1:
        msg = (
            f"the operator must have shape (n_omega,) or (n_batch, n_modes, n_pol, "
            f"n_omega), got {values.shape}"
        )
        raise TranslationError(msg)
    if n_pol not in (1, 2):
        msg = f"n_pol must be 1 or 2 (conventions.md section 5), got {n_pol!r}"
        raise TranslationError(msg)
    return np.broadcast_to(values, (1, 1, n_pol, values.size))


def tabulate(
    func: Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]],
    z0: float,
    z1: float,
    *,
    kinks: Sequence[float] | np.ndarray = (),
    tol: float = DEFAULT_TOL,
) -> PiecewiseChebyshev:
    """Tabulate a scalar function of z as a frozen Chebyshev table.

    The seam's representation of anything that varies along z. Declared kinks outside
    ``[z0, z1]`` are dropped rather than refused, because a path is asked for slightly
    beyond the fibre end (the margin of :data:`DEFAULT_MARGIN`) where it holds its last
    value.

    Parameters
    ----------
    func : callable
        ``z -> value`` over arrays of positions in m.
    z0, z1 : float
        Ends of the range in m, with ``z1 > z0``.
    kinks : array_like, optional
        Positions that must be interval edges; those outside the range are dropped.
    tol : float, optional
        Target bound on the uniform error relative to the value scale.

    Returns
    -------
    solprop.translate.chebyshev.PiecewiseChebyshev
        The table, with read-only, C-contiguous arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import tabulate
    >>> table = tabulate(lambda z: 2.0 * z, 0.0, 1.0)
    >>> float(table(0.5)), table.coefficients.flags.writeable
    (1.0, False)
    """
    edges = sorted(
        {
            float(value)
            for value in np.atleast_1d(np.asarray(kinks, dtype=np.float64))
            if z0 < float(value) < z1
        }
    )
    table = build(func, z0, z1, tol=tol, mandatory_breakpoints=edges)
    return PiecewiseChebyshev(
        _readonly(table.coefficients),
        _readonly(table.breakpoints),
        _readonly(table.degrees),
        _readonly(table.hard_kinks),
        table.budget_exceeded,
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, read-only copy of an array.

    Parameters
    ----------
    values : numpy.ndarray
        The array to freeze.

    Returns
    -------
    numpy.ndarray
        The same values, C-contiguous and not writeable.
    """
    frozen = np.ascontiguousarray(values)
    frozen.setflags(write=False)
    return frozen


def profile_table(
    profile: ZProfile, z0: float, z1: float, *, tol: float = DEFAULT_TOL
) -> PiecewiseChebyshev:
    """Tabulate a scalar path along z in the form the seam carries.

    A ``ZProfile`` is a Python object with a ``__call__``; a backend cannot evaluate one
    inside a compiled loop. The seam's representation of a path is therefore a
    scalar-valued :class:`solprop.translate.chebyshev.PiecewiseChebyshev` over the same
    z range as the cumulative operator — plain arrays, one Clenshaw pass, and the
    profile's own breakpoints as mandatory edges so a stage boundary is never crossed.
    This is what "``ZProfile`` table" means in ``docs/dev/kernel-contract.md``.

    Where the path is exactly zero at an interval's edge -- a declared vacuum, such as
    the end of a pressure profile that falls to 0 bar -- the table is exactly zero
    there too. The fit alone misses that zero by a few ulp of the value scale, of
    either sign -- its end samples are inset by one ulp and its coefficients are
    rounded -- and a density below zero is refused by the problem's domain check. The
    correction moves only those ends, and only while it changes the interval by at most
    `tol`, so a path that is negative anywhere stays negative in its table and is still
    refused. A path that leaves zero too steeply for its inset sample to be round-off
    (a square-root vacuum end) keeps the table the fit gave.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        The path: a density, a core radius, an effective area, a taper rescaling.
    z0, z1 : float
        Ends of the range in m, with ``z1 > z0``. Outside its own span every profile in
        :mod:`solprop.profiles` holds its end value, so the margin beyond the fibre is
        well defined.
    tol : float, optional
        Target bound on the uniform error relative to the value scale.

    Returns
    -------
    solprop.translate.chebyshev.PiecewiseChebyshev
        The path as a scalar table, with its interior breakpoints as hard kinks.

    Examples
    --------
    >>> from solprop.profiles import Linear
    >>> from solprop.translate.linop import profile_table
    >>> table = profile_table(Linear(1.0, 3.0, 2.0), 0.0, 2.0)
    >>> float(table(1.0))
    2.0
    """
    table = tabulate(profile.__call__, z0, z1, kinks=profile.breakpoints, tol=tol)
    vacuum = np.asarray(profile(table.breakpoints)) == 0.0
    if not np.any(vacuum):
        return table
    coefficients = table.coefficients.copy()
    for index, degree in enumerate(table.degrees):
        # A degree-0 interval with a zero edge is either exactly zero already or a
        # constant that is not, which no correction within tol could make zero.
        if degree >= 1 and (vacuum[index] or vacuum[index + 1]):
            row = coefficients[index, : degree + 1]
            row[:] = _zero_edges(row, bool(vacuum[index]), bool(vacuum[index + 1]), tol)
    if np.array_equal(coefficients, table.coefficients):
        return table
    return replace(table, coefficients=_readonly(coefficients))


def _zero_edges(
    row: Float[np.ndarray, " n_deg"], left: bool, right: bool, tol: float
) -> Float[np.ndarray, " n_deg"]:
    """Return one interval's coefficients with its chosen ends evaluating to exactly 0.

    The ends are ``x = -1`` (`left`) and ``x = 1`` (`right`), evaluated as
    :func:`numpy.polynomial.chebyshev.chebval` and every backend's Clenshaw pass
    evaluate them. The coefficients are returned unchanged when those ends are zero
    already, or when the correction would change the interval by more than `tol` of its
    largest coefficient: that is not rounding, and the domain check decides.

    Parameters
    ----------
    row : numpy.ndarray
        The interval's Chebyshev coefficients up to its degree, at least two.
    left, right : bool
        Whether the path is exactly zero at the interval's left or right end.
    tol : float
        The table's tolerance, relative to the interval's largest coefficient.

    Returns
    -------
    numpy.ndarray
        The corrected coefficients, or `row` itself when nothing is corrected.
    """
    ends = chebyshev_basis.chebval(np.array([-1.0, 1.0]), row)
    if (not left or ends[0] == 0.0) and (not right or ends[1] == 0.0):
        return row
    # At x = +-1 the Clenshaw recurrence only adds, subtracts and doubles, and each of
    # its intermediates is at most 3 R, R = sum_j (j + 1) |c_j| (b_k is the sum of
    # (j - k + 1) c_j there). Rounded to multiples of a power of two `quantum` of at
    # least 2**-50 R, every intermediate, after the corrections below as well, is a
    # multiple of quantum / 2 smaller than 2**53 of them. The recurrence is then exact,
    # the ends are the exact sums c_0 -+ c_1 + c_2 -+ ..., and half-quantum
    # corrections to c_0 and c_1, linear in x, zero the chosen ends exactly. The other
    # end moves only by the rounding to the quantum.
    reach = float(np.sum((np.arange(row.size) + 1.0) * np.abs(row)))
    quantum = 2.0 ** (math.ceil(math.log2(8.0 * reach)) - 53)
    pinned = np.round(row / quantum) * quantum
    at_left, at_right = chebyshev_basis.chebval(np.array([-1.0, 1.0]), pinned)
    lower = at_left if left else 0.0
    upper = at_right if right else 0.0
    pinned[0] -= 0.5 * (upper + lower)
    pinned[1] -= 0.5 * (upper - lower)
    if float(np.sum(np.abs(pinned - row))) > tol * float(np.max(np.abs(row))):
        return row
    return pinned


def constant_phi(  # noqa: PLR0913 - the response kinks join the frame record
    operator: Complex[np.ndarray, "*value n_omega"],
    z_max: float,
    *,
    n_pol: int = 1,
    margin: float = DEFAULT_MARGIN,
    beta0: PiecewiseChebyshev | None = None,
    beta1: PiecewiseChebyshev | None = None,
    kinks: Sequence[float] | np.ndarray = (),
) -> LinearOperator:
    """Return the exact table of an operator that does not vary along z.

    ``Phi(z) = D z`` is a polynomial of degree one, so a single interval represents it
    exactly and no adaptive search is needed. The operator itself is kept in
    ``phi_constant`` so that :func:`delta_phi` returns ``D (z2 - z1)`` to the last bit
    rather than the difference of two Clenshaw sums — the linear flow of a constant
    operator is then bitwise reproducible, which is the property the reference backend's
    P0 test rests on.

    Something other than ``D`` may still kink along z -- a response's pre-ionisation
    path -- and steps land only on the hard kinks of ``Phi``. Such positions
    are passed as `kinks`: the one interval is then split there into exact degree-1
    pieces that carry them as hard kinks, while ``phi_constant`` and therefore
    :func:`delta_phi` are unchanged.

    Parameters
    ----------
    operator : numpy.ndarray
        ``D(omega)`` in 1/m, complex, shape ``(n_omega,)`` or ``(n_batch, n_modes,
        n_pol, n_omega)``.
    z_max : float
        Length the run covers, m, positive and finite.
    n_pol : int, optional
        Polarisation components to broadcast a one-dimensional operator over, 1 or 2.
    margin : float, optional
        Fraction of `z_max` the table extends beyond the end. Default
        :data:`DEFAULT_MARGIN`.
    beta0, beta1 : solprop.translate.chebyshev.PiecewiseChebyshev or None, optional
        The frame quantities as scalar tables, for the record. ``None`` when the
        operator was supplied without a mode and a frame.
    kinks : sequence of float or numpy.ndarray, optional
        Positions in m where steps must land although ``D`` does not change there.
        Those strictly inside the table become interval edges and hard kinks; none
        (the default) gives the single interval.

    Returns
    -------
    LinearOperator
        Representation ``"constant"``: one interval of degree 1 per stretch between
        kinks, and ``D`` itself.

    Raises
    ------
    TranslationError
        If `z_max` or `margin` is invalid, or the operator has an unusable shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi
    >>> table = constant_phi(np.array([0.0, -1j, -2j]), 1.0)
    >>> table.phi_coefficients.shape, table.z_span
    ((1, 2, 1, 1, 1, 3), (0.0, 1.05))
    >>> table.phi_representation
    'constant'
    """
    span = _checked_span(z_max, margin)
    values = np.asarray(_value_shape(operator, n_pol), dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        msg = (
            f"the operator must be finite; {int(np.sum(~np.isfinite(values)))} of "
            f"{values.size} entries are not"
        )
        raise TranslationError(msg)
    interior = np.unique(
        np.asarray([k for k in np.ravel(kinks) if 0.0 < float(k) < span], dtype=float)
    )
    if interior.size == 0:
        # Phi(z) = D z on [0, span] is c0 T0 + c1 T1 with c0 = c1 = D span / 2, since
        # z = span (x + 1) / 2 in the interval's own coordinate.
        half = 0.5 * span
        coefficients = np.stack([values * half, values * half])[np.newaxis]
        edges = np.array([0.0, span])
    else:
        # On [a, b], z = (a + b)/2 + (b - a) x / 2, so Phi = D z has c0 = D (a + b)/2
        # and c1 = D (b - a)/2: still exact, and continuous across every edge.
        edges = np.concatenate(([0.0], interior, [span]))
        coefficients = np.stack(
            [
                np.stack([values * (0.5 * (lo + hi)), values * (0.5 * (hi - lo))])
                for lo, hi in itertools.pairwise(edges)
            ]
        )
    return LinearOperator(
        phi_representation="constant",
        phi_breakpoints=edges,
        phi_coefficients=coefficients,
        phi_degrees=np.ones(edges.size - 1, dtype=np.intp),
        hard_kinks=interior,
        phi_constant=values,
        beta0=beta0,
        beta1=beta1,
    )


def _checked_span(z_max: float, margin: float) -> float:
    """Return the end of the tabulated range, validating both arguments.

    Parameters
    ----------
    z_max : float
        Length the run covers, m.
    margin : float
        Fraction of `z_max` to extend beyond it.

    Returns
    -------
    float
        ``z_max (1 + margin)``.

    Raises
    ------
    TranslationError
        If `z_max` is not positive and finite or `margin` is negative or not finite.
    """
    if not (math.isfinite(z_max) and z_max > 0.0):
        msg = f"z_max must be positive and finite, got {z_max!r}"
        raise TranslationError(msg)
    if not (math.isfinite(margin) and margin >= 0.0):
        msg = f"margin must be non-negative and finite, got {margin!r}"
        raise TranslationError(msg)
    return z_max * (1.0 + margin)


def _as_modes(mode: Mode | Sequence[Mode]) -> tuple[Mode, ...]:
    """Return the modes of a problem as a tuple, accepting one mode or a set.

    Parameters
    ----------
    mode : Mode or sequence of Mode
        One mode, or the ordered set of a multimode problem.

    Returns
    -------
    tuple of Mode
        The modes, in order.

    Raises
    ------
    TranslationError
        If an empty set is given.
    """
    modes = tuple(mode) if isinstance(mode, Sequence) else (mode,)
    if not modes:
        msg = "a problem needs at least one mode"
        raise TranslationError(msg)
    return modes


def build_phi(  # noqa: PLR0913 - the documented builder takes every input by name
    mode: Mode | Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    *,
    z_max: float,
    n_pol: int = 1,
    absorber_rate: Float[np.ndarray, " n_omega"] | None = None,
    loss: LossPolicy = DEFAULT_LOSS_POLICY,
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
    n_max: int = DEFAULT_MAX_DEGREE,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
    representation: RepresentationChoice = "auto",
    reuse: ParameterExpansion | None = None,
    index: float | None = None,
    kinks: Sequence[float] | np.ndarray = (),
) -> LinearOperator:
    """Build the cumulative linear operator of a mode set over a run.

    Three representations, in the order they are tried.

    A mode set that reports ``z_constant`` takes :func:`constant_phi`: one interval,
    degree 1, and a :func:`delta_phi` that is exact to the last bit.

    A mode set whose whole z dependence runs through scalar paths it declares
    (:class:`solprop.modes.protocol.ParameterisedMode`) takes the **expansion in those
    parameters**: ``D`` is resolved once over the range each path spans, and ``Phi`` is
    that expansion against scalar integral tables along z
    (:mod:`solprop.translate.expansion`). The paths' breakpoints are mandatory edges of
    the scalar tables, so a stage boundary or a taper corner is never crossed and
    becomes a hard kink, exactly as before; what changes is that the cusps and the
    corners are resolved in tables of ``n_terms`` scalars rather than of the state.

    Anything else takes the **z table**: the scalar paths are resolved first, because
    they are cheap and carry the structure that makes the operator hard, and the
    operator itself is then resolved on the union of their edges and integrated exactly
    in the Chebyshev basis. That is the general representation and the fallback for a
    mode whose parameters do not resolve. The approximations in ``Phi`` are the
    interpolation of ``D`` and of the cumulative logarithmic power weight, each
    resolved to its measured round-off floor
    (``docs/guide/numerics/linear-operator.md``).

    Parameters
    ----------
    mode : Mode or sequence of Mode
        The mode, or the ordered set of a multimode problem. Only ``neff``,
        ``dneff_domega``, the declared ``core_radius`` and ``density`` paths and, for
        the expansion, ``at_parameters`` are used.
    grid : solprop.grid.TimeGrid
        The grid; supplies the frequency axis, the active band and the layout.
    frame : solprop.linear.Frame
        The co-moving frame, usually from :meth:`solprop.linear.Frame.comoving`.
    z_max : float
        Length the run covers, m, positive and finite.
    n_pol : int, optional
        Polarisation components, 1 or 2. ``D`` is the same for both components of one
        mode, and is stored for each so that a birefringent mode needs no new layout.
    absorber_rate : numpy.ndarray or None, optional
        Spectral absorber rate in 1/m, shape ``(n_omega,)``, folded into ``D`` before
        the integration so the collar is exact as a rate.
    loss : solprop.linear.LossPolicy, optional
        What to do with an attenuation too large to represent. Applied to ``D`` before
        ``Phi`` is integrated, never to an accumulated exponent afterwards.
    margin : float, optional
        Fraction of `z_max` the table extends beyond the end. Default
        :data:`DEFAULT_MARGIN`.
    tol : float, optional
        Target bound on the uniform error of the table relative to the value scale. The
        request is raised to :data:`ROUNDOFF_SAFETY` times the measured round-off floor
        of ``D`` when that is larger, because no interpolant can resolve a function
        below the noise of its own evaluation, and relaxed further, by decades up to
        :data:`MAX_TOL`, if the interval budget is exhausted anyway
        (``docs/dev/stepper.md`` section 4, step 4). Both are logged.
    n_max : int, optional
        Highest Chebyshev degree tried on an interval, or on a parameter, before the
        interval is bisected or the expansion gives up.
    max_intervals : int, optional
        Interval budget; exceeding it sets the fallback flag and logs a warning.
    representation : {'auto', 'expansion', 'table'}, optional
        Which representation to build. ``"auto"``, the default, prefers the expansion;
        ``"table"`` forces the z table, which is what an equivalence test between the
        two needs; ``"expansion"`` refuses to fall back. A ``z_constant`` mode set takes
        :func:`constant_phi` whatever this says, since that path is exact.
    reuse : ParameterExpansion or None, optional
        Coefficients of the identical operator function on the same grid, frame and
        fixed parameters. The caller must establish function identity. Covered paths
        receive new scalar weights; other paths rebuild normally, with a logged reason.
    index : float or None, optional
        Power-weight override matching the nonlinear prefactor. None uses the
        mode's real effective index; a positive finite float makes the cumulative
        normalisation term identically zero. The GNLSE route passes this through
        ``assemble(prefactor_index=1.0)`` and retains its bitwise Phi.
    kinks : sequence of float or numpy.ndarray, optional
        Further positions in m where steps must land, beside the mode set's own
        breakpoints: where a response's path kinks although the modes do not (a
        pre-ionisation path). They become interval edges and hard kinks of
        every representation; none (the default) builds exactly what the modes alone
        declare.

    Returns
    -------
    LinearOperator
        Section 3 of the kernel contract, covering ``[0, z_max (1 + margin)]``.

    Raises
    ------
    TranslationError
        If `z_max`, `margin`, `representation` or the mode set is invalid, or
        ``representation="expansion"`` is asked for and cannot be built.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.linear import Frame
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.translate.linop import build_phi
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0, 0.0, 2e-26], omega_ref=grid.omega0
    ... )
    >>> table = build_phi(mode, grid, Frame.comoving(grid), z_max=1.0)
    >>> table.phi_representation, table.n_intervals
    ('constant', 1)
    """
    if index is not None and (not math.isfinite(index) or index <= 0):
        msg = "index must be positive and finite"
        raise TranslationError(msg)
    if reuse is not None and (reuse.endpoint_degrees is not None) != (index is None):
        msg = "Phi expansion reuse refused: index override differs"
        raise TranslationError(msg)
    _check_dispersion_span(_as_modes(mode), z_max)
    linear = _dispersion_phi(
        mode,
        grid,
        frame,
        z_max=z_max,
        n_pol=n_pol,
        absorber_rate=absorber_rate,
        loss=loss,
        margin=margin,
        tol=tol,
        n_max=n_max,
        max_intervals=max_intervals,
        representation=representation,
        reuse=reuse,
        kinks=kinks,
    )
    return _power_phi(linear, _as_modes(mode), grid, index)


def _dispersion_phi(  # noqa: PLR0913 - the dispersion builder shares the public controls
    mode: Mode | Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    *,
    z_max: float,
    n_pol: int = 1,
    absorber_rate: Float[np.ndarray, " n_omega"] | None = None,
    loss: LossPolicy = DEFAULT_LOSS_POLICY,
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
    n_max: int = DEFAULT_MAX_DEGREE,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
    representation: RepresentationChoice = "auto",
    reuse: ParameterExpansion | None = None,
    kinks: Sequence[float] | np.ndarray = (),
) -> LinearOperator:
    """Build the cumulative linear operator of a mode set over a run.

    Three representations, in the order they are tried.

    A mode set that reports ``z_constant`` takes :func:`constant_phi`: one interval,
    degree 1, and a :func:`delta_phi` that is exact to the last bit.

    A mode set whose whole z dependence runs through scalar paths it declares
    (:class:`solprop.modes.protocol.ParameterisedMode`) takes the **expansion in those
    parameters**: ``D`` is resolved once over the range each path spans, and ``Phi`` is
    that expansion against scalar integral tables along z
    (:mod:`solprop.translate.expansion`). The paths' breakpoints are mandatory edges of
    the scalar tables, so a stage boundary or a taper corner is never crossed and
    becomes a hard kink, exactly as before; what changes is that the cusps and the
    corners are resolved in tables of ``n_terms`` scalars rather than of the state.

    Anything else takes the **z table**: the scalar paths are resolved first, because
    they are cheap and carry the structure that makes the operator hard, and the
    operator itself is then resolved on the union of their edges and integrated exactly
    in the Chebyshev basis. That is the general representation and the fallback for a
    mode whose parameters do not resolve; the only approximation anywhere in ``Phi`` is
    the interpolation of ``D`` (``docs/guide/numerics/linear-operator.md``).

    Parameters
    ----------
    mode : Mode or sequence of Mode
        The mode, or the ordered set of a multimode problem. Only ``neff``,
        ``dneff_domega``, the declared ``core_radius`` and ``density`` paths and, for
        the expansion, ``at_parameters`` are used.
    grid : solprop.grid.TimeGrid
        The grid; supplies the frequency axis, the active band and the layout.
    frame : solprop.linear.Frame
        The co-moving frame, usually from :meth:`solprop.linear.Frame.comoving`.
    z_max : float
        Length the run covers, m, positive and finite.
    n_pol : int, optional
        Polarisation components, 1 or 2. ``D`` is the same for both components of one
        mode, and is stored for each so that a birefringent mode needs no new layout.
    absorber_rate : numpy.ndarray or None, optional
        Spectral absorber rate in 1/m, shape ``(n_omega,)``, folded into ``D`` before
        the integration so the collar is exact as a rate.
    loss : solprop.linear.LossPolicy, optional
        What to do with an attenuation too large to represent. Applied to ``D`` before
        ``Phi`` is integrated, never to an accumulated exponent afterwards.
    margin : float, optional
        Fraction of `z_max` the table extends beyond the end. Default
        :data:`DEFAULT_MARGIN`.
    tol : float, optional
        Target bound on the uniform error of the table relative to the value scale. The
        request is raised to :data:`ROUNDOFF_SAFETY` times the measured round-off floor
        of ``D`` when that is larger, because no interpolant can resolve a function
        below the noise of its own evaluation, and relaxed further, by decades up to
        :data:`MAX_TOL`, if the interval budget is exhausted anyway
        (``docs/dev/stepper.md`` section 4, step 4). Both are logged.
    n_max : int, optional
        Highest Chebyshev degree tried on an interval, or on a parameter, before the
        interval is bisected or the expansion gives up.
    max_intervals : int, optional
        Interval budget; exceeding it sets the fallback flag and logs a warning.
    representation : {'auto', 'expansion', 'table'}, optional
        Which representation to build. ``"auto"``, the default, prefers the expansion;
        ``"table"`` forces the z table, which is what an equivalence test between the
        two needs; ``"expansion"`` refuses to fall back. A ``z_constant`` mode set takes
        :func:`constant_phi` whatever this says, since that path is exact.
    reuse : ParameterExpansion or None, optional
        Coefficients of the identical operator function on the same grid, frame and
        fixed parameters. The caller must establish function identity. Covered paths
        receive new scalar weights; other paths rebuild normally, with a logged reason.
    kinks : sequence of float or numpy.ndarray, optional
        Further hard kinks in m beside the mode set's breakpoints, as for
        :func:`build_phi`.

    Returns
    -------
    LinearOperator
        Section 3 of the kernel contract, covering ``[0, z_max (1 + margin)]``.

    Raises
    ------
    TranslationError
        If `z_max`, `margin`, `representation` or the mode set is invalid, or
        ``representation="expansion"`` is asked for and cannot be built.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.linear import Frame
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> from solprop.translate.linop import build_phi
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    >>> mode = SpecifiedMode.from_taylor(
    ...     [0.0, 0.0, 2e-26], omega_ref=grid.omega0
    ... )
    >>> table = build_phi(mode, grid, Frame.comoving(grid), z_max=1.0)
    >>> table.phi_representation, table.n_intervals
    ('constant', 1)
    """
    _checked_representation(representation)
    span = _checked_span(z_max, margin)
    modes = _as_modes(mode)
    operator_at = _operator_sampler(modes, grid, frame, n_pol, loss, absorber_rate)
    paths = declared_paths(modes)
    edges = _interior_edges(paths, span)
    beta0, beta1 = _frame_tables(modes[0], frame, span, tol, edges)
    # The modes' own edges alone set the frame record; the further kinks only split
    # the tables steps are placed by, and are absent unless a response declared them.
    extra = [float(k) for k in np.ravel(kinks) if 0.0 < float(k) < span]
    if extra:
        edges = sorted({*edges, *extra})

    constant = all(one.z_constant for one in modes)
    if constant and reuse is None:
        table = constant_phi(
            operator_at(0.0),
            z_max,
            margin=margin,
            beta0=beta0,
            beta1=beta1,
            kinks=extra,
        )
        logger.debug(
            "Phi: constant operator over [0, %.6g] m, %d bytes", span, table.nbytes
        )
        return table

    # The floor is a property of one evaluation of D, not of the variable it is
    # differenced in, so it is measured once and serves both representations.
    floor = _roundoff_floor(operator_at, span)
    working = max(tol, ROUNDOFF_SAFETY * floor)
    if working > tol:
        logger.info(
            "Phi: D is evaluated to a relative round-off floor of %.2e, so the "
            "tolerance is raised from %.1e to %.1e (%.0fx the floor)",
            floor,
            tol,
            working,
            ROUNDOFF_SAFETY,
        )

    if representation != "table":
        if (
            reuse is not None
            and isinstance(modes[0], ParameterisedMode)
            and parameters_own_z(modes)
        ):
            shared = reweight_expansion(
                reuse,
                modes[0].parameter_paths,
                integrate=True,
                tol=working,
                kinks=edges,
            )
            if shared is not None:
                return _expansion_operator(
                    shared, operator_at, span, beta0=beta0, beta1=beta1
                )
        if constant:
            return constant_phi(
                operator_at(0.0),
                z_max,
                margin=margin,
                beta0=beta0,
                beta1=beta1,
                kinks=extra,
            )
        expansion = _phi_expansion(
            modes,
            grid,
            frame,
            n_pol,
            loss,
            absorber_rate,
            span=span,
            tol=working,
            kinks=edges,
            n_max=n_max,
            max_intervals=max_intervals,
        )
        if expansion is not None:
            return _expansion_operator(
                expansion, operator_at, span, beta0=beta0, beta1=beta1
            )
        if representation == "expansion":
            msg = (
                "an expansion in the scalar parameters was asked for and could not be "
                f"built over [0, {span:.6g}] m; the debug log says why, and the z "
                "table is the general representation"
            )
            raise TranslationError(msg)

    return _z_table_operator(
        operator_at,
        paths,
        span,
        tol=working,
        kinks=edges,
        n_max=n_max,
        max_intervals=max_intervals,
        beta0=beta0,
        beta1=beta1,
    )


def _weight_sampler(
    modes: Sequence[Mode], grid: TimeGrid, n_pol: int, index: float | None
) -> Callable[[float], Float[np.ndarray, "n_modes n_pol n_omega"]]:
    """Sample the prefactor's real index; unused spectral bins have unit weight.

    Parameters
    ----------
    modes : sequence of solprop.modes.protocol.Mode
        Modes in canonical state order.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    n_pol : int
        Number of state polarisations, one or two.
    index : float or None
        Constant index override, or None for the physical effective index.

    Returns
    -------
    callable
        Positive real index in shape (n_modes, n_pol, n_omega) at one z.
    """

    def sample(z: float) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
        """Sample the index at the requested position.

        Parameters
        ----------
        z : float or numpy.ndarray
            Positions within the translated span, in metres.

        Returns
        -------
        numpy.ndarray
            Values of the sampled weight polynomial at the requested positions.
        """
        weight = np.ones((len(modes), n_pol, grid.n_omega))
        for number, mode in enumerate(modes):
            weight[number][:, grid.active_mask] = power_weight(
                mode, grid.omega[grid.active_mask], z, index=index
            )
        return weight

    return sample


def with_power_weight(
    linear: LinearOperator,
    modes: Sequence[Mode],
    grid: TimeGrid,
    index: float | None,
) -> LinearOperator:
    """Mirror the prefactor representations for statistics, outside the step loop.

    Parameters
    ----------
    linear : LinearOperator
        Cumulative dispersion operator and its representation.
    modes : sequence of solprop.modes.protocol.Mode
        Modes in canonical state order.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    index : float or None
        Constant index override, or None for the physical effective index.

    Returns
    -------
    LinearOperator
        Operator with a constant, parameter-expansion or z-table weight row.
    """
    n_pol = linear.value_shape[2]
    sample = _weight_sampler(modes, grid, n_pol, index)
    if index is not None or all(mode.z_constant for mode in modes):
        constant = sample(0.0)
        table = PiecewiseChebyshev(
            _readonly(constant[None, None]),
            _readonly(np.array(linear.z_span)),
            _readonly(np.array([0], dtype=np.intp)),
            _readonly(np.array([])),
        )
        return replace(
            linear,
            power_weight_representation="constant",
            power_weight=table,
            power_weight_constant=constant,
        )
    paths = parameter_paths(modes, *linear.z_span)
    if linear.phi_representation == "expansion" and paths is not None:
        frozen = [mode for mode in modes if isinstance(mode, ParameterisedMode)]

        def value_at(
            values: Mapping[str, float],
        ) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
            """Sample the weight inside the declared parameter box.

            Parameters
            ----------
            values : collections.abc.Mapping
                Physical parameter values inside the declared expansion box.

            Returns
            -------
            numpy.ndarray
                Weight values in the mode and frequency layout.
            """
            return _weight_sampler(
                [mode.at_parameters(values) for mode in frozen], grid, n_pol, index
            )(0.0)

        expansion = build_expansion(
            value_at, paths, *linear.z_span, tol=DEFAULT_TOL, kinks=linear.hard_kinks
        )
        if expansion is not None:
            return replace(
                linear,
                power_weight_representation="expansion",
                power_weight_expansion=expansion,
            )

    def samples(
        z: Float[np.ndarray, " n_z"],
    ) -> Float[np.ndarray, "n_z n_modes n_pol n_omega"]:
        """Stack weight values along the position axis.

        Parameters
        ----------
        z : float or numpy.ndarray
            Positions within the translated span, in metres.

        Returns
        -------
        numpy.ndarray
            Power weights with a leading position axis.
        """
        return np.stack([sample(float(one)) for one in z])

    table = build(
        samples,
        *linear.z_span,
        tol=DEFAULT_TOL,
        mandatory_breakpoints=linear.hard_kinks,
    )
    table = replace(
        table,
        coefficients=_readonly(table.coefficients),
        breakpoints=_readonly(table.breakpoints),
        degrees=_readonly(table.degrees),
        hard_kinks=_readonly(table.hard_kinks),
    )
    return replace(linear, power_weight_representation="table", power_weight=table)


# Nothing in the package calls isinstance on this, but the test session's beartype hook
# does: it compiles an isinstance check for every annotation, and a plain Protocol is
# not isinstanceable, so dropping the decorator makes every caller of
# `power_weight_at` fail to import under the hook.
@runtime_checkable
class _PowerWeightedProblem(Protocol):
    """Expose the linear group without a circular import of KernelProblem."""

    @property
    def linear(self) -> LinearOperator:
        """Return the translated linear group.

        Returns
        -------
        LinearOperator
            Operator and optional modal power-weight rows.
        """
        ...


def power_weight_at(
    problem: _PowerWeightedProblem, z: float
) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
    """Evaluate the modal power weight at one statistics or observation plane.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Validated modal problem carrying the weight row.
    z : float
        Position in m inside the translated range.

    Returns
    -------
    numpy.ndarray
        Positive dimensionless weights of shape ``(n_modes, n_pol, n_omega)``.

    Raises
    ------
    TranslationError
        If no modal row is present or the position is outside the translated range.
    """
    return power_weight_row(problem.linear, z)


def power_weight_row(
    linear: LinearOperator, z: float
) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
    """Evaluate the modal power weight of an assembled linear group at one position.

    :func:`power_weight_at` on a problem and the entrance conversion of the launch in
    :mod:`solprop.translate.problem` are the same quantity, so they share this one
    arithmetic path: a second, independent evaluation of the mode's index would leave
    the launch and the output sites disagreeing at the last few bits and break the
    exactness of the entrance round trip.

    Parameters
    ----------
    linear : LinearOperator
        Assembled linear group carrying the weight row.
    z : float
        Position in m inside the translated range.

    Returns
    -------
    numpy.ndarray
        Positive dimensionless weights of shape ``(n_modes, n_pol, n_omega)``.

    Raises
    ------
    TranslationError
        If no modal row is present or the position is outside the translated range.
    """
    _check_inside(linear, z)
    if linear.power_weight_constant is not None:
        return linear.power_weight_constant
    if linear.power_weight_expansion is not None:
        return linear.power_weight_expansion(z)
    if linear.power_weight is not None:
        return linear.power_weight(z)
    msg = "power_weight is absent for spatial geometry"
    raise TranslationError(msg)


def power_weight_factors(
    problem: _PowerWeightedProblem, positions: Float[np.ndarray, " n_z"]
) -> tuple[
    Float[np.ndarray, "n_terms n_modes n_pol n_omega"], Float[np.ndarray, "n_z n_terms"]
]:
    """Factor the modal power weight at many planes into terms and per-plane weights.

    A saved run records the weight at every save plane so that an analysis can recover
    the internal amplitude without the modes. The dense stack is ``n_z`` state-sized
    rows; the factored form is what the problem already holds: one row for a constant
    fill, the expansion's few terms along a declared parameter path, and the dense
    stack itself only for the z table. ``numpy.tensordot(basis[i], terms, axes=1)``
    reproduces :func:`power_weight_at` at ``positions[i]`` bit for bit, because it is
    the arithmetic that function performs.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Validated modal problem carrying the weight row.
    positions : numpy.ndarray
        Positions in m inside the translated range, shape ``(n_z,)``.

    Returns
    -------
    terms : numpy.ndarray
        Dimensionless rows of shape ``(n_terms, n_modes, n_pol, n_omega)``.
    basis : numpy.ndarray
        Dimensionless per-plane weights of the rows, shape ``(n_z, n_terms)``.

    Raises
    ------
    TranslationError
        If no modal row is present or a position is outside the translated range.
    """
    linear = problem.linear
    for z in positions:
        _check_inside(linear, float(z))
    if linear.power_weight_constant is not None:
        return linear.power_weight_constant[np.newaxis], np.ones((positions.size, 1))
    expansion = linear.power_weight_expansion
    if expansion is not None:
        basis = np.stack([expansion.weights(float(z)) for z in positions])
        return np.asarray(expansion.terms, dtype=float), basis
    if linear.power_weight is not None:
        # The z table has no separable form: each plane is its own term.
        terms = np.stack([linear.power_weight(float(z)) for z in positions])
        return terms, np.eye(positions.size)
    msg = "power_weight is absent for spatial geometry"
    raise TranslationError(msg)


def _log_weight_sampler(
    modes: Sequence[Mode], grid: TimeGrid, n_pol: int, index: float | None
) -> _OperatorAt:
    """Sample the real endpoint exponent on the active band, zero elsewhere.

    Parameters
    ----------
    modes : sequence of solprop.modes.protocol.Mode
        Modes in canonical state order.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    n_pol : int
        Number of state polarisations, one or two.
    index : float or None
        Constant index override, or None for the physical effective index.

    Returns
    -------
    callable
        Minus one half the logarithmic weight in canonical state shape.
    """

    def sample(z: float) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Sample the index at the requested position.

        Parameters
        ----------
        z : float or numpy.ndarray
            Positions within the translated span, in metres.

        Returns
        -------
        numpy.ndarray
            Values of the sampled weight polynomial at the requested positions.
        """
        columns = np.zeros((1, len(modes), n_pol, grid.n_omega), dtype=complex)
        for number, mode in enumerate(modes):
            columns[0, number][:, grid.active_mask] = -0.5 * log_power_weight(
                mode, grid.omega[grid.active_mask], z, index=index
            )
        return columns

    return sample


def _power_phi(
    linear: LinearOperator,
    modes: Sequence[Mode],
    grid: TimeGrid,
    index: float | None,
) -> LinearOperator:
    """Add the exact cumulative power conversion without changing constant Phi.

    Parameters
    ----------
    linear : LinearOperator
        Cumulative dispersion operator and its representation.
    modes : sequence of solprop.modes.protocol.Mode
        Modes in canonical state order.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.
    index : float or None
        Constant index override, or None for the physical effective index.

    Returns
    -------
    LinearOperator
        Operator including the exact cumulative logarithmic term.
    """
    if linear.phi_representation == "constant" or index is not None:
        return linear
    n_pol = linear.value_shape[2]
    sample = _log_weight_sampler(modes, grid, n_pol, index)
    # Unlike D, this logarithm has no frame cancellation. Measure its own floor;
    # eight rounding units cover sampling and the coefficient transform.
    tol = max(8 * np.finfo(float).eps, 8 * _roundoff_floor(sample, linear.z_span[1]))
    if linear.phi_expansion is not None:
        if linear.phi_expansion.endpoint_degrees is not None:
            return linear
        frozen = [mode for mode in modes if isinstance(mode, ParameterisedMode)]
        paths = parameter_paths(modes, *linear.z_span)
        assert paths is not None

        def value_at(
            values: Mapping[str, float],
        ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
            """Sample the weight inside the declared parameter box.

            Parameters
            ----------
            values : collections.abc.Mapping
                Physical parameter values inside the declared expansion box.

            Returns
            -------
            numpy.ndarray
                Weight values in the mode and frequency layout.
            """
            return _log_weight_sampler(
                [mode.at_parameters(values) for mode in frozen], grid, n_pol, index
            )(0.0)

        # The tail test runs per element on the whole box, whose corners the path
        # never visits and where an element can be far smaller than anywhere along
        # z, so the floor along z does not bound the one the test meets.
        box_floor = parameter_roundoff_floor(
            value_at, linear.phi_expansion.parameters, linear.phi_expansion.ranges
        )
        box_tol = max(tol, 8 * box_floor)
        if box_tol > tol:
            logger.info(
                "Phi: the logarithmic power weight is evaluated to a relative "
                "round-off floor of %.2e on the parameter box, so its tolerance is "
                "raised from %.1e to %.1e",
                box_floor,
                tol,
                box_tol,
            )
        expansion = append_endpoint_expansion(
            linear.phi_expansion, value_at, paths, tol=box_tol
        )
        if expansion is None:
            msg = (
                "the logarithmic power weight could not resolve on Phi's parameter box"
            )
            raise TranslationError(msg)
        return replace(
            linear,
            phi_expansion=expansion,
            phi_breakpoints=expansion.weights.breakpoints,
            phi_degrees=expansion.weights.degrees,
            hard_kinks=expansion.weights.hard_kinks,
        )
    entrance = sample(0.0)

    def correction(
        z: Float[np.ndarray, " n_z"],
    ) -> Complex[np.ndarray, "n_z n_batch n_modes n_pol n_omega"]:
        """Sample the logarithmic contribution along z.

        Parameters
        ----------
        z : float or numpy.ndarray
            Positions within the translated span, in metres.

        Returns
        -------
        numpy.ndarray
            Logarithmic values with a leading position axis.
        """
        return np.stack([sample(float(one)) for one in z])

    # Resolve the log itself before subtracting its entrance value. A relative
    # tolerance on the vanishing endpoint difference would chase round-off at z=0.
    endpoint = build(
        correction,
        *linear.z_span,
        tol=tol,
        mandatory_breakpoints=linear.phi_breakpoints[1:-1],
    )
    coefficients = endpoint.coefficients.copy()
    coefficients[:, 0] -= entrance
    endpoint = replace(endpoint, coefficients=coefficients)
    combined = _add_phi_tables(linear.polynomial, endpoint)
    return replace(
        linear,
        phi_coefficients=combined.coefficients,
        phi_breakpoints=combined.breakpoints,
        phi_degrees=combined.degrees,
    )


def _add_phi_tables(
    integral: PiecewiseChebyshev, endpoint: PiecewiseChebyshev
) -> PiecewiseChebyshev:
    """Add independently resolved coefficients on the endpoint table's refinement.

    The endpoint table is built on the integral's own breakpoints as mandatory edges, so
    it refines them and the integral is a single polynomial across each interval of the
    result. Reprojecting it there at *its* degree is therefore exact, and the sum on
    interval ``i`` has degree ``max(integral degree there, endpoint degree i)``. The
    coefficient array stays rectangular at the common width -- that is already the
    narrowest width that holds every interval, and the padding is exact zeros, which a
    Clenshaw pass reproduces bit for bit -- but `degrees` is per interval, as the field
    documents, rather than the maximum everywhere.

    Parameters
    ----------
    integral : PiecewiseChebyshev
        Original integrated polynomial on its z intervals.
    endpoint : PiecewiseChebyshev
        Endpoint polynomial on the same physical span.

    Returns
    -------
    PiecewiseChebyshev
        Sum of both polynomials on their common refinement.
    """
    width = max(integral.coefficients.shape[1], endpoint.coefficients.shape[1])
    coefficients = np.zeros(
        (endpoint.degrees.size, width, *integral.value_shape), dtype=complex
    )
    degrees = np.empty(endpoint.degrees.shape, dtype=np.intp)
    for number, (low, high) in enumerate(
        zip(endpoint.breakpoints[:-1], endpoint.breakpoints[1:], strict=True)
    ):
        old = integral.interval_containing(float((low + high) / 2))
        reprojected = max(1, int(integral.degrees[old]))
        points = lobatto_points(reprojected, float(low), float(high))
        # `resolve_axis` is used for its transform alone: the reprojection is exact by
        # construction, so the resolution flag `tol` decides is meaningless here and the
        # value 1.0 only keeps the tolerance validation happy.
        transformed, _ = resolve_axis(
            np.stack([integral(float(z)) for z in points]), 0, tol=1.0
        )
        coefficients[number, : transformed.shape[0]] = transformed
        degrees[number] = max(reprojected, int(endpoint.degrees[number]))
    coefficients[:, : endpoint.coefficients.shape[1]] += endpoint.coefficients
    return PiecewiseChebyshev(
        coefficients,
        endpoint.breakpoints,
        degrees,
        integral.hard_kinks,
        integral.budget_exceeded or endpoint.budget_exceeded,
    )


def _checked_representation(representation: str) -> None:
    """Refuse a representation the builder does not know.

    Its parameter is a plain ``str`` rather than the literal alias, because the point of
    the check is the production run, where no runtime type checker stands in front of it
    and a misspelt choice would otherwise be taken for "not the z table".

    Parameters
    ----------
    representation : str
        What the caller asked for.

    Raises
    ------
    TranslationError
        If it is not one of :data:`RepresentationChoice`.
    """
    if representation not in ("auto", "expansion", "table"):
        msg = (
            "representation must be 'auto', 'expansion' or 'table', got "
            f"{representation!r}"
        )
        raise TranslationError(msg)


def _phi_expansion(  # noqa: PLR0913 - the sampler's inputs plus the build's knobs
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    n_pol: int,
    loss: LossPolicy,
    absorber_rate: Float[np.ndarray, " n_omega"] | None,
    *,
    span: float,
    tol: float,
    kinks: Sequence[float],
    n_max: int,
    max_intervals: int,
) -> ParameterExpansion | None:
    """Try to expand ``Phi`` in the scalar paths the mode set agrees on.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set, in order.
    grid : solprop.grid.TimeGrid
        The grid.
    frame : solprop.linear.Frame
        The co-moving frame.
    n_pol : int
        Polarisation components.
    loss : solprop.linear.LossPolicy
        The loss limit.
    absorber_rate : numpy.ndarray or None
        The spectral absorber rate.
    span : float
        End of the tabulated range, m.
    tol : float
        Target bound on the uniform relative error, already raised to the round-off
        floor.
    kinks : sequence of float
        The paths' interior breakpoints, mandatory edges of the scalar tables.
    n_max : int
        Highest degree tried on a parameter.
    max_intervals : int
        Interval budget for the scalar tables.

    Returns
    -------
    solprop.translate.expansion.ParameterExpansion or None
        The expansion, or ``None`` when the mode set does not declare a usable
        parameter set or a parameter does not resolve. Either way the reason is logged.
    """
    varying = parameter_paths(modes, 0.0, span)
    if varying is None:
        return None
    operator_of = _operator_of_parameters(
        modes, grid, frame, n_pol, loss, absorber_rate
    )
    working = tol
    expansion = build_expansion(
        operator_of,
        varying,
        0.0,
        span,
        tol=working,
        kinks=kinks,
        n_max=n_max,
        max_intervals=max_intervals,
        integrate=True,
    )
    while expansion is not None and expansion.budget_exceeded and working < MAX_TOL:
        working = min(working * 10.0, MAX_TOL)
        logger.warning(
            "Phi: the scalar tables of the expansion exhausted their budget of %d over "
            "[0, %.6g] m; relaxing the tolerance to %.1e and rebuilding",
            max_intervals,
            span,
            working,
        )
        expansion = build_expansion(
            operator_of,
            varying,
            0.0,
            span,
            tol=working,
            kinks=kinks,
            n_max=n_max,
            max_intervals=max_intervals,
            integrate=True,
        )
    return expansion


def _expansion_operator(
    expansion: ParameterExpansion,
    operator_at: _OperatorAt,
    span: float,
    *,
    beta0: PiecewiseChebyshev,
    beta1: PiecewiseChebyshev,
) -> LinearOperator:
    """Wrap a resolved expansion as section 3 of the contract.

    Parameters
    ----------
    expansion : solprop.translate.expansion.ParameterExpansion
        The expansion of ``Phi``, with integrated weights.
    operator_at : callable
        ``z -> D(z)``, for the debug-level check against the quadrature.
    span : float
        End of the tabulated range, m.
    beta0, beta1 : solprop.translate.chebyshev.PiecewiseChebyshev
        The frame quantities, for the record.

    Returns
    -------
    LinearOperator
        With ``phi_representation`` ``"expansion"``.
    """
    weights = expansion.weights
    table = LinearOperator(
        phi_representation="expansion",
        phi_breakpoints=weights.breakpoints,
        phi_coefficients=None,
        phi_degrees=weights.degrees,
        hard_kinks=weights.hard_kinks,
        phi_constant=None,
        phi_expansion=expansion,
        beta0=beta0,
        beta1=beta1,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Phi: expansion in %s to degrees %s, %d terms over %d scalar intervals on "
            "[0, %.6g] m, %d hard kinks, %.3f MB; quad_vec agreement %.3e",
            expansion.parameters,
            expansion.degrees.tolist(),
            expansion.n_terms,
            table.n_intervals,
            span,
            table.hard_kinks.size,
            table.nbytes / 2**20,
            verify_phi(table, operator_at, _verify_pairs(span, DEFAULT_VERIFY_POINTS)),
        )
    return table


def _z_table_operator(  # noqa: PLR0913 - the fallback build's knobs, as `build` has them
    operator_at: _OperatorAt,
    paths: Sequence[ZProfile],
    span: float,
    *,
    tol: float,
    kinks: Sequence[float],
    n_max: int,
    max_intervals: int,
    beta0: PiecewiseChebyshev,
    beta1: PiecewiseChebyshev,
) -> LinearOperator:
    """Build the version-2 z table, the general representation.

    Parameters
    ----------
    operator_at : callable
        ``z -> D(z)`` in the canonical layout.
    paths : sequence of solprop.profiles.ZProfile
        The declared scalar paths, resolved first because they carry the structure.
    span : float
        End of the tabulated range, m.
    tol : float
        Target bound on the uniform relative error, already raised to the round-off
        floor; relaxed by decades up to :data:`MAX_TOL` if the budget is exhausted.
    kinks : sequence of float
        The paths' interior breakpoints, mandatory edges.
    n_max : int
        Highest degree tried on an interval before it is bisected.
    max_intervals : int
        Interval budget.
    beta0, beta1 : solprop.translate.chebyshev.PiecewiseChebyshev
        The frame quantities, for the record.

    Returns
    -------
    LinearOperator
        With ``phi_representation`` ``"table"``.
    """
    samplers = [path.__call__ for path in paths]

    def sample(
        z: Float[np.ndarray, " n_z"], *_: object
    ) -> Complex[np.ndarray, "n_z n_batch n_modes n_pol n_omega"]:
        """Return the operator at every position of a Chebyshev node set.

        Parameters
        ----------
        z : numpy.ndarray
            Positions along the waveguide, m, shape ``(n_z,)``.
        *_ : object
            The path values the builder resolved; the operator is evaluated from ``z``
            directly, so they are not used.

        Returns
        -------
        numpy.ndarray
            ``D`` at each position, shape ``(n_z, 1, n_modes, n_pol, n_omega)``.
        """
        return np.stack([operator_at(float(one)) for one in z])

    working = tol
    resolved = build_from_paths(
        samplers,
        sample,
        0.0,
        span,
        tol=working,
        mandatory_breakpoints=kinks,
        n_max=n_max,
        max_intervals=max_intervals,
    )
    while resolved.budget_exceeded and working < MAX_TOL:
        working = min(working * 10.0, MAX_TOL)
        logger.warning(
            "Phi: the interval budget of %d was exhausted over [0, %.6g] m; relaxing "
            "the tolerance to %.1e and rebuilding (docs/dev/stepper.md section 4)",
            max_intervals,
            span,
            working,
        )
        resolved = build_from_paths(
            samplers,
            sample,
            0.0,
            span,
            tol=working,
            mandatory_breakpoints=kinks,
            n_max=n_max,
            max_intervals=max_intervals,
        )
    phi = resolved.integral()
    if resolved.budget_exceeded:
        logger.warning(
            "Phi: the interval budget of %d is exhausted even at %.1e over [0, %.6g] "
            "m; the table carries unresolved intervals and its accuracy is not stated",
            max_intervals,
            working,
            span,
        )
    table = LinearOperator(
        phi_representation="table",
        phi_breakpoints=phi.breakpoints,
        phi_coefficients=phi.coefficients,
        phi_degrees=phi.degrees,
        hard_kinks=phi.hard_kinks,
        phi_constant=None,
        beta0=beta0,
        beta1=beta1,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Phi: %d intervals of degree up to %d over [0, %.6g] m, %d hard kinks, "
            "%.3f MB; quad_vec agreement %.3e",
            table.n_intervals,
            int(table.phi_degrees.max()),
            span,
            table.hard_kinks.size,
            table.nbytes / 2**20,
            verify_phi(table, operator_at, _verify_pairs(span, DEFAULT_VERIFY_POINTS)),
        )
    return table


def _operator_sampler(
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    n_pol: int,
    loss: LossPolicy,
    absorber_rate: Float[np.ndarray, " n_omega"] | None,
) -> _OperatorAt:
    """Return a callable giving ``D`` in the canonical layout at one position.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set, in order.
    grid : solprop.grid.TimeGrid
        The grid.
    frame : solprop.linear.Frame
        The co-moving frame.
    n_pol : int
        Polarisation components; ``D`` is repeated over them.
    loss : solprop.linear.LossPolicy
        The loss limit, applied before integration.
    absorber_rate : numpy.ndarray or None
        The spectral absorber rate, or ``None``.

    Returns
    -------
    callable
        ``z -> D`` of shape ``(1, n_modes, n_pol, n_omega)``.
    """

    def operator_at(z: float) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the linear operator at one position, in the canonical layout.

        Parameters
        ----------
        z : float
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``D(omega, z)`` in 1/m, shape ``(1, n_modes, n_pol, n_omega)``.
        """
        shared = _shared_frame(modes[0], frame, z)
        columns = [
            dispersion_operator(
                one, grid, shared, z, loss=loss, absorber_rate=absorber_rate
            )
            for one in modes
        ]
        return _canonical(columns, n_pol)

    return operator_at


def _operator_of_parameters(
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    n_pol: int,
    loss: LossPolicy,
    absorber_rate: Float[np.ndarray, " n_omega"] | None,
) -> _OperatorOf:
    """Return a callable giving ``D`` at parameter values rather than at a position.

    Every mode is frozen at the same values, which is sound because
    :func:`parameter_paths` has checked that the set agrees about every path it expands
    over, and the whole set is then written in one frame taken from the reference mode
    at those values, as :func:`_shared_frame` does along z. The
    frozen modes are z-constant, so the position handed to
    :func:`solprop.linear.dispersion_operator` is irrelevant and is zero.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set, in order; each must implement
        :class:`solprop.modes.protocol.ParameterisedMode`.
    grid : solprop.grid.TimeGrid
        The grid.
    frame : solprop.linear.Frame
        The co-moving frame. Its ``beta_0`` and ``beta_1`` follow the frozen reference
        mode, so the frame subtraction is a function of the same parameters and needs
        no dimension of its own.
    n_pol : int
        Polarisation components; ``D`` is repeated over them.
    loss : solprop.linear.LossPolicy
        The loss limit, applied before integration.
    absorber_rate : numpy.ndarray or None
        The spectral absorber rate, or ``None``.

    Returns
    -------
    callable
        ``values -> D`` of shape ``(1, n_modes, n_pol, n_omega)``.
    """
    parameterised = [one for one in modes if isinstance(one, ParameterisedMode)]

    def operator_of(
        values: Mapping[str, float],
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the linear operator at one point of the parameter space.

        Parameters
        ----------
        values : collections.abc.Mapping
            One value per expanded path, by name.

        Returns
        -------
        numpy.ndarray
            ``D(omega; values)`` in 1/m, shape ``(1, n_modes, n_pol, n_omega)``.
        """
        frozen = [one.at_parameters(values) for one in parameterised]
        # One frame for the whole set, from the reference mode at *these* parameter
        # values, exactly as `_operator_sampler` takes it from the reference mode at
        # this position: it is the same frame, reached through the same funnel, and a
        # frame per mode would give each its own time origin.
        shared = _shared_frame(frozen[0], frame, 0.0)
        columns = [
            dispersion_operator(
                one, grid, shared, 0.0, loss=loss, absorber_rate=absorber_rate
            )
            for one in frozen
        ]
        return _canonical(columns, n_pol)

    return operator_of


def _canonical(
    columns: Sequence[Complex[np.ndarray, " n_omega"]], n_pol: int
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Stack one operator per mode into the canonical state layout.

    Parameters
    ----------
    columns : sequence of numpy.ndarray
        ``D`` per mode, each of shape ``(n_omega,)``.
    n_pol : int
        Polarisation components; ``D`` is the same for both components of one mode and
        is repeated over them so a birefringent mode needs no new layout.

    Returns
    -------
    numpy.ndarray
        Shape ``(1, n_modes, n_pol, n_omega)``.
    """
    stacked = np.stack(columns)[:, np.newaxis, :]
    return np.repeat(stacked, n_pol, axis=1)[np.newaxis]


def _shared_frame(reference: Mode, frame: Frame, z: float) -> Frame:
    """Freeze the frame's quantities on the reference mode, for the whole set.

    A frame whose ``beta_0`` and ``beta_1`` are ``None`` recomputes them **from the
    mode it is handed** (:func:`solprop.linear.frame_quantities`), which is what makes
    it co-moving along a gradient. Handed one mode at a time that would give every mode
    its own time origin: the operator's ``beta_1 omega`` term is a change of the time
    variable, so a set propagated that way is no longer simultaneous, and the modes
    walk apart by ``(beta_1,p - beta_1,1) z``. The nonlinear term multiplies the modes
    at the same time sample, so that walk-off is silently lost -- and the group delay
    between HE_11 and HE_12 of a 125 um capillary is 43 fs per metre, tens of pulse
    widths over a metre-scale multimode run.

    One frame therefore serves the whole set, taken from the first mode, which is the
    operator ``docs/guide/numerics/linear-operator.md`` writes -- ``beta_0(z)`` and
    ``beta_1(z)`` carry no mode index there -- and the one whose quantities
    :func:`build_phi` records beside ``Phi``. Freezing them here also saves the
    remaining modes an ``neff`` evaluation each.

    Parameters
    ----------
    reference : solprop.modes.protocol.Mode
        The mode the frame follows; the first of the set.
    frame : solprop.linear.Frame
        The frame, whose fixed values are kept as they are.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    solprop.linear.Frame
        The same frame with both quantities fixed at this position.
    """
    if frame.beta0 is not None and frame.beta1 is not None:
        return frame
    beta0, beta1 = frame_quantities(reference, frame, z)
    return replace(frame, beta0=beta0, beta1=beta1)


def _roundoff_floor(operator_at: _OperatorAt, span: float) -> float:
    """Estimate the relative round-off of one evaluation of the operator.

    A second difference over a spacing far below any physical scale isolates the noise:
    the smooth part contributes ``delta**2 |D''|``, which at a spacing of 1e-9 of the
    span is some fifteen orders below the round-off, while the noise survives at its
    full size. The estimate is taken at three positions and the largest kept.

    It is reported in the **acceptance test's own norm**: per element against that
    element's own value, with the comparison scale floored at
    :data:`~solprop.translate.chebyshev.ENVELOPE_FLOOR` of the global envelope, exactly
    as ``_tail_ratio`` measures the coefficient tail it is meant to bound. Measured
    against the global envelope instead, a mode set holding one small operator beside
    one large one reports a floor set by the large one, the tolerance stays far below
    what the small one can be resolved to, the interval budget is exhausted and the
    whole search is paid a second time at a looser tolerance. The two norms
    differ by whatever ratio the elements span, which is four orders on a wide field
    grid and eight when a dispersion-specified mode stands beside a capillary one.

    Parameters
    ----------
    operator_at : callable
        ``z -> D(z)`` in the canonical layout.
    span : float
        End of the tabulated range, m.

    Returns
    -------
    float
        The relative round-off floor; zero if the operator vanishes everywhere probed.
    """
    delta = _FLOOR_STEP_FRACTION * span
    worst = 0.0
    for fraction in np.linspace(0.25, 0.75, _FLOOR_PROBES):
        centre = float(fraction * span)
        low = operator_at(centre - delta)
        middle = operator_at(centre)
        high = operator_at(centre + delta)
        envelope = np.abs(middle)
        global_envelope = float(envelope.max())
        if global_envelope == 0.0:
            continue
        scale = np.maximum(envelope, global_envelope * ENVELOPE_FLOOR)
        noise = np.abs(low - 2.0 * middle + high)
        worst = max(worst, float(np.max(noise / scale)))
    return worst


def declared_paths(modes: Sequence[Mode]) -> tuple[ZProfile, ...]:
    """Return the scalar paths along z the mode set declares, without duplicates.

    The union :func:`build_phi` resolves before it samples the operator, and the one the
    prefactor table of :mod:`solprop.translate.responses` resolves for the same reason:
    the paths are cheap and carry the structure the expensive array-valued function
    inherits. Paths compare by identity: a bore profile handed to two modes counts once,
    while the density path each mode derives from its own pressure argument counts per
    mode -- which costs nothing but a repeated, cheap resolution.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.

    Returns
    -------
    tuple of solprop.profiles.ZProfile
        The non-constant ``core_radius`` and ``density`` paths, each once.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.profiles import Linear
    >>> from solprop.translate.linop import declared_paths
    >>> bore = Linear(125e-6, 75e-6, 0.8)
    >>> modes = [MarcatiliMode(bore, "He", 2.0, m=m) for m in (1, 2)]
    >>> len(declared_paths(modes))
    1
    """
    paths: list[ZProfile] = []
    for one in modes:
        declared = (
            one.parameter_paths.values()
            if isinstance(one, ParameterisedMode)
            else (getattr(one, name, None) for name in ("core_radius", "density"))
        )
        for path in declared:
            if path is None or path.is_constant:
                continue
            if not any(path is seen for seen in paths):
                paths.append(path)
    return tuple(paths)


def parameter_paths(
    modes: Sequence[Mode], z0: float, z1: float
) -> dict[str, ZProfile] | None:
    """Return the varying scalar paths the whole mode set agrees on, or ``None``.

    The expansion of :mod:`solprop.translate.expansion` has one dimension per path, and
    every mode is frozen at the *same* point of that space, so the set is usable only if
    every mode declares the same paths and agrees about their values. That is the
    ordinary case -- the modes of one capillary share a bore and a fill -- but it is not
    automatic, because each mode derives its own ``density`` object from its own
    pressure argument. This checks it rather than assuming it, by value at probe
    positions, and refuses the set otherwise; the caller then builds the z table, which
    makes no such assumption.

    Unlike :func:`declared_paths`, which deduplicates by identity and returns the paths
    that drive the adaptive split, this returns them **by name**, because a name is what
    :meth:`solprop.modes.protocol.ParameterisedMode.at_parameters` takes.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.
    z0, z1 : float
        Ends of the tabulated range in m, where the paths are compared.

    Returns
    -------
    dict or None
        The non-constant paths by name, in a stable order, or ``None`` when a mode does
        not declare its parameters, the modes declare different ones, they disagree
        about a path's values, or every declared path is constant. Each case is logged
        at DEBUG with what it was.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.profiles import Linear
    >>> from solprop.translate.linop import parameter_paths
    >>> bore = Linear(125e-6, 75e-6, 0.8)
    >>> modes = [MarcatiliMode(bore, "He", 2.0, m=m) for m in (1, 2)]
    >>> sorted(parameter_paths(modes, 0.0, 0.8))
    ['core_radius']
    """
    parameterised = [one for one in modes if isinstance(one, ParameterisedMode)]
    if len(parameterised) != len(modes):
        logger.debug(
            "Phi: %d of %d modes do not declare their parameters, so the operator is "
            "not a function of a shared parameter set; the z table it is",
            len(modes) - len(parameterised),
            len(modes),
        )
        return None
    if not parameters_own_z(modes):
        logger.debug(
            "Phi: a mode declares parameters that are not the whole of its z "
            "dependence, so freezing them would silently drop the rest; the z table "
            "it is"
        )
        return None
    declared = [dict(one.parameter_paths) for one in parameterised]
    names = sorted(declared[0])
    if any(sorted(one) != names for one in declared[1:]):
        logger.debug(
            "Phi: the modes declare different parameters (%s), so there is no shared "
            "parameter space; the z table it is",
            [sorted(one) for one in declared],
        )
        return None
    probes = np.linspace(z0, z1, _AGREEMENT_PROBES)
    varying: dict[str, ZProfile] = {}
    for name in names:
        paths = [one[name] for one in declared]
        values = [np.asarray(path(probes), dtype=np.float64) for path in paths]
        if any(not np.array_equal(values[0], other) for other in values[1:]):
            logger.debug(
                "Phi: the modes disagree about the path %r over [%.6g, %.6g] m, so "
                "each has its own operator; the z table it is",
                name,
                z0,
                z1,
            )
            return None
        if not paths[0].is_constant:
            varying[name] = paths[0]
    if not varying:
        logger.debug(
            "Phi: every declared path (%s) is constant, yet the mode set is not "
            "z_constant; the z table it is",
            names,
        )
        return None
    return varying


_SPECTRAL_PATHS = (
    ("dispersion_path", "dispersion"),
    ("derivative_path", "dispersion derivative"),
    ("loss_path", "loss"),
)
"""The declared spectral paths of a specified mode, with the name each is refused by."""


def _check_dispersion_span(modes: Sequence[Mode], z_max: float) -> None:
    """Refuse a declared dispersion path that stops before the run does.

    A path along z holds its value at the nearest declared end outside its range,
    as a scalar :class:`solprop.profiles.ZProfile` does, which is what covers the
    few per cent of margin the tables carry beyond ``z_max``. A run **longer** than
    the declaration is a different matter: the dispersion of the last declared
    section would be carried over the rest of the fibre in silence, which is the
    kind of plausible, wrong answer the package refuses by name.

    All three spectral paths are checked and not the value alone: a
    :class:`solprop.nonlinear.gnlse.SpecifiedMode` takes a declared path for its
    derivative and for its loss too, and a loss declared over half the fibre would
    hold its last attenuation over the rest in exactly the same silence.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.
    z_max : float
        Length the run covers, m.

    Raises
    ------
    TranslationError
        Naming the mode, which of its paths falls short, its declared range and the
        run.
    """
    for one in modes:
        for attribute, label in _SPECTRAL_PATHS:
            path = getattr(one, attribute, None)
            if path is None or path.is_constant:
                continue
            start, end = path.z_span
            if start > 0.0 or end < z_max:
                msg = (
                    f"{type(one).__name__} declares its {label} over "
                    f"[{start:.6g}, {end:.6g}] m, which does not cover the run's "
                    f"[0, {z_max:.6g}] m: a declared path must reach the end of "
                    "the fibre rather than be held at its last value"
                )
                raise TranslationError(msg)


def parameters_own_z(modes: Sequence[Mode]) -> bool:
    """Return whether every mode's declared parameters are its whole z dependence.

    An optional trait, read by name and defaulting to ``True`` -- the assumption
    :class:`solprop.modes.protocol.ParameterisedMode` has always carried. A mode
    that declares scalar paths *and* something else that moves says so by reporting
    ``parameters_own_z = False``, and the translation layer then tabulates the
    operator along z instead of expanding it in parameters that cannot reproduce it.
    The case is a :class:`solprop.nonlinear.gnlse.SpecifiedMode` whose dispersion
    follows a path beside its declared area: the
    area is still its kinks, which is what :func:`declared_paths` reads it for, but
    freezing the area would leave the dispersion moving.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set.

    Returns
    -------
    bool
        ``True`` when no mode denies it.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.translate.linop import parameters_own_z
    >>> parameters_own_z([MarcatiliMode(125e-6, "He", 1.0)])
    True
    """
    return all(bool(getattr(one, "parameters_own_z", True)) for one in modes)


def _interior_edges(paths: Sequence[ZProfile], span: float) -> list[float]:
    """Return the declared breakpoints of the paths that lie strictly inside the range.

    Parameters
    ----------
    paths : sequence of solprop.profiles.ZProfile
        The scalar paths the modes declare.
    span : float
        End of the tabulated range, m.

    Returns
    -------
    list of float
        The interior breakpoints, sorted and deduplicated.
    """
    return sorted(
        {
            float(value)
            for path in paths
            for value in np.atleast_1d(path.breakpoints)
            if 0.0 < float(value) < span
        }
    )


def _frame_tables(
    mode: Mode, frame: Frame, span: float, tol: float, kinks: Sequence[float]
) -> tuple[PiecewiseChebyshev, PiecewiseChebyshev]:
    """Tabulate the frame quantities the operator removes, for the record.

    The mode's declared breakpoints are mandatory edges here for the same reason as in
    the operator's own build: a frame quantity kinks where the gas or the bore does, and
    a fit that is not told about the kink locates it by bisecting down to the minimum
    interval width, spending dozens of intervals and still leaving an error a hundred
    times its tolerance at the kink itself.

    Parameters
    ----------
    mode : Mode
        The reference mode; the frame follows it along z.
    frame : solprop.linear.Frame
        The frame.
    span : float
        End of the tabulated range, m.
    tol : float
        Target bound on the uniform error relative to the value scale.
    kinks : sequence of float
        Positions that must be interval edges.

    Returns
    -------
    tuple
        ``(beta0, beta1)`` as scalar Chebyshev tables, in 1/m and s/m.
    """
    quantities = [
        tabulate(
            lambda z, index=index: np.array(
                [frame_quantities(mode, frame, float(one))[index] for one in z]
            ),
            0.0,
            span,
            kinks=kinks,
            tol=tol,
        )
        for index in (0, 1)
    ]
    return quantities[0], quantities[1]


def delta_phi(
    table: LinearOperator, z1: float, z2: float, *, helpers: Helpers | None = None
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Return the exponent of the exact linear flow between two positions.

    ``Phi(z2) - Phi(z1)``, so that ``exp(delta_phi(table, z1, z2))`` is the linear
    propagator over ``[z1, z2]`` whatever the distance — the property the interaction
    picture rests on. A constant operator is handled as ``D (z2 - z1)`` rather than as
    the difference of two Chebyshev sums, which is exact. An expansion subtracts its
    scalar weights first and combines once, rather than combining twice and
    subtracting; the two agree to rounding and the first is ``n_terms`` flops per state
    element instead of two Clenshaw passes over it.

    Parameters
    ----------
    table : LinearOperator
        The cumulative operator.
    z1, z2 : float
        Positions in m, both inside the table's span.
    helpers : solprop._helpers.Helpers or None, optional
        The evaluating backend's Chebyshev helpers, for a z table read during a run;
        ``None``, the default, takes the shared translation-time evaluation.

    Returns
    -------
    numpy.ndarray
        Dimensionless exponent, complex, shape ``(n_batch, n_modes, n_pol, n_omega)``.

    Raises
    ------
    TranslationError
        If a position lies outside the table.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, delta_phi
    >>> table = constant_phi(np.array([-1j, -2j]), 1.0)
    >>> exponent = delta_phi(table, 0.25, 0.75).ravel()
    >>> bool(np.array_equal(np.exp(exponent), np.exp(np.array([-1j, -2j]) * 0.5)))
    True
    """
    if table.separable is not None:
        _check_inside(table, z1)
        _check_inside(table, z2)
        return _separable_phase(table.separable, z2 - z1, imaginary=False)
    if table.phi_constant is not None:
        _check_inside(table, z1)
        _check_inside(table, z2)
        return table.phi_constant * (z2 - z1)
    if table.phi_expansion is not None:
        return table.phi_expansion.delta(z1, z2)
    return table.polynomial(z2, helpers=helpers) - table.polynomial(z1, helpers=helpers)


def phi_rate(
    table: LinearOperator,
    z: float,
    *,
    before: bool = False,
    helpers: Helpers | None = None,
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Return the rate ``D(z) = dPhi/dz`` of the representation that defines the flow.

    A constant or separable operator returns ``D`` itself. A table's interval edges
    are continuous in ``Phi`` but, at a hard kink, not in ``D``: a position on an edge
    belongs to the interval starting there unless `before` asks for the one ending
    there -- the side a step that ends on the edge came from (SEP 16 section 4.2).

    Parameters
    ----------
    table : LinearOperator
        The cumulative operator.
    z : float
        Position in m, inside the table's span.
    before : bool, optional
        Read an edge from the interval ending there.
    helpers : solprop._helpers.Helpers or None, optional
        The stepping backend's implementation of an expansion's scalar rate rows (SEP 29
        section 6.6); ``None`` takes the shared compiled pass. A z table's rows, the
        size of the state, always take the shared pass.

    Returns
    -------
    numpy.ndarray
        ``D(z)`` in 1/m, shape ``(n_batch, n_modes, n_pol, n_omega)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, phi_rate
    >>> table = constant_phi(np.array([-1j, -2j]), 1.0)
    >>> phi_rate(table, 0.3).ravel().imag.tolist()
    [-1.0, -2.0]
    """
    if table.separable is not None:
        _check_inside(table, z)
        return _separable_phase(table.separable, 1.0, imaginary=False)
    if table.phi_constant is not None:
        _check_inside(table, z)
        return table.phi_constant.copy()
    value = _rate_value(table, z, before=before, helpers=helpers)
    if table.phi_expansion is not None:
        return np.tensordot(value, table.phi_expansion.terms, axes=1)
    return value


def rate_weights(
    table: LinearOperator,
    z: float,
    *,
    before: bool = False,
    helpers: Helpers | None = None,
) -> Float[np.ndarray, " n_phi_terms"]:
    """Return ``D(z)``'s weights over the operator's fixed terms.

    The weights of :func:`phi_rate` against the terms a device-ready backend holds
    (``translate.stage.StageTerms.phi``): ``[1]`` for a constant operator, whose single
    term is ``D``, and the derivative of an expansion's scalar integral tables. A
    resident Adams engine forms the rate from them (SEP 16 phase 3b).

    Parameters
    ----------
    table : LinearOperator
        A constant operator or an expansion.
    z : float
        Position in m, inside the table's span.
    before : bool, optional
        Read an edge from the interval ending there.
    helpers : solprop._helpers.Helpers or None, optional
        The stepping backend's implementation of the expansion's rate rows (SEP 29
        section 6.6); ``None`` takes the shared compiled pass.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_phi_terms,)``.

    Raises
    ------
    TranslationError
        For a z table or a separable operator, which have no such terms.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, rate_weights
    >>> rate_weights(constant_phi(np.array([-1j, -2j]), 1.0), 0.3).tolist()
    [1.0]
    """
    if table.phi_constant is not None:
        _check_inside(table, z)
        return np.ones(1)
    if table.phi_expansion is None:
        msg = "only a constant operator or an expansion has fixed terms to weight"
        raise TranslationError(msg)
    return np.asarray(
        _rate_value(table, z, before=before, helpers=helpers), dtype=np.float64
    )


def _rate_value(
    table: LinearOperator, z: float, *, before: bool, helpers: Helpers | None
) -> np.ndarray:
    """Evaluate the rate table at `z`, on the side of an edge `before` asks for.

    Parameters
    ----------
    table : LinearOperator
        An expansion or a z table.
    z : float
        Position in m.
    before : bool
        Read an edge from the interval ending there.
    helpers : solprop._helpers.Helpers or None
        The stepping backend's implementation for an expansion's rows, which reads the
        chosen interval as a one-interval table at the same ``x``; ``None``, or a z
        table, takes the shared compiled pass.

    Returns
    -------
    numpy.ndarray
        The expansion's term weights, or the z table's ``D``.
    """
    rate = table.rate
    index = rate.interval_containing(z)
    if before and index > 0 and z <= float(rate.breakpoints[index]):
        index -= 1
    lo = float(rate.breakpoints[index])
    hi = float(rate.breakpoints[index + 1])
    if helpers is None:
        return clenshaw((2.0 * z - lo - hi) / (hi - lo), rate.coefficients[index])
    block = np.ascontiguousarray(rate.coefficients[index : index + 1])
    flat = block.reshape(1, block.shape[1], -1)
    value = np.empty((1, flat.shape[2]), dtype=flat.dtype)
    # The interval is already chosen, edge side included: the position is never refused
    # and, on [lo, hi] alone, takes that interval and the same map onto [-1, 1].
    helpers.points(
        np.array([z], dtype=np.float64),
        np.array([lo, hi]),
        -math.inf,
        math.inf,
        flat,
        value,
    )
    return value.reshape(rate.coefficients.shape[2:])


def _separable_phase(
    coefficients: SeparableOperator, distance: float, *, imaginary: bool
) -> np.ndarray:
    """Evaluate a phase with bounded coefficient scratch and one returned volume.

    Parameters
    ----------
    coefficients : solprop.translate.spatial.SeparableOperator
        Shared constant-medium data.
    distance : float
        Integration length in m.
    imaginary : bool
        Return only the unitary phase when true.

    Returns
    -------
    numpy.ndarray
        Complex exponent or real unitary phase, in the canonical state layout.
    """
    result = np.empty(
        coefficients.value_shape, dtype=np.float64 if imaginary else np.complex128
    )
    # Six complex-equivalent arrays cover square root, masks and products.
    # This independent host evaluation uses the same 64 MiB workspace ceiling.
    chunk = max(1, (64 << 20) // (8 * 16 * len(coefficients.k_squared)))
    for first in range(0, coefficients.value_shape[1], chunk):
        rows = slice(first, first + chunk)
        operator, _ = separable_values(coefficients, rows)
        result[:, rows] = (np.imag(operator) if imaginary else operator) * distance
    return result


def unitary_phase(
    table: LinearOperator, z: float
) -> Float[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Return the phase that advances the shot-noise field to a position.

    The modified noise realisation is drawn once and carried along the fibre by the
    *unitary* part of the linear flow: dispersion yes, loss and gain no. That
    part is ``Im Phi(omega, z)``, so the same table serves the field and the noise and
    the noise costs no extra transform. Every representation answers it the same way:
    an expansion's weights are real and its integrals vanish at the start of the range,
    so its value at ``z`` is ``Phi(omega, z)`` itself.

    Parameters
    ----------
    table : LinearOperator
        The cumulative operator.
    z : float
        Position in m, inside the table's span.

    Returns
    -------
    numpy.ndarray
        Phase in radians, shape ``(n_batch, n_modes, n_pol, n_omega)``.

    Raises
    ------
    TranslationError
        If `z` lies outside the table.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, unitary_phase
    >>> table = constant_phi(np.array([-1j - 0.5, -2j]), 1.0)
    >>> np.round(unitary_phase(table, 0.5), 12).ravel().tolist()
    [-0.5, -1.0]
    """
    if table.separable is not None:
        _check_inside(table, z)
        return _separable_phase(table.separable, z, imaginary=True)
    if table.phi_constant is not None:
        _check_inside(table, z)
        return np.imag(table.phi_constant) * z
    if table.phi_expansion is not None:
        return np.imag(table.phi_expansion(z))
    return np.imag(table.polynomial(z))


def _check_inside(table: LinearOperator, z: float) -> None:
    """Refuse a position outside the tabulated range.

    Parameters
    ----------
    table : LinearOperator
        The cumulative operator.
    z : float
        Position in m.

    Raises
    ------
    TranslationError
        If `z` is outside ``[z_span[0], z_span[1]]``.
    """
    low, high = table.z_span
    if not low <= z <= high:
        msg = f"z = {z!r} m lies outside the table, which covers [{low}, {high}] m"
        raise TranslationError(msg)


def _verify_pairs(span: float, count: int) -> tuple[tuple[float, float], ...]:
    """Return evenly spread position pairs for a spot check of the table.

    Deterministic rather than random, so a debug log is reproducible: the pairs walk
    across the span with increasing separation.

    Parameters
    ----------
    span : float
        End of the tabulated range, m.
    count : int
        Number of pairs.

    Returns
    -------
    tuple
        ``(z1, z2)`` pairs with ``0 <= z1 < z2 <= span``.
    """
    starts = np.linspace(0.0, 0.5 * span, count)
    ends = np.linspace(0.5 * span, span, count)
    return tuple((float(a), float(b)) for a, b in zip(starts, ends, strict=True))


def verify_phi(
    table: LinearOperator,
    operator_at: _OperatorAt,
    points: Sequence[tuple[float, float]],
) -> float:
    """Check the table against an independent quadrature of the operator.

    Step 3 of the construction in ``docs/dev/stepper.md`` section 4: ``Phi(z2) -
    Phi(z1)`` is compared with ``scipy.integrate.quad_vec`` of ``D`` over the same
    interval, bin by bin. The quadrature is the oracle and is far more expensive than
    the table, so this is a spot check at a handful of pairs, not part of a run.

    Parameters
    ----------
    table : LinearOperator
        The cumulative operator to check.
    operator_at : callable
        ``z -> D(z)`` in the same layout as the table's value, in 1/m.
    points : sequence of tuple of float
        ``(z1, z2)`` pairs in m, inside the table's span.

    Returns
    -------
    float
        The largest deviation over all pairs and bins, relative to the scale of the
        exponent at that pair. Zero when `points` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi, verify_phi
    >>> operator = np.array([-1j, -2j])
    >>> table = constant_phi(operator, 1.0)
    >>> float(verify_phi(table, lambda z: operator, [(0.0, 1.0)])) < 1e-14
    True
    """
    worst = 0.0
    for z1, z2 in points:
        reference, error = quad_vec(
            lambda z: np.asarray(operator_at(float(z))), z1, z2, epsabs=1e-13
        )
        deviation = np.abs(delta_phi(table, z1, z2) - reference)
        scale = float(np.max(np.abs(reference)))
        if scale == 0.0:
            worst = max(worst, float(np.max(deviation)))
            continue
        worst = max(worst, float(np.max(deviation)) / scale)
        logger.debug(
            "Phi check over [%.6g, %.6g] m: deviation %.3e, quadrature error %.3e",
            z1,
            z2,
            float(np.max(deviation)),
            float(np.max(np.abs(error))),
        )
    return worst
