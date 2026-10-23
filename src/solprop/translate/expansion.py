# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""A z-dependent quantity as a Chebyshev expansion in the scalar paths it depends on.

A mode whose whole z dependence runs through named scalar paths
(:class:`solprop.modes.protocol.ParameterisedMode`) makes every quantity built from it a
function of those parameters rather than of the position:
``D(omega, z) = D(omega; rho(z), a(z))``. That distinction is worth a representation of
its own, because the two dependences are completely different in character. ``D`` is
smooth and analytic in the density and the core radius; ``rho(z)`` is what carries the
square-root cusp of a near-vacuum entrance, the stage boundaries of a fill and the
corners of a taper. Tabulating ``D`` along z pays for that structure once per state
element:

.. math::

    D(\omega; \rho, a) \approx \sum_{jk} d_{jk}(\omega)\,T_j(\hat\rho)\,T_k(\hat a),
    \qquad
    \Phi(\omega, z) = \sum_{jk} d_{jk}(\omega)\, I_{jk}(z),

with :math:`I_{jk}(z) = \int_0^z T_j(\hat\rho(z'))T_k(\hat a(z'))\,\mathrm{d}z'` a
**scalar** table along z. The expensive array-valued object is resolved where it is easy
-- one global Chebyshev box per parameter -- and the hard structure stays in the scalar
tables, which cost 8 bytes per term per coefficient instead of 16 bytes per state
element. The memory of the kernel contract's section 3 falls from
``n_intervals x n_deg x state`` to ``n_terms x state``, and evaluation
falls from two Clenshaw passes over the state to one over ``n_terms`` scalars followed
by one linear combination.

The same class serves the cumulative operator, whose weights are the integrals
:math:`I_{jk}`, and the prefactor, whose weights are the products
:math:`T_j(\hat\rho(z))T_k(\hat a(z))` themselves; ``integrate`` is the only difference
between the two builds.

**Where this does not apply.** The expansion is one global box per parameter, so it is
exact only to the truncation of an *analytic* function. A quantity that kinks in a
parameter -- the loss clamp of :class:`solprop.linear.LossPolicy` crossing its limit,
the evanescent index floor of a capillary binding partway along the range -- has no such
truncation, the degree search reaches its ceiling, and :func:`build_expansion` returns
``None``. The caller then falls back to the z table, which is the general representation
and splits at the kink as it always did. Nothing about the seam changes either way:
a backend evaluates through ``delta_phi``, ``unitary_phase`` and ``prefactor_at``.

The narrative is ``docs/guide/numerics/linear-operator.md``, "The expansion in the
parameters".
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from jaxtyping import Float, Inexact, Int
from numpy.polynomial import chebyshev as chebyshev_basis

from solprop.exceptions import TranslationError
from solprop.profiles import ZProfile
from solprop.translate.chebyshev import (
    DEFAULT_MAX_DEGREE,
    DEFAULT_MAX_INTERVALS,
    DEFAULT_TOL,
    ENVELOPE_FLOOR,
    INITIAL_DEGREE,
    SAMPLING_NOISE_FACTOR,
    PiecewiseChebyshev,
    build,
    interleave_axis,
    lobatto_points,
    refinement_points,
    resolve_axis,
    trim_axis,
)

__all__ = [
    "MAX_PARAMETERS",
    "RANGE_SAMPLES",
    "RANGE_SLACK",
    "ParameterExpansion",
    "append_endpoint_expansion",
    "build_expansion",
    "parameter_roundoff_floor",
    "reweight_expansion",
]

logger = logging.getLogger(__name__)

MAX_PARAMETERS: Final = 2
"""Most varying parameters an expansion carries before the caller must fall back.

Two is what the physics layer declares -- a core radius and a number density -- and the
tensor grid costs the product of the per-parameter node counts, so a third dimension
would be a decision rather than a detail (``docs/dev/kernel-contract.md`` section 3).
"""

RANGE_SAMPLES: Final = 4097
"""Positions sampled per declared segment when bracketing a path's range.

Every profile in :mod:`solprop.profiles` is monotone between its own breakpoints, so
the extremes are at the segment ends and the sweep is a guard rather than the method. It
is a guard worth having: a measured profile handed in through
:class:`~solprop.profiles.FromCallable` need not be monotone, and a range that misses
the values the path actually reaches would make the composition an extrapolation.
"""

RANGE_SLACK: Final = 1e-6
"""Overshoot of the mapped parameter tolerated before the composition is refused.

The range is bracketed by sampling, so a path may reach a hair beyond it. Relative to
the range, ``1e-6`` past an end changes ``T_j`` by about ``cosh(j sqrt(2e-6)) - 1``,
which is below the tail tolerance at every degree the search reaches, and the value is
clipped to the end. Anything larger means the bracket is wrong and is an error, because
a Chebyshev series must never be extrapolated.
"""

_FLOOR_STEP_FRACTION: Final = 1e-9
"""Second-difference step of :func:`parameter_roundoff_floor`, per parameter range.

The value :mod:`solprop.translate.linop` uses along z, for the same reason: the smooth
part contributes ``(1e-9)**2`` of the range-scaled second derivative, some fifteen
orders below the round-off the difference is there to isolate.
"""

_ValueAt = Callable[[Mapping[str, float]], Inexact[np.ndarray, " *value"]]


@dataclass(frozen=True, eq=False)
class ParameterExpansion:
    """A quantity as a Chebyshev expansion in its scalar paths, with weights along z.

    The value at a position is ``sum_j terms[j] * weights(z)[j]``: one Clenshaw pass
    over ``n_terms`` scalars and one linear combination over the value. Each shape
    annotation carries its own axis name because ``__post_init__`` is what enforces
    their agreement, and that check must stay reachable in production where no runtime
    type checker runs.

    Attributes
    ----------
    parameters : tuple of str
        The path names the expansion runs over, in the order of the term multi-index,
        for example ``("core_radius", "density")``.
    ranges : numpy.ndarray
        ``(lo, hi)`` per parameter in that parameter's own units, float64, shape
        ``(n_params, 2)``, strictly increasing across each row. These are the values the
        paths reach over the tabulated range.
    degrees : numpy.ndarray
        Chebyshev degree per parameter, ``numpy.intp``, shape ``(n_params,)``. The terms
        are the tensor product, flattened in C order, so this block contributes
        ``prod(degrees + 1)`` of the `n_terms` terms and `endpoint_degrees`, when
        present, contributes the rest.
    terms : numpy.ndarray
        The expansion coefficients, shape ``(n_terms, *value_shape)``. For the kernel
        contract's section 3 the value shape is ``(n_batch_or_1, n_modes, n_pol,
        n_omega)`` and the array is complex128.
    weights : solprop.translate.chebyshev.PiecewiseChebyshev
        The scalar tables along z, value shape ``(n_terms,)``, float64: the products
        ``T_j(p_hat(z))`` for a plain quantity and their integrals from the start of the
        range for a cumulative one.
    endpoint_degrees : numpy.ndarray or None, optional
        Degrees of an appended tensor block with weights equal to the plain basis
        products minus their entrance values. The first block retains integral
        weights; both blocks use the same parameter box.

    Raises
    ------
    TranslationError
        If the parameter count, the ranges, the degrees, the term count and the weights'
        value shape do not agree, or a range is not increasing.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import Linear
    >>> from solprop.translate.expansion import build_expansion
    >>> paths = {"density": Linear(1.0, 3.0, 1.0)}
    >>> expansion = build_expansion(
    ...     lambda values: np.array([values["density"] ** 2]), paths, 0.0, 1.0
    ... )
    >>> expansion.parameters, expansion.n_terms
    (('density',), 3)
    >>> round(float(expansion(0.5).item()), 12)
    4.0
    """

    parameters: tuple[str, ...]
    ranges: Float[np.ndarray, "n_params two"]
    degrees: Int[np.ndarray, " n_degrees"]
    terms: Inexact[np.ndarray, " n_terms *value"]
    weights: PiecewiseChebyshev
    endpoint_degrees: Int[np.ndarray, " n_endpoint_params"] | None = None

    def __post_init__(self) -> None:
        """Adopt read-only copies and check what the shape annotations cannot.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        for name, dtype in (("ranges", np.float64), ("degrees", np.intp)):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        terms = np.ascontiguousarray(self.terms)
        terms.setflags(write=False)
        object.__setattr__(self, "terms", terms)
        n_params = len(self.parameters)
        if self.ranges.shape != (n_params, 2) or self.degrees.size != n_params:
            msg = (
                f"{n_params} parameters {self.parameters} need ranges of shape "
                f"({n_params}, 2) and {n_params} degrees, got {self.ranges.shape} and "
                f"{self.degrees.size}"
            )
            raise TranslationError(msg)
        if n_params and not np.all(self.ranges[:, 1] > self.ranges[:, 0]):
            msg = (
                "every parameter range must be increasing; a constant path contributes "
                f"no dimension and must not be expanded. Got {self.ranges.tolist()}"
            )
            raise TranslationError(msg)
        expected = int(np.prod(self.degrees + 1))
        if self.endpoint_degrees is not None:
            endpoint = _readonly(np.asarray(self.endpoint_degrees, dtype=np.intp))
            object.__setattr__(self, "endpoint_degrees", endpoint)
            if endpoint.shape != self.degrees.shape or np.any(endpoint < 0):
                msg = (
                    "endpoint_degrees must match the parameter axes and be nonnegative"
                )
                raise TranslationError(msg)
            expected += int(np.prod(endpoint + 1))
        if self.terms.shape[0] != expected:
            msg = (
                f"degrees {self.degrees.tolist()} make {expected} terms, but terms has "
                f"{self.terms.shape[0]}"
            )
            raise TranslationError(msg)
        if self.weights.value_shape != (expected,):
            msg = (
                f"weights has value shape {self.weights.value_shape}, expected "
                f"({expected},) to match the terms"
            )
            raise TranslationError(msg)

    @property
    def n_terms(self) -> int:
        """Return the number of expansion terms.

        Returns
        -------
        int
            The length of the flattened multi-index: ``prod(degrees + 1)``, plus
            ``prod(endpoint_degrees + 1)`` for the appended endpoint block when one is
            present.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.profiles import Linear
        >>> from solprop.translate.expansion import build_expansion
        >>> build_expansion(
        ...     lambda values: np.array([values["density"]]),
        ...     {"density": Linear(1.0, 2.0, 1.0)},
        ...     0.0,
        ...     1.0,
        ... ).n_terms
        2
        """
        return int(self.terms.shape[0])

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the shape of one evaluated value.

        Returns
        -------
        tuple of int
            The trailing axes of `terms`.
        """
        return tuple(self.terms.shape[1:])

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the range of z the weights cover.

        Returns
        -------
        tuple of float
            ``(first breakpoint, last breakpoint)`` in m.
        """
        return self.weights.z_span

    @property
    def budget_exceeded(self) -> bool:
        """Return whether the weights' adaptive split ran out of intervals.

        Returns
        -------
        bool
            The flag of the scalar tables; the parameter search itself reports failure
            by returning ``None`` from :func:`build_expansion` instead.
        """
        return self.weights.budget_exceeded

    @property
    def nbytes(self) -> int:
        """Return the memory the expansion occupies.

        Returns
        -------
        int
            Bytes of the terms and of the scalar tables together.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.profiles import Linear
        >>> from solprop.translate.expansion import build_expansion
        >>> expansion = build_expansion(
        ...     lambda values: np.array([values["density"]]),
        ...     {"density": Linear(1.0, 2.0, 1.0)},
        ...     0.0,
        ...     1.0,
        ... )
        >>> expansion.nbytes > 0
        True
        """
        return int(self.terms.nbytes + self.weights.coefficients.nbytes)

    def __call__(self, z: float) -> Inexact[np.ndarray, " *value"]:
        """Evaluate the expansion at one position.

        Parameters
        ----------
        z : float
            Position in m, inside :attr:`z_span`.

        Returns
        -------
        numpy.ndarray
            The value, of shape :attr:`value_shape`.

        Raises
        ------
        TranslationError
            If `z` lies outside the weights' span.
        """
        return np.tensordot(self.weights(z), self.terms, axes=1)

    def delta(self, z1: float, z2: float) -> Inexact[np.ndarray, " *value"]:
        """Return the difference of the expansion between two positions.

        One subtraction of the scalar weights and one linear combination, rather than
        two of each: for a cumulative quantity this *is* the increment over the step.

        Parameters
        ----------
        z1, z2 : float
            Positions in m, both inside :attr:`z_span`.

        Returns
        -------
        numpy.ndarray
            ``value(z2) - value(z1)``, of shape :attr:`value_shape`.

        Raises
        ------
        TranslationError
            If a position lies outside the weights' span.
        """
        return np.tensordot(self.weights(z2) - self.weights(z1), self.terms, axes=1)


def build_expansion(  # noqa: PLR0913 - the build's knobs, as `build` has them
    value_at: _ValueAt,
    paths: Mapping[str, ZProfile],
    z0: float,
    z1: float,
    *,
    tol: float = DEFAULT_TOL,
    kinks: Sequence[float] | np.ndarray = (),
    n_max: int = DEFAULT_MAX_DEGREE,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
    integrate: bool = False,
) -> ParameterExpansion | None:
    """Expand a quantity in its scalar parameters, with weights along z.

    The parameter search is the one-dimensional search of
    :mod:`solprop.translate.chebyshev` applied to each parameter axis in turn, on one
    global box per parameter: sample the tensor grid of Chebyshev-Lobatto points, accept
    an axis when its coefficient tail is negligible *per element*, otherwise double that
    axis, reusing the samples already taken, up to `n_max`. There is no splitting in a
    parameter -- a quantity that needs one is not analytic there, and the z table is the
    representation for it.

    Parameters
    ----------
    value_at : callable
        ``values -> array``, taking one value per parameter by name and returning the
        quantity at those values. Its result must have the same shape and dtype at every
        point.
    paths : collections.abc.Mapping
        The **varying** paths, by name. A constant path contributes no dimension and
        must not be passed; its value stays inside `value_at`.
    z0, z1 : float
        Ends of the range in m, with ``z1 > z0``. The weights cover it and the parameter
        ranges are what the paths reach over it.
    tol : float, optional
        Target bound on the uniform error relative to the value scale, for both the
        parameter search and the scalar tables.
    kinks : array_like, optional
        Positions in z that must be interval edges of the scalar tables: the paths'
        declared breakpoints. Those outside the range are dropped.
    n_max : int, optional
        Highest Chebyshev degree tried on a parameter axis before the build gives up.
    max_intervals : int, optional
        Interval budget for the scalar tables.
    integrate : bool, optional
        Whether the weights are the integrals of the products from `z0`, which makes the
        expansion a cumulative quantity. Default ``False``.

    Returns
    -------
    ParameterExpansion or None
        The expansion, or ``None`` when a parameter axis is still unresolved at `n_max`
        or there are more than :data:`MAX_PARAMETERS` of them. ``None`` is the signal to
        fall back to the z table, not an error.

    Raises
    ------
    TranslationError
        If no paths are given, a path does not vary over the range, the range is empty,
        or `value_at` returns a non-finite or inconsistently shaped value.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import Linear
    >>> from solprop.translate.expansion import build_expansion
    >>> expansion = build_expansion(
    ...     lambda values: np.array([1.0, values["density"]]),
    ...     {"density": Linear(0.0, 2.0, 1.0)},
    ...     0.0,
    ...     1.0,
    ...     integrate=True,
    ... )
    >>> [round(value, 12) for value in expansion.delta(0.0, 1.0).tolist()]
    [1.0, 1.0]
    """
    if not paths:
        msg = (
            "build_expansion needs at least one varying path; a quantity with none is "
            "constant and has a cheaper representation"
        )
        raise TranslationError(msg)
    names = tuple(paths)
    if len(names) > MAX_PARAMETERS:
        logger.debug(
            "expansion: %d varying parameters %s is more than the %d the tensor grid "
            "carries; falling back to the z table",
            len(names),
            names,
            MAX_PARAMETERS,
        )
        return None
    ranges = _parameter_ranges(paths, z0, z1)
    resolved = _resolve_parameters(value_at, names, ranges, tol=tol, n_max=n_max)
    if resolved is None:
        return None
    terms, degrees = resolved
    weights = _weight_tables(
        paths,
        names,
        ranges,
        degrees,
        z0,
        z1,
        tol=tol,
        kinks=kinks,
        n_max=n_max,
        max_intervals=max_intervals,
        integrate=integrate,
    )
    expansion = ParameterExpansion(names, ranges, degrees, terms, weights)
    _log_expansion(expansion)
    return expansion


def _joined_weights(
    integral: PiecewiseChebyshev, endpoint: PiecewiseChebyshev
) -> PiecewiseChebyshev:
    """Join integral weights with endpoint differences on their common refinement.

    Parameters
    ----------
    integral : PiecewiseChebyshev
        Original integrated polynomial on its z intervals.
    endpoint : PiecewiseChebyshev
        Endpoint polynomial on the same physical span.

    Returns
    -------
    PiecewiseChebyshev
        Concatenated integral and endpoint-difference weights.
    """
    origin = endpoint(endpoint.z_span[0])

    def sample(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, "n_z n_terms"]:
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
        return np.stack(
            [
                np.concatenate((integral(float(one)), endpoint(float(one)) - origin))
                for one in z
            ]
        )

    table = build(
        sample,
        *integral.z_span,
        tol=DEFAULT_TOL,
        mandatory_breakpoints=np.union1d(integral.breakpoints, endpoint.breakpoints)[
            1:-1
        ],
    )
    return replace(
        table,
        coefficients=_readonly(table.coefficients),
        breakpoints=_readonly(table.breakpoints),
        degrees=_readonly(table.degrees),
        hard_kinks=_readonly(np.union1d(integral.hard_kinks, endpoint.hard_kinks)),
    )


def append_endpoint_expansion(
    integral: ParameterExpansion,
    value_at: _ValueAt,
    paths: Mapping[str, ZProfile],
    *,
    tol: float,
) -> ParameterExpansion | None:
    """Append endpoint differences over a cumulative expansion's parameter box.

    Parameters
    ----------
    integral : ParameterExpansion
        Cumulative expansion with integral weights and no endpoint block.
    value_at : callable
        Endpoint quantity at one point of the existing parameter box.
    paths : Mapping of str to ZProfile
        Scalar paths in the same physical units as the original expansion.
    tol : float
        Positive relative resolution tolerance for the endpoint quantity.

    Returns
    -------
    ParameterExpansion or None
        Concatenated terms and scalar weights, or None if the endpoint quantity
        cannot be resolved on the parameter box.
    """
    resolved = _resolve_parameters(
        value_at,
        integral.parameters,
        integral.ranges,
        tol=tol,
        n_max=DEFAULT_MAX_DEGREE,
    )
    if resolved is None:
        return None
    terms, degrees = resolved
    weights = _weight_tables(
        paths,
        integral.parameters,
        integral.ranges,
        degrees,
        *integral.z_span,
        tol=DEFAULT_TOL,
        kinks=integral.weights.hard_kinks,
        n_max=DEFAULT_MAX_DEGREE,
        max_intervals=DEFAULT_MAX_INTERVALS,
        integrate=False,
    )
    return ParameterExpansion(
        integral.parameters,
        integral.ranges,
        integral.degrees,
        np.concatenate((integral.terms, terms)),
        _joined_weights(integral.weights, weights),
        degrees,
    )


def parameter_roundoff_floor(
    value_at: _ValueAt,
    names: tuple[str, ...],
    ranges: Float[np.ndarray, "n_params two"],
) -> float:
    """Measure a sampler's relative round-off on a parameter box, in the tail norm.

    The parameter search accepts an axis when the coefficient tail of every element is
    small against that element's own envelope **along the axis**, the samples on the
    other axes being separate elements (``_resolve_parameters``). No degree can take
    that tail below the sampler's own noise, so a tolerance under the floor measured
    here is unreachable. A floor measured along z does not bound it: the box has
    corners the path never visits, where an element can be orders smaller than anywhere
    on the path -- a capillary's ``ln Re n_eff`` at the low-density, wide-bore corner of
    a tapered gradient, near the long-wavelength crossing of ``Re n_eff = 1`` -- while
    its absolute rounding is the same.

    The probes are the tensor grid of each range's two ends and midpoint. At each, a
    second difference of step ``_FLOOR_STEP_FRACTION`` (1e-9) of the range, taken
    inward at an end, isolates the noise; it is divided by the element's largest
    magnitude over the three probes of its line along the axis, floored at
    :data:`~solprop.translate.chebyshev.ENVELOPE_FLOOR` of the global envelope as the
    tail test floors it.

    Parameters
    ----------
    value_at : callable
        ``values -> array``, the sampler of :func:`build_expansion`, evaluated at points
        inside the box only.
    names : tuple of str
        The parameter names, in axis order.
    ranges : numpy.ndarray
        ``(lo, hi)`` per parameter, shape ``(n_params, 2)``, with ``hi > lo``.

    Returns
    -------
    float
        The largest relative noise found, dimensionless; zero for a sampler that
        vanishes at every probe.

    Examples
    --------
    A logarithm a thousand times smaller on the box's low-density line has a floor
    about a thousand times larger there, measured against that line's own values:

    >>> import numpy as np
    >>> from solprop.translate.expansion import parameter_roundoff_floor
    >>> ranges = np.array([[1e-3, 1.0], [0.0, 1.0]])
    >>> names = ("density", "radius")
    >>> def weight(values):
    ...     scaled = values["density"] * (1 + values["radius"])
    ...     return np.array([np.log(1 + 1e-4 * scaled)])
    >>> def uniform(values):
    ...     return np.array([np.log(1 + 1e-4 * (1 + values["radius"]))])
    >>> ratio = parameter_roundoff_floor(weight, names, ranges) / (
    ...     parameter_roundoff_floor(uniform, names, ranges)
    ... )
    >>> f"{ratio:.0f}"
    '1000'
    """
    n_params = len(names)
    probes = [
        [float(low), 0.5 * (float(low) + float(high)), float(high)]
        for low, high in ranges
    ]
    steps = [_FLOOR_STEP_FRACTION * float(high - low) for low, high in ranges]

    def at(point: Sequence[float]) -> Inexact[np.ndarray, " *value"]:
        """Sample the quantity at one point of the box.

        Parameters
        ----------
        point : sequence of float
            One value per parameter, in axis order.

        Returns
        -------
        numpy.ndarray
            The quantity there.
        """
        return np.asarray(value_at(dict(zip(names, point, strict=True))))

    shape = (3,) * n_params
    points = {
        index: [probes[axis][index[axis]] for axis in range(n_params)]
        for index in np.ndindex(shape)
    }
    centre = np.stack([at(points[index]) for index in np.ndindex(shape)])
    centre = centre.reshape(*shape, *centre.shape[1:])
    magnitude = np.abs(centre)
    global_envelope = float(magnitude.max())
    if global_envelope == 0.0:
        return 0.0
    # Centred at the midpoint; inward at an end, so that no probe leaves the box and
    # a vacuum end is never sampled at a negative density.
    offsets = {0: (0.0, 1.0, 2.0), 1: (-1.0, 0.0, 1.0), 2: (0.0, -1.0, -2.0)}
    worst = 0.0
    for axis in range(n_params):
        scale = np.maximum(
            magnitude.max(axis=axis, keepdims=True), global_envelope * ENVELOPE_FLOOR
        )
        noise = np.empty_like(magnitude)
        for index, point in points.items():
            stencil = []
            for offset in offsets[index[axis]]:
                shifted = list(point)
                shifted[axis] += offset * steps[axis]
                stencil.append(centre[index] if offset == 0.0 else at(shifted))
            noise[index] = np.abs(stencil[0] - 2.0 * stencil[1] + stencil[2])
        worst = max(worst, float(np.max(noise / scale)))
    return worst


def reweight_expansion(
    expansion: ParameterExpansion,
    paths: Mapping[str, ZProfile],
    *,
    integrate: bool = False,
    tol: float = DEFAULT_TOL,
    kinks: Sequence[float] | np.ndarray = (),
) -> ParameterExpansion | None:
    """Reuse coefficients for an unchanged function along covered parameter paths.

    The caller must establish that the sampled function is unchanged, including
    parameters held constant during its original construction. This function checks
    path coverage and rebuilds scalar weights; it never extrapolates coefficients.

    Parameters
    ----------
    expansion : ParameterExpansion
        Coefficients and original sampled parameter ranges.
    paths : Mapping of str to ZProfile
        New paths in each parameter's original units, covering the same z span.
    integrate : bool, optional
        Integrate weights from the start of the span, for a cumulative operator.
    tol : float, optional
        Scalar interpolation tolerance, positive and dimensionless.
    kinks : sequence of float or numpy.ndarray, optional
        Mandatory scalar-table breakpoints in m.

    Returns
    -------
    ParameterExpansion or None
        New weights sharing the exact terms array, or None when a parameter is
        missing, newly varying, or outside its original range. The reason is logged.
    """
    names = expansion.parameters
    if any(name not in paths for name in names) or any(
        not path.is_constant for name, path in paths.items() if name not in names
    ):
        logger.info("Expansion rebuild: parameter names changed from %s", names)
        return None
    z0, z1 = expansion.z_span
    selected = {name: paths[name] for name in names}
    for index, (name, path) in enumerate(selected.items()):
        if path.is_constant:
            value = float(path(np.array([z0]))[0])
            low, high = value, value
        else:
            low, high = _parameter_ranges({name: path}, z0, z1)[0]
        if low < expansion.ranges[index, 0] or high > expansion.ranges[index, 1]:
            logger.info(
                "Expansion rebuild: %s range [%g, %g] exceeds [%g, %g]",
                name,
                low,
                high,
                *expansion.ranges[index],
            )
            return None
    weights = _weight_tables(
        selected,
        names,
        expansion.ranges,
        expansion.degrees,
        z0,
        z1,
        tol=tol,
        kinks=kinks,
        n_max=DEFAULT_MAX_DEGREE,
        max_intervals=DEFAULT_MAX_INTERVALS,
        integrate=integrate,
    )
    if expansion.endpoint_degrees is not None:
        endpoint = _weight_tables(
            selected,
            names,
            expansion.ranges,
            expansion.endpoint_degrees,
            z0,
            z1,
            tol=DEFAULT_TOL,
            kinks=kinks,
            n_max=DEFAULT_MAX_DEGREE,
            max_intervals=DEFAULT_MAX_INTERVALS,
            integrate=False,
        )
        weights = _joined_weights(weights, endpoint)
    if weights.budget_exceeded:
        logger.info(
            "Expansion rebuild: replacement scalar weights exhausted their budget"
        )
        return None
    logger.debug("Reused %d expansion terms; rebuilt scalar weights", expansion.n_terms)
    return ParameterExpansion(
        names,
        expansion.ranges,
        expansion.degrees,
        expansion.terms,
        weights,
        expansion.endpoint_degrees,
    )


def _parameter_ranges(
    paths: Mapping[str, ZProfile], z0: float, z1: float
) -> Float[np.ndarray, "n_params two"]:
    """Bracket the values each path reaches over the tabulated range.

    Parameters
    ----------
    paths : collections.abc.Mapping
        The varying paths, by name.
    z0, z1 : float
        Ends of the range in m.

    Returns
    -------
    numpy.ndarray
        ``(lo, hi)`` per path in the order of `paths`, shape ``(n_params, 2)``.

    Raises
    ------
    TranslationError
        If the range is empty, or a path does not vary over it.
    """
    if not z1 > z0:
        msg = f"need z1 > z0, got z0 = {z0!r}, z1 = {z1!r}"
        raise TranslationError(msg)
    rows = []
    for name, path in paths.items():
        edges = [
            float(value)
            for value in np.atleast_1d(np.asarray(path.breakpoints, dtype=np.float64))
            if z0 < float(value) < z1
        ]
        positions = np.unique(
            np.concatenate(
                [
                    np.linspace(lo, hi, RANGE_SAMPLES)
                    for lo, hi in zip([z0, *edges], [*edges, z1], strict=True)
                ]
            )
        )
        values = np.asarray(path(positions), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            msg = f"the path {name!r} is not finite over [{z0}, {z1}] m"
            raise TranslationError(msg)
        low, high = float(values.min()), float(values.max())
        if not high > low:
            msg = (
                f"the path {name!r} does not vary over [{z0}, {z1}] m, holding "
                f"{low!r}; a constant path contributes no dimension"
            )
            raise TranslationError(msg)
        rows.append((low, high))
    return np.array(rows, dtype=np.float64)


def _resolve_parameters(
    value_at: _ValueAt,
    names: tuple[str, ...],
    ranges: Float[np.ndarray, "n_params two"],
    *,
    tol: float,
    n_max: int,
) -> tuple[Inexact[np.ndarray, " n_terms *value"], Int[np.ndarray, " n_params"]] | None:
    """Raise the degree of each parameter axis until every coefficient tail is small.

    Parameters
    ----------
    value_at : callable
        The sampler.
    names : tuple of str
        The parameter names, in axis order.
    ranges : numpy.ndarray
        ``(lo, hi)`` per parameter.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried on an axis.

    Returns
    -------
    tuple or None
        The flattened terms and the trimmed degree per parameter, or ``None`` when an
        axis is still unresolved at `n_max`.
    """
    n_params = len(names)
    degrees = [min(INITIAL_DEGREE, n_max) for _ in names]
    points = [
        lobatto_points(degree, *ranges[axis]) for axis, degree in enumerate(degrees)
    ]
    values = _sample_grid(value_at, names, points)
    while True:
        accepted = [resolve_axis(values, axis, tol=tol)[1] for axis in range(n_params)]
        if all(accepted):
            break
        axis = next(
            (
                index
                for index in range(n_params)
                if not accepted[index] and 2 * degrees[index] <= n_max
            ),
            None,
        )
        if axis is None:
            logger.debug(
                "expansion: %s is not resolved to %.1e at degree %s, the ceiling of "
                "%d; falling back to the z table",
                [name for name, ok in zip(names, accepted, strict=True) if not ok],
                tol,
                degrees,
                n_max,
            )
            return None
        fine = list(points)
        fine[axis] = refinement_points(degrees[axis], *ranges[axis])
        values = interleave_axis(values, _sample_grid(value_at, names, fine), axis)
        degrees[axis] *= 2
        points[axis] = lobatto_points(degrees[axis], *ranges[axis])
    # Transform and trim one axis at a time, in order, so that the trim of each axis
    # sees the others as values rather than as coefficients. `trim_axis` compares each
    # trailing element against its own envelope with a floor: a coefficient column that
    # is already negligible would be given a negligible comparison scale and would then
    # hold the trim open on noise, which is exactly what the tail test avoids by the
    # same means.
    coefficients = values
    for axis in range(n_params):
        coefficients, _ = resolve_axis(coefficients, axis, tol=tol)
        coefficients = trim_axis(coefficients, axis)
    kept = np.array(
        [coefficients.shape[axis] - 1 for axis in range(n_params)], dtype=np.intp
    )
    terms = coefficients.reshape(
        int(np.prod(coefficients.shape[:n_params])), *coefficients.shape[n_params:]
    )
    return np.ascontiguousarray(terms), kept


def _sample_grid(
    value_at: _ValueAt,
    names: tuple[str, ...],
    # Each axis has its own node count, so the entries share no axis name: "_" is
    # jaxtyping's anonymous dimension, which binds nothing across the sequence.
    points: Sequence[Float[np.ndarray, " _"]],
) -> Inexact[np.ndarray, " *shape"]:
    """Evaluate the quantity at every point of a tensor grid of parameter values.

    Parameters
    ----------
    value_at : callable
        The sampler.
    names : tuple of str
        The parameter names, in axis order.
    points : sequence of numpy.ndarray
        The nodes of each parameter axis.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(*node counts, *value_shape)``.

    Raises
    ------
    TranslationError
        If a sample is not finite, or the value shape or dtype changes between points.
    """
    shape = tuple(int(axis.size) for axis in points)
    samples = []
    for index in np.ndindex(shape):
        values = {
            name: float(points[axis][index[axis]]) for axis, name in enumerate(names)
        }
        sample = np.asarray(value_at(values))
        if samples and sample.shape != samples[0].shape:
            msg = (
                f"the sampler returned shape {sample.shape} at {values}, having "
                f"returned {samples[0].shape} before"
            )
            raise TranslationError(msg)
        if not np.all(np.isfinite(sample)):
            msg = f"the sampler returned a non-finite value at {values}"
            raise TranslationError(msg)
        samples.append(sample)
    stacked = np.stack(samples)
    return stacked.reshape(*shape, *stacked.shape[1:])


def _weight_tables(  # noqa: PLR0913 - one call site; every argument is a distinct choice
    paths: Mapping[str, ZProfile],
    names: tuple[str, ...],
    ranges: Float[np.ndarray, "n_params two"],
    degrees: Int[np.ndarray, " n_params"],
    z0: float,
    z1: float,
    *,
    tol: float,
    kinks: Sequence[float] | np.ndarray,
    n_max: int,
    max_intervals: int,
    integrate: bool,
) -> PiecewiseChebyshev:
    """Tabulate the products of the basis polynomials along the paths.

    Parameters
    ----------
    paths : collections.abc.Mapping
        The varying paths, by name.
    names : tuple of str
        The parameter names, in axis order.
    ranges : numpy.ndarray
        ``(lo, hi)`` per parameter.
    degrees : numpy.ndarray
        The retained degree per parameter.
    z0, z1 : float
        Ends of the range in m.
    tol : float
        Target bound on the uniform relative error.
    kinks : array_like
        Positions that must be interval edges; those outside the range are dropped.
    n_max : int
        Highest degree tried on an interval.
    max_intervals : int
        Interval budget.
    integrate : bool
        Whether to return the cumulative integrals from `z0`.

    Returns
    -------
    solprop.translate.chebyshev.PiecewiseChebyshev
        Value shape ``(n_terms,)``, float64, with read-only arrays.
    """

    def products(
        z: Float[np.ndarray, " n_z"],
    ) -> Float[np.ndarray, "n_z n_terms"]:
        """Return the tensor product of the basis polynomials at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_terms)``, the multi-index flattened in C order.
        """
        columns = [
            chebyshev_basis.chebvander(
                _mapped(paths[name](z), ranges[axis], name), int(degrees[axis])
            )
            for axis, name in enumerate(names)
        ]
        table = columns[0]
        for column in columns[1:]:
            table = (table[:, :, np.newaxis] * column[:, np.newaxis, :]).reshape(
                z.size, -1
            )
        return table

    edges = sorted(
        {
            float(value)
            for value in np.atleast_1d(np.asarray(kinks, dtype=np.float64))
            if z0 < float(value) < z1
        }
    )
    table = build(
        products,
        z0,
        z1,
        tol=max(tol, _mapping_noise(ranges)),
        mandatory_breakpoints=edges,
        n_max=n_max,
        max_intervals=max_intervals,
    )
    if integrate:
        table = table.integral()
    return PiecewiseChebyshev(
        _readonly(table.coefficients),
        _readonly(table.breakpoints),
        _readonly(table.degrees),
        _readonly(table.hard_kinks),
        table.budget_exceeded,
    )


def _mapping_noise(ranges: Float[np.ndarray, "n_params two"]) -> float:
    """Return the acceptance floor the rounding of the mapped paths sets.

    :func:`_mapped` divides by half the expansion range, so a path value ``v`` rounded
    to ``eps |v|`` moves its mapped value by ``2 eps |v| / (hi - lo)``. Over a narrow
    range (a short window of a gentle gradient) that is far above ``eps``: a density
    varying by 0.7 % maps with about 300 eps of noise, and no split can resolve the
    products below it. The floor is that noise times
    :data:`solprop.translate.chebyshev.SAMPLING_NOISE_FACTOR`, the margin the build
    applies to the noise of its node positions; it stays below
    :data:`solprop.translate.chebyshev.DEFAULT_TOL` unless a range is narrower than
    about a thousandth of its values.

    Parameters
    ----------
    ranges : numpy.ndarray
        ``(lo, hi)`` per parameter.

    Returns
    -------
    float
        The floor, relative to the products' scale; zero when every range is empty.
    """
    low, high = ranges[:, 0], ranges[:, 1]
    width = high - low
    reach = np.maximum(np.abs(low), np.abs(high))
    amplification = np.divide(
        2.0 * reach, width, out=np.zeros_like(reach), where=width > 0.0
    )
    eps = float(np.finfo(np.float64).eps)
    return SAMPLING_NOISE_FACTOR * eps * float(np.max(amplification, initial=0.0))


def _mapped(
    values: Float[np.ndarray, " n_z"],
    span: Float[np.ndarray, " two"],
    name: str,
) -> Float[np.ndarray, " n_z"]:
    """Map a path's values onto ``[-1, 1]`` over its expansion range.

    Parameters
    ----------
    values : numpy.ndarray
        The path at some positions.
    span : numpy.ndarray
        ``(lo, hi)`` of the expansion range.
    name : str
        The path's name, for the message.

    Returns
    -------
    numpy.ndarray
        The mapped values, clipped to ``[-1, 1]``.

    Raises
    ------
    TranslationError
        If a value lies more than :data:`RANGE_SLACK` of the range outside it, which
        means the bracket is wrong and the series would be extrapolated.
    """
    low, high = float(span[0]), float(span[1])
    mapped = (2.0 * np.asarray(values, dtype=np.float64) - low - high) / (high - low)
    worst = float(np.max(np.abs(mapped)))
    if worst > 1.0 + RANGE_SLACK:
        msg = (
            f"the path {name!r} reaches {worst:.6g} in units of half its expansion "
            f"range [{low!r}, {high!r}], which is beyond the {RANGE_SLACK:.0e} slack; "
            "the range does not bracket the path and the series would be extrapolated"
        )
        raise TranslationError(msg)
    return np.clip(mapped, -1.0, 1.0)


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


def _log_expansion(expansion: ParameterExpansion) -> None:
    """Log the expansion's parameters, degrees and size at DEBUG level.

    Parameters
    ----------
    expansion : ParameterExpansion
        The expansion just built.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "expansion in %s over %s: degrees %s, %d terms, %d scalar intervals, %.3f MB, "
        "budget exceeded %s",
        expansion.parameters,
        [(f"{low:.6g}", f"{high:.6g}") for low, high in expansion.ranges.tolist()],
        expansion.degrees.tolist(),
        expansion.n_terms,
        expansion.weights.degrees.size,
        expansion.nbytes / 2**20,
        expansion.budget_exceeded,
    )
