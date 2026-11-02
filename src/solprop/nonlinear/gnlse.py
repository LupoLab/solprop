# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The GNLSE as a mode-averaged envelope problem.

The generalised nonlinear Schroedinger equation is not a separate solver path in
solprop: the field is primary and the GNLSE is an interface mapping. It is the
mode-averaged envelope problem with a **specified mode**
(:class:`SpecifiedMode`) — a mode whose propagation constant is given directly, as a
Taylor series, a table, or a declared path of either along z, rather than derived from
a waveguide geometry — plus a mapping from the usual GNLSE coefficient ``gamma`` onto
the third-order susceptibility the Kerr response takes. The shock term is not a
separate operator either: it is the ``omega`` dependence of the normalisation
prefactor (:mod:`solprop.nonlinear.normalisation`).

When Raman is off and no fraction is given the **whole** of ``gamma`` acts as Kerr, so
an "N = 6" soliton is N = 6. A fraction given explicitly is always held back from the
Kerr term, with Raman on or off, so that switching Raman alone adds the delayed response
without changing the instantaneous one.

The derivation and a worked example are in ``docs/guide/waveguides/gnlse.md``.
"""

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float
from scipy import interpolate

from solprop.constants import EPS0, C
from solprop.exceptions import ModeError, SurrogateAreaWarning
from solprop.modes.base import alpha, beta, beta1
from solprop.modes.protocol import (
    GEOMETRY_PATH_NAMES,
    ChannelLabel,
    CrossSection,
    Mode,
    ParameterisedMode,
    PolarisationComponent,
)
from solprop.modes.tabulated import MIN_TABLE_POINTS, IndexTable
from solprop.nonlinear.raman import RamanResponse, bundle_response
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import chi3_from_n2
from solprop.physdata.glasses import SilicaBundle
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import FromCallable, PiecewiseLinear, ZProfile, as_profile

__all__ = [
    "MIN_TABLE_POINTS",
    "ConstantDispersion",
    "DispersionFunction",
    "DispersionPath",
    "GnlseProblem",
    "InterpolatedDispersion",
    "LossFunction",
    "SpecifiedMode",
    "TabulatedDispersion",
    "TaylorDispersion",
    "as_dispersion_path",
    "attenuation_from_db",
    "kerr_susceptibility",
    "problem_from",
    "raman_susceptibility",
]

logger = logging.getLogger(__name__)

DispersionFunction = Callable[
    [Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]
]
"""A propagation constant, or a derivative of one, as a function of frequency."""

LossFunction = Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]]
"""A power attenuation coefficient in 1/m as a function of angular frequency."""


@runtime_checkable
class DispersionPath(Protocol):
    """A spectral quantity that follows a declared path along ``z``.

    The spectral sibling of :class:`solprop.profiles.ZProfile`: where a ``ZProfile``
    carries one scalar along the fibre, this carries a whole frequency axis --
    ``beta(omega, z)``, ``d beta / d omega (omega, z)`` or ``alpha(omega, z)``. The
    three members below are the ones the translation layer reads: the
    **breakpoints** become the hard kinks the schedule lands on, the **span** is what
    a run is checked against, and **is_constant** is what keeps a fibre that does not
    move on the exact constant-operator path.

    A mode built on one is z-varying by construction, and
    :func:`solprop.translate.linop.build_phi` resolves it along z exactly as it
    resolves a gas gradient: the operator is sampled at positions, the declared
    breakpoints are mandatory interval edges, and the Chebyshev integral is exact in
    between (``docs/guide/numerics/linear-operator.md``).

    Implementations here are :class:`ConstantDispersion`, which adapts a
    :data:`DispersionFunction` of frequency alone, and
    :class:`InterpolatedDispersion`, which interpolates declared sets. Anything with
    these four members is accepted: the protocol is runtime-checkable and
    :func:`as_dispersion_path` is the one coercion.
    """

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the positions in m where the path kinks; may be empty."""
        ...

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval in m the path is declared over, ``(start, end)``."""
        ...

    @property
    def is_constant(self) -> bool:
        """Return whether the path takes the same value at every position."""
        ...

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the quantity on a frequency axis at one position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position along the fibre, m.

        Returns
        -------
        numpy.ndarray
            The value at those frequencies and that position, shape ``(n_omega,)``.
        """
        ...


@dataclass(frozen=True)
class ConstantDispersion:
    """Adapt a frequency-only dispersion or loss to the path interface.

    The counterpart of :class:`solprop.profiles.Constant` for a spectral quantity:
    it ignores ``z``, declares no breakpoints and reports itself constant, so a
    fibre whose dispersion does not move stays on the exact constant-operator path
    and its arithmetic is unchanged -- the wrapped function is called with exactly
    the argument it was called with before.

    Attributes
    ----------
    function : callable
        ``f(omega)`` returning an array of the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.gnlse import ConstantDispersion, TaylorDispersion
    >>> path = ConstantDispersion(TaylorDispersion([1.0e7], 2.0e15))
    >>> path.is_constant, path.breakpoints.size
    (True, 0)
    >>> float(path(np.array([2.0e15]), 3.0)[0])
    10000000.0
    """

    function: DispersionFunction

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the empty set of breakpoints.

        Returns
        -------
        numpy.ndarray
            An empty float array: a constant path kinks nowhere.
        """
        return np.empty(0, dtype=float)

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the path covers.

        Returns
        -------
        tuple of float
            ``(0, inf)``: a constant path covers any run.
        """
        return 0.0, math.inf

    @property
    def is_constant(self) -> bool:
        """Return whether the path varies along z.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the wrapped function, ignoring the position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position along the fibre, m; ignored.

        Returns
        -------
        numpy.ndarray
            The wrapped function's value, shape ``(n_omega,)``.
        """
        del z
        return self.function(omega)


@dataclass(frozen=True)
class _UniformLoss:
    """Keep a constant power attenuation reproducible as a callable of frequency."""

    value: float

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the same attenuation at every frequency.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            ``alpha`` in 1/m, shape ``(n_omega,)``.
        """
        return np.full(omega.shape, self.value)


@dataclass(frozen=True, eq=False)
class InterpolatedDispersion:
    """``beta(omega, z)`` from declared sets at declared positions, linear in z.

    The one mechanism a z-varying specified fibre runs through. Each member is an
    ordinary frequency-only dispersion -- a :class:`TaylorDispersion` expansion set,
    a :class:`TabulatedDispersion` column of an ``n_eff(omega, z)`` table, or any
    callable -- and between two positions the value is the straight line

    ``beta(omega, z) = (1 - t) beta_i(omega) + t beta_{i+1}(omega)``,
    ``t = (z - z_i) / (z_{i+1} - z_i)``,

    which is exact at every declared position and reproduces the members there
    bit for bit. Outside the declared range the value is held at the nearest end,
    as a :class:`solprop.profiles.PiecewiseLinear` scalar path is: that covers the
    few per cent of margin the cumulative operator tabulates beyond ``z_max``, while
    a run **longer** than the declared range is refused by name by
    :func:`solprop.translate.linop.build_phi` rather than silently extrapolated.

    A kink is a declared position and nothing else, which is why the positions are
    the :attr:`breakpoints`: the linear interpolant is smooth in between, so the
    adaptive Chebyshev split resolves each section at degree one and the integral of
    the operator over it is exact.

    Equality is by identity (``eq=False``) because the positions are an array.

    Attributes
    ----------
    positions : numpy.ndarray
        Strictly increasing positions in m, shape ``(n_positions,)``, at least two
        and all finite. Adopted as a read-only float64 array, so that two runs
        specified the same way record the same positions and serialise alike.
    members : tuple of callable
        One dispersion per position, in the same order. A sequence is adopted as a
        tuple.

    Raises
    ------
    ModeError
        If fewer than two positions are given, they are not strictly increasing or
        finite, or the number of members does not match.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.gnlse import InterpolatedDispersion, TaylorDispersion
    >>> entrance = TaylorDispersion([0.0, 5.0e-9], 2.0e15)
    >>> exit_set = TaylorDispersion([0.0, 7.0e-9], 2.0e15)
    >>> path = InterpolatedDispersion([0.0, 1.0], (entrance, exit_set))
    >>> float(path(np.array([3.0e15]), 0.5)[0])
    6000000.0
    """

    positions: Float[np.ndarray, " n_positions"] | Sequence[float]
    members: tuple[DispersionFunction, ...] | Sequence[DispersionFunction]

    def __post_init__(self) -> None:
        """Check the positions and adopt them as a read-only array.

        Raises
        ------
        ModeError
            Naming what is wrong with the declaration.
        """
        nodes = np.ascontiguousarray(self.positions, dtype=np.float64).reshape(-1)
        members = tuple(self.members)
        if nodes.size < 2:
            msg = (
                "a dispersion path needs at least two declared positions; a fibre "
                f"whose dispersion does not move takes a plain function, got {nodes!r}"
            )
            raise ModeError(msg)
        if not (np.all(np.isfinite(nodes)) and np.all(np.diff(nodes) > 0.0)):
            msg = f"the declared positions must be finite and increasing, got {nodes!r}"
            raise ModeError(msg)
        if len(members) != nodes.size:
            msg = (
                f"{len(members)} dispersion sets were declared at {nodes.size} "
                "positions: a path carries one set per position"
            )
            raise ModeError(msg)
        nodes.setflags(write=False)
        object.__setattr__(self, "positions", nodes)
        object.__setattr__(self, "members", members)

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the declared positions, every one of which is a kink.

        Returns
        -------
        numpy.ndarray
            The positions in m, shape ``(n_positions,)``.
        """
        return np.asarray(self.positions, dtype=float)

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the declared sets cover.

        Returns
        -------
        tuple of float
            ``(first position, last position)`` in m.
        """
        nodes = np.asarray(self.positions, dtype=float)
        return float(nodes[0]), float(nodes[-1])

    @property
    def is_constant(self) -> bool:
        """Return whether every position declares the same set.

        Returns
        -------
        bool
            ``True`` only when one object was declared at every position, which is
            the conservative test an arbitrary callable admits.
        """
        members = tuple(self.members)
        return all(one is members[0] for one in members[1:])

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Interpolate the declared sets at one position.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position along the fibre, m; held at the nearest end outside the
            declared range.

        Returns
        -------
        numpy.ndarray
            The interpolated value, shape ``(n_omega,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import (
        ...     InterpolatedDispersion,
        ...     TaylorDispersion,
        ... )
        >>> path = InterpolatedDispersion(
        ...     [0.0, 1.0],
        ...     (TaylorDispersion([1.0], 1.0), TaylorDispersion([3.0], 1.0)),
        ... )
        >>> [float(path(np.array([1.0]), z)[0]) for z in (-1.0, 0.25, 2.0)]
        [1.0, 1.5, 3.0]
        """
        nodes = np.asarray(self.positions, dtype=float)
        members = tuple(self.members)
        lower = int(
            np.clip(np.searchsorted(nodes, z, side="right") - 1, 0, nodes.size - 2)
        )
        width = nodes[lower + 1] - nodes[lower]
        fraction = float(np.clip((float(z) - nodes[lower]) / width, 0.0, 1.0))
        if fraction == 0.0:
            return members[lower](omega)
        if fraction == 1.0:
            return members[lower + 1](omega)
        return (1.0 - fraction) * members[lower](omega) + fraction * members[lower + 1](
            omega
        )


def as_dispersion_path(value: DispersionFunction | DispersionPath) -> DispersionPath:
    """Coerce a frequency-only dispersion to a path, and pass a path through.

    The spectral sibling of :func:`solprop.profiles.as_profile`, and the one
    coercion: a :class:`SpecifiedMode` asks for its dispersion, its derivative and
    its loss through this, so that every one of them answers the same four
    questions whether it moves along z or not.

    Parameters
    ----------
    value : callable or DispersionPath
        A function of frequency alone, or a declared path already.

    Returns
    -------
    DispersionPath
        The path, a :class:`ConstantDispersion` when the argument is a plain
        function.

    Raises
    ------
    ModeError
        If a :class:`solprop.profiles.ZProfile` is given. Its members are a
        superset of this protocol's, so it passes the structural test and would
        then fail inside :meth:`SpecifiedMode.beta` with a raw ``TypeError``; the
        two are different things and the refusal belongs at the boundary.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import TaylorDispersion, as_dispersion_path
    >>> as_dispersion_path(TaylorDispersion([1.0e7], 2.0e15)).is_constant
    True
    """
    if isinstance(value, ZProfile):
        msg = (
            f"{type(value).__name__} is a ZProfile, which carries one scalar along "
            "z; a dispersion path carries a whole frequency axis at every position. "
            "Declare the dispersion with InterpolatedDispersion, or a fixed one with "
            f"ConstantDispersion or a plain function of omega, got {value!r}"
        )
        raise ModeError(msg)
    if isinstance(value, DispersionPath):
        return value
    return ConstantDispersion(value)


def attenuation_from_db(loss_db_per_m: float) -> float:
    """Convert a power loss in dB/m to a power attenuation coefficient in 1/m.

    ``alpha = ln(10)/10 * loss``, so that the power falls as ``exp(-alpha z)`` and the
    field amplitude as ``exp(-alpha z / 2)``, which is the ``-alpha/2`` of the linear
    operator (``docs/dev/conventions.md`` section 2).

    Parameters
    ----------
    loss_db_per_m : float
        Power loss in dB/m. Negative values describe gain and are permitted.

    Returns
    -------
    float
        Power attenuation coefficient in 1/m.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import attenuation_from_db
    >>> f"{attenuation_from_db(10.0):.6f}"
    '2.302585'
    """
    return math.log(10.0) / 10.0 * loss_db_per_m


def kerr_susceptibility(n2: float, raman_fraction: float = 0.0) -> float:
    """Return the Kerr susceptibility of the GNLSE mapping.

    ``chi3 = (4/3) (1 - f_R) n2 eps0 c``: the
    ``4/3`` undoes the ``3/4`` of the envelope Kerr response, and the linear index is
    taken as one, which is :func:`solprop.physdata.nonlinear_data.chi3_from_n2` with a
    unit index.
    ``raman_fraction`` is the share of the nonlinearity a Raman response carries;
    :func:`problem_from` passes zero when Raman is off, and the share it does pass goes
    to a response of integral :func:`raman_susceptibility` times that fraction, so that
    the two together carry the whole of ``gamma``.

    Parameters
    ----------
    n2 : float
        Nonlinear index in m^2/W.
    raman_fraction : float, optional
        Share of the nonlinearity reserved for Raman, in ``[0, 1]``. Default 0.

    Returns
    -------
    float
        Third-order susceptibility in m^2/V^2.

    Raises
    ------
    ModeError
        If ``raman_fraction`` lies outside ``[0, 1]``.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import kerr_susceptibility
    >>> f"{kerr_susceptibility(2.6e-20):.6e}"
    '9.201985e-23'
    >>> f"{kerr_susceptibility(2.6e-20, 0.18) / kerr_susceptibility(2.6e-20):.2f}"
    '0.82'
    """
    if not 0.0 <= raman_fraction <= 1.0:
        msg = f"raman_fraction must lie in [0, 1], got {raman_fraction!r}"
        raise ModeError(msg)
    return (1.0 - raman_fraction) * chi3_from_n2(n2, 1.0)


def raman_susceptibility(n2: float) -> float:
    r"""Return the Raman strength of the GNLSE mapping, as an integrated response.

    A Raman response is stated by its integral: ``integral h dt`` carries the units of
    :attr:`solprop.nonlinear.responses.KerrResponse.chi3` per unit density
    (``docs/guide/media/raman-numerics.md``), so it is directly comparable with the Kerr
    susceptibility. The two are not *equal* for the same share of the nonlinearity,
    because the envelope forms differ by their numerical factors:

    .. math::

        P_\text{Kerr} = \tfrac34\,\varepsilon_0\chi^{(3)}\lvert A\rvert^2 A,
        \qquad
        P_\text{Raman} \to \tfrac12\,\varepsilon_0
        \left(\textstyle\int h\,\mathrm{d}t\right)\lvert A\rvert^2 A

    in the slowly varying limit, so a Raman term of integral ``X`` acts like a Kerr term
    of ``(2/3) X`` and carrying the whole nonlinearity needs
    ``X = (3/2) chi3_total = 2 n2 eps0 c``. A response built with this amplitude and a
    ``fraction`` of ``f_R`` therefore takes exactly the share
    :func:`kerr_susceptibility` leaves behind, and

    ``(2/3) f_R raman_susceptibility(n2) + kerr_susceptibility(n2, f_R) ==
    kerr_susceptibility(n2)``

    holds by construction, which is a test. Because the response function here is
    linear in ``eps0``, no further ``eps0`` appears; a convention with a response
    quadratic in ``eps0`` writes the same number as ``chi3R = 2 f_R n2 eps0 c``.

    Parameters
    ----------
    n2 : float
        Nonlinear index in m^2/W.

    Returns
    -------
    float
        The value ``integral h dt`` takes when the **whole** nonlinearity is Raman, in
        m^2/V^2; a :class:`solprop.nonlinear.raman.RamanResponse` scales it by its own
        ``fraction``.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import kerr_susceptibility, raman_susceptibility
    >>> f"{raman_susceptibility(2.6e-20):.6e}"
    '1.380298e-22'
    >>> total = kerr_susceptibility(2.6e-20)
    >>> share = (2 / 3) * 0.18 * raman_susceptibility(2.6e-20)
    >>> bool(abs(share + kerr_susceptibility(2.6e-20, 0.18) - total) < 1e-36)
    True
    """
    return 2.0 * n2 * EPS0 * C


@dataclass(frozen=True, eq=False)
class TaylorDispersion:
    """``beta(omega)``, or a derivative of it, as a Taylor series about a frequency.

    ``beta(omega) = sum_n betas[n] (omega - omega_ref)**n / n!``, so ``betas[n]`` is the
    n-th derivative of ``beta`` at the reference frequency and ``betas[1]`` is the
    inverse group velocity there. Differentiation is exact, in the polynomial basis,
    never by differencing, as the co-moving frame requires.

    The instance is callable and stands where a closure would. That is what makes a mode
    built from it **reproducible in a run's metadata**:
    :func:`solprop.output.schema.serialise` records a closure as its ``repr``, which
    carries the function's memory address, so a configuration holding one hashes
    differently in every process and the run could never be resumed from its own file.
    A frozen dataclass records its fields instead.

    Equality is by identity (``eq=False``) because ``betas`` is an array.

    Attributes
    ----------
    betas : numpy.ndarray
        Taylor coefficients ``beta_0, beta_1, ...`` in s^n/m, shape ``(n_deg,)``, at
        least one and all finite. A sequence is adopted as a read-only float64 array, so
        that two runs specified the same way record the same coefficients whichever type
        they passed and therefore hash alike.
    omega_ref : float
        Frequency the series is expanded about, rad/s, positive.
    derivative : int
        How many times to differentiate before evaluating: 0 for ``beta`` itself, 1 for
        ``d beta / d omega``. Non-negative.
    polynomial : numpy.polynomial.Polynomial
        The differentiated series in the shifted variable ``omega - omega_ref``, built
        once in ``__post_init__``. Derived, so it is neither a constructor argument nor
        part of what a run records.

    Raises
    ------
    ModeError
        If no coefficients are given, one is not finite, ``omega_ref`` is not positive,
        or ``derivative`` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.gnlse import TaylorDispersion
    >>> beta = TaylorDispersion([1.0e7, 5.0e-9, -2.0e-26], 2.0e15)
    >>> f"{float(beta(np.array([2.0e15]))[0]):.6e}"
    '1.000000e+07'
    >>> gradient = TaylorDispersion([1.0e7, 5.0e-9], 2.0e15, 1)
    >>> f"{float(gradient(np.array([3.0e15]))[0]):.1e}"
    '5.0e-09'
    """

    betas: Float[np.ndarray, " n_deg"] | Sequence[float]
    omega_ref: float
    derivative: int = 0
    polynomial: np.polynomial.Polynomial = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check the series and build the differentiated polynomial once.

        Raises
        ------
        ModeError
            Naming what is wrong with the series.
        """
        coefficients = np.ascontiguousarray(self.betas, dtype=np.float64).reshape(-1)
        if coefficients.size == 0:
            msg = "betas must hold at least the zeroth-order coefficient"
            raise ModeError(msg)
        if not np.all(np.isfinite(coefficients)):
            msg = f"every Taylor coefficient must be finite, got {self.betas!r}"
            raise ModeError(msg)
        if not self.omega_ref > 0.0:
            msg = f"omega_ref must be positive, got {self.omega_ref!r}"
            raise ModeError(msg)
        if self.derivative < 0:
            msg = f"derivative must be a non-negative order, got {self.derivative!r}"
            raise ModeError(msg)
        coefficients.setflags(write=False)
        object.__setattr__(self, "betas", coefficients)
        orders = np.arange(coefficients.size)
        # betas[n] is the n-th derivative at omega_ref, so the polynomial coefficient is
        # betas[n] / n!; the series is then evaluated in the shifted variable.
        series = np.polynomial.Polynomial(
            coefficients / np.vectorize(math.factorial)(orders)
        )
        object.__setattr__(self, "polynomial", series.deriv(self.derivative))

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the series, or its derivative, at the given frequencies.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            The value in s to the ``derivative`` per m, shape ``(n_omega,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import TaylorDispersion
        >>> TaylorDispersion([0.0, 5.0e-9], 2.0e15)(np.array([2.0e15, 3.0e15]))
        array([      0., 5000000.])
        """
        return np.asarray(self.polynomial(omega - self.omega_ref), dtype=float)


@dataclass(frozen=True, eq=False)
class TabulatedDispersion:
    """``beta(omega)``, or a derivative of it, from a table, by cubic spline.

    A cubic spline through the table, differentiated exactly in the spline basis, as the
    co-moving frame requires. Extrapolation is refused rather than silently produced: a
    spline dispersion evaluated outside the data it was built from is the classic way to
    get a plausible, wrong answer.

    Callable and frozen for the reason :class:`TaylorDispersion` gives: a mode built
    from it records its table, not a closure's memory address, so a run can be resumed
    from its own file in another process.

    Equality is by identity (``eq=False``) because the table is an array.

    Attributes
    ----------
    omega : numpy.ndarray
        Strictly increasing angular frequencies, rad/s, shape ``(n_nodes,)``, at least
        :data:`solprop.modes.tabulated.MIN_TABLE_POINTS` of them.
    beta : numpy.ndarray
        Propagation constant at those frequencies, 1/m, the same shape. The two arrays
        carry independent axis names on purpose: their agreement is what
        ``__post_init__`` checks, and that check must stay reachable in production,
        where the test session's runtime type checker is absent.
    derivative : int
        How many times to differentiate before evaluating: 0 for ``beta`` itself, 1 for
        ``d beta / d omega``. Non-negative.
    spline : scipy.interpolate.CubicSpline
        The differentiated spline, built once in ``__post_init__``. Derived, so it is
        neither a constructor argument nor part of what a run records.

    Raises
    ------
    ModeError
        If the two arrays disagree in shape, the table is too short, its frequencies are
        not strictly increasing, or ``derivative`` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.gnlse import TabulatedDispersion
    >>> omega = np.linspace(1.0e15, 3.0e15, 8)
    >>> beta = TabulatedDispersion(omega, 5.0e-9 * omega)
    >>> f"{float(beta(np.array([2.0e15]))[0]):.6f}"
    '10000000.000000'
    """

    omega: Float[np.ndarray, " n_nodes"]
    beta: Float[np.ndarray, " n_beta_nodes"]
    derivative: int = 0
    spline: interpolate.CubicSpline = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check the table and build the differentiated spline once.

        Raises
        ------
        ModeError
            Naming what is wrong with the table.
        """
        nodes = np.ascontiguousarray(self.omega, dtype=np.float64)
        values = np.ascontiguousarray(self.beta, dtype=np.float64)
        if nodes.shape != values.shape:
            msg = f"omega and beta have shapes {nodes.shape} and {values.shape}"
            raise ModeError(msg)
        if nodes.size < MIN_TABLE_POINTS:
            msg = (
                f"a cubic spline needs at least {MIN_TABLE_POINTS} points, got "
                f"{nodes.size}"
            )
            raise ModeError(msg)
        if not np.all(np.diff(nodes) > 0.0):
            msg = "the tabulated frequencies must be strictly increasing"
            raise ModeError(msg)
        if self.derivative < 0:
            msg = f"derivative must be a non-negative order, got {self.derivative!r}"
            raise ModeError(msg)
        for name, array in (("omega", nodes), ("beta", values)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        spline = interpolate.CubicSpline(nodes, values, extrapolate=False)
        object.__setattr__(
            self,
            "spline",
            spline if self.derivative == 0 else spline.derivative(self.derivative),
        )

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the spline, or its derivative, refusing to extrapolate.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, inside the tabulated
            range.

        Returns
        -------
        numpy.ndarray
            The value in s to the ``derivative`` per m, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If any frequency lies outside the table.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import TabulatedDispersion
        >>> omega = np.linspace(1.0e15, 3.0e15, 8)
        >>> TabulatedDispersion(omega, 5.0e-9 * omega, 1)(np.array([2.0e15]))
        array([5.e-09])
        """
        lower, upper = float(self.omega[0]), float(self.omega[-1])
        if omega.size and (omega.min() < lower or omega.max() > upper):
            msg = (
                f"omega runs from {omega.min():.6e} to {omega.max():.6e} rad/s, "
                f"outside the tabulated range {lower:.6e} to {upper:.6e} rad/s"
            )
            raise ModeError(msg)
        return np.asarray(self.spline(omega), dtype=float)


@dataclass(frozen=True)
class _TableBeta:
    """Keep an index table's propagation constant reproducible across processes."""

    table: IndexTable
    derivative: int = 0

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the propagation constant or its analytic first derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, inside the table.

        Returns
        -------
        numpy.ndarray
            Values in rad/m, or s/m for derivative one, shape ``(n_omega,)``.
        """
        return (
            self.table.beta(omega)
            if self.derivative == 0
            else self.table.dbeta_domega(omega)
        )


@dataclass(frozen=True)
class _TableAlpha:
    """Keep an index table's power attenuation reproducible across processes."""

    table: IndexTable

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the table's power attenuation.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, inside the table.

        Returns
        -------
        numpy.ndarray
            Power attenuation in 1/m, shape ``(n_omega,)``.
        """
        return self.table.alpha(omega)


def _source_paths(mode: Mode) -> tuple[ZProfile, ...]:
    """Return the varying scalar paths a source mode declares.

    The z ownership of the mode a specified fibre is built from: the paths of a
    :class:`solprop.modes.protocol.ParameterisedMode`, and otherwise the paths a
    mode declares under :data:`solprop.modes.protocol.GEOMETRY_PATH_NAMES`, which
    is the same tuple the transform's kink walk reads. Only the ones that move are
    kept.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The source mode.

    Returns
    -------
    tuple of solprop.profiles.ZProfile
        The varying declared paths, in declaration order.
    """
    declared: tuple[ZProfile | None, ...]
    if isinstance(mode, ParameterisedMode):
        declared = tuple(mode.parameter_paths.values())
    else:
        declared = tuple(getattr(mode, name, None) for name in GEOMETRY_PATH_NAMES)
    return tuple(one for one in declared if one is not None and not one.is_constant)


def _source_geometry(mode: Mode) -> tuple[tuple[float, float], Float[np.ndarray, " n"]]:
    """Return the span and the breakpoints a moving source mode declares.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The source mode, which moves along z.

    Returns
    -------
    tuple
        ``((start, end), breakpoints)`` in m: the tightest interval every varying
        path covers, and the union of their kinks inside it.

    Raises
    ------
    ModeError
        If the mode moves but declares no varying path to say where, or the paths
        it declares have no interval in common.
    """
    paths = _source_paths(mode)
    if not paths:
        msg = (
            f"{type(mode).__name__} varies along z but declares no path saying "
            "where: a specified fibre samples the dispersion and the area over the "
            "positions a path declares, so the source must declare one"
        )
        raise ModeError(msg)
    start = max(float(one.z_span[0]) for one in paths)
    end = min(float(one.z_span[1]) for one in paths)
    if not end > start:
        msg = (
            f"the declared paths of {type(mode).__name__} have no interval in "
            f"common: they meet only at [{start!r}, {end!r}] m"
        )
        raise ModeError(msg)
    edges = np.unique(
        np.concatenate(
            [np.atleast_1d(np.asarray(one.breakpoints, dtype=float)) for one in paths]
            + [np.array([start, end])]
        )
    )
    return (start, end), edges[(edges >= start) & (edges <= end)]


def _area_path(
    aeff: Float[np.ndarray, " n_positions"] | Sequence[float] | ZProfile,
    positions: Float[np.ndarray, " n_positions"],
) -> ZProfile:
    """Return the effective-area path declared beside a dispersion path.

    Parameters
    ----------
    aeff : array_like or solprop.profiles.ZProfile
        One positive area in m^2 per declared position, or a path already.
    positions : numpy.ndarray
        The dispersion path's declared positions in m, shape ``(n_positions,)``.

    Returns
    -------
    solprop.profiles.ZProfile
        A :class:`solprop.profiles.PiecewiseLinear` through the declared areas, or
        the supplied path unchanged.

    Raises
    ------
    ModeError
        If a sequence of areas does not carry one value per declared position.
    """
    if isinstance(aeff, ZProfile):
        return aeff
    areas = np.ascontiguousarray(aeff, dtype=np.float64).reshape(-1)
    if areas.size != positions.size:
        msg = (
            f"{areas.size} effective areas were declared at {positions.size} "
            "positions: a fibre whose geometry moves samples its dispersion and "
            "its area together, at the same positions"
        )
        raise ModeError(msg)
    return PiecewiseLinear(positions, areas)


def _dispersion_moves(mode: Mode) -> bool:
    """Return whether a source mode's dispersion, not merely its geometry, moves.

    A mode that reports ``z_constant`` is fixed in every respect. One that does not
    is usually moving its geometry, and its dispersion with it -- but a
    :class:`SpecifiedMode` whose only declared path is its effective area is the
    documented exception, and it says so through the optional
    ``dispersion_is_constant`` trait. Wrapping such a fibre must not turn its
    frequency-only dispersion into a path, or the cumulative operator would leave
    the parameter expansion for the z table and the compiled routes with it.

    Parameters
    ----------
    mode : solprop.modes.protocol.Mode
        The source mode.

    Returns
    -------
    bool
        ``True`` when the source's propagation constant depends on z.
    """
    declared = getattr(mode, "dispersion_is_constant", None)
    return not mode.z_constant if declared is None else not bool(declared)


def _warn_surrogate_area(label: str, omega: object) -> None:
    """Warn that a provider channel's area is its median-knot surrogate.

    Parameters
    ----------
    label : str
        The display label of the channel handed to a scalar model with no area and
        no wavelength.
    omega : object
        The frequency its table records the surrogate at, rad/s, if it records one.
    """
    where = (
        f"{omega:.6e} rad/s ({2.0 * math.pi * C / omega * 1e9:.2f} nm)"
        if isinstance(omega, float)
        else "an undeclared frequency"
    )
    warnings.warn(
        f"{label}'s effective area is the surrogate at the "
        f"median knot of its guide's band, {where}, not at the pump: pass "
        "wavelength=<the pump's> to solve it there (prop_gnlse passes wavelength), "
        "or aeff= to declare it",
        SurrogateAreaWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class _ModeArea:
    """Keep a supplied mode's effective area reproducible as a path along z."""

    mode: Mode

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Sample the source mode's effective area.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``A_eff(z)`` in m^2, shape ``(n_z,)``.
        """
        return np.array([self.mode.aeff(float(one)) for one in z], dtype=float)


@dataclass(frozen=True)
class _ModeBeta:
    """Keep a supplied mode's propagation constant reproducible across processes.

    A :class:`DispersionPath`: it forwards the position to the source mode, so a
    fibre built from a moving guide carries that guide's dispersion at every
    position rather than freezing it at the entrance, and reports the source's own
    breakpoints and span as its own.
    """

    mode: Mode
    derivative: int = 0

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the source mode's declared breakpoints.

        Returns
        -------
        numpy.ndarray
            The positions in m, empty for a source whose dispersion is fixed.
        """
        if not _dispersion_moves(self.mode):
            return np.empty(0, dtype=float)
        return _source_geometry(self.mode)[1]

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the source mode's declared paths cover.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m, ``(0, inf)`` for a source whose dispersion is
            fixed.
        """
        if not _dispersion_moves(self.mode):
            return 0.0, math.inf
        return _source_geometry(self.mode)[0]

    @property
    def is_constant(self) -> bool:
        """Return whether the source mode's dispersion moves.

        Returns
        -------
        bool
            ``True`` unless the source's propagation constant depends on z.
        """
        return not _dispersion_moves(self.mode)

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the propagation constant or its analytic first derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, inside the source mode's support.
        z : float, optional
            Position along the fibre, m.

        Returns
        -------
        numpy.ndarray
            Values in rad/m, or s/m for derivative one, shape ``(n_omega,)``.
        """
        return (
            beta(self.mode, omega, z)
            if self.derivative == 0
            else beta1(self.mode, omega, z)
        )


@dataclass(frozen=True)
class _ModeAlpha:
    """Keep a supplied mode's power attenuation reproducible across processes.

    A :class:`DispersionPath`, for the reason :class:`_ModeBeta` gives.
    """

    mode: Mode

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the source mode's declared breakpoints.

        Returns
        -------
        numpy.ndarray
            The positions in m, empty for a source that does not move.
        """
        return _ModeBeta(self.mode).breakpoints

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the source mode's declared paths cover.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m.
        """
        return _ModeBeta(self.mode).z_span

    @property
    def is_constant(self) -> bool:
        """Return whether the source mode's loss moves.

        Returns
        -------
        bool
            ``True`` unless the source's attenuation depends on z.
        """
        return not _dispersion_moves(self.mode)

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Evaluate the supplied mode's power attenuation.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, inside the source mode's support.
        z : float, optional
            Position along the fibre, m.

        Returns
        -------
        numpy.ndarray
            Power attenuation in 1/m, shape ``(n_omega,)``.
        """
        return alpha(self.mode, omega, z)


@dataclass(frozen=True)
class SpecifiedMode:
    """A mode defined by its propagation constant rather than by a waveguide.

    This is the GNLSE's "fibre": ``beta(omega)`` given as data, a power loss, and an
    effective area. It implements :class:`solprop.modes.protocol.Mode` in full, so the
    linear operator, the normalisation and the mode-averaged transform treat it exactly
    like a capillary mode and the GNLSE needs no special path (see
    ``docs/dev/decisions.md``).

    Build one with :meth:`from_taylor` or :meth:`from_table` for a fibre that does
    not move, and with :meth:`from_taylor_path`, :meth:`from_index_path` or
    :meth:`from_mode` for one that does; the constructor takes the resolved
    functions and is the escape hatch for an arbitrary analytic dispersion.

    **Two paths along z, declared together.** The dispersion, its derivative and the
    loss are each either a function of frequency alone or a :class:`DispersionPath`; the
    effective area is either a number or a :class:`solprop.profiles.ZProfile`. A fibre
    whose dispersion moves must declare an area that moves with it, at the same declared
    positions -- no real waveguide changes one without the other -- and the
    **area-only** case, a taper whose dispersion is held fixed, stays as the documented
    constant-dispersion special case. The z dependence then goes through the one
    mechanism a gas gradient goes through: the declared positions are the hard kinks,
    the cumulative operator resolves ``D(omega, z)`` between them and integrates it
    exactly (``docs/guide/numerics/linear-operator.md``).

    The transverse profile is a uniform disc of area ``effective_area``, which is the
    profile whose effective area is exactly that area, so the single-node mode-averaged
    transform is well defined and agrees with :meth:`aeff`.

    Attributes
    ----------
    dispersion : callable or DispersionPath
        ``beta(omega)`` in 1/m, taking and returning an array of shape
        ``(n_omega,)``, or a declared path ``beta(omega, z)``.
    dispersion_derivative : callable or DispersionPath
        ``d beta / d omega`` in s/m, analytic, same shapes; a path when
        ``dispersion`` is one, declared at the same positions.
    loss : float or callable or DispersionPath
        Power attenuation in 1/m: a constant, a function of ``omega`` with the same
        shapes, or a declared path. Convert a figure in dB/m with
        :func:`attenuation_from_db`.
    effective_area : float or ZProfile
        ``A_eff`` in m^2, positive: a constant, or a declared path along the fibre.
        A path makes the mode z-varying, and the area then enters exactly once, as
        the shape of the nonlinear coefficient: the mode-averaged transform
        tabulates ``sqrt(A_eff(0) / A_eff(z))`` and the response prefactor keeps the
        entrance area, so ``gamma(z) = gamma A_eff(0) / A_eff(z)`` with the supplied
        ``gamma`` as the entrance coefficient.

    Raises
    ------
    ModeError
        If the effective area is not positive and finite anywhere it is declared, if
        the dispersion moves without the area, if the value and the derivative
        disagree about whether they move, or if the dispersion and the area are
        declared at different positions.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.gnlse import SpecifiedMode
    >>> mode = SpecifiedMode.from_taylor(
    ...     [1.0e7, 5.0e-9, -2.0e-26], omega_ref=2.0e15, aeff=1e-10
    ... )
    >>> f"{float(mode.beta(np.array([2.0e15]))[0]):.6e}"
    '1.000000e+07'
    >>> mode.aeff()
    1e-10
    """

    dispersion: DispersionFunction | DispersionPath
    dispersion_derivative: DispersionFunction | DispersionPath
    loss: float | LossFunction | DispersionPath
    effective_area: float | ZProfile

    # Properties rather than class attributes: the protocol declares these three as
    # properties, and a `ClassVar` does not satisfy a protocol property for the type
    # checker.
    @property
    def z_constant(self) -> bool:
        """Whether the mode is independent of ``z``.

        Returns
        -------
        bool
            ``True`` when the effective area, the dispersion, its derivative and
            the loss are all constant along z.
        """
        return self.area_path.is_constant and self.dispersion_is_constant

    @property
    def dispersion_is_constant(self) -> bool:
        """Whether the spectral quantities are functions of frequency alone.

        Returns
        -------
        bool
            ``True`` when the dispersion, its derivative and the loss all report
            themselves constant along z.

        Examples
        --------
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> SpecifiedMode.from_taylor([1.0e7], omega_ref=2.0e15).dispersion_is_constant
        True
        """
        return all(
            path.is_constant
            for path in (self.dispersion_path, self.derivative_path, self.loss_path)
        )

    @property
    def area_path(self) -> ZProfile:
        """Return the declared effective-area path in m^2.

        Returns
        -------
        solprop.profiles.ZProfile
            The path, a :class:`~solprop.profiles.Constant` when the area is a
            number. It is the path the transform's kink walk and
            :func:`solprop.translate.linop.declared_paths` take their breakpoints
            from, so that the tables split there and the schedule lands on them.
        """
        return as_profile(self.effective_area)

    @property
    def dispersion_path(self) -> DispersionPath:
        """Return the propagation constant as a path along z.

        Returns
        -------
        DispersionPath
            The declared path, a :class:`ConstantDispersion` when the dispersion is
            a function of frequency alone. Its breakpoints are kinks of the
            cumulative operator exactly as the area path's are.
        """
        return as_dispersion_path(self.dispersion)

    @property
    def derivative_path(self) -> DispersionPath:
        """Return ``d beta / d omega`` as a path along z.

        Returns
        -------
        DispersionPath
            The declared path, a :class:`ConstantDispersion` when the derivative is
            a function of frequency alone.
        """
        return as_dispersion_path(self.dispersion_derivative)

    @property
    def loss_path(self) -> DispersionPath:
        """Return the power attenuation as a path along z.

        Returns
        -------
        DispersionPath
            The declared path; a constant number becomes a uniform attenuation at
            every frequency and every position.
        """
        loss = self.loss
        if isinstance(loss, float | int):
            return ConstantDispersion(_UniformLoss(float(loss)))
        return as_dispersion_path(loss)

    @property
    def scale_invariant(self) -> bool:
        """Whether the profile scales exactly with a core radius along ``z``.

        Returns
        -------
        bool
            Always ``True``: the surrogate profile is a disc of radius
            ``sqrt(A_eff(z) / pi)``, which is exactly ``f(r / a(z))``.
        """
        return True

    @property
    def azimuthal_order(self) -> int:
        """Highest azimuthal order of the profile.

        Returns
        -------
        int
            Always 0: the disc is azimuthally uniform.
        """
        return 0

    @property
    def channel_label(self) -> ChannelLabel:
        """Name this channel: a dispersion, not an eigenmode of a guide.

        Returns
        -------
        ChannelLabel
            Family ``specified``, with no azimuthal or radial order, no parity and
            no handedness: a mode given by ``beta(omega)`` has no quantum numbers.

        Examples
        --------
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor([1.0e7], omega_ref=2.0e15)
        >>> mode.channel_label.display_label
        'specified'
        """
        return ChannelLabel("specified", None, None, None, None, "specified")

    def principal_component(self, z: float = 0.0) -> PolarisationComponent:
        """Return the polarisation of the uniform surrogate profile.

        Parameters
        ----------
        z : float, optional
            Position along the fibre, m; ignored by the uniform profile.

        Returns
        -------
        {'x'}
            The sole nonzero Cartesian component.
        """
        del z
        return "x"

    def __post_init__(self) -> None:
        """Check the effective area, and that the two paths were declared together.

        Raises
        ------
        ModeError
            If ``effective_area`` is not positive and finite, at the entrance or
            anywhere its declared path is sampled; if the dispersion moves and the
            area does not; if the value and the derivative disagree about whether
            they move; or if the area path is declared at other positions than the
            spectral paths that move.
        """
        path = self.area_path
        sampled = path(
            np.unique(
                np.concatenate(
                    (
                        np.atleast_1d(path.breakpoints),
                        np.array([0.0]),
                        np.asarray(
                            [value for value in path.z_span if math.isfinite(value)],
                            dtype=float,
                        ),
                    )
                )
            )
        )
        if not (np.all(np.isfinite(sampled)) and np.all(sampled > 0.0)):
            msg = f"effective_area must be positive, got {self.effective_area!r}"
            raise ModeError(msg)
        self._check_paths_agree()

    def _check_paths_agree(self) -> None:
        """Refuse a dispersion that moves without its area, or against other positions.

        A waveguide whose dispersion changes has changed its geometry, and a geometry
        that has changed has a different effective area: the two come together or not at
        all. The derivative is held to the same declaration as the value it belongs to,
        because the co-moving frame takes both from the same source, and a value that
        moves beside a frozen derivative would give a frame that does not follow the
        pulse.

        The positions compared against the area path's are the **union** over every
        spectral path that moves -- the value, the derivative and the loss -- and
        not the value path's alone. That is what makes "the declared positions are
        the hard kinks" true by construction: only the area path is walked for the
        kinks (:data:`solprop.translate.modal._DECLARED_PATHS`), so a corner
        declared in the loss alone would otherwise be a corner nothing lands on.
        It is also what makes a fibre whose **loss** alone moves constructible: its
        value path is constant and declares no position at all.

        Raises
        ------
        ModeError
            Naming which of the three disagreements it is.
        """
        value, derivative = self.dispersion_path, self.derivative_path
        if value.is_constant != derivative.is_constant:
            msg = (
                "the propagation constant and its frequency derivative disagree "
                f"about moving along z ({value.is_constant=}, "
                f"{derivative.is_constant=}); the co-moving frame takes both from one "
                "source"
            )
            raise ModeError(msg)
        if self.dispersion_is_constant:
            return
        if self.area_path.is_constant:
            msg = (
                "a specified fibre whose dispersion follows a path along z must "
                "declare an effective-area path beside it: a guide whose dispersion "
                "changes has changed its geometry, and its area with it. Give "
                f"aeff= a ZProfile, got {self.effective_area!r}"
            )
            raise ModeError(msg)
        moving = [
            path
            for path in (self.dispersion_path, self.derivative_path, self.loss_path)
            if not path.is_constant
        ]
        declared = np.unique(
            np.concatenate(
                [
                    np.atleast_1d(np.asarray(path.breakpoints, dtype=float))
                    for path in moving
                ]
            )
        )
        area = np.atleast_1d(np.asarray(self.area_path.breakpoints, dtype=float))
        if not np.array_equal(declared, area):
            msg = (
                f"the dispersion paths are declared at z = {declared.tolist()} m and "
                f"the area path at z = {area.tolist()} m: a fibre whose geometry "
                "moves samples its dispersion and its area together, at the same "
                "positions"
            )
            raise ModeError(msg)

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the scalar path this mode's z dependence runs through.

        Returns
        -------
        collections.abc.Mapping
            ``{"effective_area": path}`` in m^2.

        Notes
        -----
        When the dispersion is a function of frequency alone -- the documented
        special case, a taper whose dispersion is held fixed -- the declared area
        is the whole of the z dependence and
        :class:`solprop.modes.protocol.ParameterisedMode` holds exactly: the linear
        operator is expanded in a parameter it does not depend on, at degree zero,
        and the path's breakpoints become the hard kinks the schedule lands on.

        When the dispersion follows a path the area is no longer the whole of it,
        and :func:`at_parameters` cannot return a z-constant mode. The declaration
        is still the mode's kinks, which is what
        :func:`solprop.translate.linop.declared_paths` reads it for; the expansion
        is not taken, because :func:`solprop.translate.linop.parameter_paths`
        checks that freezing a mode's parameters leaves it z-constant and falls back
        to the z table when it does not.

        Examples
        --------
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> from solprop.profiles import Linear
        >>> mode = SpecifiedMode.from_taylor(
        ...     [0.0], omega_ref=2.0e15, aeff=Linear(1e-10, 2e-10, 1.0)
        ... )
        >>> sorted(mode.parameter_paths)
        ['effective_area']
        """
        return MappingProxyType({"effective_area": self.area_path})

    @property
    def parameters_own_z(self) -> bool:
        """Whether the declared parameters are the whole of this mode's z dependence.

        The trait the translation layer reads before it expands the cumulative
        operator in a mode's parameters instead of tabulating it along z. It is
        ``True`` for the constant-dispersion special case, where the declared area
        is everything, and ``False`` once the dispersion follows a path, which no
        scalar parameter can be frozen at.

        Returns
        -------
        bool
            :attr:`dispersion_is_constant`.

        Examples
        --------
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> SpecifiedMode.from_taylor([0.0], omega_ref=2.0e15).parameters_own_z
        True
        """
        return self.dispersion_is_constant

    def at_parameters(self, values: Mapping[str, float]) -> "SpecifiedMode":  # noqa: UP037 - the hook resolves the class during construction
        """Return the same fibre with its area frozen at a value.

        Parameters
        ----------
        values : collections.abc.Mapping
            Any subset of ``{"effective_area"}``, in m^2 and positive. A name left
            out keeps its path; a name this mode does not declare is an error.

        Returns
        -------
        SpecifiedMode
            The same dispersion and loss with a constant effective area.

        Raises
        ------
        ModeError
            If a name is not declared, the value is not a positive area, or the
            dispersion follows a path along z -- there is then no parameter value
            that reproduces this mode, which is what :attr:`parameters_own_z` says
            in advance.
        """
        unknown = set(values) - {"effective_area"}
        if unknown:
            msg = f"unknown SpecifiedMode parameters: {unknown}"
            raise ModeError(msg)
        if not self.dispersion_is_constant:
            msg = (
                "a specified fibre whose dispersion follows a path along z cannot "
                "be frozen at a parameter value: its z dependence is not a scalar "
                "path, and the cumulative operator tabulates it along z instead"
            )
            raise ModeError(msg)
        return dataclasses.replace(
            self, effective_area=values.get("effective_area", self.effective_area)
        )

    @classmethod
    def from_taylor(
        cls,
        betas: Float[np.ndarray, " n_deg"] | list[float],
        omega_ref: float,
        *,
        loss: float | LossFunction = 0.0,
        aeff: float | ZProfile = 1.0,
        # Quoted so that beartype, which resolves the annotation while the class body
        # is still executing, does not look up a name that is not yet bound.
    ) -> "SpecifiedMode":  # noqa: UP037
        """Build a mode from Taylor coefficients of ``beta`` about a frequency.

        ``beta(omega) = sum_n betas[n] (omega - omega_ref)**n / n!``. There is no
        constructor that centres the series at zero frequency, because a high-order
        Taylor polynomial extrapolated from DC to the optical band is meaningless.

        Parameters
        ----------
        betas : array_like
            Taylor coefficients ``beta_0, beta_1, ...`` in s^n/m, at least one.
        omega_ref : float
            Frequency the series is expanded about, rad/s, positive.
        loss : float or callable, optional
            Power attenuation in 1/m; a constant or a function of ``omega``. Default 0.
        aeff : float or ZProfile, optional
            Effective area in m^2, or a declared path along the fibre. Default 1.0,
            which is the GNLSE convention in which the area has been folded into
            ``n2`` already.

        Returns
        -------
        SpecifiedMode
            The mode.

        Raises
        ------
        ModeError
            If no coefficients are given, ``omega_ref`` is not positive, or ``aeff`` is
            not positive.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor([0.0, 5.0e-9], omega_ref=2.0e15)
        >>> f"{float(mode.beta(np.array([3.0e15]))[0]):.6f}"
        '5000000.000000'
        """
        return cls(
            TaylorDispersion(betas, omega_ref),
            TaylorDispersion(betas, omega_ref, 1),
            loss,
            aeff if isinstance(aeff, ZProfile) else float(aeff),
        )

    @classmethod
    def from_taylor_path(
        cls,
        betas: Float[np.ndarray, "n_positions n_deg"] | Sequence[Sequence[float]],
        positions: Float[np.ndarray, " n_positions"] | Sequence[float],
        omega_ref: float,
        *,
        aeff: Float[np.ndarray, " n_positions"] | Sequence[float] | ZProfile,
        loss: float | LossFunction | DispersionPath = 0.0,
    ) -> "SpecifiedMode":  # noqa: UP037 - the hook resolves the class during construction
        """Build a fibre whose Taylor dispersion and area are declared along z.

        One expansion set and one effective area per declared position, both
        interpolated linearly in between: the two are sampled together, which is
        what makes the model consistent. Between two positions the
        propagation constant is the straight line between the two sets, so the
        cumulative operator resolves each section at degree one in ``z`` and
        integrates it exactly, and the declared positions are hard kinks the
        schedule lands on.

        Parameters
        ----------
        betas : array_like
            Taylor coefficients ``beta_0, beta_1, ...`` in s^n/m at each position,
            shape ``(n_positions, n_deg)``. Every row carries the same number of
            coefficients.
        positions : array_like
            Strictly increasing positions in m, shape ``(n_positions,)``, at least
            two. They must cover the run, which
            :func:`solprop.translate.linop.build_phi` checks by name.
        omega_ref : float
            Frequency every series is expanded about, rad/s, positive.
        aeff : array_like or ZProfile
            Effective area in m^2: one positive value per position, or a declared
            path whose breakpoints are exactly `positions`.
        loss : float or callable or DispersionPath, optional
            Power attenuation in 1/m. Default 0.

        Returns
        -------
        SpecifiedMode
            The mode, with an analytic first dispersion derivative at every
            position.

        Raises
        ------
        ModeError
            If the rows and the positions disagree, fewer than two positions are
            given, the areas are not positive, or the two paths are declared at
            different positions.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor_path(
        ...     [[0.0, 5.0e-9], [0.0, 7.0e-9]],
        ...     [0.0, 1.0],
        ...     omega_ref=2.0e15,
        ...     aeff=[1e-10, 2e-10],
        ... )
        >>> f"{float(mode.beta(np.array([3.0e15]), 0.5)[0]):.6f}"
        '6000000.000000'
        >>> mode.z_constant
        False
        """
        nodes = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1)
        coefficients = np.ascontiguousarray(betas, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[0] != nodes.size:
            msg = (
                f"betas has shape {coefficients.shape}, expected "
                f"({nodes.size}, n_deg): one expansion set per declared position"
            )
            raise ModeError(msg)
        members = tuple(TaylorDispersion(row, omega_ref) for row in coefficients)
        derivatives = tuple(TaylorDispersion(row, omega_ref, 1) for row in coefficients)
        return cls(
            InterpolatedDispersion(nodes, members),
            InterpolatedDispersion(nodes, derivatives),
            loss,
            _area_path(aeff, nodes),
        )

    @classmethod
    def from_index_path(
        cls,
        omega: Float[np.ndarray, " n_nodes"],
        neff: Complex[np.ndarray, "n_positions n_nodes"],
        positions: Float[np.ndarray, " n_positions"] | Sequence[float],
        *,
        aeff: Float[np.ndarray, " n_positions"] | Sequence[float] | ZProfile,
    ) -> "SpecifiedMode":  # noqa: UP037 - the hook resolves the class during construction
        """Build a fibre from an ``n_eff(omega, z)`` table and its area path.

        The table's real part is the propagation constant ``beta = omega Re n_eff / c``
        and its imaginary part the power attenuation ``alpha = 2 omega Im n_eff / c``
        (``docs/dev/conventions.md`` section 2, ``Im n_eff > 0`` is loss). Each row is
        one position's spectrum, a cubic spline in frequency differentiated exactly in
        the spline basis, and the rows are interpolated linearly in ``z``.

        The frequency axis is shared by every position, and it must **span the run's
        active band**: a spline evaluated outside the table it was built from is
        refused by name rather than extrapolated, which is the classic way to get a
        plausible, wrong answer.

        Parameters
        ----------
        omega : numpy.ndarray
            Strictly increasing angular frequencies, rad/s, shape ``(n_nodes,)``, at
            least :data:`solprop.modes.tabulated.MIN_TABLE_POINTS` of them.
        neff : numpy.ndarray
            Complex effective index at each position and frequency, shape
            ``(n_positions, n_nodes)``.
        positions : array_like
            Strictly increasing positions in m, shape ``(n_positions,)``, at least
            two.
        aeff : array_like or ZProfile
            Effective area in m^2: one positive value per position, or a declared
            path whose breakpoints are exactly `positions`.

        Returns
        -------
        SpecifiedMode
            The mode.

        Raises
        ------
        ModeError
            If the table's shape does not match the axes, fewer than two positions
            are given, or the two paths are declared at different positions.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.constants import C
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> omega = np.linspace(1.0e15, 3.0e15, 8)
        >>> index = np.stack([np.full(8, 1.45 + 0j), np.full(8, 1.46 + 0j)])
        >>> mode = SpecifiedMode.from_index_path(
        ...     omega, index, [0.0, 1.0], aeff=[1e-10, 2e-10]
        ... )
        >>> f"{float(mode.neff(np.array([2.0e15]), 0.5).real[0]):.6f}"
        '1.455000'
        """
        nodes = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1)
        axis = np.ascontiguousarray(omega, dtype=np.float64).reshape(-1)
        table = np.ascontiguousarray(neff, dtype=np.complex128)
        if table.ndim != 2 or table.shape != (nodes.size, axis.size):
            msg = (
                f"neff has shape {table.shape}, expected "
                f"({nodes.size}, {axis.size}): one index row per declared position"
            )
            raise ModeError(msg)
        betas = axis * np.real(table) / C
        alphas = 2.0 * axis * np.imag(table) / C
        return cls(
            InterpolatedDispersion(
                nodes, tuple(TabulatedDispersion(axis, row) for row in betas)
            ),
            InterpolatedDispersion(
                nodes, tuple(TabulatedDispersion(axis, row, 1) for row in betas)
            ),
            InterpolatedDispersion(
                nodes, tuple(TabulatedDispersion(axis, row) for row in alphas)
            ),
            _area_path(aeff, nodes),
        )

    @classmethod
    def from_table(
        cls,
        omega: Float[np.ndarray, " n_nodes"],
        beta: Float[np.ndarray, " n_nodes"],
        *,
        loss: float | LossFunction = 0.0,
        aeff: float | ZProfile = 1.0,
    ) -> "SpecifiedMode":  # noqa: UP037
        """Build a mode from a tabulated propagation constant.

        The table is interpolated by a cubic spline and differentiated exactly in the
        spline basis, so ``dneff_domega`` stays analytic, as the co-moving frame
        requires.
        Evaluation outside the tabulated range raises rather than extrapolating.

        Parameters
        ----------
        omega : numpy.ndarray
            Strictly increasing angular frequencies, rad/s, shape ``(n_nodes,)``, at
            least four.
        beta : numpy.ndarray
            Propagation constant at those frequencies, 1/m, shape ``(n_nodes,)``.
        loss : float or callable, optional
            Power attenuation in 1/m; a constant or a function of ``omega``. Default 0.
        aeff : float or ZProfile, optional
            Effective area in m^2, or a declared path along the fibre. Default 1.0.

        Returns
        -------
        SpecifiedMode
            The mode.

        Raises
        ------
        ModeError
            If the two arrays disagree in shape, the table is shorter than four points,
            the frequencies are not strictly increasing, or ``aeff`` is not positive.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> omega = np.linspace(1.0e15, 3.0e15, 8)
        >>> mode = SpecifiedMode.from_table(omega, 5.0e-9 * omega)
        >>> f"{float(mode.beta(np.array([2.0e15]))[0]):.6f}"
        '10000000.000000'
        """
        return cls(
            TabulatedDispersion(omega, beta),
            TabulatedDispersion(omega, beta, 1),
            loss,
            aeff if isinstance(aeff, ZProfile) else float(aeff),
        )

    @classmethod
    def from_index_table(
        cls, table: IndexTable, *, aeff: float | ZProfile = 1.0
    ) -> "SpecifiedMode":  # noqa: UP037 - type hook resolves the class during construction
        """Build a dispersion-specified mode from a complex-index table.

        Parameters
        ----------
        table : solprop.modes.tabulated.IndexTable
            Whole effective index and loss, with finite spectral support.
        aeff : float or ZProfile, optional
            Positive propagation area in m^2, or a declared path along the fibre.
            The driver selects the table's area at its reference wavelength;
            default 1.0 is the gamma convention.

        Returns
        -------
        SpecifiedMode
            Mode with analytic real-index derivative and reproducible callables.

        Raises
        ------
        ModeError
            If the propagation area is not positive.
        """
        return cls(
            _TableBeta(table),
            _TableBeta(table, 1),
            _TableAlpha(table),
            aeff if isinstance(aeff, ZProfile) else float(aeff),
        )

    @classmethod
    def from_mode(
        cls,
        mode: Mode,
        *,
        aeff: float | ZProfile | None = None,
        wavelength: float | None = None,
    ) -> "SpecifiedMode":  # noqa: UP037 - resolved during class construction
        """Use a mode's dispersion, loss and area for a GNLSE fibre.

        Its fill contributes to the supplied dispersion and loss, but is ignored
        as a nonlinear material. The GNLSE caller supplies gamma or n2 instead.
        Frozen callables retain the original mode for reproducible records.

        **A source whose geometry moves hands over both paths.** The frozen
        callables forward the position to the source mode, so its dispersion, its
        derivative and its loss are evaluated where they are asked for rather than
        frozen at the entrance, and the effective area is sampled from the same
        mode over the same declared positions -- the two come together, so a moving
        source is accepted rather than refused. A source
        that does not move keeps the ordinary constant path, bit for bit.

        **A provider channel's area belongs to a frequency.** A
        :class:`~solprop.modes.guide.ProviderChannelMode`'s own area is the
        surrogate at the median knot of the band its guide was sampled on, which is
        not in general the pump: on worked example 05's graded-index taper it is
        35 % above the pump's. With
        `wavelength`, the area is solved again at that wavelength instead
        (:meth:`~solprop.modes.guide.ProviderChannelMode.area_at`) -- once for a
        snapshot, and at every sampled position of a moving guide's path, so that
        a moving channel hands over an area path at the pump. With neither `aeff`
        nor `wavelength`, the median-knot surrogate is used and a
        :class:`~solprop.exceptions.SurrogateAreaWarning` says so.
        :func:`solprop.driver.interface.prop_gnlse` passes its ``wavelength``.

        Parameters
        ----------
        mode : solprop.modes.protocol.Mode
            Source mode.
        aeff : float or ZProfile or None, optional
            Positive finite effective area in m^2, or a declared path. ``None``
            takes the source's declared area path if it has one, samples the
            source's own ``aeff`` along z when its geometry moves, and otherwise
            ``mode.aeff()`` -- or, for a provider channel given `wavelength`, the
            channel's area solved at that wavelength.
        wavelength : float or None, optional
            Positive wavelength in m the area of a provider channel is taken at,
            normally the pump's; read only when `aeff` is ``None``. Every other
            mode's area is not a function of frequency in the
            :class:`~solprop.modes.protocol.Mode` protocol -- it is declared once,
            as :class:`~solprop.modes.tabulated.TabulatedMode`'s is at its own
            ``reference_wavelength`` -- so there it is accepted and **ignored**,
            which is what lets :func:`solprop.driver.interface.prop_gnlse` pass its
            ``wavelength`` for any fibre.

        Returns
        -------
        SpecifiedMode
            GNLSE mode with an analytic first dispersion derivative.

        Raises
        ------
        ModeError
            If the source's geometry moves and a constant area is asked for, if a
            moving source declares no path saying where it moves, if the area is
            not positive and finite, if `wavelength` is not positive and finite, or
            if a provider channel asked for its area at `wavelength` carries no
            provider to solve.

        Warns
        -----
        SurrogateAreaWarning
            For a provider channel given neither `aeff` nor `wavelength`.
        """
        if wavelength is not None and not (
            math.isfinite(wavelength) and wavelength > 0
        ):
            raise ModeError(
                f"wavelength must be positive and finite, got {wavelength!r}"
            )
        # The property meant, not the attribute name: a source whose whole z
        # dependence is its effective area hands that path over directly.
        path: ZProfile | None = None
        if isinstance(mode, ParameterisedMode) and set(mode.parameter_paths) == {
            "effective_area"
        }:
            path = mode.parameter_paths["effective_area"]
        # Imported here: solprop.modes.guide imports solprop.nonlinear through the
        # material regions' responses.
        from solprop.modes.guide import ProviderChannelMode  # noqa: PLC0415

        solved: float | ZProfile | None = None
        if isinstance(mode, ProviderChannelMode) and aeff is None:
            if wavelength is None:
                _warn_surrogate_area(
                    mode.label.display_label,
                    # A table sampled from a provider records the knot; one
                    # built by hand has no identity to say.
                    getattr(mode.table, "source_identity", {}).get(
                        "channel_area_omega"
                    ),
                )
            else:
                solved = mode.area_at(2.0 * math.pi * C / wavelength)
        if _dispersion_moves(mode) and path is None:
            # A moving guide: sample the area over the positions its own declared
            # paths kink at, so that the two paths carry the same breakpoints and
            # the cumulative operator splits on them. A provider channel's area
            # solved at the pump is declared over the same positions.
            span, kinks = _source_geometry(mode)
            path = FromCallable(
                solved if isinstance(solved, ZProfile) else _ModeArea(mode),
                span,
                kinks,
            )
        area: float | ZProfile
        if aeff is not None:
            area = aeff
        elif path is not None:
            area = path
        else:
            area = mode.aeff() if solved is None else solved
        if _dispersion_moves(mode) and as_profile(area).is_constant:
            msg = (
                f"{type(mode).__name__} moves along z, so a specified fibre built "
                "from it carries its dispersion along z and needs an effective-area "
                f"path beside it; aeff={aeff!r} is constant. Pass a ZProfile, or "
                "leave aeff=None to sample the source's own area"
            )
            raise ModeError(msg)
        return cls(
            _ModeBeta(mode),
            _ModeBeta(mode, 1),
            _ModeAlpha(mode),
            area if isinstance(area, ZProfile) else float(area),
        )

    def beta(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the propagation constant.

        Unlike :meth:`neff` this is defined at zero frequency, so it is the right entry
        point for anything that works over a whole field grid.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position along the waveguide, m. It reaches the declared dispersion
            path; a dispersion given as a function of frequency alone ignores it.

        Returns
        -------
        numpy.ndarray
            ``beta(omega, z)`` in 1/m, shape ``(n_omega,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor([1.0e7], omega_ref=2.0e15)
        >>> mode.beta(np.array([0.0, 3.0e15])).tolist()
        [10000000.0, 10000000.0]
        """
        return self.dispersion_path(np.asarray(omega, dtype=float), z)

    def alpha(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the power attenuation coefficient.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``.
        z : float, optional
            Position along the waveguide, m. It reaches the declared loss path; a
            constant or a function of frequency alone ignores it.

        Returns
        -------
        numpy.ndarray
            ``alpha(omega, z)`` in 1/m, shape ``(n_omega,)``, so that the power falls
            as ``exp(-alpha z)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import (
        ...     SpecifiedMode,
        ...     attenuation_from_db,
        ... )
        >>> mode = SpecifiedMode.from_taylor(
        ...     [1.0e7], omega_ref=2.0e15, loss=attenuation_from_db(10.0)
        ... )
        >>> f"{mode.alpha(np.array([2.0e15]))[0]:.6f}"
        '2.302585'
        """
        values = np.asarray(omega, dtype=float)
        return np.asarray(self.loss_path(values, z), dtype=float)

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the complex effective index.

        ``n_eff = c (beta + i alpha / 2) / omega``, the inverse of the ``beta`` and
        ``alpha`` definitions in ``docs/dev/conventions.md`` section 2, so that
        ``Im n_eff > 0`` is loss.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m. It reaches the declared dispersion
            and loss paths; quantities given as functions of frequency alone
            ignore it.

        Returns
        -------
        numpy.ndarray
            ``n_eff(omega)``, complex, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If any frequency is not strictly positive. A specified mode has
            no effective index at zero frequency. Evaluate on the grid's active band
            instead.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor(
        ...     [1.0e7, 5.0e-9], omega_ref=2.0e15
        ... )
        >>> f"{mode.neff(np.array([2.0e15]))[0].real:.6f}"
        '1.498962'
        """
        values = _positive(omega)
        return C * (self.beta(values, z) + 0.5j * self.alpha(values, z)) / values

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return ``n_eff - 1`` formed without cancellation.

        ``(c (beta + i alpha / 2) - omega) / omega``: the subtraction happens on
        quantities of the same size rather than after a division that lands near one.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m. It reaches the declared dispersion
            and loss paths; quantities given as functions of frequency alone
            ignore it.

        Returns
        -------
        numpy.ndarray
            ``n_eff - 1``, complex, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If any frequency is not strictly positive.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.constants import C
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor(
        ...     [1.45 * 2.0e15 / C], omega_ref=2.0e15
        ... )
        >>> f"{mode.neff_minus_one(np.array([2.0e15]))[0].real:.6f}"
        '0.450000'
        """
        values = _positive(omega)
        numerator = C * (self.beta(values, z) + 0.5j * self.alpha(values, z)) - values
        return numerator / values

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic frequency derivative of the effective index.

        Differentiating ``n_eff = c (beta + i alpha / 2) / omega`` gives
        ``(c beta' - n_eff) / omega`` plus ``i c alpha' / (2 omega)``. The second term
        is zero for a constant loss and is **dropped** for a callable one, which is the
        only approximation in this class: the co-moving frame quantities take the real
        part, where the loss never appears, so ``beta_0`` and ``beta_1`` are unaffected
        either way.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m. It reaches the declared dispersion
            and loss paths; quantities given as functions of frequency alone
            ignore it.

        Returns
        -------
        numpy.ndarray
            ``d n_eff / d omega`` in s/rad, complex, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If any frequency is not strictly positive.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.constants import C
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> flat = SpecifiedMode.from_taylor(
        ...     [1.5 * 2.0e15 / C, 1.5 / C], omega_ref=2.0e15
        ... )
        >>> bool(np.allclose(flat.dneff_domega(np.array([2.0e15])), 0.0, atol=1e-28))
        True
        """
        values = _positive(omega)
        derivative = self.derivative_path(values, z)
        return (C * derivative - self.neff(values, z)) / values

    def aeff(self, z: float = 0.0) -> float:
        """Return the effective area at a position.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m; it is the declared area path that is
            evaluated, and a constant area ignores it.

        Returns
        -------
        float
            ``A_eff(z)`` in m^2.
        """
        return float(self.area_path(np.array([float(z)]))[0])

    def radius(self, z: float = 0.0) -> float:
        """Return the radius of the equivalent uniform disc.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        float
            ``sqrt(A_eff(z) / pi)`` in m, the radius of the disc whose effective area
            is ``A_eff(z)``.
        """
        return math.sqrt(self.aeff(z) / math.pi)

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        """Return the cross-section parametrisation and the integration limits.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m; the surrogate disc follows the area.

        Returns
        -------
        tuple
            ``("polar", (0.0, 0.0), (radius(z), 2 pi))`` in m and rad.
        """
        return "polar", (0.0, 0.0), (self.radius(z), 2.0 * math.pi)

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return the transverse field components at points of the cross-section.

        A uniform disc polarised along x: one inside the radius, zero outside. Its
        effective area ``(int |e|^2 dA)^2 / int |e|^4 dA`` is the disc's area, which is
        :meth:`aeff` by construction, so the single-node mode-averaged transform is
        consistent with the normalisation.

        Parameters
        ----------
        points : numpy.ndarray
            ``(r, theta)`` in m and rad, shape ``(n_points, 2)``.
        z : float, optional
            Position along the waveguide, m; the disc follows the declared area.

        Returns
        -------
        numpy.ndarray
            ``(E_x, E_y)`` of the unnormalised profile, shape ``(n_points, 2)``, real.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.gnlse import SpecifiedMode
        >>> mode = SpecifiedMode.from_taylor([1.0], omega_ref=1.0, aeff=np.pi)
        >>> points = np.array([[0.5, 0.0], [1.5, 0.0]])
        >>> mode.field(points).tolist()
        [[1.0, 0.0], [0.0, 0.0]]
        """
        inside = np.asarray(points, dtype=float)[:, 0] <= self.radius(z)
        return np.stack([inside.astype(float), np.zeros(inside.shape)], axis=1)


def _positive(omega: Float[np.ndarray, " n_omega"]) -> Float[np.ndarray, " n_omega"]:
    """Return the frequencies as a float array, refusing non-positive ones.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies, rad/s, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The same values as float64, shape ``(n_omega,)``.

    Raises
    ------
    ModeError
        If any value is not strictly positive.
    """
    values = np.asarray(omega, dtype=float)
    if values.size and values.min() <= 0.0:
        msg = (
            "a dispersion-specified mode has no effective index at or below zero "
            f"frequency; the smallest omega given is {values.min():.6e} rad/s"
        )
        raise ModeError(msg)
    return values


@dataclass(frozen=True)
class GnlseProblem:
    """A GNLSE fibre expressed as a mode-averaged envelope problem.

    Everything the translation layer needs to build a kernel problem for the GNLSE, and
    nothing else: the mode carries the dispersion and the loss, the response carries the
    nonlinearity, and the two scalars say how the shock term and any future Raman
    response are configured. The coefficients that went into the mapping are kept so
    that a run records what it was asked for.

    Attributes
    ----------
    mode : solprop.modes.protocol.Mode
        The fibre: ``beta(omega)``, the loss and the effective area.
    kerr : solprop.nonlinear.responses.KerrResponse
        The Kerr response, with ``thg=False``: an envelope GNLSE grid carries no third
        harmonic.
    raman : solprop.nonlinear.raman.RamanResponse or None
        The delayed response taking ``raman_fraction`` of the nonlinearity, or ``None``
        when Raman is off: the optmat silica bundle's sampled shape, normalised to
        :func:`raman_susceptibility` of `n2` and scaled by its own fraction.
    bundle : solprop.physdata.glasses.SilicaBundle or None
        The optmat silica bundle the fraction, the shape or ``n2`` came from;
        ``None`` when none was used. Recorded in the run's material
        manifest, not in its configuration.
    fraction_origin : str
        Where ``raman_fraction`` came from: ``"user"`` (given), ``"bundle"`` (the
        bundle's ``f_R``, Raman on) or ``"raman-off"`` (0, Raman off and none
        given).
    n2_origin : str
        ``"user"`` for a given ``gamma`` or ``n2``, ``"bundle"`` for
        ``n2="recommended"``, the bundle's total ``n2`` at the carrier.
    omega0 : float
        Reference angular frequency the mapping used, rad/s.
    gamma : float
        Nonlinear coefficient in 1/(W m), ``n2 omega0 / (c A_eff)``.
    n2 : float
        Nonlinear index in m^2/W.
    chi3 : float
        Third-order susceptibility in m^2/V^2, already reduced by the Raman fraction.
    aeff : float
        Effective area in m^2.
    raman_fraction : float
        Share of ``gamma`` held back from the Kerr term, between 0 and 1. The Raman
        response carries it when there is one; with Raman off it is zero unless a
        fraction was given, and then that share of ``gamma`` is not propagated.
    shock : bool
        Whether the normalisation prefactor keeps its ``omega`` dependence.
    prefactor_index : float
        Effective index the normalisation prefactor uses, always 1.0. The
        GNLSE's ``beta`` is a fitted dispersion rather than a physical effective index,
        so ``c beta / omega`` need not be near one and must not be used here. Pass it
        to :func:`solprop.nonlinear.normalisation.uppe_prefactor` as ``index``.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> problem = problem_from([0.0, 5.0e-9, -2.0e-26], omega_ref=1.8e15, gamma=0.1)
    >>> f"{problem.gamma:.6f}", problem.raman_fraction, problem.kerr.thg
    ('0.100000', 0.0, False)
    >>> problem.responses
    (KerrResponse(chi3=..., thg=False, species=None),)
    """

    mode: Mode
    kerr: KerrResponse
    omega0: float
    gamma: float
    n2: float
    chi3: float
    aeff: float
    raman_fraction: float
    shock: bool
    prefactor_index: float = 1.0
    raman: RamanResponse | None = None
    # Provenance kept out of the configuration record, so that a run without a
    # selection records exactly what it did; the material manifest records it.
    bundle: SilicaBundle | None = field(default=None, metadata={"record": False})
    fraction_origin: str = field(default="", metadata={"record": False})
    n2_origin: str = field(default="user", metadata={"record": False})

    @property
    def responses(self) -> tuple[object, ...]:
        """Return the declared responses, in the order the compiler is given them.

        Returns
        -------
        tuple
            The Kerr response, and the Raman response beside it when one was asked for.
        """
        return (self.kerr,) if self.raman is None else (self.kerr, self.raman)


def problem_from(  # noqa: PLR0913 - the GNLSE's parameters, each named in ADR-026
    betas: Float[np.ndarray, " n_deg"] | list[float],
    *,
    omega_ref: float,
    gamma: float | None = None,
    n2: float | str | None = None,
    aeff: float = 1.0,
    loss_db_per_m: float = 0.0,
    raman: bool = False,
    raman_fraction: float | None = None,
    shock: bool = True,
    selection: MaterialSelection | None = None,
) -> GnlseProblem:
    """Map GNLSE parameters onto a mode-averaged envelope problem.

    The mapping:

    - ``k0 = omega_ref / c`` and ``n2 = gamma A_eff / k0``, with ``A_eff = 1`` unless
      one is given, so that ``gamma = n2 omega_ref / (c A_eff)`` either way;
    - ``chi3 = (4/3) (1 - f_R) n2 eps0 c``, the ``4/3`` undoing the ``3/4`` of the
      envelope Kerr response and the linear index taken as one;
    - with ``raman=True``, a fused-silica :class:`solprop.nonlinear.raman.RamanResponse`
      of integral :func:`raman_susceptibility` and fraction ``f_R`` beside it, so that
      the two together carry the whole of ``gamma``;
    - the linear operator sees ``beta(omega)`` directly and the prefactor sees
      ``n_eff = 1``, which follows from ``n_eff = c beta / omega`` only in the sense
      that the GNLSE's ``beta`` already contains the whole dispersion;
    - the shock term is the ``omega / omega_ref`` dependence of that prefactor rather
      than a time derivative, so ``shock=False`` simply freezes it.

    With ``raman=False`` and no `raman_fraction` the ``(1 - f_R)`` factor is not
    applied, so none of the Kerr nonlinearity is discarded: the textbook "no Raman",
    ``f_R = 0`` at fixed ``gamma``. With ``raman=False`` and an explicit
    `raman_fraction` the factor is applied and nothing carries the rest, so a pair of
    runs that gives the same fraction and switches only `raman` differs by the delayed
    response alone.

    Parameters
    ----------
    betas : array_like
        Taylor coefficients of ``beta`` about ``omega_ref``, ``beta_0, beta_1, ...`` in
        s^n/m. ``beta_0`` and ``beta_1`` only shift the frame and do not change the
        physics.
    omega_ref : float
        Reference angular frequency, rad/s, positive. Both the Taylor expansion and the
        ``gamma`` mapping refer to it.
    gamma : float or None, optional
        Nonlinear coefficient in 1/(W m). Give this or `n2`, not both.
    n2 : float, str or None, optional
        Nonlinear index in m^2/W. Give this or `gamma`, not both.
        ``"recommended"`` takes the silica bundle's total long-pulse ``n2``
        (optmat's ``kerr/SiO2``) at the carrier ``2 pi c / omega_ref``.
    aeff : float, optional
        Effective area in m^2. Default 1.0, the GNLSE convention in which the area is
        already folded into ``gamma``.
    loss_db_per_m : float, optional
        Power loss in dB/m, converted by :func:`attenuation_from_db`. Default 0.
    raman : bool, optional
        Whether a Raman response takes the share `raman_fraction` of ``gamma``. Default
        ``False``.
    raman_fraction : float or None, optional
        Share of ``gamma`` held back from the Kerr term, between 0 and 1. ``None``, the
        default, is the silica bundle's own ``f_R`` with `raman` true -- 0.2188 for
        optmat's default bundle (SEP 3, D4) -- and 0 with it false, which gives the
        whole of ``gamma`` to the Kerr term. A value given is always held back, with
        `raman` true or false, and overrides the bundle's.
    shock : bool, optional
        Keep the ``omega`` dependence of the normalisation prefactor. Default ``True``.
    selection : solprop.physdata.selection.MaterialSelection or None, optional
        ``None`` is ``MaterialSelection()``. Fused silica's nonlinear response is
        optmat's silica bundle (SEP 3, section 4.5): the bundle named by
        ``selection.raman`` (``"recommended"`` is the ``"default"`` bundle,
        Hollenbeck and Cantrell's shape with the Kerr fit's ``f_R``; another name is
        an optmat bundle name, an object is taken as it is), its ``f_R`` for an
        omitted fraction with Raman on, its normalised ``h_R`` sampled as the Raman
        response, and its ``n2`` for ``n2="recommended"``.

    Returns
    -------
    GnlseProblem
        The mode, the Kerr response and the coefficients they came from.

    Raises
    ------
    ModeError
        If neither or both of `gamma` and `n2` are given, if `raman_fraction` is
        outside ``[0, 1]``, if `omega_ref` or `aeff` is not positive, if `raman` is
        true with a zero fraction, or if `n2` is a string other than
        ``"recommended"``.

    Notes
    -----
    ``raman=True`` with ``raman_fraction=0`` raises rather than returning a problem
    whose Raman response carries none of the nonlinearity: the response itself refuses a
    fraction outside ``(0, 1]``, and a run that declared a term contributing nothing
    would pay for a convolution at every stage and be indistinguishable from one that
    did not declare it.

    Examples
    --------
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> problem = problem_from([0.0, 5.0e-9], omega_ref=1.8e15, gamma=0.1, aeff=1.0)
    >>> f"{problem.n2:.6e}", f"{problem.chi3:.6e}"
    ('1.665514e-08', '5.894628e-11')
    >>> raman = problem_from([0.0, 5.0e-9], omega_ref=1.8e15, gamma=0.1, raman=True)
    >>> f"{raman.chi3 / problem.chi3:.2f}", f"{raman.raman_fraction:.4f}"
    ('0.78', '0.2188')
    >>> held = problem_from(
    ...     [0.0, 5.0e-9], omega_ref=1.8e15, gamma=0.1, raman_fraction=0.18
    ... )
    >>> f"{held.chi3 / problem.chi3:.2f}", held.raman is None
    ('0.82', True)
    """
    if (gamma is None) == (n2 is None):
        msg = "give exactly one of gamma and n2"
        raise ModeError(msg, parameter=("gamma", "n2"))
    selection = MaterialSelection() if selection is None else selection
    if isinstance(n2, str) and n2 != "recommended":
        msg = (
            f"n2={n2!r}: the one name n2 takes is 'recommended', fused silica's total "
            "n2 from optmat's silica bundle at the carrier; otherwise give n2 in m^2/W"
        )
        raise ModeError(msg, parameter="n2")
    bundle = (
        SilicaBundle.resolve(selection.raman) if raman or isinstance(n2, str) else None
    )
    # An omitted fraction is the bundle's with Raman on and none with it off; a given
    # one is always held back from the Kerr term.
    fraction, origin = _held_back_fraction(raman, raman_fraction, bundle)
    if not 0.0 <= fraction <= 1.0:
        msg = f"raman_fraction must lie in [0, 1], got {raman_fraction!r}"
        raise ModeError(msg, parameter="raman_fraction")
    if not aeff > 0.0:
        msg = f"aeff must be positive, got {aeff!r}"
        raise ModeError(msg, parameter="aeff")
    if raman and fraction == 0.0:
        msg = (
            "raman=True with raman_fraction=0 declares a Raman response that carries "
            "none of the nonlinearity; call with raman=False instead, which gives the "
            "whole of gamma to the Kerr term"
        )
        raise ModeError(msg, parameter=("raman", "raman_fraction"))

    mode = SpecifiedMode.from_taylor(
        betas, omega_ref, loss=attenuation_from_db(loss_db_per_m), aeff=aeff
    )
    wavenumber = omega_ref / C
    if isinstance(n2, str):
        assert bundle is not None
        nonlinear_index: float = bundle.n2(2.0 * math.pi * C / omega_ref)
    elif gamma is not None:
        nonlinear_index = gamma * aeff / wavenumber
    else:
        assert n2 is not None
        nonlinear_index = float(n2)
    coefficient = nonlinear_index * wavenumber / aeff

    susceptibility = kerr_susceptibility(nonlinear_index, fraction)
    # thg=False on both responses: an envelope GNLSE grid carries no third harmonic, and
    # the translation layer checks that the two agree.
    delayed = None
    if raman:
        assert bundle is not None
        delayed = bundle_response(
            bundle,
            amplitude=raman_susceptibility(nonlinear_index),
            fraction=fraction,
            thg=False,
        )

    logger.debug(
        "GNLSE mapping: gamma=%.6e 1/(W m), n2=%.6e m^2/W, chi3=%.6e m^2/V^2, "
        "aeff=%.6e m^2, f_R=%.3f, shock=%s",
        coefficient,
        nonlinear_index,
        susceptibility,
        aeff,
        fraction,
        shock,
    )
    return GnlseProblem(
        mode=mode,
        kerr=KerrResponse(chi3=susceptibility, thg=False),
        omega0=float(omega_ref),
        gamma=float(coefficient),
        n2=float(nonlinear_index),
        chi3=float(susceptibility),
        aeff=float(aeff),
        raman_fraction=float(fraction),
        shock=bool(shock),
        raman=delayed,
        bundle=bundle,
        fraction_origin=origin,
        n2_origin="bundle" if isinstance(n2, str) else "user",
    )


def _held_back_fraction(
    raman: bool, raman_fraction: float | None, bundle: SilicaBundle | None
) -> tuple[float, str]:
    """Return the share of ``gamma`` held back from the Kerr term, and its origin.

    The rule of ``docs/guide/waveguides/gnlse.md`` ("The Raman fraction is not silently
    taken"):
    a given fraction is always held back; an omitted one is the silica bundle's
    ``f_R`` with Raman on (SEP 3, D4) and zero with Raman off.

    Parameters
    ----------
    raman : bool
        Whether a Raman response takes the fraction.
    raman_fraction : float or None
        The fraction given, or ``None``.
    bundle : SilicaBundle or None
        The selection's silica bundle, resolved when Raman is on.

    Returns
    -------
    tuple
        The fraction and ``"user"``, ``"bundle"`` or ``"raman-off"``.
    """
    if raman_fraction is not None:
        return float(raman_fraction), "user"
    if not raman:
        return 0.0, "raman-off"
    assert bundle is not None
    return bundle.raman_fraction, "bundle"
