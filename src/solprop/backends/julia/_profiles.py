# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lower independent profile statistics through the shared scalar representation.

The measured fit policy is SEP 47 Part II's profile-statistics policy: reuse a
compatible translated table at ``1e-13`` of the requested profile's scale, otherwise
try the shared adaptive builder at its existing tolerances and accept the closest
table within ``1e-12``. The latter allowance covers the recorded platform-libm
variation; it is a measured discrepancy, not a uniform-error certificate.

Only preparation evaluates the Python profile. Julia receives owned numerical
coefficients in the profile's existing units, with no callback or originating object.
The aggregate preparation owns caching and the ``statistics_tables`` record.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float

from solprop.backends.julia._descriptors import TableTransfer, lower_table
from solprop.exceptions import BackendError, SolpropError
from solprop.profiles import ZProfile
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.linop import DEFAULT_MARGIN, profile_table
from solprop.translate.problem import (
    PropagationProblem,
    _walk_tables,  # pyright: ignore[reportPrivateUsage] - shared translated-table inventory
)

__all__ = ["ProfileLowering", "lower_statistic_profile"]

_ProfileName = Literal["density", "pressure", "core_radius"]
_PROFILE_NAMES = ("density", "pressure", "core_radius")

# SEP 47 Part II records this target and maximum, including the platform-libm
# motivation. The check set and refinement sequence retain its native implementation.
_FIT_TARGET = 1e-13
_PROFILE_TOLERANCE = 1e-12
_FIT_TOLERANCES = (1e-12, 1e-13, 1e-14)
_CHECK_POINTS = 4097


@dataclass(frozen=True, eq=False)
class ProfileLowering:
    """Own one represented statistic and its measured discrepancy.

    Attributes
    ----------
    value : float or TableTransfer
        Exact represented constant or independently owned, read-only scalar table.
        Units are those of the supplied profile: m^-3 for density, catalogue bar
        for pressure, and m for core radius. No unit conversion is performed.
    source : str
        Dotted name of a reused translated table, or ``"lowered profile"``.
    max_error : float
        Largest absolute discrepancy on the check set, in the profile's units.
    relative_error : float
        Discrepancy divided by the profile's largest magnitude on that set; zero
        for an exactly zero profile. This is a measurement, not an error proof.
    check_points : int
        Number of distinct positions checked, including endpoints and knots.
    """

    value: float | TableTransfer
    source: str
    max_error: float
    relative_error: float
    check_points: int

    def record(self) -> dict[str, str | float | int]:
        """Return fresh JSON-compatible metadata for ``statistics_tables``.

        Returns
        -------
        dict
            The existing ``source``, ``max_error``, ``relative_error`` and
            ``check_points`` fields, without numerical arrays or Python profiles.
        """
        return {
            "source": self.source,
            "max_error": self.max_error,
            "relative_error": self.relative_error,
            "check_points": self.check_points,
        }


def _check_set(
    profile: ZProfile,
    start: float,
    end: float,
    edges: Float[np.ndarray, " n_edges"],
) -> Float[np.ndarray, " n_z"]:
    """Combine uniform positions, declared knots, table edges and midpoints.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        Requested scalar path, whose knots are in m.
    start, end : float
        Increasing measurement endpoints, in m.
    edges : numpy.ndarray
        Candidate interval boundaries, in m, shape ``(n_edges,)``.

    Returns
    -------
    numpy.ndarray
        Sorted distinct measurement positions, in m, shape ``(n_z,)``.
    """
    declared = np.asarray(profile.breakpoints)
    if (
        declared.ndim != 1
        or declared.dtype.kind not in "fiu"
        or not np.all(np.isfinite(declared))
    ):
        raise BackendError("profile breakpoints must be a finite real vector")
    knots = np.asarray(declared, dtype=np.float64)
    middle = 0.5 * edges[:-1] + 0.5 * edges[1:]
    return np.unique(
        np.concatenate(
            (
                np.linspace(start, end, _CHECK_POINTS),
                knots[(knots >= start) & (knots <= end)],
                edges[(edges >= start) & (edges <= end)],
                middle[(middle >= start) & (middle <= end)],
            )
        )
    )


def _measured(
    table: PiecewiseChebyshev,
    profile: ZProfile,
    check: Float[np.ndarray, " n_z"],
) -> tuple[float, float]:
    """Measure scalar discrepancies with the shared pure Clenshaw evaluator.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev
        Candidate real scalar representation.
    profile : solprop.profiles.ZProfile
        Exact path, in the same units as the candidate.
    check : numpy.ndarray
        Measurement positions in m, shape ``(n_z,)``.

    Returns
    -------
    tuple of float
        Maximum absolute discrepancy and profile magnitude, in the profile's units.
    """
    sampled = np.asarray(profile(check))
    if (
        sampled.shape != check.shape
        or sampled.dtype.kind not in "fiu"
        or not np.all(np.isfinite(sampled))
    ):
        raise BackendError(
            "profile samples must be finite real scalars at every position"
        )
    with np.errstate(over="raise", invalid="raise"):
        exact = np.asarray(sampled, dtype=np.float64)
    # PLAIN is the shared default: preparation must not import native code or Julia.
    tabulated = np.asarray(table.at_positions(check), dtype=np.float64).reshape(-1)
    if tabulated.shape != check.shape or not np.all(np.isfinite(tabulated)):
        raise BackendError("represented profile samples must be finite real scalars")
    scale = float(np.max(np.abs(exact), initial=0.0))
    # An unrelated finite candidate may differ by more than floatmax. That makes
    # it unsuitable; it does not make the requested profile itself invalid.
    with np.errstate(over="ignore"):
        error = float(np.max(np.abs(tabulated - exact), initial=0.0))
    return error, scale


def _covers(table: PiecewiseChebyshev, start: float, end: float) -> bool:
    """Check that a real scalar candidate spans every measurement position.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev
        A candidate of any represented shape and dtype.
    start, end : float
        Measurement endpoints, in m.

    Returns
    -------
    bool
        Whether packed width, dtype and admitted span permit scalar comparison.
    """
    _, lower, upper, coefficients = table.packed()
    return (
        coefficients.shape[2] == 1
        and not np.iscomplexobj(coefficients)
        and lower <= start
        and end <= upper
    )


def _lowered(
    table: PiecewiseChebyshev, source: str, error: float, scale: float, count: int
) -> ProfileLowering:
    """Copy an accepted representation and its measured error metadata.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev
        Accepted real scalar table.
    source : str
        Candidate's dotted name or the independent-fit description.
    error, scale : float
        Maximum absolute discrepancy and profile magnitude, in the profile's units.
    count : int
        Number of distinct measurement positions.

    Returns
    -------
    ProfileLowering
        Owned immutable data; only structurally exact constants become scalars.
    """
    _, _, _, coefficients = table.packed()
    constant = float(coefficients[0, 0, 0])
    # A one-term real series evaluates c0 + 0*x. Nonzero c0, and positive zero,
    # preserve that exact constant throughout the admitted finite-coordinate span.
    # Negative zero can change sign with x, and padded higher terms keep their
    # original evaluation, so neither is collapsed on mathematical equality alone.
    collapsible = (
        coefficients.shape[1] == 1
        and np.all(coefficients[:, 0, 0] == constant)
        and (constant != 0.0 or not np.any(np.signbit(coefficients[:, 0, 0])))
    )
    return ProfileLowering(
        constant if collapsible else lower_table(table),
        source,
        error,
        error / scale if scale > 0.0 else 0.0,
        count,
    )


def _represent(
    problem: PropagationProblem, profile: ZProfile, z_end: float
) -> ProfileLowering:
    """Reuse or refine a scalar table under the existing measurement policy.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Scientific descriptor with the launch and candidate representations.
    profile : solprop.profiles.ZProfile
        Requested callable, already in its statistic's catalogue units.
    z_end : float
        Last requested position, in m.

    Returns
    -------
    ProfileLowering
        The first target-compatible candidate or closest admissible fitted table.
    """
    start, end = float(problem.initial.z), float(z_end)
    if not math.isfinite(start) or not math.isfinite(end):
        raise BackendError("profile measurement endpoints must be finite")
    if not end > start:
        end = start + max(abs(start), 1.0) * 1e-12
    lower = min(start, 0.0)
    upper = max(end, end + abs(end) * DEFAULT_MARGIN)
    if not math.isfinite(upper) or not upper > lower:
        raise BackendError("profile representation span is not finite and increasing")
    for source, table in _walk_tables(problem):
        if not _covers(table, start, end):
            continue
        check = _check_set(profile, start, end, table.breakpoints)
        error, scale = _measured(table, profile, check)
        if error <= _FIT_TARGET * scale:
            return _lowered(table, source, error, scale, int(check.size))
    best = math.inf
    closest: ProfileLowering | None = None
    for tolerance in _FIT_TOLERANCES:
        table = profile_table(profile, lower, upper, tol=tolerance)
        check = _check_set(profile, start, end, table.breakpoints)
        error, scale = _measured(table, profile, check)
        if error <= _FIT_TARGET * scale:
            return _lowered(table, "lowered profile", error, scale, int(check.size))
        relative = error / scale if scale > 0.0 else math.inf
        if relative < best:
            best = relative
            closest = _lowered(table, "lowered profile", error, scale, int(check.size))
    if closest is not None and best <= _PROFILE_TOLERANCE:
        return closest
    raise BackendError(
        "profile cannot be tabulated to rounding level over the run "
        f"(best measured relative error {best:.3g}, above {_PROFILE_TOLERANCE:g})"
    )


def lower_statistic_profile(
    problem: PropagationProblem, name: _ProfileName, profile: ZProfile, z_end: float
) -> ProfileLowering:
    """Represent one requested profile independently for Julia statistics.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Scientific descriptor supplying the launch and candidate scalar tables.
    name : {'density', 'pressure', 'core_radius'}
        Statistic named in diagnostics and the caller's measurement record.
    profile : solprop.profiles.ZProfile
        Exact callable to represent, already in the catalogue's units. Its declared
        breakpoints are mandatory fit edges. Pressure is already in bar.
    z_end : float
        Last requested position in m. A nonmarching run retains the existing small
        measurement interval beyond its launch; the fit includes the shared margin.

    Returns
    -------
    ProfileLowering
        Owned numerical value and measured error metadata. It retains neither input
        object and is not cached here; aggregate preparation may cache its result.

    Raises
    ------
    BackendError
        Naming Julia and the statistic for malformed samples or bounds, an unsupported
        statistic, or a profile no shared adaptive fit resolves within the established
        measured tolerance. Original evaluation/translation errors remain chained.

    Notes
    -----
    Candidate tables and new fits are checked at 4097 uniform positions plus profile
    knots, table edges and midpoints. The target ``1e-13``, fallback maximum ``1e-12``
    and refinement tolerances are the existing SEP 47 Part II profile-statistics
    policy. Every accepted record contains its actual measured discrepancy. There is
    no native import, Julia startup, material inference or runtime callback.
    """
    if name not in _PROFILE_NAMES:
        raise BackendError(f"julia does not admit profile statistic {name!r}")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _represent(problem, profile, z_end)
    except (SolpropError, ValueError, TypeError, ArithmeticError) as error:
        raise BackendError(f"julia {name} statistic: {error}") from error
