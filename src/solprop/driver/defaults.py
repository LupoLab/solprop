# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The route defaults: one table, and the pure functions that resolve a default.

Each interface (``prop_gnlse``, ``prop_capillary``, ``prop_waveguide``, ``prop_radial``,
``prop_free`` and ``prop_planar``, and their ``Simulation.from_*`` constructors) steps
with its own integrator and tolerances, carries its own noise and applies the temporal
collar in its own scheme, each qualified separately (SEP 37, SEP 21). This module holds
those choices in one frozen table, :data:`ROUTE_DEFAULTS`, and resolves them in one
place, so that the constructors, the metadata a front end reads (SEP 41) and the tests
cannot disagree. The guide's table of the defaults is
``docs/guide/numerics/choosing-a-stepper.md``.

A default is resolved once, when the settings are built, and the choice is recorded
(:class:`solprop.output.schema.NoiseResolution` for the noise,
:class:`solprop.output.schema.DefaultsResolution` for the rest), so a run that took a
default says so in its summary and its record.

Layer D of ``docs/dev/architecture.md`` section 3, beside the constructors that read it.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from solprop.exceptions import DriverError, SolpropWarning
from solprop.output.schema import NoiseResolution
from solprop.stepper.controller import PIController
from solprop.stepper.multistep import Multistep
from solprop.stepper.runge_kutta import RungeKutta
from solprop.translate.masks import AbsorberScheme
from solprop.translate.precision import SINGLE_RTOL_CAP, SINGLE_RTOL_FLOOR

__all__ = [
    "CAPILLARY_RTOL",
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "FREE_ATOL",
    "FREE_NOISE",
    "FREE_RTOL",
    "INTERIOR_LAUNCH_NOISE",
    "PLANAR_NOISE",
    "ROUTE_DEFAULTS",
    "SINGLE_PRECISION_NOISE",
    "SLAB_NOISE",
    "SPATIAL_ATOL",
    "SPATIAL_RTOL",
    "TDSE_NOISE",
    "WAVEGUIDE_ATOL",
    "WAVEGUIDE_RTOL",
    "NoiseModel",
    "NoisePolicy",
    "RouteDefaults",
    "RouteName",
    "default_controller",
    "default_integrator",
    "default_rtol",
    "default_scheme",
    "resolve_noise",
    "route_defaults",
    "rtol_in_precision",
]

DEFAULT_RTOL = 1e-7
"""Relative tolerance a GNLSE run asks for unless told otherwise.

Set by SEP 37's convergence protocol on the supercontinuum of Dudley, Genty and Coen
(the benchmark suite's ``gnlse_raman_shock``): the loosest of three levels a decade
apart at which the ensemble-mean spectrum stays within 0.5 dB above -40 dB of its peak
and the coherence within 0.02, over paired seeds, with the default ``RungeKutta()``.
A noiseless soliton-recurrence study asks for its own, tighter value.
"""

DEFAULT_ATOL = 1e-14
"""Absolute tolerance, in the square root of watts, of the same default."""

CAPILLARY_RTOL = 1e-6
"""Relative tolerance a capillary run asks for unless told otherwise.

The default of :meth:`~solprop.driver.simulation.Simulation.from_capillary` and
:func:`~solprop.driver.interface.prop_capillary` with their default ``Multistep()``.
SEP 37's convergence protocol on the benchmark suite's ``rdw`` case passes it, and a
noisy comb-to-continuum check needs it with this integrator (``1e-5`` fails the
coherence budget there). It is a starting point, not a guarantee: a production run
should calibrate it (``docs/dev/stepper.md`` section 3).
"""

WAVEGUIDE_RTOL = 1e-5
"""Relative tolerance a modal waveguide run asks for unless told otherwise.

The default of :meth:`~solprop.driver.simulation.Simulation.from_waveguide` and
:func:`~solprop.driver.interface.prop_waveguide` with their default ``RungeKutta()``,
set by SEP 37's convergence protocol on the benchmark suite's ``grin_nonlinear``
window. A starting point, not a guarantee: a production run should calibrate it.
"""

WAVEGUIDE_ATOL = 1e-10
"""Absolute tolerance of a capillary or modal waveguide run, in sqrt(W).

The companion of :data:`CAPILLARY_RTOL` and :data:`WAVEGUIDE_RTOL`. It was never
binding in SEP 37's measurements.
"""

SPATIAL_RTOL = 1e-5
"""Relative tolerance of the controller a radial or planar run takes by default.

Set by SEP 37's convergence protocol on the benchmark suite's ``radial_filament_256``
and ``planar`` cases. It applies only where the caller's
:class:`~solprop.driver.simulation.SimulationSettings` gives no controller.
"""

SPATIAL_ATOL = 0.0
"""Absolute tolerance of the same default controller: none, as the suite runs it."""

FREE_RTOL = 1e-4
"""Relative tolerance of the controller a Cartesian or line run takes by default.

With the default ``RungeKutta()`` (SEP 37): the Adams history's memory rules the
multistep method out for a large transverse state. On a self-focusing Kerr beam the
spectrum stays within 0.002 dB of a tight reference even at 1e-3; 1e-4 cuts the field
error by 2.5 times for a sixth more steps. Applies only where the caller's
:class:`~solprop.driver.simulation.SimulationSettings` gives no controller.
"""

FREE_ATOL = 0.0
"""Absolute tolerance of the same default controller."""

SINGLE_PRECISION_NOISE = (
    "single precision carries no noise: binary32 rounding swamps a one-photon seed"
)
"""Why a default resolves to no noise under ``precision="single"`` (SEP 37)."""

PLANAR_NOISE = "the hybrid planar grid carries no noise: no photon measure is declared"
"""Why a planar run carries no noise (SEP 37)."""

SLAB_NOISE = (
    "a slab mode set carries no noise: no photon measure is declared for its "
    "amplitudes per unit width"
)
"""Why a waveguide run of slab modes carries no noise, as the planar grid does not.

A slab amplitude is in sqrt(W/m), and the one-photon draw is defined per mode in
sqrt(W); one photon per metre of width is a convention no derivation has fixed for
these modes (the line grid declares its own), so a default resolves to none and an
explicit model is refused.
"""

FREE_NOISE = (
    "a Cartesian or line run carries no noise by default: it is slow, and rarely about "
    "noise sensitivity"
)
"""Why a default resolves to no noise on the free-space interface (SEP 37)."""

TDSE_NOISE = (
    "a TDSE run carries no noise by default: it is expensive, rarely repeated as an "
    "ensemble, and its checks are coherent"
)
"""Why a default resolves to no noise beside the TDSE response (SEP 37)."""

INTERIOR_LAUNCH_NOISE = (
    "an interior launch takes its spectrum as given, which may already carry a draw"
)
"""Why a default noise is dropped at an interior launch (SEP 37)."""

NoiseModel = Literal["input", "modified"]
"""Which shot-noise model a run carries; ``None`` carries none.

The two models are described in ``docs/guide/media/noise.md``.
"""

RouteName = Literal[
    "gnlse",
    "capillary",
    "capillary tdse",
    "waveguide",
    "waveguide slab",
    "radial",
    "free",
    "planar",
]
"""A row of :data:`ROUTE_DEFAULTS`.

The six interfaces, with two variants whose noise differs: a capillary beside the TDSE
response, and a waveguide of slab modes. The Cartesian and line grids share ``free``.
"""

NoisePolicy = Literal["input", "quiet", "unmeasured"]
"""What a route's ``noise="default"`` does.

``"input"`` draws the one-photon-per-mode seed. ``"quiet"`` takes none by default, for
the row's reason, and accepts an explicit model. ``"unmeasured"`` takes none and refuses
an explicit model, because no photon measure is declared for the representation.
"""


@dataclass(frozen=True)
class RouteDefaults:
    """One route's defaults: how it steps, what noise it carries, how its collar acts.

    Attributes
    ----------
    integrator : {'Multistep', 'RungeKutta'}
        The method a binary64 run steps with when none is named. Single precision
        always takes ``RungeKutta()`` (SEP 33).
    rtol : float
        The binary64 relative tolerance; :func:`default_rtol` gives it in single
        precision.
    atol : float
        The absolute tolerance, in the route's field unit.
    noise : {'input', 'quiet', 'unmeasured'}
        What ``noise="default"`` resolves to (:data:`NoisePolicy`).
    noise_reason : str or None
        Why the default carries no noise; ``None`` where it carries the input noise.
    absorber_scheme : {'nonlinear', 'split_step'}
        How the temporal collar is applied when no scheme is named and planes are not
        interpolated; an interpolated run takes ``"nonlinear"`` (SEP 21).

    Examples
    --------
    >>> from solprop.driver.defaults import ROUTE_DEFAULTS
    >>> ROUTE_DEFAULTS["capillary"].integrator, ROUTE_DEFAULTS["capillary"].rtol
    ('Multistep', 1e-06)
    """

    integrator: Literal["Multistep", "RungeKutta"]
    rtol: float
    atol: float
    noise: NoisePolicy
    noise_reason: str | None
    absorber_scheme: AbsorberScheme


ROUTE_DEFAULTS: Mapping[str, RouteDefaults] = MappingProxyType(
    {
        "gnlse": RouteDefaults(
            "RungeKutta", DEFAULT_RTOL, DEFAULT_ATOL, "input", None, "nonlinear"
        ),
        "capillary": RouteDefaults(
            "Multistep", CAPILLARY_RTOL, WAVEGUIDE_ATOL, "input", None, "nonlinear"
        ),
        "capillary tdse": RouteDefaults(
            "Multistep",
            CAPILLARY_RTOL,
            WAVEGUIDE_ATOL,
            "quiet",
            TDSE_NOISE,
            "nonlinear",
        ),
        "waveguide": RouteDefaults(
            "RungeKutta", WAVEGUIDE_RTOL, WAVEGUIDE_ATOL, "input", None, "nonlinear"
        ),
        "waveguide slab": RouteDefaults(
            "RungeKutta",
            WAVEGUIDE_RTOL,
            WAVEGUIDE_ATOL,
            "unmeasured",
            SLAB_NOISE,
            "nonlinear",
        ),
        "radial": RouteDefaults(
            "Multistep", SPATIAL_RTOL, SPATIAL_ATOL, "input", None, "split_step"
        ),
        "free": RouteDefaults(
            "RungeKutta", FREE_RTOL, FREE_ATOL, "quiet", FREE_NOISE, "split_step"
        ),
        "planar": RouteDefaults(
            "Multistep",
            SPATIAL_RTOL,
            SPATIAL_ATOL,
            "unmeasured",
            PLANAR_NOISE,
            "split_step",
        ),
    }
)
"""The defaults of every route, keyed by :data:`RouteName`.

The binary64 tolerances and integrators were qualified per route by SEP 37's convergence
protocol and are deliberately not aligned (SEP 41, Q3); the collar schemes are SEP 21's.
"""


def route_defaults(route: str) -> RouteDefaults:
    """Return a route's row of :data:`ROUTE_DEFAULTS`.

    Parameters
    ----------
    route : str
        A :data:`RouteName`.

    Returns
    -------
    RouteDefaults
        The row.

    Raises
    ------
    DriverError
        If `route` is not a row of the table.

    Examples
    --------
    >>> from solprop.driver.defaults import route_defaults
    >>> route_defaults("free").noise_reason.startswith("a Cartesian or line run")
    True
    """
    try:
        return ROUTE_DEFAULTS[route]
    except KeyError:
        routes = sorted(ROUTE_DEFAULTS)
        msg = f"no route defaults for {route!r}; the routes are {routes}"
        raise DriverError(msg) from None


def default_rtol(route: str, precision: str) -> float:
    """Return a route's default relative tolerance in the run's precision.

    The one rule every route follows. Binary64 takes the route's own. Binary32 takes the
    larger of it and :data:`~solprop.translate.precision.SINGLE_RTOL_FLOOR`, so it is
    never tighter than its binary64 twin nor than binary32 rounding can serve, and at
    most :data:`~solprop.translate.precision.SINGLE_RTOL_CAP`, the loosest value the
    single-precision survival record measured (the maintainers' ruling of 2026-10-28).

    Parameters
    ----------
    route : str
        A :data:`RouteName`.
    precision : {'double', 'single'}
        The run's precision.

    Returns
    -------
    float
        The default relative tolerance.

    Examples
    --------
    >>> from solprop.driver.defaults import default_rtol
    >>> [default_rtol(route, "single") for route in ("gnlse", "capillary", "free")]
    [1e-06, 1e-06, 1e-05]
    >>> default_rtol("gnlse", "double")
    1e-07
    """
    return rtol_in_precision(route_defaults(route).rtol, precision)


def rtol_in_precision(rtol: float, precision: str) -> float:
    """Return a binary64 default tolerance as the run's precision takes it.

    The rule :func:`default_rtol` applies to a route's row, for any binary64 value.

    Parameters
    ----------
    rtol : float
        The binary64 default.
    precision : {'double', 'single'}
        The run's precision.

    Returns
    -------
    float
        `rtol` in binary64; clamped to the single-precision floor and cap otherwise.

    Examples
    --------
    >>> from solprop.driver.defaults import rtol_in_precision
    >>> [rtol_in_precision(rtol, "single") for rtol in (1e-8, 1e-5, 1e-3)]
    [1e-06, 1e-05, 1e-05]
    """
    if precision != "single":
        return rtol
    return min(max(rtol, SINGLE_RTOL_FLOOR), SINGLE_RTOL_CAP)


def default_controller(route: str, precision: str) -> PIController:
    """Return the step controller a route takes when given no tolerances.

    Parameters
    ----------
    route : str
        A :data:`RouteName`.
    precision : {'double', 'single'}
        The run's precision.

    Returns
    -------
    solprop.stepper.controller.PIController
        :func:`default_rtol` and the route's ``atol``, at the controller's own gains.

    Examples
    --------
    >>> from solprop.driver.defaults import default_controller
    >>> controller = default_controller("radial", "double")
    >>> controller.rtol, controller.atol
    (1e-05, 0.0)
    """
    return PIController(
        rtol=default_rtol(route, precision), atol=route_defaults(route).atol
    )


def default_integrator(route: str | None, precision: str) -> Multistep | RungeKutta:
    """Return the integrator a route steps with when none is named.

    Parameters
    ----------
    route : str or None
        A :data:`RouteName`, or ``None`` for settings built by hand, which step with
        ``Multistep()``.
    precision : {'double', 'single'}
        The run's precision. Single precision refuses the Adams method until its
        survival is established (SEP 33, Q2), so it always takes ``RungeKutta()``.

    Returns
    -------
    Multistep or RungeKutta
        A new instance, at the method's own defaults.

    Examples
    --------
    >>> from solprop.driver.defaults import default_integrator
    >>> type(default_integrator("gnlse", "double")).__name__
    'RungeKutta'
    >>> type(default_integrator("capillary", "single")).__name__
    'RungeKutta'
    >>> type(default_integrator(None, "double")).__name__
    'Multistep'
    """
    if precision == "single":
        return RungeKutta()
    name = "Multistep" if route is None else route_defaults(route).integrator
    return RungeKutta() if name == "RungeKutta" else Multistep()


def default_scheme(route: str | None, *, interpolate_planes: bool) -> AbsorberScheme:
    """Return how a route applies its temporal collar when no scheme is named.

    Parameters
    ----------
    route : str or None
        A :data:`RouteName`, or ``None`` for settings built by hand, which take
        ``"nonlinear"``.
    interpolate_planes : bool
        Whether planes are interpolated inside accepted steps, which the split step
        cannot serve: such a run takes ``"nonlinear"`` (SEP 21).

    Returns
    -------
    {'nonlinear', 'split_step'}
        The scheme.

    Examples
    --------
    >>> from solprop.driver.defaults import default_scheme
    >>> default_scheme("radial", interpolate_planes=False)
    'split_step'
    >>> default_scheme("radial", interpolate_planes=True)
    'nonlinear'
    """
    if route is None or interpolate_planes:
        return "nonlinear"
    return route_defaults(route).absorber_scheme


def resolve_noise(
    route: str, noise: str | None, seed: int | None, precision: str
) -> tuple[NoiseModel | None, NoiseResolution]:
    """Resolve a constructor's ``noise`` keyword into the model a run carries (SEP 37).

    ``"default"`` follows the route's :data:`NoisePolicy`: the input noise, or none for
    the row's reason. Single precision carries none whatever the route, because binary32
    rounding swamps a one-photon seed. A seed beside a default that resolves to none is
    refused rather than silently dropped, and so is a seed beside ``noise=None``. An
    explicit model is refused where the route is ``"unmeasured"`` and warned on beside
    the TDSE response; any other value is passed through, and an unknown one is refused
    where it is drawn.

    Parameters
    ----------
    route : str
        A :data:`RouteName`.
    noise : {'default', None, 'input', 'modified'}
        The keyword as given.
    seed : int or None
        The keyword ``seed`` as given.
    precision : {'double', 'single'}
        The run's arithmetic.

    Returns
    -------
    tuple
        The model to draw, or ``None``, and the record of how it was chosen.

    Warns
    -----
    SolpropWarning
        When a TDSE run asks for a noise model explicitly.

    Raises
    ------
    DriverError
        If a seed is given and the run resolves to no noise, or a model is asked for
        on a route with no photon measure.

    Examples
    --------
    >>> from solprop.driver.defaults import resolve_noise
    >>> resolve_noise("capillary", "default", None, "double")
    ('input', NoiseResolution(requested='default', resolved='input', reason=None))
    >>> resolve_noise("free", "default", None, "double")[1].reason[:20]
    'a Cartesian or line '
    """
    row = route_defaults(route)
    if row.noise == "unmeasured" and noise not in ("default", None):
        msg = f"noise={noise!r}: {row.noise_reason}; pass noise=None"
        raise DriverError(msg)
    if noise == "default":
        fallback = SINGLE_PRECISION_NOISE if precision == "single" else row.noise_reason
        if fallback is not None:
            if seed is not None:
                msg = (
                    f"a noise seed was given, but this run takes no noise by default "
                    f"({fallback}): pass a noise model with the seed, or drop the seed"
                )
                raise DriverError(msg)
            return None, NoiseResolution("default", None, fallback)
        return "input", NoiseResolution("default", "input")
    if noise is None and seed is not None:
        msg = "a noise seed requires a noise model"
        raise DriverError(msg)
    if route == "capillary tdse" and noise is not None:
        msg = (
            f"noise={noise!r} beside the TDSE response: the atomic source is expensive "
            "and a TDSE run is rarely repeated as a noise ensemble; pass noise=None "
            "unless the noise is what this run is about"
        )
        warnings.warn(msg, SolpropWarning, stacklevel=4)
    # An unknown model passes through to be refused where it is drawn.
    return cast("NoiseModel | None", noise), NoiseResolution(noise, noise)
