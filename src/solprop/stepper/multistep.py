# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Low-call Adams integrators: the method specification and its variable-step algebra.

SEP 13 (``docs/dev/seps/archive/sep-0013-expensive-response-steppers.md``) proposes
interaction-picture Adams methods for propagation whose nonlinear evaluation dominates
the cost. This module holds the backend-agnostic half: the frozen
:class:`Multistep` specification a run carries in place of a Runge-Kutta tableau, and
the scalar algebra every backend shares -- the integrated Lagrange weights of a
variable-step Adams formula on the **actual** history positions, the leading defect
constants and the Milne factor of a predictor-corrector pair (SEP 13 section 4).

Positions enter in normalised form, ``theta = (z - z_n) / h`` with ``z_n`` the anchor
and ``h`` the step being taken, so a history point behind the anchor has a negative
node and the predicted endpoint the node 1. The weights are exact for polynomials of
degree below the node count, on any distinct nodes; constant-step Adams-Bashforth and
Adams-Moulton coefficients are the special case of equally spaced nodes [1]_. Variable
coefficients recomputed on the actual history, rather than fixed ones, follow the
variable-coefficient Adams methods of [2]_; no solver code is taken from it.

References
----------
.. [1] E. Hairer, S. P. Noersett and G. Wanner, *Solving Ordinary Differential
       Equations I*, 2nd ed., Springer (1993), sections III.1 and III.5.
.. [2] P. N. Brown, G. D. Byrne and A. C. Hindmarsh, "VODE: A Variable-Coefficient
       ODE Solver", SIAM J. Sci. Stat. Comput. 10, 1038 (1989). doi:10.1137/0910062
"""

import functools
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float
from numpy.polynomial import polynomial

from solprop._helpers import NUMBA, Helpers
from solprop.exceptions import StepperError
from solprop.stepper.schedule import LANDING_ULPS

__all__ = [
    "CARRIER_PHASE_CAP",
    "COLLAR_STABILITY",
    "MAX_ADAMS_ORDER",
    "ORDER_HYSTERESIS",
    "PEC_REAL_INTERVAL",
    "REJECTION_GROWTH",
    "RESTART_AFTER",
    "CarrierFrame",
    "ComparisonMultistep",
    "Corrector",
    "Formulation",
    "Interpolant",
    "MarchPolicy",
    "Multistep",
    "defect_integral",
    "integrated_weights",
    "lagrange_coefficients",
    "march_policy",
    "milne_factor",
]

MAX_ADAMS_ORDER = 6
"""Highest admitted order; SEP 13 section 4.4 bounds the initial study at six."""

PEC_REAL_INTERVAL = (1.0, 0.545, 0.3, 0.163, 0.0877, 0.0465)
"""Negative-real-axis stability interval of PEC ``q/(q+1)`` at constant step, by ``q``.

The predicted states of the one-order-higher PEC family advance by Adams-Bashforth of
order ``q + 1`` (SEP 13), so the interval is that formula's: ``h |lambda|`` up to 1,
6/11, 0.3, 0.1633, 0.0877 and 0.0465 for ``q = 1..6`` (rounded down), against 3.3 for
DOPRI5. Computed from the spectral radius of the PEC recurrence for
``u' = lambda u`` in ``tests/stepper/test_multistep.py``. It binds a damping term
integrated explicitly -- the in-stage collar -- in either formulation: the exponential
one integrates the diagonal operator exactly, but the collar is in its source.

The interval is the one at a residual phase ``Im x' = 0``, ``x'`` the argument of the
step's moments: ``h D`` in the exponential formulation, ``h D - sigma`` under a carrier
frame (:data:`CarrierFrame`). In a bin whose residual phase per step is nonzero the
recurrence is the same with ``x'`` in place of ``x``; every ``h |lambda|`` up to the
interval stays stable for ``|Im x'|`` up to about 5.9, 5.1, 3.5, 3.2 and 3.0 at
``q = 1..5`` without loss, and further with it (SEP 35 section 4.4, reproduced in
``tests/backends/test_carrier_frame.py``). Neither formulation enforces that condition;
the temporal collar couples bins, so the bound is a heuristic in both."""

RESTART_AFTER = 3
"""Consecutive rejections after which the history is restarted at order one."""

ORDER_HYSTERESIS = 1.2
"""Factor by which a neighbouring order's predicted step must win to be adopted."""

COLLAR_STABILITY = 1.0
"""Share of an order's negative-real stability interval the order choice allows."""

REJECTION_GROWTH = 0.9
"""Largest factor a rejected step's own defect may propose for its retry.

Below one, so that a retry is always shorter than the step it replaces, whatever the
defect's exponent makes of a norm just above one."""

CARRIER_PHASE_CAP = 1.0
"""Largest carrier phase ``h |kappa|`` a step may cover under a carrier frame, rad.

Part of the :data:`CarrierFrame` policy, not a setting (SEP 35 section 4.3): the
residual rotation per step of the source components the shift does not remove --
stationary ones at ``kappa``, the third harmonic's at ``2 kappa`` -- then stays at or
below 1 and 2 rad, short of the ``pi`` at which the samples alias and the defect stops
seeing the error. The proposal is shortened before a trial; a shortened step is not a
landing."""

Interpolant = Literal["adams", "hermite"]
"""How a comparison method interpolates save planes when they are interpolated.

``"adams"`` integrates the accepted step's own derivative polynomial, the selected
method's interpolant; ``"hermite"`` is the generic cubic Hermite reference (SEP 13
section 6.4). Whether planes are interpolated at all is the run's
``interpolate_planes``.
"""

Formulation = Literal["interaction", "exponential"]
"""How an Adams method integrates its source history (SEP 16 section 4.2).

``"interaction"`` interpolates each source sample carried into the step's anchor frame
by the exact linear flow (Lawson's interaction picture, SEP 13). ``"exponential"``,
:class:`Multistep`'s default, interpolates the untransported samples and integrates the
polynomial exactly against the linear flow (exponential Adams).
"""

CarrierFrame = Literal["off", "scalar"]
"""Which carrier phase the exponential formulation removes from its history (SEP 35).

On a field grid, and on an envelope grid with the third harmonic, the operator keeps
``beta_0`` and the source that follows the carrier rotates along ``z`` at
``kappa = beta(omega_0) - beta_1 omega_0``. ``"scalar"`` interpolates, inside each step,
``G_j = exp(sigma (1 - theta_j)) F_j`` with ``sigma = i Im x`` at the reference bin, and
integrates it against the moments at ``x - sigma``: exact for any ``sigma``, with the
state, the stored history and the linear flow untouched, and each step capped at
:data:`CARRIER_PHASE_CAP`. ``"off"``, the default, is SEP 16's arithmetic. On an
envelope grid without the third harmonic the shift is the present arithmetic exactly
(``sigma = 0``). The shift saves source calls on dense gases at matched total-field
error, but the weak bands it leaves less accurate (SEP 35 phase 4,
``docs/guide/numerics/adams.md``), which is why it is not the default. A per-band shift
of the third harmonic was developed and removed by SEP 35's gate.
"""

Corrector = Literal["none", "pec", "pece"]
"""The execution contract of a step (SEP 13 section 4.2).

``"pec"`` corrects once and stores the derivative of the predicted state, the selected
method's contract. ``"pece"`` pays a second call for the derivative of the corrected,
accepted state, and ``"none"`` is explicit Adams-Bashforth; both remain only as
comparators (:class:`ComparisonMultistep`).
"""


@dataclass(frozen=True)
class Multistep:
    """A low-call Adams method, in place of a Runge-Kutta tableau.

    SEP 13's selected method: one source call per accepted step, an Adams-Bashforth
    predictor of order ``q`` and an Adams-Moulton corrector one order higher through the
    new source (PEC), controlled on the predictor-corrector difference. Its predicted
    states advance by Adams-Bashforth of order ``q + 1``, so it has that formula's
    stability, and the accepted value is the corrector's
    (``docs/guide/numerics/adams.md``). Two formulations share that recurrence, its
    history, controller and lifecycle (SEP 16): exponential Adams, the default, and the
    interaction picture (:data:`Formulation`).

    Attributes
    ----------
    order : int
        Highest predictor order ``q`` in ``[1, 6]``; the corrector uses ``q + 1``
        derivatives. The method starts at order one and raises it as history accrues.
    variable_order : bool
        Choose the order between one and :attr:`order` step by step from the
        neighbouring orders' estimates, at no extra call (SEP 13 section 4.4). On by
        default: it settled at the best fixed order on every case SEP 13 measured.
    max_calls : int or None
        Budget of charged source calls for the run: startup, accepted and rejected
        trials and restarts alike. A trial the remaining budget cannot pay for is not
        started; the run stops with :class:`solprop.exceptions.ResponseBudgetError` at
        its last accepted step. ``None`` for no budget.
    interval_cap : bool
        Cap each step at the width of the Chebyshev linear-table interval containing
        its start, as the Runge-Kutta march does. Off by default: the cumulative
        operator is exact at any position, and on SEP 13's canonical capillary the
        cap -- dozens of micrometre intervals at a choked vacuum end -- cost 20-40 %
        of the calls for no accuracy.
    equal_approach : bool
        Divide the distance to a forced landing into two equal steps when it lies
        between one and two proposals, instead of a full step and a sliver.
    max_growth : float
        Largest ratio of one step to the step before, above one. Variable-step Adams
        stability needs it bounded (SEP 13 section 4.4).
    min_factor : float
        Smallest step-size factor after a rejection, in ``(0, 1)``.
    safety : float
        Safety factor on the step proposed from an error estimate, in ``(0, 1]``.
    formulation : {'interaction', 'exponential'}
        How the source history is integrated (:data:`Formulation`); exponential by
        default. A run's record hashes the method's representation, which names it
        (output schema 15); a record made before then that names none was stepped in
        the interaction picture.
    carrier_frame : {'off', 'scalar'}
        The carrier phase the exponential formulation removes from its history inside
        each step (:data:`CarrierFrame`, ``docs/guide/numerics/adams.md``); ``"off"`` by
        default. The shift needs the exponential formulation and is refused by name
        on a backend or route that does not deliver it. The representation names it,
        so a march snapshot taken under another policy restarts its history.

    Raises
    ------
    StepperError
        If a field is outside its range, or a carrier frame is asked of the interaction
        formulation.

    Examples
    --------
    >>> from solprop.stepper.multistep import Multistep
    >>> Multistep().name
    'exponential-adams-pec5/6-variable'
    >>> Multistep(carrier_frame="scalar").name
    'exponential-adams-pec5/6-variable-scalar-frame'
    >>> Multistep(order=3, variable_order=False, formulation="interaction").name
    'adams-pec3/4'
    """

    order: int = 5
    variable_order: bool = True
    max_calls: int | None = None
    interval_cap: bool = False
    equal_approach: bool = True
    max_growth: float = 2.0
    min_factor: float = 0.2
    safety: float = 0.8
    formulation: Formulation = "exponential"
    carrier_frame: CarrierFrame = "off"

    def __post_init__(self) -> None:
        """Check the ranges.

        Raises
        ------
        StepperError
            Naming the offending field.
        """
        # The Literal annotation refuses an unknown formulation first wherever the
        # annotations are checked at run time, as in the test suite.
        if self.formulation not in ("interaction", "exponential"):  # pragma: no cover
            msg = (
                "formulation must be 'interaction' or 'exponential', got "
                f"{self.formulation!r}"
            )
            raise StepperError(msg)
        if self.carrier_frame not in ("off", "scalar"):  # pragma: no cover
            msg = f"carrier_frame must be 'off' or 'scalar', got {self.carrier_frame!r}"
            raise StepperError(msg)
        if self.carrier_frame != "off" and self.formulation != "exponential":
            # SEP 35 section 5: the interaction picture already transports the whole
            # linear phase; a shift has nothing left to remove.
            msg = (
                f"carrier_frame={self.carrier_frame!r} needs the exponential "
                f"formulation, got formulation={self.formulation!r}"
            )
            raise StepperError(msg)
        if not 1 <= self.order <= MAX_ADAMS_ORDER:
            msg = f"order must lie in [1, {MAX_ADAMS_ORDER}], got {self.order!r}"
            raise StepperError(msg)
        if self.max_calls is not None and self.max_calls < 1:
            msg = f"max_calls must be positive or None, got {self.max_calls!r}"
            raise StepperError(msg)
        if not self.max_growth > 1.0:
            msg = f"max_growth must exceed one, got {self.max_growth!r}"
            raise StepperError(msg)
        if not 0.0 < self.min_factor < 1.0:
            msg = f"min_factor must lie in (0, 1), got {self.min_factor!r}"
            raise StepperError(msg)
        if not 0.0 < self.safety <= 1.0:
            msg = f"safety must lie in (0, 1], got {self.safety!r}"
            raise StepperError(msg)

    @property
    def corrector(self) -> Corrector:
        """Return the execution contract: one call per step, the predicted source kept.

        Returns
        -------
        str
            ``"pec"``.
        """
        return "pec"

    @property
    def name(self) -> str:
        """Return the method's record name.

        Returns
        -------
        str
            ``adams-pec<q>/<q + 1>``, with ``-variable`` when the order is chosen, an
            ``exponential-`` prefix for the exponential formulation and a
            ``-<policy>-frame`` suffix under a carrier frame.
        """
        variable = "-variable" if self.variable_order else ""
        prefix = "exponential-" if self.formulation == "exponential" else ""
        frame = "" if self.carrier_frame == "off" else f"-{self.carrier_frame}-frame"
        return f"{prefix}adams-pec{self.order}/{self.order + 1}{variable}{frame}"

    @property
    def real_stability_interval(self) -> float:
        """Return the negative-real stability interval of the highest order run.

        Returns
        -------
        float
            :data:`PEC_REAL_INTERVAL` at :attr:`order`: the largest ``h |lambda|``
            for a damping rate ``lambda`` integrated explicitly.
        """
        return PEC_REAL_INTERVAL[self.order - 1]

    def corrector_nodes(self, order: int) -> int:
        """Return the corrector's derivative count at a current predictor order.

        Parameters
        ----------
        order : int
            The predictor order in force, which is below :attr:`order` during startup.

        Returns
        -------
        int
            ``order + 1``: the corrector is one order higher than the predictor.
        """
        return order + 1


@dataclass(frozen=True)
class ComparisonMultistep(Multistep):
    """An Adams variant SEP 13 compared and did not select, kept for benchmarks.

    PECE, explicit Adams-Bashforth and the matched predictor-corrector pairs controlled
    by Milne's estimate each lost to :class:`Multistep` on every case measured (SEP 13
    Part II). They stay executable so that the comparison can be rerun, and are not part
    of the documented interface.

    Attributes
    ----------
    corrector : {'none', 'pec', 'pece'}
        The execution contract (:data:`Corrector`).
    corrector_order : int or None
        Derivatives of the corrector, the predicted one included: ``order`` for a
        matched pair or ``order + 1``, the selected family. ``None`` is ``order``.
        Ignored for ``corrector='none'``.
    interpolant : {'adams', 'hermite'}
        How the planes are interpolated when the run interpolates them
        (``interpolate_planes``, ADR-036, ADR-039): the accepted step's own derivative
        polynomial, at the step's order and no call, or the generic cubic Hermite
        interpolant of the step's end states and derivatives, a reference whose error
        is one order below what the step controls (:data:`Interpolant`).
    formulation : {'interaction'}
        The variants are stepped in the interaction picture only.

    Raises
    ------
    StepperError
        If a field is outside its range, variable order is asked of a matched pair, or
        the exponential formulation is asked of a comparison variant (SEP 16 delivers
        it for the selected family only).

    Examples
    --------
    >>> from solprop.stepper.multistep import ComparisonMultistep
    >>> ComparisonMultistep(corrector="pece", order=4).name
    'adams-pece4'
    """

    corrector: Corrector = "pece"  # pyright: ignore[reportIncompatibleMethodOverride]
    corrector_order: int | None = None
    variable_order: bool = False
    interpolant: Interpolant = "adams"
    formulation: Formulation = "interaction"

    def __post_init__(self) -> None:
        """Check the ranges, the corrector's among them.

        Raises
        ------
        StepperError
            Naming the offending field.
        """
        super().__post_init__()
        if self.formulation != "interaction":
            msg = (
                "the comparison variants use the interaction formulation only; the "
                "exponential one is delivered for Multistep (SEP 16)"
            )
            raise StepperError(msg)
        # The Literal annotation refuses an unknown corrector first wherever the
        # annotations are checked at run time, as in the test suite.
        if self.corrector not in ("none", "pec", "pece"):  # pragma: no cover
            msg = f"corrector must be 'none', 'pec' or 'pece', got {self.corrector!r}"
            raise StepperError(msg)
        if self.corrector_order is not None and self.corrector_order not in (
            self.order,
            self.order + 1,
        ):
            msg = (
                f"corrector_order must be order ({self.order}) or order + 1, got "
                f"{self.corrector_order!r}"
            )
            raise StepperError(msg)
        if self.variable_order and (
            self.corrector == "none" or self.corrector_order != self.order + 1
        ):
            msg = (
                "variable_order needs the one-order-higher corrector "
                "(corrector 'pec' or 'pece' with corrector_order = order + 1)"
            )
            raise StepperError(msg)

    @property
    def name(self) -> str:
        """Return the variant's record name.

        Returns
        -------
        str
            ``adams-<corrector><order>``, with ``/<corrector order>`` for the
            higher-order corrector, ``ab`` standing for ``none``.
        """
        stem = "ab" if self.corrector == "none" else self.corrector
        extended = (
            self.corrector != "none"
            and self.corrector_order is not None
            and self.corrector_order != self.order
        )
        suffix = f"/{self.corrector_order}" if extended else ""
        variable = "-variable" if self.variable_order else ""
        return f"adams-{stem}{self.order}{suffix}{variable}"

    def corrector_nodes(self, order: int) -> int:
        """Return the corrector's derivative count at a current predictor order.

        Parameters
        ----------
        order : int
            The predictor order in force.

        Returns
        -------
        int
            ``order`` for a matched pair, ``order + 1`` for the higher-order corrector.
        """
        if self.corrector_order is None or self.corrector_order == self.order:
            return order
        return order + 1


@dataclass(frozen=True)
class MarchPolicy:
    """Every number the Adams march reads and never changes during a run (SEP 39).

    solprop forms one per run, from the method, the run's stepper settings, the
    operator's breakpoints and kinks, the in-stage collar and this module's constants
    (:func:`march_policy`). The Python march reads its controller and landing numbers
    from it. The native march receives it whole and holds no default and no constant
    of its own, so every number either execution uses has this one source (SEP 39
    section 6.2).

    Attributes
    ----------
    order : int
        Highest predictor order, ``q_max``.
    variable_order : bool
        Whether the order is chosen step by step.
    safety, max_growth, min_factor : float
        The method's safety factor, largest growth and smallest factor.
    rejection_growth : float
        Largest factor a rejection's own defect may propose (:data:`REJECTION_GROWTH`).
    hysteresis : float
        :data:`ORDER_HYSTERESIS`.
    restart_after : int
        :data:`RESTART_AFTER`.
    collar_stability : float
        :data:`COLLAR_STABILITY`.
    real_intervals : tuple of float
        :data:`PEC_REAL_INTERVAL`, by order.
    collar_rate : float
        The in-stage collar's largest rate over two, 1/m; zero without one, outside the
        exponential formulation, or under the split scheme.
    carrier : bool
        Whether a carrier frame caps each proposal (:data:`CarrierFrame`).
    carrier_cap : float
        :data:`CARRIER_PHASE_CAP`, rad.
    max_calls : int or None
        The call budget.
    equal_approach, interval_cap : bool
        The method's landing options.
    dz_min, dz_max, stretch, z_end : float
        The stepper settings: the floor, the cap, the landing stretch and the run's end,
        m (the stretch is a ratio).
    save_planes : tuple of float
        The planes to land on, m; empty when the planes are interpolated.
    hard_kinks : tuple of float
        The operator's hard kinks, m.
    breakpoints : tuple of float
        The edges whose intervals cap a step, m: the operator's breakpoints with the
        interval cap, otherwise only its first and last.
    landing_ulps : int
        :data:`~solprop.stepper.schedule.LANDING_ULPS`.
    interpolate : bool
        Whether the save planes are observed densely.
    quiet_interval : float or None
        The boundary monitor's interval, m, which caps a step while the collar rests;
        ``None`` without a monitor. The Python march reads the same number from the
        monitor through the driver (``StepDriver._quiet_capped``).

    Raises
    ------
    StepperError
        If an order, factor, interval, ulp count or position set is out of range.
    """

    order: int
    variable_order: bool
    safety: float
    max_growth: float
    min_factor: float
    rejection_growth: float
    hysteresis: float
    restart_after: int
    collar_stability: float
    real_intervals: tuple[float, ...]
    collar_rate: float
    carrier: bool
    carrier_cap: float
    max_calls: int | None
    equal_approach: bool
    interval_cap: bool
    dz_min: float
    dz_max: float
    stretch: float
    z_end: float
    save_planes: tuple[float, ...]
    hard_kinks: tuple[float, ...]
    breakpoints: tuple[float, ...]
    landing_ulps: int
    interpolate: bool
    quiet_interval: float | None

    def __post_init__(self) -> None:
        """Refuse a record the march could not execute, by field."""
        checks = (
            ("order", 1 <= self.order <= MAX_ADAMS_ORDER),
            ("real_intervals", len(self.real_intervals) >= self.order),
            ("safety", 0.0 < self.safety <= 1.0),
            ("max_growth", self.max_growth > 1.0),
            ("min_factor", 0.0 < self.min_factor < 1.0),
            ("rejection_growth", 0.0 < self.rejection_growth <= 1.0),
            ("hysteresis", self.hysteresis >= 1.0),
            ("restart_after", self.restart_after >= 1),
            ("collar_stability", self.collar_stability > 0.0),
            ("collar_rate", self.collar_rate >= 0.0),
            ("carrier_cap", self.carrier_cap > 0.0),
            ("max_calls", self.max_calls is None or self.max_calls >= 1),
            ("dz_min", 0.0 <= self.dz_min <= self.dz_max),
            ("stretch", self.stretch >= 1.0),
            ("breakpoints", len(self.breakpoints) >= 2),
            ("landing_ulps", self.landing_ulps >= 1),
            (
                "quiet_interval",
                self.quiet_interval is None or self.quiet_interval > 0.0,
            ),
        )
        for name, valid in checks:
            if not valid:
                msg = f"the Adams march's policy has {name} = {getattr(self, name)!r}"
                raise StepperError(msg)


def march_policy(  # noqa: PLR0913 - one record of a run's settings, by name
    method: Multistep,
    *,
    dz_min: float,
    dz_max: float,
    stretch: float,
    z_end: float,
    save_planes: Sequence[float],
    hard_kinks: Sequence[float],
    breakpoints: Sequence[float],
    interpolate: bool,
    collar_rate: float = 0.0,
    carrier: bool = False,
    quiet_interval: float | None = None,
) -> MarchPolicy:
    """Return the policy of one run of `method` (:class:`MarchPolicy`).

    Parameters
    ----------
    method : Multistep
        The method.
    dz_min, dz_max, stretch, z_end : float
        The run's stepper settings.
    save_planes, hard_kinks, breakpoints : sequence of float
        The run's save planes and the operator's kinks and breakpoints, m, in order.
    interpolate : bool
        Whether the planes are interpolated rather than landed on.
    collar_rate : float, optional
        The in-stage collar's largest rate over two, 1/m.
    carrier : bool, optional
        Whether a carrier frame is active.
    quiet_interval : float or None, optional
        The boundary monitor's interval, m.

    Returns
    -------
    MarchPolicy
        The record, with this module's constants filled in.

    Examples
    --------
    >>> from solprop.stepper.multistep import Multistep, march_policy
    >>> policy = march_policy(
    ...     Multistep(), dz_min=1e-9, dz_max=1e-2, stretch=1.1, z_end=1.0,
    ...     save_planes=[0.5, 1.0], hard_kinks=[], breakpoints=[0.0, 0.3, 1.0],
    ...     interpolate=False,
    ... )
    >>> policy.order, policy.hysteresis, policy.breakpoints
    (5, 1.2, (0.0, 1.0))
    """
    edges = tuple(float(edge) for edge in breakpoints)
    return MarchPolicy(
        order=method.order,
        variable_order=method.variable_order,
        safety=method.safety,
        max_growth=method.max_growth,
        min_factor=method.min_factor,
        rejection_growth=REJECTION_GROWTH,
        hysteresis=ORDER_HYSTERESIS,
        restart_after=RESTART_AFTER,
        collar_stability=COLLAR_STABILITY,
        real_intervals=PEC_REAL_INTERVAL,
        collar_rate=float(collar_rate),
        carrier=carrier,
        carrier_cap=CARRIER_PHASE_CAP,
        max_calls=method.max_calls,
        equal_approach=method.equal_approach,
        interval_cap=method.interval_cap,
        dz_min=float(dz_min),
        dz_max=float(dz_max),
        stretch=float(stretch),
        z_end=float(z_end),
        save_planes=() if interpolate else tuple(float(z) for z in save_planes),
        hard_kinks=tuple(float(z) for z in hard_kinks),
        breakpoints=edges if method.interval_cap else (edges[0], edges[-1]),
        landing_ulps=LANDING_ULPS,
        interpolate=interpolate,
        quiet_interval=None if quiet_interval is None else float(quiet_interval),
    )


def _basis_rows(
    points: np.ndarray, upper: float, integrate: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Lagrange basis on `points`: its coefficients, or its integrals.

    Scalar arithmetic on at most seven nodes, called several times a step: each basis
    numerator ``prod_{m != j} (x - theta_m)`` is built from its own roots, ascending,
    as ``numpy.polynomial.polyfromroots`` builds it -- deflating the whole node
    polynomial instead loses digits to the far nodes -- then either divided by its
    denominator (the monomial coefficients, ``integrate`` false) or integrated from 0
    to `upper` by Horner's rule and divided (the weights). The NumPy reference backend
    runs it as written, ``numba_cpu`` compiled by :func:`_basis` and the native backend
    its C++ transcription (``solprop._helpers``), each with these operations in this
    order, so the values are the same bit for bit.

    Parameters
    ----------
    points : numpy.ndarray
        The distinct nodes, float64.
    upper : float
        The integral's upper limit.
    integrate : bool
        Whether to return the integrals rather than the coefficients.

    Returns
    -------
    tuple
        The coefficients, shape ``(n, n)`` (zero when integrating), and the integrals,
        shape ``(n,)`` (unset when not).
    """
    n = points.shape[0]
    rows = np.zeros((n, n))
    weights = np.empty(n)
    coefficients = np.empty(n)
    product = np.empty(n)
    for index in range(n):
        node = points[index]
        length = 1
        coefficients[0] = 1.0
        denominator = 1.0
        for other in range(n):
            if other == index:
                continue
            root = points[other]
            product[0] = 0.0
            for power in range(length):
                product[power + 1] = coefficients[power]
            for power in range(length):
                product[power] -= root * coefficients[power]
            length += 1
            for power in range(length):
                coefficients[power] = product[power]
            denominator *= node - root
        if integrate:
            integral = 0.0
            for power in range(length - 1, -1, -1):
                integral = (integral + coefficients[power] / (power + 1)) * upper
            weights[index] = integral / denominator
        else:
            for power in range(length):
                rows[index, power] = coefficients[power] / denominator
    return rows, weights


@functools.cache
def _basis() -> Callable[[np.ndarray, float, bool], tuple[np.ndarray, np.ndarray]]:
    """Return :func:`_basis_rows` compiled.

    Six to nine times faster than the Python loops, which were most of an exponential
    trial's host work after the z tables. numba is imported here, at the first use, so
    that importing solprop does not load it.

    Returns
    -------
    collections.abc.Callable
        The compiled function.
    """
    from solprop._numba_cache import njit  # noqa: PLC0415 - numba on first use

    # The function itself, not a runtime type checker's wrapper around it (the test
    # suite's import hook adds one), which numba can neither compile nor cache.
    # Strict (SEP 25 section 4.2): the divided differences and integrated weights
    # are the stable increments and defects, small differences of large products.
    return njit(cache=True, strict=True)(inspect.unwrap(_basis_rows))


def integrated_weights(
    nodes: Sequence[float] | Float[np.ndarray, " n_nodes"],
    upper: float = 1.0,
    *,
    check: bool = True,
    helpers: Helpers = NUMBA,
) -> Float[np.ndarray, " n_nodes"]:
    """Return the weights of the integral from 0 to `upper` of an interpolant.

    ``sum_j w_j f(theta_j)`` equals ``integral_0^upper p(theta) dtheta`` for the
    polynomial ``p`` of degree ``len(nodes) - 1`` interpolating ``f`` at the nodes:
    each weight is the integral of the ``j``-th Lagrange basis polynomial. With the
    nodes in units of the step, ``h * sum_j w_j g_j`` is the Adams increment.

    Parameters
    ----------
    nodes : sequence of float
        Distinct normalised positions ``(z_j - z_n) / h``.
    upper : float, optional
        Upper limit of the integral in the same units; 1 for a whole step, a fraction
        of it for dense output.
    check : bool, optional
        Whether to refuse empty, non-finite or repeated nodes. The Adams march, whose
        nodes are accepted positions and so distinct, skips it on every step.
    helpers : solprop._helpers.Helpers, optional
        The evaluating backend's implementation of the basis (SEP 29 section 6.6); the
        shared Numba one by default, for callers outside a backend.

    Returns
    -------
    numpy.ndarray
        One weight per node.

    Raises
    ------
    StepperError
        If the nodes are empty or not distinct.

    Examples
    --------
    Constant-step Adams-Bashforth 4 and Adams-Moulton 4, times 24:

    >>> from solprop.stepper.multistep import integrated_weights
    >>> (24 * integrated_weights([0, -1, -2, -3])).round(12).tolist()
    [55.0, -59.0, 37.0, -9.0]
    >>> (24 * integrated_weights([1, 0, -1, -2])).round(12).tolist()
    [9.0, 19.0, -5.0, 1.0]
    """
    theta = np.asarray(nodes, dtype=np.float64).reshape(-1)
    if check:
        _check_nodes(theta)
    _, weights = helpers.basis(np.ascontiguousarray(theta), float(upper), True)
    return weights


def lagrange_coefficients(
    nodes: Sequence[float] | Float[np.ndarray, " n_nodes"],
    *,
    check: bool = True,
    helpers: Helpers = NUMBA,
) -> Float[np.ndarray, "n_nodes n_nodes"]:
    """Return the monomial coefficients of the Lagrange basis on distinct nodes.

    Row ``j`` holds ``c_jk`` with ``ell_j(theta) = sum_k c_jk theta**k``, ascending in
    ``k``, so that ``ell_j(theta_i)`` is one at ``i = j`` and zero elsewhere. The
    exponential Adams weights integrate each power against the linear flow
    (``w_j(x) = sum_k c_jk k! phi_{k+1}(x)``, SEP 16 section 4.2); at ``x = 0`` they
    reduce to :func:`integrated_weights`.

    Parameters
    ----------
    nodes : sequence of float
        Distinct normalised positions ``(z_j - z_n) / h``.
    check : bool, optional
        Whether to refuse empty, non-finite or repeated nodes.
    helpers : solprop._helpers.Helpers, optional
        The evaluating backend's implementation of the basis; the shared Numba one by
        default.

    Returns
    -------
    numpy.ndarray
        The coefficients, shape ``(n_nodes, n_nodes)``.

    Raises
    ------
    StepperError
        If the nodes are empty or not distinct.

    Examples
    --------
    The linear basis on the nodes 1 and 0: ``theta`` and ``1 - theta``.

    >>> from solprop.stepper.multistep import lagrange_coefficients
    >>> lagrange_coefficients([1.0, 0.0]).tolist()
    [[0.0, 1.0], [1.0, -1.0]]
    """
    theta = np.asarray(nodes, dtype=np.float64).reshape(-1)
    if check:
        _check_nodes(theta)
    rows, _ = helpers.basis(np.ascontiguousarray(theta), 1.0, False)
    return rows


def defect_integral(
    nodes: Sequence[float] | Float[np.ndarray, " n_nodes"], upper: float = 1.0
) -> float:
    """Return the integral from 0 to `upper` of the node polynomial.

    ``integral_0^upper prod_j (theta - theta_j) dtheta``. Divided by ``q!`` for ``q``
    nodes it is the leading defect constant ``C`` of the formula interpolating on those
    nodes, ``exact - formula = C h^{q+1} g^{(q)}`` (SEP 13 section 4.3): the
    interpolation remainder, integrated.

    Parameters
    ----------
    nodes : sequence of float
        Normalised positions, as for :func:`integrated_weights`.
    upper : float, optional
        Upper limit of the integral.

    Returns
    -------
    float
        The integral, dimensionless.

    Examples
    --------
    The constant-step AB4 and AM4 constants, 251/720 and -19/720, times 4!:

    >>> from solprop.stepper.multistep import defect_integral
    >>> round(defect_integral([0, -1, -2, -3]) * 30, 9)
    251.0
    >>> round(defect_integral([1, 0, -1, -2]) * 30, 9)
    -19.0
    """
    theta = np.asarray(nodes, dtype=np.float64).reshape(-1)
    antiderivative = polynomial.polyint(polynomial.polyfromroots(theta))
    return float(
        polynomial.polyval(upper, antiderivative)
        - polynomial.polyval(0.0, antiderivative)
    )


def milne_factor(
    predictor_nodes: Sequence[float] | Float[np.ndarray, " n_nodes"],
    corrector_nodes: Sequence[float] | Float[np.ndarray, " n_nodes"],
) -> float:
    """Return the factor turning a predictor-corrector difference into the defect.

    With the predictor's defect ``C_B h^{q+1} g^{(q)}`` and the corrector's
    ``C_M h^{q+1} g^{(q)}`` on the same number of nodes, ``v_C - v_P`` is
    ``(C_B - C_M) h^{q+1} g^{(q)}``, so the corrector's local defect is
    ``C_M / (C_B - C_M)`` times the difference (Milne's device, SEP 13 section 4.3).
    The common ``1/q!`` cancels.

    Parameters
    ----------
    predictor_nodes, corrector_nodes : sequence of float
        The two formulas' normalised nodes, equal in number; the corrector's include 1.

    Returns
    -------
    float
        ``C_M / (C_B - C_M)``.

    Raises
    ------
    StepperError
        If the node counts differ, or the two constants coincide so the difference
        carries no information.

    Examples
    --------
    >>> from solprop.stepper.multistep import milne_factor
    >>> round(milne_factor([0, -1, -2, -3], [1, 0, -1, -2]) * 270, 9)
    -19.0
    """
    if len(predictor_nodes) != len(corrector_nodes):
        msg = (
            f"a Milne estimate needs equal node counts, got {len(predictor_nodes)} and "
            f"{len(corrector_nodes)}"
        )
        raise StepperError(msg)
    predictor = defect_integral(predictor_nodes)
    corrector = defect_integral(corrector_nodes)
    spread = predictor - corrector
    # On equal nodes the spread is a few tenths; an irregular stencil that brought the
    # two constants together would divide round-off into the estimate.
    if abs(spread) <= 1e-12 * max(abs(predictor), abs(corrector), 1.0):
        msg = "the predictor and corrector defect constants coincide"
        raise StepperError(msg)
    return corrector / spread


def _check_nodes(theta: Float[np.ndarray, " n_nodes"]) -> None:
    """Refuse empty, non-finite or repeated nodes.

    Parameters
    ----------
    theta : numpy.ndarray
        The nodes.

    Raises
    ------
    StepperError
        If the nodes are empty, not finite or not distinct.
    """
    if theta.size == 0 or not np.all(np.isfinite(theta)):
        msg = f"nodes must be non-empty and finite, got {theta}"
        raise StepperError(msg)
    if np.unique(theta).size != theta.size:
        msg = f"nodes must be distinct, got {theta}"
        raise StepperError(msg)
