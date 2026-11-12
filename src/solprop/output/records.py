# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The record types of a saved run: its grid, modes, noise, stepper and saved planes.

Each is a frozen dataclass that a handler writes and reads back. :class:`SavedPlane`
carries one saved position with its field, its statistics and the step history since the
previous plane; :class:`RunOutcome` and :class:`TerminalCheckpoint` how a run ended.

**Conventions on the way out.** A ``SavedPlane`` holds its field in the internal FFT
convention, as a backend produces it; a handler converts it with
:func:`solprop.fourier.spectrum_to_optics` on the way to storage, so that what is stored
is the window-centred optics spectrum a spectrometer would agree with, marked
``spectral_convention = "optics"`` (:data:`SPECTRAL_CONVENTION`;
``docs/dev/conventions.md`` §2). The noise realisation and ``noise_phase`` are internal
objects and stay in the FFT convention, marked so by :data:`INTERNAL_CONVENTION`.

The manual page is ``docs/guide/results/output.md``.
"""

import copy
import dataclasses
import functools
import hashlib
import json
import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop import fourier
from solprop._display import Displayed, engineering, line, summarise
from solprop.exceptions import OutputError
from solprop.grid import TimeGrid
from solprop.modes.material import JsonValue
from solprop.modes.protocol import Mode
from solprop.noise import NoiseRealisation
from solprop.output.serialisation import (
    _decode,
    _optional_text,
    _rebuild,
    _stored_entry,
    serialise,
)
from solprop.stepper import FixedSchedule, PIController, Tableau

__all__ = [
    "FIELD_LAYOUT",
    "HISTORY_FIELDS",
    "INTERNAL_CONVENTION",
    "SPECTRAL_CONVENTION",
    "STEP_STATS_INDEX",
    "Convention",
    "DefaultsResolution",
    "GridRecord",
    "MarchSnapshot",
    "ModeRecord",
    "NoiseRecord",
    "NoiseResolution",
    "RecipeFile",
    "RecipeRecord",
    "RunOutcome",
    "SavedPlane",
    "SimulationKind",
    "StepHistory",
    "StepLimits",
    "StepStats",
    "StepperRecord",
    "StoredField",
    "TerminalCheckpoint",
]

SPECTRAL_CONVENTION = "optics"
"""Sign convention of every spectrum a handler stores."""

INTERNAL_CONVENTION = "fft"
"""Sign convention of the arrays that stay internal: the noise field and its phase."""

FIELD_LAYOUT = "(n_z, n_modes, n_pol, n_omega)"
"""Axis order of the stored field, as ``docs/dev/conventions.md`` §6 fixes it."""

SimulationKind = Literal["mode_averaged", "multimode", "radial", "free", "slab"]
"""Which transform the run used."""

Convention = Literal["optics", "fft"]
"""Which sign convention an accessor should hand back."""


@dataclass(frozen=True, kw_only=True)
class GridRecord:
    """The scalars that identify the grid a run was computed on.

    Only scalars: the four axis arrays a :class:`solprop.grid.TimeGrid` holds are
    reconstructible from these, and keeping them out of the record keeps the metadata
    JSON, and the configuration hash taken over it, small.

    Attributes
    ----------
    kind : {'field', 'envelope'}
        Spectral layout, which fixes what the stored field means and which conversion
        the storage boundary applies.
    n_t : int
        Coarse time samples.
    n_to : int
        Fine time samples.
    n_omega : int
        Coarse spectral bins, the length of the stored field's last axis.
    n_omega_fine : int
        Fine spectral bins.
    dt : float
        Coarse sample spacing, s.
    dt_fine : float
        Fine sample spacing, s.
    ffac : float
        Oversampling factor, dimensionless.
    omega0 : float
        Reference angular frequency, rad/s.
    d_omega : float
        Spectral bin spacing, rad/s.
    length : float
        Propagation length the grid was built for, m.
    reference_wavelength : float
        Wavelength of ``omega0``, m.
    omega_min : float
        Lower in-band angular frequency, rad/s.
    omega_max : float
        Upper in-band angular frequency, rad/s.
    omega_max_window : float
        Upper edge of the active band, rad/s.
    t_window_min : float
        Lower inner edge of the temporal collar, s: the window is flat between this
        and ``t_window_max`` and tapers to the ends of the axis outside them.
    t_window_max : float
        Upper inner edge of the temporal collar, s.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.records import GridRecord
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> record = GridRecord.from_grid(grid)
    >>> record.kind, record.n_omega == grid.n_omega
    ('envelope', True)
    """

    kind: fourier.GridKind
    n_t: int
    n_to: int
    n_omega: int
    n_omega_fine: int
    dt: float
    dt_fine: float
    ffac: float
    omega0: float
    d_omega: float
    length: float
    reference_wavelength: float
    omega_min: float
    omega_max: float
    omega_max_window: float
    t_window_min: float
    t_window_max: float

    @classmethod
    def from_grid(cls, grid: TimeGrid) -> "GridRecord":
        """Return the record of a grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            The grid the run used.

        Returns
        -------
        GridRecord
            Its scalar description.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> from solprop.output.records import GridRecord
        >>> grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
        >>> GridRecord.from_grid(grid).kind
        'field'
        """
        return cls(
            kind=grid.kind,
            n_t=grid.n_t,
            n_to=grid.n_to,
            n_omega=grid.n_omega,
            n_omega_fine=grid.n_omega_fine,
            dt=grid.dt,
            dt_fine=grid.dt_fine,
            ffac=grid.ffac,
            omega0=grid.omega0,
            d_omega=grid.d_omega,
            length=grid.length,
            reference_wavelength=grid.reference_wavelength,
            omega_min=grid.omega_min,
            omega_max=grid.omega_max,
            omega_max_window=grid.omega_max_window,
            t_window_min=grid.t_collar.one_low,
            t_window_max=grid.t_collar.one_high,
        )


@dataclass(frozen=True, kw_only=True)
class ModeRecord:
    """One propagated mode, by class, effective area, channel label and configuration.

    Attributes
    ----------
    kind : str
        Qualified class name of the mode, ``'module.ClassName'``.
    effective_area : float
        ``A_eff`` in m^2 at the reference position, the one number processing needs
        most often and the one a plot label carries.
    configuration : dict
        The mode object as :func:`~solprop.output.serialisation.serialise` records it.
    label : collections.abc.Mapping or None
        The physical channel this row of the mode axis is, as
        :meth:`solprop.modes.protocol.ChannelLabel.as_mapping` writes it: the
        family, the azimuthal and radial orders, the parity, the handedness and
        the display label. :meth:`from_mode` always records one; ``None`` on a
        record built without it, which names its channel only by class name and
        position, and which :func:`solprop.processing.modal.channel_labels`
        refuses. Part of :meth:`~solprop.output.metadata.RunMetadata.hashed`.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.output.records import ModeRecord
    >>> record = ModeRecord.from_mode(MarcatiliMode(125e-6, "He", 1.0))
    >>> record.kind
    'solprop.modes.capillary.MarcatiliMode'
    >>> None if record.label is None else record.label["display_label"]
    'HE1,1:even'
    """

    kind: str
    effective_area: float
    configuration: Mapping[str, JsonValue]
    label: Mapping[str, JsonValue] | None = None

    @classmethod
    def from_mode(cls, mode: Mode, z: float = 0.0) -> "ModeRecord":
        """Return the record of a mode.

        Parameters
        ----------
        mode : solprop.modes.protocol.Mode
            The mode.
        z : float, optional
            Position along the waveguide the effective area is read at, m.

        Returns
        -------
        ModeRecord
            Its description.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.output.records import ModeRecord
        >>> record = ModeRecord.from_mode(MarcatiliMode(75e-6, "He", 1.0))
        >>> f"{record.effective_area:.6e}"
        '8.421575e-09'
        """
        kind = type(mode)
        # serialise returns a dictionary for everything that is not a primitive, a
        # sequence or a callable, and a mode is none of those, so the cast is total.
        configuration = cast("dict[str, JsonValue]", serialise(mode))
        if getattr(mode, "mixture", None) is not None:
            # These legacy constructor slots are placeholders when a mixture owns
            # every species model and density path; they would imply a second fill.
            for name in ("gas", "pressure", "temperature", "source", "susceptibility"):
                configuration.pop(name, None)
        return cls(
            kind=f"{kind.__module__}.{kind.__qualname__}",
            effective_area=mode.aeff(z),
            configuration=configuration,
            label=dict(mode.channel_label.as_mapping()),
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class NoiseRecord:
    """The shot-noise realisation a run started from, and the seed that drew it.

    Equality is by identity (``eq=False``) because ``spectrum`` is an array; compare
    the seeds, which is what reproducibility means.

    The spectrum is stored in the **internal** FFT convention, unlike the field: it is
    not a user-facing spectrum but the object the stepper adds at every stage, and
    conjugating it would make it disagree with the file it has to be read back into.
    :attr:`spectral_convention` says so explicitly.

    Attributes
    ----------
    model : {'input', 'modified'}
        Which noise model the realisation belongs to.
    seed : int or None
        Entropy of the generator, or ``None`` when it could not be recovered.
    spectrum : numpy.ndarray
        The realisation, shape ``(n_modes, n_pol, n_omega)``, complex, in the square
        root of watts.
    spectral_convention : str
        Always ``'fft'``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import input_noise
    >>> from solprop.output.records import NoiseRecord
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> realisation = input_noise(grid, np.random.default_rng(3))
    >>> record = NoiseRecord.from_realisation(realisation)
    >>> record.seed, record.spectral_convention
    (3, 'fft')
    """

    model: str
    seed: int | None
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"]
    spectral_convention: str = INTERNAL_CONVENTION

    def __post_init__(self) -> None:
        """Adopt a read-only copy of the spectrum, so a record cannot drift."""
        frozen = np.array(self.spectrum, dtype=np.complex128, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "spectrum", frozen)

    @classmethod
    def from_realisation(cls, realisation: NoiseRealisation) -> "NoiseRecord":
        """Return the record of a drawn realisation.

        Parameters
        ----------
        realisation : solprop.noise.NoiseRealisation
            The realisation the run used.

        Returns
        -------
        NoiseRecord
            Its description.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.noise import modified_noise
        >>> from solprop.output.records import NoiseRecord
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
        >>> NoiseRecord.from_realisation(modified_noise(grid, np.random.default_rng(1)))
        ... # doctest: +ELLIPSIS
        NoiseRecord(model='modified', seed=1, ...)
        """
        return cls(
            model=realisation.model,
            seed=realisation.seed,
            spectrum=realisation.spectrum,
        )


@dataclass(frozen=True)
class NoiseResolution:
    """How a run's noise model was chosen: what was asked for, what it carries, and why.

    Attributes
    ----------
    requested : {'default', 'input', 'modified'} or None
        The model the caller asked for: ``"default"`` when the keyword was omitted,
        ``None`` when noise was switched off explicitly.
    resolved : {'input', 'modified'} or None
        The model the run carries, or ``None`` for a noiseless run.
    reason : str or None
        Why a default resolved to no noise, for example that the hybrid planar grid
        declares no photon measure; ``None`` otherwise.

    Examples
    --------
    >>> from solprop.output.records import NoiseResolution
    >>> NoiseResolution("default", "input")
    NoiseResolution(requested='default', resolved='input', reason=None)
    """

    requested: str | None
    resolved: str | None
    reason: str | None = None


@dataclass(frozen=True)
class DefaultsResolution:
    """Which of a run's numerics took its route's default, and which route that was.

    The companion of :class:`NoiseResolution` for the stepping (``docs/guide/numerics/
    choosing-a-stepper.md``): the integrator, the step controller and the temporal
    collar's scheme are each either the caller's or the route's default from
    :data:`solprop.driver.defaults.ROUTE_DEFAULTS`. Outside the hash: what the run
    computes is in the stepper record already, and how it was chosen is not physics.

    Attributes
    ----------
    route : str or None
        The row of the defaults table the constructor read, or ``None`` for settings
        built by hand.
    defaulted : sequence of str
        Which of ``"integrator"``, ``"rtol"`` and ``"absorber_scheme"`` took the
        route's default, in that order, adopted as a tuple.

    Examples
    --------
    >>> from solprop.output.records import DefaultsResolution
    >>> DefaultsResolution("capillary", ["integrator", "rtol"])
    DefaultsResolution(route='capillary', defaulted=('integrator', 'rtol'))
    """

    route: str | None
    defaulted: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Adopt the names as a tuple, so a record read back compares equal."""
        object.__setattr__(self, "defaulted", tuple(self.defaulted))


@dataclass(frozen=True)
class StepLimits:
    """The caps a run put on the controller's step proposal.

    Grouped into one record because they travel together and because a constructor
    that took them one by one would need more arguments than the project allows.

    Attributes
    ----------
    dz_min : float
        Smallest step the controller was allowed to propose, m. Zero when the driver
        left it to :func:`solprop.stepper.controller.default_dz_min`.
    dz_max : float
        Largest step, m; infinite when uncapped.
    dz_init : float
        First step, m; zero when the driver chose it.
    stretch : float
        Landing stretch factor, dimensionless; 1 switches landing stretch off.
    interval_cap : bool
        Whether a step was capped at its operator-table interval's width.
    auto_init : bool
        Whether the start-up rule took the step after the first from the first step's
        estimate.
    interpolate_planes : bool
        Whether the save planes but the end and the hard kinks were interpolated
        inside accepted steps; ``False`` when every plane was landed on.
    step_planes : int
        A positive ``k`` when the end of every ``k``-th accepted step was saved too;
        ``0`` for the requested planes only.

    Examples
    --------
    >>> from solprop.output.records import StepLimits
    >>> StepLimits(dz_max=0.01).dz_max
    0.01
    """

    dz_min: float = 0.0
    dz_max: float = math.inf
    dz_init: float = 0.0
    stretch: float = 1.1
    interval_cap: bool = True
    auto_init: bool = False
    interpolate_planes: bool = False
    step_planes: int = 0


@dataclass(frozen=True, eq=False, kw_only=True)
class StepperRecord:
    """The solver settings a run used, including the schedule it actually took.

    Equality is by identity (``eq=False``) because ``schedule`` is an array. Schedules
    are first class and re-loadable (see ``docs/dev/decisions.md``): a replayed run is
    bit-identical, and every ensemble member then carries the same numerical error as a
    common systematic.

    Build one from the objects a run held with :meth:`from_settings`, and from a stored
    document with :meth:`from_record`.

    Attributes
    ----------
    tableau : str
        Name of the Runge-Kutta scheme.
    method : str or None
        The integrator that stepped the run instead of the tableau, as its ``repr``:
        a low-call Adams method (SEP 13) or the low-storage march (SEP 24); ``None``
        for the Runge-Kutta march.
    controller : dict
        The step controller as :func:`~solprop.output.serialisation.serialise` records
        it: tolerances, safety and gain factors, lag and the norm weights.
    limits : StepLimits
        The caps on the controller's proposal.
    schedule : numpy.ndarray or None
        Step lengths of shape ``(n_steps,)``, m, when a fixed schedule was replayed or
        recorded; ``None`` for a purely adaptive run.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import StepLimits, StepperRecord
    >>> from solprop.stepper import DOPRI5, FixedSchedule, PIController
    >>> record = StepperRecord.from_settings(
    ...     DOPRI5,
    ...     PIController(rtol=1e-8, atol=0.0),
    ...     schedule=FixedSchedule(np.array([0.5, 0.5])),
    ...     limits=StepLimits(dz_max=0.1),
    ... )
    >>> record.tableau, record.limits.dz_max
    ('DOPRI5', 0.1)
    """

    tableau: str
    controller: Mapping[str, JsonValue]
    limits: StepLimits = dataclasses.field(default_factory=StepLimits)
    schedule: Float[np.ndarray, " n_steps"] | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        """Adopt a read-only copy of the schedule, so a record cannot drift."""
        if self.schedule is not None:
            frozen = np.array(self.schedule, dtype=np.float64, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, "schedule", frozen)

    @classmethod
    def from_settings(
        cls,
        tableau: Tableau,
        controller: PIController,
        *,
        schedule: FixedSchedule | None = None,
        limits: StepLimits | None = None,
        method: str | None = None,
    ) -> "StepperRecord":
        """Return the record of the objects a run stepped with.

        Parameters
        ----------
        tableau : solprop.stepper.tableau.Tableau
            The scheme.
        controller : solprop.stepper.controller.PIController
            The step controller, which carries the tolerances and the norm choice.
        schedule : solprop.stepper.schedule.FixedSchedule, optional
            The fixed schedule replayed, if any.
        limits : StepLimits, optional
            The caps on the controller's proposal; the defaults when omitted.
        method : str or None, optional
            The Adams method that stepped the run instead of the tableau, as its
            ``repr``; ``None`` for the Runge-Kutta march.

        Returns
        -------
        StepperRecord
            The record.

        Examples
        --------
        >>> from solprop.output.records import StepperRecord
        >>> from solprop.stepper import TSIT5, PIController
        >>> record = StepperRecord.from_settings(
        ...     TSIT5, PIController(rtol=1e-6, atol=0.0)
        ... )
        >>> record.tableau, record.schedule is None
        ('Tsit5', True)
        """
        # A PIController is a dataclass, so serialise always returns a dictionary.
        record = cast("dict[str, JsonValue]", serialise(controller))
        return cls(
            tableau=tableau.name,
            controller=record,
            limits=StepLimits() if limits is None else limits,
            schedule=None if schedule is None else schedule.steps,
            method=method,
        )

    @classmethod
    def from_record(cls, document: Mapping[str, JsonValue]) -> "StepperRecord":
        """Return the record a stored document describes.

        Parameters
        ----------
        document : collections.abc.Mapping
            The form :func:`~solprop.output.serialisation.serialise` produced from a
            record.

        Returns
        -------
        StepperRecord
            The record.

        Examples
        --------
        >>> from solprop.output.records import StepperRecord
        >>> from solprop.output.serialisation import serialise
        >>> from solprop.stepper import TSIT5, PIController
        >>> one = StepperRecord.from_settings(TSIT5, PIController(rtol=1e-6, atol=0.0))
        >>> StepperRecord.from_record(serialise(one)).tableau
        'Tsit5'
        """
        # Another build of this schema may lack an optional field.
        stored = cast(
            "Callable[..., Any]",
            functools.partial(_stored_entry, document, path=("stepper",)),
        )

        schedule = _decode(stored("schedule"))
        limits = stored("limits")
        return cls(
            tableau=str(stored("tableau", required=True)),
            controller=cast(
                "dict[str, JsonValue]", stored("controller", required=True)
            ),
            limits=StepLimits()
            if limits is None
            else _rebuild(
                StepLimits, cast("dict[str, JsonValue]", limits), ("stepper", "limits")
            ),
            schedule=None if schedule is None else np.asarray(schedule, dtype=float),
            method=_optional_text(stored("method")),
        )


HISTORY_FIELDS: Mapping[str, type] = {
    "z": np.float64,
    "h": np.float64,
    "norm": np.float64,
    "accepted": np.bool_,
    "wall_time": np.float64,
}
"""Dtype of each :class:`StepHistory` field, in the order they are laid out."""


@dataclass(frozen=True, eq=False, kw_only=True)
class StepHistory(Displayed):
    """What the stepper did between two saved planes.

    Recorded whether the run was adaptive or replayed a schedule, because the embedded
    error estimate costs nothing extra and a fixed run still needs it: a largest scaled
    norm above one means a step was too coarse for the tolerance the norm is scaled by
    (``docs/dev/stepper.md`` §8).

    Equality is by identity (``eq=False``) because every field is an array.

    Attributes
    ----------
    z : numpy.ndarray
        Position at the start of each attempt, shape ``(n_steps,)``, m.
    h : numpy.ndarray
        Length of each attempt, shape ``(n_steps,)``, m.
    norm : numpy.ndarray
        Scaled error norm each attempt scored as executed, shape ``(n_steps,)``,
        dimensionless; the controller's rescaled equivalent of a landing step is its
        own and is not recorded. The low-storage march estimates none and records 0.0,
        "not estimated", never a step that met its tolerance exactly.
    accepted : numpy.ndarray
        Whether each attempt was accepted, shape ``(n_steps,)``, boolean.
    wall_time : numpy.ndarray
        Wall-clock time each attempt took, shape ``(n_steps,)``, s.

    Raises
    ------
    OutputError
        If the five arrays do not have equal shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import StepHistory
    >>> history = StepHistory(
    ...     z=np.array([0.0, 0.5]),
    ...     h=np.array([0.5, 0.5]),
    ...     norm=np.array([0.3, 0.4]),
    ...     accepted=np.array([True, True]),
    ...     wall_time=np.array([1e-3, 1e-3]),
    ... )
    >>> history.n_steps, history.n_rejected
    (2, 0)
    """

    # Independent axis names on purpose: their agreement is checked below with an
    # OutputError naming the culprit, and that check must also hold in production,
    # where the test session's runtime type checker is absent. One shared name would
    # let the checker forestall it.
    z: Float[np.ndarray, " n_history_z"]
    h: Float[np.ndarray, " n_history_h"]
    norm: Float[np.ndarray, " n_history_norm"]
    accepted: Bool[np.ndarray, " n_history_accepted"]
    wall_time: Float[np.ndarray, " n_history_wall"]

    def __repr__(self) -> str:
        """Return a one-line description: attempts, rejections and wall time.

        Returns
        -------
        str
            For example ``StepHistory(1842 attempts, 12 rejected, 12.2 s)``.
        """
        attempts = int(np.asarray(self.z).size)
        rejected = attempts - int(np.count_nonzero(self.accepted))
        seconds = float(np.sum(self.wall_time)) if attempts else 0.0
        return line(
            "StepHistory",
            [
                ("", f"{attempts} attempts"),
                ("", f"{rejected} rejected"),
                ("", engineering(seconds, "s")),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        attempts = int(np.asarray(self.z).size)
        if not attempts:
            return "StepHistory: no step attempted", []
        accepted = np.asarray(self.accepted, dtype=bool)
        steps = np.asarray(self.h, dtype=float)[accepted]
        return "StepHistory", [
            (
                "attempts",
                f"{attempts}, of which {attempts - int(accepted.sum())} rejected",
            ),
            ("accepted steps", summarise(steps, "m").replace("values", "steps")),
            ("largest error norm", f"{float(np.max(self.norm)):.3g}"),
            ("wall time", engineering(float(np.sum(self.wall_time)), "s")),
        ]

    def __post_init__(self) -> None:
        """Check that the five arrays describe one sequence, and freeze them.

        Raises
        ------
        OutputError
            If the five shapes disagree. That they are one-dimensional is fixed by the
            annotations, as it is for :class:`solprop.stepper.tableau.Tableau`.
        """
        shapes = {}
        for name, dtype in HISTORY_FIELDS.items():
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            shapes[name] = values.shape
        if len(set(shapes.values())) > 1:
            msg = f"step history arrays must have equal shapes, got {shapes}"
            raise OutputError(msg)

    @classmethod
    def empty(cls) -> "StepHistory":
        """Return the history of a plane no step led to, such as the starting plane.

        Returns
        -------
        StepHistory
            Five empty arrays.

        Examples
        --------
        >>> from solprop.output.records import StepHistory
        >>> StepHistory.empty().n_steps
        0
        """
        return cls(
            z=np.empty(0),
            h=np.empty(0),
            norm=np.empty(0),
            accepted=np.empty(0, dtype=bool),
            wall_time=np.empty(0),
        )

    @property
    def n_steps(self) -> int:  # numpydoc ignore=RT01
        """Number of attempts recorded.

        Accepted and rejected attempts together.
        """
        return int(self.z.size)

    @property
    def n_rejected(self) -> int:  # numpydoc ignore=RT01
        """Number of rejected attempts."""
        return int(np.count_nonzero(~self.accepted))


STEP_STATS_INDEX: Mapping[str, type] = {"z": np.float64, "step": np.int64}
"""Dtype of each :class:`StepStats` index column, beside the statistics themselves."""


@dataclass(frozen=True, eq=False, kw_only=True)
class StepStats(Displayed):
    """The statistics a backend evaluated *between* two saved planes.

    A run whose :class:`solprop.stats.StatsSpec` names a ``periodic`` subset gets one
    row every ``period`` accepted steps, evaluated inside the backend's step loop and
    flushed here at each plane. Empty for a run that asked for none, which is
    the default.

    Stored the way the step history is -- flat columns concatenated over planes with a
    per-plane ``counts`` beside them -- so that a resume appends and a scan needs no NaN
    padding, which growing datasets would force on it.

    Equality is by identity (``eq=False``) because every field is an array.

    Attributes
    ----------
    z : numpy.ndarray
        Position of each row, m, float64, shape ``(n_rows,)``.
    step : numpy.ndarray
        Index of the accepted step each row was taken after, into this plane's
        :class:`StepHistory`, int64, shape ``(n_rows,)``. The history counts rejected
        attempts, so this is not ``row * period``.
    values : dict
        The statistics, keyed by catalogue name, each shaped ``(n_rows, *shape)`` with
        the statistic's own shape after the row axis, as the backend's rows are.

    Raises
    ------
    OutputError
        If the columns do not all have the same number of rows.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import StepStats
    >>> rows = StepStats(z=np.array([0.1, 0.2]), step=np.array([0, 2]),
    ...                  values={"energy": np.ones((2, 1, 1))})
    >>> rows.n_rows, rows.names
    (2, ('energy',))
    """

    z: Float[np.ndarray, " n_rows_z"]
    step: Int[np.ndarray, " n_rows_step"]
    values: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a one-line description: how many rows of which statistics.

        Returns
        -------
        str
            For example ``StepStats(1842 rows of energy, peak_power)``.
        """
        rows = int(np.asarray(self.z).size)
        return f"StepStats({rows} rows of {', '.join(self.values) or 'nothing'})"

    def __post_init__(self) -> None:
        """Freeze the columns and check that they describe the same rows.

        Raises
        ------
        OutputError
            Naming the columns whose row counts disagree.
        """
        shapes = {}
        for name, dtype in STEP_STATS_INDEX.items():
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            shapes[name] = values.shape
        adopted = {}
        for name, value in self.values.items():
            rows = np.array(value, copy=True)
            rows.setflags(write=False)
            adopted[name] = rows
            shapes[name] = rows.shape[:1]
        object.__setattr__(self, "values", adopted)
        if len(set(shapes.values())) > 1:
            msg = f"step statistics columns must have equal row counts, got {shapes}"
            raise OutputError(msg)

    @classmethod
    def empty(cls) -> "StepStats":
        """Return the channel a run without a periodic subset carries.

        Returns
        -------
        StepStats
            No rows and no statistics.
        """
        return cls(z=np.empty(0), step=np.empty(0, dtype=np.int64), values={})

    @property
    def n_rows(self) -> int:  # numpydoc ignore=RT01
        """How many rows belong to this plane."""
        return int(self.z.size)

    @property
    def names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics the channel carries, sorted."""
        return tuple(sorted(self.values))


_ADOPTED: weakref.WeakValueDictionary[int, np.ndarray] = weakref.WeakValueDictionary()
"""The arrays :func:`_adopted` marked, by id, for as long as each lives."""


def _adopted(values: np.ndarray) -> np.ndarray:
    """Mark an array a :class:`SavedPlane` takes as it is, instead of copying it.

    The driver's route (:func:`solprop.driver.run.run_problem`) for what an observation
    holds. Only for an array that nothing writes to again, as
    :meth:`solprop.backends.protocol.HostStepping.observe` promises, and never for a
    batch member, whose view would pin the whole batch.

    - Returns: the same array, read-only.
    """
    values.setflags(write=False)
    _ADOPTED[id(values)] = values
    return values


def _plane_array(
    values: np.ndarray, dtype: type[np.complex128] | type[np.float64]
) -> np.ndarray:
    """Return an array as a :class:`SavedPlane` holds it: read-only, nobody else's.

    - Returns: `values` itself when :func:`_adopted` marked it, a frozen copy otherwise.
    """
    if _ADOPTED.get(id(values)) is values:
        return values
    held = np.array(values, dtype=dtype, copy=True)
    held.setflags(write=False)
    return held


@dataclass(frozen=True, eq=False, kw_only=True)
class SavedPlane(Displayed):
    """One saved position: the field there, its statistics and how it was reached.

    The field is in the **internal** FFT convention, exactly as a backend produces it; a
    handler converts it on the way to storage. Equality is by identity (``eq=False``)
    because the field is an array. A **field-free** plane, of a run that keeps no field
    there (:class:`~solprop.output.store.FieldRetention`), carries ``field=None`` and no
    noise phase: its position, statistics and step history only.

    Attributes
    ----------
    z : float
        Position along the waveguide, m.
    field : numpy.ndarray or None
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, complex, in the square
        root of watts, internal convention; ``None`` on a field-free plane.
    stats : dict
        Statistics evaluated at this plane, keyed by catalogue name; the values are the
        arrays :func:`solprop.stats.collect` returns.
    step_history : StepHistory
        Every attempt the stepper made since the previous plane.
    step_stats : StepStats
        The rows a periodic statistics subset produced since the previous plane, empty
        unless the run asked for one.
    noise_phase : numpy.ndarray or None
        Accumulated unitary phase of the noise field at this plane, shape
        ``(n_modes, n_pol, n_omega)``, rad, internal convention, so that processing can
        restore the background; ``None`` when no noise model is active, and on a
        field-free plane.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import SavedPlane, StepHistory
    >>> plane = SavedPlane(
    ...     z=0.0,
    ...     field=np.ones((1, 1, 5), dtype=complex),
    ...     stats={"energy": np.ones((1, 1))},
    ...     step_history=StepHistory.empty(),
    ... )
    >>> plane.shape
    (1, 1, 5)
    """

    z: float
    field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None
    stats: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
    step_history: StepHistory = dataclasses.field(default_factory=StepHistory.empty)
    step_stats: StepStats = dataclasses.field(default_factory=StepStats.empty)
    noise_phase: Float[np.ndarray, "n_modes_phase n_pol_phase n_omega_phase"] | None = (
        None
    )

    def __repr__(self) -> str:
        """Return a one-line description: the position and the field's shape.

        Returns
        -------
        str
            For example ``SavedPlane(z=1.5 m, <2x1x8192 complex>, 14 statistics)``.
        """
        return line(
            "SavedPlane",
            [
                ("z", engineering(self.z, "m")),
                ("", self._field_text()),
                ("", f"{len(self.stats)} statistics"),
            ],
        )

    def _field_text(self) -> str:
        """Return the field summarised, or a note that the plane keeps none.

        - Returns: the field's shape, or ``"no field"``.
        """
        return "no field" if self.field is None else summarise(self.field)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return f"SavedPlane at z = {engineering(self.z, 'm')}", [
            ("field", self._field_text()),
            ("statistics", ", ".join(self.stats) or "none"),
            ("steps since last", repr(self.step_history)),
            ("between planes", repr(self.step_stats)),
        ]

    def __post_init__(self) -> None:
        """Freeze the arrays and check that a noise phase matches the field.

        Raises
        ------
        OutputError
            If ``z`` is not finite, ``noise_phase`` does not have the field's shape, or
            a field-free plane carries a noise phase.
        """
        if not math.isfinite(self.z):
            msg = f"a saved plane needs a finite position, got {self.z!r}"
            raise OutputError(msg)
        object.__setattr__(self, "stats", dict(self.stats))
        if self.field is None:
            if self.noise_phase is not None:
                msg = "a field-free plane carries no noise phase"
                raise OutputError(msg)
            return
        field = _plane_array(self.field, np.complex128)
        object.__setattr__(self, "field", field)
        if self.noise_phase is not None:
            phase = _plane_array(self.noise_phase, np.float64)
            if phase.shape != field.shape:
                msg = (
                    f"noise_phase has shape {phase.shape}, expected the field's "
                    f"{field.shape}"
                )
                raise OutputError(msg)
            object.__setattr__(self, "noise_phase", phase)

    @property
    def shape(self) -> tuple[int, ...] | None:  # numpydoc ignore=RT01
        """The field's shape, or ``None`` on a field-free plane.

        ``(n_modes, n_pol, n_omega)``.
        """
        return None if self.field is None else self.field.shape


@dataclass(frozen=True, eq=False)
class StoredField:
    """A kept plane's field, exported straight into a plane store's stored form.

    What a run's service hands :meth:`~solprop.output.store.PlaneStore.save_exported`
    beside a plane that carries no field of its own (SEP 47 section 5.4): the field as
    the store keeps it, and the noise phase that a plane with its field would carry.

    Attributes
    ----------
    values : numpy.ndarray
        The field in the stored form, ``(n_modes, n_pol, n_omega)``: true-power
        amplitudes (or a spatial route's field measure), sqrt(W), in the optics
        convention, as :meth:`~solprop.output.store.PlaneStore.save` writes a plane's
        field.
    noise_phase : numpy.ndarray or None
        The noise field's accumulated phase at the plane, rad, internal convention, the
        shape of `values`; ``None`` without a noise model.

    Raises
    ------
    OutputError
        If `noise_phase` does not have the shape of `values`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import StoredField
    >>> StoredField(np.ones((1, 1, 4), complex)).values.shape
    (1, 1, 4)
    """

    values: Complex[np.ndarray, "n_modes n_pol n_omega"]
    noise_phase: Float[np.ndarray, "n_modes_phase n_pol_phase n_omega_phase"] | None = (
        None
    )

    def __post_init__(self) -> None:
        """Check that a noise phase has the field's shape."""
        if self.noise_phase is not None and self.noise_phase.shape != self.values.shape:
            msg = (
                f"noise_phase has shape {self.noise_phase.shape}, expected the field's "
                f"{self.values.shape}"
            )
            raise OutputError(msg)


def _with_phase(plane: SavedPlane, phase: np.ndarray) -> SavedPlane:
    """Return a field-free plane carrying an exported field's noise phase.

    The store's own view of a plane whose field it holds apart, in the stored form
    (:class:`StoredField`); it never leaves the store, whose hooks read the noise
    phase from the plane as from one that carries its field.
    """
    held = dataclasses.replace(plane)
    object.__setattr__(held, "noise_phase", _plane_array(phase, np.float64))
    return held


@dataclass(frozen=True)
class RunOutcome:
    """Record completion and cumulative attempted work separately from run identity.

    Attributes
    ----------
    status : {'running', 'complete', 'budget_exhausted', 'failed'}
        Current execution status. A budget stop is explicitly unfinished.
    z : float
        Last accepted or saved position, m.
    message : str
        Explanation of a stop or failure; empty on ordinary progress.
    logical_calls : collections.abc.Mapping
        Cumulative logical response ledger, including startup and failed attempts.
    atomic_calls : collections.abc.Mapping
        Cumulative provider ledger; empty when no atomic service is bound.
    execution : collections.abc.Mapping
        Current backend and provider facts, captured before resources close. Kept
        separately from the original run metadata so resumes retain their actual
        device, plan and diagnostic provenance without changing scientific identity.
    previous_executions : tuple of collections.abc.Mapping
        Closing records of earlier executions of the same resumed run, in order.
    """

    status: Literal["running", "complete", "budget_exhausted", "failed"]
    z: float
    message: str = ""
    logical_calls: Mapping[str, object] = dataclasses.field(default_factory=dict)
    atomic_calls: Mapping[str, object] = dataclasses.field(default_factory=dict)
    execution: Mapping[str, object] = dataclasses.field(default_factory=dict)
    previous_executions: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous status and nonfinite positions, and adopt ledger copies."""
        if self.status not in ("running", "complete", "budget_exhausted", "failed"):
            raise OutputError(f"unknown run outcome {self.status!r}")
        if not np.isfinite(self.z) or self.z < 0:
            raise OutputError("run outcome z must be finite and nonnegative")
        # One JSON round trip validates and copies every part: the record is built
        # at every saved plane, and one per part and per amended copy, nine a plane,
        # were about 5 % of a short capillary run on MemoryOutput.
        parts: dict[str, object] = {
            "logical_calls": dict(self.logical_calls),
            "atomic_calls": dict(self.atomic_calls),
            "execution": dict(self.execution),
            "previous_executions": [
                dict(record) for record in self.previous_executions
            ],
        }
        try:
            owned = _outcome_record(parts)
        except OutputError as error:
            # Only now is each part checked alone, so that the error names the one to
            # look at: the whole record fails exactly when one of its parts does.
            name, why = next(
                (name, why)
                for name, part in parts.items()
                if (why := _json_failure(part)) is not None
            )
            msg = (
                "execution records must contain finite JSON values: the run "
                f"outcome's {name} holds {why}"
            )
            raise OutputError(msg) from error
        for name in ("logical_calls", "atomic_calls", "execution"):
            object.__setattr__(self, name, owned[name])
        object.__setattr__(
            self,
            "previous_executions",
            tuple(cast("list[dict[str, object]]", owned["previous_executions"])),
        )


def _json_failure(part: object) -> str | None:
    """Return why a part of a run outcome is not finite JSON, or ``None`` if it is."""
    try:
        json.dumps(part, allow_nan=False)
    except (TypeError, ValueError) as error:
        return str(error)
    return None


def _json_copy(value: object) -> object:
    """Return an independent copy of a value made of validated JSON values.

    Cheaper than another JSON round trip, and than :func:`copy.deepcopy`, because the
    value is known to hold only dicts, lists, strings, numbers, booleans and
    ``None``.
    """
    if isinstance(value, dict):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value


def _amended_outcome(
    outcome: RunOutcome,
    *,
    previous_executions: tuple[Mapping[str, object], ...],
) -> RunOutcome:
    """Return a copy of an outcome with closing records a store has already validated.

    :func:`dataclasses.replace` would validate the whole record again; the store's
    earlier closing records were validated when it took them, so a saved plane validates
    its record once. The caller hands over copies of them (:func:`_json_copy`), so that
    each outcome holds its own and a change to one read back reaches no other.
    """
    amended = copy.copy(outcome)
    object.__setattr__(amended, "previous_executions", previous_executions)
    return amended


def _outcome_record(record: Mapping[str, object]) -> dict[str, object]:
    """Own finite JSON values so later provider mutations cannot change a snapshot.

    - ``record``: execution facts composed of finite JSON values.
    """
    try:
        return cast(
            "dict[str, object]", json.loads(json.dumps(dict(record), allow_nan=False))
        )
    except (TypeError, ValueError) as error:
        raise OutputError(
            f"execution records must contain finite JSON values: {error}"
        ) from error


@dataclass(frozen=True, eq=False)
class TerminalCheckpoint:
    """Keep an accepted field between scheduled saves without adding an observation.

    Attributes
    ----------
    plane : SavedPlane
        Accepted field in the internal FFT convention and the output amplitude
        convention, with already available history and periodic statistics.
        Creating a checkpoint never requests fresh material statistics. It always
        carries the field, also in a run that keeps none at its planes, so that a
        stopped run resumes (SEP 22, Q1).
    next_save_index : int
        Index of the first scheduled save strictly beyond this field.
    landed : bool
        Whether the field is an accepted endpoint rather than a dense observation.
    """

    plane: SavedPlane
    next_save_index: int
    landed: bool = True

    def __post_init__(self) -> None:
        """Reject an unusable save index, a field-free or a nonfinite checkpoint."""
        if type(self.next_save_index) is not int or self.next_save_index < 0:
            raise OutputError(
                "checkpoint next_save_index must be a nonnegative integer"
            )
        if self.plane.field is None:
            raise OutputError(
                "a terminal checkpoint carries the field, also in a run that keeps "
                "none at its planes"
            )
        if (
            not np.isfinite(self.plane.z)
            or self.plane.z < 0
            or not np.all(np.isfinite(self.plane.field))
        ):
            raise OutputError("checkpoint position and field must be finite")


@dataclass(frozen=True, eq=False)
class MarchSnapshot:  # numpydoc ignore=PR02
    """An Adams march's state at an accepted position, for a resume to continue.

    What a resume needs to carry on as the uninterrupted run would: the accepted state
    exactly as the backend held it, the history's samples in their engine rows, which
    sample each history entry is, the order, the rejection count and the step the
    controller proposed next. A snapshot is never the record of what was computed, only
    of where a march stood: a resume continues it when its history key matches the
    resuming method's, and otherwise starts the history afresh.

    Equality is by identity (``eq=False``) because the fields hold arrays.

    Attributes
    ----------
    z : float
        The accepted position, m.
    state : numpy.ndarray
        The accepted state, shape ``(n_modes, n_pol, n_omega)``, in the internal FFT
        convention, exactly as the backend held it. A file stores it with a leading
        axis of one, the layout it has always had (SEP 47 section 6.7).
    proposal : float
        The step the controller proposed next, m.
    method : str
        The method's representation, kept as provenance.
    order : int
        The order chosen for the next step.
    rejections : int
        Consecutive rejections so far.
    last_rejected : bool
        Whether the last trial was rejected.
    positions : numpy.ndarray
        Each history entry's position, m, oldest first.
    identities : tuple of str
        Each entry's kind (``"accepted"`` or ``"predicted"``).
    slots : numpy.ndarray
        Each entry's engine row.
    rows : collections.abc.Mapping
        The engine's rows by name: the samples (``"store"``) and, for a compensated
        exponential march, the sampled states and rates and each sample's flow
        weights, in the engine's own layout, which keeps a leading axis of one after
        the row axis.
    collar_was_quiet : bool
        Whether the in-stage collar was left out, quiet, at the last attempt, which
        holds the first step after it returns to its stable step (SEP 39 section
        6.5.9).
    history_key : str or None
        What defines the history
        (:attr:`solprop.stepper.multistep.Multistep.history_key`), which the resuming
        method's must equal; ``None`` for a snapshot that names none, which starts the
        history afresh.

    Raises
    ------
    OutputError
        If the entries disagree in number, or a position or the state is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.records import MarchSnapshot
    >>> snapshot = MarchSnapshot(
    ...     z=0.5, state=np.ones((1, 1, 4), complex), proposal=0.1, method="M()",
    ...     order=2, rejections=0, last_rejected=False,
    ...     positions=np.array([0.4, 0.5]), identities=("accepted", "predicted"),
    ...     slots=np.array([0, 1]), rows={"store": np.zeros((4, 1, 1, 1, 4), complex)},
    ... )
    >>> snapshot.nbytes
    320
    """

    z: float
    state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    proposal: float
    method: str
    order: int
    rejections: int
    last_rejected: bool
    positions: Float[np.ndarray, " n_entries"]
    identities: tuple[str, ...]
    slots: Int[np.ndarray, " n_entries"]
    rows: Mapping[str, np.ndarray]
    collar_was_quiet: bool = False
    history_key: str | None = None

    def __post_init__(self) -> None:
        """Refuse a snapshot whose entries disagree or whose numbers are not finite.

        Raises
        ------
        OutputError
            Naming the inconsistency.
        """
        count = len(self.identities)
        if self.positions.shape != (count,) or self.slots.shape != (count,):
            raise OutputError("march snapshot entries disagree in number")
        if not (
            math.isfinite(self.z)
            and math.isfinite(self.proposal)
            and np.all(np.isfinite(self.positions))
            and np.all(np.isfinite(self.state))
        ):
            raise OutputError("march snapshot positions and state must be finite")

    @property
    def nbytes(self) -> int:  # numpydoc ignore=RT01
        """The bytes the snapshot's arrays hold.

        The state and every engine row.
        """
        return int(self.state.nbytes + sum(row.nbytes for row in self.rows.values()))


def _holds_carried(plane: SavedPlane, carried: SavedPlane) -> bool:
    """Return whether a checkpoint already begins with the attempts a resume carried.

    The checkpoint a store holds after a resume, or a copy of it, starts with the
    carried attempts, every column equal; a new one from the resumed march starts
    where the resume did, with attempts of its own and its own wall clock.

    - Returns: True when `plane`'s history and rows start with `carried`'s, or `carried`
      holds nothing to add.
    """
    before, history = carried.step_history, plane.step_history
    if before.n_steps == 0:
        return carried.step_stats.n_rows == 0
    if history.n_steps < before.n_steps:
        return False
    return all(
        np.array_equal(getattr(history, name)[: before.n_steps], getattr(before, name))
        for name in HISTORY_FIELDS
    )


def _prepend_checkpoint_history(
    previous: SavedPlane, current: SavedPlane
) -> SavedPlane:
    """Retain attempts and periodic rows preceding a fresh-start resume."""
    history = StepHistory(
        **{
            name: np.concatenate(
                (
                    getattr(previous.step_history, name),
                    getattr(current.step_history, name),
                )
            )
            for name in HISTORY_FIELDS
        }
    )
    before, after = previous.step_stats, current.step_stats
    if before.n_rows and after.n_rows and before.names != after.names:
        raise OutputError("checkpoint periodic statistic names changed on resume")
    names = before.names if before.n_rows else after.names
    values = {}
    for name in names:
        if before.n_rows and after.n_rows:
            values[name] = np.concatenate((before.values[name], after.values[name]))
        else:
            values[name] = before.values[name] if before.n_rows else after.values[name]
    rows = StepStats(
        z=np.concatenate((before.z, after.z)),
        step=np.concatenate((before.step, after.step + previous.step_history.n_steps)),
        values=values,
    )
    return dataclasses.replace(current, step_history=history, step_stats=rows)


@dataclass(frozen=True)
class RecipeFile:
    """A data file a recorded recipe refers to (SEP 41, S2).

    Attributes
    ----------
    path : str
        The path the recipe wrote, relative to its directory.
    sha256 : str
        Its SHA-256, as the recipe recorded it.
    data : bytes or None
        Its contents, embedded under the generated name it is filed by; ``None``
        when the run's files together passed the embedding cap, and only the path and
        hash are kept.
    """

    path: str
    sha256: str
    data: bytes | None = None


@dataclass(frozen=True)
class RecipeRecord:
    """The recipe a run was built from, as its output records it once (SEP 41, S2).

    Provenance outside the run record and its hash: a file without one is a file from
    a script, and a reader that does not know it ignores it.

    Attributes
    ----------
    format : int
        The recipe format.
    authored : str
        The recipe's text as loaded, or its canonical TOML if it was edited.
    resolved : str
        The recipe to rebuild the run from: the authored keys with the drawn seed, the
        keywords whose value written back gives the same record, at the values used,
        and the execution settings at run start.
    files : collections.abc.Mapping
        The data files the recipe refers to, by the generated name each is filed by.

    Examples
    --------
    >>> from solprop.output.records import RecipeRecord
    >>> record = RecipeRecord(1, "[recipe]", "[recipe]", {})
    >>> record.authored_sha256 == record.resolved_sha256
    True
    """

    format: int
    authored: str
    resolved: str
    files: Mapping[str, RecipeFile] = dataclasses.field(default_factory=dict)

    @property
    def authored_sha256(self) -> str:  # numpydoc ignore=RT01
        """SHA-256 of the authored text, in hexadecimal."""
        return hashlib.sha256(self.authored.encode("utf-8")).hexdigest()

    @property
    def resolved_sha256(self) -> str:  # numpydoc ignore=RT01
        """SHA-256 of the resolved text, in hexadecimal."""
        return hashlib.sha256(self.resolved.encode("utf-8")).hexdigest()
