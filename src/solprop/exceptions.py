# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The exception and warning hierarchies.

Every error raised by solprop is a :class:`SolpropError` and every warning it emits is a
:class:`SolpropWarning`, so callers can catch or filter the whole family with one clause
and still tell the cases apart. Raise the most specific subclass; never raise a bare
:class:`Exception`, and never return a sentinel where raising is correct.
"""

__all__ = [
    "AdiabaticTaperWarning",
    "ApertureEdgeWarning",
    "BackendError",
    "BoundaryError",
    "BoundaryWarning",
    "ConvergenceError",
    "CouplingError",
    "DiluteScalingWarning",
    "DriverError",
    "ExponentBudgetWarning",
    "GasFlowWarning",
    "GeneralModalBasisWarning",
    "GrazingIncidenceWarning",
    "GridError",
    "HostMarchWarning",
    "MaterialError",
    "MaterialRangeWarning",
    "MaterialStatusWarning",
    "MemoryBudgetError",
    "MemoryBudgetWarning",
    "ModeError",
    "NoZeroDispersionError",
    "NoiseError",
    "OpticsGeometryError",
    "OutputError",
    "ParameterPath",
    "PolarisationError",
    "ProcessingError",
    "PulseError",
    "RamanTemperatureWarning",
    "ResponseBudgetError",
    "RunTerminated",
    "SampledPeakWarning",
    "SolpropError",
    "SolpropWarning",
    "StepperError",
    "StopRequested",
    "SurrogateAreaWarning",
    "TDSEArchiveError",
    "TDSEConsistencyError",
    "TDSEConsistencyWarning",
    "TDSECostWarning",
    "TDSEDomainWarning",
    "TDSEUnqualifiedModelWarning",
    "ToolsError",
    "TranslationError",
]


type ParameterPath = str | tuple[str, ...]
"""The parameter an error or warning concerns: a recipe path, or several for a conflict.

A path names a keyword of a route, then ``.name`` for a keyword of an object, ``[i]``
for an element of an array and ``["key"]`` for a key of a mapping, as in
``"pressure"``, ``"pulses[1].energy"`` or ``'plasma["Ar"]'`` (SEP 41 section 5.1).
"""


class SolpropError(Exception):
    """Base class for every error raised by solprop.

    Parameters
    ----------
    *args : object
        The message, as for :class:`Exception`.
    parameter : str or tuple of str or None, optional
        The parameter the error concerns, as a path (:data:`ParameterPath`), or a
        tuple of paths when it is a conflict between them; ``None`` when it concerns
        no single parameter. Set where the interface refuses a value, so a front end
        can put the message next to its field (SEP 41, S10).
    rule : str or None, optional
        The identifier of the combination rule the refusal enforces, where it is one
        (SEP 41, S8); ``None`` otherwise.

    Attributes
    ----------
    parameter : str or tuple of str or None
        As given.
    rule : str or None
        As given.

    Examples
    --------
    >>> from solprop.exceptions import DriverError
    >>> error = DriverError("pressure must be positive", parameter="pressure")
    >>> str(error), error.parameter, error.rule
    ('pressure must be positive', 'pressure', None)
    """

    def __init__(
        self,
        *args: object,
        parameter: ParameterPath | None = None,
        rule: str | None = None,
    ) -> None:
        """Hold the message, and the parameter and rule it concerns."""
        super().__init__(*args)
        self.parameter = parameter
        self.rule = rule


class RunTerminated(SystemExit):
    """A run stopped by SIGTERM, after writing where it stood.

    A scheduler's time limit (Slurm's, for one) sends SIGTERM before it kills the
    process. During a run it raises this at once, or as soon as the step or the save
    being written is complete, so the run's terminal checkpoint and march snapshot are
    written as for any failure and ``resume=True`` continues the run. It is a
    :class:`SystemExit` carrying the conventional exit status ``128 + SIGTERM`` (143),
    so a script that does not catch it ends as a terminated process does.

    Parameters
    ----------
    signal_number : int
        The signal that stopped the run.

    Examples
    --------
    >>> import signal
    >>> from solprop.exceptions import RunTerminated
    >>> error = RunTerminated(signal.SIGTERM)
    >>> error.code, str(error)
    (143, 'terminated by SIGTERM')
    """

    def __init__(self, signal_number: int) -> None:
        """Hold the signal and the exit status it implies."""
        import signal  # noqa: PLC0415 - only for the signal's name

        super().__init__(128 + int(signal_number))
        self.signal_number = int(signal_number)
        self.signal_name = signal.Signals(self.signal_number).name

    def __str__(self) -> str:
        """Return what stopped the run.

        Returns
        -------
        str
            ``"terminated by <signal>"``.
        """
        return f"terminated by {self.signal_name}"


class StopRequested(BaseException):
    """Raised inside a run when its own stop flag is set (SEP 41, S17).

    The run loop writes its state as for SIGTERM, and
    :meth:`solprop.driver.simulation.Simulation.run` returns the output when the flag
    that stopped it is its own: a requested stop is not an error. It derives from
    :class:`BaseException`, as :class:`KeyboardInterrupt` does, so no ``except
    Exception`` on the way swallows it.

    Parameters
    ----------
    flag : solprop.log.StopFlag
        The flag that was set: any object with an ``is_set()`` method.

    Examples
    --------
    >>> import threading
    >>> from solprop._interrupts import poll
    >>> from solprop.exceptions import StopRequested
    >>> flag = threading.Event()
    >>> poll(flag)
    >>> flag.set()
    >>> try:
    ...     poll(flag)
    ... except StopRequested as stop:
    ...     print(stop, stop.flag is flag)
    stopped by the caller's stop flag True
    """

    def __init__(self, flag: object) -> None:
        """Hold the flag that was set."""
        super().__init__("stopped by the caller's stop flag")
        self.flag = flag


class DriverError(SolpropError, RuntimeError):
    """A run was asked for something the driver cannot execute."""


class MemoryBudgetError(DriverError):
    """A run's estimated persistent memory exceeds its declared memory budget."""


class BoundaryError(SolpropError, RuntimeError):
    """Light reached the time window's edge beyond what the boundary check allows.

    Raised by the boundary monitor (SEP 21), and for an invalid check.
    """


class GridError(SolpropError, ValueError):
    """A time or frequency grid is inconsistent or unsupported."""


class MaterialError(SolpropError, ValueError):
    """A material, gas or model is unknown or used outside its validity range."""


class OpticsGeometryError(MaterialError):
    """An optical setting has no forward, non-grazing ray or diffraction order."""


class ModeError(SolpropError, ValueError):
    """A waveguide mode or mode set is invalid."""


class NoZeroDispersionError(ModeError):
    """``beta_2`` has one sign at both ends of a zero-dispersion search window."""


class PulseError(SolpropError, ValueError):
    """An input pulse specification is invalid or does not fit the grid."""


class PolarisationError(SolpropError, ValueError):
    """A Jones or Stokes vector, optical element or polarisation angle is invalid."""


class NoiseError(SolpropError, ValueError):
    """A shot-noise request is invalid."""


class TranslationError(SolpropError, ValueError):
    """The translation layer could not build a valid kernel problem."""


class BackendError(SolpropError, RuntimeError):
    """A backend is unavailable, misused or failed during propagation."""


class OutputError(SolpropError, RuntimeError):
    """An output handler could not save, load or resume a run."""


class TDSEArchiveError(OutputError):
    """A TDSE response archive could not be opened, serialized or written (SEP 20).

    Raised before any provider call when the archive path is refused, and in place
    of the next provider call after a failed write, so a run stops at its last
    optical checkpoint. A provider failure stays the primary error when recording it
    also fails.
    """


class TDSEConsistencyError(MaterialError):
    """A haCC response failed, or could not complete, its weak-field linear check.

    Raised before any propagation call when the per-atom linear susceptibility of the
    atomic model departs from optmat's helium by more than the declared ``refuse``
    bound, or when the check's own extraction is invalid (SEP 20, section 5.4). The
    structured result is attached as ``result`` when one exists.
    """

    result: object | None = None


class ProcessingError(SolpropError, ValueError):
    """A post-processing or plotting request is invalid."""


class ConvergenceError(SolpropError, RuntimeError):
    """A numerical procedure did not converge within its budget."""


class CouplingError(SolpropError, ValueError):
    """A field plane, a remap or a launch between simulations is refused (SEP 31)."""


class ResponseBudgetError(ConvergenceError):
    """A run stopped before its end because its source-call budget was spent.

    Raised before a trial that the budget could not pay for, so no step is accepted to
    fit the budget and the run is never marked complete (SEP 13 section 5). The state
    at the last accepted step travels with the error; a file output holds every plane
    reached, from which the run resumes afresh.

    Parameters
    ----------
    message : str
        What stopped and where.
    z : float
        Position of the last accepted step, m.
    field : numpy.ndarray or None
        The state there.
    calls : int
        Source calls charged before the stop.

    Attributes
    ----------
    z : float
        Position of the last accepted step, m.
    field : numpy.ndarray or None
        The state there, in the kernel's canonical layout and FFT convention.
    calls : int
        Source calls charged before the stop.
    """

    def __init__(self, message: str, *, z: float, field: object, calls: int) -> None:
        """Carry the stop's position, state and spent calls with the message."""
        super().__init__(message)
        self.z = z
        self.field = field
        self.calls = calls


class StepperError(SolpropError, ValueError):
    """A tableau, controller, schedule or landing setting is invalid."""


class ToolsError(SolpropError, ValueError):
    """A design calculation was asked for something it cannot answer."""


class SolpropWarning(UserWarning):
    """Base class for every warning raised by solprop.

    Parameters
    ----------
    *args : object
        The message, as for :class:`Warning`.
    parameter : str or tuple of str or None, optional
        The parameter the warning concerns, as a path (:data:`ParameterPath`);
        ``None`` when it concerns no single parameter (SEP 41, S10). A warning that
        carries one is issued as an instance:
        ``warnings.warn(MaterialRangeWarning(message, parameter="gas"))``.

    Attributes
    ----------
    parameter : str or tuple of str or None
        As given.

    Examples
    --------
    >>> from solprop.exceptions import MaterialRangeWarning
    >>> MaterialRangeWarning("outside the fit", parameter="gas").parameter
    'gas'
    """

    def __init__(self, *args: object, parameter: ParameterPath | None = None) -> None:
        """Hold the message and the parameter it concerns."""
        super().__init__(*args)
        self.parameter = parameter


class MaterialRangeWarning(SolpropWarning):
    """A material model is evaluated outside the validity range of its source."""


class MaterialStatusWarning(SolpropWarning):
    """A resolved material model is flagged provisional or untested by its source."""


class DiluteScalingWarning(SolpropWarning):
    """A gas's dilute-limit index departs from its finite-density law beyond a bound."""


class ExponentBudgetWarning(SolpropWarning):
    """A step the run cannot shrink forms a propagator factor near the overflow limit.

    A replayed schedule's step, or the step floor ``dz_min``, between nine tenths of
    ``log(DBL_MAX)`` and the limit itself (SEP 22 section 4.5): the run completes only
    while its field stays small enough that the factor times it does not overflow.
    """


class GrazingIncidenceWarning(SolpropWarning):
    """A grazing-incidence model is evaluated outside the range it is valid in.

    A spatial spectral bin with an unresolved clamp branch, a coherent wall block near
    its validity limit, or a wall-pair guide (rectangular or slab) past its first-order
    ratios.
    """


class HostMarchWarning(SolpropWarning):
    """The native backend steps an Adams run with solprop's Python march.

    The native backend executes the ``Multistep()`` march natively (SEP 39) wherever
    it covers the configuration; elsewhere it runs solprop's own march,
    the reference, with the same method and accuracy at a higher host cost per step.
    The warning names the reason, and the run's backend record says which march ran
    (``adams_march``).
    """


class AdiabaticTaperWarning(SolpropWarning):
    """A multimode run on a guide whose modes move along z omits their coupling."""


class ApertureEdgeWarning(SolpropWarning):
    """A wide ``tanh`` aperture edge converges slowly for light at the aperture centre.

    Amplitude-weighted ``tanh`` with ``2 <= r/w < 4`` (SEP 23, open question 10): the
    profile slopes at its centre, so a bright beam there is collected with a lattice
    error near ``1e-5``, where ``rcos`` is flat inside ``r - w``.
    """


class MemoryBudgetWarning(SolpropWarning):
    """A run's estimated persistent memory exceeds a budget declared to warn."""


class GeneralModalBasisWarning(SolpropWarning):
    """A general modal basis leaves its budget somewhere the run's light can reach."""


class BoundaryWarning(SolpropWarning):
    """The time window's guards or collar hold more of the field than its budget allows.

    Issued by the boundary monitor (SEP 21): at the launch, at the first committed step
    whose guard occupancy exceeds the budget, and at the end of a run whose collar
    removed more than the budget.
    """


class GasFlowWarning(SolpropWarning):
    """A steady-flow gas fill leaves the regime its laminar continuum law describes."""


class RamanTemperatureWarning(SolpropWarning):
    """Rotational Raman populations are held at one temperature; the gas's varies."""


class SampledPeakWarning(SolpropWarning):
    """A node-maximum plasma statistic is read where no quadrature node is on the axis.

    A field-scale radial rule of few nodes (SEP 19) integrates the source but samples
    the axial peak at its innermost node; the ``on_axis_*`` statistics read the axis.
    """


class SurrogateAreaWarning(SolpropWarning):
    """A scalar model takes a provider channel's area at a frequency nobody chose."""


class TDSECostWarning(SolpropWarning):
    """A TDSE response runs on a many-node radial rule: each node is one atom a call."""


class TDSEDomainWarning(SolpropWarning):
    """A haCC preset is used beyond its qualifying duration, intensity or carrier."""


class TDSEConsistencyWarning(SolpropWarning):
    """A haCC weak-field linear check warned, or was skipped at the caller's request."""


class TDSEUnqualifiedModelWarning(SolpropWarning):
    """A custom haCC basis has no qualified accuracy, or lacks the current closure."""
