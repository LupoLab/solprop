# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bridge solprop's physics layer to optmat: units, identities, warnings and errors.

Every call into optmat goes through this module. It imports optmat only inside its
functions: optmat's registries take more than a second to import, and
``import solprop`` must not pay for them.

What crosses here:

- **units**: pressure in bar at solprop's interface, ``p_Pa = 1e5 p_bar`` exactly at
  optmat's; angular frequency at solprop's, vacuum wavelength ``2 pi c / omega`` at
  optmat's, with ``omega = 0`` the static limit ``inf``;
- **identities**: an optmat model's ``identity`` becomes a
  :class:`~solprop.physdata.selection.MaterialRecord`;
- **warnings**: optmat's ``ModelRangeWarning`` and ``ModelAbsorptionWarning`` become one
  :class:`~solprop.exceptions.MaterialRangeWarning` per material and evaluation
  context, naming the parts of the evaluated band outside the model's qualified
  intervals and the reasons of the gaps they fall in; a ``provisional`` or
  ``untested`` model warns once (:class:`~solprop.exceptions.MaterialStatusWarning`)
  and a ``comparison`` or ``user`` one is recorded silently;
- **errors**: optmat's ``OptmatError`` (``StateError``, ``DataError``) becomes
  :class:`~solprop.exceptions.MaterialError`, naming the material, the quantity, the
  state or band asked for, and the model.

An *evaluation context* (:func:`material_context`) is the scope within which each
warning is raised once: a run (its construction, translation and record), a fill's
sampling, a flow calculation. Outside any context every call is its own. A context
keeps a :class:`MaterialLog`: the warnings it has raised and the parts of each band
that lay outside a model's qualified intervals, which a run's material manifest
records.

An adapter keeps its last few evaluations (:class:`EvaluationCache`): a material's
index or polarizability does not depend on z, yet a translation reads it at every
sample of a z table. A repeated frequency array takes the stored arrays, and the
warnings its first evaluation captured are raised again under the same policy, so
what a context warns about and records does not depend on the cache.
"""

import contextlib
import contextvars
import itertools
import math
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from typing import Any, Final

import numpy as np
from jaxtyping import Float

from solprop.constants import BAR, C
from solprop.exceptions import (
    MaterialError,
    MaterialRangeWarning,
    MaterialStatusWarning,
)
from solprop.physdata.selection import MaterialLog, MaterialRecord, OptmatModel

__all__: list[str] = []

# optmat's model classes cannot be named in a signature without importing optmat at
# module scope, a second of import time that `import solprop` must not pay (as for
# pyfluids in `equation_of_state`). The adapters reach their models' methods through
# this one alias; a model is checked against the `OptmatModel` protocol at run time.
Untyped = Any


_LOGS: Final[contextvars.ContextVar[tuple[MaterialLog, ...]]] = contextvars.ContextVar(
    "solprop_optmat_logs", default=()
)
"""The logs of the open evaluation contexts, outermost first."""

_FLAGGED: Final = frozenset({"provisional", "untested"})
"""The statuses that warn when a model is resolved; the others are recorded only."""


def bar_to_pa(pressure_bar: float) -> float:
    """Convert a pressure from bar, solprop's interface unit, to optmat's pascals.

    Parameters
    ----------
    pressure_bar : float
        Pressure in bar.

    Returns
    -------
    float
        Pressure in Pa, exactly ``1e5`` times the input.
    """
    return float(pressure_bar) * BAR


def pa_to_bar(pressure_pa: float) -> float:
    """Convert a pressure from optmat's pascals to bar.

    Parameters
    ----------
    pressure_pa : float
        Pressure in Pa.

    Returns
    -------
    float
        Pressure in bar.
    """
    return float(pressure_pa) / BAR


def omega_to_wavelength(
    omega: Float[np.ndarray, "*shape"],
) -> Float[np.ndarray, "*shape"]:
    """Return the vacuum wavelengths of non-negative angular frequencies.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies in rad/s, >= 0, already validated.

    Returns
    -------
    numpy.ndarray
        ``2 pi c / omega`` in m; ``inf`` (optmat's static limit) where ``omega = 0``.
    """
    values = np.asarray(omega, dtype=float)
    with np.errstate(divide="ignore"):
        return np.asarray(np.where(values > 0, 2 * math.pi * C / values, math.inf))


def wavelength_to_omega(
    wavelength: Float[np.ndarray, "*shape"],
) -> Float[np.ndarray, "*shape"]:
    """Return the angular frequencies of positive vacuum wavelengths.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Vacuum wavelengths in m, > 0; ``inf`` is the static limit.

    Returns
    -------
    numpy.ndarray
        ``2 pi c / lambda`` in rad/s; 0 at ``inf``.
    """
    return np.asarray(2 * math.pi * C / np.asarray(wavelength, dtype=float))


def untyped(model: object) -> Untyped:
    """Return an optmat object for attribute access, which its protocol cannot type.

    Parameters
    ----------
    model : object
        An optmat model, response or equation of state.

    Returns
    -------
    Any
        The same object.
    """
    return model


def optmat_version() -> str:
    """Return the installed optmat version.

    Returns
    -------
    str
        ``optmat.__version__``.
    """
    import optmat  # noqa: PLC0415 - optmat is imported lazily, see the module docstring

    return str(optmat.__version__)


def record(model: object) -> MaterialRecord:
    """Return the record of a resolved optmat model.

    Parameters
    ----------
    model : OptmatModel
        Any optmat object with an ``identity``.

    Returns
    -------
    MaterialRecord
        Its identity in plain values.

    Raises
    ------
    MaterialError
        If the object has no optmat identity.
    """
    if not isinstance(model, OptmatModel):
        msg = (
            f"a {type(model).__name__} is not an optmat model: a model object carries "
            "an optmat identity"
        )
        raise MaterialError(msg)
    return MaterialRecord.from_identity(model.identity)


def open_log() -> MaterialLog | None:
    """Return the log of the innermost open evaluation context, if any.

    Returns
    -------
    MaterialLog or None
        The log, or ``None`` outside every context.
    """
    logs = _LOGS.get()
    return logs[-1] if logs else None


@contextlib.contextmanager
def material_context(log: MaterialLog | None = None) -> Iterator[MaterialLog]:
    """Open an evaluation context in which each material warning is raised once.

    Without a `log`, a context nested in another shares it. With one, the context
    resumes that log -- a run's, kept between its construction, its translation and
    its record, so that each warning is raised once per run however its work is
    spread in time. A log resumed inside another open context is stacked on it: a
    warning either has already raised is not raised again, and both record what
    happens inside.

    Parameters
    ----------
    log : MaterialLog or None, optional
        The log to resume; ``None`` shares the open context's, or starts one.

    Yields
    ------
    MaterialLog
        The context's log.
    """
    logs = _LOGS.get()
    if log is None and logs:
        yield logs[-1]
        return
    if log is not None and log in logs:
        yield log
        return
    chosen = MaterialLog() if log is None else log
    token = _LOGS.set((*logs, chosen))
    try:
        yield chosen
    finally:
        _LOGS.reset(token)


def warn_once(
    key: tuple[str, ...], message: str, category: type[Warning], stacklevel: int = 4
) -> None:
    """Raise a warning unless an open evaluation context has raised it already.

    Parameters
    ----------
    key : tuple of str
        What makes two warnings the same one: a kind and a material's digest, say.
    message : str
        The warning's text.
    category : type
        The warning's class.
    stacklevel : int, optional
        Passed to :func:`warnings.warn`.
    """
    logs = _LOGS.get()
    if any(key in log.seen for log in logs):
        return
    for log in logs:
        log.seen.add(key)
    warnings.warn(message, category, stacklevel=stacklevel)


def _record_range(
    model: object, material: str, wavelength: Float[np.ndarray, "*shape"] | None
) -> None:
    """Record, in every open context, where a band left a model's intervals.

    Parameters
    ----------
    model : OptmatModel
        The model evaluated, with ``intervals`` and possibly ``gap_reasons``.
    material : str
        The material it serves.
    wavelength : numpy.ndarray or None
        The evaluated vacuum wavelengths in m, where known.
    """
    logs = _LOGS.get()
    if not logs:
        return
    entry = record(model)
    intervals: tuple[tuple[float, float], ...] = tuple(
        getattr(untyped(model), "intervals", ())
    )
    parts = (
        outside_parts(wavelength, intervals, tuple(getattr(model, "gap_reasons", ())))
        if intervals and wavelength is not None
        else ()
    )
    for log in logs:
        log.record_range(entry.digest, material, entry.name, parts)


def check_status(model: object, material: str) -> MaterialRecord:
    """Apply the status policy to a resolved model and return its record.

    A ``provisional`` or ``untested`` model warns once per evaluation context; a
    ``validated``, ``comparison`` or ``user`` one does not, since the first is a
    curated default and the other two were chosen by the caller.

    Parameters
    ----------
    model : OptmatModel
        The resolved model.
    material : str
        The material it serves, for the message.

    Returns
    -------
    MaterialRecord
        The model's record.

    Warns
    -----
    MaterialStatusWarning
        If the model is ``provisional`` or ``untested``.
    """
    entry = record(model)
    if entry.status in _FLAGGED:
        warn_once(
            ("status", entry.digest),
            f"{material}: optmat's {entry.kind} model {entry.name} is {entry.status}"
            f" ({entry.status_reason}); it is used as selected, and recorded",
            MaterialStatusWarning,
        )
    return entry


def _nm(value: float) -> str:
    """Return a wavelength in nm as text.

    Parameters
    ----------
    value : float
        Wavelength in m; ``inf`` is the static limit.

    Returns
    -------
    str
        Four significant figures and the unit, or ``inf``.
    """
    return "inf" if math.isinf(value) else f"{value * 1e9:.4g} nm"


def outside_parts(
    wavelength: Float[np.ndarray, "*shape"],
    intervals: Sequence[tuple[float, float]],
    gap_reasons: Sequence[str] = (),
) -> tuple[str, ...]:
    """Describe the parts of a band that lie outside a model's qualified intervals.

    Parameters
    ----------
    wavelength : numpy.ndarray
        The evaluated vacuum wavelengths in m.
    intervals : sequence of tuple of float
        The model's disjoint, ascending qualified intervals in m.
    gap_reasons : sequence of str, optional
        The reason of each gap between two consecutive intervals, where known.

    Returns
    -------
    tuple of str
        One phrase per part outside: below the first interval, in each gap the band
        reaches (with its reason), and beyond the last; empty if the band is inside.
    """
    values = np.ravel(np.asarray(wavelength, dtype=float))
    values = values[~np.isnan(values)]
    parts: list[str] = []
    lowest, highest = intervals[0][0], intervals[-1][1]
    below = values[values < lowest]
    if below.size:
        parts.append(
            f"{_nm(float(below.min()))} to {_nm(lowest)}, below its shortest "
            "qualified wavelength"
        )
    for index, ((_, end), (start, _)) in enumerate(itertools.pairwise(intervals)):
        inside = values[(values > end) & (values < start)]
        if inside.size:
            reason = gap_reasons[index] if index < len(gap_reasons) else "unqualified"
            parts.append(f"{_nm(end)} to {_nm(start)}, a gap ({reason})")
    above = values[values > highest]
    if above.size:
        parts.append(
            f"{_nm(highest)} to {_nm(float(above.max()))}, beyond its longest "
            "qualified wavelength"
        )
    return tuple(parts)


def _range_message(
    model: object, material: str, wavelength: Float[np.ndarray, "*shape"] | None
) -> str:
    """Return the text of a range warning for a band outside a model's intervals.

    Parameters
    ----------
    model : OptmatModel
        The model, with ``intervals`` and, for a fit with several, ``gap_reasons``.
    material : str
        The material, for the message.
    wavelength : numpy.ndarray or None
        The evaluated vacuum wavelengths in m, where known.

    Returns
    -------
    str
        The model, its intervals, and the parts of the band outside them.
    """
    entry = record(model)
    intervals: tuple[tuple[float, float], ...] = tuple(
        getattr(untyped(model), "intervals", ())
    )
    text = f"{material}: optmat's {entry.name} is evaluated outside its qualified "
    if not intervals or wavelength is None:
        return text + "range"
    described = ", ".join(f"{_nm(low)} to {_nm(high)}" for low, high in intervals)
    parts = outside_parts(
        wavelength, intervals, tuple(getattr(model, "gap_reasons", ()))
    )
    held = getattr(model, "out_of_range", "") == "hold"
    return (
        f"{text}intervals ({described}); outside them the band reaches "
        f"{'; '.join(parts)}. "
        + (
            "The table holds its end values there"
            if held
            else "It is evaluated there as the model extrapolates"
        )
    )


@contextlib.contextmanager
def _captured(
    model: object, quantity: str, *, material: str, state: str = ""
) -> Iterator[list[warnings.WarningMessage]]:
    """Capture optmat's warnings and translate its refusals.

    Parameters
    ----------
    model : OptmatModel
        The model the body evaluates.
    quantity : str
        What it evaluates, for messages.
    material : str
        The material, for messages.
    state : str, optional
        The state asked for, for error messages.

    Yields
    ------
    list of warnings.WarningMessage
        The warnings the body raises, filled as it runs.

    Raises
    ------
    MaterialError
        If optmat refuses the evaluation.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy

    entry = record(model)
    where = f" at {state}" if state else ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield caught
        except OptmatError as error:
            msg = (
                f"{material}: optmat refuses the {quantity}{where} ({entry.name}, "
                f"digest {entry.digest[:12]}): {error}"
            )
            raise MaterialError(msg) from error


@contextlib.contextmanager
def translated(
    model: object,
    quantity: str,
    *,
    material: str,
    wavelength: Float[np.ndarray, "*shape"] | None = None,
    state: str = "",
) -> Iterator[None]:
    """Evaluate optmat under solprop's warning and error policy.

    Parameters
    ----------
    model : OptmatModel
        The model the body evaluates.
    quantity : str
        What it evaluates, for messages: ``"polarizability"``, ``"number density"``.
    material : str
        The material, for messages.
    wavelength : numpy.ndarray or None, optional
        The vacuum wavelengths in m the body evaluates, for range messages.
    state : str, optional
        The state asked for, for error messages: ``"30 bar and 293.15 K"``.

    Yields
    ------
    None
        Control, with optmat's warnings captured.

    Raises
    ------
    MaterialError
        If optmat refuses the evaluation.

    Warns
    -----
    MaterialRangeWarning
        Once per material and evaluation context, if the band leaves the model's
        qualified intervals or an absorbing model's real part is asked for.
    """
    with _captured(model, quantity, material=material, state=state) as caught:
        yield
    with material_context():
        _reemit(caught, model, material, wavelength)


EVALUATION_CACHE_SIZE: Final[int] = 8
"""Frequency arrays an adapter keeps the evaluations of, per quantity together.

A translation reads a material at a few arrays -- the grid's band, the frame's
reference frequency, a mode's own sampling -- many times each; eight covers them with
room to spare, and bounds what an adapter holds to eight pairs of spectra.
"""


class EvaluationCache:
    """Keep an adapter's last few optmat evaluations, by quantity and frequency array.

    A material model's index or polarizability is a function of frequency alone, and
    an adapter is immutable, so a frequency array whose bytes equal an earlier one's
    takes the arrays stored for it: bitwise what a fresh evaluation returns. The
    warnings that first evaluation captured are stored beside them and raised again
    at every use, under the policy of :func:`translated`, so each open evaluation
    context still warns once and records the same bands; outside every context each
    use warns, as each evaluation does. Least recently used entries are evicted
    beyond :attr:`size`. The cache is not copied with its adapter: a pickled or
    deep-copied adapter starts with an empty one.

    Parameters
    ----------
    size : int, optional
        The most entries kept, :data:`EVALUATION_CACHE_SIZE` by default; at least
        one.

    Raises
    ------
    ValueError
        If `size` is less than one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.physdata._optmat import EvaluationCache
    >>> cache = EvaluationCache(size=2)
    >>> len(cache), cache.size
    (0, 2)
    """

    __slots__ = ("_entries", "_lock", "size")

    def __init__(self, size: int = EVALUATION_CACHE_SIZE) -> None:
        """Start empty.

        Parameters
        ----------
        size : int, optional
            The most entries kept; at least one.

        Raises
        ------
        ValueError
            If `size` is less than one.
        """
        if size < 1:
            msg = f"an evaluation cache keeps at least one entry, not {size}"
            raise ValueError(msg)
        self.size = size
        self._entries: OrderedDict[
            Hashable,
            tuple[tuple[np.ndarray, ...], tuple[warnings.WarningMessage, ...]],
        ] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Return the number of entries held.

        Returns
        -------
        int
            At most :attr:`size`.
        """
        return len(self._entries)

    def __repr__(self) -> str:
        """Return the size and occupancy.

        Returns
        -------
        str
            For example ``EvaluationCache(2 of 8)``.
        """
        return f"EvaluationCache({len(self)} of {self.size})"

    def __reduce__(self) -> tuple[type["EvaluationCache"], tuple[int]]:  # noqa: UP037 - the hook resolves class names eagerly
        """Pickle and copy as an empty cache of the same size.

        Returns
        -------
        tuple
            The class and its size.
        """
        return (type(self), (self.size,))

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._entries.clear()

    def evaluate(
        self,
        quantity: str,
        omega: Float[np.ndarray, "*shape"],
        model: object,
        *,
        material: str,
        wavelength: Float[np.ndarray, "*shape"],
        compute: Callable[[], tuple[np.ndarray, ...]],
    ) -> tuple[np.ndarray, ...]:
        """Return an evaluation at `omega`, computing it only if it is not held.

        Parameters
        ----------
        quantity : str
            What `compute` evaluates, the key's first part and the messages' word.
        omega : numpy.ndarray
            The validated angular frequencies in rad/s, the key's second part.
        model : OptmatModel
            The model `compute` evaluates.
        material : str
            The material, for messages.
        wavelength : numpy.ndarray
            The vacuum wavelengths of `omega` in m, for range messages.
        compute : callable
            Evaluates optmat and returns its arrays; called under
            :func:`translated`'s policy.

        Returns
        -------
        tuple of numpy.ndarray
            Fresh, writeable copies of the arrays `compute` returns.

        Raises
        ------
        MaterialError
            If optmat refuses the evaluation; nothing is stored.

        Warns
        -----
        MaterialRangeWarning
            As :func:`translated`, at every use whether stored or not.
        """
        values = np.asarray(omega)
        key = (quantity, values.shape, values.dtype.str, values.tobytes())
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                self._entries.move_to_end(key)
        if found is None:
            with _captured(model, quantity, material=material) as caught:
                computed = compute()
            arrays = tuple(np.array(array, copy=True) for array in computed)
            for array in arrays:
                array.flags.writeable = False
            found = (arrays, tuple(caught))
            with self._lock:
                self._entries[key] = found
                self._entries.move_to_end(key)
                while len(self._entries) > self.size:
                    self._entries.popitem(last=False)
        arrays, warned = found
        if warned:
            with material_context():
                _reemit(warned, model, material, wavelength)
        return tuple(np.array(array, copy=True) for array in arrays)


def _reemit(
    caught: Sequence[warnings.WarningMessage],
    model: object,
    material: str,
    wavelength: Float[np.ndarray, "*shape"] | None,
) -> None:
    """Raise captured optmat warnings as solprop's, once each per context.

    Parameters
    ----------
    caught : sequence of warnings.WarningMessage
        What the evaluation raised.
    model : OptmatModel
        The model evaluated.
    material : str
        The material, for messages.
    wavelength : numpy.ndarray or None
        The evaluated vacuum wavelengths in m, where known.
    """
    from optmat.exceptions import (  # noqa: PLC0415 - optmat is imported lazily
        ModelAbsorptionWarning,
        ModelRangeWarning,
    )

    entry = record(model)
    for item in caught:
        if issubclass(item.category, ModelRangeWarning):
            _record_range(model, material, wavelength)
            warn_once(
                ("range", entry.digest),
                f"{_range_message(model, material, wavelength)}. optmat: "
                f"{item.message}",
                MaterialRangeWarning,
            )
        elif issubclass(item.category, ModelAbsorptionWarning):
            warn_once(
                ("absorption", entry.digest),
                f"{material}: {item.message}",
                MaterialRangeWarning,
            )
        else:
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )


def composition_fractions(name: str) -> Mapping[str, float]:
    """Return a named optmat composition's species and mole fractions.

    Parameters
    ----------
    name : str
        A composition optmat names, such as ``"dry air (N2, O2, Ar)"``.

    Returns
    -------
    collections.abc.Mapping
        Gas to mole fraction, in optmat's order, summing to one.

    Raises
    ------
    MaterialError
        If optmat has no composition of that name.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - lazily, see above
    from optmat.gas.composition import composition  # noqa: PLC0415 - as above

    try:
        constituents: tuple[Untyped, ...] = composition(name)
    except OptmatError as error:
        msg = f"optmat has no composition {name!r}: {error}"
        raise MaterialError(msg) from error
    return {c.material: float(c.mole_fraction) for c in constituents}


def record_dropped(
    model: object,
    bound: float,
    extinction: float,
    absorption: float,
    wavelength: float,
) -> None:
    """Record, in every open context, the loss a transparent index dropped.

    Parameters
    ----------
    model : OptmatModel
        The solid model whose imaginary index was dropped.
    bound : float
        The transparent approximation's absorption bound in 1/m.
    extinction : float
        The largest ``k`` dropped, dimensionless.
    absorption : float
        The largest ``4 pi k / lambda`` dropped, 1/m.
    wavelength : float
        The vacuum wavelength in m of that absorption, or NaN.
    """
    logs = _LOGS.get()
    if not logs:
        return
    key = (record(model).digest, bound)
    for log in logs:
        log.record_dropped(key, extinction, absorption, wavelength)


def record_outside(model: object, material: str, parts: Sequence[str]) -> None:
    """Record, in every open context, parts of an evaluation outside a model's range.

    For a range the model warns about at every evaluation, which an adapter records
    rather than emits (an ionisation rate tabulated above its maximum field), or one
    that is not of wavelength (a Raman response's densities and temperature).

    Parameters
    ----------
    model : OptmatModel
        The model evaluated.
    material : str
        The material it serves.
    parts : sequence of str
        The phrases naming what lay outside.
    """
    logs = _LOGS.get()
    if not logs:
        return
    entry = record(model)
    for log in logs:
        log.record_range(entry.digest, material, entry.name, parts)
