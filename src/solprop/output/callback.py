# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The output handler that gives each plane to a callable and keeps nothing.

The handler for a scan that reduces each plane to a few numbers: the callable is the
body of what would otherwise be a hand loop over the backend. The plane reaches it as
the run made it, with no copy and no conversion, so the handler costs no volume of its
own.

It is an :class:`~solprop.output.store.Output` and a
:class:`~solprop.output.store.RetainingOutput`, and deliberately neither a
:class:`~solprop.output.store.PlaneStore` nor a
:class:`~solprop.output.store.CheckpointOutput`. A store's readers have nothing to read
here, and its save path converts each field into a new volume. A checkpoint promises a
state that ``resume=True`` can continue, and with no file there is nothing to continue,
so a run neither fetches a terminal field for this handler nor says that one was
written.

The manual page is ``docs/guide/setup/running.md``.
"""

from collections.abc import Callable

import numpy as np

from solprop._display import line
from solprop.exceptions import OutputError
from solprop.output.metadata import RunMetadata, resolve_metadata_write
from solprop.output.records import Convention, SavedPlane
from solprop.output.store import FieldRetention

__all__ = ["CallbackOutput"]


def _not_held(what: str) -> OutputError:
    """Return the refusal of a read, naming what was asked for."""
    return OutputError(
        f"a CallbackOutput does not hold {what}: it gives each plane to its callable "
        "and keeps nothing. Keep what you need in the callable, or run into a "
        "MemoryOutput or a file to read the run back"
    )


class CallbackOutput:
    """Give each saved plane to a callable, and keep nothing.

    The callable is given the run's own :class:`~solprop.output.records.SavedPlane`, not
    a copy. Its arrays are read-only and nothing writes to them again, so the callable
    may keep any of them, at the cost of what it keeps; a plane it reduces and lets go
    costs one field volume on the host while it is held, the one the observation
    crossed with. The field is as a backend produces it: in the internal FFT convention,
    in the square root of watts. A reduction of the squared modulus does not depend on
    the convention, and :func:`solprop.fourier.spectrum_to_optics` gives the form a file
    stores, in a second volume.

    The run's ``fields`` setting
    (:class:`~solprop.driver.settings.SimulationSettings`) decides what a plane
    carries. With ``fields="none"``, what a reduction of statistics wants, every plane
    has ``field=None`` and ``noise_phase=None`` beside its position, its statistics and
    its step history, and no field leaves the backend. With ``fields="planes"`` the
    planes it does not name are those; with the default ``"all"`` every plane carries
    its field.

    What it cannot do, it refuses by name. Reading anything back (``handler[key]``,
    :meth:`fields`) raises :class:`~solprop.exceptions.OutputError`, and a handler that
    has served a run refuses another, which would continue it. A resume needs a file
    (:func:`~solprop.output.hdf5.check_cache`, :func:`~solprop.output.hdf5.resume`,
    ``resume=True``), as does a TDSE run's ledger. A stopped run returns the handler
    with no state written.

    A run with an optmat ionisation rate reads its peak rates back from its output to
    report the peak field and warn past the rate's range
    (:class:`~solprop.exceptions.MaterialRangeWarning`). This handler stores no
    statistic, so that report and warning do not happen: read the rate in the callable.

    Parameters
    ----------
    receive : collections.abc.Callable
        Called once per saved plane, the entrance first, with the plane. What it returns
        is ignored; what it raises stops the run and propagates, and that plane is not
        counted.

    Raises
    ------
    OutputError
        If `receive` is not callable.

    Examples
    --------
    >>> import dataclasses
    >>> from solprop.driver import Simulation
    >>> from solprop.output import CallbackOutput
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1.2e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech", n_save=3, rtol=1e-6,
    ... )
    >>> simulation.settings = dataclasses.replace(simulation.settings, fields="none")
    >>> kept = []
    >>> handler = CallbackOutput(lambda plane: kept.append((plane.z, plane.field)))
    >>> simulation.run(handler, progress="none") is handler
    True
    >>> kept
    [(0.0, None), (0.005, None), (0.01, None)]
    >>> "z" in handler, handler
    (False, CallbackOutput(3 planes, closed))
    """

    def __init__(self, receive: Callable[[SavedPlane], None]) -> None:
        """Hold the callable, with no run begun."""
        if not callable(receive):
            msg = f"a CallbackOutput needs a callable taking a plane, got {receive!r}"
            raise OutputError(msg)
        self._receive = receive
        self._metadata: RunMetadata | None = None
        self._n_saved = 0
        self._closed = False

    def __repr__(self) -> str:
        """Return a one-line description: how many planes, and whether it is open.

        Returns
        -------
        str
            For example ``CallbackOutput(3 planes, closed)``.
        """
        state = "closed" if self._closed else "open"
        return line(type(self).__name__, [("", f"{self._n_saved} planes"), ("", state)])

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, holding its record.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to replace a different record held already
            (:func:`~solprop.output.metadata.resolve_metadata_write`).

        Raises
        ------
        OutputError
            If the handler has served a run already, or a different record is held and
            `force` is false.
        """
        if self._closed or self._n_saved:
            msg = (
                "this CallbackOutput has served a run and keeps none of its planes, so "
                "no run can continue it; build a new one for each run"
            )
            raise OutputError(msg)
        resolve_metadata_write(self._metadata, metadata, force=force)
        self._metadata = metadata

    def retain(self, retention: FieldRetention, field_shape: tuple[int, ...]) -> None:
        """Accept the run's field rule: there is no storage for it to size.

        Declares :class:`~solprop.output.store.RetainingOutput`, which is what admits
        field-free planes. The rule reaches the callable as each plane's ``field``.

        Parameters
        ----------
        retention : FieldRetention
            Which planes keep their field.
        field_shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the run's state.
        """

    def save(self, plane: SavedPlane) -> None:
        """Give one plane to the callable, and count it once the callable has returned.

        Parameters
        ----------
        plane : SavedPlane
            The plane, handed on as it is.

        Raises
        ------
        OutputError
            If the handler is closed or was never opened.
        """
        if self._closed:
            msg = "this CallbackOutput is closed; build a new one for each run"
            raise OutputError(msg)
        if self._metadata is None:
            raise OutputError("call open(metadata) before saving a plane")
        self._receive(plane)
        self._n_saved += 1

    def close(self) -> None:
        """Finish the run. Calling it twice is harmless."""
        self._closed = True

    @property
    def n_saved(self) -> int:  # numpydoc ignore=RT01
        """How many planes the callable has taken."""
        return self._n_saved

    @property
    def metadata(self) -> RunMetadata | None:  # numpydoc ignore=RT01
        """Record of the run, or ``None`` before it is opened."""
        return self._metadata

    @property
    def stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics stored at every plane: none.

        Empty, so that nothing asks to read one back.
        """
        return ()

    @property
    def step_stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics stored between planes: none."""
        return ()

    def __getitem__(self, key: str) -> np.ndarray:
        """Refuse to read anything back, naming what was asked for.

        Parameters
        ----------
        key : str
            What was asked for.

        Raises
        ------
        OutputError
            Always.
        """
        raise _not_held(repr(key))

    def __contains__(self, key: object) -> bool:
        """Return whether ``self[key]`` would return an array: it never would.

        Parameters
        ----------
        key : object
            A candidate key.

        Returns
        -------
        bool
            ``False``.
        """
        return False

    def fields(self, convention: Convention = "optics") -> np.ndarray:  # noqa: ARG002 - the protocol's signature
        """Refuse to read the fields back.

        Parameters
        ----------
        convention : {'optics', 'fft'}, optional
            The convention that was asked for.

        Raises
        ------
        OutputError
            Always.
        """
        raise _not_held("the fields")
