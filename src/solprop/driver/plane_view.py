# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The lazy view of a saved plane that a plane observer is shown (SEP 41, S3).

A :class:`~solprop.log.PlaneObserver` is shown each saved plane as a
:class:`PlaneView`: the plane's position, statistics row and basis at once, and its
field, spectrum, power and spectral density only on request, computed by the same
:mod:`solprop.processing` functions that read a file, so what an observer draws is
what the file holds. On a device a plane whose field the output does not keep is
fetched only when asked for, and the fetch is counted in the run's resources.
"""

import logging
import time
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from solprop._host import ObserverSpend
from solprop.capabilities import Quantity, _quantities
from solprop.exceptions import DriverError, SolpropWarning
from solprop.log import PlaneObserver, ProgressReporter
from solprop.output.capability import BasisCapability, capability_of
from solprop.output.memory import MemoryOutput
from solprop.output.schema import RunMetadata, SavedPlane, StoredField
from solprop.processing import fields

if TYPE_CHECKING:
    from solprop.coupling.plane import FieldPlane

__all__ = ["PlaneView", "PlaneWatch"]

logger = logging.getLogger(__name__)

_EXPIRED: Final = (
    "this plane view was valid only during observe_plane: copy what you keep"
)


class PlaneView:
    """One saved plane, as a plane observer sees it; valid only during the call.

    Parameters
    ----------
    plane : solprop.output.schema.SavedPlane
        The plane as committed, its field ``None`` where the output keeps none.
    index : int
        Its index among the saved planes.
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    fetch : callable or None
        Returns the plane's field where it was not kept; ``None`` where it was.
    spend : solprop._host.ObserverSpend
        The run's observer ledger, which counts the fetches.
    stored : solprop.output.schema.StoredField or None, optional
        The kept field as the output stored it, where it was exported straight into
        the output (SEP 47 section 5.4) and the plane carries none: the view reads
        that, with no fetch.

    Attributes
    ----------
    z : float
        The plane's position, m.
    index : int
        Its index among the saved planes.
    stats : collections.abc.Mapping
        Its statistics row, by name.
    """

    def __init__(
        self,
        plane: SavedPlane,
        index: int,
        metadata: RunMetadata,
        fetch: Callable[[], np.ndarray] | None,
        spend: ObserverSpend,
        *,
        stored: StoredField | None = None,
    ) -> None:
        """Hold the plane; compute nothing yet."""
        self.z = float(plane.z)
        self.index = index
        self.stats: Mapping[str, np.ndarray] = dict(plane.stats)
        self._plane = plane
        self._metadata = metadata
        self._fetch = fetch
        self._spend = spend
        self._stored = stored
        self._output: MemoryOutput | None = None
        self._open = True

    @property
    def basis(self) -> BasisCapability:
        """Return the run's basis.

        Returns
        -------
        {'modal', 'general_modal', 'spatial'}
            The basis capability its record declares.
        """
        return capability_of(self._metadata)

    @property
    def quantities(self) -> dict[str, Quantity]:
        """Return what the plane's derived quantities are.

        Returns
        -------
        dict
            Whether its power, energy, density and field are physical fluxes.
        """
        return _quantities(self.basis)

    def _reader(self) -> MemoryOutput:
        """Return the plane as a one-plane output, fetching its field once if needed."""
        if not self._open:
            raise DriverError(_EXPIRED)
        if self._output is None:
            plane = self._plane
            if self._stored is not None:
                output = MemoryOutput()
                output.open(self._metadata)
                # A copy of what the output stored, into the view's own room.
                output.save_exported(plane, self._stored)
                self._output = output
                return output
            if plane.field is None and self._fetch is not None:
                self._spend.transfers += 1
                plane = SavedPlane(
                    z=plane.z,
                    field=self._fetch(),
                    stats=plane.stats,
                    noise_phase=plane.noise_phase,
                )
            output = MemoryOutput()
            output.open(self._metadata)
            output.save(plane)
            self._output = output
        return self._output

    def spectrum(self, **keywords: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane's spectrum, as the processing function returns it.

        The function is :func:`solprop.processing.fields.get_spectrum`.

        Parameters
        ----------
        **keywords : Any
            Its keywords, the plane excepted.

        Returns
        -------
        tuple
            The angular frequency axis and the spectrum.
        """
        return fields.get_spectrum(self._reader(), plane=0, **keywords)

    def field(self, **keywords: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane's field in time, as the processing function returns it.

        The function is :func:`solprop.processing.fields.get_field`.

        Parameters
        ----------
        **keywords : Any
            Its keywords, the plane excepted.

        Returns
        -------
        tuple
            The time axis and the field.
        """
        return fields.get_field(self._reader(), plane=0, **keywords)

    def power(self, **keywords: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane's power in time, as the processing function returns it.

        The function is :func:`solprop.processing.fields.get_power`.

        Parameters
        ----------
        **keywords : Any
            Its keywords, the plane excepted.

        Returns
        -------
        tuple
            The time axis and the power, W.

        Raises
        ------
        solprop.exceptions.OutputError
            On a spatial run, whose columns the function would read as modes:
            :func:`solprop.spatial.spatial_power` is the beam's power.
        """
        return fields.get_power(self._reader(), plane=0, **keywords)

    def spectral_density(self, **keywords: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane's spectral density, as the processing function does.

        Parameters
        ----------
        **keywords : Any
            The keywords of :func:`solprop.processing.fields.get_spectral_density`, the
            plane excepted.

        Returns
        -------
        tuple
            The axis and the density.

        Raises
        ------
        solprop.exceptions.OutputError
            On a spatial run, whose columns the function would read as modes.
        """
        return fields.get_spectral_density(self._reader(), plane=0, **keywords)

    def stored(self) -> np.ndarray:
        """Return the plane's stored spectrum, as a file holds it.

        The raw forward DFT in the internal convention, shape
        ``(n_modes, n_pol, n_omega)`` -- what ``output.fields("fft")`` returns for the
        plane, and what the functionals of :mod:`solprop.spatial` (``spatial_power``,
        ``spatial_fluence`` and the rest) take with the run's ``SpatialFlux``, which is
        how a spatial run's plane is measured while it runs.

        Returns
        -------
        numpy.ndarray
            A copy of the stored spectrum.
        """
        return np.array(self._reader().fields("fft")[0])

    def field_plane(self) -> "FieldPlane":
        """Return the plane as :func:`solprop.processing.planes.field_plane` reads it.

        Its ``spectral_energy`` is a spatial plane's energy per frequency bin, where
        :meth:`spectral_density` refuses the columns of a spatial run.

        Returns
        -------
        solprop.coupling.FieldPlane
            The plane.

        Raises
        ------
        solprop.exceptions.OutputError
            For a run :func:`~solprop.processing.planes.field_plane` refuses (a hybrid
            planar run, the modified noise model).
        """
        from solprop.processing.planes import field_plane  # noqa: PLC0415 - heavy

        return field_plane(self._reader(), 0)

    def _close(self) -> None:
        """End the view's validity and let its arrays go."""
        self._open = False
        self._output = None


class PlaneWatch:
    """Show a run's saved planes to its reporter, when it is a plane observer.

    Parameters
    ----------
    reporter : solprop.log.ProgressReporter
        The run's reporter; only a :class:`~solprop.log.PlaneObserver` is shown planes.
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    fetch : callable
        Returns the field at the current plane, for a plane the output keeps none of.
    spend : solprop._host.ObserverSpend
        The run's observer ledger.
    """

    def __init__(
        self,
        reporter: ProgressReporter,
        metadata: RunMetadata,
        fetch: Callable[[], np.ndarray],
        spend: ObserverSpend,
    ) -> None:
        """Hold the observer, or nothing for a reporter that is not one."""
        self._observer = reporter if isinstance(reporter, PlaneObserver) else None
        self._metadata = metadata
        self._fetch = fetch
        self._spend = spend

    def show(
        self, plane: SavedPlane, index: int, stored: StoredField | None = None
    ) -> None:
        """Show one committed plane, detaching an observer that fails.

        Parameters
        ----------
        plane : solprop.output.schema.SavedPlane
            The plane as committed.
        index : int
            Its index among the saved planes.
        stored : solprop.output.schema.StoredField or None, optional
            Its kept field as the output stored it, where it was exported straight
            into the output and the plane carries none.

        Warns
        -----
        SolpropWarning
            When the observer raised, which then sees no more planes in this run.
        """
        observer = self._observer
        if observer is None:
            return
        view = PlaneView(
            plane,
            index,
            self._metadata,
            self._fetch if plane.field is None and stored is None else None,
            self._spend,
            stored=stored,
        )
        started = time.perf_counter()
        try:
            observer.observe_plane(plane.z, view)
        except Exception as error:
            # Not a stop, an interrupt or a signal: those are BaseExceptions and stop
            # the run as they would anywhere.
            self._observer = None
            logger.exception("the plane observer failed at z = %.6e m", plane.z)
            msg = (
                f"the plane observer raised {type(error).__name__}: {error}; it is "
                "shown no more planes in this run, which continues"
            )
            warnings.warn(msg, SolpropWarning, stacklevel=2)
        finally:
            view._close()
            self._spend.calls += 1
            self._spend.seconds += time.perf_counter() - started
