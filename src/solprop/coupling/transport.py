# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transport a spatial field through an ordered sequence of linear elements (SEP 31).

:func:`transport` applies spatial and spectral elements in the order given and keeps
a sequential ledger in joules (SEP 31 §4.7). A frequency-only block -- an
:class:`solprop.optics.Chain` or any known :class:`solprop.optics.TransferFunction`
-- may appear directly in the sequence; it is adapted by
:class:`solprop.coupling.elements.SpectralTransfer`, reported at its position, and
never moved past a spatial element. Losses add in joules and the per-frequency
energy spectra stay in every report; broadband efficiencies are never multiplied.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from solprop.coupling.elements import (
    Element,
    ElementReport,
    SpatialState,
    SpectralTransfer,
)
from solprop.exceptions import GridError
from solprop.optics import TransferFunction
from solprop.spatial import SpatialFlux, spatial_energy

__all__ = ["TransportReport", "TransportResult", "transport"]


@dataclass(frozen=True, eq=False)
class TransportReport:
    """The sequential ledger of a transport.

    Every element's input energy is the previous element's output energy, so
    ``energy_out = energy_in - specified_loss - removed + approximation_error``
    holds for the whole sequence as for each element, to the rounding of the sums.

    Attributes
    ----------
    elements : tuple of ElementReport
        One report per element, in order, each with its position.
    energy_in, energy_out : float
        Energy crossing the first input and the last output plane, J.
    specified_loss : float
        Sum of the declared losses, J.
    removed : float
        Sum of the intentional removals, J.
    approximation_error : float
        Sum of the signed approximation errors, J.
    numerical_error : float
        Sum of the numerical error estimates, J; a conservative composition.
    """

    elements: tuple[ElementReport, ...]
    energy_in: float
    energy_out: float
    specified_loss: float
    removed: float
    approximation_error: float
    numerical_error: float

    @property
    def ledger_residual(self) -> float:
        """Return the part of the energy change the ledger does not account for.

        Returns
        -------
        float
            ``energy_out - (energy_in - specified_loss - removed +
            approximation_error)`` in J; rounding only.
        """
        explained = self.energy_in - self.specified_loss - self.removed
        return self.energy_out - (explained + self.approximation_error)


@dataclass(frozen=True, eq=False)
class TransportResult:
    """The transported plane and its ledger.

    Attributes
    ----------
    state : SpatialState
        The field at the last element's output plane.
    report : TransportReport
        The sequential accounting.
    """

    state: SpatialState
    report: TransportReport


def _as_element(item: object, position: int) -> Element:
    """Return an element, adapting a frequency-only transfer function.

    Parameters
    ----------
    item : object
        An element, or an :class:`solprop.optics.TransferFunction` such as a chain.
    position : int
        Its position, for messages.

    Returns
    -------
    Element
        The element itself, or its :class:`SpectralTransfer` adapter.

    Raises
    ------
    GridError
        If the item is neither.
    """
    if isinstance(item, Element):
        return item
    if isinstance(item, TransferFunction):
        return SpectralTransfer(item)
    msg = (
        f"item {position} of a transport sequence is neither an element nor a "
        f"transfer function: {type(item).__name__}"
    )
    raise GridError(msg)


def transport(state: SpatialState, elements: Sequence[object]) -> TransportResult:
    """Apply an ordered sequence of linear elements to a field plane.

    Parameters
    ----------
    state : solprop.coupling.elements.SpatialState
        The field at the first plane.
    elements : sequence
        Elements of :mod:`solprop.coupling.elements`, applied in order. A
        :class:`solprop.optics.Chain` or another known transfer function is one
        frequency-only block at its position; an unknown callable needs an explicit
        :class:`solprop.coupling.elements.SpectralTransfer` with its calibration.

    Returns
    -------
    TransportResult
        The last plane and the sequential ledger.

    Raises
    ------
    GridError
        If an item is neither an element nor a transfer function.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.coupling.elements import FreeSpace, SpatialState, ThinLens
    >>> from solprop.coupling.transport import transport
    >>> from solprop.grid import EnvelopeGrid, RadialGrid
    >>> from solprop.pulses import GaussianPulse, SpatialPulse
    >>> from solprop.spatial import SpatialMedium
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    >>> space, air = RadialGrid(4e-3, 128), SpatialMedium()
    >>> beam = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15,
    ...                                   energy=1e-6), 1e-3)
    >>> plane = SpatialState(grid, space, beam.build(grid, space, medium=air), air)
    >>> result = transport(plane, [ThinLens(1.0, model_error=1e-3),
    ...                            FreeSpace(0.5, air)])
    >>> f"{result.report.energy_out:.4e} J"
    '1.0000e-06 J'
    """
    current = state
    reports: list[ElementReport] = []
    for position, item in enumerate(elements):
        applied = _as_element(item, position).apply(current)
        reports.append(replace(applied.report, position=position))
        current = applied.state
    if reports:
        energy_in = reports[0].energy_in
        energy_out = reports[-1].energy_out
    else:
        flux = SpatialFlux.resolve(state.grid, state.space, state.medium)
        energy_in = energy_out = spatial_energy(state.spectrum, flux)
    report = TransportReport(
        elements=tuple(reports),
        energy_in=energy_in,
        energy_out=energy_out,
        specified_loss=sum(item.specified_loss for item in reports),
        removed=sum(item.removed for item in reports),
        approximation_error=sum(item.approximation_error for item in reports),
        numerical_error=sum(item.numerical_error for item in reports),
    )
    return TransportResult(current, report)
