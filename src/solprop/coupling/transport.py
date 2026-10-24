# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Transport a field plane through an ordered sequence of linear elements (SEP 31).

:func:`transport` takes a radial or Cartesian
:class:`~solprop.coupling.plane.FieldPlane`, applies spatial and spectral elements in
the order given and returns the plane at the last element's output with a sequential
ledger in joules (SEP 31 §4.7). A frequency-only block -- an
:class:`solprop.optics.Chain` or any known :class:`solprop.optics.TransferFunction` --
may appear directly in the sequence; it is adapted by
:class:`solprop.coupling.elements.SpectralTransfer`, reported at its position, and never
moved past a spatial element. Losses add in joules and the per-frequency energy spectra
stay in every report; broadband efficiencies are never multiplied.

The plane's record follows the field. The frame term every
:class:`~solprop.coupling.elements.FreeSpace` removes is added to the plane's
reference, so the absolute spectrum
(:meth:`~solprop.coupling.plane.FieldPlane.absolute`) stays exact over any distance;
a spectral element that drops its own constant phase or group delay leaves that term
of the reference uncalibrated; a mirror turns the plane's axes with its polarisation;
an interface changes its medium; and the power metric is resolved again with SEP 34's
:meth:`solprop.spatial.SpatialFlux.resolve` in the output medium and grid.

The elements themselves act on the plain internal record
:class:`~solprop.coupling.elements.SpatialState`, the documented interface for
writing a new element; :func:`transport` is the one public way to run a sequence.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from solprop.adiabatic import AdiabaticAdmission
from solprop.constants import C
from solprop.coupling.elements import (
    Element,
    ElementReport,
    FreeSpace,
    Mirror,
    SpatialState,
    SpectralTransfer,
)
from solprop.coupling.plane import (
    Calibration,
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneReference,
)
from solprop.exceptions import CouplingError
from solprop.fourier import spectrum_to_internal, spectrum_to_optics
from solprop.grid import TimeGrid
from solprop.linear import Frame
from solprop.optics import TransferFunction
from solprop.spatial import FluxSide, SpatialFlux, SpatialMedium, spatial_energy

__all__ = ["TransportReport", "transport"]

_AXIS_TOLERANCE = 1e-12
"""Agreement required between a declared output axis and the transported one."""


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
    reference_in, reference_out : PlaneReference or None
        The plane's reference before and after.
    frame : str
        Where each free-space frame term came from; empty without a gap.
    source_certificate : solprop.adiabatic.AdiabaticAdmission or None
        The source plane's path certificate (SEP 34 §4.5), carried as its own item:
        it bounds the producing run's linear background path, not this transport.
    """

    elements: tuple[ElementReport, ...]
    energy_in: float
    energy_out: float
    specified_loss: float
    removed: float
    approximation_error: float
    numerical_error: float
    reference_in: PlaneReference | None = None
    reference_out: PlaneReference | None = None
    frame: str = ""
    source_certificate: AdiabaticAdmission | None = None

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
    CouplingError
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
    raise CouplingError(msg)


def _frame_term(
    frame: Frame | None, medium: SpatialMedium, grid: TimeGrid, distance: float
) -> tuple[float, float, str]:
    r"""Return the frame phase and delay a gap removes, and how they were chosen.

    With a frame, the gap removes :math:`RL` with :math:`R=\beta_0+\beta_1
    (\omega-\omega_o)` of the gap's medium at the frame's reference frequency (its
    fixed values where it has them), the term a run's linear operator removes
    (:class:`solprop.linear.Frame`): :math:`\beta_0` only on an envelope grid that
    subtracts it, and :math:`\omega_o` the reference frequency on an envelope grid,
    zero on a field grid. In the reference this is the phase
    :math:`(\beta_0-\omega_o\beta_1)L` and the delay :math:`\beta_1L`, as in
    :func:`solprop.coupling.plane.reference_at`. Without a frame only the delay
    :math:`\beta_1(\omega_0)L` at the grid's reference frequency is removed.

    Parameters
    ----------
    frame : solprop.linear.Frame or None
        The plane's frame, or None.
    medium : solprop.spatial.SpatialMedium
        The gap's homogeneous medium.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    distance : float
        The gap's length, m.

    Returns
    -------
    tuple
        The phase in rad, the delay in s, and a description of the source.
    """
    reference = grid.omega0 if frame is None else frame.omega0
    at = np.array([reference])
    index = float(np.real(medium.index(at))[0])
    slope = float(np.real(medium.dindex_domega(at))[0])
    beta1 = (index + reference * slope) / C
    if frame is None:
        return 0.0, beta1 * distance, "beta1(omega0) L of the medium (no frame given)"
    beta1 = beta1 if frame.beta1 is None else frame.beta1
    envelope = grid.kind == "envelope"
    beta0 = 0.0
    if envelope and frame.subtract_beta0:
        beta0 = index * reference / C if frame.beta0 is None else frame.beta0
    origin = reference if envelope else 0.0
    return (beta0 - origin * beta1) * distance, beta1 * distance, "the plane's frame"


def _state(plane: FieldPlane) -> SpatialState:
    """Return the internal spatial state of a radial or Cartesian plane.

    Parameters
    ----------
    plane : FieldPlane
        The plane.

    Returns
    -------
    SpatialState
        Its internal-convention spectrum on its grids, medium and local frame.

    Raises
    ------
    CouplingError
        If the plane is modal or planar, or carries no medium.
    """
    if plane.kind in ("modal", "general_modal"):
        msg = (
            f"a {plane.kind} plane holds guide coefficients, not a field on a "
            "transverse grid: synthesise the guide's exit field with Outcouple "
            "before transporting it"
        )
        raise CouplingError(msg)
    space = plane.basis.space
    if plane.kind not in ("radial", "cartesian") or space is None:
        msg = f"transport takes a radial or Cartesian plane, got a {plane.kind} plane"
        raise CouplingError(msg)
    if plane.medium is None:
        msg = (
            "transport needs the plane's medium: read a saved plane with "
            "field_plane(..., source=simulation), or build it with its medium"
        )
        raise CouplingError(msg)
    geometry = plane.geometry
    frame = np.array([geometry.x_axis, geometry.y_axis, geometry.normal])
    internal = spectrum_to_internal(plane.samples, kind=plane.grid.kind)
    return SpatialState(plane.grid, space, internal, plane.medium, frame)


def _sequence(
    state: SpatialState,
    elements: Sequence[object],
    frame: Frame | None = None,
    *,
    track: bool = False,
) -> tuple[SpatialState, TransportReport]:
    """Apply an ordered sequence to an internal state and keep the ledger.

    Parameters
    ----------
    state : SpatialState
        The field at the first plane.
    elements : sequence
        Elements or transfer functions, applied in order.
    frame : solprop.linear.Frame or None, optional
        The frame of each free-space gap without an explicit ``frame_delay``.
    track : bool, optional
        Whether the caller tracks a reference, so that a gap without an explicit
        ``frame_delay`` takes its frame term from `frame`. Default False: the
        element's own default applies.

    Returns
    -------
    tuple
        The last state and the sequential report.
    """
    current = state
    reports: list[ElementReport] = []
    sources: list[str] = []
    for position, item in enumerate(elements):
        element = _as_element(item, position)
        if track and isinstance(element, FreeSpace) and element.frame_delay is None:
            phase, delay, source = _frame_term(
                frame, element.medium, state.grid, element.distance
            )
            element = dataclasses.replace(element, frame_delay=delay, frame_phase=phase)
            sources.append(f"item {position}: {source}")
        applied = element.apply(current)
        reports.append(dataclasses.replace(applied.report, position=position))
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
        frame="; ".join(sources),
    )
    return current, report


def _reference(
    reference: PlaneReference, reports: Sequence[ElementReport]
) -> PlaneReference:
    """Return the plane's reference after the elements.

    Each free-space gap adds the phase and delay it removed. A spectral element
    that drops its constant phase or its group delay, or does not say, leaves that
    term uncalibrated: the physical element adds what its model omits.

    Parameters
    ----------
    reference : PlaneReference
        The input reference.
    reports : sequence of ElementReport
        The element reports, in order.

    Returns
    -------
    PlaneReference
        The output reference.
    """
    phase, delay = reference.phase, reference.delay
    known = set(reference.calibrated)
    for report in reports:
        phase += float(report.diagnostics.get("phase_removed", 0.0))
        delay += float(report.diagnostics.get("delay_removed", 0.0))
        for record in report.calibration:
            if record.removes_phase is not False:
                known.discard("phase")
            if record.removes_delay is not False:
                known.discard("delay")
    calibrated: tuple[Calibration, ...] = tuple(
        name for name in ("phase", "delay") if name in known
    )
    return PlaneReference(phase, delay, calibrated)


def _geometry(
    plane: FieldPlane,
    state: SpatialState,
    elements: Sequence[object],
    declared: PlaneGeometry | None,
) -> PlaneGeometry:
    """Return the output plane's geometry.

    Without a declaration the plane moves forward by every gap's distance along its
    current normal and keeps its origin otherwise; a gap leaves it away from any
    declared jump (no side); its axes are the state's frame, turned by any mirror.

    Parameters
    ----------
    plane : FieldPlane
        The input plane.
    state : SpatialState
        The output state, whose frame holds the output axes.
    elements : sequence
        The applied elements, in order.
    declared : PlaneGeometry or None
        The caller's output geometry, checked against the output axes.

    Returns
    -------
    PlaneGeometry
        The output geometry.

    Raises
    ------
    CouplingError
        If a declared geometry's normal or x axis is not the transported one.
    """
    row_normal, row_x = state.frame[2], state.frame[0]
    normal = (float(row_normal[0]), float(row_normal[1]), float(row_normal[2]))
    x_axis = (float(row_x[0]), float(row_x[1]), float(row_x[2]))
    if declared is not None:
        if not (
            np.allclose(declared.normal, normal, rtol=0.0, atol=_AXIS_TOLERANCE)
            and np.allclose(declared.x_axis, x_axis, rtol=0.0, atol=_AXIS_TOLERANCE)
        ):
            msg = (
                f"the declared output plane has normal {declared.normal} and x axis "
                f"{declared.x_axis}, but the transported field arrives with normal "
                f"{normal} and x axis {x_axis}"
            )
            raise CouplingError(msg)
        return declared
    geometry = plane.geometry
    z = geometry.z
    origin = np.array(geometry.origin)
    side: FluxSide | None = geometry.side
    direction = np.array(geometry.normal)
    for item in elements:
        if isinstance(item, FreeSpace):
            z += item.distance
            origin = origin + item.distance * direction
            side = None
        elif isinstance(item, Mirror):
            outgoing = np.asarray(item.outgoing, dtype=float)
            direction = outgoing / np.linalg.norm(outgoing)
    return PlaneGeometry(
        z,
        side,
        (float(origin[0]), float(origin[1]), float(origin[2])),
        normal,
        x_axis,
    )


def transport(
    plane: FieldPlane,
    elements: Sequence[object],
    *,
    geometry: PlaneGeometry | None = None,
    frame: Frame | None = None,
) -> tuple[FieldPlane, TransportReport]:
    """Apply an ordered sequence of linear elements to a field plane.

    Parameters
    ----------
    plane : solprop.coupling.FieldPlane
        A radial or Cartesian plane with its medium. A modal plane is refused:
        its guide field is synthesised first by ``Outcouple``.
    elements : sequence
        Elements of :mod:`solprop.coupling.elements`, applied in order. A
        :class:`solprop.optics.Chain` or another known transfer function is one
        frequency-only block at its position; an unknown callable needs an explicit
        :class:`solprop.coupling.elements.SpectralTransfer` with its calibration.
    geometry : solprop.coupling.PlaneGeometry or None, optional
        The output plane, declared in the receiving system's coordinates; its axes
        must be the transported ones. None moves the input plane along its path.
    frame : solprop.linear.Frame or None, optional
        The frame the plane is written in, such as the source simulation's
        ``frame``: every :class:`~solprop.coupling.elements.FreeSpace` without
        its own ``frame_delay`` then removes the affine term the run would have
        removed over its distance. None removes only the delay
        ``beta1(omega0) L`` of the gap's medium, as the report records.

    Returns
    -------
    tuple
        The plane at the last element's output -- reference, geometry, medium,
        metric and provenance updated -- and the
        :class:`~solprop.coupling.transport.TransportReport`.

    Raises
    ------
    CouplingError
        If the plane cannot be transported, its exterior is unknown and a gap
        propagates it, an item is neither an element nor a transfer function, an
        element refuses, or a declared geometry disagrees with the transported
        axes.

    Examples
    --------
    >>> from solprop.coupling import (FieldPlane, FreeSpace, PlaneGeometry,
    ...                               ThinLens, transport)
    >>> from solprop.grid import EnvelopeGrid, RadialGrid
    >>> from solprop.pulses import GaussianPulse, SpatialPulse
    >>> from solprop.spatial import SpatialMedium
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (750e-9, 850e-9), 200e-15)
    >>> air = SpatialMedium()
    >>> beam = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15,
    ...                                   energy=1e-6), 1e-3)
    >>> plane = FieldPlane.from_pulse(beam, grid, PlaneGeometry(0.0),
    ...                               space=RadialGrid(4e-3, 128), medium=air)
    >>> focus, report = transport(plane, [ThinLens(1.0, model_error=1e-3),
    ...                                   FreeSpace(0.5, air)])
    >>> f"{report.energy_out:.4e} J at z = {focus.geometry.z} m"
    '1.0000e-06 J at z = 0.5 m'
    """
    state = _state(plane)
    if plane.exterior == "unknown" and any(
        isinstance(item, FreeSpace) for item in elements
    ):
        msg = (
            "this plane's field outside its samples is unknown, and propagation "
            "depends on the whole field: declare the exterior zero or pad the record"
        )
        raise CouplingError(msg)
    final, report = _sequence(state, elements, frame, track=True)
    reference = _reference(plane.reference, report.elements)
    output = _geometry(plane, final, elements, geometry)
    flux = SpatialFlux.resolve(
        final.grid, final.space, final.medium, z=output.z, side=output.side
    )
    basis = PlaneBasis.of_space(final.space)
    provenance = plane.provenance
    for item in report.elements:
        step = f"transport[{item.position}]: {item.element}"
        if item.reconstruction:
            step += " (reconstruction)"
        provenance = provenance.then(step)
    if report.frame:
        provenance = provenance.then(f"transport frame: {report.frame}")
    result = FieldPlane(
        basis.kind,
        spectrum_to_optics(final.spectrum, kind=final.grid.kind),
        final.grid,
        "V m" if basis.kind == "radial" else "V/m",
        basis,
        reference,
        output,
        PlaneMetric.from_flux(flux),
        plane.exterior,
        final.medium,
        plane.certificate,
        provenance=provenance,
    )
    report = dataclasses.replace(
        report,
        reference_in=plane.reference,
        reference_out=reference,
        source_certificate=plane.certificate,
    )
    return result, report
