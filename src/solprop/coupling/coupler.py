# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Couple a field plane into a simulation's input port (SEP 31, sections 4.3 and 4.7).

:func:`couple` is the one public path from a :class:`~solprop.coupling.plane.FieldPlane`
to a target-bound :class:`~solprop.coupling.port.CouplingLaunch`. It applies one of
the three models of :class:`~solprop.coupling.models.CouplingModel` and returns a
:class:`CouplingResult`: the launch and an immutable :class:`CouplingReport`.

**Measures.** Three quadratic forms are kept apart (section 4.3): ``F``, the
Hermitian E/H flux pairing of a modal basis; ``G = S^H M S``, the scalar
least-squares Gram in the common paraxial measure :math:`M_0 = Y_0\,dA/2`,
:math:`Y_0 = n_{\rm ref}\varepsilon_0c`; and ``Q``, a representation's native power
metric. Only in the common positive metric ``M_0`` does the orthogonal-residual
identity :math:`\|f\|^2_M = a^\dagger Ga + \|f - Sa\|^2_M` hold, so only the scalar
model reports an uncaptured energy; the others report the incident and launched
energies and leave the difference unattributed.

**Overlaps in the plane's own quadrature.** A spatial plane's columns are the
Hankel or Fourier spectrum of its field, so the scalar overlap of a target profile is
taken by Parseval in that spectral quadrature, with the profile's closed-form
transform (:meth:`~solprop.coupling.models.ScalarModeBasis.hankel`): the incident
norm, the overlaps and the Gram all use one quadrature, and the identity above holds
to rounding. A modal source is overlapped on a composite Gauss--Legendre rule that
breaks at both core radii.

**Admission** (section 4.3). Each approximate coupling is admitted against its
``model_error`` budget :math:`\epsilon_m` through a list of
:class:`AdmissionItem` s, each a relative energy discrepancy:

- ``angular``: :math:`[d(E_{\rm in}) + d_{\rm target}]/U` with
  :math:`1 - \cos\theta_* = \epsilon_m/8` and
  :math:`d = (1 - \cos\theta_*)U_{\rm cone} + U_{\rm tail}` (section 4.6), for the
  incident field and for the retained target subspace. The cone term is the cone
  rule's **allocation**, about :math:`\epsilon_m/8` of the cone share, not a
  measured defect; only the tail beyond :math:`\theta_*` is measured;
- ``contrast``: the weak-guidance screen
  :math:`\Delta \le \epsilon_m/8` of a solid provider, or the recorded pair
  envelope that replaces it; not applied to a capillary wall;
- ``target metric`` and ``source metric``: the relative defect between ``G`` (or
  ``F``) and the target's native ``Q``, and between ``M_0`` and the source's native
  metric, each at most :math:`\epsilon_m/8`;
- ``closure`` (vector model): :math:`2\delta + \delta^2` of the H-closure field error
  :math:`\delta`, at most :math:`\epsilon_m/8`;
- ``passivity``: the whitened certificate :math:`C^\dagger Q_BC \preceq Q_A`; an
  apparent gain within the model's own defects is reported as a model defect, never
  rescaled or clipped, and a larger one refuses;
- ``envelope``: the recorded pair envelope's allowance (scaled by the pair's leaky
  parameter for a capillary, by the fibre's contrast for a beam into a step-index
  fibre, by the squared contrast for the vector model into a step-index guide),
  plus the incident share of any bins outside its range; a general guide without
  a recorded envelope (a graded index) reports it without a value, uncounted;
- ``numerical``: quadrature, representation and domain errors, an amplitude error
  :math:`\delta` entered as :math:`2\delta + \delta^2`.

The counted items must sum to at most :math:`\epsilon_m`. A source plane's SEP 34 path
certificate is carried separately in the report. A zero field is a
valid zero launch with an undefined efficiency.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from jaxtyping import Complex, Float, Inexact, Int
from scipy.special import j0

from solprop import fourier
from solprop._display import Displayed, engineering, line
from solprop.adiabatic import AdiabaticAdmission
from solprop.constants import EPS0, C
from solprop.coupling.envelopes import (
    CAPILLARY_ENVELOPE,
    STEP_INDEX_BEAM_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    VECTOR_STEP_INDEX_ENVELOPE,
    PairEnvelope,
)
from solprop.coupling.models import CouplingModel, ScalarModeBasis, radial_rule
from solprop.coupling.plane import FieldPlane, PlaneMetric, lattice_identity
from solprop.coupling.port import (
    METRIC_TOLERANCE,
    CouplingLaunch,
    InputPort,
    check_same_medium,
    occupied_outside,
)
from solprop.exceptions import CouplingError
from solprop.grid import FreeGrid, RadialGrid

__all__ = [
    "PASSIVITY_ROUNDING",
    "AdmissionItem",
    "CouplingReport",
    "CouplingResult",
    "ItemScope",
    "couple",
    "target_columns",
]

PASSIVITY_ROUNDING = 1e-10
"""Apparent gain of a whitened map below which it is rounding, not a model defect.

The certificate's largest squared singular value is formed from sums over at most
a few thousand columns of terms within a few ulp of each other, then whitened by a
Cholesky factor of a Gram whose condition is checked; 1e-10 is well above that
rounding and well below any model defect worth naming.
"""

_OCCUPIED = 1e-14
"""Share of the peak bin's energy below which a bin is not occupied.

The per-bin admission items are maxima over occupied bins; a bin carrying less than
this share of the peak carries less energy than the rounding of the total.
"""

_TARGET_BINS = 5
"""Occupied bins at which a general guide's angular content is evaluated.

The node Fourier transform of every channel at every bin would cost nodes times
cone samples times bins; the content varies smoothly with frequency, so the two
band edges, the peak bin and two between them bound it.
"""

_PHASE_BLOCK = 1 << 22
"""Entries of the cone by node phase matrix formed at once (64 MiB of complex).

A fine node table and a wide cone make the whole matrix several GiB.
"""

_SCALE = 0.5 * EPS0 * C
"""``eps0 c / 2``: joules per weighted ``|E|^2`` sum of a field in V/m."""

ItemScope = Literal["field", "map", "model"]
"""What an admission item bounds: this field, every input of the map, or the model."""


@dataclass(frozen=True)
class AdmissionItem:
    r"""One term of a coupling's admission budget (SEP 31, section 4.3).

    Attributes
    ----------
    name : str
        ``angular``, ``contrast``, ``target metric``, ``source metric``,
        ``closure``, ``passivity``, ``envelope`` or ``numerical``.
    value : float or None
        The relative energy discrepancy it bounds, dimensionless; None where the
        item does not apply (`note` says why).
    allowance : float or None
        Its own cap, where the model gives it one (:math:`\epsilon_m/8`); None
        when only the total budget limits it.
    counted : bool
        Whether `value` is part of the budget's sum. A screen (``contrast``)
        bounds a model's validity, not an energy, and is not counted.
    scope : {'field', 'map', 'model'}
        Whether it bounds this field only, every input of the map, or the model.
    note : str, optional
        What it measured, or why it does not apply.
    """

    name: str
    value: float | None
    allowance: float | None
    counted: bool
    scope: ItemScope
    note: str = ""

    @property
    def passed(self) -> bool:
        """Return whether the item is within its own cap.

        Returns
        -------
        bool
            True when it has no value or no cap, or its value is at most the cap.
        """
        return (
            self.value is None or self.allowance is None or self.value <= self.allowance
        )


@dataclass(frozen=True, eq=False)
class CouplingReport(Displayed):
    r"""The structured account of one coupling (SEP 31, sections 4.7 and 5.3).

    Energies are per bin, J, in the measure each names; the totals are their sums.

    Attributes
    ----------
    model : str
        The coupling model.
    model_error : float or None
        Its budget; None for ``same_basis``.
    reference_index : numpy.ndarray
        :math:`n_{\rm ref}` of the common measure at each bin, shape
        ``(n_omega,)``; zero where it is not defined.
    incident : numpy.ndarray
        The incident field's energy over its whole plane, shape ``(n_omega,)``: in
        the common measure :math:`M_0` for the approximate models, the plane's own
        metric for ``same_basis``.
    launched : numpy.ndarray
        The launch's energy in the target's native metric, shape ``(n_omega,)``:
        what the receiving run's entrance plane carries.
    captured : numpy.ndarray or None
        The captured energy :math:`a^\dagger Ga` in the common metric, shape
        ``(n_omega,)``; None where the model has no common positive metric.
    captured_reason : str
        Why `captured` is None; empty otherwise.
    items : tuple of AdmissionItem
        The admission budget's terms.
    exterior : str
        What the source plane says beyond its record: ``'zero'`` or ``'unknown'``
        (unobserved tails are unknown, not lost).
    certificate : solprop.adiabatic.AdiabaticAdmission or None
        The source plane's SEP 34 path certificate, carried separately: its scope
        is the linear background's forward field, so it bounds the source's
        background path, not the coupling.
    source : collections.abc.Mapping
        The source plane's provenance.
    port : str
        The target port's identity.
    """

    model: str
    model_error: float | None
    reference_index: Float[np.ndarray, " n_omega"]
    incident: Float[np.ndarray, " n_omega"]
    launched: Float[np.ndarray, " n_omega"]
    captured: Float[np.ndarray, " n_omega"] | None
    captured_reason: str
    items: tuple[AdmissionItem, ...]
    exterior: str
    certificate: AdiabaticAdmission | None
    source: Mapping[str, object]
    port: str

    def __post_init__(self) -> None:
        """Own read-only arrays and mappings, checked against one another."""
        n_omega = np.shape(self.incident)
        for name in ("reference_index", "incident", "launched", "captured"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != n_omega or not np.all(np.isfinite(array)):
                msg = f"a coupling report's {name} is finite, one value per bin"
                raise CouplingError(msg)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))

    @property
    def uncaptured(self) -> Float[np.ndarray, " n_omega"] | None:
        """Return the incident energy each bin leaves outside the target, not clipped.

        Returns
        -------
        numpy.ndarray or None
            ``incident - captured``, J, in the common positive metric where the
            orthogonal-residual identity makes it a nonnegative physical share;
            None where `captured` is None.
        """
        return None if self.captured is None else self.incident - self.captured

    @property
    def incident_energy(self) -> float:
        """Return the whole incident energy.

        Returns
        -------
        float
            J.
        """
        return float(np.sum(self.incident))

    @property
    def launched_energy(self) -> float:
        """Return the energy the launch carries into the target.

        Returns
        -------
        float
            J.
        """
        return float(np.sum(self.launched))

    @property
    def captured_energy(self) -> float | None:
        """Return the captured energy in the common metric.

        Returns
        -------
        float or None
            J; None where it is not defined.
        """
        return None if self.captured is None else float(np.sum(self.captured))

    @property
    def uncaptured_energy(self) -> float | None:
        """Return the uncaptured energy in the common metric, not clipped.

        Returns
        -------
        float or None
            J; None where it is not defined.
        """
        uncaptured = self.uncaptured
        return None if uncaptured is None else float(np.sum(uncaptured))

    @property
    def efficiency(self) -> float | None:
        """Return the launched energy as a share of the incident energy.

        Returns
        -------
        float or None
            Dimensionless; None when nothing is incident, where it is undefined.
        """
        incident = self.incident_energy
        return None if incident == 0.0 else self.launched_energy / incident

    @property
    def budget(self) -> float:
        """Return the sum of the counted admission items.

        Returns
        -------
        float
            Dimensionless; at most `model_error` for an admitted coupling.
        """
        return float(
            sum(item.value for item in self.items if item.counted and item.value)
        )

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``CouplingReport(scalar_paraxial, 1 uJ incident, 906 nJ
            launched)``.
        """
        return line(
            "CouplingReport",
            [
                ("", self.model),
                ("", f"{engineering(self.incident_energy, 'J')} incident"),
                ("", f"{engineering(self.launched_energy, 'J')} launched"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        efficiency = self.efficiency
        rows: list[tuple[str, object]] = [
            ("model", self.model),
            ("incident", engineering(self.incident_energy, "J")),
            ("launched", engineering(self.launched_energy, "J")),
            (
                "launched share",
                "undefined (nothing incident)"
                if efficiency is None
                else f"{efficiency:.4f} of the incident energy",
            ),
        ]
        uncaptured = self.uncaptured_energy
        rows.append(
            (
                "uncaptured",
                engineering(uncaptured, "J")
                if uncaptured is not None
                else "not defined",
            )
        )
        if self.model_error is not None:
            rows.append(
                ("budget", f"{self.budget:.3e} of model_error {self.model_error:g}")
            )
        for item in self.items:
            value = "n/a" if item.value is None else f"{item.value:.3e}"
            rows.append((item.name, f"{value} ({item.scope}) {item.note}".rstrip()))
        rows.append(("exterior", self.exterior))
        if self.certificate is not None:
            certificate = self.certificate
            rows.append(
                (
                    "path certificate",
                    f"field error {certificate.field_error:.3e}, reflection "
                    f"{certificate.reflection:.3e} (source's linear background path)",
                )
            )
        return "CouplingReport: a field coupled into an input port", rows


@dataclass(frozen=True, eq=False)
class CouplingResult:
    """A launch bound to its port, and the account of how it was computed.

    Attributes
    ----------
    launch : solprop.coupling.port.CouplingLaunch
        What :meth:`solprop.driver.simulation.Simulation.with_input` takes.
    report : CouplingReport
        The energies, the admission budget and the provenance.
    """

    launch: CouplingLaunch
    report: CouplingReport


def couple(
    plane: FieldPlane, port: InputPort, *, model: CouplingModel
) -> CouplingResult:
    r"""Couple a field plane into an input port under a declared model.

    SEP 31 section 4.3. The plane must already be on the port's temporal lattice
    (:func:`~solprop.coupling.remap.regrid` moves it), at an identically oriented
    plane, and inside the port's active band. Every pair the model is not
    qualified for is refused by name; nothing restores energy the target does not
    take, and nothing is rescaled.

    ``same_basis``
        Exact recovery: a copy for an identical basis; a declared
        :class:`~solprop.coupling.models.BasisChange`; or, for a general modal
        plane carrying E/H at the port's own nodes, :math:`a = F^{-1}B`.
    ``scalar_paraxial``
        A radial or Cartesian plane into a capillary port or a weakly guiding
        step-index fibre's, a capillary plane into another capillary (any radii and
        mode sets), or a weakly guiding step-index fibre's plane into another's,
        within the recorded envelopes.
    ``vector_paraxial``
        A radial or Cartesian plane, or an ordinary modal plane with its profiles,
        into a general modal guide that carries E/H at fixed nodes, with the
        forward closure :math:`H_t = Y_0\hat z\times E_t`.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        The receiving port.
    model : solprop.coupling.models.CouplingModel
        The model and its budget.

    Returns
    -------
    CouplingResult
        The launch, bound to `port`, and its report.

    Raises
    ------
    CouplingError
        If the port does not admit the model, the lattice, orientation or band
        differ, the pair is not one the model is qualified for, or an admission
        item or the total budget is exceeded.
    """
    if model.kind not in port.models:
        msg = f"this port admits {port.models}, not {model.kind}"
        raise CouplingError(msg)
    _check_common(plane, port)
    if model.kind == "same_basis":
        samples, report = _same_basis(plane, port, model)
    elif model.kind == "scalar_paraxial":
        samples, report = _scalar_paraxial(plane, port, model)
    else:
        samples, report = _vector_paraxial(plane, port, model)
    provenance = {
        "plane": plane.provenance.record(),
        "reference": plane.reference.record(),
    }
    launch = CouplingLaunch(
        port, samples, plane.reference.minus(port.frame), model.kind, provenance
    )
    return CouplingResult(launch, report)


def _check_common(plane: FieldPlane, port: InputPort) -> None:
    """Refuse a plane on another lattice, orientation or outside the port's band.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        The receiving port.

    Raises
    ------
    CouplingError
        Naming the disagreement.
    """
    if lattice_identity(plane.grid) != port.lattice or not np.array_equal(
        plane.grid.omega, port.grid.omega
    ):
        msg = "the plane's temporal lattice is not the port's: regrid it first"
        raise CouplingError(msg)
    if not plane.geometry.oriented_as(port.geometry):
        msg = "the plane and the entrance are oriented differently"
        raise CouplingError(msg)
    outside = occupied_outside(plane.samples, port.grid)
    floor = (plane.grid.n_omega * float(np.finfo(float).eps)) ** 2
    if outside > floor:
        msg = (
            f"{outside:.3e} of the plane lies outside the port's active band, which "
            "the receiving run would remove: regrid with crop_band=True to remove "
            "it explicitly"
        )
        raise CouplingError(msg)


def _report(  # noqa: PLR0913 - the account's terms, each named at the call
    plane: FieldPlane,
    port: InputPort,
    model: CouplingModel,
    *,
    index: Float[np.ndarray, " n_omega"],
    incident: Float[np.ndarray, " n_omega"],
    launched: Float[np.ndarray, " n_omega"],
    captured: Float[np.ndarray, " n_omega"] | None,
    captured_reason: str = "",
    items: tuple[AdmissionItem, ...] = (),
) -> CouplingReport:
    """Assemble a report, refusing a coupling whose budget is exceeded.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        The receiving port.
    model : solprop.coupling.models.CouplingModel
        The model.
    index : numpy.ndarray
        The common measure's reference index per bin.
    incident, launched : numpy.ndarray
        Per-bin energies, J.
    captured : numpy.ndarray or None
        Per-bin captured energy in the common metric, J.
    captured_reason : str, optional
        Why `captured` is None.
    items : tuple of AdmissionItem, optional
        The admission terms.

    Returns
    -------
    CouplingReport
        The report.

    Raises
    ------
    CouplingError
        If an item exceeds its own cap or the counted items exceed `model_error`.
    """
    report = CouplingReport(
        model.kind,
        model.model_error,
        index,
        incident,
        launched,
        captured,
        captured_reason,
        items,
        plane.exterior,
        plane.certificate,
        plane.provenance.record(),
        port.identity,
    )
    failed = [item for item in items if not item.passed]
    if failed:
        named = ", ".join(
            f"{item.name} {cast('float', item.value):.3e} > {item.allowance:.3e}"
            f" ({item.note})"
            for item in failed
        )
        msg = f"{model.kind} is not admitted for this pair: {named}"
        raise CouplingError(msg)
    if model.model_error is not None and report.budget > model.model_error:
        terms = ", ".join(
            f"{item.name} {item.value:.3e}"
            for item in items
            if item.counted and item.value
        )
        msg = (
            f"{model.kind}'s defects sum to {report.budget:.3e}, more than its "
            f"model_error {model.model_error:g}: {terms}"
        )
        raise CouplingError(msg)
    return report


def _bin_weights(port: InputPort) -> Float[np.ndarray, " n_omega"]:
    """Return the energy weight of each stored bin of the port's lattice.

    Parameters
    ----------
    port : solprop.coupling.port.InputPort
        The port.

    Returns
    -------
    numpy.ndarray
        :func:`solprop.fourier.energy_weights`, s per bin.
    """
    grid = port.grid
    return fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)


def _occupied(energy: Float[np.ndarray, " n_omega"]) -> np.ndarray:
    """Return the bins that carry energy above rounding of the peak.

    Parameters
    ----------
    energy : numpy.ndarray
        Per-bin energy, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_omega,)``.
    """
    peak = float(np.max(energy)) if energy.size else 0.0
    return energy > _OCCUPIED * peak if peak > 0.0 else np.zeros(energy.shape, bool)


def _weighted(
    defect: Float[np.ndarray, " n_omega"], energy: Float[np.ndarray, " n_omega"]
) -> float:
    """Return a per-bin relative defect weighted by energy: a bound on the total.

    Parameters
    ----------
    defect : numpy.ndarray
        Relative defect per bin.
    energy : numpy.ndarray
        Energy per bin.

    Returns
    -------
    float
        ``sum(defect energy) / sum(energy)``; zero for a zero field.
    """
    total = float(np.sum(energy))
    return 0.0 if total == 0.0 else float(np.sum(np.abs(defect) * energy) / total)


def _metric_matrices(
    metric: PlaneMetric, n_modes: int, n_omega: int
) -> Complex[np.ndarray, "n_omega n_modes n_modes"]:
    """Return a modal metric as one Hermitian matrix per bin.

    Parameters
    ----------
    metric : solprop.coupling.plane.PlaneMetric
        An ``identity`` or ``dense`` metric.
    n_modes : int
        The number of columns.
    n_omega : int
        The number of bins.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_omega, n_modes, n_modes)``, optics convention.

    Raises
    ------
    CouplingError
        For a spatial or unknown metric.
    """
    if metric.kind == "identity":
        return np.broadcast_to(
            np.eye(n_modes, dtype=np.complex128), (n_omega, n_modes, n_modes)
        )
    if metric.kind == "dense":
        return np.moveaxis(cast("np.ndarray", metric.matrix), 2, 0)
    msg = f"a modal metric is identity or dense, got {metric.kind}"
    raise CouplingError(msg)


def _whitened_extremes(
    form: Inexact[np.ndarray, "n n"], metric: Inexact[np.ndarray, "n n"]
) -> tuple[float, float]:
    """Return the extreme generalised eigenvalues of ``form`` against ``metric``.

    Parameters
    ----------
    form : numpy.ndarray
        Hermitian quadratic form.
    metric : numpy.ndarray
        Hermitian positive-definite metric.

    Returns
    -------
    tuple of float
        The smallest and largest eigenvalue of ``metric^{-1/2} form metric^{-1/2}``.

    Raises
    ------
    CouplingError
        If the metric is not positive definite.
    """
    try:
        factor = np.linalg.cholesky(0.5 * (metric + metric.conj().T))
    except np.linalg.LinAlgError as error:
        msg = "a native metric is not positive definite: it cannot whiten a map"
        raise CouplingError(msg) from error
    left = np.linalg.solve(factor, 0.5 * (form + form.conj().T))
    whitened = np.linalg.solve(factor, left.conj().T).conj().T
    values = np.linalg.eigvalsh(0.5 * (whitened + whitened.conj().T))
    return float(values[0]), float(values[-1])


# --------------------------------------------------------------------- same basis


def _same_basis(
    plane: FieldPlane, port: InputPort, model: CouplingModel
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], CouplingReport]:
    """Recover a plane's coordinates exactly in the port's basis.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        The receiving port.
    model : solprop.coupling.models.CouplingModel
        The ``same_basis`` model, with its optional basis change.

    Returns
    -------
    tuple
        The target samples and the report.

    Raises
    ------
    CouplingError
        If the bases differ with no declared change and no common E/H nodes, or a
        declared change does not preserve energy, or the recovered field is not in
        the port's span.
    """
    incident = plane.spectral_energy()
    recovery: tuple[AdmissionItem, ...] = ()
    if model.basis_change is not None:
        samples = _declared_change(plane, port, model)
        note = "declared basis change"
    elif plane.kind == port.route and plane.basis.identity == port.basis.identity:
        if plane.samples.shape != port.shape:
            msg = (
                f"the plane's shape {plane.samples.shape} is not the port's "
                f"{port.shape}"
            )
            raise CouplingError(msg)
        check_same_medium(plane.metric, port.metric)
        samples = np.array(plane.samples)
        note = "identical basis: exact copy"
    elif plane.kind == "general_modal" and port.route == "general_modal":
        samples, recovery = _flux_recovery(plane, port)
        note = "F^-1 B recovery at common nodes"
    else:
        msg = (
            f"a {plane.kind} plane of basis {plane.basis.identity[:12]} is not the "
            f"{port.route} port's basis {port.basis.identity[:12]}: a different basis "
            "is a coupling, not a relaunch; declare a BasisChange or choose an "
            "approximate model"
        )
        raise CouplingError(msg)
    launched = port.metric.spectral_energy(samples, port.grid)
    return samples, _report(
        plane,
        port,
        model,
        index=np.zeros(port.grid.n_omega)
        if port.reference_index is None
        else np.asarray(port.reference_index),
        incident=incident,
        launched=launched,
        captured=launched,
        items=(AdmissionItem("numerical", None, None, False, "map", note), *recovery),
    )


def _declared_change(
    plane: FieldPlane, port: InputPort, model: CouplingModel
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Apply a declared basis change after checking it preserves energy.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane, modal.
    port : solprop.coupling.port.InputPort
        The receiving port, modal.
    model : solprop.coupling.models.CouplingModel
        The model carrying the change.

    Returns
    -------
    numpy.ndarray
        The target samples, shape ``port.shape``.

    Raises
    ------
    CouplingError
        If the change names other bases, has the wrong shape, or does not map the
        plane's metric onto the port's.
    """
    change = model.basis_change
    assert change is not None
    if change.source != plane.basis.identity or change.target != port.basis.identity:
        msg = "the basis change was declared for another pair of bases"
        raise CouplingError(msg)
    n_target, n_pol, n_omega = port.shape
    n_source = plane.samples.shape[0]
    matrices = change.at(n_omega)
    if matrices.shape[1:] != (n_target, n_source) or plane.samples.shape[1] != n_pol:
        msg = (
            f"the basis change maps {matrices.shape[2]} columns to "
            f"{matrices.shape[1]}, "
            f"not the plane's {n_source} to the port's {n_target}"
        )
        raise CouplingError(msg)
    source = _metric_matrices(plane.metric, n_source, n_omega)
    target = _metric_matrices(port.metric, n_target, n_omega)
    pulled = np.einsum("wtm,wtu,wun->wmn", matrices.conj(), target, matrices)
    scale = np.max(np.abs(source), axis=(1, 2))
    defect = np.max(np.abs(pulled - source), axis=(1, 2))
    if np.any(defect > METRIC_TOLERANCE * np.maximum(scale, 1e-300)):
        msg = (
            f"the declared basis change does not preserve energy (largest relative "
            f"defect {float(np.max(defect / np.maximum(scale, 1e-300))):.3e}): it is "
            "not a relabelling of the same field"
        )
        raise CouplingError(msg)
    return np.einsum("wtm,mpw->tpw", matrices, plane.samples)


def _flux_recovery(
    plane: FieldPlane, port: InputPort
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], tuple[AdmissionItem, ...]]:
    r"""Recover a general modal plane's coordinates in the port's basis by F.

    :math:`a = F^{-1}B` with
    :math:`B_m = \frac14\int(\mathbf E\times\mathbf h_m^* +
    \mathbf e_m^*\times\mathbf H)\cdot\hat z\,dA` on the common nodes, using the
    plane's own E and H. The field must lie in the port's span, which is checked
    against its flux :math:`P = \frac12\operatorname{Re}\int(\mathbf E\times
    \mathbf H^*)\cdot\hat z\,dA` on the same nodes. A field in the span has
    :math:`P = a^\dagger Fa` exactly, so its recovered energy :math:`a^\dagger Q_Ba`
    is within the port's metric defect :math:`d_B` (the whitened extremes of F
    against :math:`Q_B`) of P, and the plane's own energy :math:`c^\dagger Q_Ac` is
    within its measured defect :math:`|c^\dagger Q_Ac - P|` of P. The recovered and
    the plane's energies must therefore agree to the sum of the two defects, or the
    field is refused as outside the span. Both defects are reported (not counted:
    ``same_basis`` has no budget).

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        A general modal plane read with its source simulation (``nodal`` E and H).
    port : solprop.coupling.port.InputPort
        A general modal port carrying E/H at fixed nodes.

    Returns
    -------
    tuple
        The target samples, shape ``port.shape``, and the ``target metric`` and
        ``source metric`` items.

    Raises
    ------
    CouplingError
        If either side lacks E/H, the nodes differ, or the field is not in the span.
    """
    nodal = plane.nodal
    if nodal is None or nodal.magnetic is None:
        msg = "an F^-1 B recovery needs the plane's E and H: read it with source="
        raise CouplingError(msg)
    if port.electric is None or port.magnetic is None or port.nodes is None:
        msg = "an F^-1 B recovery needs the port's E/H at fixed nodes"
        raise CouplingError(msg)
    if nodal.nodes.shape != port.nodes.shape or not np.allclose(
        nodal.nodes,
        port.nodes,
        rtol=0.0,
        atol=1e-12 * float(np.max(np.abs(port.nodes))),
    ):
        msg = "the plane's nodes are not the port's: F^-1 B needs a common quadrature"
        raise CouplingError(msg)
    weights = cast("np.ndarray", port.node_weights)
    electric, magnetic = port.electric, port.magnetic
    flux = _flux_matrix(electric, magnetic, weights)
    pairing = _pairing(nodal.electric, nodal.magnetic, electric, magnetic, weights)
    original = plane.spectral_energy()
    occupied = np.flatnonzero(_occupied(original))
    # Only the occupied bins are solved: off the guide's support its stored E/H
    # are explicit zeros, F is singular there and the plane carries nothing.
    coordinates = np.zeros(pairing.shape, dtype=np.complex128)
    for k in occupied:
        coordinates[:, k] = np.linalg.solve(flux[k], pairing[:, k])
    samples = coordinates[:, None, :]
    recovered = port.metric.spectral_energy(samples, port.grid)
    q_matrices = _metric_matrices(port.metric, port.shape[0], port.grid.n_omega)
    defect = 0.0
    for k in occupied:
        low, high = _whitened_extremes(flux[k], q_matrices[k])
        defect = max(defect, abs(low - 1.0), abs(high - 1.0))
    # The field's own flux on the common nodes, per bin (J): 1/2 Re (E x H*).z.
    cross = (
        nodal.electric[:, 0] * nodal.magnetic[:, 1].conj()
        - nodal.electric[:, 1] * nodal.magnetic[:, 0].conj()
    )
    field_flux = (
        _bin_weights(port) * 0.5 * np.real(np.einsum("q,qw->w", weights, cross))
    )
    total = max(float(np.sum(original)), 1e-300)
    source_defect = float(np.sum(np.abs(original - field_flux)[occupied])) / total
    allowed = defect * float(np.sum(recovered)) + (source_defect + 1e-10) * total
    mismatch = abs(float(np.sum(recovered)) - float(np.sum(original)))
    if mismatch > allowed:
        msg = (
            f"the recovered energy differs from the plane's by "
            f"{mismatch / total:.3e}, beyond the port's metric defect {defect:.3e} "
            f"and the plane's {source_defect:.3e}: the field is not in the port's span"
        )
        raise CouplingError(msg)
    items = (
        AdmissionItem(
            "target metric",
            defect,
            None,
            False,
            "map",
            "whitened extremes of the port's F against its native Q",
        ),
        AdmissionItem(
            "source metric",
            source_defect,
            None,
            False,
            "field",
            "the plane's native energy against its flux on the common nodes",
        ),
    )
    return samples, items


def _flux_matrix(
    electric: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"],
    magnetic: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"],
    weights: Float[np.ndarray, " n_nodes"],
) -> Complex[np.ndarray, "n_omega n_modes n_modes"]:
    r"""Return the flux pairing :math:`F_{mn}` of a basis at its nodes, per bin.

    :math:`F_{mn} = \frac14\sum_q w_q[(\mathbf e_n\times\mathbf h_m^*) +
    (\mathbf e_m^*\times\mathbf h_n)]_z`; conjugating every field (optics against
    internal convention) conjugates F, so the formula holds in either.

    Parameters
    ----------
    electric, magnetic : numpy.ndarray
        Basis E and H per √W, shape ``(n_nodes, 3, n_modes, n_omega)``.
    weights : numpy.ndarray
        Node area weights, m^2.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_omega, n_modes, n_modes)``, dimensionless.
    """
    ex, ey = electric[:, 0], electric[:, 1]
    hx, hy = magnetic[:, 0], magnetic[:, 1]
    first = np.einsum("q,qnw,qmw->wmn", weights, ex, hy.conj()) - np.einsum(
        "q,qnw,qmw->wmn", weights, ey, hx.conj()
    )
    second = np.einsum("q,qmw,qnw->wmn", weights, ex.conj(), hy) - np.einsum(
        "q,qmw,qnw->wmn", weights, ey.conj(), hx
    )
    return 0.25 * (first + second)


def _pairing(
    field_e: Complex[np.ndarray, "n_nodes 3 n_omega"],
    field_h: Complex[np.ndarray, "n_nodes 3 n_omega"],
    electric: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"],
    magnetic: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"],
    weights: Float[np.ndarray, " n_nodes"],
) -> Complex[np.ndarray, "n_modes n_omega"]:
    r"""Return the flux pairing of a field against each basis member, per bin.

    :math:`B_m = \frac14\sum_q w_q[\mathbf E\times\mathbf h_m^* +
    \mathbf e_m^*\times\mathbf H]_z`.

    Parameters
    ----------
    field_e, field_h : numpy.ndarray
        The field's E (V/m) and H (A/m) spectra at the nodes, shape
        ``(n_nodes, 3, n_omega)``.
    electric, magnetic : numpy.ndarray
        Basis E and H per √W, shape ``(n_nodes, 3, n_modes, n_omega)``.
    weights : numpy.ndarray
        Node area weights, m^2.

    Returns
    -------
    numpy.ndarray
        In √W, shape ``(n_modes, n_omega)``.
    """
    first = np.einsum("q,qw,qmw->mw", weights, field_e[:, 0], magnetic[:, 1].conj())
    first -= np.einsum("q,qw,qmw->mw", weights, field_e[:, 1], magnetic[:, 0].conj())
    second = np.einsum("q,qmw,qw->mw", weights, electric[:, 0].conj(), field_h[:, 1])
    second -= np.einsum("q,qmw,qw->mw", weights, electric[:, 1].conj(), field_h[:, 0])
    return 0.25 * (first + second)


# ---------------------------------------------------------------- scalar paraxial


def target_columns(
    profiles: ScalarModeBasis, space: RadialGrid | FreeGrid
) -> Complex[np.ndarray, "n_modes n_columns"]:
    r"""Return scalar profiles in a spatial grid's own column representation.

    The columns a profile centred on the grid's axis would have if it were that
    grid's field: on a radial grid its Hankel transform at the grid's
    :math:`k_r`; on a Cartesian grid its Fourier transform
    :math:`2\pi\tilde v(|\mathbf k|)` at the grid's :math:`\mathbf k`, with the
    raw DFT's origin phase :math:`e^{i\mathbf k\cdot\mathbf x_0}/(\Delta x\Delta y)`.
    Both are the continuous transform, in closed form, so a profile's kink at the
    core radius is not sampled. Then
    :math:`\int v^*f\,dA = \sum_j w_j\,c_j^*s_j` with the grid's spectral weights.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The profiles.
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_modes, n_columns)``: in V m per (V/m) on a radial grid, V/m per
        (V/m) on a Cartesian one, in the internal convention of the grid's forward
        transform (the optics-convention columns of a field are conjugated in the
        transverse phase as well).
    """
    if isinstance(space, RadialGrid):
        return profiles.hankel(np.asarray(space.k_r)).astype(np.complex128)
    kx = np.tile(space.k_x, space.n_y)
    ky = np.repeat(space.k_y, space.n_x)
    transform = 2.0 * np.pi * profiles.hankel(np.hypot(kx, ky))
    area = float(space.spatial_weights[0])
    origin = np.exp(1j * (kx * space.x[0] + ky * space.y[0])) / area
    return transform * origin[None, :]


def _channels(
    profiles: ScalarModeBasis, n_pol: int
) -> tuple[Float[np.ndarray, "n_channels 2"], Int[np.ndarray, " n_channels"]]:
    """Return each channel's Jones vector and mode index, channel order m-major.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The profiles.
    n_pol : int
        1: each mode with its own polarisation; 2: each mode in x and in y.

    Returns
    -------
    tuple of numpy.ndarray
        Jones vectors, shape ``(n_channels, 2)``, and the mode of each channel.
    """
    if n_pol == 1:
        return np.asarray(profiles.jones), np.arange(profiles.n_modes)
    jones = np.tile(np.eye(2), (profiles.n_modes, 1))
    return jones, np.repeat(np.arange(profiles.n_modes), 2)


@dataclass
class _Fit:
    """The per-bin quantities of a scalar least-squares fit.

    Attributes
    ----------
    overlap : numpy.ndarray
        ``S^H M f`` per bin, shape ``(n_omega, n_channels)``.
    norm : numpy.ndarray
        ``||f||_M^2`` per bin, shape ``(n_omega,)``.
    gram : numpy.ndarray
        ``G`` per bin, shape ``(n_omega, n_channels, n_channels)``.
    items : list of AdmissionItem
        The admission items the fit measured.
    gain_form : numpy.ndarray or None
        ``S^H M Q_A^-1 M S`` per bin, from which the passivity certificate is
        whitened.
    """

    overlap: np.ndarray
    norm: np.ndarray
    gram: np.ndarray
    items: list[AdmissionItem] = field(default_factory=list)
    gain_form: np.ndarray | None = None


def _scalar_paraxial(
    plane: FieldPlane, port: InputPort, model: CouplingModel
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], CouplingReport]:
    """Fit a scalar field onto a port's scalar profiles in the common measure.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        The receiving port, carrying scalar profiles and a reference index.
    model : solprop.coupling.models.CouplingModel
        The model and its budget.

    Returns
    -------
    tuple
        The target samples and the report.

    Raises
    ------
    CouplingError
        For an unqualified pair or an exceeded budget.
    """
    profiles = port.profiles
    if profiles is None or port.reference_index is None:
        msg = (
            f"this {port.route} port has no scalar profiles: scalar_paraxial couples "
            "into a capillary's HE1m modes or a weakly guiding step-index fibre's"
        )
        raise CouplingError(msg)
    epsilon = cast("float", model.model_error)
    index = np.asarray(port.reference_index)
    bins = _bin_weights(port)
    n_modes, n_pol, n_omega = port.shape
    jones, mode_of = _channels(profiles, n_pol)
    if plane.kind in ("radial", "cartesian"):
        fit, native, envelope_parameters = _spatial_fit(
            plane, profiles, jones, mode_of, index, epsilon
        )
    elif plane.kind == "modal":
        source = plane.profiles
        if source is None:
            msg = (
                "this modal plane carries no mode profiles: read it with "
                "field_plane(run, source=simulation)"
            )
            raise CouplingError(msg)
        if source.family != profiles.family:
            msg = (
                f"scalar_paraxial is not qualified for a {source.family} plane into a "
                f"{profiles.family} port: no recorded envelope covers the pair"
            )
            raise CouplingError(msg)
        fit, native, envelope_parameters = _modal_fit(
            plane,
            source,
            profiles,
            jones=jones,
            mode_of=mode_of,
            index=index,
            epsilon=epsilon,
        )
    else:
        msg = (
            "scalar_paraxial takes a radial, Cartesian or modal plane, not "
            f"{plane.kind}"
        )
        raise CouplingError(msg)
    coordinates = np.zeros((n_omega, jones.shape[0]), dtype=np.complex128)
    occupied = _occupied(fit.norm)
    for k in np.flatnonzero(occupied):
        coordinates[k] = np.linalg.solve(fit.gram[k], fit.overlap[k])
    captured = bins * np.real(np.einsum("wc,wc->w", fit.overlap.conj(), coordinates))
    # The fit is formed in the internal convention, where a Cartesian plane's
    # transverse DFT carries its origin phase; the launch is stored in the optics one.
    samples = fourier.spectrum_to_optics(
        coordinates.T.reshape(n_modes, n_pol, n_omega), kind=port.grid.kind
    )
    launched = port.metric.spectral_energy(samples, port.grid)
    incident = bins * fit.norm
    items = list(fit.items)
    items.append(_target_metric(profiles, mode_of, index, occupied, epsilon))
    items.append(
        _target_angular_scalar(
            profiles, (jones, mode_of), index, port, occupied, epsilon
        )
    )
    items.append(_source_metric_item(incident, native, epsilon))
    if profiles.family == "capillary":
        envelope_parameters.update(_leaky_parameters(profiles, plane, incident))
    elif plane.kind == "modal":
        # A step-index modal source is a step-index fibre with its profiles.
        envelope_parameters["uncovered"] = _uncovered_v_share(
            {"v_source": cast("ScalarModeBasis", plane.profiles), "v_target": profiles},
            STEP_INDEX_ENVELOPE,
            plane.grid.omega,
            incident,
        )
    else:
        envelope_parameters["uncovered"] = _uncovered_v_share(
            {"v_target": profiles},
            STEP_INDEX_BEAM_ENVELOPE,
            plane.grid.omega,
            incident,
        )
    items.extend(
        _contrast_items(profiles, plane, index, occupied, epsilon, envelope_parameters)
    )
    items.append(_gain_item(_scalar_gain(fit, occupied), items))
    return samples, _report(
        plane,
        port,
        model,
        index=index,
        incident=incident,
        launched=launched,
        captured=captured,
        items=tuple(items),
    )


def _kappa_star(
    index: Float[np.ndarray, " n_omega"],
    omega: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> Float[np.ndarray, " n_omega"]:
    r"""Return the admitted cone's transverse wavenumber at each bin.

    :math:`\kappa_* = n_{\rm ref}(\omega/c)\sin\theta_*`, with
    :math:`1 - \cos\theta_* = \epsilon_m/8`.

    Parameters
    ----------
    index : numpy.ndarray
        Reference index per bin.
    omega : numpy.ndarray
        Angular frequency per bin, rad/s.
    epsilon : float
        The model budget.

    Returns
    -------
    numpy.ndarray
        Transverse wavenumber in rad/m, zero where the index or the frequency is
        not positive.
    """
    sine = math.sin(math.acos(1.0 - epsilon / 8.0))
    return np.where((index > 0) & (omega > 0), index * omega / C * sine, 0.0)


def _spatial_fit(
    plane: FieldPlane,
    profiles: ScalarModeBasis,
    jones: Float[np.ndarray, "n_channels 2"],
    mode_of: np.ndarray,
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> tuple[_Fit, Float[np.ndarray, " n_omega"], dict[str, float]]:
    """Return the scalar fit of a spatial plane onto a port's scalar profiles.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        A radial or Cartesian plane.
    profiles : solprop.coupling.models.ScalarModeBasis
        The target profiles.
    jones : numpy.ndarray
        Each target channel's Jones vector.
    mode_of : numpy.ndarray
        Each target channel's mode.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    tuple
        The fit, the plane's native energy per bin (J) and the envelope's range
        parameters.

    Raises
    ------
    CouplingError
        If the plane occupies nonpropagating columns, or its metric is not an
        admittance.
    """
    space = cast("RadialGrid | FreeGrid", plane.basis.space)
    metric = plane.metric
    if metric.kind != "admittance":
        msg = f"a spatial plane's metric is an admittance, got {metric.kind}"
        raise CouplingError(msg)
    weights = np.asarray(space.spectral_weights)
    columns = target_columns(profiles, space)
    samples = fourier.spectrum_to_internal(plane.samples, kind=plane.grid.kind)
    n_pol_source = samples.shape[1]
    # The field's projection onto each target channel's polarisation, per column.
    projected = np.einsum(
        "cp,jpw->cjw", jones[:, :n_pol_source].astype(complex), samples
    )
    rows = columns[mode_of]
    overlap_raw = np.einsum("j,cj,cjw->wc", weights, rows.conj(), projected)
    # The profiles are orthonormal over the whole plane, so their exact Gram is the
    # identity (times the Jones overlaps). It, and not the Gram of their transforms
    # truncated to the plane's window, is the least-squares matrix: the overlaps are
    # exact for a field the plane resolves, whatever of a profile lies beyond the
    # window, and then the orthogonal-residual identity holds in the continuum.
    exact = profiles.gram()[np.ix_(mode_of, mode_of)] * (jones @ jones.T)
    effective = profiles.effective_index[mode_of]
    root = np.sqrt(np.where(effective > 0, effective, 1.0))
    density = np.einsum("j,jpw->w", weights, np.abs(samples) ** 2)
    norm = _SCALE * index * density
    overlap = index[:, None] * math.sqrt(_SCALE) * overlap_raw / root.T
    gram = (
        index[:, None, None] * exact[None] / (root.T[:, :, None] * root.T[:, None, :])
    )
    native = plane.spectral_energy()
    fit = _Fit(overlap, norm, gram)
    occupied = _occupied(norm)
    # Nonpropagating content carries no energy and no coupling model.
    admittance = np.real(cast("np.ndarray", metric.admittance))
    column_energy = weights[:, None] * np.sum(np.abs(samples) ** 2, axis=1)
    _refuse_nonpropagating(column_energy, admittance, occupied)
    fit.items.append(_spatial_angular(plane, column_energy, index, epsilon))
    # The overlaps are exact for a field the plane resolves: what the plane holds
    # near its spectral edge bounds what it may miss beyond it.
    edge = _spectral_edge_share(space, column_energy)
    analytic = plane.analytic
    domain = 0.0 if analytic is None else analytic.error
    fit.items.append(
        AdmissionItem(
            "numerical",
            edge + domain,
            None,
            True,
            "field",
            f"source content at the plane's spectral edge {edge:.1e}, "
            f"source domain/resolution {domain:.1e}",
        )
    )
    # The passivity form: C Q_A^-1 C^H = G^-1 X G^-1 with X weighted by 1/Re Y.
    live = admittance > 0.0
    inverse = np.where(live, 1.0 / np.where(live, admittance, 1.0), 0.0)
    cross = np.einsum("j,cj,dj,jw->wcd", weights, rows.conj(), rows, inverse)
    cross = cross * (jones @ jones.T)[None]
    fit.gain_form = (
        index[:, None, None] ** 2 * cross / (root.T[:, :, None] * root.T[:, None, :])
    )
    if profiles.family == "capillary":
        parameters = {"modes": float(profiles.n_modes), "radius_ratio": 1.0}
    else:
        parameters = _beam_step_index_parameters(profiles)
    return fit, native, parameters


def _spectral_edge_share(
    space: RadialGrid | FreeGrid, column_energy: Float[np.ndarray, "n_columns n_omega"]
) -> float:
    """Return a spatial field's energy share in the outer fifth of its spectral window.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.
    column_energy : numpy.ndarray
        Each column's weighted squared samples per bin.

    Returns
    -------
    float
        Dimensionless; columns with a transverse wavenumber above 0.8 of the
        window's largest along an axis.
    """
    if isinstance(space, RadialGrid):
        k_r = np.asarray(space.k_r)
        outer = k_r > 0.8 * float(np.max(k_r))
    else:
        kx = np.abs(np.tile(space.k_x, space.n_y))
        ky = np.abs(np.repeat(space.k_y, space.n_x))
        outer = (kx > 0.8 * float(np.max(kx))) | (ky > 0.8 * float(np.max(ky)))
    total = float(np.sum(column_energy))
    return 0.0 if total == 0.0 else float(np.sum(column_energy[outer]) / total)


def _bin_like(plane: FieldPlane) -> Float[np.ndarray, " n_omega"]:
    """Return the plane's bin energy weights.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The plane.

    Returns
    -------
    numpy.ndarray
        :func:`solprop.fourier.energy_weights` of its grid.
    """
    grid = plane.grid
    return fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)


def _modal_fit(  # noqa: PLR0913 - the plane, both profile sets and the measure
    plane: FieldPlane,
    source: ScalarModeBasis,
    profiles: ScalarModeBasis,
    *,
    jones: Float[np.ndarray, "n_channels 2"],
    mode_of: np.ndarray,
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> tuple[_Fit, Float[np.ndarray, " n_omega"], dict[str, float]]:
    """Return the scalar fit of a modal plane's synthesised field onto target profiles.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        An ordinary modal plane with its profiles.
    source : solprop.coupling.models.ScalarModeBasis
        Its profiles.
    profiles : solprop.coupling.models.ScalarModeBasis
        The target profiles.
    jones : numpy.ndarray
        Each target channel's Jones vector.
    mode_of : numpy.ndarray
        Each target channel's mode.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    tuple
        The fit, the plane's native energy per bin (J) and the envelope's range
        parameters.
    """
    source_jones, source_mode = _channels(source, plane.samples.shape[1])
    overlap_matrix, gram_source, _, error = _profile_overlaps(source, profiles)
    pol = jones @ source_jones.T
    cross = overlap_matrix[np.ix_(mode_of, source_mode)] * pol
    k_source = gram_source[np.ix_(source_mode, source_mode)] * (
        source_jones @ source_jones.T
    )
    n_source = source.effective_index[source_mode]
    n_target = profiles.effective_index[mode_of]
    root_source = np.sqrt(np.where(n_source > 0, n_source, 1.0)).T
    root_target = np.sqrt(np.where(n_target > 0, n_target, 1.0)).T
    coefficients = (
        fourier.spectrum_to_internal(plane.samples, kind=plane.grid.kind)
        .reshape(-1, plane.grid.n_omega)
        .T
    )
    scaled = coefficients / root_source
    overlap = index[:, None] * np.einsum("cs,ws->wc", cross, scaled) / root_target
    norm = index * np.real(np.einsum("ws,st,wt->w", scaled.conj(), k_source, scaled))
    # The closed-form normalisation makes both profile sets orthonormal; their
    # quadrature Grams' departure from the identity is part of the numerical item.
    exact = profiles.gram()[np.ix_(mode_of, mode_of)] * (jones @ jones.T)
    gram = (
        index[:, None, None]
        * exact[None]
        / (root_target[:, :, None] * root_target[:, None, :])
    )
    fit = _Fit(overlap, norm, gram)
    fit.items.append(
        _modal_angular(
            source,
            jones=source_jones,
            mode_of=source_mode,
            scaled=scaled.T,
            norm=norm,
            index=index,
            plane=plane,
            epsilon=epsilon,
        )
    )
    # The quadrature error bounds overlap (field-amplitude) entries; as an energy
    # discrepancy it enters as 2 delta + delta^2 (section 4.7).
    fit.items.append(
        AdmissionItem(
            "numerical",
            2.0 * error + error**2,
            None,
            True,
            "map",
            f"profile overlap quadrature, amplitude error {error:.1e}",
        )
    )
    # Native source metric is the identity, so C Q_A^-1 C^H = G^-1 R R^H G^-1.
    cross_scaled = (
        index[:, None, None]
        * cross[None]
        / (root_target[:, :, None] * root_source[:, None, :])
    )
    fit.gain_form = np.einsum("wcs,wds->wcd", cross_scaled, cross_scaled.conj())
    parameters = {
        "modes": float(max(source.n_modes, profiles.n_modes)),
        "radius_ratio": source.radius / profiles.radius,
    }
    if profiles.family == "step_index":
        parameters.update(_step_index_parameters(source, profiles))
    return fit, plane.spectral_energy(), parameters


def _profile_overlaps(
    source: ScalarModeBasis, target: ScalarModeBasis
) -> tuple[
    Float[np.ndarray, "n_target n_source"],
    Float[np.ndarray, "n_source n_source"],
    Float[np.ndarray, "n_target n_target"],
    float,
]:
    """Return the profiles' overlaps on a composite rule, with its error estimate.

    The rule breaks at both core radii and extends to where both sets are
    negligible; panels hold at most a quarter period of the most oscillatory
    profile. The same overlaps on the rule refined by halving every panel bound the
    quadrature error, together with each Gram's departure from the identity the
    profiles' closed-form normalisation fixes.

    Parameters
    ----------
    source, target : solprop.coupling.models.ScalarModeBasis
        The two profile sets.

    Returns
    -------
    tuple
        The target-by-source overlaps, both Grams and the error estimate.
    """
    radii = (source.radius, target.radius)
    outer = max(source.outer(), target.outer())
    wavenumber = max(
        float(np.max(source.u)) / source.radius, float(np.max(target.u)) / target.radius
    )
    panel = 0.5 * math.pi / wavenumber

    def overlaps(step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the three overlap matrices on a rule of the given panel length.

        Parameters
        ----------
        step : float
            Largest panel length, m.

        Returns
        -------
        tuple of numpy.ndarray
            Target-by-source overlaps and the two Grams.
        """
        r, w = radial_rule(radii, outer, panel=step)
        vs = source.values(r)
        vt = target.values(r)
        return (vt * w) @ vs.T, (vs * w) @ vs.T, (vt * w) @ vt.T

    cross, gram_source, gram_target = overlaps(panel)
    fine = overlaps(0.5 * panel)
    error = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip((cross, gram_source, gram_target), fine, strict=True)
    )
    error = max(
        error,
        float(np.max(np.abs(gram_source - source.gram()))),
        float(np.max(np.abs(gram_target - target.gram()))),
    )
    return cross, gram_source, gram_target, error


def _step_index_parameters(
    source: ScalarModeBasis, target: ScalarModeBasis
) -> dict[str, float]:
    """Return a step-index pair's envelope parameters at each profile wavelength.

    Parameters
    ----------
    source, target : solprop.coupling.models.ScalarModeBasis
        The two LP sets.

    Returns
    -------
    dict
        ``v_source`` and ``v_target``: each fibre's V at its profile wavelength,
        from its LP01 U and W.
    """
    return {
        "v_source": float(math.hypot(source.u[0], source.w[0])),
        "v_target": float(math.hypot(target.u[0], target.w[0])),
    }


def _beam_step_index_parameters(profiles: ScalarModeBasis) -> dict[str, float]:
    """Return a beam-into-fibre pair's envelope parameters at the profile wavelength.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The target fibre's LP set.

    Returns
    -------
    dict
        ``v_target``: the fibre's V from its LP01 U and W; ``modes``: the number of
        distinct LP0m profiles (two polarisations of one profile count once).
    """
    return {
        "v_target": float(math.hypot(profiles.u[0], profiles.w[0])),
        "modes": float(np.unique(profiles.u).size),
    }


def _target_metric(
    profiles: ScalarModeBasis,
    mode_of: np.ndarray,
    index: Float[np.ndarray, " n_omega"],
    occupied: np.ndarray,
    epsilon: float,
) -> AdmissionItem:
    r"""Return the relative defect between the exact G and the target's native Q.

    The profiles are orthonormal, so in true-power coordinates
    :math:`G = \operatorname{diag}(n_{\rm ref}/\operatorname{Re}n_{\rm eff})` and
    ``Q`` is the identity: the whitened eigenvalues are the ratios.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The target profiles.
    mode_of : numpy.ndarray
        Each channel's mode.
    index : numpy.ndarray
        Reference index per bin.
    occupied : numpy.ndarray
        Occupied bins.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``target metric``, capped at ``epsilon / 8``.
    """
    effective = profiles.effective_index[mode_of][:, occupied]
    if effective.size == 0:
        value = 0.0
    else:
        value = float(np.max(np.abs(index[occupied][None, :] / effective - 1.0)))
    return AdmissionItem(
        "target metric",
        value,
        epsilon / 8.0,
        True,
        "map",
        "G (n_ref/Re n_eff) against the native identity",
    )


def _target_angular_scalar(
    profiles: ScalarModeBasis,
    channels: tuple[np.ndarray, np.ndarray],
    index: Float[np.ndarray, " n_omega"],
    port: InputPort,
    occupied: np.ndarray,
    epsilon: float,
) -> AdmissionItem:
    """Return the retained target subspace's angular defect, its worst unit field.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The target profiles.
    channels : tuple of numpy.ndarray
        Each channel's Jones vector and mode (:func:`_channels`).
    index : numpy.ndarray
        Reference index per bin.
    port : solprop.coupling.port.InputPort
        The port.
    occupied : numpy.ndarray
        Occupied bins.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``angular`` of the target subspace: the largest eigenvalue of
        ``(1 - cos theta*) K + (1 - (1 - cos theta*)) T`` against the channel Gram
        ``K``, with T the tail Gram.
    """
    jones, mode_of = channels
    polarisation = jones @ jones.T
    gram = profiles.gram()[np.ix_(mode_of, mode_of)] * polarisation
    star = _kappa_star(index, port.grid.omega, epsilon)
    worst = 0.0
    for k in np.flatnonzero(occupied):
        cone = profiles.cone_gram(float(star[k]))[np.ix_(mode_of, mode_of)]
        tail = gram - cone * polarisation
        defect = (epsilon / 8.0) * gram + (1.0 - epsilon / 8.0) * tail
        worst = max(worst, _whitened_extremes(defect, gram)[1])
    return AdmissionItem(
        "angular",
        worst if np.any(occupied) else 0.0,
        None,
        True,
        "map",
        "retained target subspace: cone-rule allocation (1 - cos theta*) = "
        "model_error/8 of the cone, not a measured defect, plus the tail",
    )


def _source_metric_item(
    incident: Float[np.ndarray, " n_omega"],
    native: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> AdmissionItem:
    """Return the relative defect between the common measure and the source's own.

    Parameters
    ----------
    incident : numpy.ndarray
        The incident energy per bin in the common measure, J.
    native : numpy.ndarray
        The plane's energy per bin in its native metric, J.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``source metric``: ``sum |M_0 - Q_A| / sum Q_A``, capped at ``epsilon/8``.
    """
    total = float(np.sum(native))
    value = 0.0 if total == 0.0 else float(np.sum(np.abs(incident - native)) / total)
    return AdmissionItem(
        "source metric",
        value,
        epsilon / 8.0,
        True,
        "field",
        "common measure M0 against the plane's native metric",
    )


def _leaky_parameters(
    profiles: ScalarModeBasis,
    plane: FieldPlane,
    incident: Float[np.ndarray, " n_omega"],
) -> dict[str, float]:
    r"""Return a capillary pair's leaky-field parameter and its uncovered share.

    At each bin the larger :math:`\delta = |\nu_n|c/(\omega a)` of the target and a
    capillary source. ``leaky`` is its mean weighted by the incident energy, as the
    other per-bin items are: the envelope's allowance is linear in
    :math:`\delta`, so the weighted mean bounds the whole field's discrepancy over
    the bins inside the range. ``uncovered`` is the incident energy share of the
    bins beyond the range, which the envelope does not bound and the item counts
    in full.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The target's capillary profiles.
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    incident : numpy.ndarray
        Incident energy per bin, J.

    Returns
    -------
    dict
        ``leaky`` and ``uncovered``, dimensionless.
    """
    omega = plane.grid.omega
    delta = cast("np.ndarray", profiles.leaky(omega))
    source = None if plane.profiles is None else plane.profiles.leaky(omega)
    if source is not None:
        delta = np.maximum(delta, source)
    beyond = delta > CAPILLARY_ENVELOPE.ranges["leaky"][1]
    return {
        "leaky": _weighted(delta, incident),
        "uncovered": _weighted(beyond.astype(float), incident),
    }


def _uncovered_v_share(
    fibres: Mapping[str, ScalarModeBasis],
    envelope: PairEnvelope,
    omega: Float[np.ndarray, " n_omega"],
    incident: Float[np.ndarray, " n_omega"],
) -> float:
    r"""Return the incident share of bins where a fibre's V leaves the range.

    The profiles are frozen at the profile wavelength, which the envelope's range
    must contain, but the vector physics it bounds depends on the actual
    :math:`V(\omega) = (\omega a/c)\sqrt{n_{\rm core}^2 - n_{\rm clad}^2}` at
    each bin. A bin where any fibre's V is outside the range is not bounded by
    the envelope, so its share of the incident energy is counted in full.

    Parameters
    ----------
    fibres : collections.abc.Mapping
        Each fibre's LP set, with its core and cladding indices, by the name of
        its V range in `envelope` (``v_source``, ``v_target``).
    envelope : solprop.coupling.envelopes.PairEnvelope
        The pair's envelope.
    omega : numpy.ndarray
        Absolute angular frequency of each bin, rad/s.
    incident : numpy.ndarray
        Incident energy per bin, J.

    Returns
    -------
    float
        Dimensionless.
    """
    beyond = np.zeros(omega.shape, bool)
    for name, basis in fibres.items():
        core = cast("np.ndarray", basis.core_index)
        cladding = cast("np.ndarray", basis.cladding_index)
        aperture = np.sqrt(np.maximum(core**2 - cladding**2, 0.0))
        v = omega * basis.radius * aperture / C
        low, high = envelope.ranges[name]
        beyond |= (incident > 0.0) & ((v < low) | (v > high))
    return _weighted(beyond.astype(float), incident)


def _contrast_items(
    profiles: ScalarModeBasis,
    plane: FieldPlane,
    index: Float[np.ndarray, " n_omega"],
    occupied: np.ndarray,
    epsilon: float,
    parameters: dict[str, float],
) -> list[AdmissionItem]:
    r"""Return the contrast screen and the pair envelope of a scalar coupling.

    A capillary wall is not a weakly guiding dielectric, so the screen does not
    apply there; a step-index pair is screened by
    :math:`\Delta \le \epsilon_m/8` over both fibres' regions (the target's alone
    for an incident beam), unless its recorded envelope covers it. Every pair needs
    its envelope to cover it: the capillary one, the fibre-pair one, or for a beam
    into a fibre :data:`~solprop.coupling.envelopes.STEP_INDEX_BEAM_ENVELOPE`.

    Parameters
    ----------
    profiles : solprop.coupling.models.ScalarModeBasis
        The target profiles.
    plane : solprop.coupling.plane.FieldPlane
        The source plane; for a step-index target, a spatial plane or a modal plane
        with its profiles.
    index : numpy.ndarray
        Reference index per bin.
    occupied : numpy.ndarray
        Occupied bins.
    epsilon : float
        The model budget.
    parameters : dict
        The pair's envelope parameters.

    Returns
    -------
    list of AdmissionItem
        ``contrast`` and ``envelope``.

    Raises
    ------
    CouplingError
        If no recorded envelope covers the pair.
    """
    envelope: PairEnvelope = CAPILLARY_ENVELOPE
    if profiles.family == "step_index":
        envelope = (
            STEP_INDEX_ENVELOPE if plane.kind == "modal" else STEP_INDEX_BEAM_ENVELOPE
        )
    if profiles.family == "capillary":
        screen = AdmissionItem(
            "contrast",
            None,
            None,
            False,
            "model",
            "not applied: a capillary wall is not a weakly guiding dielectric",
        )
    else:
        contrast = 0.0
        # A step-index modal source is a fibre with its own profiles; a beam's
        # medium is measured by the source-metric item instead.
        fibres = (
            (profiles, cast("ScalarModeBasis", plane.profiles))
            if plane.kind == "modal"
            else (profiles,)
        )
        for one in fibres:
            for values in (one.core_index, one.cladding_index):
                values = cast("np.ndarray", values)
                ratio = np.abs(values[occupied] ** 2 - index[occupied] ** 2) / (
                    2.0 * index[occupied] ** 2
                )
                contrast = max(contrast, float(np.max(ratio)) if ratio.size else 0.0)
        parameters["contrast"] = contrast
        within = contrast <= epsilon / 8.0
        screen = AdmissionItem(
            "contrast",
            contrast,
            None if not within else epsilon / 8.0,
            False,
            "model",
            "weak-guidance screen Delta"
            + ("" if within else f" > model_error/8, replaced by {envelope.identity}"),
        )
    uncovered = parameters.get("uncovered", 0.0)
    outside = envelope.covers(parameters)
    if outside is not None:
        msg = (
            f"scalar_paraxial is not qualified for this {profiles.family} pair: "
            f"{outside}"
        )
        raise CouplingError(msg)
    return [
        screen,
        AdmissionItem(
            "envelope",
            envelope.allowance(parameters) + uncovered,
            None,
            True,
            "model",
            envelope.identity
            + (
                f", plus the {uncovered:.1e} incident share outside its range"
                if uncovered > 0.0
                else ""
            ),
        ),
    ]


def _scalar_gain(fit: _Fit, occupied: np.ndarray) -> float:
    """Return the largest apparent gain of a scalar fit's map over occupied bins.

    The largest whitened squared singular value of ``C = G^-1 S^H M`` from the
    source's native coordinates to the target's, ``C Q_A^-1 C^H`` against ``Q_B``
    (the identity), minus one.

    Parameters
    ----------
    fit : _Fit
        The fit, with its ``gain_form``.
    occupied : numpy.ndarray
        Occupied bins.

    Returns
    -------
    float
        Dimensionless; negative for a strictly contracting map.
    """
    form = cast("np.ndarray", fit.gain_form)
    gain = -1.0
    for k in np.flatnonzero(occupied):
        inverse = np.linalg.inv(fit.gram[k])
        whitened = inverse @ form[k] @ inverse.conj().T
        largest = float(np.linalg.eigvalsh(0.5 * (whitened + whitened.conj().T))[-1])
        gain = max(gain, largest - 1.0)
    return gain


@dataclass(frozen=True, eq=False)
class _Source:
    """How a plane's coordinates make a field at a general guide's nodes.

    Attributes
    ----------
    transfer : collections.abc.Callable
        ``transfer(k)``: the internal-convention transverse field at the nodes per
        unit source coordinate at bin k, shape ``(n_nodes, 2, n_source)``.
    coordinates : numpy.ndarray
        Internal-convention source coordinates, shape ``(n_source, n_omega)``.
    native : numpy.ndarray
        Each coordinate's native metric weight, J per squared coordinate before the
        bin weight, shape ``(n_source, n_omega)``.
    norm : numpy.ndarray
        The incident norm in the common measure per bin, before the bin weight.
    items : tuple of AdmissionItem
        The source's admission items.
    """

    transfer: Callable[[int], np.ndarray]
    coordinates: np.ndarray
    native: np.ndarray
    norm: np.ndarray
    items: tuple[AdmissionItem, ...]


def _vector_paraxial(
    plane: FieldPlane, port: InputPort, model: CouplingModel
) -> tuple[Complex[np.ndarray, "n_modes n_pol n_omega"], CouplingReport]:
    r"""Recover a beam's coordinates in a general guide with the flux pairing F.

    The incident E is the plane's field at the port's nodes (a spatial plane's
    spectral interpolant, zero beyond its window, or an ordinary modal plane's
    synthesis from its profiles); its H is the forward closure
    :math:`H_t = Y_0\hat z\times E_t`. Then :math:`a = F^{-1}B`. Everything is formed
    in the internal convention, where the port's tables live, and the launch is
    converted to the optics convention at the end.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    port : solprop.coupling.port.InputPort
        A general modal port carrying E/H at fixed nodes and its contrast.
    model : solprop.coupling.models.CouplingModel
        The model and its budget.

    Returns
    -------
    tuple
        The target samples and the report.

    Raises
    ------
    CouplingError
        For an unqualified pair or an exceeded budget.
    """
    if (
        port.route != "general_modal"
        or port.electric is None
        or port.magnetic is None
        or port.nodes is None
        or port.node_weights is None
        or port.reference_index is None
    ):
        msg = (
            "vector_paraxial couples into a general modal guide that carries its E/H "
            "at fixed nodes; this port does not"
        )
        raise CouplingError(msg)
    epsilon = cast("float", model.model_error)
    index = np.asarray(port.reference_index)
    weights = np.asarray(port.node_weights)
    electric = np.conj(port.electric)
    magnetic = np.conj(port.magnetic)
    admittance = index * EPS0 * C
    source = _vector_source(plane, port, index, epsilon)
    occupied = _occupied(source.norm)
    amplitudes, gain, metric_defect = _vector_coordinates(
        source, port, admittance, occupied
    )
    samples = fourier.spectrum_to_optics(amplitudes, kind=port.grid.kind)[:, None, :]
    contrast_values = (
        np.zeros(0) if port.contrast is None else np.asarray(port.contrast)[occupied]
    )
    probes = _probe_bins(occupied, source.norm)
    incident = _bin_weights(port) * source.norm
    items = [
        *source.items,
        AdmissionItem(
            "target metric",
            metric_defect,
            epsilon / 8.0,
            True,
            "map",
            "F at the port's nodes against the native Q",
        ),
        AdmissionItem(
            "contrast",
            float(np.max(contrast_values)) if contrast_values.size else 0.0,
            epsilon / 8.0,
            False,
            "model",
            "weak-guidance screen Delta over the guide's regions",
        ),
        _closure_item(electric, magnetic, weights, admittance, probes, epsilon),
        _target_angular_vector(electric, weights, port, index, probes, epsilon),
        _vector_envelope_item(port, incident),
    ]
    items.append(_gain_item(gain, items, "field"))
    return samples, _report(
        plane,
        port,
        model,
        index=index,
        incident=incident,
        launched=port.metric.spectral_energy(samples, port.grid),
        captured=None,
        captured_reason=(
            "vector_paraxial recovers coordinates with the flux pairing F; F and the "
            "incident measure M0 are not one positive metric, so no orthogonal "
            "remainder is a physical loss (section 4.3)"
        ),
        items=tuple(items),
    )


def _vector_envelope_item(
    port: InputPort, incident: Float[np.ndarray, " n_omega"]
) -> AdmissionItem:
    r"""Return the vector model's recorded facet envelope item, or refuse the guide.

    Only a :class:`~solprop.modes.guide.StepIndexProvider` guide has a recorded
    envelope (:data:`~solprop.coupling.envelopes.VECTOR_STEP_INDEX_ENVELOPE`); its
    port carries the guide's normalised frequency. Its parameters are the contrast,
    V and squared contrast weighted by the incident energy, as the other per-bin
    items are: the allowance is linear in the squared contrast, so the weighted
    mean bounds the whole field over the bins inside the range. Bins whose contrast
    or V leave the range are not bounded, and their incident share is counted in
    full. Any other general guide (a graded index) keeps an uncounted item without
    a value: its omitted facet reflection is bounded only by the contrast screen
    and the closure.

    Parameters
    ----------
    port : solprop.coupling.port.InputPort
        The general modal port.
    incident : numpy.ndarray
        Incident energy per bin, J.

    Returns
    -------
    AdmissionItem
        ``envelope``.

    Raises
    ------
    CouplingError
        If the weighted parameters lie outside the envelope's range.
    """
    if port.normalised_frequency is None:
        return AdmissionItem(
            "envelope",
            None,
            None,
            False,
            "model",
            "no recorded pair envelope for this provider (only a step-index guide "
            "has one): omitted facet reflection is bounded only by the contrast "
            "screen and the closure",
        )
    envelope = VECTOR_STEP_INDEX_ENVELOPE
    frequency = np.asarray(port.normalised_frequency)
    # A step-index port carries its contrast beside its frequency.
    contrast = np.asarray(port.contrast)
    parameters = {
        "contrast": _weighted(contrast, incident),
        "v": _weighted(frequency, incident),
        "contrast_squared": _weighted(contrast**2, incident),
    }
    if float(np.sum(incident)) > 0.0:
        outside = envelope.covers(parameters)
        if outside is not None:
            msg = (
                f"vector_paraxial is not qualified for this step-index guide: {outside}"
            )
            raise CouplingError(msg)
    low, high = envelope.ranges["v"]
    beyond = (
        (contrast > envelope.ranges["contrast"][1])
        | (frequency < low)
        | (frequency > high)
    )
    uncovered = _weighted(beyond.astype(float), incident)
    return AdmissionItem(
        "envelope",
        envelope.allowance(parameters) + uncovered,
        None,
        True,
        "model",
        envelope.identity
        + (
            f", plus the {uncovered:.1e} incident share outside its range"
            if uncovered > 0.0
            else ""
        ),
    )


def _vector_coordinates(
    source: _Source,
    port: InputPort,
    admittance: Float[np.ndarray, " n_omega"],
    occupied: np.ndarray,
) -> tuple[Complex[np.ndarray, "n_modes n_omega"], float, float]:
    r"""Return the recovered coordinates, the map's apparent gain and F's defect.

    At each occupied bin :math:`a = F^{-1}B` with
    :math:`B_m = \frac14\sum_q w_q[\mathbf E\times\mathbf h_m^* +
    \mathbf e_m^*\times Y_0\hat z\times\mathbf E]_z`, and the whitened certificate
    of ``C = F^-1 L`` (``L`` the pairing per unit source coordinate) against the
    native metrics.

    Parameters
    ----------
    source : _Source
        The source's field map.
    port : solprop.coupling.port.InputPort
        The general modal port.
    admittance : numpy.ndarray
        :math:`Y_0` per bin, A/V.
    occupied : numpy.ndarray
        Occupied bins.

    Returns
    -------
    tuple
        Internal-convention amplitudes, shape ``(n_modes, n_omega)``; the largest
        apparent gain; the largest whitened defect of F against the native Q.
    """
    weights = np.asarray(port.node_weights)
    electric = np.conj(cast("np.ndarray", port.electric))
    magnetic = np.conj(cast("np.ndarray", port.magnetic))
    n_modes = port.shape[0]
    flux = _flux_matrix(electric, magnetic, weights)
    target_q = _metric_matrices(port.metric, n_modes, port.grid.n_omega)
    amplitudes = np.zeros((n_modes, port.grid.n_omega), dtype=np.complex128)
    gain = 0.0
    metric_defect = 0.0
    for k in np.flatnonzero(occupied):
        # The bracket of B for a unit transverse field along x (last index 0) or y.
        hx, hy = magnetic[:, 0, :, k], magnetic[:, 1, :, k]
        ex, ey = electric[:, 0, :, k], electric[:, 1, :, k]
        bracket = np.stack(
            (
                hy.conj() + admittance[k] * ex.conj(),
                -hx.conj() + admittance[k] * ey.conj(),
            ),
            axis=2,
        )
        pairing = 0.25 * np.einsum(
            "q,qmp,qps->ms", weights, bracket, source.transfer(int(k))
        )
        amplitudes[:, k] = np.linalg.solve(flux[k], pairing @ source.coordinates[:, k])
        # The certificate covers the coordinates this field occupies: a column the
        # guide's node rule cannot resolve has an aliased pairing, and a field with
        # nothing in it is not mapped through it (section 4.6, a single field's
        # certificate is labelled as such).
        energy = source.native[:, k] * np.abs(source.coordinates[:, k]) ** 2
        live = (source.native[:, k] > 0.0) & (
            energy > _OCCUPIED * float(np.max(energy))
        )
        mapped = np.linalg.solve(flux[k], pairing[:, live])
        form = (mapped / source.native[live, k][None, :]) @ mapped.conj().T
        native_q = np.conj(target_q[k])
        gain = max(gain, _whitened_extremes(form, native_q)[1] - 1.0)
        low, high = _whitened_extremes(flux[k], native_q)
        metric_defect = max(metric_defect, abs(low - 1.0), abs(high - 1.0))
    return amplitudes, gain, metric_defect


def _gain_item(
    gain: float, items: list[AdmissionItem], scope: ItemScope = "map"
) -> AdmissionItem:
    """Return the passivity item of a map's apparent gain, or refuse it.

    Its excess over one is rounding below :data:`PASSIVITY_ROUNDING`, a model
    defect within the model's own metric, angular and closure items, and refused
    beyond them. It is never rescaled or clipped.

    Parameters
    ----------
    gain : float
        The largest whitened squared singular value minus one.
    items : list of AdmissionItem
        The items already measured.
    scope : {'map', 'field'}, optional
        Whether the certificate covers the whole map or the coordinates this field
        occupies.

    Returns
    -------
    AdmissionItem
        ``passivity``.

    Raises
    ------
    CouplingError
        If the gain exceeds the model's own defects.
    """
    allowance = sum(
        item.value
        for item in items
        if item.value is not None
        and item.name in ("target metric", "source metric", "angular", "closure")
    )
    if gain <= PASSIVITY_ROUNDING:
        return AdmissionItem("passivity", 0.0, None, True, scope, "passive to rounding")
    if gain > allowance:
        msg = (
            f"the coupling map has an apparent gain {gain:.3e}, beyond the model's "
            f"own defects {allowance:.3e}: refine the planes or refuse the pair"
        )
        raise CouplingError(msg)
    return AdmissionItem(
        "passivity",
        gain,
        None,
        True,
        scope,
        "apparent gain within the model's defects, reported as a model defect",
    )


def _vector_source(
    plane: FieldPlane,
    port: InputPort,
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> _Source:
    """Return how a plane's coordinates make a field at a port's nodes.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        A radial or Cartesian plane, or an ordinary modal plane with profiles.
    port : solprop.coupling.port.InputPort
        The general modal port.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    _Source
        The field map, coordinates, native weights, incident norm and items.

    Raises
    ------
    CouplingError
        For an unsupported plane.
    """
    nodes = cast("np.ndarray", port.nodes)
    if plane.kind in ("radial", "cartesian"):
        return _spatial_source(plane, nodes, index, epsilon)
    if plane.kind == "modal" and plane.profiles is not None:
        return _modal_source(plane, nodes, index, epsilon)
    msg = (
        f"vector_paraxial takes a radial or Cartesian plane, or an ordinary modal "
        f"plane with its profiles, not a {plane.kind} plane"
        + (" without profiles" if plane.kind == "modal" else "")
    )
    raise CouplingError(msg)


def _spatial_angular(
    plane: FieldPlane,
    column_energy: Float[np.ndarray, "n_columns n_omega"],
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> AdmissionItem:
    """Return a spatial plane's angular defect, energy-weighted (section 4.6).

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        A radial or Cartesian plane.
    column_energy : numpy.ndarray
        Each column's weighted squared samples per bin.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``angular`` of the incident field.
    """
    space = cast("RadialGrid | FreeGrid", plane.basis.space)
    kappa = np.sqrt(
        np.asarray(space.k_r) ** 2
        if isinstance(space, RadialGrid)
        else np.asarray(space.kperp_sq)
    )
    star = _kappa_star(index, plane.grid.omega, epsilon)
    tail = kappa[:, None] > star[None, :]
    density = np.sum(column_energy, axis=0)
    defect = (epsilon / 8.0) * np.sum(np.where(tail, 0.0, column_energy), axis=0)
    defect += np.sum(np.where(tail, column_energy, 0.0), axis=0)
    relative = np.where(density > 0, defect / np.where(density > 0, density, 1.0), 0.0)
    return AdmissionItem(
        "angular",
        _weighted(relative, density * _bin_like(plane)),
        None,
        True,
        "field",
        "incident field: cone-rule allocation (1 - cos theta*) = model_error/8 "
        "of the cone, not a measured defect, plus the tail",
    )


def _refuse_nonpropagating(
    column_energy: Float[np.ndarray, "n_columns n_omega"],
    admittance: Float[np.ndarray, "n_columns n_omega"],
    occupied: np.ndarray,
) -> None:
    """Refuse a plane with more than rounding in nonpropagating columns.

    Parameters
    ----------
    column_energy : numpy.ndarray
        Each column's weighted squared samples per bin.
    admittance : numpy.ndarray
        Each column's real admittance per bin.
    occupied : numpy.ndarray
        Occupied bins.

    Raises
    ------
    CouplingError
        If the nonpropagating share exceeds 1e-12.
    """
    dead = (admittance <= 0.0) & occupied[None, :]
    total = float(np.sum(column_energy[:, occupied]))
    share = 0.0 if total == 0.0 else float(np.sum(column_energy[dead])) / total
    if share > 1e-12:
        msg = (
            f"the plane puts {share:.3e} of its field norm in columns that do not "
            "propagate: remove that content with its own report before coupling"
        )
        raise CouplingError(msg)


def _spatial_source(
    plane: FieldPlane,
    nodes: Float[np.ndarray, "n_nodes 2"],
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> _Source:
    """Return a radial or Cartesian plane's field map at a guide's nodes.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        A radial or Cartesian plane.
    nodes : numpy.ndarray
        The guide's nodes, m.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    _Source
        The field map and its items.

    Raises
    ------
    CouplingError
        If the nodes reach beyond a window whose exterior is unknown, or the plane
        occupies nonpropagating columns.
    """
    grid = plane.grid
    space = cast("RadialGrid | FreeGrid", plane.basis.space)
    inside = np.hypot(nodes[:, 0], nodes[:, 1]) <= _window(space)
    if not np.all(inside) and plane.exterior != "zero":
        msg = (
            "the port's nodes reach beyond the plane's window, where its exterior "
            "is unknown: widen the plane or declare its exterior zero"
        )
        raise CouplingError(msg)
    samples = fourier.spectrum_to_internal(plane.samples, kind=grid.kind)
    n_columns, n_pol = samples.shape[:2]
    spectral = np.asarray(space.spectral_weights)
    admittance = np.real(cast("np.ndarray", plane.metric.admittance))
    column_energy = spectral[:, None] * np.sum(np.abs(samples) ** 2, axis=1)
    norm = _SCALE * index * np.sum(column_energy, axis=0)
    _refuse_nonpropagating(column_energy, admittance, _occupied(norm))
    functions = _column_functions(space, nodes, inside)
    cache = np.zeros((nodes.shape[0], 2, n_columns * n_pol), dtype=np.complex128)
    for p in range(n_pol):
        cache[:, p, p::n_pol] = functions
    analytic = plane.analytic
    domain = 0.0 if analytic is None else analytic.error
    edge = _window_edge_share(space, samples, inside)
    items = (
        _spatial_angular(plane, column_energy, index, epsilon),
        AdmissionItem(
            "numerical",
            domain + edge,
            None,
            True,
            "field",
            f"source domain/resolution {domain:.1e}, field beyond the window "
            f"{edge:.1e}",
        ),
        _source_metric_item(_bin_like(plane) * norm, plane.spectral_energy(), epsilon),
    )
    return _Source(
        lambda _bin: cache,
        samples.reshape(n_columns * n_pol, grid.n_omega),
        np.repeat(_SCALE * spectral[:, None] * admittance, n_pol, axis=0),
        norm,
        items,
    )


def _modal_source(
    plane: FieldPlane,
    nodes: Float[np.ndarray, "n_nodes 2"],
    index: Float[np.ndarray, " n_omega"],
    epsilon: float,
) -> _Source:
    """Return an ordinary modal plane's synthesised field map at a guide's nodes.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        An ordinary modal plane with its profiles.
    nodes : numpy.ndarray
        The guide's nodes, m.
    index : numpy.ndarray
        Reference index per bin.
    epsilon : float
        The model budget.

    Returns
    -------
    _Source
        The field map and its items.
    """
    grid = plane.grid
    source = cast("ScalarModeBasis", plane.profiles)
    jones, mode_of = _channels(source, plane.samples.shape[1])
    values = source.values(np.hypot(nodes[:, 0], nodes[:, 1]))[mode_of]
    effective = source.effective_index[mode_of]
    root = np.sqrt(np.where(effective > 0, effective, 1.0))
    coordinates = fourier.spectrum_to_internal(plane.samples, kind=grid.kind).reshape(
        -1, grid.n_omega
    )

    def transfer(k: int) -> np.ndarray:
        """Return the field at the nodes per unit true-power coefficient.

        Parameters
        ----------
        k : int
            The bin.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_nodes, 2, n_channels)``, V/m per √W.
        """
        scale = np.sqrt(2.0 / (EPS0 * C)) / root[:, k]
        return values.T[:, None, :] * (jones.T * scale[None, :])[None, :, :]

    scaled = coordinates / root
    norm = index * np.sum(np.abs(scaled) ** 2, axis=0)
    bins = _bin_like(plane)
    items = (
        _modal_angular(
            source,
            jones=jones,
            mode_of=mode_of,
            scaled=scaled,
            norm=norm,
            index=index,
            plane=plane,
            epsilon=epsilon,
        ),
        _source_metric_item(bins * norm, plane.spectral_energy(), epsilon),
    )
    return _Source(transfer, coordinates, np.ones(coordinates.shape), norm, items)


def _modal_angular(  # noqa: PLR0913 - the source set, its field and the measure
    source: ScalarModeBasis,
    *,
    jones: Float[np.ndarray, "n_channels 2"],
    mode_of: np.ndarray,
    scaled: Complex[np.ndarray, "n_channels n_omega"],
    norm: Float[np.ndarray, " n_omega"],
    index: Float[np.ndarray, " n_omega"],
    plane: FieldPlane,
    epsilon: float,
) -> AdmissionItem:
    """Return a modal source field's angular defect, energy-weighted.

    Parameters
    ----------
    source : solprop.coupling.models.ScalarModeBasis
        The source profiles.
    jones : numpy.ndarray
        Each source channel's Jones vector.
    mode_of : numpy.ndarray
        Each source channel's mode.
    scaled : numpy.ndarray
        The vacuum-normalised source amplitudes ``b / sqrt(Re n_eff)``.
    norm : numpy.ndarray
        The incident norm in the common measure per bin.
    index : numpy.ndarray
        Reference index per bin.
    plane : solprop.coupling.plane.FieldPlane
        The source plane.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``angular`` of the incident field.
    """
    star = _kappa_star(index, plane.grid.omega, epsilon)
    defect = np.zeros(plane.grid.n_omega)
    for k in np.flatnonzero(_occupied(norm)):
        tail = source.gram() - source.cone_gram(float(star[k]))
        tail_c = tail[np.ix_(mode_of, mode_of)] * (jones @ jones.T)
        tail_energy = index[k] * float(
            np.real(scaled[:, k].conj() @ tail_c @ scaled[:, k])
        )
        defect[k] = (epsilon / 8.0) * (norm[k] - tail_energy) + tail_energy
    relative = np.where(norm > 0, defect / np.where(norm > 0, norm, 1.0), 0.0)
    return AdmissionItem(
        "angular",
        _weighted(relative, norm * _bin_like(plane)),
        None,
        True,
        "field",
        "incident field: cone-rule allocation (1 - cos theta*) = model_error/8 "
        "of the cone, not a measured defect, plus the tail",
    )


def _window(space: RadialGrid | FreeGrid) -> float:
    """Return the radius inside which a spatial plane's interpolant is its field, m.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.

    Returns
    -------
    float
        The radial window, or the smaller Cartesian half-width.
    """
    if isinstance(space, RadialGrid):
        return float(space.hankel.radius)
    return float(min(np.max(np.abs(space.x)), np.max(np.abs(space.y))))


def _column_functions(
    space: RadialGrid | FreeGrid,
    nodes: Float[np.ndarray, "n_nodes 2"],
    inside: np.ndarray,
) -> Complex[np.ndarray, "n_nodes n_columns"]:
    """Return each column's internal-convention field at given nodes, per unit sample.

    On a radial grid the inverse Hankel series
    ``f(r) = sum_j w_j H_j J0(k_j r) / (2 pi)``; on a Cartesian grid the inverse
    DFT's trigonometric interpolant ``f(x) = sum_j S_j exp(i k_j (x - x0)) / N``.
    Zero at nodes outside the window.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.
    nodes : numpy.ndarray
        Node positions ``(x, y)``, m, shape ``(n_nodes, 2)``.
    inside : numpy.ndarray
        Nodes inside the window.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_columns)``.
    """
    if isinstance(space, RadialGrid):
        radius = np.hypot(nodes[:, 0], nodes[:, 1])
        values = (
            np.asarray(space.spectral_weights)[None, :]
            * j0(np.outer(radius, space.k_r))
            / (2.0 * np.pi)
        )
        return np.where(inside[:, None], values, 0.0).astype(np.complex128)
    kx = np.tile(space.k_x, space.n_y)
    ky = np.repeat(space.k_y, space.n_x)
    phase = np.exp(
        1j
        * (
            np.outer(nodes[:, 0] - space.x[0], kx)
            + np.outer(nodes[:, 1] - space.y[0], ky)
        )
    ) / (space.n_x * space.n_y)
    return np.where(inside[:, None], phase, 0.0)


def _window_edge_share(
    space: RadialGrid | FreeGrid,
    samples: Complex[np.ndarray, "n_columns n_pol n_omega"],
    inside: np.ndarray,
) -> float:
    """Return a bound on the share of the field the port's nodes miss beyond the window.

    Zero when every node lies inside the window; otherwise the plane's energy
    share in the outer tenth of its window, which bounds the tail the zero
    exterior discards to leading order.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.
    samples : numpy.ndarray
        Internal-convention columns.
    inside : numpy.ndarray
        Port nodes inside the window.

    Returns
    -------
    float
        Dimensionless.
    """
    if np.all(inside):
        return 0.0
    if isinstance(space, RadialGrid):
        nodal = space.hankel.inverse(np.moveaxis(samples, 0, -1))
        weights = np.asarray(space.radial_weights)
        radius = np.asarray(space.r)
        energy = np.sum(np.abs(nodal) ** 2, axis=(0, 1)) * weights
        outer = radius > 0.9 * _window(space)
    else:
        shaped = samples.reshape(space.n_y, space.n_x, *samples.shape[1:])
        nodal = np.fft.ifft2(shaped, axes=(0, 1)).reshape(samples.shape)
        energy = np.sum(np.abs(nodal) ** 2, axis=(1, 2))
        outer = np.asarray(space.r) > 0.9 * _window(space)
    total = float(np.sum(energy))
    return 0.0 if total == 0.0 else float(np.sum(energy[outer]) / total)


def _probe_bins(
    occupied: np.ndarray, norm: Float[np.ndarray, " n_omega"]
) -> np.ndarray:
    """Return up to :data:`_TARGET_BINS` occupied bins spanning the band.

    Parameters
    ----------
    occupied : numpy.ndarray
        Occupied bins.
    norm : numpy.ndarray
        Incident norm per bin.

    Returns
    -------
    numpy.ndarray
        Bin indices: the edges, the peak and points between.
    """
    live = np.flatnonzero(occupied)
    if live.size == 0:
        return live
    picks = {int(live[0]), int(live[-1]), int(np.argmax(norm))}
    picks.update(
        int(one)
        for one in live[np.linspace(0, live.size - 1, _TARGET_BINS).astype(int)]
    )
    return np.array(sorted(picks))[: max(_TARGET_BINS, 3)]


def _closure_item(
    electric: np.ndarray,
    magnetic: np.ndarray,
    weights: Float[np.ndarray, " n_nodes"],
    admittance: Float[np.ndarray, " n_omega"],
    probes: np.ndarray,
    epsilon: float,
) -> AdmissionItem:
    r"""Return the H-closure's energy discrepancy over the retained target subspace.

    :math:`\delta^2 = \lambda_{\max}(X, Y)` with :math:`X` the Gram of
    :math:`\mathbf h_t - Y_0\hat z\times\mathbf e_t` and :math:`Y` that of
    :math:`Y_0\hat z\times\mathbf e_t`; the energy discrepancy is
    :math:`2\delta + \delta^2` (section 4.7).

    Parameters
    ----------
    electric, magnetic : numpy.ndarray
        Basis E and H at the nodes, internal convention.
    weights : numpy.ndarray
        Node area weights.
    admittance : numpy.ndarray
        :math:`Y_0` per bin, A/V.
    probes : numpy.ndarray
        The bins evaluated.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``closure``, capped at ``epsilon / 8``.
    """
    worst = 0.0
    for k in probes:
        ex, ey = electric[:, 0, :, k], electric[:, 1, :, k]
        hx, hy = magnetic[:, 0, :, k], magnetic[:, 1, :, k]
        closed_x, closed_y = -admittance[k] * ey, admittance[k] * ex
        dx, dy = hx - closed_x, hy - closed_y
        gram_d = np.einsum("q,qm,qn->mn", weights, dx.conj(), dx) + np.einsum(
            "q,qm,qn->mn", weights, dy.conj(), dy
        )
        gram_c = np.einsum(
            "q,qm,qn->mn", weights, closed_x.conj(), closed_x
        ) + np.einsum("q,qm,qn->mn", weights, closed_y.conj(), closed_y)
        worst = max(worst, _whitened_extremes(gram_d, gram_c)[1])
    delta = math.sqrt(max(worst, 0.0))
    return AdmissionItem(
        "closure",
        2.0 * delta + delta * delta,
        epsilon / 8.0,
        True,
        "map",
        f"H_t = Y0 z x E_t against provider H, field error {delta:.2e}",
    )


def _target_angular_vector(
    electric: np.ndarray,
    weights: Float[np.ndarray, " n_nodes"],
    port: InputPort,
    index: Float[np.ndarray, " n_omega"],
    probes: np.ndarray,
    epsilon: float,
) -> AdmissionItem:
    """Return a general guide's retained subspace angular defect from its node FT.

    The transverse E of each channel is Fourier transformed by its node quadrature
    on a polar rule inside the cone; the tail is the node Gram minus the cone.

    Parameters
    ----------
    electric : numpy.ndarray
        Basis E at the nodes, internal convention.
    weights : numpy.ndarray
        Node area weights.
    port : solprop.coupling.port.InputPort
        The port.
    index : numpy.ndarray
        Reference index per bin.
    probes : numpy.ndarray
        The bins evaluated.
    epsilon : float
        The model budget.

    Returns
    -------
    AdmissionItem
        ``angular`` of the target subspace.
    """
    nodes = cast("np.ndarray", port.nodes)
    extent = float(np.max(np.hypot(nodes[:, 0], nodes[:, 1])))
    star = _kappa_star(index, port.grid.omega, epsilon)
    angles = 2.0 * np.pi * np.arange(16) / 16
    worst = 0.0
    for k in probes:
        radial, area = radial_rule([], float(star[k]) * extent, panel=0.5 * math.pi)
        kappa_r = radial / extent
        area_k = area / extent**2 / 16
        kx = np.outer(kappa_r, np.cos(angles)).ravel()
        ky = np.outer(kappa_r, np.sin(angles)).ravel()
        cone_w = np.repeat(area_k, 16) / (2.0 * np.pi) ** 2
        transverse = electric[:, :2, :, k]
        gram = np.einsum("q,qcm,qcn->mn", weights, transverse.conj(), transverse)
        weighted = weights[:, None, None] * transverse
        cone = np.zeros_like(gram)
        # Cone points in blocks: the cone x node phase matrix is never whole.
        step = max(1, _PHASE_BLOCK // nodes.shape[0])
        for start in range(0, kx.size, step):
            block = slice(start, start + step)
            phase = np.exp(
                -1j
                * (np.outer(kx[block], nodes[:, 0]) + np.outer(ky[block], nodes[:, 1]))
            )
            spectra = np.einsum("kq,qcm->kcm", phase, weighted)
            cone += np.einsum("k,kcm,kcn->mn", cone_w[block], spectra.conj(), spectra)
        tail = gram - cone
        defect = (epsilon / 8.0) * gram + (1.0 - epsilon / 8.0) * tail
        worst = max(worst, _whitened_extremes(defect, gram)[1])
    return AdmissionItem(
        "angular",
        worst,
        None,
        True,
        "map",
        "retained target subspace: cone-rule allocation (1 - cos theta*) = "
        "model_error/8 of the cone, not a measured defect, plus the tail",
    )
