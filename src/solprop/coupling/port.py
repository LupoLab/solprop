# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The receiving side of a transfer: a simulation's input port and its launch.

An :class:`InputPort` describes what a simulation can receive at its entrance,
already sampled (SEP 31, section 5.1): the route, the temporal lattice and basis
with their identities, the entrance plane and the frame accumulated there, the
native metric Q, the reference index of the medium, and the coupling models it
admits. :meth:`solprop.driver.simulation.Simulation.input_port` returns one; the
translation layer fills it. It is a P-layer value type, so that a coupling computed
with NumPy alone never imports the translation layer.

A :class:`CouplingLaunch` is bound to one port: it carries the port, the target
coefficients and the reference origin the launched run inherits, and
:meth:`solprop.driver.simulation.Simulation.with_input` refuses it for any other port,
whatever the array shapes.

:func:`same_basis` is the identity model of section 4.3: a plane whose basis,
lattice, orientation and metric are the port's is copied as it stands, and its
reference becomes the launched run's origin. Phase 3 adds the general ``couple``.
"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.coupling.plane import (
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    PlaneKind,
    PlaneMetric,
    PlaneReference,
    lattice_identity,
)
from solprop.exceptions import CouplingError
from solprop.grid import TimeGrid

__all__ = [
    "ENTRANCE_TOLERANCE",
    "METRIC_TOLERANCE",
    "CouplingLaunch",
    "InputPort",
    "same_basis",
]

ENTRANCE_TOLERANCE = 1e-12
"""Relative agreement of two entrance positions, of the larger of 1 m and abs(z).

A launch names its entrance by position; within this, two positions are the same
plane, as a plane the run reached by summing steps sits within a few ulp of it.
"""

METRIC_TOLERANCE = 1e-10
"""Largest relative difference of two admittance metrics that counts as one medium.

The same medium at the same plane evaluates its admittance through one arithmetic
path (SEP 34) and agrees to rounding; a different medium differs by its index, a
few parts in a million for a gas. Anything between is a different medium.
"""


def _occupied_outside(
    samples: Complex[np.ndarray, "n_modes n_pol n_omega"], grid: TimeGrid
) -> float:
    """Return the share of a record's squared samples outside a grid's active band.

    Parameters
    ----------
    samples : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``.
    grid : solprop.grid.TimeGrid
        The receiving grid, on the same lattice.

    Returns
    -------
    float
        The share, zero for a zero record.
    """
    power = np.sum(np.abs(samples) ** 2, axis=(0, 1))
    total = float(np.sum(power))
    return 0.0 if total == 0.0 else float(np.sum(power[~grid.active_mask]) / total)


@dataclass(frozen=True, eq=False)
class InputPort:
    """What a simulation can receive at its entrance, already sampled.

    Attributes
    ----------
    route : {'modal', 'general_modal', 'radial', 'cartesian', 'planar'}
        The receiving representation.
    grid : solprop.grid.TimeGrid
        The receiving temporal grid.
    basis : PlaneBasis
        The receiving basis, with its identity, labels and transverse grid.
    shape : tuple of int
        The launch's ``(n_modes, n_pol, n_omega)``.
    geometry : PlaneGeometry
        The entrance plane: its position along the receiving path and side.
    frame : PlaneReference
        The receiving run's own frame accumulated from its path origin to the
        entrance, without any inherited origin: zero at a path's start.
    metric : PlaneMetric
        The native metric Q at the entrance: the identity for ordinary
        true-power coefficients, the basis Gram (optics convention) for a general
        guide, the axial admittance for a spatial grid.
    reference_index : numpy.ndarray or None, optional
        Real phase index of the entrance medium at each bin, shape
        ``(n_omega,)``, where the route defines one (a spatial medium).
    nodes : numpy.ndarray or None, optional
        Transverse nodes ``(x, y)`` the target basis is sampled at, m, shape
        ``(n_nodes, 2)``; None where phase 2 samples no profiles.
    node_weights : numpy.ndarray or None, optional
        Their area weights, m^2, shape ``(n_nodes,)``.
    electric : numpy.ndarray or None, optional
        The target basis's electric field at the nodes, optics convention, in
        V/(m sqrt(W)), shape ``(n_nodes, 3, n_modes, n_omega)``: a general guide's
        E reference at its entrance.
    magnetic : numpy.ndarray or None, optional
        Its magnetic field in A/(m sqrt(W)), the same shape.
    models : tuple of str, optional
        The coupling models the port admits. Phase 2 admits ``'same_basis'``.

    Raises
    ------
    CouplingError
        If the shape disagrees with the grid or the basis kind with the route.
    """

    route: PlaneKind
    grid: TimeGrid
    basis: PlaneBasis
    shape: tuple[int, int, int]
    geometry: PlaneGeometry
    frame: PlaneReference
    metric: PlaneMetric
    reference_index: Float[np.ndarray, " n_omega"] | None = None
    nodes: Float[np.ndarray, "n_nodes 2"] | None = None
    node_weights: Float[np.ndarray, " n_nodes"] | None = None
    electric: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"] | None = None
    magnetic: Complex[np.ndarray, "n_nodes 3 n_modes n_omega"] | None = None
    models: tuple[str, ...] = ("same_basis",)

    def __post_init__(self) -> None:
        """Check the shape, the basis and the reference index."""
        if self.basis.kind != self.route:
            msg = f"a {self.route} port needs a {self.route} basis"
            raise CouplingError(msg)
        shape = tuple(int(one) for one in self.shape)
        if len(shape) != 3 or shape[2] != self.grid.n_omega or min(shape) < 1:
            msg = f"a port's shape is (n_modes, n_pol, n_omega), got {self.shape}"
            raise CouplingError(msg)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "models", tuple(self.models))
        if self.reference_index is not None:
            index = np.array(self.reference_index, dtype=np.float64, copy=True)
            if index.shape != (self.grid.n_omega,) or not np.all(np.isfinite(index)):
                msg = "a reference index is finite, one value per bin"
                raise CouplingError(msg)
            index.setflags(write=False)
            object.__setattr__(self, "reference_index", index)
        for name, dtype in (
            ("nodes", np.float64),
            ("node_weights", np.float64),
            ("electric", np.complex128),
            ("magnetic", np.complex128),
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=dtype, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, name, array)

    @property
    def lattice(self) -> str:
        """Return the identity of the receiving temporal lattice.

        Returns
        -------
        str
            :func:`~solprop.coupling.plane.lattice_identity` of `grid`.
        """
        return lattice_identity(self.grid)

    @property
    def identity(self) -> str:
        """Return the port's identity: route, lattice, basis and entrance.

        Returns
        -------
        str
            A SHA-256 hex digest; two ports with the same identity accept the same
            launch.
        """
        record = {
            "route": self.route,
            "lattice": self.lattice,
            "basis": self.basis.identity,
            "shape": list(self.shape),
            "z": self.geometry.z,
            "side": self.geometry.side,
        }
        text = json.dumps(record, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def accepts(self, other: "InputPort") -> str | None:  # noqa: UP037 - beartype evaluates the class body
        """Return why a launch bound to `other` cannot enter this port, or None.

        Parameters
        ----------
        other : InputPort
            The port a launch was bound to.

        Returns
        -------
        str or None
            The first disagreement -- route, lattice, basis, shape, entrance,
            orientation or metric (the medium) -- named; None when the two are the
            same port.
        """
        for name, mine, theirs in (
            ("route", self.route, other.route),
            ("lattice", self.lattice, other.lattice),
            ("basis", self.basis.identity, other.basis.identity),
            ("shape", self.shape, other.shape),
            ("side", self.geometry.side, other.geometry.side),
        ):
            if mine != theirs:
                return f"the launch's {name} is not this simulation's"
        scale = max(1.0, abs(self.geometry.z))
        if abs(self.geometry.z - other.geometry.z) > ENTRANCE_TOLERANCE * scale:
            return (
                f"the launch enters at z = {other.geometry.z!r} m and this simulation "
                f"at z = {self.geometry.z!r} m"
            )
        if not self.geometry.oriented_as(other.geometry):
            return "the launch's entrance plane is oriented differently"
        try:
            _check_same_medium(other.metric, self.metric)
        except CouplingError as refused:
            return f"the launch's metric is not this simulation's: {refused}"
        return None


@dataclass(frozen=True, eq=False)
class CouplingLaunch:
    """A launch bound to one input port.

    Attributes
    ----------
    port : InputPort
        The port it was computed for; its grid, basis, metric and entrance are the
        launch's identity.
    samples : numpy.ndarray
        The target coefficients, optics-convention stored spectrum, shape
        ``port.shape``, read-only, in the port's native units.
    origin : PlaneReference
        The reference of the launched run's path origin: the launched spectrum's
        reference minus the frame the port has accumulated at its entrance. The
        launched run's planes add their own frame to it.
    model : str
        The coupling model that produced it.
    provenance : collections.abc.Mapping
        JSON-ready record of where it came from: the source plane's provenance and
        the model, recorded in the launched run's configuration.

    Raises
    ------
    CouplingError
        If the samples do not have the port's shape or are not finite.
    """

    port: InputPort
    samples: Complex[np.ndarray, "n_modes n_pol n_omega"]
    origin: PlaneReference
    model: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        """Own read-only samples of the port's shape."""
        samples = np.array(self.samples, dtype=np.complex128, copy=True)
        if samples.shape != self.port.shape or not np.all(np.isfinite(samples)):
            msg = (
                f"a launch holds finite samples of its port's shape {self.port.shape}, "
                f"got {samples.shape}"
            )
            raise CouplingError(msg)
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def __repr__(self) -> str:
        """Return a one-line description without arrays.

        Returns
        -------
        str
            The model, route and entrance.
        """
        return (
            f"CouplingLaunch({self.model}, {self.port.route}, "
            f"z={self.port.geometry.z:.6g} m)"
        )

    def internal(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the samples in the internal convention a simulation takes.

        Returns
        -------
        numpy.ndarray
            :func:`solprop.fourier.spectrum_to_internal` of `samples`.
        """
        return np.asarray(
            fourier.spectrum_to_internal(self.samples, kind=self.port.grid.kind),
            dtype=np.complex128,
        )

    def record(self) -> dict[str, object]:
        """Return the coupling provenance a launched run records.

        Returns
        -------
        dict
            The model, the port identity, the origin and the source provenance.
        """
        return {
            "model": self.model,
            "port": self.port.identity,
            "origin": self.origin.record(),
            "source": dict(self.provenance),
        }


def same_basis(plane: FieldPlane, port: InputPort) -> CouplingLaunch:
    """Launch a plane into a port of the same basis, unchanged.

    The identity model of SEP 31 section 4.3: exact coordinate recovery in a
    checked common reference, which here is the identical basis on the identical
    lattice, at an identically oriented plane, in the same medium. The coefficients
    are copied as they stand and the plane's reference becomes the launched run's
    origin, so the launched run continues the source's absolute time and carrier
    phase without moving the pulse in its window. Anything else is a different
    calculation, refused by name: a different lattice needs
    :func:`~solprop.coupling.remap.regrid` first, a different basis or medium a coupling
    model or an interface.

    Parameters
    ----------
    plane : FieldPlane
        The source plane.
    port : InputPort
        The receiving port.

    Returns
    -------
    CouplingLaunch
        The launch, bound to `port`.

    Raises
    ------
    CouplingError
        If the port does not admit the model; if the route, basis, lattice, shape,
        orientation or medium differ; or if the plane carries light outside the
        port's active band.
    """
    if "same_basis" not in port.models:
        msg = f"this port admits {port.models}, not same_basis"
        raise CouplingError(msg)
    if plane.kind != port.route or plane.basis.identity != port.basis.identity:
        msg = (
            f"a {plane.kind} plane of basis {plane.basis.identity[:12]} is not the "
            f"{port.route} port's basis {port.basis.identity[:12]}: a different basis "
            "is a coupling, not a relaunch"
        )
        raise CouplingError(msg)
    if lattice_identity(plane.grid) != port.lattice or not np.array_equal(
        plane.grid.omega, port.grid.omega
    ):
        msg = "the plane's temporal lattice is not the port's: regrid it first"
        raise CouplingError(msg)
    if plane.samples.shape != port.shape:
        msg = f"the plane's shape {plane.samples.shape} is not the port's {port.shape}"
        raise CouplingError(msg)
    if not plane.geometry.oriented_as(port.geometry):
        msg = "the plane and the entrance are oriented differently"
        raise CouplingError(msg)
    _check_same_medium(plane.metric, port.metric)
    outside = _occupied_outside(plane.samples, port.grid)
    floor = (plane.grid.n_omega * float(np.finfo(float).eps)) ** 2
    if outside > floor:
        msg = (
            f"{outside:.3e} of the plane lies outside the port's active band, which "
            "the receiving run would remove: regrid with crop_band=True to remove "
            "it explicitly"
        )
        raise CouplingError(msg)
    provenance = {
        "plane": plane.provenance.record(),
        "reference": plane.reference.record(),
    }
    return CouplingLaunch(
        port,
        plane.samples,
        plane.reference.minus(port.frame),
        "same_basis",
        provenance,
    )


def _check_same_medium(source: PlaneMetric, target: PlaneMetric) -> None:
    """Refuse a relaunch across a change of medium or metric.

    Parameters
    ----------
    source : PlaneMetric
        The plane's metric.
    target : PlaneMetric
        The port's metric.

    Raises
    ------
    CouplingError
        If the kinds differ, or the admittances or Grams differ by more than
        :data:`METRIC_TOLERANCE`.
    """
    if source.kind == "unknown":
        msg = (
            f"the plane's metric is unknown ({source.reason}), so its medium cannot "
            "be checked"
        )
        raise CouplingError(msg)
    if source.kind != target.kind:
        msg = f"the plane's metric is {source.kind} and the port's {target.kind}"
        raise CouplingError(msg)
    for name in ("admittance", "matrix", "weights"):
        mine = getattr(source, name)
        theirs = getattr(target, name)
        if mine is None:
            continue
        mine = cast("np.ndarray", mine)
        theirs = cast("np.ndarray", theirs)
        scale = float(np.max(np.abs(theirs))) if theirs.size else 0.0
        if mine.shape != theirs.shape or (
            scale > 0.0
            and float(np.max(np.abs(mine - theirs))) > METRIC_TOLERANCE * scale
        ):
            msg = (
                f"the plane's {name} differs from the port's: a change of medium "
                "needs an interface, not a relaunch"
            )
            raise CouplingError(msg)
    if not math.isclose(source.scale, target.scale, rel_tol=1e-15):
        msg = "the plane's metric scale is not the port's"
        raise CouplingError(msg)
