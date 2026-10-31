# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sample a translated problem's entrance into a coupling port (SEP 31).

Two products of the translation layer feed plane exchange. :func:`frame_record` is
the persisted frame integral of a run: the antiderivatives of the frame quantities
its linear operator removed, so that any saved plane's reference can be rebuilt
without the run (SEP 31). :func:`input_port` fills the
:class:`solprop.coupling.port.InputPort` of a translated problem's entrance: its native
metric at that plane, the frame it has accumulated there, the entrance medium's
index, the coupling models it admits and what they need: an ordinary mode set's
scalar profiles, and for a general guide that does not move its E/H reference at the
nodes and the weak-guidance contrast of its regions (SEP 31, section 4.3). Both read
the contract's own leaves, so a port and a saved plane of the same run see the same
numbers.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
from jaxtyping import Float

from solprop.constants import EPS0, C
from solprop.coupling import (
    InputPort,
    PlaneBasis,
    PlaneGeometry,
    PlaneKind,
    PlaneMetric,
    PlaneReference,
    ScalarModeBasis,
    reference_at,
)
from solprop.exceptions import CouplingError, TranslationError
from solprop.grid import TimeGrid
from solprop.modes.guide import ModalGuide, StepIndexProvider
from solprop.modes.protocol import Mode
from solprop.spatial import FluxSide, SpatialMedium
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.coupled import metric_at
from solprop.translate.linop import LinearOperator
from solprop.translate.modal_tables import ModalTransformTables
from solprop.translate.problem import KernelProblem
from solprop.translate.spatial import axial_admittance

__all__ = [
    "frame_record",
    "guide_contrast",
    "input_port",
    "port_metric",
    "step_index_frequency",
]


def frame_record(
    linear: LinearOperator, frequency_origin: float
) -> dict[str, object] | None:
    r"""Return the persisted frame integral of a run's linear operator.

    The operator removed :math:`R = \beta_0(z) + \beta_1(z)(\omega - \omega_o)`
    (``docs/dev/conventions.md`` section 2; :class:`solprop.linear.Frame`), with
    :math:`\beta_0` zero where the frame keeps it. The record holds the exact
    Chebyshev antiderivatives :math:`\Phi_0 = \int_0^z\beta_0` and
    :math:`T = \int_0^z\beta_1` of the contract's own frame tables and the frequency
    origin, which is all :func:`solprop.coupling.plane.reference_at` needs.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        The contract's linear operator.
    frequency_origin : float
        :math:`\omega_o` in rad/s: the frame's reference frequency on an envelope
        grid, zero on a field grid.

    Returns
    -------
    dict or None
        JSON-ready ``frequency_origin`` and the ``phase`` (rad) and ``delay`` (s)
        antiderivatives, each its ``breakpoints`` (m) and per-interval
        ``coefficients``, lowest degree first; None for an operator supplied
        without its frame tables.
    """
    if linear.beta0 is None or linear.beta1 is None:
        return None
    return {
        "frequency_origin": float(frequency_origin),
        "phase": _antiderivative(linear.beta0),
        "delay": _antiderivative(linear.beta1),
    }


def _antiderivative(values: PiecewiseChebyshev) -> dict[str, object]:
    """Return the edges and coefficients of a scalar table's antiderivative.

    Parameters
    ----------
    values : solprop.translate.chebyshev.PiecewiseChebyshev
        A scalar frame table.

    Returns
    -------
    dict
        JSON-ready ``breakpoints`` (m) and per-interval ``coefficients``, lowest
        degree first, of the exact antiderivative vanishing at the first edge.
    """
    integral = values.integral()
    coefficients = np.real(integral.coefficients).reshape(integral.degrees.size, -1)
    return {
        "breakpoints": [float(one) for one in integral.breakpoints],
        "coefficients": coefficients.tolist(),
    }


def port_metric(problem: KernelProblem, grid: TimeGrid, z: float) -> PlaneMetric:
    """Return the native metric of a translated problem's state at a plane.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    z : float
        The plane's position, m.

    Returns
    -------
    solprop.coupling.PlaneMetric
        The axial admittance of a spatial run's plane measure at ``z`` (SEP 34),
        the optics-convention Gram ``conj(W)`` of a modal basis that declares one,
        and the identity for true-power coefficients otherwise.
    """
    geometry = problem.transform.geometry
    if geometry is not None:
        if problem.flux is None:
            msg = "a spatial problem carries its plane measure"
            raise TranslationError(msg)
        admittance = axial_admittance(problem.flux.at(z), grid.omega, slice(None))
        return PlaneMetric(
            "admittance",
            admittance=admittance,
            weights=geometry.spectral_weights,
            scale=1.0 if geometry.kind == "slab" else 0.5 * EPS0 * C,
        )
    matrix = metric_at(problem.linear.modal_metric, z)
    if matrix is None:
        return PlaneMetric("identity")
    # The state is internal-convention; a plane's samples are optics-convention, a
    # conjugation away, so the same quadratic form takes conj(W).
    return PlaneMetric("dense", matrix=np.conj(matrix))


def input_port(  # noqa: PLR0913 - the problem and each part of the entrance it does not hold
    problem: KernelProblem,
    grid: TimeGrid,
    *,
    route: PlaneKind,
    basis: PlaneBasis,
    frame: dict[str, object] | None,
    medium: SpatialMedium | None = None,
    tables: ModalTransformTables | None = None,
    side: FluxSide | None = None,
    z: float | None = None,
    modes: Sequence[Mode] = (),
    guide: ModalGuide | None = None,
) -> InputPort:
    """Fill the input port of a translated problem's entrance.

    The models it admits: ``same_basis`` always; ``scalar_paraxial`` for an
    ordinary mode set whose scalar profiles exist (a capillary's HE1m modes, a
    step-index fibre's HE1m modes in the weak-guidance limit), with the core's index
    as the reference; ``vector_paraxial`` for a general guide whose E/H the port
    carries, with its core's index and its regions' contrast.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem; its launch position is the entrance.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    route : {'modal', 'general_modal', 'radial', 'cartesian', 'planar'}
        The receiving representation.
    basis : solprop.coupling.PlaneBasis
        The receiving basis and its identity, as the driver computes it from the
        run's record.
    frame : dict or None
        :func:`frame_record` of the problem; None leaves the entrance frame
        uncalibrated.
    medium : solprop.spatial.SpatialMedium or None, optional
        The bulk medium of a spatial route, for the reference index.
    tables : solprop.translate.modal_tables.ModalTransformTables or None, optional
        The E/H reference of a general guide; sampled into the port when the guide
        does not move.
    side : {'before', 'after'} or None, optional
        Side of a declared jump at the entrance.
    z : float or None, optional
        An interior entrance, m, inside the problem's span; None is its launch
        position.
    modes : sequence of solprop.modes.protocol.Mode, optional
        An ordinary modal run's modes, for their scalar profiles.
    guide : solprop.modes.guide.ModalGuide or None, optional
        A general modal run's guide, for its core index and contrast.

    Returns
    -------
    solprop.coupling.InputPort
        The port and the models it admits.
    """
    z = float(problem.initial.z) if z is None else float(z)
    record: dict[str, object] = {
        "origin": PlaneReference().record(),
        "frame": frame,
    }
    entrance = reference_at(record, z)
    index = None
    if medium is not None:
        live = grid.omega > 0
        values = np.zeros(grid.n_omega)
        values[live] = np.real(medium.index(grid.omega[live], z))
        index = values
    nodes = weights = electric = magnetic = None
    models: tuple[str, ...] = ("same_basis",)
    profiles = contrast = frequency = None
    if tables is not None:
        synthesis = tables.synthesis
        moves = synthesis is not None and synthesis.shape[0] * synthesis.shape[1] > 1
        if not moves:
            nodes = tables.node_xy[0]
            weights = tables.weights[0]
            # Internal-convention reference, conjugated to the optics convention of
            # a plane's samples.
            electric = np.conj(tables.electric[0])
            magnetic = np.conj(tables.magnetic[0])
            if guide is not None:
                index, contrast = guide_contrast(guide, grid)
                frequency = step_index_frequency(guide, grid, z)
                models = ("same_basis", "vector_paraxial")
    if route == "modal" and modes:
        try:
            profiles = ScalarModeBasis.from_modes(modes, grid, z)
        except CouplingError:
            # A family with angular structure has no scalar profiles; the port then
            # admits only the identity model, and couple names why.
            profiles = None
        if profiles is not None:
            index = np.asarray(profiles.core_index)
            models = ("same_basis", "scalar_paraxial")
            if profiles.family == "step_index":
                contrast = _scalar_contrast(profiles, index)
    shape = cast("tuple[int, int, int]", tuple(int(one) for one in problem.shape[1:]))
    return InputPort(
        route,
        grid,
        basis,
        shape,
        PlaneGeometry(z, side),
        entrance,
        port_metric(problem, grid, z),
        index,
        nodes,
        weights,
        electric,
        magnetic,
        models,
        profiles,
        contrast,
        frequency,
    )


def _scalar_contrast(profiles: ScalarModeBasis, index: np.ndarray) -> np.ndarray:
    """Return a step-index profile set's weak-guidance contrast at each bin.

    Parameters
    ----------
    profiles : solprop.coupling.ScalarModeBasis
        A step-index LP set, carrying its core and cladding indices.
    index : numpy.ndarray
        The reference index per bin.

    Returns
    -------
    numpy.ndarray
        ``max |n^2 - n_ref^2| / (2 n_ref^2)`` over core and cladding, zero where
        the reference index is zero, shape ``(n_omega,)``.
    """
    live = index > 0
    contrast = np.zeros(index.shape)
    for values in (profiles.core_index, profiles.cladding_index):
        values = cast("np.ndarray", values)
        contrast[live] = np.maximum(
            contrast[live],
            np.abs(values[live] ** 2 - index[live] ** 2) / (2.0 * index[live] ** 2),
        )
    return contrast


def step_index_frequency(
    guide: ModalGuide, grid: TimeGrid, z: float
) -> Float[np.ndarray, " n_omega"] | None:
    r"""Return a step-index guide's normalised frequency at its entrance, per bin.

    :math:`V = (\omega/c)\,a(z)\sqrt{n_{\rm co}^2 - n_{\rm cl}^2}`, with the core
    radius at the entrance and the real parts of the core's and the cladding's
    declared laws, is what the recorded vector envelope's range
    (:data:`solprop.coupling.envelopes.VECTOR_STEP_INDEX_ENVELOPE`) is stated in.

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        The guide.
    grid : solprop.grid.TimeGrid
        The run's grid.
    z : float
        The entrance, m.

    Returns
    -------
    numpy.ndarray or None
        V at each bin, shape ``(n_omega,)``, zero off the active positive band;
        None for a guide whose provider is not a
        :class:`~solprop.modes.guide.StepIndexProvider`.
    """
    provider = guide.field_provider
    if not isinstance(provider, StepIndexProvider):
        return None
    live = grid.active_mask & (grid.omega > 0)
    omega = grid.omega[live]
    radius = float(provider.modes[0].core_radius(np.asarray([z]))[0])
    core, cladding = (
        np.real(np.asarray(region.material.permittivity(omega)))
        for region in provider.regions
    )
    frequency = np.zeros(grid.n_omega)
    frequency[live] = omega / C * radius * np.sqrt(np.maximum(core - cladding, 0.0))
    return frequency


def guide_contrast(guide: ModalGuide, grid: TimeGrid) -> tuple[np.ndarray, np.ndarray]:
    r"""Return a general guide's core index and weak-guidance contrast at each bin.

    The core is the region on the axis, whose real phase index is the reference
    :math:`n_{\rm ref}`; the contrast is
    :math:`\Delta = \max|n^2 - n_{\rm ref}^2|/(2n_{\rm ref}^2)` over every region's
    law and, for a radial-law provider, over its core profile sampled at 33 radii
    (SEP 31, section 4.3).

    Parameters
    ----------
    guide : solprop.modes.guide.ModalGuide
        The guide.
    grid : solprop.grid.TimeGrid
        The run's grid.

    Returns
    -------
    tuple of numpy.ndarray
        The reference index and the contrast, each shape ``(n_omega,)``, zero off
        the active positive band.
    """
    live = grid.active_mask & (grid.omega > 0)
    omega = grid.omega[live]
    index = np.zeros(grid.n_omega)
    index[live] = np.real(np.sqrt(guide.regions[0].material.permittivity(omega)))
    squares = [
        np.real(np.asarray(region.material.permittivity(omega)))
        for region in guide.regions
    ]
    radial = getattr(guide.field_provider, "permittivity", None)
    if radial is not None and callable(getattr(radial, "permittivity", None)):
        fractions = np.linspace(0.0, 1.0, 33)
        profile = np.array(
            [np.real(radial.permittivity(float(one), fractions)) for one in omega]
        )
        squares.extend([profile.min(axis=1), profile.max(axis=1)])
    reference = index[live] ** 2
    contrast = np.zeros(grid.n_omega)
    contrast[live] = np.max(
        [np.abs(one - reference) / (2.0 * reference) for one in squares], axis=0
    )
    return index, contrast
