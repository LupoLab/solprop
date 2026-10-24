# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Synthesise a guide's exit field on a spatial grid (SEP 31, sections 4.1 and 5.2).

:class:`Outcouple` turns an ordinary modal plane -- true-power coefficients with
their scalar profiles (:class:`~solprop.coupling.models.ScalarModeBasis`) -- into a
radial or Cartesian :class:`~solprop.coupling.plane.FieldPlane` of the physical
field, under an **explicitly declared** forward interface model. No Fresnel factor
is implied: the one model offered, ``"matched"``, is the guide's exit into a medium
of its own core index, where the transverse field is continuous; a change of index is
a separate ``PlanarInterface`` element in the transport sequence that follows,
applied once.

The guide's field per true-power coefficient is
:math:`\sqrt{2/(\varepsilon_0c\operatorname{Re}n_{\rm eff})}\,v_m` (``docs/dev/
conventions.md`` section 4). It is written into the grid's columns through each
profile's closed-form transform (:func:`~solprop.coupling.coupler.target_columns`),
so a profile's kink at the core radius is not sampled; what the grid's spectral
window cannot hold is reported, not renormalised.
"""

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from jaxtyping import Float

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.coupling.coupler import target_columns
from solprop.coupling.plane import FieldPlane, PlaneBasis, PlaneMetric
from solprop.coupling.port import METRIC_TOLERANCE
from solprop.exceptions import CouplingError
from solprop.grid import FreeGrid, RadialGrid
from solprop.spatial import SpatialFlux, SpatialMedium

__all__ = ["Outcouple", "OutcoupleResult", "OutcouplingModel"]

OutcouplingModel = Literal["matched"]
"""A declared forward interface model of :class:`Outcouple`."""


@dataclass(frozen=True, eq=False)
class OutcoupleResult:
    """The synthesised exit plane and its energy account.

    Attributes
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The radial or Cartesian plane of the exit field.
    modal : numpy.ndarray
        The modal plane's energy per bin in its native metric, J, shape
        ``(n_omega,)``.
    synthesised : numpy.ndarray
        The synthesised plane's energy per bin in its axial-admittance metric, J,
        shape ``(n_omega,)``. It differs from `modal` by the scalar model's metric
        defect (core index against effective index), the profiles' angular content
        and what the grid's window misses; nothing is renormalised.
    window : float
        The largest departure of the profiles' Gram in the grid's spectral window
        from the identity: the share of a profile the grid cannot represent.
    """

    plane: FieldPlane
    modal: Float[np.ndarray, " n_omega"]
    synthesised: Float[np.ndarray, " n_omega"]
    window: float


@dataclass(frozen=True, eq=False)
class Outcouple:
    """Synthesise an ordinary guide's exit field onto a spatial grid.

    Attributes
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid of the synthesised plane, centred on the guide's axis.
    interface : {'matched'}
        The declared forward interface model. ``'matched'``: the exit medium has
        the guide's core index, so the transverse field is continuous and there is
        no reflection; any other exit medium is refused, and its index change is a
        separate planar interface.
    medium : solprop.spatial.SpatialMedium or None, optional
        The exit medium, whose index must match the core's.
    index : numpy.ndarray or None, optional
        Its real phase index at each bin instead, shape ``(n_omega,)``; by default
        the guide's own core index.

    Raises
    ------
    CouplingError
        For both a medium and an index.

    Examples
    --------
    >>> from solprop.grid import RadialGrid
    >>> Outcouple(RadialGrid(300e-6, 128))
    Outcouple(matched, RadialGrid)
    """

    space: RadialGrid | FreeGrid
    interface: OutcouplingModel = "matched"
    medium: SpatialMedium | None = None
    index: Float[np.ndarray, " n_omega"] | None = None

    def __post_init__(self) -> None:
        """Check that the exit medium is given once."""
        if self.medium is not None and self.index is not None:
            msg = "give the exit medium or its index, not both"
            raise CouplingError(msg)

    def __repr__(self) -> str:
        """Return a one-line description without arrays.

        Returns
        -------
        str
            The interface model and the grid's class.
        """
        return f"Outcouple({self.interface}, {type(self.space).__name__})"

    def apply(self, plane: FieldPlane) -> OutcoupleResult:
        """Synthesise a modal plane's exit field on the grid.

        Parameters
        ----------
        plane : solprop.coupling.plane.FieldPlane
            An ordinary modal plane carrying its scalar profiles (read with
            ``field_plane(run, source=simulation)``).

        Returns
        -------
        OutcoupleResult
            The spatial plane and its energy account.

        Raises
        ------
        CouplingError
            If the plane is not modal or has no profiles, or the exit medium's index
            is not the core's.
        """
        profiles = plane.profiles
        if plane.kind != "modal" or profiles is None:
            msg = (
                "Outcouple synthesises an ordinary modal plane with its profiles: read "
                "it with field_plane(run, source=simulation)"
            )
            raise CouplingError(msg)
        grid = plane.grid
        core = cast("np.ndarray", profiles.core_index)
        if self.medium is not None:
            flux = SpatialFlux.resolve(grid, self.space, self.medium, z=0.0)
            metric = PlaneMetric.from_flux(flux)
            live = (grid.omega > 0) & grid.active_mask
            outside = np.zeros(grid.n_omega)
            outside[live] = np.real(
                np.asarray(self.medium.index(grid.omega[live], 0.0))
            )
        else:
            outside = (
                core if self.index is None else np.asarray(self.index, dtype=np.float64)
            )
            metric = PlaneMetric.paraxial(self.space, outside, grid)
        occupied = np.sum(np.abs(plane.samples) ** 2, axis=(0, 1)) > 0.0
        scale = np.maximum(np.abs(core[occupied]), 1e-300)
        if np.any(
            np.abs(outside[occupied] - core[occupied]) > METRIC_TOLERANCE * scale
        ):
            msg = (
                "the exit medium's index is not the guide's core index: the 'matched' "
                "model has no index change; put a PlanarInterface after it"
            )
            raise CouplingError(msg)
        columns = target_columns(profiles, self.space)
        jones = np.asarray(profiles.jones)
        n_pol_in = plane.samples.shape[1]
        if n_pol_in == 2:
            jones_c = np.tile(np.eye(2), (profiles.n_modes, 1))
            mode_of = np.repeat(np.arange(profiles.n_modes), 2)
        else:
            jones_c, mode_of = jones, np.arange(profiles.n_modes)
        n_pol = 2 if np.any(jones_c[:, 1] != 0.0) else 1
        effective = profiles.effective_index[mode_of]
        factor = np.where(
            effective > 0,
            np.sqrt(2.0 / (EPS0 * C * np.where(effective > 0, effective, 1.0))),
            0.0,
        )
        coefficients = fourier.spectrum_to_internal(plane.samples, kind=grid.kind)
        amplitudes = coefficients.reshape(-1, grid.n_omega) * factor
        internal = np.einsum(
            "cj,cp,cw->jpw",
            columns[mode_of],
            jones_c[:, :n_pol].astype(complex),
            amplitudes,
        )
        samples = fourier.spectrum_to_optics(internal, kind=grid.kind)
        weights = np.asarray(self.space.spectral_weights)
        gram = np.einsum("j,mj,nj->mn", weights, columns.conj(), columns)
        values = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
        window = float(max(abs(values[0] - 1.0), abs(values[-1] - 1.0)))
        basis = PlaneBasis.of_space(self.space)
        synthesised_plane = FieldPlane(
            basis.kind,
            samples,
            grid,
            "V m" if basis.kind == "radial" else "V/m",
            basis,
            plane.reference,
            plane.geometry,
            metric,
            plane.exterior,
            self.medium,
            plane.certificate,
            provenance=plane.provenance.then(
                f"Outcouple({self.interface}) onto {type(self.space).__name__}"
            ),
        )
        return OutcoupleResult(
            synthesised_plane,
            plane.spectral_energy(),
            synthesised_plane.spectral_energy(),
            window,
        )
