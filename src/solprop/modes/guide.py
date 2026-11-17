# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build a general modal guide from a vector provider, sampled over frequency and z.

:class:`ModalGuide` pairs a :class:`~solprop.modes.vector.ModalProvider` with the
ordered mode collection and the material regions a general modal run propagates.
:meth:`ModalGuide.from_provider` samples the provider into tracked tables
(:mod:`solprop.modes.tables`) and, for a moving geometry, along its path
(:mod:`solprop.modes.path`), and records what it solved on and measured in
:class:`ModalSampling` and its :class:`~solprop.modes.path.ModalPath`. Each channel is a
:class:`ProviderChannelMode`, carrying its sampled dispersion and the
:class:`ChannelSource` it can be solved again from; the guide's tables are checked
against the provider to :data:`PROVIDER_BETA_RTOL`. The basis declarations
(:class:`ChannelBasis`, :func:`circular_pair`, :func:`provider_channels`) are
re-exported from :mod:`solprop.modes.vector`, and the path record :class:`ModalPath`
from :mod:`solprop.modes.path`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._display import Displayed, engineering, line
from solprop._ownership import frozen_copy
from solprop._process import host_blas
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.grid import TimeGrid
from solprop.modes.collection import ModeCollection
from solprop.modes.material import (
    JsonValue,
    MaterialRegion,
    PermittivityLaw,
    quantised_measurement,
)
from solprop.modes.path import (
    PATH_RTOL_FLOOR,
    ModalPath,
    path_rule,
    path_snapshot,
    sample_modal_path,
)
from solprop.modes.protocol import ChannelLabel, CrossSection, PolarisationComponent
from solprop.modes.stepindex import StepIndexMode
from solprop.modes.stepindex_vector import StepIndexProvider
from solprop.modes.tables import (
    DenseModalTable,
    align_modal_sample,
    apply_channel_basis,
    area_knot,
    channel_area,
    channel_index_table,
    generator_is_coupled,
    sample_modal_table,
    solved_in_table_basis,
)
from solprop.modes.tabulated import IndexTable
from solprop.modes.vector import (
    BASIS_UNITARY_ATOL,
    ChannelBasis,
    ModalNodeRule,
    ModalProvider,
    ModalSample,
    circular_pair,
    provider_channels,
)
from solprop.profiles import FromCallable, ZProfile, as_profile

__all__ = [
    "BASIS_UNITARY_ATOL",
    "PROVIDER_BETA_RTOL",
    "ChannelBasis",
    "ChannelSource",
    "ModalGuide",
    "ModalPath",
    "ModalSampling",
    "ProviderChannelMode",
    "circular_pair",
    "provider_channels",
]

PROVIDER_BETA_RTOL = 1e-6
"""Largest relative disagreement a guide admits between its tables and its provider.

:meth:`ModalGuide.from_provider` reads each channel's propagation constant out of
a tracked dense table and then asks the provider again, at the entrance frequency
and at three frequencies the table did not sample -- midway across the first, the
middle and the last gap between knots. The held-out ones are what means something:
they measure the frequency interpolation the table performs, which is the model
the guide's linear operator will use, and the ends of a band are where a
not-a-knot cubic is worst.

It certifies **``beta`` alone**. ``beta_1`` is the interpolant's and is not checked
against anything here; ``beta_2`` is uncertified and diagnostic
(:class:`ProviderChannelMode`).

The margin is measured, with this guard's own quantity on the reference parabolic
sector of ``tests/modes/test_modal_tables.py``, by a development study of the
provider-table derivatives. At the frequency the
guard tests: 1.11e-12 over a +-2 % band with five knots, 7.03e-14 with nine,
1.96e-8 over +-20 % with four knots and 2.08e-7 over +-40 % with four. So 1e-6 has
orders of margin on any band a run would use and **a factor of five** on the
widest band measured at the middle gap. Taken over every knot midpoint, that widest
case reaches 3.10e-6 at an end gap, above this budget, which is why the guard now
samples the first and the last gap as well: a four-knot table over +-40 % is
refused, and needs more knots.
"""


@dataclass(frozen=True, eq=False)
class ChannelSource:
    """What a provider channel is solved again from: its provider, index and rule.

    A :class:`ProviderChannelMode` is a sampled snapshot, and some questions need
    the fields again at a frequency or a position the snapshot did not keep -- a
    scalar model's effective area **at its pump**, above all.
    :meth:`ModalGuide.from_provider` gives every channel one, so that the
    channel can ask the provider itself rather than its caller reaching into the
    translation layer.

    Attributes
    ----------
    provider : ModalProvider
        The guide's full-vector provider.
    index : int
        The channel's index on the provider's mode axis, after `basis`, from zero.
    nodes : ModalNodeRule
        The rule the guide was sampled on: the entrance rule of a moving guide,
        which the provider is asked for again at every position
        (:func:`solprop.modes.path.path_rule`).
    z : float
        The position in m the snapshot, or a moving guide's entrance, was solved at.
    basis : ChannelBasis or None
        The guide's declared change of channel basis, applied to every fresh solve
        before the channel is read, or ``None`` for the provider's own.
    """

    provider: ModalProvider
    index: int
    nodes: ModalNodeRule
    z: float
    basis: ChannelBasis | None = None


@dataclass(frozen=True, eq=False)
class ProviderChannelMode(Displayed):
    """One labelled channel of a modal provider, carrying its sampled dispersion.

    The dispersion carrier the ordered ``modes`` of a :class:`ModalGuide` need for a
    provider that solves its own fields: its index comes from the table bridge
    (:func:`solprop.modes.tables.sample_provider_table`), and its identity
    is the provider's own channel label. It is **not** a second mode solver behind a
    :class:`~solprop.modes.protocol.Mode` interface, and it has no transverse profile
    of its own: :meth:`field` refuses, because the physical fields of this channel are
    the provider's three-component E and H, not two real transverse components.

    **What is certified and what is not.** :meth:`ModalGuide.from_provider` certifies
    ``beta`` and nothing else: it compares the table's propagation constant with a
    fresh provider solve at the entrance and at one held-out frequency, against
    :data:`PROVIDER_BETA_RTOL`. ``beta_1`` is the interpolant's own derivative, and
    nothing checks it at construction; measured held-out against fresh-solve
    five-point differences on the reference parabolic sector it is 4.4e-13 relative
    at nine knots over a +-2 % band. ``beta_2`` -- what
    :func:`solprop.modes.base.beta2` differences out of that derivative, and so what
    :func:`solprop.modes.base.zdw`, :func:`solprop.tools.soliton.dispersion_length` and
    :func:`solprop.tools.soliton.soliton_order` are given for this channel -- is
    **diagnostic only and uncertified**: on the same sector and the same held-out
    frequencies it is 2.8e-2 relative at nine knots, 3.2e-3 at seventeen and 2.1e-3
    at thirty-three over +-2 %, and worse than unity over a +-20 % band at any knot
    count measured. A table's cubic reproduces a nearly constant group delay to
    twelve digits and still has no useful curvature; read ``beta_2`` here as an
    order of magnitude, and take a dispersion length or a soliton order from it as
    the same (measured by a development study of the provider-table derivatives).

    Attributes
    ----------
    table : solprop.modes.tabulated.IndexTable
        Snapshot of this channel's complex index, with the interpolant's slopes.
    label : ChannelLabel
        The provider's own label for this channel.
    outer_radius : float
        Positive radius in m of the provider's declared physical domain, shared by
        every channel of one guide, so that a collection has one cross-section.
    area : float
        Positive surrogate effective area in m²
        (:func:`solprop.modes.tables.channel_area`) at the **median knot**
        of the band the guide was sampled on, the middle of that band. The table's
        ``source_identity["channel_area_omega"]`` names that frequency. The
        frequency is part of the number: on the reference parabolic sector the
        lowest knot's area is 2.0 %, 25 % and 66 % above the median knot's over
        +-2 %, +-20 % and +-40 % bands. On a moving guide, the entrance's.
    path : ModalPath or None
        How a guide whose geometry moves was sampled along z, or ``None`` for a
        snapshot. With it the channel is **not** z-constant:
        its index and its surrogate area are the path's, interpolated between the
        sampled positions exactly as the run's maps are.
    path_tables : tuple of IndexTable
        The channel's index table at every position of `path`, in
        :attr:`ModalPath.positions` order flattened; empty for a snapshot.
    path_areas : tuple of float
        Its surrogate area at every position of `path`, in m², same order; empty
        for a snapshot.
    source : ChannelSource or None
        The provider, index and rule the channel can be solved again from, set by
        :meth:`ModalGuide.from_provider`; ``None`` for a channel built by hand,
        which then cannot answer :meth:`solved_area`. A live handle, not
        configuration: it is left out of the channel's serialised record, since
        the run records the provider's identity beside its channels.
    """

    table: IndexTable
    label: ChannelLabel
    outer_radius: float
    area: float
    path: ModalPath | None = None
    path_tables: tuple[IndexTable, ...] = ()
    path_areas: tuple[float, ...] = ()
    source: ChannelSource | None = field(default=None, metadata={"record": False})

    def __repr__(self) -> str:
        """Return a one-line description: the channel and the domain.

        Returns
        -------
        str
            For example ``ProviderChannelMode(TE0,1, 50 um domain)``.
        """
        return line(
            "ProviderChannelMode",
            [
                ("", self.label.display_label),
                ("", f"{engineering(self.outer_radius, 'm')} domain"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        low, high = self.table.wavelength_range
        return f"ProviderChannelMode: {self.label.display_label}, from a provider", [
            ("family", self.label.family),
            ("support", f"{engineering(low, 'm')} to {engineering(high, 'm')}"),
            ("domain", engineering(self.outer_radius, "m")),
            ("effective area", f"{self.area * 1e12:.4g} um^2"),
        ]

    def __post_init__(self) -> None:
        """Check the declared domain and area.

        Raises
        ------
        ModeError
            If the domain radius or the effective area is not positive and finite.
        """
        for name, value in (
            ("outer_radius", self.outer_radius),
            ("area", self.area),
            *(("path_areas", one) for one in self.path_areas),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ModeError(f"{name} must be positive and finite, got {value!r}")
        count = 0 if self.path is None else self.path.positions.size
        if len(self.path_tables) != count or len(self.path_areas) != count:
            raise ModeError(
                f"a provider channel carries one index table and one area per position "
                f"of its path, {count} of each, and got {len(self.path_tables)} and "
                f"{len(self.path_areas)}"
            )

    @property
    def z_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the channel is independent of position.

        ``True`` for a snapshot at one declared position; ``False`` on a guide
        whose geometry moves, whose channel carries a sampled path.
        """
        return self.path is None

    def entrance(self) -> "ProviderChannelMode":
        """Return this channel's snapshot at the entrance, with no path.

        What a moving guide's cumulative ``Phi`` is carried by: the run puts one
        reference channel's entrance dispersion in ``Phi`` and every channel's
        departure from it, the motion included, in the coupled flow
        (``solprop.translate.assembly._general_assembly``).

        Returns
        -------
        ProviderChannelMode
            The same channel, z-constant, with its entrance table and area.
        """
        return ProviderChannelMode(
            self.table, self.label, self.outer_radius, self.area, source=self.source
        )

    @property
    def area_path(self) -> ZProfile | None:  # numpydoc ignore=RT01
        """Surrogate area along a moving guide's path, m².

        Declared under the cross-section name a moving mode declares its area by
        (:data:`solprop.modes.protocol.GEOMETRY_PATH_NAMES`), so that the tables of
        a run split on the path's section edges, where the interpolant's slope may
        jump.

        ``None`` for a snapshot.
        """
        path = self.path
        if path is None:
            return None
        edges = path.z_edges
        return FromCallable(
            _PathArea(self),
            (float(edges[0]), float(edges[-1])),
            np.asarray(edges[1:-1], dtype=float),
        )

    def _along(self, values: Sequence[np.ndarray], z: float) -> np.ndarray:
        """Interpolate per-position values of this channel's path at one position.

        - ``z``: position in m.
        """
        path = self.path
        assert path is not None
        return path.interpolate(np.asarray(values), z)

    @property
    def scale_invariant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the profile scales exactly with a core radius.

        Always ``False``: a provider channel has no scaling law, and its
        normalised frequency changes with the geometry.
        """
        return False

    @property
    def azimuthal_order(self) -> int:  # numpydoc ignore=RT01
        """The label's azimuthal order, or zero where it has none."""
        return self.label.azimuthal_order or 0

    @property
    def channel_label(self) -> ChannelLabel:  # numpydoc ignore=RT01
        """The provider's own label for this channel.

        The label the provider published in its identity.
        """
        return self.label

    def principal_component(self, z: float = 0.0) -> PolarisationComponent | None:
        """Declare that a vector channel needs both transverse components.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored, the snapshot is one position.

        Returns
        -------
        None
            Always ``None``.
        """
        del z
        return None

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the tabulated complex effective index.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s inside the table, shape (n_omega,).
        z : float, optional
            Position in m; ignored for a snapshot, interpolated along a moving
            guide's path.

        Returns
        -------
        numpy.ndarray
            ``n_eff(omega)``, complex, shape (n_omega,).
        """
        if self.path is None:
            return self.table.neff(omega)
        return self._along([table.neff(omega) for table in self.path_tables], z)

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return ``n_eff - 1`` formed without cancellation.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s inside the table, shape (n_omega,).
        z : float, optional
            Position in m; ignored for a snapshot, interpolated along a moving
            guide's path.

        Returns
        -------
        numpy.ndarray
            ``n_eff - 1``, complex, shape (n_omega,), the first of the table's
            refractivity pair; its second is the derivative
            :meth:`dneff_domega` returns.
        """
        if self.path is None:
            return self.table.refractivity(omega)[0]
        return self._along(
            [table.refractivity(omega)[0] for table in self.path_tables], z
        )

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the frequency derivative of the effective index.

        It is the **interpolant's**, not an analytic model derivative (ADR-006): a
        provider declares no closed form, so the table carries the slopes of the
        dense generator interpolant and this differentiates them. Nothing certifies
        it: :func:`solprop.modes.tables.validate_modal_table` measures no
        frequency derivative, and
        :meth:`ModalGuide.from_provider` checks ``beta`` alone. Measured held-out
        against fresh-solve five-point differences on the reference parabolic
        sector, the ``beta_1`` it gives is 4.4e-13 relative at nine knots over a
        +-2 % band; the second derivative taken from it is a diagnostic
        (:class:`ProviderChannelMode`).

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s inside the table, shape (n_omega,).
        z : float, optional
            Position in m; ignored for a snapshot, interpolated along a moving
            guide's path.

        Returns
        -------
        numpy.ndarray
            ``d n_eff / d omega`` in s/rad, complex, shape (n_omega,).
        """
        if self.path is None:
            return self.table.dneff_domega(omega)
        return self._along([table.dneff_domega(omega) for table in self.path_tables], z)

    def aeff(self, z: float = 0.0) -> float:
        """Return the declared surrogate effective area.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored for a snapshot, interpolated along a moving
            guide's path.

        Returns
        -------
        float
            ``A_eff`` in m², the scalar surrogate of
            :func:`solprop.modes.tables.channel_area` at the median knot
            of the sampled band (``source_identity["channel_area_omega"]``), not
            the nonlinear coefficient of this vector channel and not a function of
            frequency: a scalar route has one area and this is the one declared.
        """
        if self.path is None:
            return self.area
        return float(self._along([np.asarray(one) for one in self.path_areas], z))

    def solved_area(self, omega: float, z: float = 0.0) -> float:
        """Return the surrogate effective area from a fresh provider solve.

        The same definition as :attr:`area` and :meth:`aeff` --
        :func:`solprop.modes.tables.channel_area`, ``(int |E|^2 dA)^2 /
        int |E|^4 dA`` over the three Cartesian components, the longitudinal one
        included -- but at the frequency asked for rather than at the median knot of
        the sampled band, which is where a scalar model's pump usually is not. The
        provider is solved at `omega` on the guide's own rule: the sampling rule
        for a snapshot, and for a moving guide the rule the provider builds at `z`
        on the same request (:func:`solprop.modes.path.path_rule`), which
        is the rule the path's own samples were taken on. The declared channel
        basis is applied first; no tracking gauge is, so a channel inside a
        degenerate block that no basis declares has the provider's own member of
        that block.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s, inside the provider's domain.
        z : float, optional
            Position in m; ignored for a snapshot, which is solved where it was
            sampled.

        Returns
        -------
        float
            ``A_eff`` in m².

        Raises
        ------
        ModeError
            If the channel carries no :class:`ChannelSource`, or the provider
            refuses the frequency or the position.
        """
        source = self.source
        if source is None:
            raise ModeError(
                f"{self.label.display_label} was built without its provider, so its "
                "area cannot be solved again: build the guide with "
                "ModalGuide.from_provider, or pass the area explicitly"
            )
        if self.path is None:
            position, rule = source.z, source.nodes
        else:
            position = float(z)
            rule = path_rule(source.provider, source.nodes, position)
        sample = source.provider.solve(float(omega), position, nodes=rule)
        if source.basis is not None:
            sample = apply_channel_basis(sample, source.basis)
        return channel_area(sample, rule, source.index)

    def area_at(self, omega: float) -> float | ZProfile:
        """Return the surrogate area at one frequency, along z for a moving guide.

        :meth:`solved_area` at every position the channel's own area is sampled at:
        once for a snapshot, and at each position of a moving guide's
        :attr:`path` -- the positions :attr:`path_areas` holds -- interpolated
        between them exactly as :attr:`area_path` interpolates the median-knot
        areas, with the same span and the same section edges. It is what
        :meth:`solprop.nonlinear.gnlse.SpecifiedMode.from_mode` hands a scalar
        model given the pump's `wavelength`.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s, inside the provider's domain.

        Returns
        -------
        float or solprop.profiles.ZProfile
            The area in m², a number for a snapshot and a path along z for a
            moving guide.

        Raises
        ------
        ModeError
            As :meth:`solved_area`.
        """
        path = self.path
        if path is None:
            return self.solved_area(omega)
        areas = tuple(
            self.solved_area(omega, z) for z in path.positions.ravel().tolist()
        )
        edges = path.z_edges
        return FromCallable(
            _SolvedPathArea(path, areas),
            (float(edges[0]), float(edges[-1])),
            np.asarray(edges[1:-1], dtype=float),
        )

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        """Return the provider's declared physical domain.

        Parameters
        ----------
        z : float, optional
            Position in m; ignored, the snapshot is one position.

        Returns
        -------
        tuple
            ``("polar", (0, 0), (outer_radius, 2 pi))`` in m and rad, the same for
            every channel of one guide.
        """
        del z
        return "polar", (0.0, 0.0), (self.outer_radius, 2.0 * np.pi)

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Refuse a two-component transverse profile this channel does not have.

        Parameters
        ----------
        points : numpy.ndarray
            Cross-section positions, shape (n_points, 2).
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Never returns.

        Raises
        ------
        ModeError
            Always: the physical fields of a provider channel are the provider's
            three-component E and H on its node rule, and squeezing them into two
            real transverse components would be a different, unqualified mode.
        """
        del points, z
        raise ModeError(
            f"{self.label.display_label} is a provider channel: ask its provider for "
            "the three-component E and H on a node rule; it has no two-component "
            "real transverse profile"
        )


@dataclass(frozen=True, eq=False)
class _PathArea:
    """Evaluate a moving channel's surrogate area over an array of positions."""

    channel: ProviderChannelMode

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the area at every position, m².

        - ``z``: positions in m.
        """
        return np.array([self.channel.aeff(float(one)) for one in np.asarray(z)])


@dataclass(frozen=True, eq=False)
class _SolvedPathArea:
    """Evaluate areas solved at a moving channel's path positions, along z."""

    path: ModalPath
    areas: tuple[float, ...]

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the area at every position, m².

        - ``z``: positions in m.
        """
        values = np.asarray(self.areas)
        return np.array(
            [float(self.path.interpolate(values, float(one))) for one in np.asarray(z)]
        )


def _profile_defect(interpolated: ModalSample, solved: ModalSample) -> float:
    """Return the relative maximum-norm defect between two aligned samples' fields.

    ``solved`` is a fresh provider solve, already rotated onto the table's frequency
    interpolation by :func:`solprop.modes.tables.align_modal_sample`.

    - Returns: the larger of the electric and magnetic relative defects, dimensionless
      and non-negative.
    """
    # Neither scale can vanish: a validated sample's flux metric is positive
    # definite, which no identically zero E or H produces.
    return max(
        float(np.max(np.abs(left - right))) / float(np.max(np.abs(right)))
        for left, right in (
            (interpolated.electric, solved.electric),
            (interpolated.magnetic, solved.magnetic),
        )
    )


@dataclass(frozen=True, eq=False)
class ModalSampling(Displayed):
    """Record what a guide's channels were solved on, and what that measured.

    Everything a dense reference table of this guide is rebuilt from -- the knots,
    the position, the node rule and the declared degenerate blocks -- together with
    the profile interpolation error those knots were measured to give. A guide
    built by :meth:`ModalGuide.from_provider` carries one; a guide built from
    constructed modes does not, and cannot be propagated, because nothing declares
    the resolution its reference would be sampled at.

    Attributes
    ----------
    omega : numpy.ndarray
        The knots the provider was solved at, increasing positive angular
        frequencies in rad/s, shape (n_omega_sampling,), at least four.
    z : float
        The one position in m the snapshot was taken at.
    nodes : ModalNodeRule
        The common Cartesian rule every knot was solved on, with its own certified
        tail bound and material ownership.
    degenerate_blocks : tuple of tuple of int
        Channel indices the provider's own order declares degenerate, as the
        tracking took them.
    basis : ChannelBasis or None
        The declared unitary change of channel coordinates the reference carries
        on top of the tracking gauge, or ``None`` for the provider's own. It is
        here because it is part of *what the reference is*, exactly as the knots
        and the node rule are: a table rebuilt from this sampling without it is a
        different basis.
    profile_error : float
        Largest relative departure, in the maximum norm over nodes, components and
        channels, between the table's frequency interpolation of E and H and a
        fresh provider solve, measured at the **entrance knot**, where the table
        interpolates nothing, and at the midpoints of the first, the middle and
        the last knot gap, where a not-a-knot cubic is worst.
        Non-negative and dimensionless. It is the budget
        :func:`solprop.translate.modal_tables.modal_transform_tables` records on
        the run's own record, and it is a measurement at four frequencies, not a
        proof over the band.
    path : ModalPath or None
        How a guide whose geometry moves was sampled along z -- its sections, the
        degree its accuracy check demanded and what that measured -- or ``None``
        for a snapshot. With it `z` is the path's entrance
        and `nodes` the entrance rule, which the provider is asked for again at
        every sampled position.

    Raises
    ------
    ModeError
        If the knots are not at least four increasing positive finite values, the
        position is not finite, or the profile error is negative or not finite.
    """

    omega: Float[np.ndarray, " n_omega_sampling"]
    z: float
    nodes: ModalNodeRule
    degenerate_blocks: tuple[tuple[int, ...], ...]
    profile_error: float
    basis: ChannelBasis | None = None
    path: ModalPath | None = None

    def __post_init__(self) -> None:
        """Adopt a read-only knot array and check it.

        Raises
        ------
        ModeError
            Naming what is out of range.
        """
        knots = frozen_copy(np.reshape(self.omega, -1), np.float64)
        if (
            knots.size < 4
            or not np.all(np.isfinite(knots))
            or knots[0] <= 0.0
            or not np.all(np.diff(knots) > 0.0)
        ):
            raise ModeError(
                "a guide's sampling knots must be at least four increasing "
                f"positive finite frequencies, got {knots}"
            )
        object.__setattr__(self, "omega", knots)
        if not np.isfinite(self.z):
            raise ModeError(f"a guide's snapshot position must be finite, got {self.z}")
        if not np.isfinite(self.profile_error) or self.profile_error < 0.0:
            raise ModeError(
                "a guide's measured profile error must be non-negative and finite, "
                f"got {self.profile_error!r}"
            )

    @property
    def omega_span(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Band the guide was solved over: lowest and highest knot, rad/s."""
        return float(self.omega[0]), float(self.omega[-1])

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """What a run records of this sampling, and a resume compares.

        Everything a reader needs to say **what resolution the reference was built
        at** -- the knots themselves, the snapshot position, the node rule's own
        construction and tail certificate, the declared degenerate blocks and the
        measured profile error -- and no array the rule owns by value: a rule's
        node coordinates and area weights are the megabytes
        :class:`~solprop.translate.modal_tables.ModalTransformTables` already
        describes by extent, and the rule's own ``identity`` is a mapping of
        scalars, as the run record holds it.

        The profile error is **measured**, by a dense solve and a reduction, so it
        enters at :data:`~solprop.modes.material.MEASURED_DIGITS` significant
        digits: the same guide re-solved on another BLAS differs in the last bits
        of it and in nothing else, and a resume key and a reconstruction gate must
        not refuse a run that did not change. The knots, the position, the blocks
        and a declared basis are declared and enter exactly.

        JSON-serialisable scalars and lists of scalars, stable across
        processes.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes.guide import ModalSampling
        >>> from solprop.modes.vector import ModalNodeRule
        >>> rule = ModalNodeRule(
        ...     np.zeros((1, 2)), np.ones(1), np.zeros(1, np.int32), {"rule": "demo"}
        ... )
        >>> sampling = ModalSampling(
        ...     np.array([1.0, 2.0, 3.0, 4.0]), 0.0, rule, ((0,),), 1e-5
        ... )
        >>> sampling.identity["n_knots"], sampling.identity["n_nodes"]
        (4, 1)
        """
        record: dict[str, JsonValue] = {
            "n_knots": int(self.omega.size),
            "omega": [float(one) for one in self.omega],
            "z": float(self.z),
            "n_nodes": int(self.nodes.weights.size),
            "nodes": dict(self.nodes.identity),
            "degenerate_blocks": [
                [int(one) for one in block] for block in self.degenerate_blocks
            ],
            "profile_error": quantised_measurement(self.profile_error),
        }
        # Present only where one is declared, so that every general modal file
        # written without a basis keeps the identity it was stamped with and
        # still resumes. That is not a hole: a declared basis composes its
        # `gauge_id` into both `tables.gauge_id` and `metric.gauge_id`, which are
        # always in the key, so the two directions of a basis mismatch are each
        # refused by a field that is always compared.
        if self.basis is not None:
            record["basis"] = dict(self.basis.identity)
        # Present only on a moving guide, for the same reason: a snapshot's file keeps
        # the key it was stamped with, and a moving guide's is refused a resume by
        # the path that changed.
        if self.path is not None:
            record["path"] = dict(self.path.identity)
        return record

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalSampling(9 knots, 1.1 to 2.5 Prad/s, 288 nodes)``.
        """
        low, high = self.omega_span
        return line(
            "ModalSampling",
            [
                ("", f"{self.omega.size} knots"),
                (
                    "",
                    f"{engineering(low, 'rad/s')} to {engineering(high, 'rad/s')}",
                ),
                ("", f"{self.nodes.weights.size} nodes"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        low, high = self.omega_span
        blocks = self.degenerate_blocks
        return "ModalSampling: what a guide's channels were solved on", [
            ("knots", self.omega.size),
            ("band", f"{engineering(low, 'rad/s')} to {engineering(high, 'rad/s')}"),
            (
                "wavelengths",
                f"{engineering(2 * np.pi * C / high, 'm')} to "
                f"{engineering(2 * np.pi * C / low, 'm')}",
            ),
            ("position", engineering(self.z, "m")),
            ("nodes", self.nodes.weights.size),
            (
                "degenerate",
                "none" if not blocks else ", ".join(str(list(b)) for b in blocks),
            ),
            ("basis", "the provider's own" if self.basis is None else repr(self.basis)),
            ("profile error", f"{self.profile_error:.3e}"),
            ("path", "none (a snapshot)" if self.path is None else repr(self.path)),
        ]


@dataclass(frozen=True)
class ModalGuide(Displayed):
    """Associate a vector provider with the existing ordered mode collection.

    Attributes
    ----------
    field_provider : ModalProvider
        Internal-convention full-vector field provider.
    mode_collection : ModeCollection
        Physical modes and their channel labels in matching order.
    regions : tuple of MaterialRegion
        Ordered material regions owned by the guide, each declaring its own
        response (:class:`~solprop.modes.material.MaterialKerr`) or none, which
        is how a linear cladding is said to be linear.
    sampling : ModalSampling or None
        What the guide's channels were solved on, filled by
        :meth:`from_provider` and ``None`` for a guide built from constructed
        modes. A guide without it can be inspected and compared against, and
        cannot be propagated:
        :meth:`solprop.driver.simulation.Simulation.from_waveguide` refuses it by
        name, because the resolution its dense reference would be sampled at is
        nowhere declared.
    """

    field_provider: ModalProvider
    mode_collection: ModeCollection
    regions: tuple[MaterialRegion, ...]
    sampling: ModalSampling | None = None

    def __repr__(self) -> str:
        """Return a one-line description: the provider and the channel count.

        Returns
        -------
        str
            For example ``ModalGuide(StepIndexProvider(HE11, 2 regions), 1 channel)``.
        """
        count = len(self.mode_collection)
        return line(
            "ModalGuide",
            [
                ("", repr(self.field_provider)),
                ("", f"{count} channel{'' if count == 1 else 's'}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        labels = [label.display_label for label in self.mode_collection.channel_labels]
        sampling = self.sampling
        return "ModalGuide: a mode collection and its full-vector provider", [
            ("provider", repr(self.field_provider)),
            ("channels", ", ".join(labels)),
            (
                "regions",
                ", ".join(
                    f"{region.name}"
                    + (
                        ""
                        if (electronic := region.kerr or region.gas) is None
                        else f" ({electronic!r})"
                    )
                    for region in self.regions
                ),
            ),
            (
                "sampling",
                "none (built from constructed modes; cannot be propagated)"
                if sampling is None
                else repr(sampling),
            ),
        ]

    def __post_init__(self) -> None:
        """Require the provider's own regions and channels, in order and count.

        Raises
        ------
        ModeError
            If the declared regions or the ordered channel labels disagree with
            the field provider's, or if the provider names no channels.
        """
        if tuple(self.regions) != tuple(self.field_provider.regions):
            raise ModeError(
                "guide regions must be the field provider's own material regions"
            )
        declared = self.field_provider.identity.get("channels")
        if not isinstance(declared, list):
            raise ModeError(
                "a modal guide needs a provider that names its channels; "
                f"{type(self.field_provider).__name__} declares none, so the "
                "mode order cannot be checked against it"
            )
        labels = [
            dict(label.as_mapping()) for label in self.mode_collection.channel_labels
        ]
        sampling = self.sampling
        # A declared change of basis relabels the channels it mixes -- a parity
        # pair becomes a handedness pair -- so the guide's labels are then the
        # declaration's and not the provider's, and this check compares against
        # the declaration. The declaration itself was built from the provider's
        # labels, so the chain back to the provider is unbroken.
        expected = (
            list(declared)
            if sampling is None or sampling.basis is None
            else [dict(label.as_mapping()) for label in sampling.basis.labels]
        )
        if expected != labels:
            raise ModeError(
                "guide channel labels disagree with the provider's channels in "
                f"order or count: {labels} against {expected}"
            )

    @classmethod
    def from_step_index(
        cls,
        modes: Sequence[StepIndexMode],
        *,
        core: PermittivityLaw,
        cladding: PermittivityLaw,
        regions: Sequence[MaterialRegion] = (),
    ) -> "ModalGuide":
        """Build a bound provider from existing exact step-index modes.

        Parameters
        ----------
        modes : sequence of StepIndexMode
            Ordered unique bound channels of one guide.
        core : PermittivityLaw
            Core law, matching the modes' core index.
        cladding : PermittivityLaw
            Explicit exterior law, matching the modes' cladding index.
        regions : sequence of MaterialRegion, optional
            Core/exterior response ownership; empty builds two linear regions.

        Returns
        -------
        ModalGuide
            Guide with a separate full-vector physics provider.
        """
        collection = ModeCollection(modes)
        radius = modes[0].core_radius
        material_regions = tuple(regions) or (
            MaterialRegion("core", 0.0, radius, core),
            MaterialRegion("cladding", radius, None, cladding),
        )
        return cls(
            StepIndexProvider(tuple(modes), material_regions),
            collection,
            material_regions,
        )

    @classmethod
    def on_grid(
        cls,
        grid: TimeGrid,
        provider: ModalProvider,
        *,
        knots: int,
        n_r: int,
    ) -> "ModalGuide":
        """Build a guide from a provider, sampled across a time grid's active band.

        :meth:`from_provider` with its frequencies and node rule taken from the grid:
        `knots` frequencies evenly spaced from 1 % below the active band's lowest
        frequency to 1 % above its highest, so that the interpolant is never asked to
        extrapolate, and the provider's own rule at the lowest, where a bound field
        reaches furthest (:meth:`from_provider`). Both counts are the caller's: the
        general modal route refuses too few knots and warns about a marginal count
        by name, for the band it runs on.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            The run's time grid.
        provider : ModalProvider
            Internal-convention provider naming its channels in ``identity``.
        knots : int
            Number of frequency knots, at least four.
        n_r : int
            Radial quadrature nodes per element of the provider's rule.

        Returns
        -------
        ModalGuide
            The guide, as :meth:`from_provider` returns it.

        Raises
        ------
        ModeError
            If there are fewer than four knots, or :meth:`from_provider` refuses them
            or the band's channels.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.grid import FieldGrid
        >>> from solprop.modes import (
        ...     GradedIndexProvider,
        ...     IndexMaterial,
        ...     ModalGuide,
        ...     PowerLawPermittivity,
        ... )
        >>> provider = GradedIndexProvider(
        ...     permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        ...     exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        ...     radius=25e-6,
        ...     family="LP",
        ...     count=2,
        ...     elements=32,
        ...     extent=2.0,
        ... )
        >>> grid = FieldGrid.from_limits(1e-3, 1.03e-6, (1.0e-6, 1.06e-6), 100e-15)
        >>> guide = ModalGuide.on_grid(grid, provider, knots=13, n_r=2)
        >>> len(guide.mode_collection.channel_labels)
        2
        """
        if knots < 4:
            msg = f"a guide needs at least four frequency knots, got {knots}"
            raise ModeError(msg)
        # An envelope grid holds its frequencies in FFT order: the ends of the array
        # are beside the carrier, not the edges of the band.
        active = grid.omega[grid.active_mask]
        low = 0.99 * float(np.min(active))
        high = 1.01 * float(np.max(active))
        return cls.from_provider(
            provider,
            np.linspace(low, high, knots),
            nodes=provider.node_rule(low, n_r=n_r),
        )

    @classmethod
    @host_blas()
    def from_provider(
        cls,
        provider: ModalProvider,
        omega: Float[np.ndarray, " n_omega"],
        *,
        z: float = 0.0,
        nodes: ModalNodeRule | None = None,
        degenerate_blocks: Sequence[Sequence[int]] = (),
        basis: ChannelBasis | None = None,
        beta_rtol: float = PROVIDER_BETA_RTOL,
        sections: int = 1,
        path_rtol: float | None = None,
    ) -> "ModalGuide":
        """Build a guide from a provider that solves its own fields.

        The route for a :class:`~solprop.modes.graded_index.GradedIndexProvider`, and
        for any provider that publishes its channels: the provider owns its sector,
        channel count, elements, extent and radius path, exactly as
        :meth:`from_step_index` takes constructed modes, and nothing here invents a
        resolution. Dispersion comes through
        the table bridge, so each channel becomes a
        :class:`ProviderChannelMode` whose propagation constant is the provider's own
        generator at the knots and whose derivatives are the dense table's
        interpolant.

        The propagation constants are then checked back against the provider at the
        entrance frequency, where the table interpolates nothing, and midway across
        the first, the middle and the last gap between knots, where it interpolates
        as it will in a run; the ends are where a not-a-knot cubic is worst.

        **A provider whose geometry moves** -- a declared radius path that is not
        constant, or any longitudinal breakpoint -- is sampled along its path as
        well: its z domain is cut into sections at every declared kink, each divided
        `sections` times, and every section is solved at the Chebyshev-Lobatto points
        of the degree its own accuracy check demands, on the provider's own rule at
        each position and in its own gauge
        (:func:`solprop.modes.path.sample_modal_path`). The checks below are
        then taken at the entrance, which is the provider's own z domain start, and
        each channel carries its index and area along the path
        (:class:`ModalPath`, :attr:`ProviderChannelMode.path`). A guide whose
        geometry does not move is a **snapshot at one position** and is bitwise what
        it was.

        Parameters
        ----------
        provider : ModalProvider
            Internal-convention provider naming its channels in ``identity``.
        omega : numpy.ndarray
            At least four increasing positive frequencies in rad/s covering the band
            the guide will be used over, shape (n_omega,).
        z : float, optional
            Snapshot position in m inside the provider's z domain.
        nodes : ModalNodeRule or None, optional
            Common physical rule; ``None`` asks the provider for one at the
            **lowest** frequency of the band, where a bound field reaches furthest
            and the power a rule discards is largest, so that the rule's
            ``tail_power_bound`` is a bound over the whole band and not only at the
            frequency it was taken.
        degenerate_blocks : sequence of sequence of int, optional
            Channel indices the provider's own order declares degenerate, as
            :func:`solprop.modes.tables.sample_modal_table` takes them.
            Tracking rotates freely **inside** a declared block and matches
            one-to-one outside it, and the generator is rotated with the fields, so
            a degenerate pair -- the two parities of an HE_1m channel -- stays
            physically equivalent instead of being matched by an ordering the
            eigenvalues cannot resolve. The blocks are retained on the table, so the
            propagation-constant checks below align a fresh solve the same way.
            Empty by default: an arbitrary provider declares no degeneracy.
        basis : ChannelBasis or None, optional
            A declared unitary change of the channel coordinates, applied to every
            knot after the tracking gauge (:func:`circular_pair` builds the
            circular partners of a parity pair). It must act only inside blocks
            named in `degenerate_blocks` and only on blocks whose propagation
            constants are equal to the rounding of the largest; both are refused by
            name. The guide's channels then carry the declaration's labels --
            handedness rather than parity -- and its ``ProviderChannelMode`` rows
            carry the transformed channels' surrogate areas. ``None`` keeps the
            provider's own coordinates, and the guide is then bitwise what it was.
        beta_rtol : float, optional
            Budget on the two propagation-constant checks, default
            :data:`PROVIDER_BETA_RTOL`.
        sections : int, optional
            Sections each kink interval of a moving guide is divided into, positive;
            one by default. Ignored by nothing: a snapshot refuses any other value,
            because it has no path to divide.
        path_rtol : float or None, optional
            Budget on a moving guide's held-out departure along z; ``None`` takes
            the frequency interpolation's own measured profile error, floored at
            :data:`solprop.modes.path.PATH_RTOL_FLOOR`, so that the
            sampling in z is never the coarser of the two. A snapshot refuses one.

        Returns
        -------
        ModalGuide
            The provider, one channel mode per declared channel in the provider's
            own order, and the provider's material regions.

        Raises
        ------
        ModeError
            If the provider names no channels, the budget is invalid, a declared
            `basis` is not admitted, a channel's tabulated propagation constant
            disagrees with a fresh solve by more than `beta_rtol`; for a moving
            guide, if `z` is not its entrance, `nodes` is not a rule its provider
            builds, its gauge is not continuous or its path does not resolve; for a
            snapshot, if `sections` or `path_rtol` is declared.
        """
        if not np.isfinite(beta_rtol) or beta_rtol <= 0:
            raise ModeError("beta_rtol must be positive and finite")
        declared = provider.identity.get("channels")
        if not isinstance(declared, list) or not declared:
            raise ModeError(
                "a modal guide needs a provider that names its channels; "
                f"{type(provider).__name__} declares none"
            )
        frequencies = np.asarray(omega, dtype=float)
        moving, entrance, nodes, table = _entrance_table(
            provider,
            frequencies,
            z,
            nodes,
            degenerate_blocks=degenerate_blocks,
            basis=basis,
            path_declared=sections != 1 or path_rtol is not None,
        )
        assert isinstance(table, DenseModalTable)
        middle = len(frequencies) // 2
        # Between two knots, where the table interpolates as it will in a run. The
        # first and the last gap as well as the middle one: a not-a-knot cubic is at
        # its best in the middle of a band and worst at its ends, where the measured
        # error is up to fifteen times the middle gap's.
        gaps = {
            "first held-out": (0, 1),
            "held-out": (middle, middle + 1),
            "last held-out": (len(frequencies) - 2, len(frequencies) - 1),
        }
        checks: list[tuple[str, float, ModalSample, ModalSample]] = []
        for name, frequency in (
            ("entrance", float(frequencies[0])),
            *(
                (name, float((frequencies[low] + frequencies[high]) / 2))
                for name, (low, high) in gaps.items()
            ),
        ):
            interpolated = table.evaluate(frequency, entrance)
            checks.append(
                (
                    name,
                    frequency,
                    interpolated,
                    align_modal_sample(
                        interpolated,
                        solved_in_table_basis(table, provider, frequency, entrance),
                        nodes,
                        degenerate_blocks=table.degenerate_blocks,
                    ),
                )
            )
        # The same four frequencies -- the entrance knot and the three held-out
        # midpoints -- certify the profiles the run's synthesis and projection
        # maps are built from, which nothing else measures: the beta guard below
        # is a scalar statement about the operator, and a table can reproduce beta
        # to twelve digits with a visibly wrong field. The entrance knot is in it
        # because the table's own sampling there is not exact either.
        profile_error = max(
            _profile_defect(interpolated, aligned)
            for _, _, interpolated, aligned in checks
        )
        # The scalar surrogate area is taken at the median knot, the middle of the
        # band the guide was sampled over, and the table's identity records which
        # frequency that was.
        reference_sample = table.samples[0][area_knot(frequencies)]
        modes: list[ProviderChannelMode] = []
        # The rule's own declared truncation radius where it has one, because the
        # outermost Gauss node sits inside the last panel edge; otherwise the
        # furthest node, which is all an arbitrary rule says about its domain.
        declared_radius = nodes.identity.get("outer_radius")
        radius = (
            float(declared_radius)
            if isinstance(declared_radius, float | int)
            and not isinstance(declared_radius, bool)
            else float(np.max(np.hypot(nodes.xy[:, 0], nodes.xy[:, 1])))
        )
        # A declared basis of a *split* block leaves the generator coupled, and then
        # every per-channel index table is a surrogate: `i G_cc` is the channel's
        # mean propagation constant and not its propagation, because the block
        # beats. The run propagates the block through `linear.coupled_flow` and uses
        # only the reference channel's Phi.
        coupled = generator_is_coupled(table)
        mixed: frozenset[int] = (
            frozenset()
            if basis is None
            else frozenset(one for block in basis.blocks for one in block)
        )
        for channel, mapping in enumerate(declared):
            if not isinstance(mapping, dict):
                raise ModeError("a declared channel must be a label mapping")
            # A declared basis relabels the channels it mixes, and its labels were
            # built from these, so the chain back to the provider is unbroken.
            # Outside its blocks it changes nothing, and a hand-built declaration
            # that disagrees there is relabelling a channel it does not touch:
            # `__post_init__` compares the guide against these same labels, so a
            # silent rename would pass every check downstream of it.
            label = ChannelLabel.from_mapping(mapping)
            if basis is not None and channel in mixed:
                label = basis.labels[channel]
            elif basis is not None and basis.labels[channel] != label:
                raise ModeError(
                    f"a declared channel basis renames channel {channel}, which it "
                    f"does not mix: the provider declares "
                    f"{label.display_label!r} and the basis carries "
                    f"{basis.labels[channel].display_label!r}. A declaration "
                    "relabels the blocks it acts inside and nothing else"
                )
            # From the one table built above: a table per channel would solve the
            # whole band once per channel for the same knots.
            index = channel_index_table(
                table, channel, include_area=True, coupled=coupled
            )
            for name, frequency, _, reference in checks:
                expected = float(np.real(1j * reference.generator[channel, channel]))
                measured = float(index.beta(np.array([frequency]))[0])
                if abs(measured - expected) > beta_rtol * abs(expected):
                    raise ModeError(
                        f"channel {channel} disagrees with the provider at the "
                        f"{name} frequency {frequency:.6e} rad/s: the table gives "
                        f"{measured!r} 1/m and a solve gives {expected!r} 1/m, "
                        f"a relative {abs(measured - expected) / abs(expected):.3g} "
                        f"above the declared {beta_rtol:.3g}; refine the knots"
                    )
            modes.append(
                ProviderChannelMode(
                    index,
                    label,
                    radius,
                    channel_area(reference_sample, nodes, channel),
                    source=ChannelSource(provider, channel, nodes, entrance, basis),
                )
            )
        path = None
        if moving:
            path, modes = _sampled_path(
                provider,
                table,
                modes,
                sections=sections,
                rtol=profile_error if path_rtol is None else path_rtol,
            )
        return cls(
            provider,
            ModeCollection(modes),
            tuple(provider.regions),
            ModalSampling(
                frequencies,
                entrance,
                nodes,
                tuple(table.degenerate_blocks),
                profile_error,
                basis,
                path,
            ),
        )

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Delegate a solve to the guide's full-vector provider.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s.
        z : float
            Position in m.
        nodes : ModalNodeRule
            Common physical nodes and material indices.

        Returns
        -------
        ModalSample
            The provider's validated E/H sample.
        """
        return self.field_provider.solve(omega, z, nodes=nodes)

    @property
    def omega_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Provider frequency interval ``(lower, upper)``, rad/s."""
        return self.field_provider.omega_domain

    @property
    def z_domain(self) -> tuple[float, float]:  # numpydoc ignore=RT01
        """Provider longitudinal interval ``(lower, upper)``, m."""
        return self.field_provider.z_domain

    @property
    def z_breakpoints(self) -> tuple[float, ...]:  # numpydoc ignore=RT01
        """Provider physical breakpoints (path kinks), m."""
        return self.field_provider.z_breakpoints

    @property
    def identity(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """Provider physical identity.

        Serializable physical model and construction metadata.
        """
        return self.field_provider.identity


def _moving_geometry(provider: ModalProvider) -> list[str]:
    """Return what moves along z in a provider's declared geometry, by name.

    - Returns: each region radius whose path is not constant, and the interior kinks;
      empty for a guide whose geometry does not move.
    """
    moving = [
        f"{region.name}.{name}"
        for region in provider.regions
        for name, value in (
            ("inner_radius", region.inner_radius),
            ("outer_radius", region.outer_radius),
        )
        if value is not None and not as_profile(value).is_constant
    ]
    low, high = (float(one) for one in provider.z_domain)
    kinks = [float(one) for one in provider.z_breakpoints if low < float(one) < high]
    if kinks:
        moving.append(f"path kinks at {kinks} m")
    return moving


def _check_entrance_rule(
    provider: ModalProvider, nodes: ModalNodeRule, z: float
) -> None:
    """Refuse an entrance rule the provider would not build again at every position.

    - ``z``: the entrance, m.
    """
    again = path_rule(provider, nodes, z)
    if not (
        np.array_equal(again.xy, nodes.xy)
        and np.array_equal(again.weights, nodes.weights)
    ):
        raise ModeError(
            f"a moving guide asks {type(provider).__name__} for its rule again at "
            "every position of the path, and asked the way this entrance rule says "
            "it was asked for, the provider returns a different rule at the entrance: "
            "pass nodes=None, or the provider's own node_rule at the entrance"
        )


def _sampled_path(
    provider: ModalProvider,
    table: object,
    modes: Sequence[ProviderChannelMode],
    *,
    sections: int,
    rtol: float,
) -> tuple[ModalPath, list[ProviderChannelMode]]:
    """Sample a moving guide's path and give each channel its index along it.

    - Returns: the path record and the channels carrying it.
    """
    assert isinstance(table, DenseModalTable)
    sampled = sample_modal_path(
        provider,
        table.omega,
        table.nodes,
        rtol=max(float(rtol), PATH_RTOL_FLOOR),
        sections=sections,
        degenerate_blocks=table.degenerate_blocks,
        basis=table.basis,
        gram_atol=table.gram_atol,
        known={float(table.z[0]): table},
    )
    path = sampled.path
    knot = area_knot(table.omega)
    outer = max(
        float(cast("float", one.nodes.identity.get("outer_radius", 0.0)))
        for one in sampled.tables
    )
    moved = []
    for channel, mode in enumerate(modes):
        moved.append(
            ProviderChannelMode(
                mode.table,
                mode.label,
                max(outer, mode.outer_radius),
                mode.area,
                path,
                tuple(
                    channel_index_table(
                        one,
                        channel,
                        include_area=False,
                        coupled=generator_is_coupled(one),
                    )
                    for one in sampled.tables
                ),
                tuple(
                    channel_area(one.samples[0][knot], one.nodes, channel)
                    for one in sampled.tables
                ),
                mode.source,
            )
        )
    return path, moved


def _entrance_table(
    provider: ModalProvider,
    frequencies: Float[np.ndarray, " n_omega"],
    z: float,
    nodes: ModalNodeRule | None,
    *,
    degenerate_blocks: Sequence[Sequence[int]],
    basis: ChannelBasis | None,
    path_declared: bool,
) -> tuple[list[str], float, ModalNodeRule, object]:
    """Solve a guide's snapshot table, at its entrance when its geometry moves.

    - ``frequencies``: the knots in rad/s.
    - ``z``: the declared snapshot position, m.
    - ``nodes``: the declared rule, or ``None`` for the provider's own at the lowest
      knot.
    - Returns: what moves (empty for a snapshot), the position the table is at, the rule
      and the table.
    """
    # The radius paths are on the protocol, as the regions' own radii, and a
    # declared kink is motion too: the slope jumps there.
    moving = _moving_geometry(provider)
    if not moving and path_declared:
        raise ModeError(
            f"{type(provider).__name__} declares no moving geometry, so the guide "
            "is a snapshot with no path to divide or to sample: sections and "
            "path_rtol belong to a provider whose radius path moves"
        )
    entrance = float(provider.z_domain[0]) if moving else float(z)
    if moving and float(z) != entrance:
        raise ModeError(
            f"{type(provider).__name__} declares moving geometry "
            f"({', '.join(moving)}), so the guide is sampled along its whole path "
            f"from its entrance at z = {entrance!r} m, and a snapshot position of "
            f"z = {float(z)!r} m means nothing for it: leave z at its default"
        )
    if nodes is None:
        # The lowest frequency: a bound field reaches furthest there, so the
        # rule's certified tail is a bound over the band rather than at one
        # frequency inside it.
        nodes = provider.node_rule(float(np.min(frequencies)), entrance)
    if moving:
        _check_entrance_rule(provider, nodes, entrance)
        table = path_snapshot(
            provider,
            frequencies,
            entrance,
            nodes,
            degenerate_blocks=degenerate_blocks,
            basis=basis,
        )
    else:
        table = sample_modal_table(
            provider,
            frequencies,
            np.array([float(z)]),
            nodes,
            degenerate_blocks=degenerate_blocks,
            basis=basis,
        )
    return moving, entrance, nodes, table
