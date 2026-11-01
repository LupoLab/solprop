# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build Poynting-normalised bound-mode providers with explicit material ownership."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.modes._stepindex_vector import _fields, _power
from solprop.modes.collection import ModeCollection
from solprop.modes.material import (
    JsonValue,
    MaterialRegion,
    PermittivityLaw,
    quantised_measurement,
)
from solprop.modes.protocol import ChannelLabel, CrossSection, PolarisationComponent
from solprop.modes.quadrature import quadrature_rule
from solprop.modes.stepindex import StepIndexMode, _material
from solprop.modes.tabulated import IndexTable
from solprop.modes.vector import (
    ModalNodeRule,
    ModalProvider,
    ModalSample,
    flux_product,
)
from solprop.profiles import FromCallable, ZProfile, as_profile, profile_identity

__all__ = [
    "BASIS_UNITARY_ATOL",
    "MATERIAL_AGREEMENT_RTOL",
    "PROVIDER_BETA_RTOL",
    "ChannelBasis",
    "ChannelSource",
    "ModalGuide",
    "ModalPath",
    "ModalSampling",
    "ProviderChannelMode",
    "StepIndexProvider",
    "circular_pair",
    "provider_channels",
    "radial_node_rule",
]

BASIS_UNITARY_ATOL = 1e-12
"""Largest ``norm(U.H U - I, 2)`` a declared change of basis is still called unitary.

Dimensionless. A declared transform is admitted only where it is a **unitary** change
of coordinates of an exactly degenerate block: the flux metric then transforms as
``W' = U.H W U``, which is the identity wherever ``W`` was, and the generator
``U.H diag(beta) U`` stays diagonal. A nonunitary transform does neither -- it is a
leaky or skewed basis whose metric is genuinely off the identity and whose flow is a
matrix exponential -- and nothing in this package has measured one, so it is refused
here rather than approximated.

**The number.** :func:`circular_pair`'s matrix has entries ``+-1/sqrt(2)`` and
``+-i/sqrt(2)``; ``1/sqrt(2)`` is exact to a half ulp and the products that form
``U.H U`` cancel to the last bit, so its measured defect is at the 1e-16 level.
A budget of 1e-12 is four orders above that -- room for a matrix assembled from
measured numbers rather than from a literal -- and twelve orders below any
departure a reader would call a different basis.

**The room is real, because every later gate honours this budget.** A transform
that is unitary only to ``E`` moves a knot's measured ``norm(W - I, 2)`` by
``norm(E + E.H, 2)``, which is this defect and not the metric's, so the per-knot
invariance check of :func:`solprop.translate.modal_tables.apply_channel_basis`
takes this constant as its floor rather than the rounding of its own product
(2.8e-14 for two channels, which would refuse a declaration that had just passed
here). The four orders are therefore the room a declaration actually has, not the
room it has until the next gate.
"""

MATERIAL_AGREEMENT_RTOL = 1e-6
"""Largest relative disagreement a step-index provider admits between a region's
declared permittivity law and the index its channels were solved with.

The modes own the eigenproblem's indices (``StepIndexMode.core`` and
``cladding``) and the regions own the material laws; the two describe the same
material and nothing else made them agree. The regions' laws are what a declared
``n2`` is converted against (:meth:`solprop.modes.material.MaterialKerr.susceptibility`)
and what a guide's identity records, so a core region declared at 2.0 beside
channels solved at 1.45 would propagate the 1.45 fields with a ``chi3`` 90 % too
large and nothing downstream would say so.

The budget is on the **permittivity**, and 1e-6 is chosen to admit a law evaluated
by two paths -- a glass law asked for its index and the same law asked for its
permittivity -- while refusing any disagreement a reader would call physical: the
smallest index step this package's own weakly guiding fixtures use is 0.001 in the
index, which is 1.4e-3 in the permittivity, three orders above this.
"""

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


def radial_node_rule(
    interfaces: Sequence[float],
    outer_radius: float,
    *,
    n_r: int,
    n_theta: int,
) -> ModalNodeRule:
    """Build interface-split Cartesian nodes from the existing polar panel rule.

    Parameters
    ----------
    interfaces : sequence of float
        Strictly increasing positive material radii in m, inside outer_radius.
    outer_radius : float
        Positive finite truncation radius in m; tail certification is separate.
    n_r : int
        At least two Gauss nodes in each radial panel.
    n_theta : int
        At least two angular nodes; caller certifies angular resolution.

    Returns
    -------
    ModalNodeRule
        Positive area weights and one int32 material index per node.

    Raises
    ------
    ModeError
        If the radius or interface arrangement is invalid.
    """
    if not np.isfinite(outer_radius) or outer_radius <= 0:
        raise ModeError("outer_radius must be positive and finite")
    rule = quadrature_rule(
        "polar",
        full=True,
        n_r=n_r,
        n_theta=n_theta,
        breaks=tuple(float(r / outer_radius) for r in interfaces),
    )
    limits = ("polar", (0.0, 0.0), (outer_radius, 2 * np.pi))
    polar = rule.nodes(limits)
    xy = np.column_stack(
        (polar[:, 0] * np.cos(polar[:, 1]), polar[:, 0] * np.sin(polar[:, 1]))
    )
    return ModalNodeRule(
        xy,
        rule.weights(limits),
        np.searchsorted(interfaces, polar[:, 0], side="left").astype(np.int32),
        {
            "interfaces": list(interfaces),
            "outer_radius": outer_radius,
            "n_r": n_r,
            "n_theta": n_theta,
        },
    )


@dataclass(frozen=True, eq=False)
class ChannelBasis(Displayed):
    """Declare the unitary coordinates a guide's degenerate channels are given in.

    A guide's channels arrive in whatever basis its provider solved them in, which
    for an azimuthally degenerate family is the real parity pair: the ``cos`` and
    ``sin`` patterns :class:`~solprop.modes.protocol.ChannelLabel` defines. Any
    unitary combination of an
    **exactly degenerate** block is as good a basis, and one of them -- the circular
    pair of :func:`circular_pair` -- diagonalises the isotropic Kerr tensor's
    polarisation dependence. This record is the declaration of that choice: the
    matrix, the blocks it acts inside, the labels the transformed channels carry and
    the identity a file records and a resume compares.

    **The transform acts on the reference, not on the state.** With ``e`` the
    provider's nodal field rows and ``U`` this matrix, the declared basis is
    ``e' = e U``, so a physical field whose reference-basis amplitudes are ``c`` has
    declared-basis amplitudes ``d = U.H c``: the columns of ``U`` are the new basis
    vectors written in the old one. Every derived object follows from that one rule
    -- ``W' = U.H W U``, ``G' = U.H G U``, ``Q' = U.H Q``, ``S' = S U`` -- and the
    noise draw made in the reference basis reaches the declared one as ``U.H`` times
    itself.

    **Whether a block is degenerate is decided where the table is built**, not here
    (:func:`solprop.translate.modal_tables.apply_channel_basis`): ``U.H diag(beta) U``
    is diagonal when the block's propagation constants are equal and is a coupled
    matrix flow, ``linear.coupled_flow``, when they are not.

    Attributes
    ----------
    transform : numpy.ndarray
        The unitary ``U``, shape (n_modes, n_modes), dimensionless. It is the
        identity outside `blocks`, and unitary to :data:`BASIS_UNITARY_ATOL`.
    blocks : tuple of tuple of int
        Disjoint channel-index blocks ``U`` acts inside, each with at least two
        members. They must be blocks the guide declared degenerate.
    labels : tuple of ChannelLabel
        One label per channel of the declared basis, in row order: the provider's
        own outside the blocks and the transformed ones inside them.
    gauge_id : str
        Non-empty identity of this declaration, composed into the tables' and the
        metric's ``gauge_id`` so that a run in one basis cannot resume in another.

    Raises
    ------
    solprop.exceptions.ModeError
        If the matrix is not square, finite, unitary or block-diagonal on `blocks`,
        if the blocks are not disjoint valid sets of at least two channels, if the
        label count disagrees with the matrix, or if the identity is empty.

    Examples
    --------
    >>> from solprop.modes.guide import circular_pair
    >>> from solprop.modes.protocol import ChannelLabel
    >>> pair = (
    ...     ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
    ...     ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
    ... )
    >>> basis = circular_pair(pair, (0, 1))
    >>> basis.gauge_id
    'circular(0,1)'
    >>> [label.display_label for label in basis.labels]
    ['HE1,1:left', 'HE1,1:right']
    """

    transform: Complex[np.ndarray, "n_modes_basis_out n_modes_basis_in"]
    blocks: tuple[tuple[int, ...], ...]
    labels: tuple[ChannelLabel, ...]
    gauge_id: str

    def __repr__(self) -> str:
        """Return a one-line description, never the matrix.

        Returns
        -------
        str
            For example ``ChannelBasis(circular(0,1), 2 channels, 1 block)``.
        """
        blocks = len(self.blocks)
        return line(
            "ChannelBasis",
            [
                ("", self.gauge_id),
                (
                    "",
                    f"{len(self.labels)} channel{'' if len(self.labels) == 1 else 's'}",
                ),
                ("", f"{blocks} block{'' if blocks == 1 else 's'}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "ChannelBasis: a declared unitary change of a guide's channel basis", [
            ("gauge", self.gauge_id),
            ("channels", ", ".join(label.display_label for label in self.labels)),
            ("blocks", ", ".join(str(list(block)) for block in self.blocks)),
            ("unitary defect", f"{self.unitary_defect:.3e}"),
        ]

    def __post_init__(self) -> None:
        """Copy, validate and freeze the matrix, and check the blocks and labels.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming what is out of range.
        """
        matrix = np.array(self.transform, dtype=np.complex128, order="C", copy=True)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not matrix.shape[0]
            or not np.all(np.isfinite(matrix))
        ):
            raise ModeError(
                "a channel basis transform must be a finite non-empty square "
                f"matrix, got shape {matrix.shape}"
            )
        matrix.setflags(write=False)
        object.__setattr__(self, "transform", matrix)
        object.__setattr__(
            self,
            "blocks",
            tuple(tuple(int(one) for one in block) for block in self.blocks),
        )
        object.__setattr__(self, "labels", tuple(self.labels))
        if not self.gauge_id:
            raise ModeError("a channel basis needs a non-empty gauge_id")
        count = matrix.shape[0]
        if len(self.labels) != count:
            raise ModeError(
                f"a channel basis needs one label per channel: the transform is "
                f"{count} wide and {len(self.labels)} label(s) were given"
            )
        self._check_blocks(count)
        defect = self.unitary_defect
        if not (defect <= BASIS_UNITARY_ATOL):
            raise ModeError(
                f"a declared change of basis must be unitary: this one is "
                f"{defect:.4g} from it, above the {BASIS_UNITARY_ATOL:.4g} budget. "
                "A nonunitary transform is a leaky or skewed basis whose flux "
                "metric is genuinely off the identity and whose flow is a coupled "
                "matrix exponential; neither has been measured in this package, and "
                "the coupled flow carries a unitary change of basis only"
            )

    def _check_blocks(self, count: int) -> None:
        """Require disjoint valid blocks and a matrix that is the identity outside.

        Parameters
        ----------
        count : int
            Validated channel count, dimensionless.

        Raises
        ------
        solprop.exceptions.ModeError
            Naming what is out of range.
        """
        seen: set[int] = set()
        membership = np.full(count, -1, dtype=int)
        for position, block in enumerate(self.blocks):
            if (
                len(block) < 2
                or len(set(block)) != len(block)
                or any(not 0 <= one < count for one in block)
                or seen.intersection(block)
            ):
                raise ModeError(
                    "a channel basis acts inside disjoint blocks of at least two "
                    f"distinct channels of the {count} it describes, got "
                    f"{[list(one) for one in self.blocks]}"
                )
            seen.update(block)
            membership[list(block)] = position
        # Outside a declared block the transform must be the identity: that is what
        # makes "this basis mixes only these channels" a checkable statement, and it
        # is what the exact-degeneracy gate on each block is then enough to cover.
        same = membership[:, None] == membership[None, :]
        allowed = same & (membership[:, None] >= 0)
        outside = self.transform - np.eye(count)
        if np.any(outside[~allowed] != 0.0):
            raise ModeError(
                "a channel basis transform must be the identity outside its "
                f"declared blocks {[list(one) for one in self.blocks]}: it mixes "
                "channels this guide has not declared degenerate, and the exact "
                "degeneracy that keeps the generator diagonal is checked per block"
            )

    @property
    def unitary_defect(self) -> float:
        """Return ``norm(U.H U - I, 2)``, how far the declaration is from unitary.

        Returns
        -------
        float
            Non-negative, dimensionless.
        """
        matrix = self.transform
        product = matrix.conj().T @ matrix
        return float(np.linalg.norm(product - np.eye(len(matrix)), ord=2))

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return what a run records of this declaration, and a resume compares.

        The matrix enters **by value**: two different unitary transforms of the same
        block are two different bases, and the block list alone does not tell them
        apart. Its entries are declared numbers, not measurements, so they enter
        exactly, as a declared ``chi3`` or a knot does
        (:meth:`solprop.modes.guide.ModalSampling.identity`).

        Returns
        -------
        collections.abc.Mapping
            JSON-serialisable scalars and lists of scalars, stable across processes.
        """
        return {
            "gauge_id": self.gauge_id,
            "blocks": [[int(one) for one in block] for block in self.blocks],
            # The two parts by name, as lists of floats: a complex number has no
            # JSON form, and a pair of real matrices is what survives a round trip.
            "transform_real": [
                [float(one) for one in row] for row in np.real(self.transform)
            ],
            "transform_imag": [
                [float(one) for one in row] for row in np.imag(self.transform)
            ],
            "labels": [dict(label.as_mapping()) for label in self.labels],
        }


def provider_channels(provider: ModalProvider) -> tuple[ChannelLabel, ...]:
    """Return a provider's declared channels as labels, in its own order.

    A provider publishes its channels as mappings inside ``identity``, which is what
    survives JSON; this is the one place they are turned back into
    :class:`~solprop.modes.protocol.ChannelLabel` objects for a caller that has to
    name one -- to declare a change of basis over a pair, most of all.

    Parameters
    ----------
    provider : solprop.modes.vector.ModalProvider
        A provider that names its channels.

    Returns
    -------
    tuple
        One label per channel.

    Raises
    ------
    ModeError
        If the provider names no channels.
    """
    declared = provider.identity.get("channels")
    if not isinstance(declared, list) or not declared:
        raise ModeError(
            "this provider names no channels, so its labels cannot be read; "
            f"{type(provider).__name__} declares none"
        )
    labels = []
    for mapping in declared:
        if not isinstance(mapping, dict):
            raise ModeError("a declared channel must be a label mapping")
        labels.append(ChannelLabel.from_mapping(mapping))
    return tuple(labels)


def circular_pair(labels: Sequence[ChannelLabel], block: Sequence[int]) -> ChannelBasis:
    """Declare the circular partners of one exactly degenerate parity pair.

    The change of basis that turns the real ``cos``/``sin`` pair of an azimuthally
    degenerate family into its two circular partners, which is the basis an
    isotropic Kerr tensor is diagonal in: for a field that is circular at every
    node ``E . E`` vanishes, so ``(E . E) E*`` drops out of
    ``P = eps0 chi3 (2 |E|^2 E + (E . E) E*) / 4`` and the self coefficient falls to
    two thirds of the linear one.

    **The convention, derived here.** The even partner is the ``cos((n-1) theta)``
    pattern about **x**, which for ``n = 1`` is the x-polarised member, and the odd
    one is its quarter-turn partner about **y**
    (:class:`~solprop.modes.protocol.ChannelLabel`). A provider's
    fields are in the **internal** convention of ``docs/dev/conventions.md``
    section 2, whose plane wave is ``exp(i(omega t - beta z))``, so the physical
    transverse field of a channel with nodal rows ``(e_even, e_odd)`` and amplitude
    one is ``Re[(e_even + a e_odd) exp(i omega t)]``. With ``a = -i`` that is
    ``(cos(omega t), sin(omega t))`` in ``(x, y)``: the vector turns from x towards
    y, which an observer looking **towards the source** sees turn anticlockwise.
    That is positive helicity, which optics calls **left** circular (Jackson,
    *Classical Electrodynamics*, section 7.2), and its optics-convention Jones
    vector is the conjugate ``(1, i)/sqrt(2)``, which
    :mod:`solprop.polarisation` records as positive Stokes ``V``. So

    ``e_left  = (e_even - i e_odd)/sqrt(2)``, ``e_right = (e_even + i e_odd)/sqrt(2)``

    and the two columns of the returned matrix are exactly those, in that order:
    ``block[0]`` becomes the **left** channel and ``block[1]`` the **right** one.
    The order is not free -- ``block[0]`` must be the **even** member and
    ``block[1]`` the odd one, and a block given the other way round is refused by
    name. The names *left* and *right* are a convention, fixed by the derivation
    above, not a derived fact.

    Parameters
    ----------
    labels : collections.abc.Sequence
        Every channel's label, in the guide's own row order.
    block : collections.abc.Sequence
        Exactly two channel indices, one labelled ``"even"`` and one ``"odd"``,
        of the same family and quantum numbers.

    Returns
    -------
    ChannelBasis
        The declaration, with ``gauge_id`` ``"circular(i,j)"`` for the block's own
        two indices in ascending order and the two labels relabelled by handedness.

    Raises
    ------
    solprop.exceptions.ModeError
        If the block is not two distinct valid indices, or the two labels are not a
        parity pair of one family.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.guide import circular_pair
    >>> from solprop.modes.protocol import ChannelLabel
    >>> pair = (
    ...     ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
    ...     ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
    ... )
    >>> basis = circular_pair(pair, (0, 1))
    >>> bool(np.allclose(basis.transform[:, 0], [1, -1j] / np.sqrt(2)))
    True
    >>> basis.labels[1].handedness
    'right'
    """
    indices = [int(one) for one in block]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ModeError(
            f"a circular pair is exactly two distinct channels, got {list(block)}"
        )
    if any(not 0 <= one < len(labels) for one in indices):
        raise ModeError(
            f"circular pair {indices} is outside the {len(labels)} channel(s) given"
        )
    even, odd = (labels[one] for one in indices)
    if (even.parity, odd.parity) != ("even", "odd") or (
        even.family,
        even.azimuthal_order,
        even.radial_order,
    ) != (odd.family, odd.azimuthal_order, odd.radial_order):
        raise ModeError(
            "a circular pair is the even and the odd parity of one family, in that "
            f"order: got {even.display_label!r} ({even.parity}) and "
            f"{odd.display_label!r} ({odd.parity})"
        )
    root = 1.0 / np.sqrt(2.0)
    matrix = np.eye(len(labels), dtype=np.complex128)
    matrix[np.ix_(indices, indices)] = np.array(
        [[root, root], [-1j * root, 1j * root]], dtype=np.complex128
    )
    base = even.display_label.removesuffix(":even")
    turned = [
        ChannelLabel(
            even.family,
            even.azimuthal_order,
            even.radial_order,
            None,
            hand,
            f"{base}:{hand}",
        )
        for hand in ("left", "right")
    ]
    relabelled = list(labels)
    for position, label in zip(indices, turned, strict=True):
        relabelled[position] = label
    low, high = sorted(indices)
    return ChannelBasis(
        matrix,
        ((indices[0], indices[1]),),
        tuple(relabelled),
        f"circular({low},{high})",
    )


@dataclass(frozen=True, eq=False)
class StepIndexProvider(Displayed):
    """Evaluate exact bound step-index E/H using independent full-domain power norms.

    A returned :class:`~solprop.modes.vector.ModalSample` is always fully valid:
    an unsupported channel is refused by name rather than masked out, so
    ``valid`` is all ``True`` in every sample this provider returns.

    Attributes
    ----------
    modes : tuple of StepIndexMode
        Unique physical channels of one core/cladding geometry, in state order.
    regions : tuple of MaterialRegion
        Core and unbounded exterior, with material laws matching the modes.
    """

    modes: tuple[StepIndexMode, ...]
    regions: tuple[MaterialRegion, ...]

    def __repr__(self) -> str:
        """Return a one-line description: the channels and the region count.

        Returns
        -------
        str
            For example ``StepIndexProvider(HE11, 2 regions)``.
        """
        return line(
            "StepIndexProvider",
            [
                ("", ", ".join(f"{m.kind}{m.n}{m.m}" for m in self.modes)),
                ("", f"{len(self.regions)} regions"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        low, high = self.z_domain
        return "StepIndexProvider: exact bound E/H, unit full-domain power", [
            ("channels", ", ".join(f"{m.kind}{m.n}{m.m}" for m in self.modes)),
            ("regions", ", ".join(region.name for region in self.regions)),
            ("z domain", f"{engineering(low, 'm')} to {engineering(high, 'm')}"),
        ]

    def __post_init__(self) -> None:
        """Validate the common guide and the two material regions.

        Raises
        ------
        ModeError
            If the channels or material ownership are inconsistent.
        """
        ModeCollection(self.modes)
        if any(
            not ModeCollection(self.modes)._same_stepindex_guide(m) for m in self.modes
        ):
            raise ModeError("step-index channels must share one physical guide")
        if len(self.regions) != 2 or self.regions[-1].outer_radius is not None:
            raise ModeError("step-index provider requires core and unbounded cladding")
        self._check_material_agreement()

    def _check_material_agreement(self) -> None:
        """Require each region's law to be the index its channels were solved with.

        Raises
        ------
        ModeError
            If a region's law does not cover the channels' profile wavelength, or
            its permittivity there departs from the channels' own index squared by
            more than :data:`MATERIAL_AGREEMENT_RTOL`.
        """
        mode = self.modes[0]
        omega = float(2 * np.pi * C / mode.profile_wavelength)
        # The resolved laws, not the declared ones: `core=None` means fused silica
        # and the mode has already substituted it.
        solved = (mode._core, mode._cladding)
        for region, law in zip(self.regions, solved, strict=True):
            low, high = region.material.omega_domain
            if not low <= omega <= high:
                raise ModeError(
                    f"region {region.name!r} declares a law valid from {low:.6e} to "
                    f"{high:.6e} rad/s, which does not cover the channels' profile "
                    f"wavelength {mode.profile_wavelength:.6e} m ({omega:.6e} "
                    "rad/s): the region's law is what a declared n2 is converted "
                    "against and what the guide's identity records"
                )
            index = float(_material(law, np.array([omega]))[0][0])
            declared = float(
                np.real(region.material.permittivity(np.array([omega]))[0])
            )
            if abs(declared - index**2) > MATERIAL_AGREEMENT_RTOL * index**2:
                raise ModeError(
                    f"region {region.name!r} declares relative permittivity "
                    f"{declared!r} at the channels' profile wavelength "
                    f"{mode.profile_wavelength:.6e} m, and the channels were solved "
                    f"with index {index!r} there, which is {index**2!r}: a relative "
                    f"{abs(declared - index**2) / index**2:.3g}, above "
                    f"{MATERIAL_AGREEMENT_RTOL:.3g}. The eigenproblem uses the "
                    "modes' indices and the region's law is what a declared n2 is "
                    "converted against, so the two must be the same material"
                )

    @property
    def z_breakpoints(self) -> tuple[float, ...]:
        """Return radius-path breakpoints in m.

        Returns
        -------
        tuple of float
            Positions of physical path kinks in m.
        """
        return tuple(float(x) for x in self.modes[0].core_radius.breakpoints)

    @property
    def z_domain(self) -> tuple[float, float]:
        """Return the radius-path interval in m.

        Returns
        -------
        tuple of float
            Lower and upper position in m.
        """
        return self.modes[0].core_radius.z_span

    @property
    def omega_domain(self) -> tuple[float, float]:
        """Return material frequency support; solve checks each mode cutoff.

        Returns
        -------
        tuple of float
            Lower and upper angular frequency in rad/s.
        """
        return max(r.material.omega_domain[0] for r in self.regions), min(
            r.material.omega_domain[1] for r in self.regions
        )

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return internal convention, bound-model and channel identities.

        Returns
        -------
        collections.abc.Mapping
            Serializable physical model and construction metadata.
        """
        return {
            "spectral_convention": "internal",
            "model": "bound-step-index",
            "channels": [
                dict(label.as_mapping())
                for label in ModeCollection(self.modes).channel_labels
            ],
            "materials": [dict(r.material.identity) for r in self.regions],
            # A structural path identity, not a repr: numpy summarises an array over
            # 1000 elements, so two different long tapers would share an identity.
            "radius_path": cast(
                "JsonValue", profile_identity(self.modes[0].core_radius)
            ),
        }

    def node_rule(
        self,
        omega: float,
        z: float = 0.0,
        *,
        n_r: int | None = None,
        n_theta: int | None = None,
    ) -> ModalNodeRule:
        """Build a common rule and certify the discarded cladding power.

        Parameters
        ----------
        omega : float
            Positive frequency in rad/s in the guided band.
        z : float, optional
            Position in m within the radius path.
        n_r : int or None, optional
            Gauss nodes per radial panel, at least two; ``None`` uses the 64 this
            provider's exponential cladding tail needs.
        n_theta : int or None, optional
            Angular node count; None uses 4*(maximum n+1)+1 for quartic products.

        Returns
        -------
        ModalNodeRule
            Material-split rule with independently integrated tail bounds.

        Raises
        ------
        ModeError
            If the bound field or tail cannot be certified.
        """
        omega, z = float(omega), float(z)
        n_r = 64 if n_r is None else n_r
        radius = self.modes[0]._radius(z)
        decay = []
        for mode in self.modes:
            neff = float(mode.neff(np.array([omega], dtype=float), z)[0].real)
            clad = float(
                _material(mode._cladding, np.array([omega], dtype=float))[0][0]
            )
            decay.append(omega / C * np.sqrt((neff - clad) * (neff + clad)))
        outer = radius + 24 / min(decay)
        rule = radial_node_rule(
            (radius,),
            outer,
            n_r=n_r,
            n_theta=4 * (max(m.n for m in self.modes) + 1) + 1
            if n_theta is None
            else n_theta,
        )
        tails = []
        for mode in self.modes:
            total, error = _power(mode, omega, z)
            tail, tail_error = _power(mode, omega, z, outer / radius)
            if total <= error or (abs(tail) + tail_error) / (total - error) > 1e-12:
                raise ModeError(
                    "cladding tail could not be certified below 1e-12 of full power"
                )
            tails.append((abs(tail) + tail_error) / (total - error))
        return ModalNodeRule(
            rule.xy,
            rule.weights,
            rule.region_index,
            {
                **rule.identity,
                "tail_power_bound": tails,
                # Beside the bound, so that a consumer reads the certificate
                # without having to know that "omega" is the frequency it holds
                # at. A tail bound is only a bound at the frequency it was taken.
                "tail_power_bound_omega": omega,
                "omega": omega,
                "z": z,
            },
        )

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Solve bound channels on explicitly owned physical nodes.

        Parameters
        ----------
        omega : float
            Positive angular frequency in rad/s within material and guided support.
        z : float
            Position in m in the radius path.
        nodes : ModalNodeRule
            Common Cartesian nodes with correct core/cladding ownership.

        Returns
        -------
        ModalSample
            Unit full-domain power E/H and their sampled physical Gram matrix. Its
            ``valid`` mask is all ``True``: an unsupported channel is refused here,
            not masked.

        Raises
        ------
        ModeError
            If domains, material ownership, norm or a boundary residual is invalid.
        """
        if (
            not self.omega_domain[0] < omega <= self.omega_domain[1]
            or not self.z_domain[0] <= z <= self.z_domain[1]
        ):
            raise ModeError("sample is outside provider frequency or z domain")
        omega, z = float(omega), float(z)
        radius = self.modes[0]._radius(z)
        boundaries = [
            float(as_profile(value)(np.array([z], dtype=float))[0])
            for value in (
                self.regions[0].inner_radius,
                self.regions[0].outer_radius,
                self.regions[1].inner_radius,
            )
            if value is not None
        ]
        if boundaries != [0.0, radius, radius]:
            raise ModeError("material region radii disagree with the core interface")
        expected = (np.hypot(nodes.xy[:, 0], nodes.xy[:, 1]) > radius).astype(np.int32)
        if not np.array_equal(nodes.region_index, expected):
            raise ModeError("node region ownership disagrees with the core interface")
        for region, material in zip(
            self.regions, (self.modes[0]._core, self.modes[0]._cladding), strict=True
        ):
            eps = region.material.permittivity(np.array([omega], dtype=float))
            target = _material(material, np.array([omega], dtype=float))[0] ** 2
            if np.any(np.imag(eps) != 0) or not np.allclose(
                np.real(eps), target, rtol=16 * np.finfo(float).eps, atol=0
            ):
                raise ModeError(
                    "provider material differs from its lossless step-index law"
                )
        electric, magnetic, beta, residual = [], [], [], []

        angles = np.arange(17) * 2 * np.pi / 17
        boundary = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        # Compare exact one-sided traces, without a finite radial offset.
        radial = boundary / radius
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        for mode in self.modes:
            e, h = _fields(mode, nodes.xy, omega, z)
            norm, error = _power(mode, omega, z)
            if norm <= error:
                raise ModeError("full-domain power norm is non-positive or unresolved")
            ei, hi = _fields(mode, boundary, omega, z, exterior=False)
            eo, ho = _fields(mode, boundary, omega, z, exterior=True)
            eps_core = (
                float(_material(mode._core, np.array([omega], dtype=float))[0][0]) ** 2
            )
            eps_clad = (
                float(_material(mode._cladding, np.array([omega], dtype=float))[0][0])
                ** 2
            )
            e_defect = np.column_stack(
                (
                    np.sum((ei[:, :2] - eo[:, :2]) * tangent, axis=1),
                    ei[:, 2] - eo[:, 2],
                    np.sum(
                        (eps_core * ei[:, :2] - eps_clad * eo[:, :2]) * radial, axis=1
                    )
                    / eps_core,
                )
            )
            defect = max(
                float(np.max(abs(e_defect))) / float(np.max(abs(ei))),
                float(np.max(abs(hi - ho))) / float(np.max(abs(hi))),
            )
            if defect > 1e-9:
                raise ModeError("step-index Maxwell boundary residual exceeds 1e-9")
            electric.append(e / np.sqrt(norm))
            magnetic.append(h / np.sqrt(norm))
            beta.append(
                float(mode.neff(np.array([omega], dtype=float), z)[0].real) * omega / C
            )
            residual.append(defect)
        e, h = np.stack(electric, axis=-1), np.stack(magnetic, axis=-1)
        metric = flux_product(e, h, nodes.weights)
        return ModalSample(
            omega,
            z,
            e,
            h,
            -1j * np.diag(beta),
            metric,
            np.array(residual),
            np.ones(len(beta), dtype=bool),
            e.conj(),
            -h.conj(),
            -4 * metric,
            "analytic-real-transverse",
            json.dumps(dict(self.identity), sort_keys=True, allow_nan=False),
        )


@dataclass(frozen=True, eq=False)
class ModalPath(Displayed):
    """Record how a moving guide was sampled along z, and what that measured.

    The z half of a guide's sampling (:class:`ModalSampling`), filled by
    :meth:`ModalGuide.from_provider` for a provider whose geometry moves. The path
    is cut into **sections** -- the provider's declared kink intervals, each divided
    `sections` times -- and every section is
    solved at the Chebyshev-Lobatto points of one degree along z, the degree the
    accuracy check demanded: doubled until the expansion reproduced a fresh solve at
    the points a doubling adds to within `rtol`
    (:func:`solprop.translate.modal_path.sample_modal_path`). A run rebuilds the same
    positions from this record rather than searching again.

    Attributes
    ----------
    z_edges : numpy.ndarray
        Section edges in m, strictly increasing, shape ``(n_edges_path,)``: the
        provider's z domain split at its interior breakpoints and subdivided.
    degree : int
        The Chebyshev degree along z every section is sampled at, positive; each
        section holds ``degree + 1`` positions, its two ends included.
    sections : int
        How many sections each kink interval was divided into, positive.
    rtol : float
        The budget the degree was searched against, positive: by default the
        frequency interpolation's own measured profile error, so that the sampling in
        z is never the coarser of the two.
    z_error : float
        The measured held-out departure at the degree kept, dimensionless and
        non-negative: the largest relative maximum-norm departure of E, H or the
        generator from a fresh solve between the sampled positions.
    gram_departure : float
        The largest ``norm(W - I, 2)`` of the expanded basis at those same held-out
        positions, dimensionless and non-negative.

    Raises
    ------
    ModeError
        Naming the field that is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.guide import ModalPath
    >>> path = ModalPath(np.array([0.0, 1e-3]), 2, 1, 1e-6, 3e-7, 1e-8)
    >>> path.positions.round(6).tolist()
    [[0.0, 0.0005, 0.001]]
    >>> round(float(path.interpolate(np.array([0.0, 1.0, 2.0]), 2.5e-4)), 12)
    0.5
    """

    z_edges: Float[np.ndarray, " n_edges_path"]
    degree: int
    sections: int
    rtol: float
    z_error: float
    gram_departure: float

    def __post_init__(self) -> None:
        """Freeze the edges and check every declaration and measurement.

        Raises
        ------
        ModeError
            Naming the field that is out of range.
        """
        edges = np.array(self.z_edges, dtype=np.float64, copy=True).reshape(-1)
        if (
            edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ModeError(
                "a path's section edges are at least two finite, strictly increasing "
                f"positions, got {edges.tolist()}"
            )
        edges.setflags(write=False)
        object.__setattr__(self, "z_edges", edges)
        for name in ("degree", "sections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModeError(f"a path's {name} is a positive integer, got {value!r}")
        for name, strict in (
            ("rtol", True),
            ("z_error", False),
            ("gram_departure", False),
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or (strict and value == 0.0):
                raise ModeError(
                    f"a path's {name} is a {'positive' if strict else 'non-negative'} "
                    f"finite number, got {getattr(self, name)!r}"
                )

    @property
    def n_intervals(self) -> int:
        """Return how many sections the path is cut into.

        Returns
        -------
        int
            The section count.
        """
        return int(self.z_edges.size - 1)

    @property
    def positions(self) -> Float[np.ndarray, "n_intervals_path n_deg_path"]:
        """Return every sampled position, section by section.

        Returns
        -------
        numpy.ndarray
            Positions in m, shape ``(n_intervals, degree + 1)``, increasing inside a
            section, so a position on an interior edge appears twice.
        """
        # Imported here: solprop.translate imports solprop.modes. The positions are
        # the translation layer's own Chebyshev-Lobatto points, bit for bit, because
        # the run solves exactly these positions again.
        from solprop.translate.chebyshev import lobatto_points  # noqa: PLC0415

        return np.stack(
            [
                lobatto_points(self.degree, float(lo), float(hi))
                for lo, hi in zip(self.z_edges[:-1], self.z_edges[1:], strict=True)
            ]
        )

    def section(self, z: float) -> int:
        """Return the section a position lies in, clamped to the path's ends.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        int
            The section index; a position on an interior edge takes the section
            starting there, and one outside the path the nearest end's.
        """
        raw = int(np.searchsorted(self.z_edges, float(z), side="right")) - 1
        return min(max(raw, 0), self.n_intervals - 1)

    def interpolate(
        self,
        values: Inexact[np.ndarray, " n_positions *value"],
        z: float,
    ) -> np.ndarray:
        """Evaluate values sampled at the path's positions at any position.

        The barycentric form of the interpolating polynomial through one section's
        Chebyshev-Lobatto samples -- weights ``(-1)^j``, halved at the two ends -- which
        is the same polynomial the translation layer's Chebyshev coefficients carry,
        evaluated stably. A position outside the path **holds** the nearest end's
        value, as a :class:`~solprop.profiles.ZProfile` holds its own beyond its span:
        a run's tables reach a few per cent past its end, and the geometry is not
        declared there.

        Parameters
        ----------
        values : numpy.ndarray
            One value per position, in :attr:`positions` order flattened, position
            axis first, shape ``(n_intervals * (degree + 1), *value_shape)``.
        z : float
            Position in m.

        Returns
        -------
        numpy.ndarray
            The value at `z`, shape ``value_shape``.

        Raises
        ------
        ModeError
            If `values` does not carry one row per position.
        """
        rows = np.asarray(values)
        width = self.degree + 1
        if rows.shape[0] != self.n_intervals * width:
            raise ModeError(
                f"a path of {self.n_intervals} section(s) at degree {self.degree} "
                f"interpolates one value per position, {self.n_intervals * width} in "
                f"all, and got {rows.shape[0]}"
            )
        index = self.section(z)
        lo, hi = float(self.z_edges[index]), float(self.z_edges[index + 1])
        x = min(max((2.0 * float(z) - lo - hi) / (hi - lo), -1.0), 1.0)
        section = rows[index * width : (index + 1) * width]
        nodes = -np.cos(np.pi * np.arange(width) / self.degree)
        # A position that maps onto a node to within a few ulp -- the mid-section
        # node is -cos(pi/2) = -6e-17, which no position maps to exactly -- is that
        # node, whose value is returned as sampled rather than through a quotient.
        exact = np.flatnonzero(np.abs(nodes - x) <= 4.0 * np.finfo(float).eps)
        if exact.size:
            return np.asarray(section[int(exact[0])])
        weights = (-1.0) ** np.arange(width)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        terms = weights / (x - nodes)
        return np.asarray(np.tensordot(terms, section, axes=1) / float(np.sum(terms)))

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return what a run records of the path, and a resume compares.

        The declared edges, degree, subdivision and budget exactly; the two measured
        departures at :data:`~solprop.modes.material.MEASURED_DIGITS`, as every other
        measured entry of a guide's identity is.

        Returns
        -------
        collections.abc.Mapping
            JSON-serialisable scalars and lists of scalars.
        """
        return {
            "z_edges": [float(one) for one in self.z_edges],
            "degree": int(self.degree),
            "sections": int(self.sections),
            "rtol": float(self.rtol),
            "z_error": quantised_measurement(self.z_error),
            "gram_departure": quantised_measurement(self.gram_departure),
        }

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``ModalPath(1 section, degree 8, 0 to 1 mm)``.
        """
        count = self.n_intervals
        return line(
            "ModalPath",
            [
                ("", f"{count} section{'' if count == 1 else 's'}"),
                ("", f"degree {self.degree}"),
                (
                    "",
                    f"{engineering(float(self.z_edges[0]), 'm')} to "
                    f"{engineering(float(self.z_edges[-1]), 'm')}",
                ),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "ModalPath: how a moving guide was sampled along z", [
            ("sections", f"{self.n_intervals} ({self.sections} per kink interval)"),
            (
                "span",
                f"{engineering(float(self.z_edges[0]), 'm')} to "
                f"{engineering(float(self.z_edges[-1]), 'm')}",
            ),
            ("degree", self.degree),
            ("positions", self.n_intervals * (self.degree + 1)),
            ("budget", f"{self.rtol:.3e}"),
            ("held-out error", f"{self.z_error:.3e}"),
            ("gram departure", f"{self.gram_departure:.3e}"),
        ]


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
        (:func:`solprop.translate.modal_path.path_rule`).
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
    (:func:`solprop.translate.modal_tables.sample_provider_table`), and its identity
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
    :func:`solprop.modes.base.zdw`, :func:`solprop.tools.dispersion_length` and
    :func:`solprop.tools.soliton_order` are given for this channel -- is
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
        (:func:`solprop.translate.modal_tables.channel_area`) at the **median knot**
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
    def z_constant(self) -> bool:
        """Whether the channel is independent of position.

        Returns
        -------
        bool
            ``True`` for a snapshot at one declared position; ``False`` on a guide
            whose geometry moves, whose channel carries a sampled path.
        """
        return self.path is None

    def entrance(self) -> "ProviderChannelMode":  # noqa: UP037 - the hook resolves it while the body runs
        """Return this channel's snapshot at the entrance, with no path.

        What a moving guide's cumulative ``Phi`` is carried by: the run puts one
        reference channel's entrance dispersion in ``Phi`` and every channel's
        departure from it, the motion included, in the coupled flow
        (``solprop.translate.problem._general_assembly``).

        Returns
        -------
        ProviderChannelMode
            The same channel, z-constant, with its entrance table and area.
        """
        return ProviderChannelMode(
            self.table, self.label, self.outer_radius, self.area, source=self.source
        )

    @property
    def area_path(self) -> ZProfile | None:
        """Return the surrogate area along a moving guide's path, or ``None``.

        Declared under the cross-section name a moving mode declares its area by
        (:data:`solprop.modes.protocol.GEOMETRY_PATH_NAMES`), so that the tables of
        a run split on the path's section edges, where the interpolant's slope may
        jump.

        Returns
        -------
        solprop.profiles.ZProfile or None
            The area in m² along z, or ``None`` for a snapshot.
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

        Parameters
        ----------
        values : sequence of numpy.ndarray
            One value per path position.
        z : float
            Position in m.

        Returns
        -------
        numpy.ndarray
            The value at `z`.
        """
        path = self.path
        assert path is not None
        return path.interpolate(np.asarray(values), z)

    @property
    def scale_invariant(self) -> bool:
        """Whether the profile scales exactly with a core radius.

        Returns
        -------
        bool
            Always ``False``: a provider channel has no scaling law, and its
            normalised frequency changes with the geometry.
        """
        return False

    @property
    def azimuthal_order(self) -> int:
        """Return the label's azimuthal order, or zero where it has none.

        Returns
        -------
        int
            Azimuthal order, dimensionless.
        """
        return self.label.azimuthal_order or 0

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the provider's own label for this channel.

        Returns
        -------
        ChannelLabel
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
        it: :func:`solprop.translate.modal_tables.validate_modal_table` measures no
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
            :func:`solprop.translate.modal_tables.channel_area` at the median knot
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
        :func:`solprop.translate.modal_tables.channel_area`, ``(int |E|^2 dA)^2 /
        int |E|^4 dA`` over the three Cartesian components, the longitudinal one
        included -- but at the frequency asked for rather than at the median knot of
        the sampled band, which is where a scalar model's pump usually is not. The
        provider is solved at `omega` on the guide's own rule: the sampling rule
        for a snapshot, and for a moving guide the rule the provider builds at `z`
        on the same request (:func:`solprop.translate.modal_path.path_rule`), which
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
        # Imported here: solprop.translate imports solprop.modes.
        from solprop.translate.modal_path import path_rule  # noqa: PLC0415
        from solprop.translate.modal_tables import (  # noqa: PLC0415
            apply_channel_basis,
            channel_area,
        )

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

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Areas in m², same shape.
        """
        return np.array([self.channel.aeff(float(one)) for one in np.asarray(z)])


@dataclass(frozen=True, eq=False)
class _SolvedPathArea:
    """Evaluate areas solved at a moving channel's path positions, along z."""

    path: ModalPath
    areas: tuple[float, ...]

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the area at every position, m².

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Areas in m², same shape.
        """
        values = np.asarray(self.areas)
        return np.array(
            [float(self.path.interpolate(values, float(one))) for one in np.asarray(z)]
        )


def _profile_defect(interpolated: ModalSample, solved: ModalSample) -> float:
    """Return the relative maximum-norm defect between two aligned samples' fields.

    Parameters
    ----------
    interpolated : solprop.modes.vector.ModalSample
        The table's frequency interpolation at one frequency.
    solved : solprop.modes.vector.ModalSample
        A fresh provider solve there, already rotated onto the interpolation by
        :func:`solprop.translate.modal_tables.align_modal_sample`.

    Returns
    -------
    float
        The larger of the electric and magnetic relative defects, dimensionless
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
        knots = np.ascontiguousarray(self.omega, dtype=np.float64).reshape(-1)
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
        knots.setflags(write=False)
        object.__setattr__(self, "omega", knots)
        if not np.isfinite(self.z):
            raise ModeError(f"a guide's snapshot position must be finite, got {self.z}")
        if not np.isfinite(self.profile_error) or self.profile_error < 0.0:
            raise ModeError(
                "a guide's measured profile error must be non-negative and finite, "
                f"got {self.profile_error!r}"
            )

    @property
    def omega_span(self) -> tuple[float, float]:
        """Return the band the guide was solved over.

        Returns
        -------
        tuple of float
            Lowest and highest knot in rad/s.
        """
        return float(self.omega[0]), float(self.omega[-1])

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return what a run records of this sampling, and a resume compares.

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

        Returns
        -------
        collections.abc.Mapping
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
    ) -> "ModalGuide":  # noqa: UP037 - runtime hook evaluates class annotations eagerly
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
    def from_provider(  # noqa: PLR0913 - a provider and five declarations of the sampling
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
    ) -> "ModalGuide":  # noqa: UP037 - runtime hook evaluates class annotations eagerly
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
        (:func:`solprop.translate.modal_path.sample_modal_path`). The checks below are
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
            :func:`solprop.translate.modal_tables.sample_modal_table` takes them.
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
            :data:`solprop.translate.modal_path.PATH_RTOL_FLOOR`, so that the
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
        # Imported here: solprop.translate imports solprop.modes, so a module-level
        # import of the bridge would close a cycle through solprop.translate's own
        # package body.
        from solprop.translate.modal_tables import (  # noqa: PLC0415
            DenseModalTable,
            _area_knot,
            _channel_index_table,
            align_modal_sample,
            channel_area,
            generator_is_coupled,
            solved_in_table_basis,
        )

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
        reference_sample = table.samples[0][_area_knot(frequencies)]
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
            index = _channel_index_table(
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
    def omega_domain(self) -> tuple[float, float]:
        """Return the provider frequency interval in rad/s.

        Returns
        -------
        tuple of float
            Lower and upper angular frequency in rad/s.
        """
        return self.field_provider.omega_domain

    @property
    def z_domain(self) -> tuple[float, float]:
        """Return the provider longitudinal interval in m.

        Returns
        -------
        tuple of float
            Lower and upper position in m.
        """
        return self.field_provider.z_domain

    @property
    def z_breakpoints(self) -> tuple[float, ...]:
        """Return the provider physical breakpoints in m.

        Returns
        -------
        tuple of float
            Positions of physical path kinks in m.
        """
        return self.field_provider.z_breakpoints

    @property
    def identity(self) -> Mapping[str, JsonValue]:
        """Return the provider physical identity.

        Returns
        -------
        collections.abc.Mapping
            Serializable physical model and construction metadata.
        """
        return self.field_provider.identity


def _moving_geometry(provider: ModalProvider) -> list[str]:
    """Return what moves along z in a provider's declared geometry, by name.

    Parameters
    ----------
    provider : ModalProvider
        The guide.

    Returns
    -------
    list of str
        Each region radius whose path is not constant, and the interior kinks; empty
        for a guide whose geometry does not move.
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

    Parameters
    ----------
    provider : ModalProvider
        The moving guide.
    nodes : ModalNodeRule
        The declared entrance rule.
    z : float
        The entrance, m.

    Raises
    ------
    ModeError
        If the provider, asked for the rule the way `nodes` says it was asked for,
        returns a different one at the entrance.
    """
    # Imported here: solprop.translate imports solprop.modes.
    from solprop.translate.modal_path import path_rule  # noqa: PLC0415

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

    Parameters
    ----------
    provider : ModalProvider
        The moving guide.
    table : solprop.translate.modal_tables.DenseModalTable
        The entrance snapshot, already solved and checked.
    modes : sequence of ProviderChannelMode
        The channels, built from the entrance snapshot.
    sections : int
        Sections per kink interval.
    rtol : float
        The held-out budget, before the floor.

    Returns
    -------
    tuple
        The path record and the channels carrying it.
    """
    # Imported here: solprop.translate imports solprop.modes.
    from solprop.translate.modal_path import (  # noqa: PLC0415
        PATH_RTOL_FLOOR,
        path_sections,
        sample_modal_path,
    )
    from solprop.translate.modal_tables import (  # noqa: PLC0415
        DenseModalTable,
        _area_knot,
        _channel_index_table,
        channel_area,
        generator_is_coupled,
    )

    assert isinstance(table, DenseModalTable)
    edges = path_sections(provider, sections)
    budget = max(float(rtol), PATH_RTOL_FLOOR)
    sampled = sample_modal_path(
        provider,
        table.omega,
        table.nodes,
        edges,
        rtol=budget,
        degenerate_blocks=table.degenerate_blocks,
        basis=table.basis,
        gram_atol=table.gram_atol,
        known={float(table.z[0]): table},
    )
    path = ModalPath(
        edges,
        sampled.degree,
        sections,
        budget,
        sampled.z_error,
        sampled.gram_departure,
    )
    knot = _area_knot(table.omega)
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
                    _channel_index_table(
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


def _entrance_table(  # noqa: PLR0913 - a provider, a band, a position, a rule and three declarations
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

    Parameters
    ----------
    provider : ModalProvider
        The guide.
    frequencies : numpy.ndarray
        The knots in rad/s.
    z : float
        The declared snapshot position, m.
    nodes : ModalNodeRule or None
        The declared rule, or ``None`` for the provider's own at the lowest knot.
    degenerate_blocks : sequence of sequence of int
        The declared degenerate blocks.
    basis : ChannelBasis or None
        The declared change of basis.
    path_declared : bool
        Whether a path's sections or budget were declared.

    Returns
    -------
    tuple
        What moves (empty for a snapshot), the position the table is at, the rule
        and the table.

    Raises
    ------
    ModeError
        If a path is declared for a snapshot, a moving guide is asked for a
        snapshot position other than its entrance, or its rule is not the
        provider's own.
    """
    # Imported here: solprop.translate imports solprop.modes.
    from solprop.translate.modal_path import path_snapshot  # noqa: PLC0415
    from solprop.translate.modal_tables import sample_modal_table  # noqa: PLC0415

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
