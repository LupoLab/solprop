# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Condense a capillary's outgoing wall condition onto its retained Marcatili modes.

The coherent wall replaces each mode's own diagonal Marcatili loss by the rank-one
Robin wall term that couples every retained mode to every other with a definite
phase. :class:`CoherentWall` is the declaration a capillary run takes; the operators
it builds are :class:`WallOperator` (:func:`wall_operator`, one scalar HE1m block) and
:class:`VectorWallOperator` (:func:`vector_wall_operator`, the vector families of
:class:`WallFamily`), each with its validity ratios in :class:`WallLimits`
(:func:`worst_wall_limits` over a path). Along a taper or a pressure gradient the
moving basis carries the connection of :func:`wall_connection` and
:func:`taper_connection` (blocks chosen by :func:`taper_coupling_block`), and
:func:`wall_square_root_metric` is the metric those blocks are measured in. The
modes themselves are :class:`~solprop.modes.capillary.MarcatiliMode`; the manual
pages are ``docs/guide/waveguides/coherent-wall.md`` and ``vector-wall.md``.
"""

import logging
import math
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float
from scipy.linalg import sqrtm
from scipy.special import jv

from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import GrazingIncidenceWarning, ModeError, TranslationError
from solprop.modes.capillary import (
    MarcatiliMode,
    ModeKind,
    core_susceptibility_at,
)
from solprop.modes.protocol import (
    Mode,
)
from solprop.profiles import (
    ZProfile,
    profile_identity,
)

__all__ = [
    "CoherentWall",
    "VectorWallOperator",
    "WallFamily",
    "WallImpedance",
    "WallLimits",
    "WallOperator",
    "taper_connection",
    "taper_coupling_block",
    "vector_wall_operator",
    "wall_connection",
    "wall_operator",
    "wall_square_root_metric",
    "worst_wall_limits",
]

logger = logging.getLogger(__name__)

WallImpedance = Literal["mean", "te_tm"]
"""Whether a coherent wall condenses one mean impedance or separate TE and TM ones."""


@dataclass(frozen=True, eq=False)
class CoherentWall(Displayed):
    r"""Declare the coherent wall block of a capillary mode set.

    The capillary route's diagonal operator gives every mode its own Marcatili
    attenuation and no coupling at all. The **coherent** wall instead condenses the
    outgoing Robin boundary condition onto the retained basis, which makes the wall a
    rank-one term that couples every mode to every other one with a definite phase.
    Declaring one on
    :meth:`solprop.driver.simulation.Simulation.from_capillary` replaces the diagonal
    Marcatili loss of the block's channels by that operator, exactly as
    :func:`wall_operator` builds it.

    Its physical content, in the optics convention of ``docs/dev/conventions.md``
    section 2 and with the carrier removed, is the paraxial generator

    .. math::
       G = D + H,\qquad D = -\frac{i}{2ka^2}\operatorname{diag}(u_m^2),
       \qquad H = g(a)\,hh^{T},

    with :math:`h_m = -\sqrt2\,\operatorname{sgn}J_1(u_m)\,u_m` the radial derivative of
    the orthonormal basis function at the wall, and
    :math:`g(a) = i/[2ka^2(2N - i k_{\rm eff}a)]`. The manual page is
    ``docs/guide/waveguides/coherent-wall.md``.

    **A guide that varies along z** -- a pressure gradient, a taper -- is carried by
    a frozen-residual split: the complete generator ``G_s = D(z_s) + H(z_s)`` is
    frozen at each section's midpoint inside the exponential flow, and its variation
    joins the HE1m connection

    .. math::
       C_{mn} = \frac{a'}{a}\frac{2s_ms_nu_mu_n}{u_m^2-u_n^2}\ (m\ne n),
       \qquad C_{nn} = 0

    in the **explicit residual**
    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``, which the interaction-picture
    Runge-Kutta source carries. Freezing is a rearrangement of the same equation and
    not a second-order approximation, because the residual is evaluated exactly;
    `sections` is what trades the residual's stiffness against the number of frozen
    blocks (:func:`solprop.translate.coupled.frozen_residual_flow`).

    **What it replaces, and what it therefore refuses.** The block is the run's loss
    model, so a capillary carrying one must keep the model's loss: ``loss_scale=0``,
    another scale or a measured ``alpha`` is refused by name rather than discarded. It
    is derived for the full capillary model, so ``model="reduced"`` is refused too. It
    needs the contiguous ``HE_11`` to ``HE_1N`` set, whose every channel declares the
    same radius and density profiles and whose profiles carry analytic derivatives,
    and a scalar run. It carries the default ``"input"`` noise, drawn in the channel
    basis at the entrance and propagated like any field, and refuses the
    ``"modified"`` model, whose transport through a nonunitary coupled flow is not
    measured. ``numpy_ref`` and ``native`` execute the ``linear.coupled_flow`` leaf it
    becomes.

    **The vector boundary response** is ``impedance="te_tm"``. The scalar
    block above condenses one *impedance-averaged* Robin condition, which is what a
    linearly polarised ``HE_1m`` field sees on azimuthal average; the vector block
    condenses the two conditions the wall actually imposes -- the radial (TM-like)
    field sees ``k_o/n_r^2`` and the tangential (TE-like) field sees
    ``k_o`` -- and is therefore a block **per total azimuthal order**, coupling the two
    radial families of that order rather than one. See :func:`vector_wall_operator`;
    the averaged impedance is recovered from it as a limit rather than assumed.

    Attributes
    ----------
    impedance : str
        ``"mean"`` for the impedance-averaged block, ``"te_tm"`` for the coupled
        azimuthal block, which gives the wall's TE (tangential) and TM (radial) fields
        their own impedances. It names the **wall's** two impedances and not the state's
        polarisation axis: a ``"te_tm"`` run still carries one transverse component per
        channel. On a guide that varies along z -- a taper, a pressure gradient -- the
        vector block takes the same frozen-residual route as the scalar one, with a
        connection that is each radial family's own and nothing between the families
        (:func:`wall_connection`). Which families the run carries is the **mode set's**
        business and not this field's: a set retaining one hybrid family alone is a
        legitimate vector block with no partner to couple to, and is the scalar block to
        the last bit, because that family's Bessel order is zero and its diagonal
        impedance *is* the average. Declare the partner family -- the hybrids of
        azimuthal index ``-nu`` -- to have a conversion at all. The transverse order
        ``nu = 0`` is not like that: ``TE_0m`` and ``TM_0m`` carry their own impedances
        whether or not the other is retained.
    condition_limit : float
        Largest admitted ``norm(U) norm(U^-1)`` of one step's flow action,
        dimensionless and at least one. The inverse action is what the
        interaction-picture source is carried back by, so an unbounded inverse is an
        unbounded amplification of that source; the
        backend measures the product per step and refuses above this. The default
        1e4 admits a step that attenuates the most lossy retained mode by about
        nine e-foldings.
    validity_limit : float
        Largest admitted value of the three dimensionless validity ratios of
        :class:`WallLimits`. Above it the block is **refused** by name; above a tenth
        of it the construction warns, naming the ratio. The default 0.25 keeps the
        retained transverse scale well inside the paraxial and quasi-static regimes.
    sections : int
        How many equal sections each interval of the run's own linear-operator table
        is divided into, a positive integer. The **placement** is not this number's:
        the edges are the table's breakpoints, which already land on every declared
        kink of the radius and the density, subdivided `sections` times. One, the
        default, freezes one generator per declared interval; raising it reduces the
        residual's norm and the inverse amplification, and is not permission to
        discard the residual. It has no effect on a constant guide, whose residual is
        exactly zero and which takes the single-section constant-Schur flow whatever
        this says.

    Raises
    ------
    ModeError
        If a limit or the section count is out of range.

    Examples
    --------
    >>> from solprop.modes import CoherentWall
    >>> CoherentWall().impedance, CoherentWall(impedance="te_tm").impedance
    ('mean', 'te_tm')
    >>> repr(CoherentWall())[:43]
    'CoherentWall(impedance=mean, condition_limi'
    """

    condition_limit: float = 1.0e4
    validity_limit: float = 0.25
    sections: int = 1
    impedance: WallImpedance = "mean"

    def __post_init__(self) -> None:
        """Check the wall impedance, both limits and the section count.

        Raises
        ------
        ModeError
            If the wall impedance is not one this package derives, if a limit is
            not finite or outside the range its meaning allows, or if the section
            count is not a positive integer.
        """
        if self.impedance not in ("mean", "te_tm"):
            msg = (
                "CoherentWall.impedance is 'mean' for the impedance-averaged block "
                "or 'te_tm' for the coupled azimuthal one with the wall's own TE and "
                f"TM impedances, got {self.impedance!r}"
            )
            raise ModeError(msg)
        try:
            # `operator.index` is the integer test, not `int(x) != x`: the latter
            # admits 2.0, which then translates, runs and only fails in `__repr__`,
            # where the count is formatted with "d". A numpy integer is still an
            # integer here, and a bool is excluded by name below.
            count = operator.index(self.sections)
        except TypeError:
            count = 0
        if isinstance(self.sections, bool) or count < 1:
            msg = (
                "CoherentWall.sections is how many equal parts each interval of the "
                "operator table is divided into: a positive integer, and a flow "
                f"carries at least one frozen section. Got {self.sections!r}"
            )
            raise ModeError(msg)
        if not math.isfinite(self.condition_limit) or self.condition_limit < 1.0:
            msg = (
                "CoherentWall.condition_limit is the conditioning of a flow action "
                f"and is at least one, got {self.condition_limit!r}"
            )
            raise ModeError(msg)
        if not 0.0 < self.validity_limit <= 1.0:
            msg = (
                "CoherentWall.validity_limit is a dimensionless ratio in (0, 1], got "
                f"{self.validity_limit!r}"
            )
            raise ModeError(msg)

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            For example ``CoherentWall(impedance=mean, condition_limit=10000,
            validity_limit=0.25, sections=1)``.
        """
        rows = [
            ("impedance", self.impedance),
            ("condition_limit", f"{self.condition_limit:g}"),
            ("validity_limit", f"{self.validity_limit:g}"),
            ("sections", f"{self.sections:d}"),
        ]
        return line("CoherentWall", rows)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "CoherentWall: the condensed Robin wall of a capillary block", [
            ("wall impedance", self.impedance),
            ("condition limit", f"{self.condition_limit:g}"),
            ("validity limit", f"{self.validity_limit:g}"),
            ("sections per interval", f"{self.sections:d}"),
            ("model", self._model_text()),
        ]

    def _model_text(self) -> str:
        """Return the printed one-line model the declaration builds."""
        wall = "TE/TM impedances" if self.impedance == "te_tm" else "mean impedance"
        return f"{wall}, paraxial, frozen per section"


@dataclass(frozen=True, eq=False)
class WallLimits(Displayed):
    r"""The dimensionless validity ratios of one coherent wall block.

    Each is evaluated with the **highest retained** mode over the band the run scores
    (``band`` below), which is where every approximation in it is worst there.
    None of them is a proof of Maxwell accuracy: they are the stated conditions under
    which the condensed paraxial model is the model it claims to be. In particular a
    converged paraxial Robin equation is not by itself Maxwell accuracy, and many
    retained basis components can exceed the in-core wavenumber without being
    propagating channels.

    Attributes
    ----------
    transverse_ratio : float
        ``u_N / (|k| a)`` with ``k`` the **in-core** wavenumber: how far the highest
        retained transverse scale is from paraxial. The paraxial generator's own
        error is a quarter of its fourth power relative to the carrier.
    vacuum_ratio : float
        ``u_N / (k_0 a)`` with ``k_0`` the vacuum wavenumber: the grazing-incidence
        ratio the Marcatili expansion is asymptotic in, which the cladding term is
        referred to.
    quasi_static : float
        ``2N / |a k_eff|``: the fraction of the eliminated boundary coordinate the
        condensation discards in the quasi-static wall limit.
        Raising ``N`` raises this, so it cannot by itself validate the condensation.
        A **vector** block takes the same ratio family by family and keeps the worst:
        ``(2N_f + p_f) sum_g |Z_fg| / a``, with ``p_f`` the family's Bessel order and
        ``Z`` its impedance matrix in m, which is the row sum the condensed
        ``S = (C - i a Z^-1)^-1`` departs from ``i Z/a`` by
        (:func:`vector_wall_operator`).
    band : tuple of float
        Lowest and highest angular frequency in rad/s the ratios were **taken** over.
        On a run this is the error norm's **scored band**
        (:attr:`solprop.grid.TimeGrid.inband_mask`, the requested wavelength limits),
        not the active band the operator is built on: a field grid's active band
        reaches about twice the longest requested wavelength, where every ratio (each
        proportional to the wavelength) is about twice its in-band value, and those
        collar bins are absorbed rather than scored.
        Without a scored mask, or with one that holds no evaluated bin, it is the
        whole evaluated band.
    n_modes : int
        How many channels the block carries, over every family in it.
    radius : float
        Core radius in m the ratios were taken at.

    Examples
    --------
    >>> from solprop.modes.wall import WallLimits
    >>> limits = WallLimits(0.01, 0.01, 0.02, (1e15, 3e15), 4, 125e-6)
    >>> f"{limits.worst:.3f}"
    '0.020'
    """

    transverse_ratio: float
    vacuum_ratio: float
    quasi_static: float
    band: tuple[float, float]
    n_modes: int
    radius: float

    @property
    def worst(self) -> float:  # numpydoc ignore=RT01
        """Largest of the three ratios.

        The one a refusal or a warning is taken on.
        """
        return max(self.transverse_ratio, self.vacuum_ratio, self.quasi_static)

    def named(self) -> tuple[str, float]:
        """Return the name and the value of the ratio that is worst.

        Returns
        -------
        tuple
            The ratio's name as it is written here, and its value.

        Examples
        --------
        >>> from solprop.modes.wall import WallLimits
        >>> WallLimits(0.3, 0.1, 0.2, (1e15, 3e15), 4, 125e-6).named()[0]
        'transverse_ratio'
        """
        rows = (
            ("transverse_ratio", self.transverse_ratio),
            ("vacuum_ratio", self.vacuum_ratio),
            ("quasi_static", self.quasi_static),
        )
        return max(rows, key=lambda row: row[1])

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            For example ``WallLimits(4 channels, worst transverse_ratio=0.0123)``.
        """
        name, value = self.named()
        return line(
            "WallLimits",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("worst", f"{name}={value:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        return "WallLimits: where the condensed paraxial wall model stops being one", [
            ("channels", self.n_modes),
            ("radius", engineering(self.radius, "m")),
            ("u_N/(k a)", f"{self.transverse_ratio:.4g}"),
            ("u_N/(k_0 a)", f"{self.vacuum_ratio:.4g}"),
            ("2N/|a k_eff|", f"{self.quasi_static:.4g}"),
            (
                "band",
                f"{engineering(self.band[0], 'rad/s')} to "
                f"{engineering(self.band[1], 'rad/s')}",
            ),
        ]


@dataclass(frozen=True, eq=False)
class WallOperator(Displayed):
    r"""The condensed Robin wall generator of a contiguous ``HE_1m`` block.

    The output of :func:`wall_operator`: the dense paraxial generator in the **optics**
    convention and in 1/m, together with the factored pieces it was built from and the
    validity ratios it was checked against. Crossing to the internal FFT convention is
    the complex conjugate of the **whole** equation, ``c' = G c`` becoming
    ``c' = conj(G) c``, which is what the translation layer stores; it is not ``-G``,
    and a passive mode propagated with the wrong one grows
    (``docs/dev/conventions.md`` section 2, ADR-017).

    Attributes
    ----------
    omega : numpy.ndarray
        The angular frequencies in rad/s the block is tabulated on, shape
        ``(n_omega,)``; the whole grid, with the inactive bins left exactly zero.
    active : numpy.ndarray
        Where the operator was evaluated, bool, shape ``(n_omega,)``.
    generator : numpy.ndarray
        ``G = D + H`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, in the
        optics convention with the carrier ``k`` removed.
    trace_vector : numpy.ndarray
        ``h_m``, dimensionless, shape ``(n_modes,)``: the radial derivative of the
        ``m``-th orthonormal basis function at the wall.
    coupling : numpy.ndarray
        ``g(a)`` in 1/m, complex, shape ``(n_omega,)``, zero off the active band.
    wavenumber : numpy.ndarray
        The in-core wavenumber ``k = n_core omega / c`` in 1/m, complex, shape
        ``(n_omega,)``, zero off the active band.
    impedance : numpy.ndarray
        ``k_eff = k_0 / v_n`` in 1/m, complex, shape ``(n_omega,)``, zero off the
        active band: the vacuum wavenumber over the **current** filled-core cladding
        term, which is what makes the isolated block reproduce this package's own
        capillary loss.
    limits : WallLimits
        The validity ratios over the active band.
    n_modes : int
        How many channels the block carries.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> block.generator.shape
    (2, 2, 1)
    >>> bool(block.generator[0, 0, 0].real < 0.0)
    True
    """

    omega: Float[np.ndarray, " n_omega_wall"]
    active: Bool[np.ndarray, " n_omega_wall_a"]
    generator: Complex[np.ndarray, "n_modes_wall_out n_modes_wall_in n_omega_wall_g"]
    trace_vector: Float[np.ndarray, " n_modes_wall_h"]
    coupling: Complex[np.ndarray, " n_omega_wall_g2"]
    wavenumber: Complex[np.ndarray, " n_omega_wall_k"]
    impedance: Complex[np.ndarray, " n_omega_wall_e"]
    limits: WallLimits

    @property
    def n_modes(self) -> int:  # numpydoc ignore=RT01
        """How many channels the block couples."""
        return len(self.trace_vector)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``WallOperator(2 channels, 1 frequency, worst 0.0123)``.
        """
        count = int(np.count_nonzero(self.active))
        return line(
            "WallOperator",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{count} frequenc{'y' if count == 1 else 'ies'}"),
                ("worst", f"{self.limits.worst:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        band = self.generator[:, :, self.active]
        loss = -np.real(np.einsum("mmw->mw", band)) if band.size else np.zeros((1, 1))
        return "WallOperator: the condensed Robin wall of an HE_1m block", [
            ("channels", self.n_modes),
            ("frequencies", f"{int(np.count_nonzero(self.active))} active"),
            ("largest field loss", f"{float(np.max(loss)):.4g} 1/m"),
            ("largest coupling", f"{float(np.max(abs(self.coupling))):.4g} 1/m"),
            ("limits", repr(self.limits)),
        ]


@dataclass(frozen=True)
class WallFamily:
    r"""One radial family of a vector coherent wall block.

    A vector block is one **total azimuthal order** :math:`\nu` of a capillary, and
    that order has at most two radial families, because the transverse field of an
    order-:math:`\nu` mode is a combination of the two circular components
    :math:`\hat e_\pm`, whose Cartesian angular harmonics are :math:`\nu\mp1` and
    whose radial functions are therefore :math:`J_{\nu-1}` and :math:`J_{\nu+1}`.
    Each family is a contiguous ``m = 1 .. N_f`` set of
    this package's own Marcatili channels.

    Attributes
    ----------
    label : str
        How the family prints: ``"HE_1m"``, ``"EH_1m"``, ``"TE_0m"``, ``"TM_0m"`` and
        the higher orders in the same spelling.
    azimuthal_order : int
        The channels' own :attr:`~solprop.modes.capillary.MarcatiliMode.n`. Within one
        sector the two families carry ``+nu`` and ``-nu``; ``nu = 0`` carries the two
        transverse kinds.
    bessel_order : int
        ``p``, the order of the Bessel function the family's radial basis is built
        from, ``>= 0``: ``|n - 1|`` for a hybrid family and ``1`` for a transverse
        one.
    n_channels : int
        How many radial orders of this family the block retains, ``>= 1``.
    angular : tuple of float
        The family's **angular content at the wall**, ``(radial, tangential)``, a unit
        two-vector: the fraction of its boundary field that is normal to the wall and
        the fraction tangential to it. ``(1, 0)`` for ``TM_0m`` and ``(0, 1)`` for
        ``TE_0m``, which is why those two see the wall's own two impedances; and
        ``(1, -1)/sqrt 2`` and ``(1, 1)/sqrt 2`` for the two hybrid families of an
        order ``nu >= 1``, which is why *those* see the impedance **average** on the
        diagonal and the impedance **difference** between them.
    radial_sign : float
        The sign of the family's **implemented** radial profile against ``J_p``,
        ``+1`` or ``-1``. :meth:`~solprop.modes.capillary.MarcatiliMode.field` builds a
        hybrid from
        ``jv(n - 1, .)``, and ``J_{-p} = (-1)^p J_p``, so the ``n = -nu`` family of an
        **even** order is implemented with the opposite sign to ``J_{nu+1}``. It is a
        gauge of the mode and cancels out of every diagonal quantity, but it is the
        sign of the **coupling** between the two families, so the wall's boundary
        traces carry it.

    Examples
    --------
    >>> from solprop.modes.wall import WallFamily
    >>> family = WallFamily("TE_0m", 0, 1, 4, (0.0, 1.0), 1.0)
    >>> family.truncation
    9
    """

    label: str
    azimuthal_order: int
    bessel_order: int
    n_channels: int
    angular: tuple[float, float]
    radial_sign: float

    @property
    def truncation(self) -> int:  # numpydoc ignore=RT01
        """The family's boundary truncation constant ``2 N_f + p_f``.

        The constant the condensation's denominator carries for this family: the
        truncated boundary self term ``sum_m h_m^2/u_m^2 = 2 N_f`` of the retained
        orders plus the ``p_f`` of the exact recurrence
        ``q J_{p+1}(q) = (p - i eta) J_p(q)``. Dropping the ``p_f`` leaves a
        residue that does **not** fall with ``N``.
        """
        return 2 * self.n_channels + self.bessel_order


@dataclass(frozen=True, eq=False)
class VectorWallOperator(Displayed):
    r"""The condensed vector Robin wall generator of one azimuthal order.

    The output of :func:`vector_wall_operator`, and the vector counterpart of
    :class:`WallOperator`: the dense paraxial generator in the **optics** convention
    and in 1/m, the pieces it was factored from and the validity ratios it was checked
    against. Crossing to the internal FFT convention is the complex conjugate of the
    **whole** equation, exactly as for the scalar block.

    The wall term is **rank two, not rank one**: one boundary trace vector per family,
    weighted by the condensed ``2 x 2`` impedance of the order.

    Attributes
    ----------
    omega : numpy.ndarray
        The angular frequencies in rad/s the block is tabulated on, shape
        ``(n_omega,)``; the whole grid, with the inactive bins left exactly zero.
    active : numpy.ndarray
        Where the operator was evaluated, bool, shape ``(n_omega,)``.
    generator : numpy.ndarray
        ``G = D + H`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, in the
        optics convention with the carrier ``k`` removed.
    families : tuple of WallFamily
        The one or two radial families of this order, in the state's channel order.
    sector : int
        The total azimuthal order ``nu >= 0`` every channel of the block belongs to.
    family_traces : numpy.ndarray
        The boundary traces ``h_m`` arranged one column per family, dimensionless,
        shape ``(n_modes, n_families)``, exactly zero outside each family's own
        channels. The wall term is ``V W V^T`` with ``V`` this matrix.
    weight : numpy.ndarray
        ``W = i S/(2 k a^2)`` in 1/m, complex, shape
        ``(n_families, n_families, n_omega)``: the condensed impedance of the order,
        scaled by the paraxial reduction. Zero off the active band.
    wavenumber : numpy.ndarray
        The in-core wavenumber ``k = n_core omega / c`` in 1/m, complex, shape
        ``(n_omega,)``, zero off the active band.
    impedance : numpy.ndarray
        The wall's own two impedances in m, complex, shape ``(2, n_omega)``: the
        radial (TM-like) ``v_TM/k_0`` and the tangential (TE-like) ``v_TE/k_0``, built
        from this package's own Marcatili cladding terms. Their mean is the scalar
        block's ``1/k_eff`` and their half-difference is the anisotropy.
    limits : WallLimits
        The validity ratios over the active band.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, vector_wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, kind=k, n=0) for k in ("TM", "TE")]
    >>> block = vector_wall_operator(modes, np.array([2.356e15]))
    >>> block.sector, block.n_families, block.generator.shape
    (0, 2, (2, 2, 1))
    >>> bool(abs(block.generator[0, 1, 0]) == 0.0)
    True
    """

    omega: Float[np.ndarray, " n_omega_vec"]
    active: Bool[np.ndarray, " n_omega_vec_a"]
    generator: Complex[np.ndarray, "n_modes_vec_out n_modes_vec_in n_omega_vec_g"]
    families: tuple[WallFamily, ...]
    sector: int
    family_traces: Float[np.ndarray, "n_modes_vec_h n_families_vec_h"]
    weight: Complex[np.ndarray, "n_families_w_out n_families_w_in n_omega_vec_w"]
    wavenumber: Complex[np.ndarray, " n_omega_vec_k"]
    impedance: Complex[np.ndarray, "n_wall_components n_omega_vec_z"]
    limits: WallLimits

    @property
    def n_modes(self) -> int:  # numpydoc ignore=RT01
        """How many channels the block couples.

        The total over its families.
        """
        return sum(family.n_channels for family in self.families)

    @property
    def n_families(self) -> int:  # numpydoc ignore=RT01
        """How many radial families the block carries.

        One or two.
        """
        return len(self.families)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``VectorWallOperator(nu=1, HE_1m+EH_1m, 4 channels,
            1 frequency, worst 0.0123)``.
        """
        count = int(np.count_nonzero(self.active))
        return line(
            "VectorWallOperator",
            [
                ("nu", f"{self.sector}"),
                ("", "+".join(family.label for family in self.families)),
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{count} frequenc{'y' if count == 1 else 'ies'}"),
                ("worst", f"{self.limits.worst:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        band = self.generator[:, :, self.active]
        loss = -np.real(np.einsum("mmw->mw", band)) if band.size else np.zeros((1, 1))
        anisotropy = self.impedance[0] - self.impedance[1]
        mean = self.impedance[0] + self.impedance[1]
        ratio = abs(anisotropy[self.active]) / abs(mean[self.active])
        return "VectorWallOperator: the condensed vector wall of one azimuthal order", [
            ("azimuthal order", self.sector),
            (
                "families",
                ", ".join(
                    f"{family.label} (J_{family.bessel_order}, {family.n_channels})"
                    for family in self.families
                ),
            ),
            ("channels", self.n_modes),
            ("frequencies", f"{int(np.count_nonzero(self.active))} active"),
            ("largest field loss", f"{float(np.max(loss)):.4g} 1/m"),
            ("wall anisotropy", f"{float(np.max(ratio)):.4g} of the mean impedance"),
            ("limits", repr(self.limits)),
        ]


def _wall_block_modes(
    modes: Sequence[Mode], *, wall: bool = True
) -> tuple[MarcatiliMode, ...]:
    """Check a mode set is one contiguous scalar ``HE_11`` to ``HE_1N`` block.

    The scalar regulariser condenses the boundary coordinate
    onto the *complete* set of ``J_0`` radial orders, so a sparse or mixed set is not a
    smaller block but a different operator: the ``2N`` self term counts the retained
    orders, and a missing order leaves its share of the boundary trace nowhere to go.
    The set is therefore rejected by name rather than promoted, because promoting it
    would silently change the channels the caller asked to propagate and save.

    Two of the mode's own settings are refused here rather than ignored, because the
    block does ignore them. **The wall is the loss model**: the condensed Robin
    boundary condition *is* where every channel's attenuation comes from, the
    assembly subtracts the whole ``HE_11`` wavenumber and re-adds the block's own
    diagonal, and :func:`wall_operator` builds its impedance from the cladding term
    whatever :attr:`MarcatiliMode.loss_scale` and :attr:`MarcatiliMode.alpha` say --
    so ``loss_scale=0``, another scale or a measured ``alpha`` beside a wall would
    vanish without a word.
    And the condensation inverts the **full** model's transverse eigenvalue, so
    ``model="reduced"``, whose leading-order expansion of ``n_eff`` is a different
    carrier, would be overridden by the block it sits beside.

    - ``modes``: typed as the protocol rather than as :class:`MarcatiliMode` so that a
      set carrying another kind of mode reaches the refusal here and is named, instead
      of failing the runtime type check the test session installs.
    - ``wall``: whether the block is a coherent wall's. The taper coupling alone
      (:func:`taper_connection`) reads the radial orders and the radius, never the loss
      or the capillary model, so ``False`` admits any loss and either model and keeps
      every other condition.
    """
    block = tuple(modes)
    if not block:
        raise ModeError("a coherent wall block needs at least one HE_1m mode")
    for index, mode in enumerate(block):
        if isinstance(mode, MarcatiliMode) and type(mode) is not MarcatiliMode:
            msg = (
                "a coherent wall block assumes one uniform dielectric wall at a "
                f"single radius, and channel {index} is a {type(mode).__name__}: a "
                "structured cladding's loss is its own resonances, not a Robin "
                "condition on a bare capillary, and condensing this basis would give "
                "the structure the bare capillary's wall. Leave wall=None and keep "
                "the guide's own loss (a structured wall is not implemented)"
            )
            raise ModeError(msg)
    labels = []
    for mode in block:
        if not isinstance(mode, MarcatiliMode):
            labels.append(type(mode).__name__)
        else:
            labels.append(f"{mode.kind}_{mode.n}{mode.m}")
    wanted = [f"HE_1{index + 1}" for index in range(len(block))]
    if labels != wanted:
        msg = (
            "a coherent wall block needs the contiguous Marcatili set HE_11 to "
            f"HE_1{len(block)} in that order, and this run carries "
            f"{', '.join(labels)}. Propagate the complete block, or leave wall=None "
            "and keep the uncoupled Marcatili loss"
        )
        raise ModeError(msg)
    _check_wall_channels(cast("tuple[MarcatiliMode, ...]", block), wall=wall)
    return cast("tuple[MarcatiliMode, ...]", block)


def _wall_channel_settings(mode: MarcatiliMode) -> tuple[tuple[str, object], ...]:
    """Return everything the wall block reads from one channel but its radial order.

    The operator is built from ``block[0]`` alone -- its radius, its density, its
    cladding term and its capillary model -- and the retained orders enter only
    through ``u_m``. Every other channel must therefore declare the *same* guide, or
    the block silently propagates the first channel's one. These are the settings
    compared: the two **profiles** by their identity rather than by a value at one
    position, because a block frozen per section reads them at every section midpoint
    and two tapers that agree at the entrance are two different guides everywhere
    else.

    - Returns: ``(name, value)`` pairs, comparable and printable.
    """
    return (
        ("phi", mode.phi),
        (
            "temperature",
            _profile_key(mode.temperature)
            if isinstance(mode.temperature, ZProfile)
            else mode.temperature,
        ),
        ("core radius", _profile_key(mode.core_radius)),
        ("density", _profile_key(mode.density)),
    )


def _profile_key(profile: ZProfile) -> str:
    """Return a comparable, printable identity of one declared z profile.

    - Returns: the sorted identity mapping of :func:`solprop.profiles.profile_identity`,
      rendered so that two paths compare equal exactly when the identity does. A path
      that has no structural identity -- an arbitrary callable -- is keyed on the object
      itself, so two channels sharing one declared path agree and two that were handed
      separate callables are refused, which is the conservative answer for something
      this package cannot inspect.
    """
    try:
        identity = profile_identity(profile)
    except TranslationError:
        return f"a declared callable at 0x{id(profile):x}"
    return repr(sorted(identity.items()))


def _check_wall_channels(
    block: tuple[MarcatiliMode, ...], *, wall: bool = True
) -> None:
    """Refuse a block whose channels do not declare one guide, by name.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set, in channel order.
    - ``wall``: ``False`` (the taper coupling alone) skips the loss and capillary-model
      conditions, which only the wall needs, and names the taper coupling in the rest.
    """
    owner = "a coherent wall block" if wall else "the taper coupling"
    for index, mode in enumerate(block):
        if wall and mode.model != "full":
            msg = (
                "a coherent wall block needs one capillary model, the full one, and "
                f"channel {index} declares model={mode.model!r}: the condensation "
                "inverts the full model's transverse eigenvalue, and the reduced "
                "model's leading-order carrier beside it would be overridden rather "
                "than propagated"
            )
            raise ModeError(msg)
        if wall and (mode.alpha is not None or mode.loss_scale != 1.0):
            msg = (
                "a coherent wall block is the loss model, and channel "
                f"{index} declares loss={mode._loss_text()}: the condensed Robin "
                "boundary condition replaces the Marcatili attenuation of every "
                "channel, so a loss setting beside it would be discarded in silence. "
                "Leave alpha and loss_scale at their defaults with a wall, or leave "
                "wall=None and keep the uncoupled loss you asked for"
            )
            raise ModeError(msg)
        for name, path in (
            ("core radius", mode.core_radius),
            ("density", mode.density),
        ):
            if not path.derivative_is_analytic:
                msg = (
                    f"{owner} frozen per section differentiates the "
                    f"declared profiles, and channel {index}'s {name} reports no "
                    "analytic derivative: the taper connection is a'/a and a "
                    "difference of a sampled profile is noise, not a slope. Declare "
                    "a profile with an analytic derivative, or supply the slope"
                    + (
                        ""
                        if wall
                        else "; or pass taper_coupling=False for the adiabatic route"
                    )
                )
                raise ModeError(msg)
    reference = _wall_channel_settings(block[0])
    for index, mode in enumerate(block[1:], start=1):
        for (name, mine), (_, theirs) in zip(
            _wall_channel_settings(mode), reference, strict=True
        ):
            if mine != theirs:
                msg = (
                    f"{owner} is built from one guide and reads it from "
                    f"the lowest channel alone; channel {index} declares "
                    f"{name}={mine!r} against {theirs!r}. Every channel shares one "
                    "capillary, one fill and one polarisation angle, the "
                    "mean-impedance block and the TE/TM one (impedance='te_tm') "
                    "alike"
                )
                raise ModeError(msg)


def _check_wall_guide(
    block: tuple[MarcatiliMode, ...],
    band: Float[np.ndarray, " n_active"],
    *,
    probe: ModeKind | None = None,
) -> None:
    """Refuse a block whose channels sample different cladding or fill, by name.

    The declared settings are compared in :func:`_check_wall_channels`; this compares
    what the operator actually reads, which is the only decisive test, because two
    channels can name their cladding differently and mean the same glass. The
    cladding term and the per-particle susceptibility come out of the same code on
    the same frequencies, so equal guides give **bitwise** equal arrays and the
    comparison is exact.

    It is a probe of at most three frequencies rather than the whole band: sampling
    the gas and the cladding models is the expensive part of building a mode's terms
    (:class:`MarcatiliTerms`), and a cladding that agrees at the two ends and the
    middle of the band and differs between them is not a thing this package can
    build.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set, in channel order.
    - ``band``: the active frequencies in rad/s, ascending, non-empty.
    - ``probe``: which family's cladding term to compare the channels through. ``None``
      compares each channel's own, which is right when they share one family. A
      **vector** block does not: its families are the point of it, and their cladding
      terms differ by construction, so it compares every channel through one named kind,
      which is the same permittivity read the same way.
    """
    if len(block) == 1:
        return
    frequencies = band[np.unique([0, band.size // 2, band.size - 1])]
    sampled = [
        (mode if probe is None else replace(mode, kind=probe, n=0, m=1)).sampled_terms(
            frequencies
        )
        for mode in block
    ]
    reference = sampled[0]
    for index, terms in enumerate(sampled[1:], start=1):
        for name, mine, theirs in (
            ("cladding", terms.cladding_term, reference.cladding_term),
            (
                "fill",
                terms.core_susceptibility,
                reference.core_susceptibility,
            ),
        ):
            if not np.array_equal(mine, theirs):
                msg = (
                    "a coherent wall block is built from one guide and reads it from "
                    f"the lowest channel alone; channel {index} samples a different "
                    f"{name} over the active band. Every channel of the block shares "
                    "one capillary and one fill"
                )
                raise ModeError(msg)


def wall_operator(
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float = 0.0,
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
) -> WallOperator:
    r"""Build the condensed Robin wall generator of a contiguous ``HE_1m`` block.

    In outline (the manual page is ``docs/guide/waveguides/coherent-wall.md``), the
    orthonormal radial basis is :math:`\phi_m(x) = \sqrt2 J_0(u_m x)/|J_1(u_m)|` on
    :math:`x = r/a`, whose derivative at the wall is
    :math:`h_m = -\sqrt2\,\operatorname{sgn}J_1(u_m)\,u_m`. Truncating the
    Mittag-Leffler expansion of the exact outgoing root equation
    :math:`qJ_1(q) + i\eta J_0(q) = 0` at ``N`` orders and eliminating the boundary
    coordinate gives the condensed transverse operator
    :math:`L_N = -\operatorname{diag}(u_m^2) + hh^{T}/(2N - i\eta)`, and the paraxial
    reduction of :math:`M = k^2I + L_N/a^2` about the in-core carrier is
    :math:`G = i(M - k^2)/(2k) = D + H` in 1/m.

    Two normalisations are re-derived here rather than assumed. The **amplitude map**
    is the identity up to one mode-independent positive scale: this package's internal
    amplitude is ``a_m = E_m sqrt(N_m)`` with
    ``N_m = (pi/2) a^2 J_1(u_m)^2 eps_0 c`` (``docs/dev/conventions.md`` section 4),
    and both that normalisation and the orthonormal basis above carry the same
    ``|J_1(u_m)|``, so the two sets of coordinates differ by ``a sqrt(pi eps_0 c)``
    for every channel alike and the generator is unchanged by the map. The
    **impedance** is ``k_eff = k_0 / v_n``, the *vacuum* wavenumber over the cladding
    term this package's own capillary index already uses, because that is the
    combination whose ``N = 1`` limit is this package's Marcatili transverse
    eigenvalue; the carrier ``k`` of the paraxial reduction is the **in-core**
    wavenumber. Referring the impedance to the core instead is identical for a vacuum
    core and differs at the order of half the gas susceptibility for a filled one.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous block ``HE_11`` to ``HE_1N``, in the state's channel order,
        sharing one polarisation angle, one capillary and one constant section.
        Anything else is refused by name.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    z : float, optional
        Position along the capillary in m the block is frozen at.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``. ``None`` takes every
        strictly positive frequency. An empty out-of-band bin is never evaluated: the
        wall formula divides by ``omega`` and its validity ratios are meaningless
        there, so the generator is left exactly zero.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the validity ratios are **taken** over, shape
        ``(n_omega,)``, intersected with the evaluated bins; a run passes the error
        norm's scored band (:attr:`solprop.grid.TimeGrid.inband_mask`), so the absorbed
        collar a field grid adds below the requested band, at up to twice the longest
        requested wavelength, neither refuses nor warns. The generator is still built on
        every evaluated bin. ``None``, or a mask that holds no evaluated bin, takes the
        ratios over every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio; above it the block is refused, above a tenth
        of it it warns. See :class:`CoherentWall`.

    Returns
    -------
    WallOperator
        The dense optics-convention generator, its factored pieces and its limits.

    Raises
    ------
    ModeError
        If the set is not one contiguous scalar block, if a channel declares a loss
        setting or the reduced model, if the active band is empty or holds a
        non-positive frequency, if the block is not passive on the branch the
        cladding term lands on, or if a validity ratio exceeds `validity_limit`.

    Warns
    -----
    GrazingIncidenceWarning
        When a validity ratio exceeds a tenth of `validity_limit`, naming it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2, 3)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> block.n_modes, f"{block.limits.worst:.4f}"
    (3, '0.0090')
    """
    block = _wall_block_modes(modes)
    values = np.asarray(omega, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ModeError("a coherent wall block needs a non-empty frequency vector")
    mask = (
        values > 0.0
        if active is None
        else np.asarray(active, dtype=bool) & (values > 0.0)
    )
    if not np.any(mask):
        msg = (
            "a coherent wall block has no active bin to evaluate: the mask leaves no "
            "strictly positive frequency, and a hollow capillary has no mode at zero"
        )
        raise ModeError(msg)
    count = len(block)
    first = block[0]
    band = values[mask]
    judged = _judged_bins(mask, scored)
    _check_wall_guide(block, band)
    terms = first.sampled_terms(band)
    radius, density = first._scalars(z)
    chi = core_susceptibility_at(terms, density)
    vacuum = band / C
    wavenumber = np.sqrt(1.0 + chi) * vacuum
    impedance = vacuum / terms.cladding_term
    eta = radius * impedance
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    trace = -np.sqrt(2.0) * np.sign(jv(1, zeros)) * zeros
    # g(a) and D share the prefactor 1/(2 k a^2) of the paraxial reduction; the
    # condensed rank-one term carries the eliminated boundary denominator.
    prefactor = 1.0 / (2.0 * wavenumber * radius**2)
    coupling = 1j * prefactor / (2.0 * count - 1j * eta)
    diagonal = -1j * prefactor[None, :] * (zeros**2)[:, None]
    generator = coupling[None, None, :] * np.outer(trace, trace)[:, :, None]
    generator[np.arange(count), np.arange(count)] += diagonal
    limits = WallLimits(
        transverse_ratio=float(np.max(zeros[-1] / (abs(wavenumber[judged]) * radius))),
        vacuum_ratio=float(np.max(zeros[-1] / (vacuum[judged] * radius))),
        quasi_static=float(np.max(2.0 * count / abs(eta[judged]))),
        band=(float(band[judged][0]), float(band[judged][-1])),
        n_modes=count,
        radius=radius,
    )
    _check_wall_passivity(generator, wavenumber, radius, count)
    _check_wall_limits(limits, validity_limit)
    return WallOperator(
        omega=values,
        active=mask,
        generator=_scattered(generator, mask, (count, count)),
        trace_vector=trace,
        coupling=_scattered(coupling, mask, ()),
        wavenumber=_scattered(wavenumber, mask, ()),
        impedance=_scattered(impedance, mask, ()),
        limits=limits,
    )


def _vector_wall_families(
    block: tuple[MarcatiliMode, ...],
) -> tuple[int, tuple[WallFamily, ...]]:
    """Return the azimuthal order of a vector wall block and its radial families.

    The set must be **one** total azimuthal order, because the wall is rotationally
    symmetric and therefore exactly diagonal in that order, and each of its families
    must be the contiguous ``m = 1 .. N_f`` set the condensation truncates.
    Everything else is refused by name.

    - Returns: the order ``nu`` and its families, in channel order.
    """
    runs: list[list[MarcatiliMode]] = []
    for mode in block:
        if runs and (runs[-1][0].kind, runs[-1][0].n) == (mode.kind, mode.n):
            runs[-1].append(mode)
        else:
            runs.append([mode])
    labels = [f"{run[0].kind}_{run[0].n}m x {len(run)}" for run in runs]
    for run in runs:
        if [mode.m for mode in run] != list(range(1, len(run) + 1)):
            msg = (
                "every family of a vector coherent wall block is the contiguous "
                f"radial set m = 1 to N of its own azimuthal order, and this run "
                f"carries {', '.join(f'{mode.kind}_{mode.n}{mode.m}' for mode in run)}"
                ": the boundary self term counts the retained orders, so a missing "
                "one leaves its share of the trace nowhere to go"
            )
            raise ModeError(msg)
    if len(runs) > 2:
        msg = (
            "a vector coherent wall block is one total azimuthal order, which has at "
            f"most two radial families, and this run carries {len(runs)}: "
            f"{', '.join(labels)}. The wall is rotationally symmetric, so orders do "
            "not couple and each is a block of its own"
        )
        raise ModeError(msg)
    kinds = {run[0].kind for run in runs}
    if kinds <= {"TE", "TM"}:
        sector = 0
        families = tuple(
            WallFamily(
                label=f"{run[0].kind}_0m",
                azimuthal_order=0,
                bessel_order=1,
                n_channels=len(run),
                angular=(1.0, 0.0) if run[0].kind == "TM" else (0.0, 1.0),
                # Both transverse families are built from +J_1 (`MarcatiliMode.field`).
                radial_sign=1.0,
            )
            for run in runs
        )
        return sector, families
    if kinds != {"HE"}:
        msg = (
            "a vector coherent wall block is one total azimuthal order, and this run "
            f"mixes the transverse order nu = 0 with a hybrid one: {', '.join(labels)}"
            ". TE_0m and TM_0m are the nu = 0 block; the hybrids of order nu are its "
            "own block"
        )
        raise ModeError(msg)
    sector = abs(runs[0][0].n)
    if len(runs) == 2 and runs[1][0].n != -runs[0][0].n:
        msg = (
            f"the two hybrid families of the azimuthal order nu = {sector} are "
            f"n = {sector} (radial J_{abs(sector - 1)}) and n = {-sector} (radial "
            f"J_{sector + 1}), and this run carries {', '.join(labels)}: the wall "
            "couples those two and no others, so a block is one of them or both"
        )
        raise ModeError(msg)
    scale = 1.0 / math.sqrt(2.0)
    families = tuple(
        WallFamily(
            label=("HE" if run[0].n > 0 else "EH") + f"_{sector}m",
            azimuthal_order=run[0].n,
            bessel_order=abs(run[0].n - 1),
            n_channels=len(run),
            angular=(scale, -scale if run[0].n > 0 else scale),
            radial_sign=-1.0 if (run[0].n - 1) < 0 and (run[0].n - 1) % 2 else 1.0,
        )
        for run in runs
    )
    return sector, families


def _vector_wall_block(
    modes: Sequence[Mode],
) -> tuple[int, tuple[WallFamily, ...], tuple[MarcatiliMode, ...]]:
    """Check a mode set is one azimuthal order of a bare capillary, and split it.

    - Returns: the azimuthal order, its families and the same modes as a tuple.
    """
    block = tuple(modes)
    if not block:
        msg = "a vector coherent wall block needs at least one capillary mode"
        raise ModeError(msg)
    for index, mode in enumerate(block):
        if type(mode) is not MarcatiliMode:
            msg = (
                "a vector coherent wall block assumes one uniform dielectric wall at "
                f"a single radius, and channel {index} is a {type(mode).__name__}: a "
                "structured cladding's loss is its own resonances, not a Robin "
                "condition on a bare capillary. Leave wall=None and keep the guide's "
                "own loss"
            )
            raise ModeError(msg)
    checked = cast("tuple[MarcatiliMode, ...]", block)
    sector, families = _vector_wall_families(checked)
    _check_wall_channels(checked)
    return sector, families, checked


def _vector_impedances(
    mode: MarcatiliMode, band: Float[np.ndarray, " n_active"]
) -> Complex[np.ndarray, "n_wall_components n_active"]:
    """Return the wall's radial and tangential impedances in m.

    They are this package's **own** Marcatili cladding terms of the two transverse
    families over the vacuum wavenumber: ``v_TM/k_0`` for a field normal to the wall
    and ``v_TE/k_0`` for one tangential to it. Sampling them through a replacement
    mode rather than through
    :func:`~solprop.modes.capillary.cladding_term_and_derivative` directly is
    deliberate: a constant, tabulated or mixed cladding then reaches the wall model by
    exactly the path any TE or TM channel of this guide would take.

    - ``band``: the active frequencies in rad/s, ascending, strictly positive.
    - Returns: the radial and tangential impedances in m, complex, shape ``(2,
      n_active)``.
    """
    vacuum = band / C
    return np.stack(
        [
            np.asarray(
                replace(mode, kind=kind, n=0, m=1).sampled_terms(band).cladding_term
            )
            / vacuum
            for kind in ("TM", "TE")
        ]
    )


def vector_wall_operator(
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float = 0.0,
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
) -> VectorWallOperator:
    r"""Build the condensed **vector** Robin wall generator of one azimuthal order.

    In outline (the manual page is ``docs/guide/waveguides/vector-wall.md``), three
    steps.

    **The boundary response is a matrix, not a scalar.** At grazing incidence the wall
    imposes :math:`\partial_rE = ik_oE` on the field *tangential* to it and
    :math:`\partial_rE = i(k_o/n_r^2)E` on the field *normal* to it. In the
    near-Dirichlet regime the condensation is linear in the **impedance**, so what
    enters is :math:`Z = \operatorname{diag}(1/k_{\rm TM}, 1/k_{\rm TE})` in the
    :math:`(\hat r,\hat\theta)` components. Rotating it into the radial families of a
    total azimuthal order :math:`\nu` -- the angular quadrature
    :math:`Z_{fg} = \int_0^{2\pi}[Z_{rr}P_fP_g + Z_{\theta\theta}Q_fQ_g]\,d\theta`
    over the unit-normalised angular patterns of :attr:`WallFamily.angular` -- gives,
    for :math:`\nu \ge 1`,

    .. math::
       Z = \bar z\,I + \delta\,\sigma_x,\qquad
       \bar z = \tfrac12(1/k_{\rm TM} + 1/k_{\rm TE}),\qquad
       \delta = \tfrac12(1/k_{\rm TM} - 1/k_{\rm TE}),

    and, for :math:`\nu = 0`, the same matrix diagonalised: TM sees :math:`\bar z +
    \delta` and TE sees :math:`\bar z - \delta`, with no coupling at all. The mean
    :math:`\bar z` is exactly the scalar block's :math:`1/k_{\rm eff}`, because
    ``v_HE`` is the mean of ``v_TE`` and ``v_TM``; the **anisotropy** :math:`\delta` is
    what the scalar average discards, and it is ``(n_r^2 - 1)/(n_r^2 + 1)`` of the
    mean -- 35 % for silica, not a small correction.

    **The condensation is rank two.** Truncating the Mittag-Leffler expansion of the
    exact recurrence :math:`qJ_{p+1}(q) = (p - i\eta)J_p(q)` family by family gives
    :math:`T_p(\lambda) \simeq 2N_f + p_f + h_f^{T}(\lambda - D_f)^{-1}h_f`, and the
    exact two-family secular equation
    :math:`\det[\operatorname{diag}(T_{p_A}, T_{p_B}) - iaZ^{-1}] = 0` is **exactly**
    the secular equation of

    .. math::
       L_N = -\operatorname{diag}(u_m^2) + VSV^{T},\qquad
       S = (C - i\,aZ^{-1})^{-1},\qquad C = \operatorname{diag}(2N_f + p_f),

    with ``V`` one boundary trace column per family. The truncation constant carries
    the family's Bessel order ``p_f``: without it the condensed eigenvalues stop
    converging to the exact roots. The paraxial reduction of
    :math:`M = k^2I + L_N/a^2` is :math:`G = i(M - k^2)/(2k)` in 1/m, as for the
    scalar block.

    **What is new beyond two copies of the scalar block.** The scalar block is one
    family with one averaged impedance, and two copies of it for two polarisations are
    two copies of the same average. This block is per azimuthal order: it gives
    ``TE_0m`` and ``TM_0m`` their **own** impedances, which differ by ``n_r^2``, and it
    **couples** the ``J_{nu-1}`` and ``J_{nu+1}`` families of every order ``nu >= 1``
    through :math:`\delta`, which converts power between them. Each family in
    isolation still reduces to this package's own Marcatili loss for its own ``v_n``.

    Parameters
    ----------
    modes : sequence of Mode
        One total azimuthal order of one capillary, in the state's channel order:
        either ``TM_0m`` and ``TE_0m`` sets, or the hybrid ``n = +nu`` set followed by
        the ``n = -nu`` one, each contiguous from ``m = 1``. Either family may be
        retained alone, in which case the block carries no conversion and the report's
        single-family loss limits are what it is. Every channel declares the same
        guide, fill and polarisation angle; anything else is refused by name.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    z : float, optional
        Position along the capillary in m the block is frozen at.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``. ``None`` takes every
        strictly positive frequency; an out-of-band bin is never evaluated.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the validity ratios are taken over, as for
        :func:`wall_operator`; ``None`` takes every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio; above it the block is refused, above a tenth
        of it it warns. See :class:`CoherentWall`.

    Returns
    -------
    VectorWallOperator
        The dense optics-convention generator, its factored pieces and its limits.

    Raises
    ------
    ModeError
        If the set is not one azimuthal order of a bare capillary, if a channel
        declares a loss setting or the reduced model, if the active band is empty, if
        the block is not passive on the branch the cladding terms land on, or if a
        validity ratio exceeds `validity_limit`.

    Warns
    -----
    GrazingIncidenceWarning
        When a validity ratio exceeds a tenth of `validity_limit`, naming it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, vector_wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, n=n, m=1) for n in (1, -1)]
    >>> block = vector_wall_operator(modes, np.array([2.356e15]))
    >>> block.sector, [family.label for family in block.families]
    (1, ['HE_1m', 'EH_1m'])
    >>> bool(abs(block.generator[0, 1, 0]) > 0.0)
    True
    """
    sector, families, block = _vector_wall_block(modes)
    values = np.asarray(omega, dtype=float)
    if values.ndim != 1 or not values.size:
        msg = "a vector coherent wall block needs a non-empty frequency vector"
        raise ModeError(msg)
    mask = (
        values > 0.0
        if active is None
        else np.asarray(active, dtype=bool) & (values > 0.0)
    )
    if not np.any(mask):
        msg = (
            "a vector coherent wall block has no active bin to evaluate: the mask "
            "leaves no strictly positive frequency, and a hollow capillary has no "
            "mode at zero"
        )
        raise ModeError(msg)
    first = block[0]
    band = values[mask]
    judged = _judged_bins(mask, scored)
    # The families' own cladding terms differ by construction, so the channels are
    # compared through one of them: what must agree is the permittivity, not v_n.
    _check_wall_guide(block, band, probe="TE")
    terms = first.sampled_terms(band)
    radius, density = first._scalars(z)
    chi = core_susceptibility_at(terms, density)
    vacuum = band / C
    wavenumber = np.sqrt(1.0 + chi) * vacuum
    impedance = _vector_impedances(first, band)
    angular = np.array([family.angular for family in families], dtype=float)
    # Z_fg(omega) = sum_i A_fi A_gi z_i(omega): the wall's two impedances rotated into
    # the families' angular content, which is the whole vector boundary response.
    gram = np.einsum("fi,iw,gi->fgw", angular, impedance, angular)
    truncation = np.diag([float(family.truncation) for family in families])
    admittance = radius * np.linalg.inv(np.moveaxis(gram, -1, 0))
    condensed = np.linalg.inv(truncation[None] - 1j * admittance)
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    orders = np.repeat(
        [family.bessel_order for family in families],
        [family.n_channels for family in families],
    )
    traces = np.zeros((len(block), len(families)))
    columns = np.repeat(np.arange(len(families)), [f.n_channels for f in families])
    # The derivative at the wall of each channel's **implemented** orthonormal radial
    # function, which is `jv(n - 1, .)` and so carries the family's own sign.
    signs = np.repeat(
        [family.radial_sign for family in families],
        [family.n_channels for family in families],
    )
    traces[np.arange(len(block)), columns] = (
        -np.sqrt(2.0) * signs * np.sign(jv(orders + 1, zeros)) * zeros
    )
    prefactor = 1.0 / (2.0 * wavenumber * radius**2)
    weight = 1j * prefactor[None, None, :] * np.moveaxis(condensed, 0, -1)
    generator = np.einsum("mf,fgw,ng->mnw", traces, weight, traces)
    generator[np.arange(len(block)), np.arange(len(block))] += (
        -1j * prefactor[None, :] * (zeros**2)[:, None]
    )
    rows = np.max(np.sum(abs(gram[..., judged]), axis=1), axis=-1) / radius
    # The **highest retained** order over every family, which is not the last channel:
    # a block may retain more radial orders of the J_{nu-1} family than of the
    # J_{nu+1} one, and then the largest zero is not the one the channel order ends on.
    highest = float(np.max(zeros))
    limits = WallLimits(
        transverse_ratio=float(np.max(highest / (abs(wavenumber[judged]) * radius))),
        vacuum_ratio=float(np.max(highest / (vacuum[judged] * radius))),
        quasi_static=float(
            np.max(
                [
                    family.truncation * row
                    for family, row in zip(families, rows, strict=True)
                ]
            )
        ),
        band=(float(band[judged][0]), float(band[judged][-1])),
        n_modes=len(block),
        radius=radius,
    )
    _check_wall_passivity(generator, wavenumber, radius, len(block))
    _check_wall_limits(limits, validity_limit)
    return VectorWallOperator(
        omega=values,
        active=mask,
        generator=_scattered(generator, mask, (len(block), len(block))),
        families=families,
        sector=sector,
        family_traces=traces,
        weight=_scattered(weight, mask, (len(families), len(families))),
        wavenumber=_scattered(wavenumber, mask, ()),
        impedance=_scattered(impedance, mask, (2,)),
        limits=limits,
    )


def wall_connection(
    modes: Sequence[Mode], z: float = 0.0, *, impedance: WallImpedance = "mean"
) -> Float[np.ndarray, "n_modes_link_out n_modes_link_in"]:
    r"""Return the moving-basis connection of a wall block at one position.

    The basis of the condensed wall block is attached to the core radius, so a taper
    moves it. Writing the field as :math:`E = \sum_n c_n \psi_n(r; a(z))` with
    :math:`\psi_n = \sqrt2 J_p(u_n r/a)/(a|J_{p+1}(u_n)|)` orthonormal under
    :math:`\int_0^a \cdot\, r\,dr` at every ``z``, :math:`u_n` the zeros of
    :math:`J_p`, the coefficients pick up :math:`c' = (iB - C)c` with
    :math:`C_{mn} = \langle\psi_m|\partial_z\psi_n\rangle`:

    .. math::
       C_{mn} = \frac{a'}{a}\,\frac{2s_ms_nu_mu_n}{u_m^2-u_n^2}\quad(m\ne n),
       \qquad C_{nn} = 0,

    with :math:`s_m = \operatorname{sgn}J_{p+1}(u_m)` the sign of the basis slope at
    the wall, :math:`-J_p'(u_m) = J_{p+1}(u_m)`. For the ``HE_1m`` block ``p = 0``
    and this is the scalar block's formula. It is **real** -- so the optics/internal
    conjugation of ``docs/dev/conventions.md`` section 2 leaves it alone -- and
    **antisymmetric**, because swapping the indices flips the denominator alone, so
    the connection on its own is norm preserving and moves no power out of the
    retained set. The sign is checked by differentiating the implemented profile
    (``tests/modes/test_wall_operator.py``), not asserted.

    **The vector block** (``impedance="te_tm"``) is one
    total azimuthal order with at most two radial families, and its connection is
    **block diagonal**: each family carries the formula above with its **own** Bessel
    order ``p_f`` (:attr:`WallFamily.bessel_order`), zeros and signs, and the two
    families do not connect at all. The reason is exact rather than small: a taper
    moves the radius and nothing else, and every channel's field is its radial
    function of ``r/a`` times an angular pattern that does not depend on ``a``, so
    :math:`\langle\psi^A_m|\partial_z\psi^B_n\rangle` is a radial integral times
    the angular overlap of the two families' patterns, which is zero -- the two
    hybrid families of an order ``nu >= 1`` overlap as :math:`\cos 2\nu\theta`, and
    ``TE_0m`` and ``TM_0m`` are orthogonal at every point. A family's implemented
    sign (:attr:`WallFamily.radial_sign`) enters :math:`s_m s_n` squared and cancels.
    Both statements are measured on the implemented fields, not assumed.

    The whole matrix scales with ``a'(z)/a(z)``, and nothing in it depends on
    frequency: the basis is the same set of Bessel orders at every bin.

    Parameters
    ----------
    modes : sequence of Mode
        The block in the state's channel order: the contiguous ``HE_11`` to ``HE_1N``
        set for ``impedance="mean"``, one azimuthal order of a capillary for
        ``"te_tm"``. Refused by name on exactly the conditions :func:`wall_operator`
        and :func:`vector_wall_operator` refuse.
    z : float, optional
        Position along the capillary in m.
    impedance : str, optional
        Which block the connection is of, ``"mean"`` or ``"te_tm"``, as
        :attr:`CoherentWall.impedance`.

    Returns
    -------
    numpy.ndarray
        ``C`` in 1/m, real, shape ``(n_modes, n_modes)``, with a zero diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode
    >>> from solprop.modes.wall import wall_connection
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=index) for index in (1, 2)]
    >>> link = wall_connection(modes, 5e-3)
    >>> bool(np.all(link == -link.T)), float(link[0, 0])
    (True, 0.0)
    >>> vector = [MarcatiliMode(taper, "He", 1.0, n=n, m=1) for n in (1, -1)]
    >>> float(wall_connection(vector, 5e-3, impedance="te_tm")[0, 1])
    0.0
    """
    if impedance == "mean":
        return _scalar_connection(_wall_block_modes(modes), z)
    _, grouped, block = _vector_wall_block(modes)
    families = tuple((family.bessel_order, family.n_channels) for family in grouped)
    first = block[0]
    radius = float(first.core_radius(np.array([float(z)]))[0])
    slope = float(first.core_radius.derivative(np.array([float(z)]))[0])
    link = np.zeros((len(block), len(block)), dtype=float)
    start = 0
    for order, count in families:
        stop = start + count
        zeros = np.array([mode.u_nm for mode in block[start:stop]], dtype=float)
        link[start:stop, start:stop] = _bessel_connection(zeros, order, slope / radius)
        start = stop
    return link


def taper_coupling_block(modes: Sequence[Mode]) -> tuple[MarcatiliMode, ...] | None:
    """Return the mode set the taper coupling is carried on, or ``None``.

    The taper coupling -- the moving-basis connection ``<psi_m|d_z psi_n>`` of
    :func:`taper_connection` in a coupled flow, with each channel's own propagation
    constant on the flow's diagonal -- is carried by default on a **multimode capillary
    whose radius varies along z**, lossless or not. That is a set of two or more plain
    :class:`~solprop.modes.capillary.MarcatiliMode` channels forming the contiguous
    ``HE_11`` to ``HE_1N`` block, whose core radius is not constant.
    Every other set is ``None``: one channel needs no connection; a constant bore --
    a pressure gradient included, which changes the effective index and not the
    ``J_0(u_m r/a)`` field -- has none; and a structured cladding (a Vincetti mode), a
    tabulated mode or a set of several azimuthal orders is not the block the
    connection is derived for, and keeps the adiabatic route with its
    :class:`~solprop.exceptions.AdiabaticTaperWarning`.

    Parameters
    ----------
    modes : sequence of Mode
        The run's mode set, in the state's channel order.

    Returns
    -------
    tuple of MarcatiliMode or None
        The block, or ``None`` when the coupling does not apply.

    Examples
    --------
    >>> from solprop.modes.capillary import MarcatiliMode
    >>> from solprop.modes.wall import taper_coupling_block
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=m) for m in (1, 2)]
    >>> len(taper_coupling_block(modes))
    2
    >>> taper_coupling_block([MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)])
    >>> taper_coupling_block([MarcatiliMode(taper, "He", 1.0, m=m) for m in (1, 3)])
    """
    block = tuple(modes)
    if len(block) < 2 or any(type(mode) is not MarcatiliMode for mode in block):
        return None
    channels = cast("tuple[MarcatiliMode, ...]", block)
    labels = [(mode.kind, mode.n, mode.m) for mode in channels]
    if labels != [("HE", 1, index + 1) for index in range(len(block))]:
        return None
    if channels[0].radius_is_constant:
        return None
    return channels


def taper_connection(
    modes: Sequence[Mode], z: float = 0.0
) -> Float[np.ndarray, "n_modes_taper_out n_modes_taper_in"]:
    r"""Return the taper connection of an ``HE_1m`` block at one position.

    The connection :math:`C_{mn} = \langle\psi_m|\partial_z\psi_n\rangle` of the
    field-normalised ``J_0(u_m r/a)`` basis, exactly :func:`wall_connection`'s scalar
    block (``impedance="mean"``) -- real, antisymmetric, proportional to ``a'/a``
    -- without its two conditions that belong to the wall: it admits any loss and
    either capillary model, because the basis is the Dirichlet one whatever the
    modes' effective index says. It is what the **taper coupling** carries on a
    multimode capillary without a wall.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set in state order, every channel
        declaring the same radius and density profiles, each with an analytic
        derivative.
    z : float, optional
        Position along the capillary in m.

    Returns
    -------
    numpy.ndarray
        ``C`` in 1/m, real, shape ``(n_modes, n_modes)``, with a zero diagonal.

    Raises
    ------
    ModeError
        If the set is not such a block.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode
    >>> from solprop.modes.wall import taper_connection
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=m, loss_scale=0.0) for m in (1, 2)]
    >>> link = taper_connection(modes, 5e-3)
    >>> bool(np.all(link == -link.T)), float(link[0, 0])
    (True, 0.0)
    """
    return _scalar_connection(_wall_block_modes(modes, wall=False), z)


def _scalar_connection(
    block: tuple[MarcatiliMode, ...], z: float
) -> Float[np.ndarray, "n_modes_scalar_out n_modes_scalar_in"]:
    """Return the connection of a checked ``HE_1m`` block at one position.

    - ``z``: position in m.
    - Returns: ``C`` in 1/m.
    """
    first = block[0]
    radius = float(first.core_radius(np.array([float(z)]))[0])
    slope = float(first.core_radius.derivative(np.array([float(z)]))[0])
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    return _bessel_connection(zeros, 0, slope / radius)


def _bessel_connection(
    zeros: Float[np.ndarray, " n_family"], order: int, rate: float
) -> Float[np.ndarray, "n_family_out n_family_in"]:
    """Return the connection of one radial family of Bessel order `order`.

    - ``order``: the family's Bessel order ``p``, ``>= 0``.
    - ``rate``: ``a'/a`` in 1/m.
    - Returns: ``C`` of the family in 1/m, real, antisymmetric, with a zero diagonal.
    """
    count = zeros.size
    signs = np.sign(jv(order + 1, zeros))
    squared = zeros**2
    # The diagonal of the difference is zero, so the reciprocal is taken where the
    # denominator is not: C_nn is zero by the derivation, not by a limit.
    difference = squared[:, None] - squared[None, :]
    off = ~np.eye(count, dtype=bool)
    link = np.zeros((count, count), dtype=float)
    weight = 2.0 * np.outer(signs * zeros, signs * zeros)
    link[off] = rate * weight[off] / difference[off]
    return link


def worst_wall_limits(
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    positions: Sequence[float],
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
    impedance: WallImpedance = "mean",
) -> WallLimits:
    """Return the validity ratios at the position where they are worst.

    A guide that varies along z has a different block at every section, and the three
    ratios of :class:`WallLimits` are not monotone in the declared parameters in
    general -- the radius, the fill's index and the cladding term all enter them. The
    scan is therefore over the positions the caller actually builds the flow from, and
    the **one** call that carries the caller's `validity_limit` is the one at the worst
    of them, so a run is refused or warned about once and names the geometry that did
    it rather than the entrance's.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous block ``HE_11`` to ``HE_1N``, in the state's channel order.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    positions : sequence of float
        Positions in m to scan, non-empty.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the ratios are taken over, shape ``(n_omega,)``
        (:func:`wall_operator`); ``None`` takes every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio, applied at the worst position alone.
    impedance : str, optional
        Which block to build, ``"mean"`` (:func:`wall_operator`) or ``"te_tm"``
        (:func:`vector_wall_operator`). The scan is the same either way.

    Returns
    -------
    WallLimits
        The ratios at that position.

    Raises
    ------
    ModeError
        If `positions` is empty, or on everything the block's own constructor refuses.

    Warns
    -----
    GrazingIncidenceWarning
        Once, from the worst position, when a ratio exceeds a tenth of the limit.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode
    >>> from solprop.modes.wall import worst_wall_limits
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 100e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=index) for index in (1, 2)]
    >>> limits = worst_wall_limits(
    ...     modes, np.array([2.356e15]), positions=[0.0, 5e-3, 1e-2]
    ... )
    >>> f"{limits.radius * 1e6:.1f} um"
    '100.0 um'
    """
    scan = [float(value) for value in positions]
    if not scan:
        msg = "a coherent wall block needs at least one position to scan its limits"
        raise ModeError(msg)
    build = wall_operator if impedance == "mean" else vector_wall_operator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        ratios = [
            build(
                modes,
                omega,
                z=z,
                active=active,
                scored=scored,
                validity_limit=math.inf,
            ).limits.worst
            for z in scan
        ]
    return build(
        modes,
        omega,
        z=scan[int(np.argmax(ratios))],
        active=active,
        scored=scored,
        validity_limit=validity_limit,
    ).limits


def _judged_bins(
    mask: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega_scored"] | None,
) -> Bool[np.ndarray, " n_band"]:
    """Return which evaluated bins a wall block's validity ratios are taken over.

    - ``mask``: the evaluated bins, bool, shape ``(n_omega,)``, at least one true.
    - ``scored``: the scored bins, bool, shape ``(n_omega,)``; ``None`` takes every
      evaluated bin.
    - Returns: bool over the evaluated bins alone, at least one true. A scored mask that
      holds **no** evaluated bin -- requested limits narrower than one bin, which a
      monochromatic study grid can declare -- takes every evaluated bin, the
      conservative choice: it never admits a block the whole band would refuse.
    """
    if scored is None:
        return np.ones(int(np.count_nonzero(mask)), dtype=bool)
    chosen = np.asarray(scored, dtype=bool)
    if chosen.shape != mask.shape:
        msg = (
            f"a coherent wall block's scored mask has shape {chosen.shape}, not the "
            f"frequency axis's {mask.shape}"
        )
        raise ModeError(msg)
    judged = chosen[mask]
    return judged if np.any(judged) else np.ones(judged.shape, dtype=bool)


def _scattered(
    values: np.ndarray,
    mask: Bool[np.ndarray, " n_omega"],
    leading: tuple[int, ...],
) -> np.ndarray:
    """Return active-band values placed back on the whole frequency axis.

    - ``values``: the values on the active bins, with the frequency axis last.
    - ``leading``: the shape in front of the frequency axis.
    - Returns: complex, shape ``(*leading, n_omega)``, exactly zero off the mask.
    """
    out = np.zeros((*leading, len(mask)), dtype=complex)
    out[..., mask] = values
    return out


def _check_wall_passivity(
    generator: Complex[np.ndarray, "n_modes n_modes n_omega"],
    wavenumber: Complex[np.ndarray, " n_omega"],
    radius: float,
    count: int,
) -> None:
    """Refuse a wall block whose Hermitian part is not negative semidefinite.

    The Hermitian part of the paraxial generator is the rank-one
    ``-Re(eta) h h^T / (|k| a^2 (4N^2 + |eta|^2))`` for the scalar block and rank two
    for the vector one (one rank per family: ``(S - S^H)/2i = S Re(H) S^H``), so it is
    negative exactly when the impedance lands on the passive branch. A cladding whose
    permittivity puts the square root elsewhere would give a wall that supplies power,
    which :class:`MarcatiliMode` clamps to zero for a diagonal loss; a clamp is
    not available here, because zeroing one entry of a rank-one operator is not a
    boundary condition. The block is refused instead.

    **No cladding model this package ships can reach the refusal**, and the proof is
    worth writing down rather than leaving as an untested branch. With
    ``s = sqrt(e - 1)`` on the principal root, ``Re s = x >= 0`` and
    ``1/v_HE = 2s/(s^2 + 2)``, whose real part is
    ``2x(|s|^2 + 2)/|s^2 + 2|^2 >= 0``; equality holds only for a real ``e < 1``,
    where the wall is exactly lossless rather than active. The guard is therefore a
    defence against a **future** impedance that does not take that root (the vector
    block's two impedances take it, and the guard is shared with it) and
    against a sign slipped into this arithmetic, not against a cladding a caller can
    pass today.

    - ``wavenumber``: the in-core wavenumber on the active band, 1/m.
    - ``radius``: core radius in m.
    """
    moved = np.moveaxis(generator, (0, 1), (-2, -1))
    hermitian = 0.5 * (moved + np.conj(np.swapaxes(moved, -1, -2)))
    largest = float(np.max(np.linalg.eigvalsh(hermitian)))
    scale = float(np.max(abs(generator)))
    budget = 64.0 * count * float(np.finfo(float).eps) * scale
    if largest > budget:
        rate = float(np.max(1.0 / (abs(wavenumber) * radius**2)))
        msg = (
            "the coherent wall block is not passive on the branch this cladding term "
            "lands on: the Hermitian part of the generator has eigenvalue "
            f"{largest:.4g} 1/m, above the {budget:.4g} 1/m rounding of a block "
            f"whose own scale is {rate:.4g} 1/m. A cladding outside Marcatili's "
            "regime has no outgoing-wall impedance to condense"
        )
        raise ModeError(msg)


def _check_wall_limits(limits: WallLimits, validity_limit: float) -> None:
    """Refuse or warn on the validity ratios of a wall block, by name."""
    name, value = limits.named()
    if value > validity_limit:
        msg = (
            f"the coherent wall block's {name} is {value:.4g} over "
            f"{limits.band[0]:.4g} to {limits.band[1]:.4g} rad/s, above the "
            f"{validity_limit:.4g} this block admits: the condensed paraxial model is "
            "not valid there. Narrow the band, widen the core, or retain fewer "
            "channels"
        )
        raise ModeError(msg)
    if value > 0.1 * validity_limit:
        warnings.warn(
            f"the coherent wall block's {name} reaches {value:.4g}, within a factor "
            f"ten of the {validity_limit:.4g} limit; the condensed paraxial model is "
            "marginal over this band",
            GrazingIncidenceWarning,
            stacklevel=3,
        )


def wall_square_root_metric(
    generator: Complex[np.ndarray, "n_modes_sqrt_out n_modes_sqrt_in"],
    wavenumber: float,
) -> tuple[
    Complex[np.ndarray, "n_modes_b_out n_modes_b_in"],
    Complex[np.ndarray, "n_modes_w_out n_modes_w_in"],
]:
    r"""Return the square-root generator and its candidate flux metric, as a diagnostic.

    For a **constant, real, positive** carrier ``k``, the square-root model
    ``c' = iBc`` with ``B^2 = M = k^2 I - 2ikG`` satisfies

    .. math::
       \frac{d}{dz}\left(c^\dagger W c\right)
       = -c^\dagger\frac{M - M^\dagger}{2ik}c,\qquad
       W = \frac{B + B^\dagger}{2k},

    which is the flux balance of this scalar model. **Its domain is exactly that
    statement.** Production semantics are the paraxial generator, not this square root;
    ``W`` is positive semidefinite only when the Hermitian part of ``B`` is, which
    positive eigenvalues of ``B`` do not establish, and which the principal branch does
    not establish either; a complex ``k``, an evanescent sector, a varying cross-section
    -- which adds ``W'`` -- and a varying ``k`` all fall outside it. Use it to measure
    the identity, not to define an error norm or to transport noise.

    Parameters
    ----------
    generator : numpy.ndarray
        The optics-convention paraxial generator at one frequency in 1/m, complex,
        shape ``(n_modes, n_modes)``.
    wavenumber : float
        The carrier ``k`` in 1/m, real and positive.

    Returns
    -------
    tuple of numpy.ndarray
        ``B`` in 1/m on the principal branch, and the dimensionless ``W``.

    Raises
    ------
    ModeError
        If the carrier is not real, finite and positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> from solprop.modes.wall import wall_square_root_metric
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> k = float(block.wavenumber[0].real)
    >>> root, metric = wall_square_root_metric(block.generator[:, :, 0], k)
    >>> bool(np.max(abs(root @ root - (k**2 * np.eye(2)
    ...      - 2j * k * block.generator[:, :, 0]))) < 1e-6 * k**2)
    True
    """
    carrier = float(wavenumber)
    if not math.isfinite(carrier) or carrier <= 0.0:
        msg = (
            "the square-root flux metric is derived for a constant real positive "
            f"carrier, got k = {wavenumber!r}"
        )
        raise ModeError(msg)
    block = np.asarray(generator, dtype=complex)
    identity = np.eye(block.shape[0], dtype=complex)
    squared = carrier**2 * identity - 2j * carrier * block
    root = np.asarray(sqrtm(squared), dtype=complex)
    return root, (root + np.conj(root.T)) / (2.0 * carrier)
