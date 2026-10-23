# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The mode protocol: what the rest of solprop needs from a waveguide mode.

A mode is anything with an effective index and a transverse field profile. The linear
operator needs ``neff`` and its analytic frequency derivative (the co-moving frame,
ADR-006); the multimode transform needs the field at quadrature nodes and the geometry
of the cross-section; the mode-averaged normalisation needs the effective area; the
translation layer needs the three traits that say how the mode changes along z, and, for
a mode whose whole z dependence is a handful of scalar paths, the two members of
:class:`ParameterisedMode` that let the operator be evaluated at parameter *values*
rather than at positions. Nothing here computes anything: implementations live in
:mod:`solprop.modes` and, for a dispersion-specified GNLSE mode, in
``solprop.nonlinear.gnlse``.

Conventions (``docs/dev/conventions.md`` section 5): ``Im n_eff > 0`` is loss;
``n_eff - 1`` is available without cancellation for gases through
:meth:`Mode.neff_minus_one`; points are ``(r, theta)`` for a polar cross-section and
``(x, y)`` for a Cartesian or slab one, in metres and radians; fields are the transverse
Cartesian components ``(E_x, E_y)`` of the *unnormalised* profile, whose normalisation
``N`` and effective area follow from them.

A mode **uniform in y** -- the hollow planar guide of :mod:`solprop.modes.slab` --
declares the ``"slab"`` cross-section and the optional ``transverse_measure`` trait
``"per_unit_width"``: its normalisation is a power per metre of y, its effective
"area" is an effective width in metres, and the powers and energies of a run on it are
per unit width (:data:`TransverseMeasure`). A mode without the trait is ``"area"``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float

from solprop._display import Displayed, line
from solprop.exceptions import ModeError
from solprop.profiles import ZProfile

__all__ = [
    "GEOMETRY_PATH_NAMES",
    "ChannelLabel",
    "CrossSection",
    "Handedness",
    "Mode",
    "ParameterisedMode",
    "Parity",
    "PolarisationComponent",
    "TransverseMeasure",
]

GEOMETRY_PATH_NAMES = ("core_radius", "width_x", "width_y", "area_path")
"""The attribute names under which a mode declares its cross-section along z.

A hollow capillary declares ``core_radius``, a rectangular guide its two widths, and
a :class:`solprop.nonlinear.gnlse.SpecifiedMode` its ``area_path``: the
effective area is the whole of its cross-section, so its breakpoints are
cross-section kinks exactly as a bore corner is. Each is a
:class:`solprop.profiles.ZProfile`.

The names live here because two modules must agree on them and neither may import
the other: :mod:`solprop.translate.modal` walks them for the kinks the tables and the
schedule land on, and :mod:`solprop.nonlinear.gnlse` walks them for the z ownership of
a mode a specified fibre is built from. The translation layer's tuple is these names
plus ``dispersion_path``.
"""

CrossSection = Literal["polar", "cartesian", "slab"]
"""How a mode parametrises its cross-section: ``(r, theta)``, ``(x, y)`` or ``x`` alone.

A ``"slab"`` cross-section is uniform in y: its limits are one-tuples ``(-a,)`` and
``(a,)`` in x, and its points are still ``(x, y)`` pairs, with ``y = 0`` on the rule's
nodes and the profile independent of it.
"""

TransverseMeasure = Literal["area", "per_unit_width"]
"""What the transverse integral of a mode or a grid measures.

``"area"`` is an integral over the whole cross-section: powers in W, energies in J,
effective areas in m². ``"per_unit_width"`` is an integral over x alone for a geometry
uniform in y: powers in W/m, energies in J/m, the effective "area" an effective width
in m. Intensities are W/m² either way. A mode declares it with an optional
``transverse_measure`` property, read by name, whose absence means ``"area"``.
"""

PolarisationComponent = Literal["x", "y"]
"""Name one transverse Cartesian field component."""

Parity = Literal["even", "odd"]
"""Which of a degenerate azimuthal pair a channel is; :class:`ChannelLabel` defines it.

Not restated here: one definition, in one place.
"""

Handedness = Literal["left", "right"]
"""Which circular partner a complex degenerate channel is."""


@dataclass(frozen=True)
class ChannelLabel(Displayed):
    """Name one physical channel of a waveguide, independently of its position.

    The identity the mode axis carries: the family and the quantum numbers a person
    quotes, with ``None`` wherever a family has no such number -- an axially
    symmetric ``TE_0m`` has no parity partner, so its parity is ``None``, and a real
    basis has no handedness. The label belongs to the mode, not to its index in a
    collection: a set's order is the order of its tuple, and two members may carry
    the same label only where they are genuinely the same physical channel in
    different gauges.

    **Parity is defined once, here, for every family.** ``"even"`` is the member
    whose transverse pattern is the ``cos((n - 1) theta)`` one about the **x** axis
    -- for ``n = 1`` that is the x-polarised member -- and ``"odd"`` is its
    quarter-turn partner, ``sin((n - 1) theta)`` about x. A family reports the
    parity of the physical pattern, whatever its own construction is written in: an
    exact step-index ``HE_nm`` builds the pair from ``cos(n theta)`` and
    ``sin(n theta)`` on the *radial* component and a Marcatili ``HE_nm`` from a
    rotation angle, and both land on this one definition. Note that
    :attr:`azimuthal_order` is the family index ``n`` while the parity names the
    ``(n - 1)``-fold pattern the transverse vector field actually has; the two
    differ by one, and that is the physics of a hybrid mode rather than a slip.

    Attributes
    ----------
    family : str
        Non-empty family name: ``HE``, ``EH``, ``TE``, ``TM``, ``LP``, ``rect`` or a
        provider's own declared family.
    azimuthal_order : int or None
        Non-negative azimuthal index ``n``, or ``None`` where the family has none.
    radial_order : int or None
        Positive radial index ``m``, or ``None`` where the family has none.
    parity : {'even', 'odd'} or None
        Which of a degenerate azimuthal pair, by the definition above, or ``None``
        for a channel with no parity partner or in an unlabelled gauge.
    handedness : {'left', 'right'} or None
        Which circular partner of a complex degenerate pair, or ``None`` for a real
        basis. The convention is defined once, by
        :func:`solprop.modes.guide.circular_pair`: ``left`` is
        ``(e_even - i e_odd)/sqrt(2)``, positive helicity in the internal convention
        and positive Stokes ``V`` in the optics one
        (``docs/guide/setup/polarisation.md``).
    display_label : str
        Non-empty stable text for a person, a legend or a file: ``HE1,1:even``.

    Examples
    --------
    >>> from solprop.modes.protocol import ChannelLabel
    >>> label = ChannelLabel("TE", 0, 1, None, None, "TE0,1")
    >>> label.display_label, label.parity
    ('TE0,1', None)
    """

    family: str
    azimuthal_order: int | None
    radial_order: int | None
    parity: Parity | None
    handedness: Handedness | None
    display_label: str

    def __repr__(self) -> str:
        """Return a one-line description: the display label and the family.

        Returns
        -------
        str
            For example ``ChannelLabel(HE1,1:even, family=HE)``.
        """
        return line("ChannelLabel", [("", self.display_label), ("family", self.family)])

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return f"ChannelLabel: {self.display_label}", [
            ("family", self.family),
            ("azimuthal order", self.azimuthal_order),
            ("radial order", self.radial_order),
            ("parity", self.parity or "none"),
            ("handedness", self.handedness or "none"),
        ]

    def __post_init__(self) -> None:
        """Validate the family, the orders and the two partner labels.

        Raises
        ------
        solprop.exceptions.ModeError
            If the family or the display label is empty, an order is negative, or a
            parity or handedness is not one of its two names.
        """
        if not self.family or not self.display_label:
            raise ModeError("a channel label needs a family and a display label")
        for name, lowest in (("azimuthal_order", 0), ("radial_order", 1)):
            value = getattr(self, name)
            # bool is an int in Python, so True would otherwise become an order
            # here while :meth:`from_mapping` refuses it: one rule, both doors.
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < lowest
            ):
                raise ModeError(
                    f"{name} must be None or an integer of at least {lowest}, "
                    f"got {value!r}"
                )
        if self.parity not in (None, "even", "odd"):
            raise ModeError(
                f"parity must be 'even', 'odd' or None, got {self.parity!r}"
            )
        if self.handedness not in (None, "left", "right"):
            raise ModeError(
                f"handedness must be 'left', 'right' or None, got {self.handedness!r}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ChannelLabel":  # noqa: UP037 - the runtime hook resolves this while the class body runs
        """Rebuild a label from the mapping :meth:`as_mapping` produced.

        The round trip a provider's ``identity["channels"]`` and output schema 7
        need: the mapping has been through JSON, so every value arrives as an
        ordinary Python object and is checked here before it becomes a label.

        Parameters
        ----------
        mapping : collections.abc.Mapping
            Exactly the six keys of :meth:`as_mapping`, with a string family and
            display label, integer or ``None`` orders, and string or ``None``
            parity and handedness.

        Returns
        -------
        ChannelLabel
            The label.

        Raises
        ------
        solprop.exceptions.ModeError
            If a key is missing or extra, or a value has the wrong type.

        Examples
        --------
        >>> from solprop.modes.protocol import ChannelLabel
        >>> label = ChannelLabel("TE", 0, 1, None, None, "TE0,1")
        >>> ChannelLabel.from_mapping(label.as_mapping()) == label
        True
        """
        allowed: Mapping[str, tuple[type, ...]] = {
            "family": (str,),
            "azimuthal_order": (int, type(None)),
            "radial_order": (int, type(None)),
            "parity": (str, type(None)),
            "handedness": (str, type(None)),
            "display_label": (str,),
        }
        if set(mapping) != set(allowed):
            raise ModeError(
                f"a channel label mapping needs exactly {sorted(allowed)}, got "
                f"{sorted(mapping)}"
            )
        for name, kinds in allowed.items():
            # bool is an int in Python and would silently become an order.
            if not isinstance(mapping[name], kinds) or isinstance(mapping[name], bool):
                names = " or ".join(kind.__name__ for kind in kinds)
                raise ModeError(
                    f"a channel label's {name} must be {names}, got {mapping[name]!r}"
                )
        return cls(
            cast("str", mapping["family"]),
            cast("int | None", mapping["azimuthal_order"]),
            cast("int | None", mapping["radial_order"]),
            cast("Parity | None", mapping["parity"]),
            cast("Handedness | None", mapping["handedness"]),
            cast("str", mapping["display_label"]),
        )

    def as_mapping(self) -> Mapping[str, str | int | None]:
        """Return the label as the serialisable mapping an identity carries.

        Returns
        -------
        collections.abc.Mapping
            The six fields by name, in the order they are declared, with ``None``
            where a family has no such quantum number. This is what a provider's
            ``identity`` publishes and what output schema 7 will persist.

        Examples
        --------
        >>> from solprop.modes.protocol import ChannelLabel
        >>> mapping = ChannelLabel("TE", 0, 1, None, None, "TE0,1").as_mapping()
        >>> mapping["display_label"], mapping["parity"]
        ('TE0,1', None)
        """
        return {
            "family": self.family,
            "azimuthal_order": self.azimuthal_order,
            "radial_order": self.radial_order,
            "parity": self.parity,
            "handedness": self.handedness,
            "display_label": self.display_label,
        }


@runtime_checkable
class Mode(Protocol):
    """A guided mode as consumed by the linear operator, the transforms and the stats.

    Every method takes the position ``z`` in metres, so that tapers and pressure
    gradients are ordinary cases; a z-independent mode ignores it and reports
    ``z_constant``.
    """

    @property
    def z_constant(self) -> bool:
        """Whether the mode is independent of ``z`` (no taper, no gradient)."""
        ...

    @property
    def scale_invariant(self) -> bool:
        """Whether the profile scales exactly with the core radius along ``z``.

        A Marcatili mode has ``e(r; a) = f(r / a)`` and ``N ∝ a²``, so the synthesis
        and projection matrices of the multimode transform rescale by two scalars per
        ``z`` instead of being re-evaluated.
        """
        ...

    @property
    def channel_label(self) -> ChannelLabel:
        """Name the physical channel this mode is, with its quantum numbers.

        The label is a property of the mode itself, not of its place in a set:
        :attr:`solprop.modes.collection.ModeCollection.channel_labels` is the ordered
        tuple of its members' labels, and output schema 7 persists them beside the
        mode axis.
        """
        ...

    @property
    def azimuthal_order(self) -> int:
        """Highest azimuthal order ``h`` of the profile, which sets the theta rule size.

        The periodic trapezoid rule is exact for trigonometric polynomials of degree
        at most ``n_theta - 1`` and cubic products of order-``h`` modes reach degree
        ``4 h``, so the transform needs ``n_theta > 4 h``.
        """
        ...

    def principal_component(self, z: float = 0.0) -> PolarisationComponent | None:
        """Return the sole nonzero Cartesian component, or both components as None.

        This is a property of the continuous profile, not a test at quadrature nodes:
        an aliased rule must not hide a component that the mode actually carries.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        {'x', 'y'} or None
            The sole nonzero component; ``None`` means both components are needed.
        """
        ...

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the complex effective index.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive: no
            waveguide has a mode at zero frequency, and the linear operator evaluates
            on the active band, which excludes it.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``n_eff(omega, z)``, complex, shape ``(n_omega,)``; ``Im n_eff >
            0`` is loss.
        """
        ...

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return ``n_eff - 1`` formed without cancellation.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive: no
            waveguide has a mode at zero frequency, and the linear operator evaluates
            on the active band, which excludes it.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``n_eff - 1``, complex, shape ``(n_omega,)``, exact to rounding for a
            gas-filled waveguide (``docs/dev/conventions.md`` section 5).
        """
        ...

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic frequency derivative of the effective index.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive: no
            waveguide has a mode at zero frequency, and the linear operator evaluates
            on the active band, which excludes it.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``d n_eff / d omega`` in s/rad, complex, shape ``(n_omega,)``.
            Analytic where
            the model allows (ADR-006); an implementation that differences must say so
            in its docstring and report the accuracy it reaches.
        """
        ...

    def aeff(self, z: float = 0.0) -> float:
        """Return the effective area.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        float
            ``A_eff = (∫|e|² dA)² / ∫|e|⁴ dA`` in m²; for a mode uniform in y
            (``transverse_measure = "per_unit_width"``) the same ratio over x
            alone, an effective width in m.
        """
        ...

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, ...], tuple[float, ...]]:
        """Return the cross-section parametrisation and the integration limits.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        tuple
            ``(kind, lower, upper)``: ``"polar"`` with ``(r, theta)`` limits,
            ``"cartesian"`` with ``(x, y)`` limits, or ``"slab"`` with the one-tuple
            ``x`` limits of a guide uniform in y, in m and rad.
        """
        ...

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return the transverse field components at points of the cross-section.

        Parameters
        ----------
        points : numpy.ndarray
            Shape ``(n_points, 2)``: ``(r, theta)`` or ``(x, y)`` per :meth:`dimlimits`;
            a slab mode reads ``x`` and ignores ``y``.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``(E_x, E_y)`` of the unnormalised profile at each point, shape
            ``(n_points, 2)``, real.
        """
        ...


@runtime_checkable
class ParameterisedMode(Mode, Protocol):
    """A mode whose whole z dependence runs through named scalar paths.

    A Marcatili mode reaches ``z`` only through its core radius ``a(z)`` and its number
    density ``rho(z)``, so its effective index is a function of those two scalars and
    the frequency: ``n_eff(omega, z) = n_eff(omega; a(z), rho(z))``. Stating that here
    is what lets :func:`solprop.linear.dispersion_operator` be called on parameter
    values rather than on positions, and so lets the translation layer expand the
    cumulative operator in the parameters instead of tabulating it along z
    (``docs/dev/kernel-contract.md`` section 3).

    A declared path is a statement of **z ownership** -- this is where the mode's z
    dependence lives, so these are its kinks and this is what an expansion may be
    taken in -- and not a statement that the linear operator depends on it: a
    specified mode declares its effective area, whose whole effect is on the nonlinear
    coefficient, and the operator is then expanded in a parameter it does not depend
    on, at degree zero.

    A mode whose declared paths are **not** the whole of its z dependence says so with
    the optional ``parameters_own_z`` trait, which
    :func:`solprop.translate.linop.parameters_own_z` reads by name. It defaults to
    ``True`` -- what every mode written before the trait existed implied -- and a mode
    that reports ``False`` keeps the translation layer on the z table instead of an
    expansion its parameters cannot reproduce. The case is a
    :class:`solprop.nonlinear.gnlse.SpecifiedMode` whose dispersion follows a path
    beside its declared area: the area is still its
    kinks, but freezing the area would leave the dispersion moving.

    A mode that does not implement this is not less capable, only less exploitable: the
    translation layer falls back to the z table for it, which is the general
    representation. A z-constant mode need not implement it either, having a cheaper
    path still.
    """

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the named scalar paths the effective index depends on.

        Returns
        -------
        collections.abc.Mapping
            Path name to :class:`~solprop.profiles.ZProfile`, for example
            ``{"core_radius": ..., "density": ...}``. Every key is accepted by
            :meth:`at_parameters`; constant paths are included, so that a caller sees
            the whole dependence and decides for itself which of them vary.
        """
        ...

    def at_parameters(self, values: Mapping[str, float]) -> "Mode":  # noqa: UP037 - beartype evaluates the annotation eagerly
        """Return the same mode with the named paths frozen at the given values.

        Parameters
        ----------
        values : collections.abc.Mapping
            Path name to value, in the units of :attr:`parameter_paths`. Every name
            must be one of that mapping's keys; a name it does not know is an error
            rather than a silent no-op. Names left out keep their path.

        Returns
        -------
        Mode
            A mode that reports ``z_constant`` and reproduces this one at every
            position where its paths take the given values.

        Raises
        ------
        solprop.exceptions.ModeError
            If a name is not a declared path, or a value is outside the path's range
            of validity.
        """
        ...
