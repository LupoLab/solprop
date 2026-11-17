# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The catalogue of plane statistics: one :class:`StatDefinition` per name.

- :data:`CATALOGUE`: every statistic solprop knows, keyed by name, with its shape,
  unit, requirements, weighting and reference evaluator.
- The name sets derived from it, which the spec, the backends, the output capability
  and the general modal route read: :data:`GENERAL_MODAL_ACCESSORS`,
  :data:`WITHHELD_WITH_A_FLOW`, :data:`ACCESSOR_ADVICE`,
  :data:`METRIC_WEIGHTED_STATISTICS`, :data:`PROFILE_STATISTICS`,
  :data:`ON_AXIS_PLASMA_STATISTICS` and :data:`PLASMA_INTEGRATION_STATISTICS`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from solprop.stats.definitions import (
    analytic_flux,
    energy,
    energy_window,
    fwhm_t,
    fwhm_t_outer,
    omega0,
    peak_power,
    peak_power_total,
    peak_power_window,
    physical_energy,
    spectral_flux,
)

__all__ = [
    "ACCESSOR_ADVICE",
    "CATALOGUE",
    "GENERAL_MODAL_ACCESSORS",
    "METRIC_WEIGHTED_STATISTICS",
    "ON_AXIS_PLASMA_STATISTICS",
    "PLASMA_INTEGRATION_STATISTICS",
    "PROFILE_STATISTICS",
    "WITHHELD_WITH_A_FLOW",
    "StatDefinition",
]


@dataclass(frozen=True)
class StatDefinition:
    """What one statistic is called, what it returns and what it needs.

    Attributes
    ----------
    name : str
        Key the statistic is stored under.
    summary : str
        One line saying what the number is, used by the manual page.
    shape : str
        Shape of the value in the standard axis names, ``'()'`` for a scalar.
    units : str
        SI units of the value.
    requires : tuple
        What :func:`~solprop.stats.collect` must be given for this statistic:
        ``'windows'``, ``'modes'``, the name of a z-profile, ``'spatial_geometry'``, the
        far-field tables ``'apertures'`` and ``'regions'`` of the spec, or
        ``'separable_operator'``, the run's linear operator, which only a backend holds.
    backend : bool
        Whether only a backend can compute it. These are the statistics of quantities
        the saved field does not carry -- the electron density, the ionisation rate, the
        on-axis intensity, the quadrature error -- which a backend forms from its own
        buffers in one pass. :func:`~solprop.stats.collect` refuses them by name rather
        than pulling the state back across the seam to reconstruct them.
    between_planes : bool
        Whether a backend can also evaluate it *between* save planes, from what an
        accepted step already leaves behind, for the periodic channel of
        :attr:`~solprop.stats.StatsSpec.periodic`. What disqualifies a statistic is cost
        per step, not difficulty: ``zdw`` needs a root search per mode, and paying that
        on every step would cost a large share of the step; ``peak_intensity`` is in the
        subset because its on-axis matrix is built once and rescaled rather than
        rebuilt. :class:`~solprop.stats.StatsSpec` refuses a periodic name without it,
        by name.
    evaluate : collections.abc.Callable or None
        The function :func:`~solprop.stats.collect` forms it with from a saved plane's
        field: called ``(grid, field)``, with the spec's windows when it requires them,
        or with ``metric=`` when it is metric-weighted; ``None`` for one formed
        otherwise.
    metric : bool
        Whether its value depends on the basis's flux metric
        (:data:`~solprop.stats.catalogue.METRIC_WEIGHTED_STATISTICS`).
    weight : {'poynting', 'unit'} or None
        How it depends on the modal power weight: ``'poynting'`` for one a backend forms
        in its weighted Poynting pass, ``'unit'`` for a general modal statistic valid
        only under the constant weight (:func:`~solprop.stats.weight_refusal_for`).
    profile : bool
        Whether it reads the z-profile of its own name
        (:data:`~solprop.stats.catalogue.PROFILE_STATISTICS`).
    plasma : {'on_axis', 'integration'} or None
        Its family of plasma statistics
        (:data:`~solprop.stats.catalogue.ON_AXIS_PLASMA_STATISTICS`,
        :data:`~solprop.stats.catalogue.PLASMA_INTEGRATION_STATISTICS`).
    general : {'admitted', 'no_flow', 'diagnostic', 'medium'} or None
        How a general modal basis treats it: admitted, admitted unless the block couples
        (:data:`~solprop.stats.catalogue.WITHHELD_WITH_A_FLOW`), a step diagnostic the
        basis does not decide, refused because the guide has no gas fill or spatial
        geometry, or (``None``) refused because of the basis.
    advice : str or None
        What to use instead on a general modal basis
        (:data:`~solprop.stats.catalogue.ACCESSOR_ADVICE`).

    Examples
    --------
    >>> from solprop.stats import CATALOGUE
    >>> CATALOGUE["energy"].units, CATALOGUE["energy"].shape
    ('J', '(n_modes, n_pol); () for spatial beams')
    >>> CATALOGUE["electron_density"].backend
    True
    >>> CATALOGUE["energy"].between_planes, CATALOGUE["zdw"].between_planes
    (True, False)
    """

    name: str
    summary: str
    shape: str
    units: str
    requires: tuple[str, ...] = ()
    backend: bool = False
    between_planes: bool = False
    evaluate: Callable[..., np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    metric: bool = False
    weight: Literal["poynting", "unit"] | None = None
    profile: bool = False
    plasma: Literal["on_axis", "integration"] | None = None
    general: Literal["admitted", "no_flow", "diagnostic", "medium"] | None = None
    advice: str | None = None


_DETECTOR_ADVICE = (
    "analytic_flux or reconstructed E/H with an explicit detector definition"
)
"""What replaces a peak power or intensity on a general modal basis."""


def _catalogue() -> dict[str, StatDefinition]:
    """Return the fixed catalogue of statistics, keyed by name."""
    entries = (
        StatDefinition(
            name="omega0",
            summary="Centroid of the spectral energy density.",
            shape="(n_modes, n_pol)",
            units="rad/s",
            between_planes=True,
            evaluate=omega0,
            metric=True,
            weight="poynting",
            general="admitted",
        ),
        StatDefinition(
            name="energy",
            summary="Modal column energies or whole spatial beam energy.",
            shape="(n_modes, n_pol); () for spatial beams",
            units="J",
            between_planes=True,
            evaluate=energy,
            weight="poynting",
            advice=(
                "physical_energy, which integrates the same-frequency metric product"
            ),
        ),
        StatDefinition(
            name="energy_window",
            summary="Pulse energy inside each spectral window.",
            shape="(n_windows, n_modes, n_pol)",
            units="J",
            requires=("windows",),
            between_planes=True,
            evaluate=energy_window,
            weight="poynting",
            advice=(
                "physical_energy with an explicit band, or the reconstructed E/H of a "
                "saved plane"
            ),
        ),
        StatDefinition(
            name="peak_power",
            summary="Peak analytic Poynting power of each mode during a run.",
            shape="(n_modes,)",
            units="W",
            between_planes=True,
            evaluate=peak_power,
            weight="poynting",
            advice=_DETECTOR_ADVICE,
        ),
        StatDefinition(
            name="peak_power_window",
            summary="Peak analytic Poynting power per mode and spectral window.",
            shape="(n_windows, n_modes)",
            units="W",
            requires=("windows",),
            between_planes=True,
            evaluate=peak_power_window,
            weight="poynting",
            advice=_DETECTOR_ADVICE,
        ),
        StatDefinition(
            name="peak_power_total",
            summary="Peak total analytic Poynting power during a run.",
            shape="()",
            units="W",
            between_planes=True,
            evaluate=peak_power_total,
            weight="poynting",
            advice=_DETECTOR_ADVICE,
        ),
        StatDefinition(
            name="physical_energy",
            summary="Physical energy of a general modal basis, metric-weighted.",
            shape="()",
            units="J",
            between_planes=True,
            evaluate=physical_energy,
            metric=True,
            weight="unit",
            general="admitted",
        ),
        StatDefinition(
            name="spectral_flux",
            summary="Spectral density of a general modal basis's physical energy.",
            shape="(n_omega,)",
            units="J/(rad/s)",
            evaluate=spectral_flux,
            metric=True,
            weight="unit",
            general="admitted",
        ),
        StatDefinition(
            name="analytic_flux",
            summary="Peak total analytic Poynting flux of a general modal basis.",
            shape="()",
            units="W",
            between_planes=True,
            evaluate=analytic_flux,
            metric=True,
            weight="unit",
            general="admitted",
        ),
        StatDefinition(
            name="fwhm_t",
            summary="Central half-maximum width of the total Poynting power.",
            shape="()",
            units="s",
            between_planes=True,
            evaluate=fwhm_t,
            metric=True,
            weight="poynting",
            general="admitted",
        ),
        StatDefinition(
            name="fwhm_t_outer",
            summary="Total extent of Poynting power above half its maximum.",
            shape="()",
            units="s",
            between_planes=True,
            evaluate=fwhm_t_outer,
            metric=True,
            weight="poynting",
            general="admitted",
        ),
        StatDefinition(
            name="density",
            summary="Number density of the filling gas at the plane.",
            shape="()",
            units="1/m^3",
            requires=("density",),
            between_planes=True,
            profile=True,
            general="medium",
        ),
        StatDefinition(
            name="pressure",
            summary="Pressure of the filling gas at the plane.",
            shape="()",
            units="bar",
            requires=("pressure",),
            between_planes=True,
            profile=True,
            general="medium",
        ),
        StatDefinition(
            name="core_radius",
            summary="Core radius of the waveguide at the plane.",
            shape="()",
            units="m",
            requires=("core_radius",),
            between_planes=True,
            profile=True,
            general="medium",
        ),
        StatDefinition(
            name="zdw",
            summary="Zero-dispersion wavelength of each mode at the plane.",
            shape="(n_modes,)",
            units="m",
            requires=("modes",),
            general="no_flow",
        ),
        StatDefinition(
            name="species_ionisation",
            summary="Final ionised fraction of each species at every node.",
            shape="(n_species, n_nodes)",
            units="1",
            backend=True,
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="species_electron_density",
            summary="Species electron densities at the densest total-electron node.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="species_peak_ionisation_rate",
            summary="Species peak rate over all nodes; cycle-averaged for envelopes.",
            shape="(n_species,)",
            units="1/s",
            backend=True,
            between_planes=False,
            general="medium",
        ),
        StatDefinition(
            name="species_density",
            summary="Component number densities at this plane.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=False,
            general="medium",
        ),
        StatDefinition(
            name="species_ionisation_energy",
            summary="Final ionised inventory energy per propagation length.",
            shape="(n_species,)",
            units="J/m",
            backend=True,
            between_planes=False,
            general="medium",
        ),
        StatDefinition(
            name="electron_density",
            summary="Free-electron density left behind by the pulse.",
            shape="()",
            units="1/m^3",
            backend=True,
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="peak_ionisation_rate",
            summary="Peak ionisation rate; cycle-averaged for envelopes.",
            shape="()",
            units="1/s",
            backend=True,
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="on_axis_electron_density",
            summary="Free-electron density left behind on the axis.",
            shape="()",
            units="1/m^3",
            backend=True,
            between_planes=False,
            plasma="on_axis",
        ),
        StatDefinition(
            name="on_axis_peak_ionisation_rate",
            summary="Peak rate on the axis; cycle-averaged for envelopes.",
            shape="()",
            units="1/s",
            backend=True,
            between_planes=False,
            plasma="on_axis",
        ),
        StatDefinition(
            name="on_axis_species_electron_density",
            summary="Each species' electron density left behind on the axis.",
            shape="(n_species,)",
            units="1/m^3",
            backend=True,
            between_planes=False,
            plasma="on_axis",
        ),
        StatDefinition(
            name="on_axis_species_peak_ionisation_rate",
            summary="Species peak rate on the axis; cycle-averaged for envelopes.",
            shape="(n_species,)",
            units="1/s",
            backend=True,
            between_planes=False,
            plasma="on_axis",
        ),
        StatDefinition(
            name="plasma_undershoot",
            summary="Largest exposure undershoot of the plasma column's band integral.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
            plasma="integration",
            general="medium",
        ),
        StatDefinition(
            name="plasma_work_defect",
            summary="Loss-current work against the ionisation it paid, relative.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
            plasma="integration",
            general="medium",
        ),
        StatDefinition(
            name="plasma_boundary_level",
            summary="Free-current integrand at the window start, relative to its peak.",
            shape="()",
            units="1",
            backend=True,
            between_planes=False,
            plasma="integration",
            general="medium",
        ),
        StatDefinition(
            name="peak_intensity",
            summary="Spatial vacuum intensity; modal on-axis analytic Poynting flux.",
            shape="()",
            units="W/m^2",
            backend=True,
            between_planes=True,
            weight="poynting",
            advice=_DETECTOR_ADVICE,
        ),
        StatDefinition(
            name="fwhm_r",
            summary="Central fluence FWHM; NaN if undefined.",
            shape="()",
            units="m",
            requires=("spatial_geometry",),
            general="medium",
        ),
        StatDefinition(
            name="on_axis_intensity",
            summary="Largest intensity in time on the spatial beam axis.",
            shape="()",
            units="W/m^2",
            requires=("spatial_geometry",),
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="on_axis_spectrum",
            summary="On-axis spectral fluence density summed over polarisation.",
            shape="(n_omega,)",
            units="J s/m^2",
            requires=("spatial_geometry",),
            general="medium",
        ),
        StatDefinition(
            name="beam_radius",
            summary="Fluence second-moment and 1/e^2 radii; NaN if undefined.",
            shape="(2,)",
            units="m",
            requires=("spatial_geometry",),
            general="medium",
        ),
        StatDefinition(
            name="aperture_spectrum",
            summary="Spectrum collected through each far-field aperture.",
            shape="(n_apertures, n_omega)",
            units="J/(rad/s)",
            requires=("apertures", "spatial_geometry"),
            general="medium",
        ),
        StatDefinition(
            name="aperture_on_axis_spectrum",
            summary="On-axis spectral fluence behind each aperture in a Fourier plane.",
            shape="(n_apertures, n_omega)",
            units="J s/m^2",
            requires=("apertures", "spatial_geometry"),
            general="medium",
        ),
        StatDefinition(
            name="region_spectrum",
            summary="Spectrum in each box of transverse wavevector.",
            shape="(n_regions, n_omega)",
            units="J/(rad/s)",
            requires=("regions", "spatial_geometry"),
            general="medium",
        ),
        StatDefinition(
            name="beam_spectrum",
            summary="Whole-beam spectral energy density.",
            shape="(n_omega,)",
            units="J/(rad/s)",
            requires=("spatial_geometry",),
            general="medium",
        ),
        StatDefinition(
            name="absorbed_power",
            summary="Energy per metre each absorbing channel of the operator removes.",
            shape="(5,)",
            units="J/m",
            requires=("spatial_geometry", "separable_operator"),
            backend=True,
            between_planes=True,
            general="medium",
        ),
        StatDefinition(
            name="quadrature_error",
            summary="Relative error of the transverse quadrature on the polarisation.",
            shape="()",
            units="1",
            backend=True,
            between_planes=True,
            general="diagnostic",
        ),
    )
    return {entry.name: entry for entry in entries}


CATALOGUE: Mapping[str, StatDefinition] = _catalogue()
"""Every statistic solprop knows about, keyed by name."""

GENERAL_MODAL_ACCESSORS: frozenset[str] = frozenset(
    name
    for name, entry in CATALOGUE.items()
    if entry.general in ("admitted", "no_flow")
) | frozenset(
    # Processing accessors of the general basis (solprop.processing.modal).
    (
        "reconstructed_flux",
        "channel_labels",
        "coordinate_intensity",
        "instantaneous_flux",
        "electric_field",
        "magnetic_field",
        "plane_nodes",
    )
)
"""Every accessor a three-component general modal basis admits, by name.

The sets live here, beside the catalogue whose names they select from, and
:mod:`solprop.output.capability` imports them to build the gate: they are a statement
about statistics, and keeping them here is what lets
:mod:`solprop.translate.general` record the list a file publishes without a
translation module importing an output one.

``omega0``, ``fwhm_t`` and ``fwhm_t_outer`` are functionals of the same-frequency flux
``a.H W a`` and take the basis metric
(:data:`solprop.stats.catalogue.METRIC_WEIGHTED_STATISTICS`), so they are exact on a
declared basis as well as on the identity one; ``zdw`` is a property of the channels'
propagation constants and of no basis at all -- until a declared basis couples channels
whose propagation constants differ, when the block executes through a coupled flow and
each channel's constant is the block's mean; the route then withholds it
(:data:`WITHHELD_WITH_A_FLOW`).

``spectral_flux`` is the one array of the family -- the same flux as a density in
frequency rather than integrated. It
is **admitted but not a default** of the route: it is ``n_omega`` wide per plane where
the other five are scalars (``omega0`` one per channel), so a run asks for it by name.

Two names of :mod:`solprop.processing.modal` are deliberately absent.
``quadrature_error`` is refused separately, by the driver, because what disqualifies
it is the absent embedded coarse rule and not the basis -- a general guide's node rule
has no embedded pair to difference, so such a run has no transverse-resolution alarm,
which is a known limitation -- and it sits
inside the run's resume key, so publishing it here would promise a statistic the
same record refuses. ``channel_amplitudes`` takes two arrays and a table rather than
an opened handler, so no door ever asks the gate about it; it is in
``solprop.output.capability._GENERAL_ONLY`` all the same, so that the one
direction that could be reached -- an ordinary run asked for it -- would be refused
for the right reason.
"""

WITHHELD_WITH_A_FLOW: frozenset[str] = frozenset(
    name for name, entry in CATALOGUE.items() if entry.general == "no_flow"
)
"""The general-basis accessors a route withholds when its block couples, by name.

A declared change of basis over channels whose propagation constants differ executes
through ``linear.coupled_flow``, and the per-channel propagation constant the
statistics would read is then the block's mean: ``zdw`` from it is a number that
propagates nothing. The route leaves these names out of the record's published list and
every accessor door refuses them with :func:`solprop.output.capability.refusal_for`'s
``withheld``.
"""

ACCESSOR_ADVICE: Mapping[str, str] = {
    name: entry.advice for name, entry in CATALOGUE.items() if entry.advice is not None
}
"""What to use instead of each per-channel name on a general modal basis.

A name with no entry here gets the generic advice of
:mod:`solprop.output.capability`: a general modal basis has ``physical_energy``,
``analytic_flux`` and the reconstruction accessors, and nothing else of that family.
"""

_GAS_OR_SPATIAL_NAMES: frozenset[str] = frozenset(
    name for name, entry in CATALOGUE.items() if entry.general == "medium"
)
"""Names a general modal guide fails for a reason that is not its basis at all.

A plasma density, a gas pressure and a beam radius are unavailable because this guide
has no gas fill and no spatial geometry, not because its coordinate powers do not add
up; advising a reader to use ``analytic_flux`` instead of ``electron_density`` would
be nonsense. The gate names that reason instead.
"""

_STEP_DIAGNOSTICS: frozenset[str] = frozenset(
    name for name, entry in CATALOGUE.items() if entry.general == "diagnostic"
)
"""Diagnostics of the step, which are the same quantity in any basis.

Basis-independent step diagnostics are admitted on a general basis, so the gate never
refuses one: whoever owns the real reason refuses it
instead. ``quadrature_error`` is refused on a general guide by the run loop, because a
provider's node rule carries no embedded coarse rule to difference --
a statement about the rule, not about the basis -- and it is deliberately not in
:data:`GENERAL_MODAL_ACCESSORS`, so no run record publishes as admitted a name the
same record's run refuses.
"""

METRIC_WEIGHTED_STATISTICS: frozenset[str] = frozenset(
    name for name, entry in CATALOGUE.items() if entry.metric
)
"""Statistics whose value depends on the basis's flux metric, by name.

These are the functionals of the physical same-frequency flux
``sum_mn conj(a_m) W_mn a_n`` -- its integral (``physical_energy``), its density in
frequency (``spectral_flux``), its peak (``analytic_flux``), its two widths
(``fwhm_t``, ``fwhm_t_outer``) and the spectral centroid of its per-channel share
(``omega0``). On an orthonormal basis ``W`` is the identity and they are the unweighted
quantities they have always been, bit for bit; on a general modal basis whose Gram is
not the identity they are wrong without it, so a caller that has a metric must hand it
over (:func:`~solprop.stats.collect`) and one that cannot must refuse rather than
answer (:mod:`solprop.processing.modal`).

``zdw`` is deliberately absent: it is a property of the channels' propagation
constants and of no basis at all, and a unitary change of basis of an exactly
degenerate block leaves every one of them unchanged.
"""

PROFILE_STATISTICS: tuple[str, ...] = tuple(
    name for name, entry in CATALOGUE.items() if entry.profile
)
"""Statistics that read a :class:`solprop.profiles.ZProfile` at the plane's position."""

ON_AXIS_PLASMA_STATISTICS: tuple[str, ...] = tuple(
    name for name, entry in CATALOGUE.items() if entry.plasma == "on_axis"
)
"""The plasma statistics evaluated on the waveguide's axis (SEP 19, section 5.2).

Opt-in and saved-plane only. The axis field is synthesised from the mode profiles, as
for ``peak_intensity``, and its populations are scanned with exactly the rate tables,
preionisation, densities and running-integral rule of the node scan, once per
observation; a rule with a node on the axis supplies that row instead. They sit beside
the unchanged node reductions -- ``electron_density`` is the densest node's -- because a
few-node field-scale rule may have no node near the axis, and a multimode field's
largest density need not lie on it. Defined on resolved modal routes; the spatial
routes, general modal guides, the mode-averaged route and TDSE refuse them by name.
"""

PLASMA_INTEGRATION_STATISTICS: tuple[str, ...] = tuple(
    name for name, entry in CATALOGUE.items() if entry.plasma == "integration"
)
"""The field plasma column's run-time integration diagnostics (SEP 18, section 4.5).

Evaluated with the other plasma statistics at each save plane. Each is a ratio of two
maxima over every sample, node and population, so that a node the field barely
ionises cannot set it: the undershoot ``max(-dS) / max(dS)`` of the exposure
increments; the work defect ``max |paid - freed| / max freed``, the loss current's work
per ionisation potential against the fraction it freed; and the boundary level, the
free-current integrand's largest magnitude in the ``2K`` samples the filter reads past
the window's start against its largest anywhere.
"""
