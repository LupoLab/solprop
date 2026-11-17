# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluate a :class:`~solprop.stats.StatsSpec` on a saved plane.

- :func:`collect`: every statistic a spec names, from one saved plane, through the
  reference definitions, the catalogue and the spatial measures of
  :mod:`solprop.spatial`.
- :func:`default_spec`: the statistics a route records unless told otherwise.
- :data:`WEIGHT_DEPENDENT_STATISTICS`, whose values depend on the modal power weight,
  and :func:`weight_refusal_for`, why a general statistic cannot be formed under one.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Complex, Float

from solprop.constants import EPS0, C
from solprop.exceptions import NoZeroDispersionError, OutputError, SolpropWarning
from solprop.grid import RadialGrid, TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.slab import PlanarGrid
from solprop.output.records import SavedPlane, SimulationKind
from solprop.profiles import ZProfile
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    OPERATOR_STATISTICS,
    SpatialFlux,
    aperture_statistics,
    beam_radius,
    fwhm_r,
    on_axis_intensity,
    on_axis_spectrum,
    planar_intensity,
    spatial_energy,
    spatial_intensity,
)
from solprop.stats.catalogue import CATALOGUE, PROFILE_STATISTICS
from solprop.stats.definitions import (
    _power_half_width,
    _window_masks,
    analytic_field,
    inband_wavelengths,
    profile_value,
    zdw,
)
from solprop.stats.spec import StatsSpec, far_field_tables

__all__ = [
    "WEIGHT_DEPENDENT_STATISTICS",
    "collect",
    "default_spec",
    "weight_refusal_for",
]

logger = logging.getLogger(__name__)


def _profile_for(profiles: Mapping[str, ZProfile] | None, name: str) -> ZProfile:
    """Return the z-profile one statistic needs, or say which one is missing.

    - ``profiles``: the profiles the run has, keyed by statistic name.
    """
    if profiles is None or name not in profiles:
        available = sorted(profiles) if profiles else []
        msg = (
            f"the {name!r} statistic needs a {name!r} z-profile; "
            f"collect was given {available}"
        )
        raise OutputError(msg)
    return profiles[name]


def _zdw_or_nan(
    modes: Sequence[Mode], z: float, grid: TimeGrid
) -> Float[np.ndarray, " n_modes"]:
    """Return each mode's zero-dispersion wavelength in a grid's band, or ``nan``.

    Of several zeros in the band, the one nearest the grid's reference wavelength. A
    run asked for this statistic is not stopped because its band holds no zero: the
    plane records ``nan`` with a warning, as :func:`fwhm_t` does for a missing crossing.
    A direct call to :func:`zdw` still raises.

    - ``z``: position along the waveguide, m.
    - Returns: Zero-dispersion wavelength or ``nan``, shape ``(n_modes,)``, m.
    """
    window = inband_wavelengths(grid)
    values = np.full(len(modes), np.nan)
    for index, mode in enumerate(modes):
        try:
            values[index] = zdw(
                [mode], z, wavelength_limits=window, near=grid.reference_wavelength
            )[0]
        except NoZeroDispersionError:
            continue
    missing = np.flatnonzero(np.isnan(values))
    if missing.size:
        warnings.warn(
            f"the 'zdw' statistic records nan for modes {missing.tolist()}: the "
            "search found no zero of beta_2 in the grid's "
            f"{window[0] * 1e9:.4g}-{window[1] * 1e9:.4g} nm band that no resonance "
            "separates from its reference wavelength",
            SolpropWarning,
            stacklevel=3,
        )
    return values


def collect(
    spec: StatsSpec,
    grid: TimeGrid,
    plane: SavedPlane,
    *,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
    flux: SpatialFlux | None = None,
    metric: Complex[np.ndarray, "n_modes_metric_out n_modes_metric_in n_omega_metric"]
    | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate every statistic a spec asks for at one saved plane.

    Each statistic uses its named public array function and deterministic reductions.
    For saved true-power modal amplitudes, temporal array functions use ``|b(t)|²``.
    Backend run statistics use analytic Poynting flux: their time integrals agree,
    but broadband peaks and widths can differ (``docs/dev/conventions.md`` section 4).

    Parameters
    ----------
    spec : StatsSpec
        Which statistics to evaluate, and the windows they need.
    grid : solprop.grid.TimeGrid
        The grid the plane's field lives on.
    plane : solprop.output.records.SavedPlane
        The plane, whose field is in the internal FFT convention.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order. Needed by ``zdw``.
    profiles : collections.abc.Mapping, optional
        The z-profiles, keyed by statistic name. Needed by ``density``, ``pressure``
        and ``core_radius``.
    flux : solprop.spatial.SpatialFlux or None, optional
        The plane's measure for a radial or Cartesian spectrum, resolved at the
        plane's own position (and jump side) in its medium: every spatial statistic
        is that medium's axial Poynting flux (SEP 34), and widths use the signed
        time-integrated fluence.
        The far-field statistics ``aperture_spectrum``,
        ``aperture_on_axis_spectrum``, ``region_spectrum`` and ``beam_spectrum``
        need a Cartesian grid or a line (SEP 23), and are formed in one pass by
        :func:`solprop.spatial.aperture_statistics`, the function a run uses.
    metric : numpy.ndarray or None, optional
        The basis flux metric ``W(omega)`` at this plane, shape
        ``(n_modes, n_modes, n_omega)``, for the names in
        :data:`~solprop.stats.catalogue.METRIC_WEIGHTED_STATISTICS`; ``None``, the
        default, is the identity and every statistic is then bitwise what it was. A
        caller that holds a general modal problem gets it from
        :func:`solprop.translate.coupled.metric_at`.

    Returns
    -------
    dict
        The value of each requested statistic, keyed by name, always as a NumPy array
        and zero-dimensional for a scalar. The shape of each is the ``shape`` field of
        its :class:`~solprop.stats.StatDefinition`.

    Raises
    ------
    OutputError
        If a statistic's dependency is missing -- no windows for a windowed statistic,
        no modes for ``zdw``, or no profile for a profile statistic -- or if it is one
        a backend owns, which this function cannot form from a saved plane
        (``absorbed_power`` among them, refused by name because a plane does not
        carry the linear operator's rates); if a far-field statistic
        is asked of a radial grid, or of a line with an aperture off ``y = 0`` or a
        region excluding ``kappa_y = 0``; or if the plane kept no field (SEP 22).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.records import SavedPlane
    >>> from solprop.pulses import GaussianPulse
    >>> from solprop.stats import collect, default_spec
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> built = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6).build(grid)
    >>> plane = SavedPlane(z=0.0, field=built)
    >>> values = collect(default_spec("mode_averaged"), grid, plane)
    >>> sorted(values)
    ['energy', 'fwhm_t', 'fwhm_t_outer', 'omega0', 'peak_power']
    >>> values["energy"].shape, values["fwhm_t"].shape
    ((1, 1), ())
    """
    field = plane.field
    if field is None:
        msg = (
            "this plane kept no field (a run saved with fields='none' or 'planes'), "
            "so no statistic can be formed from it; read the statistics the run "
            "stored, output['stats/<name>']"
        )
        raise OutputError(msg)
    values: dict[str, np.ndarray] = {}
    # The far-field statistics share one pass over the columns (SEP 23).
    far_field = (
        _far_field_statistics(spec, flux, field)
        if flux is not None and _SAVED_FAR_FIELD & set(spec.names)
        else {}
    )
    for name in spec.names:
        entry = CATALOGUE[name]
        if flux is not None and not entry.profile:
            values[name] = (
                far_field[name]
                if name in far_field
                else _spatial_statistic(name, flux, field)
            )
        elif "spatial_geometry" in entry.requires:
            msg = f"the {name!r} statistic requires spatial geometry"
            raise OutputError(msg)
        elif entry.evaluate is not None and entry.metric:
            values[name] = entry.evaluate(grid, field, metric=metric)
        elif entry.evaluate is not None and "windows" in entry.requires:
            if spec.n_windows == 0:
                msg = f"the {name!r} statistic needs at least one energy window"
                raise OutputError(msg)
            values[name] = entry.evaluate(grid, field, spec.energy_windows)
        elif entry.evaluate is not None:
            values[name] = entry.evaluate(grid, field)
        elif entry.backend:
            msg = (
                f"the {name!r} statistic is computed by the backend from its own "
                "buffers, not from a saved plane; ask for it in the run's "
                "statistics (a StatsSpec)"
            )
            raise OutputError(msg)
        elif entry.profile:
            values[name] = profile_value(_profile_for(profiles, name), plane.z)
        else:
            # Only "zdw" is left: the spec has already checked every name against the
            # catalogue, and every other entry is routed above.
            if modes is None:
                msg = "the 'zdw' statistic needs the propagated modes"
                raise OutputError(msg)
            values[name] = _zdw_or_nan(modes, plane.z, grid)
    logger.debug("collected %d statistics at z = %.6e m", len(values), plane.z)
    return values


_POYNTING_STATISTICS = frozenset(
    name for name, entry in CATALOGUE.items() if entry.weight == "poynting"
)
"""The statistics a backend forms in its weighted Poynting pass."""

_GENERAL_FLUX_STATISTICS = frozenset(
    name for name, entry in CATALOGUE.items() if entry.weight == "unit"
)
"""The three general modal statistics, which are **not** in `_POYNTING_STATISTICS`.

They are evaluated by :func:`collect` from the saved field alone, because the general
modal route's ``linear.power_weight`` is the constant one: the
Poynting amplitude is the true-power amplitude there. A weight that is not one would
make their arithmetic silently wrong rather than obviously so -- none of them is in
the set a backend routes through :func:`_poynting_statistics` -- so
:func:`weight_refusal_for` guards them.
"""

WEIGHT_DEPENDENT_STATISTICS = _POYNTING_STATISTICS | _GENERAL_FLUX_STATISTICS
"""Names whose value depends on the modal power weight, so a backend evaluates it."""


def weight_refusal_for(
    names: Sequence[str], weight: Float[np.ndarray, "n_modes n_pol n_omega"]
) -> str | None:
    """Return why a general statistic cannot be formed under this weight, or ``None``.

    :func:`~solprop.stats.physical_energy`, :func:`~solprop.stats.spectral_flux` and
    :func:`~solprop.stats.analytic_flux` evaluate the identity-metric, unit-weight
    arithmetic, which is what the general modal route certifies before it will run
    (``linear.power_weight`` is the constant one there). This is **unreachable today**:
    no route carries both a modal table and a non-constant weight, and the translation
    layer refuses to build one. It is written because the failure would be silent rather
    than loud if a later change loosened either side -- the three names are outside the
    set a backend routes through the weighted Poynting pass, so they would be answered
    from the unweighted field.

    Parameters
    ----------
    names : collections.abc.Sequence
        The statistic names asked for.
    weight : numpy.ndarray
        The modal power weight at this plane, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    str or None
        The refusal, or ``None`` when no general statistic is asked for or the
        weight is the constant one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.stats import weight_refusal_for
    >>> weight_refusal_for(["physical_energy"], np.ones((1, 1, 4))) is None
    True
    >>> weight_refusal_for(["energy"], np.full((1, 1, 4), 2.0)) is None
    True
    """
    asked = [name for name in names if name in _GENERAL_FLUX_STATISTICS]
    if not asked or bool(np.all(weight == 1.0)):
        return None
    return (
        f"{sorted(asked)[0]} is unavailable for this run: it is the identity-metric, "
        "unit-power-weight quantity of a general modal basis, and this problem's "
        "linear.power_weight is not the constant one, so the arithmetic would be the "
        "unweighted field's and the answer would be silently wrong. A general modal "
        "guide's Poynting amplitude is its true-power amplitude; use energy or "
        "peak_power_total, which carry the weight"
    )


def _poynting_statistics(
    spec: StatsSpec,
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    weight: Float[np.ndarray, "n_modes n_pol n_omega"],
    axis: Float[np.ndarray, "2 n_modes"] | None = None,
) -> dict[str, np.ndarray]:
    """Share cycle-averaged analytic Poynting power and on-axis intensity.

    The real instantaneous sum-frequency term is excluded. Negative broadband
    excursions of the Hermitian flux are retained, without clipping.

    - ``axis``: physical on-axis synthesis matrix, shape (2, n_modes).
    - Returns: requested energy, temporal flux and axial intensity statistics.
    """
    names = set(spec.names) & _POYNTING_STATISTICS
    values: dict[str, np.ndarray] = {}
    density = np.abs(field) ** 2 * weight
    if "omega0" in names:
        total = np.sum(density, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values["omega0"] = np.sum(density * grid.omega, axis=-1) / total
    spectral = (
        density
        if grid.kind == "envelope"
        else density * np.concatenate(([1.0], np.full(grid.n_omega - 2, 4.0), [1.0]))
    )
    if "energy" in names:
        values["energy"] = (grid.dt / grid.n_t) * np.sum(spectral, axis=-1)
    if names & {"energy_window", "peak_power_window"} and spec.n_windows == 0:
        msg = "windowed Poynting statistics need at least one energy window"
        raise OutputError(msg)
    masks = _window_masks(grid, spec.energy_windows)
    if "energy_window" in names:
        values["energy_window"] = np.stack(
            [(grid.dt / grid.n_t) * np.sum(spectral * mask, axis=-1) for mask in masks]
        )
    if names - {"omega0", "energy", "energy_window", "peak_power_window"}:
        analytic = analytic_field(grid, field)
        weighted = analytic_field(grid, field * weight)
        power = np.sum(np.real(analytic * np.conj(weighted)), axis=1)
        total_power = np.sum(power, axis=0)
        if "peak_power" in names:
            values["peak_power"] = np.max(power, axis=-1)
        if "peak_power_total" in names:
            values["peak_power_total"] = np.asarray(np.max(total_power))
        for name in ("fwhm_t", "fwhm_t_outer"):
            if name in names:
                values[name] = _power_half_width(
                    grid, total_power, outermost=name.endswith("outer")
                )
        if "peak_intensity" in names:
            if axis is None:
                msg = "Poynting peak_intensity requires the on-axis synthesis matrix"
                raise OutputError(msg)
            electric = np.tensordot(axis, analytic, axes=([1], [0]))
            magnetic = np.tensordot(axis, weighted, axes=([1], [0]))
            flux = np.sum(np.real(electric * np.conj(magnetic)), axis=(0, 1))
            values["peak_intensity"] = np.asarray(0.5 * EPS0 * C * np.max(flux))
    if "peak_power_window" in names:
        values["peak_power_window"] = np.stack(
            [
                np.max(
                    np.sum(
                        np.real(
                            analytic_field(grid, field * mask)
                            * np.conj(analytic_field(grid, field * weight * mask))
                        ),
                        axis=1,
                    ),
                    axis=-1,
                )
                for mask in masks
            ]
        )
    return values


_SAVED_FAR_FIELD = frozenset(FAR_FIELD_STATISTICS) - frozenset(OPERATOR_STATISTICS)
"""The far-field statistics a saved Cartesian plane determines (SEP 23 section 5)."""


def _far_field_statistics(
    spec: StatsSpec,
    flux: SpatialFlux,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> dict[str, np.ndarray]:
    """Form every saved-field far-field statistic a spec asks for, in one pass.

    - ``field``: internal Cartesian spectrum in V/m, shape ``(n_x*n_y, n_pol,
      n_omega)``.
    - Returns: the requested statistics of :data:`_SAVED_FAR_FIELD`, keyed by name.
    """
    names = [name for name in spec.names if name in _SAVED_FAR_FIELD]
    space = flux.space
    if isinstance(space, PlanarGrid):
        msg = "the far-field statistics read a Cartesian or line grid, not a planar one"
        raise OutputError(msg)
    if isinstance(space, RadialGrid):
        msg = (
            f"the far-field statistics {FAR_FIELD_STATISTICS} need a Cartesian grid or "
            "a line, and a radial grid has no Cartesian transverse wavevector to map "
            "onto a mask plane"
        )
        raise OutputError(msg)
    tables = far_field_tables(
        spec,
        flux.grid.omega,
        k_x=space.k_x,
        k_y=space.k_y,
        weights=space.spectral_weights,
        on_axis=space.on_axis,
    )
    return aperture_statistics(
        names,
        flux.grid,
        field,
        tables,
        axial=lambda columns: (flux.admittance(columns), None),
    )


def _spatial_statistic(
    name: str,
    flux: SpatialFlux,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> np.ndarray:
    """Evaluate a spatial saved-field statistic through its physics reference.

    - ``field``: internal spectrum, shape (n_modes, n_pol, n_omega): a Hankel spectrum
      in V m or a raw Cartesian DFT spectrum in V/m.
    """
    if name == "energy":
        return np.asarray(spatial_energy(field, flux))
    if name == "on_axis_intensity":
        return np.asarray(on_axis_intensity(field, flux))
    if name == "on_axis_spectrum":
        return on_axis_spectrum(field, flux)
    if name == "peak_intensity":
        local = (
            planar_intensity(field, flux)
            if isinstance(flux.space, PlanarGrid)
            else spatial_intensity(field, flux)
        )
        return np.asarray(np.max(local))
    if name == "beam_radius":
        return np.asarray(
            [beam_radius(field, flux), beam_radius(field, flux, method="1/e2")]
        )
    if name == "fwhm_r":
        return np.asarray(fwhm_r(field, flux))
    if name in OPERATOR_STATISTICS:
        msg = (
            f"the {name!r} statistic needs the run's separable linear operator, which "
            "a saved plane does not carry; a backend forms it during the run, so read "
            f"the value the run stored, output['stats/{name}']"
        )
        raise OutputError(msg)
    msg = f"the {name!r} statistic is not a spatial saved-field statistic"
    raise OutputError(msg)


def default_spec(
    kind: SimulationKind, *, with_profiles: bool = False, with_zdw: bool = False
) -> StatsSpec:
    """Return the statistics a run of the given kind records unless told otherwise.

    The field statistics every run wants, plus -- for a multimode run, where the modes
    peak at different positions in time -- the peak of the total power. The z-profile
    statistics and the zero-dispersion wavelength are opt-in: the first needs profiles
    the driver may not have, and the second is a root search per mode per plane.

    Parameters
    ----------
    kind : {'mode_averaged', 'multimode', 'radial', 'free'}
        Which transform the run uses.
    with_profiles : bool, optional
        Whether to record the gas density, the pressure and the core radius for
        modal runs, or only the gas density for spatial runs.
    with_zdw : bool, optional
        Whether to record the zero-dispersion wavelength of each mode. Spatial
        runs reject this option because they carry no modal dispersion.

    Returns
    -------
    StatsSpec
        The spec, with no energy windows; add those for the run's bands of interest.

    Examples
    --------
    >>> from solprop.stats import default_spec
    >>> default_spec("mode_averaged").names
    ('omega0', 'energy', 'peak_power', 'fwhm_t', 'fwhm_t_outer')
    >>> "peak_power_total" in default_spec("multimode").names
    True
    """
    if kind in ("radial", "free"):
        if with_zdw:
            msg = "spatial statistics do not define a modal zdw"
            raise OutputError(msg)
        names = [
            "energy",
            "on_axis_intensity",
            "peak_intensity",
            "beam_radius",
            "fwhm_r",
            "on_axis_spectrum",
        ]
        if with_profiles:
            names.append("density")
        return StatsSpec(names=tuple(names))
    names = ["omega0", "energy", "peak_power"]
    if kind == "multimode":
        names.append("peak_power_total")
    names += ["fwhm_t", "fwhm_t_outer"]
    if with_profiles:
        names += list(PROFILE_STATISTICS)
    if with_zdw:
        names.append("zdw")
    return StatsSpec(names=tuple(names))
