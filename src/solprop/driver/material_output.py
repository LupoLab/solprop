# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record the material and physical reconstruction data needed by saved-run analysis.

Two records are built here. :func:`material_record` and :func:`processing_record`
describe a run's mixture for the configuration and for processing after reload.
:func:`material_manifest` collects a run's **material manifest** (SEP 3, output schema
8): every optmat-backed object the run carries, wherever it sits -- the mixture, a
mode's fill, a wall, a cladding, a core, a general guide's material regions, a bulk
medium -- found by walking the run's own objects, with the selection and the policies
in force.
"""

import dataclasses
import enum
import functools
import hashlib
import json
import math
import types
from collections.abc import Iterator, Mapping, Sequence
from typing import Final, cast

import numpy as np
from jaxtyping import Float

from solprop.constants import EPS0, C
from solprop.grid import TimeGrid
from solprop.modes.fill import GasFillMixin
from solprop.modes.material import GasKerr, MaterialRegion
from solprop.modes.protocol import Mode
from solprop.nonlinear.gnlse import GnlseProblem
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.optmat_rates import OptmatRate, PeakFieldReport
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import Mixture, _optmat
from solprop.physdata.glasses import GlassIndex, OptmatGlass
from solprop.physdata.mixtures import DILUTE_SCALING_THRESHOLD
from solprop.physdata.selection import (
    MaterialLog,
    MaterialManifest,
    MaterialRecord,
    MaterialSelection,
)
from solprop.physdata.sellmeier import GasSusceptibility, OptmatSusceptibility
from solprop.physdata.species import Species
from solprop.profiles import Constant
from solprop.translate.linop import power_weight_factors
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import DispersiveKerrTerms

__all__ = [
    "MIXTURE_FILTER_RULE",
    "material_manifest",
    "material_record",
    "processing_record",
]

_LEAVES = (
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    np.ndarray,
    np.generic,
    enum.Enum,
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
)
"""Values the manifest's walk never descends into. A function, a lambda or a closure
is one: what it captures is not walked, so a material reached only through a callable's
closure is not found."""

_STOP_PACKAGES: Final = frozenset(
    {
        "optmat",
        "numpy",
        "scipy",
        "h5py",
        "builtins",
        "logging",
        "threading",
        "jaxtyping",
        "CoolProp",
        "pyfluids",
        "numba",
    }
)
"""Packages whose objects the walk does not enter. optmat's own objects are recorded
through the solprop adapters that hold them, never walked; the rest hold no material."""


def material_record(mixture: Mixture) -> dict[str, object]:
    """Record explicit composition units, paths and complete supplied species data.

    Parameters
    ----------
    mixture : Mixture
        Ordered populations and temperature.

    Returns
    -------
    dict
        Constructor data or declared path samples, together with every species model.
    """
    return mixture.serialise() | {
        "species_data": tuple(component.species for component in mixture.components)
    }


def processing_record(
    problem: KernelProblem, grid: TimeGrid, mixture: Mixture, modes: Sequence[Mode]
) -> dict[str, object]:
    """Record compact data for species absorption and node fluence after reload.

    Parameters
    ----------
    problem : KernelProblem
        Translated geometry and save positions, with no backend state.
    grid : TimeGrid
        Time and angular-frequency axes and temporal padding scale.
    mixture : Mixture
        The same ordered material used by the operator and responses.
    modes : sequence of Mode
        Capillary modes, or an empty sequence for spatial propagation.

    Returns
    -------
    dict
        Density in m^-3 at requested planes, absorption cross sections in m^2 in
        the active band, and physical field synthesis with temporal padding removed.
        A replaced total loss is recorded as unavailable species attribution.
    """
    positions = np.concatenate(([0.0], problem.stepper.save_planes))
    transform = problem.transform
    geometry = transform.geometry
    omega = np.sort(grid.omega[grid.active_mask])
    cross_sections = _cross_sections(mixture, modes, omega)
    # The weight at every plane, factored: the dense stack is n_z state-sized rows of
    # JSON in a hashed record (26 MB for 201 planes of four modes on 2049 bins).
    terms, basis = (None, None)
    if geometry is None:
        terms, basis = power_weight_factors(problem, positions)
    return {
        "labels": mixture.labels,
        "z_m": positions,
        "density_m-3": mixture.density(positions).T,
        "omega_rad_s": omega,
        "cross_section_m2": cross_sections,
        "absorption_definition": "rho_s omega Im(gamma_s)/c, first order in chi",
        "absorption_unavailable": None
        if cross_sections is not None
        else "callable total loss has no species attribution",
        "node_synthesis": transform.synthesis / grid.pad_scale,
        "power_weight_terms": terms,
        "power_weight_basis": basis,
        "n_nodes": transform.n_nodes,
        "node_polarisations": transform.n_pol * problem.shape[2],
        "taper": np.ones(positions.size)
        if transform.taper_scale is None
        else np.asarray([float(transform.taper_scale(float(z))) for z in positions]),
        "column_energy_weights": np.ones(problem.shape[1])
        if geometry is None
        else EPS0 * C / 2 * geometry.spectral_weights,
    }


def _cross_sections(
    mixture: Mixture, modes: Sequence[Mode], omega: np.ndarray
) -> np.ndarray | None:
    """Factor the mode's first-order attribution into density-independent rows.

    Parameters
    ----------
    mixture : Mixture
        Ordered material populations.
    modes : sequence of Mode
        Capillary modes, or an empty sequence for bulk propagation.
    omega : numpy.ndarray
        Positive angular frequencies in rad/s within the active band.

    Returns
    -------
    numpy.ndarray or None
        Cross sections in m^2, shape (n_species, n_modes_or_one, n_omega), or None
        when the total loss policy supplies no species attribution.
    """
    if not modes:
        return np.asarray(
            [
                [
                    np.zeros_like(omega)
                    if (
                        (
                            isinstance(
                                component.species.susceptibility, GasSusceptibility
                            )
                            and not component.species.susceptibility.absorbing
                        )
                        or (
                            component.density.is_constant
                            and float(component.density(np.array([0.0]))[0]) == 0.0
                        )
                    )
                    else omega
                    * np.imag(component.species.susceptibility.susceptibility(omega))
                    / C
                ]
                for component in mixture.components
            ]
        )
    if any(not isinstance(mode, GasFillMixin) or callable(mode.loss) for mode in modes):
        return None
    # An absent population must not demand optical data outside its table support.
    unit_density = mixture.with_densities(
        {
            component.label: Constant(
                0.0
                if component.density.is_constant
                and float(component.density(np.array([0.0]))[0]) == 0.0
                else 1.0
            )
            for component in mixture.components
        }
    )
    return np.stack(
        [
            mode.with_mixture(unit_density).absorption_by_species(omega)
            for mode in modes
            if isinstance(mode, GasFillMixin)
        ],
        axis=1,
    )


def _children(value: object) -> list[object]:
    """Return what the manifest's walk visits next from one object.

    Parameters
    ----------
    value : object
        A dataclass, a ``functools.partial``, a bound method, or any object with a
        ``__dict__``.

    Returns
    -------
    list
        A dataclass's field values in declaration order; a partial's function,
        arguments and keywords; a bound method's instance; otherwise the instance
        attributes, none for an object that has none.
    """
    if isinstance(value, functools.partial):
        return [value.func, value.args, value.keywords]
    if isinstance(value, types.MethodType):
        return [value.__self__]
    if dataclasses.is_dataclass(value):
        return [getattr(value, one.name, None) for one in dataclasses.fields(value)]
    return list(getattr(value, "__dict__", {}).values())


def _walk(roots: Sequence[object]) -> Iterator[object]:
    """Yield every object reachable from the roots, once each.

    The walk descends into mappings, sequences, dataclasses, partials, bound methods
    and any object with a ``__dict__`` -- solprop's own, a caller's subclass, a
    user's mode -- but never into an array, a number, a string, a function or an
    object of a package in :data:`_STOP_PACKAGES`. Each object is visited once, so a
    cycle ends, and the order is fixed by the objects themselves.

    Parameters
    ----------
    roots : sequence
        The run's objects: its configuration, modes, responses, media and grids.

    Yields
    ------
    object
        Each reachable object outside the stopped packages, in depth-first order.
    """
    seen: set[int] = set()
    stack: list[object] = list(reversed(roots))
    while stack:
        value = stack.pop()
        if value is None or isinstance(value, _LEAVES) or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Mapping):
            stack.extend(reversed(list(value.values())))
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            items = (
                sorted(value, key=repr)
                if isinstance(value, (set, frozenset))
                else value
            )
            stack.extend(reversed(list(items)))
            continue
        if (
            not isinstance(value, (functools.partial, types.MethodType))
            and type(value).__module__.split(".")[0] in _STOP_PACKAGES
        ):
            continue
        yield value
        stack.extend(reversed(_children(value)))


def _single_species_fills(
    found: Sequence[object],
) -> list[tuple[Species, GasFillMixin]]:
    """Return each distinct single-species fill among the run's modes.

    Parameters
    ----------
    found : sequence
        The objects the walk reached.

    Returns
    -------
    list of tuple
        ``(species, mode)`` for each gas fill without a mixture whose species
        carries optmat records, one per distinct species, density path and
        temperature: the modes of one guide share their fill.
    """
    fills: list[tuple[Species, GasFillMixin]] = []
    seen: list[tuple[Species, str, object]] = []
    for value in found:
        if not isinstance(value, GasFillMixin):
            continue
        material = value.fill_species
        if material is None or not material.material_records:
            continue
        key = (material, repr(value.density), value.temperature)
        if key in seen:
            continue
        seen.append(key)
        fills.append((material, value))
    return fills


def _records(value: object) -> tuple[MaterialRecord, ...]:
    """Return an object's optmat records, or none for one that carries none.

    Parameters
    ----------
    value : object
        Any of solprop's objects.

    Returns
    -------
    tuple of MaterialRecord
        Its ``material_records``, where it has them.
    """
    found = getattr(value, "material_records", ())
    return tuple(found) if isinstance(found, tuple) else ()


def _solid_entry(glass: OptmatGlass, log: MaterialLog | None) -> dict[str, object]:
    """Return an optmat solid's manifest entry.

    Parameters
    ----------
    glass : OptmatGlass
        The adapter.
    log : MaterialLog or None
        The run's evaluation log, whose dropped loss is the run's own; ``None``
        takes what the adapter has dropped over its life.

    Returns
    -------
    dict
        Its material, axis, model, digest, transparency policy and measurement.
    """
    entry = glass.material_records[0]
    return {
        "material": glass.name,
        "axis": glass.axis,
        "model": entry.name,
        "digest": entry.digest,
        "transparent": glass.transparent,
        "absorption_bound": glass.absorption_bound if glass.transparent else None,
        "transparency": dict(glass.run_transparency(log)),
    }


def _citation_entry(value: object) -> dict[str, object] | None:
    """Return the citation of a material object that is not optmat's, if any.

    Parameters
    ----------
    value : object
        Any of solprop's objects.

    Returns
    -------
    dict or None
        ``object``, ``name`` and ``citation`` for a glass, a gas susceptibility or
        a species that carries no optmat record; ``None`` otherwise.
    """
    if not isinstance(value, GlassIndex | GasSusceptibility | Species) or _records(
        value
    ):
        return None
    return {
        "object": type(value).__name__,
        "name": str(getattr(value, "name", "") or getattr(value, "source", "")),
        "citation": str(getattr(value, "citation", "")),
    }


def _window(omega: Float[np.ndarray, " n_omega"]) -> float | None:
    """Return the time window a frequency grid resolves.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies in rad/s, uniformly spaced in some order.

    Returns
    -------
    float or None
        ``2 pi / d_omega`` in s, or ``None`` for fewer than two frequencies.
    """
    spacing = np.diff(np.unique(np.asarray(omega, dtype=float)))
    return None if spacing.size == 0 else float(2.0 * np.pi / np.min(spacing))


def _raman_entry(
    response: OptmatRamanResponse, omega: Float[np.ndarray, " n_omega"]
) -> dict[str, object]:
    """Return what was measured about an optmat Raman response's assembly.

    Parameters
    ----------
    response : OptmatRamanResponse
        The response.
    omega : numpy.ndarray
        The run's angular frequencies in rad/s, whose spacing fixes the window over
        which a held line shift's phase error is bounded.

    Returns
    -------
    dict
        The assembly's record with the labels, and the shift's phase bound in rad
        over the run's window (``None`` without a held shift).
    """
    assembly = response.assembly
    entry: dict[str, object] = {"labels": list(response.labels)}
    if assembly is None:
        return entry
    window = _window(omega)
    shift = assembly.shift
    entry |= assembly.to_dict()
    entry["window_s"] = window
    entry["shift_phase_bound_rad"] = (
        None if shift is None or window is None else shift.phase_bound(window)
    )
    return entry


def _gnlse_policy(problems: Sequence[GnlseProblem]) -> dict[str, object] | None:
    """Return a GNLSE run's held-back Raman fraction and where it came from.

    Parameters
    ----------
    problems : sequence of GnlseProblem
        The run's GNLSE mappings: none, or one.

    Returns
    -------
    dict or None
        ``fraction``, its ``origin`` (``"user"``, ``"bundle"`` or
        ``"raman-off"``), whether a Raman response carries it, the silica
        ``bundle`` (or ``None``) and the origin of ``n2``; ``None`` for a run
        without a GNLSE mapping.
    """
    if not problems:
        return None
    problem = problems[0]
    return {
        "fraction": problem.raman_fraction,
        "origin": problem.fraction_origin,
        "raman": problem.raman is not None,
        "bundle": None if problem.bundle is None else problem.bundle.name,
        "n2_origin": problem.n2_origin,
        "carrier_wavelength": 2.0 * math.pi * C / problem.omega0,
    }


def _policies(
    selection: MaterialSelection | None,
    *,
    dilute: bool,
    carrier: float | None,
    gnlse: dict[str, object] | None,
    raman: Sequence[OptmatRamanResponse] = (),
    dispersive: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the approximation policies a run's materials are evaluated under.

    Parameters
    ----------
    selection : MaterialSelection or None
        The run's selection.
    dilute : bool
        Whether a gas susceptibility from optmat is scaled linearly in density.
    carrier : float or None
        The carrier in m an optmat Kerr coefficient was evaluated at, or ``None``
        when the run has no optmat Kerr model.
    gnlse : dict or None
        A GNLSE run's held-back fraction (:func:`_gnlse_policy`).
    raman : sequence of OptmatRamanResponse, optional
        The run's optmat Raman responses.
    dispersive : dict or None, optional
        The dispersive electronic law in force (:func:`_dispersive_policy`), which
        replaces the carrier law; ``None`` when the run has none.

    Returns
    -------
    dict
        The policies; one a later phase resolves is recorded as selected.
    """
    assemblies = [one.assembly for one in raman if one.assembly is not None]
    temperatures = sorted({assembly.temperature for assembly in assemblies})
    return {
        "linear_density_scaling": {
            "law": "N alpha / eps0",
            "warning_threshold": DILUTE_SCALING_THRESHOLD,
        }
        if dilute
        else None,
        "electronic": None if selection is None else selection.electronic,
        "carrier_wavelength": None
        if selection is None
        else selection.carrier_wavelength,
        # SEP 3, section 4.3: one frozen coefficient per species, at the carrier,
        # for every process the cubic term drives.
        "electronic_response": dispersive
        if dispersive is not None
        else None
        if carrier is None
        else {"law": "carrier-kerr", "carrier_wavelength": carrier},
        "gnlse_raman": gnlse,
        "raman_shift": None if selection is None else selection.raman_shift,
        # The temperature an optmat Raman response holds its populations and its
        # widths' temperature laws at (SEP 3, section 4.4): one per response.
        "raman_reference_temperature": None
        if not temperatures
        else temperatures[0]
        if len(temperatures) == 1
        else temperatures,
        "raman_terms": [
            {
                "material": assembly.material,
                "labels": list(one.labels),
                "groups": assembly.groups,
                "terms": assembly.terms,
                "shift": None if assembly.shift is None else assembly.shift.policy,
            }
            for one in raman
            if (assembly := one.assembly) is not None
        ],
        "energy_loss": None if selection is None else selection.energy_loss,
    }


def _dispersive_label(response: DispersiveKerrResponse) -> str:
    """Return the population a dispersive response is recorded under.

    Parameters
    ----------
    response : DispersiveKerrResponse
        The response.

    Returns
    -------
    str
        Its component label, or its gas for a single-gas run.
    """
    return response.model.material if response.species is None else response.species


def _dispersive_entries(
    responses: Sequence[DispersiveKerrResponse],
    band: tuple[float, float] | None,
    *,
    vector: bool,
) -> list[dict[str, object]]:
    """Return the manifest's electronic entries of a run's dispersive responses.

    Parameters
    ----------
    responses : sequence of DispersiveKerrResponse
        The run's dispersive responses.
    band : tuple of float or None
        The run's active band in rad/s, or ``None`` when it is not known.
    vector : bool
        Whether the run's field has two polarisation components at a node.

    Returns
    -------
    list of dict
        Per response: its label, gas, evaluator, sector and tensor approximation, the
        model and support digests, the leg bound and both band edges in m^-1, and the
        regions of the band (:meth:`ElectronicRegions.to_dict`), in label order.
    """
    entries = []
    for response in responses:
        model = response.model
        regions = None if band is None else model.regions(*band, vector=vector)
        entries.append(
            {
                "label": _dispersive_label(response),
                "material": model.material,
                "model": model.material_records[0].name,
                "digest": model.model_digest,
                "support_digest": model.support_digest,
                "evaluator": response.evaluator,
                "sector": response.sector,
                "tensor": "isotropic-kleinman",
                "leg_bound": model.leg_bound,
                "band_edge_linear": model.band_edge_linear,
                "band_edge_any": model.band_edge_any,
                "regions": None if regions is None else regions.to_dict(),
            }
        )
    return sorted(entries, key=lambda entry: str(entry["label"]))


MIXTURE_FILTER_RULE: Final = (
    "g_mix = [sum_s rho_s gamma0_s g_s^4 / sum_s rho_s gamma0_s]^(1/4), "
    "rebuilt from the densities at every stage"
)
"""The symmetric evaluator's filter of a mixture, as a run records it (SEP 6, 6)."""


def _density_identities(found: Sequence[object]) -> dict[str, str]:
    """Return the identity of each population's density profile, by channel label.

    Parameters
    ----------
    found : sequence of object
        The run's objects: its mixtures, and a general modal guide's regions, whose
        gas fills' channels are labelled ``"<region>:<population>"``.

    Returns
    -------
    dict
        Per label, the SHA-256 of the canonical JSON of the profile's record -- its
        samples, breakpoints and declarations, as the run's configuration records a
        mixture (:meth:`solprop.physdata.mixtures.Mixture.serialise`).
    """
    identities: dict[str, str] = {}

    def add(prefix: str, mixture: Mixture) -> None:
        """Record a mixture's populations under a label prefix.

        Parameters
        ----------
        prefix : str
            Prepended to every population label.
        mixture : Mixture
            The mixture.
        """
        record = cast("list[dict[str, object]]", mixture.serialise()["components"])
        for component in record:
            identities.setdefault(
                f"{prefix}{component['label']}",
                hashlib.sha256(
                    _canonical_entry(
                        cast("Mapping[str, object]", component["density"])
                    ).encode()
                ).hexdigest(),
            )

    for value in found:
        if isinstance(value, Mixture):
            add("", value)
        elif isinstance(value, MaterialRegion) and isinstance(value.gas, GasKerr):
            add(f"{value.name}:", value.gas.fill)
    return identities


def _mixture_filters(
    leaf: DispersiveKerrTerms | None, densities: Mapping[str, str]
) -> list[dict[str, object]]:
    """Return the symmetric mixture filters a run's leaf builds, for its record.

    Parameters
    ----------
    leaf : solprop.translate.responses.DispersiveKerrTerms or None
        The run's compiled dispersive channels.
    densities : Mapping of str to str
        Each label's density-profile identity (:func:`_density_identities`).

    Returns
    -------
    list of dict
        One per group of several species under the symmetric evaluator
        (:meth:`~solprop.translate.responses.DispersiveKerrTerms.groups`): the rule,
        the general modal region it acts in (``None`` on every other route), and the
        composition it is built from -- each species' label and its density profile's
        identity. Empty for the polynomial and for a pure gas.
    """
    if leaf is None or leaf.evaluator != "symmetric":
        return []
    filters: list[dict[str, object]] = []
    for group in leaf.groups():
        if len(group) < 2:
            continue
        filters.append(
            {
                "rule": MIXTURE_FILTER_RULE,
                "region": None if leaf.regions is None else int(leaf.regions[group[0]]),
                "composition": [
                    {
                        "label": leaf.labels[channel],
                        "density": densities.get(leaf.labels[channel]),
                    }
                    for channel in group
                ],
            }
        )
    return filters


def _dispersive_policy(
    entries: Sequence[Mapping[str, object]],
    mixtures: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return the dispersive electronic law a run's entries record, for its key.

    Parameters
    ----------
    entries : sequence of Mapping
        :func:`_dispersive_entries` of the run.
    mixtures : sequence of Mapping, optional
        The symmetric evaluator's mixture filters (:func:`_mixture_filters`), written
        under ``mixture_filters`` only when there is one, so that a pure gas's record
        is what it was.

    Returns
    -------
    dict
        The law, its evaluator, sector and tensor approximation, the per-leg bound
        rule, each population's support digest and regions, and each mixture filter
        with the composition it is built from: what the resume key covers, so that a
        changed model, support record, band or composition is refused.
    """
    evaluator = str(entries[0]["evaluator"])
    policy: dict[str, object] = {
        "law": "symmetric-kerr" if evaluator == "symmetric" else "gas-polynomial",
        "evaluator": evaluator,
        "sector": entries[0]["sector"],
        "tensor": "isotropic-kleinman",
        "leg_bound": "per-leg at sqrt(3 W2_max / 4)",
        "species": [
            {
                "label": entry["label"],
                "support_digest": entry["support_digest"],
                "regions": entry["regions"],
            }
            for entry in entries
        ],
    }
    if mixtures:
        policy["mixture_filters"] = [dict(one) for one in mixtures]
    return policy


def _rate_policies(found: Sequence[object]) -> list[dict[str, object]]:
    """Return the policies of a run's optmat ionisation rates.

    Parameters
    ----------
    found : sequence of object
        The run's objects.

    Returns
    -------
    list of dict
        :meth:`solprop.nonlinear.optmat_rates.OptmatRate.manifest_entry` of each
        rate, each once, in canonical order: the threshold kind, the reference field,
        the constant energy per event and its bound, the maximum field.
    """
    entries = [
        value.manifest_entry() for value in found if isinstance(value, OptmatRate)
    ]
    distinct = {_canonical_entry(entry): entry for entry in entries}
    return [distinct[key] for key in sorted(distinct)]


def _canonical_entry(entry: Mapping[str, object]) -> str:
    """Return an entry's canonical JSON, for ordering and deduplication.

    Parameters
    ----------
    entry : collections.abc.Mapping
        Plain values.

    Returns
    -------
    str
        Sorted-key, compact JSON.
    """
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


def material_manifest(  # noqa: PLR0913 - the run's objects and its band, one each
    roots: Sequence[object],
    *,
    selection: MaterialSelection | None,
    omega: Float[np.ndarray, " n_omega"],
    log: MaterialLog | None = None,
    peak_fields: Sequence[PeakFieldReport] = (),
    carrier: float | None = None,
    active_band: tuple[float, float] | None = None,
    vector: bool = False,
    leaf: DispersiveKerrTerms | None = None,
) -> MaterialManifest:
    """Collect a run's material manifest from its objects.

    Every one of solprop's objects reachable from `roots` is visited once. Those
    that carry optmat records (a species, a mixture, a gas susceptibility or
    equation of state, a solid) contribute them; a mixture contributes its
    optmat populations, its named composition and its dilute-scaling check over
    `omega`, measured here once for the run; an optmat solid its axis and its
    transparent approximation; a species with optmat's Kerr model its coefficient
    at `carrier`, beside the third-harmonic one; a GNLSE mapping its held-back
    Raman fraction and its origin; and any other glass, gas susceptibility or
    species its citation. A run with no optmat object and no selection has the
    empty manifest, whatever else it carries.

    Parameters
    ----------
    roots : sequence
        The run's objects: configuration, modes, responses, media, guide.
    selection : MaterialSelection or None
        The selection the run was built with.
    omega : numpy.ndarray
        Angular frequencies in rad/s of the run's band, collars included; the
        positive ones are checked.
    log : solprop.physdata.selection.MaterialLog or None, optional
        The run's evaluation log, whose out-of-interval bands are recorded.
    peak_fields : sequence of PeakFieldReport, optional
        The post-run diagnostic of each optmat rate: the largest field each
        population saw, against the model's maximum field and the constant energy
        (:func:`solprop.nonlinear.optmat_rates.peak_field_report`); a measurement.
    carrier : float or None, optional
        The run's carrier, its grid's reference wavelength in m, at which an
        optmat Kerr coefficient is recorded; ``None`` records none.
    active_band : tuple of float or None, optional
        The run's active band in rad/s, ``omega_min / 2`` to ``1.1 omega_max``, over
        which a dispersive electronic response's regions are classified (SEP 6,
        section 4.10); ``None`` records no regions.
    vector : bool, optional
        Whether the run's field has two polarisation components at a node, which
        takes the any-polarisation band edge.
    leaf : solprop.translate.responses.DispersiveKerrTerms or None, optional
        The run's compiled dispersive channels, whose groups name the symmetric
        evaluator's mixture filters: each is recorded, with the composition it is built
        from, in the electronic law's policy (and so in the resume key), and its
        species' electronic entries name it. ``None`` records none.

    Returns
    -------
    solprop.physdata.selection.MaterialManifest
        The manifest, deduplicated and in canonical order.
    """
    found = list(_walk(roots))
    records = [entry for value in found for entry in _records(value)]
    if not records and selection is None:
        return MaterialManifest()
    mixtures = [value for value in found if isinstance(value, Mixture)]
    band = np.asarray(omega, dtype=float)
    # A fill without a mixture (a planar run's slab) carries its gas as one species,
    # whose records and dilute-scaling check are recorded as a population's are; the
    # modes of one guide share their fill, so it is recorded once (SEP 3, Q27).
    fills = _single_species_fills(found)
    species = [
        {
            "label": component.label,
            "species": component.species.name,
            "records": [entry.digest for entry in component.species.material_records],
        }
        for mixture in mixtures
        for component in mixture.components
        if component.species.material_records
    ] + [
        {
            "label": material.name,
            "species": material.name,
            "records": [entry.digest for entry in material.material_records],
        }
        for material, _ in fills
        if material.material_records
    ]
    compositions = [
        {
            "name": mixture.composition,
            "labels": list(mixture.labels),
            "fractions": list(mixture.fractions or ()),
        }
        for mixture in mixtures
        if mixture.composition is not None
    ]
    checks = [
        check.to_dict()
        for mixture in mixtures
        for check in mixture.dilute_scaling_checks(band[band > 0])
    ] + [
        check.to_dict()
        for _, mode in fills
        for check in mode._dilute_checks(band[band > 0])
    ]
    citations = [entry for value in found if (entry := _citation_entry(value))]
    dispersive = _dispersive_entries(
        [value for value in found if isinstance(value, DispersiveKerrResponse)],
        active_band,
        vector=vector,
    )
    mixture_filters = _mixture_filters(leaf, _density_identities(found))
    # Each population of a symmetric mixture names the filter it belongs to.
    membership = {
        str(member["label"]): {
            "kind": "mixture",
            "labels": [one["label"] for one in composition],
        }
        for mixture_filter in mixture_filters
        for composition in [
            cast("list[dict[str, object]]", mixture_filter["composition"])
        ]
        for member in composition
    }
    dispersive = [
        entry | ({"filter": membership[label]} if label in membership else {})
        for entry in dispersive
        for label in [str(entry["label"])]
    ]
    # Under a dispersive law no carrier coefficient is used, so none is recorded.
    electronic = dispersive or [
        {"label": component.label}
        | component.species.kerr.at_carrier(carrier).to_dict()
        for mixture in mixtures
        for component in mixture.components
        if component.species.kerr is not None and carrier is not None
    ]
    raman = [value for value in found if isinstance(value, OptmatRamanResponse)]
    return MaterialManifest(
        optmat_version=_optmat.optmat_version(),
        selection=None if selection is None else selection.to_dict(),
        records=tuple(records),
        species=tuple(species),
        compositions=tuple(compositions),
        solids=tuple(
            _solid_entry(value, log)
            for value in found
            if isinstance(value, OptmatGlass)
        ),
        policies={
            **_policies(
                selection,
                dilute=any(isinstance(value, OptmatSusceptibility) for value in found),
                carrier=carrier if electronic and not dispersive else None,
                gnlse=_gnlse_policy(
                    [value for value in found if isinstance(value, GnlseProblem)]
                ),
                raman=raman,
                dispersive=_dispersive_policy(dispersive, mixture_filters)
                if dispersive
                else None,
            ),
            "ionisation": _rate_policies(found),
        },
        raman=tuple(_raman_entry(one, band) for one in raman),
        dilute_scaling=tuple(checks),
        ranges=() if log is None else log.range_entries(),
        citations=tuple(citations),
        electronic=tuple(electronic),
        peak_fields=tuple(report.to_dict() for report in peak_fields),
    )
