# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read population inventories and reconstruct physical deposition from saved planes."""

from collections.abc import Mapping
from typing import cast

import numpy as np
from jaxtyping import Float

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.exceptions import ProcessingError
from solprop.grid import TimeGrid
from solprop.output.schema import Output
from solprop.processing.axes import grid_of
from solprop.processing.fields import get_spectral_density
from solprop.translate.flux import flux_from_record
from solprop.translate.spatial import axial_admittance

__all__ = [
    "absorbed_spectral_energy",
    "get_power_weight",
    "node_fluence",
    "species_ionisation",
    "species_ionisation_energy",
]


def species_ionisation(output: Output) -> Float[np.ndarray, "n_z n_species n_nodes"]:
    """Return final ionised fractions at every saved plane and transverse node.

    Component order is the metadata's ``medium.labels`` order, including components
    without a plasma channel. Fractions include the initial ionised inventory;
    subtract the declared initial fraction to obtain newly ionised fractions.

    Parameters
    ----------
    output : Output
        Opened memory or file output with the species_ionisation statistic.

    Returns
    -------
    numpy.ndarray
        Dimensionless final fractions in [0, 1], shape (n_z, n_species, n_nodes).

    Raises
    ------
    ProcessingError
        If the run did not record this statistic.
    """
    return _statistic(output, "species_ionisation")


def species_ionisation_energy(output: Output) -> Float[np.ndarray, "n_z n_species"]:
    """Return ionisation energy stored in the final inventory, per propagation length.

    The recorded definition is ``I_p,s sum_node w_node rho_s f_s`` in J/m, with
    transverse weights in m^2. It includes preionised particles. Pulse work instead
    uses ``f_s - f0_s``; this inventory is not a longitudinally integrated energy.

    Parameters
    ----------
    output : Output
        Opened memory or file output with the species_ionisation_energy statistic.

    Returns
    -------
    numpy.ndarray
        J/m at each plane, shape (n_z, n_species), in mixture order.

    Raises
    ------
    ProcessingError
        If the run did not record this statistic.
    """
    return _statistic(output, "species_ionisation_energy")


def absorbed_spectral_energy(
    output: Output, *, species: str | None = None
) -> tuple[Float[np.ndarray, " n_omega"], Float[np.ndarray, "n_z n_species n_omega"]]:
    """Return species-resolved linear absorption per metre and angular frequency.

    Multiply each mode's saved spectral energy density by its first-order material
    attenuation ``rho_s omega Im(gamma_s)/c`` and sum over modes. Spatial runs use
    their physical spectral quadrature. The active band is returned in ascending
    angular frequency. Wall leakage, plasma work and nonlinear absorption are not
    assigned to species by this definition. A numeric total-loss control scales the
    attribution; a callable replacement has no species attribution and is refused.

    Parameters
    ----------
    output : Output
        Opened mixture output with reconstruction data.
    species : str or None, optional
        One component label, or None for all labels in mixture order. A selected
        component retains a species axis of length one.

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency in rad/s, shape (n_omega,).
    absorbed : numpy.ndarray
        Energy loss density in J/(m rad/s), shape (n_z, n_species, n_omega).

    Raises
    ------
    ProcessingError
        If the label, saved-plane data or species attribution is unavailable.
    """
    record = _record(output)
    if record["cross_section_m2"] is None:
        raise ProcessingError(str(record["absorption_unavailable"]))
    labels = tuple(cast("list[str]", record["labels"]))
    if species is not None and species not in labels:
        raise ProcessingError(f"unknown species label {species!r}; available: {labels}")
    selected = list(range(len(labels))) if species is None else [labels.index(species)]
    omega, density = get_spectral_density(output, axis="omega", band="active")
    n_modes = output.fields().shape[1]
    modal = density.reshape(output.n_saved, n_modes, omega.size)
    weights = _array(record, "column_energy_weights")
    cross_sections = _array(record, "cross_section_m2")[selected]
    positions = _plane_indices(output, record)
    populations = _array(record, "density_m-3")[positions][:, selected]
    # On a spatial column the physical flux removed is 2 Im(k_z) times the axial
    # flux, (eps0 c / 2)(c Re k_z / omega)|E|^2, and 2 Im(k_z) Re(k_z) = Im(k**2):
    # the admittance cancels exactly, at every angle, leaving this cross section
    # times eps0 c |E|^2 / 2 (SEP 34).
    absorbed = np.einsum("smw,zmw,m->zsw", cross_sections, modal, weights)
    return omega, absorbed * populations[:, :, None]


def get_power_weight(output: Output) -> Float[np.ndarray, "n_z n_modes n_pol n_omega"]:
    """Return the modal power weight ``Re n_eff`` at every saved plane.

    The weight that relates the saved true-power amplitude ``b`` to the internal
    vacuum-normalised amplitude ``a`` of ``docs/guide/results/power-and-intensity.md``:
    ``b = sqrt(w) a``, with ``w = Re n_eff(omega, z)`` and ``w = 1`` on the GNLSE
    route. Divide a saved plane by its ``sqrt(w)`` before applying a
    vacuum-normalised synthesis matrix or an ionisation rate. The run record holds
    the weight factored -- a few frequency-resolved terms and one small basis row
    per plane -- and the product rebuilt here is bit for bit the translation's own
    evaluation at that plane.

    Parameters
    ----------
    output : solprop.output.schema.Output
        Opened modal output whose record carries the power-weight factors.

    Returns
    -------
    numpy.ndarray
        Dimensionless positive weights, shape (n_z, n_modes, n_pol, n_omega), in
        saved-plane order.

    Raises
    ------
    ProcessingError
        If the run recorded no power weight -- a radial, Cartesian or free-space
        run, which propagates a physical field and has no modal weight -- or if its
        saved planes are not recorded positions.
    """
    record = _record(output)
    if record.get("power_weight_terms") is None:
        raise ProcessingError(
            "the run recorded no modal power weight: only a modal run carries one"
        )
    terms = _array(record, "power_weight_terms")
    basis = _array(record, "power_weight_basis")[_plane_indices(output, record)]
    return np.stack([np.tensordot(row, terms, axes=1) for row in basis])


def node_fluence(output: Output) -> Float[np.ndarray, "n_z n_nodes"]:
    """Reconstruct the pulse fluence at every transverse quadrature node.

    Apply the saved synthesis matrix and taper to the electric and magnetic-field
    equivalents, then integrate analytic Poynting flux in J/m^2. A spatial run's
    magnetic field is the closure of the medium at each plane (SEP 34). The
    temporal sum uses
    the grid's fine spacing and the complete saved band, with no added window. A
    mode-averaged run has one effective node and returns energy divided by A_eff.
    Radial and Cartesian runs reconstruct the physical electric field before the sum.

    Parameters
    ----------
    output : Output
        Opened mixture output with physical reconstruction data.

    Returns
    -------
    numpy.ndarray
        Fluence in J/m^2, shape (n_z, n_nodes), in the statistics' node order.

    Raises
    ------
    ProcessingError
        If physical reconstruction data or recorded save positions are absent.
    """
    record = _record(output)
    metadata = output.metadata
    assert metadata is not None
    grid = grid_of(output)
    if metadata.simulation_kind in ("multimode", "mode_averaged"):
        return _modal_fluence(output, record, grid)
    return _spatial_fluence(output, record, grid)


def _spatial_fluence(
    output: Output, record: Mapping[str, object], grid: TimeGrid
) -> np.ndarray:
    """Integrate the signed axial flux of a radial, Cartesian or planar run.

    Each plane's field is paired with its magnetic closure, the conjugate of the
    plane's own admittance applied to the stored spectrum (SEP 34), rebuilt from the
    run's recorded plane measure at the plane's position.

    Parameters
    ----------
    output : solprop.output.schema.Output
        Saved spatial planes, whose record carries the plane measure.
    record : collections.abc.Mapping
        Saved synthesis and taper.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.

    Returns
    -------
    numpy.ndarray
        Fluence in J/m^2, shape (n_z, n_nodes).

    Raises
    ------
    ProcessingError
        If the record carries no plane measure.
    """
    metadata = output.metadata
    assert metadata is not None
    spatial = metadata.spatial
    if spatial is None or "flux_plan" not in spatial:
        msg = (
            "this spatial output carries no plane measure (flux_plan, SEP 34) and "
            "cannot be measured as an axial flux"
        )
        raise ProcessingError(msg)
    plan = flux_from_record(cast("Mapping[str, object]", spatial["flux_plan"]))
    n_nodes = cast("int", record["n_nodes"])
    taper = _array(record, "taper")[_plane_indices(output, record)]
    fields = output.fields("fft")
    result = []
    for field, z in zip(fields, np.asarray(output["z"]), strict=True):
        closure = (
            np.conj(axial_admittance(plan.at(float(z)), grid.omega, slice(None)))[
                :, None
            ]
            * field
        )
        pair = []
        for spectrum in (field, closure):
            fine = fourier.oversample(spectrum, grid.n_to, kind=grid.kind)
            samples = fourier.inverse(fine, kind=grid.kind)
            analytic = (
                fourier.analytic_signal(samples) if grid.kind == "field" else samples
            )
            if metadata.simulation_kind == "free":
                assert metadata.spatial is not None
                nx, ny = cast("tuple[int, int]", metadata.spatial["shape"])
                physical = np.fft.ifft2(
                    analytic.reshape(ny, nx, *analytic.shape[1:]), axes=(0, 1)
                ).reshape(n_nodes, -1, analytic.shape[-1])
            else:
                physical = np.einsum(
                    "rm,mpt->rpt", _array(record, "node_synthesis"), analytic
                ).reshape(n_nodes, -1, analytic.shape[-1])
            pair.append(physical)
        result.append(
            EPS0
            * C
            / 2
            * grid.dt_fine
            * np.sum(np.real(pair[0] * np.conj(pair[1])), axis=(-2, -1))
        )
    return np.asarray(result) * taper[:, None] ** 2


def _modal_fluence(
    output: Output, record: Mapping[str, object], grid: TimeGrid
) -> np.ndarray:
    """Reconstruct physical electric and magnetic fields from true-power spectra.

    Parameters
    ----------
    output : solprop.output.schema.Output
        Saved planes in true-power units with reconstruction metadata.
    record : collections.abc.Mapping
        Saved per-plane power weights, physical synthesis and taper.
    grid : solprop.grid.TimeGrid
        Temporal grid defining frequency and Parseval weights.

    Returns
    -------
    numpy.ndarray
        Integrated Poynting flux, shape (n_z, n_nodes), in J/m^2.
    """
    weights = get_power_weight(output)
    indices = _plane_indices(output, record)
    fields = output.fields("fft")
    synthesis = _array(record, "node_synthesis")
    result = []
    for field, weight in zip(fields, weights, strict=True):
        physical = []
        for inverse in (True, False):
            spectrum = fourier.power_amplitude(field, weight, inverse=inverse)
            fine = fourier.oversample(spectrum, grid.n_to, kind=grid.kind)
            samples = fourier.inverse(fine, kind=grid.kind)
            analytic = (
                fourier.analytic_signal(samples) if grid.kind == "field" else samples
            )
            values = np.einsum("rm,mpt->rpt", synthesis, analytic)
            physical.append(
                values.reshape(cast("int", record["n_nodes"]), -1, grid.n_to)
            )
        result.append(
            EPS0
            * C
            / 2
            * grid.dt_fine
            * np.sum(np.real(physical[0] * np.conj(physical[1])), axis=(-2, -1))
        )
    taper = _array(record, "taper")[indices]
    return np.asarray(result) * taper[:, None] ** 2


def _statistic(output: Output, name: str) -> np.ndarray:
    """Require an explicitly recorded statistic, keeping every population axis.

    Parameters
    ----------
    output : Output
        Opened saved run.
    name : str
        Statistic name whose units and dimensions follow the catalogue.

    Returns
    -------
    numpy.ndarray
        Recorded values, including all population and node axes.
    """
    if name not in output.stat_names:
        raise ProcessingError(f"the run did not record {name}")
    return np.asarray(output[f"stats/{name}"], dtype=float)


def _record(output: Output) -> Mapping[str, object]:
    """Return the run's mixture reconstruction data, refusing a run without it.

    Parameters
    ----------
    output : Output
        Opened saved run.

    Returns
    -------
    Mapping
        Material processing data in the units the record declares.
    """
    metadata = output.metadata
    record = (
        None if metadata is None else metadata.configuration.get("mixture_processing")
    )
    if not isinstance(record, Mapping):
        raise ProcessingError("the run has no mixture reconstruction data")
    return record


def _array(record: Mapping[str, object], key: str) -> np.ndarray:
    """Read either a live array or its JSON representation from a reloaded file.

    Parameters
    ----------
    record : Mapping
        Material processing metadata.
    key : str
        Array name, including units where applicable.

    Returns
    -------
    numpy.ndarray
        Float64 values with their recorded shape.
    """
    value = record[key]
    if isinstance(value, Mapping):
        value = value["__ndarray__"]
    return np.asarray(value, dtype=float)


def _plane_indices(output: Output, record: Mapping[str, object]) -> np.ndarray:
    """Locate exact saved planes without interpolating data from another position.

    Parameters
    ----------
    output : Output
        Opened saved run with z in m.
    record : Mapping
        Material processing data with the requested save positions in m.

    Returns
    -------
    numpy.ndarray
        Integer indices into the recorded material positions.
    """
    recorded = _array(record, "z_m")
    positions = np.asarray(output["z"])
    indices = np.searchsorted(recorded, positions)
    if np.any(indices >= recorded.size) or not np.array_equal(
        recorded[indices], positions
    ):
        raise ProcessingError(
            "saved planes do not match mixture reconstruction positions"
        )
    return indices
