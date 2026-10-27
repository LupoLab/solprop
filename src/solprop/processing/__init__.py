# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Post-processing of saved runs.

Read a run back through the :class:`solprop.output.schema.Output` protocol -- a memory
handler or an HDF5 file, through the same accessors -- and get fields, spectra, spectral
densities, pulse metrics and the coherence of an ensemble, in the optics convention and
in SI units. The manual page is ``docs/guide/results/processing.md``.

Nothing here touches a backend, and no window is ever applied unless the call asks for
one.
"""

from solprop.processing.axes import (
    BANDS,
    SPECTRAL_AXES,
    Band,
    ModeSelection,
    SpectralAxis,
    TransverseUnits,
    band_mask,
    grid_from_record,
    grid_of,
    layout,
    spectral_axis,
    spectral_order,
    transverse_units,
)
from solprop.processing.coherence import g12, g12_of
from solprop.processing.compression import (
    CepOptimum,
    Compression,
    apply_transfer,
    compress_gratings,
    compress_material,
    compress_mirrors,
    compress_prisms,
    compress_taylor,
    optimise_cep,
    transform_limit,
)
from solprop.processing.fields import (
    FIELD_KINDS,
    NOISE_FLOORS,
    FieldKind,
    NoiseFloor,
    get_field,
    get_phase,
    get_power,
    get_spectral_density,
    get_spectrum,
    nearest_plane,
)
from solprop.processing.mixtures import (
    absorbed_spectral_energy,
    get_power_weight,
    node_fluence,
    species_ionisation,
    species_ionisation_energy,
)
from solprop.processing.modal import (
    channel_amplitudes,
    channel_labels,
    coordinate_intensity,
    electric_field,
    instantaneous_flux,
    magnetic_field,
    physical_energy,
    reconstructed_flux,
)
from solprop.processing.planes import PLANE_TOLERANCE, field_plane
from solprop.processing.spectra import (
    DEFAULT_BANDPASS_BINS,
    DEFAULT_BANDPASS_FRACTION,
    Passband,
    bandpass,
    specres,
)
from solprop.processing.time import (
    ARRIVAL_METHODS,
    ArrivalMethod,
    arrival_time,
    coherence_time,
    energy,
    field_autocorrelation,
    fwhm,
    fwhm_f,
    fwhm_t,
    half_max_range,
    intensity_autocorrelation,
    moment,
    peak_power,
    time_bandwidth,
)

__all__ = [
    "ARRIVAL_METHODS",
    "BANDS",
    "DEFAULT_BANDPASS_BINS",
    "DEFAULT_BANDPASS_FRACTION",
    "FIELD_KINDS",
    "NOISE_FLOORS",
    "PLANE_TOLERANCE",
    "SPECTRAL_AXES",
    "ArrivalMethod",
    "Band",
    "CepOptimum",
    "Compression",
    "FieldKind",
    "ModeSelection",
    "NoiseFloor",
    "Passband",
    "SpectralAxis",
    "TransverseUnits",
    "absorbed_spectral_energy",
    "apply_transfer",
    "arrival_time",
    "band_mask",
    "bandpass",
    "channel_amplitudes",
    "channel_labels",
    "coherence_time",
    "compress_gratings",
    "compress_material",
    "compress_mirrors",
    "compress_prisms",
    "compress_taylor",
    "coordinate_intensity",
    "electric_field",
    "energy",
    "field_autocorrelation",
    "field_plane",
    "fwhm",
    "fwhm_f",
    "fwhm_t",
    "g12",
    "g12_of",
    "get_field",
    "get_phase",
    "get_power",
    "get_power_weight",
    "get_spectral_density",
    "get_spectrum",
    "grid_from_record",
    "grid_of",
    "half_max_range",
    "instantaneous_flux",
    "intensity_autocorrelation",
    "layout",
    "magnetic_field",
    "moment",
    "nearest_plane",
    "node_fluence",
    "optimise_cep",
    "peak_power",
    "physical_energy",
    "reconstructed_flux",
    "species_ionisation",
    "species_ionisation_energy",
    "specres",
    "spectral_axis",
    "spectral_order",
    "time_bandwidth",
    "transform_limit",
    "transverse_units",
]
