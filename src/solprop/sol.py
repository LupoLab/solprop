# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The session namespace: everything a person types at a prompt, behind one import.

For a terminal or a notebook, **not** for library code::

    import solprop.sol as sol          # or: from solprop.sol import *

    sol.capillary_params(125e-6, "He", 1.0, energy=275e-6, fwhm=7.5e-15,
                         wavelength=800e-9)
    run = sol.prop_capillary(125e-6, 1.0, "He", 1.0, wavelength=800e-9, ...)
    sol.prop_2d(run)
    sol.compress_taylor(run)

This is the one module of the package whose import has side effects, which is its whole
purpose. Importing it calls :func:`start_session`, which

1. configures logging at ``INFO`` through :func:`solprop.log.configure_logging`, so
   that a run reports its set-up and draws its progress bar; and
2. silences :class:`~solprop.exceptions.MaterialRangeWarning`. Every capillary run asks
   its gas and its cladding for an index over a band wider than any published fit
   covers, so at a prompt the notice is noise. A script that imports from the home
   modules keeps it: the library's default is unchanged, and
   :func:`material_warnings` turns it back on here.

Both are idempotent, and :func:`end_session` undoes them. Nothing is printed, and
matplotlib's backend and interactive mode are left alone.

The names are flat and explicit. Where two home modules use one name, the processing
function keeps it and the plot takes a ``plot_`` prefix: ``energy`` measures,
``plot_energy`` draws; likewise ``plot_stats``. The names are this package's snake
case, such as ``prop_2d``; ``docs/guide/workflows/interactive.md`` has the table.
``numpy`` and
``matplotlib.pyplot`` are deliberately not re-exported, so a star import never shadows
your own ``np`` and ``plt``.
"""

import logging
import warnings
from typing import cast

from solprop.driver.executors import Batch, Local, Pool, Queue, Slurm
from solprop.driver.interface import (
    prop_capillary,
    prop_free,
    prop_gnlse,
    prop_planar,
    prop_radial,
    prop_waveguide,
)
from solprop.driver.scan import Scan, ScanPoint, run_scan
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.driver.tdse import HaccPreset, hacc_basis, hacc_preset
from solprop.driver.tdse_consistency import ConsistencyResult, check_consistency
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.log import PACKAGE_LOGGER, configure_logging
from solprop.modes import (
    IndexTable,
    MarcatiliMode,
    ModeCollection,
    PlanarGrid,
    RectMode,
    SlabMode,
    StepIndexMode,
    TabulatedHollowMode,
    TabulatedMode,
    TubeLattice,
    VincettiMode,
    gaussian_coupling,
    gaussian_projection,
    gaussian_to_per_unit_width,
    per_unit_width_to_gaussian,
)
from solprop.modes.base import (
    aeff,
    alpha,
    beta,
    beta1,
    beta2,
    db_per_m,
    loss_length,
    transmission,
    zdw,
)
from solprop.nonlinear import (
    DampedOscillator,
    DispersiveKerrResponse,
    EnvelopePlasmaResponse,
    HaccBasis,
    KerrResponse,
    PlasmaResponse,
    RamanResponse,
    RotationalLines,
    SpecifiedMode,
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEConsistency,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.optics import (
    Chain,
    ChirpedMirror,
    GratingPair,
    Material,
    PrismGeometry,
    PrismPair,
    PrismPaths,
    TaylorPhase,
    TransferFunction,
    brewster_angle,
    brewster_apex,
    chirped_mirrors,
    littrow_angle,
)
from solprop.output import HDF5Output, MemoryOutput, TDSEArchive, open_tdse_archive
from solprop.output.hdf5 import open_file, resume
from solprop.physdata import (
    Gas,
    MaterialSelection,
    Mixture,
    barium_fluoride,
    bk7,
    calcium_fluoride,
    density,
    fused_silica,
    glass,
    magnesium_fluoride,
    n2,
    refractive_index,
    sapphire,
    species,
)
from solprop.plotting import energy as plot_energy
from solprop.plotting import prop_2d, spec_1d, spectrogram, time_1d
from solprop.plotting import stats as plot_stats
from solprop.processing import (
    CepOptimum,
    Compression,
    Passband,
    apply_transfer,
    arrival_time,
    band_coherence,
    bandpass,
    channel_amplitudes,
    channel_labels,
    coherence_time,
    compress_gratings,
    compress_material,
    compress_mirrors,
    compress_prisms,
    compress_taylor,
    coordinate_intensity,
    electric_field,
    energy,
    example_output,
    field_autocorrelation,
    fwhm,
    fwhm_f,
    fwhm_t,
    g12,
    g12_of,
    get_field,
    get_phase,
    get_power,
    get_power_weight,
    get_spectral_density,
    get_spectrum,
    grid_of,
    half_max_range,
    instantaneous_flux,
    intensity_autocorrelation,
    magnetic_field,
    moment,
    nearest_plane,
    optimise_cep,
    peak_power,
    physical_energy,
    reconstructed_flux,
    specres,
    spectral_bands,
    time_bandwidth,
    transform_limit,
)
from solprop.processing.scans import ScanOutput
from solprop.processing.scans import open as open_scan
from solprop.profiles import (
    Constant,
    DensityPath,
    Linear,
    PathVariation,
    PiecewiseLinear,
    PressureGradient,
)
from solprop.pulses import (
    CustomPulse,
    CWPulse,
    DataPulse,
    GaussianPulse,
    PlanarPulse,
    SechPulse,
    SpatialPulse,
)
from solprop.stepper import Multistep, RungeKutta
from solprop.tools import (
    GasRatio,
    NonlinearPhase,
    SolitonContent,
    SolitonParams,
    capillary_design,
    capillary_params,
    dispersion_length,
    field_to_intensity,
    fission_length,
    gas_ratio,
    intensity_to_field,
    lambda_rdw,
    lambda_rdw_all,
    nonlinear_coefficient,
    nonlinear_length,
    phase_mismatch,
    pressure_rdw,
    pressure_zdw,
    pulse_peak_power,
    pulse_width,
    self_focusing_power,
    soliton_content,
    soliton_order,
    soliton_params,
)

__all__ = [
    "Batch",
    "CWPulse",
    "CepOptimum",
    "Chain",
    "ChirpedMirror",
    "Compression",
    "ConsistencyResult",
    "Constant",
    "CustomPulse",
    "DampedOscillator",
    "DataPulse",
    "DensityPath",
    "DispersiveKerrResponse",
    "EnvelopeGrid",
    "EnvelopePlasmaResponse",
    "FieldGrid",
    "FreeGrid",
    "Gas",
    "GasRatio",
    "GaussianPulse",
    "GratingPair",
    "HDF5Output",
    "HaccBasis",
    "HaccPreset",
    "IndexTable",
    "KerrResponse",
    "LineGrid",
    "Linear",
    "Local",
    "MarcatiliMode",
    "Material",
    "MaterialSelection",
    "MemoryOutput",
    "Mixture",
    "ModeCollection",
    "Multistep",
    "NonlinearPhase",
    "OptmatRate",
    "Passband",
    "PathVariation",
    "PiecewiseLinear",
    "PlanarGrid",
    "PlanarPulse",
    "PlasmaResponse",
    "Pool",
    "PressureGradient",
    "PrismGeometry",
    "PrismPair",
    "PrismPaths",
    "Queue",
    "RadialGrid",
    "RamanResponse",
    "RectMode",
    "RotationalLines",
    "RungeKutta",
    "Scan",
    "ScanOutput",
    "ScanPoint",
    "SechPulse",
    "Simulation",
    "SimulationSettings",
    "SlabMode",
    "Slurm",
    "SolitonContent",
    "SolitonParams",
    "SpatialPulse",
    "SpecifiedMode",
    "StepIndexMode",
    "TDSEArchive",
    "TDSEAtomicWindow",
    "TDSEBasis",
    "TDSECalibration",
    "TDSEConsistency",
    "TDSEExecution",
    "TDSEGasResponse",
    "TDSESampling",
    "TabulatedHollowMode",
    "TabulatedMode",
    "TaylorPhase",
    "TransferFunction",
    "TubeLattice",
    "VincettiMode",
    "aeff",
    "alpha",
    "apply_transfer",
    "arrival_time",
    "band_coherence",
    "bandpass",
    "barium_fluoride",
    "beta",
    "beta1",
    "beta2",
    "bk7",
    "brewster_angle",
    "brewster_apex",
    "calcium_fluoride",
    "capillary_design",
    "capillary_params",
    "channel_amplitudes",
    "channel_labels",
    "check_consistency",
    "chirped_mirrors",
    "coherence_time",
    "compress_gratings",
    "compress_material",
    "compress_mirrors",
    "compress_prisms",
    "compress_taylor",
    "configure_logging",
    "coordinate_intensity",
    "db_per_m",
    "density",
    "dispersion_length",
    "electric_field",
    "end_session",
    "energy",
    "example_output",
    "field_autocorrelation",
    "field_to_intensity",
    "fission_length",
    "fused_silica",
    "fwhm",
    "fwhm_f",
    "fwhm_t",
    "g12",
    "g12_of",
    "gas_ratio",
    "gaussian_coupling",
    "gaussian_projection",
    "gaussian_to_per_unit_width",
    "get_field",
    "get_phase",
    "get_power",
    "get_power_weight",
    "get_spectral_density",
    "get_spectrum",
    "glass",
    "grid_of",
    "hacc_basis",
    "hacc_preset",
    "half_max_range",
    "instantaneous_flux",
    "intensity_autocorrelation",
    "intensity_to_field",
    "lambda_rdw",
    "lambda_rdw_all",
    "littrow_angle",
    "loss_length",
    "magnesium_fluoride",
    "magnetic_field",
    "material_warnings",
    "moment",
    "n2",
    "nearest_plane",
    "nonlinear_coefficient",
    "nonlinear_length",
    "open_file",
    "open_scan",
    "open_tdse_archive",
    "optimise_cep",
    "peak_power",
    "per_unit_width_to_gaussian",
    "phase_mismatch",
    "physical_energy",
    "plot_energy",
    "plot_stats",
    "pressure_rdw",
    "pressure_zdw",
    "prop_2d",
    "prop_capillary",
    "prop_free",
    "prop_gnlse",
    "prop_planar",
    "prop_radial",
    "prop_waveguide",
    "pulse_peak_power",
    "pulse_width",
    "reconstructed_flux",
    "refractive_index",
    "resume",
    "run_scan",
    "sapphire",
    "self_focusing_power",
    "set_log_level",
    "soliton_content",
    "soliton_order",
    "soliton_params",
    "spec_1d",
    "species",
    "specres",
    "spectral_bands",
    "spectrogram",
    "start_session",
    "time_1d",
    "time_bandwidth",
    "transform_limit",
    "transmission",
    "zdw",
]

logger = logging.getLogger(__name__)


_OWN_FILTER = ("ignore", None, MaterialRangeWarning, None, 0)
"""The entry :func:`warnings.filterwarnings` makes for this module's one filter."""


def material_warnings(enabled: bool) -> None:
    """Turn the material-range warnings on or off for this session.

    Off installs one ``"ignore"`` filter for
    :class:`~solprop.exceptions.MaterialRangeWarning`; on removes exactly that entry, so
    whatever was in force before this module was imported is in force again -- Python's
    default, or a stricter filter of your own, which is never touched or shadowed. (An
    ``"ignore"`` filter of your own for the same category is the same entry, and goes
    with it.)

    Parameters
    ----------
    enabled : bool
        ``True`` to see the warnings again; ``False`` to silence them, as importing
        this module does.

    Examples
    --------
    >>> import warnings
    >>> from solprop import sol
    >>> from solprop.exceptions import MaterialRangeWarning
    >>> with warnings.catch_warnings():
    ...     sol.material_warnings(False)
    ...     warnings.warn("outside the fit", MaterialRangeWarning)
    """
    if not enabled:
        warnings.filterwarnings("ignore", category=MaterialRangeWarning)
        return
    # `warnings` has no public call that removes one filter. Removing an "ignore" entry
    # from the list is safe without the module's private bookkeeping, which exists to
    # invalidate the once-per-location registries: an ignored warning is never entered
    # in one.
    filters = cast("list[object]", warnings.filters)  # typeshed calls it a Sequence
    while _OWN_FILTER in filters:
        filters.remove(_OWN_FILTER)


def set_log_level(level: int | str) -> None:
    """Set the level of the package's log output for this session.

    Parameters
    ----------
    level : int or str
        A :mod:`logging` level or its name: ``"DEBUG"`` for the translation
        summaries, ``"WARNING"`` for a quiet session.
    """
    configure_logging(level)


def start_session(level: int | str = logging.INFO) -> None:
    """Configure logging and silence the material-range warnings.

    What importing :mod:`solprop.sol` does. Calling it again changes nothing: one
    handler, one filter.

    Parameters
    ----------
    level : int or str, optional
        Level of the package's log output.
    """
    configure_logging(level)
    material_warnings(False)
    logger.debug("interactive session started")


def end_session() -> None:
    """Undo :func:`start_session`: remove the log handler and show the warnings.

    For a test, or a long-lived process that wants the library's defaults back. The
    handler installed by :func:`solprop.log.configure_logging` is removed and the
    package logger returns to inheriting its level.
    """
    package = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(package.handlers):
        if getattr(handler, "_solprop_installed", False):
            package.removeHandler(handler)
            handler.close()
    package.setLevel(logging.NOTSET)
    material_warnings(True)


start_session()
