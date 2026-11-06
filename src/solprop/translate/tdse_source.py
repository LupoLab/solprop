# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Admit the scalar helium TDSE route and translate its macroscopic source.

The microscopic provider supplies per-atom current and dipole histories. The
ordinary Kerr compiler remains the authority for the optmat cubic. The current
factor is constructed independently of the polarisation factor, including at
zero frequency; no division by frequency is involved (SEP 12 section 4.6).
"""

import functools
import json
from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Float

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid, TimeGrid, planck_taper
from solprop.modes.capillary import MarcatiliMode
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import TransverseQuadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.nonlinear.tdse import TDSEGasResponse
from solprop.physdata import Gas
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.physdata.species import Species, species
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.tdse import TDSEDescriptor, lattice_times

__all__ = [
    "STEADY_CURRENT_CERTIFICATE_LIMIT",
    "TDSE_UNAVAILABLE_STATISTICS",
    "admit_tdse",
    "current_numerator",
    "steady_current_certificate",
]

STEADY_CURRENT_CERTIFICATE_LIMIT = 1e-10
"""Largest admitted steady-current certificate of a gated native source's gate.

``kappa_ss`` is the active-band norm of the edge-free source of ``S = 1``, which is
exactly zero in the continuous identity, relative to the full transform of the gate
(:func:`steady_current_certificate`). It is the whole admission: no fixed taper sample
count is imposed.
"""

TDSE_UNAVAILABLE_STATISTICS = frozenset(
    {
        "electron_density",
        "peak_ionisation_rate",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_ionisation_energy",
        "quadrature_error",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
        "on_axis_electron_density",
        "on_axis_peak_ionisation_rate",
        "on_axis_species_electron_density",
        "on_axis_species_peak_ionisation_rate",
    }
)
"""Legacy rate/dipole-only statistics that do not identify a TDSE observable.

The complete-source quadrature diagnostic is recorded with its actual stage
identity; a PEC predictor is not the accepted field named by a saved-plane row.
"""


def _helium(mode: MarcatiliMode) -> Species:
    """Resolve the same helium material that supplies the mode's dispersion.

    Parameters
    ----------
    mode : MarcatiliMode
        Declared scalar capillary mode and its material.

    Returns
    -------
    Species
        Admitted optmat helium species with no initial ionisation.
    """
    if mode.mixture is not None:
        if len(mode.mixture.components) != 1:
            raise TranslationError(
                "TDSE refuses gas mixtures",
                parameter=("tdse", "gas"),
                rule="tdse-helium",
            )
        component = mode.mixture.components[0]
        if component.preionisation != 0.0:
            raise TranslationError(
                "TDSE refuses preionised populations",
                parameter=("tdse", "gas"),
                rule="tdse-helium",
            )
        material = component.species
    elif mode.gas == Gas.He:
        material = species(Gas.He)
    else:
        raise TranslationError(
            "TDSE requires a declared helium fill",
            parameter=("tdse", "gas"),
            rule="tdse-helium",
        )
    if material.name != "He" or not isinstance(material.kerr, OptmatKerr):
        raise TranslationError(
            "TDSE requires the selected optmat helium material",
            parameter=("tdse", "gas"),
            rule="tdse-helium",
        )
    return material


def admit_tdse(  # noqa: PLR0913 - admission checks the declared route together.
    responses: Sequence[object],
    grid: TimeGrid,
    modes: Sequence[Mode],
    *,
    shape: tuple[int, int, int, int],
    quadrature: TransverseQuadrature | None,
    integrator: Multistep | LowStorageRungeKutta | None,
    modified_noise: bool,
    shock: bool,
    prefactor_index: float | None,
    interpolate_planes: bool = False,
) -> tuple[tuple[object, ...], TDSEDescriptor | None]:
    """Replace the TDSE declaration with its explicitly owned cubic coefficient.

    Parameters
    ----------
    responses : sequence
        Scientific declarations, including at most one TDSE gas response.
    grid : solprop.grid.TimeGrid
        Resolved response grid, retaining the parent's coarse bins and window.
    modes : sequence of Mode
        Propagated modes in state order.
    shape : tuple of int
        Canonical state dimensions, batch, mode, polarisation and frequency.
    quadrature : TransverseQuadrature or None
        Declared transverse integration rule.
    integrator : Multistep or LowStorageRungeKutta or None
        The run's integrator; TDSE requires an Adams method.
    modified_noise : bool
        Whether the ``'modified'`` shot-noise model is present. Its draw is kept apart
        from the field and added at every stage, which the atom's drive does not
        carry, so it is refused. The ``'input'`` draw is part of the launch: the atom
        sees it inside the field it is driven by, and it is admitted (SEP 37,
        section 6).
    shock : bool
        Whether the full frequency-dependent polarisation numerator is retained.
    prefactor_index : float or None
        Requested override of the physical effective index.
    interpolate_planes : bool, optional
        Whether save planes are interpolated inside accepted steps. TDSE requires
        landed observations, so this must be ``False``.

    Returns
    -------
    tuple
        Ordinary responses for the existing compiler and immutable atomic identity.

    Raises
    ------
    TranslationError
        For unsupported geometry, material, sampling or response ownership.
    """
    atomic = [one for one in responses if isinstance(one, TDSEGasResponse)]
    if not atomic:
        return tuple(responses), None
    if len(atomic) != 1 or len(responses) != 1:
        raise TranslationError(
            "TDSE owns the complete electronic response: duplicate Kerr, plasma "
            "or other response declarations are unsupported"
        )
    if not isinstance(grid, FieldGrid) or shape[:3] != (1, 1, 1):
        raise TranslationError("TDSE requires an unbatched scalar one-mode field grid")
    if len(modes) != 1 or not isinstance(modes[0], MarcatiliMode):
        raise TranslationError("TDSE requires the resolved HE_11 capillary mode")
    mode = modes[0]
    if (mode.kind, mode.n, mode.m, mode.phi) != ("HE", 1, 1, 0.0):
        raise TranslationError("TDSE requires the x-polarised HE_11 capillary mode")
    if not mode.core_radius.is_constant:
        raise TranslationError("TDSE refuses tapered capillaries")
    if quadrature is None:
        raise TranslationError("TDSE requires a resolved radial quadrature")
    if not isinstance(integrator, Multistep):
        raise TranslationError("TDSE requires Multistep Adams integration")
    if interpolate_planes:
        raise TranslationError(
            "TDSE requires landed save planes; set interpolate_planes=False"
        )
    if modified_noise:
        raise TranslationError(
            "TDSE refuses the 'modified' noise model, whose draw is kept apart from "
            "the field the atom is driven by; use noise='input' or noise=None"
        )
    if not shock or prefactor_index is not None:
        raise TranslationError(
            "TDSE requires physical frequency-dependent normalisation"
        )
    material = _helium(mode)
    response = atomic[0]
    descriptor = TDSEDescriptor.from_response(response, grid)
    if response.kerr_model != "optmat_cubic":
        return (), descriptor
    coefficient = material.kerr_coefficient(grid.reference_wavelength)
    assert coefficient is not None
    return (KerrResponse(chi3=coefficient, thg=True),), descriptor


def current_numerator(grid: TimeGrid) -> Float[np.ndarray, "1 1 n_omega"]:
    """Construct the current prefactor numerator directly in the FFT convention.

    Divide this real, negative array by the stage's positive modal power weight
    ``Re(neff)``. The current and polarisation then have the same projection and
    crop normalization without introducing a spurious zero-frequency singularity.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Resolved field grid supplying active bins and the transform crop scale.

    Returns
    -------
    numpy.ndarray
        Read-only ``-crop_scale / (2 eps0 c)`` on active bins, zero elsewhere,
        shape ``(1, 1, n_omega)``.
    """
    values = np.zeros((1, 1, grid.n_omega), dtype=np.float64)
    values[..., grid.active_mask] = -grid.crop_scale / (2 * EPS0 * C)
    values.setflags(write=False)
    return values


def steady_current_certificate(
    descriptor: TDSEDescriptor,
    omega: Float[np.ndarray, " n_omega"],
    active: Bool[np.ndarray, " n_omega"],
) -> float:
    r"""Compute the steady-current certificate of a gated TDSE source (SEP 26, 4.4).

    .. math::

        \kappa_{\rm ss} = \frac{\lVert \mathrm{DFT}[g] - \mathrm{DFT}[g']/(i\omega)
        \rVert_A}{\lVert \mathrm{DFT}[g]\rVert_2},

    the edge-free source of a constant current ``S = 1`` on the active bins ``A`` of
    the coarse spectrum, relative to the full one-sided transform of the gate on the
    fine lattice. The raw forward DFTs of the sampled gate and of its analytic
    derivative are cropped to the coarse bins exactly as a source is. The numerator is
    zero in the continuous identity; what is left is the taper's alias error on this
    lattice and band, plus rounding. It is one transform of a fixed gate per
    configuration and is cached.

    The denominator is the full transform, not the active band (a correction of SEP
    26 section 4.4, which divided by ``||DFT[g]||_A``). A long gate puts almost none of
    its spectrum in the band, so the band-relative ratio compared one rounding-level
    norm with another and refused well-resolved gates: a 1 ps window at 0.4-2 um gave
    1.4e-7. Against the full transform the residual is measured on the scale of the
    gate itself, ``sqrt(N / 2) ||g||_2`` up to the one-sided weighting, which is the
    scale of the truncating source of a unit current before the band crop; the same
    window gives 5.9e-17, while an under-resolved taper still fails by orders of
    magnitude (``docs/guide/media/tdse.md``).

    Parameters
    ----------
    descriptor : TDSEDescriptor
        A gated descriptor: its window's gate, or its recorded fine temporal collar.
    omega : numpy.ndarray
        Coarse angular frequencies in rad/s, shape ``(n_omega,)``.
    active : numpy.ndarray
        Active-bin mask of the same shape; every active frequency is positive.

    Returns
    -------
    float
        Nonnegative dimensionless certificate; zero for an empty band.
    """
    return _steady_current_certificate(
        descriptor, np.asarray(omega).tobytes(), np.asarray(active).tobytes()
    )


@functools.cache
def _steady_current_certificate(
    descriptor: TDSEDescriptor, omega: bytes, active: bytes
) -> float:
    """Evaluate one configuration's certificate once.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Hashable gated descriptor.
    omega, active : bytes
        Raw float64 frequencies and boolean mask, the cache key of the band.

    Returns
    -------
    float
        The certificate.
    """
    frequencies = np.frombuffer(omega, dtype=np.float64)
    mask = np.frombuffer(active, dtype=np.bool_)
    lattice = json.loads(descriptor.configuration_json)["lattice"]
    times = lattice_times(descriptor.samples, descriptor.spacing)
    breakpoints = descriptor.source_breakpoints()
    gate = planck_taper(times, *breakpoints)
    crop = fourier.crop_index(
        lattice["coarse_samples"], descriptor.samples, kind="field"
    )[mask]
    spectrum = fourier.forward(gate, kind="field")
    truncating = spectrum[crop]
    edge = fourier.forward(descriptor.source_gate_derivative(), kind="field")[crop]
    band = frequencies[mask]
    corrected = truncating - edge * (-1j / band)
    # Never zero: a resolved window holds at least four samples, the interior ones
    # with g > 0, and a grid collar has a plateau of ones.
    denominator = float(np.linalg.norm(spectrum))
    assert denominator > 0.0
    return float(np.linalg.norm(corrected)) / denominator
