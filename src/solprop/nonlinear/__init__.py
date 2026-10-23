# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Nonlinear responses and their models.

Response definitions with declared traits, ionisation rates, Raman models, the
GNLSE mapping and the normalisation prefactors (physics layer).
"""

from solprop.nonlinear.envelope_plasma import (
    EnvelopePlasmaResponse,
    EnvelopePlasmaSources,
)
from solprop.nonlinear.gnlse import (
    GnlseProblem,
    SpecifiedMode,
    attenuation_from_db,
    kerr_susceptibility,
    problem_from,
)
from solprop.nonlinear.mixtures import electronic_response, responses_for
from solprop.nonlinear.normalisation import (
    modal_kerr_coefficient,
    mode_average_prefactor,
    sqrtw_to_vm,
    uppe_prefactor,
)
from solprop.nonlinear.raman import (
    DampedOscillator,
    Dephasing,
    IntermediateBroadening,
    MolecularOptions,
    RamanModel,
    RamanResponse,
    RamanTensorTerm,
    RotationalLines,
    SampledShape,
    SilicaTensorResponse,
    bundle_response,
    normalised_damped_oscillator,
    raman_drive,
    response_for,
    rotational_nonrigid,
    undamped_dephasing,
    vibrational,
)
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    Domain,
    KerrResponse,
    PlasmaResponse,
    Polarisation,
    ResponseTraits,
    thg_carrier,
)
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEConsistency,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)

__all__ = [
    "DampedOscillator",
    "Dephasing",
    "DispersiveKerrResponse",
    "Domain",
    "EnvelopePlasmaResponse",
    "EnvelopePlasmaSources",
    "GnlseProblem",
    "HaccBasis",
    "IntermediateBroadening",
    "KerrResponse",
    "MolecularOptions",
    "PlasmaResponse",
    "Polarisation",
    "RamanModel",
    "RamanResponse",
    "RamanTensorTerm",
    "ResponseTraits",
    "RotationalLines",
    "SampledShape",
    "SilicaTensorResponse",
    "SpecifiedMode",
    "TDSEAtomicWindow",
    "TDSEBasis",
    "TDSECalibration",
    "TDSEConsistency",
    "TDSEExecution",
    "TDSEGasResponse",
    "TDSESampling",
    "attenuation_from_db",
    "bundle_response",
    "electronic_response",
    "kerr_susceptibility",
    "modal_kerr_coefficient",
    "mode_average_prefactor",
    "normalised_damped_oscillator",
    "problem_from",
    "raman_drive",
    "response_for",
    "responses_for",
    "rotational_nonrigid",
    "sqrtw_to_vm",
    "thg_carrier",
    "undamped_dephasing",
    "uppe_prefactor",
    "vibrational",
]
