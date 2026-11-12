# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Design calculators: soliton numbers and dispersive-wave phase matching.

The numbers a person works out before a run. For a pulse of energy :math:`E` and
full width at half maximum :math:`\tau_{\rm fwhm}` in a mode with group-velocity
dispersion :math:`\beta_2`, effective area :math:`A_{\rm eff}` and a fill of nonlinear
index :math:`n_2`,

.. math::
   \gamma = \frac{n_2\omega_0}{cA_{\rm eff}},\quad
   L_D = \frac{\tau_0^2}{|\beta_2|},\quad
   L_{NL} = \frac{1}{\gamma P_0},\quad
   N = \sqrt{L_D/L_{NL}},\quad
   L_{\rm fiss} = L_D/N,

with the natural width :math:`\tau_0` and the peak power :math:`P_0` of the chosen
pulse shape (:func:`pulse_width`, :func:`pulse_peak_power`). One shape runs through the
whole chain, so the Gaussian numbers are never computed with a ``sech`` width.

A dispersive wave is phase matched to a soliton at the frequencies where

.. math::
   \Delta\beta(\omega) = \beta_{\rm dw}(\omega) - \beta_{\rm s}(\omega_0)
   - \beta_{1,\rm s}(\omega_0)(\omega-\omega_0) - k_{\rm NL}

vanishes (:func:`phase_mismatch`). The last term is the soliton's nonlinear wavenumber
and uses the largest unchirped constituent when pulse data are given
(:func:`soliton_content`). Without pulse data the condition is linear.
``docs/guide/workflows/design-tools.md`` derives both and says what the nonlinear term
assumes.

Everything is SI except pressure, which is in bar as at the rest of the user interface
(``docs/dev/conventions.md`` section 1).
"""

from solprop.tools.beam import (
    BeamParams,
    beam_params,
    beam_through_lens,
    field_to_intensity,
    intensity_to_field,
    self_focusing_power,
)
from solprop.tools.capillary import (
    GasRatio,
    SolitonParams,
    capillary_design,
    capillary_params,
    gas_ratio,
    soliton_params,
)
from solprop.tools.design import (
    RDW,
    ZDW,
    DesignBuild,
    DesignRoots,
    DesignTarget,
    RootChoice,
    SolitonOrder,
    pressure_rdw,
    pressure_zdw,
    radius_zdw,
    solve_parameter,
)
from solprop.tools.phase_matching import (
    NonlinearPhase,
    lambda_rdw,
    lambda_rdw_all,
    phase_mismatch,
)
from solprop.tools.pulse import PulseShape, pulse_energy, pulse_peak_power, pulse_width
from solprop.tools.soliton import (
    SolitonContent,
    dispersion_length,
    fission_length,
    nonlinear_coefficient,
    nonlinear_length,
    soliton_content,
    soliton_order,
    soliton_peak_power,
)
from solprop.tools.waveguide import Fill

__all__ = [
    "RDW",
    "ZDW",
    "BeamParams",
    "DesignBuild",
    "DesignRoots",
    "DesignTarget",
    "Fill",
    "GasRatio",
    "NonlinearPhase",
    "PulseShape",
    "RootChoice",
    "SolitonContent",
    "SolitonOrder",
    "SolitonParams",
    "beam_params",
    "beam_through_lens",
    "capillary_design",
    "capillary_params",
    "dispersion_length",
    "field_to_intensity",
    "fission_length",
    "gas_ratio",
    "intensity_to_field",
    "lambda_rdw",
    "lambda_rdw_all",
    "nonlinear_coefficient",
    "nonlinear_length",
    "phase_mismatch",
    "pressure_rdw",
    "pressure_zdw",
    "pulse_energy",
    "pulse_peak_power",
    "pulse_width",
    "radius_zdw",
    "self_focusing_power",
    "soliton_content",
    "soliton_order",
    "soliton_params",
    "soliton_peak_power",
    "solve_parameter",
]
