# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design calculators: the numbers a person works out before a run.

The commonly used names of the submodules are re-exported here.

- :mod:`~solprop.tools.pulse`: the natural width, peak power and energy of a ``sech`` or
  Gaussian pulse, one shape through the whole chain.
- :mod:`~solprop.tools.soliton`: dispersion and nonlinear lengths, soliton orders, the
  fission length, the soliton content of an unchirped pulse and the fill's nonlinear
  coefficient.
- :mod:`~solprop.tools.phase_matching`: the phase mismatch between a soliton and a
  dispersive wave, and its roots in wavelength.
- :mod:`~solprop.tools.capillary`: every design number of a pulse in a mode or a
  gas-filled capillary, the capillary that gives a soliton design, and gas mixing
  ratios.
- :mod:`~solprop.tools.design`: the guide or fill parameter that meets a design target
  (a zero-dispersion wavelength, a dispersive-wave wavelength, a soliton order).
- :mod:`~solprop.tools.beam`: the critical power for self-focusing, Gaussian-beam
  numbers and the conversion between field and intensity.
- :mod:`~solprop.tools.waveguide`: the fill (:data:`~solprop.tools.waveguide.Fill`) and
  the mode a design evaluates.

Everything is SI except pressure, which is in bar as at the rest of the user interface
(``docs/dev/conventions.md`` §1). ``docs/guide/workflows/design-tools.md`` derives the
soliton numbers and the phase-matching condition and says what each assumes.
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
