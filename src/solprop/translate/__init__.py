# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The translation layer.

Physics objects become validated kernel-ready arrays and the kernel contract
(``docs/dev/kernel-contract.md``). :func:`solprop.translate.problem.assemble` is the
entry point: it builds the cumulative linear operator, the masks, the modal transform
and the compiled responses, and returns the single
:class:`~solprop.translate.problem.KernelProblem` that crosses into a backend.
"""

from solprop.translate.chebyshev import (
    DEFAULT_MAX_DEGREE,
    DEFAULT_MAX_INTERVALS,
    DEFAULT_TOL,
    PiecewiseChebyshev,
    build,
    build_from_paths,
)
from solprop.translate.expansion import ParameterExpansion, build_expansion
from solprop.translate.ionisation import RateTable, build_rate_table
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    LinearOperator,
    PhiRepresentation,
    RepresentationChoice,
    build_phi,
    constant_phi,
    delta_phi,
    profile_table,
    tabulate,
    unitary_phase,
    verify_phi,
)
from solprop.translate.masks import (
    AbsorberScheme,
    Grids,
    build_masks,
    noise_spectrum,
    temporal_rates,
)
from solprop.translate.modal import (
    DEFAULT_NODE_BUDGET,
    DEFAULT_SPATIAL_NODE_BUDGET,
    Transform,
    mode_averaged_transform,
    multimode_transform,
)
from solprop.translate.problem import (
    SCHEMA_VERSION,
    Flags,
    Initial,
    KernelProblem,
    StepperSettings,
    assemble,
)
from solprop.translate.raman import (
    MAX_DECAY_EXPONENT,
    RamanKernels,
    build_raman_kernels,
    decay_rate_profiles,
    kernel_spectrum,
    kernel_taper,
    sample_kernel,
    undamped_kernels,
)
from solprop.translate.responses import (
    CompiledResponses,
    KerrCoefficients,
    PlasmaCoefficients,
    RamanCoefficients,
    ResponseStage,
    compile_responses,
    prefactor_at,
    raman_coefficients,
)
from solprop.translate.stage import (
    StageScalars,
    StageTerms,
    StepAnchor,
    anchor,
    combine,
    require_device_ready,
    scalars,
    terms,
)

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_MAX_DEGREE",
    "DEFAULT_MAX_INTERVALS",
    "DEFAULT_NODE_BUDGET",
    "DEFAULT_SPATIAL_NODE_BUDGET",
    "DEFAULT_TOL",
    "MAX_DECAY_EXPONENT",
    "SCHEMA_VERSION",
    "AbsorberScheme",
    "CompiledResponses",
    "Flags",
    "Grids",
    "Initial",
    "KernelProblem",
    "KerrCoefficients",
    "LinearOperator",
    "ParameterExpansion",
    "PhiRepresentation",
    "PiecewiseChebyshev",
    "PlasmaCoefficients",
    "RamanCoefficients",
    "RamanKernels",
    "RateTable",
    "RepresentationChoice",
    "ResponseStage",
    "StageScalars",
    "StageTerms",
    "StepAnchor",
    "StepperSettings",
    "Transform",
    "anchor",
    "assemble",
    "build",
    "build_expansion",
    "build_from_paths",
    "build_masks",
    "build_phi",
    "build_raman_kernels",
    "build_rate_table",
    "combine",
    "compile_responses",
    "constant_phi",
    "decay_rate_profiles",
    "delta_phi",
    "kernel_spectrum",
    "kernel_taper",
    "mode_averaged_transform",
    "multimode_transform",
    "noise_spectrum",
    "prefactor_at",
    "profile_table",
    "raman_coefficients",
    "require_device_ready",
    "sample_kernel",
    "scalars",
    "tabulate",
    "temporal_rates",
    "terms",
    "undamped_kernels",
    "unitary_phase",
    "verify_phi",
]
