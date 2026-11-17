# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The translation layer: physics objects turned into the kernel problem a backend runs.

:func:`~solprop.translate.assembly.assemble` is the entry point. It builds the
cumulative linear operator, the masks, the modal or spatial transform and the compiled
responses, and returns a :class:`~solprop.translate.problem.Translation` ``(problem,
request)``: the validated :class:`~solprop.translate.problem.PropagationProblem` that
crosses into a backend (``docs/dev/kernel-contract.md``) and the
:class:`~solprop.translate.run_request.RunRequest` of a fresh run of it. The modules, by
role:

- Contract and assembly: ``problem``, ``routes``, ``run_request``, ``assembly``,
  ``validation``, ``walls`` (a capillary's coupled flows), ``admission`` and ``stage``
  (the per-stage terms and scalars a backend evaluates).
- Representations along z: ``chebyshev``, ``expansion`` and ``linop``.
- Grids and boundaries: ``masks``, ``transverse`` and ``collar_check``.
- Transforms and the plane measure: ``modal``, ``radial``, ``free``, ``planar``,
  ``spatial``, ``flux`` and ``coupling`` (a coupling port's entrance).
- The general modal route: ``general``, ``coupled``, ``modal_tables`` and
  ``modal_path``.
- Responses: ``responses``, ``raman``, ``ionisation``, ``plasma_integration``,
  ``plasma_source``, ``envelope_plasma``, ``tdse`` and ``tdse_source``.
"""

from solprop.translate.assembly import assemble
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
    Transform,
    mode_averaged_transform,
    multimode_transform,
)
from solprop.translate.problem import (
    Flags,
    Initial,
    PropagationProblem,
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
from solprop.translate.routes import (
    GeneralRoute,
    ModalRoute,
    Route,
    SpatialRoute,
    TDSESource,
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
    "DEFAULT_TOL",
    "MAX_DECAY_EXPONENT",
    "AbsorberScheme",
    "CompiledResponses",
    "Flags",
    "GeneralRoute",
    "Grids",
    "Initial",
    "KerrCoefficients",
    "LinearOperator",
    "ModalRoute",
    "ParameterExpansion",
    "PhiRepresentation",
    "PiecewiseChebyshev",
    "PlasmaCoefficients",
    "PropagationProblem",
    "RamanCoefficients",
    "RamanKernels",
    "RateTable",
    "RepresentationChoice",
    "ResponseStage",
    "Route",
    "SpatialRoute",
    "StageScalars",
    "StageTerms",
    "StepAnchor",
    "TDSESource",
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
