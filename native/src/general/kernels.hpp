// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The general workspace's helpers: device-callable functions, kernel functors and the
// workspace's view and plan types, shared by its translation units (there is no
// relocatable device code, so a helper a kernel calls is defined in a header, inline).
// Contracting code: a strict translation unit never includes it; its few strict
// helpers (`decayed_fraction`, `compensation_residual`, `StageTerms::phase_sum`, ...)
// carry fp.hpp's macros themselves. Everything lives in solprop_native::general.
// This umbrella holds the include block and the lookup of the helpers, and includes
// its parts in their original order, so a unit that includes it sees one token
// sequence (each part's own summary lists its names):
//   reductions.hpp        NormSums, FluenceMoments (reduction values)
//   adams_trial.hpp       FlowAside, SwappedAside, upload_weights, apply_generator
//   views.hpp             the view and plan aliases, EnvelopeStream, FieldCurrentStream
//   traversal.hpp         cartesian_for, crossing_tiles, crossing_for, PropagatorTile,
//                         tiled_crossing, tiled_update
//   separable_layout.hpp  TransverseLayout, PhysicalSpectrum, RegisterIndex,
//                         TransverseChunk, SpatialCoefficients
//   exponentials.hpp      exponential, cosine_sine, propagator_exp, entry_exponential
//   ionisation.hpp        the ionised fractions and the rate table, band_loss_scale
//   uploads.hpp           upload_span, copy_vector
//   norm_update.hpp       UpdateColumn, reduction_slots, accept_update
//   stage_terms.hpp       StageTerms, a stage's entry and finish, EntryFactors
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "general/kernels.hpp contracts: a strict translation unit must not include it"
#endif

#include "solprop_native/march/adams_phi.hpp"
#include "solprop_native/march/z_table.hpp"
#include "solprop_native/response/coupled_flow.hpp"
#include "solprop_native/response/dispersive.hpp"
#include "solprop_native/response/envelope_plasma.hpp"
#include "solprop_native/response/far_field.hpp"
#include "solprop_native/response/flow_functions.hpp"
#include "solprop_native/response/general_modal.hpp"
#include "solprop_native/response/plasma.hpp"
#include "solprop_native/runtime/build.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/dispatch.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/fp.hpp"
#include "solprop_native/runtime/refuse.hpp"
#include "solprop_native/runtime/setup_data.hpp"
#include "solprop_native/runtime/vmath.hpp"
#include "solprop_native/transforms/fft_plan.hpp"
#include "solprop_native/transforms/gemm.hpp"
#include "solprop_native/transforms/raman_fft.hpp"
#include "solprop_native/transforms/slab.hpp"

#include <KokkosFFT.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <fftw3.h>
#ifdef __SSE2__
#include <immintrin.h>
#define SOLPROP_NATIVE_STREAMING_STORES 1
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
namespace solprop_native {
namespace general {}
// The helpers are found as an unnamed namespace's would be, under their own name so
// that every unit that includes them sees one definition.
using namespace general;
} // namespace solprop_native

// The parts, in their original order (not sorted): each builds on those before it.
// clang-format off
#include "reductions.hpp"
#include "adams_trial.hpp"
#include "views.hpp"
#include "traversal.hpp"
#include "separable_layout.hpp"
#include "exponentials.hpp"
#include "ionisation.hpp"
#include "uploads.hpp"
#include "norm_update.hpp"
#include "stage_terms.hpp"
// clang-format on
