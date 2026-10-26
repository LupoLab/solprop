// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

/*
 * Precision policies and role-typed views (solprop SEP 36 section 6.3, option A).
 *
 * A policy names the scalar of each role a buffer can play. The kernels and the
 * classes that own a route's buffers are templated on the policy and explicitly
 * instantiated once per policy the build selects (`SOLPROP_NATIVE_PRECISIONS`), so
 * that "no binary64 array in a binary32 pass" is a property of the compiled code.
 *
 * The roles, with their scalar under Double / Single / Mixed:
 *
 * - state (double / float / double): the propagated state and every row stored in
 *   the interaction variable -- the Runge-Kutta stages, solution, error and FSAL rows,
 *   the low-storage registers, the Adams history with its compensation and dense
 *   rows, the noise, the held plane -- and the split-step absorber's transforms of the
 *   state (section 4.2).
 * - source (double / float / float): the source pass -- the transforms and their
 *   buffers, the node, polarisation and projection rows, Raman, the plasma scans, the
 *   held and envelope streams and the in-stage absorber terms.
 * - table (double / float / float): narrowed per-run tables a source pass multiplies
 *   by -- the prefactors, the carrier and windows, the synthesis and projection
 *   matrices, the Raman kernels, the plasma rate tables, the envelope rule.
 * - reduction (double / double / double): norm and statistics accumulators. Under
 *   Single it is chosen per target (section 4.5); binary64 on CPUs, the only target
 *   so far.
 * - propagator (double / float / double): the stage propagators and their inverses
 *   as stored. Under Single it is rounded once from a binary64 argument (section
 *   4.3).
 * - argument (double under every policy): the propagator's argument, the linear
 *   operator's terms and coefficients, the z tables, and the host scalars and
 *   combined coefficients (section 4.5). It is not a policy parameter.
 *
 * The routes SEP 33 refuses under reduced precision (general modal, coupled flow,
 * dispersive Kerr, far-field statistics, the slab kind, TDSE) keep binary64 types
 * written out (the `Binary64` aliases below) and exist in the `Double` instantiation
 * alone: `require_policy_route` (general.cpp) refuses them by name under any other
 * policy.
 *
 * The aliases `Complex`, `ComplexView`, `RealView` and the rest below are the binary64
 * types of the seam with solprop (host vectors, bindings) and of those routes.
 */

namespace solprop_native {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Layout = Kokkos::LayoutRight;

/** Every role binary64: the arithmetic every route had before SEP 36. */
struct Double {
  using state = double;
  using source = double;
  using table = double;
  using reduction = double;
  using propagator = double;
  static constexpr const char* name = "double";
};

/** SEP 33's single precision: binary32 everywhere but the reductions (section 4.5). */
struct Single {
  using state = float;
  using source = float;
  using table = float;
  using reduction = double;
  using propagator = float;
  static constexpr const char* name = "single";
};

/** SEP 36's mixed precision: binary64 state and propagator, binary32 source (4.2). */
struct Mixed {
  using state = double;
  using source = float;
  using table = float;
  using reduction = double;
  using propagator = double;
  static constexpr const char* name = "mixed";
};

/** Whether `Policy` is one of the three policies. */
template <class Policy>
inline constexpr bool is_precision_policy_v =
    std::is_same_v<Policy, Double> || std::is_same_v<Policy, Single> ||
    std::is_same_v<Policy, Mixed>;

/** Whether `Policy` keeps every role binary64: the routes refused under reduced
    precision exist under this policy alone. */
template <class Policy>
inline constexpr bool is_binary64_policy_v = std::is_same_v<Policy, Double>;

/** The device views of one scalar, as `Layout` stores them in `MemorySpace`. */
template <class Scalar> struct ScalarViews {
  using Real = Scalar;
  using Complex = Kokkos::complex<Scalar>;
  using ComplexView = Kokkos::View<Complex*, Layout, MemorySpace>;
  using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
  using ComplexView3 = Kokkos::View<Complex***, Layout, MemorySpace>;
  using ComplexView4 = Kokkos::View<Complex****, Layout, MemorySpace>;
  using RealView = Kokkos::View<Scalar*, Layout, MemorySpace>;
  using RealView2 = Kokkos::View<Scalar**, Layout, MemorySpace>;
  using RealView3 = Kokkos::View<Scalar***, Layout, MemorySpace>;
  using RealScalarView = Kokkos::View<Scalar, Layout, MemorySpace>;
};

/** The binary64 views: the seam's types, and every role under `Double`. */
using Binary64 = ScalarViews<double>;

using Complex = Binary64::Complex;
using ComplexView = Binary64::ComplexView;
using ConstComplexView = Kokkos::View<const Complex*, Layout, MemorySpace>;
using ComplexView2 = Binary64::ComplexView2;
using ComplexView3 = Binary64::ComplexView3;
using ComplexView4 = Binary64::ComplexView4;
using RealView = Binary64::RealView;
using ConstRealView = Kokkos::View<const double*, Layout, MemorySpace>;
using RealView2 = Binary64::RealView2;
using RealView3 = Binary64::RealView3;
using ByteView = Kokkos::View<unsigned char*, Layout, MemorySpace>;
using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;

/**
 * The role-typed views of `Policy`, defined once per policy. A class templated on a
 * policy takes its buffer types from here (`typename Roles<Policy>::StateView2`), so
 * each buffer's declaration names its role. Complex views carry no suffix and real
 * ones `Real`; the trailing digit is the rank.
 */
template <class Policy> struct Roles {
  static_assert(is_precision_policy_v<Policy>, "not a solprop_native precision policy");
  using State = ScalarViews<typename Policy::state>;
  using Source = ScalarViews<typename Policy::source>;
  using Table = ScalarViews<typename Policy::table>;
  using Reduction = ScalarViews<typename Policy::reduction>;
  using Propagator = ScalarViews<typename Policy::propagator>;
  using Argument = Binary64;

  using StateComplex = State::Complex;
  using StateView = State::ComplexView;
  using StateView2 = State::ComplexView2;
  using StateView3 = State::ComplexView3;
  using StateRealView = State::RealView;
  using StateRealView2 = State::RealView2;

  using SourceComplex = Source::Complex;
  using SourceView = Source::ComplexView;
  using SourceView2 = Source::ComplexView2;
  using SourceView3 = Source::ComplexView3;
  using SourceRealView = Source::RealView;
  using SourceRealView2 = Source::RealView2;
  using SourceRealView3 = Source::RealView3;

  using TableComplex = Table::Complex;
  using TableView = Table::ComplexView;
  using TableView2 = Table::ComplexView2;
  using TableView3 = Table::ComplexView3;
  using TableRealView = Table::RealView;
  using TableRealView2 = Table::RealView2;
  using TableRealView3 = Table::RealView3;

  using ReductionComplex = Reduction::Complex;
  using ReductionView = Reduction::ComplexView;
  using ReductionView2 = Reduction::ComplexView2;
  using ReductionView3 = Reduction::ComplexView3;
  using ReductionRealView = Reduction::RealView;
  using ReductionRealView2 = Reduction::RealView2;
  using ReductionRealView3 = Reduction::RealView3;
  using ReductionScalarView = Reduction::RealScalarView;

  using PropagatorComplex = Propagator::Complex;
  using PropagatorView = Propagator::ComplexView;
  using PropagatorView2 = Propagator::ComplexView2;
  using PropagatorView3 = Propagator::ComplexView3;

  using ArgumentView = Argument::ComplexView;
  using ArgumentView2 = Argument::ComplexView2;
  using ArgumentView3 = Argument::ComplexView3;
  using ArgumentRealView = Argument::RealView;
  using ArgumentRealView2 = Argument::RealView2;
  using ArgumentRealView3 = Argument::RealView3;
};

/** The policies this build instantiates (`SOLPROP_NATIVE_PRECISIONS`). */
#if !defined(SOLPROP_NATIVE_PRECISION_DOUBLE) || !SOLPROP_NATIVE_PRECISION_DOUBLE
#error "solprop-native: every build instantiates the Double precision policy"
#endif
#if defined(SOLPROP_NATIVE_PRECISION_SINGLE) && SOLPROP_NATIVE_PRECISION_SINGLE
#define SOLPROP_NATIVE_SINGLE_BUILT 1
#else
#define SOLPROP_NATIVE_SINGLE_BUILT 0
#endif
#if defined(SOLPROP_NATIVE_PRECISION_MIXED) && SOLPROP_NATIVE_PRECISION_MIXED
#define SOLPROP_NATIVE_MIXED_BUILT 1
#else
#define SOLPROP_NATIVE_MIXED_BUILT 0
#endif

/** Whether this build instantiated `Policy` (`SOLPROP_NATIVE_PRECISIONS`). */
template <class Policy>
inline constexpr bool precision_built_v = std::is_same_v<Policy, Double>
#if SOLPROP_NATIVE_SINGLE_BUILT
                                          || std::is_same_v<Policy, Single>
#endif
#if SOLPROP_NATIVE_MIXED_BUILT
                                          || std::is_same_v<Policy, Mixed>
#endif
    ;

/*
 * `Kokkos::complex`'s operators promote a binary32 operand silently when the other is
 * binary64, which would compute a binary32 pass in binary64 without a diagnostic. In
 * this namespace those mixed operations do not compile: a pass converts an operand
 * explicitly where its role allows it (solprop SEP 36 section 6.3). Scalars mixing the
 * two formats are caught by `-Wdouble-promotion` in the binary32 translation units.
 */
#define SOLPROP_NATIVE_DELETE_MIXED(op)                                                \
  Kokkos::complex<double> operator op(const Kokkos::complex<float>&,                   \
                                      const Kokkos::complex<double>&) = delete;        \
  Kokkos::complex<double> operator op(const Kokkos::complex<double>&,                  \
                                      const Kokkos::complex<float>&) = delete;         \
  Kokkos::complex<double> operator op(const Kokkos::complex<float>&, const double&) =  \
      delete;                                                                          \
  Kokkos::complex<double> operator op(const double&, const Kokkos::complex<float>&) =  \
      delete;                                                                          \
  Kokkos::complex<double> operator op(const Kokkos::complex<double>&, const float&) =  \
      delete;                                                                          \
  Kokkos::complex<double> operator op(const float&, const Kokkos::complex<double>&) =  \
      delete;
SOLPROP_NATIVE_DELETE_MIXED(+)
SOLPROP_NATIVE_DELETE_MIXED(-)
SOLPROP_NATIVE_DELETE_MIXED(*)
SOLPROP_NATIVE_DELETE_MIXED(/)
#undef SOLPROP_NATIVE_DELETE_MIXED

} // namespace solprop_native
