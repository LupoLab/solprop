// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/precision.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

namespace {
namespace native = solprop_native;

template <class Policy, class State, class Source, class Table, class Reduction,
          class Propagator>
constexpr bool roles_are() {
  using R = native::Roles<Policy>;
  return std::is_same_v<typename R::StateView2,
                        Kokkos::View<Kokkos::complex<State>**, native::Layout,
                                     native::MemorySpace>> &&
         std::is_same_v<typename R::SourceRealView2,
                        Kokkos::View<Source**, native::Layout, native::MemorySpace>> &&
         std::is_same_v<typename R::TableView2,
                        Kokkos::View<Kokkos::complex<Table>**, native::Layout,
                                     native::MemorySpace>> &&
         std::is_same_v<
             typename R::ReductionRealView,
             Kokkos::View<Reduction*, native::Layout, native::MemorySpace>> &&
         std::is_same_v<typename R::PropagatorView2,
                        Kokkos::View<Kokkos::complex<Propagator>**, native::Layout,
                                     native::MemorySpace>> &&
         // The propagator's argument is binary64 under every policy.
         std::is_same_v<typename R::ArgumentView3, native::ComplexView3>;
}

// solprop SEP 36 section 6.3's table.
static_assert(roles_are<native::Double, double, double, double, double, double>());
static_assert(roles_are<native::Single, float, float, float, double, float>());
static_assert(roles_are<native::Mixed, double, float, float, double, double>());

// The Double policy's views are the binary64 seam types: its instantiation is the
// code the backend had before the policies.
static_assert(
    std::is_same_v<native::Roles<native::Double>::StateView2, native::ComplexView2>);
static_assert(
    std::is_same_v<native::Roles<native::Double>::SourceRealView3, native::RealView3>);
static_assert(native::is_binary64_policy_v<native::Double>);
static_assert(!native::is_binary64_policy_v<native::Single>);
static_assert(!native::is_binary64_policy_v<native::Mixed>);
} // namespace

TEST(PrecisionPolicy, PoliciesAreNamed) {
  EXPECT_EQ(std::string(native::Double::name), "double");
  EXPECT_EQ(std::string(native::Single::name), "single");
  EXPECT_EQ(std::string(native::Mixed::name), "mixed");
}

namespace {

// solprop SEP 36 section 6.3: a binary32 subnormal survives a pass on the execution
// space (no build flushes to zero): the smallest normal halved, quartered and summed,
// and a complex product, each exactly representable as a subnormal.
void subnormals_survive_a_pass() {
  using Value = Kokkos::complex<float>;
  Kokkos::View<float*, native::Layout, native::MemorySpace> real("subnormal.real", 3);
  Kokkos::View<Value*, native::Layout, native::MemorySpace> complex("subnormal.complex",
                                                                    1);
  Kokkos::parallel_for(
      "subnormal.pass", Kokkos::RangePolicy<native::ExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        const float smallest = Kokkos::Experimental::norm_min_v<float>;
        const float half = smallest * 0.5F;
        real(0) = half;
        real(1) = smallest * 0.25F + smallest * 0.25F;
        real(2) = half - smallest * 0.25F;
        complex(0) = Value(half, -half) * Value(1.0F, 0.0F);
      });
  const auto host_real = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, real);
  const auto host_complex =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, complex);
  const float smallest = std::numeric_limits<float>::min();
  EXPECT_EQ(host_real(0), std::ldexp(smallest, -1));
  EXPECT_EQ(host_real(1), std::ldexp(smallest, -1));
  EXPECT_EQ(host_real(2), std::ldexp(smallest, -2));
  EXPECT_EQ(host_complex(0).real(), std::ldexp(smallest, -1));
  EXPECT_EQ(host_complex(0).imag(), -std::ldexp(smallest, -1));
}

} // namespace

// CUDA extended lambdas require a publicly accessible enclosing function.
TEST(PrecisionPolicy, Binary32SubnormalsSurviveAPass) { subnormals_survive_a_pass(); }
