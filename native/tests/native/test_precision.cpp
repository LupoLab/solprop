// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/precision.hpp"

#include <gtest/gtest.h>

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
