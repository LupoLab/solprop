// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/gemm.hpp"
#include "solprop_native/precision.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
#include <KokkosBlas_tpl_spec.hpp>
#endif

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * The vendor BLAS's math mode under a reduced policy, set and probed (solprop SEP 36
 * section 6.3).
 *
 * cuBLAS computes binary32 products in TF32 where the handle's math mode allows it, and
 * rocBLAS in xf32 where its mode asks for it. Native's library products all go through
 * Kokkos Kernels' TPL layer, which issues them on its process-wide handle
 * (`KokkosBlas::Impl::CudaBlasSingleton`, `RocBlasSingleton`). A reduced-precision
 * workspace sets that handle's mode to cuBLAS's pedantic math (rocBLAS's default, not
 * xf32) at construction, then probes each product family the run issues against a
 * binary64 host reference: a product beyond `gemm_probe_bound` refuses the run.
 */

namespace solprop_native {

/** A reduced-precision matrix product the vendor BLAS computes below binary32. */
class GemmProbeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** The normwise relative error a binary32 product must stay below (SEP 33 7.1). */
inline constexpr double gemm_probe_bound = 0x1p-17;

/** The math mode a reduced-precision workspace sets on the vendor handle. */
enum class BlasMathMode { pedantic, reduced };

/**
 * The mode `set_blas_math_mode` sets: pedantic, unless a test selects the reduced mode
 * (TF32, xf32) to see the probe refuse it.
 */
inline BlasMathMode& blas_math_mode_selection() {
  static BlasMathMode mode = BlasMathMode::pedantic;
  return mode;
}

/** Whether this build issues products on a vendor handle whose mode can be set. */
[[nodiscard]] constexpr bool blas_math_mode_settable() noexcept {
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS) || defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  return true;
#else
  return false;
#endif
}

/**
 * Set the selected math mode on Kokkos Kernels' vendor handle; a host build's BLAS has
 * no reduced mode and is left alone.
 */
inline void set_blas_math_mode() {
  [[maybe_unused]] const bool pedantic =
      blas_math_mode_selection() == BlasMathMode::pedantic;
#if defined(SOLPROP_NATIVE_TPL_BLAS_CUBLAS)
  auto& blas = KokkosBlas::Impl::CudaBlasSingleton::singleton();
  if (cublasSetMathMode(blas.handle,
                        pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_TF32_TENSOR_OP_MATH) !=
      CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasSetMathMode failed on Kokkos Kernels' handle");
#elif defined(SOLPROP_NATIVE_TPL_BLAS_ROCBLAS)
  auto& blas = KokkosBlas::Impl::RocBlasSingleton::singleton();
  if (rocblas_set_math_mode(blas.handle, pedantic ? rocblas_default_math
                                                  : rocblas_xf32_xdl_math_op) !=
      rocblas_status_success)
    throw std::runtime_error("rocblas_set_math_mode failed on Kokkos Kernels' handle");
#endif
}

/** One product family a run issues: its scalar and its shape, `m x k` by `k x n`. */
struct GemmFamily {
  bool complex{};
  std::size_t rows{}, inner{}, columns{};
};

/**
 * One family's binary32 product of fixed-seed Gaussian factors through `selected_gemm`
 * against the binary64 host product of the same rounded factors: the normwise relative
 * error.
 */
template <class Real>
double gemm_probe_error(const ExecutionSpace& execution_space, const GemmFamily& family,
                        const std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  std::normal_distribution<double> normal;
  const auto m = family.rows, k = family.inner, n = family.columns;
  const auto draw = [&] { return static_cast<Real>(normal(engine)); };
  std::vector<std::complex<double>> a(m * k), b(k * n);
  const auto product = [&](const auto& c_host) {
    double difference = 0.0, reference = 0.0;
    for (std::size_t row = 0; row < m; ++row)
      for (std::size_t column = 0; column < n; ++column) {
        std::complex<double> exact{};
        for (std::size_t index = 0; index < k; ++index)
          exact += a[row * k + index] * b[index * n + column];
        const std::complex<double> found = c_host(row, column);
        difference += std::norm(found - exact);
        reference += std::norm(exact);
      }
    return std::sqrt(difference / reference);
  };
  if (family.complex) {
    using Value = Kokkos::complex<Real>;
    Kokkos::View<Value**, Layout, MemorySpace> left("solprop_native.probe.a", m, k),
        right("solprop_native.probe.b", k, n), out("solprop_native.probe.c", m, n);
    auto host_left = Kokkos::create_mirror_view(left);
    auto host_right = Kokkos::create_mirror_view(right);
    for (std::size_t index = 0; index < m * k; ++index) {
      const Value value(draw(), draw());
      host_left.data()[index] = value;
      a[index] = {static_cast<double>(value.real()), static_cast<double>(value.imag())};
    }
    for (std::size_t index = 0; index < k * n; ++index) {
      const Value value(draw(), draw());
      host_right.data()[index] = value;
      b[index] = {static_cast<double>(value.real()), static_cast<double>(value.imag())};
    }
    Kokkos::deep_copy(execution_space, left, host_left);
    Kokkos::deep_copy(execution_space, right, host_right);
    selected_gemm(execution_space, left, right, out, Value{}, true);
    const auto host_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    return product([&](const std::size_t row, const std::size_t column) {
      const auto value = host_out(row, column);
      return std::complex<double>(static_cast<double>(value.real()),
                                  static_cast<double>(value.imag()));
    });
  }
  Kokkos::View<Real**, Layout, MemorySpace> left("solprop_native.probe.a", m, k),
      right("solprop_native.probe.b", k, n), out("solprop_native.probe.c", m, n);
  auto host_left = Kokkos::create_mirror_view(left);
  auto host_right = Kokkos::create_mirror_view(right);
  for (std::size_t index = 0; index < m * k; ++index) {
    host_left.data()[index] = draw();
    a[index] = static_cast<double>(host_left.data()[index]);
  }
  for (std::size_t index = 0; index < k * n; ++index) {
    host_right.data()[index] = draw();
    b[index] = static_cast<double>(host_right.data()[index]);
  }
  Kokkos::deep_copy(execution_space, left, host_left);
  Kokkos::deep_copy(execution_space, right, host_right);
  selected_gemm(execution_space, left, right, out, Real{}, true);
  const auto host_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  return product([&](const std::size_t row, const std::size_t column) {
    return std::complex<double>(static_cast<double>(host_out(row, column)), 0.0);
  });
}

/**
 * Probe every family; refuse, naming the family and the observed error, a product
 * beyond `gemm_probe_bound`. Returns the largest error. The buffers are released on
 * return, before the march.
 */
template <class Real>
double probe_gemm_families(const ExecutionSpace& execution_space,
                           const std::vector<GemmFamily>& families) {
  double worst = 0.0;
  std::uint64_t seed = 36;
  for (const auto& family : families) {
    const double error = gemm_probe_error<Real>(execution_space, family, seed++);
    worst = error > worst ? error : worst;
    if (!(error <= gemm_probe_bound)) {
      char observed[64];
      std::snprintf(observed, sizeof observed, "%.3e", error);
      throw GemmProbeError(
          std::string("the vendor BLAS computed a ") +
          (family.complex ? "complex" : "real") + " binary32 product of " +
          std::to_string(family.rows) + "x" + std::to_string(family.inner) + " by " +
          std::to_string(family.inner) + "x" + std::to_string(family.columns) +
          " to a relative error of " + observed +
          " against binary64, beyond 2^-17: the run is refused (solprop SEP 36 section "
          "6.3). A reduced math mode (TF32, xf32) on the BLAS handle gives errors of "
          "this size; on CUDA, NVIDIA_TF32_OVERRIDE=0 is one remedy");
    }
  }
  return worst;
}

} // namespace solprop_native
