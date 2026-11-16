// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The Adams trial's helpers: the scope guards FlowAside (a coupled flow set aside for
// a stage evaluation, so that the Adams sample is the material source alone) and
// SwappedAside (a trial's prediction standing in the state for its source evaluation
// alone), and a frozen-residual coupled flow's weight row upload (upload_weights) and
// generator (apply_generator). A part of kernels.hpp, included through it.
#ifdef SOLPROP_NATIVE_STRICT_TU
#error "a strict translation unit must not include general/adams_trial.hpp"
#endif

#include "solprop_native/response/coupled_flow.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/graph.hpp"
#include "solprop_native/runtime/launch.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solprop_native::general {

// Sets a coupled flow aside for a stage evaluation and puts it back however the scope
// is left: the Adams sample is the material source alone (solprop SEP 16 phase 5).
class FlowAside {
public:
  explicit FlowAside(std::unique_ptr<CoupledFlowAction>& slot)
      : slot_(slot), held_(std::exchange(slot, nullptr)) {}
  // Restores the slot; clang-tidy 18 misreads it as trivial when GCC 13's intrinsic
  // headers fail to parse.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~FlowAside() { slot_ = std::move(held_); }
  FlowAside(const FlowAside&) = delete;
  FlowAside& operator=(const FlowAside&) = delete;
  FlowAside(FlowAside&&) = delete;
  FlowAside& operator=(FlowAside&&) = delete;

private:
  std::unique_ptr<CoupledFlowAction>& slot_;
  std::unique_ptr<CoupledFlowAction> held_;
};

// Swaps two views for a scope and back however it is left: an Adams trial's
// prediction stands in the state for its source evaluation alone, so that a failure
// there leaves the accepted state in place (solprop issue #129).
template <class View> class SwappedAside {
public:
  SwappedAside(View& first, View& second) : first_(first), second_(second) {
    std::swap(first_, second_);
  }
  ~SwappedAside() { std::swap(first_, second_); }
  SwappedAside(const SwappedAside&) = delete;
  SwappedAside& operator=(const SwappedAside&) = delete;
  SwappedAside(SwappedAside&&) = delete;
  SwappedAside& operator=(SwappedAside&&) = delete;

private:
  View& first_;
  View& second_;
};

// One frozen-residual weight row from the host into a View's row.
template <class Row>
void upload_weights(const ExecutionSpace& space, const Row& out,
                    const std::vector<double>& weights) {
  if (weights.size() < out.extent(0))
    throw std::invalid_argument("a coupled flow's weights are shorter than its terms");
  const Kokkos::View<const double*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(weights.data(), out.extent(0));
  graph::deep_copy(space, out, host);
}

// A frozen-residual flow's generator at one position applied to a field, `sum_t w_t
// G_t u` on the block's bins and zero off them (the terms are zero off the scored
// bins). A zero weight is skipped, which leaves every bit as the full sum has it.
template <class Out>
void apply_generator(const ExecutionSpace& space, const std::size_t n,
                     const std::size_t modes,
                     const Kokkos::View<std::size_t*, Layout, MemorySpace>& slots,
                     const Kokkos::View<Complex****, Layout, MemorySpace>& terms,
                     const Kokkos::View<double*, Layout, MemorySpace>& weights,
                     const Kokkos::View<Complex**, Layout, MemorySpace>& input,
                     const Out& out) {
  const auto n_terms = weights.extent(0);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_for(
      "solprop_native.adams.generator", Range2(space, {0, 0}, {n, modes}),
      KOKKOS_LAMBDA(const std::size_t index, const std::size_t i) {
        const auto one = slots(index);
        Complex value{};
        if (one != 0) {
          const auto bin = one - 1;
          for (std::size_t entry = 0; entry < n_terms; ++entry) {
            const double weight = weights(entry);
            if (weight == 0.0)
              continue;
            for (std::size_t j = 0; j < modes; ++j)
              value += weight * terms(entry, bin, i, j) * input(j, index);
          }
        }
        out(i, index) = value;
      });
#else
  launch::parallel_for(
      "solprop_native.adams.generator",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(const std::size_t index) {
        const auto one = slots(index);
        for (std::size_t i = 0; i < modes; ++i) {
          Complex value{};
          if (one != 0) {
            const auto bin = one - 1;
            for (std::size_t entry = 0; entry < n_terms; ++entry) {
              const double weight = weights(entry);
              if (weight == 0.0)
                continue;
              for (std::size_t j = 0; j < modes; ++j)
                value += weight * terms(entry, bin, i, j) * input(j, index);
            }
          }
          out(i, index) = value;
        }
      });
#endif
}

} // namespace solprop_native::general
