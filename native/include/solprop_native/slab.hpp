// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/core.hpp"
#include "solprop_native/general_modal.hpp"

#include <cstddef>

namespace solprop_native {

/**
 * Apply the hybrid planar kind's real slab rule in x to whole y rows, in time.
 *
 * `target(u * out + o, t) = sum_i matrix(o, i) source(u * in + i, t)` over `units`
 * blocks: the slab synthesis (matrix `(n_x, n_p)`, in = n_p, out = n_x) and
 * projection (`(n_p, n_x)`, in = n_x, out = n_p) of whole y rows. One team per output
 * row zeroes it and accumulates each term in increasing i, a zero and then each
 * product as solprop's reference sums, with the vector loop over contiguous samples;
 * the Views are captured by value as unmanaged copies, so the host compiler sees the
 * loop's loads and stores leave them unmoved and vectorises it (the general modal
 * source's pattern). Each lane owns its samples in both loops, so the width moves no
 * bit. The engine's planar stage and SEP 6's dispersive response share it.
 */
template <class Source, class Target>
void slab_contract(const char* label, const ExecutionSpace& execution,
                   const Kokkos::View<double**, Layout, MemorySpace>& matrix,
                   const Source& source, const Target& target, const std::size_t units,
                   const std::size_t n_time) {
  using Value = Target::non_const_value_type;
  using SourceValue = Source::non_const_value_type;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  const Kokkos::View<const double**, Layout, MemorySpace, Unmanaged> weights = matrix;
  const Kokkos::View<const SourceValue**, Layout, MemorySpace, Unmanaged> input =
      source;
  const Kokkos::View<Value**, Layout, MemorySpace, Unmanaged> output = target;
  const auto out = matrix.extent(0);
  const auto in = matrix.extent(1);
  const auto samples = static_cast<int>(n_time);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int lanes = TeamPolicy::vector_length_max();
#else
  const int lanes = static_cast<int>(general_vector_lanes);
#endif
  launch::parallel_for(
      label, TeamPolicy(execution, static_cast<int>(units * out), 1, lanes),
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank());
        const auto first = (row / out) * in;
        const auto local = row % out;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, samples),
                             [=](const int time) { output(row, time) = Value{}; });
        for (std::size_t index = 0; index < in; ++index) {
          const double weight = weights(local, index);
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, samples), [=](const int time) {
                output(row, time) += weight * input(first + index, time);
              });
        }
      });
}

} // namespace solprop_native
