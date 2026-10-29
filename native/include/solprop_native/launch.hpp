// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <utility>

/** The backend's top-level dispatches: `Kokkos::parallel_for`, `parallel_reduce` and
    `parallel_scan`, with the kernel's functor passed as its launch argument on a
   device.

    Kokkos's HIP backend copies a functor of 512 bytes to 32 kB into constant memory
    before its launch, through a host-to-device copy (a blit kernel on an AMD Instinct
    device) behind an event wait for the previous such launch, so that the host cannot
    queue a kernel ahead of one still reading the buffer. Most of a trial's kernels
    capture that much, and on an AMD Instinct MI325X these waits and blits left the
    device idle between kernels of a few microseconds. With
    `WorkItemProperty::HintLightWeight` a functor under the 4 kB kernel-argument limit
    goes as the kernel's argument instead, which needs neither. A functor near the limit
    keeps Kokkos's choice: above it a light-weight launch would stage the functor in
    global memory behind a stream synchronisation. CUDA builds that pass functors as
    grid constants already launch this way, and the hint changes nothing there. Host
    builds dispatch exactly as Kokkos does. */
namespace solprop_native::launch {

/** The largest functor given the light-weight hint, in bytes: the 4 kB argument limit
    less room for the policy and the dispatch's own members. */
inline constexpr std::size_t argument_functor_limit = 3840;

/** `policy`, hinted light-weight on a device build when `Functor` fits as an argument.
 */
template <typename Functor, typename Policy>
[[nodiscard]] auto policy_for(const Policy& policy) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (Kokkos::is_execution_policy_v<Policy> &&
                sizeof(Functor) <= argument_functor_limit)
    return Kokkos::Experimental::require(
        policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  else
    return policy;
#else
  return policy;
#endif
}

template <typename Label, typename Policy, typename Functor>
void parallel_for(const Label& label, const Policy& policy, const Functor& functor) {
  Kokkos::parallel_for(label, policy_for<Functor>(policy), functor);
}

template <typename Label, typename Policy, typename Functor, typename... Results>
void parallel_reduce(const Label& label, const Policy& policy, const Functor& functor,
                     Results&&... results) {
  Kokkos::parallel_reduce(label, policy_for<Functor>(policy), functor,
                          std::forward<Results>(results)...);
}

template <typename Label, typename Policy, typename Functor, typename... Results>
void parallel_scan(const Label& label, const Policy& policy, const Functor& functor,
                   Results&&... results) {
  Kokkos::parallel_scan(label, policy_for<Functor>(policy), functor,
                        std::forward<Results>(results)...);
}

} // namespace solprop_native::launch
