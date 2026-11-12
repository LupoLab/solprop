// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Macros.hpp>

#include <cstddef>
#include <utility>

namespace solprop_native {

/** `function.template operator()<Value>()` for the first of `Values` equal to `value`,
    else `otherwise()`: a run-time extent as a compile-time one, the `switch` ladder
    written once. Device-callable; every branch returns the same type. */
template <std::size_t First, std::size_t... Rest, class Function, class Otherwise>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto) dispatch_among(const std::size_t value,
                                                          const Function& function,
                                                          const Otherwise& otherwise) {
  if (value == First)
    return function.template operator()<First>();
  if constexpr (sizeof...(Rest) == 0)
    return otherwise();
  else
    return dispatch_among<Rest...>(value, function, otherwise);
}

namespace detail {

template <std::size_t First, std::size_t... Offsets, class Function, class Otherwise>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto)
dispatch_width(std::index_sequence<Offsets...> /*offsets*/, const std::size_t width,
               const Function& function, const Otherwise& otherwise) {
  return dispatch_among<(First + Offsets)...>(width, function, otherwise);
}

} // namespace detail

/** `dispatch_among` over the widths `First, First + 1, ..., Last`. */
template <std::size_t First, std::size_t Last, class Function, class Otherwise>
KOKKOS_FORCEINLINE_FUNCTION decltype(auto) dispatch_width(const std::size_t width,
                                                          const Function& function,
                                                          const Otherwise& otherwise) {
  static_assert(First <= Last, "a width range holds at least one width");
  return detail::dispatch_width<First>(std::make_index_sequence<Last - First + 1>{},
                                       width, function, otherwise);
}

} // namespace solprop_native
