// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/execution.hpp"
#include "solprop_native/graph.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Setup-time checks and copies of a problem's host data, shared by the units that build
// a workspace's views: extents and their size arithmetic, finiteness, and one copy of a
// host vector into a view. Host code, run before any step.
namespace solprop_native {

inline void require_extent(const std::size_t actual, const std::size_t expected,
                           const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string(name) + " has the wrong extent");
}

inline std::size_t checked_product(const std::size_t left, const std::size_t right,
                                   const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left * right;
}

inline std::size_t checked_sum(const std::size_t left, const std::size_t right,
                               const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left + right;
}

inline void require_allocation(const std::size_t elements,
                               const std::size_t element_bytes, const char* name) {
  static_cast<void>(checked_product(elements, element_bytes, name));
}

template <class Value> bool finite(const std::vector<Value>& values) {
  return std::all_of(values.begin(), values.end(), [](const Value& value) {
    if constexpr (std::is_same_v<Value, Complex>)
      return std::isfinite(value.real()) && std::isfinite(value.imag());
    else
      return std::isfinite(value);
  });
}

template <class View, class Value>
void upload(const ExecutionSpace& execution, const View& destination,
            const std::vector<Value>& source) {
  auto mirror = Kokkos::create_mirror_view(destination);
  auto* data = mirror.data();
  for (std::size_t index = 0; index < source.size(); ++index)
    data[index] = source[index];
  graph::deep_copy(execution, destination, mirror);
}

} // namespace solprop_native
