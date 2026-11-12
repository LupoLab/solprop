// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace solprop_native {

namespace detail {

template <class Part> void append_part(std::string& text, const Part& part) {
  if constexpr (std::is_arithmetic_v<Part> && !std::is_same_v<Part, bool> &&
                !std::is_same_v<Part, char>)
    text += std::to_string(part);
  else
    text += std::string_view(part);
}

} // namespace detail

/** `parts` concatenated: text as given, numbers as `std::to_string` writes them (a
    double in `%f`), so that a message built from them keeps its exact text. */
template <class... Parts> [[nodiscard]] std::string concat(const Parts&... parts) {
  std::string text;
  (detail::append_part(text, parts), ...);
  return text;
}

/** Throw `Error` with the message `concat(parts...)`. */
template <class Error = std::invalid_argument, class... Parts>
[[noreturn]] void refuse(const Parts&... parts) {
  throw Error(concat(parts...));
}

} // namespace solprop_native
