// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/*
 * The field tables of the records that cross the seam (solprop SEP 52 section 5.3).
 *
 * Each record declares its fields once, next to its definition, as a function found by
 * argument-dependent lookup:
 *
 *     constexpr auto describe(const Policy*) {
 *       return fields::table<Policy>("march.Policy",
 *                                    fields::value("order", &Policy::order),
 *                                    fields::array("steps", &Policy::steps, ...));
 *     }
 *
 * A field gives its name and its member. A value's conversion follows the member's
 * C++ type (a number, a bool, a string, a sequence, an optional, a nested record, a
 * list of records, a mapping of numbers, a variant of records). An array gives its
 * dtype, its rank and its extents, each written as a literal, as the record's own
 * count, or as a symbol another array's extent binds (the Raman table's
 * `(n_terms, 2, n_to)`). A count field takes its value from a symbol the arrays bound,
 * so that a count an array's extents give is read from the array and never sent beside
 * it. The binding layer's converter (`src/records.hpp`) reads a Python dataclass or
 * mapping into the record and back, and exports the tables (`_core.schemas()`).
 *
 * Python-free and Kokkos-free: descriptors only.
 */
namespace solprop_native::fields {

/** The NumPy dtype an array field takes. */
enum class Dtype : std::uint8_t { float64, complex128, uint64, uint8, boolean };

/** The dtype's NumPy name. */
[[nodiscard]] constexpr const char* dtype_name(const Dtype dtype) noexcept {
  switch (dtype) {
  case Dtype::float64:
    return "float64";
  case Dtype::complex128:
    return "complex128";
  case Dtype::uint64:
    return "uint64";
  case Dtype::uint8:
    return "uint8";
  case Dtype::boolean:
    return "bool";
  }
  return "?";
}

/** The dtype an element type takes by default, beyond the numbers and the bools:
    `core.hpp` adds the complex one. */
template <class T> struct ElementDtype;

template <class T> [[nodiscard]] constexpr Dtype element_dtype() noexcept {
  if constexpr (std::is_same_v<T, double>)
    return Dtype::float64;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return Dtype::boolean;
  else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> &&
                     sizeof(T) == sizeof(std::uint64_t))
    return Dtype::uint64;
  else
    return ElementDtype<T>::value;
}

/** The element an array field stores: a vector's, or an optional vector's (an array
    that may be `None`), or a borrowed span's: a view of the Python array itself, valid
    only while the call that read it runs, for a large array the call consumes at once
    (a run's initial field), so that it is not copied twice. */
template <class Member> struct ArrayElement;
template <class T> struct ArrayElement<std::vector<T>> {
  using type = T;
  static constexpr bool optional = false;
  static constexpr bool borrowed = false;
};
template <class T> struct ArrayElement<std::optional<std::vector<T>>> {
  using type = T;
  static constexpr bool optional = true;
  static constexpr bool borrowed = false;
};
template <class T> struct ArrayElement<std::span<const T>> {
  using type = T;
  static constexpr bool optional = false;
  static constexpr bool borrowed = true;
};

/** One axis of an array field: free (`any`), a literal (`fixed`), or a symbol with an
    offset (`of`), which the first array to name it binds and every other one meets. */
struct Extent {
  const char* symbol = nullptr;
  std::size_t literal = 0;
  std::ptrdiff_t offset = 0;
  bool fixed_size = false;

  [[nodiscard]] static constexpr Extent any() noexcept { return {}; }
  [[nodiscard]] static constexpr Extent fixed(const std::size_t size) noexcept {
    return {.symbol = nullptr, .literal = size, .offset = 0, .fixed_size = true};
  }
  [[nodiscard]] static constexpr Extent of(const char* symbol,
                                           const std::ptrdiff_t offset = 0) noexcept {
    return {.symbol = symbol, .literal = 0, .offset = offset, .fixed_size = false};
  }
};

/** A field whose conversion follows its member's type. */
template <class Record, class Member> struct Value {
  const char* name;
  Member Record::* member;
};

/** An optional value held as a flag beside it: `None` clears the flag. */
template <class Record, class Member> struct Flagged {
  const char* name;
  bool Record::* flag;
  Member Record::* member;
};

/** An array of `Element` stored flat in a vector, C order. `Rank` zero takes any rank,
    and the record keeps the shape in its `shapes` map when it has one. */
template <class Record, class Element, std::size_t Rank> struct Array {
  const char* name;
  Element Record::* member; // a std::vector of the stored element
  Dtype dtype;
  std::array<Extent, Rank> extents;
};

/** A count the arrays' extents give: set from `symbol`, never read from Python. */
template <class Record, class Member> struct Count {
  const char* symbol;
  Member Record::* member;
};

/** A record's table: its name and its fields, in order. */
template <class Record, class... Fields> struct Table {
  using record_type = Record;
  const char* name;
  std::tuple<Fields...> fields;
};

template <class Record, class... Fields>
[[nodiscard]] constexpr Table<Record, Fields...> table(const char* name,
                                                       Fields... entries) {
  return {name, std::tuple<Fields...>{entries...}};
}

template <class Record, class Member>
[[nodiscard]] constexpr Value<Record, Member> value(const char* name,
                                                    Member Record::* member) {
  return {name, member};
}

template <class Record, class Member>
[[nodiscard]] constexpr Flagged<Record, Member>
flagged(const char* name, bool Record::* flag, Member Record::* member) {
  return {name, flag, member};
}

/** An array field of `rank` axes with the given extents (or any rank, none given). */
template <class Record, class Vector, class... Extents>
[[nodiscard]] constexpr Array<Record, Vector, sizeof...(Extents)>
array(const char* name, Vector Record::* member, const Dtype dtype,
      Extents... extents) {
  return {name, member, dtype, {Extent(extents)...}};
}

/** An array field with its element's default dtype. */
template <class Record, class Vector, class... Extents>
[[nodiscard]] constexpr Array<Record, Vector, sizeof...(Extents)>
array(const char* name, Vector Record::* member, Extents... extents) {
  return {name,
          member,
          element_dtype<typename ArrayElement<Vector>::type>(),
          {Extent(extents)...}};
}

template <class Record, class Member>
[[nodiscard]] constexpr Count<Record, Member> count(const char* symbol,
                                                    Member Record::* member) {
  return {symbol, member};
}

/** Whether `Record` has a table (`describe`, found by argument-dependent lookup). */
template <class Record>
concept Described = requires { describe(static_cast<const Record*>(nullptr)); };

/** `Record`'s table. */
template <Described Record> [[nodiscard]] constexpr auto table_of() {
  return describe(static_cast<const Record*>(nullptr));
}

/** Call `visit` on each of `table`'s fields, in order. */
template <class Table, class Visit>
constexpr void each(const Table& table, Visit&& visit) {
  std::apply([&](const auto&... entry) { (visit(entry), ...); }, table.fields);
}

} // namespace solprop_native::fields
