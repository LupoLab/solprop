// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The field-table converter between Python records and the C++ records of `fields.hpp`
// (solprop SEP 52 section 5.3), as one set of templates: `read` a dataclass, mapping or
// tuple into a new record, `write_record` and `write_value` a record back as a
// dictionary, and `collect` each table's schema for `_core.schemas()`. The note on the
// namespace below gives what reading checks and how a refusal names its field.
//
// Types: `Source` (the fields a Python record offers), `Borrowed` (a checked array),
// `Symbols` (the symbols the extents bind), `Extern` (the hook for a bound type a table
// names) and the trait `FieldKind`. The non-template helpers are in records.cpp.

#include "solprop_native/runtime/fields.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/*
 * The field-table converter (solprop SEP 52 section 5.3): one template that reads a
 * Python dataclass, a mapping or a tuple into a C++ record by its table
 * (`solprop_native/runtime/fields.hpp`) and writes the record back as a dictionary, and
 * the schema export of every table.
 *
 * Reading takes one pass over the input, under the interpreter lock, into a new
 * record: each field is checked -- a missing or unknown field, a wrong type or dtype,
 * an array that is not C-contiguous or overlaps itself, a wrong rank, extents that
 * disagree with their relations, a size whose product overflows -- before its value is
 * copied, and a refusal discards the record, so nothing reaches native state from a
 * malformed one. A dataclass type's field names are read once and held, and each field
 * is looked up by an interned name, since a run's records cross at every start. Inputs
 * are borrowed for the call:
 * read-only arrays are admitted, since nothing writes them. A refusal names the
 * field's path (`run.records.planes.planes`). Arrays written back own their storage
 * through a capsule.
 */
namespace solprop_native::records {

namespace nb = nanobind;

/** The symbols an array's extents bind within one record, by name. */
using Symbols = std::map<std::string, std::size_t>;

/** How a binding reads a type the tables name but cannot describe (a bound object such
    as a statistics plan); specialised before use. */
template <class T> struct Extern;

// ------------------------------------------------------------- non-template helpers

/** Refuse with Python's `TypeError`, naming `path`. */
[[noreturn]] void type_refusal(const std::string& path, const std::string& message);
/** Refuse with Python's `ValueError`, naming `path`. */
[[noreturn]] void value_refusal(const std::string& path, const std::string& message);
/** A child's path. */
[[nodiscard]] std::string child(const std::string& path, const std::string& name);
[[nodiscard]] std::string child(const std::string& path, std::size_t index);

/** The fields a Python record offers: a dataclass's, a mapping's keys or a tuple's
    positions. */
class Source {
public:
  Source(nb::handle value, const std::string& path);
  /** Whether a field of this name (at this position) is given. */
  [[nodiscard]] bool has(const char* name, std::size_t index) const;
  [[nodiscard]] nb::object get(const char* name, std::size_t index) const;
  /** The given names (empty for a tuple). */
  [[nodiscard]] const std::vector<std::string>& names() const { return *names_; }
  /** Whether the input is a dataclass instance, whose type's fields were read once. */
  [[nodiscard]] bool dataclass() const { return dataclass_; }
  /** The input's type. */
  [[nodiscard]] PyTypeObject* type() const { return Py_TYPE(value_.ptr()); }
  [[nodiscard]] bool positional() const { return positional_; }
  [[nodiscard]] std::size_t size() const { return size_; }

private:
  nb::handle value_;
  bool positional_ = false;
  bool mapping_ = false;
  std::size_t size_ = 0;
  bool dataclass_ = false;
  // A dataclass type's names, held for the process; a mapping's, owned here.
  const std::vector<std::string>* names_ = &owned_;
  std::vector<std::string> owned_;
};

/** A checked array: its data, shape and dtype, borrowed for the conversion. */
struct Borrowed {
  const void* data = nullptr;
  std::vector<std::size_t> shape;
  std::size_t count = 0;
};

/** Check `value` as a C-contiguous CPU array of `dtype` (`rank` axes, or any rank for
    zero) whose element count does not overflow; refuse naming `path`. */
[[nodiscard]] Borrowed borrow(nb::handle value, fields::Dtype dtype, std::size_t rank,
                              const std::string& path);
/** Hold `shape` to `extents`, binding the symbols they name the first time. */
void relate(const std::vector<std::size_t>& shape, const fields::Extent* extents,
            std::size_t rank, Symbols& symbols, const std::string& path);
/** One extent as text: `n_terms`, `n_edges-1`, `2` or `*`. */
[[nodiscard]] std::string extent_text(const fields::Extent& extent);
/** A NumPy array owning a copy of `bytes` bytes of `dtype`, of `shape`. */
[[nodiscard]] nb::object owned_array(const void* data, std::size_t bytes,
                                     fields::Dtype dtype,
                                     const std::vector<std::size_t>& shape);

[[nodiscard]] bool is_bool(nb::handle value);
[[nodiscard]] bool is_integer(nb::handle value);
[[nodiscard]] bool is_number(nb::handle value);
[[nodiscard]] bool is_sequence(nb::handle value);
[[nodiscard]] bool is_mapping(nb::handle value);

// ------------------------------------------------------------------ type traits

template <class T> struct IsVector : std::false_type {};
template <class T> struct IsVector<std::vector<T>> : std::true_type {};
template <class T> struct IsVector<std::deque<T>> : std::true_type {};
template <class T> struct IsOptional : std::false_type {};
template <class T> struct IsOptional<std::optional<T>> : std::true_type {};
template <class T> struct IsStdArray : std::false_type {};
template <class T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};
template <class T> struct IsVariant : std::false_type {};
template <class... T> struct IsVariant<std::variant<T...>> : std::true_type {};
template <class T> struct IsNumberMap : std::false_type {};
template <> struct IsNumberMap<std::map<std::string, double>> : std::true_type {};

template <class T>
concept Named =
    std::is_enum_v<T> && requires { names(static_cast<const T*>(nullptr)); };
template <class T>
concept HasExtern = requires { Extern<T>::kind; };
template <class T>
concept Presence = requires(const T& value) {
  { value.present() } -> std::convertible_to<bool>;
};
template <class R>
concept HasShapes = requires(R& record) { record.shapes; };

template <class Field> struct FieldKind;
template <class R, class M> struct FieldKind<fields::Value<R, M>> {
  static constexpr int kind = 0;
};
template <class R, class M> struct FieldKind<fields::Flagged<R, M>> {
  static constexpr int kind = 1;
};
template <class R, class M, std::size_t N> struct FieldKind<fields::Array<R, M, N>> {
  static constexpr int kind = 2;
};
template <class R, class M> struct FieldKind<fields::Count<R, M>> {
  static constexpr int kind = 3;
};

// --------------------------------------------------------------------- reading

template <class R> void read_record(nb::handle value, R* out, const std::string& path);

/** The names a record's table reads from Python (its counts are not read). */
template <class R> std::vector<const char*> table_names() {
  std::vector<const char*> out;
  fields::each(fields::table_of<R>(), [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    if constexpr (FieldKind<Field>::kind != 3)
      out.push_back(field.name);
  });
  return out;
}

/** Whether a dataclass type was already found to name exactly the fields of the table
    named `table`, a pointer that identifies it (one conversion checks each pair once).
 */
[[nodiscard]] bool checked(PyTypeObject* type, const char* table);
void mark_checked(PyTypeObject* type, const char* table);

/** Whether `source` names exactly `R`'s fields (every one, nothing else). */
template <class R> bool matches(const Source& source) {
  const auto wanted = table_names<R>();
  if (source.positional())
    return source.size() == wanted.size();
  if (source.names().size() != wanted.size())
    return false;
  return std::ranges::all_of(
      wanted, [&source](const char* name) { return source.has(name, 0); });
}

template <class T> void read_value(nb::handle value, T* out, const std::string& path);

template <class... Alternatives, class Variant>
void read_variant(nb::handle value, Variant* out, const std::string& path) {
  const Source source(value, path);
  bool found = false;
  const auto attempt = [&]<class A> {
    if (found || !matches<A>(source))
      return;
    found = true;
    if (out == nullptr) {
      read_record<A>(value, static_cast<A*>(nullptr), path);
    } else {
      A held;
      read_record<A>(value, &held, path);
      *out = std::move(held);
    }
  };
  (attempt.template operator()<Alternatives>(), ...);
  if (!found) {
    std::string names;
    ((names +=
      std::string(names.empty() ? "" : " or ") + fields::table_of<Alternatives>().name),
     ...);
    type_refusal(path, "matches none of " + names + " by its fields");
  }
}

template <class T, class Variant> struct VariantReader;
template <class Variant, class... A> struct VariantReader<Variant, std::variant<A...>> {
  static void read(nb::handle value, Variant* out, const std::string& path) {
    read_variant<A...>(value, out, path);
  }
};

template <class T> void read_value(nb::handle value, T* out, const std::string& path) {
  if constexpr (std::is_same_v<T, bool>) {
    if (!is_bool(value))
      type_refusal(path, "must be a bool");
    if (out != nullptr)
      *out = nb::cast<bool>(value, true);
  } else if constexpr (Named<T>) {
    constexpr auto table = names(static_cast<const T*>(nullptr));
    std::optional<T> held;
    nb::handle name = value;
    nb::object attribute;
    if (!nb::isinstance<nb::str>(value) && nb::hasattr(value, "name")) {
      attribute = value.attr("name");
      name = attribute;
    }
    // By name only (SEP 52 section 5.3): an integer would tie the Python side to the
    // C++ enumerators' values.
    if (nb::isinstance<nb::str>(name)) {
      const auto text = nb::cast<std::string>(name);
      for (const auto& [label, entry] : table)
        if (text == label)
          held = entry;
    }
    if (!held) {
      std::string labels;
      for (const auto& [label, entry] : table)
        labels += std::string(labels.empty() ? "" : ", ") + label;
      value_refusal(path, "must name one of " + labels);
    }
    if (out != nullptr)
      *out = *held;
  } else if constexpr (std::is_integral_v<T>) {
    if (!is_integer(value) || is_bool(value))
      type_refusal(path, "must be an integer");
    T held{};
    if (!nb::try_cast<T>(value, held, true))
      value_refusal(path, "is out of range");
    if (out != nullptr)
      *out = held;
  } else if constexpr (std::is_floating_point_v<T>) {
    if (!is_number(value) || is_bool(value))
      type_refusal(path, "must be a number");
    if (out != nullptr)
      *out = nb::cast<T>(value, true);
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (!nb::isinstance<nb::str>(value))
      type_refusal(path, "must be a str");
    if (out != nullptr)
      *out = nb::cast<std::string>(value);
  } else if constexpr (IsOptional<T>::value) {
    // nvcc still needs the `typename` that C++20 makes redundant on the dependent type.
    // NOLINTNEXTLINE(readability-redundant-typename)
    using Inner = typename T::value_type;
    if (value.is_none()) {
      if (out != nullptr)
        out->reset();
      return;
    }
    if (out == nullptr) {
      read_value<Inner>(value, nullptr, path);
    } else {
      Inner held{};
      read_value<Inner>(value, &held, path);
      *out = std::move(held);
    }
  } else if constexpr (HasExtern<T>) {
    Extern<T>::read(value, out, path);
  } else if constexpr (IsVariant<T>::value) {
    VariantReader<T, T>::read(value, out, path);
  } else if constexpr (IsNumberMap<T>::value) {
    if (!is_mapping(value))
      type_refusal(path, "must be a mapping of names to numbers");
    PyObject* listed = PyMapping_Items(value.ptr());
    if (listed == nullptr)
      throw nb::python_error();
    const auto items = nb::steal<nb::list>(listed);
    if (out != nullptr)
      out->clear();
    for (const auto item : items) {
      const auto pair = nb::borrow<nb::tuple>(item);
      if (!nb::isinstance<nb::str>(pair[0]))
        type_refusal(path, "must have str keys");
      const auto key = nb::cast<std::string>(pair[0]);
      double number = 0.0;
      read_value<double>(pair[1], &number, child(path, key));
      if (out != nullptr)
        (*out)[key] = number;
    }
  } else if constexpr (IsStdArray<T>::value || std::is_array_v<T>) {
    using Element = std::remove_cvref_t<decltype(std::declval<T&>()[0])>;
    constexpr std::size_t size = sizeof(T) / sizeof(Element);
    if (!is_sequence(value))
      type_refusal(path, "must be a sequence of " + std::to_string(size));
    const auto items = nb::borrow<nb::sequence>(value);
    if (nb::len(items) != size)
      value_refusal(path, "must hold " + std::to_string(size) + " values");
    for (std::size_t index = 0; index < size; ++index)
      read_value<Element>(items[index], out == nullptr ? nullptr : &(*out)[index],
                          child(path, index));
  } else if constexpr (IsVector<T>::value) {
    // nvcc still needs the `typename` that C++20 makes redundant on the dependent type.
    // NOLINTNEXTLINE(readability-redundant-typename)
    using Element = typename T::value_type;
    if (!is_sequence(value))
      type_refusal(path, "must be a sequence");
    const auto items = nb::borrow<nb::sequence>(value);
    const std::size_t size = nb::len(items);
    if (out != nullptr)
      out->clear();
    for (std::size_t index = 0; index < size; ++index) {
      if (out == nullptr) {
        read_value<Element>(items[index], nullptr, child(path, index));
      } else {
        Element held{};
        read_value<Element>(items[index], &held, child(path, index));
        out->push_back(std::move(held));
      }
    }
  } else if constexpr (fields::Described<T>) {
    // A table with a notion of absence (a stage table) reads `None` as absent.
    if constexpr (Presence<T>) {
      if (value.is_none()) {
        if (out != nullptr)
          *out = T{};
        return;
      }
    }
    read_record<T>(value, out, path);
  } else {
    static_assert(sizeof(T) == 0, "a field type the converter cannot read");
  }
}

template <class R, class M, std::size_t N>
void read_array(const fields::Array<R, M, N>& field, nb::handle value, R* out,
                Symbols& symbols, const std::string& path) {
  using Traits = fields::ArrayElement<M>;
  // nvcc still needs the `typename` that C++20 makes redundant on the dependent type.
  // NOLINTNEXTLINE(readability-redundant-typename)
  using Element = typename Traits::type;
  if constexpr (Traits::optional) {
    if (value.is_none()) {
      if (out != nullptr)
        (out->*field.member).reset();
      return;
    }
  }
  const auto borrowed = borrow(value, field.dtype, N, path);
  if constexpr (N > 0)
    relate(borrowed.shape, field.extents.data(), N, symbols, path);
  if (out == nullptr)
    return;
  if constexpr (Traits::borrowed) {
    // A view of the caller's array, which it keeps alive for the call.
    out->*field.member = std::span<const Element>(
        static_cast<const Element*>(borrowed.data), borrowed.count);
    return;
  }
  std::vector<Element> held(borrowed.count);
  static_assert(std::is_trivially_copyable_v<Element>);
  if (borrowed.count != 0)
    std::memcpy(static_cast<void*>(held.data()), borrowed.data,
                borrowed.count * sizeof(Element));
  out->*field.member = std::move(held);
  if constexpr (N == 0 && HasShapes<R>)
    out->shapes[field.name] = borrowed.shape;
}

/** Bind an integral field's value as the symbol of its name, for the arrays. */
template <class T>
void bind_count(const char* name, const T& value, Symbols& symbols,
                const std::string& path) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if (value < 0)
      return;
    const auto count = static_cast<std::size_t>(value);
    const auto found = symbols.find(name);
    if (found != symbols.end() && found->second != count)
      value_refusal(path, "is " + std::to_string(count) + ", and the arrays give " +
                              std::to_string(found->second));
    symbols[name] = count;
  }
}

template <class R> void read_record(nb::handle value, R* out, const std::string& path) {
  const Source source(value, path);
  constexpr auto table = fields::table_of<R>();
  const auto wanted = table_names<R>();
  if (source.positional()) {
    if (source.size() != wanted.size())
      value_refusal(path, std::string("is a ") + table.name + " of " +
                              std::to_string(wanted.size()) + " fields, given " +
                              std::to_string(source.size()));
  } else if (!source.dataclass() || !checked(source.type(), table.name)) {
    for (const auto& name : source.names()) {
      bool known = false;
      for (const char* entry : wanted)
        known = known || name == entry;
      if (!known)
        value_refusal(path, std::string("has an unknown field '") + name + "' for a " +
                                table.name);
    }
    if (source.dataclass())
      mark_checked(source.type(), table.name);
  }
  Symbols symbols;
  std::size_t index = 0;
  fields::each(table, [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    constexpr int kind = FieldKind<Field>::kind;
    if constexpr (kind == 3) {
      return;
    } else {
      const std::size_t position = index++;
      const auto where = child(path, field.name);
      if (!source.has(field.name, position)) {
        // An optional field may be left out of a mapping.
        if constexpr (kind == 1) {
          if (out != nullptr)
            out->*field.flag = false;
          return;
        } else if constexpr (kind == 0) {
          using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
          if constexpr (IsOptional<M>::value) {
            if (out != nullptr)
              (out->*field.member).reset();
            return;
          }
        } else if constexpr (kind == 2) {
          using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
          if constexpr (fields::ArrayElement<M>::optional) {
            if (out != nullptr)
              (out->*field.member).reset();
            return;
          }
        }
        value_refusal(where, std::string("is missing from a ") + table.name);
      }
      const auto item = source.get(field.name, position);
      if constexpr (kind == 0) {
        using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
        if (out == nullptr) {
          if constexpr (std::is_integral_v<M> && !std::is_same_v<M, bool>) {
            M held{};
            read_value<M>(item, &held, where);
            bind_count(field.name, held, symbols, where);
          } else {
            read_value<M>(item, static_cast<M*>(nullptr), where);
          }
        } else {
          read_value<M>(item, &(out->*field.member), where);
          bind_count(field.name, out->*field.member, symbols, where);
        }
      } else if constexpr (kind == 1) {
        using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
        if (item.is_none()) {
          if (out != nullptr) {
            out->*field.flag = false;
            out->*field.member = M{};
          }
        } else {
          read_value<M>(item, out == nullptr ? nullptr : &(out->*field.member), where);
          if (out != nullptr)
            out->*field.flag = true;
        }
      } else {
        read_array(field, item, out, symbols, where);
      }
    }
  });
  if (out == nullptr)
    return;
  fields::each(table, [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    if constexpr (FieldKind<Field>::kind == 3) {
      const auto found = symbols.find(field.symbol);
      using M = std::remove_cvref_t<decltype(out->*field.member)>;
      out->*field.member = found == symbols.end() ? M{} : static_cast<M>(found->second);
    }
  });
}

/** Read `value` into a new `R`, which a refusal discards. */
template <class R> R read(nb::handle value, const std::string& path) {
  R out{};
  read_value<R>(value, &out, path);
  return out;
}

// --------------------------------------------------------------------- writing

template <class R> nb::dict write_record(const R& record);

template <class T> nb::object write_value(const T& value) {
  if constexpr (std::is_same_v<T, bool>) {
    return nb::bool_(value);
  } else if constexpr (Named<T>) {
    constexpr auto table = names(static_cast<const T*>(nullptr));
    for (const auto& [label, entry] : table)
      if (entry == value)
        return nb::str(label);
    return nb::int_(static_cast<long long>(value));
  } else if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T> ||
                       std::is_same_v<T, std::string>) {
    return nb::cast(value);
  } else if constexpr (IsOptional<T>::value) {
    return value ? write_value(*value) : nb::none();
  } else if constexpr (HasExtern<T>) {
    return Extern<T>::write(value);
  } else if constexpr (IsVariant<T>::value) {
    return std::visit([](const auto& held) -> nb::object { return write_value(held); },
                      value);
  } else if constexpr (IsNumberMap<T>::value) {
    nb::dict out;
    for (const auto& [key, number] : value)
      out[key.c_str()] = number;
    return out;
  } else if constexpr (IsStdArray<T>::value || std::is_array_v<T> ||
                       IsVector<T>::value) {
    nb::list out;
    for (const auto& item : value)
      out.append(write_value(item));
    return out;
  } else if constexpr (fields::Described<T>) {
    if constexpr (Presence<T>) {
      if (!value.present())
        return nb::none();
    }
    return write_record(value);
  } else {
    static_assert(sizeof(T) == 0, "a field type the converter cannot write");
  }
}

/** The symbols a record's counts, integral fields and arrays give back. */
template <class R> Symbols written_symbols(const R& record) {
  Symbols symbols;
  constexpr auto table = fields::table_of<R>();
  fields::each(table, [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    constexpr int kind = FieldKind<Field>::kind;
    if constexpr (kind == 3) {
      symbols[field.symbol] = static_cast<std::size_t>(record.*field.member);
    } else if constexpr (kind == 0) {
      using M = std::remove_cvref_t<decltype(record.*field.member)>;
      if constexpr (std::is_integral_v<M> && !std::is_same_v<M, bool>)
        symbols[field.name] = static_cast<std::size_t>(record.*field.member);
    }
  });
  // An array with one axis left to resolve gives it, twice round for chains.
  for (int round = 0; round < 2; ++round)
    fields::each(table, [&](const auto& field) {
      using Field = std::remove_cvref_t<decltype(field)>;
      if constexpr (FieldKind<Field>::kind == 2) {
        constexpr std::size_t rank = std::tuple_size_v<decltype(field.extents)>;
        if constexpr (rank > 0) {
          const auto* values = &(record.*field.member);
          std::size_t size = 0;
          if constexpr (fields::ArrayElement<
                            std::remove_cvref_t<decltype(*values)>>::optional) {
            if (!*values)
              return;
            size = (*values)->size();
          } else {
            size = values->size();
          }
          std::size_t known = 1;
          const fields::Extent* open = nullptr;
          for (const auto& extent : field.extents) {
            if (extent.fixed_size) {
              known *= extent.literal;
            } else if (extent.symbol != nullptr && symbols.contains(extent.symbol)) {
              known *= static_cast<std::size_t>(
                  static_cast<std::ptrdiff_t>(symbols[extent.symbol]) + extent.offset);
            } else if (open == nullptr && extent.symbol != nullptr) {
              open = &extent;
            } else {
              return;
            }
          }
          if (open != nullptr && known != 0 && size % known == 0)
            symbols[open->symbol] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(size / known) - open->offset);
        }
      }
    });
  return symbols;
}

template <class R, class M, std::size_t N>
nb::object write_array(const fields::Array<R, M, N>& field, const R& record,
                       const Symbols& symbols) {
  using Traits = fields::ArrayElement<M>;
  std::span<const typename Traits::type> held;
  if constexpr (Traits::optional) {
    if (!(record.*field.member))
      return nb::none();
    held = (record.*field.member).value();
  } else {
    held = record.*field.member;
  }
  const auto* values = &held;
  std::vector<std::size_t> shape{values->size()};
  if constexpr (N == 0 && HasShapes<R>) {
    const auto found = record.shapes.find(field.name);
    if (found != record.shapes.end())
      shape = found->second;
  } else if constexpr (N > 0) {
    std::vector<std::size_t> resolved;
    std::size_t product = 1;
    bool known = true;
    for (const auto& extent : field.extents) {
      std::size_t size = 0;
      if (extent.fixed_size) {
        size = extent.literal;
      } else if (extent.symbol != nullptr && symbols.contains(extent.symbol)) {
        size = static_cast<std::size_t>(
            static_cast<std::ptrdiff_t>(symbols.at(extent.symbol)) + extent.offset);
      } else {
        known = false;
        break;
      }
      resolved.push_back(size);
      product *= size;
    }
    if (known && product == values->size())
      shape = std::move(resolved);
  }
  std::size_t product = 1;
  for (const auto size : shape)
    product *= size;
  if (product != values->size())
    shape = {values->size()};
  return owned_array(values->data(), values->size() * sizeof(typename Traits::type),
                     field.dtype, shape);
}

template <class R> nb::dict write_record(const R& record) {
  nb::dict out;
  const auto symbols = written_symbols(record);
  fields::each(fields::table_of<R>(), [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    constexpr int kind = FieldKind<Field>::kind;
    if constexpr (kind == 0) {
      out[field.name] = write_value(record.*field.member);
    } else if constexpr (kind == 1) {
      out[field.name] =
          record.*field.flag ? write_value(record.*field.member) : nb::none();
    } else if constexpr (kind == 2) {
      out[field.name] = write_array(field, record, symbols);
    }
  });
  return out;
}

// --------------------------------------------------------------------- schemas

template <class R> void collect(nb::dict& tables);

template <class T> nb::dict describe_value(nb::dict& tables) {
  nb::dict out;
  if constexpr (std::is_same_v<T, bool>) {
    out["kind"] = "bool";
  } else if constexpr (Named<T>) {
    out["kind"] = "enum";
    nb::list labels;
    for (const auto& [label, entry] : names(static_cast<const T*>(nullptr)))
      labels.append(label);
    out["values"] = labels;
  } else if constexpr (std::is_integral_v<T>) {
    out["kind"] = "int";
  } else if constexpr (std::is_floating_point_v<T>) {
    out["kind"] = "float";
  } else if constexpr (std::is_same_v<T, std::string>) {
    out["kind"] = "str";
  } else if constexpr (IsOptional<T>::value) {
    out["kind"] = "optional";
    out["of"] = describe_value<typename T::value_type>(tables);
  } else if constexpr (HasExtern<T>) {
    out["kind"] = "object";
    out["type"] = Extern<T>::kind;
  } else if constexpr (IsVariant<T>::value) {
    out["kind"] = "variant";
    nb::list names;
    [&]<class... A>(std::variant<A...>* /*unused*/) {
      (collect<A>(tables), ...);
      (names.append(fields::table_of<A>().name), ...);
    }(static_cast<T*>(nullptr));
    out["records"] = names;
  } else if constexpr (IsNumberMap<T>::value) {
    out["kind"] = "mapping";
    out["of"] = describe_value<double>(tables);
  } else if constexpr (IsStdArray<T>::value || std::is_array_v<T>) {
    using Element = std::remove_cvref_t<decltype(std::declval<T&>()[0])>;
    out["kind"] = "sequence";
    out["size"] = sizeof(T) / sizeof(Element);
    out["of"] = describe_value<Element>(tables);
  } else if constexpr (IsVector<T>::value) {
    out["kind"] = "sequence";
    out["of"] = describe_value<typename T::value_type>(tables);
  } else if constexpr (fields::Described<T>) {
    collect<T>(tables);
    out["kind"] = "record";
    out["record"] = fields::table_of<T>().name;
    out["nullable"] = Presence<T>;
  }
  return out;
}

/** `R`'s table as data: each field's name and kind, an array's dtype, rank and extent
    relations, and the counts its arrays give. */
template <class R> nb::dict schema(nb::dict& tables) {
  constexpr auto table = fields::table_of<R>();
  nb::list entries;
  nb::dict counts;
  fields::each(table, [&](const auto& field) {
    using Field = std::remove_cvref_t<decltype(field)>;
    constexpr int kind = FieldKind<Field>::kind;
    if constexpr (kind == 3) {
      counts[field.symbol] = field.symbol;
    } else {
      nb::dict entry;
      if constexpr (kind == 0) {
        using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
        entry = describe_value<M>(tables);
      } else if constexpr (kind == 1) {
        using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
        entry["kind"] = "optional";
        entry["of"] = describe_value<M>(tables);
      } else {
        using M = std::remove_cvref_t<decltype(std::declval<R&>().*field.member)>;
        constexpr std::size_t rank = std::tuple_size_v<decltype(field.extents)>;
        entry["kind"] = "array";
        entry["dtype"] = fields::dtype_name(field.dtype);
        entry["rank"] = rank == 0 ? nb::none() : nb::cast(rank);
        nb::list extents;
        for (const auto& extent : field.extents)
          extents.append(extent_text(extent));
        entry["extents"] = extents;
        entry["optional"] = fields::ArrayElement<M>::optional;
      }
      entry["name"] = field.name;
      entries.append(entry);
    }
  });
  nb::dict out;
  out["fields"] = entries;
  out["counts"] = counts;
  return out;
}

template <class R> void collect(nb::dict& tables) {
  constexpr auto name = fields::table_of<R>().name;
  if (tables.contains(name))
    return;
  tables[name] = nb::none(); // held while its fields are described
  tables[name] = schema<R>(tables);
}

} // namespace solprop_native::records
