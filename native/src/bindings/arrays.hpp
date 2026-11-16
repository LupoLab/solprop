// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The arrays the extension's binding files share: the exact NumPy inputs each entry
// point accepts, read into native vectors without hidden conversions, and the host
// arrays it returns, each owning its memory through a capsule.
//
// Types: the `nb::ndarray` aliases (`ComplexInput`, `RealInput`, `ComplexOutput`,
// `HostField` and the rest) and `ExactArray`, a problem field's element as NumPy names
// it. Functions: the readers (`complex_span`, `complex_vector`, `real_vector`,
// `bool_vector`, `index_vector`, `exact_vector`) and the owning outputs (`array_of`,
// `complex_output`, `owned_complex`, `complex_buffer_output`, `vector_output`,
// `real_output`).

#include "solprop_native/runtime/crossing.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/refuse.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace solprop_native::bindings {

namespace nb = nanobind;
namespace native = solprop_native;

using ComplexInput =
    nb::ndarray<const std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealInput = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using BoolInput = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
using ByteInput = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;
using IndexInput = nb::ndarray<const std::uint64_t, nb::c_contig, nb::device::cpu>;
using ComplexOutput =
    nb::ndarray<nb::numpy, std::complex<double>, nb::c_contig, nb::device::cpu>;
using RealOutput = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
// A read-only NumPy view of the engine's memory (solprop SEP 49's host source field).
using HostField =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::c_contig, nb::device::cpu>;
using ComplexStorage = std::vector<std::complex<double>>;
using RealStorage = std::vector<double>;

template <class Array> Array exact_array(const nb::handle value) {
  return nb::cast<Array>(value, false);
}

template <class T>
std::size_t element_count(const nb::ndarray<T, nb::c_contig, nb::device::cpu>& values) {
  std::size_t count = 1;
  for (std::size_t axis = 0; axis < values.ndim(); ++axis) {
    count *= values.shape(axis);
  }
  return count;
}

// NumPy's complex128 and `native::Complex` are both two doubles, real part first.
static_assert(sizeof(native::Complex) == sizeof(std::complex<double>));
static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ >= alignof(native::Complex));

/** A complex array's own memory as native values, read in place. */
inline std::span<const native::Complex> complex_span(const ComplexInput values) {
  return {reinterpret_cast<const native::Complex*>(values.data()),
          element_count(values)};
}

inline std::vector<native::Complex> complex_vector(const ComplexInput values) {
  const auto count = element_count(values);
  std::vector<native::Complex> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] =
        native::Complex(values.data()[index].real(), values.data()[index].imag());
  }
  return result;
}

inline std::vector<double> real_vector(const RealInput values) {
  const auto count = element_count(values);
  return {values.data(), values.data() + count};
}

inline std::vector<unsigned char> bool_vector(const BoolInput values) {
  const auto count = element_count(values);
  std::vector<unsigned char> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = values.data()[index] ? 1U : 0U;
  }
  return result;
}

inline std::vector<std::size_t> index_vector(const IndexInput values) {
  const auto count = element_count(values);
  std::vector<std::size_t> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = static_cast<std::size_t>(values.data()[index]);
  }
  return result;
}

// ------------------------------------------------------------- the problem's fields
// A vector field of a bound struct crosses as an exact NumPy array: float64,
// complex128, uint64 or bool, C-contiguous on the CPU and of any rank, copied in, and
// out as a one-dimensional copy. Anything else is refused, naming the field.

template <class Element> struct ExactArray;
template <> struct ExactArray<double> {
  using Input = RealInput;
  using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
  using Stored = double;
  static constexpr const char* dtype = "float64";
  static constexpr const char* scalar = "numpy.float64";
};
template <> struct ExactArray<native::Complex> {
  using Input = ComplexInput;
  using Output =
      nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;
  using Stored = std::complex<double>;
  static constexpr const char* dtype = "complex128";
  static constexpr const char* scalar = "numpy.complex128";
};
template <> struct ExactArray<std::size_t> {
  using Input = IndexInput;
  using Output = nb::ndarray<nb::numpy, std::uint64_t, nb::ndim<1>, nb::c_contig>;
  using Stored = std::uint64_t;
  static constexpr const char* dtype = "uint64";
  static constexpr const char* scalar = "numpy.uint64";
};
template <> struct ExactArray<unsigned char> {
  using Input = BoolInput;
  using Output = nb::ndarray<nb::numpy, bool, nb::ndim<1>, nb::c_contig>;
  using Stored = bool;
  static constexpr const char* dtype = "bool";
  static constexpr const char* scalar = "numpy.bool_";
};

template <class Element>
std::vector<Element> exact_vector(const nb::handle value, const std::string& field,
                                  const std::size_t rank,
                                  std::vector<std::size_t>* shape = nullptr) {
  typename ExactArray<Element>::Input array;
  if (!nb::try_cast(value, array, false) || (rank != 0 && array.ndim() != rank))
    native::refuse(field, " must be a C-contiguous ", ExactArray<Element>::dtype,
                   " array", rank == 0 ? "" : native::concat(" of rank ", rank));
  if (shape != nullptr) {
    shape->resize(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis)
      (*shape)[axis] = array.shape(axis);
  }
  if constexpr (std::is_same_v<Element, double>)
    return real_vector(array);
  else if constexpr (std::is_same_v<Element, native::Complex>)
    return complex_vector(array);
  else if constexpr (std::is_same_v<Element, std::size_t>)
    return index_vector(array);
  else
    return bool_vector(array);
}

template <class Element>
// NOLINTNEXTLINE(readability-redundant-typename): nvcc's front end needs it
typename ExactArray<Element>::Output array_of(const std::vector<Element>& values) {
  // NOLINTNEXTLINE(readability-redundant-typename): nvcc's front end needs it
  using Stored = typename ExactArray<Element>::Stored;
  auto storage = std::make_unique<Stored[]>(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    if constexpr (std::is_same_v<Element, native::Complex>)
      storage[index] = {values[index].real(), values[index].imag()};
    else
      storage[index] = static_cast<Stored>(values[index]);
  }
  nb::capsule owner(storage.get(), [](void* held) noexcept {
    const std::unique_ptr<Stored[]> owned(static_cast<Stored*>(held));
  });
  const std::array<std::size_t, 1> shape{values.size()};
  return {storage.release(), 1, shape.data(), owner};
}

inline ComplexOutput complex_output(const std::vector<native::Complex>& values,
                                    std::vector<std::size_t> shape) {
  auto storage = std::make_unique<ComplexStorage>(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    (*storage)[index] =
        std::complex<double>(values[index].real(), values[index].imag());
  }
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<ComplexStorage> held(static_cast<ComplexStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  return {pointer->data(), shape.size(), shape.data(), owner};
}

/** A complex array owning `storage`. */
inline ComplexOutput owned_complex(std::unique_ptr<ComplexStorage> storage,
                                   std::vector<std::size_t> shape) {
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<ComplexStorage> held(static_cast<ComplexStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  return {pointer->data(), shape.size(), shape.data(), owner};
}

/** A complex array owning `storage`, uninitialised doubles a native copy filled. */
inline ComplexOutput complex_buffer_output(native::crossing::HostBuffer storage,
                                           std::vector<std::size_t> shape) {
  auto* pointer = storage.release();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const native::crossing::HostBuffer held(static_cast<double*>(value));
  });
  return {reinterpret_cast<std::complex<double>*>(pointer), shape.size(), shape.data(),
          owner};
}

/** `values` as a one-dimensional array that owns them. */
inline RealOutput vector_output(std::vector<double> values) {
  auto storage = std::make_unique<RealStorage>(std::move(values));
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<RealStorage> held(static_cast<RealStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  const std::array<std::size_t, 1> shape{pointer->size()};
  return {pointer->data(), shape.size(), shape.data(), owner};
}

inline RealOutput real_output(std::vector<double> values, const std::size_t rows) {
  if (rows == 0 || values.size() % rows != 0)
    throw std::invalid_argument("native output has an invalid packed shape");
  auto storage = std::make_unique<RealStorage>(std::move(values));
  auto* pointer = storage.get();
  nb::capsule owner(pointer, [](void* value) noexcept {
    const std::unique_ptr<RealStorage> held(static_cast<RealStorage*>(value));
  });
  static_cast<void>(storage.release()); // the capsule owns it now
  const std::array<std::size_t, 2> shape{rows, pointer->size() / rows};
  return {pointer->data(), shape.size(), shape.data(), owner};
}

} // namespace solprop_native::bindings
