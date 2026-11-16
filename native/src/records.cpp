// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "records.hpp"

#include <nanobind/stl/string.h>

#include <limits>
#include <stdexcept>

namespace solprop_native::records {

void type_refusal(const std::string& path, const std::string& message) {
  throw nb::type_error((path + " " + message).c_str());
}

void value_refusal(const std::string& path, const std::string& message) {
  throw std::invalid_argument(path + " " + message);
}

std::string child(const std::string& path, const std::string& name) {
  return path.empty() ? name : path + "." + name;
}

std::string child(const std::string& path, const std::size_t index) {
  return path + "[" + std::to_string(index) + "]";
}

bool is_bool(const nb::handle value) {
  if (PyBool_Check(value.ptr()))
    return true;
  // NumPy's bool scalar.
  const auto type =
      nb::borrow<nb::object>(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())));
  return nb::hasattr(type, "__module__") &&
         nb::cast<std::string>(nb::str(type.attr("__module__"))) == "numpy" &&
         nb::cast<std::string>(nb::str(type.attr("__name__"))) == "bool";
}

bool is_integer(const nb::handle value) { return PyIndex_Check(value.ptr()) != 0; }

bool is_number(const nb::handle value) {
  // A float (NumPy's float64 among them) or an integer; never a complex or a str.
  return PyFloat_Check(value.ptr()) || PyIndex_Check(value.ptr()) != 0;
}

bool is_sequence(const nb::handle value) {
  return PySequence_Check(value.ptr()) != 0 && !nb::isinstance<nb::str>(value) &&
         !nb::isinstance<nb::bytes>(value) && !nb::isinstance<nb::dict>(value);
}

bool is_mapping(const nb::handle value) {
  return nb::isinstance<nb::dict>(value) ||
         (PyMapping_Check(value.ptr()) != 0 && nb::hasattr(value, "keys") &&
          !is_sequence(value));
}

Source::Source(const nb::handle value, const std::string& path) : value_(value) {
  const auto type = nb::handle(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())));
  if (nb::hasattr(type, "__dataclass_fields__") &&
      !nb::isinstance<nb::type_object>(value)) {
    for (const auto key :
         nb::borrow<nb::dict>(type.attr("__dataclass_fields__")).keys())
      names_.push_back(nb::cast<std::string>(key));
    return;
  }
  if (is_mapping(value)) {
    mapping_ = true;
    PyObject* keys = PyMapping_Keys(value.ptr());
    if (keys == nullptr)
      throw nb::python_error();
    for (const auto key : nb::steal<nb::list>(keys)) {
      if (!nb::isinstance<nb::str>(key))
        type_refusal(path, "must have str keys");
      names_.push_back(nb::cast<std::string>(key));
    }
    return;
  }
  if (nb::isinstance<nb::tuple>(value) || nb::isinstance<nb::list>(value)) {
    positional_ = true;
    size_ = nb::len(value);
    return;
  }
  type_refusal(path, "must be a dataclass, a mapping or a tuple of fields");
}

bool Source::has(const char* name, const std::size_t index) const {
  if (positional_)
    return index < size_;
  for (const auto& entry : names_)
    if (entry == name)
      return true;
  return false;
}

nb::object Source::get(const char* name, const std::size_t index) const {
  if (positional_)
    return nb::borrow<nb::sequence>(value_)[index];
  if (mapping_)
    return value_[name];
  return value_.attr(name);
}

namespace {

nb::dlpack::dtype numpy_dtype(const fields::Dtype dtype) {
  switch (dtype) {
  case fields::Dtype::float64:
    return nb::dtype<double>();
  case fields::Dtype::complex128:
    return nb::dtype<std::complex<double>>();
  case fields::Dtype::uint64:
    return nb::dtype<std::uint64_t>();
  case fields::Dtype::uint8:
    return nb::dtype<std::uint8_t>();
  case fields::Dtype::boolean:
    return nb::dtype<bool>();
  }
  return {};
}

std::size_t element_size(const fields::Dtype dtype) {
  switch (dtype) {
  case fields::Dtype::float64:
  case fields::Dtype::uint64:
    return 8;
  case fields::Dtype::complex128:
    return 16;
  case fields::Dtype::uint8:
  case fields::Dtype::boolean:
    return 1;
  }
  return 0;
}

std::string shape_text(const std::vector<std::size_t>& shape) {
  std::string text = "(";
  for (std::size_t axis = 0; axis < shape.size(); ++axis)
    text += (axis == 0 ? "" : ", ") + std::to_string(shape[axis]);
  return text + (shape.size() == 1 ? ",)" : ")");
}

} // namespace

Borrowed borrow(const nb::handle value, const fields::Dtype dtype,
                const std::size_t rank, const std::string& path) {
  using Any = nb::ndarray<nb::ro, nb::device::cpu>;
  const std::string wanted = fields::dtype_name(dtype);
  const std::string axes = rank == 0 ? "" : " of rank " + std::to_string(rank);
  Any array;
  if (!nb::try_cast(value, array, false))
    type_refusal(path, "must be a C-contiguous " + wanted + " array" + axes);
  if (array.dtype() != numpy_dtype(dtype))
    type_refusal(path, "must be a C-contiguous " + wanted + " array" + axes +
                           ", not another dtype");
  Borrowed out;
  out.shape.resize(array.ndim());
  for (std::size_t axis = 0; axis < array.ndim(); ++axis)
    out.shape[axis] = array.shape(axis);
  if (rank != 0 && array.ndim() != rank)
    value_refusal(path, "must be a C-contiguous " + wanted + " array" + axes +
                            ", given shape " + shape_text(out.shape));
  // C order, element strides: an axis of one may take any stride; a stride of zero
  // elsewhere is an overlapping (broadcast) view.
  // An empty array holds nothing to overlap, whatever its strides.
  bool empty = false;
  for (const auto size : out.shape)
    empty = empty || size == 0;
  std::size_t count = 1;
  for (std::size_t axis = array.ndim(); axis-- > 0;) {
    const std::size_t size = out.shape[axis];
    if (!empty && size != 1 && array.stride(axis) != static_cast<std::int64_t>(count))
      value_refusal(path, "must be a C-contiguous " + wanted + " array" + axes +
                              ", not a strided or overlapping view");
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size)
      value_refusal(path, "has a size whose product overflows");
    count *= size;
  }
  if (count > std::numeric_limits<std::size_t>::max() / element_size(dtype))
    value_refusal(path, "has a size whose product overflows");
  out.count = count;
  out.data = array.data();
  return out;
}

std::string extent_text(const fields::Extent& extent) {
  if (extent.fixed_size)
    return std::to_string(extent.literal);
  if (extent.symbol == nullptr)
    return "*";
  std::string text = extent.symbol;
  if (extent.offset > 0)
    text += "+" + std::to_string(extent.offset);
  else if (extent.offset < 0)
    text += std::to_string(extent.offset);
  return text;
}

void relate(const std::vector<std::size_t>& shape, const fields::Extent* extents,
            const std::size_t rank, Symbols& symbols, const std::string& path) {
  std::string relation = "(";
  for (std::size_t axis = 0; axis < rank; ++axis)
    relation += (axis == 0 ? "" : ", ") + extent_text(extents[axis]);
  relation += rank == 1 ? ",)" : ")";
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const auto& extent = extents[axis];
    const std::size_t size = shape[axis];
    const auto refuse_axis = [&](const std::string& why) {
      value_refusal(path, "has shape " + shape_text(shape) + ", and its layout is " +
                              relation + ": axis " + std::to_string(axis) + " " + why);
    };
    if (extent.fixed_size) {
      if (size != extent.literal)
        refuse_axis("must be " + std::to_string(extent.literal));
      continue;
    }
    if (extent.symbol == nullptr)
      continue;
    const auto signed_size = static_cast<std::ptrdiff_t>(size);
    if (signed_size < extent.offset)
      refuse_axis("is shorter than its offset");
    const auto value = static_cast<std::size_t>(signed_size - extent.offset);
    const auto found = symbols.find(extent.symbol);
    if (found == symbols.end())
      symbols[extent.symbol] = value;
    else if (found->second != value)
      refuse_axis("gives " + std::string(extent.symbol) + " = " +
                  std::to_string(value) + ", where the record has " +
                  std::to_string(found->second));
  }
}

nb::object owned_array(const void* data, const std::size_t bytes,
                       const fields::Dtype dtype,
                       const std::vector<std::size_t>& shape) {
  auto storage = std::make_unique<unsigned char[]>(bytes == 0 ? 1 : bytes);
  if (bytes != 0)
    std::memcpy(storage.get(), data, bytes);
  nb::capsule owner(storage.get(), [](void* held) noexcept {
    const std::unique_ptr<unsigned char[]> owned(static_cast<unsigned char*>(held));
  });
  void* pointer = storage.release();
  const nb::ndarray<nb::numpy> array(pointer, shape.size(), shape.data(), owner,
                                     nullptr, numpy_dtype(dtype));
  return nb::cast(array);
}

} // namespace solprop_native::records
