// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/dispersive.hpp"
#include "solprop_native/gemm.hpp"
#include "solprop_native/slab.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace solprop_native {
namespace {

using Matrix = Kokkos::View<double**, Layout, MemorySpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

std::size_t product_of(const std::size_t left, const std::size_t right,
                       const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string(name) + " extent overflows size_t");
  return left * right;
}

void require_size(const std::size_t actual, const std::size_t expected,
                  const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string("dispersive ") + name +
                                " has the wrong extent");
}

template <class Value>
void upload(const ExecutionSpace& space,
            const Kokkos::View<Value*, Layout, MemorySpace>& target,
            const std::vector<Value>& source) {
  require_size(source.size(), target.extent(0), "upload");
  const Kokkos::View<const Value*, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), source.size());
  Kokkos::deep_copy(space, target, host);
}

template <class Value>
void upload(const ExecutionSpace& space,
            const Kokkos::View<Value**, Layout, MemorySpace>& target,
            const std::vector<Value>& source) {
  require_size(source.size(), target.size(), "upload");
  const Kokkos::View<const Value**, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), target.extent(0), target.extent(1));
  Kokkos::deep_copy(space, target, host);
}

// One product of the field with itself twice at every node of a chunk: the symmetric
// evaluator's product of its filtered field, and the polynomial's `E^3` (solprop
// `make_dispersive_self`), times the collar. It may run in place: each sample reads
// every component before it writes. A shifted source has a constant per row
// subtracted as it is read: the zero-frequency bin of a field grid's bare analytic
// signal, which `Pi_+` excludes and the engine's analytic family keeps.
template <class Value, std::size_t Components, DispersiveForm Form> struct SelfProduct {
  const Value* source;
  Value* target;
  const double* window;
  const Complex* carrier;
  const Complex* offset;
  std::size_t n_to;
  bool shifted;

  KOKKOS_INLINE_FUNCTION void sample(const std::size_t node,
                                     const std::size_t time) const {
    const std::size_t base = node * Components;
    const double collar = window[time];
    if constexpr (Form == DispersiveForm::real) {
      if constexpr (Components == 2) {
        const double x0 = source[base * n_to + time];
        const double x1 = source[(base + 1) * n_to + time];
        double p0 = 0.0;
        double p1 = 0.0;
        dispersive_isotropic(x0, x1, x0, x1, x0, x1, p0, p1);
        target[base * n_to + time] = p0 * collar;
        target[(base + 1) * n_to + time] = p1 * collar;
      } else {
        target[base * n_to + time] =
            dispersive_real_self(source[base * n_to + time]) * collar;
      }
    } else {
      if constexpr (Components == 2) {
        Complex x0 = source[base * n_to + time];
        Complex x1 = source[(base + 1) * n_to + time];
        if (shifted) {
          x0 = x0 - offset[base];
          x1 = x1 - offset[base + 1];
        }
        Complex p0, p1;
        dispersive_conserving(x0, x1, x0, x1, x0, x1, p0, p1);
        if constexpr (Form == DispersiveForm::harmonic) {
          Complex h0, h1;
          dispersive_harmonic(x0, x1, x0, x1, x0, x1, carrier[time], h0, h1);
          p0 = p0 + h0;
          p1 = p1 + h1;
        }
        target[base * n_to + time] = p0 * collar;
        target[(base + 1) * n_to + time] = p1 * collar;
      } else {
        Complex u = source[base * n_to + time];
        if (shifted)
          u = u - offset[base];
        Complex value = dispersive_conserving_self(u);
        if constexpr (Form == DispersiveForm::harmonic)
          value = value + dispersive_harmonic_self(u, carrier[time]);
        target[base * n_to + time] = value * collar;
      }
    }
  }
  // A host node owns its whole row of samples; a device thread one sample.
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node) const {
    for (std::size_t time = 0; time < n_to; ++time)
      sample(node, time);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node,
                                         const std::size_t time) const {
    sample(node, time);
  }
};

// The polynomial's two derivative products at every node of a chunk, in place: with
// `E` the bare field, `F = D E` and `H = D^2 E`, `T(E, E, F)` over `F` and
// `c0 T(E, E, H) + c1 T(E, F, F)` over `H` (solprop `make_dispersive_pairs`), each
// times the collar.
template <class Value, std::size_t Components, DispersiveForm Form>
struct PairProducts {
  const Value* field;
  Value* first;
  Value* second;
  const double* window;
  const Complex* carrier;
  const Complex* offset;
  double c0;
  double c1;
  std::size_t n_to;
  bool shifted;

  KOKKOS_INLINE_FUNCTION void sample(const std::size_t node,
                                     const std::size_t time) const {
    const std::size_t base = node * Components;
    const std::size_t at0 = base * n_to + time;
    const double collar = window[time];
    if constexpr (Form == DispersiveForm::real) {
      if constexpr (Components == 2) {
        const std::size_t at1 = (base + 1) * n_to + time;
        const double x0 = field[at0], x1 = field[at1];
        const double f0 = first[at0], f1 = first[at1];
        const double h0 = second[at0], h1 = second[at1];
        double p0 = 0.0, p1 = 0.0, a0 = 0.0, a1 = 0.0, b0 = 0.0, b1 = 0.0;
        dispersive_isotropic(x0, x1, x0, x1, f0, f1, p0, p1);
        dispersive_isotropic(x0, x1, x0, x1, h0, h1, a0, a1);
        dispersive_isotropic(x0, x1, f0, f1, f0, f1, b0, b1);
        first[at0] = p0 * collar;
        first[at1] = p1 * collar;
        second[at0] = (c0 * a0 + c1 * b0) * collar;
        second[at1] = (c0 * a1 + c1 * b1) * collar;
      } else {
        const double x = field[at0], f = first[at0], h = second[at0];
        first[at0] = dispersive_real_pair(x, f) * collar;
        second[at0] =
            (c0 * dispersive_real_pair(x, h) + c1 * dispersive_real_cross(x, f)) *
            collar;
      }
    } else {
      if constexpr (Components == 2) {
        const std::size_t at1 = (base + 1) * n_to + time;
        Complex x0 = field[at0], x1 = field[at1];
        if (shifted) {
          x0 = x0 - offset[base];
          x1 = x1 - offset[base + 1];
        }
        const Complex f0 = first[at0], f1 = first[at1];
        const Complex h0 = second[at0], h1 = second[at1];
        Complex p0, p1, a0, a1, b0, b1;
        dispersive_conserving(x0, x1, x0, x1, f0, f1, p0, p1);
        dispersive_conserving(x0, x1, x0, x1, h0, h1, a0, a1);
        dispersive_conserving(x0, x1, f0, f1, f0, f1, b0, b1);
        if constexpr (Form == DispersiveForm::harmonic) {
          const Complex c = carrier[time];
          Complex g0, g1;
          dispersive_harmonic(x0, x1, x0, x1, f0, f1, c, g0, g1);
          p0 = p0 + g0;
          p1 = p1 + g1;
          dispersive_harmonic(x0, x1, x0, x1, h0, h1, c, g0, g1);
          a0 = a0 + g0;
          a1 = a1 + g1;
          dispersive_harmonic(x0, x1, f0, f1, f0, f1, c, g0, g1);
          b0 = b0 + g0;
          b1 = b1 + g1;
        }
        first[at0] = p0 * collar;
        first[at1] = p1 * collar;
        second[at0] = (c0 * a0 + c1 * b0) * collar;
        second[at1] = (c0 * a1 + c1 * b1) * collar;
      } else {
        Complex u = field[at0];
        if (shifted)
          u = u - offset[base];
        const Complex f = first[at0];
        const Complex h = second[at0];
        Complex pair = dispersive_conserving_pair(u, f);
        Complex weighted = c0 * dispersive_conserving_pair(u, h) +
                           c1 * dispersive_conserving_cross(u, f);
        if constexpr (Form == DispersiveForm::harmonic) {
          const Complex c = carrier[time];
          pair = pair + dispersive_harmonic_pair(u, f, c);
          weighted = weighted + (c0 * dispersive_harmonic_pair(u, h, c) +
                                 c1 * dispersive_harmonic_cross(u, f, c));
        }
        first[at0] = pair * collar;
        second[at0] = weighted * collar;
      }
    }
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node) const {
    for (std::size_t time = 0; time < n_to; ++time)
      sample(node, time);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node,
                                         const std::size_t time) const {
    sample(node, time);
  }
};

template <class Functor>
void over_nodes(const char* label, const ExecutionSpace& space,
                const std::size_t node_count, const std::size_t n_to,
                const Functor& functor) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using Policy = Kokkos::MDRangePolicy<
      ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  Kokkos::parallel_for(label, Policy(space, {0, 0}, {node_count, n_to}), functor);
#else
  static_cast<void>(n_to);
  Kokkos::parallel_for(label, Kokkos::RangePolicy<ExecutionSpace>(space, 0, node_count),
                       functor);
#endif
}

using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;
using RealView2 = Kokkos::View<double**, Layout, MemorySpace>;
using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

// The product functors over a chunk's nodes, chosen by the form and the components.
template <bool Real, std::size_t Components>
void run_self(const ExecutionSpace& space, const DispersiveForm form,
              const std::size_t node_count, const std::size_t n_to,
              const double* const source, double* const target,
              const double* const window, const Complex* const carrier,
              const Complex* const offset, const bool shifted) {
  using Value = std::conditional_t<Real, double, Complex>;
  const auto* const from = reinterpret_cast<const Value*>(source);
  auto* const to = reinterpret_cast<Value*>(target);
  if constexpr (Real) {
    over_nodes("solprop_native.dispersive.self_product", space, node_count, n_to,
               SelfProduct<double, Components, DispersiveForm::real>{
                   from, to, window, carrier, offset, n_to, shifted});
  } else if (form == DispersiveForm::harmonic) {
    over_nodes("solprop_native.dispersive.self_product", space, node_count, n_to,
               SelfProduct<Complex, Components, DispersiveForm::harmonic>{
                   from, to, window, carrier, offset, n_to, shifted});
  } else {
    over_nodes("solprop_native.dispersive.self_product", space, node_count, n_to,
               SelfProduct<Complex, Components, DispersiveForm::conserving>{
                   from, to, window, carrier, offset, n_to, shifted});
  }
}

template <bool Real, std::size_t Components>
void run_pairs(const ExecutionSpace& space, const DispersiveForm form,
               const std::size_t node_count, const std::size_t n_to,
               const double* const field, double* const first, double* const second,
               const double c0, const double c1, const double* const window,
               const Complex* const carrier, const Complex* const offset,
               const bool shifted) {
  using Value = std::conditional_t<Real, double, Complex>;
  const auto* const bare = reinterpret_cast<const Value*>(field);
  auto* const one = reinterpret_cast<Value*>(first);
  auto* const two = reinterpret_cast<Value*>(second);
  if constexpr (Real) {
    over_nodes("solprop_native.dispersive.pair_products", space, node_count, n_to,
               PairProducts<double, Components, DispersiveForm::real>{
                   bare, one, two, window, carrier, offset, c0, c1, n_to, shifted});
  } else if (form == DispersiveForm::harmonic) {
    over_nodes("solprop_native.dispersive.pair_products", space, node_count, n_to,
               PairProducts<Complex, Components, DispersiveForm::harmonic>{
                   bare, one, two, window, carrier, offset, c0, c1, n_to, shifted});
  } else {
    over_nodes("solprop_native.dispersive.pair_products", space, node_count, n_to,
               PairProducts<Complex, Components, DispersiveForm::conserving>{
                   bare, one, two, window, carrier, offset, c0, c1, n_to, shifted});
  }
}

// `E^3` (or the filtered field's product) at a chunk's nodes of one or two
// components; a real form's real products take no offset.
void self_products(const ExecutionSpace& space, const DispersiveForm form,
                   const std::size_t components, const std::size_t node_count,
                   const std::size_t n_to, const double* const source,
                   double* const target, const double* const window,
                   const Complex* const carrier, const Complex* const offset,
                   const bool shifted) {
  if (form != DispersiveForm::real) {
    if (components == 2)
      run_self<false, 2>(space, form, node_count, n_to, source, target, window, carrier,
                         offset, shifted);
    else
      run_self<false, 1>(space, form, node_count, n_to, source, target, window, carrier,
                         offset, shifted);
  } else if (components == 2) {
    run_self<true, 2>(space, form, node_count, n_to, source, target, window, carrier,
                      offset, shifted);
  } else {
    run_self<true, 1>(space, form, node_count, n_to, source, target, window, carrier,
                      offset, shifted);
  }
}

void pair_products(const ExecutionSpace& space, const DispersiveForm form,
                   const std::size_t components, const std::size_t node_count,
                   const std::size_t n_to, const double* const field,
                   double* const first, double* const second, const double c0,
                   const double c1, const double* const window,
                   const Complex* const carrier, const Complex* const offset,
                   const bool shifted) {
  if (form != DispersiveForm::real) {
    if (components == 2)
      run_pairs<false, 2>(space, form, node_count, n_to, field, first, second, c0, c1,
                          window, carrier, offset, shifted);
    else
      run_pairs<false, 1>(space, form, node_count, n_to, field, first, second, c0, c1,
                          window, carrier, offset, shifted);
  } else if (components == 2) {
    run_pairs<true, 2>(space, form, node_count, n_to, field, first, second, c0, c1,
                       window, carrier, offset, false);
  } else {
    run_pairs<true, 1>(space, form, node_count, n_to, field, first, second, c0, c1,
                       window, carrier, offset, false);
  }
}

// Row `field * rows + row` of `out` is `inputs(field) * (scale * source(row))` on the
// first `bins` bins and zero beyond, where an analytic family's full-length spectrum
// holds nothing (solprop `filter_spectra`). A complex-to-real plan destroys its input,
// so every bin is written. `scale` is one but on the general route, whose synthesis
// keeps the inverse transform's `1 / n_to`.
void filter_rows(const ExecutionSpace& space, const RealView2& inputs,
                 const ComplexView2& source, const ComplexView2& out,
                 const std::size_t rows, const std::size_t fields,
                 const std::size_t bins, const double scale, std::size_t active = 0) {
  if (active == 0)
    active = rows;
  Kokkos::parallel_for(
      "solprop_native.dispersive.filter",
      Range2(space, {0, 0}, {fields * active, out.extent(1)}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t k) {
        const auto field = row / active;
        const auto local = row % active;
        out(field * rows + local, k) =
            k < bins ? inputs(field, k) * (scale * source(local, k)) : Complex{};
      });
}

// A lone symmetric response's filter, on the stage's own spectrum.
void filter_in_place(const ExecutionSpace& space, const RealView2& inputs,
                     const ComplexView2& spectrum, const std::size_t rows,
                     const std::size_t bins) {
  Kokkos::parallel_for(
      "solprop_native.dispersive.filter_in_place", Range2(space, {0, 0}, {rows, bins}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t k) {
        spectrum(row, k) = inputs(0, k) * spectrum(row, k);
      });
}

// `target(row, k) (+)= sum_p weights(p, k) spectra(p * rows + row, k)`: the products'
// sum first, in product order, then landed once, so the result does not depend on what
// the target held (solprop `add_spectra`, `combine_spectra`).
void land_rows(const ExecutionSpace& space, const RealView2& weights,
               const ComplexView2& spectra, const ComplexView2& target,
               const std::size_t rows, const std::size_t bins,
               const std::size_t products, const bool assign, std::size_t active = 0) {
  if (active == 0)
    active = rows;
  Kokkos::parallel_for(
      "solprop_native.dispersive.land", Range2(space, {0, 0}, {active, bins}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t k) {
        Complex total = weights(0, k) * spectra(row, k);
        for (std::size_t product = 1; product < products; ++product)
          total += weights(product, k) * spectra(product * rows + row, k);
        if (assign)
          target(row, k) = total;
        else
          target(row, k) += total;
      });
}

// The one product's multiplier on a spectrum the stage's own transform made.
void weigh_rows(const ExecutionSpace& space, const RealView2& weights,
                const ComplexView2& target, const std::size_t rows,
                const std::size_t bins) {
  Kokkos::parallel_for(
      "solprop_native.dispersive.weigh", Range2(space, {0, 0}, {rows, bins}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t k) {
        target(row, k) = weights(0, k) * target(row, k);
      });
}

// Row `row` of a chunk's real or complex time buffer: its first double.
template <class View> double* row_of(const View& view, const std::size_t row) {
  return reinterpret_cast<double*>(view.data() + row * view.extent(1));
}

// The general route's products of `Components` Cartesian components at the nodes of
// one region, zero at the others (solprop `make_general_dispersive`): the cube of the
// field `T(E, E, E)`, and the polynomial's pairs `T(E, E, F)` and `c0 T(E, E, H) + c1
// T(E, F, F)`. The fields are complex analytic signals; the full sector reads their
// real parts and forms real products. A field read with `scale` is the route's bare
// signal, which keeps the inverse transform's `1 / n_to`; a shifted one has its
// zero-frequency constant subtracted per row.
template <bool Real, std::size_t Components> struct GeneralCube {
  using Value = std::conditional_t<Real, double, Complex>;
  const Complex* source;
  Value* target;
  const std::size_t* region;
  std::size_t key;
  std::size_t first;
  std::size_t n_to;
  double scale;
  const Complex* offset;
  bool shifted;

  KOKKOS_INLINE_FUNCTION void sample(const std::size_t node,
                                     const std::size_t time) const {
    const std::size_t base = node * Components;
    if (region[first + node] != key) {
      for (std::size_t c = 0; c < Components; ++c)
        target[(base + c) * n_to + time] = Value{};
      return;
    }
    Value x[Components];
    for (std::size_t c = 0; c < Components; ++c) {
      const Complex value = source[(base + c) * n_to + time];
      if constexpr (Real) {
        x[c] = value.real() * scale;
      } else {
        x[c] = value * scale;
        if (shifted)
          x[c] = x[c] - offset[base + c];
      }
    }
    Value p[Components];
    if constexpr (Real)
      dispersive_isotropic_n(x, x, x, p);
    else
      dispersive_conserving_n(x, x, x, p);
    for (std::size_t c = 0; c < Components; ++c)
      target[(base + c) * n_to + time] = p[c];
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node) const {
    for (std::size_t time = 0; time < n_to; ++time)
      sample(node, time);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node,
                                         const std::size_t time) const {
    sample(node, time);
  }
};

template <bool Real, std::size_t Components> struct GeneralPairs {
  using Value = std::conditional_t<Real, double, Complex>;
  const Complex* field;
  const Complex* first_field;
  const Complex* second_field;
  Value* out_first;
  Value* out_second;
  double c0;
  double c1;
  const std::size_t* region;
  std::size_t key;
  std::size_t first;
  std::size_t n_to;
  double scale;
  const Complex* offset;
  bool shifted;

  KOKKOS_INLINE_FUNCTION void sample(const std::size_t node,
                                     const std::size_t time) const {
    const std::size_t base = node * Components;
    if (region[first + node] != key) {
      for (std::size_t c = 0; c < Components; ++c) {
        out_first[(base + c) * n_to + time] = Value{};
        out_second[(base + c) * n_to + time] = Value{};
      }
      return;
    }
    Value x[Components], f[Components], h[Components];
    for (std::size_t c = 0; c < Components; ++c) {
      const std::size_t at = (base + c) * n_to + time;
      if constexpr (Real) {
        x[c] = field[at].real() * scale;
        f[c] = first_field[at].real();
        h[c] = second_field[at].real();
      } else {
        x[c] = field[at] * scale;
        if (shifted)
          x[c] = x[c] - offset[base + c];
        f[c] = first_field[at];
        h[c] = second_field[at];
      }
    }
    Value p[Components], a[Components], b[Components];
    if constexpr (Real) {
      dispersive_isotropic_n(x, x, f, p);
      dispersive_isotropic_n(x, x, h, a);
      dispersive_isotropic_n(x, f, f, b);
    } else {
      dispersive_conserving_n(x, x, f, p);
      dispersive_conserving_n(x, x, h, a);
      dispersive_conserving_n(x, f, f, b);
    }
    for (std::size_t c = 0; c < Components; ++c) {
      out_first[(base + c) * n_to + time] = p[c];
      out_second[(base + c) * n_to + time] = c0 * a[c] + c1 * b[c];
    }
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node) const {
    for (std::size_t time = 0; time < n_to; ++time)
      sample(node, time);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t node,
                                         const std::size_t time) const {
    sample(node, time);
  }
};

} // namespace

void require_dispersive(const Problem& problem) {
  if (!problem.dispersive) {
    if (!problem.dispersive_inputs.empty() ||
        !problem.dispersive_coefficients.empty() ||
        !problem.dispersive_term_rows.empty() ||
        !problem.dispersive_group_offsets.empty() ||
        !problem.dispersive_group_keys.empty() ||
        !problem.dispersive_node_region.empty())
      throw std::invalid_argument(
          "dispersive data were supplied while the dispersive response is disabled");
    return;
  }
  if (problem.dispersive_form < 0 || problem.dispersive_form > 2)
    throw std::invalid_argument("dispersive_form must be 0, 1 or 2");
  const auto form = static_cast<DispersiveForm>(problem.dispersive_form);
  const bool general = problem.general_modal;
  // The general route synthesises its own analytic signal on the real field family;
  // elsewhere the family follows the sector.
  const bool family_fits = general
                               ? problem.family == FieldFamily::field_real &&
                                     form != DispersiveForm::harmonic
                               : (problem.family == FieldFamily::field_real &&
                                  form == DispersiveForm::real) ||
                                     (problem.family == FieldFamily::field_analytic &&
                                      form == DispersiveForm::conserving) ||
                                     (problem.family == FieldFamily::envelope &&
                                      form != DispersiveForm::real);
  if (!family_fits)
    throw std::invalid_argument(
        "the dispersive form does not fit the field family: a field grid's full cubic "
        "takes the real family, its number-conserving sector the analytic one (the "
        "general modal route's both the real one), and an envelope's both sectors the "
        "envelope family");
  const std::size_t bins = problem.family == FieldFamily::envelope && !general
                               ? problem.n_to
                               : problem.n_to / 2 + 1;
  const std::size_t channels = problem.dispersive_coefficients.size();
  if (channels == 0)
    throw std::invalid_argument("the dispersive response needs a channel");
  require_size(problem.dispersive_input_scale.size(), bins, "input_scale");
  require_size(problem.dispersive_output_scale.size(), bins, "output_scale");
  require_size(problem.dispersive_pair_constants.size(),
               product_of(2, channels, "pair constants"), "pair_constants");
  const auto& offsets = problem.dispersive_group_offsets;
  const auto& members = problem.dispersive_group_members;
  if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != members.size() ||
      members.size() != channels)
    throw std::invalid_argument(
        "dispersive groups must partition the channels, from a zero offset");
  std::vector<unsigned char> seen(channels, 0U);
  for (const auto member : members) {
    if (member >= channels || seen[member] != 0U)
      throw std::invalid_argument("dispersive group members must be distinct channels");
    seen[member] = 1U;
  }
  std::size_t mixtures = 0;
  for (std::size_t group = 0; group + 1 < offsets.size(); ++group) {
    if (offsets[group + 1] <= offsets[group])
      throw std::invalid_argument("every dispersive group needs a channel");
    if (problem.dispersive_symmetric && offsets[group + 1] - offsets[group] > 1)
      ++mixtures;
  }
  const std::size_t groups = offsets.size() - 1;
  // Keys and node regions belong to the general route, where a channel acts at its own
  // region's nodes; elsewhere every channel acts at every node.
  const auto& keys = problem.dispersive_group_keys;
  if (general) {
    require_size(keys.size(), groups, "group_keys");
    require_size(problem.dispersive_node_region.size(), problem.general_nodes,
                 "node_region");
  } else if (!keys.empty() || !problem.dispersive_node_region.empty()) {
    throw std::invalid_argument(
        "dispersive group keys and node regions belong to the general modal route");
  }
  std::vector<std::size_t> distinct;
  for (std::size_t group = 0; group < groups; ++group) {
    const auto key = keys.empty() ? 0 : keys[group];
    if (std::find(distinct.begin(), distinct.end(), key) == distinct.end())
      distinct.push_back(key);
  }
  const std::size_t fields = problem.dispersive_symmetric ? groups : 2 * groups;
  const std::size_t products =
      problem.dispersive_symmetric ? groups : distinct.size() + 2 * groups;
  require_size(problem.dispersive_inputs.size(), product_of(fields, bins, "inputs"),
               "inputs");
  const std::size_t terms = problem.dispersive_term_rows.size();
  if (terms == 0)
    throw std::invalid_argument("the dispersive response needs an output term");
  require_size(problem.dispersive_term_channels.size(), terms, "term_channels");
  require_size(problem.dispersive_term_multipliers.size(),
               product_of(terms, bins, "term multipliers"), "term_multipliers");
  for (std::size_t term = 0; term < terms; ++term)
    if (problem.dispersive_term_rows[term] >= products ||
        problem.dispersive_term_channels[term] > channels + mixtures)
      throw std::invalid_argument("a dispersive term names no product or factor");
  require_size(problem.dispersive_quartics.size(),
               mixtures == 0 ? 0 : product_of(channels, bins, "quartics"), "quartics");
  // The products and their spectra, and the node slots: the general route's chunk, or
  // at most the modal columns.
  const auto rows = general ? product_of(problem.general_node_chunk,
                                         problem.general_components, "dispersive rows")
                            : problem.n_columns();
  const auto field =
      product_of(product_of(products + fields, rows, "dispersive products"),
                 problem.n_to, "dispersive products");
  static_cast<void>(product_of(field, sizeof(Complex), "dispersive product bytes"));
  static_cast<void>(product_of(
      product_of(2 * problem.node_chunk, problem.effective_node_pol(), "slots"),
      product_of(problem.n_to, sizeof(Complex), "slot bytes"), "dispersive slots"));
}

void dispersive_mixture_rows(
    const ExecutionSpace& space,
    const Kokkos::View<const double**, Layout, MemorySpace>& quartics,
    const Kokkos::View<const std::size_t*, Layout, MemorySpace>& members,
    const Kokkos::View<const double*, Layout, MemorySpace>& factors, const double total,
    const Kokkos::View<const double*, Layout, MemorySpace>& input_scale,
    const Kokkos::View<const double*, Layout, MemorySpace>& output_scale,
    const Kokkos::View<double*, Layout, MemorySpace>& inputs,
    const Kokkos::View<double*, Layout, MemorySpace>& outputs) {
  const auto n_members = members.extent(0);
  const bool vacuum = !(total > 0.0);
  Kokkos::parallel_for(
      "solprop_native.dispersive.mixture_rows",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, inputs.extent(0)),
      KOKKOS_LAMBDA(const std::size_t k) {
        if (vacuum) {
          inputs(k) = input_scale(k);
          outputs(k) = output_scale(k);
          return;
        }
        double quartic = 0.0;
        for (std::size_t member = 0; member < n_members; ++member) {
          const auto channel = members(member);
          quartic += factors(channel) * quartics(channel, k);
        }
        const double curve = Kokkos::sqrt(Kokkos::sqrt(quartic / total));
        inputs(k) = curve * input_scale(k);
        outputs(k) = curve * output_scale(k);
      });
}

// --- the layout and the stage's weights ----------------------------------------------

DispersiveWeights::DispersiveWeights(const Problem& problem,
                                     const ExecutionSpace& kernels)
    : kernels_(kernels) {
  require_dispersive(problem);
  if (!problem.dispersive)
    throw std::invalid_argument("the problem does not carry the dispersive response");
  channels_ = problem.dispersive_coefficients.size();
  bins_ = problem.dispersive_input_scale.size();
  n_stages_ = problem.n_stages;
  symmetric_ = problem.dispersive_symmetric;
  coefficients_ = problem.dispersive_coefficients;
  pair_constants_ = problem.dispersive_pair_constants;
  group_offsets_ = problem.dispersive_group_offsets;
  group_members_ = problem.dispersive_group_members;
  groups_ = group_offsets_.size() - 1;
  keys_ = problem.dispersive_group_keys.empty() ? std::vector<std::size_t>(groups_, 0)
                                                : problem.dispersive_group_keys;
  if (!symmetric_) {
    // One cube per distinct key, in the order the groups first name it (solprop
    // `dispersive_terms_layout`).
    cube_of_.assign(groups_, 0);
    for (std::size_t group = 0; group < groups_; ++group) {
      const auto at = std::find(cube_keys_.begin(), cube_keys_.end(), keys_[group]);
      cube_of_[group] = static_cast<std::size_t>(at - cube_keys_.begin());
      if (at == cube_keys_.end())
        cube_keys_.push_back(keys_[group]);
    }
  }
  cubes_ = cube_keys_.size();
  fields_ = symmetric_ ? groups_ : 2 * groups_;
  products_ = symmetric_ ? groups_ : cubes_ + 2 * groups_;
  for (std::size_t group = 0; group < groups_ && symmetric_; ++group)
    if (group_offsets_[group + 1] - group_offsets_[group] > 1)
      mixture_groups_.push_back(group);
  mixtures_ = mixture_groups_.size();
  static_inputs_ = RealView2("solprop_native.dispersive.static_inputs", fields_, bins_);
  inputs_ = RealView2("solprop_native.dispersive.inputs", fields_, bins_);
  const auto terms = problem.dispersive_term_rows.size();
  term_multipliers_ =
      RealView2("solprop_native.dispersive.term_multipliers", terms, bins_);
  outputs_ = RealView2("solprop_native.dispersive.outputs", terms, bins_);
  weights_ = RealView2("solprop_native.dispersive.weights", products_, bins_);
  quartics_ = RealView2("solprop_native.dispersive.quartics",
                        mixtures_ == 0 ? 0 : channels_, mixtures_ == 0 ? 0 : bins_);
  input_scale_ = RealView("solprop_native.dispersive.input_scale", bins_);
  output_scale_ = RealView("solprop_native.dispersive.output_scale", bins_);
  factors_ = RealView("solprop_native.dispersive.factors", channels_ + 1 + mixtures_);
  for (auto& staging : host_factors_)
    staging = Kokkos::create_mirror_view(factors_);
  term_rows_ = IndexView("solprop_native.dispersive.term_rows", terms);
  term_channels_ = IndexView("solprop_native.dispersive.term_channels", terms);
  members_ = IndexView("solprop_native.dispersive.members", channels_);
  upload(kernels_, static_inputs_, problem.dispersive_inputs);
  Kokkos::deep_copy(kernels_, inputs_, static_inputs_);
  upload(kernels_, term_multipliers_, problem.dispersive_term_multipliers);
  Kokkos::deep_copy(kernels_, outputs_, term_multipliers_);
  if (mixtures_ != 0)
    upload(kernels_, quartics_, problem.dispersive_quartics);
  upload(kernels_, input_scale_, problem.dispersive_input_scale);
  upload(kernels_, output_scale_, problem.dispersive_output_scale);
  upload(kernels_, term_rows_, problem.dispersive_term_rows);
  upload(kernels_, term_channels_, problem.dispersive_term_channels);
  upload(kernels_, members_, problem.dispersive_group_members);
  pairs_.assign(2 * groups_, 0.0);
  densities_.assign(n_stages_ * channels_, 0.0);
  last_.assign(channels_, 0.0);
  kernels_.fence("solprop_native dispersive weights");
}

void DispersiveWeights::require_densities(const std::vector<double>& densities) const {
  require_size(densities.size(), densities_.size(), "densities");
  for (const double value : densities)
    if (!(value >= 0.0) || value > std::numeric_limits<double>::max())
      throw std::invalid_argument(
          "dispersive densities must be finite and non-negative");
}

void DispersiveWeights::set_densities(const std::vector<double>& densities) {
  require_densities(densities);
  std::copy(densities.begin(), densities.end(), densities_.begin());
}

void DispersiveWeights::update(const std::size_t stage) {
  if (stage >= n_stages_)
    throw std::out_of_range("dispersive stage");
  const double* const density = densities_.data() + stage * channels_;
  if (weighted_ && std::equal(density, density + channels_, last_.begin()))
    return;
  weighted_ = true;
  std::copy(density, density + channels_, last_.begin());
  // A ring of host buffers, as the engine's stage scalars: a copy's source is never
  // rewritten before it has surely been read (a host build's mirror is the view).
  auto& factors = host_factors_[next_factors_];
  next_factors_ = (next_factors_ + 1) % host_factors_.size();
  for (std::size_t channel = 0; channel < channels_; ++channel)
    factors(channel) = coefficients_[channel] * density[channel];
  // The unit factor of a product that carries its channels' factors already.
  factors(channels_) = 1.0;
  for (std::size_t mixture = 0; mixture < mixtures_; ++mixture) {
    const auto group = mixture_groups_[mixture];
    double total = 0.0;
    for (auto member = group_offsets_[group]; member < group_offsets_[group + 1];
         ++member)
      total += factors(group_members_[member]);
    // The mixture's coefficient, zero at a vacuum (solprop `mixture_rows`).
    factors(channels_ + 1 + mixture) = total > 0.0 ? total : 0.0;
  }
  if (!symmetric_)
    for (std::size_t group = 0; group < groups_; ++group) {
      double c0 = 0.0;
      double c1 = 0.0;
      for (auto member = group_offsets_[group]; member < group_offsets_[group + 1];
           ++member) {
        const auto channel = group_members_[member];
        c0 += factors(channel) * pair_constants_[2 * channel];
        c1 += factors(channel) * pair_constants_[2 * channel + 1];
      }
      pairs_[2 * group] = c0;
      pairs_[2 * group + 1] = c1;
    }
  Kokkos::deep_copy(kernels_, factors_, factors);
  for (std::size_t mixture = 0; mixture < mixtures_; ++mixture) {
    const auto group = mixture_groups_[mixture];
    const auto members = Kokkos::subview(
        members_, std::make_pair(group_offsets_[group], group_offsets_[group + 1]));
    // A positive total is the coefficient itself; a vacuum's is zero, not positive.
    dispersive_mixture_rows(
        kernels_, quartics_, members, factors_, factors(channels_ + 1 + mixture),
        input_scale_, output_scale_, Kokkos::subview(inputs_, group, Kokkos::ALL()),
        Kokkos::subview(outputs_, group, Kokkos::ALL()));
  }
  // `weights(p, k) = sum_{t: row_t = p} factor(channel_t) outputs(t, k)`, in term order
  // from zero (solprop `weigh_terms`).
  const auto rows = term_rows_;
  const auto channels = term_channels_;
  const auto outputs = outputs_;
  const auto weights = weights_;
  const auto device_factors = factors_;
  const auto products = products_;
  const auto terms = rows.extent(0);
  Kokkos::parallel_for(
      "solprop_native.dispersive.weigh_terms",
      Kokkos::RangePolicy<ExecutionSpace>(kernels_, 0, bins_),
      KOKKOS_LAMBDA(const std::size_t k) {
        for (std::size_t product = 0; product < products; ++product)
          weights(product, k) = 0.0;
        for (std::size_t term = 0; term < terms; ++term)
          weights(rows(term), k) += device_factors(channels(term)) * outputs(term, k);
      });
}

std::size_t DispersiveWeights::bytes() const {
  const auto real = sizeof(double);
  const auto index = sizeof(std::size_t);
  std::size_t host = 0;
  for (const auto& staging : host_factors_)
    if (staging.data() != factors_.data())
      host += staging.size() * real;
  host += (coefficients_.capacity() + pair_constants_.capacity() + pairs_.capacity() +
           densities_.capacity() + last_.capacity()) *
              real +
          (group_offsets_.capacity() + group_members_.capacity() +
           mixture_groups_.capacity() + keys_.capacity() + cube_of_.capacity() +
           cube_keys_.capacity()) *
              index;
  return (static_inputs_.size() + inputs_.size() + term_multipliers_.size() +
          outputs_.size() + weights_.size() + quartics_.size() + input_scale_.size() +
          output_scale_.size() + factors_.size()) *
             real +
         (term_rows_.size() + term_channels_.size() + members_.size()) * index + host;
}

void DispersiveWeights::close() noexcept {
  static_inputs_ = {};
  inputs_ = {};
  term_multipliers_ = {};
  outputs_ = {};
  weights_ = {};
  quartics_ = {};
  input_scale_ = {};
  output_scale_ = {};
  factors_ = {};
  for (auto& staging : host_factors_)
    staging = {};
  term_rows_ = {};
  term_channels_ = {};
  members_ = {};
}

// --- the fused node path -------------------------------------------------------------

DispersiveResponse::DispersiveResponse(const Problem& problem, CpuExecutionPool& pool,
                                       StageViews views)
    : n_modes_(problem.n_modes), n_pol_state_(problem.n_pol_state),
      n_pol_transform_(problem.n_pol_transform),
      components_(problem.effective_node_pol()), columns_(problem.n_columns()),
      n_nodes_(problem.n_nodes), node_chunk_(problem.node_chunk),
      n_chunks_((problem.n_nodes + problem.node_chunk - 1) / problem.node_chunk),
      n_to_(problem.n_to), form_(static_cast<DispersiveForm>(problem.dispersive_form)),
      stage_(std::move(views)),
      weights_(problem,
               pool.get(CpuOperation::stage, problem.dispersive_inputs.size())) {
  if (problem.general_modal || problem.separable_geometry())
    throw std::invalid_argument(
        "the fused dispersive response runs on the legacy transform's node path");
  const bool complex = form_ != DispersiveForm::real;
  spectrum_bins_ = complex ? n_to_ : n_to_ / 2 + 1;
  const bool symmetric = weights_.symmetric();
  const auto fields = weights_.fields();
  const auto products = weights_.products();
  kernels_ = pool.get(CpuOperation::stage, products * columns_ * n_to_);
  response_chunk_ = problem.node_chunk;
  const auto tail = problem.n_nodes % response_chunk_;
  response_ =
      pool.get(CpuOperation::response, response_chunk_ * components_ * n_to_, n_to_);
  tail_response_ = pool.get(CpuOperation::response, tail * components_ * n_to_, n_to_);
  const auto element = complex ? 2U : 1U;
  const auto dense_elements =
      response_chunk_ * components_ * n_to_ * n_modes_ * element;
  const auto tail_dense_elements = tail * components_ * n_to_ * n_modes_ * element;
  matrix_copy_ = pool.get(CpuOperation::copy, n_modes_ * n_nodes_ * n_pol_transform_);
  dense_ = pool.get(CpuOperation::dense, dense_elements);
  tail_dense_ = pool.get(CpuOperation::dense, tail_dense_elements);
  analytic_ = problem.family == FieldFamily::field_analytic;
  embedded_ = !problem.projection_coarse_matrix.empty();
  // Alone, nothing else is projected or transformed beside the response; a lone one is
  // also a single product, which reads the stage's own node buffer.
  alone_ = !problem.nonlinear && !problem.plasma && !problem.raman &&
           !problem.envelope_plasma;
  lone_ = alone_ && symmetric && products == 1;
  in_place_ = lone_ && !problem.in_stage_absorber;
  borrowed_ = lone_ && !analytic_;
  shifted_ = analytic_ && !symmetric;

  const auto chunk_rows = node_chunk_ * components_;
  const auto field_rows = fields * columns_;
  if (lone_ && !in_place_ && borrowed_) {
    // The stage's projected spectrum is free until its forward transform: the
    // filtered spectrum's scratch.
    filtered_spectrum_ = stage_.projected_spectrum;
  } else if (!borrowed_) {
    spectra_ = ComplexView2("solprop_native.dispersive.product_spectra",
                            products * columns_, spectrum_bins_);
    // The filtered spectra live in the products' spectra, free until the products are
    // transformed; the rows have the same length on every layout.
    if (!in_place_)
      filtered_spectrum_ = Kokkos::subview(
          spectra_, std::make_pair(std::size_t{0}, field_rows), Kokkos::ALL());
  }
  if (!in_place_) {
    if (complex)
      filtered_complex_ =
          ComplexView2("solprop_native.dispersive.filtered_time", field_rows, n_to_);
    else
      filtered_real_ =
          RealView2("solprop_native.dispersive.filtered_time", field_rows, n_to_);
  }
  if (!lone_) {
    const auto slots = symmetric ? chunk_rows : 2 * chunk_rows;
    if (complex)
      slots_complex_ =
          ComplexView2("solprop_native.dispersive.node_fields", slots, n_to_);
    else
      slots_real_ = RealView2("solprop_native.dispersive.node_fields", slots, n_to_);
  }
  if (!borrowed_) {
    const auto rows = products * columns_;
    if (complex) {
      products_complex_ =
          ComplexView2("solprop_native.dispersive.products", rows, n_to_);
      if (embedded_)
        coarse_complex_ =
            ComplexView2("solprop_native.dispersive.coarse_products", rows, n_to_);
    } else {
      products_real_ = RealView2("solprop_native.dispersive.products", rows, n_to_);
      if (embedded_)
        coarse_real_ =
            RealView2("solprop_native.dispersive.coarse_products", rows, n_to_);
    }
    if (n_chunks_ > 1) {
      const auto packed = n_chunks_ * n_modes_ * node_chunk_ * n_pol_transform_;
      packed_projection_ =
          RealView("solprop_native.dispersive.projection_chunks", packed);
      if (embedded_)
        packed_coarse_ = RealView("solprop_native.dispersive.coarse_chunks", packed);
    }
  }
  if (shifted_) {
    dc_ = ComplexView("solprop_native.dispersive.zero_frequency", columns_);
    offset_ = ComplexView("solprop_native.dispersive.offsets", chunk_rows);
  }
  kernels_.fence("solprop_native dispersive construction");
}

DispersiveResponse::~DispersiveResponse() { close(); }

void DispersiveResponse::plan(CpuExecutionPool& transforms) {
  const auto space_for = [&](const std::size_t samples) -> const ExecutionSpace& {
    return transforms.get(CpuOperation::fft, samples);
  };
  constexpr int axis = -1;
  if (!in_place_) {
    if (form_ == DispersiveForm::real)
      inverse_real_ = std::make_unique<InverseRealPlan>(
          space_for(filtered_real_.size()), filtered_spectrum_, filtered_real_,
          KokkosFFT::Direction::backward, axis);
    else
      inverse_complex_ = std::make_unique<ComplexPlan>(
          space_for(filtered_complex_.size()), filtered_spectrum_, filtered_complex_,
          KokkosFFT::Direction::backward, axis);
  }
  if (borrowed_)
    return;
  if (form_ == DispersiveForm::real) {
    forward_real_ = std::make_unique<ForwardRealPlan>(
        space_for(products_real_.size()), products_real_, spectra_,
        KokkosFFT::Direction::forward, axis);
    if (embedded_)
      coarse_real_plan_ = std::make_unique<ForwardRealPlan>(
          space_for(coarse_real_.size()), coarse_real_, spectra_,
          KokkosFFT::Direction::forward, axis);
  } else {
    forward_complex_ = std::make_unique<ComplexPlan>(
        space_for(products_complex_.size()), products_complex_, spectra_,
        KokkosFFT::Direction::forward, axis);
    if (embedded_)
      coarse_complex_plan_ = std::make_unique<ComplexPlan>(
          space_for(coarse_complex_.size()), coarse_complex_, spectra_,
          KokkosFFT::Direction::forward, axis);
  }
}

void DispersiveResponse::prepare(const std::size_t stage) {
  weights_.update(stage);
  const auto fine = stage_.fine_spectrum;
  if (shifted_) {
    const auto dc = dc_;
    Kokkos::parallel_for(
        "solprop_native.dispersive.zero_frequency",
        Kokkos::RangePolicy<ExecutionSpace>(kernels_, 0, columns_),
        KOKKOS_LAMBDA(const std::size_t column) { dc(column) = fine(column, 0); });
  }
  if (in_place_) {
    filter_in_place(kernels_, weights_.inputs(), fine, columns_, weights_.bins());
    return;
  }
  filter_rows(kernels_, weights_.inputs(), fine, filtered_spectrum_, columns_,
              weights_.fields(), weights_.bins(), 1.0);
  if (inverse_real_)
    KokkosFFT::execute(*inverse_real_, filtered_spectrum_, filtered_real_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*inverse_complex_, filtered_spectrum_, filtered_complex_,
                       KokkosFFT::Normalization::none);
}

void DispersiveResponse::begin() {
  if (borrowed_ || n_chunks_ == 1)
    return;
  const auto chunk_rows = node_chunk_ * n_pol_transform_;
  const auto total_rows = n_nodes_ * n_pol_transform_;
  const auto modes = n_modes_;
  const auto packed = packed_projection_;
  const auto coarse = packed_coarse_;
  const auto projection = stage_.scaled_projection;
  const auto projection_coarse = stage_.scaled_projection_coarse;
  const bool embedded = embedded_;
  Kokkos::parallel_for(
      "solprop_native.dispersive.pack_projection",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right,
                                                         Kokkos::Iterate::Right>>(
          matrix_copy_, {0, 0}, {modes, total_rows}),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
        const auto chunk = row / chunk_rows;
        const auto local = row % chunk_rows;
        const auto rows = total_rows - chunk * chunk_rows < chunk_rows
                              ? total_rows - chunk * chunk_rows
                              : chunk_rows;
        const auto at = chunk * modes * chunk_rows + mode * rows + local;
        packed(at) = projection(mode, row);
        if (embedded)
          coarse(at) = projection_coarse(mode, row);
      });
}

void DispersiveResponse::synthesise(const std::size_t field,
                                    const std::size_t first_node,
                                    const std::size_t node_count,
                                    const bool complex_fields, double* const target) {
  const std::size_t element = complex_fields ? 2 : 1;
  const auto width = n_pol_state_ * n_to_ * element;
  const auto rows = node_count * n_pol_transform_;
  const auto first_row = first_node * n_pol_transform_;
  double* const filtered = complex_fields
                               ? reinterpret_cast<double*>(filtered_complex_.data())
                               : filtered_real_.data();
  const Matrix synthesis(stage_.scaled_synthesis.data() + first_row * n_modes_, rows,
                         n_modes_);
  const Matrix modal(filtered + field * columns_ * n_to_ * element, n_modes_, width);
  const Matrix nodes(target, rows, width);
  selected_gemm(dense_for(node_count), synthesis, modal, nodes, 0.0, true);
}

void DispersiveResponse::project(const std::size_t chunk, const std::size_t node_count,
                                 const std::size_t product, const double* const source,
                                 const bool first) {
  const bool complex = form_ != DispersiveForm::real;
  const std::size_t element = complex ? 2 : 1;
  const auto width = n_pol_state_ * n_to_ * element;
  const auto rows = node_count * n_pol_transform_;
  const auto offset = chunk * n_modes_ * node_chunk_ * n_pol_transform_;
  const Matrix nodes(const_cast<double*>(source), rows, width);
  const double beta = first ? 0.0 : 1.0;
  const auto block = product * columns_ * n_to_ * element;
  double* const fine = complex ? reinterpret_cast<double*>(products_complex_.data())
                               : products_real_.data();
  const Matrix projection(n_chunks_ == 1 ? stage_.scaled_projection.data()
                                         : packed_projection_.data() + offset,
                          n_modes_, rows);
  selected_gemm(dense_for(node_count), projection, nodes,
                Matrix(fine + block, n_modes_, width), beta, true);
  if (!embedded_)
    return;
  double* const coarse =
      complex ? reinterpret_cast<double*>(coarse_complex_.data()) : coarse_real_.data();
  const Matrix projection_coarse(n_chunks_ == 1 ? stage_.scaled_projection_coarse.data()
                                                : packed_coarse_.data() + offset,
                                 n_modes_, rows);
  selected_gemm(dense_for(node_count), projection_coarse, nodes,
                Matrix(coarse + block, n_modes_, width), beta, true);
}

void DispersiveResponse::offsets(const std::size_t first_node,
                                 const std::size_t node_count) {
  const auto synthesis = stage_.scaled_synthesis;
  const auto dc = dc_;
  const auto offset = offset_;
  const auto modes = n_modes_;
  const auto n_pol_state = n_pol_state_;
  const auto first_row = first_node * n_pol_transform_;
  Kokkos::parallel_for(
      "solprop_native.dispersive.offsets",
      Kokkos::RangePolicy<ExecutionSpace>(kernels_, 0, node_count * components_),
      KOKKOS_LAMBDA(const std::size_t row) {
        const auto transform_row = first_row + row / n_pol_state;
        const auto state_pol = row % n_pol_state;
        Complex value{};
        for (std::size_t mode = 0; mode < modes; ++mode)
          value += synthesis(transform_row, mode) * dc(mode * n_pol_state + state_pol);
        offset(row) = value;
      });
}

void DispersiveResponse::respond(const std::size_t chunk, const std::size_t first_node,
                                 const std::size_t node_count) {
  const bool complex = form_ != DispersiveForm::real;
  const bool first = chunk == 0;
  const double* const window = stage_.time_window.data();
  const Complex* const carrier = stage_.carrier.data();
  const auto self = [&](const double* source, double* target, const bool shifted) {
    self_products(response_for(node_count), form_, components_, node_count, n_to_,
                  source, target, window, carrier, offset_.data(), shifted);
  };
  double* const node = complex ? reinterpret_cast<double*>(stage_.node_complex.data())
                               : stage_.node_real.data();
  const std::size_t slot_size = node_chunk_ * components_ * n_to_ * (complex ? 2 : 1);
  if (weights_.symmetric()) {
    double* const field =
        lone_ ? node
              : (complex ? reinterpret_cast<double*>(slots_complex_.data())
                         : slots_real_.data());
    for (std::size_t group = 0; group < weights_.groups(); ++group) {
      // In place, the stage's own synthesis gave the filtered field at the nodes.
      if (!in_place_)
        synthesise(group, first_node, node_count, complex, field);
      double* const target =
          borrowed_
              ? (complex ? reinterpret_cast<double*>(stage_.polarisation_complex.data())
                         : stage_.polarisation_real.data())
              : field;
      self(field, target, false);
      if (!borrowed_)
        project(chunk, node_count, group, field, first);
    }
    return;
  }
  // The polynomial reads the bare field the stage synthesised; the cube is projected
  // before the derivative fields are formed, so two slots hold the three products in
  // turn.
  double* const derivative =
      complex ? reinterpret_cast<double*>(slots_complex_.data()) : slots_real_.data();
  double* const second = derivative + slot_size;
  if (shifted_)
    offsets(first_node, node_count);
  self(node, derivative, shifted_);
  project(chunk, node_count, 0, derivative, first);
  for (std::size_t group = 0; group < weights_.groups(); ++group) {
    synthesise(2 * group, first_node, node_count, complex, derivative);
    synthesise(2 * group + 1, first_node, node_count, complex, second);
    pair_products(response_for(node_count), form_, components_, node_count, n_to_, node,
                  derivative, second, weights_.pair(group, 0), weights_.pair(group, 1),
                  window, carrier, offset_.data(), shifted_);
    project(chunk, node_count, 1 + 2 * group, derivative, first);
    project(chunk, node_count, 2 + 2 * group, second, first);
  }
}

void DispersiveResponse::finish() {
  if (borrowed_)
    return;
  if (forward_real_)
    KokkosFFT::execute(*forward_real_, products_real_, spectra_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*forward_complex_, products_complex_, spectra_,
                       KokkosFFT::Normalization::none);
  coarse_pending_ = embedded_;
}

void DispersiveResponse::combine() {
  if (borrowed_) {
    // The stage's transforms made the product's spectra; `g` and `eps0 gamma0 rho`
    // finish them there, on both rules.
    weigh_rows(kernels_, weights_.weights(), stage_.projected_spectrum, columns_,
               weights_.bins());
    if (embedded_)
      weigh_rows(kernels_, weights_.weights(), stage_.projected_coarse_spectrum,
                 columns_, weights_.bins());
    return;
  }
  land_rows(kernels_, weights_.weights(), spectra_, stage_.projected_spectrum, columns_,
            weights_.bins(), weights_.products(), alone_);
}

void DispersiveResponse::finish_coarse() {
  if (!coarse_pending_)
    return;
  coarse_pending_ = false;
  if (coarse_real_plan_)
    KokkosFFT::execute(*coarse_real_plan_, coarse_real_, spectra_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*coarse_complex_plan_, coarse_complex_, spectra_,
                       KokkosFFT::Normalization::none);
  land_rows(kernels_, weights_.weights(), spectra_, stage_.projected_coarse_spectrum,
            columns_, weights_.bins(), weights_.products(), alone_);
}

std::map<std::string, std::size_t> DispersiveResponse::empty_ledger() {
  return {{"dispersive_filtered_time", 0},   {"dispersive_node_fields", 0},
          {"dispersive_products", 0},        {"dispersive_coarse_products", 0},
          {"dispersive_product_spectra", 0}, {"dispersive_projection_chunks", 0},
          {"dispersive_multipliers", 0}};
}

std::map<std::string, std::size_t> DispersiveResponse::memory_ledger() const {
  if (closed_)
    return empty_ledger();
  const auto real = sizeof(double);
  const auto complex = sizeof(Complex);
  return {
      {"dispersive_filtered_time",
       filtered_real_.size() * real + filtered_complex_.size() * complex},
      {"dispersive_node_fields",
       slots_real_.size() * real + slots_complex_.size() * complex},
      {"dispersive_products",
       products_real_.size() * real + products_complex_.size() * complex},
      {"dispersive_coarse_products",
       coarse_real_.size() * real + coarse_complex_.size() * complex},
      {"dispersive_product_spectra", spectra_.size() * complex},
      {"dispersive_projection_chunks",
       (packed_projection_.size() + packed_coarse_.size()) * real},
      {"dispersive_multipliers",
       weights_.bytes() + (dc_.size() + offset_.size()) * complex},
  };
}

void DispersiveResponse::close() noexcept {
  if (closed_)
    return;
  closed_ = true;
  kernels_.fence();
  inverse_real_.reset();
  inverse_complex_.reset();
  forward_real_.reset();
  forward_complex_.reset();
  coarse_real_plan_.reset();
  coarse_complex_plan_.reset();
  stage_ = {};
  weights_.close();
  filtered_spectrum_ = {};
  spectra_ = {};
  filtered_complex_ = {};
  slots_complex_ = {};
  products_complex_ = {};
  coarse_complex_ = {};
  filtered_real_ = {};
  slots_real_ = {};
  products_real_ = {};
  coarse_real_ = {};
  packed_projection_ = {};
  packed_coarse_ = {};
  dc_ = {};
  offset_ = {};
}

// --- the Cartesian free-space, line and planar routes
// ---------------------------------

SeparableDispersive::SeparableDispersive(const Problem& problem, CpuExecutionPool& pool,
                                         StageViews views)
    : n_to_(problem.n_to), n_x_(problem.n_x), n_p_(problem.n_p),
      form_(static_cast<DispersiveForm>(problem.dispersive_form)),
      slab_(problem.slab_geometry()), stage_(std::move(views)),
      weights_(problem,
               pool.get(CpuOperation::stage, problem.dispersive_inputs.size())) {
  if (!problem.separable_geometry())
    throw std::invalid_argument("the separable dispersive response runs on the "
                                "Cartesian free-space, line and planar routes");
  // A chunk's temporal rows: a free-space chunk's nodes times their components, a
  // planar block's (p, y) rows; and the rows of its node buffer.
  components_ = problem.effective_node_pol();
  node_rows_ = problem.node_chunk * components_;
  rows_ = slab_ ? (problem.node_chunk / n_x_) * n_p_ : node_rows_;
  require_size(stage_.fine_spectrum.extent(0), rows_, "chunk rows");
  const bool complex = form_ != DispersiveForm::real;
  spectrum_bins_ = complex ? n_to_ : n_to_ / 2 + 1;
  const bool symmetric = weights_.symmetric();
  const auto fields = weights_.fields();
  const auto products = weights_.products();
  kernels_ = pool.get(CpuOperation::stage, products * rows_ * n_to_);
  response_chunk_ = problem.node_chunk;
  const auto tail = problem.n_nodes % response_chunk_;
  active_rows_ = rows_;
  if constexpr (cpu_size_policy_enabled)
    tail_rows_ = slab_ ? (tail / n_x_) * n_p_ : tail * components_;
  tail_kernels_ = pool.get(CpuOperation::stage, products * tail_rows_ * n_to_);
  response_ =
      pool.get(CpuOperation::response, response_chunk_ * components_ * n_to_, n_to_);
  tail_response_ = pool.get(CpuOperation::response, tail * components_ * n_to_, n_to_);
  dense_ = pool.get(CpuOperation::dense, response_chunk_ * components_ * n_to_ * n_p_);
  tail_dense_ = pool.get(CpuOperation::dense, tail * components_ * n_to_ * n_p_);
  analytic_ = problem.family == FieldFamily::field_analytic;
  alone_ = !problem.nonlinear && !problem.plasma && !problem.raman &&
           !problem.envelope_plasma;
  lone_ = alone_ && symmetric && products == 1;
  // The in-stage absorber is a pass of its own here, which packs the chunk afresh.
  in_place_ = lone_;
  borrowed_ = lone_ && !analytic_;
  shifted_ = analytic_ && !symmetric;
  const auto product_rows = products * rows_;
  const auto field_rows = fields * rows_;
  if (!borrowed_) {
    spectra_ = ComplexView2("solprop_native.dispersive.product_spectra", product_rows,
                            spectrum_bins_);
    if (complex)
      products_complex_ =
          ComplexView2("solprop_native.dispersive.products", product_rows, n_to_);
    else
      products_real_ =
          RealView2("solprop_native.dispersive.products", product_rows, n_to_);
  }
  if (!in_place_) {
    // Out of place is never borrowed: the filtered spectra live in the products'.
    filtered_spectrum_ = Kokkos::subview(
        spectra_, std::make_pair(std::size_t{0}, field_rows), Kokkos::ALL());
    // Free space forms each product over its own filtered field, so the inverse
    // transform writes the filtered fields straight into the product blocks they
    // become: every block but the polynomial's cube. The planar route keeps them on
    // the rows, to be synthesised.
    const auto first = symmetric ? std::size_t{0} : rows_;
    const auto blocks = std::make_pair(first, first + field_rows);
    if (slab_ && complex)
      filtered_complex_ =
          ComplexView2("solprop_native.dispersive.filtered_time", field_rows, n_to_);
    else if (slab_)
      filtered_real_ =
          RealView2("solprop_native.dispersive.filtered_time", field_rows, n_to_);
    else if (complex)
      filtered_complex_ = Kokkos::subview(products_complex_, blocks, Kokkos::ALL());
    else
      filtered_real_ = Kokkos::subview(products_real_, blocks, Kokkos::ALL());
  }
  if (slab_ && !lone_) {
    // The planar route's node slots: the filtered field and its product, or the
    // polynomial's two derivative fields, at the block's nodes.
    const auto slots = (symmetric ? 1 : 2) * node_rows_;
    if (complex)
      slots_complex_ =
          ComplexView2("solprop_native.dispersive.node_fields", slots, n_to_);
    else
      slots_real_ = RealView2("solprop_native.dispersive.node_fields", slots, n_to_);
  }
  if (shifted_) {
    dc_ = ComplexView("solprop_native.dispersive.zero_frequency", rows_);
    if (slab_)
      offset_ = ComplexView("solprop_native.dispersive.offsets", node_rows_);
  }
  kernels_.fence("solprop_native separable dispersive construction");
}

SeparableDispersive::~SeparableDispersive() { close(); }

void SeparableDispersive::plan(CpuExecutionPool& transforms) {
  const auto space_for = [&](const std::size_t samples) -> const ExecutionSpace& {
    return transforms.get(CpuOperation::fft, samples);
  };
  constexpr int axis = -1;
  if (!in_place_) {
    if (form_ == DispersiveForm::real)
      inverse_real_ = std::make_unique<InverseRealPlan>(
          space_for(filtered_real_.size()), filtered_spectrum_, filtered_real_,
          KokkosFFT::Direction::backward, axis);
    else
      inverse_complex_ = std::make_unique<ComplexPlan>(
          space_for(filtered_complex_.size()), filtered_spectrum_, filtered_complex_,
          KokkosFFT::Direction::backward, axis);
  }
  if (borrowed_)
    return;
  if (form_ == DispersiveForm::real)
    forward_real_ = std::make_unique<ForwardRealPlan>(
        space_for(products_real_.size()), products_real_, spectra_,
        KokkosFFT::Direction::forward, axis);
  else
    forward_complex_ = std::make_unique<ComplexPlan>(
        space_for(products_complex_.size()), products_complex_, spectra_,
        KokkosFFT::Direction::forward, axis);
  if (tail_rows_ != 0) {
    const auto& tail_space = space_for(tail_rows_ * n_to_);
    if (!in_place_) {
      if (form_ == DispersiveForm::real)
        tail_inverse_real_.create(tail_space, filtered_spectrum_, filtered_real_, rows_,
                                  tail_rows_, KokkosFFT::Direction::backward);
      else
        tail_inverse_complex_.create(tail_space, filtered_spectrum_, filtered_complex_,
                                     rows_, tail_rows_, KokkosFFT::Direction::backward);
    }
    if (form_ == DispersiveForm::real)
      tail_forward_real_.create(tail_space, products_real_, spectra_, rows_, tail_rows_,
                                KokkosFFT::Direction::forward);
    else
      tail_forward_complex_.create(tail_space, products_complex_, spectra_, rows_,
                                   tail_rows_, KokkosFFT::Direction::forward);
  }
}

void SeparableDispersive::begin(const std::size_t stage) { weights_.update(stage); }

void SeparableDispersive::prepare(const std::size_t node_count) {
  active_tail_ = tail_rows_ != 0 && node_count != response_chunk_;
  active_rows_ = active_tail_ ? tail_rows_ : rows_;
  const auto& space = active_tail_ ? tail_kernels_ : kernels_;
  const auto fine = stage_.fine_spectrum;
  if (shifted_) {
    const auto dc = dc_;
    Kokkos::parallel_for(
        "solprop_native.dispersive.zero_frequency",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, active_rows_),
        KOKKOS_LAMBDA(const std::size_t row) { dc(row) = fine(row, 0); });
  }
  if (in_place_) {
    filter_in_place(space, weights_.inputs(), fine, active_rows_, weights_.bins());
    return;
  }
  filter_rows(space, weights_.inputs(), fine, filtered_spectrum_, rows_,
              weights_.fields(), weights_.bins(), 1.0, active_rows_);
  if (active_tail_) {
    if (inverse_real_)
      tail_inverse_real_.execute();
    else
      tail_inverse_complex_.execute();
    return;
  }
  if (inverse_real_)
    KokkosFFT::execute(*inverse_real_, filtered_spectrum_, filtered_real_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*inverse_complex_, filtered_spectrum_, filtered_complex_,
                       KokkosFFT::Normalization::none);
}

void SeparableDispersive::slab_offsets(const std::size_t units) {
  // The bare analytic signal's zero-frequency constant of each row, synthesised at the
  // block's nodes by the slab rule, as the bare field is.
  const Kokkos::View<Complex**, Layout, MemorySpace, Unmanaged> rows(dc_.data(), rows_,
                                                                     1);
  const Kokkos::View<Complex**, Layout, MemorySpace, Unmanaged> nodes(offset_.data(),
                                                                      node_rows_, 1);
  slab_contract("solprop_native.dispersive.slab_offsets", kernels_, stage_.synthesis,
                rows, nodes, units, 1);
}

void SeparableDispersive::respond(const std::size_t node_count) {
  const bool complex = form_ != DispersiveForm::real;
  const double* const window = stage_.time_window.data();
  const Complex* const carrier = stage_.carrier.data();
  const auto self = [&](const double* source, double* target, const bool shifted) {
    self_products(response_for(node_count), form_, components_, node_count, n_to_,
                  source, target, window, carrier, slab_ ? offset_.data() : dc_.data(),
                  shifted);
  };
  const auto pairs = [&](const double* field, double* first, double* second,
                         const std::size_t group) {
    pair_products(response_for(node_count), form_, components_, node_count, n_to_,
                  field, first, second, weights_.pair(group, 0),
                  weights_.pair(group, 1), window, carrier,
                  slab_ ? offset_.data() : dc_.data(), shifted_);
  };
  double* const node = complex ? reinterpret_cast<double*>(stage_.node_complex.data())
                               : stage_.node_real.data();
  double* const polarisation =
      complex ? reinterpret_cast<double*>(stage_.polarisation_complex.data())
              : stage_.polarisation_real.data();
  const auto product_row = [&](const std::size_t row) -> double* {
    return complex ? row_of(products_complex_, row) : row_of(products_real_, row);
  };
  if (!slab_) {
    // The chunk's rows are its nodes, which the stage's node buffer holds.
    if (borrowed_) {
      self(node, polarisation, false);
      return;
    }
    if (in_place_) {
      self(node, product_row(0), false);
      return;
    }
    if (weights_.symmetric()) {
      for (std::size_t group = 0; group < weights_.groups(); ++group)
        self(product_row(group * rows_), product_row(group * rows_), false);
      return;
    }
    // The derivative fields sit in the blocks their products take; the bare field is
    // the stage's node buffer, and the cube a block of its own.
    for (std::size_t group = 0; group < weights_.groups(); ++group)
      pairs(node, product_row((1 + 2 * group) * rows_),
            product_row((2 + 2 * group) * rows_), group);
    self(node, product_row(0), shifted_);
    return;
  }
  // The planar route: a block's `units` y rows, `width` (p, y) rows, `size` nodes.
  const auto units = node_count / n_x_;
  const auto width = units * n_p_;
  if (shifted_)
    slab_offsets(units);
  const auto rows_at = [](const auto& view, const std::size_t first,
                          const std::size_t count) {
    return Kokkos::subview(view, std::make_pair(first, first + count), Kokkos::ALL());
  };
  const auto synthesise = [&](const std::size_t field, const std::size_t slot) {
    if (complex)
      slab_contract("solprop_native.dispersive.slab_synthesis", dense_for(node_count),
                    stage_.synthesis, rows_at(filtered_complex_, field * rows_, width),
                    rows_at(slots_complex_, slot * node_rows_, node_count), units,
                    n_to_);
    else
      slab_contract("solprop_native.dispersive.slab_synthesis", dense_for(node_count),
                    stage_.synthesis, rows_at(filtered_real_, field * rows_, width),
                    rows_at(slots_real_, slot * node_rows_, node_count), units, n_to_);
  };
  // Project a block's node values onto its rows of one product.
  const auto project = [&](const bool from_node, const std::size_t slot,
                           const std::size_t product) {
    if (complex)
      slab_contract("solprop_native.dispersive.slab_projection", dense_for(node_count),
                    stage_.projection,
                    from_node ? rows_at(stage_.node_complex, 0, node_count)
                              : rows_at(slots_complex_, slot * node_rows_, node_count),
                    rows_at(products_complex_, product * rows_, width), units, n_to_);
    else
      slab_contract("solprop_native.dispersive.slab_projection", dense_for(node_count),
                    stage_.projection,
                    from_node ? rows_at(stage_.node_real, 0, node_count)
                              : rows_at(slots_real_, slot * node_rows_, node_count),
                    rows_at(products_real_, product * rows_, width), units, n_to_);
  };
  const auto slot_of = [&](const std::size_t slot) -> double* {
    return complex ? row_of(slots_complex_, slot * node_rows_)
                   : row_of(slots_real_, slot * node_rows_);
  };
  if (borrowed_) {
    // In place, the stage synthesised the filtered field; its product is the stage's
    // polarisation, which the stage projects.
    self(node, polarisation, false);
    return;
  }
  if (in_place_) {
    self(node, node, false);
    project(true, 0, 0);
    return;
  }
  if (weights_.symmetric()) {
    for (std::size_t group = 0; group < weights_.groups(); ++group) {
      synthesise(group, 0);
      self(slot_of(0), slot_of(0), false);
      project(false, 0, group);
    }
    return;
  }
  self(node, slot_of(0), shifted_);
  project(false, 0, 0);
  for (std::size_t group = 0; group < weights_.groups(); ++group) {
    synthesise(2 * group, 0);
    synthesise(2 * group + 1, 1);
    pairs(node, slot_of(0), slot_of(1), group);
    project(false, 0, 1 + 2 * group);
    project(false, 1, 2 + 2 * group);
  }
}

void SeparableDispersive::finish() {
  if (borrowed_)
    return;
  if (active_tail_) {
    if (forward_real_)
      tail_forward_real_.execute();
    else
      tail_forward_complex_.execute();
    return;
  }
  if (forward_real_)
    KokkosFFT::execute(*forward_real_, products_real_, spectra_,
                       KokkosFFT::Normalization::none);
  else
    KokkosFFT::execute(*forward_complex_, products_complex_, spectra_,
                       KokkosFFT::Normalization::none);
}

void SeparableDispersive::combine() {
  const auto& space = active_tail_ ? tail_kernels_ : kernels_;
  if (borrowed_) {
    weigh_rows(space, weights_.weights(), stage_.projected_spectrum, active_rows_,
               weights_.bins());
    return;
  }
  land_rows(space, weights_.weights(), spectra_, stage_.projected_spectrum, rows_,
            weights_.bins(), weights_.products(), alone_, active_rows_);
}

std::map<std::string, std::size_t> SeparableDispersive::memory_ledger() const {
  if (closed_)
    return DispersiveResponse::empty_ledger();
  const auto real = sizeof(double);
  const auto complex = sizeof(Complex);
  // Free space's filtered fields are blocks of its products; the planar route's its
  // own.
  const auto filtered =
      slab_ ? filtered_real_.size() * real + filtered_complex_.size() * complex : 0;
  auto ledger = DispersiveResponse::empty_ledger();
  ledger["dispersive_filtered_time"] = filtered;
  ledger["dispersive_node_fields"] =
      slots_real_.size() * real + slots_complex_.size() * complex;
  ledger["dispersive_products"] =
      products_real_.size() * real + products_complex_.size() * complex;
  ledger["dispersive_product_spectra"] = spectra_.size() * complex;
  ledger["dispersive_multipliers"] =
      weights_.bytes() + (dc_.size() + offset_.size()) * complex;
  return ledger;
}

void SeparableDispersive::close() noexcept {
  if (closed_)
    return;
  closed_ = true;
  kernels_.fence();
  tail_inverse_real_.clear();
  tail_inverse_complex_.clear();
  tail_forward_real_.clear();
  tail_forward_complex_.clear();
  inverse_real_.reset();
  inverse_complex_.reset();
  forward_real_.reset();
  forward_complex_.reset();
  stage_ = {};
  weights_.close();
  filtered_spectrum_ = {};
  spectra_ = {};
  filtered_complex_ = {};
  slots_complex_ = {};
  products_complex_ = {};
  filtered_real_ = {};
  slots_real_ = {};
  products_real_ = {};
  dc_ = {};
  offset_ = {};
}

// --- the general modal route --------------------------------------------------------

GeneralDispersive::GeneralDispersive(const Problem& problem, CpuExecutionPool& pool,
                                     StageViews views)
    : rows_(problem.general_node_chunk * problem.general_components),
      components_(problem.general_components), n_to_(problem.n_to),
      real_(static_cast<DispersiveForm>(problem.dispersive_form) ==
            DispersiveForm::real),
      stage_(std::move(views)),
      weights_(problem,
               pool.get(CpuOperation::stage, problem.dispersive_inputs.size())) {
  if (!problem.general_modal)
    throw std::invalid_argument(
        "the general dispersive response runs on the general modal route");
  require_size(stage_.spectrum.extent(0), rows_, "chunk rows");
  spectrum_bins_ = real_ ? n_to_ / 2 + 1 : n_to_;
  shifted_ = !real_ && !weights_.symmetric();
  const auto fields = weights_.fields() * rows_;
  const auto products = weights_.products() * rows_;
  kernels_ = pool.get(CpuOperation::stage, products * n_to_);
  response_chunk_ = problem.general_node_chunk;
  const auto tail = problem.general_nodes % response_chunk_;
  active_rows_ = rows_;
  last_chunk_ = problem.general_nodes / response_chunk_;
  if constexpr (cpu_size_policy_enabled)
    tail_rows_ = tail * components_;
  tail_kernels_ =
      pool.get(CpuOperation::stage, weights_.products() * tail_rows_ * n_to_);
  response_ =
      pool.get(CpuOperation::response, response_chunk_ * components_ * n_to_, n_to_);
  tail_response_ = pool.get(CpuOperation::response, tail * components_ * n_to_, n_to_);
  filtered_spectrum_ =
      ComplexView2("solprop_native.dispersive.filtered_spectrum", fields, n_to_);
  filtered_ = ComplexView2("solprop_native.dispersive.filtered_time", fields, n_to_);
  if (real_)
    products_real_ = RealView2("solprop_native.dispersive.products", products, n_to_);
  else
    products_complex_ =
        ComplexView2("solprop_native.dispersive.products", products, n_to_);
  spectra_ = ComplexView2("solprop_native.dispersive.product_spectra", products,
                          spectrum_bins_);
  if (shifted_)
    dc_ = ComplexView("solprop_native.dispersive.zero_frequency", rows_);
  node_region_ =
      IndexView("solprop_native.dispersive.node_region", problem.general_nodes);
  upload(kernels_, node_region_, problem.dispersive_node_region);
  // Which chunks have a node of any channel's region, decided once.
  const auto& region = problem.dispersive_node_region;
  const auto chunk = problem.general_node_chunk;
  for (std::size_t first = 0; first < problem.general_nodes; first += chunk) {
    const auto last = std::min(first + chunk, problem.general_nodes);
    bool any = false;
    for (std::size_t node = first; node < last && !any; ++node)
      for (std::size_t group = 0; group < weights_.groups() && !any; ++group)
        any = region[node] == weights_.key(group);
    live_.push_back(any ? 1U : 0U);
  }
  kernels_.fence("solprop_native general dispersive construction");
}

GeneralDispersive::~GeneralDispersive() { close(); }

void GeneralDispersive::plan(CpuExecutionPool& transforms) {
  constexpr int axis = -1;
  inverse_ = std::make_unique<ComplexPlan>(
      transforms.get(CpuOperation::fft, filtered_.size()), filtered_spectrum_,
      filtered_, KokkosFFT::Direction::backward, axis);
  if (real_)
    forward_real_ = std::make_unique<ForwardRealPlan>(
        transforms.get(CpuOperation::fft, products_real_.size()), products_real_,
        spectra_, KokkosFFT::Direction::forward, axis);
  else
    forward_complex_ = std::make_unique<ComplexPlan>(
        transforms.get(CpuOperation::fft, products_complex_.size()), products_complex_,
        spectra_, KokkosFFT::Direction::forward, axis);
  if (tail_rows_ != 0) {
    const auto& space = transforms.get(CpuOperation::fft, tail_rows_ * n_to_);
    tail_inverse_.create(space, filtered_spectrum_, filtered_, rows_, tail_rows_,
                         KokkosFFT::Direction::backward);
    if (real_)
      tail_forward_real_.create(space, products_real_, spectra_, rows_, tail_rows_,
                                KokkosFFT::Direction::forward);
    else
      tail_forward_complex_.create(space, products_complex_, spectra_, rows_,
                                   tail_rows_, KokkosFFT::Direction::forward);
  }
}

void GeneralDispersive::begin(const std::size_t stage) { weights_.update(stage); }

void GeneralDispersive::prepare(const std::size_t chunk) {
  active_tail_ = tail_rows_ != 0 && chunk == last_chunk_;
  active_rows_ = active_tail_ ? tail_rows_ : rows_;
  const auto& space = active_tail_ ? tail_kernels_ : kernels_;
  if (!live(chunk))
    return;
  // The route's synthesised spectrum keeps the inverse transform's `1 / n_to`.
  const double inverse = 1.0 / static_cast<double>(n_to_);
  const auto spectrum = stage_.spectrum;
  if (shifted_) {
    const auto dc = dc_;
    Kokkos::parallel_for(
        "solprop_native.dispersive.zero_frequency",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, active_rows_),
        KOKKOS_LAMBDA(const std::size_t row) { dc(row) = inverse * spectrum(row, 0); });
  }
  filter_rows(space, weights_.inputs(), spectrum, filtered_spectrum_, rows_,
              weights_.fields(), weights_.bins(), inverse, active_rows_);
  if (active_tail_) {
    tail_inverse_.execute();
    return;
  }
  KokkosFFT::execute(*inverse_, filtered_spectrum_, filtered_,
                     KokkosFFT::Normalization::none);
}

template <bool Real, std::size_t Components>
void GeneralDispersive::products(const std::size_t first, const std::size_t count) {
  using Value = std::conditional_t<Real, double, Complex>;
  Value* const products = Real ? reinterpret_cast<Value*>(products_real_.data())
                               : reinterpret_cast<Value*>(products_complex_.data());
  const Complex* const filtered = filtered_.data();
  const Complex* const field = stage_.field.data();
  const std::size_t* const region = node_region_.data();
  const Complex* const offset = dc_.data();
  const double inverse = 1.0 / static_cast<double>(n_to_);
  const auto block = rows_ * n_to_;
  if (weights_.symmetric()) {
    for (std::size_t group = 0; group < weights_.groups(); ++group)
      over_nodes("solprop_native.dispersive.general_cube", response_for(count), count,
                 n_to_,
                 GeneralCube<Real, Components>{
                     filtered + group * block, products + group * block, region,
                     weights_.key(group), first, n_to_, 1.0, offset, false});
    return;
  }
  // One cube per region, on the bare signal; each group's pairs at its region.
  for (std::size_t cube = 0; cube < weights_.cubes(); ++cube)
    over_nodes("solprop_native.dispersive.general_cube", response_for(count), count,
               n_to_,
               GeneralCube<Real, Components>{field, products + cube * block, region,
                                             weights_.cube_key(cube), first, n_to_,
                                             inverse, offset, shifted_});
  const auto cubes = weights_.cubes();
  for (std::size_t group = 0; group < weights_.groups(); ++group)
    over_nodes(
        "solprop_native.dispersive.general_pairs", response_for(count), count, n_to_,
        GeneralPairs<Real, Components>{
            field, filtered + 2 * group * block, filtered + (2 * group + 1) * block,
            products + (cubes + 2 * group) * block,
            products + (cubes + 2 * group + 1) * block, weights_.pair(group, 0),
            weights_.pair(group, 1), region, weights_.key(group), first, n_to_, inverse,
            offset, shifted_});
}

void GeneralDispersive::respond(const std::size_t chunk, const std::size_t first,
                                const std::size_t count) {
  if (!live(chunk))
    return;
  if (real_) {
    if (components_ == 3)
      products<true, 3>(first, count);
    else if (components_ == 2)
      products<true, 2>(first, count);
    else
      products<true, 1>(first, count);
    if (active_tail_) {
      tail_forward_real_.execute();
      return;
    }
    KokkosFFT::execute(*forward_real_, products_real_, spectra_,
                       KokkosFFT::Normalization::none);
    return;
  }
  if (components_ == 3)
    products<false, 3>(first, count);
  else if (components_ == 2)
    products<false, 2>(first, count);
  else
    products<false, 1>(first, count);
  if (active_tail_) {
    tail_forward_complex_.execute();
    return;
  }
  KokkosFFT::execute(*forward_complex_, products_complex_, spectra_,
                     KokkosFFT::Normalization::none);
}

void GeneralDispersive::combine(const bool overwrite) {
  land_rows(active_tail_ ? tail_kernels_ : kernels_, weights_.weights(), spectra_,
            stage_.source, rows_, weights_.bins(), weights_.products(), overwrite,
            active_rows_);
}

std::map<std::string, std::size_t> GeneralDispersive::memory_ledger() const {
  if (closed_)
    return DispersiveResponse::empty_ledger();
  const auto real = sizeof(double);
  const auto complex = sizeof(Complex);
  auto ledger = DispersiveResponse::empty_ledger();
  ledger["dispersive_filtered_time"] = filtered_.size() * complex;
  ledger["dispersive_products"] =
      products_real_.size() * real + products_complex_.size() * complex;
  // As `numba_cpu` counts them: the filtered spectra beside the products'.
  ledger["dispersive_product_spectra"] =
      (spectra_.size() + filtered_spectrum_.size()) * complex;
  ledger["dispersive_multipliers"] = weights_.bytes() + dc_.size() * complex +
                                     node_region_.size() * sizeof(std::size_t) +
                                     live_.capacity();
  return ledger;
}

void GeneralDispersive::close() noexcept {
  if (closed_)
    return;
  closed_ = true;
  kernels_.fence();
  tail_inverse_.clear();
  tail_forward_real_.clear();
  tail_forward_complex_.clear();
  inverse_.reset();
  forward_real_.reset();
  forward_complex_.reset();
  stage_ = {};
  weights_.close();
  node_region_ = {};
  filtered_spectrum_ = {};
  filtered_ = {};
  products_complex_ = {};
  products_real_ = {};
  spectra_ = {};
  dc_ = {};
}

} // namespace solprop_native
