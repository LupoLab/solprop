// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/coupled_flow.hpp"
#include "solprop_native/launch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace solprop_native {
namespace {

std::size_t product(const std::size_t left, const std::size_t right, const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string("coupled flow extent overflows: ") + name);
  return left * right;
}

void require_size(const std::size_t actual, const std::size_t expected,
                  const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string("coupled flow ") + name + " has " +
                                std::to_string(actual) + " values, expected " +
                                std::to_string(expected));
}

bool finite(const std::vector<Complex>& values) {
  return std::all_of(values.begin(), values.end(), [](const Complex& value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
  });
}

template <class View, class Value>
void upload(const ExecutionSpace& execution, const View& destination,
            const std::vector<Value>& source) {
  auto mirror = Kokkos::create_mirror_view(destination);
  auto* data = mirror.data();
  for (std::size_t index = 0; index < source.size(); ++index)
    data[index] = source[index];
  Kokkos::deep_copy(execution, destination, mirror);
}

KOKKOS_INLINE_FUNCTION Complex flow_exponential(const Complex value) {
  const double scale = Kokkos::exp(value.real());
  return {scale * Kokkos::cos(value.imag()), scale * Kokkos::sin(value.imag())};
}

std::size_t block_size(const std::size_t count, const Problem& problem,
                       const char* name) {
  return product(product(count, problem.flow_active, name),
                 product(problem.n_modes, problem.n_modes, name), name);
}

using ComplexView3 = CoupledFlowAction::ComplexView3;
using ComplexView4 = CoupledFlowAction::ComplexView4;

// solprop's reference `_TAYLOR_ORDER` and `_TAYLOR_SCALE`
// (`numpy_ref/backend.py`): the series degree and the row-sum norm the exponent is
// scaled below; the host refuses a length needing more than 30 squarings first.
constexpr int kTaylorOrder = 16;
constexpr double kTaylorScale = 0.5;
// Cyclic Jacobi sweeps a Gram matrix is given; a few channels settle in four to six,
// and whatever is left is added to the bound (Weyl), so it never under-measures.
constexpr int kJacobiSweeps = 30;
constexpr double kUnit = 0x1p-53;

// `W exp(l Lambda) Y` and `W exp(-l Lambda) Y` of one bin (solprop
// `numba_cpu.coupled_flow.form_flow_actions`): one real exponential each way and one
// cosine and sine per eigenvalue, then every entry summed from zero over the
// eigenvalue index.
KOKKOS_INLINE_FUNCTION void
eigenbasis_actions(const double length, const std::size_t section,
                   const std::size_t bin, const std::size_t slot,
                   const std::size_t modes, const ComplexView4& left,
                   const ComplexView4& right, const ComplexView3& eigenvalues,
                   const ComplexView3& exponents, const ComplexView4& forward,
                   const ComplexView4& inverse) {
  for (std::size_t mode = 0; mode < modes; ++mode) {
    const Complex value = eigenvalues(section, bin, mode);
    const double real = length * value.real();
    const double imaginary = length * value.imag();
    const double cosine = Kokkos::cos(imaginary);
    const double sine = Kokkos::sin(imaginary);
    const double grow = Kokkos::exp(real);
    const double decay = Kokkos::exp(-real);
    exponents(bin, 0, mode) = Complex(grow * cosine, grow * sine);
    exponents(bin, 1, mode) = Complex(decay * cosine, -(decay * sine));
  }
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      Complex ahead{};
      Complex back{};
      for (std::size_t inner = 0; inner < modes; ++inner) {
        const Complex entry = left(section, bin, row, inner);
        const Complex weight = right(section, bin, inner, column);
        ahead += (entry * exponents(bin, 0, inner)) * weight;
        back += (entry * exponents(bin, 1, inner)) * weight;
      }
      forward(slot, bin, row, column) = ahead;
      inverse(slot, bin, row, column) = back;
    }
}

// `Z exp(l T) Z^H` of one bin by solprop's reference scaling, series and squaring
// (`_triangular_exponential` and `_flow_action`), with the bin's own squaring count.
KOKKOS_INLINE_FUNCTION void
squared_series_action(const double length, const std::size_t fallback,
                      const std::size_t modes, const ComplexView3& vectors,
                      const ComplexView3& triangular, const ComplexView4& work,
                      const ComplexView4& out, const std::size_t slot,
                      const std::size_t bin) {
  double largest = 0.0;
  for (std::size_t row = 0; row < modes; ++row) {
    double total = 0.0;
    for (std::size_t column = 0; column < modes; ++column)
      total += Kokkos::abs(length * triangular(fallback, row, column));
    largest = total > largest ? total : largest;
  }
  int squarings = 0;
  if (largest > kTaylorScale)
    squarings = static_cast<int>(Kokkos::ceil(Kokkos::log2(largest / kTaylorScale)));
  double divisor = 1.0;
  for (int index = 0; index < squarings; ++index)
    divisor *= 2.0;
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      work(fallback, 0, row, column) =
          (length * triangular(fallback, row, column)) / divisor;
      work(fallback, 1, row, column) = row == column ? Complex(1.0, 0.0) : Complex{};
    }
  for (int term = kTaylorOrder; term > 0; --term) {
    for (std::size_t row = 0; row < modes; ++row)
      for (std::size_t column = 0; column < modes; ++column) {
        Complex total{};
        for (std::size_t inner = 0; inner < modes; ++inner)
          total += work(fallback, 0, row, inner) * work(fallback, 1, inner, column);
        work(fallback, 2, row, column) = total;
      }
    for (std::size_t row = 0; row < modes; ++row)
      for (std::size_t column = 0; column < modes; ++column) {
        const Complex diagonal = row == column ? Complex(1.0, 0.0) : Complex{};
        work(fallback, 1, row, column) =
            diagonal + work(fallback, 2, row, column) / static_cast<double>(term);
      }
  }
  for (int index = 0; index < squarings; ++index) {
    for (std::size_t row = 0; row < modes; ++row)
      for (std::size_t column = 0; column < modes; ++column) {
        Complex total{};
        for (std::size_t inner = 0; inner < modes; ++inner)
          total += work(fallback, 1, row, inner) * work(fallback, 1, inner, column);
        work(fallback, 2, row, column) = total;
      }
    for (std::size_t row = 0; row < modes; ++row)
      for (std::size_t column = 0; column < modes; ++column)
        work(fallback, 1, row, column) = work(fallback, 2, row, column);
  }
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      Complex total{};
      for (std::size_t inner = 0; inner < modes; ++inner)
        total += vectors(fallback, row, inner) * work(fallback, 1, inner, column);
      work(fallback, 2, row, column) = total;
    }
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      Complex total{};
      for (std::size_t inner = 0; inner < modes; ++inner)
        total += work(fallback, 2, row, inner) *
                 Kokkos::conj(vectors(fallback, column, inner));
      out(slot, bin, row, column) = total;
    }
}

// The largest diagonal entry of one bin's Gram matrix, and the sum of the squared
// moduli of its entries above the diagonal.
KOKKOS_INLINE_FUNCTION void gram_extent(const ComplexView3& gram, const std::size_t bin,
                                        const std::size_t modes, double& largest,
                                        double& off) {
  largest = 0.0;
  off = 0.0;
  for (std::size_t row = 0; row < modes; ++row) {
    const double diagonal = gram(bin, row, row).real();
    largest = diagonal > largest ? diagonal : largest;
    for (std::size_t column = row + 1; column < modes; ++column) {
      const Complex value = gram(bin, row, column);
      off += value.real() * value.real() + value.imag() * value.imag();
    }
  }
}

// An upper bound on the spectral norm of one action: the largest eigenvalue of its
// Gram matrix by cyclic Jacobi rotations (a phase making the pivot real, then the
// symmetric rotation zeroing it; Golub and Van Loan section 8.5) plus the Frobenius
// norm of what is left off the diagonal (solprop
// `numba_cpu.coupled_flow._spectral_norm`).
KOKKOS_INLINE_FUNCTION double spectral_norm(const ComplexView4& actions,
                                            const std::size_t slot,
                                            const std::size_t bin,
                                            const std::size_t modes,
                                            const ComplexView3& gram) {
  for (std::size_t row = 0; row < modes; ++row)
    for (std::size_t column = 0; column < modes; ++column) {
      Complex total{};
      for (std::size_t inner = 0; inner < modes; ++inner)
        total += Kokkos::conj(actions(slot, bin, inner, row)) *
                 actions(slot, bin, inner, column);
      gram(bin, row, column) = total;
    }
  double largest = 0.0;
  double off = 0.0;
  gram_extent(gram, bin, modes, largest, off);
  for (int sweep = 0;
       sweep < kJacobiSweeps && off > (kUnit * largest) * (kUnit * largest); ++sweep) {
    for (std::size_t first = 0; first < modes; ++first)
      for (std::size_t second = first + 1; second < modes; ++second) {
        const Complex pivot = gram(bin, first, second);
        const double size = Kokkos::abs(pivot);
        if (size == 0.0)
          continue;
        const double low = gram(bin, first, first).real();
        const double high = gram(bin, second, second).real();
        const Complex phase = pivot / size;
        const double spread = (high - low) / (2.0 * size);
        double tangent =
            1.0 / (Kokkos::abs(spread) + Kokkos::sqrt(1.0 + spread * spread));
        if (spread < 0.0)
          tangent = -tangent;
        const double cosine = 1.0 / Kokkos::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        const Complex back = Kokkos::conj(phase);
        for (std::size_t index = 0; index < modes; ++index) {
          const Complex one = gram(bin, index, first);
          const Complex two = gram(bin, index, second);
          gram(bin, index, first) = cosine * one - sine * back * two;
          gram(bin, index, second) = sine * one + cosine * back * two;
        }
        for (std::size_t index = 0; index < modes; ++index) {
          const Complex one = gram(bin, first, index);
          const Complex two = gram(bin, second, index);
          gram(bin, first, index) = cosine * one - sine * phase * two;
          gram(bin, second, index) = sine * one + cosine * phase * two;
        }
        gram(bin, first, second) = Complex{};
        gram(bin, second, first) = Complex{};
        gram(bin, first, first) = Complex(low - tangent * size, 0.0);
        gram(bin, second, second) = Complex(high + tangent * size, 0.0);
      }
    gram_extent(gram, bin, modes, largest, off);
  }
  return Kokkos::sqrt(largest + Kokkos::sqrt(2.0 * off));
}

} // namespace

namespace {

std::size_t fallback_count(const Problem& problem) {
  return static_cast<std::size_t>(
      std::count_if(problem.flow_route.begin(), problem.flow_route.end(),
                    [](const std::size_t value) { return value != 0; }));
}

void require_flow_eigenbasis(const Problem& problem) {
  if (problem.flow_left.empty() && problem.flow_right.empty() &&
      problem.flow_eigenvalues.empty() && problem.flow_route.empty() &&
      problem.flow_fallback_vectors.empty() && problem.flow_fallback_triangular.empty())
    return;
  const auto sections = problem.flow_sections;
  require_size(problem.flow_left.size(), block_size(sections, problem, "eigenbasis"),
               "left eigenbasis");
  require_size(problem.flow_right.size(), block_size(sections, problem, "eigenbasis"),
               "right eigenbasis");
  const auto bins = product(sections, problem.flow_active, "route");
  require_size(problem.flow_eigenvalues.size(),
               product(bins, problem.n_modes, "eigenvalues"), "eigenvalues");
  require_size(problem.flow_route.size(), bins, "route");
  std::size_t expected = 1;
  for (const auto route : problem.flow_route) {
    if (route == 0)
      continue;
    if (route != expected)
      throw std::invalid_argument("flow_route must number the scaled-and-squared bins "
                                  "1, 2, ... in row-major order");
    ++expected;
  }
  const auto fallback = expected - 1;
  const auto blocks = product(
      fallback, product(problem.n_modes, problem.n_modes, "fallback"), "fallback");
  require_size(problem.flow_fallback_vectors.size(), blocks, "fallback vectors");
  require_size(problem.flow_fallback_triangular.size(), blocks, "fallback triangular");
  if (!finite(problem.flow_left) || !finite(problem.flow_right) ||
      !finite(problem.flow_eigenvalues) || !finite(problem.flow_fallback_vectors) ||
      !finite(problem.flow_fallback_triangular))
    throw std::invalid_argument("coupled flow eigenbasis data must be finite");
}

} // namespace

void require_coupled_flow(const Problem& problem) {
  if (!problem.coupled_flow) {
    if (problem.flow_active != 0 || problem.flow_sections != 0 ||
        problem.flow_terms != 0 || !problem.flow_slot.empty() ||
        !problem.flow_frozen.empty() || !problem.flow_residual_terms.empty() ||
        !problem.flow_left.empty() || !problem.flow_route.empty())
      throw std::invalid_argument("coupled flow arrays require coupled_flow");
    return;
  }
  // The action contracts the channel axis of a scalar state; the synthesis may resolve
  // two transverse components, which the flow never reads.
  if (problem.n_pol_state != 1 || problem.separable_geometry())
    throw std::invalid_argument("a coupled flow mixes one channel axis of a scalar "
                                "modal state, with no polarisation axis");
  if (problem.modified_noise)
    throw std::invalid_argument("a coupled flow carries no noise model");
  if (problem.n_stages == 0)
    throw std::invalid_argument("a coupled flow needs a tableau");
  if (problem.flow_active == 0 || problem.flow_active > problem.n_omega)
    throw std::invalid_argument("flow_active must lie in [1, n_omega]");
  if (problem.flow_sections == 0)
    throw std::invalid_argument("a coupled flow has at least one section");
  require_size(problem.flow_slot.size(), problem.n_omega, "slot");
  std::size_t expected = 1;
  for (const auto slot : problem.flow_slot) {
    if (slot == 0)
      continue;
    if (slot != expected)
      throw std::invalid_argument("flow_slot must number the active bins 1, 2, ... "
                                  "in increasing frequency");
    ++expected;
  }
  if (expected - 1 != problem.flow_active)
    throw std::invalid_argument("flow_slot names a different number of active bins "
                                "than flow_active");
  static_cast<void>(block_size(problem.n_stages + 1, problem, "actions"));
  require_flow_eigenbasis(problem);
  if (problem.flow_terms == 0) {
    if (!problem.flow_frozen.empty() || !problem.flow_residual_terms.empty())
      throw std::invalid_argument("a coupled flow with no explicit residual carries "
                                  "no frozen blocks or residual terms");
    return;
  }
  require_size(problem.flow_frozen.size(),
               block_size(problem.flow_sections, problem, "frozen"), "frozen");
  require_size(problem.flow_residual_terms.size(),
               block_size(problem.flow_terms, problem, "residual terms"),
               "residual terms");
  if (!finite(problem.flow_frozen) || !finite(problem.flow_residual_terms))
    throw std::invalid_argument("coupled flow residual data must be finite");
}

CoupledFlowAction::CoupledFlowAction(const Problem& problem,
                                     const ExecutionSpace& kernels)
    : n_modes_(problem.n_modes), n_omega_(problem.n_omega), n_stages_(problem.n_stages),
      active_(problem.flow_active), sections_(problem.flow_sections),
      terms_(problem.flow_terms), identity_(problem.n_stages + 1, 0U),
      listed_counts_(problem.n_stages, 0U), kernels_(kernels) {
  require_coupled_flow(problem);
  if (!problem.coupled_flow)
    throw std::invalid_argument("the problem does not declare a coupled flow");
  const auto slots = n_stages_ + 1;
  slot_ = IndexView("solprop_native.flow.slot", n_omega_);
  forward_ =
      ComplexView4("solprop_native.flow.forward", slots, active_, n_modes_, n_modes_);
  inverse_ =
      ComplexView4("solprop_native.flow.inverse", slots, active_, n_modes_, n_modes_);
  input_ = ComplexView2("solprop_native.flow.input", n_modes_, n_omega_);
  drive_ = ComplexView2("solprop_native.flow.drive", n_modes_, n_omega_);
  state_ = ComplexView2("solprop_native.flow.carried_state", n_modes_, n_omega_);
  solution_ = ComplexView2("solprop_native.flow.carried_solution", n_modes_, n_omega_);
  error_ = ComplexView2("solprop_native.flow.carried_error", n_modes_, n_omega_);
  if (terms_ != 0) {
    frozen_ = ComplexView4("solprop_native.flow.frozen", sections_, active_, n_modes_,
                           n_modes_);
    terms_view_ = ComplexView4("solprop_native.flow.residual_terms", terms_, active_,
                               n_modes_, n_modes_);
    weights_ = RealView2("solprop_native.flow.weights", n_stages_, terms_);
    active_terms_ = IndexView2("solprop_native.flow.active_terms", n_stages_, terms_);
    active_counts_ = IndexView("solprop_native.flow.active_counts", n_stages_);
    // The same Views on a host build; persistent host copies on a device one, so a
    // trial's listing allocates nothing either way.
    host_active_terms_ = Kokkos::create_mirror_view(active_terms_);
    host_active_counts_ = Kokkos::create_mirror_view(active_counts_);
    lab_ = ComplexView2("solprop_native.flow.lab", n_modes_, n_omega_);
    upload(kernels_, frozen_, problem.flow_frozen);
    upload(kernels_, terms_view_, problem.flow_residual_terms);
  }
  upload(kernels_, slot_, problem.flow_slot);
  forms_ = !problem.flow_left.empty();
  if (forms_) {
    fallback_ = fallback_count(problem);
    left_ = ComplexView4("solprop_native.flow.left", sections_, active_, n_modes_,
                         n_modes_);
    right_ = ComplexView4("solprop_native.flow.right", sections_, active_, n_modes_,
                          n_modes_);
    eigenvalues_ =
        ComplexView3("solprop_native.flow.eigenvalues", sections_, active_, n_modes_);
    route_ = IndexView2("solprop_native.flow.route", sections_, active_);
    fallback_vectors_ = ComplexView3("solprop_native.flow.fallback_vectors", fallback_,
                                     n_modes_, n_modes_);
    fallback_triangular_ = ComplexView3("solprop_native.flow.fallback_triangular",
                                        fallback_, n_modes_, n_modes_);
    fallback_work_ = ComplexView4("solprop_native.flow.fallback_work", fallback_, 3,
                                  n_modes_, n_modes_);
    exponents_ = ComplexView3("solprop_native.flow.exponents", active_, 2, n_modes_);
    gram_ = ComplexView3("solprop_native.flow.gram", active_, n_modes_, n_modes_);
    fractions_ = RealView("solprop_native.flow.fractions", slots);
    measured_ = RealView("solprop_native.flow.measured", active_);
    host_fractions_ = Kokkos::create_mirror_view(fractions_);
    host_measured_ = Kokkos::create_mirror_view(measured_);
    measured_host_.assign(active_, 0.0);
    upload(kernels_, left_, problem.flow_left);
    upload(kernels_, right_, problem.flow_right);
    upload(kernels_, eigenvalues_, problem.flow_eigenvalues);
    upload(kernels_, route_, problem.flow_route);
    upload(kernels_, fallback_vectors_, problem.flow_fallback_vectors);
    upload(kernels_, fallback_triangular_, problem.flow_fallback_triangular);
  }
  kernels_.fence("solprop_native coupled flow upload");
}

const std::vector<double>&
CoupledFlowAction::form_actions(const std::size_t section, const double dz,
                                const std::vector<double>& fractions) {
  if (closed_)
    throw std::logic_error("the coupled flow is closed");
  if (!forms_)
    throw std::logic_error("this coupled flow carries no eigenbasis to form its "
                           "actions from");
  if (section >= sections_)
    throw std::invalid_argument("flow_section names no section of the coupled flow");
  require_size(fractions.size(), n_stages_ + 1, "stage fractions");
  if (!std::isfinite(dz) || dz <= 0.0 ||
      !std::all_of(fractions.begin(), fractions.end(), [](const double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
      }))
    throw std::invalid_argument("coupled flow step and fractions must be finite, the "
                                "step positive and the fractions in [0, 1]");
  if (fractions.back() == 0.0)
    throw std::invalid_argument("the whole step's coupled flow action is never the "
                                "identity");
  for (std::size_t slot = 0; slot <= n_stages_; ++slot) {
    host_fractions_(slot) = fractions[slot];
    identity_.at(slot) = fractions[slot] == 0.0 ? 1U : 0U;
  }
  // Forgotten first: a refusal of what this forms leaves no action set behind.
  ready_ = false;
  Kokkos::deep_copy(kernels_, fractions_, host_fractions_);
  const auto slots = n_stages_ + 1;
  const auto whole = n_stages_;
  const auto modes = n_modes_;
  const auto held_fractions = fractions_;
  const auto route = route_;
  const auto left = left_;
  const auto right = right_;
  const auto eigenvalues = eigenvalues_;
  const auto exponents = exponents_;
  const auto vectors = fallback_vectors_;
  const auto triangular = fallback_triangular_;
  const auto work = fallback_work_;
  const auto forward = forward_;
  const auto inverse = inverse_;
  const auto gram = gram_;
  const auto measured = measured_;
  launch::parallel_for(
      "solprop_native.flow.form_actions",
      Kokkos::RangePolicy<ExecutionSpace>(kernels_, 0, active_),
      KOKKOS_LAMBDA(const std::size_t bin) {
        const auto path = route(section, bin);
        for (std::size_t slot = 0; slot < slots; ++slot) {
          const double fraction = held_fractions(slot);
          if (fraction == 0.0)
            continue;
          // A fraction the tableau repeats (DOPRI5's last node and the whole step)
          // is the same matrix; it is copied, as the reference's memo reuses it.
          std::size_t earlier = slot;
          for (std::size_t other = 0; other < slot; ++other)
            if (held_fractions(other) == fraction) {
              earlier = other;
              break;
            }
          if (earlier != slot) {
            for (std::size_t row = 0; row < modes; ++row)
              for (std::size_t column = 0; column < modes; ++column) {
                forward(slot, bin, row, column) = forward(earlier, bin, row, column);
                inverse(slot, bin, row, column) = inverse(earlier, bin, row, column);
              }
            continue;
          }
          const double length = fraction * dz;
          if (path == 0) {
            eigenbasis_actions(length, section, bin, slot, modes, left, right,
                               eigenvalues, exponents, forward, inverse);
          } else {
            squared_series_action(length, path - 1, modes, vectors, triangular, work,
                                  forward, slot, bin);
            squared_series_action(-length, path - 1, modes, vectors, triangular, work,
                                  inverse, slot, bin);
          }
        }
        measured(bin) = spectral_norm(forward, whole, bin, modes, gram) *
                        spectral_norm(inverse, whole, bin, modes, gram);
      });
  Kokkos::deep_copy(kernels_, host_measured_, measured_);
  kernels_.fence("solprop_native coupled flow formation");
  std::copy(host_measured_.data(), host_measured_.data() + active_,
            measured_host_.begin());
  ready_ = true;
  return measured_host_;
}

CoupledFlowAction::~CoupledFlowAction() { close(); }

void CoupledFlowAction::set_actions(const std::vector<Complex>& forward,
                                    const std::vector<Complex>& inverse,
                                    const std::vector<unsigned char>& identity) {
  if (closed_)
    throw std::logic_error("the coupled flow is closed");
  require_size(forward.size(), forward_.size(), "forward actions");
  require_size(inverse.size(), inverse_.size(), "inverse actions");
  require_size(identity.size(), n_stages_ + 1, "identity flags");
  if (!finite(forward) || !finite(inverse))
    throw std::invalid_argument("coupled flow actions must be finite");
  if (identity.back() != 0U)
    throw std::invalid_argument("the whole step's coupled flow action is never the "
                                "identity");
  upload(kernels_, forward_, forward);
  upload(kernels_, inverse_, inverse);
  identity_ = identity;
  ready_ = true;
}

void CoupledFlowAction::set_trial(const std::size_t section,
                                  const std::vector<double>& weights) {
  if (closed_)
    throw std::logic_error("the coupled flow is closed");
  if (!ready_)
    throw std::logic_error("no coupled flow actions have been loaded");
  if (section >= sections_)
    throw std::invalid_argument("flow_section names no section of the coupled flow");
  section_ = section;
  if (terms_ == 0) {
    if (!weights.empty())
      throw std::invalid_argument("a coupled flow with no explicit residual takes no "
                                  "flow_weights");
    return;
  }
  require_size(weights.size(), n_stages_ * terms_, "flow_weights");
  if (!std::all_of(weights.begin(), weights.end(),
                   [](const double value) { return std::isfinite(value); }))
    throw std::invalid_argument("flow_weights must be finite");
  upload(kernels_, weights_, weights);
  const auto& listed = host_active_terms_;
  const auto& counts = host_active_counts_;
  for (std::size_t stage = 0; stage < n_stages_; ++stage) {
    std::size_t count = 0;
    for (std::size_t term = 0; term < terms_; ++term)
      if (weights[stage * terms_ + term] != 0.0)
        listed(stage, count++) = term;
    counts(stage) = count;
    listed_counts_.at(stage) = count;
  }
  Kokkos::deep_copy(kernels_, active_terms_, listed);
  Kokkos::deep_copy(kernels_, active_counts_, counts);
}

std::vector<Complex> CoupledFlowAction::actions_to_host(const bool inverse) const {
  if (closed_)
    throw std::logic_error("the coupled flow is closed");
  if (!ready_)
    throw std::logic_error("no coupled flow actions have been loaded");
  kernels_.fence("solprop_native coupled flow observation");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        inverse ? inverse_ : forward_);
  return {host.data(), host.data() + host.size()};
}

bool CoupledFlowAction::identity(const std::size_t stage) const {
  return identity_.at(stage) != 0U;
}

void CoupledFlowAction::carry_input(const std::size_t stage, const double dz,
                                    const ComplexView2& current,
                                    const ComplexView3& stages,
                                    const RealView& tableau) {
  const auto slot = slot_;
  const auto forward = forward_;
  const auto input = input_;
  const auto modes = n_modes_;
  const auto n_stages = n_stages_;
  // The reference's `stage_input(U u, U k, ...)`: the state and the stage sum are
  // each carried and only then combined, in that order.
  launch::parallel_for(
      "solprop_native.flow.carry_input", Range2(kernels_, {0, 0}, {modes, n_omega_}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        const auto one = slot(index);
        if (one == 0) {
          Complex accumulated{};
          for (std::size_t previous = 0; previous < stage; ++previous)
            accumulated +=
                tableau(stage * n_stages + previous) * stages(previous, row, index);
          input(row, index) = current(row, index) + dz * accumulated;
          return;
        }
        const auto bin = one - 1;
        Complex carried_state{};
        Complex carried_sum{};
        for (std::size_t column = 0; column < modes; ++column) {
          Complex accumulated{};
          for (std::size_t previous = 0; previous < stage; ++previous)
            accumulated +=
                tableau(stage * n_stages + previous) * stages(previous, column, index);
          const Complex action = forward(stage, bin, row, column);
          carried_state += action * current(column, index);
          carried_sum += action * accumulated;
        }
        input(row, index) = carried_state + dz * carried_sum;
      });
}

void CoupledFlowAction::add_residual(const std::size_t stage,
                                     const ComplexView2& stage_phase) {
  if (terms_ == 0)
    return;
  const auto slot = slot_;
  const auto frozen = frozen_;
  const auto terms = terms_view_;
  const auto weights = weights_;
  const auto listed = active_terms_;
  const auto count = listed_counts_.at(stage);
  const auto lab = lab_;
  const auto drive = drive_;
  const auto modes = n_modes_;
  const auto section = section_;
  // `R_s(z_i) = sum_t w_t G_t - G_s` on the active bins, zero off them (solprop
  // `_residual_action` and `_residual_drive`), carried back by the stage's own
  // `exp(-dPhi)` before the block's inverse action is applied. The sum runs over the
  // stage's non-zero weights in increasing term order: a zero weight adds an exact
  // zero to a running sum that starts at +0 and so is never -0, which leaves every
  // bit as the full sum has it.
  launch::parallel_for(
      "solprop_native.flow.add_residual", Range2(kernels_, {0, 0}, {modes, n_omega_}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        const auto one = slot(index);
        if (one == 0)
          return;
        const auto bin = one - 1;
        Complex residual{};
        for (std::size_t column = 0; column < modes; ++column) {
          Complex combined{};
          for (std::size_t entry = 0; entry < count; ++entry) {
            const auto term = listed(stage, entry);
            combined += weights(stage, term) * terms(term, bin, row, column);
          }
          residual +=
              (combined - frozen(section, bin, row, column)) * lab(column, index);
        }
        drive(row, index) += flow_exponential(-stage_phase(row, index)) * residual;
      });
}

void CoupledFlowAction::carry_back(const std::size_t stage,
                                   const double solution_weight,
                                   const double error_weight,
                                   const ComplexView3& stages,
                                   const ComplexView2& solution,
                                   const ComplexView2& error) {
  const auto slot = slot_;
  const auto inverse = inverse_;
  const auto drive = drive_;
  const auto modes = n_modes_;
  const bool identity = identity_.at(stage) != 0U;
  launch::parallel_for(
      "solprop_native.flow.carry_back", Range2(kernels_, {0, 0}, {modes, n_omega_}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        const auto one = slot(index);
        Complex value{};
        if (identity || one == 0) {
          value = drive(row, index);
        } else {
          const auto bin = one - 1;
          for (std::size_t column = 0; column < modes; ++column)
            value += inverse(stage, bin, row, column) * drive(column, index);
        }
        stages(stage, row, index) = value;
        if (stage == 0) {
          solution(row, index) = solution_weight * value;
          error(row, index) = error_weight * value;
        } else {
          solution(row, index) += solution_weight * value;
          error(row, index) += error_weight * value;
        }
      });
}

void CoupledFlowAction::carry_step(const ComplexView2& current,
                                   const ComplexView2& solution,
                                   const ComplexView2& error) {
  const auto slot = slot_;
  const auto forward = forward_;
  const auto state = state_;
  const auto carried_solution = solution_;
  const auto carried_error = error_;
  const auto modes = n_modes_;
  const auto whole = n_stages_;
  launch::parallel_for(
      "solprop_native.flow.carry_step", Range2(kernels_, {0, 0}, {modes, n_omega_}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        const auto one = slot(index);
        if (one == 0) {
          state(row, index) = current(row, index);
          carried_solution(row, index) = solution(row, index);
          carried_error(row, index) = error(row, index);
          return;
        }
        const auto bin = one - 1;
        Complex state_value{};
        Complex solution_value{};
        Complex error_value{};
        for (std::size_t column = 0; column < modes; ++column) {
          const Complex action = forward(whole, bin, row, column);
          state_value += action * current(column, index);
          solution_value += action * solution(column, index);
          error_value += action * error(column, index);
        }
        state(row, index) = state_value;
        carried_solution(row, index) = solution_value;
        carried_error(row, index) = error_value;
      });
}

std::map<std::string, std::size_t> CoupledFlowAction::empty_ledger() {
  return {
      {"flow_slot", 0},
      {"flow_forward", 0},
      {"flow_inverse", 0},
      {"flow_frozen", 0},
      {"flow_residual_terms", 0},
      {"flow_weights", 0},
      {"flow_active_terms", 0},
      {"flow_input", 0},
      {"flow_lab", 0},
      {"flow_drive", 0},
      {"flow_carried_state", 0},
      {"flow_carried_solution", 0},
      {"flow_carried_error", 0},
      {"flow_eigenbasis", 0},
      {"flow_fallback", 0},
      {"flow_formation", 0},
  };
}

std::map<std::string, std::size_t> CoupledFlowAction::memory_ledger() const {
  constexpr auto complex_bytes = sizeof(Complex);
  constexpr auto real_bytes = sizeof(double);
  constexpr auto index_bytes = sizeof(std::size_t);
  return {
      {"flow_slot", slot_.size() * index_bytes},
      {"flow_forward", forward_.size() * complex_bytes},
      {"flow_inverse", inverse_.size() * complex_bytes},
      {"flow_frozen", frozen_.size() * complex_bytes},
      {"flow_residual_terms", terms_view_.size() * complex_bytes},
      {"flow_weights", weights_.size() * real_bytes},
      {"flow_active_terms",
       (active_terms_.size() + active_counts_.size()) * sizeof(std::size_t)},
      {"flow_input", input_.size() * complex_bytes},
      {"flow_lab", lab_.size() * complex_bytes},
      {"flow_drive", drive_.size() * complex_bytes},
      {"flow_carried_state", state_.size() * complex_bytes},
      {"flow_carried_solution", solution_.size() * complex_bytes},
      {"flow_carried_error", error_.size() * complex_bytes},
      {"flow_eigenbasis",
       (left_.size() + right_.size() + eigenvalues_.size()) * complex_bytes +
           route_.size() * index_bytes},
      {"flow_fallback", (fallback_vectors_.size() + fallback_triangular_.size() +
                         fallback_work_.size()) *
                            complex_bytes},
      {"flow_formation", (exponents_.size() + gram_.size()) * complex_bytes +
                             (fractions_.size() + measured_.size()) * real_bytes},
  };
}

void CoupledFlowAction::close() noexcept {
  if (closed_)
    return;
  kernels_.fence();
  slot_ = {};
  forward_ = {};
  inverse_ = {};
  frozen_ = {};
  terms_view_ = {};
  weights_ = {};
  active_terms_ = {};
  active_counts_ = {};
  host_active_terms_ = {};
  host_active_counts_ = {};
  input_ = {};
  lab_ = {};
  drive_ = {};
  state_ = {};
  solution_ = {};
  error_ = {};
  left_ = {};
  right_ = {};
  eigenvalues_ = {};
  route_ = {};
  fallback_vectors_ = {};
  fallback_triangular_ = {};
  fallback_work_ = {};
  exponents_ = {};
  gram_ = {};
  fractions_ = {};
  measured_ = {};
  host_fractions_ = {};
  host_measured_ = {};
  ready_ = false;
  forms_ = false;
  closed_ = true;
}

} // namespace solprop_native
