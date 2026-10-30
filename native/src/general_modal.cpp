// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/general_modal.hpp"
#include "solprop_native/fft_plan.hpp"
#include "solprop_native/launch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace solprop_native {
namespace {

using ComplexView2 = GeneralModalSource::ComplexView2;
using RealView2 = GeneralModalSource::RealView2;

constexpr std::size_t max_components = 3;

// Unroll the loop that follows completely, on GCC. A sample's contraction over a
// compile-time component count runs as nested loops whose inner bounds follow the outer
// index. GCC vectorises the enclosing loop over samples only when those loops are gone
// before its vectoriser runs, and its early complete unrolling leaves such a nest alone
// unless asked: the loop then ran a sample at a time, packing pairs through the stack
// (a failed store-to-load forward per sample), at about 60 % of a general modal run.
// Clang and icpx unroll the nest themselves and device builds keep their own unrolling,
// so the macro is empty there. Unrolling moves no rounding.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__clang__)
#define SOLPROP_NATIVE_UNROLL
#elif defined(__GNUC__)
#define SOLPROP_NATIVE_UNROLL _Pragma("GCC unroll 16")
#else
#define SOLPROP_NATIVE_UNROLL
#endif

// The source's vector loops capture their Views by value, once per loop: unmanaged
// copies carry no reference count to maintain, and the by-value capture lets the
// compiler see that the loop's loads and stores do not move the Views themselves.
template <class Data>
using UnmanagedView =
    Kokkos::View<Data, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// The stored entries of the symmetric pair history, in solprop's `_UPPER` order:
// xx, yy, zz, xy, xz, yz.
KOKKOS_INLINE_FUNCTION std::size_t pair_row(const std::size_t entry) {
  return entry < 3 ? entry : (entry == 5 ? 1 : 0);
}
KOKKOS_INLINE_FUNCTION std::size_t pair_column(const std::size_t entry) {
  return entry < 3 ? entry : (entry == 3 ? 1 : 2);
}

std::size_t product(const std::size_t left, const std::size_t right, const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string("general modal extent overflows: ") + name);
  return left * right;
}

void require_size(const std::size_t actual, const std::size_t expected,
                  const char* name) {
  if (actual != expected)
    throw std::invalid_argument(std::string("general modal ") + name + " has " +
                                std::to_string(actual) + " values, expected " +
                                std::to_string(expected));
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
  Kokkos::deep_copy(execution, destination, mirror);
}

constexpr std::size_t fourth_power(const std::size_t value) {
  return value * value * value * value;
}

// The real cubic's monomials `(E_j E_k) E_l`, `j <= k <= l`, in lexicographic order
// (solprop `symmetric_monomials`): ten for three components, four for two, one for one.
constexpr std::size_t symmetric_monomials(const std::size_t components) {
  return components * (components + 1) * (components + 2) / 6;
}

// The position of the monomial an ordered triple sorts to, among `components`'
// monomials in their order.
KOKKOS_INLINE_FUNCTION constexpr std::size_t
symmetric_monomial(const std::size_t components, std::size_t j, std::size_t k,
                   std::size_t l) {
  if (j > k) {
    const auto swap = j;
    j = k;
    k = swap;
  }
  if (k > l) {
    const auto swap = k;
    k = l;
    l = swap;
  }
  if (j > k) {
    const auto swap = j;
    j = k;
    k = swap;
  }
  std::size_t position = 0;
  for (std::size_t a = 0; a < components; ++a)
    for (std::size_t b = a; b < components; ++b)
      for (std::size_t c = b; c < components; ++c) {
        if (a == j && b == k && c == l)
          return position;
        ++position;
      }
  return position;
}
static_assert(symmetric_monomials(3) == 10 && symmetric_monomial(3, 2, 0, 1) == 4 &&
              symmetric_monomial(3, 2, 2, 2) == 9 &&
              symmetric_monomial(2, 1, 0, 1) == 2);

bool general_crop_is_contiguous(const Problem& problem) {
  const auto& crop = problem.crop_index;
  if (crop.size() != problem.n_omega || crop.empty())
    return false;
  for (std::size_t omega = 0; omega < crop.size(); ++omega)
    if (crop[omega] != crop.front() + omega)
      return false;
  return true;
}

// The vector length of the source's team kernels. A host build runs a team's vector
// range as one vectorised loop and ignores it; a device gets its whole warp or
// wavefront, where eight lanes left three quarters of a CUDA warp and seven eighths of
// an AMD wavefront idle. Every lane computes its own bin or
// sample alone, with no reduction across lanes, so the width moves no bit.
int general_team_lanes() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return Kokkos::TeamPolicy<ExecutionSpace>::vector_length_max();
#else
  return static_cast<int>(general_vector_lanes);
#endif
}
} // namespace

void require_general_modal(const Problem& problem) {
  if (!problem.general_modal) {
    if (!problem.general_synthesis.empty() || !problem.general_projection.empty() ||
        !problem.general_tensor.empty() || !problem.general_frequency_factor.empty() ||
        !problem.general_delayed_tensor.empty() ||
        !problem.general_delayed_spectrum.empty() ||
        !problem.general_node_response.empty())
      throw std::invalid_argument("general modal arrays require general_modal");
    return;
  }
  const auto components = problem.general_components;
  const auto nodes = problem.general_nodes;
  if (components == 0 || components > max_components)
    throw std::invalid_argument("the general modal route resolves one to three "
                                "Cartesian components");
  if (nodes == 0 || problem.general_node_chunk == 0 ||
      problem.general_node_chunk > nodes)
    throw std::invalid_argument("general_node_chunk must lie in [1, general_nodes]");
  if (problem.family != FieldFamily::field_real || problem.n_pol_state != 1 ||
      problem.n_pol_transform != 1 || problem.separable_geometry())
    throw std::invalid_argument("the general modal route runs one channel axis on a "
                                "field grid with no polarisation axis");
  if (problem.nonlinear || problem.plasma || problem.raman || problem.envelope_plasma)
    throw std::invalid_argument("the general modal route carries its own regional "
                                "responses and no legacy nonlinear term");
  if (problem.n_to == 0 || problem.n_to % 2 != 0)
    throw std::invalid_argument("the general modal route needs an even fine grid");
  // A field grid's coarse bins are the leading bins of the fine spectrum (solprop
  // `fourier.crop_index`), so the source reads and writes them as one contiguous
  // block, which its vector loops need.
  if (!general_crop_is_contiguous(problem))
    throw std::invalid_argument("the general modal route needs the coarse bins to be "
                                "one contiguous block of the fine spectrum");
  if (!std::isfinite(problem.general_pad_scale) || problem.general_pad_scale <= 0.0 ||
      !std::isfinite(problem.general_epsilon0) || problem.general_epsilon0 <= 0.0)
    throw std::invalid_argument(
        "general modal pad scale and vacuum permittivity must be finite and positive");
  if (problem.general_map_intervals == 0 || problem.general_map_n_deg == 0)
    throw std::invalid_argument("general modal maps carry at least one section of at "
                                "least one Chebyshev term along z");
  const auto map_terms =
      product(problem.general_map_intervals, problem.general_map_n_deg, "map terms");
  if (map_terms > 1 && problem.n_stages == 0)
    throw std::invalid_argument("moving general modal maps need at least one stage");
  const auto rows = product(nodes, components, "node rows");
  const auto maps =
      product(product(rows, problem.n_modes, "maps"), problem.n_omega, "maps");
  require_size(problem.general_synthesis.size(), product(map_terms, maps, "map terms"),
               "synthesis");
  require_size(problem.general_projection.size(), product(map_terms, maps, "map terms"),
               "projection");
  if (map_terms > 1)
    static_cast<void>(product(product(problem.n_stages, map_terms, "map weights"),
                              sizeof(double), "map weight bytes"));
  require_size(problem.general_tensor.size(),
               product(nodes, fourth_power(components), "tensor"), "tensor");
  require_size(problem.general_frequency_factor.size(), problem.n_omega,
               "frequency factor");
  if (!finite(problem.general_synthesis) || !finite(problem.general_projection) ||
      !finite(problem.general_tensor) || !finite(problem.general_frequency_factor))
    throw std::invalid_argument("general modal maps must be finite");
  const auto chunk_rows = product(problem.general_node_chunk, components, "chunk rows");
  static_cast<void>(product(product(chunk_rows, problem.n_to, "chunk buffers"),
                            sizeof(Complex), "chunk bytes"));
  if (problem.general_terms == 0) {
    if (problem.general_responses != 0 || !problem.general_delayed_tensor.empty() ||
        !problem.general_delayed_spectrum.empty() ||
        !problem.general_node_response.empty())
      throw std::invalid_argument("a general modal problem with no delayed term "
                                  "carries no delayed arrays");
    return;
  }
  if (components != max_components)
    throw std::invalid_argument(
        "a delayed general modal response needs the three Cartesian components");
  if (problem.general_responses == 0)
    throw std::invalid_argument("a delayed general modal term needs a response");
  const auto channels =
      product(problem.general_responses, problem.general_terms, "delayed channels");
  require_size(problem.general_delayed_tensor.size(),
               product(channels, fourth_power(components), "delayed tensor"),
               "delayed tensor");
  require_size(problem.general_delayed_spectrum.size(),
               product(channels, problem.n_to + 1, "delayed spectrum"),
               "delayed spectrum");
  require_size(problem.general_node_response.size(), nodes, "node response");
  if (!finite(problem.general_delayed_tensor) ||
      !finite(problem.general_delayed_spectrum))
    throw std::invalid_argument("general modal delayed data must be finite");
  for (const auto owner : problem.general_node_response)
    if (owner > problem.general_responses)
      throw std::invalid_argument("general modal node response names no response");
  static_cast<void>(product(
      product(product(problem.general_node_chunk, general_pair_histories, "pairs"),
              2 * problem.n_to, "pair buffers"),
      sizeof(Complex), "pair bytes"));
}

GeneralModalSource::GeneralModalSource(const Problem& problem)
    : n_modes_(problem.n_modes), n_omega_(problem.n_omega), n_to_(problem.n_to),
      components_(problem.general_components), nodes_(problem.general_nodes),
      chunk_(problem.general_node_chunk), terms_(problem.general_terms),
      map_terms_(problem.general_map_intervals * problem.general_map_n_deg),
      map_n_deg_(problem.general_map_n_deg), n_stages_(problem.n_stages),
      crop_offset_(problem.crop_index.empty() ? 0 : problem.crop_index.front()),
      real_form_(problem.general_real_form), epsilon0_(problem.general_epsilon0) {
  require_general_modal(problem);
  if (!problem.general_modal)
    throw std::invalid_argument("the problem does not declare the general modal route");
  kernels_ = kernel_pool_.get(CpuOperation::stage, n_modes_ * n_omega_);
  response_ =
      kernel_pool_.get(CpuOperation::response, chunk_ * components_ * n_to_, n_to_);
  tail_response_ = kernel_pool_.get(CpuOperation::response,
                                    (nodes_ % chunk_) * components_ * n_to_, n_to_);
  dense_ =
      kernel_pool_.get(CpuOperation::modal, chunk_ * components_ * n_omega_ * n_modes_);
  tail_dense_ = kernel_pool_.get(CpuOperation::modal,
                                 (nodes_ % chunk_) * components_ * n_omega_ * n_modes_);
  projection_lanes_ = general_projection_lanes(
      n_modes_, n_omega_, static_cast<std::size_t>(dense_.concurrency()));
  // A gas region's dispersive channels respond where every regional tensor may be
  // zero; with no regional tensor and no delayed response the contraction is
  // identically zero, and it and its transform are skipped (solprop `numba_cpu`).
  const bool tensor =
      std::any_of(problem.general_tensor.begin(), problem.general_tensor.end(),
                  [](const double value) { return value != 0.0; });
  regional_ = tensor || terms_ != 0;
  responds_ = tensor || problem.dispersive;
  const auto chunk_rows = chunk_ * components_;
  if (moving()) {
    // A moving guide holds every coefficient of both maps and one evaluated map of
    // each, which `combine` rewrites per stage; the weights and their host copy are
    // allocated here so that a trial allocates nothing.
    const auto one_map = problem.general_synthesis.size() / map_terms_;
    maps_ = kernel_pool_.get(CpuOperation::modal, 2 * one_map * map_n_deg_);
    synthesis_terms_ = ComplexView("solprop_native.general.synthesis_terms",
                                   problem.general_synthesis.size());
    projection_terms_ = ComplexView("solprop_native.general.projection_terms",
                                    problem.general_projection.size());
    synthesis_ = ComplexView("solprop_native.general.stage_synthesis", one_map);
    projection_ = ComplexView("solprop_native.general.stage_projection", one_map);
    map_weights_ =
        RealView2("solprop_native.general.map_weights", n_stages_, map_terms_);
    host_map_weights_ = Kokkos::create_mirror_view(map_weights_);
    stage_sections_.assign(n_stages_, 0);
  } else {
    synthesis_ = ComplexView("solprop_native.general.synthesis",
                             problem.general_synthesis.size());
    projection_ = ComplexView("solprop_native.general.projection",
                              problem.general_projection.size());
  }
  frequency_factor_ = ComplexView("solprop_native.general.frequency_factor", n_omega_);
  tensor_ = RealView("solprop_native.general.tensor", problem.general_tensor.size());
  scaled_ = ComplexView2("solprop_native.general.scaled_state", n_modes_, n_omega_);
  source_ = ComplexView2("solprop_native.general.source", n_modes_, n_omega_);
  node_spectrum_ =
      ComplexView2("solprop_native.general.node_spectrum", chunk_rows, n_to_);
  node_field_ = ComplexView2("solprop_native.general.node_field", chunk_rows, n_to_);
  real_signal_ = general_real_signal_enabled && real_form_ && !problem.dispersive;
  if (real_signal_) {
    half_spectrum_ =
        AliasComplexView2(node_spectrum_.data(), chunk_rows, n_to_ / 2 + 1);
    real_field_ = AliasRealView2(reinterpret_cast<double*>(node_field_.data()),
                                 chunk_rows, n_to_);
  }
  node_polarisation_ =
      RealView2("solprop_native.general.node_polarisation", chunk_rows, n_to_);
  node_source_ =
      ComplexView2("solprop_native.general.node_source", chunk_rows, n_to_ / 2 + 1);
  if (terms_ != 0) {
    const auto pair_rows = chunk_ * general_pair_histories;
    delayed_tensor_ = RealView("solprop_native.general.delayed_tensor",
                               problem.general_delayed_tensor.size());
    delayed_spectrum_ = ComplexView("solprop_native.general.delayed_spectrum",
                                    problem.general_delayed_spectrum.size());
    node_response_ = IndexView("solprop_native.general.node_response", nodes_);
    node_drive_ = RealView2("solprop_native.general.node_drive", pair_rows, 2 * n_to_);
    node_drive_spectrum_ = ComplexView2("solprop_native.general.node_drive_spectrum",
                                        pair_rows, n_to_ + 1);
    node_product_ =
        ComplexView2("solprop_native.general.node_product", pair_rows, n_to_ + 1);
    node_histories_ =
        RealView2("solprop_native.general.node_histories", pair_rows, 2 * n_to_);
    node_delayed_ = RealView2("solprop_native.general.node_delayed", chunk_rows, n_to_);
    upload(kernels_, delayed_tensor_, problem.general_delayed_tensor);
    upload(kernels_, delayed_spectrum_, problem.general_delayed_spectrum);
    upload(kernels_, node_response_, problem.general_node_response);
  }
  upload(kernels_, moving() ? synthesis_terms_ : synthesis_, problem.general_synthesis);
  upload(kernels_, moving() ? projection_terms_ : projection_,
         problem.general_projection);
  upload(kernels_, frequency_factor_, problem.general_frequency_factor);
  upload(kernels_, tensor_, problem.general_tensor);
  if (problem.dispersive)
    dispersive_ = std::make_unique<GeneralDispersive>(
        problem, kernel_pool_,
        GeneralDispersive::StageViews{node_spectrum_, node_field_, node_source_});
  kernels_.fence("solprop_native general modal upload");
}

GeneralModalSource::~GeneralModalSource() { close(); }

void GeneralModalSource::plan() {
  transforms_ = fft_pool_.get(CpuOperation::fft, node_field_.size());
  delayed_transforms_ = raman_pool_.get(CpuOperation::raman, node_drive_.size());
  constexpr int axis = -1;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if (real_signal_)
    seed_row_fft_wisdom(transforms_, half_spectrum_, real_field_,
                        KokkosFFT::Direction::backward);
  else
    seed_row_fft_wisdom(transforms_, node_spectrum_, node_field_,
                        KokkosFFT::Direction::backward);
  seed_row_fft_wisdom(transforms_, node_polarisation_, node_source_,
                      KokkosFFT::Direction::forward);
#endif
  if (real_signal_)
    signal_plan_ = std::make_unique<SignalPlan>(
        transforms_, half_spectrum_, real_field_, KokkosFFT::Direction::backward, axis);
  else
    inverse_plan_ = std::make_unique<ComplexPlan>(
        transforms_, node_spectrum_, node_field_, KokkosFFT::Direction::backward, axis);
  forward_plan_ =
      std::make_unique<RealPlan>(transforms_, node_polarisation_, node_source_,
                                 KokkosFFT::Direction::forward, axis);
  const auto tail = nodes_ % chunk_;
  if (tail != 0) {
    const auto& tail_transforms =
        fft_pool_.get(CpuOperation::fft, tail * components_ * n_to_);
    const auto rows = std::make_pair(std::size_t{0}, tail * components_);
    tail_spectrum_ = Kokkos::subview(node_spectrum_, rows, Kokkos::ALL());
    tail_field_ = Kokkos::subview(node_field_, rows, Kokkos::ALL());
    tail_source_ = Kokkos::subview(node_source_, rows, Kokkos::ALL());
    tail_polarisation_ = Kokkos::subview(node_polarisation_, rows, Kokkos::ALL());
    if (real_signal_) {
      tail_half_spectrum_ =
          AliasComplexView2(node_spectrum_.data(), tail * components_, n_to_ / 2 + 1);
      tail_real_field_ = AliasRealView2(reinterpret_cast<double*>(node_field_.data()),
                                        tail * components_, n_to_);
    }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    if (real_signal_)
      seed_row_fft_wisdom(tail_transforms, tail_half_spectrum_, tail_real_field_,
                          KokkosFFT::Direction::backward);
    else
      seed_row_fft_wisdom(tail_transforms, tail_spectrum_, tail_field_,
                          KokkosFFT::Direction::backward);
    seed_row_fft_wisdom(tail_transforms, tail_polarisation_, tail_source_,
                        KokkosFFT::Direction::forward);
#endif
    if (real_signal_)
      tail_signal_plan_ = std::make_unique<SignalPlan>(
          tail_transforms, tail_half_spectrum_, tail_real_field_,
          KokkosFFT::Direction::backward, axis);
    else
      tail_inverse_plan_ =
          std::make_unique<ComplexPlan>(tail_transforms, tail_spectrum_, tail_field_,
                                        KokkosFFT::Direction::backward, axis);
    tail_forward_plan_ =
        std::make_unique<RealPlan>(tail_transforms, tail_polarisation_, tail_source_,
                                   KokkosFFT::Direction::forward, axis);
  }
  if (dispersive_)
    dispersive_->plan(fft_pool_);
  if (terms_ != 0) {
    drive_plan_ = std::make_unique<DelayedForwardPlan>(
        delayed_transforms_, node_drive_, node_drive_spectrum_,
        KokkosFFT::Direction::forward, axis);
    history_plan_ = std::make_unique<DelayedInversePlan>(
        delayed_transforms_, node_product_, node_histories_,
        KokkosFFT::Direction::backward, axis);
    if (cpu_size_policy_enabled && tail != 0) {
      const auto rows = std::make_pair(std::size_t{0}, tail * general_pair_histories);
      tail_drive_ = Kokkos::subview(node_drive_, rows, Kokkos::ALL());
      tail_histories_ = Kokkos::subview(node_histories_, rows, Kokkos::ALL());
      tail_drive_spectrum_ = Kokkos::subview(node_drive_spectrum_, rows, Kokkos::ALL());
      tail_product_ = Kokkos::subview(node_product_, rows, Kokkos::ALL());
      const auto& tail_delayed =
          raman_tail_pool_.get(CpuOperation::raman, tail_drive_.size());
      tail_drive_plan_ = std::make_unique<DelayedForwardPlan>(
          tail_delayed, tail_drive_, tail_drive_spectrum_,
          KokkosFFT::Direction::forward, axis);
      tail_history_plan_ = std::make_unique<DelayedInversePlan>(
          tail_delayed, tail_product_, tail_histories_, KokkosFFT::Direction::backward,
          axis);
    }
  }
}

std::string GeneralModalSource::delayed_fft_provider() const {
  return drive_plan_ ? drive_plan_->provider() : "none";
}

void GeneralModalSource::evaluate() {
  if (closed_)
    throw std::logic_error("the general modal source is closed");
  if (!inverse_plan_ && !signal_plan_)
    throw std::logic_error("the general modal source has not been planned");
  Kokkos::deep_copy(kernels_, source_, Complex{});
  std::size_t chunk = 0;
  for (std::size_t first = 0; first < nodes_; first += chunk_, ++chunk) {
    const auto count = std::min(chunk_, nodes_ - first);
    synthesise(first, count);
    // SEP 6's dispersive channels filter the synthesised spectrum first, in synthesis.
    const bool live = dispersive_ && dispersive_->live(chunk);
    if (live)
      dispersive_->prepare(chunk);
    if (real_signal_ && regional_) {
      if (count < chunk_)
        KokkosFFT::execute(*tail_signal_plan_, tail_half_spectrum_, tail_real_field_,
                           KokkosFFT::Normalization::none);
      else
        KokkosFFT::execute(*signal_plan_, half_spectrum_, real_field_,
                           KokkosFFT::Normalization::none);
    } else if (regional_ || (live && dispersive_->reads_bare())) {
      if (count < chunk_)
        KokkosFFT::execute(*tail_inverse_plan_, tail_spectrum_, tail_field_,
                           KokkosFFT::Normalization::none);
      else
        KokkosFFT::execute(*inverse_plan_, node_spectrum_, node_field_,
                           KokkosFFT::Normalization::none);
    }
    if (regional_) {
      respond(first, count);
      if (terms_ != 0)
        delayed(first, count);
    }
    if (live)
      dispersive_->respond(chunk, first, count);
    if (regional_) {
      if (count < chunk_)
        KokkosFFT::execute(*tail_forward_plan_, tail_polarisation_, tail_source_,
                           KokkosFFT::Normalization::none);
      else
        KokkosFFT::execute(*forward_plan_, node_polarisation_, node_source_,
                           KokkosFFT::Normalization::none);
    } else if (!live)
      Kokkos::deep_copy(kernels_, node_source_, Complex{});
    // The products' spectra join the regional polarisation's, or are the chunk's.
    if (live)
      dispersive_->combine(!regional_);
    project(first, count);
  }
  finish();
}

void GeneralModalSource::set_weights(const std::vector<double>& weights) {
  if (closed_)
    throw std::logic_error("the general modal source is closed");
  if (!moving()) {
    if (!weights.empty())
      throw std::invalid_argument("a stage carries transform_weights for general modal "
                                  "maps that do not move");
    return;
  }
  weights_ready_ = false;
  require_size(weights.size(), n_stages_ * map_terms_, "transform_weights");
  if (!finite(weights))
    throw std::invalid_argument("transform_weights must be finite");
  // The contract's weights are zero outside the section a stage lies in (solprop
  // kernel schema 14), so the combination sums that section's terms alone; a zero
  // weight's product adds an exact zero, so skipping the others moves no bit.
  for (std::size_t stage = 0; stage < n_stages_; ++stage) {
    const auto* row = weights.data() + stage * map_terms_;
    std::size_t section = map_terms_;
    for (std::size_t term = 0; term < map_terms_; ++term) {
      if (row[term] == 0.0)
        continue;
      const auto owner = term / map_n_deg_;
      if (section != map_terms_ && owner != section)
        throw std::invalid_argument("a stage's transform_weights span more than one "
                                    "section of the general modal maps");
      section = owner;
    }
    stage_sections_[stage] = section == map_terms_ ? 0 : section;
  }
  auto* data = host_map_weights_.data();
  for (std::size_t index = 0; index < weights.size(); ++index)
    data[index] = weights[index];
  Kokkos::deep_copy(kernels_, map_weights_, host_map_weights_);
  weights_ready_ = true;
}

void GeneralModalSource::combine(const std::size_t stage) {
  if (!moving())
    return;
  if (closed_)
    throw std::logic_error("the general modal source is closed");
  if (!weights_ready_)
    throw std::logic_error("a moving guide's maps need the trial's transform_weights");
  if (stage >= n_stages_)
    throw std::invalid_argument("the general modal maps have no such stage");
  combine_maps(stage);
}

void GeneralModalSource::combine_maps(const std::size_t stage) {
  // One team per row of both maps -- a (node, component, mode) row of S and a (mode,
  // node, component) row of Q, the same count -- whose vector lanes run over the
  // row's contiguous bins; each bin is the weighted sum of the section's terms in
  // term order from zero, solprop's `GeneralMaps.at`. One output row per vector loop
  // and Views captured by value, as the source's other kernels.
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const UnmanagedView<const Complex*> synthesis_terms = synthesis_terms_;
  const UnmanagedView<const Complex*> projection_terms = projection_terms_;
  const UnmanagedView<Complex*> synthesis = synthesis_;
  const UnmanagedView<Complex*> projection = projection_;
  const UnmanagedView<const double**> weights = map_weights_;
  const auto n_omega = n_omega_;
  const auto one_map = synthesis_.size();
  const auto n_deg = map_n_deg_;
  const auto first = stage_sections_[stage] * n_deg;
  const auto rows = one_map / n_omega;
  const auto policy =
      TeamPolicy(maps_, static_cast<int>(rows), 1, general_team_lanes());
  launch::parallel_for(
      "solprop_native.general.combine_maps", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto base = static_cast<std::size_t>(team.league_rank()) * n_omega;
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, static_cast<int>(n_omega)),
            [=](const int lane) {
              const auto element = base + static_cast<std::size_t>(lane);
              double real = 0.0;
              double imag = 0.0;
              for (std::size_t term = 0; term < n_deg; ++term) {
                const double weight = weights(stage, first + term);
                const auto& value = synthesis_terms((first + term) * one_map + element);
                real += weight * value.real();
                imag += weight * value.imag();
              }
              synthesis(element) = Complex(real, imag);
            });
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, static_cast<int>(n_omega)),
                             [=](const int lane) {
                               const auto element =
                                   base + static_cast<std::size_t>(lane);
                               double real = 0.0;
                               double imag = 0.0;
                               for (std::size_t term = 0; term < n_deg; ++term) {
                                 const double weight = weights(stage, first + term);
                                 const auto& value = projection_terms(
                                     (first + term) * one_map + element);
                                 real += weight * value.real();
                                 imag += weight * value.imag();
                               }
                               projection(element) = Complex(real, imag);
                             });
      });
}

void GeneralModalSource::synthesise(const std::size_t first, const std::size_t count) {
  // Every fine bin outside the crop multiplies a zero amplitude, and a partial
  // chunk's unused rows transform to zero rather than to the previous chunk. One
  // team per row zeroes it and then accumulates its crop mode by mode, in the
  // reference's order (a zero, then each mode's product), while the row is in cache;
  // each vector loop runs over contiguous bins, and the barrier orders the zeroing
  // before the sums where lanes are threads. The real signal's one-sided rows take
  // each product halved where the doubling doubled it (exact) and no imaginary part
  // at the zero and Nyquist bins, which its real transform does not read; its
  // transform overwrites the rows, so they are zeroed every time as well.
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const UnmanagedView<const Complex*> synthesis = synthesis_;
  const UnmanagedView<const Complex**> scaled = scaled_;
  const bool real_signal = real_signal_;
  const UnmanagedView<Complex**> spectrum =
      real_signal ? UnmanagedView<Complex**>(half_spectrum_)
                  : UnmanagedView<Complex**>(node_spectrum_);
  const auto components = components_;
  const auto modes = n_modes_;
  const auto n_omega = n_omega_;
  const auto n_to = n_to_;
  const auto rows = count * components;
  const auto crop = crop_offset_;
  const auto samples = static_cast<int>(spectrum.extent(1));
  const auto policy = TeamPolicy(
      dense_for(count), static_cast<int>(chunk_ * components), 1, general_team_lanes());
  launch::parallel_for(
      "solprop_native.general.synthesise", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto row = static_cast<std::size_t>(team.league_rank());
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, samples),
                             [=](const int bin) { spectrum(row, bin) = Complex{}; });
        if (row >= rows)
          return;
        team.team_barrier();
        const auto base = ((first * components + row) * modes) * n_omega;
        for (std::size_t mode = 0; mode < modes; ++mode) {
          if (real_signal)
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, static_cast<int>(n_omega)),
                [=](const int lane) {
                  const auto omega = static_cast<std::size_t>(lane);
                  const auto map = base + mode * n_omega + omega;
                  const double map_real = synthesis(map).real();
                  const double map_imag = synthesis(map).imag();
                  const double state_real = scaled(mode, omega).real();
                  const double state_imag = scaled(mode, omega).imag();
                  const double product_real =
                      map_real * state_real - map_imag * state_imag;
                  const double product_imag =
                      map_real * state_imag + map_imag * state_real;
                  const auto fine = crop + omega;
                  const bool edge = fine == 0 || 2 * fine == n_to;
                  const double scale = edge ? 1.0 : 0.5;
                  auto& bin = spectrum(row, fine);
                  bin = Complex(bin.real() + scale * product_real,
                                edge ? 0.0 : bin.imag() + scale * product_imag);
                });
          else
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, static_cast<int>(n_omega)),
                [=](const int lane) {
                  const auto omega = static_cast<std::size_t>(lane);
                  // Kokkos::complex's product and sum, written on the parts.
                  const auto map = base + mode * n_omega + omega;
                  const double map_real = synthesis(map).real();
                  const double map_imag = synthesis(map).imag();
                  const double state_real = scaled(mode, omega).real();
                  const double state_imag = scaled(mode, omega).imag();
                  const double product_real =
                      map_real * state_real - map_imag * state_imag;
                  const double product_imag =
                      map_real * state_imag + map_imag * state_real;
                  auto& bin = spectrum(row, crop + omega);
                  bin = Complex(bin.real() + product_real, bin.imag() + product_imag);
                });
        }
      });
}

void GeneralModalSource::respond(const std::size_t first, const std::size_t count) {
  // A compile-time component count unrolls the contraction completely.
  if (real_signal_) {
    if (components_ == 3)
      respond_components<3, true>(first, count);
    else if (components_ == 2)
      respond_components<2, true>(first, count);
    else
      respond_components<1, true>(first, count);
  } else if (components_ == 3)
    respond_components<3, false>(first, count);
  else if (components_ == 2)
    respond_components<2, false>(first, count);
  else
    respond_components<1, false>(first, count);
}

template <std::size_t Components, bool RealSignal>
void GeneralModalSource::respond_components(const std::size_t first,
                                            const std::size_t count) {
  // One team per node, with independent samples in vector lanes. The real
  // contraction shares each field load and cubic monomial across output components;
  // each component still accumulates the monomials in their original order.
  //
  // The real cubic is symmetric in `(j, k, l)`, so it runs as solprop's symmetric
  // contraction (`symmetric_cubic`): the node's coefficients `C_im`, the tensor summed
  // over each monomial's orderings in lexicographic order from zero, are formed once
  // per team, and each sample sums `C_im (E_j E_k) E_l` over the ten monomials in their
  // order -- 3 + 10 products a row in place of 9 + 81 + 3.
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const UnmanagedView<const Complex**> field = node_field_;
  const UnmanagedView<const double**> signal = real_field_;
  const UnmanagedView<double**> polarisation = node_polarisation_;
  const UnmanagedView<double**> drive = node_drive_;
  const auto tensor = tensor_;
  constexpr auto components = Components;
  const auto n_to = n_to_;
  const auto samples = static_cast<int>(n_to_);
  const bool real_form = real_form_;
  const bool delayed = terms_ != 0;
  const double epsilon0 = epsilon0_;
  const double inverse = 1.0 / static_cast<double>(n_to_);
  constexpr auto stride = fourth_power(Components);
  constexpr auto monomials = symmetric_monomials(Components);
  const auto policy =
      TeamPolicy(response_for(count), static_cast<int>(count), 1, general_team_lanes());
  launch::parallel_for(
      "solprop_native.general.respond", policy, KOKKOS_LAMBDA(const TeamMember& team) {
        const auto node = static_cast<std::size_t>(team.league_rank());
        const auto row = node * components;
        Kokkos::Array<double, stride> entries{};
        for (std::size_t entry = 0; entry < stride; ++entry)
          entries[entry] = tensor((first + node) * stride + entry);
        Kokkos::Array<double, components * monomials> coefficients{};
        if (real_form)
          for (std::size_t j = 0; j < components; ++j)
            for (std::size_t k = 0; k < components; ++k)
              for (std::size_t l = 0; l < components; ++l) {
                const auto position = symmetric_monomial(components, j, k, l);
                for (std::size_t i = 0; i < components; ++i)
                  coefficients[i * monomials + position] +=
                      entries[((i * components + j) * components + k) * components + l];
              }
        if (real_form) {
          // Each monomial is shared by the Cartesian output components. Every
          // output retains the same ordered sum over the symmetric monomials.
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, samples), [=](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                double local[components]{};
                double values[components]{};
                SOLPROP_NATIVE_UNROLL
                for (std::size_t component = 0; component < components; ++component)
                  if constexpr (RealSignal)
                    local[component] = signal(row + component, time) * inverse;
                  else
                    local[component] = field(row + component, time).real() * inverse;
                std::size_t position = 0;
                SOLPROP_NATIVE_UNROLL
                for (std::size_t j = 0; j < components; ++j) {
                  SOLPROP_NATIVE_UNROLL
                  for (std::size_t k = j; k < components; ++k) {
                    const double pair = local[j] * local[k];
                    SOLPROP_NATIVE_UNROLL
                    for (std::size_t l = k; l < components; ++l, ++position) {
                      const double monomial = pair * local[l];
                      SOLPROP_NATIVE_UNROLL
                      for (std::size_t i = 0; i < components; ++i)
                        values[i] += coefficients[i * monomials + position] * monomial;
                    }
                  }
                }
                SOLPROP_NATIVE_UNROLL
                for (std::size_t i = 0; i < components; ++i)
                  polarisation(row + i, time) = epsilon0 * values[i];
              });
        } else
          for (std::size_t i = 0; i < components; ++i) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, samples), [=](const int sample) {
                  const auto time = static_cast<std::size_t>(sample);
                  Complex local[max_components]{};
                  for (std::size_t component = 0; component < components; ++component)
                    local[component] = field(row + component, time) * inverse;
                  Complex pair[max_components][max_components]{};
                  for (std::size_t k = 0; k < components; ++k)
                    for (std::size_t l = 0; l < components; ++l)
                      pair[k][l] = local[k] * Kokkos::conj(local[l]);
                  Complex value{};
                  for (std::size_t j = 0; j < components; ++j) {
                    Complex weighted{};
                    for (std::size_t k = 0; k < components; ++k)
                      for (std::size_t l = 0; l < components; ++l)
                        weighted += entries[((i * components + j) * components + k) *
                                                components +
                                            l] *
                                    pair[k][l];
                    value += weighted * local[j];
                  }
                  polarisation(row + i, time) = (epsilon0 * value).real();
                });
          }
        if (!delayed)
          return;
        // The delayed response's pair drive: the real field's `E_k E_l`, or the
        // analytic signal's `Re(E_k conj(E_l)) / 2`, of the same pair product.
        for (std::size_t entry = 0; entry < general_pair_histories; ++entry) {
          const auto history = node * general_pair_histories + entry;
          const auto left_row = row + pair_row(entry);
          const auto right_row = row + pair_column(entry);
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, samples), [=](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                if constexpr (RealSignal) {
                  const double left = signal(left_row, time) * inverse;
                  const double right = signal(right_row, time) * inverse;
                  drive(history, time) = left * right;
                } else if (real_form) {
                  const double left = field(left_row, time).real() * inverse;
                  const double right = field(right_row, time).real() * inverse;
                  drive(history, time) = left * right;
                } else {
                  const Complex left = field(left_row, time) * inverse;
                  const Complex right = field(right_row, time) * inverse;
                  drive(history, time) = 0.5 * (left * Kokkos::conj(right)).real();
                }
                drive(history, time + n_to) = 0.0;
              });
        }
      });
}

void GeneralModalSource::delayed(const std::size_t first, const std::size_t count) {
  if (!tail_drive_plan_ || count == chunk_)
    execute_raman(*drive_plan_, node_drive_, node_drive_spectrum_,
                  KokkosFFT::Normalization::none);
  else
    execute_raman(*tail_drive_plan_, tail_drive_, tail_drive_spectrum_,
                  KokkosFFT::Normalization::none);
  Kokkos::deep_copy(response_for(count), node_delayed_, 0.0);
  const auto drive_spectrum = node_drive_spectrum_;
  const auto product_view = node_product_;
  const auto histories = node_histories_;
  const auto accumulated = node_delayed_;
  const auto field = node_field_;
  const auto signal = real_field_;
  const bool real_signal = real_signal_;
  const auto response = node_response_;
  const auto spectra = delayed_spectrum_;
  const auto tensors = delayed_tensor_;
  const auto terms = terms_;
  const auto n_to = n_to_;
  const auto bins = n_to_ + 1;
  const double inverse = 1.0 / static_cast<double>(n_to_);
  const double inverse_doubled = 1.0 / static_cast<double>(2 * n_to_);
  constexpr auto components = max_components;
  constexpr auto stride = components * components * components * components;
  for (std::size_t term = 0; term < terms_; ++term) {
    launch::parallel_for(
        "solprop_native.general.delayed_product",
        Range2(response_for(count), {0, 0}, {count * general_pair_histories, bins}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t bin) {
          const auto owner = response(first + row / general_pair_histories);
          product_view(row, bin) =
              owner == 0 ? Complex{}
                         : spectra(((owner - 1) * terms + term) * bins + bin) *
                               drive_spectrum(row, bin);
        });
    if (!tail_history_plan_ || count == chunk_)
      execute_raman(*history_plan_, node_product_, node_histories_,
                    KokkosFFT::Normalization::none);
    else
      execute_raman(*tail_history_plan_, tail_product_, tail_histories_,
                    KokkosFFT::Normalization::none);
    launch::parallel_for(
        "solprop_native.general.delayed_contract",
        Range2(response_for(count), {0, 0}, {count, n_to}),
        KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
          const auto owner = response(first + node);
          if (owner == 0)
            return;
          double history[components][components]{};
          for (std::size_t entry = 0; entry < general_pair_histories; ++entry) {
            const double value =
                histories(node * general_pair_histories + entry, time) *
                inverse_doubled;
            history[pair_row(entry)][pair_column(entry)] = value;
            history[pair_column(entry)][pair_row(entry)] = value;
          }
          double local[components]{};
          for (std::size_t component = 0; component < components; ++component)
            local[component] =
                (real_signal ? signal(node * components + component, time)
                             : field(node * components + component, time).real()) *
                inverse;
          const auto base = ((owner - 1) * terms + term) * stride;
          for (std::size_t i = 0; i < components; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < components; ++j)
              for (std::size_t k = 0; k < components; ++k)
                for (std::size_t l = 0; l < components; ++l) {
                  const double entry = tensors(
                      base + ((i * components + j) * components + k) * components + l);
                  sum += entry * history[k][l] * local[j];
                }
            accumulated(node * components + i, time) += sum;
          }
        });
  }
  const auto polarisation = node_polarisation_;
  const double epsilon0 = epsilon0_;
  launch::parallel_for(
      "solprop_native.general.delayed_add",
      Range2(response_for(count), {0, 0}, {count * components, n_to}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        polarisation(row, time) += epsilon0 * accumulated(row, time);
      });
}

void GeneralModalSource::project(const std::size_t first, const std::size_t count) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  project_device(first, count);
#else
  switch (projection_lanes_) {
  case 2048:
    project_lanes<2048>(first, count);
    break;
  case 1024:
    project_lanes<1024>(first, count);
    break;
  case 512:
    project_lanes<512>(first, count);
    break;
  case 256:
    project_lanes<256>(first, count);
    break;
  case 128:
    project_lanes<128>(first, count);
    break;
  case 64:
    project_lanes<64>(first, count);
    break;
  case 32:
    project_lanes<32>(first, count);
    break;
  case 16:
    project_lanes<16>(first, count);
    break;
  default:
    project_lanes<general_vector_lanes>(first, count);
  }
#endif
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
void GeneralModalSource::project_device(const std::size_t first,
                                        const std::size_t count) {
  // The host kernel's one sequential accumulator per bin, which keeps the CPU build
  // thread-invariant, leaves a device with a few hundred lanes each walking thousands
  // of rows: 86 % of a graded-index step's device time on one GPU measured. Here a
  // team owns one mode's block of adjacent bins, its vector
  // lanes the bins (coalesced over the map and the spectrum), and its threads split the
  // rows, each summing every `threads`-th row in order; the threads' partial sums are
  // then added in thread order from team scratch. The order is fixed by the launch
  // shape, so a run repeats bitwise. It is not the host's order: the sum's rounding is
  // the gamma over the rows that solprop's parity bound carries for any order.
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  using Scratch = Kokkos::View<double**, ExecutionSpace::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const UnmanagedView<const Complex*> projection = projection_;
  const UnmanagedView<const Complex**> spectrum = node_source_;
  const auto source = source_;
  const auto components = components_;
  const auto nodes = nodes_;
  const auto n_omega = n_omega_;
  const auto crop = crop_offset_;
  const auto rows = count * components;
  const auto lanes = static_cast<std::size_t>(TeamPolicy::vector_length_max());
  const auto threads = std::size_t{1024} / lanes;
  const auto blocks = (n_omega + lanes - 1) / lanes;
  const auto policy =
      TeamPolicy(dense_for(count), static_cast<int>(n_modes_ * blocks),
                 static_cast<int>(threads), static_cast<int>(lanes))
          .set_scratch_size(0,
                            Kokkos::PerTeam(Scratch::shmem_size(threads, 2 * lanes)));
  launch::parallel_for(
      "solprop_native.general.project_device", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto block = static_cast<std::size_t>(team.league_rank());
        const auto mode = block / blocks;
        const auto start = (block % blocks) * lanes;
        const auto width = static_cast<int>(Kokkos::min(lanes, n_omega - start));
        const auto thread = static_cast<std::size_t>(team.team_rank());
        const Scratch partial(team.team_scratch(0), threads, 2 * lanes);
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, width), [&](const int lane) {
              const auto bin = start + static_cast<std::size_t>(lane);
              double real = 0.0;
              double imag = 0.0;
              for (std::size_t row = thread; row < rows; row += threads) {
                const Complex map = projection(
                    ((mode * nodes + first) * components + row) * n_omega + bin);
                const Complex value = spectrum(row, crop + bin);
                real += map.real() * value.real() - map.imag() * value.imag();
                imag += map.real() * value.imag() + map.imag() * value.real();
              }
              partial(thread, 2 * static_cast<std::size_t>(lane)) = real;
              partial(thread, 2 * static_cast<std::size_t>(lane) + 1) = imag;
            });
        team.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, width), [&](const int lane) {
          const auto column = 2 * static_cast<std::size_t>(lane);
          double real = 0.0;
          double imag = 0.0;
          for (std::size_t other = 0; other < threads; ++other) {
            real += partial(other, column);
            imag += partial(other, column + 1);
          }
          Kokkos::single(Kokkos::PerThread(team), [&] {
            source(mode, start + static_cast<std::size_t>(lane)) += Complex(real, imag);
          });
        });
      });
}
#endif

template <std::size_t Lanes>
void GeneralModalSource::project_lanes(const std::size_t first,
                                       const std::size_t count) {
  // Each output bin's sum runs over the chunk's rows in the reference's order, one
  // sequential accumulator per bin, so the only parallelism that keeps that order is
  // across the output bins. A team owns `Lanes` adjacent bins of one mode and walks
  // the rows once, updating the bins' accumulators as its vector lanes: the lanes are
  // independent sums over contiguous bins of the map and the spectrum, so they
  // vectorise (and coalesce on a device), where one bin per iteration was a serial
  // chain of dependent additions on a strided walk over the map, in a policy whose
  // single tile left the threads idle. The accumulators are split into
  // parts and Kokkos::complex's product is written on them, so no aggregate load
  // stops the vectoriser and no rounding moves. Bin `bin` is only ever touched by
  // the lane the vector range gives it, so the arrays are correct whether the lanes
  // share them (the host) or each holds its own copy (a device).
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const UnmanagedView<const Complex*> projection = projection_;
  const UnmanagedView<const Complex**> spectrum = node_source_;
  const auto source = source_;
  const auto components = components_;
  const auto nodes = nodes_;
  const auto n_omega = n_omega_;
  const auto crop = crop_offset_;
  const auto blocks = (n_omega + Lanes - 1) / Lanes;
  const auto policy =
      TeamPolicy(dense_for(count), static_cast<int>(n_modes_ * blocks), 1,
                 Kokkos::min(static_cast<int>(Lanes), TeamPolicy::vector_length_max()));
  launch::parallel_for(
      "solprop_native.general.project", policy, KOKKOS_LAMBDA(const TeamMember& team) {
        const auto block = static_cast<std::size_t>(team.league_rank());
        const auto mode = block / blocks;
        const auto start = (block % blocks) * Lanes;
        const auto width = static_cast<int>(Kokkos::min(Lanes, n_omega - start));
        Kokkos::Array<double, Lanes> real{};
        Kokkos::Array<double, Lanes> imag{};
        for (std::size_t row = 0; row < count * components; ++row) {
          const auto map =
              ((mode * nodes + first) * components + row) * n_omega + start;
          const auto fine = crop + start;
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(team, width),
              [=, &real, &imag](const int lane) {
                const auto bin = static_cast<std::size_t>(lane);
                const double map_real = projection(map + bin).real();
                const double map_imag = projection(map + bin).imag();
                const double spectrum_real = spectrum(row, fine + bin).real();
                const double spectrum_imag = spectrum(row, fine + bin).imag();
                const double product_real =
                    map_real * spectrum_real - map_imag * spectrum_imag;
                const double product_imag =
                    map_real * spectrum_imag + map_imag * spectrum_real;
                real[bin] += product_real;
                imag[bin] += product_imag;
              });
        }
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange(team, width), [&](const int lane) {
              const auto bin = static_cast<std::size_t>(lane);
              source(mode, start + bin) += Complex(real[bin], imag[bin]);
            });
      });
}

void GeneralModalSource::finish() {
  const auto source = source_;
  const auto factor = frequency_factor_;
  launch::parallel_for(
      "solprop_native.general.frequency_factor",
      Range2(kernels_, {0, 0}, {n_modes_, n_omega_}),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t omega) {
        source(mode, omega) = source(mode, omega) * factor(omega);
      });
}

std::map<std::string, std::size_t> GeneralModalSource::empty_ledger() {
  return {
      {"general_synthesis", 0},
      {"general_projection", 0},
      {"general_stage_maps", 0},
      {"general_map_weights", 0},
      {"general_tensor", 0},
      {"general_frequency_factor", 0},
      {"general_scaled_state", 0},
      {"general_source", 0},
      {"general_node_spectrum", 0},
      {"general_node_field", 0},
      {"general_node_polarisation", 0},
      {"general_node_source", 0},
      {"general_node_drive", 0},
      {"general_node_drive_spectrum", 0},
      {"general_node_product", 0},
      {"general_node_histories", 0},
      {"general_node_delayed", 0},
      {"general_delayed_tensor", 0},
      {"general_delayed_spectrum", 0},
      {"general_node_response", 0},
  };
}

std::map<std::string, std::size_t> GeneralModalSource::memory_ledger() const {
  constexpr auto complex_bytes = sizeof(Complex);
  constexpr auto real_bytes = sizeof(double);
  constexpr auto index_bytes = sizeof(std::size_t);
  return {
      // A moving guide's rows are every coefficient of each map, the two maps a
      // stage evaluates and the trial's weights, as solprop's reference counts them.
      {"general_synthesis",
       (moving() ? synthesis_terms_.size() : synthesis_.size()) * complex_bytes},
      {"general_projection",
       (moving() ? projection_terms_.size() : projection_.size()) * complex_bytes},
      {"general_stage_maps",
       moving() ? (synthesis_.size() + projection_.size()) * complex_bytes : 0},
      {"general_map_weights", map_weights_.size() * real_bytes},
      {"general_tensor", tensor_.size() * real_bytes},
      {"general_frequency_factor", frequency_factor_.size() * complex_bytes},
      {"general_scaled_state", scaled_.size() * complex_bytes},
      {"general_source", source_.size() * complex_bytes},
      {"general_node_spectrum", node_spectrum_.size() * complex_bytes},
      {"general_node_field", node_field_.size() * complex_bytes},
      {"general_node_polarisation", node_polarisation_.size() * real_bytes},
      {"general_node_source", node_source_.size() * complex_bytes},
      {"general_node_drive", node_drive_.size() * real_bytes},
      {"general_node_drive_spectrum", node_drive_spectrum_.size() * complex_bytes},
      {"general_node_product", node_product_.size() * complex_bytes},
      {"general_node_histories", node_histories_.size() * real_bytes},
      {"general_node_delayed", node_delayed_.size() * real_bytes},
      {"general_delayed_tensor", delayed_tensor_.size() * real_bytes},
      {"general_delayed_spectrum", delayed_spectrum_.size() * complex_bytes},
      {"general_node_response", node_response_.size() * index_bytes},
  };
}

void GeneralModalSource::close() noexcept {
  if (closed_)
    return;
  kernels_.fence();
  if (dispersive_)
    dispersive_->close();
  dispersive_.reset();
  tail_inverse_plan_.reset();
  tail_signal_plan_.reset();
  tail_forward_plan_.reset();
  tail_spectrum_ = {};
  tail_field_ = {};
  tail_source_ = {};
  tail_polarisation_ = {};
  inverse_plan_.reset();
  signal_plan_.reset();
  forward_plan_.reset();
  tail_drive_plan_.reset();
  tail_history_plan_.reset();
  tail_drive_ = {};
  tail_histories_ = {};
  tail_drive_spectrum_ = {};
  tail_product_ = {};
  drive_plan_.reset();
  history_plan_.reset();
  synthesis_ = {};
  projection_ = {};
  synthesis_terms_ = {};
  projection_terms_ = {};
  map_weights_ = {};
  host_map_weights_ = {};
  weights_ready_ = false;
  frequency_factor_ = {};
  delayed_spectrum_ = {};
  tensor_ = {};
  delayed_tensor_ = {};
  node_response_ = {};
  scaled_ = {};
  source_ = {};
  half_spectrum_ = {};
  tail_half_spectrum_ = {};
  real_field_ = {};
  tail_real_field_ = {};
  node_spectrum_ = {};
  node_field_ = {};
  node_source_ = {};
  node_polarisation_ = {};
  node_drive_ = {};
  node_histories_ = {};
  node_delayed_ = {};
  node_drive_spectrum_ = {};
  node_product_ = {};
  kernels_ = ExecutionSpace();
  transforms_ = ExecutionSpace();
  delayed_transforms_ = ExecutionSpace();
  response_ = ExecutionSpace();
  tail_response_ = ExecutionSpace();
  maps_ = ExecutionSpace();
  dense_ = ExecutionSpace();
  tail_dense_ = ExecutionSpace();
  kernel_pool_.clear();
  fft_pool_.clear();
  raman_pool_.clear();
  raman_tail_pool_.clear();
  closed_ = true;
}

} // namespace solprop_native
