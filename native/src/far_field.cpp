// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/far_field.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace solprop_native {

namespace {

// The most rows a request may carry, solprop's `MAX_APERTURES` and `MAX_REGIONS`.
constexpr std::size_t max_rows = 16;

RealView sized(const char* label, const std::size_t count) {
  return count == 0 ? RealView{} : RealView(label, count);
}

} // namespace

FarFieldLayout far_field_layout(const std::uint64_t flags,
                                const std::size_t n_apertures,
                                const std::size_t n_regions, const std::size_t n_pol) {
  FarFieldLayout layout;
  layout.collected = (flags & far_field_collected) != 0 ? 0 : -1;
  if ((flags & far_field_amplitude) != 0)
    layout.amplitude = layout.collected == 0 ? 1 : 0;
  layout.stride = (layout.collected == 0 ? 1 : 0) +
                  (layout.amplitude >= 0 ? 4 * static_cast<long>(n_pol) : 0);
  long slots = 0;
  const auto place = [&slots](long& first, const long count) {
    first = count != 0 ? slots : -1;
    slots += count;
  };
  place(layout.beam, (flags & far_field_beam) != 0 ? 1 : 0);
  place(layout.absorbed, (flags & far_field_absorbed) != 0
                             ? static_cast<long>(far_field_absorbed_channels)
                             : 0);
  place(layout.regions,
        (flags & far_field_regions) != 0 ? static_cast<long>(n_regions) : 0);
  place(layout.apertures, layout.stride * static_cast<long>(n_apertures));
  place(layout.axis, (flags & far_field_axis) != 0 ? 4 * static_cast<long>(n_pol) : 0);
  layout.n_slots = static_cast<std::size_t>(slots);
  return layout;
}

std::size_t far_field_band_count(const std::size_t n_x, const std::size_t n_y,
                                 const std::size_t n_slots, const std::size_t n_omega) {
  if (n_x == 0 || n_y == 0 || n_slots == 0 || n_omega == 0)
    throw std::invalid_argument("the far-field bands need positive extents");
  const std::size_t columns = n_x * n_y;
  const std::size_t unit = n_y > 1 ? n_x : far_field_line_block;
  const std::size_t units = (columns + unit - 1) / unit;
  const std::size_t admitted = far_field_band_budget / (8 * n_slots * n_omega);
  return std::max<std::size_t>(1, std::min(units, admitted));
}

FarFieldPass::FarFieldPass(const Problem& problem, const ExecutionSpace& execution)
    : execution_(execution), n_x_(problem.n_x), n_y_(problem.n_y),
      n_omega_(problem.n_omega), n_pol_(problem.n_pol_state), n_t_(problem.n_t),
      dt_(problem.dt), envelope_(problem.family == FieldFamily::envelope) {
  if (!problem.free_geometry() || problem.spatial_k_x.size() != n_x_ ||
      problem.spatial_k_y.size() != n_y_)
    throw std::invalid_argument(
        "the far-field pass needs a Cartesian or line grid and its column axes");
  for (const double value : problem.spatial_k_x)
    if (!std::isfinite(value))
      throw std::invalid_argument("spatial_k_x must be finite");
  for (const double value : problem.spatial_k_y)
    if (!std::isfinite(value))
      throw std::invalid_argument("spatial_k_y must be finite");
  k_x_ = RealView("solprop_native.far_field.k_x", n_x_);
  k_y_ = RealView("solprop_native.far_field.k_y", n_y_);
  const auto host_x = Kokkos::create_mirror_view(k_x_);
  const auto host_y = Kokkos::create_mirror_view(k_y_);
  for (std::size_t index = 0; index < n_x_; ++index)
    host_x(index) = problem.spatial_k_x[index];
  for (std::size_t index = 0; index < n_y_; ++index)
    host_y(index) = problem.spatial_k_y[index];
  Kokkos::deep_copy(execution_, k_x_, host_x);
  Kokkos::deep_copy(execution_, k_y_, host_y);
  if (problem.spatial_far_field_slots != 0) {
    const auto slots = problem.spatial_far_field_slots;
    const auto bands = far_field_band_count(n_x_, n_y_, slots, n_omega_);
    partials_ = RealView("solprop_native.far_field.partials", bands * slots * n_omega_);
  }
  execution_.fence("solprop_native far-field axes");
}

std::size_t FarFieldPass::partial_bytes() const noexcept {
  return partials_.size() * sizeof(double);
}

std::size_t FarFieldPass::table_bytes() const noexcept {
  return (k_x_.size() + k_y_.size() + tables_.size() + packed_.size()) * sizeof(double);
}

void FarFieldPass::close() noexcept {
  execution_.fence();
  k_x_ = {};
  k_y_ = {};
  partials_ = {};
  tables_ = {};
  packed_ = {};
  host_tables_ = {};
  host_packed_ = {};
}

std::vector<double> FarFieldPass::run(const std::uint64_t flags,
                                      const std::vector<double>& apertures,
                                      const int profile_code, const int exponent,
                                      const std::vector<double>& regions,
                                      const double intensity_factor,
                                      const ComplexView2& field, const Inputs& inputs) {
  if (flags == 0 || (flags & ~far_field_flags) != 0)
    throw std::invalid_argument("invalid far-field statistic flags");
  if (!std::isfinite(intensity_factor) || intensity_factor <= 0.0)
    throw std::invalid_argument("the far-field intensity factor must be positive");
  if (profile_code < 0 || profile_code > 3 || (exponent != 1 && exponent != 2))
    throw std::invalid_argument("invalid aperture profile code or exponent");
  const std::size_t row = far_field_aperture_constants + n_omega_;
  if (apertures.size() % row != 0 || regions.size() % 4 != 0)
    throw std::invalid_argument(
        "aperture rows must hold 6 + n_omega values and region rows 4");
  const std::size_t n_apertures = apertures.size() / row;
  const std::size_t n_regions = regions.size() / 4;
  if (n_apertures > max_rows || n_regions > max_rows)
    throw std::invalid_argument("at most 16 apertures and 16 regions");
  if ((flags & (far_field_collected | far_field_amplitude)) != 0 && n_apertures == 0)
    throw std::invalid_argument("the aperture statistics need at least one aperture");
  for (std::size_t index = 0; index < apertures.size(); ++index) {
    const double value = apertures[index];
    const bool scale = index % row >= far_field_aperture_constants;
    if (!std::isfinite(value) || (scale && value < 0.0))
      throw std::invalid_argument("aperture tables must be finite, scales nonnegative");
  }
  for (const double value : regions)
    if (std::isnan(value))
      throw std::invalid_argument("region edges must not be NaN");
  if (field.extent(0) != n_x_ * n_y_ * n_pol_ || field.extent(1) != n_omega_ ||
      inputs.weights.extent(0) != n_x_ * n_y_ ||
      inputs.on_axis.extent(0) != n_x_ * n_y_ || inputs.omega.extent(0) != n_omega_ ||
      inputs.active.extent(0) != n_omega_)
    throw std::invalid_argument(
        "the far-field state or its inputs have the wrong shape");
  const auto layout = far_field_layout(flags, n_apertures, n_regions, n_pol_);
  if (layout.n_slots == 0)
    return {};
  const auto n_bands = far_field_band_count(n_x_, n_y_, layout.n_slots, n_omega_);
  const auto needed = n_bands * layout.n_slots * n_omega_;
  // Spectral rows (beam, regions, collected, the apertures' axes, the axis), then
  // the four absorbed powers.
  std::size_t spectral =
      (layout.beam >= 0 ? 1U : 0U) + (layout.regions >= 0 ? n_regions : 0) +
      (layout.collected >= 0 ? n_apertures : 0) +
      (layout.amplitude >= 0 ? n_apertures : 0) + (layout.axis >= 0 ? 1U : 0U);
  const std::size_t packed_size =
      spectral * n_omega_ + (layout.absorbed >= 0 ? far_field_absorbed_channels : 0);
  const std::size_t table_size = apertures.size() + regions.size();
  if (partials_.size() < needed || tables_.size() < table_size ||
      packed_.size() < packed_size) {
    execution_.fence("solprop_native far-field resize");
    if (partials_.size() < needed)
      partials_ = RealView("solprop_native.far_field.partials", needed);
    if (tables_.size() < table_size) {
      tables_ = sized("solprop_native.far_field.tables", table_size);
      host_tables_ = Kokkos::create_mirror_view(tables_);
    }
    if (packed_.size() < packed_size) {
      packed_ = sized("solprop_native.far_field.packed", packed_size);
      host_packed_ = Kokkos::create_mirror_view(packed_);
    }
  }
  bool shared = n_apertures > 0;
  for (std::size_t index = 0; index < apertures.size(); ++index)
    host_tables_(index) = apertures[index];
  for (std::size_t index = 0; index < regions.size(); ++index)
    host_tables_(apertures.size() + index) = regions[index];
  // The image's distance from the origin is evaluated once per element when every
  // aperture shares its distance and scale row, and the radius too when they also
  // share a centre: the same strict expression on the same inputs, so every hard-edge
  // decision is the general path's exactly.
  bool one_distance = shared;
  for (std::size_t index = 1; index < n_apertures; ++index) {
    for (std::size_t column = far_field_aperture_constants - 1; column < row; ++column)
      one_distance =
          one_distance && apertures[index * row + column] == apertures[column];
    for (const std::size_t constant : {std::size_t{0}, std::size_t{1}})
      shared = shared && apertures[index * row + constant] == apertures[constant];
  }
  shared = shared && one_distance;
  if (table_size != 0)
    Kokkos::deep_copy(execution_, tables_, host_tables_);
  accumulate(layout, n_bands, n_apertures, n_regions, profile_code, exponent,
             Sharing{shared, one_distance}, field, inputs);
  finish(layout, n_bands, n_apertures, n_regions, intensity_factor, packed_size);
  Kokkos::deep_copy(execution_, host_packed_, packed_);
  execution_.fence("solprop_native far-field statistics boundary");
  return {host_packed_.data(), host_packed_.data() + packed_size};
}

void FarFieldPass::accumulate(const FarFieldLayout& layout, const std::size_t n_bands,
                              const std::size_t n_apertures,
                              const std::size_t n_regions, const int profile_code,
                              const int exponent, const Sharing sharing,
                              const ComplexView2& field, const Inputs& inputs) {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  const auto partials = partials_;
  const auto tables = tables_;
  const auto k_x = k_x_;
  const auto k_y = k_y_;
  const auto n_x = n_x_;
  const auto n_y = n_y_;
  const auto n = n_omega_;
  const auto n_pol = n_pol_;
  const auto slots = layout.n_slots;
  const auto row = far_field_aperture_constants + n;
  const auto region_table = n_apertures * row;
  const auto weights = inputs.weights;
  const auto on_axis = inputs.on_axis;
  const auto square = inputs.square;
  const auto square_low = inputs.square_low;
  const auto transverse = inputs.transverse;
  const auto transverse_low = inputs.transverse_low;
  const auto spectral_rate = inputs.spectral_rate;
  const auto transverse_rate = inputs.transverse_rate;
  const auto omega = inputs.omega;
  const auto active = inputs.active;
  const double evanescent = inputs.evanescent;
  const std::size_t n_axes = layout.axis >= 0 ? n_pol : 0;
  const auto regions_used = layout.regions >= 0 ? n_regions : 0;
  // Apertures declared but no aperture statistic asked for (a periodic beam spectrum,
  // say) cost nothing: no weight is evaluated.
  const auto apertures_used = layout.apertures >= 0 ? n_apertures : 0;
  const bool shared = sharing.centre;
  const bool one_distance = sharing.distance;
  // A host team is one thread, which then walks its band's columns with frequency
  // innermost, as solprop's numba_cpu pass does; a device team's threads share the
  // band's frequencies. Either way one thread owns each (band, bin).
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const Policy policy(execution_, static_cast<int>(n_bands), Kokkos::AUTO());
#else
  const Policy policy(execution_, static_cast<int>(n_bands), 1);
#endif
  Kokkos::parallel_for(
      "solprop_native.far_field.bands", policy, KOKKOS_LAMBDA(const Member& team) {
        const auto band = static_cast<std::size_t>(team.league_rank());
        const auto base = band * slots * n;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n),
                             [&](const std::size_t bin) {
                               for (std::size_t slot = 0; slot < slots; ++slot)
                                 partials(base + slot * n + bin) = 0.0;
                             });
        const auto first = far_field_band_edge(band, n_bands, n_x, n_y);
        const auto last = far_field_band_edge(band + 1, n_bands, n_x, n_y);
        for (std::size_t column = first; column < last; ++column) {
          const double kx = k_x(column % n_x);
          const double ky = k_y(column / n_x);
          const double weight = weights(column);
          const Complex alpha = on_axis(column);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, n), [&](const std::size_t bin) {
                const auto at = [&](const long slot) -> double& {
                  return partials(base + static_cast<std::size_t>(slot) * n + bin);
                };
                double power = 0.0;
                for (std::size_t pol = 0; pol < n_pol; ++pol) {
                  const Complex value = field(column * n_pol + pol, bin);
                  power += value.real() * value.real() + value.imag() * value.imag();
                }
                const Complex longitudinal = (square(bin) - transverse(column)) +
                                             (square_low(bin) - transverse_low(column));
                const Complex root = Kokkos::sqrt(longitudinal);
                const bool propagating = longitudinal.real() > 0.0;
                const double attenuation =
                    propagating ? root.imag()
                                : Kokkos::min(Kokkos::abs(root.imag()), evanescent);
                const bool live = active(bin) != 0U;
                // The conjugate admittance the internal spectrum's closure takes
                // (solprop SEP 34), zero off the forward power ports.
                const Complex closing =
                    live && propagating && omega(bin) > 0.0
                        ? Complex(far_field_speed_of_light * root.real(),
                                  -(far_field_speed_of_light * root.imag())) /
                              omega(bin)
                        : Complex{};
                const double weighted = weight * power * closing.real();
                if (layout.beam >= 0)
                  at(layout.beam) += weighted;
                if (layout.absorbed >= 0) {
                  at(layout.absorbed) += weighted * spectral_rate(bin);
                  if (live) {
                    at(layout.absorbed + 1) += weighted * transverse_rate(column);
                    at(propagating ? layout.absorbed + 3 : layout.absorbed + 2) +=
                        weighted * (2.0 * attenuation);
                  }
                }
                for (std::size_t region = 0; region < regions_used; ++region) {
                  const auto edges = region_table + 4 * region;
                  if (far_field_member(kx, ky, tables(edges), tables(edges + 1),
                                       tables(edges + 2), tables(edges + 3)))
                    at(layout.regions + static_cast<long>(region)) += weighted;
                }
                for (std::size_t pol = 0; pol < n_axes; ++pol) {
                  const Complex value = field(column * n_pol + pol, bin);
                  const Complex product = alpha * value;
                  const Complex closed = alpha * (closing * value);
                  const long slot = layout.axis + 2 * static_cast<long>(pol);
                  const long again = slot + 2 * static_cast<long>(n_pol);
                  at(slot) += product.real();
                  at(slot + 1) += product.imag();
                  at(again) += closed.real();
                  at(again + 1) += closed.imag();
                }
                double radius = 0.0;
                double origin = 0.0;
                if (apertures_used > 0 && one_distance) {
                  const double reach = tables(far_field_aperture_constants + bin);
                  origin = far_field_radius(reach, kx, ky, 0.0, 0.0);
                  if (shared)
                    radius = far_field_radius(reach, kx, ky, tables(0), tables(1));
                }
                for (std::size_t index = 0; index < apertures_used; ++index) {
                  const auto constants = index * row;
                  const double reach =
                      tables(constants + far_field_aperture_constants + bin);
                  if (!shared)
                    radius = far_field_radius(reach, kx, ky, tables(constants),
                                              tables(constants + 1));
                  if (!one_distance)
                    origin = far_field_radius(reach, kx, ky, 0.0, 0.0);
                  const double transmission =
                      far_field_weight(radius, origin, reach, tables(constants + 2),
                                       tables(constants + 3), tables(constants + 4),
                                       tables(constants + 5), profile_code);
                  if (transmission == 0.0)
                    continue;
                  const long first_slot =
                      layout.apertures + static_cast<long>(index) * layout.stride;
                  if (layout.collected >= 0)
                    at(first_slot + layout.collected) +=
                        weighted *
                        (exponent == 1 ? transmission : transmission * transmission);
                  if (layout.amplitude >= 0)
                    for (std::size_t pol = 0; pol < n_pol; ++pol) {
                      const Complex through =
                          transmission * field(column * n_pol + pol, bin);
                      const Complex product = alpha * through;
                      const Complex closed = alpha * (closing * through);
                      const long slot =
                          first_slot + layout.amplitude + 2 * static_cast<long>(pol);
                      const long again = slot + 2 * static_cast<long>(n_pol);
                      at(slot) += product.real();
                      at(slot + 1) += product.imag();
                      at(again) += closed.real();
                      at(again + 1) += closed.imag();
                    }
                }
              });
          // Each thread owns the same bins in every column's range; the barrier
          // keeps that true whatever the range's distribution.
          team.team_barrier();
        }
      });
}

void FarFieldPass::finish(const FarFieldLayout& layout, const std::size_t n_bands,
                          const std::size_t n_apertures, const std::size_t n_regions,
                          const double intensity_factor,
                          const std::size_t packed_size) {
  const auto partials = partials_;
  const auto packed = packed_;
  const auto n = n_omega_;
  const auto n_pol = n_pol_;
  const auto slots = layout.n_slots;
  const bool envelope = envelope_;
  // fourier.spectral_density's prefactor: 2 dt^2 / pi on a field grid and
  // dt^2 / (2 pi) on an envelope grid, on every bin.
  const double density = envelope ? dt_ * dt_ / (2.0 * std::numbers::pi)
                                  : 2.0 * (dt_ * dt_) / std::numbers::pi;
  const double energy = dt_ / static_cast<double>(n_t_);
  const std::size_t leading = static_cast<std::size_t>(layout.beam >= 0);
  const std::size_t absorbed = layout.absorbed >= 0 ? far_field_absorbed_channels : 0;
  const std::size_t spectral = (packed_size - absorbed) / n;
  const std::size_t regions = layout.regions >= 0 ? n_regions : 0;
  const std::size_t collected = layout.collected >= 0 ? n_apertures : 0;
  const auto band_sum = KOKKOS_LAMBDA(const std::size_t slot, const std::size_t bin) {
    double total = 0.0;
    for (std::size_t band = 0; band < n_bands; ++band)
      total += partials((band * slots + slot) * n + bin);
    return total;
  };
  if (spectral != 0)
    Kokkos::parallel_for(
        "solprop_native.far_field.finish", Range2(execution_, {0, 0}, {spectral, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t bin) {
          // Rows in packed order: beam, regions, collected, the apertures' axes, the
          // axis; the absorbed powers sit after the beam.
          std::size_t rest = row;
          long slot = -1;
          bool amplitude = false;
          if (rest < leading) {
            slot = layout.beam;
          } else if ((rest -= leading) < regions) {
            slot = layout.regions + static_cast<long>(rest);
          } else if ((rest -= regions) < collected) {
            slot = layout.apertures + static_cast<long>(rest) * layout.stride +
                   layout.collected;
          } else if ((rest -= collected) < (layout.amplitude >= 0 ? n_apertures : 0)) {
            slot = layout.apertures + static_cast<long>(rest) * layout.stride +
                   layout.amplitude;
            amplitude = true;
          } else {
            slot = layout.axis;
            amplitude = true;
          }
          const auto first = static_cast<std::size_t>(slot);
          double value = 0.0;
          if (amplitude) {
            // Re(A conj(G)), the amplitude's same-frequency pairing with its closure.
            for (std::size_t pol = 0; pol < n_pol; ++pol) {
              const double real = band_sum(first + 2 * pol, bin);
              const double imag = band_sum(first + 2 * pol + 1, bin);
              const double closed_real = band_sum(first + 2 * (n_pol + pol), bin);
              const double closed_imag = band_sum(first + 2 * (n_pol + pol) + 1, bin);
              value += density * (real * closed_real + imag * closed_imag);
            }
          } else {
            value = density * band_sum(first, bin);
          }
          const auto position = row * n + bin + (row >= leading ? absorbed : 0);
          packed(position) = intensity_factor * value;
        });
  if (absorbed != 0) {
    const auto first = static_cast<std::size_t>(layout.absorbed);
    const auto offset = leading * n;
    Kokkos::parallel_for(
        "solprop_native.far_field.absorbed",
        Kokkos::RangePolicy<ExecutionSpace>(execution_, 0, absorbed),
        KOKKOS_LAMBDA(const std::size_t channel) {
          // fourier.energy_weights: [1, 4, ..., 4, 1] dt / n_t on a field grid.
          double sum = 0.0;
          for (std::size_t bin = 0; bin < n; ++bin) {
            const double weight =
                envelope || bin == 0 || bin + 1 == n ? energy : 4.0 * energy;
            sum += band_sum(first + channel, bin) * weight;
          }
          packed(offset + channel) = intensity_factor * sum;
        });
  }
}

} // namespace solprop_native
