// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The general workspace's statistics and, in statistics_monitor.inc (included at the
// end), its boundary monitor's readings, each formed on the state where it lives.
// Functions: statistics, a plane's packed record over the analytic time rows
// (fill_statistics_time, form_statistics_power; pack_energy, statistics_window_energy,
// statistics_intensity, statistics_plasma); whitened_statistics, a general modal
// basis's; the magnetic closure (require_flux_closure, fill_statistics_closure,
// solprop SEP 34); the separable geometries' time statistics and fluence
// (separable_time_statistics, separable_fluence, plane_fluence_moments,
// plane_fluence_trace); the on-axis spectrum and populations (axis_spectrum,
// statistics_modal_time, axis_populations, solprop SEP 19); spatial_statistics, the
// transverse plane's; and aperture_statistics, the far-field pass's (solprop SEP 23).

#include "general_workspace.hpp"
#include "memory.hpp"

namespace solprop_native {

void GeneralWorkspace::fill_statistics_time(const bool windowed, const double lower,
                                            const double upper, const bool weighted) {
  if (omega_.empty())
    throw std::logic_error("native statistics need the coarse frequency axis");
  graph::deep_copy(execution_space_, statistics_spectrum_, Complex{});
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto omega = omega_;
  const auto power_weight = statistics_power_weight_;
  const auto n = problem_.dims.n_omega;
  const auto n_t = problem_.dims.n_t;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  launch::parallel_for(
      "solprop_native.statistics.prepare_analytic",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (windowed && !(omega(index) >= lower && omega(index) < upper))
          return;
        const double analytic_scale =
            envelope || index == 0 || index + 1 == n ? 1.0 : 2.0;
        spectrum(column, index) =
            analytic_scale / static_cast<double>(n_t) * current(column, index);
        if (weighted)
          spectrum(column, index) *= power_weight(column, index);
      });
  graph::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                 KokkosFFT::Normalization::none);
}

void GeneralWorkspace::fill_statistics_pair(const bool windowed, const double lower,
                                            const double upper) {
  if (!statistics_unit_weight_) {
    fill_statistics_time(windowed, lower, upper, true);
    graph::deep_copy(execution_space_, statistics_weighted_time_, statistics_time_);
  }
  fill_statistics_time(windowed, lower, upper);
}

void GeneralWorkspace::form_statistics_power() {
  const auto time_field = statistics_time_;
  const auto weighted_field = statistics_weighted_time_;
  const bool unit_weight = statistics_unit_weight_;
  const auto power = statistics_power_;
  const auto n_pol = problem_.dims.n_pol_state;
  launch::parallel_for(
      "solprop_native.statistics.power",
      Range2(execution_space_, {0, 0}, {problem_.dims.n_modes, problem_.dims.n_t}),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t time) {
        double sum = 0.0;
        for (std::size_t pol = 0; pol < n_pol; ++pol)
          sum += unit_weight ? squared_abs(time_field(mode * n_pol + pol, time))
                             : (time_field(mode * n_pol + pol, time) *
                                Kokkos::conj(weighted_field(mode * n_pol + pol, time)))
                                   .real();
        power(mode, time) = sum;
      });
}

void GeneralWorkspace::form_statistics_total_power() {
  const auto power = statistics_power_;
  const auto total = statistics_total_power_;
  const auto modes = problem_.dims.n_modes;
  launch::parallel_for(
      "solprop_native.statistics.total_power",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_t),
      KOKKOS_LAMBDA(const std::size_t time) {
        double sum = 0.0;
        for (std::size_t mode = 0; mode < modes; ++mode)
          sum += power(mode, time);
        total(time) = sum;
      });
}

namespace {

// The offset of a statistic not asked for.
constexpr auto absent = std::numeric_limits<std::size_t>::max();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
using StatisticsTeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
using StatisticsTeamMember = StatisticsTeamPolicy::member_type;
#endif

} // namespace

std::vector<double> GeneralWorkspace::statistics(
    const std::uint64_t flags, const std::vector<double>& windows,
    const std::vector<double>& on_axis, const double density, const double taper,
    const double preionisation, const double intensity_factor,
    const std::vector<double>& channel_preionisation,
    const std::vector<double>& power_weight) {
  const auto& selected_weight =
      power_weight.empty() ? problem_.grid.power_weight : power_weight;
  statistics_unit_weight_ =
      std::ranges::all_of(selected_weight, [](double value) { return value == 1.0; });
  if (!selected_weight.empty()) {
    require_extent(selected_weight.size(), statistics_power_weight_.size(),
                   "statistics power weight");
    auto host_weight = Kokkos::create_mirror_view(statistics_power_weight_);
    for (std::size_t column = 0; column < problem_.n_columns(); ++column)
      for (std::size_t index = 0; index < problem_.dims.n_omega; ++index) {
        const double value = selected_weight[column * problem_.dims.n_omega + index];
        if (!(value > 0.0) || !std::isfinite(value))
          throw std::invalid_argument(
              "statistics power weight must be positive and finite");
        host_weight(column, index) = value;
      }
    graph::deep_copy(execution_space_, statistics_power_weight_, host_weight);
  } else {
    graph::deep_copy(execution_space_, statistics_power_weight_, 1.0);
  }
  const auto power_weights = statistics_power_weight_;
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (windows.size() % 2 != 0)
    throw std::invalid_argument("statistics windows must contain lower/upper pairs");
  const auto requested = [flags](const StatisticsFlag flag) {
    return (flags & static_cast<std::uint64_t>(flag)) != 0U;
  };
  // The temporal statistics reconstruct whole columns; on free geometry their
  // buffers hold one node chunk, and the reference refuses these names on any
  // spatial problem, so the adapter never asks. Refuse here as well rather than
  // write past the chunk.
  // The modal power weight has no rows on a spatial problem, radial included, and
  // the energy kernels below read it for every column.
  if ((problem_.separable_geometry() || statistics_power_weight_.extent(0) == 0) &&
      (requested(statistics_omega0) || requested(statistics_energy) ||
       requested(statistics_energy_window) || requested(statistics_peak_power) ||
       requested(statistics_peak_power_window) ||
       requested(statistics_peak_power_total) || requested(statistics_fwhm)))
    throw std::invalid_argument(
        "temporal statistics are not spatial saved-field statistics");
  const auto n_windows = windows.size() / 2;
  std::size_t packed_size = 0;
  const auto reserve = [&packed_size, requested](const StatisticsFlag flag,
                                                 const std::size_t count) {
    if (!requested(flag))
      return absent;
    const auto offset = packed_size;
    packed_size = checked_sum(packed_size, count, "statistics packed output");
    return offset;
  };
  const auto omega0_offset = reserve(statistics_omega0, problem_.n_columns());
  const auto energy_offset = reserve(statistics_energy, problem_.n_columns());
  const auto energy_window_offset =
      reserve(statistics_energy_window, checked_product(n_windows, problem_.n_columns(),
                                                        "windowed-energy output"));
  const auto peak_offset = reserve(statistics_peak_power, problem_.dims.n_modes);
  const auto peak_window_offset = reserve(
      statistics_peak_power_window,
      checked_product(n_windows, problem_.dims.n_modes, "windowed-peak output"));
  const auto total_peak_offset = reserve(statistics_peak_power_total, 1);
  const auto density_offset = reserve(statistics_electron_density, 1);
  const auto rate_offset = reserve(statistics_peak_ionisation_rate, 1);
  const auto intensity_offset = reserve(statistics_peak_intensity, 1);
  const auto quadrature_offset = reserve(statistics_quadrature_error, 1);
  const auto fwhm_offset = reserve(statistics_fwhm, problem_.dims.n_t);
  const auto populations_offset = reserve(
      statistics_populations,
      checked_product(2,
                      checked_product(problem_.plasma.n_channels, problem_.dims.n_nodes,
                                      "population output"),
                      "population output"));
  // solprop SEP 18's run diagnostics, packed last: six maxima over every node.
  const auto diagnostics_offset =
      reserve(statistics_plasma_integration, plasma_diagnostic_parts);
  const bool diagnose = diagnostics_offset != absent;
  if (populations_offset != absent || (diagnose && problem_.plasma.n_channels > 1)) {
    require_extent(channel_preionisation.size(), problem_.plasma.n_channels,
                   "statistics channel preionisation");
    for (double value : channel_preionisation)
      if (!std::isfinite(value) || value < 0.0 || value > 1.0)
        throw std::invalid_argument(
            "statistics channel preionisation must be in [0, 1]");
    auto initial = Kokkos::create_mirror_view(stage_channel_preionisation_);
    for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
      initial(0, channel) = channel_preionisation[channel];
    graph::deep_copy(execution_space_, stage_channel_preionisation_, initial);
  }

  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  const auto packed = statistics_packed_;
  if (packed_size != 0)
    graph::deep_copy(execution_space_, packed,
                     Kokkos::Experimental::quiet_NaN_v<double>);

  pack_energy(packed, power_weights, omega0_offset, energy_offset);
  statistics_window_energy(packed, power_weights, n_windows, energy_window_offset,
                           windows);
  const bool needs_base_power = peak_offset != absent || total_peak_offset != absent ||
                                intensity_offset != absent || fwhm_offset != absent;
  if (needs_base_power) {
    fill_statistics_pair(false, 0.0, 0.0);
    form_statistics_power();
  }
  if (peak_offset != absent) {
    const auto power = statistics_power_;
    const auto n_t = problem_.dims.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_,
                                      static_cast<int>(problem_.dims.n_modes),
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    launch::parallel_for(
        "solprop_native.statistics.device_pack_peak_power", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          const auto mode = static_cast<std::size_t>(team.league_rank());
          double peak = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
              [&](const int time_value, double& update) {
                const auto time = static_cast<std::size_t>(time_value);
                if (power(mode, time) > update)
                  update = power(mode, time);
              },
              Kokkos::Max<double>(peak));
          Kokkos::single(Kokkos::PerTeam(team),
                         [&]() { packed(peak_offset + mode) = peak; });
        });
#else
    launch::parallel_for(
        "solprop_native.statistics.pack_peak_power",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          double peak = 0.0;
          for (std::size_t time = 0; time < n_t; ++time)
            if (power(mode, time) > peak)
              peak = power(mode, time);
          packed(peak_offset + mode) = peak;
        });
#endif
  }
  if (total_peak_offset != absent || fwhm_offset != absent) {
    form_statistics_total_power();
    if (total_peak_offset != absent) {
      const auto total = statistics_total_power_;
      const auto n_t = problem_.dims.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, 1,
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      launch::parallel_for(
          "solprop_native.statistics.device_pack_peak_power_total", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            double peak = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
                [&](const int time_value, double& update) {
                  const auto time = static_cast<std::size_t>(time_value);
                  if (total(time) > update)
                    update = total(time);
                },
                Kokkos::Max<double>(peak));
            Kokkos::single(Kokkos::PerTeam(team),
                           [&]() { packed(total_peak_offset) = peak; });
          });
#else
      launch::parallel_for(
          "solprop_native.statistics.pack_peak_power_total",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
          KOKKOS_LAMBDA(const std::size_t) {
            double peak = 0.0;
            for (std::size_t time = 0; time < n_t; ++time)
              if (total(time) > peak)
                peak = total(time);
            packed(total_peak_offset) = peak;
          });
#endif
    }
    if (fwhm_offset != absent) {
      const auto destination = Kokkos::subview(
          packed, std::make_pair(fwhm_offset, fwhm_offset + problem_.dims.n_t));
      graph::deep_copy(execution_space_, destination, statistics_total_power_);
    }
  }
  statistics_intensity(packed, intensity_offset, on_axis, intensity_factor);
  if (peak_window_offset != absent) {
    for (std::size_t window = 0; window < n_windows; ++window) {
      fill_statistics_pair(true, windows[2 * window], windows[2 * window + 1]);
      form_statistics_power();
      const auto power = statistics_power_;
      const auto n_t = problem_.dims.n_t;
      const auto modes = problem_.dims.n_modes;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, static_cast<int>(modes),
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      launch::parallel_for(
          "solprop_native.statistics.device_pack_peak_power_window", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            const auto mode = static_cast<std::size_t>(team.league_rank());
            double peak = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
                [&](const int time_value, double& update) {
                  const auto time = static_cast<std::size_t>(time_value);
                  if (power(mode, time) > update)
                    update = power(mode, time);
                },
                Kokkos::Max<double>(peak));
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              packed(peak_window_offset + window * modes + mode) = peak;
            });
          });
#else
      launch::parallel_for(
          "solprop_native.statistics.pack_peak_power_window",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, modes),
          KOKKOS_LAMBDA(const std::size_t mode) {
            double peak = 0.0;
            for (std::size_t time = 0; time < n_t; ++time)
              if (power(mode, time) > peak)
                peak = power(mode, time);
            packed(peak_window_offset + window * modes + mode) = peak;
          });
#endif
    }
  }

  statistics_plasma(packed, density_offset, rate_offset, populations_offset,
                    diagnostics_offset, diagnose, density, taper, preionisation);
  if (quadrature_offset != absent) {
    const double value = quadrature_error();
    launch::parallel_for(
        "solprop_native.statistics.pack_quadrature",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { packed(quadrature_offset) = value; });
  }

  if (packed_size == 0)
    return {};
  const auto active =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(active);
  graph::deep_copy(execution_space_, host, active);
  execution_space_.fence("solprop_native statistics packed boundary");
  std::vector<double> result(packed_size);
  for (std::size_t index = 0; index < packed_size; ++index)
    result[index] = host(index);
  return result;
}

// The columns' centre frequency and energy (`omega0`, `energy`).
void GeneralWorkspace::pack_energy(const RealView packed, const RealView2 power_weights,
                                   const std::size_t omega0_offset,
                                   const std::size_t energy_offset) {
  if (omega0_offset != absent || energy_offset != absent) {
    if (omega_.empty() || !(problem_.grid.dt > 0.0))
      throw std::logic_error("native statistics need the coarse frequency axis and dt");
    const auto current = current_;
    const auto omega = omega_;
    const auto n = problem_.dims.n_omega;
    const auto n_t = problem_.dims.n_t;
    const auto dt = problem_.grid.dt;
    const bool envelope = problem_.dims.family == FieldFamily::envelope;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_,
                                      static_cast<int>(problem_.n_columns()),
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    launch::parallel_for(
        "solprop_native.statistics.device_pack_spectral", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          const auto column = static_cast<std::size_t>(team.league_rank());
          double density_sum = 0.0;
          double weighted_sum = 0.0;
          double energy_sum = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                update += (squared_abs(current(column, index)) *
                           power_weights(column, index));
              },
              density_sum);
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                update += omega(index) * (squared_abs(current(column, index)) *
                                          power_weights(column, index));
              },
              weighted_sum);
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n)),
              [&](const int index_value, double& update) {
                const auto index = static_cast<std::size_t>(index_value);
                const double weight =
                    envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
                update += weight * (squared_abs(current(column, index)) *
                                    power_weights(column, index));
              },
              energy_sum);
          Kokkos::single(Kokkos::PerTeam(team), [&]() {
            if (omega0_offset != absent)
              packed(omega0_offset + column) =
                  density_sum > 0.0 ? weighted_sum / density_sum
                                    : Kokkos::Experimental::quiet_NaN_v<double>;
            if (energy_offset != absent)
              packed(energy_offset + column) =
                  dt / static_cast<double>(n_t) * energy_sum;
          });
        });
#else
    launch::parallel_for(
        "solprop_native.statistics.pack_spectral",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
        KOKKOS_LAMBDA(const std::size_t column) {
          double density_sum = 0.0;
          double weighted_sum = 0.0;
          double energy_sum = 0.0;
          for (std::size_t index = 0; index < n; ++index) {
            const double power =
                (squared_abs(current(column, index)) * power_weights(column, index));
            density_sum += power;
            weighted_sum += omega(index) * power;
            const double weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
            energy_sum += weight * power;
          }
          if (omega0_offset != absent)
            packed(omega0_offset + column) =
                density_sum > 0.0 ? weighted_sum / density_sum
                                  : Kokkos::Experimental::quiet_NaN_v<double>;
          if (energy_offset != absent)
            packed(energy_offset + column) = dt / static_cast<double>(n_t) * energy_sum;
        });
#endif
  }
}

// Each column's energy within each time window (`energy_window`).
void GeneralWorkspace::statistics_window_energy(const RealView packed,
                                                const RealView2 power_weights,
                                                const std::size_t n_windows,
                                                const std::size_t energy_window_offset,
                                                const std::vector<double>& windows) {
  if (energy_window_offset != absent) {
    if (omega_.empty() || !(problem_.grid.dt > 0.0))
      throw std::logic_error("native statistics need the coarse frequency axis and dt");
    const auto current = current_;
    const auto omega = omega_;
    const auto n = problem_.dims.n_omega;
    const auto n_t = problem_.dims.n_t;
    const auto dt = problem_.grid.dt;
    const auto columns = problem_.n_columns();
    const bool envelope = problem_.dims.family == FieldFamily::envelope;
    for (std::size_t window = 0; window < n_windows; ++window) {
      const double lower = windows[2 * window];
      const double upper = windows[2 * window + 1];
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
      const StatisticsTeamPolicy policy(execution_space_, static_cast<int>(columns),
                                        SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
      launch::parallel_for(
          "solprop_native.statistics.device_pack_energy_window", policy,
          KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
            const auto column = static_cast<std::size_t>(team.league_rank());
            double sum = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(n)),
                [&](const int index_value, double& update) {
                  const auto index = static_cast<std::size_t>(index_value);
                  if (!(omega(index) >= lower && omega(index) < upper))
                    return;
                  const double weight =
                      envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
                  update += weight * (squared_abs(current(column, index)) *
                                      power_weights(column, index));
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              packed(energy_window_offset + window * columns + column) =
                  dt / static_cast<double>(n_t) * sum;
            });
          });
#else
      launch::parallel_for(
          "solprop_native.statistics.pack_energy_window",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns),
          KOKKOS_LAMBDA(const std::size_t column) {
            double sum = 0.0;
            for (std::size_t index = 0; index < n; ++index) {
              if (!(omega(index) >= lower && omega(index) < upper))
                continue;
              const double weight =
                  envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
              sum += weight * (squared_abs(current(column, index)) *
                               power_weights(column, index));
            }
            packed(energy_window_offset + window * columns + column) =
                dt / static_cast<double>(n_t) * sum;
          });
#endif
    }
  }
}

// The peak on-axis intensity (`peak_intensity`).
void GeneralWorkspace::statistics_intensity(const RealView packed,
                                            const std::size_t intensity_offset,
                                            const std::vector<double>& on_axis,
                                            const double intensity_factor) {
  if (intensity_offset != absent) {
    if (on_axis.empty() || on_axis.size() % problem_.dims.n_modes != 0 ||
        on_axis.size() / problem_.dims.n_modes > 2)
      throw std::invalid_argument(
          "on-axis synthesis must have one or two rows and n_modes columns");
    graph::deep_copy(execution_space_, statistics_on_axis_, 0.0);
    const auto destination = Kokkos::subview(
        statistics_on_axis_, std::make_pair<std::size_t>(0, on_axis.size()));
    copy_vector(execution_space_, destination, on_axis);
    const auto matrix = statistics_on_axis_;
    const auto field = statistics_time_;
    const auto weighted_field = statistics_weighted_time_;
    const bool unit_weight = statistics_unit_weight_;
    const auto rows = on_axis.size() / problem_.dims.n_modes;
    const auto modes = problem_.dims.n_modes;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto n_t = problem_.dims.n_t;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    const StatisticsTeamPolicy policy(execution_space_, 1,
                                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
    launch::parallel_for(
        "solprop_native.statistics.device_pack_peak_intensity", policy,
        KOKKOS_LAMBDA(const StatisticsTeamMember& team) {
          double peak = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, static_cast<int>(n_t)),
              [&](const int time_value, double& update) {
                const auto time = static_cast<std::size_t>(time_value);
                double power = 0.0;
                for (std::size_t row = 0; row < rows; ++row) {
                  for (std::size_t pol = 0; pol < n_pol; ++pol) {
                    Complex value{}, weighted_value{};
                    for (std::size_t mode = 0; mode < modes; ++mode) {
                      value +=
                          matrix(row * modes + mode) * field(mode * n_pol + pol, time);
                      if (!unit_weight)
                        weighted_value += matrix(row * modes + mode) *
                                          weighted_field(mode * n_pol + pol, time);
                    }
                    power += unit_weight
                                 ? squared_abs(value)
                                 : (value * Kokkos::conj(weighted_value)).real();
                  }
                }
                if (power > update)
                  update = power;
              },
              Kokkos::Max<double>(peak));
          Kokkos::single(Kokkos::PerTeam(team),
                         [&]() { packed(intensity_offset) = intensity_factor * peak; });
        });
#else
    launch::parallel_for(
        "solprop_native.statistics.pack_peak_intensity",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double peak = 0.0;
          for (std::size_t time = 0; time < n_t; ++time) {
            double power = 0.0;
            for (std::size_t row = 0; row < rows; ++row) {
              for (std::size_t pol = 0; pol < n_pol; ++pol) {
                Complex value{};
                Complex weighted_value{};
                for (std::size_t mode = 0; mode < modes; ++mode) {
                  value += matrix(row * modes + mode) * field(mode * n_pol + pol, time);
                  if (!unit_weight)
                    weighted_value += matrix(row * modes + mode) *
                                      weighted_field(mode * n_pol + pol, time);
                }
                power += unit_weight ? squared_abs(value)
                                     : (value * Kokkos::conj(weighted_value)).real();
              }
            }
            if (power > peak)
              peak = power;
          }
          packed(intensity_offset) = intensity_factor * peak;
        });
#endif
  }
}

// The plasma statistics: the electron density, the peak ionisation rate, the
// populations and the run diagnostics.
void GeneralWorkspace::statistics_plasma(
    const RealView packed, const std::size_t density_offset,
    const std::size_t rate_offset, const std::size_t populations_offset,
    const std::size_t diagnostics_offset, const bool diagnose, const double density,
    const double taper, const double preionisation) {
  if (density_offset != absent || rate_offset != absent ||
      populations_offset != absent || diagnose) {
    if (!problem_.plasma.enabled)
      throw std::logic_error("this problem declares no compiled plasma response");
    if (diagnose && problem_.plasma.envelope)
      throw std::invalid_argument("the plasma integration diagnostics read the "
                                  "field-grid plasma column, and this problem "
                                  "carries none");
    RealView2 source = stash_current_;
    // The diagnostics rescan the plane, and the plasma statistics then read the scan.
    const bool source_is_fraction = stash_is_current_ && !diagnose;
    if (!source_is_fraction) {
      source = plasma_statistics_scratch_;
      if (problem_.separable_geometry()) {
        separable_load_current();
        for (std::size_t unit = 0; unit < transverse_units();
             unit += transverse_units_per_chunk()) {
          const auto chunk = transverse_chunk(unit);
          separable_pack_chunk(chunk, true);
          separable_to_time(chunk);
          separable_nodes(chunk);
          scan_plasma_statistics(chunk.first_node, chunk.nodes, source, diagnose,
                                 preionisation);
        }
      } else {
        statistics_modal_time();
        graph::deep_copy(execution_space_, stage_taper_, taper);
        scale_transform(problem_.stepper.n_stages - 1, taper);
        for (std::size_t first = 0; first < problem_.dims.n_nodes;
             first += problem_.dims.node_chunk) {
          const auto count =
              std::min(problem_.dims.node_chunk, problem_.dims.n_nodes - first);
          synthesise(first, count);
          scan_plasma_statistics(first, count, source, diagnose, preionisation);
        }
      }
    }
    if (populations_offset != absent) {
      const bool envelope = problem_.plasma.envelope;
      const auto channels = problem_.plasma.n_channels;
      const auto initial = stage_channel_preionisation_;
      launch::parallel_for(
          "solprop_native.statistics.pack_populations",
          Range2(execution_space_, {0, 0}, {problem_.dims.n_nodes, channels}),
          KOKKOS_LAMBDA(const std::size_t node, const std::size_t channel) {
            const auto offset = populations_offset + 2 * (node * channels + channel);
            packed(offset) =
                source_is_fraction ? source(node, 2 * channel)
                : envelope
                    ? envelope_fraction(source(node, 2 * channel), initial(0, channel))
                    : ionised_fraction(source(node, 2 * channel), initial(0, channel));
            packed(offset + 1) = source(node, 2 * channel + 1);
          });
    }
    const auto nodes = problem_.dims.n_nodes;
    launch::parallel_for(
        "solprop_native.statistics.pack_plasma",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          std::size_t densest = 0;
          double largest_fraction = -1.0;
          for (std::size_t node = 0; node < nodes; ++node) {
            const double fraction =
                source_is_fraction ? source(node, 0)
                                   : ionised_fraction(source(node, 0), preionisation);
            if (fraction > largest_fraction) {
              densest = node;
              largest_fraction = fraction;
            }
          }
          if (density_offset != absent)
            packed(density_offset) = density * largest_fraction;
          if (rate_offset != absent)
            packed(rate_offset) = source(densest, 1);
        });
    if (diagnose) {
      // Each part is a maximum, so the nodes combine in any order.
      const auto parts = plasma_diagnostics_;
      launch::parallel_for(
          "solprop_native.statistics.pack_plasma_integration",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                              plasma_diagnostic_parts),
          KOKKOS_LAMBDA(const std::size_t part) {
            double largest = 0.0;
            for (std::size_t node = 0; node < nodes; ++node)
              largest = Kokkos::max(largest, parts(node, part));
            packed(diagnostics_offset + part) = largest;
          });
    }
  }
}

std::vector<double> GeneralWorkspace::whitened_statistics(
    const std::vector<Complex>& root, const std::uint64_t flags,
    const std::vector<double>& windows, const double intensity_factor,
    const bool spectral) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  constexpr std::uint64_t admitted = statistics_omega0 | statistics_energy |
                                     statistics_peak_power_total | statistics_fwhm;
  if ((flags & ~admitted) != 0U)
    throw std::invalid_argument("a general modal basis's statistics are omega0, the "
                                "energy, the peak total power and the widths");
  const auto columns = problem_.n_columns();
  const auto n = problem_.dims.n_omega;
  const auto modes = problem_.dims.n_modes;
  const auto n_pol = problem_.dims.n_pol_state;
  const bool whiten = !root.empty();
  if (whiten) {
    require_extent(
        root.size(),
        checked_product(n, checked_product(modes, modes, "metric root"), "metric root"),
        "metric root");
    if (statistics_whitened_.extent(0) != columns ||
        statistics_whitened_.extent(1) != n)
      throw std::logic_error("whitened statistics are a general modal basis's");
    copy_vector(execution_space_, statistics_root_, root);
    const auto field = current_;
    const auto whitened = statistics_whitened_;
    const auto matrix = statistics_root_;
    // `c_mp(w) = sum_n R_mn(w) a_np(w)`, `numpy.einsum("kmn,npk->mpk", ...)`'s terms.
    launch::parallel_for(
        "solprop_native.statistics.whiten",
        Range2(execution_space_, {0, 0}, {columns, n}),
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          const auto mode = column / n_pol;
          const auto pol = column % n_pol;
          Complex sum{};
          for (std::size_t other = 0; other < modes; ++other)
            sum += matrix((index * modes + mode) * modes + other) *
                   field(other * n_pol + pol, index);
          whitened(column, index) = sum;
        });
  }
  const auto sums_size = columns + (spectral ? n : 0);
  if (statistics_whitened_sums_.size() < sums_size)
    throw std::logic_error("whitened statistics are a general modal basis's");
  std::vector<double> result;
  {
    // The statistics passes read `current_`: they read the whitened columns while
    // this scope holds them there, and the state's own afterwards, even on a throw.
    struct Swap {
      ComplexView2& first;
      ComplexView2& second;
      bool active;
      Swap(ComplexView2& a, ComplexView2& b, const bool on)
          : first(a), second(b), active(on) {
        if (active)
          std::swap(first, second);
      }
      ~Swap() {
        if (active)
          std::swap(first, second);
      }
      Swap(const Swap&) = delete;
      Swap& operator=(const Swap&) = delete;
      Swap(Swap&&) = delete;
      Swap& operator=(Swap&&) = delete;
    } swap(current_, statistics_whitened_, whiten);
    if (flags != 0U)
      result = statistics(flags, windows, {}, 0.0, 1.0, 0.0, intensity_factor, {}, {});
    const auto field = current_;
    const auto sums = statistics_whitened_sums_;
    launch::parallel_for(
        "solprop_native.statistics.whitened_column_sums",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns),
        KOKKOS_LAMBDA(const std::size_t column) {
          double sum = 0.0;
          for (std::size_t index = 0; index < n; ++index)
            sum += squared_abs(field(column, index));
          sums(column) = sum;
        });
    if (spectral)
      launch::parallel_for(
          "solprop_native.statistics.whitened_bin_sums",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
          KOKKOS_LAMBDA(const std::size_t index) {
            double sum = 0.0;
            for (std::size_t column = 0; column < columns; ++column)
              sum += squared_abs(field(column, index));
            sums(columns + index) = sum;
          });
  }
  const auto active = Kokkos::subview(statistics_whitened_sums_,
                                      std::make_pair<std::size_t>(0, sums_size));
  auto host = Kokkos::create_mirror_view(active);
  graph::deep_copy(execution_space_, host, active);
  execution_space_.fence("solprop_native whitened statistics boundary");
  result.reserve(result.size() + sums_size);
  for (std::size_t index = 0; index < sums_size; ++index)
    result.push_back(host(index));
  return result;
}

// solprop SEP 34: the buffers of the magnetic closure's synthesis, allocated at
// construction when the problem's own request has a local flux statistic and here,
// outside any trial step, when a later request first needs them.
void GeneralWorkspace::require_flux_closure() {
  const auto rows = statistics_time_.extent(0);
  const bool time = statistics_weighted_time_.extent(0) != rows;
  const bool magnitude = !problem_.separable_geometry() &&
                         statistics_magnitude_.extent(0) != statistics_power_.extent(0);
  const bool volume = problem_.separable_geometry() && flux_transverse_.empty();
  if (!time && !magnitude && !volume)
    return;
  execution_space_.fence("solprop_native flux closure allocation");
  if (time)
    statistics_weighted_time_ = ComplexView2("solprop_native.statistics.weighted_time",
                                             rows, problem_.dims.n_t);
  if (magnitude)
    statistics_magnitude_ =
        RealView2("solprop_native.statistics.flux_magnitude",
                  statistics_power_.extent(0), statistics_power_.extent(1));
  if (volume)
    flux_transverse_ =
        ComplexView3("solprop_native.flux.closure_transverse", transverse_.extent(0),
                     transverse_.extent(1), transverse_.extent(2));
}

// The magnetic closure of every column in time (a radial run's, solprop SEP 34): the
// analytic spectrum times the conjugate admittance, inverted into the statistics'
// time rows and kept in `statistics_weighted_time_`.
void GeneralWorkspace::fill_statistics_closure() {
  graph::deep_copy(execution_space_, statistics_spectrum_, Complex{});
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto omega = omega_;
  const auto active = active_mask_;
  const auto flux = flux_;
  const auto n = problem_.dims.n_omega;
  const auto n_t = problem_.dims.n_t;
  const auto n_pol = problem_.dims.n_pol_state;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  launch::parallel_for(
      "solprop_native.statistics.prepare_closure",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const double analytic_scale =
            envelope || index == 0 || index + 1 == n ? 1.0 : 2.0;
        const Complex closing =
            flux.closing(column / n_pol, index, active(index) != 0U, omega(index));
        spectrum(column, index) = analytic_scale / static_cast<double>(n_t) *
                                  (closing * current(column, index));
      });
  graph::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                 KokkosFFT::Normalization::none);
  graph::deep_copy(execution_space_, statistics_weighted_time_, statistics_time_);
}

void GeneralWorkspace::separable_time_statistics(const std::size_t axis_peak,
                                                 const std::size_t peak,
                                                 const bool fluence,
                                                 const double intensity_factor) {
  constexpr double lowest = std::numeric_limits<double>::lowest();
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto field = statistics_time_;
  const auto closure = statistics_weighted_time_;
  const auto packed = statistics_packed_;
  const auto axis_weights = spatial_on_axis_;
  const auto scratch = transverse_;
  const auto closure_scratch = flux_transverse_;
  const auto power = statistics_power_;
  const auto partials = spatial_statistics_partials_;
  const auto synthesis = synthesis_;
  const auto flux = flux_;
  const auto active = active_mask_;
  const auto omega = omega_;
  const auto n_modes = problem_.dims.n_modes;
  const auto n_nodes = problem_.dims.n_nodes;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.dims.n_omega;
  const auto n_t = problem_.dims.n_t;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  if (axis_peak != absent) {
    // The axis's closure first, kept in the closure rows, then its field: each the
    // coherent on-axis sum of the analytic spectrum (solprop SEP 34). On free space the
    // far-field pass forms those sums in its team-local bands. A sum per bin over every
    // column, as the slab's few columns take here, is a thread's serial walk across a
    // whole state on a device. A plane's on-axis spectrum reads the same slots.
    if (far_field_)
      static_cast<void>(far_field_->run(far_field_axis, {}, 2, {}, intensity_factor,
                                        current, far_field_inputs()));
    for (const bool closed : {true, false}) {
      graph::deep_copy(execution_space_, spectrum, Complex{});
      if (far_field_)
        far_field_->axis_spectrum(closed, spectrum);
      else
        launch::parallel_for(
            "solprop_native.free.statistics_axis_input",
            Range2(execution_space_, {0, 0}, {n_pol, n}),
            KOKKOS_LAMBDA(const std::size_t pol, const std::size_t index) {
              const bool live = active(index) != 0U;
              Complex sum{};
              for (std::size_t mode = 0; mode < n_modes; ++mode) {
                const Complex value = current(mode * n_pol + pol, index);
                sum += axis_weights(mode) *
                       (closed ? flux.closing(mode, index, live, omega(index)) * value
                               : value);
              }
              const double scale =
                  (envelope || index == 0 || index + 1 == n ? 1.0 : 2.0) /
                  static_cast<double>(n_t);
              spectrum(pol, index) = scale * sum;
            });
      graph::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                     KokkosFFT::Normalization::none);
      if (closed)
        graph::deep_copy(
            execution_space_,
            Kokkos::subview(closure, std::make_pair<std::size_t>(0, n_pol),
                            Kokkos::ALL),
            Kokkos::subview(field, std::make_pair<std::size_t>(0, n_pol), Kokkos::ALL));
    }
    launch::parallel_for(
        "solprop_native.free.statistics_axis_reduce",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          double maximum = lowest;
          for (std::size_t time = 0; time < n_t; ++time) {
            double value = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol)
              value += (field(pol, time) * Kokkos::conj(closure(pol, time))).real();
            maximum = Kokkos::max(maximum, value);
          }
          packed(axis_peak) = intensity_factor * maximum;
        });
  }
  if (peak == absent && !fluence)
    return;
  const auto transverse_count =
      static_cast<double>(problem_.slab_geometry() ? problem_.spatial.n_y : n_modes);
  launch::parallel_for(
      "solprop_native.free.statistics_transverse_input",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const auto mode = column / n_pol;
        const Complex value = current(column, index) / transverse_count;
        const auto plane = index * n_pol + column % n_pol;
        layout.cell(scratch, plane, mode) = value;
        layout.cell(closure_scratch, plane, mode) =
            flux.closing(mode, index, active(index) != 0U, omega(index)) * value;
      });
  transverse_inverse();
  transverse_inverse(flux_transverse_);
  if (fluence)
    separable_fluence(intensity_factor);
  if (peak == absent)
    return;
  const bool slab = problem_.slab_geometry();
  const auto n_x = problem_.spatial.n_x;
  const auto n_p = problem_.spatial.n_p;
  // The planar synthesis carries the temporal pad scale; the statistics read the
  // coarse field, so it is divided out (solprop's `collect_spatial`).
  const double crop_scale =
      static_cast<double>(problem_.dims.n_t) / static_cast<double>(problem_.dims.n_to);
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_node;
    const auto first_row = chunk.first_row;
    const auto count = chunk.nodes;
    // The chunk's closure first, kept in the closure rows, then its field.
    for (const bool closed : {true, false}) {
      const auto source = closed ? closure_scratch : scratch;
      graph::deep_copy(execution_space_, spectrum, Complex{});
      launch::parallel_for(
          "solprop_native.free.statistics_temporal_input",
          Range2(execution_space_, {0, 0}, {count * n_pol, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
            const double scale =
                (envelope || index == 0 || index + 1 == n ? 1.0 : 2.0) /
                static_cast<double>(n_t);
            if (slab) {
              // Node ix of the chunk's y row iy: the slab synthesis of that row.
              const auto ix = row % n_x;
              const auto rows = first_row + (row / n_x) * n_p;
              Complex sum{};
              for (std::size_t channel = 0; channel < n_p; ++channel)
                sum +=
                    synthesis(ix, channel) * source(index, layout.first(rows + channel),
                                                    layout.second(rows + channel));
              spectrum(row, index) = scale * (crop_scale * sum);
            } else {
              const auto node = first + row / n_pol;
              spectrum(row, index) =
                  scale * layout.cell(source, index * n_pol + row % n_pol, node);
            }
          });
      graph::execute(*statistics_inverse_plan_, statistics_spectrum_, statistics_time_,
                     KokkosFFT::Normalization::none);
      if (closed)
        graph::deep_copy(
            execution_space_,
            Kokkos::subview(closure, std::make_pair<std::size_t>(0, count * n_pol),
                            Kokkos::ALL),
            Kokkos::subview(field, std::make_pair<std::size_t>(0, count * n_pol),
                            Kokkos::ALL));
    }
    launch::parallel_for(
        "solprop_native.free.statistics_intensity",
        Range2(execution_space_, {0, 0}, {count, n_t}),
        KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
          double value = 0.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol)
            value += (field(node * n_pol + pol, time) *
                      Kokkos::conj(closure(node * n_pol + pol, time)))
                         .real();
          power(node, time) = intensity_factor * value;
        });
    launch::parallel_for(
        "solprop_native.free.statistics_node_peak",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, count),
        KOKKOS_LAMBDA(const std::size_t node) {
          double maximum = lowest;
          for (std::size_t time = 0; time < n_t; ++time)
            maximum = Kokkos::max(maximum, power(node, time));
          partials(first + node, 1) = maximum;
        });
  }
  launch::parallel_reduce(
      "solprop_native.free.statistics_peak",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
      KOKKOS_LAMBDA(const std::size_t node, double& maximum) {
        maximum = Kokkos::max(maximum, partials(node, 1));
      },
      Kokkos::Max<double, MemorySpace>(Kokkos::subview(packed, peak)));
}

// Each node's signed fluence into `partials(node, 0)` and a bound on its magnitude into
// `partials(node, 2)`, from the physical spectra of the transverse volumes, with no
// time transform. The analytic signal's spectrum is `s_k a_k`, with `s_k` one or two
// over `n_t`, so Parseval's theorem over the `n_t` samples gives `sum_t Re(E(t)
// conj(G(t))) = (1 / n_t) sum_k w_k Re(a_k conj(b_k))`, with `w_k` one or four: exactly
// the time sum the reference takes. The same pairs bound that sum's magnitude: by
// Cauchy-Schwarz over time and polarisation, `sum_t |E(t)| |G(t)| <= (1 / n_t)
// sqrt(sum_k w_k |a_k|^2 sum_k w_k |b_k|^2)`, with equality where `G` is proportional
// to `E`, as in a beam whose admittance varies little across its band. So a rounding
// budget set from it is never below the reference's.
void GeneralWorkspace::separable_fluence(const double intensity_factor) {
  const auto scratch = transverse_;
  const auto closure_scratch = flux_transverse_;
  const auto partials = spatial_statistics_partials_;
  const PhysicalSpectrum spectrum{.layout = transverse_layout(),
                                  .synthesis = synthesis_,
                                  .n_x = problem_.spatial.n_x,
                                  .n_p = problem_.spatial.n_p,
                                  .crop_scale = static_cast<double>(problem_.dims.n_t) /
                                                static_cast<double>(problem_.dims.n_to),
                                  .slab = problem_.slab_geometry()};
  const auto n_nodes = problem_.dims.n_nodes;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto n = problem_.dims.n_omega;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  const double measure =
      intensity_factor * problem_.grid.dt / static_cast<double>(problem_.dims.n_t);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // One node per thread, whole planes per block: adjacent threads read adjacent cells.
  launch::parallel_for(
      "solprop_native.free.statistics_fluence",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
      KOKKOS_LAMBDA(const std::size_t node) {
        double sum = 0.0, electric_size = 0.0, magnetic_size = 0.0;
        for (std::size_t index = 0; index < n; ++index) {
          const double weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol) {
            const Complex electric = spectrum.read(scratch, index * n_pol + pol, node);
            const Complex magnetic =
                spectrum.read(closure_scratch, index * n_pol + pol, node);
            sum += weight * (electric.real() * magnetic.real() +
                             electric.imag() * magnetic.imag());
            electric_size += weight * squared_abs(electric);
            magnetic_size += weight * squared_abs(magnetic);
          }
        }
        partials(node, 0) = measure * sum;
        partials(node, 2) =
            measure * Kokkos::sqrt(electric_size) * Kokkos::sqrt(magnetic_size);
      });
#else
  // A task takes a run of adjacent nodes and reads each plane's cells of the run in
  // order: one node at a time would touch a new plane at every sample.
  constexpr std::size_t run = 256;
  const std::size_t runs = (n_nodes + run - 1) / run;
  launch::parallel_for(
      "solprop_native.free.statistics_fluence",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, runs),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t first = item * run;
        const std::size_t count = n_nodes - first < run ? n_nodes - first : run;
        double sum[run] = {};
        double electric_size[run] = {};
        double magnetic_size[run] = {};
        for (std::size_t index = 0; index < n; ++index) {
          const double weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol) {
            const auto plane = index * n_pol + pol;
            for (std::size_t offset = 0; offset < count; ++offset) {
              const Complex electric = spectrum.read(scratch, plane, first + offset);
              const Complex magnetic =
                  spectrum.read(closure_scratch, plane, first + offset);
              sum[offset] += weight * (electric.real() * magnetic.real() +
                                       electric.imag() * magnetic.imag());
              electric_size[offset] += weight * squared_abs(electric);
              magnetic_size[offset] += weight * squared_abs(magnetic);
            }
          }
        }
        for (std::size_t offset = 0; offset < count; ++offset) {
          partials(first + offset, 0) = measure * sum[offset];
          partials(first + offset, 2) = measure * Kokkos::sqrt(electric_size[offset]) *
                                        Kokkos::sqrt(magnetic_size[offset]);
        }
      });
#endif
}

// What a plane's widths read of its fluence (`separable_fluence`), each a reduction
// over the nodes, so that the profile itself stays on the device: `moments` takes the
// lowest value and its node, the largest magnitude, the largest value and the count of
// values that aren't finite and, when the plane has a positive value and none that
// isn't finite, the scaled profile's sum and index moments. That scaled profile is left
// in `partials(node, 0)` for `plane_fluence_trace`. Each reduction blocks: one plane's
// work.
void GeneralWorkspace::plane_fluence_moments(double (&moments)[plane_moment_slots]) {
  const auto partials = spatial_statistics_partials_;
  const auto n_nodes = problem_.dims.n_nodes;
  const auto n_x = problem_.spatial.n_x;
  const Kokkos::RangePolicy<ExecutionSpace> nodes(execution_space_, 0, n_nodes);
  std::size_t nonfinite = 0;
  launch::parallel_reduce(
      "solprop_native.free.plane_fluence_finite", nodes,
      KOKKOS_LAMBDA(const std::size_t node, std::size_t& update) {
        if (!Kokkos::isfinite(partials(node, 0)))
          ++update;
      },
      Kokkos::Sum<std::size_t>(nonfinite));
  Kokkos::MinLoc<double, std::size_t>::value_type worst{};
  launch::parallel_reduce(
      "solprop_native.free.plane_fluence_worst", nodes,
      KOKKOS_LAMBDA(const std::size_t node,
                    Kokkos::MinLoc<double, std::size_t>::value_type& update) {
        if (partials(node, 0) < update.val) {
          update.val = partials(node, 0);
          update.loc = node;
        }
      },
      Kokkos::MinLoc<double, std::size_t>(worst));
  double size = 0.0;
  launch::parallel_reduce(
      "solprop_native.free.plane_fluence_size", nodes,
      KOKKOS_LAMBDA(const std::size_t node, double& update) {
        update = Kokkos::max(update, partials(node, 2));
      },
      Kokkos::Max<double>(size));
  double highest = 0.0;
  launch::parallel_reduce(
      "solprop_native.free.plane_fluence_highest", nodes,
      KOKKOS_LAMBDA(const std::size_t node, double& update) {
        update = Kokkos::max(update, partials(node, 0));
      },
      Kokkos::Max<double>(highest));
  // A negative value within its budget is read as zero, as the reference reads it.
  const double maximum = Kokkos::max(highest, 0.0);
  FluenceMoments sums;
  plane_fluence_held_ = nonfinite == 0 && maximum > 0.0;
  if (plane_fluence_held_)
    launch::parallel_reduce(
        "solprop_native.free.plane_fluence_moments", nodes,
        KOKKOS_LAMBDA(const std::size_t node, FluenceMoments& update) {
          // Scaling cancels in the centroid and the widths and keeps them from
          // overflowing on a bright beam.
          const double sample = Kokkos::max(partials(node, 0), 0.0) / maximum;
          partials(node, 0) = sample;
          update.total += sample;
          const std::size_t row = node / n_x;
          update.x += sample * static_cast<double>(node % n_x);
          update.y += sample * static_cast<double>(row);
        },
        Kokkos::Sum<FluenceMoments>(sums));
  moments[0] = worst.val;
  moments[1] = static_cast<double>(worst.loc);
  moments[2] = size;
  moments[3] = maximum;
  moments[4] = static_cast<double>(nonfinite);
  moments[5] = sums.total;
  moments[6] = sums.x;
  moments[7] = sums.y;
}

// The second moment and the azimuthal average of a plane's scaled fluence, left by a
// call that asked for its moments. The rule is solprop's
// (`solprop.spatial.cartesian_fluence_widths`), which owns the rings and hands them
// over: each ring samples the bilinear interpolant at `cosines.size()` angles about the
// centroid, with the centre in metres from the axes' first samples, so that only the
// trace crosses, not the profile. Returned: the second-moment sum, then the trace.
std::vector<double> GeneralWorkspace::plane_fluence_trace(
    const double center_x, const double center_y, const double dx, const double dy,
    const double extent, const double spacing, const std::size_t n_radii,
    const std::vector<double>& cosines, const std::vector<double>& sines) {
  if (!plane_fluence_held_)
    throw std::logic_error("a plane's fluence trace follows the call that holds it");
  plane_fluence_held_ = false;
  if (cosines.size() != sines.size() || (n_radii != 0 && cosines.empty()) ||
      !(dx > 0.0) || !(dy > 0.0) || !(extent > 0.0) ||
      (n_radii != 0 && !(spacing > 0.0)))
    throw std::invalid_argument("invalid fluence rings");
  const auto n_angles = cosines.size();
  const auto packed_size =
      checked_sum(checked_sum(1, n_radii, "fluence trace"),
                  checked_product(2, n_angles, "fluence rings"), "fluence trace");
  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native spatial statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  const auto packed = statistics_packed_;
  const auto samples = spatial_statistics_partials_;
  const auto n_x = problem_.spatial.n_x;
  const auto n_y = problem_.spatial.n_y;
  const auto first_ring = n_radii + 1;
  if (n_angles != 0) {
    std::vector<double> rings(cosines);
    rings.insert(rings.end(), sines.begin(), sines.end());
    copy_vector(execution_space_,
                Kokkos::subview(packed, std::make_pair(first_ring, packed_size)),
                rings);
  }
  double second = 0.0;
  launch::parallel_reduce(
      "solprop_native.free.plane_fluence_second",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_nodes),
      KOKKOS_LAMBDA(const std::size_t node, double& update) {
        const double x = (static_cast<double>(node % n_x) * dx - center_x) / extent;
        const std::size_t row = node / n_x;
        const double y = (static_cast<double>(row) * dy - center_y) / extent;
        update += samples(node, 0) * (x * x + y * y);
      },
      Kokkos::Sum<double>(second));
  const auto last_x = static_cast<double>(n_x - 1);
  const auto last_y = static_cast<double>(n_y - 1);
  // One thread per ring: a ring's samples are few, and a thread's sum needs no
  // reduction.
  launch::parallel_for(
      "solprop_native.free.plane_fluence_rings",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_radii),
      KOKKOS_LAMBDA(const std::size_t ring) {
        const double distance = spacing * static_cast<double>(ring);
        double sum = 0.0;
        for (std::size_t angle = 0; angle < n_angles; ++angle) {
          const double horizontal = Kokkos::min(
              Kokkos::max((center_x + distance * packed(first_ring + angle)) / dx, 0.0),
              last_x);
          const double vertical = Kokkos::min(
              Kokkos::max(
                  (center_y + distance * packed(first_ring + n_angles + angle)) / dy,
                  0.0),
              last_y);
          const auto ix =
              Kokkos::min(static_cast<std::size_t>(horizontal), n_x - std::size_t{2});
          const auto iy =
              Kokkos::min(static_cast<std::size_t>(vertical), n_y - std::size_t{2});
          const double fx = horizontal - static_cast<double>(ix);
          const double fy = vertical - static_cast<double>(iy);
          const auto low = iy * n_x + ix;
          const auto high = low + n_x;
          sum +=
              (1.0 - fy) * ((1.0 - fx) * samples(low, 0) + fx * samples(low + 1, 0)) +
              fy * ((1.0 - fx) * samples(high, 0) + fx * samples(high + 1, 0));
        }
        packed(1 + ring) = sum / static_cast<double>(n_angles);
      });
  const auto trace =
      Kokkos::subview(packed, std::make_pair(std::size_t{1}, first_ring));
  auto host = Kokkos::create_mirror_view(trace);
  graph::deep_copy(execution_space_, host, trace);
  execution_space_.fence("solprop_native fluence trace boundary");
  std::vector<double> result(first_ring);
  result[0] = second;
  for (std::size_t ring = 0; ring < n_radii; ++ring)
    result[1 + ring] = host(ring);
  return result;
}

// The field's spectrum at one physical point, the matrix `(n_rows, n_modes)` contracted
// with the modal state: `(n_rows * n_pol, n_omega)` samples, each a real and an
// imaginary part in turn. The transitional session's on-axis statistics scan this row
// on the host (solprop SEP 19), so that it is all a plane's scan brings across; the
// native session scans it in place (`axis_populations`).
std::vector<Complex>
GeneralWorkspace::axis_spectrum(const std::vector<double>& matrix) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (problem_.separable_geometry() || matrix.empty() ||
      matrix.size() % problem_.dims.n_modes != 0 ||
      matrix.size() / problem_.dims.n_modes >
          statistics_on_axis_.size() / problem_.dims.n_modes)
    throw std::invalid_argument(
        "an on-axis row needs a modal state and at most two rows of n_modes columns");
  const auto n_rows = matrix.size() / problem_.dims.n_modes;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto n_modes = problem_.dims.n_modes;
  const auto n = problem_.dims.n_omega;
  const auto columns = checked_product(n_rows, n_pol, "on-axis columns");
  const auto samples = checked_product(columns, n, "on-axis samples");
  const auto packed_size = checked_product(2, samples, "on-axis samples");
  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  copy_vector(execution_space_,
              Kokkos::subview(statistics_on_axis_,
                              std::make_pair<std::size_t>(0, matrix.size())),
              matrix);
  const auto weights = statistics_on_axis_;
  const auto packed = statistics_packed_;
  const auto current = current_;
  launch::parallel_for(
      "solprop_native.statistics.axis_spectrum",
      Range2(execution_space_, {0, 0}, {columns, n}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const auto row = column / n_pol;
        const auto pol = column % n_pol;
        Complex sum{};
        for (std::size_t mode = 0; mode < n_modes; ++mode)
          sum += weights(row * n_modes + mode) * current(mode * n_pol + pol, index);
        const auto at = 2 * (column * n + index);
        packed(at) = sum.real();
        packed(at + 1) = sum.imag();
      });
  const auto filled =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(filled);
  graph::deep_copy(execution_space_, host, filled);
  execution_space_.fence("solprop_native on-axis row boundary");
  std::vector<Complex> result(samples);
  for (std::size_t index = 0; index < samples; ++index)
    result[index] = Complex(host(2 * index), host(2 * index + 1));
  return result;
}

// The modal state on the fine time grid, in `modal_real_` or `modal_complex_`: the
// coarse bins padded at `crop_index_` and transformed back, the node scans' own first
// pass (`statistics`); a retained synthesis transforms at its nodes instead.
void GeneralWorkspace::statistics_modal_time() {
  graph::deep_copy(execution_space_, fine_spectrum_, Complex{});
  const auto current = current_;
  const auto fine = fine_spectrum_;
  const auto crop = crop_index_;
  const auto inverse_n_to = 1.0 / static_cast<double>(problem_.dims.n_to);
  const auto n_to = problem_.dims.n_to;
  const bool analytic = problem_.dims.family == FieldFamily::field_analytic;
  launch::parallel_for(
      "solprop_native.plasma.statistics_pad",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        const double analytic_scale =
            analytic && index > 0 && 2 * crop(index) < n_to ? 2.0 : 1.0;
        fine(column, crop(index)) =
            analytic_scale * inverse_n_to * current(column, index);
      });
  if (problem_.dims.family == FieldFamily::field_real)
    graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                   KokkosFFT::Normalization::none);
  else if (!retained_synthesis_)
    graph::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                   KokkosFFT::Normalization::none);
}

// solprop SEP 19's on-axis plasma populations, scanned natively (solprop SEP 47 phase
// 4): the modal state on the fine grid (`statistics_modal_time`), contracted with
// `matrix` -- one or two rows of `n_modes` columns, the pad scale and the taper in
// them, as `solprop.backends.populations.axis_row_matrix` forms them -- into the first
// node's buffer, and scanned there by the node scan itself, every channel with its
// tables and running-integral rule. Returns each channel's final fraction and peak
// rate, `(n_channels, 2)`. Two components on a one-component node buffer are the
// field's magnitude, `hypot(E_x, E_y)`, which is all a field grid's scan reads; an
// envelope's scan reads their polarisation ellipse, so they go to a row of their own
// and the envelope scan reads it as a node of two components
// (`solprop.backends.populations.envelope_populations` on the host's row).
std::vector<double>
GeneralWorkspace::axis_populations(const std::vector<double>& matrix,
                                   const std::vector<double>& channel_preionisation) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!problem_.plasma.enabled || problem_.plasma.n_channels == 0)
    throw std::invalid_argument(
        "the on-axis plasma statistics need a compiled plasma response");
  if (problem_.separable_geometry() || problem_.general.enabled || retained_synthesis_)
    throw std::invalid_argument(
        "the on-axis plasma statistics read a resolved modal state");
  ++axis_scans_;
  const auto n_modes = problem_.dims.n_modes;
  const auto n_pol_state = problem_.dims.n_pol_state;
  const auto node_pol = problem_.effective_node_pol();
  if (matrix.empty() || matrix.size() % n_modes != 0 || matrix.size() / n_modes > 2)
    throw std::invalid_argument(
        "an on-axis row needs one or two rows of n_modes columns");
  auto rows = matrix.size() / n_modes;
  // A y row of zeros on a one-component node scan is dropped: `hypot(E_x, 0)` is
  // `|E_x|`, and an envelope's ellipse with no y axis is its x component's.
  if (rows == 2 && n_pol_state == 1 && node_pol == 1 &&
      std::all_of(matrix.begin() + static_cast<std::ptrdiff_t>(n_modes), matrix.end(),
                  [](const double value) { return value == 0.0; }))
    rows = 1;
  const auto components = rows * n_pol_state;
  const bool pair = components == 2 && node_pol == 1;
  if (components != node_pol && !pair)
    throw std::invalid_argument(
        "an on-axis row has one component on each of the node scan's");
  // A pair on a field grid is combined into its magnitude in the node buffer; an
  // envelope's is kept, x and y, in its own row.
  const bool ellipse = pair && problem_.plasma.envelope;
  const bool combine = pair && !ellipse;
  if (ellipse && statistics_axis_row_.extent(0) == 0) {
    execution_space_.fence("solprop_native on-axis row allocation");
    statistics_axis_row_ =
        ComplexView2("solprop_native.statistics.axis_row", 2, problem_.dims.n_to);
  }
  require_extent(channel_preionisation.size(), problem_.plasma.n_channels,
                 "on-axis channel preionisation");
  for (const double value : channel_preionisation)
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
      throw std::invalid_argument("on-axis channel preionisation must be in [0, 1]");
  auto initial_host = Kokkos::create_mirror_view(stage_channel_preionisation_);
  for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
    initial_host(0, channel) = channel_preionisation[channel];
  graph::deep_copy(execution_space_, stage_channel_preionisation_, initial_host);
  const std::vector<double> held(
      matrix.begin(), matrix.begin() + static_cast<std::ptrdiff_t>(rows * n_modes));
  copy_vector(
      execution_space_,
      Kokkos::subview(statistics_on_axis_, std::make_pair<std::size_t>(0, held.size())),
      held);
  statistics_modal_time();
  const auto weights = statistics_on_axis_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto axis_row = statistics_axis_row_;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto written = combine ? std::size_t{1} : components;
  launch::parallel_for(
      "solprop_native.statistics.axis_row",
      Range2(execution_space_, {0, 0}, {written, problem_.dims.n_to}),
      KOKKOS_LAMBDA(const std::size_t component, const std::size_t time) {
        // Component `row * n_pol_state + pol`, `passes.synthesise`'s layout.
        const auto sum_real = [&](const std::size_t row, const std::size_t pol) {
          double value = 0.0;
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += weights(row * n_modes + mode) *
                     modal_real(mode * n_pol_state + pol, time);
          return value;
        };
        const auto sum_complex = [&](const std::size_t row, const std::size_t pol) {
          Complex value{};
          for (std::size_t mode = 0; mode < n_modes; ++mode)
            value += weights(row * n_modes + mode) *
                     modal_complex(mode * n_pol_state + pol, time);
          return value;
        };
        const auto row = component / n_pol_state;
        const auto pol = component % n_pol_state;
        if (ellipse) {
          axis_row(component, time) = sum_complex(row, pol);
        } else if (combine) {
          // Both rows on a scalar state: the magnitude the scan would form.
          if (real_family)
            node_real(0, time) = Kokkos::hypot(sum_real(0, 0), sum_real(1, 0));
          else
            node_complex(0, time) = Complex(
                Kokkos::hypot(sum_complex(0, 0).real(), sum_complex(1, 0).real()), 0.0);
        } else if (real_family) {
          node_real(component, time) = sum_real(row, pol);
        } else {
          node_complex(component, time) = sum_complex(row, pol);
        }
      });
  const auto source = plasma_statistics_scratch_;
  if (ellipse)
    for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
      envelope_statistics(channel, 0, 1, source, statistics_axis_row_, 2);
  else
    scan_plasma_statistics(0, 1, source, false, channel_preionisation[0]);
  const auto channels = problem_.plasma.n_channels;
  const auto packed_size = 2 * channels;
  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  const auto packed = statistics_packed_;
  const bool envelope = problem_.plasma.envelope;
  const auto initial = stage_channel_preionisation_;
  // `statistics`' own packing of a node's populations.
  launch::parallel_for(
      "solprop_native.statistics.pack_axis_populations",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, channels),
      KOKKOS_LAMBDA(const std::size_t channel) {
        packed(2 * channel) =
            envelope ? envelope_fraction(source(0, 2 * channel), initial(0, channel))
                     : ionised_fraction(source(0, 2 * channel), initial(0, channel));
        packed(2 * channel + 1) = source(0, 2 * channel + 1);
      });
  const auto filled =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(filled);
  graph::deep_copy(execution_space_, host, filled);
  execution_space_.fence("solprop_native on-axis populations boundary");
  std::vector<double> result(packed_size);
  for (std::size_t index = 0; index < packed_size; ++index)
    result[index] = host(index);
  return result;
}

std::vector<double> GeneralWorkspace::aperture_statistics(
    const std::uint64_t flags, const std::vector<double>& apertures, const int exponent,
    const std::vector<double>& regions, const double intensity_factor) {
  if (!initialized_ || !far_field_)
    throw std::logic_error(
        "far-field statistics need an initial field on a Cartesian or line grid");
  return far_field_->run(flags, apertures, exponent, regions, intensity_factor,
                         current_, far_field_inputs());
}

FarFieldPass::Inputs GeneralWorkspace::far_field_inputs() const {
  return FarFieldPass::Inputs{.weights = spatial_spectral_weights_,
                              .on_axis = spatial_on_axis_,
                              .square = spatial_.square,
                              .square_low = spatial_.square_low,
                              .transverse = spatial_.transverse,
                              .transverse_low = spatial_.transverse_low,
                              .spectral_rate = spatial_.spectral_rate,
                              .user_rate = spatial_.user_rate,
                              .transverse_rate = spatial_.transverse_rate,
                              .omega = omega_,
                              .active = active_mask_,
                              .evanescent = spatial_.evanescent};
}

std::vector<double> GeneralWorkspace::spatial_statistics(const std::uint64_t flags,
                                                         const double intensity_factor,
                                                         const double z) {
  if (!initialized_ || spatial_spectral_weights_.empty())
    throw std::logic_error(
        "spatial statistics need an initial field and spatial geometry");
  if (flux_.square.empty())
    throw std::logic_error("spatial statistics need the run's plane measure");
  if ((flags & ~spatial_flags) != 0 || !std::isfinite(intensity_factor) ||
      intensity_factor <= 0.0 || !std::isfinite(z))
    throw std::invalid_argument(
        "invalid spatial statistic flags, intensity factor or position");
  if ((flags & spatial_moments) != 0 &&
      (!problem_.free_geometry() || problem_.slab_geometry() ||
       problem_.spatial.n_y < 2))
    throw std::invalid_argument("a plane's fluence moments need a Cartesian plane");
  constexpr double lowest = std::numeric_limits<double>::lowest();
  std::size_t packed_size = 0;
  const auto reserve = [flags, &packed_size](const std::uint64_t bit,
                                             const std::size_t count) {
    if ((flags & bit) == 0)
      return absent;
    const auto offset = packed_size;
    packed_size = checked_sum(packed_size, count, "spatial statistics output");
    return offset;
  };
  const auto energy_offset = reserve(spatial_energy, 1);
  const auto axis_peak_offset = reserve(spatial_axis, 1);
  const auto peak_offset = reserve(spatial_peak, 1);
  const auto spectrum_offset = reserve(spatial_spectrum, problem_.dims.n_omega);
  // Widths read the signed fluence at the nodes, the separately reconstructed axis,
  // which only a radial width reads, and the largest magnitude, which fixes the
  // fluence's rounding budget (solprop SEP 34).
  const auto fluence_offset =
      reserve(spatial_fluence,
              checked_sum(problem_.dims.n_nodes, 2, "spatial statistics output"));
  // On a Cartesian plane the widths reduce here instead, and the moments cross.
  const auto moments_offset = reserve(spatial_moments, plane_moment_slots);
  if (packed_size == 0)
    return {};
  if (statistics_packed_.size() < packed_size) {
    execution_space_.fence("solprop_native spatial statistics output resize");
    statistics_packed_ = RealView("solprop_native.statistics.packed", packed_size);
  }
  // Every field statistic is the axial flux of the medium at the plane z.
  flux_at(z);
  const bool local =
      (flags & (spatial_axis | spatial_peak | spatial_fluence | spatial_moments)) != 0;
  if (local)
    require_flux_closure();
  const auto packed = statistics_packed_;
  const auto current = current_;
  const auto spectral_weights = spatial_spectral_weights_;
  const auto axis_weights = spatial_on_axis_;
  const auto partials = spatial_statistics_partials_;
  const auto flux = flux_;
  const auto active = active_mask_;
  const auto omega = omega_;
  const auto n_modes = problem_.dims.n_modes;
  const auto n_nodes = problem_.dims.n_nodes;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto n = problem_.dims.n_omega;
  const auto n_t = problem_.dims.n_t;
  const auto dt = problem_.grid.dt;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  if (energy_offset != absent) {
    launch::parallel_for(
        "solprop_native.spatial.energy_partials",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          double sum = 0.0;
          for (std::size_t index = 0; index < n; ++index) {
            const double admittance =
                flux.closing(mode, index, active(index) != 0U, omega(index)).real();
            if (admittance == 0.0)
              continue;
            double squared = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol)
              squared += squared_abs(current(mode * n_pol + pol, index));
            const auto weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
            sum += weight * admittance * squared;
          }
          partials(mode, 0) = intensity_factor * spectral_weights(mode) * dt /
                              static_cast<double>(n_t) * sum;
        });
    launch::parallel_reduce(
        "solprop_native.spatial.pack_energy",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_modes),
        KOKKOS_LAMBDA(const std::size_t mode, double& sum) {
          sum += partials(mode, 0);
        },
        Kokkos::Sum<double, MemorySpace>(Kokkos::subview(packed, energy_offset)));
  }
  if (spectrum_offset != absent) {
    // The coherent on-axis amplitude paired at each frequency with its closure.
    launch::parallel_for(
        "solprop_native.spatial.axis_spectrum",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
        KOKKOS_LAMBDA(const std::size_t index) {
          const bool live = active(index) != 0U;
          double power = 0.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol) {
            Complex electric{};
            Complex magnetic{};
            for (std::size_t mode = 0; mode < n_modes; ++mode) {
              const Complex value =
                  axis_weights(mode) * current(mode * n_pol + pol, index);
              electric += value;
              magnetic += flux.closing(mode, index, live, omega(index)) * value;
            }
            power += (electric * Kokkos::conj(magnetic)).real();
          }
          const auto weight = envelope || index == 0 || index + 1 == n ? 1.0 : 4.0;
          packed(spectrum_offset + index) =
              intensity_factor * dt * dt /
              (2.0 * Kokkos::numbers::pi_v<double>)*weight * power;
        });
  }
  double moments[plane_moment_slots] = {};
  if (problem_.separable_geometry()) {
    separable_time_statistics(axis_peak_offset, peak_offset,
                              fluence_offset != absent || moments_offset != absent,
                              intensity_factor);
    if (fluence_offset != absent) {
      launch::parallel_for(
          "solprop_native.free.fluence_rows",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
          KOKKOS_LAMBDA(const std::size_t node) {
            packed(fluence_offset + node) = partials(node, 0);
          });
      launch::parallel_reduce(
          "solprop_native.free.fluence_largest",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
          KOKKOS_LAMBDA(const std::size_t node, double& maximum) {
            maximum = Kokkos::max(maximum, partials(node, 2));
          },
          Kokkos::Max<double, MemorySpace>(
              Kokkos::subview(packed, fluence_offset + n_nodes + 1)));
    }
    if (moments_offset != absent)
      plane_fluence_moments(moments);
  } else if (local) {
    if (n_nodes != n_modes)
      throw std::logic_error("radial statistics need one node per Hankel column");
    fill_statistics_closure();
    fill_statistics_time(false, 0.0, 0.0);
    const auto field = statistics_time_;
    const auto closure = statistics_weighted_time_;
    const auto axis_power = statistics_total_power_;
    if (axis_peak_offset != absent || fluence_offset != absent) {
      launch::parallel_for(
          "solprop_native.spatial.axis_intensity",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_t),
          KOKKOS_LAMBDA(const std::size_t time) {
            double power = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol) {
              Complex electric{};
              Complex magnetic{};
              for (std::size_t mode = 0; mode < n_modes; ++mode) {
                electric += axis_weights(mode) * field(mode * n_pol + pol, time);
                magnetic += axis_weights(mode) * closure(mode * n_pol + pol, time);
              }
              power += (electric * Kokkos::conj(magnetic)).real();
            }
            axis_power(time) = intensity_factor * power;
          });
      launch::parallel_for(
          "solprop_native.spatial.pack_axis",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
          KOKKOS_LAMBDA(const std::size_t) {
            double peak = lowest;
            double sum = 0.0;
            for (std::size_t time = 0; time < n_t; ++time) {
              peak = Kokkos::max(peak, axis_power(time));
              sum += axis_power(time);
            }
            if (axis_peak_offset != absent)
              packed(axis_peak_offset) = peak;
            if (fluence_offset != absent)
              packed(fluence_offset + n_nodes) = dt * sum;
          });
    }
    if (peak_offset != absent || fluence_offset != absent) {
      const auto power = statistics_power_;
      const auto magnitude = statistics_magnitude_;
      const auto synthesis = synthesis_;
      const double crop_scale = static_cast<double>(problem_.dims.n_t) /
                                static_cast<double>(problem_.dims.n_to);
      // Each node's field and closure, `S F` and `S G` over the modes, as GEMMs over
      // interleaved real columns (each sample's real and imaginary parts; a mode's
      // polarisations are adjacent rows): the field's into the spectrum buffer, free
      // once both time fields are formed, then the closure's over the field, which
      // nothing reads after its synthesis. Local statistics hold one node per mode, so
      // each buffer has the rows.
      using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      static_assert(sizeof(Complex) == 2 * sizeof(double));
      const std::size_t width = 2 * n_pol * n_t;
      if (statistics_spectrum_.size() < n_nodes * n_pol * n_t ||
          field.size() < n_nodes * n_pol * n_t)
        throw std::logic_error("radial statistics need a row of each buffer per node");
      const Interleaved weights(synthesis.data(), n_nodes, n_modes);
      const Interleaved field_rows(reinterpret_cast<double*>(field.data()), n_modes,
                                   width);
      const Interleaved closure_rows(reinterpret_cast<double*>(closure.data()), n_modes,
                                     width);
      const Interleaved electric_rows(
          reinterpret_cast<double*>(statistics_spectrum_.data()), n_nodes, width);
      const Interleaved magnetic_rows(reinterpret_cast<double*>(field.data()), n_nodes,
                                      width);
      engine_gemm(transform_execution_space_, weights, field_rows, electric_rows, 0.0);
      engine_gemm(transform_execution_space_, weights, closure_rows, magnetic_rows,
                  0.0);
      launch::parallel_for(
          "solprop_native.spatial.physical_intensity",
          Range2(execution_space_, {0, 0}, {n_nodes, n_t}),
          KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
            double total = 0.0;
            double size = 0.0;
            for (std::size_t pol = 0; pol < n_pol; ++pol) {
              const std::size_t at = 2 * (pol * n_t + time);
              const Complex electric =
                  crop_scale *
                  Complex(electric_rows(node, at), electric_rows(node, at + 1));
              const Complex magnetic =
                  crop_scale *
                  Complex(magnetic_rows(node, at), magnetic_rows(node, at + 1));
              total += (electric * Kokkos::conj(magnetic)).real();
              size += Kokkos::sqrt(squared_abs(electric) * squared_abs(magnetic));
            }
            power(node, time) = intensity_factor * total;
            magnitude(node, time) = intensity_factor * size;
          });
      launch::parallel_for(
          "solprop_native.spatial.fluence_partials",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
          KOKKOS_LAMBDA(const std::size_t node) {
            double peak = lowest;
            double sum = 0.0;
            double size = 0.0;
            for (std::size_t time = 0; time < n_t; ++time) {
              peak = Kokkos::max(peak, power(node, time));
              sum += power(node, time);
              size += magnitude(node, time);
            }
            partials(node, 1) = peak;
            partials(node, 2) = dt * size;
            if (fluence_offset != absent)
              packed(fluence_offset + node) = dt * sum;
          });
      if (fluence_offset != absent)
        launch::parallel_reduce(
            "solprop_native.spatial.fluence_largest",
            Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n_nodes),
            KOKKOS_LAMBDA(const std::size_t node, double& maximum) {
              maximum = Kokkos::max(maximum, partials(node, 2));
            },
            Kokkos::Max<double, MemorySpace>(
                Kokkos::subview(packed, fluence_offset + n_nodes + 1)));
      if (peak_offset != absent)
        launch::parallel_for(
            "solprop_native.spatial.pack_peak",
            Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
            KOKKOS_LAMBDA(const std::size_t) {
              double peak = lowest;
              for (std::size_t node = 0; node < n_nodes; ++node)
                peak = Kokkos::max(peak, partials(node, 1));
              packed(peak_offset) = peak;
            });
    }
  }
  const auto filled =
      Kokkos::subview(packed, std::make_pair<std::size_t>(0, packed_size));
  auto host = Kokkos::create_mirror_view(filled);
  graph::deep_copy(execution_space_, host, filled);
  execution_space_.fence("solprop_native spatial statistics packed boundary");
  std::vector<double> result(packed_size);
  for (std::size_t index = 0; index < packed_size; ++index)
    result[index] = host(index);
  // The separable routes don't form the axis's fluence: only a radial width reads it.
  if (fluence_offset != absent && problem_.separable_geometry())
    result[fluence_offset + n_nodes] = Kokkos::Experimental::quiet_NaN_v<double>;
  if (moments_offset != absent)
    std::copy_n(moments, plane_moment_slots, result.data() + moments_offset);
  return result;
}

#include "statistics_monitor.inc"
} // namespace solprop_native
