// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The responses: the template dispatch, response_impl, the device response and
// the device trapezoid plasma, then the band plasma, the Raman convolution and
// the envelope plasma, in that order.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash>
void GeneralWorkspace::response_harmonic(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count) {
  if constexpr (Nonlinear && Family != FieldFamily::field_real) {
    if (problem_.kerr.harmonic_factor != 0.0) {
      response_raman<Family, Components, Plasma, Nonlinear, Stash, true>(
          stage, first_node, node_count);
      return;
    }
  }
  response_raman<Family, Components, Plasma, Nonlinear, Stash, false>(stage, first_node,
                                                                      node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic>
void GeneralWorkspace::response_raman(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if (problem_.raman.enabled) {
    if (problem_.raman.thg) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, true,
                    false>(stage, first_node, node_count);
    } else {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, true, false,
                    false>(stage, first_node, node_count);
    }
    return;
  }
  response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, false, false,
                false>(stage, first_node, node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear>
void GeneralWorkspace::response_stash(const std::size_t stage,
                                      const std::size_t first_node,
                                      const std::size_t node_count) {
  if constexpr (Plasma) {
    // Clang's host loop is measurably faster when the maximum-rate accumulator is
    // present. Keep that loop shape for every plasma problem, while the view and its
    // final-stage stores remain conditional on requested periodic statistics.
    response_harmonic<Family, Components, Plasma, Nonlinear, true>(stage, first_node,
                                                                   node_count);
    return;
  }
  response_harmonic<Family, Components, Plasma, Nonlinear, false>(stage, first_node,
                                                                  node_count);
}

template <FieldFamily Family, std::size_t Components, bool Plasma>
void GeneralWorkspace::response_nonlinear(const std::size_t stage,
                                          const std::size_t first_node,
                                          const std::size_t node_count) {
  if (problem_.kerr.nonlinear) {
    response_stash<Family, Components, Plasma, true>(stage, first_node, node_count);
  } else {
    response_stash<Family, Components, Plasma, false>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components>
void GeneralWorkspace::response_plasma(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  if (problem_.plasma.enabled && !problem_.plasma.envelope) {
    if constexpr (Family != FieldFamily::envelope) {
      // SEP 27: the band column writes the physical current separately. The
      // ordinary traversal forms and collars only the Kerr/Raman polarisation.
      if (problem_.plasma.band) {
        if (band_tiled_) {
          if (problem_.plasma.n_channels > 1)
            tiled_band_plasma_column<Family, Components, true>(stage, first_node,
                                                               node_count);
          else
            tiled_band_plasma_column<Family, Components, false>(stage, first_node,
                                                                node_count);
        } else if (problem_.plasma.n_channels > 1) {
          team_band_plasma_column<Family, Components, true>(stage, first_node,
                                                            node_count);
        } else {
          team_band_plasma_column<Family, Components, false>(stage, first_node,
                                                             node_count);
        }
        response_nonlinear<Family, Components, false>(stage, first_node, node_count);
        return;
      }
    }
    response_nonlinear<Family, Components, true>(stage, first_node, node_count);
  } else {
    response_nonlinear<Family, Components, false>(stage, first_node, node_count);
  }
}

void GeneralWorkspace::response(const std::size_t stage, const std::size_t first_node,
                                const std::size_t node_count) {
  const auto components = problem_.effective_node_pol();
  switch (problem_.dims.family) {
  case FieldFamily::field_real:
    if (components == 1)
      response_plasma<FieldFamily::field_real, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_real, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::field_analytic:
    if (components == 1)
      response_plasma<FieldFamily::field_analytic, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::field_analytic, 2>(stage, first_node, node_count);
    return;
  case FieldFamily::envelope:
    if (components == 1)
      response_plasma<FieldFamily::envelope, 1>(stage, first_node, node_count);
    else
      response_plasma<FieldFamily::envelope, 2>(stage, first_node, node_count);
  }
}

template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg, bool Multiple,
          bool Folded>
void GeneralWorkspace::response_impl(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count) {
  // A single node's transforms fold into the column (`folds_single_node`), compiled
  // apart so that the other columns' arithmetic and layout are untouched.
  if constexpr (Components == 1 && !Raman && !Folded) {
    if (folds_single_node()) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, Multiple, true>(stage, first_node, node_count);
      return;
    }
  }
  if constexpr (!Multiple) {
    if (problem_.plasma.n_channels > 1 && Plasma) {
      response_impl<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg, true>(stage, first_node, node_count);
      return;
    }
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto held_current = held_current_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  // The modal products replace the nodes' drives (`raman_modal_products`).
  const bool node_drives = raman_products_ == 0;
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto rate_coefficients = rate_coefficients_;
  const auto stash = stash_candidate_;
  const auto density = stage_plasma_density_;
  const auto kerr_density = stage_kerr_density_;
  const auto channels = problem_.plasma.n_channels;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto scratch = channel_scratch_;
  const auto preionisation = stage_preionisation_;
  const auto n_to = problem_.dims.n_to;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double harmonic_factor = problem_.kerr.harmonic_factor;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const auto n_stages = problem_.stepper.n_stages;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // The device's form (`device_response`); a mixture of channels takes the generic
  // column below.
  if constexpr (!Multiple) {
    device_response<Family, Components, Plasma, Nonlinear, Stash, Harmonic, Raman,
                    RamanThg>(stage, first_node, node_count);
    return;
  }
#endif
  // A folded single node reads the modal field and writes the projected one, each
  // through its one-element transform (`folds_single_node`); otherwise both scales
  // are one, which changes no bit.
  constexpr bool folded = Folded;
  const auto input_real = folded ? modal_real_ : node_real_;
  const auto input_complex = folded ? modal_complex_ : node_complex_;
  const auto target_real = folded ? projected_real_ : polarisation_real_;
  const auto target_complex = folded ? projected_complex_ : polarisation_complex_;
  const auto synthesis = synthesis_;
  const auto projection = projection_;
  const auto taper = stage_taper_;
  launch::parallel_for(
      "solprop_native.response.column",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        // CUDA extended-lambda restriction 12 requires the first capture outside
        // if-constexpr blocks. Reference binding keeps every compile-time branch.
        capture_anchor(channels, scratch, channel_preionisation, node_real,
                       node_complex, channel_coefficients, channel_parameters, dt,
                       channel_density, rate_coefficients, log_field_min, d_log_field,
                       field_min, field_max, rate_max, rate_intervals,
                       ionisation_potential, charge_mass_ratio, density,
                       conjugate_factor, self_factor, harmonic_factor, carrier,
                       raman_drive, raman_drives, node_drives, out_real, held_current,
                       window, out_complex, stash, n_stages, first_node);
        const std::size_t base = node * Components;
        const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
        const double output_scale = folded ? projection(0, 0) / taper(stage) : 1.0;
        double exposure = 0.0;
        double last_rate = 0.0;
        double largest_rate = 0.0;
        double fraction = Plasma && !Multiple ? preionisation(stage) : 0.0;
        double free_first = 0.0;
        double free_second = 0.0;
        double last_first = 0.0;
        double last_second = 0.0;
        if constexpr (Multiple) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            scratch(channel, node, 0) = 0.0;
            scratch(channel, node, 1) = 0.0;
            scratch(channel, node, 2) = channel_preionisation(stage, channel);
            scratch(channel, node, 3) = 0.0;
          }
        }
        // Cache contiguous pointwise addresses to avoid repeated View stride loads.
        // Plasma keeps its measured recurrence-loop layout: the same change there made
        // the helium plasma case slower.
        auto* const real_input = input_real.data();
        auto* const complex_input = input_complex.data();
        auto* const real_target = target_real.data();
        auto* const complex_target = target_complex.data();
        auto* const window_data = window.data();
        const double pointwise_factor =
            Plasma ? 0.0 : kerr_coefficient * kerr_density(stage);
        for (std::size_t time = 0; time < n_to; ++time) {
          Complex first_value{};
          Complex second_value{};
          if constexpr (Family == FieldFamily::field_real) {
            first_value =
                Complex(input_scale * (Plasma ? input_real(base, time)
                                              : real_input[base * n_to + time]),
                        0.0);
            if constexpr (Components == 2)
              second_value =
                  Complex(input_scale * (Plasma ? input_real(base + 1, time)
                                                : real_input[(base + 1) * n_to + time]),
                          0.0);
          } else {
            first_value = input_scale * (Plasma ? input_complex(base, time)
                                                : complex_input[base * n_to + time]);
            if constexpr (Components == 2)
              second_value =
                  input_scale * (Plasma ? input_complex(base + 1, time)
                                        : complex_input[(base + 1) * n_to + time]);
          }
          const double modulus = squared_abs(first_value) +
                                 (Components == 2 ? squared_abs(second_value) : 0.0);
          const Complex square =
              first_value * first_value +
              (Components == 2 ? second_value * second_value : Complex{});
          if constexpr (Plasma) {
            const double magnitude =
                Components == 2 ? Kokkos::hypot(first_value.real(), second_value.real())
                                : Kokkos::abs(first_value.real());
            double guard = 0.0;
            double weight = 0.0;
            if constexpr (Multiple) {
              double work = 0.0;
              for (std::size_t channel = 0; channel < channels; ++channel) {
                const double rate = ionisation_rate(
                    magnitude, channel_coefficients, channel_parameters(channel, 0),
                    channel_parameters(channel, 1), channel_parameters(channel, 2),
                    channel_parameters(channel, 3), channel_parameters(channel, 4),
                    static_cast<std::size_t>(channel_parameters(channel, 5)),
                    static_cast<std::size_t>(channel_parameters(channel, 6)));
                double channel_exposure = scratch(channel, node, 0);
                const double previous_exposure = channel_exposure;
                if (time > 0)
                  channel_exposure += trapezoid(rate, scratch(channel, node, 1), dt);
                double channel_fraction = scratch(channel, node, 2);
                if (time == 0 || channel_exposure != previous_exposure)
                  channel_fraction = ionised_fraction(
                      channel_exposure, channel_preionisation(stage, channel));
                scratch(channel, node, 0) = channel_exposure;
                scratch(channel, node, 1) = rate;
                scratch(channel, node, 2) = channel_fraction;
                if (rate > scratch(channel, node, 3))
                  scratch(channel, node, 3) = rate;
                const double partial = channel_density(stage, channel);
                weight += partial * channel_fraction * channel_parameters(channel, 7);
                work += partial * channel_parameters(channel, 8) * rate *
                        (1.0 - channel_fraction);
              }
              guard = work > 0.0 && magnitude * magnitude > 0.0
                          ? work / (magnitude * magnitude)
                          : 0.0;
            } else {
              const double rate = ionisation_rate(magnitude, rate_coefficients,
                                                  log_field_min, d_log_field, field_min,
                                                  field_max, rate_max, rate_intervals);
              const double previous_exposure = exposure;
              if (time > 0)
                exposure += trapezoid(rate, last_rate, dt);
              last_rate = rate;
              if constexpr (Stash) {
                if (rate > largest_rate)
                  largest_rate = rate;
              }
              // Preserve the first subtraction's rounding even for pre-ionized
              // input; afterwards an unchanged rounded exposure has the same exp.
              if (time == 0 || exposure != previous_exposure)
                fraction = ionised_fraction(exposure, preionisation(stage));
              guard = rate > 0.0 && magnitude * magnitude > 0.0
                          ? ionisation_potential * rate * (1.0 - fraction) /
                                (magnitude * magnitude)
                          : 0.0;
              weight = fraction * charge_mass_ratio;
            }
            double integrand = weight * first_value.real();
            if (time > 0)
              free_first += trapezoid(integrand, last_first, dt);
            double current = free_first + guard * first_value.real();
            last_first = integrand;
            held_current(base, time) = Multiple ? current : density(stage) * current;
            if constexpr (Components == 2) {
              integrand = weight * second_value.real();
              if (time > 0)
                free_second += trapezoid(integrand, last_second, dt);
              current = free_second + guard * second_value.real();
              last_second = integrand;
              held_current(base + 1, time) =
                  Multiple ? current : density(stage) * current;
            }
          }
          const double factor =
              Plasma ? kerr_coefficient * kerr_density(stage) : pointwise_factor;
          Complex first{};
          Complex second{};
          if constexpr (Nonlinear) {
            if constexpr (Family == FieldFamily::field_real) {
              first += Complex(factor * modulus * first_value.real(), 0.0);
              if constexpr (Components == 2)
                second += Complex(factor * modulus * second_value.real(), 0.0);
            } else {
              const auto analytic = [&](const Complex value) {
                const double weighted_real = conjugate_factor * square.real();
                const double weighted_imag = conjugate_factor * square.imag();
                const double weighted_self = self_factor * modulus;
                Complex result(factor * (weighted_self * value.real() +
                                         (weighted_real * value.real() +
                                          weighted_imag * value.imag())),
                               factor * (weighted_self * value.imag() +
                                         (weighted_imag * value.real() -
                                          weighted_real * value.imag())));
                if constexpr (Harmonic)
                  result +=
                      ((factor * harmonic_factor) * carrier(time)) * square * value;
                return result;
              };
              first += analytic(first_value);
              if constexpr (Components == 2)
                second += analytic(second_value);
            }
          }
          if constexpr (Raman) {
            if (node_drives)
              write_raman_drive<Components, RamanThg>(
                  raman_drive, node, time, first_value, second_value, raman_drives);
          }
          if constexpr (Family != FieldFamily::envelope) {
            (Plasma ? target_real(base, time) : real_target[base * n_to + time]) =
                Raman ? first.real()
                      : output_scale * (first.real() *
                                        (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_real(base + 1, time)
                      : real_target[(base + 1) * n_to + time]) =
                  Raman ? second.real()
                        : output_scale * (second.real() *
                                          (Plasma ? window(time) : window_data[time]));
          } else {
            (Plasma ? target_complex(base, time) : complex_target[base * n_to + time]) =
                Raman ? first
                      : output_scale *
                            (first * (Plasma ? window(time) : window_data[time]));
            if constexpr (Components == 2)
              (Plasma ? target_complex(base + 1, time)
                      : complex_target[(base + 1) * n_to + time]) =
                  Raman ? second
                        : output_scale *
                              (second * (Plasma ? window(time) : window_data[time]));
          }
        }
        if constexpr (Stash) {
          if (stash.empty() || stage + 1 != n_stages)
            return;
          if constexpr (Multiple) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
              stash(first_node + node, 2 * channel) = scratch(channel, node, 2);
              stash(first_node + node, 2 * channel + 1) = scratch(channel, node, 3);
            }
          } else {
            stash(first_node + node, 0) = fraction;
            stash(first_node + node, 1) = largest_rate;
          }
        }
      });
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
// The device's response with no team tied to a node, so that a one-node fibre fills the
// device:
//
//   - the pointwise terms (Kerr, the harmonic, the Raman drive and the windowed
//     polarisation) as one two-dimensional range over (node, time), in the column's
//     arithmetic sample by sample;
//   - the trapezoid plasma as two-level scans over tiles of `plasma_tile_samples`:
//     each running integral is a kernel of one team per (row, tile) that sums the
//     tile's increments into its total (`plasma_tile_totals_`), then one per (row,
//     tile) that adds the row's earlier totals and scans the tile again from that
//     offset, forming the exposure's fraction or the current from each prefix.
template <FieldFamily Family, std::size_t Components, bool Plasma, bool Nonlinear,
          bool Stash, bool Harmonic, bool Raman, bool RamanThg>
void GeneralWorkspace::device_response(const std::size_t stage,
                                       const std::size_t first_node,
                                       const std::size_t node_count) {
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto raman_drive = node_raman_drive_;
  const auto raman_drives = problem_.raman_drives();
  const auto window = time_window_;
  const auto carrier = carrier_;
  const auto kerr_density = stage_kerr_density_;
  const auto n_to = problem_.dims.n_to;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double harmonic_factor = problem_.kerr.harmonic_factor;
  if constexpr (Plasma)
    device_trapezoid_plasma<Family, Components, Stash>(stage, first_node, node_count);
  launch::parallel_for(
      Plasma ? "solprop_native.response.device_pointwise_after_plasma"
             : "solprop_native.response.device_pointwise",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        capture_anchor(self_factor, conjugate_factor, node_real, node_complex, out_real,
                       out_complex, raman_drive, raman_drives, window, carrier,
                       kerr_density, kerr_coefficient, harmonic_factor, stage);
        const std::size_t base = node * Components;
        const auto value_at = [&](const std::size_t row) {
          if constexpr (Family == FieldFamily::field_real)
            return Complex(node_real(row, time), 0.0);
          else
            return Complex(node_complex(row, time));
        };
        const Complex first_value = value_at(base);
        Complex second_value{};
        if constexpr (Components == 2)
          second_value = value_at(base + 1);
        const double modulus = squared_abs(first_value) +
                               (Components == 2 ? squared_abs(second_value) : 0.0);
        const Complex square =
            first_value * first_value +
            (Components == 2 ? second_value * second_value : Complex{});
        const double factor = kerr_coefficient * kerr_density(stage);
        Complex first{};
        Complex second{};
        if constexpr (Nonlinear) {
          if constexpr (Family == FieldFamily::field_real) {
            first += Complex(factor * modulus * first_value.real(), 0.0);
            if constexpr (Components == 2)
              second += Complex(factor * modulus * second_value.real(), 0.0);
          } else {
            const auto analytic = [&](const Complex value) {
              const double weighted_real = conjugate_factor * square.real();
              const double weighted_imag = conjugate_factor * square.imag();
              const double weighted_self = self_factor * modulus;
              Complex result(
                  factor *
                      (weighted_self * value.real() +
                       (weighted_real * value.real() + weighted_imag * value.imag())),
                  factor *
                      (weighted_self * value.imag() +
                       (weighted_imag * value.real() - weighted_real * value.imag())));
              if constexpr (Harmonic)
                result += ((factor * harmonic_factor) * carrier(time)) * square * value;
              return result;
            };
            first += analytic(first_value);
            if constexpr (Components == 2)
              second += analytic(second_value);
          }
        }
        if constexpr (Raman)
          write_raman_drive<Components, RamanThg>(raman_drive, node, time, first_value,
                                                  second_value, raman_drives);
        if constexpr (Family != FieldFamily::envelope) {
          out_real(base, time) = Raman ? first.real() : first.real() * window(time);
          if constexpr (Components == 2)
            out_real(base + 1, time) =
                Raman ? second.real() : second.real() * window(time);
        } else {
          out_complex(base, time) = Raman ? first : first * window(time);
          if constexpr (Components == 2)
            out_complex(base + 1, time) = Raman ? second : second * window(time);
        }
      });
}

// The trapezoid plasma's two-level scans on a device (`device_response`). Rows of the
// rate, exposure and fraction views are the chunk's nodes; rows of the integrand and
// of the tile totals are its component rows.
template <FieldFamily Family, std::size_t Components, bool Stash>
void GeneralWorkspace::device_trapezoid_plasma(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto held_current = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto fractions = plasma_fraction_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = plasma_tile_totals_;
  const auto stash = stash_candidate_;
  const auto rate_coefficients = rate_coefficients_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const auto n_to = problem_.dims.n_to;
  const auto samples = static_cast<int>(n_to);
  const auto width = static_cast<int>(plasma_tile_samples);
  const auto tiles = static_cast<int>(plasma_tiles_);
  const auto last = n_to - 1;
  const bool stashing =
      Stash && !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const std::size_t rows = node_count * Components;
  const auto team_size = static_cast<int>(response_team_size());
  // The field a row reads: a real row's value, or an analytic or envelope row's real
  // part.
  const auto raw = [=] KOKKOS_FUNCTION(const std::size_t row, const std::size_t time) {
    capture_anchor(node_real, node_complex);
    if constexpr (Family == FieldFamily::field_real)
      return node_real(row, time);
    else
      return node_complex(row, time).real();
  };

  // 1. The rates.
  launch::parallel_for(
      "solprop_native.response.device_plasma_rate",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const std::size_t base = node * Components;
        const double first = raw(base, time);
        double magnitude = 0.0;
        if constexpr (Components == 2)
          magnitude = Kokkos::hypot(first, raw(base + 1, time));
        else
          magnitude = Kokkos::abs(first);
        rates(node, time) =
            ionisation_rate(magnitude, rate_coefficients, log_field_min, d_log_field,
                            field_min, field_max, rate_max, rate_intervals);
      });
  // 2. The exposure: tile totals, then each tile's prefixes from its offset, and the
  // fraction from each.
  const auto exposure_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, samples, width, tiles, dt);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(node, static_cast<std::size_t>(tile)) = total; });
  };
  launch::parallel_for("solprop_native.response.device_plasma_exposure_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_local);
  const auto exposure_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(rates, totals, exposures, fractions, preionisation, samples, width,
                   tiles, dt, stage);
    const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, node, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) { return rates(node, static_cast<std::size_t>(sample)); },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          exposures(node, time) = prefix;
          fractions(node, time) = ionised_fraction(prefix, preionisation(stage));
        });
  };
  launch::parallel_for("solprop_native.response.device_plasma_exposure",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(node_count) * tiles, team_size),
                       exposure_finish);
  // 3. The free-current integrand, per component row.
  launch::parallel_for(
      "solprop_native.response.device_plasma_integrand",
      Range2(response_space_for(node_count), {0, 0}, {rows, n_to}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        integrands(row, time) =
            fractions(row / Components, time) * charge_mass_ratio * raw(row, time);
      });
  // 4. The free current: tile totals, then each tile's prefixes from its offset and
  // the current from each.
  const auto current_local = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, samples, width, tiles, dt);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double total = team_trapezoid_total(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt);
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(tile)) = total; });
  };
  launch::parallel_for("solprop_native.response.device_plasma_current_tiles",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_local);
  const auto current_finish = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(integrands, totals, held_current, rates, fractions, density, samples,
                   width, tiles, dt, ionisation_potential, stage);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int tile = team.league_rank() % tiles;
    const int first = tile * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, tile);
    team_trapezoid_finish(
        team,
        [&](const int sample) {
          return integrands(row, static_cast<std::size_t>(sample));
        },
        first, end, dt, offset,
        [&](const int sample, const double prefix) {
          const auto time = static_cast<std::size_t>(sample);
          const std::size_t node = row / Components;
          const std::size_t base = node * Components;
          const double first_field = raw(base, time);
          const double second_field = Components == 2 ? raw(base + 1, time) : 0.0;
          const double magnitude_squared =
              first_field * first_field + second_field * second_field;
          const double rate = rates(node, time);
          const double guard = rate > 0.0 && magnitude_squared > 0.0
                                   ? ionisation_potential * rate *
                                         (1.0 - fractions(node, time)) /
                                         magnitude_squared
                                   : 0.0;
          held_current(row, time) = density(stage) * (prefix + guard * raw(row, time));
        });
  };
  launch::parallel_for("solprop_native.response.device_plasma_current",
                       TeamPolicy(response_space_for(node_count),
                                  static_cast<int>(rows) * tiles, team_size),
                       current_finish);
  // 5. The stash: the last sample's fraction and the largest rate, per node.
  if (stashing) {
    const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(rates, fractions, stash, samples, last, first_node);
      const auto node = static_cast<std::size_t>(team.league_rank());
      double largest = 0.0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, samples),
          [&](const int sample, double& update) {
            const double rate = rates(node, static_cast<std::size_t>(sample));
            if (rate > update)
              update = rate;
          },
          Kokkos::Max<double>(largest));
      Kokkos::single(Kokkos::PerTeam(team), [&] {
        stash(first_node + node, 0) = fractions(node, last);
        stash(first_node + node, 1) = largest;
      });
    };
    launch::parallel_for("solprop_native.response.device_plasma_stash",
                         TeamPolicy(response_space_for(node_count),
                                    static_cast<int>(node_count), team_size),
                         peak);
  }
}
#endif

static_assert(band_max_half + 1 == static_cast<int>(plasma_max_taps),
              "the team band filter's tap bound is the problem's");

namespace {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
/** Whether the band column skips each row's quiet samples (`band_support`): host only.
 */
constexpr bool band_windows = false;
#else
constexpr bool band_windows = true;
/**
 * The fewest samples a host thread of a band team takes from a row: below it a thread's
 * share of a pass no longer repays the team's barriers between the passes. Measured on
 * the mode-averaged helium RDW case (n_to = 8192, one node) on a loaded Apple M-series
 * host: teams of two and four threads ran the band column faster than one, six slower;
 * 2048 caps that row at four. To be re-measured on each host class (solprop SEP 25).
 */
constexpr std::size_t band_host_samples_per_thread = 2048;
#endif

/** The first and last indices whose value is non-zero, as a Kokkos MinMax reduction. */
using BandLive = Kokkos::MinMaxScalar<int>;

/** The band window of a row whose non-zero samples `live` found. */
KOKKOS_INLINE_FUNCTION BandWindow band_window_of(const BandLive& live,
                                                 const int samples, const int half) {
  if constexpr (band_windows)
    return band_support(live.min_val, live.max_val, samples, half);
  else
    return {.first = 0, .last = samples - 1};
}

/** Count `index` into `live` when `value` is not exactly zero. */
KOKKOS_INLINE_FUNCTION void band_note(BandLive& live, const int index,
                                      const double value) {
  if (value != 0.0) {
    live.min_val = Kokkos::min(live.min_val, index);
    live.max_val = Kokkos::max(live.max_val, index);
  }
}

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
/**
 * Count a host block's non-zero values into `live`, as `band_note` of each:
 * `values[k]` is the sample `offset + k`. Only the first and the last non-zero value
 * matter, so the block is searched from either end.
 */
inline void band_note_block(BandLive& live, const double* values,
                            const std::size_t count, const int offset) {
  std::size_t low = 0;
  while (low < count && values[low] == 0.0)
    ++low;
  if (low == count)
    return;
  std::size_t high = count - 1;
  while (values[high] == 0.0)
    --high;
  live.min_val = std::min(live.min_val, offset + static_cast<int>(low));
  live.max_val = std::max(live.max_val, offset + static_cast<int>(high));
}

/**
 * `ionisation_rate` of a host block's `count <= rate_block` magnitudes into `row`,
 * with the table's limits `lower`, `upper` and `highest` as it takes them (a NaN
 * magnitude is live); `within(values, live)` runs `ionisation_rates_within` on the
 * live ones. A mostly live block runs every sample through the layer, a quiet one's
 * magnitude replaced by the lower limit, and then keeps the limits' values: each value
 * is the same function of its own magnitude, so the passes' lanes do not change it,
 * and no sample is gathered. A mostly quiet block gathers its live magnitudes.
 */
template <class Within>
void band_rates_block(const double* magnitudes, double* row, const std::size_t count,
                      const double lower, const double upper, const double highest,
                      const Within& within) {
  double values[rate_block];
  std::size_t live = 0;
  for (std::size_t k = 0; k < count; ++k)
    live += ((magnitudes[k] > upper) | (magnitudes[k] < lower)) ? 0 : 1;
  if (2 * live >= count && lower > 0.0) {
    for (std::size_t k = 0; k < count; ++k) {
      const double magnitude = magnitudes[k];
      values[k] = ((magnitude > upper) | (magnitude < lower)) ? lower : magnitude;
    }
    within(values, count);
    for (std::size_t k = 0; k < count; ++k) {
      const double magnitude = magnitudes[k];
      row[k] = magnitude > upper ? highest : magnitude < lower ? 0.0 : values[k];
    }
    return;
  }
  if (live == 0) {
    // A quiet block takes only the limits.
    for (std::size_t k = 0; k < count; ++k)
      row[k] = magnitudes[k] > upper ? highest : 0.0;
    return;
  }
  std::size_t active[rate_block];
  std::size_t gathered = 0;
  for (std::size_t k = 0; k < count; ++k) {
    const double magnitude = magnitudes[k];
    row[k] = magnitude > upper ? highest : 0.0;
    if (!(magnitude > upper) && !(magnitude < lower)) {
      values[gathered] = magnitude;
      active[gathered] = k;
      ++gathered;
    }
  }
  within(values, gathered);
  for (std::size_t k = 0; k < gathered; ++k)
    row[active[k]] = values[k];
}
#endif
} // namespace

int GeneralWorkspace::band_host_team_size(const std::size_t node_count) const {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(node_count);
  return 1;
#else
  // solprop SEP 25 section 6.7.5: share the threads between the chunk's nodes, so a
  // chunk of fewer nodes than threads runs each node's passes over chunks of t.
  const auto threads = static_cast<std::size_t>(
      std::max(1, response_space_for(node_count).concurrency()));
  const auto per_node =
      std::max<std::size_t>(1, threads / std::max<std::size_t>(1, node_count));
  const auto by_samples =
      std::max<std::size_t>(1, problem_.dims.n_to / band_host_samples_per_thread);
  return static_cast<int>(std::min(per_node, by_samples));
#endif
}

std::size_t GeneralWorkspace::band_league(const std::size_t node_count) const {
  return build::device ? node_count : std::min(node_count, band_slots_);
}

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::band_team_policy(const std::size_t node_count,
                                   const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (node_count > policy_limit || problem_.dims.n_to > policy_limit)
    throw std::length_error("band team column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(band_league(node_count));
  const auto scratch =
      Kokkos::PerTeam(band_staging_bytes<band_staging, ExecutionSpace>());
  // The widest team this kernel can launch with its scratch: a device's registers per
  // thread can hold the band column below the configured width.
  Policy probe(response_space_for(node_count), league, 1);
  probe.set_scratch_size(0, scratch);
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int requested = static_cast<int>(response_team_size());
#else
  const int requested = band_host_team_size(node_count);
#endif
  Policy policy(response_space_for(node_count), league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, scratch);
  return policy;
}

// solprop SEP 18's band rule as a team column in whole-row passes (solprop SEP 25
// section 6.7.4; include/solprop_native/response/plasma.hpp): per channel the rates,
// their filter and scan give the exposure; the fraction (floored where it is formed),
// the loss guard and the free-current weight follow pointwise, a mixture adding each
// channel's share in channel order. Per component the free-current integrand is
// filtered and scanned, and the scan's store forms the current from it, the guard and
// the field. SEP 27 uses this current directly, without a third running integral.
//
// The rows, by slot (a team's own; the node on a device): `plasma_rate_` the raw rates,
// `plasma_fraction_` the filtered row, `plasma_exposure_` the exposure,
// `plasma_free_current_` the loss guard (a mixture's work until the last channel),
// `plasma_outer_current_` a mixture's weight, `plasma_current_integrand_` the
// free-current integrand. The separate `held_current_` stores the total current. Each
// filter and scan runs on its row's live window (`band_support`) and writes the rest:
// zero before it, the final prefix after it, and an unchanged exposure's fraction.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
// NOLINTNEXTLINE(readability-function-size): hot path, split pending an A/B
void GeneralWorkspace::team_band_plasma_column(const std::size_t stage,
                                               const std::size_t first_node,
                                               const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      team_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                  node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  // A folded single node reads the modal field through its one-element synthesis
  // (`folds_single_node`).
  constexpr bool folded = Folded;
  const auto node_real = folded ? modal_real_ : node_real_;
  const auto node_complex = folded ? modal_complex_ : node_complex_;
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const auto rate_coefficients = rate_coefficients_;
  const auto channel_parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const double log_field_min = problem_.plasma.log_field_min;
  const double d_log_field = problem_.plasma.d_log_field;
  const double field_min = problem_.plasma.field_min;
  const double field_max = problem_.plasma.field_max;
  const double rate_max = problem_.plasma.rate_max;
  const auto rate_intervals = problem_.plasma.rate_intervals;
  const std::size_t channels = Multiple ? problem_.plasma.n_channels : 1;
  const auto samples = static_cast<int>(problem_.dims.n_to);
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  const std::size_t sample_count = problem_.dims.n_to;
#endif
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const std::size_t last = problem_.dims.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto column = KOKKOS_LAMBDA(const TeamMember& team) {
    // CUDA extended-lambda restriction 12 requires the first capture outside
    // if-constexpr blocks.
    capture_anchor(node_real, node_complex, out, rates, filtered, exposures, guards,
                   weights, integrands, taps_view, stash, stashing, rate_coefficients,
                   channel_parameters, channel_coefficients, channel_density,
                   channel_preionisation, density, preionisation, charge_mass_ratio,
                   ionisation_potential, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, rate_intervals, channels, samples, half, last,
                   first_node, stage, league, synthesis, taper);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    const std::size_t own = slot * Components;
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    const double input_scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    for (std::size_t node = slot; node < node_count; node += league) {
      const std::size_t base = node * Components;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        if constexpr (Family == FieldFamily::field_real)
          return input_scale * node_real(row, time);
        else
          return input_scale * node_complex(row, time).real();
      };
      const auto magnitude_at = [&](const std::size_t time) {
        const double first = field_at(base, time);
        if constexpr (Components == 2)
          return Kokkos::hypot(first, field_at(base + 1, time));
        else
          return Kokkos::abs(first);
      };
      [[maybe_unused]] const auto rate_of = [&](const double magnitude,
                                                const std::size_t channel) {
        if constexpr (Multiple)
          return ionisation_rate(
              magnitude, channel_coefficients, channel_parameters(channel, 0),
              channel_parameters(channel, 1), channel_parameters(channel, 2),
              channel_parameters(channel, 3), channel_parameters(channel, 4),
              static_cast<std::size_t>(channel_parameters(channel, 5)),
              static_cast<std::size_t>(channel_parameters(channel, 6)));
        else
          return ionisation_rate(magnitude, rate_coefficients, log_field_min,
                                 d_log_field, field_min, field_max, rate_max,
                                 rate_intervals);
      };
      BandLive integrand_live;
      integrand_live.min_val = samples;
      integrand_live.max_val = -1;

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
      // With vector maths a host team thread takes blocks of samples and gathers the
      // live ones' transcendentals into passes of the layer (`ionisation_rates_within`,
      // the decays below), writing the rest as it goes; with the system libm each
      // sample keeps its own calls, as on a device.
      [[maybe_unused]] const auto blocks =
          static_cast<int>((sample_count + rate_block - 1) / rate_block);
      [[maybe_unused]] const auto rates_into =
          [&](const std::size_t first, const std::size_t count,
              const std::size_t channel, BandLive& update) {
            const double lower = Multiple ? channel_parameters(channel, 2) : field_min;
            const double upper = Multiple ? channel_parameters(channel, 3) : field_max;
            const double highest = Multiple ? channel_parameters(channel, 4) : rate_max;
            const auto within = [&](double* values, const std::size_t live) {
              if constexpr (Multiple)
                ionisation_rates_within(
                    values, live, channel_coefficients, channel_parameters(channel, 0),
                    channel_parameters(channel, 1),
                    static_cast<std::size_t>(channel_parameters(channel, 5)),
                    static_cast<std::size_t>(channel_parameters(channel, 6)));
              else
                ionisation_rates_within(values, live, rate_coefficients, log_field_min,
                                        d_log_field, rate_intervals);
            };
            double magnitudes[rate_block];
            for (std::size_t k = 0; k < count; ++k)
              magnitudes[k] = magnitude_at(first + k);
            double* const row = &rates(slot, first);
            band_rates_block(magnitudes, row, count, lower, upper, highest, within);
            band_note_block(update, row, count, static_cast<int>(first));
          };
#endif

      // 1. Per channel: rates -> filter -> scan -> exposure, then the fraction, the
      // guard and the weight pointwise.
      for (std::size_t channel = 0; channel < channels; ++channel) {
        BandLive rate_live;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        if constexpr (vmath::vectorised) {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, blocks),
              [&](const int block, BandLive& update) {
                const auto first = static_cast<std::size_t>(block) * rate_block;
                rates_into(first, std::min(rate_block, sample_count - first), channel,
                           update);
              },
              Kokkos::MinMax<int>(rate_live));
        } else
#endif
        {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, BandLive& update) {
                const auto time = static_cast<std::size_t>(sample);
                const double rate = rate_of(magnitude_at(time), channel);
                rates(slot, time) = rate;
                band_note(update, sample, rate);
              },
              Kokkos::MinMax<int>(rate_live));
        }
        team.team_barrier();
        const BandWindow window = band_window_of(rate_live, samples, half);
        if (window.first <= window.last)
          team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot,
                                         samples, taps, half, window.first,
                                         window.last + 1);
        const double total =
            team_trapezoid_window(team, filtered, slot, window, samples, dt,
                                  [&](const int sample, const double prefix) {
                                    exposures(slot, static_cast<std::size_t>(sample)) =
                                        prefix;
                                  });
        const double initial =
            Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
        // Outside [window.first, window.last + 1] the exposure is 0 before and `total`
        // after, so the fraction there is one of two values (band_support).
        const int changing_end = Kokkos::min(window.last + 2, samples);
        const double before = band_fraction(0.0, initial);
        const double after = band_fraction(total, initial);
        [[maybe_unused]] const auto fraction_at = [&](const int sample) {
          if (sample < window.first)
            return before;
          if (sample >= changing_end)
            return after;
          return band_fraction(exposures(slot, static_cast<std::size_t>(sample)),
                               initial);
        };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        // A host block's fractions with vector maths, each handed to `apply` with its
        // sample: the changing samples' decays `exp(-max(X, 0))` in one pass, each
        // fraction then `band_fraction`'s strict arithmetic.
        [[maybe_unused]] const auto fractions_into =
            [&](const std::size_t first, const std::size_t count, const auto& apply) {
              double decay[rate_block];
              std::size_t active[rate_block];
              std::size_t live = 0;
              for (std::size_t time = first; time < first + count; ++time) {
                const auto sample = static_cast<int>(time);
                if (sample < window.first) {
                  apply(time, before);
                } else if (sample >= changing_end) {
                  apply(time, after);
                } else {
                  const double exposure = exposures(slot, time);
                  decay[live] = -(exposure < 0.0 ? 0.0 : exposure);
                  active[live] = time;
                  ++live;
                }
              }
              vmath::exp(decay, decay, live);
              for (std::size_t k = 0; k < live; ++k)
                apply(active[k], decayed_fraction(decay[k], initial));
            };
        // A single gas's host block of free-current integrands and loss guards as
        // loops over the block's samples: the fractions as `fractions_into` forms
        // them (the changing samples' decays in one pass, `decayed_fractions` in
        // another), then `band_loss_scale` and the weights with their branch taken
        // as a selection, and the integrand rows' live samples searched afterwards.
        [[maybe_unused]] const auto integrands_into = [&](const std::size_t first,
                                                          const std::size_t count,
                                                          BandLive& update) {
          double fraction[rate_block];
          const auto start = static_cast<int>(first);
          const auto stop = static_cast<int>(first + count);
          for (std::size_t k = 0; k < count; ++k)
            fraction[k] = start + static_cast<int>(k) < window.first ? before : after;
          const int begin = Kokkos::max(window.first, start);
          const int end = Kokkos::min(changing_end, stop);
          if (begin < end) {
            double* const changing = fraction + (begin - start);
            const auto changing_count = static_cast<std::size_t>(end - begin);
            const double* const exposure =
                &exposures(slot, static_cast<std::size_t>(begin));
            for (std::size_t k = 0; k < changing_count; ++k)
              changing[k] = -(exposure[k] < 0.0 ? 0.0 : exposure[k]);
            vmath::exp(changing, changing, changing_count);
            decayed_fractions(changing, changing_count, initial);
          }
          const double* const rate_row = &rates(slot, first);
          double* const guard_row = &guards(slot, first);
          for (std::size_t k = 0; k < count; ++k) {
            const std::size_t time = first + k;
            const double magnitude = magnitude_at(time);
            const double squared = magnitude * magnitude;
            const double rate = rate_row[k];
            const bool loses = rate > 0.0 && squared > 0.0;
            const double scale = ionisation_potential * rate * (1.0 - fraction[k]) /
                                 (loses ? squared : 1.0);
            guard_row[k] = loses ? scale : 0.0;
            const double weight = fraction[k] * charge_mass_ratio;
            for (std::size_t component = 0; component < Components; ++component)
              integrands(own + component, time) =
                  weight * field_at(base + component, time);
          }
          for (std::size_t component = 0; component < Components; ++component)
            band_note_block(update, &integrands(own + component, first), count, start);
        };
#endif
        if constexpr (Multiple) {
          const double partial = channel_density(stage, channel);
          const double charge = channel_parameters(channel, 7);
          const double potential = channel_parameters(channel, 8);
          const auto share_at = [&](const std::size_t time, const double fraction) {
            const double rate = rates(slot, time);
            const double share = partial * fraction * charge;
            const double paid = partial * potential * rate * (1.0 - fraction);
            const double weight = channel == 0 ? 0.0 : weights(slot, time);
            const double work = channel == 0 ? 0.0 : guards(slot, time);
            weights(slot, time) = weight + share;
            guards(slot, time) = work + paid;
          };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          if constexpr (vmath::vectorised) {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                  const auto first = static_cast<std::size_t>(block) * rate_block;
                  fractions_into(first, std::min(rate_block, sample_count - first),
                                 share_at);
                });
          } else
#endif
          {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                  share_at(static_cast<std::size_t>(sample), fraction_at(sample));
                });
          }
        } else {
          const auto integrand_at = [&](const std::size_t time, const double fraction,
                                        BandLive& update) {
            const double magnitude = magnitude_at(time);
            const double squared = magnitude * magnitude;
            guards(slot, time) = band_loss_scale(rates(slot, time), fraction, squared,
                                                 ionisation_potential);
            const double weight = fraction * charge_mass_ratio;
            for (std::size_t component = 0; component < Components; ++component) {
              const double value = weight * field_at(base + component, time);
              integrands(own + component, time) = value;
              band_note(update, static_cast<int>(time), value);
            }
          };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          if constexpr (vmath::vectorised) {
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, blocks),
                [&](const int block, BandLive& update) {
                  const auto first = static_cast<std::size_t>(block) * rate_block;
                  integrands_into(first, std::min(rate_block, sample_count - first),
                                  update);
                },
                Kokkos::MinMax<int>(integrand_live));
          } else
#endif
          {
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, samples),
                [&](const int sample, BandLive& update) {
                  integrand_at(static_cast<std::size_t>(sample), fraction_at(sample),
                               update);
                },
                Kokkos::MinMax<int>(integrand_live));
          }
        }
        if (stashing) {
          double largest = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, samples),
              [&](const int sample, double& update) {
                const double rate = rates(slot, static_cast<std::size_t>(sample));
                if (rate > update)
                  update = rate;
              },
              Kokkos::Max<double>(largest));
          Kokkos::single(Kokkos::PerTeam(team), [&] {
            stash(first_node + node, 2 * channel) =
                band_fraction(exposures(slot, last), initial);
            stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
          });
        }
        team.team_barrier();
      }
      if constexpr (Multiple) {
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, BandLive& update) {
              const auto time = static_cast<std::size_t>(sample);
              const double magnitude = magnitude_at(time);
              const double squared = magnitude * magnitude;
              guards(slot, time) =
                  band_loss_scale(1.0, 0.0, squared, guards(slot, time));
              const double weight = weights(slot, time);
              for (std::size_t component = 0; component < Components; ++component) {
                const double value = weight * field_at(base + component, time);
                integrands(own + component, time) = value;
                band_note(update, sample, value);
              }
            },
            Kokkos::MinMax<int>(integrand_live));
      }
      team.team_barrier();

      // 2. Per component: the free current, and from it the current.
      const BandWindow free_window = band_window_of(integrand_live, samples, half);
      // A single gas's density, read once: inside the store below the compiler must
      // reload it after each store to the current, which might alias it.
      const double stage_density = Multiple ? 0.0 : density(stage);
      for (std::size_t component = 0; component < Components; ++component) {
        const std::size_t row = base + component;
        if (free_window.first <= free_window.last)
          team_band_filter<band_staging>(team, tile, integrands, own + component,
                                         filtered, slot, samples, taps, half,
                                         free_window.first, free_window.last + 1);
        team_trapezoid_window(team, filtered, slot, free_window, samples, dt,
                              [&](const int sample, const double prefix) {
                                const auto time = static_cast<std::size_t>(sample);
                                const double drive =
                                    guards(slot, time) * field_at(row, time);
                                out(row, time) = Multiple
                                                     ? prefix + drive
                                                     : stage_density * (prefix + drive);
                              });
      }
    }
  };
  launch::parallel_for("solprop_native.response.team_band_plasma_column",
                       band_team_policy(node_count, column), column);
}

// BandField and BandRate are not in an anonymous namespace: the band columns' lambdas
// capture them, and GCC's -Wsubobject-linkage treats a type with internal linkage
// captured in an included file (this fragment of trial.cpp) as a header's ODR hazard.

/**
 * A band column's node field: `scale` times the real row of a field-family node (or of
 * the folded single node's modal row), and its magnitude over a node's components.
 */
// NOLINTNEXTLINE(misc-use-internal-linkage): internal linkage trips GCC's -Wsubobject-linkage
template <FieldFamily Family, std::size_t Components> struct BandField {
  RealView2 real;
  ComplexView2 complex;
  KOKKOS_INLINE_FUNCTION double at(const std::size_t row, const std::size_t time,
                                   const double scale) const {
    if constexpr (Family == FieldFamily::field_real)
      return scale * real(row, time);
    else
      return scale * complex(row, time).real();
  }
  KOKKOS_INLINE_FUNCTION double
  magnitude(const std::size_t base, const std::size_t time, const double scale) const {
    const double first = at(base, time, scale);
    if constexpr (Components == 2)
      return Kokkos::hypot(first, at(base + 1, time, scale));
    else
      return Kokkos::abs(first);
  }
};

/** A band column's ionisation rate: the single gas's table, or a mixture channel's. */
// NOLINTNEXTLINE(misc-use-internal-linkage): internal linkage trips GCC's -Wsubobject-linkage
template <bool Multiple> struct BandRate {
  RealView coefficients;
  RealView2 parameters;
  RealView channel_coefficients;
  double log_field_min{}, d_log_field{}, field_min{}, field_max{}, rate_max{};
  std::size_t intervals{};
  KOKKOS_INLINE_FUNCTION double operator()(const double magnitude,
                                           const std::size_t channel) const {
    if constexpr (Multiple)
      return ionisation_rate(magnitude, channel_coefficients, parameters(channel, 0),
                             parameters(channel, 1), parameters(channel, 2),
                             parameters(channel, 3), parameters(channel, 4),
                             static_cast<std::size_t>(parameters(channel, 5)),
                             static_cast<std::size_t>(parameters(channel, 6)));
    else
      return ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                             field_min, field_max, rate_max, intervals);
  }
};

template <class Functor>
Kokkos::TeamPolicy<ExecutionSpace>
GeneralWorkspace::tiled_band_policy(const std::size_t rows, const bool scratch,
                                    const Functor& functor) const {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  constexpr auto policy_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (band_tile_count_ == 0 || rows > policy_limit / band_tile_count_ ||
      problem_.dims.n_to > policy_limit)
    throw std::length_error("tiled band column extent exceeds signed 32-bit range");
  const auto league = static_cast<int>(rows * band_tile_count_);
  const auto bytes = scratch ? band_tile_bytes<ExecutionSpace>(band_tile_) : 0;
  Policy probe(response_execution_space_, league, 1);
  probe.set_scratch_size(0, Kokkos::PerTeam(bytes));
  const int widest = probe.team_size_max(functor, Kokkos::ParallelForTag());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto requested = static_cast<int>(std::min(response_team_size(), band_tile_));
#else
  const int requested = std::max(1, band_tile_threads_);
#endif
  Policy policy(response_execution_space_, league,
                std::max(1, std::min(requested, widest)));
  policy.set_scratch_size(0, Kokkos::PerTeam(bytes));
  return policy;
}

// The band column in tiles (solprop SEP 25 section 6.7.6;
// include/solprop_native/response/plasma.hpp): each running integral is a local kernel
// of one team per (row, tile), which loads, filters and scans its tile from zero
// (`team_band_tile`), then a finish kernel of one team per (row, tile), which adds the
// row's earlier tile totals (`team_tile_offset`) and forms the pass's product from the
// prefix, in the team column's arithmetic:
//
//   1. per channel: the local kernel forms the rates in its window (keeping its own
//      tile's in `plasma_rate_`) and the exposure's local prefixes
//      (`plasma_exposure_`); the finish forms the fraction, the loss guard
//      (`plasma_free_current_`) and the free-current integrand
//      (`plasma_current_integrand_`), a mixture accumulating its weight
//      (`plasma_outer_current_`) and work channel by channel;
//   2. per component row: the integrand's local prefixes (`plasma_outer_current_`,
//      free after the first pass), then the current (`held_current_`).
//
// Rows are by node within the chunk, the totals by row (`band_tile_totals_`). The
// stash's fraction comes from the last sample's finish and its largest rate from a
// reduction over the rate row. No window is skipped: every sample is formed.
template <FieldFamily Family, std::size_t Components, bool Multiple, bool Folded>
void GeneralWorkspace::tiled_band_plasma_column(const std::size_t stage,
                                                const std::size_t first_node,
                                                const std::size_t node_count) {
  if constexpr (Components == 1 && !Folded) {
    if (folds_single_node()) {
      tiled_band_plasma_column<Family, Components, Multiple, true>(stage, first_node,
                                                                   node_count);
      return;
    }
  }
  static_assert(Family != FieldFamily::envelope,
                "the band rule integrates a field-grid plasma column");
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  constexpr bool folded = Folded;
  const BandField<Family, Components> field{folded ? modal_real_ : node_real_,
                                            folded ? modal_complex_ : node_complex_};
  const BandRate<Multiple> rate{rate_coefficients_,
                                channel_parameters_,
                                channel_coefficients_,
                                problem_.plasma.log_field_min,
                                problem_.plasma.d_log_field,
                                problem_.plasma.field_min,
                                problem_.plasma.field_max,
                                problem_.plasma.rate_max,
                                problem_.plasma.rate_intervals};
  const auto synthesis = synthesis_;
  const auto taper = stage_taper_;
  const auto out = held_current_;
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  const auto guards = plasma_free_current_;
  const auto weights = plasma_outer_current_;
  const auto locals = plasma_outer_current_;
  const auto integrands = plasma_current_integrand_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const auto stash = stash_candidate_;
  const bool stashing = !stash.empty() && stage + 1 == problem_.stepper.n_stages;
  const auto channel_parameters = channel_parameters_;
  const auto channel_density = stage_channel_density_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto density = stage_plasma_density_;
  const auto preionisation = stage_preionisation_;
  const double charge_mass_ratio = problem_.plasma.charge_mass_ratio;
  const double ionisation_potential = problem_.plasma.ionisation_potential;
  const double dt = problem_.grid.dt_fine;
  const std::size_t channels = Multiple ? problem_.plasma.n_channels : 1;
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const int last = samples - 1;
  const std::size_t rows = node_count * Components;

  // 1. Per channel: the exposure, then the fraction, the guard and the integrand.
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rate, rates, exposures, totals, taps_view, synthesis, taper,
                     stage, channel, samples, half, width, tiles, dt);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      double taps[plasma_max_taps] = {};
      for (int k = 0; k <= half; ++k)
        taps[k] = taps_view(static_cast<std::size_t>(k));
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double total = team_band_tile(
          team,
          [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double value = rate(field.magnitude(base, time, scale), channel);
            if (sample >= first && sample < end)
              rates(node, time) = value;
            return value;
          },
          first, end, samples, taps, half, dt,
          [&](const int sample, const double prefix) {
            exposures(node, static_cast<std::size_t>(sample)) = prefix;
          });
      Kokkos::single(Kokkos::PerTeam(team),
                     [&] { totals(node, static_cast<std::size_t>(index)) = total; });
    };
    launch::parallel_for("solprop_native.response.tiled_band_exposure",
                         tiled_band_policy(node_count, true, local), local);
    const auto finish = KOKKOS_LAMBDA(const TeamMember& team) {
      capture_anchor(field, rates, exposures, guards, weights, integrands, totals,
                     stash, stashing, synthesis, taper, channel_parameters,
                     channel_density, channel_preionisation, preionisation,
                     charge_mass_ratio, ionisation_potential, channels, channel,
                     samples, width, tiles, last, first_node, stage);
      const auto node = static_cast<std::size_t>(team.league_rank() / tiles);
      const int index = team.league_rank() % tiles;
      const int first = index * width;
      const int end = Kokkos::min(first + width, samples);
      const double offset = team_tile_offset(team, totals, node, index);
      const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
      const std::size_t base = node * Components;
      const double initial =
          Multiple ? channel_preionisation(stage, channel) : preionisation(stage);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction =
                band_fraction(offset + exposures(node, time), initial);
            const double value = rates(node, time);
            if constexpr (Multiple) {
              const double partial = channel_density(stage, channel);
              const double charge = channel_parameters(channel, 7);
              const double potential = channel_parameters(channel, 8);
              const double share = partial * fraction * charge;
              const double paid = partial * potential * value * (1.0 - fraction);
              const double weight = (channel == 0 ? 0.0 : weights(node, time)) + share;
              const double work = (channel == 0 ? 0.0 : guards(node, time)) + paid;
              if (channel + 1 == channels) {
                const double magnitude = field.magnitude(base, time, scale);
                guards(node, time) =
                    band_loss_scale(1.0, 0.0, magnitude * magnitude, work);
                for (std::size_t component = 0; component < Components; ++component)
                  integrands(base + component, time) =
                      weight * field.at(base + component, time, scale);
              } else {
                weights(node, time) = weight;
                guards(node, time) = work;
              }
            } else {
              const double magnitude = field.magnitude(base, time, scale);
              guards(node, time) = band_loss_scale(
                  value, fraction, magnitude * magnitude, ionisation_potential);
              const double weight = fraction * charge_mass_ratio;
              for (std::size_t component = 0; component < Components; ++component)
                integrands(base + component, time) =
                    weight * field.at(base + component, time, scale);
            }
            if (stashing && sample == last)
              stash(first_node + node, 2 * channel) = fraction;
          });
    };
    launch::parallel_for("solprop_native.response.tiled_band_fraction",
                         tiled_band_policy(node_count, false, finish), finish);
    if (stashing) {
      const auto peak = KOKKOS_LAMBDA(const TeamMember& team) {
        capture_anchor(rates, stash, channel, samples, first_node);
        const auto node = static_cast<std::size_t>(team.league_rank());
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double value = rates(node, static_cast<std::size_t>(sample));
              if (value > update)
                update = value;
            },
            Kokkos::Max<double>(largest));
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          stash(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
      };
      launch::parallel_for("solprop_native.response.tiled_band_peak_rate",
                           Kokkos::TeamPolicy<ExecutionSpace>(
                               response_space_for(node_count),
                               static_cast<int>(node_count), Kokkos::AUTO()),
                           peak);
    }
  }

  // 2. Per component row: the free current, followed by the pointwise loss current.
  tiled_band_local(integrands, rows, "solprop_native.response.tiled_band_free_current");
  const auto current = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(field, guards, locals, out, density, totals, synthesis, taper, stage,
                   samples, width, tiles);
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    const double offset = team_tile_offset(team, totals, row, index);
    const double scale = folded ? synthesis(0, 0) * taper(stage) : 1.0;
    const std::size_t node = row / Components;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, first, end), [&](const int sample) {
          const auto time = static_cast<std::size_t>(sample);
          const double prefix = offset + locals(row, time);
          const double drive = guards(node, time) * field.at(row, time, scale);
          out(row, time) =
              Multiple ? prefix + drive : density(stage) * (prefix + drive);
        });
  };
  launch::parallel_for("solprop_native.response.tiled_band_current",
                       tiled_band_policy(rows, false, current), current);
}

// The tiled column's local kernel over component rows: `source`'s band-filtered tile
// prefixes from zero into the free-current rows (`plasma_outer_current_`),
// and each tile's total into `band_tile_totals_`.
void GeneralWorkspace::tiled_band_local(const RealView2& source, const std::size_t rows,
                                        const char* label) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto locals = plasma_outer_current_;
  const auto totals = band_tile_totals_;
  const auto taps_view = plasma_taps_;
  const double dt = problem_.grid.dt_fine;
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const auto width = static_cast<int>(band_tile_);
  const auto tiles = static_cast<int>(band_tile_count_);
  const auto local = KOKKOS_LAMBDA(const TeamMember& team) {
    const auto row = static_cast<std::size_t>(team.league_rank() / tiles);
    const int index = team.league_rank() % tiles;
    const int first = index * width;
    const int end = Kokkos::min(first + width, samples);
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const double total = team_band_tile(
        team,
        [&](const int sample) { return source(row, static_cast<std::size_t>(sample)); },
        first, end, samples, taps, half, dt,
        [&](const int sample, const double prefix) {
          locals(row, static_cast<std::size_t>(sample)) = prefix;
        });
    Kokkos::single(Kokkos::PerTeam(team),
                   [&] { totals(row, static_cast<std::size_t>(index)) = total; });
  };
  launch::parallel_for(label, tiled_band_policy(rows, true, local), local);
}

// The band rule's statistics scan: per channel the rates, their band filter and
// trapezoid scan, then team reductions over the rows. The floored final exposure and
// the largest rate always; with `Diagnose` the six parts of SEP 18's run diagnostics,
// the largest over the channels: the exposure's largest dip and rise (maxima of the
// scan's increments), the largest |paid - freed| and freed (the paid work a sum of the
// trapezoid's increments of the raw rate's owed work), and the largest |f E| in the
// first 2K samples and anywhere. Maxima combine exactly; the sums associate as the
// team's scan does. The rows are the band column's (`plasma_rate_`,
// `plasma_fraction_` filtered and then owed work, `plasma_exposure_` exposure and then
// fraction), by slot as the column walks the nodes. Not a step's hot path: it runs
// whole rows, without the column's live windows.
template <bool Diagnose>
void GeneralWorkspace::team_band_statistics(const std::size_t first_node,
                                            const std::size_t node_count,
                                            const RealView2& output,
                                            const double preionisation) {
  using TeamMember = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto rates = plasma_rate_;
  const auto filtered = plasma_fraction_;
  const auto exposures = plasma_exposure_;
  const auto taps_view = plasma_taps_;
  const auto diagnostics = plasma_diagnostics_;
  const auto coefficients = rate_coefficients_;
  const auto parameters = channel_parameters_;
  const auto channel_coefficients = channel_coefficients_;
  const auto channel_preionisation = stage_channel_preionisation_;
  const auto channels = problem_.plasma.n_channels;
  const bool multiple = channels > 1;
  const auto n_pol = problem_.effective_node_pol();
  const auto samples = static_cast<int>(problem_.dims.n_to);
  const auto intervals = problem_.plasma.rate_intervals;
  const auto dt = problem_.grid.dt_fine;
  const auto log_field_min = problem_.plasma.log_field_min;
  const auto d_log_field = problem_.plasma.d_log_field;
  const auto field_min = problem_.plasma.field_min;
  const auto field_max = problem_.plasma.field_max;
  const auto rate_max = problem_.plasma.rate_max;
  const auto half = static_cast<int>(problem_.plasma.half_width());
  const int reach = Kokkos::min(2 * half, samples);
  const std::size_t last = problem_.dims.n_to - 1;
  const std::size_t league = band_league(node_count);
  const auto scan = KOKKOS_LAMBDA(const TeamMember& team) {
    capture_anchor(node_real, node_complex, real_family, rates, filtered, exposures,
                   taps_view, diagnostics, coefficients, parameters,
                   channel_coefficients, channel_preionisation, channels, multiple,
                   n_pol, samples, intervals, dt, log_field_min, d_log_field, field_min,
                   field_max, rate_max, half, reach, last, first_node, output,
                   preionisation, league);
    const auto slot = static_cast<std::size_t>(team.league_rank());
    double taps[plasma_max_taps] = {};
    for (int k = 0; k <= half; ++k)
      taps[k] = taps_view(static_cast<std::size_t>(k));
    const auto tile = band_tile<band_staging>(team);
    for (std::size_t node = slot; node < node_count; node += league) {
      const auto base = node * n_pol;
      const auto field_at = [&](const std::size_t row, const std::size_t time) {
        return real_family ? node_real(row, time) : node_complex(row, time).real();
      };
      double parts[plasma_diagnostic_parts] = {};
      for (std::size_t channel = 0; channel < channels; ++channel) {
        const double initial =
            multiple ? channel_preionisation(0, channel) : preionisation;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        // With vector maths a host thread forms blocks of rates as the column does
        // (`band_rates_block`).
        if constexpr (vmath::vectorised) {
          const double lower = multiple ? parameters(channel, 2) : field_min;
          const double upper = multiple ? parameters(channel, 3) : field_max;
          const double highest = multiple ? parameters(channel, 4) : rate_max;
          const auto within = [&](double* values, const std::size_t live) {
            if (multiple)
              ionisation_rates_within(values, live, channel_coefficients,
                                      parameters(channel, 0), parameters(channel, 1),
                                      static_cast<std::size_t>(parameters(channel, 5)),
                                      static_cast<std::size_t>(parameters(channel, 6)));
            else
              ionisation_rates_within(values, live, coefficients, log_field_min,
                                      d_log_field, intervals);
          };
          const auto sample_count = static_cast<std::size_t>(samples);
          const auto blocks =
              static_cast<int>((sample_count + rate_block - 1) / rate_block);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                const auto start = static_cast<std::size_t>(block) * rate_block;
                const auto count = std::min(rate_block, sample_count - start);
                double magnitudes[rate_block];
                for (std::size_t k = 0; k < count; ++k) {
                  const double first = field_at(base, start + k);
                  magnitudes[k] =
                      n_pol == 2 ? Kokkos::hypot(first, field_at(base + 1, start + k))
                                 : Kokkos::abs(first);
                }
                band_rates_block(magnitudes, &rates(slot, start), count, lower, upper,
                                 highest, within);
              });
        } else {
#endif
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                const auto time = static_cast<std::size_t>(sample);
                const double first = field_at(base, time);
                const double magnitude =
                    n_pol == 2 ? Kokkos::hypot(first, field_at(base + 1, time))
                               : Kokkos::abs(first);
                rates(slot, time) =
                    multiple
                        ? ionisation_rate(
                              magnitude, channel_coefficients, parameters(channel, 0),
                              parameters(channel, 1), parameters(channel, 2),
                              parameters(channel, 3), parameters(channel, 4),
                              static_cast<std::size_t>(parameters(channel, 5)),
                              static_cast<std::size_t>(parameters(channel, 6)))
                        : ionisation_rate(magnitude, coefficients, log_field_min,
                                          d_log_field, field_min, field_max, rate_max,
                                          intervals);
              });
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
        }
#endif
        team.team_barrier();
        team_band_filter<band_staging>(team, tile, rates, slot, filtered, slot, samples,
                                       taps, half);
        team_trapezoid_prefix(team, filtered, slot, samples, dt,
                              [&](const int sample, const double prefix) {
                                exposures(slot, static_cast<std::size_t>(sample)) =
                                    prefix;
                              });
        double largest = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const int sample, double& update) {
              const double rate = rates(slot, static_cast<std::size_t>(sample));
              if (rate > update)
                update = rate;
            },
            Kokkos::Max<double>(largest));
        const double exposure = exposures(slot, last);
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          output(first_node + node, 2 * channel) = exposure > 0.0 ? exposure : 0.0;
          output(first_node + node, 2 * channel + 1) = largest > 0.0 ? largest : 0.0;
        });
        if constexpr (Diagnose) {
          // The exposure's increments, as the scan formed them.
          const auto increment = [&](const int sample) {
            const auto time = static_cast<std::size_t>(sample);
            const double current = filtered(slot, time);
            const double previous = filtered(slot, time - 1);
            return dt * (current + previous) / 2.0;
          };
          double dip = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double fall = -increment(sample);
                if (fall > update)
                  update = fall;
              },
              Kokkos::Max<double>(dip));
          double rise = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& update) {
                const double step = increment(sample);
                if (step > update)
                  update = step;
              },
              Kokkos::Max<double>(rise));
          team.team_barrier();
          // The filtered row becomes the owed work, the exposure row the fraction.
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          // With vector maths a host thread takes blocks: the decays in one pass,
          // then `decayed_fractions`, as the column forms them.
          if constexpr (vmath::vectorised) {
            const auto sample_count = static_cast<std::size_t>(samples);
            const auto blocks =
                static_cast<int>((sample_count + rate_block - 1) / rate_block);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, blocks), [&](const int block) {
                  const auto start = static_cast<std::size_t>(block) * rate_block;
                  const auto count = std::min(rate_block, sample_count - start);
                  double* const exposed = &exposures(slot, start);
                  double* const owed = &filtered(slot, start);
                  const double* const rate = &rates(slot, start);
                  double fraction[rate_block];
                  for (std::size_t k = 0; k < count; ++k)
                    fraction[k] = -(exposed[k] < 0.0 ? 0.0 : exposed[k]);
                  vmath::exp(fraction, fraction, count);
                  decayed_fractions(fraction, count, initial);
                  for (std::size_t k = 0; k < count; ++k) {
                    owed[k] = rate[k] * (1.0 - fraction[k]);
                    exposed[k] = fraction[k];
                  }
                });
          } else {
#endif
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, samples), [&](const int sample) {
                  const auto time = static_cast<std::size_t>(sample);
                  const double fraction = band_fraction(exposures(slot, time), initial);
                  filtered(slot, time) = rates(slot, time) * (1.0 - fraction);
                  exposures(slot, time) = fraction;
                });
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
          }
#endif
          team.team_barrier();
          double paid = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, 1, samples),
              [&](const int sample, double& sum) { sum += increment(sample); }, paid);
          const auto level = [&](const int sample, double& update) {
            const auto time = static_cast<std::size_t>(sample);
            const double fraction = exposures(slot, time);
            for (std::size_t component = 0; component < n_pol; ++component) {
              const double value =
                  Kokkos::abs(fraction * field_at(base + component, time));
              if (value > update)
                update = value;
            }
          };
          double early = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, reach), level,
                                  Kokkos::Max<double>(early));
          double peak = 0.0;
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, samples), level,
                                  Kokkos::Max<double>(peak));
          const double freed = exposures(slot, last) - exposures(slot, 0);
          const double values[plasma_diagnostic_parts] = {
              dip > 0.0 ? dip : 0.0,     rise > 0.0 ? rise : 0.0,
              Kokkos::abs(paid - freed), freed,
              early > 0.0 ? early : 0.0, peak > 0.0 ? peak : 0.0};
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            parts[part] = Kokkos::max(parts[part], values[part]);
        }
        team.team_barrier();
      }
      if constexpr (Diagnose)
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          for (std::size_t part = 0; part < plasma_diagnostic_parts; ++part)
            diagnostics(first_node + node, part) = parts[part];
        });
      team.team_barrier();
    }
  };
  launch::parallel_for("solprop_native.plasma.team_band_statistics",
                       band_team_policy(node_count, scan), scan);
}

void GeneralWorkspace::convolve_raman(const std::size_t stage,
                                      const std::size_t node_count) {
  const bool tail = node_count != problem_.dims.node_chunk;
  Kokkos::Profiling::pushRegion("solprop_native.raman.drive_fft");
  execute_raman(tail ? *raman_tail_forward_plan_ : *raman_drive_forward_plan_,
                tail ? raman_tail_drive_ : node_raman_drive_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.spectral_multiply");
  const auto node_spectrum = node_raman_spectrum_;
  const auto kernel_spectrum = raman_kernel_spectrum_;
  const auto drives = problem_.raman_drives();
  const double inverse_doubled = 1.0 / static_cast<double>(2 * problem_.dims.n_to);
  launch::parallel_for(
      "solprop_native.raman.spectral_multiply",
      Range2(tail ? raman_tail_multiply_space_ : raman_multiply_space_, {0, 0},
             {node_count * drives, problem_.dims.n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t index) {
        node_spectrum(node, index) =
            (node_spectrum(node, index) *
             kernel_spectrum(node % drives == 0 ? 0 : 1, index)) *
            inverse_doubled;
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("solprop_native.raman.inverse_fft");
  execute_raman(tail ? *raman_tail_inverse_plan_ : *raman_convolution_inverse_plan_,
                tail ? raman_tail_spectrum_ : node_raman_spectrum_,
                tail ? raman_tail_convolved_ : node_raman_convolved_,
                KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
  finish_raman(stage, node_count);
}

void GeneralWorkspace::finish_raman(const std::size_t stage,
                                    const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.raman.finish");
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto convolved = node_raman_convolved_;
  const auto window = time_window_;
  const auto density = stage_raman_density_;
  const auto family = problem_.dims.family;
  const auto components = problem_.effective_node_pol();
  const auto drives = problem_.raman_drives();
  const double epsilon0 = problem_.raman.epsilon0;
  launch::parallel_for(
      "solprop_native.raman.finish",
      Range2(response_space_for(node_count), {0, 0}, {node_count, problem_.dims.n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        if (components == 2) {
          const auto base = 2 * node;
          const auto row = drives * node;
          const double u = convolved(row, time);
          const double v = drives == 3 ? convolved(row + 1, time) : 0.0;
          const double w = drives == 3 ? convolved(row + 2, time) : 0.0;
          const double factor = epsilon0 * density(stage);
          if (family == FieldFamily::envelope) {
            const Complex x = node_complex(base, time);
            const Complex y = node_complex(base + 1, time);
            out_complex(base, time) =
                (out_complex(base, time) + factor * ((u + v) * x + w * y)) *
                window(time);
            out_complex(base + 1, time) =
                (out_complex(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          } else {
            const double x = family == FieldFamily::field_real
                                 ? node_real(base, time)
                                 : node_complex(base, time).real();
            const double y = family == FieldFamily::field_real
                                 ? node_real(base + 1, time)
                                 : node_complex(base + 1, time).real();
            out_real(base, time) =
                (out_real(base, time) + factor * ((u + v) * x + w * y)) * window(time);
            out_real(base + 1, time) =
                (out_real(base + 1, time) + factor * (w * x + (u - v) * y)) *
                window(time);
          }
          return;
        }
        const double factor = epsilon0 * density(stage) * convolved(node, time);
        if (family == FieldFamily::envelope) {
          out_complex(node, time) =
              (out_complex(node, time) + factor * node_complex(node, time)) *
              window(time);
        } else {
          const double field = family == FieldFamily::field_real
                                   ? node_real(node, time)
                                   : node_complex(node, time).real();
          out_real(node, time) = (out_real(node, time) + factor * field) * window(time);
        }
      });
  Kokkos::Profiling::popRegion();
}

// The modal-product convolution (after the CPU experiment's Raman port). With a real
// static synthesis S, a scalar node field is x_j = sum_m S_jm a_m, so the Raman drive
// x_j^2 (or the cycle average |x_j|^2 / 2) is sum_{m <= n} c_mn S_jm S_jn P_mn with
// P_mn = Re(a_m) Re(a_n) (or (Re(a_m conj a_n)) / 2) and c_mn = 2 off the diagonal.
// The causal convolution is linear: convolving each of the M (M + 1) / 2 products once
// and reconstructing each node's convolved drive from them replaces one drive per node.
// It is chosen where it transforms fewer rows than the nodes and its rows fit the node
// drive's own scratch; the stage-dependent kernel, its support and the taper (inside
// S) are those of the node path. The reconstruction sums the products in a different
// order from squaring the synthesised field: the two agree to rounding of the drive's
// scale, sum_m |S_jm a_m| squared. Host builds only: a device keeps the node drives.
std::size_t GeneralWorkspace::raman_modal_products() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return 0;
#else
  if (!problem_.raman.enabled || problem_.general.enabled ||
      problem_.dispersive.enabled || problem_.separable_geometry() ||
      problem_.effective_node_pol() != 1 || problem_.dims.n_pol_state != 1 ||
      problem_.dims.n_pol_transform != 1 || problem_.raman_drives() != 1 ||
      problem_.dims.n_nodes < 2)
    return 0;
  const auto products = problem_.dims.n_modes * (problem_.dims.n_modes + 1) / 2;
  return products < problem_.dims.n_nodes && products <= problem_.dims.node_chunk
             ? products
             : 0;
#endif
}

void GeneralWorkspace::convolve_raman_products() {
  Kokkos::Profiling::pushRegion("solprop_native.raman.products");
  const auto drive = raman_product_drive_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto modes = problem_.dims.n_modes;
  const auto n_to = problem_.dims.n_to;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const bool squared = problem_.raman.thg;
  launch::parallel_for(
      "solprop_native.raman.products",
      Range2(raman_product_space_, {0, 0}, {raman_products_, 2 * n_to}),
      KOKKOS_LAMBDA(const std::size_t product, const std::size_t time) {
        if (time >= n_to) {
          drive(product, time) = 0.0;
          return;
        }
        // The unordered pair (first, second), first <= second, in row-major order.
        std::size_t first = 0;
        std::size_t second = product;
        while (second >= modes - first) {
          second -= modes - first;
          ++first;
        }
        second += first;
        const double multiplicity = first == second ? 1.0 : 2.0;
        double value;
        if (real_family) {
          value = modal_real(first, time) * modal_real(second, time);
          if (!squared)
            value *= 0.5;
        } else {
          const Complex a = modal_complex(first, time);
          const Complex b = modal_complex(second, time);
          value = squared ? a.real() * b.real()
                          : 0.5 * (a.real() * b.real() + a.imag() * b.imag());
        }
        drive(product, time) = multiplicity * value;
      });
  execute_raman(*raman_product_forward_plan_, raman_product_drive_,
                raman_product_spectrum_, KokkosFFT::Normalization::none);
  const auto spectrum = raman_product_spectrum_;
  const auto kernel_spectrum = raman_kernel_spectrum_;
  const double inverse_doubled = 1.0 / static_cast<double>(2 * n_to);
  launch::parallel_for(
      "solprop_native.raman.products_multiply",
      Range2(raman_product_space_, {0, 0}, {raman_products_, n_to + 1}),
      KOKKOS_LAMBDA(const std::size_t product, const std::size_t index) {
        spectrum(product, index) =
            (spectrum(product, index) * kernel_spectrum(0, index)) * inverse_doubled;
      });
  execute_raman(*raman_product_inverse_plan_, raman_product_spectrum_,
                raman_product_drive_, KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
}

// `finish_raman`'s scalar form with each node's convolved drive reconstructed from
// the convolved modal products.
void GeneralWorkspace::finish_raman_products(const std::size_t stage,
                                             const std::size_t first_node,
                                             const std::size_t node_count) {
  Kokkos::Profiling::pushRegion("solprop_native.raman.finish_products");
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const auto out_real = polarisation_real_;
  const auto out_complex = polarisation_complex_;
  const auto convolved = raman_product_drive_;
  const auto synthesis = scaled_synthesis_;
  const auto window = time_window_;
  const auto density = stage_raman_density_;
  const auto family = problem_.dims.family;
  const auto modes = problem_.dims.n_modes;
  const double epsilon0 = problem_.raman.epsilon0;
  const auto n_to = problem_.dims.n_to;
  // Blocks of samples per node, the products accumulated in their order over each
  // block as contiguous rows: one pass over the block per product, which vectorises,
  // where a per-sample loop over the products strides between rows.
  constexpr std::size_t block = 512;
  const std::size_t blocks = (n_to + block - 1) / block;
  const double* const products = convolved.data();
  const std::size_t stride = convolved.stride(0);
  launch::parallel_for(
      "solprop_native.raman.finish_products",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count * blocks),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t node = item / blocks;
        const std::size_t begin = (item % blocks) * block;
        const std::size_t count = n_to - begin < block ? n_to - begin : block;
        const std::size_t row = first_node + node;
        double drive[block];
        for (std::size_t offset = 0; offset < count; ++offset)
          drive[offset] = 0.0;
        std::size_t product = 0;
        for (std::size_t first = 0; first < modes; ++first) {
          const double weight = synthesis(row, first);
          for (std::size_t second = first; second < modes; ++second, ++product) {
            const double pair = weight * synthesis(row, second);
            const double* const values = products + product * stride + begin;
            for (std::size_t offset = 0; offset < count; ++offset)
              drive[offset] += pair * values[offset];
          }
        }
        const double scale = epsilon0 * density(stage);
        for (std::size_t offset = 0; offset < count; ++offset) {
          const std::size_t time = begin + offset;
          const double factor = scale * drive[offset];
          if (family == FieldFamily::envelope) {
            out_complex(node, time) =
                (out_complex(node, time) + factor * node_complex(node, time)) *
                window(time);
          } else {
            const double field = family == FieldFamily::field_real
                                     ? node_real(node, time)
                                     : node_complex(node, time).real();
            out_real(node, time) =
                (out_real(node, time) + factor * field) * window(time);
          }
        }
      });
  Kokkos::Profiling::popRegion();
}

// One channel's cycle-averaged rates at each node and their running integral, from
// the polarisation ellipse of `field_rows`' `n_pol` rows per node: the node buffer
// with its components, or an envelope's on-axis row (`axis_populations`).
void GeneralWorkspace::envelope_cycle(const std::size_t channel,
                                      const std::size_t node_count,
                                      const ComplexView2& field_rows,
                                      const std::size_t n_pol) {
  // The kernels capture the view by value, never the caller's reference, which a
  // device cannot read.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto field = field_rows;
  const auto parameters = channel_parameters_;
  const auto coefficients = channel_coefficients_;
  const auto nodes =
      Kokkos::subview(envelope_rule_nodes_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto weights =
      Kokkos::subview(envelope_rule_weights_, channel, Kokkos::ALL(), Kokkos::ALL());
  const auto moments = envelope_moments_;
  const auto rates = envelope_rates_;
  const auto failures = envelope_failures_;
  const auto n_to = problem_.dims.n_to;
  launch::parallel_for(
      "solprop_native.envelope.cycle_moments",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto rate_at = [&](const double value) {
          return ionisation_rate(value, coefficients, parameters(channel, 0),
                                 parameters(channel, 1), parameters(channel, 2),
                                 parameters(channel, 3), parameters(channel, 4),
                                 static_cast<std::size_t>(parameters(channel, 5)),
                                 static_cast<std::size_t>(parameters(channel, 6)));
        };
        const auto pair =
            envelope_moments(axes.major, axes.ratio, parameters(channel, 2),
                             parameters(channel, 3), rate_at, nodes, weights);
        moments(0, node, time) = pair.cosine;
        moments(1, node, time) = pair.sine;
        rates(node, time) = pair.cosine + axes.ratio * axes.ratio * pair.sine;
        failures(node * n_to + time) = pair.failed ? 1 : 0;
      });
  const auto status = envelope_status_;
  std::size_t no_failure = 0;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A serial device scan dominated the short combined Raman/plasma profiles.
  // Integer minimum preserves the first failing sample exactly;
  // its identity denotes success without an additional normalization launch.
  no_failure = std::numeric_limits<std::size_t>::max();
  using StatusPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using StatusMember = StatusPolicy::member_type;
  const auto samples = node_count * n_to;
  // One team uses only launch-time shared memory; a global reduction could grow
  // the execution-space scratch allocator inside the first trial.
  launch::parallel_for(
      "solprop_native.envelope.angular_status",
      StatusPolicy(response_space_for(node_count), 1,
                   SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE),
      KOKKOS_LAMBDA(const StatusMember& team) {
        std::size_t first_failure = Kokkos::reduction_identity<std::size_t>::min();
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, samples),
            [&](const std::size_t sample, std::size_t& minimum) {
              if (failures(sample) != 0)
                minimum = Kokkos::min(minimum, sample + 1);
            },
            Kokkos::Min<std::size_t>(first_failure));
        Kokkos::single(Kokkos::PerTeam(team), [&]() { status(0) = first_failure; });
      });
#else
  launch::parallel_for(
      "solprop_native.envelope.angular_status",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        status(0) = 0;
        for (std::size_t sample = 0; sample < node_count * n_to; ++sample)
          if (failures(sample) != 0) {
            status(0) = sample + 1;
            break;
          }
      });
#endif
  // This planned scalar boundary rejects failed refinement before a current FFT.
  graph::deep_copy(response_space_for(node_count), envelope_host_status_,
                   envelope_status_);
  response_space_for(node_count).fence("solprop_native envelope refinement boundary");
  if (envelope_host_status_(0) != no_failure)
    refuse<EnvelopeConvergenceError>("envelope plasma channel ", channel,
                                     " exceeded the angular refinement budget");
  const auto rows = std::make_pair(std::size_t{0}, node_count);
  portable_team_trapezoid_scan<true>(
      response_space_for(node_count),
      Kokkos::subview(envelope_rates_, rows, Kokkos::ALL()),
      Kokkos::subview(envelope_exposure_, rows, Kokkos::ALL()), problem_.grid.dt_fine);
}

// One channel's exposure and peak cycle-averaged rate at each node of `field`
// (`envelope_cycle`), into the rows of `statistics` from `first_node`.
void GeneralWorkspace::envelope_statistics(const std::size_t channel,
                                           const std::size_t first_node,
                                           const std::size_t node_count,
                                           const RealView2& statistics,
                                           const ComplexView2& field,
                                           const std::size_t n_pol) {
  envelope_cycle(channel, node_count, field, n_pol);
  const auto rates = envelope_rates_;
  const auto exposure = envelope_exposure_;
  const auto n_to = problem_.dims.n_to;
  launch::parallel_for(
      "solprop_native.envelope.population_statistics",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        double peak = 0.0;
        for (std::size_t time = 0; time < n_to; ++time)
          peak = Kokkos::max(peak, rates(node, time));
        statistics(first_node + node, 2 * channel) = exposure(node, n_to - 1);
        statistics(first_node + node, 2 * channel + 1) = peak;
      });
}

void GeneralWorkspace::envelope_population(const std::size_t channel,
                                           const std::size_t stage,
                                           const std::size_t node_count) {
  const auto n_pol = problem_.effective_node_pol();
  envelope_cycle(channel, node_count, node_complex_, n_pol);
  const auto exposure = envelope_exposure_;
  const auto n_to = problem_.dims.n_to;
  const auto parameters = channel_parameters_;
  const auto densities = stage_channel_density_;
  const auto initials = stage_channel_preionisation_;
  const auto moments = envelope_moments_;
  const auto field = node_complex_;
  const auto window = time_window_;
  const auto drive = envelope_streams_[0].node;
  const auto loss = envelope_streams_[1].node;
  launch::parallel_for(
      "solprop_native.envelope.currents",
      Range2(response_space_for(node_count), {0, 0}, {node_count, n_to}),
      KOKKOS_LAMBDA(const std::size_t node, const std::size_t time) {
        const double initial = initials(stage, channel);
        const double fraction = envelope_fraction(exposure(node, time), initial);
        const double survival = (1.0 - initial) * Kokkos::exp(-exposure(node, time));
        const double density = densities(stage, channel);
        const auto axes =
            envelope_axes(field(node * n_pol, time),
                          n_pol == 2 ? field(node * n_pol + 1, time) : Complex{});
        const auto current =
            envelope_loss(axes, moments(0, node, time), moments(1, node, time),
                          parameters(channel, 8) * density * survival);
        const double drude = parameters(channel, 7) * density * fraction;
        for (std::size_t pol = 0; pol < n_pol; ++pol) {
          const auto row = node * n_pol + pol;
          const auto drive_value = window(time) * drude * field(row, time);
          const auto loss_value = window(time) * current[pol];
          if (channel == 0) {
            drive(row, time) = drive_value;
            loss(row, time) = loss_value;
          } else {
            drive(row, time) += drive_value;
            loss(row, time) += loss_value;
          }
        }
      });
}

void GeneralWorkspace::envelope_response(const std::size_t stage,
                                         const std::size_t first_node,
                                         const std::size_t node_count,
                                         const TransverseChunk* chunk) {
  for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
    envelope_population(channel, stage, node_count);
  for (auto& stream : envelope_streams_) {
    if (problem_.slab_geometry()) {
      // As the polarisation: the chunk's y rows, zero beyond a final partial chunk.
      graph::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      slab_project(*chunk, &stream);
    } else if (problem_.free_geometry()) {
      // The persistent maximum-chunk plan transforms zero padding in a final
      // partial chunk; only its actual rows are ever copied to the spatial FFT.
      graph::deep_copy(response_space_for(node_count), stream.projected, Complex{});
      const auto rows =
          std::make_pair(std::size_t{0}, node_count * problem_.effective_node_pol());
      graph::deep_copy(response_space_for(node_count),
                       Kokkos::subview(stream.projected, rows, Kokkos::ALL()),
                       Kokkos::subview(stream.node, rows, Kokkos::ALL()));
    } else {
      project(first_node, node_count, first_node == 0, &stream);
    }
  }
}
