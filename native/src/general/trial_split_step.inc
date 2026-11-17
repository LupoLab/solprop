// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The split-step absorber: its factor, the column and state passes, the removal
// it adds and the separable split step.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

namespace {
// The absorber split's factor `exp(-rate h / 2)` on `rows` rows of `time`. The factor
// depends on the sample alone: a device forms it once a step for every sample, and a
// host task forms it once for a tile of samples, through the vector-maths layer
// (solprop ADR-059), and applies it to a block of rows, the same expression to the bit
// on the libm route. A device reads each sample's factor from `factor`, formed once a
// step by `form_split_factor` with the same expression: one binary64 exponential per
// sample rather than per element, which on a device with a reduced binary64 rate is
// most of the pass.
template <class TimeView>
void split_apply(const char* label, const ExecutionSpace& execution,
                 const TimeView& time, const RealView& rate, const RealView& factors,
                 const double dz, const std::size_t rows, const std::size_t samples,
                 const double scale = 1.0) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(rate);
  static_cast<void>(dz);
  launch::parallel_for(
      label, Range2(execution, {0, 0}, {rows, samples}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        time(row, index) *= scale * factors(index);
      });
#else
  static_cast<void>(factors);
  constexpr std::size_t tile = 256;
  constexpr std::size_t row_block = 64;
  const std::size_t tiles = (samples + tile - 1) / tile;
  const std::size_t blocks = (rows + row_block - 1) / row_block;
  launch::parallel_for(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, tiles * blocks),
      KOKKOS_LAMBDA(const std::size_t task) {
        const std::size_t first = (task % tiles) * tile;
        const std::size_t count = samples - first < tile ? samples - first : tile;
        const std::size_t row_first = (task / tiles) * row_block;
        const std::size_t row_last =
            rows - row_first < row_block ? rows : row_first + row_block;
        double factor[tile];
        for (std::size_t sample = 0; sample < count; ++sample)
          factor[sample] = -0.5 * rate(first + sample) * dz;
        vmath::exp(factor, factor, count);
        for (std::size_t sample = 0; sample < count; ++sample)
          factor[sample] *= scale;
        for (std::size_t row = row_first; row < row_last; ++row)
          for (std::size_t sample = 0; sample < count; ++sample)
            time(row, first + sample) *= factor[sample];
      });
#endif
}
} // namespace

void GeneralWorkspace::split_columns(const double dz, std::size_t active_rows,
                                     const double parseval,
                                     const std::size_t first_column) {
  const bool tail = temporal_tail_rows_ != 0 && active_rows == temporal_tail_rows_;
  if (active_rows == 0)
    active_rows = coarse_spectrum_.extent(0);
  if (problem_.dims.family == FieldFamily::envelope) {
    if (tail)
      coarse_tail_inverse_complex_.execute();
    else
      graph::execute(*coarse_inverse_complex_plan_, coarse_spectrum_,
                     coarse_time_complex_, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval, first_column, coarse_time_complex_,
                        1.0);
    split_apply("solprop_native.absorber.split_apply_complex", execution_space_,
                coarse_time_complex_, absorber_coarse_, split_factor_, dz, active_rows,
                problem_.dims.n_t);
    if (tail)
      coarse_tail_forward_complex_.execute();
    else
      graph::execute(*coarse_forward_complex_plan_, coarse_time_complex_,
                     coarse_spectrum_, KokkosFFT::Normalization::none);
  } else {
    if (tail)
      coarse_tail_inverse_real_.execute();
    else
      graph::execute(*coarse_inverse_real_plan_, coarse_spectrum_, coarse_time_real_,
                     KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, active_rows, parseval, first_column, {}, 1.0);
    split_apply("solprop_native.absorber.split_apply_real", execution_space_,
                coarse_time_real_, absorber_coarse_, split_factor_, dz, active_rows,
                problem_.dims.n_t);
    if (tail)
      coarse_tail_forward_real_.execute();
    else
      graph::execute(*coarse_forward_real_plan_, coarse_time_real_, coarse_spectrum_,
                     KokkosFFT::Normalization::none);
  }
}

// Free space's complex rows: the split's factor depends on time alone, so it acts alike
// on every column of any transverse basis, and the temporal pair runs in place on the
// state's own rows, a block of them at a time. Nothing is copied to a coarse buffer and
// back, the transform's normalisation goes with the factor, and the band's mask is the
// one pass after the blocks. Equal to the reference's split in real space up to the
// transforms' rounding, at any block size.
void GeneralWorkspace::split_state_rows(const double dz, const ComplexView2& state) {
  const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
  const auto columns = problem_.n_columns();
  for (std::size_t first = 0; first < columns; first += split_rows_) {
    const auto rows = std::min(split_rows_, columns - first);
    const ComplexView2 block =
        Kokkos::subview(state, std::make_pair(first, first + rows), Kokkos::ALL());
    const bool tail = rows != split_rows_;
    graph::execute(tail ? *split_tail_inverse_plan_ : *split_inverse_plan_, block,
                   block, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz, rows, 0.0, first, block, inverse_time * inverse_time);
    split_apply("solprop_native.absorber.split_apply_complex", execution_space_, block,
                absorber_coarse_, split_factor_, dz, rows, problem_.dims.n_t,
                inverse_time);
    graph::execute(tail ? *split_tail_forward_plan_ : *split_forward_plan_, block,
                   block, KokkosFFT::Normalization::none);
  }
  mask(state);
}

// What one split takes of the power it finds in the coarse time rows, `sum P (1 -
// exp(-alpha h))` with the boundary monitor's column weights, added to the run's
// accumulator in the memory space (solprop SEP 21, ADR-057): the monitor's reading
// drains it at a check, so every accepted step is counted, checked or not. The same
// pass adds to the step's `sum P'` and the guards' and the collar's sums of `P'`, the
// power found halfway through the split's factor, `P exp(-alpha h / 2)`, which the
// monitor judges the occupancies on: the symmetric splitting's state at the step's
// end. A separable route's rows are real-space rows, measured by `parseval`; zero
// means the rows are the stored columns from `first_column` on, weighted by their
// quadrature. `complex_time` holds a complex route's rows, whose power `power_scale`
// brings to the state's measure: the state's own rows carry no transform normalisation.
void GeneralWorkspace::add_split_removal(const double dz, const std::size_t active_rows,
                                         const double parseval,
                                         const std::size_t first_column,
                                         const ComplexView2& complex_time,
                                         const double power_scale) {
  const auto rate = absorber_coarse_;
  const auto share = monitor_share_;
  const auto half = monitor_half_;
  const auto weights = monitor_column_weights_;
  const bool stored = parseval == 0.0;
  // The share the split takes, and half its power factor, depend on time alone:
  // formed once, not per row.
  launch::parallel_for(
      "solprop_native.monitor.split_share",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_t),
      KOKKOS_LAMBDA(const std::size_t index) {
        share(index) = -Kokkos::expm1(-rate(index) * dz);
        half(index) = Kokkos::exp(-0.5 * rate(index) * dz);
      });
  const auto guards = monitor_weights_;
  const auto samples =
      Range2(execution_space_, {0, 0}, {active_rows, problem_.dims.n_t});
  if (problem_.dims.family == FieldFamily::envelope) {
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_complex", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index,
                      SplitSums& update) {
          const double power = (stored ? weights(first_column + row) : parseval) *
                               (power_scale * squared_abs(complex_time(row, index)));
          update.values[0] += power * share(index);
          const double found = half(index) * power;
          update.values[1] += found;
          update.values[2] += guards(0, index) * found;
          update.values[3] += guards(1, index) * found;
          update.values[4] += guards(2, index) * found;
        },
        Kokkos::Sum<SplitSums, MemorySpace>(monitor_split_sum_));
  } else {
    const auto time = coarse_time_real_;
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_real", samples,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index,
                      SplitSums& update) {
          const double value = time(row, index);
          const double power =
              (stored ? weights(first_column + row) : parseval) * value * value;
          update.values[0] += power * share(index);
          const double found = half(index) * power;
          update.values[1] += found;
          update.values[2] += guards(0, index) * found;
          update.values[3] += guards(1, index) * found;
          update.values[4] += guards(2, index) * found;
        },
        Kokkos::Sum<SplitSums, MemorySpace>(monitor_split_sum_));
  }
  const auto removed = monitor_removed_;
  const auto split = monitor_split_sum_;
  const auto found = monitor_found_;
  launch::parallel_for(
      "solprop_native.monitor.split_removal_add",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        removed() += split().values[0];
        found() += split();
      });
}

// The split-step temporal absorber on a separable geometry: the rate is the same at
// every node, so it acts on the temporal rows (free space's nodes, the planar kind's
// y rows of slab amplitudes) between the two transverse transforms.
void GeneralWorkspace::separable_split_step(const double dz,
                                            const ComplexView2& advanced) {
  if (split_rows_ != 0) {
    split_state_rows(dz, advanced);
    return;
  }
  if (problem_.free_geometry()) {
    // The split's factor depends on time alone, so it acts alike on every column of
    // any transverse basis (solprop's reference measures its removal on the stored
    // columns for this reason): free space splits its stored transverse-spectrum
    // columns directly, chunk by chunk of the temporal rows, as a modal route splits
    // its modes, with no transverse transform pair. Equal to the reference's split in
    // real space up to the transforms' rounding.
    const auto coarse = coarse_spectrum_;
    const auto active = active_mask_;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto n = problem_.dims.n_omega;
    const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
    for (std::size_t unit = 0; unit < transverse_units();
         unit += transverse_units_per_chunk()) {
      const auto chunk = transverse_chunk(unit);
      const auto first = chunk.first_row * n_pol;
      const auto rows = chunk.rows * n_pol;
      launch::parallel_for(
          "solprop_native.free.split_columns_input",
          Range2(execution_space_, {0, 0}, {rows, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
            coarse(row, omega) = inverse_time * advanced(first + row, omega);
          });
      split_columns(dz, rows, 0.0, first);
      launch::parallel_for(
          "solprop_native.free.split_columns_output",
          Range2(execution_space_, {0, 0}, {rows, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
            advanced(first + row, omega) =
                active(omega) != 0U ? coarse(row, omega) : Complex{};
          });
    }
    return;
  }
  const auto scratch = transverse_;
  const auto coarse = coarse_spectrum_;
  const auto active = active_mask_;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto layout = transverse_layout();
  const auto n = problem_.dims.n_omega;
  const double inverse_nodes = inverse_transverse_count();
  const double inverse_time = 1.0 / static_cast<double>(problem_.dims.n_t);
  launch::parallel_for(
      "solprop_native.free.split_input",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        layout.cell(scratch, omega * n_pol + row % n_pol, mode) =
            inverse_nodes * advanced(row, omega);
      });
  transverse_inverse();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    const auto count = chunk.rows;
    graph::deep_copy(execution_space_, coarse, Complex{});
    launch::parallel_for(
        "solprop_native.free.split_temporal_input",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          coarse(row, omega) =
              inverse_time * layout.cell(scratch, omega * n_pol + row % n_pol, mode);
        });
    // The rows are real-space rows here: by Parseval their power is the stored
    // columns' over the transverse count, which the monitor's measure is in.
    split_columns(dz, count * n_pol, 1.0 / inverse_nodes);
    launch::parallel_for(
        "solprop_native.free.split_temporal_output",
        Range2(execution_space_, {0, 0}, {count * n_pol, n}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          const auto mode = first + row / n_pol;
          layout.cell(scratch, omega * n_pol + row % n_pol, mode) = coarse(row, omega);
        });
  }
  transverse_forward();
  launch::parallel_for(
      "solprop_native.free.split_output",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), n}),
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto mode = row / n_pol;
        advanced(row, omega) =
            active(omega) != 0U
                ? layout.cell(scratch, omega * n_pol + row % n_pol, mode)
                : Complex{};
      });
}

void GeneralWorkspace::form_split_factor(const double dz) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto rate = absorber_coarse_;
  const auto factor = split_factor_;
  launch::parallel_for(
      "solprop_native.absorber.split_factor",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_t),
      KOKKOS_LAMBDA(const std::size_t index) {
        factor(index) = Kokkos::exp(-0.5 * rate(index) * dz);
      });
#else
  static_cast<void>(dz);
#endif
}

void GeneralWorkspace::split_step(const double dz, const ComplexView2& advanced) {
  form_split_factor(dz);
  if (monitoring_split()) {
    // The chunks below add what this step's split finds; the last step's is dropped.
    const auto found = monitor_found_;
    launch::parallel_for(
        "solprop_native.monitor.split_found_clear",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { found() = SplitSums{}; });
  }
  if (problem_.separable_geometry()) {
    separable_split_step(dz, advanced);
    return;
  }
  const auto coarse = coarse_spectrum_;
  const double inverse_n = 1.0 / static_cast<double>(problem_.dims.n_t);
  launch::parallel_for(
      "solprop_native.absorber.split_pad",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        coarse(column, index) = inverse_n * advanced(column, index);
      });
  split_columns(dz);
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.absorber.split_commit",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        advanced(column, index) =
            active(index) != 0U ? coarse(column, index) : Complex{};
      });
}
