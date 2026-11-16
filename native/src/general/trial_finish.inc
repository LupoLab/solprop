// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The stage's finish: the held and gated responses, the transforms and the
// projection, the absorber, finish_stage, the flow-action copies and the weighted
// error; then the plasma statistics scan, which sits here in the unit's order;
// then the step's finish, the quadrature, FSAL and the mask.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

// SEP 27 forms two independently collared current products. Every maximum-chunk
// row is written, including a partial chunk's padding, before a persistent FFT plan
// can read it. Modal projection precedes the FFT; radial projection follows it.
void GeneralWorkspace::held_response(const std::size_t stage,
                                     const std::size_t first_node,
                                     const std::size_t node_count,
                                     const TransverseChunk* chunk) {
  const auto current = held_current_;
  const auto window = time_window_;
  const auto derivative = held_derivative_;
  const auto product = held_streams_[0].node;
  const auto edge = held_streams_[1].node;
  const auto live = node_count * problem_.effective_node_pol();
  const auto n_to = problem_.dims.n_to;
  cartesian_for(
      "solprop_native.held.products", response_space_for(node_count), current.extent(0),
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        const double value = row < live ? current(row, time) : 0.0;
        product(row, time) = window(time) * value;
        edge(row, time) = derivative(time) * value;
      });
  const bool folded = folds_single_node();
  const bool radial = held_radial();
  for (std::size_t index = 0; index < 2; ++index) {
    auto& stream = held_streams_[index];
    if (folded) {
      const auto source = stream.node;
      const auto target = stream.projected;
      const auto projection = projection_;
      const auto taper = stage_taper_;
      cartesian_for(
          "solprop_native.held.folded_projection", response_space_for(node_count), 1,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            target(row, time) = projection(0, 0) / taper(stage) * source(row, time);
          });
    } else if (problem_.slab_geometry()) {
      graph::deep_copy(response_space_for(node_count), stream.projected, 0.0);
      slab_contract("solprop_native.held.slab_projection",
                    response_space_for(node_count), projection_, stream.node,
                    stream.projected, chunk->nodes / problem_.spatial.n_x, n_to);
    } else if (problem_.free_geometry() || radial) {
      graph::deep_copy(response_space_for(node_count), stream.projected, stream.node);
    } else {
      project(first_node, node_count, first_node == 0, nullptr, &stream);
    }
  }
  if (radial) {
    const auto source = polarisation_real_;
    const auto target = held_streams_[2].projected;
    cartesian_for(
        "solprop_native.held.nonplasma", response_space_for(node_count),
        target.extent(0), n_to,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
          target(row, time) = row < live ? source(row, time) : 0.0;
        });
    held_radial_projection(first_node, node_count);
  }
}

// solprop's private gated benchmark (`NumpyRefBackend._column_sources`): the chunk's
// live current rows are integrated by the problem's running-integral rule, the band
// rule's filter then the trapezoid (solprop SEP 18), and w(t) times each prefix joins
// the local polarisation, which is then projected as the held current's streams would
// have been. The scans are the device column's portable primitives; no hot column
// changes, and a run without the benchmark never reaches this.
void GeneralWorkspace::gated_response(const std::size_t node_count) {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.gated.response");
  const auto live = std::pair<std::size_t, std::size_t>(
      0, node_count * problem_.effective_node_pol());
  const auto current = Kokkos::subview(held_current_, live, Kokkos::ALL);
  const auto prefix = Kokkos::subview(gated_prefix_, live, Kokkos::ALL);
  const auto& space = response_space_for(node_count);
  if (problem_.plasma.band)
    portable_team_band_scan(space, current,
                            Kokkos::subview(gated_filtered_, live, Kokkos::ALL), prefix,
                            plasma_taps_, problem_.grid.dt_fine);
  else
    portable_team_trapezoid_scan(space, current, prefix, problem_.grid.dt_fine);
  const auto window = time_window_;
  const auto polarisation = polarisation_real_;
  cartesian_for(
      "solprop_native.gated.polarisation", space, live.second, problem_.dims.n_to,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
        polarisation(row, time) += window(time) * prefix(row, time);
      });
}

// A stage's forward rows in one transform: all of them when the absorber's are due,
// else those after the absorber's (or the projection's alone). A host deferring its
// quadrature transforms the fine block only; `finish_quadrature` takes the coarse.
void GeneralWorkspace::transform_forward_pack() {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.forward_pack");
  const bool fine = defer_quadrature_ && forward_pack_fine_plan_;
  if (absorbing() || forward_pack_absorber_rows_ == 0) {
    if (fine)
      graph::execute(*forward_pack_fine_plan_, forward_pack_fine_,
                     forward_pack_fine_spectrum_, KokkosFFT::Normalization::none);
    else
      graph::execute(*forward_pack_plan_, forward_pack_, forward_pack_spectrum_,
                     KokkosFFT::Normalization::none);
  } else if (fine ? static_cast<bool>(forward_pack_fine_tail_plan_)
                  : static_cast<bool>(forward_pack_tail_plan_)) {
    if (fine)
      graph::execute(*forward_pack_fine_tail_plan_, forward_pack_fine_tail_,
                     forward_pack_fine_tail_spectrum_, KokkosFFT::Normalization::none);
    else
      graph::execute(*forward_pack_tail_plan_, forward_pack_tail_,
                     forward_pack_tail_spectrum_, KokkosFFT::Normalization::none);
  } else {
    graph::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                   KokkosFFT::Normalization::none);
  }
}

void GeneralWorkspace::held_transform(const bool coarse_only, const bool transformed) {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.fft.held_streams");
  // The packed rows are every transform this call takes when it takes the fine ones
  // and, where there are coarse rows, those too. The stage's forward pack, when it
  // took them, has transformed the streams' rows already.
  const bool every_row =
      !coarse_only && (!held_streams_[0].coarse_forward || !defer_quadrature_);
  if (held_pack_forward_ && every_row) {
    graph::execute(*held_pack_forward_, held_pack_, held_pack_spectrum_,
                   KokkosFFT::Normalization::none);
  } else if (!transformed) {
    for (std::size_t index = 0; index < 2; ++index) {
      auto& stream = held_streams_[index];
      if (!coarse_only)
        graph::execute(*stream.forward, stream.projected, stream.spectrum,
                       KokkosFFT::Normalization::none);
      if (stream.coarse_forward && (coarse_only || !defer_quadrature_))
        graph::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                       KokkosFFT::Normalization::none);
    }
  }
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto coarse_product = held_streams_[0].coarse_spectrum;
  const auto coarse_edge = held_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const bool include_coarse = !coarse.empty() && (coarse_only || !defer_quadrature_);
  cartesian_for(
      "solprop_native.held.spectral_assembly", execution_space_, projected.extent(0),
      problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        const auto bin = crop(omega);
        if (active(omega)) {
          const double value = inverse(omega);
          const Complex factor(0.0, -value);
          if (!coarse_only)
            projected(row, bin) +=
                factor * product(row, bin) + value * value * edge(row, bin);
          if (include_coarse)
            coarse(row, bin) += factor * coarse_product(row, bin) +
                                value * value * coarse_edge(row, bin);
        } else {
          if (!coarse_only)
            projected(row, bin) = Complex{};
          if (include_coarse)
            coarse(row, bin) = Complex{};
        }
      });
}

void GeneralWorkspace::held_radial_projection(const std::size_t first_node,
                                              const std::size_t node_count) {
  if (held_pack_forward_)
    graph::execute(*held_pack_forward_, held_pack_, held_pack_spectrum_,
                   KokkosFFT::Normalization::none);
  else
    for (auto& stream : held_streams_)
      graph::execute(*stream.forward, stream.projected, stream.spectrum,
                     KokkosFFT::Normalization::none);
  const auto product = held_streams_[0].spectrum;
  const auto edge = held_streams_[1].spectrum;
  const auto polarisation = held_streams_[2].spectrum;
  const auto nodal = held_nodal_spectrum_;
  const auto inverse = held_inverse_;
  const auto active = active_mask_;
  const auto crop = crop_index_;
  const auto n = problem_.dims.n_omega;
  const auto live = node_count * problem_.effective_node_pol();
  cartesian_for(
      "solprop_native.held.radial_assembly", node_spectral_space_, nodal.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        Complex value{};
        if (row < live && active(omega)) {
          const auto bin = crop(omega);
          const double inv = inverse(omega);
          value = polarisation(row, bin) + Complex(0.0, -inv) * product(row, bin) +
                  inv * inv * edge(row, bin);
        }
        nodal(row, omega) = value;
      });
  const auto packed = held_projection_;
  const auto projection = scaled_projection_;
  const auto first_row = first_node * problem_.dims.n_pol_transform;
  const auto live_rows = node_count * problem_.dims.n_pol_transform;
  cartesian_for(
      "solprop_native.held.radial_projection_matrix", matrix_copy_space_,
      packed.extent(0), packed.extent(1),
      KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
        packed(mode, row) = row < live_rows ? projection(mode, first_row + row) : 0.0;
      });
  // Interleaved real and imaginary parts share the scalar radial projection. This
  // is one GEMM of the combined cropped source, independent of state components.
  using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static_assert(sizeof(Complex) == 2 * sizeof(double));
  const auto columns = 2 * problem_.dims.n_pol_state * n;
  const Interleaved input(reinterpret_cast<double*>(nodal.data()), packed.extent(1),
                          columns);
  const Interleaved output(reinterpret_cast<double*>(held_projected_spectrum_.data()),
                           problem_.dims.n_modes, columns);
  selected_gemm(transform_space_for(node_count), packed, input, output, 0.0, true);
  if (first_node == 0)
    graph::deep_copy(temporal_execution_space_, projected_spectrum_, Complex{});
  const auto source = held_projected_spectrum_;
  const auto target = projected_spectrum_;
  cartesian_for(
      "solprop_native.held.radial_accumulate", execution_space_, target.extent(0), n,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        if (active(omega))
          target(row, crop(omega)) += source(row, omega);
      });
}

void GeneralWorkspace::envelope_transform() {
  for (auto& stream : envelope_streams_) {
    graph::execute(*stream.forward, stream.projected, stream.spectrum,
                   KokkosFFT::Normalization::none);
    if (stream.coarse_forward)
      graph::execute(*stream.coarse_forward, stream.coarse, stream.coarse_spectrum,
                     KokkosFFT::Normalization::none);
  }
  const auto drive = envelope_streams_[0].spectrum;
  const auto loss = envelope_streams_[1].spectrum;
  const auto coarse_drive = envelope_streams_[0].coarse_spectrum;
  const auto coarse_loss = envelope_streams_[1].coarse_spectrum;
  const auto projected = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto drude_factor = envelope_drude_factor_;
  const auto loss_factor = envelope_loss_factor_;
  const auto crop = crop_index_;
  launch::parallel_for(
      "solprop_native.envelope.spectral_assembly",
      Range2(execution_space_, {0, 0}, {projected.extent(0), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const auto index = crop(omega);
        projected(column, index) += drude_factor(omega) * drive(column, index) +
                                    loss_factor(omega) * loss(column, index);
        if (!coarse.empty())
          coarse(column, index) += drude_factor(omega) * coarse_drive(column, index) +
                                   loss_factor(omega) * coarse_loss(column, index);
      });
}

void GeneralWorkspace::project(const std::size_t first_node,
                               const std::size_t node_count, const bool first_chunk,
                               const EnvelopeStream* stream,
                               const FieldCurrentStream* field_stream) {
  const auto real_source =
      field_stream == nullptr ? polarisation_real_ : field_stream->node;
  const auto real_target =
      field_stream == nullptr ? projected_real_ : field_stream->projected;
  const auto real_coarse_target =
      field_stream == nullptr ? projected_coarse_real_ : field_stream->coarse;
  const auto real_stacked_target =
      field_stream == nullptr ? projected_stacked_real_ : field_stream->stacked;
  const auto node_source = stream == nullptr ? polarisation_complex_ : stream->node;
  const auto projected_target =
      stream == nullptr ? projected_complex_ : stream->projected;
  const auto coarse_target =
      stream == nullptr ? projected_coarse_complex_ : stream->coarse;
  const auto stacked_target =
      stream == nullptr ? projected_stacked_complex_ : stream->stacked;
  const bool whole_transform = first_node == 0 && node_count == problem_.dims.n_nodes &&
                               problem_.dims.n_pol_state == 1;
  if (whole_transform && !radial_blas_input_.empty()) {
    packed_radial_gemm(scaled_projection_, node_source, projected_target);
    return;
  }
  if (whole_transform && use_interleaved_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_interleaved");
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const Interleaved nodes(reinterpret_cast<double*>(node_source.data()),
                            problem_.dims.n_nodes, 2 * problem_.dims.n_to);
    const Interleaved fine(reinterpret_cast<double*>(projected_target.data()),
                           problem_.dims.n_modes, 2 * problem_.dims.n_to);
    engine_gemm(transform_space_for(node_count), scaled_projection_, nodes, fine, 0.0);
    if (project_coarse_ && !problem_.transform.projection_coarse_matrix.empty()) {
      const Interleaved coarse(reinterpret_cast<double*>(coarse_target.data()),
                               problem_.dims.n_modes, 2 * problem_.dims.n_to);
      engine_gemm(transform_space_for(node_count), scaled_projection_coarse_, nodes,
                  coarse, 0.0);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  // The stacked projections' splits copy on the stage's instance: on the temporal
  // copy's serial instance one took about 90 us at 16 threads on the four-mode
  // capillary, beside full-width kernels (docs/dev/benchmarks.md, "Stacked projection
  // splits").
  if (whole_transform && use_blas_for_projection()) {
    Kokkos::Profiling::pushRegion("solprop_native.transform.projection.kokkosblas");
    if (problem_.dims.family == FieldFamily::envelope && project_coarse_ &&
        !scaled_projection_stacked_complex_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_complex_,
                    node_source, stacked_target, Complex{}, true);
      const auto& stacked = stacked_target;
      const auto& fine = projected_target;
      const auto& coarse = coarse_target;
      const auto columns = problem_.n_columns();
      launch::parallel_for(
          "solprop_native.transform.projection.split_stacked_complex",
          Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else if (problem_.dims.family == FieldFamily::envelope) {
      selected_gemm(transform_space_for(node_count), scaled_projection_complex_,
                    node_source, projected_target, Complex{}, true);
    } else if (one_mode_products_) {
      const bool stacked = project_coarse_ && !scaled_projection_stacked_.empty();
      one_mode_projection(response_space_for(node_count),
                          stacked ? scaled_projection_stacked_ : scaled_projection_,
                          real_source, real_target, real_coarse_target);
    } else if (project_coarse_ && !scaled_projection_stacked_.empty()) {
      selected_gemm(transform_space_for(node_count), scaled_projection_stacked_,
                    real_source, real_stacked_target, 0.0, true);
      const auto& stacked = real_stacked_target;
      const auto& fine = real_target;
      const auto& coarse = real_coarse_target;
      const auto columns = problem_.n_columns();
      launch::parallel_for(
          "solprop_native.transform.projection.split_stacked",
          Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
          KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
            fine(column, time) = stacked(column, time);
            coarse(column, time) = stacked(columns + column, time);
          });
    } else {
      selected_gemm(transform_space_for(node_count), scaled_projection_, real_source,
                    real_target, 0.0, true);
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  if (use_chunked_blas_for_projection()) {
    Kokkos::Profiling::pushRegion(
        "solprop_native.transform.projection.kokkosblas_chunked");
    const auto rows = node_count * problem_.dims.n_pol_transform;
    const auto first_row = first_node * problem_.dims.n_pol_transform;
    const bool tail = node_count != problem_.dims.node_chunk;
    const bool last_chunk = first_node + node_count == problem_.dims.n_nodes;
    const bool stacked =
        project_coarse_ && !problem_.transform.projection_coarse_matrix.empty();
    if (problem_.dims.family == FieldFamily::envelope) {
      const auto polarisation = Kokkos::subview(
          node_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const auto source = scaled_projection_stacked_complex_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_stacked_complex",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      stacked_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
        if (last_chunk) {
          const auto& projected = stacked_target;
          const auto& fine = projected_target;
          const auto& coarse = coarse_target;
          const auto columns = problem_.n_columns();
          launch::parallel_for(
              "solprop_native.transform.projection.split_stacked_complex",
              Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        // The first half of the existing stacked scratch is a contiguous fine
        // matrix. Alias it when this trial stage does not need the coarse rows.
        const auto scratch =
            tail ? tail_projection_stacked_complex_ : chunk_projection_stacked_complex_;
        const ComplexView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_complex_ : chunk_projection_complex_)
                : ComplexView2(Kokkos::subview(
                      scratch, std::pair{std::size_t{0}, problem_.dims.n_modes},
                      Kokkos::ALL()));
        const auto source = scaled_projection_complex_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_complex",
            Range2(matrix_copy_space_for(node_count), {0, 0},
                   {problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      projected_target, first_chunk ? Complex{} : Complex{1.0, 0.0},
                      true);
      }
    } else {
      const auto polarisation = Kokkos::subview(
          real_source, std::make_pair(std::size_t{0}, rows), Kokkos::ALL());
      if (stacked) {
        const auto packed = tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const auto source = scaled_projection_stacked_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection_stacked",
            Range2(matrix_copy_space_for(node_count, true), {0, 0},
                   {2 * problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_stacked_target, first_chunk ? 0.0 : 1.0, true);
        if (last_chunk) {
          const auto& projected = real_stacked_target;
          const auto& fine = real_target;
          const auto& coarse = real_coarse_target;
          const auto columns = problem_.n_columns();
          launch::parallel_for(
              "solprop_native.transform.projection.split_stacked",
              Range2(execution_space_, {0, 0}, {columns, problem_.dims.n_to}),
              KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
                fine(column, time) = projected(column, time);
                coarse(column, time) = projected(columns + column, time);
              });
        }
      } else {
        const auto scratch =
            tail ? tail_projection_stacked_ : chunk_projection_stacked_;
        const RealView2 packed =
            scratch.empty()
                ? (tail ? tail_projection_ : chunk_projection_)
                : RealView2(Kokkos::subview(
                      scratch, std::pair{std::size_t{0}, problem_.dims.n_modes},
                      Kokkos::ALL()));
        const auto source = scaled_projection_;
        launch::parallel_for(
            "solprop_native.transform.pack_projection",
            Range2(matrix_copy_space_for(node_count), {0, 0},
                   {problem_.dims.n_modes, rows}),
            KOKKOS_LAMBDA(const std::size_t mode, const std::size_t row) {
              packed(mode, row) = source(mode, first_row + row);
            });
        selected_gemm(transform_space_for(node_count), packed, polarisation,
                      real_target, first_chunk ? 0.0 : 1.0, true);
      }
    }
    Kokkos::Profiling::popRegion();
    return;
  }
  Kokkos::Profiling::pushRegion("solprop_native.transform.projection.custom");
  const auto projection = scaled_projection_;
  const auto projection_coarse = scaled_projection_coarse_;
  const auto& polar_real = real_source;
  const auto& polar_complex = node_source;
  const auto& projected_real = real_target;
  const auto& projected_complex = projected_target;
  const auto& projected_coarse_real = real_coarse_target;
  const auto& projected_coarse_complex = coarse_target;
  const auto n_pol_state = problem_.dims.n_pol_state;
  const auto n_pol_transform = problem_.dims.n_pol_transform;
  const auto n_to = problem_.dims.n_to;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  const bool coarse = project_coarse_ && !projection_coarse.empty();
  launch::parallel_for(
      "solprop_native.transform.projection.custom",
      Range2(transform_space_for(node_count), {0, 0}, {problem_.n_columns(), n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        const std::size_t mode = column / n_pol_state;
        const std::size_t state_pol = column % n_pol_state;
        if (envelope) {
          Complex fine{};
          Complex rough{};
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const Complex value = polar_complex(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_complex(column, time) = fine;
            if (coarse)
              projected_coarse_complex(column, time) = rough;
          } else {
            projected_complex(column, time) += fine;
            if (coarse)
              projected_coarse_complex(column, time) += rough;
          }
        } else {
          double fine = 0.0;
          double rough = 0.0;
          for (std::size_t node = 0; node < node_count; ++node) {
            for (std::size_t transform_pol = 0; transform_pol < n_pol_transform;
                 ++transform_pol) {
              const std::size_t global_row =
                  (first_node + node) * n_pol_transform + transform_pol;
              const std::size_t local_row =
                  (node * n_pol_transform + transform_pol) * n_pol_state + state_pol;
              const double value = polar_real(local_row, time);
              fine += projection(mode, global_row) * value;
              if (coarse)
                rough += projection_coarse(mode, global_row) * value;
            }
          }
          if (first_chunk) {
            projected_real(column, time) = fine;
            if (coarse)
              projected_coarse_real(column, time) = rough;
          } else {
            projected_real(column, time) += fine;
            if (coarse)
              projected_coarse_real(column, time) += rough;
          }
        }
      });
  Kokkos::Profiling::popRegion();
}

void GeneralWorkspace::form_absorber() {
  const bool spatial = !problem_.spatial.absorber_matrix.empty();
  const auto modes = problem_.dims.n_modes;
  const auto polarisations = problem_.dims.n_pol_state;
  // Flatten (polarisation,time) together: the same scalar radial matrix acts
  // independently on each component. The views alias existing FFT work buffers.
  if (spatial && problem_.dims.family == FieldFamily::envelope) {
    using Matrix = Kokkos::View<Complex**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_complex_.data(), modes,
                       polarisations * problem_.dims.n_to);
    const Matrix output(absorber_complex_.data(), modes,
                        polarisations * problem_.dims.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_complex_, input, output,
                  Complex{}, true);
  } else if (spatial && problem_.dims.family == FieldFamily::field_real) {
    using Matrix = Kokkos::View<double**, Layout, MemorySpace,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const Matrix input(modal_real_.data(), modes, polarisations * problem_.dims.n_to);
    const Matrix output(absorber_real_.data(), modes,
                        polarisations * problem_.dims.n_to);
    selected_gemm(absorber_execution_space_, spatial_absorber_real_, input, output, 0.0,
                  true);
  }
  const auto rate = absorber_fine_;
  const auto spatial_matrix = spatial_absorber_real_;
  const auto modal_real = modal_real_;
  const auto modal_complex = modal_complex_;
  const auto absorber_real = absorber_real_;
  const auto absorber_complex = absorber_complex_;
  const auto family = problem_.dims.family;
  launch::parallel_for(
      "solprop_native.absorber.nonlinear_term",
      Range2(temporal_execution_space_, {0, 0},
             {problem_.n_columns(), problem_.dims.n_to}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t time) {
        if (family == FieldFamily::envelope) {
          absorber_complex(column, time) =
              (spatial ? absorber_complex(column, time) : Complex{}) -
              0.5 * rate(time) * modal_complex(column, time);
        } else if (family == FieldFamily::field_real) {
          absorber_real(column, time) = (spatial ? absorber_real(column, time) : 0.0) -
                                        0.5 * rate(time) * modal_real(column, time);
        } else {
          double transverse = 0.0;
          if (spatial)
            for (std::size_t source = 0; source < modes; ++source)
              transverse +=
                  spatial_matrix(column / polarisations, source) *
                  modal_complex(source * polarisations + column % polarisations, time)
                      .real();
          absorber_real(column, time) =
              transverse - 0.5 * rate(time) * modal_complex(column, time).real();
        }
      });
}

void GeneralWorkspace::finish_stage(const std::size_t stage) {
  const auto terms = stage_terms(stage);
  const double solution_weight = problem_.stepper.solution_weights[stage];
  const double error_weight = problem_.stepper.error_weights[stage];
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // The host's blocks of bins, as the stage's entry kernel takes them.
  if (terms.blocks_vectorise()) {
    const auto n = problem_.dims.n_omega;
    const std::size_t blocks = (n + stage_block - 1) / stage_block;
    launch::parallel_for(
        "solprop_native.stage.finish_accumulate",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                            problem_.n_columns() * blocks),
        KOKKOS_LAMBDA(const std::size_t item) {
          const std::size_t first = (item % blocks) * stage_block;
          terms.finish_block(stage, solution_weight, error_weight, item / blocks, first,
                             n - first < stage_block ? n - first : stage_block);
        });
    return;
  }
#endif
  launch::parallel_for(
      "solprop_native.stage.finish_accumulate",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        terms.finish(stage, solution_weight, error_weight, column, index);
      });
}

void GeneralWorkspace::set_flow_actions(const std::vector<Complex>& forward,
                                        const std::vector<Complex>& inverse,
                                        const std::vector<unsigned char>& identity) {
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  flow_->set_actions(forward, inverse, identity);
}

std::vector<Complex> GeneralWorkspace::adams_block_to_host() const {
  require_adams();
  if (adams_block_.extent(0) == 0)
    throw std::logic_error("the Adams history holds no coupled block");
  std::vector<Complex> values(adams_block_.size());
  const auto mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, adams_block_);
  std::copy(mirror.data(), mirror.data() + mirror.size(), values.begin());
  return values;
}

std::vector<Complex> GeneralWorkspace::adams_exponent_to_host() const {
  require_adams();
  std::vector<Complex> values(adams_exponent_.size());
  const auto mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, adams_exponent_);
  std::copy(mirror.data(), mirror.data() + mirror.size(), values.begin());
  return values;
}

std::vector<Complex> GeneralWorkspace::flow_actions_to_host(const bool inverse) const {
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->actions_to_host(inverse);
}

std::vector<double>
GeneralWorkspace::form_flow_actions(const std::size_t section, const double dz,
                                    const std::vector<double>& fractions) {
  if (!flow_)
    throw std::logic_error("this state carries no coupled flow");
  return flow_->form_actions(section, dz, fractions);
}

std::vector<double> GeneralWorkspace::weighted_error(const std::vector<double>& scale) {
  if (!problem_.stepper.weighted_controller)
    throw std::logic_error("this state has no weighted controller workspace");
  require_extent(scale.size(), problem_.dims.n_omega, "weighted controller scale");
  copy_vector(execution_space_, weighted_scale_, scale);
  const auto residual = residual_;
  const auto in_norm = norm_mask_;
  const auto native_scale = weighted_scale_;
  const auto partials = weighted_partials_;
  const auto n = problem_.dims.n_omega;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const int team_size = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE;
#else
  // A host team per column, sharing the threads when the columns are fewer.
  const int team_size =
      host_reduction_team_size(execution_space_, problem_.n_columns(), n);
#endif
  // On a device each column is `reduction_slots_` teams over its own bins; the
  // caller sums the slots.
  const auto slots = reduction_slots_;
  const TeamPolicy policy(execution_space_,
                          static_cast<int>(problem_.n_columns() * slots), team_size);
  launch::parallel_for(
      "solprop_native.step.weighted_error_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto league = static_cast<std::size_t>(team.league_rank());
        const auto column = league / slots;
        std::size_t first = 0;
        std::size_t end = 0;
        slot_range(n, slots, league % slots, first, end);
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(first),
                                    static_cast<int>(end)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              if (in_norm(index) == 0U)
                return;
              const Complex value = residual(column, index) / native_scale(index);
              update += squared_abs(value);
            },
            sum);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(league) = sum; });
      });
  execution_space_.fence("solprop_native weighted-error boundary");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials);
  std::vector<double> result(problem_.n_columns() * slots);
  for (std::size_t slot = 0; slot < result.size(); ++slot)
    result[slot] = host(slot);
  return result;
}

void GeneralWorkspace::scan_plasma_statistics(const std::size_t first_node,
                                              const std::size_t node_count,
                                              const RealView2& output,
                                              const bool diagnose,
                                              const double preionisation) {
  if (problem_.plasma.envelope) {
    if (diagnose)
      throw std::invalid_argument("the plasma integration diagnostics read the "
                                  "field-grid plasma column, and this problem "
                                  "carries none");
    for (std::size_t channel = 0; channel < problem_.plasma.n_channels; ++channel)
      envelope_statistics(channel, first_node, node_count, output, node_complex_,
                          problem_.effective_node_pol());
    return;
  }
  // solprop SEP 18: the band rule's statistics, and every problem's run diagnostics,
  // come from the band scan; the trapezoid's own scan below is unchanged.
  if (problem_.plasma.band || diagnose) {
    if (diagnose)
      team_band_statistics<true>(first_node, node_count, output, preionisation);
    else
      team_band_statistics<false>(first_node, node_count, output, preionisation);
    return;
  }
  const auto node_real = node_real_;
  const auto node_complex = node_complex_;
  const bool real_family = problem_.dims.family == FieldFamily::field_real;
  const auto coefficients = rate_coefficients_;
  const auto n_pol = problem_.effective_node_pol();
  const auto n_to = problem_.dims.n_to;
  const auto intervals = problem_.plasma.rate_intervals;
  const auto dt = problem_.grid.dt_fine;
  const auto log_field_min = problem_.plasma.log_field_min;
  const auto d_log_field = problem_.plasma.d_log_field;
  const auto field_min = problem_.plasma.field_min;
  const auto field_max = problem_.plasma.field_max;
  const auto rate_max = problem_.plasma.rate_max;
  if (problem_.plasma.n_channels > 1) {
    const auto parameters = channel_parameters_;
    const auto channel_coefficients = channel_coefficients_;
    const auto channels = problem_.plasma.n_channels;
    launch::parallel_for(
        "solprop_native.plasma.channel_statistics_scan",
        Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                            node_count),
        KOKKOS_LAMBDA(const std::size_t node) {
          for (std::size_t channel = 0; channel < channels; ++channel) {
            double exposure = 0.0;
            double previous_rate = 0.0;
            double largest_rate = 0.0;
            for (std::size_t time = 0; time < n_to; ++time) {
              const double first =
                  (real_family ? node_real(node * n_pol, time)
                               : node_complex(node * n_pol, time).real());
              const double second =
                  n_pol == 2
                      ? (real_family ? node_real(node * n_pol + 1, time)
                                     : node_complex(node * n_pol + 1, time).real())
                      : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              const double rate = ionisation_rate(
                  magnitude, channel_coefficients, parameters(channel, 0),
                  parameters(channel, 1), parameters(channel, 2),
                  parameters(channel, 3), parameters(channel, 4),
                  static_cast<std::size_t>(parameters(channel, 5)),
                  static_cast<std::size_t>(parameters(channel, 6)));
              if (time > 0)
                exposure += trapezoid(rate, previous_rate, dt);
              previous_rate = rate;
              if (rate > largest_rate)
                largest_rate = rate;
            }
            output(first_node + node, 2 * channel) = exposure;
            output(first_node + node, 2 * channel + 1) = largest_rate;
          }
        });
    return;
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const auto rates = plasma_rate_;
  const auto exposures = plasma_exposure_;
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(response_space_for(node_count), static_cast<int>(node_count),
                          static_cast<int>(response_team_size()));
  launch::parallel_for(
      "solprop_native.plasma.device_statistics_scan", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto node = static_cast<std::size_t>(team.league_rank());
        const auto base = node * n_pol;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value) {
              const auto time = static_cast<std::size_t>(time_value);
              const double first = (real_family ? node_real(base, time)
                                                : node_complex(base, time).real());
              const double second =
                  n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                            : node_complex(base + 1, time).real())
                             : 0.0;
              const double magnitude =
                  n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
              rates(node, time) =
                  ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                                  field_min, field_max, rate_max, intervals);
            });
        team.team_barrier();
        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& update, const bool final) {
              const auto time = static_cast<std::size_t>(time_value);
              if (time != 0)
                update += trapezoid(rates(node, time), rates(node, time - 1), dt);
              if (final)
                exposures(node, time) = update;
            });
        double largest_rate = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n_to)),
            [&](const int time_value, double& largest) {
              const auto time = static_cast<std::size_t>(time_value);
              if (rates(node, time) > largest)
                largest = rates(node, time);
            },
            Kokkos::Max<double>(largest_rate));
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          output(first_node + node, 0) = exposures(node, n_to - 1);
          output(first_node + node, 1) = largest_rate;
        });
      });
#else
  launch::parallel_for(
      "solprop_native.plasma.statistics_scan",
      Kokkos::RangePolicy<ExecutionSpace>(response_space_for(node_count), 0,
                                          node_count),
      KOKKOS_LAMBDA(const std::size_t node) {
        const auto base = node * n_pol;
        double exposure = 0.0;
        double previous_rate = 0.0;
        double largest_rate = 0.0;
        for (std::size_t time = 0; time < n_to; ++time) {
          const double first =
              (real_family ? node_real(base, time) : node_complex(base, time).real());
          const double second =
              n_pol == 2 ? (real_family ? node_real(base + 1, time)
                                        : node_complex(base + 1, time).real())
                         : 0.0;
          const double magnitude =
              n_pol == 2 ? Kokkos::hypot(first, second) : Kokkos::abs(first);
          const double rate =
              ionisation_rate(magnitude, coefficients, log_field_min, d_log_field,
                              field_min, field_max, rate_max, intervals);
          if (time > 0)
            exposure += trapezoid(rate, previous_rate, dt);
          previous_rate = rate;
          if (rate > largest_rate)
            largest_rate = rate;
        }
        output(first_node + node, 0) = exposure;
        output(first_node + node, 1) = largest_rate;
      });
#endif
}

void GeneralWorkspace::finish(const double dz) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (problem_.stepper.low_storage) {
    // The march accepted the step in place (solprop SEP 24 section 6.2): the
    // boundary treatments act on the state itself, and the stash a response pass
    // left is the last evaluation's, at a predicted field, not this one's.
    if (problem_.absorber.split_step)
      split_step(dz, current_);
    else
      mask(current_);
    stash_is_current_ = false;
    execution_space_.fence("solprop_native accepted-step boundary");
    return;
  }
  if (problem_.absorber.split_step)
    split_step(dz, advanced_);
  else
    mask(advanced_);
  std::swap(current_, advanced_);
  if (!stash_current_.empty())
    graph::deep_copy(execution_space_, stash_current_, stash_candidate_);
  stash_is_current_ = !stash_current_.empty();
  execution_space_.fence("solprop_native accepted-step boundary");
}

void GeneralWorkspace::finish_quadrature() {
  const Kokkos::Profiling::ScopedRegion region("solprop_native.quadrature.finish");
  if (!coarse_pending_)
    return;
  if (problem_.dims.family == FieldFamily::envelope)
    graph::execute(*forward_coarse_complex_plan_, projected_coarse_complex_,
                   projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  else if (forward_pack_coarse_plan_)
    graph::execute(*forward_pack_coarse_plan_, forward_pack_coarse_,
                   forward_pack_coarse_spectrum_, KokkosFFT::Normalization::none);
  else
    graph::execute(*forward_coarse_real_plan_, projected_coarse_real_,
                   projected_coarse_spectrum_, KokkosFFT::Normalization::none);
  if (held_plasma())
    held_transform(true, static_cast<bool>(forward_pack_coarse_plan_));
  coarse_pending_ = false;
}

void GeneralWorkspace::score_quadrature() {
  quadrature_error_ = Kokkos::Experimental::quiet_NaN_v<double>;
  quadrature_stale_ = false;
  if (quadrature_partials_.empty() ||
      (!problem_.kerr.nonlinear && !problem_.plasma.enabled && !dispersive_))
    return;
  finish_quadrature();
  // SEP 6's dispersive products join the embedded rule's spectrum once, when scored.
  if (dispersive_)
    dispersive_->finish_coarse();

  const auto fine = projected_spectrum_;
  const auto coarse = projected_coarse_spectrum_;
  const auto crop = crop_index_;
  const auto window = omega_window_;
  const auto prefactor_terms = prefactor_terms_;
  const auto prefactor_weights = stage_prefactor_;
  const auto partials = quadrature_partials_;
  const auto last = problem_.stepper.n_stages - 1;
  const auto n = problem_.dims.n_omega;
  const auto n_prefactor = problem_.linear.n_prefactor_terms;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
  using TeamMember = TeamPolicy::member_type;
  const TeamPolicy policy(execution_space_, static_cast<int>(problem_.n_columns()),
                          SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  launch::parallel_for(
      "solprop_native.transform.device_quadrature_partials", policy,
      KOKKOS_LAMBDA(const TeamMember& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              const Complex fine_value = weight * fine(column, crop(index));
              const Complex coarse_value = weight * coarse(column, crop(index));
              update += squared_abs(coarse_value - fine_value);
            },
            difference_sum);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index_value, double& update) {
              const auto index = static_cast<std::size_t>(index_value);
              Complex prefactor{};
              for (std::size_t term = 0; term < n_prefactor; ++term)
                prefactor += prefactor_weights(last * n_prefactor + term) *
                             prefactor_terms(term, column, index);
              const Complex weight = prefactor * window(index);
              update += squared_abs(weight * fine(column, crop(index)));
            },
            fine_sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          partials(0, column) = difference_sum;
          partials(1, column) = fine_sum;
        });
      });
#else
  launch::parallel_for(
      "solprop_native.transform.quadrature_partials",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
      KOKKOS_LAMBDA(const std::size_t column) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t index = 0; index < n; ++index) {
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term)
            prefactor += prefactor_weights(last * n_prefactor + term) *
                         prefactor_terms(term, column, index);
          const Complex weight = prefactor * window(index);
          const Complex fine_value = weight * fine(column, crop(index));
          const Complex coarse_value = weight * coarse(column, crop(index));
          fine_sum += squared_abs(fine_value);
          difference_sum += squared_abs(coarse_value - fine_value);
        }
        partials(0, column) = difference_sum;
        partials(1, column) = fine_sum;
      });
#endif
  // The columns' sums in their order, on the device: the last score and the peak
  // since the backend last read it.
  const auto scores = quadrature_scores_;
  const auto columns = problem_.n_columns();
  launch::parallel_for(
      "solprop_native.transform.quadrature_score",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        double difference_sum = 0.0;
        double fine_sum = 0.0;
        for (std::size_t column = 0; column < columns; ++column) {
          difference_sum += partials(0, column);
          fine_sum += partials(1, column);
        }
        double value = Kokkos::Experimental::quiet_NaN_v<double>;
        if (fine_sum != 0.0)
          value = Kokkos::sqrt(difference_sum / fine_sum);
        scores(0) = value;
        if (fine_sum != 0.0 && !(scores(1) >= value))
          scores(1) = value;
      });
  quadrature_stale_ = true;
}

double GeneralWorkspace::quadrature_error() const {
  if (quadrature_stale_) {
    const auto last = Kokkos::subview(quadrature_scores_, 0);
    Kokkos::deep_copy(execution_space_, quadrature_error_, last);
    execution_space_.fence("solprop_native quadrature score");
    quadrature_stale_ = false;
  }
  return quadrature_error_;
}

double GeneralWorkspace::quadrature_peak() {
  if (quadrature_scores_.extent(0) == 0)
    return Kokkos::Experimental::quiet_NaN_v<double>;
  double peak = 0.0;
  const auto stored = Kokkos::subview(quadrature_scores_, 1);
  graph::deep_copy(execution_space_, peak, stored);
  execution_space_.fence("solprop_native quadrature peak");
  graph::deep_copy(execution_space_, stored, Kokkos::Experimental::quiet_NaN_v<double>);
  return peak;
}

void GeneralWorkspace::promote_fsal() {
  score_quadrature();
  if (!pending_fsal_)
    return;
  const auto stages = stages_;
  const auto whole = whole_propagator_;
  const auto recycled = recycled_;
  const auto last = problem_.stepper.n_stages - 1;
  launch::parallel_for(
      "solprop_native.stage.promote_fsal",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        recycled(column, index) = whole(column, index) * stages(last, column, index);
      });
  has_recycled_ = true;
  pending_fsal_ = false;
}

void GeneralWorkspace::clear_fsal() {
  has_recycled_ = false;
  pending_fsal_ = false;
  coarse_pending_ = false;
  if (!stash_current_.empty())
    graph::deep_copy(execution_space_, stash_current_, 0.0);
  stash_is_current_ = false;
}

void GeneralWorkspace::mask(const ComplexView2& values) {
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.mask.accepted",
      Range2(execution_space_, {0, 0}, {problem_.n_columns(), problem_.dims.n_omega}),
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (active(index) == 0U)
          values(column, index) = Complex{};
      });
}
