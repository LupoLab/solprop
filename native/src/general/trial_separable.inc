// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The separable geometries' stage path (free space and the hybrid planar kind):
// transverse chunks and transforms, the stage input, packing and nodes, the finishing
// transform, evaluate_separable_stage, the moves between time and frequency, and the
// volume, spatial and separable cores.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

TransverseChunk
GeneralWorkspace::transverse_chunk(const std::size_t first_unit) const noexcept {
  const auto units =
      std::min(transverse_units_per_chunk(), transverse_units() - first_unit);
  if (problem_.slab_geometry())
    return {.first_row = first_unit * problem_.spatial.n_p,
            .rows = units * problem_.spatial.n_p,
            .first_node = first_unit * problem_.spatial.n_x,
            .nodes = units * problem_.spatial.n_x};
  return {
      .first_row = first_unit, .rows = units, .first_node = first_unit, .nodes = units};
}

void GeneralWorkspace::transverse_forward() {
  if (slab_forward_)
    graph::execute(*slab_forward_, transverse_, transverse_,
                   KokkosFFT::Normalization::none);
  else
    graph::execute(*transverse_forward_, transverse_, transverse_,
                   KokkosFFT::Normalization::none);
}

void GeneralWorkspace::transverse_inverse() { transverse_inverse(transverse_); }

// The in-place inverse transverse plan on another volume of the same extents: the
// plan's own scratch or the magnetic closure's (solprop SEP 34).
void GeneralWorkspace::transverse_inverse(const ComplexView3& scratch) {
  if (slab_inverse_)
    graph::execute(*slab_inverse_, scratch, scratch, KokkosFFT::Normalization::none);
  else
    graph::execute(*transverse_inverse_, scratch, scratch,
                   KokkosFFT::Normalization::none);
}

void GeneralWorkspace::separable_stage_input(const std::size_t stage, const double dz) {
  const auto current = current_;
  const auto stages = stages_;
  const auto tableau = tableau_;
  const auto spatial = spatial_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const auto whole = whole_propagator_;
  const auto index = stage_index();
  const auto n_stages = problem_.stepper.n_stages;
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double inverse_area_count = inverse_transverse_count();
  // The stage input at one element, given its spatial value and its propagator.
  const auto input = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                   const Complex linear, const Complex propagator) {
    Complex accumulated{};
    for (std::size_t previous = 0; previous < stage; ++previous)
      accumulated +=
          tableau(stage * n_stages + previous) * stages(previous, column, omega);
    Complex value = propagator * (current(column, omega) + dz * accumulated);
    if (add_noise) {
      const double angle = at(stage) * linear.imag();
      value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
    }
    scratch.data()[index.cell(column, omega)] = value * inverse_area_count;
  };
  // One element's propagators: devices, and a host whose vector maths is the system
  // libm, where the per-element form is the faster (solprop ADR-059).
  const auto element =
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto values = spatial.value(index.mode(column), omega, active(omega) != 0U);
    input(column, omega, values.linear, propagator_exp(delta(stage) * values.linear));
    if (stage + 1 == n_stages)
      whole(column, omega) = propagator_exp(delta(n_stages) * values.linear);
  };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (vmath::vectorised) {
    // Each tile's propagators in one pass of the vector maths, then the elements in
    // `crossing_for`'s order: the propagators' cosines and sines took a fifth of a
    // one-thread `line` run one element at a time (solprop issue #338).
    const bool last = stage + 1 == n_stages;
    crossing_tiles(
        "solprop_native.free.stage_input", execution_space_, problem_.n_columns(),
        problem_.dims.n_omega, [=](const CrossingTile& tile) {
          PropagatorTile step;
          PropagatorTile finish;
          Complex linear[PropagatorTile::capacity];
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            linear[k] =
                spatial.value(index.mode(column), omega, active(omega) != 0U).linear;
            const Complex argument = delta(stage) * linear[k];
            step.real[k] = argument.real();
            step.imag[k] = argument.imag();
            if (last) {
              const Complex whole_argument = delta(n_stages) * linear[k];
              finish.real[k] = whole_argument.real();
              finish.imag[k] = whole_argument.imag();
            }
          });
          step.form(tile.size());
          if (last)
            finish.form(tile.size());
          tile.visit<false>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            input(column, omega, linear[k], step[k]);
            if (last)
              whole(column, omega) = finish[k];
          });
        });
    return;
  }
#endif
  crossing_for("solprop_native.free.stage_input", execution_space_,
               problem_.n_columns(), problem_.dims.n_omega, element);
}

void GeneralWorkspace::separable_load_current() {
  const auto current = current_;
  const auto scratch = transverse_;
  const auto index = register_index();
  const double scale = inverse_transverse_count();
  crossing_for(
      "solprop_native.free.current_transverse_input", execution_space_,
      problem_.n_columns(), problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] = scale * current(column, omega);
      });
  transverse_inverse();
}

void GeneralWorkspace::separable_pack_chunk(const TransverseChunk& chunk,
                                            const bool material) {
  const auto fine = fine_spectrum_;
  // Every crop bin of the chunk's rows is written below; only the bins outside the
  // crop need zeros, each chunk, since the inverse transform may destroy its input.
  // Rows past a short last chunk keep earlier finite values nothing reads back.
  if (!separable_gaps_.empty()) {
    const auto gaps = separable_gaps_;
    cartesian_for(
        "solprop_native.free.zero_fine_gaps", execution_space_,
        chunk.rows * problem_.dims.n_pol_state, gaps.extent(0),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t gap) {
          fine(row, gaps(gap)) = Complex{};
        });
  }
  const auto scratch = transverse_;
  const auto crop = crop_index_;
  const auto n_pol = problem_.dims.n_pol_state;
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto first = chunk.first_row;
  const auto n = problem_.dims.n_omega;
  const auto n_to = problem_.dims.n_to;
  const bool analytic = problem_.dims.family == FieldFamily::field_analytic;
  // Free space's material field carries the pad scale here; the hybrid planar kind's
  // slab synthesis carries it (solprop's `Transform.synthesis`).
  const double scale = 1.0 / static_cast<double>(material && !problem_.slab_geometry()
                                                     ? problem_.dims.n_t
                                                     : problem_.dims.n_to);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  tiled_crossing(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, true,
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        return layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol);
      },
      KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                    const Complex value) {
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) = factor * scale * value;
      });
#else
  // The chunk's row `row` is the state's column `first_column + row`.
  const auto index = register_index();
  const auto first_column = first * n_pol;
  crossing_for<false, false, true>(
      "solprop_native.free.pack_temporal_chunk", execution_space_, chunk.rows * n_pol,
      n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
        // The analytic signal doubles the bins strictly between zero and the fine
        // grid's Nyquist bin, the same rule as the modal pad and the reference.
        const double factor =
            analytic && omega > 0 && 2 * crop(omega) < n_to ? 2.0 : 1.0;
        fine(row, crop(omega)) =
            factor * scale * scratch.data()[index.cell(first_column + row, omega)];
      });
#endif
}

// The chunk's physical nodes from its temporal rows: free space's are the rows
// themselves; the hybrid planar kind's are the slab synthesis of each y row.
void GeneralWorkspace::separable_nodes(const TransverseChunk& chunk) {
  const bool real_field = problem_.dims.family == FieldFamily::field_real;
  const auto n_to = problem_.dims.n_to;
  const auto n_pol = problem_.dims.n_pol_state;
  if (problem_.slab_geometry()) {
    const auto units = chunk.nodes / problem_.spatial.n_x;
    if (real_field)
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_real_, node_real_, units, n_to);
    else
      slab_contract("solprop_native.slab.synthesis", execution_space_, synthesis_,
                    modal_complex_, node_complex_, units, n_to);
    return;
  }
  if (separable_aliased_)
    return;
  const auto time = modal_real_;
  const auto nodes = node_real_;
  const auto complex_time = modal_complex_;
  const auto complex_nodes = node_complex_;
  cartesian_for(
      "solprop_native.free.physical_columns", execution_space_, chunk.rows * n_pol,
      n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
        if (real_field)
          nodes(row, index) = time(row, index);
        else
          complex_nodes(row, index) = complex_time(row, index);
      });
}

// The hybrid planar kind's slab projection of a chunk's node source onto its y rows'
// columns: the polarisation, or one envelope-plasma current stream.
void GeneralWorkspace::slab_project(const TransverseChunk& chunk,
                                    const EnvelopeStream* stream) {
  const auto units = chunk.nodes / problem_.spatial.n_x;
  const auto n_to = problem_.dims.n_to;
  if (stream != nullptr)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  stream->node, stream->projected, units, n_to);
  else if (problem_.dims.family == FieldFamily::envelope)
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_complex_, projected_complex_, units, n_to);
  else
    slab_contract("solprop_native.slab.projection", execution_space_, projection_,
                  polarisation_real_, projected_real_, units, n_to);
}

void GeneralWorkspace::separable_finish_transform(const std::size_t stage,
                                                  const bool material) {
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto stages = stages_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto delta = stage_phi_delta_;
  const auto index = stage_index();
  // The finish at one element, given its prefactor and its inverse propagator.
  const auto finish = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                    const Complex prefactor, const Complex inverse) {
    Complex value = scratch.data()[index.cell(column, omega)];
    value *=
        material ? prefactor * window(omega) : Complex(active(omega) != 0U ? 1.0 : 0.0);
    value *= inverse;
    if (material)
      stages(stage, column, omega) = value;
    else
      stages(stage, column, omega) += value;
  };
  // One element, on devices and without host vector maths (as the stage input).
  const auto element =
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto values = spatial.value(index.mode(column), omega, active(omega) != 0U);
    finish(column, omega, values.prefactor,
           propagator_exp(-delta(stage) * values.linear));
  };
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (vmath::vectorised) {
    // As the stage input: each tile's inverse propagators in one pass (issue #338).
    crossing_tiles(
        "solprop_native.free.finish_transform", execution_space_, problem_.n_columns(),
        problem_.dims.n_omega, [=](const CrossingTile& tile) {
          PropagatorTile inverse;
          Complex prefactor[PropagatorTile::capacity];
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            const auto values =
                spatial.value(index.mode(column), omega, active(omega) != 0U);
            prefactor[k] = values.prefactor;
            const Complex argument = -delta(stage) * values.linear;
            inverse.real[k] = argument.real();
            inverse.imag[k] = argument.imag();
          });
          inverse.form(tile.size());
          tile.visit<true>([&](const std::size_t column, const std::size_t omega) {
            const auto k = tile.at(column, omega);
            finish(column, omega, prefactor[k], inverse[k]);
          });
        });
    return;
  }
#endif
  crossing_for<false, false, true>("solprop_native.free.finish_transform",
                                   execution_space_, problem_.n_columns(),
                                   problem_.dims.n_omega, element);
}

// Free space (solprop `NumpyRefBackend._nonlinear_free`) and the hybrid planar kind
// (`_nonlinear_slab`): the stage input to the transverse spectrum, the inverse
// transverse transform, then chunk by chunk the temporal rows to time, the material
// response at the physical nodes (for the planar kind, synthesised from and projected
// back onto the rows in time) or the in-stage absorber on the rows, and back; the
// forward transverse transform and each column's prefactor and propagator.
void GeneralWorkspace::evaluate_separable_stage(const std::size_t stage,
                                                const double dz,
                                                const bool accumulate) {
  auto* const dispersive = separable_dispersive_.get();
  const bool nonlinear = problem_.kerr.nonlinear || problem_.plasma.enabled ||
                         problem_.raman.enabled || dispersive != nullptr;
  const auto stages = stages_;
  const auto solution = solution_;
  const auto error = error_;
  const auto n = problem_.dims.n_omega;
  // A material finish assigns the stage row; only an absorber-only or a linear stage
  // accumulates into (or leaves) a cleared row.
  if (!nonlinear)
    cartesian_for(
        "solprop_native.free.clear_stage", execution_space_, problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          stages(stage, column, omega) = Complex{};
        });
  separable_stage_input(stage, dz);
  for (const bool material : {true, false}) {
    if ((material && !nonlinear) || (!material && !problem_.absorber.in_stage))
      continue;
    if (!material)
      separable_stage_input(stage, dz);
    separable_core(stage, material);
    separable_finish_transform(stage, material);
  }
  if (!accumulate)
    return;
  const double weight = problem_.stepper.solution_weights[stage];
  const double error_weight = problem_.stepper.error_weights[stage];
  cartesian_for(
      "solprop_native.free.accumulate_stage", execution_space_, problem_.n_columns(), n,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        const Complex value = stages(stage, column, omega);
        if (stage == 0) {
          solution(column, omega) = weight * value;
          error(column, omega) = error_weight * value;
        } else {
          solution(column, omega) += weight * value;
          error(column, omega) += error_weight * value;
        }
      });
}

void GeneralWorkspace::separable_to_time(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.dims.n_pol_state == temporal_tail_rows_;
  if (problem_.dims.family == FieldFamily::field_real) {
    if (tail)
      temporal_tail_inverse_real_.execute();
    else
      graph::execute(*inverse_real_plan_, fine_spectrum_, modal_real_,
                     KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_inverse_complex_.execute();
    else
      graph::execute(*inverse_complex_plan_, fine_spectrum_, modal_complex_,
                     KokkosFFT::Normalization::none);
  }
}

void GeneralWorkspace::separable_to_frequency(const TransverseChunk& chunk) {
  const bool tail = temporal_tail_rows_ != 0 &&
                    chunk.rows * problem_.dims.n_pol_state == temporal_tail_rows_;
  if (problem_.dims.family == FieldFamily::envelope) {
    if (tail)
      temporal_tail_forward_complex_.execute();
    else
      graph::execute(*forward_complex_plan_, projected_complex_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
  } else {
    if (tail)
      temporal_tail_forward_real_.execute();
    else
      graph::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                     KokkosFFT::Normalization::none);
  }
}

// `volume_kerr_`'s stage core: the scratch's (omega, y, x) volume to time and the
// plane at once, the Kerr polarisation in place sample by sample -- the device
// column's arithmetic (`device_response`), the pad scale 1/n_t that the chunk's pack
// would apply taken with the field -- and back. The rank-3 transform is the chunks'
// temporal transforms and the plane's together, the same DFT to rounding, without the
// pack's and the crop's transposes.
void GeneralWorkspace::separable_volume_core(const std::size_t stage) {
  Kokkos::Profiling::pushRegion("solprop_native.free.volume_core");
  const auto scratch = volume_order_ ? volume_scratch_ : transverse_;
  graph::execute(*volume_inverse_, scratch, scratch, KokkosFFT::Normalization::none);
  const auto window = time_window_;
  const auto kerr_density = stage_kerr_density_;
  const double kerr_coefficient = problem_.kerr.coefficient;
  const double self_factor = problem_.kerr.self_factor;
  const double conjugate_factor = problem_.kerr.conjugate_factor;
  const double scale = 1.0 / static_cast<double>(problem_.dims.n_t);
  const auto plane = transverse_.extent(1) * transverse_.extent(2);
  const bool time_fastest = volume_order_;
  const auto n_t = problem_.dims.n_t;
  launch::parallel_for(
      "solprop_native.free.volume_kerr",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, scratch.size()),
      KOKKOS_LAMBDA(const std::size_t item) {
        const std::size_t time = time_fastest ? item % n_t : item / plane;
        Complex& cell = scratch.data()[item];
        const Complex value = scale * cell;
        const double modulus = squared_abs(value);
        const Complex square = value * value;
        const double factor = kerr_coefficient * kerr_density(stage);
        const double weighted_real = conjugate_factor * square.real();
        const double weighted_imag = conjugate_factor * square.imag();
        const double weighted_self = self_factor * modulus;
        const Complex result(
            factor * (weighted_self * value.real() +
                      (weighted_real * value.real() + weighted_imag * value.imag())),
            factor * (weighted_self * value.imag() +
                      (weighted_imag * value.real() - weighted_real * value.imag())));
        cell = result * window(time);
      });
  graph::execute(*volume_forward_, scratch, scratch, KokkosFFT::Normalization::none);
  Kokkos::Profiling::popRegion();
}

// The in-stage collar's term alone (solprop SEP 40 section 4.7): with no temporal
// rate beside it, the rate depends on the column alone and commutes with the temporal
// transforms, so the term is the rate times the scratch's spectrum, in the transverse
// physical space, and the pass takes only the transverse pair. The temporal pair
// would return the crop's bins to rounding; the bins it would drop lie outside the
// active band, which the finish masks.
void GeneralWorkspace::separable_spatial_core() {
  Kokkos::Profiling::pushRegion("solprop_native.free.spatial_core");
  transverse_inverse();
  const auto index = register_index();
  const auto spatial = spatial_;
  const auto scratch = transverse_;
  const auto n_pol = problem_.dims.n_pol_state;
  cartesian_for(
      "solprop_native.free.absorber_spectrum", execution_space_, problem_.n_columns(),
      problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] *=
            -0.5 * spatial.absorber(column / n_pol);
      });
  transverse_forward();
  Kokkos::Profiling::popRegion();
}

// The separable stage's core (solprop SEP 24 Q5): from its input in the transverse
// scratch, the material response (or, `material` false, the in-stage absorber's term)
// back into the scratch, in frequency, chunk by chunk through the temporal transforms.
// Shared by the Runge-Kutta stage and the low-storage march's passes, which differ in
// how they form the input and fold the output.
void GeneralWorkspace::separable_core(const std::size_t stage, const bool material) {
  if (material && volume_kerr_) {
    separable_volume_core(stage);
    return;
  }
  if (!material && spatial_only_absorber_) {
    separable_spatial_core();
    return;
  }
  // SEP 6's dispersive response runs chunk by chunk beside the material's response
  // column, or alone in its place (`SeparableDispersive`).
  auto* const dispersive = separable_dispersive_.get();
  const bool alone = dispersive != nullptr && dispersive->alone();
  const auto n_pol = problem_.dims.n_pol_state;
  // The host's scratch positions without a division (`RegisterIndex`).
  [[maybe_unused]] const auto index = register_index();
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto n = problem_.dims.n_omega;
  const auto n_to = problem_.dims.n_to;
  const auto real_time = modal_real_;
  const auto complex_time = modal_complex_;
  const auto polarisation_real = polarisation_real_;
  const auto polarisation_complex = polarisation_complex_;
  const auto projected_real = projected_real_;
  const auto projected_complex = projected_complex_;
  const auto temporal_rate = absorber_fine_;
  const auto spatial = spatial_;
  const auto spectrum = projected_spectrum_;
  const auto crop = crop_index_;
  const auto scratch = transverse_;
  const bool real_field = problem_.dims.family == FieldFamily::field_real;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  const bool slab = problem_.slab_geometry();
  // Profiling regions name the core's stages for a Kokkos Tools timer (SEP 28 phase
  // 0); with no tool loaded each is one test of a flag. The kernels inside carry
  // their own labels.
  Kokkos::Profiling::pushRegion("solprop_native.free.core");
  if (material && dispersive != nullptr)
    dispersive->begin(stage);
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_inverse");
  transverse_inverse();
  Kokkos::Profiling::popRegion();
  for (std::size_t unit = 0; unit < transverse_units();
       unit += transverse_units_per_chunk()) {
    const auto chunk = transverse_chunk(unit);
    const auto first = chunk.first_row;
    Kokkos::Profiling::pushRegion("solprop_native.free.pack");
    separable_pack_chunk(chunk, material);
    Kokkos::Profiling::popRegion();
    // The response filters the packed spectrum first: the inverse transform destroys
    // it. In place, that transform gives the filtered field.
    if (material && dispersive != nullptr)
      dispersive->prepare(chunk.nodes);
    Kokkos::Profiling::pushRegion("solprop_native.free.to_time");
    separable_to_time(chunk);
    Kokkos::Profiling::popRegion();
    // The projected rows are overwritten below by the polarisation (or are the
    // polarisation, aliased) or by the absorber's term; only a planar projection or a
    // response that projects nothing of the stage's needs them zeroed first.
    const bool overwrites =
        !slab &&
        (!material || !alone || (dispersive != nullptr && dispersive->borrowed()));
    Kokkos::Profiling::pushRegion("solprop_native.free.clear_projected");
    if (!overwrites) {
      if (envelope)
        graph::deep_copy(execution_space_, projected_complex_, Complex{});
      else
        graph::deep_copy(execution_space_, projected_real_, 0.0);
    }
    Kokkos::Profiling::popRegion();
    if (material) {
      separable_nodes(chunk);
      Kokkos::Profiling::pushRegion("solprop_native.free.response");
      if (!alone)
        response(stage, chunk.first_node, chunk.nodes);
      if (problem_.plasma.envelope)
        envelope_response(stage, chunk.first_node, chunk.nodes, &chunk);
      if (problem_.raman.enabled)
        convolve_raman(stage, chunk.nodes);
      if (dispersive != nullptr)
        dispersive->respond(chunk.nodes);
      if (held_plasma())
        held_response(stage, chunk.first_node, chunk.nodes, &chunk);
      Kokkos::Profiling::popRegion();
      // Alone with products of its own, nothing of the stage's is projected.
      const bool stage_projects = !alone || dispersive->borrowed();
      if (stage_projects && slab)
        slab_project(chunk, nullptr);
      else if (stage_projects && !separable_aliased_)
        cartesian_for(
            "solprop_native.free.polarisation_columns", execution_space_,
            chunk.rows * n_pol, n_to,
            KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
              if (envelope)
                projected_complex(row, time) = polarisation_complex(row, time);
              else
                projected_real(row, time) = polarisation_real(row, time);
            });
    } else {
      // The collar's rate depends on y alone on the planar kind, so it acts on the
      // rows, whose time signals are the slab amplitudes' (solprop's reference).
      cartesian_for(
          "solprop_native.free.absorber_columns", execution_space_, chunk.rows * n_pol,
          n_to, KOKKOS_LAMBDA(const std::size_t row, const std::size_t time) {
            const double rate =
                -0.5 * (spatial.absorber(first + row / n_pol) + temporal_rate(time));
            if (envelope)
              projected_complex(row, time) = rate * complex_time(row, time);
            else
              projected_real(row, time) =
                  rate *
                  (real_field ? real_time(row, time) : complex_time(row, time).real());
          });
    }
    const bool responds = material && dispersive != nullptr;
    if (responds)
      dispersive->finish();
    Kokkos::Profiling::pushRegion("solprop_native.free.to_frequency");
    if (!responds || !dispersive->replaces_transform()) {
      separable_to_frequency(chunk);
    }
    Kokkos::Profiling::popRegion();
    // The response's spectrum joins the chunk's, before the envelope currents'.
    if (responds)
      dispersive->combine();
    if (material && problem_.plasma.envelope)
      envelope_transform();
    if (material && held_plasma())
      held_transform();
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    tiled_crossing(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, false,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          return spectrum(row, crop(omega));
        },
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega,
                      const Complex value) {
          layout.cell(scratch, omega * n_pol + row % n_pol, first + row / n_pol) =
              value;
        });
#else
    const auto first_column = first * n_pol;
    crossing_for<true>(
        "solprop_native.free.crop_temporal_chunk", execution_space_, chunk.rows * n_pol,
        n, KOKKOS_LAMBDA(const std::size_t row, const std::size_t omega) {
          stream_store(scratch.data()[index.cell(first_column + row, omega)],
                       spectrum(row, crop(omega)));
        });
#endif
  }
  Kokkos::Profiling::pushRegion("solprop_native.free.transverse_forward");
  transverse_forward();
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}
