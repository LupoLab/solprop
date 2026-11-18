// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The low-storage RK4IP march (solprop SEP 24): its register folds, the step, the
// flows it forms and its separable and modal passes.
//
// Part of trial.cpp's translation unit (included there, in order; never
// compiled alone).

// --- solprop SEP 24 phase 3: the low-storage RK4IP march -----------------------------
//
// Classical RK4 in the interaction picture anchored at the step's midpoint (solprop's
// backends/low_storage.py, SEP 24 section 4.3). The state is the register A; B
// accumulates the bracket of the accepted step and C holds the next pass's lab-frame
// input, unless it is elided (solprop's `low_storage_stage_register`): then each of the
// first three finishes writes that input straight into the transverse scratch. A step
// is four passes, each forming its input in the route's scratch, running the stage core
// the Runge-Kutta march shares (`separable_core`, `modal_core`; SEP 24 Q5) and folding
// the source into the registers per element:
//
//   pass 1, F1 at z_n:  A <- P A; B <- A + h/6 P S; C <- A + h/2 P S
//   pass 2, F2 at z_m:  B <- B + h/3 S; C <- A + h/2 S
//   pass 3, F3 at z_m:  B <- B + h/3 S; C <- Q (A + h S)
//   pass 4, F4 at end:  A <- Q B + h/6 S
//
// Every pass reads the source at its own position, so nothing carries it back to an
// anchor, and only forward flows are applied. The step is accepted in place; the
// caller's finish masks it or applies the split-step collar.

namespace {

constexpr int fold_complete = 0;
constexpr int fold_exchange = 1;
constexpr int fold_second = 2;
constexpr int term_none = 0;
constexpr int term_material = 1;
constexpr int term_absorber = 2;

// The scalar row each pass reads: the step's start, its midpoint twice, its end.
constexpr std::size_t low_storage_rows[4] = {0, 1, 1, 2};

// One evaluation's fold into the registers at one element: solprop's
// `numpy_ref.engines.NumpyLowStorage._step`, the finish column of SEP 24 section 4.3.
// A pass that evaluates twice -- the material response, then the in-stage absorber's
// term, from the same input -- folds the first at the exchange (pass 3's C left
// unflowed) and the second after it, whose sum is the pass's.
KOKKOS_INLINE_FUNCTION void
low_storage_fold(const int stage, const int mode, const double step,
                 const Complex source, Complex& state, Complex& accumulator,
                 Complex& stage_register, const Complex first, const Complex second) {
  if (stage == 0) {
    const Complex slope = first * source;
    if (mode == fold_second) {
      accumulator = accumulator + (step / 6.0) * slope;
      stage_register = stage_register + (0.5 * step) * slope;
      return;
    }
    const Complex moved = first * state;
    state = moved;
    accumulator = moved + (step / 6.0) * slope;
    stage_register = moved + (0.5 * step) * slope;
    return;
  }
  if (stage == 1) {
    accumulator = accumulator + (step / 3.0) * source;
    stage_register =
        (mode == fold_second ? stage_register : state) + (0.5 * step) * source;
    return;
  }
  if (stage == 2) {
    accumulator = accumulator + (step / 3.0) * source;
    if (mode == fold_complete)
      stage_register = second * (state + step * source);
    else if (mode == fold_exchange)
      stage_register = state + step * source;
    else
      stage_register = second * (stage_register + step * source);
    return;
  }
  if (mode == fold_second) {
    state = state + (step / 6.0) * source;
    return;
  }
  state = second * accumulator + (step / 6.0) * source;
}

// A register traversal over the state's elements. The last fold of a step also reduces
// the accepted state's squared norm per column into `partials(0, column)`, one team per
// column as the Runge-Kutta accept reduces its own (its sums associate as the team
// splits them, solprop SEP 25).
template <bool Crossing = false, class Element>
void register_traversal(const char* label, const ExecutionSpace& execution,
                        const std::size_t columns, const std::size_t n,
                        const RealView2& partials, const bool reduce,
                        const Element& element) {
  const auto traversal_for = [](const char* name, const ExecutionSpace& space,
                                const std::size_t rows, const std::size_t width,
                                const auto& body) {
    if constexpr (Crossing)
      crossing_for(name, space, rows, width, body);
    else
      cartesian_for(name, space, rows, width, body);
  };
  if (!reduce) {
    traversal_for(
        label, execution, columns, n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
          static_cast<void>(element(column, index));
        });
    return;
  }
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  if constexpr (Crossing) {
    // Blocks of adjacent columns, each column's sum in ascending bin order.
    constexpr std::size_t block = crossing_tile_columns;
    constexpr std::size_t bins = crossing_tile_bins;
    const std::size_t blocks = (columns + block - 1) / block;
    launch::parallel_for(
        label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, blocks),
        KOKKOS_LAMBDA(const std::size_t which) {
          const std::size_t first = which * block;
          const std::size_t count = columns - first < block ? columns - first : block;
          double totals[block] = {};
          for (std::size_t start = 0; start < n; start += bins) {
            const std::size_t stop = n - start < bins ? n : start + bins;
            for (std::size_t index = start; index < stop; ++index)
              for (std::size_t offset = 0; offset < count; ++offset)
                totals[offset] += element(first + offset, index);
          }
          for (std::size_t offset = 0; offset < count; ++offset)
            partials(0, first + offset) = totals[offset];
        });
    return;
  }
#endif
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  const Policy policy(execution, static_cast<int>(columns),
                      SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE);
  launch::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              update += element(column, static_cast<std::size_t>(index));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { partials(0, column) = total; });
      });
#else
  const Policy policy(execution, static_cast<int>(columns),
                      host_reduction_team_size(execution, columns, n));
  launch::parallel_for(
      label, policy, KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, n),
            [&](const std::size_t index, double& update) {
              update += element(column, index);
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
#endif
}

} // namespace

double GeneralWorkspace::low_storage_step(const double dz, const TrialScalars& scalars,
                                          const bool rebuild_table) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!problem_.stepper.low_storage)
    throw std::logic_error("this native state was not built for the low-storage march");
  if (!(dz > 0.0) || !std::isfinite(dz))
    throw std::invalid_argument("the low-storage step must be positive and finite");
  if (problem_.stepper.n_stages < 3)
    throw std::invalid_argument("the low-storage march reads three scalar rows");
  if (rebuild_table && problem_.stepper.low_storage_flows != 1)
    throw std::logic_error("only a constant operator's table is rebuilt");
  if (problem_.stepper.low_storage_flows == 1 && !rebuild_table && !table_ready_)
    throw std::logic_error("the low-storage table has not been built since the initial "
                           "field was loaded");
  require_trial_scalars(scalars);
  upload_trial(scalars);
  if (general_)
    general_->set_weights(scalars.transform_weights);
  low_storage_flows_formed(rebuild_table);
  if (rebuild_table)
    table_ready_ = true;
  for (std::size_t pass = 0; pass < 4; ++pass) {
    const auto row = low_storage_rows[pass];
    if (problem_.raman.enabled)
      prepare_raman_stage(
          row, scalars.raman_rates.data() + row * problem_.raman.n_terms,
          problem_.raman.n_components > 1
              ? scalars.raman_densities.data() + row * problem_.raman.n_components
              : nullptr);
    if (problem_.separable_geometry())
      low_storage_separable_pass(pass, dz);
    else
      low_storage_modal_pass(pass, dz);
  }
  // No Runge-Kutta stage exists to observe or recycle.
  stages_ready_ = false;
  stash_is_current_ = false;
  execution_space_.fence("solprop_native low-storage step");
  double total = 0.0;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // One number crosses, not the whole of the partials' table.
  const auto partials = partials_;
  launch::parallel_reduce(
      "solprop_native.low_storage.norm_total",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_columns()),
      KOKKOS_LAMBDA(const std::size_t column, double& sum) {
        sum += partials(0, column);
      },
      Kokkos::Sum<double>(total));
#else
  for (std::size_t column = 0; column < problem_.n_columns(); ++column)
    total += partials_(0, column);
#endif
  return total;
}

// The step's held flows: a constant operator's table `exp(h_t D / 2)`, built from the
// table's interval (scalar row 2 of `phi_delta`) when the caller's key changed, or a
// varying operator's `P` and `Q` from rows 0 and 1, every step. Recomputed flows are
// formed in the finishes that apply them and hold nothing.
void GeneralWorkspace::low_storage_flows_formed(const bool rebuild_table) {
  const auto flows = low_storage_flows_;
  const auto count = problem_.stepper.low_storage_flows;
  if (count == 0 || (count == 1 && !rebuild_table))
    return;
  const auto n = problem_.dims.n_omega;
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  // With vector maths a host task forms a column's exponents a block of bins at a
  // time and their exponentials in one pass (solprop ADR-059); a separable route's
  // task, a block of adjacent columns at one bin, its table's run in the scratch's
  // layout (`RegisterIndex`).
  if constexpr (vmath::vectorised) {
    if (problem_.separable_geometry()) {
      const auto spatial = spatial_;
      const auto delta = stage_phi_delta_;
      const auto active = active_mask_;
      const auto index = register_index();
      const auto columns = problem_.n_columns();
      const std::size_t blocks = (columns + stage_block - 1) / stage_block;
      launch::parallel_for(
          "solprop_native.low_storage.separable_table",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n * blocks),
          KOKKOS_LAMBDA(const std::size_t item) {
            const std::size_t omega = item / blocks;
            const std::size_t first = (item % blocks) * stage_block;
            const std::size_t width =
                columns - first < stage_block ? columns - first : stage_block;
            const bool in_band = active(omega) != 0U;
            double real[stage_block];
            double imag[stage_block];
            for (std::size_t offset = 0; offset < width; ++offset) {
              const Complex x =
                  delta(2) *
                  spatial.value(index.mode(first + offset), omega, in_band).linear;
              real[offset] = x.real();
              imag[offset] = x.imag();
            }
            vmath::exp_complex(real, imag, real, imag, width);
            for (std::size_t offset = 0; offset < width; ++offset)
              flows.data()[index(first + offset, omega)] =
                  Complex(real[offset], imag[offset]);
          });
      return;
    }
    const auto terms = stage_terms(0);
    const auto n_phi = problem_.linear.n_phi_terms;
    const std::size_t blocks = (n + stage_block - 1) / stage_block;
    launch::parallel_for(
        "solprop_native.low_storage.modal_flows",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0,
                                            problem_.n_columns() * blocks),
        KOKKOS_LAMBDA(const std::size_t item) {
          const std::size_t column = item / blocks;
          const std::size_t first = (item % blocks) * stage_block;
          const std::size_t bins = n - first < stage_block ? n - first : stage_block;
          double real[stage_block];
          double imag[stage_block];
          const auto form = [&](const std::size_t flow, const std::size_t offset) {
            terms.phase_block(offset, column, first, bins, real, imag);
            vmath::exp_complex(real, imag, real, imag, bins);
            for (std::size_t bin = 0; bin < bins; ++bin)
              flows(flow, column, first + bin) = Complex(real[bin], imag[bin]);
          };
          if (count == 1) {
            form(0, 2 * n_phi);
          } else {
            form(0, 0);
            form(1, n_phi);
          }
        });
    return;
  }
#endif
  if (problem_.separable_geometry()) {
    const auto spatial = spatial_;
    const auto delta = stage_phi_delta_;
    const auto active = active_mask_;
    const auto n_pol = problem_.dims.n_pol_state;
    const auto index = register_index();
    cartesian_for(
        "solprop_native.low_storage.separable_table", execution_space_,
        problem_.n_columns(), n,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          const auto values = spatial.value(column / n_pol, omega, active(omega) != 0U);
          flows.data()[index(column, omega)] = propagator_exp(delta(2) * values.linear);
        });
    return;
  }
  const auto terms = stage_terms(0);
  const auto n_phi = problem_.linear.n_phi_terms;
  cartesian_for(
      "solprop_native.low_storage.modal_flows", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        if (count == 1) {
          flows(0, column, index) =
              propagator_exp(terms.phase_sum(2 * n_phi, column, index));
        } else {
          flows(0, column, index) = propagator_exp(terms.phase_sum(0, column, index));
          flows(1, column, index) =
              propagator_exp(terms.phase_sum(n_phi, column, index));
        }
      });
}

// One pass on a separable geometry (free space and the hybrid planar kind). Its input
// is the register A (pass 1) or C, with the noise draw at the pass's position, in the
// transverse scratch -- already written by the previous finish when C is elided. Under
// an in-stage absorber the pass evaluates twice from the same input: the material
// source, then the absorber's term, the exchange between them taking the material
// output out of the scratch and re-forming the input (SEP 24 section 4.3).
void GeneralWorkspace::low_storage_separable_pass(const std::size_t pass,
                                                  const double dz) {
  const bool material = problem_.kerr.nonlinear || problem_.plasma.enabled ||
                        problem_.raman.enabled || separable_dispersive_ != nullptr;
  const bool absorber = problem_.absorber.in_stage;
  const bool elided = !problem_.stepper.low_storage_stage_register;
  const auto row = low_storage_rows[pass];
  if ((material || absorber) && (pass == 0 || !elided))
    low_storage_separable_input(pass);
  if (material && absorber) {
    separable_core(row, true);
    low_storage_separable_fold(pass, fold_exchange, term_material, dz);
    separable_core(row, false);
    low_storage_separable_fold(pass, fold_second, term_absorber, dz);
  } else if (material || absorber) {
    separable_core(row, material);
    low_storage_separable_fold(pass, fold_complete,
                               material ? term_material : term_absorber, dz);
  } else {
    low_storage_separable_fold(pass, fold_complete, term_none, dz);
  }
}

namespace {
// Each column's masked squared norm of `state` into `partials(0, column)`, in a team
// per column: a device's final fold traverses like the others and reduces here, over
// one volume, instead of reducing inside the fold's crossing (SEP 28).
void column_norms(const char* label, const ExecutionSpace& execution,
                  const ComplexView2& state, const ByteView& in_norm,
                  const RealView2& partials) {
  using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
  using Member = Policy::member_type;
  const auto n = state.extent(1);
  launch::parallel_for(
      label, Policy(execution, static_cast<int>(state.extent(0)), 128),
      KOKKOS_LAMBDA(const Member& team) {
        const auto column = static_cast<std::size_t>(team.league_rank());
        double total = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, static_cast<int>(n)),
            [&](const int index, double& update) {
              const auto omega = static_cast<std::size_t>(index);
              if (in_norm(omega) != 0U)
                update += squared_abs(state(column, omega));
            },
            total);
        Kokkos::single(Kokkos::PerTeam(team), [&] { partials(0, column) = total; });
      });
}
} // namespace

void GeneralWorkspace::low_storage_separable_input(const std::size_t pass) {
  const auto state = current_;
  const auto stage_register = register_stage_;
  const auto index = stage_index();
  const auto spatial = spatial_;
  const auto at = stage_phi_at_;
  const auto active = active_mask_;
  const auto noise = noise_;
  const auto scratch = transverse_;
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const auto row = low_storage_rows[pass];
  const auto input = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
    const auto mode = index.mode(column);
    Complex value =
        pass == 0 ? state(column, omega) : stage_register.data()[index(column, omega)];
    if (add_noise) {
      const auto values = spatial.value(mode, omega, active(omega) != 0U);
      const double angle = at(row) * values.linear.imag();
      value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
    }
    return value * scale;
  };
  // In the volume route's order the scratch is the state's, so the input streams.
  if (problem_.stepper.low_storage_tiled_input && !volume_order_) {
    // Read along the state's bins and written along the scratch's rows, a tile at a
    // time.
    tiled_crossing(
        "solprop_native.low_storage.separable_input", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega, false, input,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      const Complex value) {
          scratch.data()[index.cell(column, omega)] = value;
        });
    return;
  }
  crossing_for(
      "solprop_native.low_storage.separable_input", execution_space_,
      problem_.n_columns(), problem_.dims.n_omega,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
        scratch.data()[index.cell(column, omega)] = input(column, omega);
      });
}

void GeneralWorkspace::low_storage_separable_fold(const std::size_t pass,
                                                  const int mode, const int term,
                                                  const double dz) {
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const bool tabulated = problem_.stepper.low_storage_flows != 0;
  const auto scratch = transverse_;
  const auto spatial = spatial_;
  const auto window = omega_window_;
  const auto active = active_mask_;
  const auto in_norm = norm_mask_;
  const auto noise = noise_;
  const auto delta = stage_phi_delta_;
  const auto at = stage_phi_at_;
  // A device gathers and scatters the scratch through its own tiles.
  [[maybe_unused]] const auto n_pol = problem_.dims.n_pol_state;
  [[maybe_unused]] const auto layout = transverse_layout();
  const auto index = stage_index();
  const bool add_noise = problem_.stepper.modified_noise && has_noise_;
  const double scale = inverse_transverse_count();
  const int stage = static_cast<int>(pass);
  const bool exchange = mode == fold_exchange;
  const bool second = mode == fold_second;
  // Elided, each of the first three passes writes the next pass's input in place of C.
  const bool forward = !problem_.stepper.low_storage_stage_register && pass < 3;
  const bool final = pass == 3 && !exchange;
  // What this fold reads and writes, invariant over the traversal: each moves only the
  // registers its row of the flow table names.
  const bool flows_here =
      stage == 0 || (stage == 2 && !exchange) || (stage == 3 && !second);
  const bool reads_register = (second && stage < 3) || (exchange && stage > 0);
  const bool reads_accumulator = stage > 0 || second;
  const bool writes_state = (stage == 0 && !second) || stage == 3;
  const bool writes_register = stage < 3 && !forward;
  const auto own = low_storage_rows[pass];
  const auto next = low_storage_rows[pass < 3 ? pass + 1 : 3];
  const bool linear_here = add_noise || (flows_here && !tabulated);
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // A device reduces the accepted state's norms in a pass of their own.
  const bool fused_norm = false;
#else
  const bool fused_norm = final;
#endif
  // The element, given its cell of the transverse scratch and its state (`held`).
  const auto body = KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                                  Complex& cell, Complex& held) {
    const auto column_mode = index.mode(column);
    const bool in_band = active(omega) != 0U;
    // Tabulated flows without noise, or a pass with no flow, read only the prefactor of
    // the element.
    SpatialElement values{};
    if (linear_here)
      values = spatial.value(column_mode, omega, in_band);
    else if (term == term_material)
      values.prefactor = spatial.prefactor(column_mode, omega, in_band);
    Complex source{};
    if (term == term_material)
      source = values.prefactor * window(omega) * cell;
    else if (term == term_absorber && in_band)
      source = cell;
    const std::size_t slot = index(column, omega);
    Complex flow(1.0, 0.0);
    if (flows_here)
      flow = tabulated ? flows.data()[slot]
                       : propagator_exp(delta(flow_row) * values.linear);
    Complex a = held;
    Complex b = reads_accumulator ? accumulator.data()[slot] : Complex{};
    Complex c = reads_register ? stage_register.data()[slot] : Complex{};
    if (exchange) {
      Complex value = stage == 0 ? a : c;
      if (add_noise) {
        const double angle = at(own) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    low_storage_fold(stage, mode, step, source, a, b, c, flow, flow);
    if (forward) {
      Complex value = c;
      if (add_noise) {
        const double angle = at(next) * values.linear.imag();
        value += noise(column, omega) * Complex(Kokkos::cos(angle), Kokkos::sin(angle));
      }
      cell = value * scale;
    }
    if (writes_state)
      held = a;
    if (stage < 3)
      accumulator.data()[slot] = b;
    if (writes_register)
      stage_register.data()[slot] = c;
    return final && in_norm(omega) != 0U ? squared_abs(a) : 0.0;
  };
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  // In the volume route's order, the scratch's cell sits beside the registers' and the
  // state's, so the fold streams all three.
  if (volume_order_)
    cartesian_for(
        "solprop_native.low_storage.separable_fold", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          static_cast<void>(body(column, omega,
                                 scratch.data()[index.cell(column, omega)],
                                 state(column, omega)));
        });
  else
    tiled_update(
        "solprop_native.low_storage.separable_fold", execution_space_,
        problem_.n_columns(), problem_.dims.n_omega, exchange || forward,
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega) {
          return layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol);
        },
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      Complex& cell) {
          static_cast<void>(body(column, omega, cell, state(column, omega)));
        },
        KOKKOS_LAMBDA(const std::size_t column, const std::size_t omega,
                      const Complex value) {
          layout.cell(scratch, omega * n_pol + column % n_pol, column / n_pol) = value;
        });
#else
  // The host's registers sit at the scratch cell's own position (`RegisterIndex`), so a
  // block of adjacent columns reads them, and the scratch, in contiguous runs at every
  // bin; only the state is column-major. A work item takes a block of columns over
  // every bin, a tile of bins at a time: it loads the tile's state a column (a
  // contiguous run) at a time, folds the tile in the scratch's order, and stores the
  // state back. The final fold sums each column's norm in ascending bin order (as
  // `register_traversal`).
  constexpr std::size_t block = crossing_tile_columns;
  constexpr std::size_t bins = crossing_tile_bins;
  const std::size_t columns = problem_.n_columns();
  const std::size_t n = problem_.dims.n_omega;
  const std::size_t blocks = (columns + block - 1) / block;
  const auto partials = partials_;
  launch::parallel_for(
      "solprop_native.low_storage.separable_fold",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, blocks),
      KOKKOS_LAMBDA(const std::size_t which) {
        const std::size_t first = which * block;
        const std::size_t count = columns - first < block ? columns - first : block;
        double totals[block] = {};
        Complex tile[bins][block];
        for (std::size_t start = 0; start < n; start += bins) {
          const std::size_t width = n - start < bins ? n - start : bins;
          for (std::size_t offset = 0; offset < count; ++offset) {
            const Complex* row = &state(first + offset, start);
            for (std::size_t bin = 0; bin < width; ++bin)
              tile[bin][offset] = row[bin];
          }
          for (std::size_t bin = 0; bin < width; ++bin)
            for (std::size_t offset = 0; offset < count; ++offset) {
              const std::size_t column = first + offset;
              const std::size_t omega = start + bin;
              totals[offset] +=
                  body(column, omega, scratch.data()[index(column, omega)],
                       tile[bin][offset]);
            }
          if (writes_state)
            for (std::size_t offset = 0; offset < count; ++offset) {
              Complex* row = &state(first + offset, start);
              for (std::size_t bin = 0; bin < width; ++bin)
                row[bin] = tile[bin][offset];
            }
        }
        if (fused_norm)
          for (std::size_t offset = 0; offset < count; ++offset)
            partials(0, first + offset) = totals[offset];
      });
#endif
  if (final && !fused_norm)
    column_norms("solprop_native.low_storage.separable_norms", execution_space_, state,
                 in_norm, partials_);
}

// One pass on a modal route (mode sets, radial, general modal): its input, the
// register A (pass 1) or C with the noise draw at the pass's position, padded into the
// fine spectrum (and the general route's scaled channels); the shared core; and one
// fold of the source -- prefactor, window and the in-stage absorber's term at the
// pass's position, with no flow back to an anchor.
void GeneralWorkspace::low_storage_modal_pass(const std::size_t pass, const double dz) {
  const auto row = low_storage_rows[pass];
  const bool responds = general_
                            ? general_->responds() || problem_.absorber.in_stage
                            : problem_.kerr.nonlinear || problem_.plasma.enabled ||
                                  problem_.raman.enabled ||
                                  problem_.absorber.in_stage || dispersive_ != nullptr;
  if (responds)
    low_storage_modal_input(pass);
  const bool evaluated = modal_core(row);
  const auto terms = stage_terms(row);
  const auto state = current_;
  const auto accumulator = register_accumulator_;
  const auto stage_register = register_stage_;
  const auto flows = low_storage_flows_;
  const auto count = problem_.stepper.low_storage_flows;
  const auto in_norm = norm_mask_;
  const auto n_phi = problem_.linear.n_phi_terms;
  const auto n_prefactor = problem_.linear.n_prefactor_terms;
  const int stage = static_cast<int>(pass);
  const bool final = pass == 3;
  const bool flows_here = stage != 1;
  const std::size_t flow_row = stage == 0 ? 0 : 1;
  const double step = dz;
  register_traversal(
      "solprop_native.low_storage.modal_fold", execution_space_, problem_.n_columns(),
      problem_.dims.n_omega, partials_, final,
      KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex source{};
        if (evaluated) {
          // Zero weights skipped, as in `StageTerms::finish`.
          Complex prefactor{};
          for (std::size_t term = 0; term < n_prefactor; ++term) {
            const double weight = terms.prefactor_weights(row * n_prefactor + term);
            if (weight != 0.0)
              prefactor += weight * terms.prefactor_terms(term, column, index);
          }
          source = prefactor * terms.window(index) *
                   (terms.general ? terms.general_source(column, index)
                                  : terms.projected(column, terms.crop(index)));
          if (terms.in_stage_absorber && terms.active(index) != 0U)
            source += terms.absorber(column, terms.crop(index));
        }
        Complex flow(1.0, 0.0);
        if (flows_here)
          flow = count == 1 ? flows(0, column, index)
                 : count == 2
                     ? flows(flow_row, column, index)
                     : propagator_exp(terms.phase_sum(flow_row * n_phi, column, index));
        Complex a = state(column, index);
        Complex b = stage > 0 ? accumulator(column, index) : Complex{};
        Complex c{};
        low_storage_fold(stage, fold_complete, step, source, a, b, c, flow, flow);
        if (stage == 0 || stage == 3)
          state(column, index) = a;
        if (stage < 3) {
          accumulator(column, index) = b;
          stage_register(column, index) = c;
        }
        return final && in_norm(index) != 0U ? squared_abs(a) : 0.0;
      });
}

void GeneralWorkspace::low_storage_modal_input(const std::size_t pass) {
  const auto source = pass == 0 ? current_ : register_stage_;
  const auto row = low_storage_rows[pass];
  const auto terms = stage_terms(row);
  const auto n = problem_.dims.n_omega;
  const auto gaps = fine_spectrum_size() - n;
  cartesian_for(
      "solprop_native.low_storage.modal_input", execution_space_, problem_.n_columns(),
      n, KOKKOS_LAMBDA(const std::size_t column, const std::size_t index) {
        Complex value = source(column, index);
        if (terms.add_noise) {
          const Complex phase_at = terms.noise_phase(row, column, index);
          value += terms.noise(column, index) *
                   Complex(Kokkos::cos(phase_at.imag()), Kokkos::sin(phase_at.imag()));
        }
        const double analytic_scale =
            terms.analytic && index > 0 && 2 * terms.crop(index) < terms.n_to ? 2.0
                                                                              : 1.0;
        terms.fine(column, terms.crop(index)) =
            analytic_scale * terms.inverse_n_to * value;
        if (terms.general)
          terms.general_scaled(column, index) =
              value *
              (terms.general_pad * general_doubling(terms.crop(index), terms.n_to));
        for (std::size_t gap = index; gap < gaps; gap += n)
          terms.zero_gaps(column, gap, gap + 1);
      });
}
