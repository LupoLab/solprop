// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

// The boundary monitor's readings (solprop SEP 21) and the transverse edge's (solprop
// SEP 44): bind_monitor and the monitor's sums, the edge's bind_edge, bind_edge_lines
// and edge_reading, split_sums, monitor_reading and bind_occupancy, the split collar's
// walk-off occupancy. These passes only form sums; the judgements are made elsewhere.
//
// Part of statistics.cpp's translation unit (included there, in place; never
// compiled alone).

// Bind solprop's boundary monitor (SEP 21): its four time weights, row-major `(4,
// n_t)`, and its columns' quadrature by mode (empty for unit weights). The views are
// allocated at the first bind and kept; the split's accumulator starts at zero.
void GeneralWorkspace::bind_monitor(const std::vector<double>& weights,
                                    const std::vector<double>& column_weights) {
  const auto n_t = problem_.dims.n_t;
  const auto columns = problem_.n_columns();
  require_extent(weights.size(), 4 * n_t, "monitor weights");
  if (!column_weights.empty())
    require_extent(column_weights.size(), problem_.dims.n_modes,
                   "monitor column weights");
  if (monitor_weights_.extent(0) == 0) {
    monitor_weights_ = RealView2("solprop_native.monitor.weights", 4, n_t);
    monitor_column_weights_ =
        RealView("solprop_native.monitor.column_weights", columns);
    monitor_share_ = RealView("solprop_native.monitor.split_share", n_t);
    monitor_half_ = RealView("solprop_native.monitor.split_half", n_t);
    monitor_removed_ =
        Kokkos::View<double, Layout, MemorySpace>("solprop_native.monitor.removed");
    monitor_split_sum_ =
        Kokkos::View<SplitSums, MemorySpace>("solprop_native.monitor.split_sum");
    monitor_found_ =
        Kokkos::View<SplitSums, MemorySpace>("solprop_native.monitor.split_found");
  }
  auto host_weights = Kokkos::create_mirror_view(monitor_weights_);
  for (std::size_t row = 0; row < 4; ++row)
    for (std::size_t index = 0; index < n_t; ++index)
      host_weights(row, index) = weights[row * n_t + index];
  graph::deep_copy(execution_space_, monitor_weights_, host_weights);
  // A state column is `mode * n_pol + pol`.
  std::vector<double> expanded(columns, 1.0);
  if (!column_weights.empty())
    for (std::size_t column = 0; column < columns; ++column)
      expanded[column] = column_weights[column / problem_.dims.n_pol_state];
  copy_vector(execution_space_, monitor_column_weights_, expanded);
  graph::deep_copy(execution_space_, monitor_removed_, 0.0);
  graph::deep_copy(execution_space_, monitor_found_, SplitSums{});
  execution_space_.fence("solprop_native monitor bind");
  monitor_bound_ = true;
}

// The boundary monitor's reading of the accepted state (solprop SEP 21, ADR-057):
// `sum_t P`, the four weighted sums `sum_t w_k P` and the split's accumulator, which
// this drains; `P(t)` is the columns' weighted `|E(t)|^2` on the coarse time axis.
// The state is transformed chunk by chunk through buffers idle between steps: on a
// field grid under the split scheme the collar's coarse pair, whose inverse real
// transform is the one solprop's reference takes; otherwise the statistics buffers,
// with the analytic signal's spectrum, whose real part is that transform's result.
namespace {
// The boundary monitor's sums over `count` rows of `n_t` samples: each sample's power,
// `power(row, index)`, in total and weighed by each of the four regions' weights. A
// host work item takes a run of one row's samples, its sums in sample order, so that
// no sample divides its flat index by the row length; a device takes one sample.
template <class Weights, class Power>
MonitorSums monitor_sums(const char* label, const ExecutionSpace& execution,
                         const std::size_t count, const std::size_t n_t,
                         const Weights& weights, const Power& power) {
  MonitorSums sums;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  launch::parallel_reduce(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count * n_t),
      KOKKOS_LAMBDA(const std::size_t item, MonitorSums& update) {
        const auto index = item % n_t;
        const double value = power(item / n_t, index);
        update.values[0] += value;
        for (std::size_t region = 0; region < 4; ++region)
          update.values[region + 1] += weights(region, index) * value;
      },
      Kokkos::Sum<MonitorSums>(sums));
#else
  constexpr std::size_t run = 512;
  const std::size_t runs = (n_t + run - 1) / run;
  launch::parallel_reduce(
      label, Kokkos::RangePolicy<ExecutionSpace>(execution, 0, count * runs),
      KOKKOS_LAMBDA(const std::size_t item, MonitorSums& update) {
        const std::size_t row = item / runs;
        const std::size_t first = (item % runs) * run;
        const std::size_t last = n_t - first < run ? n_t : first + run;
        double total = 0.0;
        double regions[4] = {0.0, 0.0, 0.0, 0.0};
        for (std::size_t index = first; index < last; ++index) {
          // NOLINTNEXTLINE(readability-suspicious-call-argument): the names only coincide
          const double value = power(row, index);
          total += value;
          for (std::size_t region = 0; region < 4; ++region)
            regions[region] += weights(region, index) * value;
        }
        update.values[0] += total;
        for (std::size_t region = 0; region < 4; ++region)
          update.values[region + 1] += regions[region];
      },
      Kokkos::Sum<MonitorSums>(sums));
#endif
  return sums;
}
} // namespace

namespace {
// The field on a line's node, in a row of the transverse scratch (solprop SEP 44): a
// row is a plane and a position across the line, and no route has a branch here.
struct EdgeScratchValue {
  ComplexView3 scratch;
  std::size_t units{}, across{}, row_stride{}, node_stride{};
  KOKKOS_INLINE_FUNCTION Complex operator()(const std::size_t row,
                                            const std::size_t node) const {
    return scratch.data()[(row / across) * units + (row % across) * row_stride +
                          node * node_stride];
  }
};

// The same radially: `sqrt(r) E`, a dense product per value, in a row of (pol, omega).
struct EdgeRadialValue {
  RealView2 synthesis;
  ComplexView2 current;
  RealView scale;
  std::size_t n_pol{}, n{}, modes{};
  KOKKOS_INLINE_FUNCTION Complex operator()(const std::size_t row,
                                            const std::size_t node) const {
    const auto pol = row / n;
    const auto omega = row % n;
    Complex value{};
    for (std::size_t mode = 0; mode < modes; ++mode)
      value += synthesis(node, mode) * current(mode * n_pol + pol, omega);
    return scale(node) * value;
  }
};

// One line's `(inward, windowed)`, unscaled: a Kaiser-windowed transform along its
// normal, padded twice, batch by batch so that the buffer stays bounded.
template <class Value>
MonitorSums edge_strip_line(const ExecutionSpace& execution, const Value& value,
                            const std::size_t rows, const std::size_t batch,
                            const std::size_t nodes, const std::size_t first,
                            const std::size_t last, const IndexView& positions,
                            const RealView& window, const ComplexView2& buffer,
                            StripPlan& plan) {
  const std::size_t count = 2 * nodes;
  MonitorSums sums;
  for (std::size_t start = 0; start < rows; start += batch) {
    launch::parallel_for(
        "solprop_native.edge.strip_fill",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, batch),
        KOKKOS_LAMBDA(const std::size_t row) {
          // A short last batch leaves its other rows zero, whose energy is none.
          const bool live = start + row < rows;
          for (std::size_t index = 0; index < nodes; ++index)
            buffer(row, index) =
                live ? window(index) * value(start + row, positions(index)) : Complex{};
          for (std::size_t index = nodes; index < count; ++index)
            buffer(row, index) = Complex{};
        });
    graph::execute(plan, buffer, buffer, KokkosFFT::Normalization::none);
    MonitorSums part;
    launch::parallel_reduce(
        "solprop_native.edge.strip_sums",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, batch),
        KOKKOS_LAMBDA(const std::size_t row, MonitorSums& update) {
          for (std::size_t index = 0; index < count; ++index) {
            const double power = squared_abs(buffer(row, index));
            update.values[1] += power;
            if (index >= first && index < last)
              update.values[0] += power;
          }
        },
        Kokkos::Sum<MonitorSums>(part));
    sums += part;
  }
  return sums;
}

// One line's rectified inward current: `Im(E* dE/ds) / k`, rectified per row before
// it's summed, as a collar's current is defined (solprop SEP 40 section 5.3).
template <class Value>
double edge_current_line(const ExecutionSpace& execution, const Value& value,
                         const std::size_t rows, const std::size_t node,
                         const std::size_t before, const std::size_t after,
                         const double step, const double k_min) {
  double total = 0.0;
  launch::parallel_reduce(
      "solprop_native.edge.current",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, rows),
      KOKKOS_LAMBDA(const std::size_t row, double& update) {
        const Complex here = value(row, node);
        const Complex slope = (value(row, after) - value(row, before)) / step;
        const double flux =
            (here.real() * slope.imag() - here.imag() * slope.real()) / k_min;
        update += flux > 0.0 ? flux : 0.0;
      },
      Kokkos::Sum<double>(total));
  return total;
}

// The transverse edge's region weights (solprop SEP 44), from the rows given on: a
// radial read synthesises only the outer rows its regions can touch.
struct EdgeWeights {
  RealView2 weights;
  std::size_t first{};
  KOKKOS_INLINE_FUNCTION double operator()(const std::size_t region,
                                           const std::size_t index) const {
    return weights(region, first + index);
  }
};
} // namespace

// Bind what the transverse edge's reading sums with (solprop SEP 44): three tables of
// four regions' weights on the nodes, and radially the synthesis and the first row a
// read uses. Native holds no face, mask or judgement. The views are allocated at the
// first bind and kept.
void GeneralWorkspace::bind_edge(const std::vector<double>& faces,
                                 const std::vector<double>& clip,
                                 const std::vector<double>& totals,
                                 const std::vector<double>& synthesis,
                                 const std::size_t first_row) {
  const auto nodes = faces.size() / 4;
  require_extent(faces.size(), 4 * nodes, "edge faces");
  require_extent(clip.size(), 4 * nodes, "edge clip");
  require_extent(totals.size(), 4 * nodes, "edge totals");
  if (problem_.separable_geometry()) {
    require_extent(nodes, transverse_units(), "edge nodes");
    require_extent(synthesis.size(), 0, "a separable edge's synthesis");
  } else {
    require_extent(nodes, problem_.dims.n_modes, "edge nodes");
    require_extent(synthesis.size(), nodes * nodes, "edge synthesis");
  }
  if (first_row >= nodes)
    throw std::invalid_argument("the edge's first row lies past its nodes");
  if (edge_faces_.extent(0) == 0) {
    edge_faces_ = RealView2("solprop_native.edge.faces", 4, nodes);
    edge_clip_ = RealView2("solprop_native.edge.clip", 4, nodes);
    edge_totals_ = RealView2("solprop_native.edge.totals", 4, nodes);
    edge_columns_ = RealView("solprop_native.edge.columns", problem_.dims.n_modes);
    if (!synthesis.empty())
      edge_synthesis_ = RealView2("solprop_native.edge.synthesis", nodes, nodes);
    if (!synthesis.empty() && problem_.dims.n_pol_state == 1 &&
        current_.extent(1) == problem_.dims.n_omega)
      edge_nodes_ = RealView2(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "solprop_native.edge.nodes"),
          nodes, 2 * problem_.dims.n_omega);
  }
  copy_vector(execution_space_, edge_faces_, faces);
  copy_vector(execution_space_, edge_clip_, clip);
  copy_vector(execution_space_, edge_totals_, totals);
  if (!synthesis.empty())
    copy_vector(execution_space_, edge_synthesis_, synthesis);
  edge_first_row_ = first_row;
  execution_space_.fence("solprop_native edge bind");
}

// Bind a requested collar's lines (solprop SEP 44), which Python describes: their
// nodes, windows, inward bins and onsets. A batch of one shape serves every line that
// has it.
void GeneralWorkspace::bind_edge_lines(const EdgeLines& lines) {
  const auto n_lines = lines.axes.size();
  require_extent(lines.counts.size(), n_lines, "edge line counts");
  require_extent(lines.inward.size(), 2 * n_lines, "edge line bins");
  require_extent(lines.factors.size(), n_lines, "edge line factors");
  require_extent(lines.measures.size(), n_lines, "edge line measures");
  require_extent(lines.onsets.size(), 3 * n_lines, "edge line onsets");
  require_extent(lines.steps.size(), n_lines, "edge line steps");
  require_extent(lines.windows.size(), lines.positions.size(), "edge line windows");
  if (std::accumulate(lines.counts.begin(), lines.counts.end(), std::size_t{0}) !=
      lines.positions.size())
    throw std::invalid_argument("the edge's lines don't account for their nodes");
  const auto planes = problem_.dims.n_pol_state * problem_.dims.n_omega;
  // A runner's next point reuses the plans of its shapes, and only those stay held.
  decltype(edge_batches_) batches;
  std::vector<EdgeLine> bound;
  bound.reserve(n_lines);
  std::size_t offset = 0;
  for (std::size_t index = 0; index < n_lines; ++index) {
    EdgeLine line;
    line.axis = lines.axes[index];
    line.nodes = lines.counts[index];
    line.rows = planes * (problem_.slab_geometry()         ? problem_.spatial.n_p
                          : !problem_.separable_geometry() ? std::size_t{1}
                          : line.axis == 1                 ? problem_.spatial.n_y
                                                           : problem_.spatial.n_x);
    line.first = lines.inward[2 * index];
    line.last = lines.inward[2 * index + 1];
    line.factor = lines.factors[index];
    line.measure = lines.measures[index];
    line.step = lines.steps[index];
    line.onset = lines.onsets[3 * index];
    line.before = lines.onsets[3 * index + 1];
    line.after = lines.onsets[3 * index + 2];
    if (line.nodes != 0) {
      const auto first = static_cast<std::ptrdiff_t>(offset);
      const auto last = static_cast<std::ptrdiff_t>(offset + line.nodes);
      line.positions = IndexView("solprop_native.edge.line_nodes", line.nodes);
      line.window = RealView("solprop_native.edge.line_window", line.nodes);
      copy_vector(execution_space_, line.positions,
                  std::vector<std::size_t>(lines.positions.begin() + first,
                                           lines.positions.begin() + last));
      copy_vector(execution_space_, line.window,
                  std::vector<double>(lines.windows.begin() + first,
                                      lines.windows.begin() + last));
      offset += line.nodes;
      const std::size_t count = 2 * line.nodes;
      line.batch =
          std::min(line.rows, std::max<std::size_t>(1, lines.batch_bytes /
                                                           (sizeof(Complex) * count)));
    }
    bound.push_back(std::move(line));
  }
  // The strips run in turn, so one allocation serves every shape: a positive and a
  // negative face's lines commonly differ by a node, and would otherwise hold two.
  std::size_t largest = 0;
  for (const auto& line : bound)
    largest = std::max(largest, line.batch * 2 * line.nodes);
  if (edge_strip_.size() != largest)
    edge_strip_ =
        largest == 0
            ? ComplexView{}
            : ComplexView(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             std::string("solprop_native.edge.strip")),
                          largest);
  for (const auto& line : bound) {
    if (line.nodes == 0)
      continue;
    const std::size_t count = 2 * line.nodes;
    const std::pair<std::size_t, std::size_t> shape{line.batch, count};
    if (!batches.contains(shape)) {
      EdgeBatch batch;
      if (auto kept = edge_batches_.extract(shape); kept)
        batch = std::move(kept.mapped());
      // Not tracked: `edge_strip_` owns the memory, and outlives every shape's view.
      batch.buffer = ComplexView2(edge_strip_.data(), line.batch, count);
      if (!batch.plan) {
        // A plan's own memory is measured as the workspace's are, and listed with them.
        const auto heap_before = heap_bytes();
        const auto device_before = device_free_bytes();
        constexpr int along = -1;
        batch.plan = std::make_unique<StripPlan>(fft_space_for_rows(batch.buffer),
                                                 batch.buffer, batch.buffer,
                                                 KokkosFFT::Direction::forward, along);
        execution_space_.fence("solprop_native edge strip plan");
        const auto heap_after = heap_bytes();
        const auto device_after = device_free_bytes();
        batch.heap_growth = heap_after > heap_before ? heap_after - heap_before : 0;
        batch.device_growth =
            device_before > device_after ? device_before - device_after : 0;
      }
      batches.emplace(shape, std::move(batch));
    }
  }
  edge_lines_ = std::move(bound);
  edge_batches_ = std::move(batches);
  edge_k_min_ = lines.k_min;
  if (lines.scale.empty()) {
    edge_scale_ = {};
  } else {
    edge_scale_ = RealView("solprop_native.edge.line_scale", lines.scale.size());
    copy_vector(execution_space_, edge_scale_, lines.scale);
  }
  execution_space_.fence("solprop_native edge lines");
}

// The state's reading for the transverse edge: its region sums and each mode's energy,
// with only those numbers crossing. With `whole` (a launch) every table on every node;
// otherwise the faces', and with `totals` a collar's, on the rows a read uses; and with
// a requested collar its strip and current, on the same transform. The transverse
// scratch is free at a committed position: a step's first pass writes it again.
EdgeNumbers GeneralWorkspace::edge_reading(const bool faces, const bool whole,
                                           const bool totals, const bool strip,
                                           const bool want_current) {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (edge_faces_.extent(0) == 0)
    throw std::logic_error("no transverse edge is bound");
  const auto current = current_;
  const auto n_pol = problem_.dims.n_pol_state;
  const auto n = problem_.dims.n_omega;
  EdgeNumbers reading;
  const bool separable = problem_.separable_geometry();
  if (separable)
    separable_load_current();
  const auto take = [&reading](const MonitorSums& sums) {
    reading.sums.insert(reading.sums.end(), sums.values.data() + 1,
                        sums.values.data() + 5);
  };
  if (!faces) {
    // Only what follows was asked for.
  } else if (separable) {
    const auto scratch = transverse_;
    const auto units = transverse_units();
    const auto planes = scratch.size() / units;
    const auto power = KOKKOS_LAMBDA(const std::size_t plane, const std::size_t node) {
      return squared_abs(scratch.data()[plane * units + node]);
    };
    take(monitor_sums("solprop_native.edge.faces", execution_space_, planes, units,
                      edge_faces_, power));
    if (whole)
      take(monitor_sums("solprop_native.edge.clip", execution_space_, planes, units,
                        edge_clip_, power));
    if (whole || totals)
      take(monitor_sums("solprop_native.edge.totals", execution_space_, planes, units,
                        edge_totals_, power));
  } else if (edge_nodes_.extent(0) != 0) {
    // One state polarisation: the rows wanted synthesised in one product, the state's
    // complex samples read as twice as many real columns, and every table's sums then
    // from those fields. A launch reads every node: its total is the grid's. The
    // per-sample sums this replaces synthesised each row once a table; on an Apple M5
    // Max the 256- and 512-node radial filaments ran 0.83 and 0.75 of their one-thread
    // windows, 0.92 and 0.87 at six threads.
    using Interleaved = Kokkos::View<double**, Layout, MemorySpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    static_assert(sizeof(Complex) == 2 * sizeof(double));
    const auto nodes = problem_.dims.n_modes;
    const auto first = whole ? std::size_t{0} : edge_first_row_;
    const auto rows = nodes - first;
    const Interleaved state(reinterpret_cast<double*>(current_.data()), nodes, 2 * n);
    const Interleaved synthesis(edge_synthesis_.data() + first * nodes, rows, nodes);
    const Interleaved fields(edge_nodes_.data(), rows, 2 * n);
    Kokkos::Profiling::pushRegion("solprop_native.edge.synthesis");
    engine_gemm(transform_space_for(rows), synthesis, state, fields, 0.0);
    Kokkos::Profiling::popRegion();
    const auto power = KOKKOS_LAMBDA(const std::size_t plane, const std::size_t row) {
      const double real = fields(row, 2 * plane);
      const double imaginary = fields(row, 2 * plane + 1);
      return real * real + imaginary * imaginary;
    };
    const auto sums = [&](const char* label, const RealView2& weights) {
      return monitor_sums(label, execution_space_, n, rows,
                          EdgeWeights{.weights = weights, .first = first}, power);
    };
    take(sums("solprop_native.edge.faces", edge_faces_));
    if (whole)
      take(sums("solprop_native.edge.clip", edge_clip_));
    if (whole || totals)
      take(sums("solprop_native.edge.totals", edge_totals_));
  } else {
    // Each plane's synthesis on the rows wanted. A launch reads every node: its total
    // is the grid's. A dense product per sample, on a read's few outer rows.
    const auto synthesis = edge_synthesis_;
    const auto nodes = problem_.dims.n_modes;
    const auto first = whole ? std::size_t{0} : edge_first_row_;
    const auto power = KOKKOS_LAMBDA(const std::size_t plane, const std::size_t row) {
      const auto pol = plane / n;
      const auto omega = plane % n;
      Complex value{};
      for (std::size_t mode = 0; mode < nodes; ++mode)
        value += synthesis(first + row, mode) * current(mode * n_pol + pol, omega);
      return squared_abs(value);
    };
    const auto sums = [&](const char* label, const RealView2& weights) {
      return monitor_sums(label, execution_space_, n_pol * n, nodes - first,
                          EdgeWeights{.weights = weights, .first = first}, power);
    };
    take(sums("solprop_native.edge.faces", edge_faces_));
    if (whole)
      take(sums("solprop_native.edge.clip", edge_clip_));
    if (whole || totals)
      take(sums("solprop_native.edge.totals", edge_totals_));
  }
  if (faces) {
    const auto energies = edge_columns_;
    launch::parallel_for(
        "solprop_native.edge.columns",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.dims.n_modes),
        KOKKOS_LAMBDA(const std::size_t mode) {
          double sum = 0.0;
          for (std::size_t pol = 0; pol < n_pol; ++pol)
            for (std::size_t index = 0; index < n; ++index)
              sum += squared_abs(current(mode * n_pol + pol, index));
          energies(mode) = sum;
        });
    execution_space_.fence("solprop_native edge columns");
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, energies);
    reading.columns.assign(host.data(), host.data() + host.size());
  }
  // The same two kernels on any route: only where a row's value comes from differs.
  const auto take_line = [&](const EdgeLine& line, const auto& value) {
    if (strip) {
      MonitorSums sums;
      if (line.nodes != 0) {
        auto& batch = edge_batches_.at({line.batch, 2 * line.nodes});
        sums = edge_strip_line(execution_space_, value, line.rows, line.batch,
                               line.nodes, line.first, line.last, line.positions,
                               line.window, batch.buffer, *batch.plan);
      }
      reading.strip.push_back(sums.values[0] * line.factor);
      reading.strip.push_back(sums.values[1] * line.factor);
    }
    if (want_current && line.onset != EdgeLines::no_onset)
      reading.current +=
          line.measure * edge_current_line(execution_space_, value, line.rows,
                                           line.onset, line.before, line.after,
                                           line.step, edge_k_min_);
  };
  if (strip || want_current) {
    for (const auto& line : edge_lines_) {
      if (!separable) {
        take_line(line, EdgeRadialValue{.synthesis = edge_synthesis_,
                                        .current = current,
                                        .scale = edge_scale_,
                                        .n_pol = n_pol,
                                        .n = n,
                                        .modes = problem_.dims.n_modes});
      } else if (problem_.slab_geometry()) {
        take_line(line, EdgeScratchValue{.scratch = transverse_,
                                         .units = transverse_units(),
                                         .across = 1,
                                         .row_stride = 0,
                                         .node_stride = 1});
      } else if (line.axis == 1) {
        take_line(line, EdgeScratchValue{.scratch = transverse_,
                                         .units = transverse_units(),
                                         .across = problem_.spatial.n_y,
                                         .row_stride = problem_.spatial.n_x,
                                         .node_stride = 1});
      } else {
        take_line(line, EdgeScratchValue{.scratch = transverse_,
                                         .units = transverse_units(),
                                         .across = problem_.spatial.n_x,
                                         .row_stride = 1,
                                         .node_stride = problem_.spatial.n_x});
      }
    }
  }
  return reading;
}

std::optional<std::array<double, 5>> GeneralWorkspace::split_sums() {
  if (!monitoring_split())
    return std::nullopt;
  // The last step's split adds its chunks' sums here, the removal first (`split_step`):
  // read where they are kept, nothing drained.
  execution_space_.fence("solprop_native split sums");
  const auto found =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, monitor_found_)();
  return std::array<double, 5>{found.values[0], found.values[1], found.values[2],
                               found.values[3], found.values[4]};
}

std::vector<double> GeneralWorkspace::monitor_reading() {
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!monitor_bound_)
    throw std::logic_error("no boundary monitor is bound");
  const auto current = current_;
  const auto spectrum = statistics_spectrum_;
  const auto time = statistics_time_;
  const auto weights = monitor_weights_;
  const auto column_weights = monitor_column_weights_;
  const auto n = problem_.dims.n_omega;
  const auto n_t = problem_.dims.n_t;
  const bool envelope = problem_.dims.family == FieldFamily::envelope;
  // A one-sided spectrum's last bin is the Nyquist bin only on an even axis.
  const bool nyquist = n_t % 2 == 0;
  const auto columns = problem_.n_columns();
  MonitorSums sums;
  // Half the analytic signal's transform work, on buffers that already exist.
  if (problem_.absorber.split_step && !envelope) {
    const auto coarse = coarse_spectrum_;
    const auto samples = coarse_time_real_;
    const double inverse_n = 1.0 / static_cast<double>(n_t);
    const auto chunk_rows = coarse_spectrum_.extent(0);
    for (std::size_t first = 0; first < columns; first += chunk_rows) {
      const auto count = std::min(chunk_rows, columns - first);
      launch::parallel_for(
          "solprop_native.monitor.coarse_spectrum",
          Range2(execution_space_, {0, 0}, {count, n}),
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
            coarse(row, index) = inverse_n * current(first + row, index);
          });
      graph::execute(*coarse_inverse_real_plan_, coarse, samples,
                     KokkosFFT::Normalization::none);
      sums += monitor_sums(
          "solprop_native.monitor.sums_real", execution_space_, count, n_t, weights,
          KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
            const double value = samples(row, index);
            return column_weights(first + row) * value * value;
          });
    }
  }
  const auto rows = statistics_spectrum_.extent(0);
  const bool analytic = !(problem_.absorber.split_step && !envelope);
  for (std::size_t first = 0; analytic && first < columns; first += rows) {
    const auto count = std::min(rows, columns - first);
    // Every bin of the rows read is written, the empty half of a one-sided spectrum
    // with zeros; a partial chunk's other rows are transformed but never read.
    launch::parallel_for(
        "solprop_native.monitor.spectrum",
        Range2(execution_space_, {0, 0}, {count, n_t}),
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
          if (index >= n) {
            spectrum(row, index) = Complex{};
            return;
          }
          const bool single = envelope || index == 0 || (nyquist && index + 1 == n);
          spectrum(row, index) = (single ? 1.0 : 2.0) / static_cast<double>(n_t) *
                                 current(first + row, index);
        });
    graph::execute(*statistics_inverse_plan_, spectrum, time,
                   KokkosFFT::Normalization::none);
    sums += monitor_sums(
        "solprop_native.monitor.sums", execution_space_, count, n_t, weights,
        KOKKOS_LAMBDA(const std::size_t row, const std::size_t index) {
          const Complex value = time(row, index);
          return column_weights(first + row) *
                 (envelope ? squared_abs(value) : value.real() * value.real());
        });
  }
  double removed = 0.0;
  SplitSums found;
  if (problem_.absorber.split_step) {
    execution_space_.fence("solprop_native monitor reading");
    removed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, monitor_removed_)();
    found = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, monitor_found_)();
    graph::deep_copy(execution_space_, monitor_removed_, 0.0);
  }
  std::vector<double> reading{sums.values[0],  sums.values[1],  sums.values[2],
                              sums.values[3],  sums.values[4],  removed,
                              found.values[1], found.values[2], found.values[3],
                              found.values[4]};
  if (occupancy_count_ != 0) {
    // The state's energy per walk-off bucket, beside the five: a state column is
    // `mode * n_pol + pol`, and the buckets are by mode. A reduction per column, whose
    // per-thread histograms need no atomics: on a device an atomic per element took
    // most of a reading, several steps' time on an AMD Instinct MI325X.
    const auto buckets = occupancy_buckets_;
    const auto n_pol = problem_.dims.n_pol_state;
    for (std::size_t first = 0; first < occupancy_count_;
         first += occupancy_reduction_slots) {
      OccupancySums occupancy;
      launch::parallel_reduce(
          "solprop_native.monitor.occupancy",
          Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, columns),
          KOKKOS_LAMBDA(const std::size_t row, OccupancySums& update) {
            const std::size_t offset = (row / n_pol) * n;
            const double weight = column_weights(row);
            for (std::size_t index = 0; index < n; ++index) {
              const std::size_t bucket = buckets(offset + index);
              if (bucket >= first && bucket - first < occupancy_reduction_slots)
                update.values[bucket - first] +=
                    weight * squared_abs(current(row, index));
            }
          },
          Kokkos::Sum<OccupancySums>(occupancy));
      reading.insert(reading.end(), occupancy.values.data(),
                     occupancy.values.data() +
                         std::min(occupancy_reduction_slots, occupancy_count_ - first));
    }
  }
  execution_space_.fence("solprop_native monitor reading");
  return reading;
}

// Bind the split collar's walk-off bucket of each (mode, frequency) bin (solprop
// #440): from here on a monitor reading also returns the state's energy per bucket.
// The table is allocated at the first bind and kept.
void GeneralWorkspace::bind_occupancy(const std::vector<unsigned char>& buckets,
                                      const std::size_t count) {
  require_extent(buckets.size(), problem_.dims.n_modes * problem_.dims.n_omega,
                 "occupancy buckets");
  if (count == 0 || count > occupancy_bucket_limit)
    throw std::invalid_argument("the occupancy needs between 1 and 128 buckets");
  for (const auto bucket : buckets)
    if (bucket >= count)
      throw std::invalid_argument("an occupancy bucket lies past the bucket count");
  if (occupancy_count_ == 0)
    occupancy_buckets_ =
        ByteView("solprop_native.monitor.occupancy_buckets", buckets.size());
  occupancy_count_ = count;
  auto host = Kokkos::create_mirror_view(occupancy_buckets_);
  for (std::size_t index = 0; index < buckets.size(); ++index)
    host(index) = buckets[index];
  graph::deep_copy(execution_space_, occupancy_buckets_, host);
  execution_space_.fence("solprop_native occupancy bind");
}
