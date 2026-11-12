// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/launch.hpp"

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace solprop_native {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;
using Layout = Kokkos::LayoutRight;
/** A two-dimensional range whose last index runs fastest, as `Layout` stores it.

    Kokkos's CUDA and HIP default runs the first index fastest, which puts adjacent
    threads a whole row apart in a `LayoutRight` view; a host's default is already
    this order. */
using Range2Policy = Kokkos::MDRangePolicy<
    ExecutionSpace, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

/** Samples in one host tile of a `Range2`. Swept from 64 to 1024 on the four-mode RDW
    at 16 threads (an Apple M5 Max and an AMD EPYC 7763): 64 to 256 were within 2 %
    of each other and 1024 the slowest; 64 balances best, a single 2561-bin column
    making 41 tiles rather than 11. */
inline constexpr std::int64_t host_range2_tile = 64;

/** `Range2Policy` on `space` over `[lower, upper)`.

    On the host a tile holds about `host_range2_tile` samples: part of one row, or as
    many whole rows as fit when a row is shorter. Kokkos's host default tiles two rows
    by the whole of the last extent, which leaves a kernel over `(columns, n)` only
    ceil(columns / 2) tiles to share: one for a single column, two for four modes,
    whatever the thread count. A one-thread instance keeps Kokkos's default: it has
    nothing to share, and the small tiles cost the four-mode RDW about 1 % at one
    thread on an AMD EPYC 7763. Device builds keep Kokkos's default too. A kernel over
    a `Range2` that is elementwise changes no value with the tiling; a reduction over
    one (the boundary monitor's split removal) may add in another order, within its
    rounding (solprop SEP 25). */
template <typename LT, std::size_t LN, typename UT, std::size_t UN>
[[nodiscard]] Range2Policy Range2(const ExecutionSpace& space, const LT (&lower)[LN],
                                  const UT (&upper)[UN]) {
  static_assert(LN == 2 && UN == 2, "Range2 takes two lower and two upper bounds");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return Range2Policy(space, lower, upper);
#else
  if (space.concurrency() == 1)
    return Range2Policy(space, lower, upper);
  const auto row = std::clamp<std::int64_t>(static_cast<std::int64_t>(upper[1]) -
                                                static_cast<std::int64_t>(lower[1]),
                                            1, host_range2_tile);
  const std::int64_t tile[2] = {host_range2_tile / row, row};
  return Range2Policy(space, lower, upper, tile);
#endif
}
using Complex = Kokkos::complex<double>;

/** The boundary monitor's five sums of a state (solprop SEP 21), one reduction value:
    the energy, then the low guard's, the high guard's, the collar's and the rate's. */
struct MonitorSums {
  Kokkos::Array<double, 5> values{};
  KOKKOS_INLINE_FUNCTION MonitorSums& operator+=(const MonitorSums& other) {
    for (std::size_t slot = 0; slot < 5; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};

/** A state's whole-state transfers, in bytes, each way (solprop SEP 44): what lets a
   test hold a field-free run to no transfer at all, whichever path would have made one.
 */
struct CrossedBytes {
  std::size_t to_host{};
  std::size_t from_host{};
};

/** What the transverse edge reads of a state (solprop SEP 44): four numbers per weight
    table, and each mode's energy. */
struct EdgeNumbers {
  std::vector<double> sums;
  std::vector<double> columns;
  /** A requested collar's `(inward, windowed)` per line. */
  std::vector<double> strip;
  /** A requested collar's rectified inward current. */
  double current{};
};

/** A requested collar's lines, as Python describes them (solprop SEP 44): for each, its
    axis (0 `r`, 1 `x`, 2 `y`), its count of strip nodes and those nodes, its window,
   its inward bins `[first, last)`, the factor on its strip's sums and the measure on
   its current, and its onset's node, before and after, with the distance between the
    last two. An onset whose node is `no_onset` has no slope to read. */
struct EdgeLines {
  static constexpr std::size_t no_onset = static_cast<std::size_t>(-1);
  std::vector<std::size_t> axes, counts, positions, inward, onsets;
  std::vector<double> windows, factors, measures, steps, scale;
  double k_min{};
  /** The most one batch of a padded transform holds. */
  std::size_t batch_bytes{};
};

/** What a split-step collar's finish reads of the power it finds (solprop SEP 21,
    ADR-057), one reduction value: the energy it takes, then the energy, the low
    guard's, the high guard's and the collar's, which the monitor judges the
    occupancies on. */
struct SplitSums {
  Kokkos::Array<double, 5> values{};
  KOKKOS_INLINE_FUNCTION SplitSums& operator+=(const SplitSums& other) {
    for (std::size_t slot = 0; slot < 5; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};

/** The boundary monitor's occupancy of a state (solprop #440), one reduction value: its
    energy per walk-off bucket. A device's reduction keeps one of these per thread in
    team memory, with no atomics, so its size bounds the buckets (`SPLIT_RATE_BUCKETS`).
 */
inline constexpr std::size_t occupancy_bucket_limit = 128;
// The buckets one reduction holds. A reduction's value has to fit the device's shared
// memory per block, which on HIP's smallest block of 64 threads is `(64 + 2) *
// sizeof(value)`, with the kernel's own static memory beside it within 64 KiB. Kokkos
// checks only the first: a value that fills the block passes its check and the dispatch
// then faults on an AMD Instinct MI325X ("Illegal opcode in command stream"). So a
// reading there takes two passes, leaving half the block free.
#ifdef KOKKOS_ENABLE_HIP
inline constexpr std::size_t occupancy_reduction_slots = 64;
#else
inline constexpr std::size_t occupancy_reduction_slots = occupancy_bucket_limit;
#endif
struct OccupancySums {
  Kokkos::Array<double, occupancy_reduction_slots> values{};
  KOKKOS_INLINE_FUNCTION OccupancySums& operator+=(const OccupancySums& other) {
    for (std::size_t slot = 0; slot < occupancy_reduction_slots; ++slot)
      values[slot] += other.values[slot];
    return *this;
  }
};
#ifdef KOKKOS_ENABLE_HIP
static_assert((64 + 2) * sizeof(OccupancySums) <= 48 * 1024);
#endif

using ComplexView = Kokkos::View<Complex*, Layout, MemorySpace>;
using ConstComplexView = Kokkos::View<const Complex*, Layout, MemorySpace>;
using RealView = Kokkos::View<double*, Layout, MemorySpace>;
using ConstRealView = Kokkos::View<const double*, Layout, MemorySpace>;

/** Signal exhaustion of the validated angular rule before source assembly. */
class EnvelopeConvergenceError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** Refuse a step solprop's reference refuses with `BackendError` (solprop SEP 45
    phase 4): a coupled flow's action past its squarings or its certificate. */
class BackendRefusal : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** Process-start CPU policy (solprop SEP 46 section 5's process settings): the pool
    capacities, fixed when the runtime starts. Execution instances never exceed the
    runtime budget. */
struct ThreadPolicy {
  int kernels{}, fft{}, raman_fft{}, raman_tail_fft{};
  // The field plasma column's band form (solprop SEP 25 section 6.7.6): negative takes
  // the build's (tiled on a device, the team column on the host), zero the team
  // column, and N > 0 the tiled column with host teams of N threads.
  int band_tiles{-1};
  // Bit per pool: keep explicit requests distinct from automatic capacities.
  unsigned explicit_pools{};
  bool operator==(const ThreadPolicy&) const = default;
};
void configure_thread_policy(ThreadPolicy requested);

/** Whether the field plasma column runs its tiled form under `policy`. */
[[nodiscard]] inline bool band_tiled(const ThreadPolicy& policy) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  return policy.band_tiles != 0;
#else
  return policy.band_tiles > 0;
#endif
}

/** A tiled band column's host team width under `policy`; zero on a device, whose
    teams are its response teams. */
[[nodiscard]] inline int band_tile_threads(const ThreadPolicy& policy) noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static_cast<void>(policy);
  return 0;
#else
  return policy.band_tiles > 0 ? policy.band_tiles : 0;
#endif
}
[[nodiscard]] ThreadPolicy thread_policy();
[[nodiscard]] ExecutionSpace execution_instance(int threads);

/** Independent CPU operation classes; device dispatch is unchanged. */
enum class CpuOperation {
  stage,
  response,
  dense,
  fft,
  raman,
  copy,
  flow,
  modal,
  adams
};
inline constexpr std::size_t cpu_operation_count =
    static_cast<std::size_t>(CpuOperation::adams) + 1;
/** Process-start pool supplying an operation's capacity and explicit override. */
enum class CpuPool { kernels, fft, raman_fft, raman_tail_fft };

/** One operation class's grain: below `threshold` work it runs on one thread, below
    `full` on the medium width, and from `full` on its pool's capacity. Work is
    elements for stages and copies, active columns * fine samples for responses,
    rows * inner * samples for dense products, active bins * modes^3 for matrix
    functions, elements * history rows for the exponential Adams trial's passes, and
    actual real-side batch samples for FFT and Raman plans (tails included); the FFT
    class's threshold is where a modal or spatial FFTW plan starts to thread. */
struct OperationGrain {
  std::size_t threshold{}, full{};
  bool operator==(const OperationGrain&) const = default;
};

/** A workspace's CPU execution settings (solprop SEP 46 section 5's workspace
    settings), copied into the workspace when it is built and never read from anywhere
    global, so two live workspaces may hold different ones.

    Native code holds no values for them (solprop_native's built-in table,
    `_execution.py`, or a machine calibration supplies every field, with the
    measurements behind each built-in value) and refuses a record that leaves one
    unset (`require_cpu_execution`).

    - `grain`, indexed by `CpuOperation`.
    - A response of `response_full_samples` fine samples or more, or of
      `response_full_columns` columns or more, takes the full pool at any size above
      its threshold.
    - `medium_threads`: the medium width.
    - A compact problem (`compact_problem`: one column on one node with fewer than
      `compact_max_fine_samples` fine samples) runs its kernels on a team of at most
      `compact_threads` and its threaded transforms on that whole team.
    - `stable_kernel_team`: once a kernel pool holds a full-width instance, a
      medium-width kernel takes it too (`CpuExecutionPool::get`).
    - A batch of transforms whose row length is known threads by its shape (solprop
      issue #587): from the FFT class's threshold, one thread per
      `fft_thread_elements` samples and at most one a row -- or `medium_threads`, if
      more, when its rows are at least `fft_long_row` samples long -- and its pool's
      capacity from the class's full size. FFTW's threaded plans divide the rows
      among the threads, and split a row only where it is long. */
struct CpuExecution {
  std::array<OperationGrain, cpu_operation_count> grain{};
  std::size_t response_full_samples{}, response_full_columns{};
  int medium_threads{}, compact_threads{};
  std::size_t compact_max_fine_samples{};
  bool stable_kernel_team{};
  std::size_t fft_thread_elements{}, fft_long_row{};
  bool operator==(const CpuExecution&) const = default;
  [[nodiscard]] const OperationGrain&
  operator[](CpuOperation operation) const noexcept {
    return grain[static_cast<std::size_t>(operation)];
  }
};

/** Throw `std::invalid_argument` unless every threshold, bound and width of
    `execution` is positive and each class's full size is at least its threshold. */
void require_cpu_execution(const CpuExecution& execution);

/** How a workspace plans its host FFTW transforms (solprop SEP 46 phase 3).

    - `legacy`: the plan policy of `fftw_seed_effort`, with the process's wisdom.
    - `profile`: a machine calibration's. The workspace forgets every wisdom before it
      plans and after; a transform whose exact plan the profile holds imports that
      plan's wisdom and is seeded with `FFTW_WISDOM_ONLY` (a plan it no longer builds
      is listed in `missing`, and planned at estimate); any other is seeded at its
      group's effort, as the policy would, but with no wisdom from elsewhere.
    - `capture`: each transform is seeded alone, from no wisdom, at `capture_effort`
      within `time_limit` seconds (none when not positive), and its wisdom is kept in
      `captured`, for a calibration to time and store.

    A transform's group is `rows` (one thread), `threaded_rows` or `planes`; its key
    names its kind, sizes, strides, placement, alignment and threads
    (`fft_plan_key`). Efforts are "estimate", "measure" or "patient".

    Plans runs measured before (solprop ADR-076), in the legacy and profile modes:
    `stored` holds each transform's own wisdom by key, below the profile's `wisdom`,
    which is consulted first. A transform seeded above estimate that `stored` holds, or
    any when `keep` is set, is planned alone: the workspace's wisdom is set aside, the
    transform's stored wisdom alone imported (or, for one to measure, what the
    workspace measured so far, whose sub-plans it may share), and after the seed the
    workspace's wisdom comes back over the transform's. A transform seeded twice is
    measured once. A plan the stored wisdom built is listed in
    `reused`, as is one left at estimate whose stored wisdom was imported for its
    estimate plan; one that measured is kept in `measured` when `keep` is set. */
struct FftPlanning {
  enum class Mode { legacy, profile, capture };
  Mode mode{Mode::legacy};
  std::string rows{"measure"}, threaded_rows{"estimate"}, planes{"measure"};
  std::map<std::string, std::string> wisdom;
  std::string capture_effort{"measure"};
  double time_limit{-1.0};
  std::map<std::string, std::string> stored;
  bool keep{false};
  // What planning found: each transform's group, the wisdom captured, and the
  // profile's plans that no longer build; the stored plans reused and the transforms
  // measured, each with its own wisdom.
  std::map<std::string, std::string> groups, captured, measured;
  std::vector<std::string> missing, reused;
  // While one transform plans alone: the workspace's wisdom, set aside.
  bool alone{false};
  std::string held;
};

/** Choose a width without changing the runtime or a vendor library's pool. */
[[nodiscard]] int operation_threads(const ThreadPolicy& policy,
                                    const CpuExecution& execution, CpuPool pool,
                                    CpuOperation operation, std::size_t elements,
                                    std::size_t fine_samples = 0);

using CpuSizing = std::map<std::string, std::map<std::string, std::size_t>>;
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA) &&                   \
    !defined(KOKKOS_ENABLE_HIP)
inline constexpr bool cpu_size_policy_enabled = true;
#else
inline constexpr bool cpu_size_policy_enabled = false;
#endif

/** Own execution instances constructed only while a workspace and its plans form. */
class CpuExecutionPool {
public:
  /** A pool of `pool`'s capacity choosing widths by `execution`, which it copies. */
  CpuExecutionPool(CpuPool pool, const CpuExecution& execution)
      : pool_(pool), execution_(execution) {}
  const ExecutionSpace& get(CpuOperation operation, std::size_t elements,
                            std::size_t fine_samples = 0);
  /** Size a compact problem's instances (`CpuExecution::compact_threads`), before any
      `get`; explicit pool requests still apply as given. */
  void compact() noexcept { compact_ = cpu_size_policy_enabled; }
  /** The execution settings this pool chooses widths by. */
  [[nodiscard]] const CpuExecution& execution() const noexcept { return execution_; }
  /** The widest instance handed out so far; zero before any. */
  [[nodiscard]] int widest() const noexcept {
    return instances_.empty() ? 0 : instances_.rbegin()->first;
  }
  void clear() { instances_.clear(); }
  [[nodiscard]] CpuSizing sizing() const { return sizing_; }

private:
  CpuPool pool_;
  CpuExecution execution_;
  bool compact_{};
  std::map<int, ExecutionSpace> instances_;
  CpuSizing sizing_;
};

/** A transform batch's row length, as its CPU width and sizing record count it
    (`CpuExecutionPool::get`'s `fine_samples` for the FFT and Raman classes): extent 0
    counts the rows and the remaining extents make one row, so a transverse plane is
    one row; a rank-1 view is one row. */
template <class View> [[nodiscard]] std::size_t row_length(const View& rows) noexcept {
  const std::size_t count = View::rank() == 1 ? 1 : rows.extent(0);
  return count == 0 ? 0 : rows.size() / count;
}

/** Whether the host FFTW's threaded loops take a scoped team (`CpuTeamScope`): the
    OpenMP FFTW with its loop callback (FFTW 3.3.9 and later). Without it a compact
    problem's transforms stay serial. The first call installs the callback, whose
    loops take the runtime's whole team, as FFTW's own do, outside a scope. */
[[nodiscard]] bool fftw_team_loops() noexcept;

/** Record that the build's BLAS is an OpenMP OpenBLAS, so that a compact problem's
    scope holds it at one thread (`CpuTeamScope`). */
void register_openmp_blas(bool registered) noexcept;

/** While it lives, the calling thread's threaded FFTW loops run on a team of
    `threads` (at least their own job count), so that a compact problem's transforms
    keep its kernels' team instead of waking the whole runtime; zero restores the
    runtime's team. A nonzero team also holds an OpenMP OpenBLAS at one thread, through
    the calling thread's OpenMP default width: a compact problem's products are a row
    or two deep, and an OpenBLAS region at the fitted width between the team's narrower
    ones made GNU OpenMP end and restart the surplus threads at every pair (16 cores of
    a Neoverse V2: about 100,000 thread creations in a short GNLSE window). The
    outermost scope on a thread also sets its OpenMP default to the runtime's width
    and restores it at its end. Scopes nest. */
class CpuTeamScope {
public:
  explicit CpuTeamScope(int threads) noexcept;
  ~CpuTeamScope();
  CpuTeamScope(const CpuTeamScope&) = delete;
  CpuTeamScope& operator=(const CpuTeamScope&) = delete;
  CpuTeamScope(CpuTeamScope&&) = delete;
  CpuTeamScope& operator=(CpuTeamScope&&) = delete;

private:
  // The enclosing scope's team; unread where FFTW has no loop callback.
  [[maybe_unused]] int saved_;
  // Whether this scope holds the registered OpenBLAS at one thread.
  bool holds_blas_{};
};

/** How long, in milliseconds, LLVM's OpenMP runtime keeps idle threads spinning while
    a native call runs, from the outermost `CpuTeamScope` on a thread to its end, where
    the caller's block time returns; zero leaves the process's (solprop ADR-071). GNU's
    runtime reads its spin count once, at start, from GOMP_SPINCOUNT. */
void configure_native_spin(int milliseconds);

/** LLVM's runtime's block time inside a scope, in milliseconds; -1 with another
    runtime. A test hook. */
[[nodiscard]] int scoped_block_time() noexcept;

/** The calling thread's OpenMP default width inside a scope (`CpuTeamScope`): the
    runtime's, on any thread (solprop issue #130); one without OpenMP. A test hook. */
[[nodiscard]] int scoped_openmp_default() noexcept;

/** solprop SEP 28 section 4.2's per-evaluation reference, independent of any
    workspace: the seconds of each timed forward-then-backward pair of an in-place,
    unnormalised rank-3 complex transform of `values` (row-major
    `(planes, n_y, n_x)`, copied in once) through Kokkos-FFT on the FFT execution
    space, after planning and `warm` untimed pairs. */
[[nodiscard]] std::vector<double>
reference_fft3_pairs(const std::complex<double>* values, std::size_t planes,
                     std::size_t n_y, std::size_t n_x, std::size_t warm,
                     std::size_t pairs);

/** The same reference's streaming volume-pass: the seconds of each timed scaled copy
    `b = s a` between two complex volumes of `elements` first touched in parallel on
    the kernels' execution space, after `warm` untimed copies. */
[[nodiscard]] std::vector<double>
reference_stream_copies(std::size_t elements, std::size_t warm, std::size_t repeats);

enum class FieldFamily { field_real, field_analytic, envelope };

enum StatisticsFlag : std::uint64_t {
  statistics_omega0 = 1ULL << 0U,
  statistics_energy = 1ULL << 1U,
  statistics_energy_window = 1ULL << 2U,
  statistics_peak_power = 1ULL << 3U,
  statistics_peak_power_window = 1ULL << 4U,
  statistics_peak_power_total = 1ULL << 5U,
  statistics_fwhm = 1ULL << 6U,
  statistics_electron_density = 1ULL << 7U,
  statistics_peak_ionisation_rate = 1ULL << 8U,
  statistics_peak_intensity = 1ULL << 9U,
  statistics_quadrature_error = 1ULL << 10U,
  statistics_populations = 1ULL << 11U,
  // solprop SEP 18's run diagnostics: the six maxima over every node of the field
  // plasma column's integration (exposure dip and rise, |paid - freed| and freed, the
  // free-current integrand in the first 2K samples and anywhere).
  statistics_plasma_integration = 1ULL << 12U,
};

/** The widest band rule solprop designs (SEP 18 section 5): K <= 8, so 9 taps. */
inline constexpr std::size_t plasma_max_taps = 9;

/** The six parts of SEP 18's run diagnostics, in solprop's order. */
inline constexpr std::size_t plasma_diagnostic_parts = 6;

/** Fixed, Python-free problem data uploaded once by a native state. */
struct Problem {
  std::size_t n_x{}, n_y{};
  std::vector<Complex> spatial_k_squared, spatial_k_squared_low, spatial_numerator,
      spatial_clamped;
  std::vector<double> spatial_kperp_sq, spatial_kperp_sq_low, spatial_frame,
      spatial_spectral_rate;
  // solprop kernel schema 45 (#231): the user's share of the spectral rate, read only
  // by `absorbed_power`; the operator takes the total.
  std::vector<double> spatial_user_rate;
  std::vector<double> spatial_absorber_rate;
  // solprop kernel schema 31 (SEP 22 phase 2): a transverse collar's power rate per
  // column, added to the field attenuation on active bins; zeros leave the operator.
  std::vector<double> spatial_kperp_rate;
  double spatial_evanescent_rate{};
  // solprop SEP 23 (free space only): the stored transverse axes k_x (n_x) and k_y
  // (n_y) in rad/m, FFT order, column ix + n_x iy, which the far-field statistics read;
  // and the slots of the problem's own far-field request, which size the band partials
  // at construction (zero for none). Empty and zero elsewhere.
  std::vector<double> spatial_k_x, spatial_k_y;
  std::size_t spatial_far_field_slots{};
  // solprop's hybrid planar kind (kernel schema 15): n_p slab
  // channels in x on the real (n_x, n_p) synthesis and (n_p, n_x) projection, times a
  // raw DFT in y. Columns are (p, k_y) with the channel fastest, n_modes = n_p * n_y,
  // and nodes (x, y) with x fastest, n_nodes = n_x * n_y; node_chunk holds whole y
  // rows. Row-major (n_p, n_omega) wall terms kappa_p^2 enter the separable operator,
  // and spatial_absorber_rate is then per column. Zero n_p and empty kappa elsewhere.
  std::size_t n_p{};
  std::vector<Complex> spatial_kappa_sq;
  // solprop kernel schema 35 (SEP 34): a non-separable spatial run's plane measure,
  // the entrance's material squares (n_omega) and transverse squares (n_modes) whose
  // root is solprop's `longitudinal_root`; `flux_table` when the material square
  // varies along z and is uploaded as a `TableKind::flux` table of n_omega values. A
  // separable run's measure is its own operator's (these stay empty). `flux_closure`
  // allocates the magnetic closure's buffers at construction, for a request with a
  // local flux statistic.
  std::vector<Complex> flux_k_squared, flux_k_squared_low;
  std::vector<double> flux_kperp_sq, flux_kperp_sq_low;
  bool flux_table{};
  bool flux_closure{};
  /** Free space or the hybrid planar kind: the separable operator and a y FFT. */
  [[nodiscard]] bool separable_geometry() const noexcept { return n_x != 0; }
  [[nodiscard]] bool slab_geometry() const noexcept { return n_p != 0; }
  [[nodiscard]] bool free_geometry() const noexcept { return n_x != 0 && n_p == 0; }
  std::size_t n_omega{};
  std::size_t n_t{};
  std::size_t n_to{};
  std::size_t n_modes{1};
  std::size_t n_pol_state{1};
  std::size_t n_pol_transform{1};
  std::size_t n_nodes{1};
  std::size_t node_chunk{1};
  // How the workspace's CPU operations thread (solprop SEP 46): set by the caller,
  // refused when unset (`require_cpu_execution`).
  CpuExecution cpu_execution{};
  // How the workspace plans its host transforms (solprop SEP 46 phase 3).
  FftPlanning fft_planning{};
  std::size_t n_stages{};
  // solprop SEP 24 phase 3 (kernel schema 32): the low-storage RK4IP march. With it no
  // Runge-Kutta view is allocated (stages, accumulators, accept buffer, residual,
  // recycled stage, whole-step propagator, stage inverse); the accumulator register
  // is, the stage register unless it is elided (solprop's
  // `low_storage_stage_register`), and `low_storage_flows` flow volumes (0
  // recomputed, 1 a constant operator's table, 2 a varying operator's step pair).
  bool low_storage{};
  bool low_storage_stage_register{true};
  std::size_t low_storage_flows{};
  // Whether a separable route's first input crosses to the transverse scratch through
  // tiles of team scratch, as a device's does.
  bool low_storage_tiled_input{};
  std::size_t n_phi_terms{};
  std::size_t n_prefactor_terms{};
  // solprop SEP 29 G1: Phi or the prefactor is the general z table. Its terms are then
  // `n_phi_terms` (or `n_prefactor_terms`) slots that `table_evaluate` fills with the
  // table's values at a trial's positions, and `phi_terms` (`prefactor_terms`) is
  // empty.
  bool phi_table{};
  bool prefactor_table{};
  FieldFamily family{FieldFamily::field_real};
  bool nonlinear{};
  bool modified_noise{};
  bool in_stage_absorber{};
  bool split_step_absorber{};
  bool first_same_as_last{};
  std::vector<std::size_t> crop_index;
  std::vector<unsigned char> active_mask;
  std::vector<unsigned char> norm_mask;
  std::vector<double> omega_window;
  std::vector<double> omega;
  std::vector<double> power_weight;
  std::vector<double> time_window;
  std::vector<double> time_window_derivative;
  std::vector<double> absorber_fine;
  std::vector<double> absorber_coarse;
  std::vector<double> spatial_absorber_matrix;
  std::vector<double> spatial_spectral_weights;
  std::vector<Complex> spatial_on_axis;
  std::vector<Complex> carrier;
  std::vector<Complex> phi_terms;
  std::vector<Complex> prefactor_terms;
  std::vector<double> synthesis_matrix;
  std::vector<double> projection_matrix;
  std::vector<double> projection_coarse_matrix;
  std::vector<double> tableau_matrix;
  std::vector<double> tableau_nodes;
  std::vector<double> solution_weights;
  std::vector<double> error_weights;
  double synthesis{};
  double projection{};
  double kerr_coefficient{};
  double self_factor{};
  double conjugate_factor{};
  double harmonic_factor{};
  bool take_real{};
  bool plasma{};
  bool envelope_plasma{};
  // solprop's private, test-only gated plasma benchmark: the field plasma's current
  // enters as w(t) times its running integral, added to the local polarisation,
  // instead of the held current's two collared spectra. Never set by a user run.
  bool plasma_gated{};
  // Channel-major 4 x 512 split angular rules and coarse-axis current factors.
  std::vector<double> envelope_rule_nodes, envelope_rule_weights, envelope_drude_factor;
  std::vector<Complex> envelope_loss_factor;
  bool raman{};
  bool raman_thg{};
  bool raman_tensor_capable{};
  bool stash_plasma{};
  bool weighted_controller{};
  double dt_fine{};
  double dt{};
  std::size_t n_raman_terms{};
  double raman_fraction{};
  double raman_epsilon0{};
  std::vector<double> raman_undamped;
  std::vector<double> raman_tensor_weights;
  double charge_mass_ratio{};
  double ionisation_potential{};
  double log_field_min{};
  double d_log_field{};
  double field_min{};
  double field_max{};
  double rate_max{};
  std::size_t rate_intervals{};
  std::vector<double> rate_coefficients;
  std::size_t n_channels{1};
  // solprop SEP 18 (kernel schema 25): the field plasma column's running-integral
  // rule, one per problem and shared by every channel. Each running integral is the
  // cumulative trapezoid of the integrand filtered by the symmetric taps
  // (c_0, ..., c_K), centre first, with the window's ends closed by point reflection,
  // and the exposure floored at zero where the fraction is formed. The trapezoid rule
  // is `plasma_band == false` with the one identity tap, and runs the version-24
  // column unchanged.
  bool plasma_band{};
  std::vector<double> plasma_taps{1.0};
  [[nodiscard]] std::size_t plasma_half_width() const noexcept {
    return plasma_taps.empty() ? 0 : plasma_taps.size() - 1;
  }
  // Channel-major rows: log_min, d_log, field_min, field_max, rate_max,
  // interval count, coefficient offset, charge/mass ratio, ionisation potential.
  std::vector<double> channel_parameters, channel_coefficients;
  std::size_t n_raman_components{1};
  std::vector<double> raman_row_component, raman_component_fraction;
  // solprop's general modal route. Row-major: synthesis (map terms,
  // nodes, components, modes, n_omega); projection (map terms, modes, nodes,
  // components, n_omega); tensor (nodes, C, C, C, C); delayed tensor (responses,
  // terms, C, C, C, C); delayed spectrum (responses, terms, n_to + 1); node response
  // (nodes), the owning response plus one and zero for none. Empty otherwise. The
  // map terms are general_map_intervals x general_map_n_deg Chebyshev coefficients
  // along z, one for a constant guide; a moving guide's are combined per stage over
  // TrialScalars::transform_weights.
  bool general_modal{};
  bool general_real_form{};
  std::size_t general_map_intervals{1}, general_map_n_deg{1};
  std::size_t general_components{}, general_nodes{}, general_node_chunk{};
  std::size_t general_responses{}, general_terms{};
  double general_pad_scale{}, general_epsilon0{};
  std::vector<Complex> general_synthesis, general_projection, general_frequency_factor,
      general_delayed_spectrum;
  std::vector<double> general_tensor, general_delayed_tensor;
  std::vector<std::size_t> general_node_response;
  // solprop's linear.coupled_flow. flow_slot (n_omega)
  // numbers the bins a section's generator is non-zero at 1, 2, ... in increasing
  // frequency, zero elsewhere. With an explicit residual (flow_terms > 0), row-major
  // frozen blocks (sections, active, modes, modes) and residual terms (terms, active,
  // modes, modes); both empty otherwise. The actions are loaded per step.
  bool coupled_flow{};
  std::size_t flow_active{}, flow_sections{}, flow_terms{};
  std::vector<std::size_t> flow_slot;
  std::vector<Complex> flow_frozen, flow_residual_terms;
  // The actions' native formation, all present or all
  // empty: each section's eigenbasis of its Schur factor, row-major left `W` and
  // right `Y` (sections, active, modes, modes) and eigenvalues (sections, active,
  // modes); the route (sections, active), zero where a bin takes its eigenbasis and
  // 1, 2, ... numbering in row-major order the bins formed by scaling and squaring,
  // whose Schur factors `Z` and `T` are the rows of the fallback arrays (fallback,
  // modes, modes).
  std::vector<Complex> flow_left, flow_right, flow_eigenvalues;
  std::vector<std::size_t> flow_route;
  std::vector<Complex> flow_fallback_vectors, flow_fallback_triangular;

  // solprop SEP 6's frequency-dependent electronic response
  // (`responses.dispersive_kerr`, kernel schema 24) on the fused node path, laid out by
  // the adapter as solprop's `dispersive_terms_layout` lays it out (`dispersive.hpp`).
  // The form is 0 (a field grid's full cubic), 1 (the number-conserving sector) or 2
  // (an envelope's full cubic). Row-major on the leaf's spectral layout of `bins` (n_to
  // / 2 + 1 on a field grid, n_to on an envelope): the input multipliers (fields,
  // bins), each output term's multiplier (terms, bins) with its product row and factor,
  // the static coefficients and the polynomial's pair constants (channels, 2) per
  // channel, the groups as offsets (groups + 1) into members (channels), each symmetric
  // mixture's channels' `g^4` (channels, bins) and the input and output scales (bins).
  // On the general modal route (SEP 6 Phase 3b) each group's key is its node region
  // (groups), the polynomial forms one cube per distinct key, and every general node's
  // region (general_nodes) says where a channel acts; both are empty elsewhere, where
  // every channel acts at every node. Empty otherwise.
  bool dispersive{};
  bool dispersive_symmetric{};
  int dispersive_form{};
  std::vector<double> dispersive_inputs, dispersive_term_multipliers,
      dispersive_coefficients, dispersive_pair_constants, dispersive_quartics,
      dispersive_input_scale, dispersive_output_scale;
  std::vector<std::size_t> dispersive_term_rows, dispersive_term_channels,
      dispersive_group_offsets, dispersive_group_members, dispersive_group_keys,
      dispersive_node_region;

  [[nodiscard]] std::size_t n_columns() const noexcept { return n_modes * n_pol_state; }
  [[nodiscard]] std::size_t effective_node_pol() const noexcept {
    return n_pol_transform * n_pol_state;
  }
  [[nodiscard]] std::size_t raman_drives() const noexcept {
    if (effective_node_pol() == 2)
      for (std::size_t index = 1; index < raman_tensor_weights.size(); index += 2)
        if (raman_tensor_weights[index] != 0.0)
          return 3;
    return 1;
  }
  [[nodiscard]] std::size_t raman_kernels() const noexcept {
    return raman_drives() == 3 ? 2 : 1;
  }
};

/** The most samples an Adams formula of solprop's shipped orders combines. */
inline constexpr std::size_t adams_max_samples = 8;

/** The most defect norms an Adams trial scores: its order's and its two neighbours'. */
inline constexpr std::size_t adams_max_norms = 3;

/**
 * The most channels of a coupled block whose channel vectors a host task of the
 * resident Adams engine holds on its stack (solprop SEP 16 phase 6). A wider block's
 * host passes hold them in a scratch row per bin, allocated with the history (solprop
 * SEP 47 phase 4). A device build's passes take a thread per (bin, channel) and
 * hold no channel vectors, so it steps any width.
 */
inline constexpr std::size_t adams_max_channels = 16;

/**
 * One Adams trial's coefficient tables, passed by value to the resident engine's
 * kernels (solprop SEP 16 phase 3b). `slot` names history rows newest first;
 * `predictor` weights the first `order` of them; `corrector` and each of `n_rows`
 * defect rows weight `(source, slot[0], ...)` over `width` samples.
 */
struct AdamsWeights {
  std::size_t slot[adams_max_samples]{};
  double predictor[adams_max_samples]{};
  double corrector[adams_max_samples]{};
  double rows[adams_max_norms][adams_max_samples]{};
  std::size_t order{};
  std::size_t width{};
  std::size_t n_rows{};
};

/**
 * One exponential Adams trial's coefficient tables (solprop SEP 16 phase 3b): the
 * predictor's Lagrange coefficients `predictor[sample][monomial]` over the `order`
 * newest rows, the corrector's over `(source, slot[0], ...)` and, per scored defect,
 * the divided-difference coefficients over the same `width` samples followed by the
 * node polynomial's `count` monomial coefficients.
 */
struct ExponentialWeights {
  std::size_t slot[adams_max_samples]{};
  double predictor[adams_max_samples][adams_max_samples]{};
  double corrector[adams_max_samples][adams_max_samples]{};
  double rows[adams_max_norms][2 * adams_max_samples]{};
  std::size_t order{};
  std::size_t width{};
  std::size_t count{};
  std::size_t n_rows{};
  // A varying operator's transported compensation (solprop SEP 16 section 4.2): the
  // carried samples' spans 1 - theta_j, and the predictor's, the corrector's and each
  // defect's weights over (endpoint, carried...), still to be multiplied by the step.
  bool compensated{};
  double span[adams_max_samples]{};
  double comp_predictor[adams_max_samples]{};
  double comp_corrector[adams_max_samples + 1]{};
  double comp_rows[3][adams_max_samples + 1]{};
  // Dense observation's weights over (endpoint, history...) at the fraction, and the
  // samples' normalised positions.
  double dense_weight[adams_max_samples + 1]{};
  double dense_theta[adams_max_samples + 1]{};
  // A scalar carrier frame (solprop SEP 35): sigma = i shift. The moments are those
  // of x - sigma, and history row t (the t-th of `slot`) is read rotated by
  // factor[t] = exp(sigma (1 - theta_t)); the new source's factor is one.
  bool shifted{};
  double shift{};
  // exp(-sigma), formed once a trial.
  double unshift_re{1.0};
  double unshift_im{};
  double factor_re[adams_max_samples]{};
  double factor_im[adams_max_samples]{};
};

/** Small host bundle for all stages and the whole-step propagator. */
struct TrialScalars {
  std::vector<double> phi_delta;
  std::vector<double> phi_at;
  std::vector<double> prefactor;
  std::vector<double> density;
  std::vector<double> taper;
  std::vector<double> preionisation;
  std::vector<double> raman_rates;
  std::vector<double> kerr_density, plasma_densities, plasma_preionisation,
      raman_densities;
  // A coupled flow's step section and its stages' residual weights (n_stages, terms).
  std::size_t flow_section{};
  std::vector<double> flow_weights;
  // A moving guide's map weights (n_stages, map terms), zero outside the section each
  // stage lies in (solprop kernel schema 14); empty for maps that do not move.
  std::vector<double> transform_weights;
  // Each stage's channel densities of SEP 6's dispersive response (n_stages,
  // channels), m^-3; empty without it.
  std::vector<double> dispersive_densities;
};

/** Which general z table a call names (solprop SEP 29 G1). */
enum class TableKind { phi, prefactor, rate, flux };

/** Whether `problem` runs on a compact CPU team (`CpuExecution::compact_threads`): one
    column on one node, fewer than its record's `compact_max_fine_samples` fine
    samples, and neither the general modal route, a coupled flow nor a separable
    geometry. */
[[nodiscard]] bool compact_problem(const Problem& problem) noexcept;

class GeneralWorkspace;

/** Multicolumn general state with batched FFTs and chunked node transforms. */
class GeneralState {
public:
  explicit GeneralState(Problem problem);
  ~GeneralState();
  GeneralState(const GeneralState&) = delete;
  GeneralState& operator=(const GeneralState&) = delete;
  GeneralState(GeneralState&&) = delete;
  GeneralState& operator=(GeneralState&&) = delete;

  /** Load the field and noise from host memory, one copy into the memory space. */
  void set_initial(std::span<const Complex> field, std::span<const Complex> noise);
  [[nodiscard]] std::vector<double> trial_step(double dz, const TrialScalars& scalars);
  void prepare_trial(double dz, const TrialScalars& scalars);
  /** solprop's Adams source at an anchor-frame field (SEP 14 phase 2). */
  [[nodiscard]] std::vector<Complex> adams_source(const std::vector<Complex>& input,
                                                  const TrialScalars& scalars);
  /** Adopt an Adams step's end state through the shared finish. */
  void adopt(const std::vector<Complex>& field, double dz);
  /**
   * One step of solprop's low-storage RK4IP march (SEP 24 phase 3): four register
   * passes accepted in place, returning the raw accepted state's squared norm over the
   * norm mask. `scalars` holds one row per position -- 0 the step's start, 1 its
   * midpoint, 2 its end, any later rows ignored -- and its `phi_delta` rows 0 and 1
   * the flows' intervals `P` and `Q`, row 2 the table's, read only when
   * `rebuild_table` builds a constant operator's table.
   */
  [[nodiscard]] double low_storage_step(double dz, const TrialScalars& scalars,
                                        bool rebuild_table);
  /** The resident Adams history (solprop SEP 16 phase 3b, solprop ADR-043). */
  void adams_allocate(std::size_t slots, bool compensated, bool interpolate = true);
  void adams_compensation_start(std::size_t slot, const std::vector<double>& rate);
  void adams_start(std::size_t slot, const TrialScalars& scalars);
  [[nodiscard]] double adams_magnitude();
  void adams_scale(double rtol, double atol, const std::vector<double>& weights);
  [[nodiscard]] std::vector<double>
  adams_fused_trial(double dz, const AdamsWeights& weights,
                    const std::vector<double>& divisor, const TrialScalars& scalars);
  void adams_fused_accept(double dz, const AdamsWeights& keep, std::size_t free,
                          const AdamsWeights& dense);
  void adams_hold();
  void adams_restore();
  void adams_observe(const AdamsWeights& weights, const TrialScalars& scalars);
  [[nodiscard]] std::vector<double>
  adams_exponential_trial(double dz, const ExponentialWeights& weights,
                          const std::vector<double>& divisor, const TrialScalars& step,
                          const TrialScalars& sample, const std::vector<double>& rate);
  void adams_exponential_accept(double dz, std::size_t free, const AdamsWeights& dense);
  void adams_exponential_observe(double xi, double dz,
                                 const ExponentialWeights& weights);
  /**
   * A Runge-Kutta march's dense output in a native session (solprop SEP 47 section
   * 5.3), DOPRI5's continuous extension (Hairer, Norsett and Wanner II.6) as solprop's
   * `extension_coefficients` and `_observe_inside` form it. `rk_dense_allocate` keeps
   * the tableau's dense-output weights, one per stage, and the views the extension
   * needs; `rk_dense_hold` contracts the accepted, unfinished trial over `dz` -- the
   * state before it and its stages -- into the extension's five coefficients and marks
   * the bins the step's flow changes by more than an e-fold;
   * `rk_dense_observe` holds the committed state aside and makes the state the
   * extension at the fraction `theta`, carried from the anchor by the flow of `carry`
   * (a trial's arrays from the anchor to the plane) and masked; `rk_dense_restore`
   * makes the held state current again; `rk_dense_release` frees the views.
   */
  void rk_dense_allocate(const std::vector<double>& weights);
  void rk_dense_hold(double dz);
  void rk_dense_observe(double theta, const TrialScalars& carry);
  void rk_dense_restore();
  void rk_dense_release() noexcept;
  /**
   * The resident Adams history's persistent rows, for a resume (solprop SEP 29 G2):
   * `"store"`, and `"states"`, `"rates"` and `"applied"` where allocated, each
   * `(slots, columns, n_omega)`. Scratch views are not rows: a trial rebuilds them.
   */
  [[nodiscard]] std::vector<std::string> adams_row_names() const;
  /** A named row view's extents; refuses an unknown or unallocated name. */
  [[nodiscard]] std::array<std::size_t, 3>
  adams_row_extents(const std::string& name) const;
  /** A lossless host copy of a named row view, in its LayoutRight order. */
  [[nodiscard]] std::vector<Complex> adams_rows_to_host(const std::string& name) const;
  /** The same copy written straight into `destination`, of `size` elements. */
  void adams_rows_to_host(const std::string& name, std::complex<double>* destination,
                          std::size_t size) const;
  /** Replace a named row view with a host copy of its extents; forget any trial. */
  void adams_rows_from_host(const std::string& name,
                            const std::vector<Complex>& values);
  /**
   * The general z tables (solprop SEP 29 G1). `table_upload` makes a table resident:
   * Phi's, the prefactor's, or Phi's rate (the derivative series, for the compensated
   * Adams march); `table_evaluate` writes Phi's and the prefactor's values at each
   * position into the given term slots, and `table_rate` the rate's value into a Phi
   * term slot. `table_terms_to_host` copies one term slot to the host, and
   * `phase_sum_to_host` evaluates the stage kernels' phase sum over the Phi terms for
   * one row of weights; both are diagnostics.
   */
  void table_upload(TableKind kind, std::vector<double> edges, double lower,
                    double upper, const std::complex<double>* coefficients,
                    std::size_t n_intervals, std::size_t n_terms, std::size_t n_values);
  void table_evaluate(const std::vector<std::size_t>& slots,
                      const std::vector<double>& positions);
  void table_rate(std::size_t slot, double z, bool before);
  [[nodiscard]] std::vector<Complex> table_terms_to_host(TableKind kind,
                                                         std::size_t slot) const;
  [[nodiscard]] std::vector<Complex>
  phase_sum_to_host(const std::vector<double>& weights);
  [[nodiscard]] std::vector<double> weighted_error(const std::vector<double>& scale);
  /** Load a coupled flow's actions; see `CoupledFlowAction::set_actions`. */
  void set_flow_actions(const std::vector<Complex>& forward,
                        const std::vector<Complex>& inverse,
                        const std::vector<unsigned char>& identity);
  /** A coupled flow's held actions; see `CoupledFlowAction::actions_to_host`. */
  [[nodiscard]] std::vector<Complex> flow_actions_to_host(bool inverse) const;
  /** The last exponential Adams trial's coupled block functions, row-major
      `(slots + 1, n_active, n_modes, n_modes)`, the flow first: an observation for
      tests (solprop SEP 47 phase 4). */
  [[nodiscard]] std::vector<Complex> adams_block_to_host() const;
  /** The last Adams trial's exponent `Delta Phi`, row-major `(n_columns, n_omega)`:
      an observation for tests. */
  [[nodiscard]] std::vector<Complex> adams_exponent_to_host() const;
  /** Form a coupled flow's actions natively; see `CoupledFlowAction::form_actions`. */
  [[nodiscard]] std::vector<double>
  form_flow_actions(std::size_t section, double dz,
                    const std::vector<double>& fractions);
  void prepare_raman(const std::vector<double>& rates,
                     const std::vector<double>& densities = {});
  /**
   * The spatial statistics of the accepted state at the plane `z`, measured by the
   * medium's axial flux (solprop SEP 34): flags 1 energy, 2 on-axis peak intensity, 4
   * peak intensity, 8 on-axis spectrum, 16 the widths' fluence block (the signed
   * fluence at each node, the axis fluence, which a separable route leaves NaN since
   * only a radial width reads it, then the bound on the largest magnitude), 32 a
   * Cartesian plane's fluence moments, which hold its scaled profile for
   * `plane_fluence_trace` (the lowest value, its node, the bound on the largest
   * magnitude, the largest value, the count of values that aren't finite, then the
   * scaled sum and its x and y index moments), packed in that order.
   */
  [[nodiscard]] std::vector<double>
  spatial_statistics(std::uint64_t flags, double intensity_factor, double z);
  /**
   * The second moment and the azimuthal average of the plane's scaled fluence that a
   * `spatial_statistics` call with flag 32 left, on rings solprop's own rule fixes,
   * with the centre in metres from the axes' first samples. Returns the second-moment
   * sum, then one average per ring. The profile itself stays in the execution space.
   */
  [[nodiscard]] std::vector<double>
  plane_fluence_trace(double center_x, double center_y, double dx, double dy,
                      double extent, double spacing, std::size_t n_radii,
                      const std::vector<double>& cosines,
                      const std::vector<double>& sines);
  /**
   * The modal state's spectrum at one physical point: `matrix`, one or two rows of
   * `n_modes` columns, contracted with the modes, as `(n_rows * n_pol, n_omega)`
   * samples. What an on-axis scan needs of the state.
   */
  [[nodiscard]] std::vector<Complex> axis_spectrum(const std::vector<double>& matrix);
  /**
   * solprop SEP 19's on-axis plasma populations of the accepted state: `matrix`'s row
   * (as `axis_spectrum` takes it, the pad scale and the taper in it) on the fine time
   * grid, scanned by the node scan with each channel's preionisation. Returns each
   * channel's final ionised fraction and peak rate, `(n_channels, 2)`.
   */
  [[nodiscard]] std::vector<double>
  axis_populations(const std::vector<double>& matrix,
                   const std::vector<double>& channel_preionisation);
  /**
   * solprop SEP 23's far-field statistics of the accepted state on a Cartesian or line
   * grid, one banded pass; see `FarFieldPass::run` (`far_field.hpp`) for the tables,
   * the flags (from 32) and the packed order.
   */
  [[nodiscard]] std::vector<double>
  aperture_statistics(std::uint64_t flags, const std::vector<double>& apertures,
                      int exponent, const std::vector<double>& regions,
                      double intensity_factor);
  [[nodiscard]] std::vector<double>
  statistics(std::uint64_t flags, const std::vector<double>& windows,
             const std::vector<double>& on_axis, double density, double taper,
             double preionisation, double intensity_factor,
             const std::vector<double>& channel_preionisation = {},
             const std::vector<double>& power_weight = {});
  /**
   * solprop SEP 47: a general modal basis's statistics, functionals of the physical
   * flux `a.H W a` (`solprop.stats._whitened`). The state's columns are whitened by
   * `root`, `W^(1/2)` at each frequency, `(n_omega, n_modes, n_modes)` row-major
   * (empty for the identity), into a statistics scratch, and the temporal `flags`
   * (omega0, energy, peak power total and the width's trace) are evaluated on
   * `c = W^(1/2) a`. Returns their packed values, then each column's `sum |c|^2`,
   * then, with `spectral`, each bin's `sum |c|^2` over the columns. The state is
   * unchanged.
   */
  [[nodiscard]] std::vector<double>
  whitened_statistics(const std::vector<Complex>& root, std::uint64_t flags,
                      const std::vector<double>& windows, double intensity_factor,
                      bool spectral);
  void finish(double dz);
  void promote_fsal();
  void clear_fsal();
  void set_absorber_quiet(bool quiet);
  /**
   * Bind solprop's boundary monitor (SEP 21): its four time weights, row-major
   * `(4, n_t)`, and its columns' quadrature by mode (empty for unit weights). From
   * here on each split-step finish adds what it takes to an accumulator, which
   * `monitor_reading` drains, and keeps the energies of the power it found.
   */
  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept;
  /**
   * The bound monitor's reading of the accepted state, over every column: `sum_t P`,
   * the four weighted sums `sum_t w_k P`, what the split collar took since the last
   * reading, and `sum_t P` and the two guards' and the collar's sums of the power the
   * last step's split found (zeros before the first such step).
   */
  [[nodiscard]] std::vector<double> monitor_reading();
  /**
   * Bind the split collar's walk-off bucket of each frequency bin, by mode
   * `(n_modes, n_omega)`, `count` buckets (solprop #440): from here on
   * `monitor_reading` also returns the state's energy per bucket, the fastest first.
   */
  void bind_occupancy(const std::vector<unsigned char>& buckets, std::size_t count);
  [[nodiscard]] double quadrature_error() const;
  [[nodiscard]] double quadrature_peak();
  [[nodiscard]] std::string synthesis_provider() const;
  [[nodiscard]] std::string projection_provider() const;
  [[nodiscard]] std::string gemm_shape_policy() const;
  [[nodiscard]] std::size_t response_team_size() const noexcept;
  [[nodiscard]] std::vector<Complex> field_to_host() const;
  /** The field into `destination`, of the field's size: one copy from the space. */
  void field_to_host(std::span<Complex> destination) const;
  /** The committed field where the host can read it in place, `columns * n_omega`
      elements in row-major order, after the state's work is fenced; null where the
      memory space is not host-accessible. A reader that passes it through the host
      adds its bytes with `count_to_host` (solprop SEP 47 section 5.4). */
  [[nodiscard]] const Complex* host_field() const;
  /** Count `bytes` read out of the state by a host reader of `host_field`. */
  void count_to_host(std::size_t bytes) const noexcept;
  /** What the transverse edge's reading sums with (solprop SEP 44). */
  void bind_edge(const std::vector<double>& faces, const std::vector<double>& clip,
                 const std::vector<double>& totals,
                 const std::vector<double>& synthesis, std::size_t first_row);
  /** The bytes through this state's crossings since it was built, each way. */
  [[nodiscard]] CrossedBytes crossed_bytes() const noexcept;
  /** The edge's region sums and each mode's energy, where the state lives. */
  /** A requested collar's lines, whose strip and current a reading then forms. */
  void bind_edge_lines(const EdgeLines& lines);
  EdgeNumbers edge_reading(bool faces, bool whole, bool totals, bool strip,
                           bool current);
  /** The threads a large crossing takes (`crossing.hpp`), from solprop's budget. */
  void set_crossing_workers(std::size_t workers) noexcept;
  [[nodiscard]] std::vector<Complex> stage_to_host(std::size_t stage) const;
  void stage_to_host(std::size_t stage, std::span<Complex> destination) const;
  /**
   * The last evaluation's projected polarisation spectrum on the fine rule, or on the
   * embedded one (finishing SEP 6's dispersive products there first), a diagnostic.
   */
  [[nodiscard]] std::vector<Complex> projected_to_host(bool coarse);
  /** How SEP 6's dispersive response runs: empty without it. */
  [[nodiscard]] std::map<std::string, bool> dispersive_path() const;
  [[nodiscard]] std::map<std::string, std::size_t> memory_ledger() const;
  [[nodiscard]] CpuSizing execution_sizing() const;
  /** The CPU execution record the workspace's pools choose widths by. */
  [[nodiscard]] const CpuExecution& cpu_execution() const;
  /** The workspace's FFT planning, with what its planning found. */
  [[nodiscard]] const FftPlanning& fft_planning() const;
  [[nodiscard]] double planning_seconds() const noexcept;
  [[nodiscard]] double raman_planning_seconds() const noexcept;
  [[nodiscard]] std::string raman_fft_provider() const;
  [[nodiscard]] std::map<std::string, int> raman_fft_tasks() const;
  [[nodiscard]] std::size_t raman_kernel_transforms() const noexcept;
  [[nodiscard]] bool fsal_ready() const noexcept;
  [[nodiscard]] std::size_t rk_source_evaluations() const noexcept;
  /** On-axis row scans made so far (`axis_populations`), cumulative. */
  [[nodiscard]] std::size_t axis_scans() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  void close() noexcept;

private:
  /** `function(workspace, arguments...)` inside the workspace's CPU team scope
      (`CpuTeamScope`); defined and used in general.cpp alone, so that no other unit
      sees the workspace class. */
  template <class Function, class... Arguments>
  decltype(auto) with_team(Function function, Arguments&&... arguments) const;

  std::unique_ptr<GeneralWorkspace> workspace_;
};

void fft(const ExecutionSpace& execution_space, ComplexView input, ComplexView output);
void multiply_phase(const ExecutionSpace& execution_space, ConstComplexView values,
                    ConstRealView phase, ComplexView output);

} // namespace solprop_native

template <> struct Kokkos::reduction_identity<solprop_native::MonitorSums> {
  KOKKOS_FUNCTION static solprop_native::MonitorSums sum() { return {}; }
};
template <> struct Kokkos::reduction_identity<solprop_native::OccupancySums> {
  KOKKOS_FUNCTION static solprop_native::OccupancySums sum() { return {}; }
};
template <> struct Kokkos::reduction_identity<solprop_native::SplitSums> {
  KOKKOS_FUNCTION static solprop_native::SplitSums sum() { return {}; }
};
