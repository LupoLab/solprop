// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/execution.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace solprop_native {

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

} // namespace solprop_native
