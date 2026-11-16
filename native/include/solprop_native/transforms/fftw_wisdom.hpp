// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"

/*
 * The host FFTW plan policy and wisdom (solprop SEP 46, ADR-076): the effort a
 * transform is seeded at, the workspace's planning and the process's wisdom as text.
 * Plan-time code only. fftw3.h stays in fftw_wisdom.cpp and in fft_plan.hpp, which
 * seeds row transforms through these.
 */
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace solprop_native {

// The two kinds of batched transform the plan policy tells apart: one-dimensional rows
// (the temporal, modal, Raman and plasma transforms) and free space's two-dimensional
// transverse planes.
enum class FftShape { rows, planes };

// The one plan policy for the host FFTW plans: the effort at which a transform of this
// shape, run on `threads` threads, is seeded. Kokkos-FFT plans at FFTW_ESTIMATE and
// consumes the wisdom a higher-effort temporary plan of the same transform left, so
// FFTW_ESTIMATE here means "do not seed". An explicit SOLPROP_FFTW_PLAN applies to
// every transform at every width. Unset, it is `measure`, except for threaded rows,
// which stay at ESTIMATE: there measured planning chose a slower threaded decomposition
// (solprop's docs/dev/benchmarks.md, "Threaded FFT plan selection"). Transverse planes
// are measured at any width (solprop SEP 28: 3.2-3.4 times faster than ESTIMATE). An
// empty value is unset, as for the thread variables; any other is refused, since a typo
// would otherwise change a run's plans in silence.
unsigned fftw_seed_effort(FftShape shape, int threads);

// An effort's FFTW flag.
unsigned fftw_effort_flag(std::string_view effort);

// FFTW's process-wide wisdom as text, for the stored plans (solprop ADR-076) and a
// workspace's planning: the Python adapter owns the file, its key and its lock, and
// calls these with the GIL held, as planning does, so they never race the planner.
// Empty when the provider implements FFTW's API without its wisdom (MKL). Both first
// register the threaded solvers, as every seed does: they are part of the planner's
// configuration, whose signature heads exported wisdom and must match for an import.
std::string export_fftw_wisdom();

// Merge `text` into the process's wisdom. FFTW refuses, leaving its wisdom as it was,
// text that is not wisdom or that another FFTW version or configuration wrote (its
// preamble carries the planner's signature). Wisdom only chooses among FFTW's exact
// algorithms, so a plan taken from it computes the same transform.
bool import_fftw_wisdom(const std::string& text);

// Forget the process's wisdom, the threaded solvers registered first as above.
void forget_fftw_wisdom();

/** While it lives, the calling thread's seeds follow `planning` (solprop SEP 46). In
    profile and capture modes FFTW's process-wide wisdom is forgotten at the start and
    at the end, so that no wisdom from an earlier workspace, a file or another
    candidate reaches this one, and none of this one's outlives it. Planning holds the
    GIL, which serialises FFTW's planner. */
class FftPlanningScope {
public:
  explicit FftPlanningScope(FftPlanning& planning);
  ~FftPlanningScope();
  FftPlanningScope(const FftPlanningScope&) = delete;
  FftPlanningScope& operator=(const FftPlanningScope&) = delete;
  FftPlanningScope(FftPlanningScope&&) = delete;
  FftPlanningScope& operator=(FftPlanningScope&&) = delete;

private:
  FftPlanning* saved_;
  bool forgets_;
};

// The flags a transform of `key` and `shape` on `threads` threads is seeded with, or
// none to leave it to Kokkos-FFT's estimate plan: the policy, or the workspace's
// planning (`FftPlanning`), which this records the transform in. Capture forgets the
// wisdom before the seed and sets the planner's time limit; profile imports the
// transform's own wisdom. A seed above estimate that the stored plans hold, or any
// that are kept, plans alone (`plan_alone`); a transform left at estimate that they
// hold is not seeded, its stored plan imported for Kokkos-FFT's estimate plan to take,
// as FFTW takes wisdom of any effort for an estimate plan.
std::optional<unsigned> begin_seed(const std::string& key, FftShape shape, int threads);

// Finish a seed begun with `flags`: a seed planned alone gives the workspace its
// wisdom back (`release_seed`); a capture keeps the transform's wisdom and lifts the
// time limit; a profile's plan that no longer builds from its wisdom is recorded, and
// left to the estimate plan. Any other failure to plan is an error.
void end_seed(const std::string& key, unsigned flags, bool planned);

// A transform's identity for its wisdom: its kind, the sizes and distances, in- or
// out-of-place, the arrays' alignment and the threads (solprop SEP 46 section 6.7).
std::string fft_plan_key(std::string_view kind, const std::string& sizes, int howmany,
                         int input_distance, int output_distance, const void* input,
                         const void* output, int threads);

/** A plan's execution time alone (solprop issue #595): the seconds of one execution
    of the transform a plan key names (`fft_plan_key`), planned from `wisdom` at
    `effort` with `FFTW_WISDOM_ONLY` (`plan`; NaN when the wisdom no longer builds
    it), and of the same transform at FFTW_ESTIMATE (`estimate`), each the median of
    five batches after a warm execution, on the key's threads and on arrays of its
    alignment, inside a native scope (`CpuTeamScope`) as a step's transforms run, so
    that idle threads wait as they do there. FFTW's wisdom is forgotten before each
    plan and after. */
struct PlanAlone {
  double plan{}, estimate{};
};

PlanAlone time_fft_plan(const std::string& key, const std::string& wisdom,
                        const std::string& effort);

// Plan, then discard, a batched complex transform of transverse planes at the plan
// policy's effort, leaving its wisdom. It overwrites both arrays; the caller clears
// them. Row transforms are seeded by `seed_row_fft_wisdom` (fft_plan.hpp).
void seed_fftw_wisdom(const ExecutionSpace& space, Complex* input, Complex* output,
                      const std::vector<int>& extents, int howmany, int input_distance,
                      int output_distance, int sign);

} // namespace solprop_native
#else
namespace solprop_native {
// Device builds plan with cuFFT or rocFFT, which keep no wisdom.
class FftPlanningScope {
public:
  explicit FftPlanningScope(FftPlanning& /*planning*/) noexcept {}
};
} // namespace solprop_native
#endif
