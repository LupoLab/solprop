// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"

#include <KokkosFFT.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace solprop_native {
namespace {
ThreadPolicy configured_threads;
// The runtime's width, which every unscoped OpenMP region on any thread takes
// (`CpuTeamScope`): Kokkos sets the default only on the thread that initialised it.
std::atomic<int> runtime_width{0};
} // namespace

// EXPERIMENT (branch exp/native-one-team, not for merging): run-time switches over the
// CPU thread policy, read once from the environment.
//   SOLPROP_NATIVE_EXP_THRESHOLD_SCALE=f  every operation's thresholds times f
//   SOLPROP_NATIVE_EXP_NO_MEDIUM=1        no four-thread band: serial or full width
//   SOLPROP_NATIVE_EXP_COMPACT=n          the compact team's cap (0: none)
//   SOLPROP_NATIVE_EXP_STABLE=0|1         the stable kernel team on or off
//   SOLPROP_NATIVE_EXP_FFT_JOBS=1         FFTW's unscoped loops on a team of their jobs
//   SOLPROP_NATIVE_EXP_SCOPED_BLOCKTIME=n libomp only: idle threads spin n ms while a
//                                         native call runs (kmp_set_blocktime), and the
//                                         process's setting outside
struct Experiment {
  double scale{1.0};
  bool no_medium{};
  int compact{-1};
  int stable{-1};
  bool fft_jobs{};
  int scoped_blocktime{-1};
};
const Experiment& experiment() {
  static const Experiment value = [] {
    Experiment e;
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_THRESHOLD_SCALE"))
      e.scale = std::stod(v);
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_NO_MEDIUM"))
      e.no_medium = v[0] == '1';
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_COMPACT"))
      e.compact = std::stoi(v);
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_STABLE"))
      e.stable = std::stoi(v);
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_FFT_JOBS"))
      e.fft_jobs = v[0] == '1';
    if (const char* v = std::getenv("SOLPROP_NATIVE_EXP_SCOPED_BLOCKTIME"))
      e.scoped_blocktime = std::stoi(v);
    return e;
  }();
  return value;
}

ThreadPolicy thread_policy() {
  if (configured_threads.kernels != 0)
    return configured_threads;
  const int count = Kokkos::DefaultHostExecutionSpace().concurrency();
  return {count, count, count, count, default_threaded_fft_min_samples, -1};
}

void configure_thread_policy(ThreadPolicy requested) {
  const int limit = Kokkos::DefaultHostExecutionSpace().concurrency();
  unsigned pool_bit = 1;
  requested.explicit_pools = 0;
  for (int* count : {&requested.kernels, &requested.fft, &requested.raman_fft,
                     &requested.raman_tail_fft}) {
    if (*count < 0)
      throw std::invalid_argument("CPU thread requests cannot be negative");
    if (*count != 0)
      requested.explicit_pools |= pool_bit;
    pool_bit <<= 1;
    *count = *count == 0 ? limit : std::min(*count, limit);
  }
  if (requested.band_tiles < -1)
    throw std::invalid_argument("the band column request must be -1, 0 or positive");
  if (requested.threaded_fft_min_samples == 0)
    requested.threaded_fft_min_samples = default_threaded_fft_min_samples;
  if (configured_threads.kernels != 0 && configured_threads != requested)
    throw std::runtime_error("native CPU thread policy was already initialized");
  configured_threads = requested;
  runtime_width = limit;
}

ExecutionSpace execution_instance(const int threads) {
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA) &&                   \
    !defined(KOKKOS_ENABLE_HIP)
  if (threads != ExecutionSpace().concurrency())
    return ExecutionSpace(threads);
#else
  static_cast<void>(threads);
#endif
  return ExecutionSpace();
}

int operation_threads(const ThreadPolicy& policy, const CpuPool pool,
                      const CpuOperation operation, const std::size_t elements,
                      const std::size_t fine_samples) {
  const int capacities[] = {policy.kernels, policy.fft, policy.raman_fft,
                            policy.raman_tail_fft};
  const auto index = static_cast<unsigned>(pool);
  const int capacity = capacities[index];
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA) &&                   \
    !defined(KOKKOS_ENABLE_HIP)
  if ((policy.explicit_pools & (1U << index)) != 0)
    return capacity;
  std::size_t threshold = stage_thread_min_elements;
  std::size_t full = stage_full_thread_elements;
  switch (operation) {
  case CpuOperation::stage:
    break;
  case CpuOperation::copy:
    threshold = copy_thread_min_elements;
    full = copy_thread_min_elements;
    break;
  case CpuOperation::response:
    threshold = response_thread_min_elements;
    full = response_full_thread_elements;
    break;
  case CpuOperation::modal:
    threshold = modal_thread_min_elements;
    full = modal_thread_min_elements;
    break;
  case CpuOperation::flow:
    threshold = flow_thread_min_elements;
    full = flow_thread_min_elements;
    break;
  case CpuOperation::dense:
    threshold = dense_thread_min_elements;
    full = dense_full_thread_elements;
    break;
  case CpuOperation::fft:
    threshold = policy.threaded_fft_min_samples;
    full = fft_full_thread_elements;
    break;
  case CpuOperation::raman:
    threshold = raman_thread_min_elements;
    full = raman_full_thread_elements;
    break;
  case CpuOperation::adams:
    threshold = adams_thread_min_elements;
    full = adams_full_thread_elements;
    break;
  }
  const Experiment& exp = experiment();
  if (exp.scale != 1.0) {
    threshold = std::max<std::size_t>(
        1, static_cast<std::size_t>(static_cast<double>(threshold) * exp.scale));
    full = std::max<std::size_t>(
        threshold, static_cast<std::size_t>(static_cast<double>(full) * exp.scale));
  }
  if (elements < threshold)
    return 1;
  if (exp.no_medium)
    return capacity;
  if (operation == CpuOperation::response &&
      (fine_samples >= response_full_thread_samples ||
       (fine_samples != 0 && elements / fine_samples >= response_full_thread_columns)))
    return capacity;
  return elements < full ? std::min(capacity, cpu_medium_threads) : capacity;
#else
  static_cast<void>(operation);
  static_cast<void>(elements);
  static_cast<void>(fine_samples);
  return capacity;
#endif
}

#if defined(KOKKOS_ENABLE_OPENMP) && defined(SOLPROP_NATIVE_FFTW_OPENMP) &&            \
    !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
namespace {
thread_local int fftw_loop_team = 0;

// FFTW's threaded loop (its `threads_set_callback`): the jobs on the scoped team, or,
// outside a scope, on the runtime's whole team as FFTW's own OpenMP loop runs them.
void fftw_team_loop(void* (*work)(char*), char* jobs, const std::size_t size,
                    const int count, void* /*data*/) {
  const int team = fftw_loop_team;
  if (team == 0) {
    // The runtime's width, named: on a thread other than Kokkos', the OpenMP default
    // is every CPU (solprop issue #130).
    int width = runtime_width.load(std::memory_order_relaxed);
    if (width <= 0)
      width = omp_get_max_threads();
    if (experiment().fft_jobs)
      width = std::min(width, count);
#pragma omp parallel for num_threads(width)
    for (int job = 0; job < count; ++job)
      work(jobs + size * static_cast<std::size_t>(job));
  } else {
#pragma omp parallel for num_threads(std::max(team, count))
    for (int job = 0; job < count; ++job)
      work(jobs + size * static_cast<std::size_t>(job));
  }
}

// FFTW before 3.3.9 has no loop callback; the call is dependent so that only a header
// that declares it compiles it.
template <class Loop>
concept LoopCallback =
    requires(Loop loop) { fftw_threads_set_callback(loop, nullptr); };

template <class Loop> bool install_team_loop(Loop loop) {
  if constexpr (LoopCallback<Loop>) {
    fftw_init_threads();
    fftw_threads_set_callback(loop, nullptr);
    return true;
  } else {
    static_cast<void>(loop);
    return false;
  }
}
} // namespace

bool fftw_team_loops() noexcept {
  static const bool installed = install_team_loop(&fftw_team_loop);
  return installed;
}
namespace {
int enter_fftw_team(const int threads) noexcept {
  const int saved = fftw_loop_team;
  fftw_loop_team = threads;
  return saved;
}
void leave_fftw_team(const int saved) noexcept { fftw_loop_team = saved; }
} // namespace
#else
bool fftw_team_loops() noexcept { return false; }
namespace {
int enter_fftw_team(const int threads) noexcept { return threads; }
void leave_fftw_team(int /*saved*/) noexcept {}
} // namespace
#endif

namespace {
// Whether the build's BLAS is an OpenMP OpenBLAS (`register_openmp_blas`), how many of
// the calling thread's scopes hold it at one thread, and the thread's OpenMP default
// width before the outermost of them (an OpenMP build only: Clang refuses the unused
// variable in a device build under -Werror).
std::atomic<bool> openmp_blas{false};
thread_local int blas_holds = 0;
#if defined(KOKKOS_ENABLE_OPENMP)
thread_local int blas_saved_default = 0;
// The calling thread's scopes, and its OpenMP default before the outermost of them.
thread_local int scope_depth = 0;
thread_local int scope_saved_default = 0;
thread_local int scope_saved_blocktime = -1;
#endif

// libomp's run-time blocktime control, where the loaded runtime has it (looked up by
// name, so that a GCC build running on libomp through its GOMP layer finds it too).
using SetBlocktime = void (*)(int);
using GetBlocktime = int (*)();
SetBlocktime set_blocktime() {
  static const auto set =
      reinterpret_cast<SetBlocktime>(dlsym(RTLD_DEFAULT, "kmp_set_blocktime"));
  return set;
}
GetBlocktime get_blocktime() {
  static const auto get =
      reinterpret_cast<GetBlocktime>(dlsym(RTLD_DEFAULT, "kmp_get_blocktime"));
  return get;
}

// The outermost scope on a thread sets the thread's OpenMP default width to the
// runtime's, and its end restores the caller's. Kokkos sets the default only on the
// thread that initialised it; on any other thread it is every CPU, which an OpenMP
// OpenBLAS and an unnamed region read, so that a run on a Python thread switched GNU
// OpenMP between that width and Kokkos' at every pair: 9 to 19 times slower on two
// Intel Xeon Gold 6240 (solprop issue #130).
void enter_runtime_default() noexcept {
#if defined(KOKKOS_ENABLE_OPENMP)
  if (scope_depth++ != 0)
    return;
  scope_saved_blocktime = -1;
  if (const int wanted = experiment().scoped_blocktime;
      wanted >= 0 && set_blocktime() != nullptr && get_blocktime() != nullptr) {
    scope_saved_blocktime = get_blocktime()();
    set_blocktime()(wanted);
  }
  scope_saved_default = 0;
  const int width = runtime_width.load(std::memory_order_relaxed);
  const int current = omp_get_max_threads();
  if (width > 0 && current != width) {
    scope_saved_default = current;
    omp_set_num_threads(width);
  }
#endif
}
void leave_runtime_default() noexcept {
#if defined(KOKKOS_ENABLE_OPENMP)
  if (--scope_depth != 0)
    return;
  if (scope_saved_default != 0)
    omp_set_num_threads(scope_saved_default);
  if (scope_saved_blocktime >= 0)
    set_blocktime()(scope_saved_blocktime);
#endif
}

// An OpenMP OpenBLAS runs a call serially when the caller's OpenMP default width
// (`omp_get_max_threads`) is one, before it reads its own count, so the hold sets only
// that: `openblas_set_num_threads` frees the per-thread buffers above the new count and
// allocates them again when the count returns, at every scope. Every region inside a
// scope names its width (Kokkos's instances, `fftw_team_loop`), so the default reaches
// only OpenBLAS.
void hold_blas() noexcept {
#if defined(KOKKOS_ENABLE_OPENMP)
  blas_saved_default = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
}
void release_blas() noexcept {
#if defined(KOKKOS_ENABLE_OPENMP)
  omp_set_num_threads(blas_saved_default);
#endif
}
} // namespace

void register_openmp_blas(const bool registered) noexcept { openmp_blas = registered; }

CpuTeamScope::CpuTeamScope(const int threads) noexcept
    : saved_(enter_fftw_team(threads)) {
  enter_runtime_default();
  if (threads != 0 && openmp_blas.load()) {
    holds_blas_ = true;
    if (blas_holds++ == 0)
      hold_blas();
  }
}
CpuTeamScope::~CpuTeamScope() {
  leave_fftw_team(saved_);
  if (holds_blas_ && --blas_holds == 0)
    release_blas();
  leave_runtime_default();
}

int scoped_openmp_default() noexcept {
  const CpuTeamScope scope(0);
#if defined(KOKKOS_ENABLE_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

bool compact_problem(const Problem& problem) noexcept {
  return problem.n_columns() == 1 && problem.n_nodes == 1 &&
         problem.n_pol_transform == 1 && !problem.general_modal &&
         !problem.coupled_flow && !problem.separable_geometry() &&
         problem.n_to < compact_max_fine_samples;
}

const ExecutionSpace& CpuExecutionPool::get(const CpuOperation operation,
                                            const std::size_t elements,
                                            const std::size_t fine_samples) {
  const ThreadPolicy policy = thread_policy();
  int width = operation_threads(policy, pool_, operation, elements, fine_samples);
  int capacity = policy.kernels;
  if (compact_ && policy.explicit_pools == 0) {
    const int cap = experiment().compact < 0    ? cpu_compact_threads
                    : experiment().compact == 0 ? capacity
                                                : experiment().compact;
    capacity =
        pool_ == CpuPool::kernels || fftw_team_loops() ? std::min(capacity, cap) : 1;
    // On the compact team a threaded transform takes the whole team: its loops run on
    // that team anyway. The Raman transforms keep their own start: threading the
    // 18432-sample ones of the averaged hydrogen case cost it 17 % on an AMD EPYC 7763.
    if (pool_ != CpuPool::kernels && width > 1)
      width = capacity;
    width = std::min(width, capacity);
  }
  // A stable kernel team. GNU OpenMP ends the surplus threads of its pool whenever a
  // parallel region asks for fewer than the last one, and starts them again for the
  // next wider region, so a four-thread kernel between full-width ones costs a thread
  // creation per surplus thread every time (about 0.45 ms for 12 threads). Once this
  // pool has a full-width kernel, a medium one takes the full width too. FFTW's
  // threaded loops keep the runtime's team and only distribute fewer iterations, so the
  // FFT pools keep their medium widths. Workspaces ask for their stage instance first.
  // Not on macOS (`stable_kernel_team`).
  const bool stable =
      experiment().stable < 0 ? stable_kernel_team : experiment().stable == 1;
  if (stable && pool_ == CpuPool::kernels && width > 1 && width < capacity &&
      instances_.contains(capacity))
    width = capacity;
  auto found = instances_.find(width);
  if (found == instances_.end())
    found = instances_.emplace(width, execution_instance(width)).first;
  if (elements != 0) {
    static constexpr const char* pools[] = {"kernels", "fft", "raman_fft",
                                            "raman_tail_fft"};
    static constexpr const char* operations[] = {
        "stage", "response", "dense", "fft", "raman", "copy", "flow", "modal", "adams"};
    static_assert(std::size(operations) ==
                  static_cast<unsigned>(CpuOperation::adams) + 1);
    const auto name =
        std::string(pools[static_cast<unsigned>(pool_)]) + "." +
        operations[static_cast<unsigned>(operation)] + "." + std::to_string(elements) +
        (fine_samples == 0 ? "" : ".samples." + std::to_string(fine_samples));
    sizing_[name] = {
        {"elements", elements},
        {"threads", static_cast<std::size_t>(found->second.concurrency())}};
    if (fine_samples != 0) {
      sizing_[name]["fine_samples"] = fine_samples;
      sizing_[name]["columns"] = elements / fine_samples;
    }
  }
  return found->second;
}

std::vector<double> reference_fft3_pairs(const std::complex<double>* values,
                                         const std::size_t planes,
                                         const std::size_t n_y, const std::size_t n_x,
                                         const std::size_t warm,
                                         const std::size_t pairs) {
  using Volume = Kokkos::View<Complex***, Layout, MemorySpace>;
  using Plan3 = KokkosFFT::Plan<ExecutionSpace, Volume, Volume, 3>;
  const ExecutionSpace space = execution_instance(thread_policy().fft);
  Volume volume(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "solprop_native.reference.volume"),
                planes, n_y, n_x);
  const Kokkos::View<const Complex***, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(reinterpret_cast<const Complex*>(values), planes, n_y, n_x);
  auto mirror = Kokkos::create_mirror_view(Kokkos::HostSpace{}, volume);
  Kokkos::deep_copy(mirror, host);
  Kokkos::deep_copy(volume, mirror);
  const KokkosFFT::axis_type<3> axes{-3, -2, -1};
  const Plan3 forward(space, volume, volume, KokkosFFT::Direction::forward, axes);
  const Plan3 backward(space, volume, volume, KokkosFFT::Direction::backward, axes);
  std::vector<double> seconds;
  seconds.reserve(pairs);
  for (std::size_t index = 0; index < warm + pairs; ++index) {
    space.fence();
    const auto start = std::chrono::steady_clock::now();
    KokkosFFT::execute(forward, volume, volume, KokkosFFT::Normalization::none);
    KokkosFFT::execute(backward, volume, volume, KokkosFFT::Normalization::none);
    space.fence();
    if (index >= warm)
      seconds.push_back(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
  }
  return seconds;
}

std::vector<double> reference_stream_copies(const std::size_t elements,
                                            const std::size_t warm,
                                            const std::size_t repeats) {
  const ExecutionSpace space = execution_instance(thread_policy().kernels);
  const ComplexView source(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "solprop_native.reference.source"),
                           elements);
  const ComplexView target(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              "solprop_native.reference.target"),
                           elements);
  using Policy = Kokkos::RangePolicy<ExecutionSpace>;
  // First touch in parallel, on the threads that copy, so that each page lands where
  // the copy reads or writes it.
  launch::parallel_for(
      "solprop_native.reference.first_touch", Policy(space, 0, elements),
      KOKKOS_LAMBDA(const std::size_t index) {
        source(index) = Complex(static_cast<double>(index % 7U), 1.0);
        target(index) = Complex{};
      });
  std::vector<double> seconds;
  seconds.reserve(repeats);
  const double scale = 0.5;
  for (std::size_t index = 0; index < warm + repeats; ++index) {
    space.fence();
    const auto start = std::chrono::steady_clock::now();
    launch::parallel_for(
        "solprop_native.reference.scaled_copy", Policy(space, 0, elements),
        KOKKOS_LAMBDA(const std::size_t element) {
          target(element) = scale * source(element);
        });
    space.fence();
    if (index >= warm)
      seconds.push_back(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
  }
  return seconds;
}

void fft(const ExecutionSpace& execution_space, const ComplexView input,
         const ComplexView output) {
  const std::size_t size = input.extent(0);
  if (size == 0)
    throw std::invalid_argument("fft input must not be empty");
  if (output.extent(0) != size) {
    throw std::invalid_argument("fft input and output must have the same length");
  }
  KokkosFFT::fft(execution_space, input, output, KokkosFFT::Normalization::none);
}

void multiply_phase(const ExecutionSpace& execution_space,
                    const ConstComplexView values, const ConstRealView phase,
                    const ComplexView output) {
  const std::size_t size = values.extent(0);
  if (phase.extent(0) != size || output.extent(0) != size) {
    throw std::invalid_argument("values, phase, and output must have the same length");
  }
  launch::parallel_for(
      "solprop_native.multiply_phase",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, size),
      KOKKOS_LAMBDA(const std::size_t index) {
        output(index) = values(index) *
                        Complex(Kokkos::cos(phase(index)), Kokkos::sin(phase(index)));
      });
}

} // namespace solprop_native
