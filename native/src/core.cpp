// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/core.hpp"
#include "solprop_native/fp.hpp"

#include <KokkosFFT.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace solprop_native {
namespace {
ThreadPolicy configured_threads;
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
  if (elements < threshold)
    return 1;
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
#pragma omp parallel for
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
// width before the outermost of them.
std::atomic<bool> openmp_blas{false};
thread_local int blas_holds = 0;
thread_local int blas_saved_default = 0;

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
    capacity = pool_ == CpuPool::kernels || fftw_team_loops()
                   ? std::min(capacity, cpu_compact_threads)
                   : 1;
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
  if (stable_kernel_team && pool_ == CpuPool::kernels && width > 1 &&
      width < capacity && instances_.contains(capacity))
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

namespace {

using RealPlan = KokkosFFT::Plan<ExecutionSpace, RealView, ComplexView>;
using InverseRealPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView, RealView>;
using ComplexPlan = KokkosFFT::Plan<ExecutionSpace, ComplexView, ComplexView>;
using ByteView = Kokkos::View<unsigned char*, Layout, MemorySpace>;
using IndexView = Kokkos::View<std::size_t*, Layout, MemorySpace>;
using ComplexView2 = Kokkos::View<Complex**, Layout, MemorySpace>;

KOKKOS_INLINE_FUNCTION Complex exponential(const Complex value) {
  const double magnitude = Kokkos::exp(value.real());
  return Complex(magnitude * Kokkos::cos(value.imag()),
                 magnitude * Kokkos::sin(value.imag()));
}

// A host span as an unmanaged view, for one copy between it and a memory-space view
// of the same extent: no staging mirror, no element loop. A span whose data is not
// aligned as `Complex` (a NumPy buffer need not be) is staged in an aligned vector.
void upload_span(const ExecutionSpace& execution_space, const ComplexView& destination,
                 const std::span<const Complex> source) {
  if (destination.extent(0) != source.size())
    throw std::invalid_argument("fixed problem array has the wrong extent");
  if (reinterpret_cast<std::uintptr_t>(source.data()) % alignof(Complex) != 0) {
    const std::vector<Complex> aligned(source.begin(), source.end());
    upload_span(execution_space, destination, aligned);
    execution_space.fence("solprop_native unaligned upload");
    return;
  }
  const Kokkos::View<const Complex*, Layout, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(source.data(), source.size());
  Kokkos::deep_copy(execution_space, destination, host);
}

template <class View, class Value>
void copy_vector(const ExecutionSpace& execution_space, const View& destination,
                 const std::vector<Value>& source) {
  if (destination.extent(0) != source.size()) {
    throw std::invalid_argument("fixed problem array has the wrong extent");
  }
  auto mirror = Kokkos::create_mirror_view(destination);
  for (std::size_t index = 0; index < source.size(); ++index) {
    mirror(index) = source[index];
  }
  Kokkos::deep_copy(execution_space, destination, mirror);
}

void require_extent(const std::size_t actual, const std::size_t expected,
                    const char* name) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(name) + " has the wrong extent");
  }
}

void require_problem(const Problem& problem) {
  if (problem.n_omega == 0 || problem.n_t == 0 || problem.n_to == 0) {
    throw std::invalid_argument("FFT extents must be positive");
  }
  if (problem.n_stages == 0) {
    throw std::invalid_argument("the tableau must contain a stage");
  }
  require_extent(problem.crop_index.size(), problem.n_omega, "crop_index");
  require_extent(problem.active_mask.size(), problem.n_omega, "active_mask");
  require_extent(problem.norm_mask.size(), problem.n_omega, "norm_mask");
  require_extent(problem.omega_window.size(), problem.n_omega, "omega_window");
  require_extent(problem.time_window.size(), problem.n_to, "time_window");
  require_extent(problem.phi_terms.size(), problem.n_phi_terms * problem.n_omega,
                 "phi_terms");
  require_extent(problem.prefactor_terms.size(),
                 problem.n_prefactor_terms * problem.n_omega, "prefactor_terms");
  require_extent(problem.tableau_matrix.size(), problem.n_stages * problem.n_stages,
                 "tableau_matrix");
  require_extent(problem.tableau_nodes.size(), problem.n_stages, "tableau_nodes");
  require_extent(problem.solution_weights.size(), problem.n_stages, "solution_weights");
  require_extent(problem.error_weights.size(), problem.n_stages, "error_weights");
  require_extent(problem.carrier.size(), problem.n_to, "carrier");
  if (problem.in_stage_absorber) {
    require_extent(problem.absorber_fine.size(), problem.n_to, "absorber_fine");
  }
  if (problem.split_step_absorber) {
    require_extent(problem.absorber_coarse.size(), problem.n_t, "absorber_coarse");
  }
  const std::size_t fine_spectrum =
      problem.family == FieldFamily::field_real ? problem.n_to / 2 + 1 : problem.n_to;
  std::vector<bool> seen(fine_spectrum, false);
  for (const auto index : problem.crop_index) {
    if (index >= fine_spectrum) {
      throw std::invalid_argument("crop_index lies outside the fine spectrum");
    }
    if (seen[index]) {
      throw std::invalid_argument("crop_index contains a duplicate fine-spectrum bin");
    }
    seen[index] = true;
  }
}

} // namespace

class Workspace {
public:
  explicit Workspace(Problem problem) : problem_(std::move(problem)) {
    require_problem(problem_);
    compact_ = compact_problem(problem_);
    if (compact_) {
      kernel_pool_.compact();
      fft_pool_.compact();
    }
    execution_space_ = kernel_pool_.get(CpuOperation::stage, problem_.n_omega);
    response_execution_space_ =
        kernel_pool_.get(CpuOperation::response, problem_.n_to, problem_.n_to);
    fft_execution_space_ = fft_pool_.get(CpuOperation::fft, problem_.n_to);
    coarse_fft_execution_space_ = fft_pool_.get(CpuOperation::fft, problem_.n_t);
    allocate();
    upload();
    const auto started = std::chrono::steady_clock::now();
    plan();
    planning_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    execution_space_.fence("solprop_native construction");
  }

  void set_initial(const std::span<const Complex> field,
                   const std::span<const Complex> noise) {
    require_open();
    require_extent(field.size(), problem_.n_omega, "initial field");
    if (!noise.empty())
      require_extent(noise.size(), problem_.n_omega, "noise");
    upload_span(execution_space_, current_, field);
    if (noise.empty()) {
      if (problem_.modified_noise)
        Kokkos::deep_copy(execution_space_, noise_, Complex{});
      has_noise_ = false;
    } else {
      if (!problem_.modified_noise)
        throw std::invalid_argument(
            "a native noise field is valid only for modified noise");
      upload_span(execution_space_, noise_, noise);
      has_noise_ = true;
    }
    mask(current_);
    execution_space_.fence("solprop_native set_initial");
    initialized_ = true;
    rk_source_evaluations_ = 0;
    clear_fsal();
  }

  std::array<double, 3> trial_step(const double dz, const TrialScalars& scalars) {
    require_open();
    if (!initialized_)
      throw std::logic_error("no initial field has been loaded");
    require_trial_scalars(scalars);
    upload_trial(scalars);
    std::size_t first = 0;
    if (has_recycled_) {
      reuse_fsal();
      first = 1;
    }
    for (std::size_t stage = first; stage < problem_.n_stages; ++stage) {
      evaluate_stage(stage, dz);
    }
    accept_and_score(dz);
    pending_fsal_ = problem_.first_same_as_last && !problem_.split_step_absorber;
    execution_space_.fence("solprop_native trial boundary");
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, partials_);
    return {host(0), host(1), host(2)};
  }

  void finish(const double dz) {
    require_open();
    if (!initialized_)
      throw std::logic_error("no initial field has been loaded");
    if (problem_.split_step_absorber) {
      split_step(dz);
    } else {
      mask(advanced_);
    }
    std::swap(current_, advanced_);
    execution_space_.fence("solprop_native accepted-step boundary");
  }

  void promote_fsal() {
    require_open();
    if (!pending_fsal_)
      return;
    const auto stages = stages_;
    const auto whole = whole_propagator_;
    const auto recycled = recycled_;
    const std::size_t last = problem_.n_stages - 1;
    launch::parallel_for(
        "solprop_native.stage.promote_fsal",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
        KOKKOS_LAMBDA(const std::size_t index) {
          recycled(index) = whole(index) * stages(last, index);
        });
    has_recycled_ = true;
    pending_fsal_ = false;
  }

  void clear_fsal() {
    has_recycled_ = false;
    pending_fsal_ = false;
  }

  // The boundary monitor found the temporal collar quiet (solprop SEP 21): its
  // in-stage term, below the monitor's threshold, is left out of the stages.
  void set_absorber_quiet(const bool quiet) noexcept { absorber_quiet_ = quiet; }

  void bind_monitor(const std::vector<double>& weights,
                    const std::vector<double>& column_weights);
  void unbind_monitor() noexcept { monitor_bound_ = false; }
  std::vector<double> monitor_reading();

  std::vector<Complex> field_to_host() const {
    std::vector<Complex> result(problem_.n_omega);
    field_to_host(result);
    return result;
  }

  // One copy from the memory space into the caller's buffer (solprop SEP 39 section
  // 6.7: no staging mirror and no element loop).
  void field_to_host(const std::span<Complex> destination) const {
    require_open();
    if (!initialized_)
      throw std::logic_error("no initial field has been loaded");
    require_extent(destination.size(), problem_.n_omega, "field destination");
    execution_space_.fence("solprop_native field_to_host");
    const Kokkos::View<Complex*, Layout, Kokkos::HostSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host(destination.data(), destination.size());
    // On the stage's instance, so that a compact problem keeps its team.
    Kokkos::deep_copy(execution_space_, host, current_);
    execution_space_.fence("solprop_native field_to_host copy");
  }

  std::map<std::string, std::size_t> memory_ledger() const;
  CpuSizing execution_sizing() const {
    auto result = kernel_pool_.sizing();
    result.merge(fft_pool_.sizing());
    return result;
  }
  /** The team a compact problem's FFTW loops take (`CpuTeamScope`); zero otherwise. */
  [[nodiscard]] int cpu_team() const noexcept {
    return compact_ ? std::max(kernel_pool_.widest(), fft_pool_.widest()) : 0;
  }
  double planning_seconds() const noexcept { return planning_seconds_; }
  bool fsal_ready() const noexcept { return has_recycled_; }
  std::size_t rk_source_evaluations() const noexcept { return rk_source_evaluations_; }
  bool closed() const noexcept { return closed_; }
  void close() noexcept;

  // NVCC extended device lambdas require their enclosing member to be public.
  // Workspace is translation-unit-local, so this does not widen the native API.
  std::size_t fine_spectrum_size() const {
    return problem_.family == FieldFamily::field_real ? problem_.n_to / 2 + 1
                                                      : problem_.n_to;
  }
  std::size_t projected_spectrum_size() const {
    return problem_.family == FieldFamily::envelope ? problem_.n_to
                                                    : problem_.n_to / 2 + 1;
  }
  void require_open() const {
    if (closed_)
      throw std::logic_error("native state is closed");
  }
  void require_trial_scalars(const TrialScalars& scalars) const;
  void allocate();
  void upload();
  void upload_trial(const TrialScalars& scalars);
  void plan();
  void reuse_fsal();
  void evaluate_stage(std::size_t stage, double dz);
  void inverse_transform();
  void response(std::size_t stage);
  void forward_response();
  void form_absorber();
  void forward_absorber();
  [[nodiscard]] bool absorbing() const noexcept {
    return problem_.in_stage_absorber && !absorber_quiet_;
  }
  bool absorber_quiet_ = false;
  void finish_stage(std::size_t stage);
  void accept_and_score(double dz);
  void mask(const ComplexView& values);
  void split_step(double dz);
  void plan_coarse_inverse();
  void add_split_removal(double dz);
  [[nodiscard]] bool monitoring_split() const noexcept {
    return monitor_bound_ && problem_.split_step_absorber;
  }
  // solprop's boundary monitor (SEP 21, ADR-057): its four time weights `(4, n_t)`
  // and the column's quadrature weight, and what the split-step collar took since the
  // last reading beside one split's sum. Allocated at the first bind, with the coarse
  // pair a split collar would have planned, and kept.
  bool monitor_bound_{};
  double monitor_column_weight_ = 1.0;

  Problem problem_;
  bool compact_{};
  CpuExecutionPool kernel_pool_{CpuPool::kernels};
  CpuExecutionPool fft_pool_{CpuPool::fft};
  mutable ExecutionSpace execution_space_;
  ExecutionSpace response_execution_space_, fft_execution_space_,
      coarse_fft_execution_space_;
  bool closed_{};
  bool initialized_{};
  bool has_noise_{};
  bool has_recycled_{};
  std::size_t rk_source_evaluations_{};
  bool pending_fsal_{};
  double planning_seconds_{};
  ComplexView current_, advanced_, solution_, error_, recycled_, noise_;
  ComplexView whole_propagator_, fine_spectrum_, projected_spectrum_;
  ComplexView absorber_spectrum_, carrier_, fine_time_complex_;
  ComplexView projected_complex_, absorber_complex_, coarse_spectrum_;
  ComplexView coarse_time_complex_;
  ComplexView2 stages_, phi_terms_, prefactor_terms_;
  RealView partials_, omega_window_, time_window_, absorber_fine_;
  RealView absorber_coarse_, fine_time_real_, projected_real_, absorber_real_;
  RealView coarse_time_real_, tableau_, stage_phi_delta_, stage_phi_at_;
  RealView monitor_weights_;
  Kokkos::View<double, Layout, MemorySpace> monitor_removed_, monitor_split_sum_;
  RealView stage_prefactor_, stage_density_, stage_taper_;
  ByteView active_mask_, norm_mask_;
  IndexView crop_index_;
  std::unique_ptr<InverseRealPlan> inverse_real_plan_;
  std::unique_ptr<ComplexPlan> inverse_complex_plan_;
  std::unique_ptr<RealPlan> forward_real_plan_;
  std::unique_ptr<ComplexPlan> forward_complex_plan_;
  std::unique_ptr<RealPlan> absorber_real_plan_;
  std::unique_ptr<ComplexPlan> absorber_complex_plan_;
  std::unique_ptr<InverseRealPlan> coarse_inverse_real_plan_;
  std::unique_ptr<ComplexPlan> coarse_inverse_complex_plan_;
  std::unique_ptr<RealPlan> coarse_forward_real_plan_;
  std::unique_ptr<ComplexPlan> coarse_forward_complex_plan_;
};

void Workspace::require_trial_scalars(const TrialScalars& scalars) const {
  require_extent(scalars.phi_delta.size(),
                 (problem_.n_stages + 1) * problem_.n_phi_terms, "phi_delta");
  require_extent(scalars.phi_at.size(), problem_.n_stages * problem_.n_phi_terms,
                 "phi_at");
  require_extent(scalars.prefactor.size(),
                 problem_.n_stages * problem_.n_prefactor_terms, "prefactor");
  require_extent(scalars.density.size(), problem_.n_stages, "density");
  require_extent(scalars.taper.size(), problem_.n_stages, "taper");
}

void Workspace::allocate() {
  const auto n = problem_.n_omega;
  current_ = ComplexView("solprop_native.state.current", n);
  advanced_ = ComplexView("solprop_native.state.advanced", n);
  stages_ = ComplexView2("solprop_native.step.stages", problem_.n_stages, n);
  solution_ = ComplexView("solprop_native.step.solution", n);
  error_ = ComplexView("solprop_native.step.error", n);
  recycled_ = ComplexView("solprop_native.step.recycled", n);
  if (problem_.modified_noise)
    noise_ = ComplexView("solprop_native.noise", n);
  whole_propagator_ = ComplexView("solprop_native.step.whole_propagator", n);
  partials_ = RealView("solprop_native.step.error_partials", 3);
  crop_index_ = IndexView("solprop_native.transform.crop_index", n);
  active_mask_ = ByteView("solprop_native.mask.active", n);
  norm_mask_ = ByteView("solprop_native.mask.norm", n);
  omega_window_ = RealView("solprop_native.window.omega", n);
  time_window_ = RealView("solprop_native.window.time", problem_.n_to);
  carrier_ = ComplexView("solprop_native.kerr.carrier", problem_.n_to);
  phi_terms_ = ComplexView2("solprop_native.linear.phi_terms", problem_.n_phi_terms, n);
  prefactor_terms_ = ComplexView2("solprop_native.response.prefactor_terms",
                                  problem_.n_prefactor_terms, n);
  tableau_ =
      RealView("solprop_native.step.tableau", problem_.n_stages * problem_.n_stages);
  stage_phi_delta_ = RealView("solprop_native.step.phi_delta",
                              (problem_.n_stages + 1) * problem_.n_phi_terms);
  stage_phi_at_ =
      RealView("solprop_native.step.phi_at", problem_.n_stages * problem_.n_phi_terms);
  stage_prefactor_ = RealView("solprop_native.step.prefactor",
                              problem_.n_stages * problem_.n_prefactor_terms);
  stage_density_ = RealView("solprop_native.step.density", problem_.n_stages);
  stage_taper_ = RealView("solprop_native.step.taper", problem_.n_stages);
  fine_spectrum_ =
      ComplexView("solprop_native.fft.fine_spectrum", fine_spectrum_size());
  projected_spectrum_ =
      ComplexView("solprop_native.fft.projected_spectrum", projected_spectrum_size());
  absorber_spectrum_ =
      ComplexView("solprop_native.fft.absorber_spectrum", projected_spectrum_size());
  if (problem_.family == FieldFamily::field_real) {
    fine_time_real_ = RealView("solprop_native.fft.modal_time_real", problem_.n_to);
  } else {
    fine_time_complex_ =
        ComplexView("solprop_native.fft.modal_time_complex", problem_.n_to);
  }
  if (problem_.family == FieldFamily::envelope) {
    projected_complex_ =
        ComplexView("solprop_native.response.projected_complex", problem_.n_to);
    absorber_complex_ =
        ComplexView("solprop_native.absorber.fine_complex", problem_.n_to);
  } else {
    projected_real_ = RealView("solprop_native.response.projected_real", problem_.n_to);
    absorber_real_ = RealView("solprop_native.absorber.fine_real", problem_.n_to);
  }
  absorber_fine_ =
      RealView("solprop_native.absorber.rate_fine", problem_.absorber_fine.size());
  absorber_coarse_ =
      RealView("solprop_native.absorber.rate_coarse", problem_.absorber_coarse.size());
  if (!problem_.split_step_absorber)
    return;
  coarse_spectrum_ = ComplexView("solprop_native.absorber.coarse_spectrum", n);
  if (problem_.family == FieldFamily::envelope) {
    coarse_time_complex_ =
        ComplexView("solprop_native.absorber.coarse_time_complex", problem_.n_t);
  } else {
    coarse_time_real_ =
        RealView("solprop_native.absorber.coarse_time_real", problem_.n_t);
  }
}

void Workspace::upload() {
  copy_vector(execution_space_, crop_index_, problem_.crop_index);
  copy_vector(execution_space_, active_mask_, problem_.active_mask);
  copy_vector(execution_space_, norm_mask_, problem_.norm_mask);
  copy_vector(execution_space_, omega_window_, problem_.omega_window);
  copy_vector(execution_space_, time_window_, problem_.time_window);
  copy_vector(execution_space_, carrier_, problem_.carrier);
  copy_vector(execution_space_, tableau_, problem_.tableau_matrix);
  copy_vector(execution_space_, absorber_fine_, problem_.absorber_fine);
  copy_vector(execution_space_, absorber_coarse_, problem_.absorber_coarse);
  auto phi = Kokkos::create_mirror_view(phi_terms_);
  for (std::size_t term = 0; term < problem_.n_phi_terms; ++term) {
    for (std::size_t index = 0; index < problem_.n_omega; ++index) {
      phi(term, index) = problem_.phi_terms[term * problem_.n_omega + index];
    }
  }
  Kokkos::deep_copy(execution_space_, phi_terms_, phi);
  auto prefactor = Kokkos::create_mirror_view(prefactor_terms_);
  for (std::size_t term = 0; term < problem_.n_prefactor_terms; ++term) {
    for (std::size_t index = 0; index < problem_.n_omega; ++index) {
      prefactor(term, index) =
          problem_.prefactor_terms[term * problem_.n_omega + index];
    }
  }
  Kokkos::deep_copy(execution_space_, prefactor_terms_, prefactor);
}

void Workspace::upload_trial(const TrialScalars& scalars) {
  copy_vector(execution_space_, stage_phi_delta_, scalars.phi_delta);
  copy_vector(execution_space_, stage_phi_at_, scalars.phi_at);
  copy_vector(execution_space_, stage_prefactor_, scalars.prefactor);
  copy_vector(execution_space_, stage_density_, scalars.density);
  copy_vector(execution_space_, stage_taper_, scalars.taper);
}

void Workspace::plan() {
  constexpr int axis = -1;
  if (problem_.family == FieldFamily::field_real) {
    inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        fft_execution_space_, fine_spectrum_, fine_time_real_,
        KokkosFFT::Direction::backward, axis);
  } else {
    inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_execution_space_, fine_spectrum_, fine_time_complex_,
        KokkosFFT::Direction::backward, axis);
  }
  if (problem_.family == FieldFamily::envelope) {
    forward_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_execution_space_, projected_complex_, projected_spectrum_,
        KokkosFFT::Direction::forward, axis);
    absorber_complex_plan_ = std::make_unique<ComplexPlan>(
        fft_execution_space_, absorber_complex_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
  } else {
    forward_real_plan_ = std::make_unique<RealPlan>(
        fft_execution_space_, projected_real_, projected_spectrum_,
        KokkosFFT::Direction::forward, axis);
    absorber_real_plan_ = std::make_unique<RealPlan>(
        fft_execution_space_, absorber_real_, absorber_spectrum_,
        KokkosFFT::Direction::forward, axis);
  }
  if (!problem_.split_step_absorber)
    return;
  plan_coarse_inverse();
  if (problem_.family == FieldFamily::envelope) {
    coarse_forward_complex_plan_ = std::make_unique<ComplexPlan>(
        coarse_fft_execution_space_, coarse_time_complex_, coarse_spectrum_,
        KokkosFFT::Direction::forward, axis);
  } else {
    coarse_forward_real_plan_ = std::make_unique<RealPlan>(
        coarse_fft_execution_space_, coarse_time_real_, coarse_spectrum_,
        KokkosFFT::Direction::forward, axis);
  }
}

// The coarse inverse transform on its buffers, which the split-step collar and the
// boundary monitor read; allocated here when the constructor did not.
void Workspace::plan_coarse_inverse() {
  constexpr int axis = -1;
  if (coarse_spectrum_.extent(0) == 0)
    coarse_spectrum_ =
        ComplexView("solprop_native.absorber.coarse_spectrum", problem_.n_omega);
  if (problem_.family == FieldFamily::envelope) {
    if (coarse_time_complex_.extent(0) == 0)
      coarse_time_complex_ =
          ComplexView("solprop_native.absorber.coarse_time_complex", problem_.n_t);
    coarse_inverse_complex_plan_ = std::make_unique<ComplexPlan>(
        coarse_fft_execution_space_, coarse_spectrum_, coarse_time_complex_,
        KokkosFFT::Direction::backward, axis);
  } else {
    if (coarse_time_real_.extent(0) == 0)
      coarse_time_real_ =
          RealView("solprop_native.absorber.coarse_time_real", problem_.n_t);
    coarse_inverse_real_plan_ = std::make_unique<InverseRealPlan>(
        coarse_fft_execution_space_, coarse_spectrum_, coarse_time_real_,
        KokkosFFT::Direction::backward, axis);
  }
}

// What one split takes of the power it finds, `sum P (1 - exp(-alpha h))`, added to
// the run's accumulator in the memory space (solprop SEP 21, ADR-057): the monitor's
// reading drains it at a check, so every accepted step is counted, checked or not.
void Workspace::add_split_removal(const double dz) {
  const auto rate = absorber_coarse_;
  const double weight = monitor_column_weight_;
  const Kokkos::RangePolicy<ExecutionSpace> samples(execution_space_, 0, problem_.n_t);
  if (problem_.family == FieldFamily::envelope) {
    const auto time = coarse_time_complex_;
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_complex", samples,
        KOKKOS_LAMBDA(const std::size_t index, double& update) {
          const Complex value = time(index);
          update += weight *
                    (value.real() * value.real() + value.imag() * value.imag()) *
                    -Kokkos::expm1(-rate(index) * dz);
        },
        monitor_split_sum_);
  } else {
    const auto time = coarse_time_real_;
    launch::parallel_reduce(
        "solprop_native.monitor.split_removal_real", samples,
        KOKKOS_LAMBDA(const std::size_t index, double& update) {
          update +=
              weight * time(index) * time(index) * -Kokkos::expm1(-rate(index) * dz);
        },
        monitor_split_sum_);
  }
  const auto removed = monitor_removed_;
  const auto split = monitor_split_sum_;
  launch::parallel_for(
      "solprop_native.monitor.split_removal_add",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) { removed() += split(); });
}

// Bind solprop's boundary monitor (SEP 21): its four time weights, row-major `(4,
// n_t)`, and the column's quadrature (empty for a unit weight). The first bind
// allocates its few views, and the coarse pair when no split collar planned one;
// the split's accumulator starts at zero.
void Workspace::bind_monitor(const std::vector<double>& weights,
                             const std::vector<double>& column_weights) {
  require_open();
  require_extent(weights.size(), 4 * problem_.n_t, "monitor weights");
  if (!column_weights.empty())
    require_extent(column_weights.size(), 1, "monitor column weights");
  if (monitor_weights_.extent(0) == 0) {
    monitor_weights_ = RealView("solprop_native.monitor.weights", 4 * problem_.n_t);
    monitor_removed_ =
        Kokkos::View<double, Layout, MemorySpace>("solprop_native.monitor.removed");
    monitor_split_sum_ =
        Kokkos::View<double, Layout, MemorySpace>("solprop_native.monitor.split_sum");
    if (!coarse_inverse_complex_plan_ && !coarse_inverse_real_plan_)
      plan_coarse_inverse();
  }
  copy_vector(execution_space_, monitor_weights_, weights);
  monitor_column_weight_ = column_weights.empty() ? 1.0 : column_weights[0];
  Kokkos::deep_copy(execution_space_, monitor_removed_, 0.0);
  execution_space_.fence("solprop_native monitor bind");
  monitor_bound_ = true;
}

// The bound monitor's reading of the accepted state: `sum_t P`, the four weighted
// sums and the split's accumulator, which this drains (solprop SEP 21, ADR-057).
std::vector<double> Workspace::monitor_reading() {
  require_open();
  if (!initialized_)
    throw std::logic_error("no initial field has been loaded");
  if (!monitor_bound_)
    throw std::logic_error("no boundary monitor is bound");
  const auto current = current_;
  const auto coarse = coarse_spectrum_;
  const auto n_t = problem_.n_t;
  const double inverse_n = 1.0 / static_cast<double>(n_t);
  launch::parallel_for(
      "solprop_native.monitor.spectrum",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        coarse(index) = inverse_n * current(index);
      });
  const auto weights = monitor_weights_;
  const double weight = monitor_column_weight_;
  MonitorSums sums;
  const Kokkos::RangePolicy<ExecutionSpace> samples(execution_space_, 0, n_t);
  if (problem_.family == FieldFamily::envelope) {
    KokkosFFT::execute(*coarse_inverse_complex_plan_, coarse_spectrum_,
                       coarse_time_complex_, KokkosFFT::Normalization::none);
    const auto time = coarse_time_complex_;
    launch::parallel_reduce(
        "solprop_native.monitor.sums", samples,
        KOKKOS_LAMBDA(const std::size_t index, MonitorSums& update) {
          const Complex value = time(index);
          const double power =
              weight * (value.real() * value.real() + value.imag() * value.imag());
          update.values[0] += power;
          for (std::size_t region = 0; region < 4; ++region)
            update.values[region + 1] += weights(region * n_t + index) * power;
        },
        Kokkos::Sum<MonitorSums>(sums));
  } else {
    KokkosFFT::execute(*coarse_inverse_real_plan_, coarse_spectrum_, coarse_time_real_,
                       KokkosFFT::Normalization::none);
    const auto time = coarse_time_real_;
    launch::parallel_reduce(
        "solprop_native.monitor.sums", samples,
        KOKKOS_LAMBDA(const std::size_t index, MonitorSums& update) {
          const double power = weight * time(index) * time(index);
          update.values[0] += power;
          for (std::size_t region = 0; region < 4; ++region)
            update.values[region + 1] += weights(region * n_t + index) * power;
        },
        Kokkos::Sum<MonitorSums>(sums));
  }
  double removed = 0.0;
  if (problem_.split_step_absorber) {
    execution_space_.fence("solprop_native monitor reading");
    removed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, monitor_removed_)();
    Kokkos::deep_copy(execution_space_, monitor_removed_, 0.0);
  }
  execution_space_.fence("solprop_native monitor reading");
  return {sums.values[0], sums.values[1], sums.values[2],
          sums.values[3], sums.values[4], removed};
}

void Workspace::reuse_fsal() {
  const auto stages = stages_;
  const auto recycled = recycled_;
  const auto solution = solution_;
  const auto error = error_;
  const double solution_weight = problem_.solution_weights[0];
  const double error_weight = problem_.error_weights[0];
  launch::parallel_for(
      "solprop_native.stage.reuse_fsal",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        const Complex value = recycled(index);
        stages(0, index) = value;
        solution(index) = solution_weight * value;
        error(index) = error_weight * value;
      });
}

void Workspace::evaluate_stage(const std::size_t stage, const double dz) {
  ++rk_source_evaluations_;
  if (problem_.n_omega != fine_spectrum_size()) {
    Kokkos::deep_copy(execution_space_, fine_spectrum_, Complex{});
  }
  const auto current = current_;
  const auto stages = stages_;
  const auto tableau = tableau_;
  const auto phi_terms = phi_terms_;
  const auto phi_delta = stage_phi_delta_;
  const auto phi_at = stage_phi_at_;
  const auto noise = noise_;
  const auto crop = crop_index_;
  const auto fine = fine_spectrum_;
  const auto whole = whole_propagator_;
  const std::size_t n = problem_.n_omega;
  const std::size_t n_phi = problem_.n_phi_terms;
  const std::size_t n_stages = problem_.n_stages;
  const bool add_noise = problem_.modified_noise && has_noise_;
  const bool analytic = problem_.family == FieldFamily::field_analytic;
  const double inverse_n_to = 1.0 / static_cast<double>(problem_.n_to);
  launch::parallel_for(
      "solprop_native.stage.combine_pad",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
      KOKKOS_LAMBDA(const std::size_t index) {
        // The phase increments: strict (solprop SEP 25 section 4.2).
        const auto phases = [&](Complex& phase,
                                Complex& phase_at) SOLPROP_NATIVE_STRICT_FUNCTION {
          SOLPROP_NATIVE_STRICT_BLOCK
          for (std::size_t term = 0; term < n_phi; ++term) {
            const Complex fixed = phi_terms(term, index);
            phase = strict_accumulate(phase, phi_delta(stage * n_phi + term), fixed);
            if (add_noise)
              phase_at =
                  strict_accumulate(phase_at, phi_at(stage * n_phi + term), fixed);
          }
        };
        Complex phase{};
        Complex phase_at{};
        phases(phase, phase_at);
        const Complex propagator = exponential(phase);
        Complex accumulated{};
        for (std::size_t previous = 0; previous < stage; ++previous) {
          accumulated += tableau(stage * n_stages + previous) * stages(previous, index);
        }
        Complex value = propagator * (current(index) + dz * accumulated);
        if (add_noise) {
          value += noise(index) *
                   Complex(Kokkos::cos(phase_at.imag()), Kokkos::sin(phase_at.imag()));
        }
        const double analytic_scale = analytic && index > 0 ? 2.0 : 1.0;
        fine(crop(index)) = analytic_scale * inverse_n_to * value;
        if (stage + 1 == n_stages) {
          const auto whole_phase = [&]() SOLPROP_NATIVE_STRICT_FUNCTION {
            SOLPROP_NATIVE_STRICT_BLOCK
            Complex sum{};
            for (std::size_t term = 0; term < n_phi; ++term)
              sum = strict_accumulate(sum, phi_delta(n_stages * n_phi + term),
                                      phi_terms(term, index));
            return sum;
          };
          whole(index) = exponential(whole_phase());
        }
      });
  if (!problem_.nonlinear && !absorbing()) {
    const auto solution = solution_;
    const auto error = error_;
    const double solution_weight = problem_.solution_weights[stage];
    const double error_weight = problem_.error_weights[stage];
    launch::parallel_for(
        "solprop_native.stage.linear_zero",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, n),
        KOKKOS_LAMBDA(const std::size_t index) {
          stages(stage, index) = Complex{};
          if (stage == 0) {
            solution(index) = Complex{};
            error(index) = Complex{};
          } else {
            solution(index) += solution_weight * Complex{};
            error(index) += error_weight * Complex{};
          }
        });
    return;
  }
  inverse_transform();
  response(stage);
  forward_response();
  if (absorbing()) {
    form_absorber();
    forward_absorber();
  }
  finish_stage(stage);
}

void Workspace::inverse_transform() {
  if (problem_.family == FieldFamily::field_real) {
    KokkosFFT::execute(*inverse_real_plan_, fine_spectrum_, fine_time_real_,
                       KokkosFFT::Normalization::none);
  } else {
    KokkosFFT::execute(*inverse_complex_plan_, fine_spectrum_, fine_time_complex_,
                       KokkosFFT::Normalization::none);
  }
}

void Workspace::response(const std::size_t stage) {
  const auto real_time = fine_time_real_;
  const auto complex_time = fine_time_complex_;
  const auto projected_real = projected_real_;
  const auto projected_complex = projected_complex_;
  const auto time_window = time_window_;
  const auto carrier = carrier_;
  const auto density = stage_density_;
  const auto taper = stage_taper_;
  const auto family = problem_.family;
  const bool nonlinear = problem_.nonlinear;
  const bool cubic_real = problem_.cubic_real;
  const double synthesis = problem_.synthesis;
  const double projection = problem_.projection;
  const double coefficient = problem_.kerr_coefficient;
  const double self_factor = problem_.self_factor;
  const double conjugate_factor = problem_.conjugate_factor;
  const double harmonic_factor = problem_.harmonic_factor;
  launch::parallel_for(
      "solprop_native.response.kerr_1x1",
      Kokkos::RangePolicy<ExecutionSpace>(response_execution_space_, 0, problem_.n_to),
      KOKKOS_LAMBDA(const std::size_t index) {
        const double taper_value = taper(stage);
        const double project_scale = projection / taper_value;
        if (family == FieldFamily::field_real) {
          const double node = synthesis * taper_value * real_time(index);
          const double polarisation =
              nonlinear ? coefficient * density(stage) * node * node * node : 0.0;
          projected_real(index) = project_scale * polarisation * time_window(index);
          return;
        }
        const Complex node = synthesis * taper_value * complex_time(index);
        Complex polarisation{};
        if (nonlinear) {
          const double factor = coefficient * density(stage);
          if (cubic_real) {
            polarisation = factor * node * node * node;
          } else {
            const double modulus =
                node.real() * node.real() + node.imag() * node.imag();
            const Complex square = node * node;
            polarisation = factor * (self_factor * modulus * node +
                                     conjugate_factor * square * Kokkos::conj(node) +
                                     harmonic_factor * carrier(index) * square * node);
          }
        }
        polarisation *= project_scale * time_window(index);
        if (family == FieldFamily::envelope) {
          projected_complex(index) = polarisation;
        } else {
          projected_real(index) = polarisation.real();
        }
      });
}

void Workspace::forward_response() {
  if (problem_.family == FieldFamily::envelope) {
    KokkosFFT::execute(*forward_complex_plan_, projected_complex_, projected_spectrum_,
                       KokkosFFT::Normalization::none);
  } else {
    KokkosFFT::execute(*forward_real_plan_, projected_real_, projected_spectrum_,
                       KokkosFFT::Normalization::none);
  }
}

void Workspace::form_absorber() {
  const auto rate = absorber_fine_;
  const auto real_time = fine_time_real_;
  const auto complex_time = fine_time_complex_;
  const auto absorber_real = absorber_real_;
  const auto absorber_complex = absorber_complex_;
  const auto family = problem_.family;
  launch::parallel_for(
      "solprop_native.absorber.nonlinear_term",
      Kokkos::RangePolicy<ExecutionSpace>(response_execution_space_, 0, problem_.n_to),
      KOKKOS_LAMBDA(const std::size_t index) {
        if (family == FieldFamily::envelope) {
          absorber_complex(index) = -0.5 * rate(index) * complex_time(index);
        } else if (family == FieldFamily::field_real) {
          absorber_real(index) = -0.5 * rate(index) * real_time(index);
        } else {
          absorber_real(index) = -0.5 * rate(index) * complex_time(index).real();
        }
      });
}

void Workspace::forward_absorber() {
  if (problem_.family == FieldFamily::envelope) {
    KokkosFFT::execute(*absorber_complex_plan_, absorber_complex_, absorber_spectrum_,
                       KokkosFFT::Normalization::none);
  } else {
    KokkosFFT::execute(*absorber_real_plan_, absorber_real_, absorber_spectrum_,
                       KokkosFFT::Normalization::none);
  }
}

void Workspace::finish_stage(const std::size_t stage) {
  const auto projected = projected_spectrum_;
  const auto absorber = absorber_spectrum_;
  const auto crop = crop_index_;
  const auto active = active_mask_;
  const auto window = omega_window_;
  const auto phi_terms = phi_terms_;
  const auto phi_delta = stage_phi_delta_;
  const auto prefactor_terms = prefactor_terms_;
  const auto prefactor_weights = stage_prefactor_;
  const auto stages = stages_;
  const auto solution = solution_;
  const auto error = error_;
  const std::size_t n_phi = problem_.n_phi_terms;
  const std::size_t n_prefactor = problem_.n_prefactor_terms;
  const bool in_stage_absorber = absorbing();
  const double solution_weight = problem_.solution_weights[stage];
  const double error_weight = problem_.error_weights[stage];
  launch::parallel_for(
      "solprop_native.stage.finish_accumulate",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        // The phase increment: strict (solprop SEP 25 section 4.2).
        const auto stage_phase = [&]() SOLPROP_NATIVE_STRICT_FUNCTION {
          SOLPROP_NATIVE_STRICT_BLOCK
          Complex sum{};
          for (std::size_t term = 0; term < n_phi; ++term)
            sum = strict_accumulate(sum, phi_delta(stage * n_phi + term),
                                    phi_terms(term, index));
          return sum;
        };
        const Complex phase = stage_phase();
        Complex prefactor{};
        for (std::size_t term = 0; term < n_prefactor; ++term) {
          prefactor += prefactor_weights(stage * n_prefactor + term) *
                       prefactor_terms(term, index);
        }
        Complex value = prefactor * window(index) * projected(crop(index));
        if (in_stage_absorber && active(index) != 0U) {
          value += absorber(crop(index));
        }
        value *= exponential(-phase);
        stages(stage, index) = value;
        if (stage == 0) {
          solution(index) = solution_weight * value;
          error(index) = error_weight * value;
        } else {
          solution(index) += solution_weight * value;
          error(index) += error_weight * value;
        }
      });
}

namespace {
/**
 * The Kerr engine's accepted update and its three squared norms in one pass over the
 * bins: `advanced`, then the old, new and error norms' partial sums, reduced in
 * whatever order measures fastest (solprop SEP 25).
 */
struct AcceptScore {
  using value_type = double[];
  using size_type = std::size_t;
  size_type value_count = 3;
  ComplexView current, advanced, solution, error, whole;
  ByteView in_norm;
  double dz;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t index,
                                         value_type sums) const {
    const Complex candidate = whole(index) * (current(index) + dz * solution(index));
    advanced(index) = candidate;
    if (in_norm(index) == 0U)
      return;
    const Complex residual = whole(index) * (dz * error(index));
    sums[0] += current(index).real() * current(index).real() +
               current(index).imag() * current(index).imag();
    sums[1] +=
        candidate.real() * candidate.real() + candidate.imag() * candidate.imag();
    sums[2] += residual.real() * residual.real() + residual.imag() * residual.imag();
  }
  KOKKOS_INLINE_FUNCTION void join(value_type destination,
                                   const value_type source) const {
    for (size_type row = 0; row < value_count; ++row)
      destination[row] += source[row];
  }
  KOKKOS_INLINE_FUNCTION void init(value_type values) const {
    for (size_type row = 0; row < value_count; ++row)
      values[row] = 0.0;
  }
};
} // namespace

void Workspace::accept_and_score(const double dz) {
  const AcceptScore score{3,      current_,          advanced_,  solution_,
                          error_, whole_propagator_, norm_mask_, dz};
  launch::parallel_reduce(
      "solprop_native.step.accept_error_partials",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega), score,
      partials_);
}

void Workspace::mask(const ComplexView& values) {
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.mask.accepted",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        if (active(index) == 0U)
          values(index) = Complex{};
      });
}

void Workspace::split_step(const double dz) {
  const auto advanced = advanced_;
  const auto coarse = coarse_spectrum_;
  const double inverse_n = 1.0 / static_cast<double>(problem_.n_t);
  launch::parallel_for(
      "solprop_native.absorber.split_pad",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        coarse(index) = inverse_n * advanced(index);
      });
  if (problem_.family == FieldFamily::envelope) {
    KokkosFFT::execute(*coarse_inverse_complex_plan_, coarse_spectrum_,
                       coarse_time_complex_, KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz);
    const auto time = coarse_time_complex_;
    const auto rate = absorber_coarse_;
    launch::parallel_for(
        "solprop_native.absorber.split_apply_complex",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_t),
        KOKKOS_LAMBDA(const std::size_t index) {
          time(index) *= Kokkos::exp(-0.5 * rate(index) * dz);
        });
    KokkosFFT::execute(*coarse_forward_complex_plan_, coarse_time_complex_,
                       coarse_spectrum_, KokkosFFT::Normalization::none);
  } else {
    KokkosFFT::execute(*coarse_inverse_real_plan_, coarse_spectrum_, coarse_time_real_,
                       KokkosFFT::Normalization::none);
    if (monitoring_split())
      add_split_removal(dz);
    const auto time = coarse_time_real_;
    const auto rate = absorber_coarse_;
    launch::parallel_for(
        "solprop_native.absorber.split_apply_real",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_t),
        KOKKOS_LAMBDA(const std::size_t index) {
          time(index) *= Kokkos::exp(-0.5 * rate(index) * dz);
        });
    KokkosFFT::execute(*coarse_forward_real_plan_, coarse_time_real_, coarse_spectrum_,
                       KokkosFFT::Normalization::none);
  }
  const auto active = active_mask_;
  launch::parallel_for(
      "solprop_native.absorber.split_commit",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space_, 0, problem_.n_omega),
      KOKKOS_LAMBDA(const std::size_t index) {
        advanced(index) = active(index) != 0U ? coarse(index) : Complex{};
      });
}

std::map<std::string, std::size_t> Workspace::memory_ledger() const {
  const std::size_t complex_bytes = sizeof(Complex);
  const std::size_t real_bytes = sizeof(double);
  const std::size_t fine_time =
      problem_.n_to *
      (problem_.family == FieldFamily::field_real ? real_bytes : complex_bytes);
  const std::size_t projected_time =
      problem_.n_to *
      (problem_.family == FieldFamily::envelope ? complex_bytes : real_bytes);
  return {
      {"state", problem_.n_omega * complex_bytes},
      {"advanced", problem_.n_omega * complex_bytes},
      {"stages", problem_.n_stages * problem_.n_omega * complex_bytes},
      {"accumulator_solution", problem_.n_omega * complex_bytes},
      {"accumulator_error", problem_.n_omega * complex_bytes},
      {"recycled_stage", problem_.n_omega * complex_bytes},
      {"noise", problem_.modified_noise ? problem_.n_omega * complex_bytes : 0},
      {"propagators", problem_.n_omega * complex_bytes},
      {"fine_spectrum", fine_spectrum_size() * complex_bytes},
      {"modal_time", fine_time},
      {"node_field", fine_time},
      {"node_polarisation", projected_time},
      {"projected", projected_time},
      {"projected_coarse", 0},
      {"projected_spectrum", projected_spectrum_size() * complex_bytes},
      {"absorber_time", problem_.in_stage_absorber ? projected_time : 0},
      {"absorber_spectrum",
       problem_.in_stage_absorber ? projected_spectrum_size() * complex_bytes : 0},
      {"coarse_spectrum", coarse_spectrum_.size() * complex_bytes},
      {"coarse_time", coarse_time_real_.size() * real_bytes +
                          coarse_time_complex_.size() * complex_bytes},
      {"monitor_weights",
       (monitor_weights_.size() + monitor_removed_.size() + monitor_split_sum_.size()) *
           real_bytes},
      {"fixed_terms", (problem_.n_phi_terms + problem_.n_prefactor_terms) *
                          problem_.n_omega * complex_bytes},
      {"masks_and_transform_data",
       problem_.crop_index.size() * sizeof(std::size_t) +
           (problem_.active_mask.size() + problem_.norm_mask.size()) *
               sizeof(unsigned char) +
           (problem_.omega_window.size() + problem_.time_window.size() +
            problem_.absorber_fine.size() + problem_.absorber_coarse.size()) *
               real_bytes +
           problem_.carrier.size() * complex_bytes},
      {"tableau", (problem_.tableau_matrix.size() + problem_.tableau_nodes.size() +
                   problem_.solution_weights.size() + problem_.error_weights.size()) *
                      real_bytes},
      {"trial_scalars",
       ((problem_.n_stages + 1) * problem_.n_phi_terms +
        problem_.n_stages * problem_.n_phi_terms +
        problem_.n_stages * problem_.n_prefactor_terms + 2 * problem_.n_stages) *
           real_bytes},
      {"error_partials", 3 * real_bytes},
      {"fft_workspace", 0},
  };
}

void Workspace::close() noexcept {
  if (closed_)
    return;
  execution_space_.fence();
  inverse_real_plan_.reset();
  inverse_complex_plan_.reset();
  forward_real_plan_.reset();
  forward_complex_plan_.reset();
  absorber_real_plan_.reset();
  absorber_complex_plan_.reset();
  coarse_inverse_real_plan_.reset();
  coarse_inverse_complex_plan_.reset();
  coarse_forward_real_plan_.reset();
  coarse_forward_complex_plan_.reset();
  current_ = {};
  advanced_ = {};
  stages_ = {};
  solution_ = {};
  error_ = {};
  recycled_ = {};
  noise_ = {};
  whole_propagator_ = {};
  fine_spectrum_ = {};
  fine_time_real_ = {};
  fine_time_complex_ = {};
  projected_real_ = {};
  projected_complex_ = {};
  projected_spectrum_ = {};
  absorber_real_ = {};
  absorber_complex_ = {};
  absorber_spectrum_ = {};
  coarse_spectrum_ = {};
  coarse_time_real_ = {};
  coarse_time_complex_ = {};
  monitor_bound_ = false;
  monitor_weights_ = {};
  monitor_removed_ = {};
  monitor_split_sum_ = {};
  partials_ = {};
  omega_window_ = {};
  time_window_ = {};
  absorber_fine_ = {};
  absorber_coarse_ = {};
  tableau_ = {};
  stage_phi_delta_ = {};
  stage_phi_at_ = {};
  stage_prefactor_ = {};
  stage_density_ = {};
  stage_taper_ = {};
  carrier_ = {};
  phi_terms_ = {};
  prefactor_terms_ = {};
  active_mask_ = {};
  norm_mask_ = {};
  crop_index_ = {};
  // Release private OpenMP pools while the process runtime is still alive.
  coarse_fft_execution_space_ = ExecutionSpace();
  fft_execution_space_ = ExecutionSpace();
  response_execution_space_ = ExecutionSpace();
  execution_space_ = ExecutionSpace();
  fft_pool_.clear();
  kernel_pool_.clear();
  closed_ = true;
  initialized_ = false;
  clear_fsal();
}

State::State(Problem problem)
    : workspace_(std::make_unique<Workspace>(std::move(problem))) {}
State::~State() { close(); }
void State::set_initial(const std::span<const Complex> field,
                        const std::span<const Complex> noise) {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->set_initial(field, noise);
}
std::array<double, 3> State::trial_step(const double dz, const TrialScalars& scalars) {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->trial_step(dz, scalars);
}
void State::finish(const double dz) {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->finish(dz);
}
void State::promote_fsal() {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->promote_fsal();
}
void State::clear_fsal() {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->clear_fsal();
}
void State::set_absorber_quiet(const bool quiet) {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->set_absorber_quiet(quiet);
}
void State::bind_monitor(const std::vector<double>& weights,
                         const std::vector<double>& column_weights) {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->bind_monitor(weights, column_weights);
}
void State::unbind_monitor() noexcept {
  if (workspace_)
    workspace_->unbind_monitor();
}
std::vector<double> State::monitor_reading() {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->monitor_reading();
}
std::vector<Complex> State::field_to_host() const {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->field_to_host();
}
void State::field_to_host(const std::span<Complex> destination) const {
  const CpuTeamScope team(workspace_->cpu_team());
  workspace_->field_to_host(destination);
}
std::map<std::string, std::size_t> State::memory_ledger() const {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->memory_ledger();
}
CpuSizing State::execution_sizing() const {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->execution_sizing();
}
double State::planning_seconds() const noexcept {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->planning_seconds();
}
bool State::fsal_ready() const noexcept {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->fsal_ready();
}
std::size_t State::rk_source_evaluations() const noexcept {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->rk_source_evaluations();
}
bool State::closed() const noexcept {
  const CpuTeamScope team(workspace_->cpu_team());
  return workspace_->closed();
}
void State::close() noexcept {
  if (workspace_)
    workspace_->close();
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
