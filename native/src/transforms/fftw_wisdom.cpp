// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/transforms/fftw_wisdom.hpp"

#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace solprop_native {

unsigned fftw_seed_effort(const FftShape shape, const int threads) {
  const char* text = std::getenv("SOLPROP_FFTW_PLAN");
  if (text == nullptr || *text == '\0')
    return shape == FftShape::rows && threads > 1 ? FFTW_ESTIMATE : FFTW_MEASURE;
  const std::string_view effort = text;
  if (effort == "estimate")
    return FFTW_ESTIMATE;
  if (effort == "measure")
    return FFTW_MEASURE;
  if (effort == "patient")
    return FFTW_PATIENT;
  throw std::invalid_argument("SOLPROP_FFTW_PLAN must be estimate, measure or patient, "
                              "and is '" +
                              std::string(effort) + "'");
}

unsigned fftw_effort_flag(const std::string_view effort) {
  if (effort == "estimate")
    return FFTW_ESTIMATE;
  if (effort == "measure")
    return FFTW_MEASURE;
  if (effort == "patient")
    return FFTW_PATIENT;
  throw std::invalid_argument("an FFTW effort is estimate, measure or patient, not '" +
                              std::string(effort) + "'");
}

std::string export_fftw_wisdom() {
  fftw_init_threads();
  // FFTW allocates the text with plain malloc and documents free() for it: fftw_free
  // pairs with fftw_malloc's aligned allocator, which on Windows is another heap.
  const auto release = [](char* pointer) { std::free(pointer); };
  const std::unique_ptr<char, decltype(release)> text(fftw_export_wisdom_to_string(),
                                                      release);
  return text ? std::string(text.get()) : std::string();
}

bool import_fftw_wisdom(const std::string& text) {
  fftw_init_threads();
  return fftw_import_wisdom_from_string(text.c_str()) != 0;
}

void forget_fftw_wisdom() {
  fftw_init_threads();
  fftw_forget_wisdom();
}

namespace detail {
namespace {
// The planning of the workspace this thread is building (`FftPlanningScope`).
thread_local FftPlanning* planning = nullptr;
} // namespace
} // namespace detail

namespace {
// Whether `planned` holds a wisdom entry `stored` lacks: a seed that measured. FFTW
// writes one entry a line, under a preamble line that both share.
bool adds_wisdom(const std::string& planned, const std::string& stored) {
  const auto lines = [](const std::string& text) {
    std::vector<std::string_view> found;
    for (std::size_t first = 0; first < text.size();) {
      const std::size_t last = std::min(text.find('\n', first), text.size());
      found.emplace_back(text.data() + first, last - first);
      first = last + 1;
    }
    std::ranges::sort(found);
    return found;
  };
  const auto had = lines(stored);
  return std::ranges::any_of(lines(planned), [&had](const auto line) {
    return !std::ranges::binary_search(had, line);
  });
}

// Merge `text` beneath the process's wisdom: where both hold an entry, the process's
// stays, so that a profile's plans keep theirs.
void import_beneath(const std::string& text) {
  const std::string held = export_fftw_wisdom();
  fftw_forget_wisdom();
  static_cast<void>(fftw_import_wisdom_from_string(text.c_str()));
  if (!held.empty())
    static_cast<void>(fftw_import_wisdom_from_string(held.c_str()));
}

// The wisdom already known for a transform: measured earlier in this workspace, or
// stored by an earlier run; none when it must be measured.
const std::string* known_wisdom(const FftPlanning& planning, const std::string& key) {
  if (const auto found = planning.measured.find(key); found != planning.measured.end())
    return &found->second;
  if (const auto found = planning.stored.find(key); found != planning.stored.end())
    return &found->second;
  return nullptr;
}

// Begin planning a transform alone (`FftPlanning::stored`): set the workspace's wisdom
// aside and import only the transform's own known wisdom; a transform to be measured
// starts from what this workspace measured so far, so that it shares their sub-plans
// as a workspace planning in one wisdom does, and from nothing else.
void plan_alone(FftPlanning& planning, const std::string& key) {
  planning.held = export_fftw_wisdom();
  fftw_forget_wisdom();
  if (const std::string* known = known_wisdom(planning, key))
    static_cast<void>(fftw_import_wisdom_from_string(known->c_str()));
  else
    for (const auto& entry : planning.measured)
      static_cast<void>(fftw_import_wisdom_from_string(entry.second.c_str()));
  planning.alone = true;
}

// End a transform planned alone: keep its own wisdom if it measured, note a reuse of a
// stored plan if not, then give the workspace back its wisdom, imported over the
// transform's so that a profile's plans keep their entries. Every seed begun alone
// ends here, planned or not.
void release_seed(const std::string& key, const bool planned) {
  FftPlanning* planning = detail::planning;
  if (planning == nullptr || !planning->alone)
    return;
  planning->alone = false;
  std::string own;
  if (planned) {
    own = export_fftw_wisdom();
    const std::string* known = known_wisdom(*planning, key);
    if (known != nullptr && !adds_wisdom(own, *known)) {
      if (!planning->measured.contains(key))
        planning->reused.push_back(key);
    } else if (planning->keep) {
      planning->measured[key] = own;
    }
  }
  fftw_forget_wisdom();
  if (!own.empty())
    static_cast<void>(fftw_import_wisdom_from_string(own.c_str()));
  if (!planning->held.empty())
    static_cast<void>(fftw_import_wisdom_from_string(planning->held.c_str()));
  planning->held.clear();
}

} // namespace

FftPlanningScope::FftPlanningScope(FftPlanning& planning)
    : saved_(detail::planning), forgets_(planning.mode != FftPlanning::Mode::legacy) {
  detail::planning = &planning;
  if (forgets_) {
    fftw_init_threads();
    fftw_forget_wisdom();
  }
}

FftPlanningScope::~FftPlanningScope() {
  if (forgets_)
    fftw_forget_wisdom();
  detail::planning = saved_;
}

std::optional<unsigned> begin_seed(const std::string& key, const FftShape shape,
                                   const int threads) {
  FftPlanning* planning = detail::planning;
  if (planning == nullptr) {
    const unsigned effort = fftw_seed_effort(shape, threads);
    if (effort == FFTW_ESTIMATE)
      return std::nullopt;
    return effort;
  }
  unsigned effort = FFTW_ESTIMATE;
  if (planning->mode == FftPlanning::Mode::legacy) {
    effort = fftw_seed_effort(shape, threads);
  } else {
    const char* group = shape == FftShape::planes ? "planes"
                        : threads > 1             ? "threaded_rows"
                                                  : "rows";
    planning->groups[key] = group;
    if (planning->mode == FftPlanning::Mode::capture) {
      const unsigned captured = fftw_effort_flag(planning->capture_effort);
      if (captured == FFTW_ESTIMATE)
        return std::nullopt;
      fftw_forget_wisdom();
      fftw_set_timelimit(planning->time_limit > 0.0 ? planning->time_limit
                                                    : FFTW_NO_TIMELIMIT);
      return captured;
    }
    const std::string& chosen = shape == FftShape::planes ? planning->planes
                                : threads > 1             ? planning->threaded_rows
                                                          : planning->rows;
    effort = fftw_effort_flag(chosen);
    if (const auto found = planning->wisdom.find(key);
        found != planning->wisdom.end()) {
      if (fftw_import_wisdom_from_string(found->second.c_str()) == 0) {
        planning->missing.push_back(key);
        return std::nullopt;
      }
      return effort | FFTW_WISDOM_ONLY;
    }
  }
  const auto stored = planning->stored.find(key);
  if (effort == FFTW_ESTIMATE) {
    if (stored != planning->stored.end()) {
      import_beneath(stored->second);
      planning->reused.push_back(key);
    }
    return std::nullopt;
  }
  if (planning->keep || known_wisdom(*planning, key) != nullptr)
    plan_alone(*planning, key);
  return effort;
}

void end_seed(const std::string& key, const unsigned flags, const bool planned) {
  FftPlanning* planning = detail::planning;
  const bool capture =
      planning != nullptr && planning->mode == FftPlanning::Mode::capture;
  if (capture)
    fftw_set_timelimit(FFTW_NO_TIMELIMIT);
  release_seed(key, planned);
  if (!planned) {
    if ((flags & FFTW_WISDOM_ONLY) != 0U && planning != nullptr) {
      planning->missing.push_back(key);
      return;
    }
    throw std::runtime_error("FFTW could not plan the transform " + key);
  }
  if (capture) {
    const auto release = [](char* pointer) { std::free(pointer); };
    const std::unique_ptr<char, decltype(release)> text(fftw_export_wisdom_to_string(),
                                                        release);
    planning->captured[key] = text ? std::string(text.get()) : std::string();
  }
}

std::string fft_plan_key(const std::string_view kind, const std::string& sizes,
                         const int howmany, const int input_distance,
                         const int output_distance, const void* input,
                         const void* output, const int threads) {
  const auto alignment = [](const void* pointer) {
    return std::to_string(
        fftw_alignment_of(static_cast<double*>(const_cast<void*>(pointer))));
  };
  return std::string(kind) + ":" + sizes + ":" + std::to_string(howmany) + ":" +
         std::to_string(input_distance) + ":" + std::to_string(output_distance) + ":" +
         (input == output ? "in" : "out") + ":" + alignment(input) + ":" +
         alignment(output) + ":t" + std::to_string(threads);
}

namespace {
std::vector<std::string> split_key(const std::string& text, const char by) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  for (std::size_t at = text.find(by); at != std::string::npos;
       start = at + 1, at = text.find(by, start))
    parts.push_back(text.substr(start, at - start));
  parts.push_back(text.substr(start));
  return parts;
}

double median_execution(fftw_plan plan) {
  using Clock = std::chrono::steady_clock;
  fftw_execute(plan);
  const auto first = Clock::now();
  fftw_execute(plan);
  const double once = std::chrono::duration<double>(Clock::now() - first).count();
  const int repeats = std::max(2, static_cast<int>(0.005 / std::max(once, 1e-7)));
  std::array<double, 5> batches{};
  for (double& batch : batches) {
    const auto start = Clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat)
      fftw_execute(plan);
    batch = std::chrono::duration<double>(Clock::now() - start).count() / repeats;
  }
  std::ranges::sort(batches);
  return batches[2];
}
} // namespace

PlanAlone time_fft_plan(const std::string& key, const std::string& wisdom,
                        const std::string& effort) {
  const auto parts = split_key(key, ':');
  if (parts.size() != 9 || parts[8].empty() || parts[8][0] != 't')
    throw std::invalid_argument("not an FFT plan key: " + key);
  const std::string& kind = parts[0];
  std::vector<int> extents;
  for (const auto& extent : split_key(parts[1], 'x'))
    extents.push_back(std::stoi(extent));
  const int howmany = std::stoi(parts[2]);
  const int input_distance = std::stoi(parts[3]);
  const int output_distance = std::stoi(parts[4]);
  const bool in_place = parts[5] == "in";
  const auto input_offset = static_cast<std::size_t>(std::stoi(parts[6]));
  const auto output_offset = static_cast<std::size_t>(std::stoi(parts[7]));
  const int threads = std::stoi(parts[8].substr(1));
  // The step's waiting policy and FFTW team: without the scope LLVM's runtime sleeps
  // its idle threads at once (`KMP_BLOCKTIME=0`) and each loop pays their wake-up.
  const CpuTeamScope scope(threads);
  const bool r2c = kind == "r2c";
  const bool c2r = kind == "c2r";
  const bool planes = kind.starts_with("planes-");
  const int sign = kind.ends_with("backward") ? FFTW_BACKWARD : FFTW_FORWARD;
  if (!r2c && !c2r && !planes && !kind.starts_with("c2c-"))
    throw std::invalid_argument("not an FFT plan key: " + key);
  const auto rank = static_cast<int>(extents.size());
  // Real input or output counts doubles, complex counts double pairs.
  const std::size_t input_bytes = static_cast<std::size_t>(howmany) *
                                  static_cast<std::size_t>(input_distance) *
                                  (r2c ? sizeof(double) : sizeof(fftw_complex));
  const std::size_t output_bytes = static_cast<std::size_t>(howmany) *
                                   static_cast<std::size_t>(output_distance) *
                                   (c2r ? sizeof(double) : sizeof(fftw_complex));
  fftw_init_threads();
  const auto release = [](void* pointer) { fftw_free(pointer); };
  const std::unique_ptr<void, decltype(release)> input_memory(
      fftw_malloc(std::max(input_bytes, output_bytes) + 64), release);
  const std::unique_ptr<void, decltype(release)> output_memory(
      in_place ? nullptr : fftw_malloc(output_bytes + 64), release);
  char* input_base = static_cast<char*>(input_memory.get()) + input_offset;
  char* output_base =
      in_place ? input_base : static_cast<char*>(output_memory.get()) + output_offset;
  const auto make = [&](const unsigned flags) -> fftw_plan {
    fftw_plan_with_nthreads(threads);
    if (r2c)
      return fftw_plan_many_dft_r2c(
          rank, extents.data(), howmany, reinterpret_cast<double*>(input_base), nullptr,
          1, input_distance, reinterpret_cast<fftw_complex*>(output_base), nullptr, 1,
          output_distance, flags);
    if (c2r)
      return fftw_plan_many_dft_c2r(
          rank, extents.data(), howmany, reinterpret_cast<fftw_complex*>(input_base),
          nullptr, 1, input_distance, reinterpret_cast<double*>(output_base), nullptr,
          1, output_distance, flags);
    return fftw_plan_many_dft(
        rank, extents.data(), howmany, reinterpret_cast<fftw_complex*>(input_base),
        nullptr, 1, input_distance, reinterpret_cast<fftw_complex*>(output_base),
        nullptr, 1, output_distance, sign, flags);
  };
  const auto timed = [&](const unsigned flags) {
    fftw_forget_wisdom();
    if ((flags & FFTW_WISDOM_ONLY) != 0U &&
        fftw_import_wisdom_from_string(wisdom.c_str()) == 0)
      return std::numeric_limits<double>::quiet_NaN();
    auto* const plan = make(flags);
    if (plan == nullptr) {
      fftw_forget_wisdom();
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Real input is overwritten by complex-to-real plans and harmless otherwise.
    std::memset(input_base, 0, input_bytes);
    const double seconds = median_execution(plan);
    fftw_destroy_plan(plan);
    fftw_forget_wisdom();
    return seconds;
  };
  PlanAlone result;
  result.plan = timed(fftw_effort_flag(effort) | FFTW_WISDOM_ONLY);
  result.estimate = timed(FFTW_ESTIMATE);
  return result;
}

void seed_fftw_wisdom(const ExecutionSpace& space, Complex* input, Complex* output,
                      const std::vector<int>& extents, const int howmany,
                      const int input_distance, const int output_distance,
                      const int sign) {
  std::string sizes;
  for (const int extent : extents)
    sizes += (sizes.empty() ? "" : "x") + std::to_string(extent);
  const std::string key = fft_plan_key(
      sign == FFTW_FORWARD ? "planes-forward" : "planes-backward", sizes, howmany,
      input_distance, output_distance, input, output, space.concurrency());
  fftw_init_threads();
  const auto chosen = begin_seed(key, FftShape::planes, space.concurrency());
  if (!chosen)
    return;
  fftw_plan_with_nthreads(space.concurrency());
  auto* in = reinterpret_cast<fftw_complex*>(input);
  auto* out = reinterpret_cast<fftw_complex*>(output);
  auto* const seeded = fftw_plan_many_dft(
      static_cast<int>(extents.size()), extents.data(), howmany, in, nullptr, 1,
      input_distance, out, nullptr, 1, output_distance, sign, *chosen);
  // The policy has always tolerated a plane it cannot seed; a profile records it.
  if (seeded == nullptr && (*chosen & FFTW_WISDOM_ONLY) == 0U &&
      (detail::planning == nullptr ||
       detail::planning->mode == FftPlanning::Mode::legacy)) {
    release_seed(key, false);
    return;
  }
  end_seed(key, *chosen, seeded != nullptr);
  if (seeded != nullptr)
    fftw_destroy_plan(seeded);
}

} // namespace solprop_native
#endif
