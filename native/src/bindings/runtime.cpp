// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bind.hpp"
#include "records.hpp"
#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/execution.hpp"
#include "solprop_native/runtime/vmath.hpp"

#include <Kokkos_Core.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solprop_native {
// The digest of what this extension is compiled from (cmake/SourceDigest.cmake).
const char* source_digest() noexcept;
} // namespace solprop_native

namespace solprop_native::bindings {

namespace {

struct RuntimeState {
  std::mutex mutex;
  bool owns_runtime = false;
  int requested_threads = 0;
  // The BLAS thread policy the package applied at start-up, if any.
  std::string blas_policy;
  std::vector<Owned*> states;
};

RuntimeState runtime_state;

void initialize_kokkos(const int threads, const int kernels, const int fft,
                       const int raman_fft, const int raman_tail_fft,
                       const int band_tiles) {
  if (threads < 1)
    throw std::invalid_argument("thread count must be positive");
  const std::scoped_lock lock(runtime_state.mutex);
  if (!Kokkos::is_initialized()) {
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(threads);
    Kokkos::initialize(settings);
    runtime_state.owns_runtime = true;
    runtime_state.requested_threads = threads;
  } else if (runtime_state.requested_threads != 0 &&
             runtime_state.requested_threads != threads) {
    throw std::runtime_error(
        "Kokkos was already initialized with another thread count");
  }
  native::configure_thread_policy({.kernels = kernels,
                                   .fft = fft,
                                   .raman_fft = raman_fft,
                                   .raman_tail_fft = raman_tail_fft,
                                   .band_tiles = band_tiles});
}

void shutdown_kokkos() {
  const std::scoped_lock lock(runtime_state.mutex);
  for (auto* state : runtime_state.states)
    state->close();
  if (runtime_state.owns_runtime && Kokkos::is_initialized() &&
      !Kokkos::is_finalized()) {
    Kokkos::finalize();
    runtime_state.owns_runtime = false;
  }
}

std::string decode_record_text(std::string value) {
  const std::string marker = "__SOLPROP_SPACE__";
  std::size_t position = 0;
  while ((position = value.find(marker, position)) != std::string::npos) {
    value.replace(position, marker.size(), " ");
    ++position;
  }
  return value;
}

#ifdef _WIN32
// Windows looks a symbol up in one module only, so the search walks the modules this
// extension imports (its FFT, BLAS and threading libraries, under whatever names the
// wheel repair gave them), as dlsym on the extension's own handle does on POSIX.
HMODULE native_module() {
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&decode_record_text), &module) == 0)
    throw std::runtime_error("cannot locate the native module for library provenance");
  return module;
}

std::pair<void*, HMODULE> find_symbol(const char* symbol) {
  const HMODULE self = native_module();
  // A library linked statically into the extension (the wheel's OpenBLAS) is found
  // through the extension's own exports.
  if (const auto address = GetProcAddress(self, symbol))
    return {reinterpret_cast<void*>(address), self};
  const auto* base = reinterpret_cast<const unsigned char*>(self);
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  const auto& directory =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (directory.VirtualAddress == 0)
    return {nullptr, nullptr};
  for (const auto* entry = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(
           base + directory.VirtualAddress);
       entry->Name != 0; ++entry) {
    const HMODULE dependency =
        GetModuleHandleA(reinterpret_cast<const char*>(base + entry->Name));
    if (dependency == nullptr)
      continue;
    if (const auto address = GetProcAddress(dependency, symbol))
      return {reinterpret_cast<void*>(address), dependency};
  }
  return {nullptr, nullptr};
}

std::string loaded_library(const char* symbol) {
  const auto [address, module] = find_symbol(symbol);
  if (address == nullptr)
    return {};
  std::wstring path(MAX_PATH, L'\0');
  const auto length =
      GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  if (length == 0)
    return {};
  path.resize(length);
  const int bytes =
      WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()),
                          nullptr, 0, nullptr, nullptr);
  std::string result(static_cast<std::size_t>(bytes), '\0');
  WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()),
                      result.data(), bytes, nullptr, nullptr);
  return result;
}

void* library_symbol(const char* symbol) { return find_symbol(symbol).first; }
#else
struct LibraryCloser {
  // Keep glibc's function attributes out of unique_ptr's template arguments.
  void operator()(void* handle) const noexcept { static_cast<void>(dlclose(handle)); }
};

auto native_library_handle() {
  Dl_info module{};
  if (dladdr(reinterpret_cast<const void*>(&decode_record_text), &module) == 0)
    throw std::runtime_error("cannot locate the native module for library provenance");
  std::unique_ptr<void, LibraryCloser> handle(
      dlopen(module.dli_fname, RTLD_NOW | RTLD_NOLOAD));
  if (!handle)
    throw std::runtime_error("cannot inspect the native module's loaded libraries");
  return handle;
}

std::string loaded_library(const char* symbol) {
  const auto handle = native_library_handle();
  const auto* address = dlsym(handle.get(), symbol);
  Dl_info provider{};
  return address != nullptr && dladdr(address, &provider) != 0
             ? std::string(provider.dli_fname)
             : std::string{};
}

void* library_symbol(const char* symbol) {
  const auto handle = native_library_handle();
  return dlsym(handle.get(), symbol);
}
#endif

// A distribution build's OpenBLAS (scipy-openblas32) prefixes every symbol.
std::string blas_symbol(const char* name) {
  return std::string(SOLPROP_NATIVE_BLAS_SYMBOL_PREFIX) + name;
}

template <class... Arguments>
std::optional<int> library_integer(const char* symbol, Arguments... arguments) {
  void* const address = library_symbol(symbol);
  if (address == nullptr)
    return std::nullopt;
  return reinterpret_cast<int (*)(Arguments...)>(address)(arguments...);
}

// Fit a host OpenBLAS to the Kokkos pool. A pthreads OpenBLAS keeps its own thread
// pool beside Kokkos's and oversubscribes the cores, so it is held to one thread; an
// OpenMP OpenBLAS shares the runtime but sizes itself from OMP_NUM_THREADS (every
// logical CPU when unset), so it is capped at the Kokkos thread count. Returns the
// policy applied, or an empty string when nothing was changed.
std::string fit_blas_threads(const int threads) {
  const std::scoped_lock lock(runtime_state.mutex);
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) != "OpenBLAS" || threads < 1)
    return {};
  const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
  void* const address = library_symbol(blas_symbol("openblas_set_num_threads").c_str());
  if (address == nullptr || (parallel != 1 && parallel != 2))
    return {};
  reinterpret_cast<void (*)(int)>(address)(parallel == 1 ? 1 : threads);
  runtime_state.blas_policy =
      parallel == 1 ? "one thread beside the Kokkos pool" : "the Kokkos thread count";
  return runtime_state.blas_policy;
}

// Let a compact problem's scope hold an OpenMP OpenBLAS at one thread (`CpuTeamScope`),
// whatever width the environment or `fit_blas_threads` gave it. A pthreads or serial
// OpenBLAS keeps no team beside Kokkos's to resize.
void register_host_blas() {
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) != "OpenBLAS")
    return;
  const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
  native::register_openmp_blas(parallel == 2);
}

} // namespace

void register_owned(Owned* owned) {
  const std::scoped_lock lock(runtime_state.mutex);
  runtime_state.states.push_back(owned);
}

void forget_owned(Owned* owned) {
  const std::scoped_lock lock(runtime_state.mutex);
  const auto found = std::ranges::find(runtime_state.states, owned);
  if (found != runtime_state.states.end())
    runtime_state.states.erase(found);
}

nb::dict configure_runtime(const nb::handle value) {
  const auto record = records::read<RuntimeRecord>(value, "runtime");
  initialize_kokkos(record.threads, record.kernels, record.fft, record.raman_fft,
                    record.raman_tail_fft, record.band_tiles);
  native::configure_native_spin(record.spin_milliseconds);
  nb::dict out;
  out["blas_policy"] =
      record.blas_threads ? fit_blas_threads(*record.blas_threads) : std::string();
  if (record.host_blas)
    register_host_blas();
  return out;
}

// Kokkos::finalize runs Kokkos-FFT's hook, fftw_cleanup_threads, which destroys
// FFTW's process-wide planner. Any other user of the same FFTW library must have
// destroyed its plans by then, so the owned
// runtime is finalised after the interpreter has released every Python object,
// rather than from a Python exit handler, which runs before. Nothing here calls
// Python.
void shutdown_after_interpreter() noexcept { shutdown_kokkos(); }

nb::dict build_record() {
  const std::scoped_lock lock(runtime_state.mutex);
  nb::dict result;
  result["backend_commit"] = SOLPROP_NATIVE_BACKEND_COMMIT;
  result["solprop_built_against"] = SOLPROP_NATIVE_SOLPROP_BUILT_AGAINST;
  result["source_digest"] = native::source_digest();
  result["kokkos_version"] = SOLPROP_NATIVE_KOKKOS_VERSION;
  result["kokkos_fft_version"] = SOLPROP_NATIVE_KOKKOSFFT_VERSION;
  result["kokkos_kernels_version"] = SOLPROP_NATIVE_KOKKOSKERNELS_VERSION;
  result["nanobind_version"] = SOLPROP_NATIVE_NANOBIND_VERSION;
  result["gemm_provider"] = "HYBRID";
  result["tpl_blas_enabled"] = (SOLPROP_NATIVE_TPL_BLAS_ENABLED != 0);
  result["blas_backend"] = SOLPROP_NATIVE_BLAS_BACKEND;
  result["blas_thread_policy"] = decode_record_text(SOLPROP_NATIVE_BLAS_THREAD_POLICY);
  result["blas_selection"] = decode_record_text(SOLPROP_NATIVE_BLAS_SELECTION);
  result["build_profile"] = SOLPROP_NATIVE_BUILD_PROFILE;
  result["cpu_baseline"] = SOLPROP_NATIVE_CPU_BASELINE;
  // The host kernels' transcendental passes (solprop ADR-059): SLEEF's u10 functions
  // in the variant the build targets, or the system libm.
  result["vector_math"] = solprop_native::vmath::vectorised
                              ? std::string("SLEEF ") + SOLPROP_NATIVE_SLEEF_VERSION +
                                    " u10, " + solprop_native::vmath::route
                              : std::string(solprop_native::vmath::route);
  std::optional<int> blas_threads;
  std::string blas_threading = "provider-managed";
  if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) == "OpenBLAS") {
    blas_threads = library_integer(blas_symbol("openblas_get_num_threads").c_str());
    const auto parallel = library_integer(blas_symbol("openblas_get_parallel").c_str());
    blas_threading = parallel == 2   ? "OpenMP"
                     : parallel == 1 ? "pthreads"
                     : parallel == 0 ? "serial"
                                     : "unknown";
    if (!runtime_state.blas_policy.empty())
      result["blas_thread_policy"] = runtime_state.blas_policy;
    else if (parallel == 2)
      result["blas_thread_policy"] = "OMP_NUM_THREADS at process start";
  } else if (std::string(SOLPROP_NATIVE_BLAS_BACKEND) == "MKL") {
    // MKL_DOMAIN_BLAS is 1 in the installed/public LP64 service interface.
    blas_threads = library_integer("MKL_Domain_Get_Max_Threads", 1);
    blas_threading = "MKL BLAS domain";
  }
  result["blas_threads"] = blas_threads;
  result["blas_threading"] = blas_threading;
  result["execution_space"] = native::ExecutionSpace::name();
#ifdef KOKKOS_ENABLE_CUDA
  result["fft_backend"] = "cuFFT";
  result["fft_library"] = loaded_library("cufftExecZ2Z");
  result["blas_library"] = loaded_library("cublasDgemm_v2");
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = Kokkos::num_devices();
#elif defined(KOKKOS_ENABLE_HIP)
  result["fft_backend"] = "rocFFT";
  result["fft_library"] = loaded_library("rocfft_execute");
  result["blas_library"] = loaded_library("rocblas_dgemm");
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = Kokkos::num_devices();
#elif defined(KOKKOS_ENABLE_OPENMP)
  result["fft_backend"] = SOLPROP_NATIVE_HOST_FFT_BACKEND;
  result["fft_library"] = loaded_library("fftw_execute");
  result["blas_library"] = loaded_library(blas_symbol("dgemm_").c_str());
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = 0;
#else
  result["fft_backend"] = SOLPROP_NATIVE_HOST_FFT_BACKEND;
  result["fft_library"] = loaded_library("fftw_execute");
  result["blas_library"] = loaded_library(blas_symbol("dgemm_").c_str());
  result["device_backend"] = SOLPROP_NATIVE_DEVICE_BACKEND;
  result["device_arch"] = SOLPROP_NATIVE_DEVICE_ARCH;
  result["device_count"] = 0;
#endif
  result["raman_fft_option"] = SOLPROP_NATIVE_RAMAN_FFT_PROVIDER;
#ifdef SOLPROP_NATIVE_RAMAN_ACCELERATE
  result["raman_fft_library"] = loaded_library("vDSP_fftm_zropD");
#else
  result["raman_fft_library"] = result["fft_library"];
#endif
  result["openmp_library"] = loaded_library("omp_get_num_threads");
  result["device_team_size"] = SOLPROP_NATIVE_DEVICE_TEAM_SIZE_RECORD;
  result["device_reduction_size"] = SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE_RECORD;
#ifdef KOKKOS_ENABLE_OPENMP
  result["openmp_enabled"] = true;
#else
  result["openmp_enabled"] = false;
#endif
  result["host_execution_space"] = Kokkos::DefaultHostExecutionSpace::name();
  const auto threads = native::thread_policy();
  result["thread_count"] = threads.kernels;
  result["runtime_threads"] = Kokkos::DefaultHostExecutionSpace().concurrency();
  result["requested_threads"] = runtime_state.requested_threads;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  result["fftw_threads"] = 0;
  result["raman_fft_threads"] = 0;
  result["raman_tail_fft_threads"] = 0;
#else
  result["fftw_threads"] = threads.fft;
  result["raman_fft_threads"] = threads.raman_fft;
  result["raman_tail_fft_threads"] = threads.raman_tail_fft;
#endif
  result["cpu_size_policy_enabled"] = native::cpu_size_policy_enabled;
  result["explicit_thread_pools"] = threads.explicit_pools;
  result["band_column"] = native::band_tiled(threads) ? "tiled" : "team";
  result["band_tile_threads"] = native::band_tile_threads(threads);
  result["cxx_standard"] = 20;
  result["compiler"] = __VERSION__;
  result["compiler_flags"] = decode_record_text(SOLPROP_NATIVE_COMPILER_FLAGS);
  result["build_type"] = SOLPROP_NATIVE_BUILD_TYPE;
  result["coverage_instrumented"] = (SOLPROP_NATIVE_COVERAGE_INSTRUMENTED != 0);
  result["coverage_source_hash"] = SOLPROP_NATIVE_COVERAGE_SOURCE_HASH;
  result["fft_normalization"] = "none (explicit inverse scaling)";
#ifdef KOKKOS_ENABLE_CUDA
  result["fft_planner"] = "Kokkos-FFT persistent Plan / cuFFT";
#elif defined(KOKKOS_ENABLE_HIP)
  result["fft_planner"] = "Kokkos-FFT persistent Plan / rocFFT";
#else
  result["fft_planner"] = std::string("Kokkos-FFT persistent Plan / ") +
                          SOLPROP_NATIVE_HOST_FFT_PLANNER +
                          "; wisdom seeded before planning: rows measured serial, "
                          "estimate threaded; transverse planes measured; "
                          "SOLPROP_FFTW_PLAN overrides";
#endif
  result["layout"] = "LayoutRight";
  result["native_arch_enabled"] = (SOLPROP_NATIVE_HOST_ARCH_ENABLED != 0);
  result["aggressive_vectorization_enabled"] =
      (SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED != 0);
  result["ipo_enabled"] = (SOLPROP_NATIVE_IPO_ENABLED != 0);
  const std::string cpu_baseline = SOLPROP_NATIVE_CPU_BASELINE;
  const std::string architecture_policy =
      SOLPROP_NATIVE_HOST_ARCH_ENABLED ? "native architecture"
      : cpu_baseline == "none"         ? "generic architecture"
                                       : cpu_baseline + " baseline";
  const std::string ipo_policy =
      SOLPROP_NATIVE_IPO_ENABLED ? "checked IPO" : "IPO disabled";
  const std::string vectorization_policy =
      SOLPROP_NATIVE_AGGRESSIVE_VECTORIZATION_ENABLED
          ? "aggressive Kokkos loop vectorization"
          : "standard Kokkos loop vectorization";
  // solprop SEP 25: contraction on (protected blocks strict), or the device compiler's.
  const std::string contraction = std::string(SOLPROP_NATIVE_FP_CONTRACT_RECORD);
  const std::string contraction_policy =
      contraction.empty() ? "compiler-default contraction"
                          : "contraction " + contraction + ", protected blocks strict";
  result["optimization_policy"] = std::string("CMake ") + SOLPROP_NATIVE_BUILD_TYPE +
                                  "; no fast-math; " + contraction_policy + "; " +
                                  architecture_policy + "; " + ipo_policy + "; " +
                                  vectorization_policy;
#ifdef __FAST_MATH__
  result["fast_math_enabled"] = true;
#else
  result["fast_math_enabled"] = false;
#endif
#ifdef __apple_build_version__
  result["compiler_family"] = "AppleClang";
#elif defined(__clang__)
  result["compiler_family"] = "Clang";
#elif defined(__GNUC__)
  result["compiler_family"] = "GNU";
#else
  result["compiler_family"] = "unknown";
#endif
  result["live_native_states"] = runtime_state.states.size();
  return result;
}

} // namespace solprop_native::bindings
