// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "solprop_native/crossing.hpp"

#include "solprop_native/build.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#endif

namespace solprop_native::crossing {
namespace {

// Kokkos holds its copies to one thread at a time per instance, so the workers call the
// vendor's synchronous copy, which is safe from many threads: it is what was measured.
void device_read(void* destination, const void* source, const std::size_t bytes) {
#ifdef KOKKOS_ENABLE_CUDA
  if (cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    throw std::runtime_error("a state's copy to the host failed");
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipMemcpy(destination, const_cast<void*>(source), bytes, hipMemcpyDeviceToHost) !=
      hipSuccess)
    throw std::runtime_error("a state's copy to the host failed");
#else
  std::memcpy(destination, source, bytes);
#endif
}

void device_write(void* destination, const void* source, const std::size_t bytes) {
#ifdef KOKKOS_ENABLE_CUDA
  if (cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("a state's copy to the device failed");
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipMemcpy(destination, const_cast<void*>(source), bytes, hipMemcpyHostToDevice) !=
      hipSuccess)
    throw std::runtime_error("a state's copy to the device failed");
#else
  std::memcpy(destination, source, bytes);
#endif
}

/** The device a new thread must select: the choice is per thread on both vendors. */
int current_device() {
#ifdef KOKKOS_ENABLE_CUDA
  return Kokkos::Cuda{}.cuda_device();
#elif defined(KOKKOS_ENABLE_HIP)
  return Kokkos::HIP{}.hip_device();
#else
  return 0;
#endif
}

void select_device([[maybe_unused]] const int device) {
#ifdef KOKKOS_ENABLE_CUDA
  if (cudaSetDevice(device) != cudaSuccess)
    throw std::runtime_error("a crossing's worker could not select its device");
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipSetDevice(device) != hipSuccess)
    throw std::runtime_error("a crossing's worker could not select its device");
#endif
}

/** The host threads this process may run on, at most `worker_limit`. */
std::size_t worker_count() {
#ifdef __linux__
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
    return std::clamp<std::size_t>(static_cast<std::size_t>(CPU_COUNT(&allowed)), 1,
                                   worker_limit);
#endif
  return std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, worker_limit);
}

} // namespace

bool takes_ring(const std::size_t bytes) noexcept {
  return build::device && bytes >= ring_threshold;
}

void BufferDeleter::operator()(double* pointer) const noexcept { std::free(pointer); }

HostBuffer host_buffer(const std::size_t count) {
  const std::size_t bytes = count * sizeof(double);
#ifdef __linux__
  if (takes_ring(bytes)) {
    // The size huge pages come in, which a large buffer is aligned to.
    constexpr std::size_t huge_page = std::size_t{2} << 20;
    const std::size_t rounded = (bytes + huge_page - 1) / huge_page * huge_page;
    auto* pointer = static_cast<double*>(std::aligned_alloc(huge_page, rounded));
    if (pointer == nullptr)
      throw std::bad_alloc{};
    // Advice only: a kernel that refuses it still gives ordinary pages.
    static_cast<void>(madvise(pointer, rounded, MADV_HUGEPAGE));
    return HostBuffer(pointer);
  }
#endif
  auto* pointer = static_cast<double*>(std::malloc(std::max(bytes, sizeof(double))));
  if (pointer == nullptr)
    throw std::bad_alloc{};
  return HostBuffer(pointer);
}

void Ring::set_workers(const std::size_t workers) noexcept {
  workers_ = std::clamp<std::size_t>(workers, 1, worker_limit);
}

void Ring::allocate() {
  if (!chunks_.empty())
    return;
  const auto workers = workers_ != 0 ? workers_ : worker_count();
  chunks_.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker)
    chunks_.emplace_back(Kokkos::view_alloc("solprop_native.crossing.chunk",
                                            Kokkos::WithoutInitializing),
                         chunk_bytes);
}

template <class Move> void Ring::run(const std::size_t bytes, const Move& move) {
  allocate();
  const int device = current_device();
  std::atomic<std::size_t> cursor{0};
  std::atomic<bool> failed{false};
  std::string failure;
  std::atomic_flag reporting = ATOMIC_FLAG_INIT;
  const auto work = [&](const std::size_t worker) {
    try {
      select_device(device);
      auto* chunk = chunks_[worker].data();
      while (!failed.load(std::memory_order_relaxed)) {
        const std::size_t first = cursor.fetch_add(chunk_bytes);
        if (first >= bytes)
          break;
        move(chunk, first, std::min(chunk_bytes, bytes - first));
      }
    } catch (const std::exception& error) {
      failed.store(true);
      if (!reporting.test_and_set())
        failure = error.what();
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(chunks_.size() - 1);
  for (std::size_t worker = 1; worker < chunks_.size(); ++worker)
    workers.emplace_back(work, worker);
  work(0);
  for (auto& worker : workers)
    worker.join();
  if (failed.load())
    throw std::runtime_error(failure);
}

void Ring::to_host(void* host, const void* memory, const std::size_t bytes) {
  run(bytes, [&](std::byte* chunk, const std::size_t first, const std::size_t size) {
    device_read(chunk, static_cast<const std::byte*>(memory) + first, size);
    std::memcpy(static_cast<std::byte*>(host) + first, chunk, size);
  });
}

void Ring::from_host(void* memory, const void* host, const std::size_t bytes) {
  run(bytes, [&](std::byte* chunk, const std::size_t first, const std::size_t size) {
    std::memcpy(chunk, static_cast<const std::byte*>(host) + first, size);
    device_write(static_cast<std::byte*>(memory) + first, chunk, size);
  });
}

std::size_t Ring::bytes() const noexcept { return chunks_.size() * chunk_bytes; }

void Ring::release() noexcept { chunks_.clear(); }

} // namespace solprop_native::crossing
