// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <memory>
#include <vector>

/** A state-sized crossing between a device and the host, and the host memory it lands
   in.

    A driver's copy to fresh pageable memory takes its first-touch page faults inside
   the copy, one thread at a time: a 5.27 GiB state crossed an NVIDIA H200's PCIe Gen 5
   x16 at 1.8 GiB/s and an AMD Instinct MI325X's link at 7.9 GiB/s, against about 52
   GiB/s pinned on both. So a large crossing goes through a ring of pinned chunks, one
   per host thread, each thread moving its chunks between the link and the ordinary
   memory, which it touches itself. That reached 33 GiB/s and 50 GiB/s into a fresh
   huge-page buffer on those two cards (`docs/dev/benchmarks.md`).

    A pinned mirror of the whole state wouldn't help: a returned array owns its memory,
   so the copy out of the mirror takes the same faults, and it would hold another state.
 */
namespace solprop_native::crossing {

/** A ring chunk, in bytes: a worker's unit of work and its pinned memory. */
inline constexpr std::size_t chunk_bytes = std::size_t{64} << 20;

/** The fewest bytes worth a ring. Below this a plain copy is a small part of a step,
   and a run that only ever makes small crossings holds no pinned memory. */
inline constexpr std::size_t ring_threshold = std::size_t{256} << 20;

/** The most workers a ring takes, in bytes `workers * chunk_bytes` held pinned. */
inline constexpr std::size_t worker_limit = 24;

/** Whether a crossing of `bytes` takes a ring on this build: never on a host build,
   whose memory space is the host's. */
[[nodiscard]] bool takes_ring(std::size_t bytes) noexcept;

/** Frees what `host_buffer` returned. */
struct BufferDeleter {
  void operator()(double* pointer) const noexcept;
};

/** Uninitialised doubles for a returned array to own. */
using HostBuffer = std::unique_ptr<double[], BufferDeleter>;

/** Return `count` uninitialised doubles for a crossing to land in, aligned for huge
   pages and asking for them where the buffer is large enough to use them on a device
   build. */
[[nodiscard]] HostBuffer host_buffer(std::size_t count);

/** The pinned chunks and the workers that move a crossing through them. Allocated at
   the first crossing, so that a run that keeps no field holds nothing. */
class Ring {
public:
  /** Take `workers` threads from the next allocation on, at most `worker_limit`.
     Without this the ring asks the scheduler, which an OpenMP runtime's binding can
     mislead. */
  void set_workers(std::size_t workers) noexcept;
  /** Copy `bytes` of memory-space data to host memory. The caller has fenced. */
  void to_host(void* host, const void* memory, std::size_t bytes);
  /** Copy `bytes` of host memory into the memory space. The caller has fenced. */
  void from_host(void* memory, const void* host, std::size_t bytes);
  /** The pinned bytes held, for the memory ledger. */
  [[nodiscard]] std::size_t bytes() const noexcept;
  /** Free the chunks. */
  void release() noexcept;

private:
  using Chunk = Kokkos::View<std::byte*, Kokkos::SharedHostPinnedSpace>;
  void allocate();
  template <class Move> void run(std::size_t bytes, const Move& move);
  std::vector<Chunk> chunks_;
  std::size_t workers_ = 0;
};

} // namespace solprop_native::crossing
