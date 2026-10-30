// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <Kokkos_Core.hpp>

#include <utility>

#ifndef SOLPROP_NATIVE_DEVICE_GRAPH
#define SOLPROP_NATIVE_DEVICE_GRAPH 1
#endif

#if (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)) &&                     \
    SOLPROP_NATIVE_DEVICE_GRAPH
#define SOLPROP_NATIVE_IMPL_GRAPH 1
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>
#else
#define SOLPROP_NATIVE_IMPL_GRAPH 0
#endif

/** A device region's operations replayed as one graph (CUDA or HIP).

    On an AMD Instinct MI325X a kernel launched on a stream behind the one before it
    costs the device about 3.4 microseconds however little it does, and about 2.2 as a
    node of an instantiated graph; a trial's source evaluation is a couple of dozen
    such kernels. A `Region` runs host code whose device work goes only through
    `launch::parallel_for`, `graph::deep_copy` and `graph::execute`, which record each
    operation's identity (a `Key`): its kind and the value bytes of its policy and
    functor, or of its views, plan and scalars. The first run captures the stream's
    work into a graph, which is instantiated and launched. A later run issues nothing
    while each operation matches the recorded one in order; when all have matched, the
    instantiated graph runs in their place. At the first operation that differs, or a
    run that ends early, the matched ones are issued again from their recorded copies
    under a new capture, which takes the rest of the run and replaces the graph. The
    host code runs every time, so that its own state advances as without a graph.

    Why a match replays exactly what issuing the run would. A graph node holds what its
    capture issued: a kernel with its argument bytes and launch shape, a copy or a fill
    with its addresses, sizes and value. A launch's argument is Kokkos's closure of the
    functor and policy, formed from their values alone, and its shape follows from the
    same values on the same device; a copy's and a fill's follow from their views'
    values (pointers, extents and strides) and the value; a transform's follow from its
    plan and views. So equal keys issue equal nodes, and equal nodes do what the run's
    own operations would. In turn:
    - Scalars captured by value are part of the functor's bytes: a changed scalar is a
      changed key.
    - A pointer in a key is compared as a value, so a view reallocated at another
      address is a different key; one reallocated at the same address with the same
      shape is the same kernel argument, and the replay addresses the new allocation
      exactly as the run would.
    - A plan is keyed by its address: the workspace that owns a region builds its plans
      once, at construction, and frees them only at `close`, so one address is one plan
      (with its own vendor workspace) for the region's life. An execution that would
      allocate (Kokkos-FFT's transposed or reshaped plans) cannot run in a region:
      Kokkos initialises an allocation with a fence, which a capture refuses.
    - Host memory never enters a graph: a copy to or from the host disables the region
      before it is issued, so that no node reads a staging buffer whose contents change
      from trial to trial; no fence can run inside one, so the host's writes that a
      region's kernels could see are those before the region, as without a graph.
    - Branches of the host code that change what is issued change the sequence of keys,
      and only an unchanged sequence replays.
    - Bytes that are not a value are left out of a key: a type's padding as far as a
      copy shows it (the rest is zeroed on the stack beforehand, `detail::scrub_stack`),
      and the host pointer that nvcc's wrapper of an extended
      `__host__ __device__` lambda appends to its captures (`detail::mask_host_copy`).
    The device sees its kernels' arguments only; what a recorded copy of an operation
    reads on the host (a label) changes no device result.

    What cannot be in a graph disables the region, which issues what it holds and
    then runs as written, as it does on every later run: reductions and scans (Kokkos
    may move their scratch), vendor GEMM, team launches with level-1 scratch, functors
    Kokkos would stage in constant or global memory, work on another stream, copies to
    or from the host, and any run while a Kokkos tool is loaded (its callbacks may fence
    or tune a policy). Every run captures, so that device work which bypassed the
    wrappers is found (the capture is not empty) before the next operation is skipped:
    what the run matched is issued, then the stray work, and the region is disabled. A
    capture holding anything but kernels, fills and device copies is refused as a
    `std::logic_error`. Host builds, and device builds configured with
    `SOLPROP_NATIVE_DEVICE_GRAPH=OFF`, run the region's code as written. */
namespace solprop_native::graph {

inline constexpr bool available = SOLPROP_NATIVE_IMPL_GRAPH != 0;

namespace detail {
/** The kinds of operation, as a key's type: a launch of `Functor` over `Policy`, a
    copy or fill, a transform. */
template <class Policy, class Functor> struct Launch {};
template <class Destination, class Source> struct Copy {};
template <class Plan, class Input, class Output> struct Transform {};
} // namespace detail

#if SOLPROP_NATIVE_IMPL_GRAPH

#if defined(KOKKOS_ENABLE_CUDA)
using Stream = cudaStream_t;
using GraphHandle = cudaGraph_t;
using ExecHandle = cudaGraphExec_t;
#else
using Stream = hipStream_t;
using GraphHandle = hipGraph_t;
using ExecHandle = hipGraphExec_t;
#endif

class Region;

namespace detail {
/** The region recording the calling thread's device operations, if any. */
inline thread_local Region* active = nullptr;

#if defined(KOKKOS_ENABLE_CUDA)
inline void check(const cudaError_t status, const char* call) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("solprop_native device graph: ") + call +
                             ": " + cudaGetErrorString(status));
}
#else
inline void check(const hipError_t status, const char* call) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("solprop_native device graph: ") + call +
                             ": " + hipGetErrorString(status));
}
#endif

template <class Space> Stream stream_of(const Space& space) {
#if defined(KOKKOS_ENABLE_CUDA)
  return space.cuda_stream();
#else
  return space.hip_stream();
#endif
}

inline void begin_capture(Stream stream) {
#if defined(KOKKOS_ENABLE_CUDA)
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
#else
  check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
        "hipStreamBeginCapture");
#endif
}

inline GraphHandle end_capture(Stream stream) {
  GraphHandle graph{};
#if defined(KOKKOS_ENABLE_CUDA)
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
#else
  check(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
#endif
  return graph;
}

inline void destroy(GraphHandle graph) noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
  static_cast<void>(cudaGraphDestroy(graph));
#else
  static_cast<void>(hipGraphDestroy(graph));
#endif
}

inline void destroy(ExecHandle exec) noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
  static_cast<void>(cudaGraphExecDestroy(exec));
#else
  static_cast<void>(hipGraphExecDestroy(exec));
#endif
}

inline void launch(ExecHandle exec, Stream stream) {
#if defined(KOKKOS_ENABLE_CUDA)
  check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
#else
  check(hipGraphLaunch(exec, stream), "hipGraphLaunch");
#endif
}

/** `graph` instantiated; the graph itself is destroyed. */
inline ExecHandle instantiate(GraphHandle graph) {
  ExecHandle exec{};
#if defined(KOKKOS_ENABLE_CUDA)
  const auto status = cudaGraphInstantiate(&exec, graph, 0);
#else
  const auto status = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
#endif
  destroy(graph);
  check(status, "graph instantiation");
  return exec;
}

/** The nodes captured on `stream` so far. */
inline std::size_t captured(Stream stream) {
  GraphHandle graph{};
  std::size_t count = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaStreamCaptureStatus status{};
  check(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph),
        "cudaStreamGetCaptureInfo");
  check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
#else
  hipStreamCaptureStatus status{};
  check(hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, nullptr, nullptr),
        "hipStreamGetCaptureInfo_v2");
  check(hipGraphGetNodes(graph, nullptr, &count), "hipGraphGetNodes");
#endif
  return count;
}

/** Whether `pointer` is known to be host memory (pinned or registered). */
inline bool host_memory(const void* pointer) noexcept {
  if (pointer == nullptr)
    return false;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  return attributes.type == cudaMemoryTypeHost;
#else
  hipPointerAttribute_t attributes{};
  if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess) {
    static_cast<void>(hipGetLastError());
    return false;
  }
  return attributes.type == hipMemoryTypeHost;
#endif
}

/** Refuse a graph holding anything but kernels, fills and copies between device
    buffers: a transfer from the host, an event or a host callback would replay what
    was true at its capture. The graph is destroyed with the refusal. */
inline void validate(GraphHandle graph) {
  std::size_t count = 0;
  bool device_only = true;
#if defined(KOKKOS_ENABLE_CUDA)
  check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
  std::vector<cudaGraphNode_t> nodes(count);
  if (count != 0)
    check(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
  for (const auto node : nodes) {
    cudaGraphNodeType type{};
    check(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    if (type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeMemset)
      continue;
    cudaMemcpy3DParms copy{};
    device_only = type == cudaGraphNodeTypeMemcpy &&
                  cudaGraphMemcpyNodeGetParams(node, &copy) == cudaSuccess &&
                  !host_memory(copy.srcPtr.ptr) && !host_memory(copy.dstPtr.ptr);
    if (!device_only)
      break;
  }
#else
  check(hipGraphGetNodes(graph, nullptr, &count), "hipGraphGetNodes");
  std::vector<hipGraphNode_t> nodes(count);
  if (count != 0)
    check(hipGraphGetNodes(graph, nodes.data(), &count), "hipGraphGetNodes");
  for (const auto node : nodes) {
    hipGraphNodeType type{};
    check(hipGraphNodeGetType(node, &type), "hipGraphNodeGetType");
    if (type == hipGraphNodeTypeKernel || type == hipGraphNodeTypeMemset)
      continue;
    // A one-dimensional copy's parameters may not read back: only a copy known to
    // touch the host is refused.
    hipMemcpy3DParms copy{};
    device_only = type == hipGraphNodeTypeMemcpy &&
                  (hipGraphMemcpyNodeGetParams(node, &copy) != hipSuccess ||
                   (!host_memory(copy.srcPtr.ptr) && !host_memory(copy.dstPtr.ptr)));
    static_cast<void>(hipGetLastError());
    if (!device_only)
      break;
  }
#endif
  if (!device_only) {
    destroy(graph);
    throw std::logic_error(
        "a device graph region captured a host transfer, event or callback");
  }
}

/** Zero the stack below the caller. Padding inside a closure holds whatever its stack
    slot held before (a copy carries it along with the members), and a key compares a
    closure's bytes: values left there by the host's work before a region, or by the
    capture's own calls, would differ from run to run, so that a repeated sequence would
    not match. Zeroing at a region's start and after each operation leaves only what
    the region's own host code writes. How often keys match depends on it; what a match
    means does not. */
[[gnu::noinline]] inline void scrub_stack() noexcept {
  unsigned char area[32768];
  std::memset(area, 0, sizeof(area));
  asm volatile("" : : "r"(area) : "memory");
}

/** Launch `graph` once; it is destroyed. */
inline void launch_once(GraphHandle graph, Stream stream) {
  const ExecHandle exec = instantiate(graph);
  launch(exec, stream);
  destroy(exec);
}

/** Leave the host's own heap copy out of the key of nvcc's wrapper of an extended
    `__host__ __device__` lambda. The wrapper holds the captures, which its device
    code reads, then a `void* data` member: the host's copy of the lambda, allocated
    afresh by every construction and copy (so it differs from trial to trial) and
    never read by the device. It is the last member, so it ends the object or is
    followed only by tail padding (alignment above a pointer's); it is found as the
    word whose bytes differ between the two live copies `zeros` and `ones` (distinct
    allocations) without all differing (the high bytes of two heap addresses agree,
    while padding a copy does not write shows each buffer's fill). Anything else is
    kept in the key: a key that differs only costs a capture. */
inline void mask_host_copy(std::vector<unsigned char>& mask, const unsigned char* zeros,
                           const unsigned char* ones, const std::size_t size,
                           const std::size_t alignment) {
  constexpr std::size_t word = sizeof(void*);
  if (size < word || alignment > 2 * word)
    return;
  const auto differing = [&](const std::size_t at) {
    std::size_t count = 0;
    for (std::size_t index = at; index < at + word; ++index)
      count += zeros[index] != ones[index] ? 1 : 0;
    return count;
  };
  auto data = size - word;
  if (differing(data) == word) {
    // Tail padding: the pointer is the word before it.
    if (alignment <= word || size < 2 * word)
      return;
    data -= word;
  }
  const auto count = differing(data);
  if (count == 0 || count == word)
    return;
  for (std::size_t index = data; index < data + word; ++index)
    mask[index] = 0;
}

/** The bytes of a `T` that are its value, as 0xFF over them and 0 elsewhere: those a
    copy writes alike into a buffer of zeros and one of ones. Padding a copy does not
    write is left out; padding a copy carries over stays in, which can only make two
    equal values differ. */
template <class T> std::vector<unsigned char> probe_mask(const T& object) {
  alignas(T) std::array<unsigned char, sizeof(T)> zeros{};
  alignas(T) std::array<unsigned char, sizeof(T)> ones{};
  ones.fill(0xFF);
  T* const first = new (zeros.data()) T(object);
  T* const second = new (ones.data()) T(object);
  std::vector<unsigned char> mask(sizeof(T));
  for (std::size_t index = 0; index < sizeof(T); ++index)
    mask[index] = zeros[index] == ones[index] ? 0xFF : 0x00;
#if defined(__CUDACC_EXTENDED_LAMBDA__)
  if constexpr (__nv_is_extended_host_device_lambda_closure_type(T))
    mask_host_copy(mask, zeros.data(), ones.data(), sizeof(T), alignof(T));
#endif
  first->~T();
  second->~T();
  return mask;
}

template <class T> const std::vector<unsigned char>& value_mask(const T& object) {
  // Formed once a type, from the first object recorded.
  static const std::vector<unsigned char> mask = probe_mask(object);
  return mask;
}

} // namespace detail

/** An operation's identity: its kind, then the value bytes of each part in turn. */
class Key {
public:
  void clear(const std::type_info& kind) noexcept {
    kind_ = &kind;
    bytes_.clear();
  }
  template <class T> void add(const T& part) {
    const auto& mask = detail::value_mask(part);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&part);
    const auto start = bytes_.size();
    bytes_.resize(start + sizeof(T));
    for (std::size_t index = 0; index < sizeof(T); ++index)
      bytes_[start + index] = static_cast<unsigned char>(bytes[index] & mask[index]);
  }
  [[nodiscard]] bool operator==(const Key& other) const noexcept {
    return kind_ != nullptr && other.kind_ != nullptr && *kind_ == *other.kind_ &&
           bytes_ == other.bytes_;
  }

private:
  const std::type_info* kind_ = nullptr;
  std::vector<unsigned char> bytes_;
};

/** What a wrapper does with its operation. */
enum class Admit {
  issue,  ///< issue it: no region records it
  skip,   ///< it matched: the region's graph issues it
  record, ///< issue it, and give the region a copy that issues it again (`keep`)
};

class Region {
public:
  Region() = default;
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;
  Region(Region&&) = delete;
  Region& operator=(Region&&) = delete;
  ~Region() { reset(); }

  /** Run `body` as the region on `space`'s stream (see the namespace description). */
  template <class Space, class Body> void run(const Space& space, Body&& body);

  /** The calling thread's recording region, or null. */
  [[nodiscard]] static Region* current() noexcept { return detail::active; }

  /** The stream the current run captures. */
  [[nodiscard]] Stream stream() const noexcept { return stream_; }
  /** What to do with the next operation, whose identity is `key`. */
  Admit admit(const Key& key);
  /** The copy of the operation just admitted to `record`. */
  void keep(const Key& key, std::function<void()>&& issue) {
    operations_.push_back(Operation{key, std::move(issue)});
    next_ = operations_.size();
  }
  /** The next operation cannot be in a graph: issue what the region holds, stop. */
  void refuse();

  /** Runs that launched the recorded graph, and runs that captured one. */
  [[nodiscard]] std::size_t replays() const noexcept { return replays_; }
  [[nodiscard]] std::size_t captures() const noexcept { return captures_; }
  [[nodiscard]] bool disabled() const noexcept { return disabled_; }
  /** Drop the graph and its records, as when the views they name change. */
  void reset() noexcept {
    if (exec_ != nullptr)
      detail::destroy(exec_);
    exec_ = nullptr;
    operations_.clear();
    next_ = 0;
  }

private:
  enum class Mode { off, verify, capture };
  struct Operation {
    Key key;
    std::function<void()> issue;
  };
  void reissue(const std::size_t count) {
    for (std::size_t index = 0; index < count; ++index)
      operations_[index].issue();
  }
  void restart();
  void stray();
  void stop() noexcept {
    reset();
    disabled_ = true;
    mode_ = Mode::off;
    detail::active = nullptr;
  }
  void finish();
  void abandon() noexcept;

  Stream stream_{};
  ExecHandle exec_{};
  Mode mode_ = Mode::off;
  bool disabled_ = false;
  std::size_t next_ = 0;
  std::vector<Operation> operations_;
  std::size_t replays_ = 0;
  std::size_t captures_ = 0;
};

// The run departs from the recorded sequence after `next_` matched operations: they
// are issued again under a fresh capture, which takes the rest of the run.
inline void Region::restart() {
  detail::destroy(detail::end_capture(stream_));
  detail::begin_capture(stream_);
  operations_.resize(next_);
  reissue(next_);
  mode_ = Mode::capture;
}

// A verifying run's capture is not empty: work bypassed the wrappers since the last
// operation, after every matched one. They run, then the stray work, and the region
// stops.
inline void Region::stray() {
  const GraphHandle graph = detail::end_capture(stream_);
  mode_ = Mode::off;
  detail::validate(graph);
  reissue(next_);
  detail::launch_once(graph, stream_);
  stop();
}

inline Admit Region::admit(const Key& key) {
  if (mode_ == Mode::capture)
    return Admit::record;
  if (detail::captured(stream_) != 0) {
    stray();
    return Admit::issue;
  }
  if (next_ < operations_.size() && operations_[next_].key == key) {
    ++next_;
    return Admit::skip;
  }
  restart();
  return Admit::record;
}

inline void Region::refuse() {
  if (mode_ == Mode::verify) {
    if (detail::captured(stream_) != 0) {
      stray();
      return;
    }
    // The operations matched so far were skipped: issue them now.
    detail::destroy(detail::end_capture(stream_));
    mode_ = Mode::off;
    reissue(next_);
  } else {
    // The captured operations, in order, run once.
    const GraphHandle graph = detail::end_capture(stream_);
    mode_ = Mode::off;
    detail::validate(graph);
    detail::launch_once(graph, stream_);
  }
  stop();
}

inline void Region::finish() {
  if (mode_ == Mode::verify) {
    if (detail::captured(stream_) != 0) {
      stray();
      return;
    }
    if (next_ == operations_.size()) {
      detail::destroy(detail::end_capture(stream_));
      mode_ = Mode::off;
      detail::launch(exec_, stream_);
      ++replays_;
      return;
    }
    // The run ended before the recorded sequence: its matched part is the graph.
    restart();
  }
  const GraphHandle graph = detail::end_capture(stream_);
  mode_ = Mode::off;
  detail::validate(graph);
  if (exec_ != nullptr)
    detail::destroy(exec_);
  exec_ = nullptr;
  exec_ = detail::instantiate(graph);
  detail::launch(exec_, stream_);
  ++captures_;
}

// A run that throws: what it issued before the throw still runs, as without a graph,
// while the capture can still be ended; the region stops.
inline void Region::abandon() noexcept {
  detail::active = nullptr;
  try {
    if (mode_ == Mode::verify && detail::captured(stream_) == 0) {
      detail::destroy(detail::end_capture(stream_));
      mode_ = Mode::off;
      reissue(next_);
    } else if (mode_ != Mode::off) {
      const GraphHandle graph = detail::end_capture(stream_);
      mode_ = Mode::off;
      detail::validate(graph);
      detail::launch_once(graph, stream_);
    }
  } catch (...) {
    // The run's own exception is the one reported.
  }
  stop();
}

template <class Space, class Body> void Region::run(const Space& space, Body&& body) {
  if (disabled_ || detail::active != nullptr || Kokkos::Tools::profileLibraryLoaded()) {
    body();
    return;
  }
  stream_ = detail::stream_of(space);
  next_ = 0;
  mode_ = exec_ != nullptr ? Mode::verify : Mode::capture;
  if (mode_ == Mode::capture)
    operations_.clear();
  detail::begin_capture(stream_);
  detail::active = this;
  detail::scrub_stack();
  try {
    body();
  } catch (...) {
    abandon();
    throw;
  }
  detail::active = nullptr;
  finish();
}

namespace detail {
/** The calling thread's key, reused from operation to operation. */
inline Key& scratch() {
  thread_local Key key;
  return key;
}
} // namespace detail

/** The next operation cannot be in a graph: the calling thread's region, if any,
    stops. */
inline void unrecordable() {
  if (Region* const region = Region::current())
    region->refuse();
}

/** Issue the operation of kind `Kind` that `parts...` identify, as `issue`, through
    the calling thread's region (which the caller has checked exists). */
template <class Kind, class Issue, class... Parts>
void operation(const Issue& issue, const Parts&... parts) {
  Region* const region = Region::current();
  Key& key = detail::scratch();
  key.clear(typeid(Kind));
  (key.add(parts), ...);
  switch (region->admit(key)) {
  case Admit::skip:
    break;
  case Admit::record:
    region->keep(key, std::function<void()>(issue));
    issue();
    break;
  case Admit::issue:
    issue();
    break;
  }
  detail::scrub_stack();
}

namespace detail {
/** A label that outlives a recorded launch: an array (a literal) is kept as its
    address, anything else as a string. */
template <typename Label> auto kept_label(const Label& label) {
  if constexpr (std::is_array_v<Label>)
    return static_cast<const char*>(label);
  else
    return std::string(label);
}

template <typename Policy> struct is_team_policy : std::false_type {};
template <typename... Properties>
struct is_team_policy<Kokkos::TeamPolicy<Properties...>> : std::true_type {};

/** Kokkos passes a light-weight launch's closure (the functor, the policy, sometimes
    a second policy and a few bounds) as the kernel's argument below 4 kB and stages a
    larger one in global memory from the host: all of it must fit. */
template <typename Functor, typename Policy>
inline constexpr bool argument_launch =
    Kokkos::is_execution_policy_v<Policy> &&
    sizeof(Functor) + 2 * sizeof(Policy) + 256 < 4096;
} // namespace detail

/** `Kokkos::parallel_for(label, policy, functor)` in the calling thread's region (which
    the caller has checked exists), `policy` hinted light-weight: recorded when a graph
    may hold it, otherwise issued after the region stops. */
template <typename Label, typename Policy, typename Functor>
void parallel_for(const Label& label, const Policy& policy, const Functor& functor) {
  bool recordable = false;
  if constexpr (detail::argument_launch<Functor, Policy>) {
    recordable = detail::stream_of(policy.space()) == Region::current()->stream();
    if constexpr (detail::is_team_policy<Policy>::value)
      recordable = recordable && policy.scratch_size(1) == 0;
  }
  if (!recordable) {
    unrecordable();
    Kokkos::parallel_for(label, policy, functor);
    return;
  }
  operation<detail::Launch<Policy, Functor>>(
      [name = detail::kept_label(label), policy, functor] {
        Kokkos::parallel_for(name, policy, functor);
      },
      policy, functor);
}

namespace detail {
/** Whether a view of `Memory` lives where `Space` runs and not on the host. */
template <class Space, class Memory>
inline constexpr bool device_resident =
    Kokkos::SpaceAccessibility<Space, Memory>::accessible &&
    !Kokkos::SpaceAccessibility<Kokkos::HostSpace, Memory>::accessible;

/** A copy or fill between device buffers, whose kernel (if Kokkos needs one) passes
    its views as its argument. */
template <class Space, class Destination, class Source>
constexpr bool recordable_copy() {
  if constexpr (!Kokkos::is_view_v<Destination> ||
                sizeof(Destination) + sizeof(Source) > 256)
    return false;
  else if constexpr (!device_resident<Space, typename Destination::memory_space>)
    return false;
  else if constexpr (Kokkos::is_view_v<Source>)
    return device_resident<Space, typename Source::memory_space>;
  else
    return std::is_trivially_copyable_v<Source>;
}
} // namespace detail

/** `Kokkos::deep_copy(space, destination, source)` -- from a view or a value, to a
    view or a scalar -- recorded in the calling thread's region when both sides live
    where `space` runs; a copy that crosses to or from the host stops the region
    first. Host builds call `Kokkos::deep_copy` itself. */
template <class Space, class Destination, class Source>
void deep_copy(const Space& space, Destination&& destination, const Source& source) {
  using Target = std::remove_cvref_t<Destination>;
  if (Region::current() == nullptr) {
    Kokkos::deep_copy(space, std::forward<Destination>(destination), source);
    return;
  }
  if constexpr (detail::recordable_copy<Space, Target, Source>()) {
    if (detail::stream_of(space) == Region::current()->stream()) {
      operation<detail::Copy<Target, Source>>(
          [space, target = Target(destination), source] {
            Kokkos::deep_copy(space, target, source);
          },
          destination, source);
      return;
    }
  }
  unrecordable();
  Kokkos::deep_copy(space, std::forward<Destination>(destination), source);
}

#else

/** Host builds (and `SOLPROP_NATIVE_DEVICE_GRAPH=OFF`): a region runs its code. */
class Region {
public:
  template <class Space, class Body> void run(const Space& /*space*/, Body&& body) {
    body();
  }
  [[nodiscard]] static Region* current() noexcept { return nullptr; }
  void reset() noexcept {}
};

inline void unrecordable() noexcept {}

template <typename Label, typename Policy, typename Functor>
void parallel_for(const Label& label, const Policy& policy, const Functor& functor) {
  Kokkos::parallel_for(label, policy, functor);
}

// The wrappers' calls are Kokkos's own.
using Kokkos::deep_copy;

#endif

} // namespace solprop_native::graph
