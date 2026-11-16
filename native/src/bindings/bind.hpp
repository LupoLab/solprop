// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "solprop_native/runtime/cpu_execution.hpp"
#include "solprop_native/runtime/fields.hpp"

#include <nanobind/nanobind.h>

#include <optional>

/*
 * The entry points of the extension's binding files and the declarations they share.
 * module.cpp's `NB_MODULE(_core, ...)` calls the entry points in one fixed order, which
 * the type names in later signatures and the generated stubs depend on: the problem
 * (problem.cpp), the session, workspace and probe classes (module.cpp), then the
 * private `_testing` module: `bind_helpers` (helpers.hpp), `bind_transverse`
 * (transverse.hpp), `bind_testing` and `bind_engines` (testing.cpp). runtime.cpp holds
 * the runtime's configuration, its library lookups and the build record; march.cpp the
 * marches' records (march.hpp); module.hpp the classes the test entry points take.
 */
namespace solprop_native::bindings {

namespace nb = nanobind;
namespace native = solprop_native;

// A native object the runtime closes before Kokkos is finalised: a state or a
// workspace, each of which closes the session that leases it first.
class Owned {
public:
  Owned() = default;
  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;
  Owned(Owned&&) = delete;
  Owned& operator=(Owned&&) = delete;
  virtual ~Owned() = default;
  virtual void close() noexcept = 0;
};

/** Hold `owned` to be closed before Kokkos is finalised (runtime.cpp); forget it. */
void register_owned(Owned* owned);
void forget_owned(Owned* owned);

// The runtime's one record (solprop SEP 52 section 5.3): the thread pools, the BLAS
// fitting and the spin, applied once at import.
struct RuntimeRecord {
  int threads = 0;
  int kernels = 0;
  int fft = 0;
  int raman_fft = 0;
  int raman_tail_fft = 0;
  int band_tiles = -1;
  int spin_milliseconds = 0;
  std::optional<int> blas_threads;
  bool host_blas = true;
};

[[nodiscard]] constexpr auto describe(const RuntimeRecord* /*unused*/) {
  using R = RuntimeRecord;
  return native::fields::table<R>(
      "RuntimeRecord", native::fields::value("threads", &R::threads),
      native::fields::value("kernels", &R::kernels),
      native::fields::value("fft", &R::fft),
      native::fields::value("raman_fft", &R::raman_fft),
      native::fields::value("raman_tail_fft", &R::raman_tail_fft),
      native::fields::value("band_tiles", &R::band_tiles),
      native::fields::value("spin_milliseconds", &R::spin_milliseconds),
      native::fields::value("blas_threads", &R::blas_threads),
      native::fields::value("host_blas", &R::host_blas));
}

/** Apply the runtime record once, at import, and return the BLAS thread policy it
    applied (`_core.configure_runtime`). */
nb::dict configure_runtime(nb::handle value);
/** The build's provenance and the runtime's settings (`_core.build_record`). */
nb::dict build_record();
/** Close what the runtime holds and finalise the Kokkos runtime it owns, after the
    interpreter has released every Python object (registered with `Py_AtExit`). */
void shutdown_after_interpreter() noexcept;

/** A workspace's FFT planning outcome and its CPU execution record, as the
    workspace's diagnostics report them (problem.cpp). */
nb::dict fft_planning_dict(const native::FftPlanning& planning);
nb::dict cpu_execution_dict(const native::CpuExecution& execution);

/** Bind `FieldFamily` and `Problem`, with its field groups, into the extension. */
void bind_problem(nb::module_& module_);
/** Bind the private `_testing` module's entry points, then its `Engines`. */
void bind_testing(nb::module_& testing);
void bind_engines(nb::module_& testing);

} // namespace solprop_native::bindings
