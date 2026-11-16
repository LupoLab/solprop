// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>

/*
 * A source the host evaluates for the resident Adams march (solprop SEP 49 section
 * 6.2): a lab-frame field in, a spectral source out, and an attempt's start and end.
 * It knows nothing of what evaluates it; solprop's TDSE response is its first user,
 * through the binding layer, which wraps a Python object and takes the interpreter
 * lock for each call. The engine calls it from the thread that called `advance`,
 * outside every parallel region, and only on a CPU build: a device build refuses a
 * problem that needs one before anything is allocated.
 *
 * Python-free: an exception the host raises reaches the march as a `HostSourceError`,
 * the original kept by the binding layer for solprop's session to raise.
 */
namespace solprop_native {

/** Why an evaluation is made, the march's label for it (solprop's `TDSEPurpose`). */
enum class HostPurpose : int { startup = 0, predictor = 1 };

/** How an attempt ended: rejected or accepted after its decision, or abandoned by a
    failure (or a floor) before one, which only releases what it reserved. */
enum class HostOutcome : int { rejected = 0, accepted = 1, abandoned = 2 };

/** A host source's failure: the host's own exception is kept where it was raised. */
class HostSourceError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** The three entry points of an attempt (SEP 49 section 6.2). */
class HostSource {
public:
  HostSource() = default;
  HostSource(const HostSource&) = delete;
  HostSource& operator=(const HostSource&) = delete;
  HostSource(HostSource&&) = delete;
  HostSource& operator=(HostSource&&) = delete;
  virtual ~HostSource() = default;
  /** Before an attempt's first evaluation, after the march's budget check: `cost`
      evaluations will follow. */
  virtual void begin(std::uint64_t cost) = 0;
  /** One evaluation: the lab-frame spectrum `field` (`n_omega` bins of one column) at
      `z`, m, into `source`, the spectral source before the spectral collar. */
  virtual void evaluate(std::span<const std::complex<double>> field, double z,
                        HostPurpose purpose,
                        std::span<std::complex<double>> source) = 0;
  /** After the decision and, for an accepted attempt, after its commit; also after a
      failure, as `abandoned`. Releases what `begin` reserved on every outcome. */
  virtual void end(HostOutcome outcome) = 0;
};

} // namespace solprop_native
