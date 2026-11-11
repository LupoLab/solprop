// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The CPU execution record the native tests build workspaces with (solprop SEP 46):
// solprop_native's built-in table for this platform with idle threads asleep, as
// `_execution.builtin` returns it without a spin. Native code holds no values of its
// own; a workspace refuses a problem without a record.

#include "solprop_native/core.hpp"

namespace solprop_native::test {

inline CpuExecution make_table() {
  CpuExecution table;
#ifdef __APPLE__
  constexpr std::size_t stage = 8192;
  constexpr std::size_t fft = 32768;
  constexpr std::size_t fft_thread = 32768;
  constexpr std::size_t fft_long_row = 65536;
  constexpr bool stable = false;
#else
  constexpr std::size_t stage = 4096;
  constexpr std::size_t fft = 8192;
  constexpr std::size_t fft_thread = 4096;
  constexpr std::size_t fft_long_row = 16384;
  constexpr bool stable = true;
#endif
  const auto set = [&](CpuOperation operation, std::size_t threshold,
                       std::size_t full) {
    table.grain[static_cast<std::size_t>(operation)] = {threshold, full};
  };
  set(CpuOperation::stage, stage, stage);
  set(CpuOperation::response, 32768, 196608);
  set(CpuOperation::dense, 1048576, 1048576);
  set(CpuOperation::fft, fft, 4194304);
  set(CpuOperation::raman, 65536, 262144);
  set(CpuOperation::copy, 65536, 65536);
  set(CpuOperation::flow, 32768, 32768);
  set(CpuOperation::modal, 65536, 65536);
  set(CpuOperation::adams, 8192, 65536);
  table.response_full_samples = 65536;
  table.response_full_columns = 40;
  table.medium_threads = 4;
  table.compact_threads = 8;
  table.compact_max_fine_samples = 65536;
  table.stable_kernel_team = stable;
  table.fft_thread_elements = fft_thread;
  table.fft_long_row = fft_long_row;
  return table;
}

/** The record every test problem and pool takes unless a test sets its own. */
inline const CpuExecution& table() {
  static const CpuExecution value = make_table();
  return value;
}

} // namespace solprop_native::test
